"""Direction-explicit fixed-design pushover diagnostics (benchmark preparation, 2026-09-20).

This is not the production pushover. ``Analysis/Pushover.py`` drives DOF 1
of the roof master under the uniform FX_FLOOR pattern of Loads/Lateral_Loads,
reads X reactions only and flags hinge yield against the interior family's
thresholds; it has no Y or reverse-direction path. This module pushes the
roof master in +X, -X, +Y or -Y under an explicit, recorded lateral pattern
on top of a declared gravity state and records, at every converged step:

* the signed floor-master displacements (ux, uy, rz) and the signed edge
  displacements of the loaded direction;
* every column's local end forces, the internal moment about the loaded
  bending axis transformed to the joint faces, and the story shear it
  carries in the push direction (sign rule below);
* every beam's end forces and its vertical-bending end moments;
* every IMK hinge's deformation and moment with yield / cap / ultimate
  events measured against the backbone actually installed on that end
  (Model.IMK_Hinges.hinge_registry), per sign;
* the base reactions, the applied lateral loads and the gravity state;
* the column-to-line story shear sums per frame line;
* equilibrium and sign checks: base shear, story shear per story, the
  moment balance at every joint for the lateral increment, and the
  linearity of every column's moment diagram, all reported as residuals.

Gravity and lateral responses are kept apart: the gravity state is step 0
and every later step also carries its increment from it. Nothing here
redesigns, rebuilds the design record or writes outside ``output_dir``.

Sign conventions (validated by tests/test_pushover_diagnostic.py on a
cantilever and on a small frame):

* Column elements use vecxz = (1, 0, 0) with local x upward: local y is
  -global Y and local z is +global X. ``localForce`` is the 12-vector
  [N, Vy, Vz, T, My, Mz] at end i (bottom) then end j (top), the forces
  acting ON the element at its ends. The story shear a column carries in
  +X is -Vz_i; in +Y it is +Vy_i.
* The internal moment along a column (no transverse load) is linear:
  M(x) = -My_i - Vz_i x for bending about local y (X push) and
  M(x) = -Mz_i + Vy_i x for bending about local z (Y push), x measured
  from end i; M(L) equals My_j (Mz_j), which is checked at every step.
* Beam elements use vecxz = (0, 0, 1): local z is +Z (up) and vertical
  bending is My with shear Vz, for beam_x (local y = +Y) and beam_y (local
  y = -X) alike.
* Global moments about the loaded bending axis at a joint: X push (about
  global Y): column -My, beam_x +My, beam_y torsion +T; Y push (about
  global X): column +Mz, beam_y -My, beam_x torsion +T. Their sum over the
  members at a joint is zero for the lateral increment.
* A hinge's moment is the physical element's end moment about the spring
  axis; its deformation is the spring's own.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import openseespy.opensees as ops

import Structure_Parameters as sp
from Model.Build_Model import apply_analysis_constraints, build_model, floor_master_node, node_tag, roof_master_node
from Analysis.Gravity import run_gravity_analysis
from Loads.Gravity_Loads import apply_gravity_loads
from Loads.Seismic_ELF import elf_story_forces
from Model.IMK_Hinges import hinge_element_tag, hinge_registry, imk_hinge_stiffness


DIAGNOSTIC_VERSION = "pushover_diagnostic_v1_direction_explicit"
PATTERN_SERIES_TAG = 31
PATTERN_TAG = 31
LOAD_PATTERNS = ("elf", "uniform", "triangular", "explicit")
DIRECTION_DOF = {"x": 1, "y": 2}
# Column story shear in the push direction from the element's local end-i forces.
COLUMN_STORY_SHEAR = {"x": lambda f: -f[2], "y": lambda f: f[1]}
# Internal moment about the loaded bending axis along a column, x from end i.
COLUMN_MOMENT_ALONG = {"x": lambda f, x: -f[4] - f[2] * x, "y": lambda f, x: -f[5] + f[1] * x}
COLUMN_END_MOMENT_INDEX = {"x": (4, 10), "y": (5, 11)}
COLUMN_SHEAR_INDEX = {"x": (2, 8), "y": (1, 7)}
# Global moment about the loaded bending axis from a member's local end moments (index, factor).
GLOBAL_MOMENT_TERMS = {
    "x": {"column": [(4, -1.0)], "beam_x": [(4, 1.0)], "beam_y": [(3, 1.0)]},
    "y": {"column": [(5, 1.0)], "beam_y": [(4, -1.0)], "beam_x": [(3, 1.0)]},
}
# Which hinge spring (deformation index) carries the loaded bending: columns'
# springs are about global X (index 0) and global Y (index 1); beams' index 0
# is vertical bending.
HINGE_ACTIVE_INDEX = {"x": {"column": 1, "beam_x": 0, "beam_y": None}, "y": {"column": 0, "beam_y": 0, "beam_x": None}}


@dataclass
class RecoveryRule:
    name: str
    algorithm: tuple
    tolerance: float
    max_iterations: int
    du_factor: float = 1.0


@dataclass
class DiagnosticSettings:
    """Every choice the diagnostic makes, recorded with its results."""
    direction: str = "y"
    sign: float = 1.0
    load_pattern: str = "elf"                     # LOAD_PATTERNS
    explicit_weights: list | None = None          # per elevated floor, for "explicit"
    model_period_sec: float | None = None         # ELF shape at this period (None: ASCE Ta)
    reference_base_shear_kip: float = 1.0         # the pattern sums to this, so lambda is the base shear / reference
    gravity_dead_factor: float = 1.0
    gravity_live_factor: float = 0.25             # ASCE 41 effective live fraction; the NTHA state uses 1.0
    gravity_self_weight_factor: float = 1.0
    live_pattern: str = "all"
    du_in: float = 0.05
    max_steps: int = 1200
    target_roof_drift_ratio: float = 0.04
    tolerance: float = 1.0e-5
    max_iterations: int = 50
    recovery: list = field(default_factory=lambda: [
        RecoveryRule("NewtonRelaxed", ("Newton",), 5.0e-5, 100, 1.0),
        RecoveryRule("ModifiedNewton", ("ModifiedNewton",), 5.0e-5, 200, 1.0),
        RecoveryRule("NewtonLineSearch", ("NewtonLineSearch",), 5.0e-5, 200, 1.0),
        RecoveryRule("HalfStepNewton", ("Newton",), 5.0e-5, 100, 0.5)])
    stop_on_strength_loss_fraction: float = 0.2   # stop once lambda <= fraction * peak lambda; 0 disables
    record_every: int = 1
    hinge_yield_tolerance: float = 1.0e-6
    equilibrium_tolerance_kip: float = 1.0e-4
    joint_balance_tolerance: float = 1.0e-4       # relative to the largest member moment at the joint
    label: str = ""

    def validate(self):
        if self.direction not in DIRECTION_DOF:
            raise ValueError("direction must be 'x' or 'y'")
        if self.sign not in (1.0, -1.0):
            raise ValueError("sign must be +1.0 or -1.0")
        if self.load_pattern not in LOAD_PATTERNS:
            raise ValueError(f"load_pattern must be one of {LOAD_PATTERNS}")
        if self.load_pattern == "explicit" and (not self.explicit_weights or len(self.explicit_weights) != sp.NUM_FLOOR):
            raise ValueError("explicit_weights must give one weight per elevated floor")
        if self.du_in <= 0 or self.max_steps <= 0 or self.target_roof_drift_ratio <= 0:
            raise ValueError("du_in, max_steps and target_roof_drift_ratio must be positive")
        if not 0.0 <= self.stop_on_strength_loss_fraction < 1.0:
            raise ValueError("stop_on_strength_loss_fraction must be in [0, 1)")

    def to_dict(self):
        data = asdict(self)
        data["recovery"] = [asdict(r) if not isinstance(r, dict) else r for r in self.recovery]
        data["case"] = f"{self.direction}{'+' if self.sign > 0 else '-'}"
        return data


# ---------------------------------------------------------------------------
# Lateral pattern
# ---------------------------------------------------------------------------

def pattern_weights(settings):
    """Per-floor lateral weights summing to the reference base shear, with their basis."""
    n = sp.NUM_FLOOR
    if settings.load_pattern == "elf":
        elf = elf_story_forces(settings.model_period_sec)
        raw = list(elf["story_forces_kip"])
        basis = {"kind": "ASCE 7-22 12.8.3 vertical distribution", "design_period_sec": elf["design_period_sec"],
                 "vertical_distribution_k": elf["vertical_distribution_k"], "asce_ta_sec": elf["asce_ta_sec"],
                 "period_capped_at_cu_ta": elf["period_capped_at_cu_ta"], "elf_base_shear_kip": elf["base_shear_kip"]}
    elif settings.load_pattern == "uniform":
        raw = [1.0] * n
        basis = {"kind": "uniform: equal force at every floor"}
    elif settings.load_pattern == "triangular":
        raw = [float(k) for k in range(1, n + 1)]
        basis = {"kind": "triangular: proportional to height"}
    else:
        raw = [float(w) for w in settings.explicit_weights]
        basis = {"kind": "explicit weights supplied by the caller"}
    total = sum(raw)
    if total <= 0:
        raise ValueError("lateral pattern weights must sum to a positive value")
    weights = [w / total * settings.reference_base_shear_kip for w in raw]
    return weights, basis


def apply_pattern(settings, weights):
    ops.timeSeries("Linear", PATTERN_SERIES_TAG)
    ops.pattern("Plain", PATTERN_TAG, PATTERN_SERIES_TAG)
    for k, w in enumerate(weights, start=1):
        fx = settings.sign * w if settings.direction == "x" else 0.0
        fy = settings.sign * w if settings.direction == "y" else 0.0
        ops.load(floor_master_node(k), fx, fy, 0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def _length(n_i, n_j):
    xi, yi, zi = ops.nodeCoord(n_i)
    xj, yj, zj = ops.nodeCoord(n_j)
    return math.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)


def element_inventory():
    """Columns, beams and joints of the frame in the builders' tag order."""
    columns, beams = [], []
    tag = 1
    for k in range(sp.NUM_FLOOR):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                columns.append({"tag": tag, "node_i": node_tag(k, i, j), "node_j": node_tag(k + 1, i, j), "story": k + 1,
                                "grid_i": i, "grid_j": j, "length_in": _length(node_tag(k, i, j), node_tag(k + 1, i, j))})
                tag += 1
    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X):
                beams.append({"tag": tag, "kind": "beam_x", "node_i": node_tag(k, i, j), "node_j": node_tag(k, i + 1, j),
                              "floor": k, "line": j, "span": i, "length_in": _length(node_tag(k, i, j), node_tag(k, i + 1, j))})
                tag += 1
    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y):
            for i in range(sp.NUM_BAY_X + 1):
                beams.append({"tag": tag, "kind": "beam_y", "node_i": node_tag(k, i, j), "node_j": node_tag(k, i, j + 1),
                              "floor": k, "line": i, "span": j, "length_in": _length(node_tag(k, i, j), node_tag(k, i, j + 1))})
                tag += 1
    joints = {}
    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                joints[(k, i, j)] = {"node": node_tag(k, i, j), "floor": k, "grid_i": i, "grid_j": j, "members": []}
    for c in columns:
        # A story-k column runs from floor k-1 (end i) to floor k (end j): it is the column below the
        # floor-k joint and, for k >= 2, the column above the floor-(k-1) joint.
        joints[(c["story"], c["grid_i"], c["grid_j"])]["members"].append({"tag": c["tag"], "kind": "column", "end": "j", "role": "column_below"})
        if c["story"] >= 2:
            joints[(c["story"] - 1, c["grid_i"], c["grid_j"])]["members"].append({"tag": c["tag"], "kind": "column", "end": "i", "role": "column_above"})
    for b in beams:
        if b["kind"] == "beam_x":
            joints[(b["floor"], b["span"], b["line"])]["members"].append({"tag": b["tag"], "kind": "beam_x", "end": "i", "role": "beam_x_right"})
            joints[(b["floor"], b["span"] + 1, b["line"])]["members"].append({"tag": b["tag"], "kind": "beam_x", "end": "j", "role": "beam_x_left"})
        else:
            joints[(b["floor"], b["line"], b["span"])]["members"].append({"tag": b["tag"], "kind": "beam_y", "end": "i", "role": "beam_y_ahead"})
            joints[(b["floor"], b["line"], b["span"] + 1)]["members"].append({"tag": b["tag"], "kind": "beam_y", "end": "j", "role": "beam_y_behind"})
    # The columns above a floor are the joint's "column_above" entries; the roof has none.
    return {"columns": columns, "beams": beams,
            "joints": [{"key": f"{k},{i},{j}", **v} for (k, i, j), v in sorted(joints.items())]}


def hinge_inventory(inventory):
    """Every IMK hinge with the installed backbone the events are measured against, per end and sign."""
    registry = hinge_registry()
    hinges = []
    for member in inventory["columns"] + inventory["beams"]:
        entry = registry.get(member["tag"])
        if entry is None:
            continue
        kind = entry["member_type"]
        length = entry["length_in"]
        # The installed spring stiffness is recorded at build time from the
        # member's own properties (its line's T-section, its axial state); a
        # recomputation here without them would disagree for edge beams.
        # Older registries without the entry fall back to the recomputation.
        ke_y = entry.get("ke_y_kip_in_per_rad") or imk_hinge_stiffness(kind, "rot_y", length)
        ke_z = entry.get("ke_z_kip_in_per_rad") or imk_hinge_stiffness(kind, "rot_z", length)
        for end_id, end in ((1, "i"), (2, "j")):
            if kind == "column":
                fy_pos = fy_neg = entry["yield_moment_y_kip_in"]
                fz = entry["yield_moment_z_kip_in"]
            else:
                # Beam springs: hogging is positive deformation at end i and negative at end j.
                hog = entry[f"yield_moment_y_hogging_{end}_kip_in"]
                sag = entry[f"yield_moment_y_sagging_{end}_kip_in"]
                fy_pos, fy_neg = (hog, sag) if end == "i" else (sag, hog)
                fz = entry["yield_moment_z_kip_in"]
            hinges.append({"hinge_tag": hinge_element_tag(member["tag"], end_id), "member_tag": member["tag"], "member_type": kind,
                           "material_type": entry.get("material_type", "IMKBilin"),
                           "installed_materials": entry.get("installed_materials", {}).get(end, {}),
                           "end": end, "joint_node": member["node_i"] if end == "i" else member["node_j"],
                           "springs": {"y": {"index": 0, "fy_positive_kip_in": fy_pos, "fy_negative_kip_in": fy_neg, "ke": ke_y,
                                             "theta_y_positive": fy_pos / ke_y if ke_y > 0 else None,
                                             "theta_y_negative": fy_neg / ke_y if ke_y > 0 else None},
                                       "z": {"index": 1, "fy_positive_kip_in": fz, "fy_negative_kip_in": fz, "ke": ke_z,
                                             "theta_y_positive": fz / ke_z if ke_z > 0 else None,
                                             "theta_y_negative": fz / ke_z if ke_z > 0 else None}},
                           "theta_p": entry["theta_p"], "theta_pc": entry["theta_pc"], "theta_u": entry["theta_u"],
                           "axial_ratio": entry.get("axial_ratio"), "backbone_source": entry.get("source"),
                           "strength_basis": entry.get("strength_basis"), "beam_family": entry.get("beam_family")})
    return hinges


def model_audit(inventory, hinges):
    """What the built model represents, stated from its own registry and parameters."""
    registry = hinge_registry()
    families = {}
    columns = []
    for entry in registry.values():
        if entry["member_type"] == "column":
            columns.append(entry)
        else:
            families.setdefault(entry.get("beam_family"), entry)
    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "element_formulation": sp.ELEMENT_FORMULATION,
        "imk": {"material": sp.IMK_MATERIAL_TYPE, "columns": sp.IMK_APPLY_TO_COLUMNS, "beams": sp.IMK_APPLY_TO_BEAMS,
                "stiffness_mode": getattr(sp, "IMK_HINGE_STIFFNESS_MODE", None), "stiffness_factor": sp.IMK_HINGE_STIFFNESS_FACTOR,
                "fmax_over_fy": (getattr(sp, "IMK_FMAXFY_POS", None), getattr(sp, "IMK_FMAXFY_NEG", None)),
                "calibrated_backbone": getattr(sp, "IMK_USE_CALIBRATED_BACKBONE", None),
                "beam_theta_y_target": sp.IMK_BEAM_THETA_Y, "column_theta_y_target": sp.IMK_COLUMN_THETA_Y},
        "stiffness_modifiers": {"beam": sp.section_stiffness_modifier("beam"), "column": sp.section_stiffness_modifier("column")},
        "transformations": {"column": "PDelta, vecxz (1, 0, 0)", "beam_x": "Linear, vecxz (0, 0, 1)", "beam_y": "Linear, vecxz (0, 0, 1)"},
        "joints": "centerline nodes; zero-length IMK springs at the member ends; no rigid end zones, no panel-zone element, "
                  "no joint shear deformation or strength",
        "diaphragm": "rigid (ux, uy, rz of every floor node tied to the floor master)",
        "base": "fixed (all six DOFs)",
        "gravity_model": sp.effective_gravity_load_model(),
        "sections": {"b_col_in": sp.B_COL, "h_col_in": sp.H_COL, "fc_col_ksi": sp.FC_COL_KSI, "b_beam_in": sp.B_BEAM,
                     "h_beam_in": sp.H_BEAM, "fc_beam_ksi": sp.FC_BEAM_KSI, "slab_thickness_in": sp.SLAB_THICKNESS_IN},
        "cage": {"col_bar_size": sp.COL_BAR_SIZE, "col_top_bars": sp.COL_TOP_BARS, "col_bot_bars": sp.COL_BOT_BARS,
                 "col_side_bars": sp.COL_SIDE_BARS, "beam_bar_size": sp.BEAM_BAR_SIZE, "beam_top_bars": sp.BEAM_TOP_BARS,
                 "beam_bot_bars": sp.BEAM_BOT_BARS, "col_hoops": (sp.COL_STIRRUP_BAR_SIZE, sp.COL_STIRRUP_LEGS, sp.COL_STIRRUP_SPACING),
                 "beam_hoops": (sp.BEAM_STIRRUP_BAR_SIZE, sp.BEAM_STIRRUP_LEGS, sp.BEAM_STIRRUP_SPACING),
                 "hoops_enter_the_model_through": "rho_sh in the Haselton rotation capacities only (IMK_Calibration)"},
        "beam_hinge_strengths_by_family": {fam: {"hogging_kip_in": e["yield_moment_y_hogging_kip_in"], "sagging_kip_in": e["yield_moment_y_sagging_kip_in"],
                                                 "basis": e.get("strength_basis"), "flange_width_in": e.get("flange_width_in"),
                                                 "exterior_slab_anchorage": e.get("exterior_slab_anchorage")}
                                           for fam, e in families.items()},
        "column_hinges": {"count": len(columns),
                          "yield_moment_range_kip_in": ([min(e["yield_moment_y_kip_in"] for e in columns), max(e["yield_moment_y_kip_in"] for e in columns)]
                                                        if columns else None),
                          "axial_ratio_range": ([min(e["axial_ratio"] for e in columns), max(e["axial_ratio"] for e in columns)] if columns else None),
                          "strength_basis": "nominal P-M surface (fy, phi = 1) at the tributary gravity axial estimate fixed at build "
                                            "time; the same value serves both bending axes (my = mz); not updated with the solved axial force",
                          "theta_p_range": ([min(e["theta_p"] for e in columns), max(e["theta_p"] for e in columns)] if columns else None)},
        "represented": ["installed final cage (sections, bars) and hoops through rho_sh",
                        "developed slab mats in the ACI 6.3.2 flange in beam hinge strengths and T/L stiffness, per family, "
                        "with undeveloped exterior ends where the perimeter anchorage fails",
                        "beam hinge asymmetry (hogging / sagging) per end", "P-Delta on the columns", "rigid diaphragms",
                        "column hinge strength from the nominal P-M surface at an estimated gravity axial load"],
        "not_represented": ["probable (1.25 fy) member strengths: hinges yield at nominal Mn and harden to FmaxFy x Mn",
                            "axial-moment interaction during the push (column hinge strength fixed at build time; no redistribution)",
                            "distinct column strengths about the two axes for unequal bar layouts (my = mz)",
                            "joint shear strength, deformation or panel zones; rigid end offsets",
                            "member shear strength or shear failure (elastic spines, flexural hinges only)",
                            "bar slip beyond the a_sl term of the Haselton rotation capacities", "foundation flexibility",
                            "transverse-beam torsional strength (elastic torsion only)"],
        "hinge_count": len(hinges),
        "element_count": {"columns": len(inventory["columns"]), "beams": len(inventory["beams"])},
    }


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def _base_nodes():
    return [node_tag(0, i, j) for j in range(sp.NUM_BAY_Y + 1) for i in range(sp.NUM_BAY_X + 1)]


def _global_forces(inventory):
    """Global end forces [Fx, Fy, Fz, Mx, My, Mz] at i then j, acting on the element (P-Delta included)."""
    forces = {}
    for member in inventory["columns"] + inventory["beams"]:
        f = list(ops.eleResponse(member["tag"], "force"))
        if len(f) != 12 or not all(math.isfinite(v) for v in f):
            raise RuntimeError(f"member {member['tag']}: global force vector is not 12 finite values")
        forces[member["tag"]] = f
    return forces


def _reactions(inventory, global_forces):
    """Base reactions from the story-1 columns' global end-i forces (the force the support applies).

    In the hinge model the base node connects to its column through a
    zero-length spring and penalty equalDOF ties, so ``ops.reactions`` sees
    no translational element force there and ``nodeReaction`` reads zero
    (the production pushover's base shear has the same blind spot). The
    column's own end forces are the reactions; the nodal values are kept
    beside them for the record.
    """
    rows, totals = {}, [0.0] * 6
    for c in inventory["columns"]:
        if c["story"] != 1:
            continue
        r = global_forces[c["tag"]][:6]
        rows[str(c["node_i"])] = {"column_tag": c["tag"], "reaction": r}
        totals = [a + b for a, b in zip(totals, r)]
    ops.reactions()
    nodal = [0.0] * 6
    for node in _base_nodes():
        nodal = [a + ops.nodeReaction(node, dof) for a, dof in zip(nodal, range(1, 7))]
    return rows, totals, nodal


def _floor_displacements():
    return {str(k): {"ux": ops.nodeDisp(floor_master_node(k), 1), "uy": ops.nodeDisp(floor_master_node(k), 2),
                     "rz": ops.nodeDisp(floor_master_node(k), 6)} for k in range(1, sp.NUM_FLOOR + 1)}


def _edge_displacements(direction):
    edges = {}
    for k in range(1, sp.NUM_FLOOR + 1):
        if direction == "x":
            a, b = node_tag(k, 0, 0), node_tag(k, 0, sp.NUM_BAY_Y)
        else:
            a, b = node_tag(k, 0, 0), node_tag(k, sp.NUM_BAY_X, 0)
        dof = DIRECTION_DOF[direction]
        edges[str(k)] = {"edge_a_in": ops.nodeDisp(a, dof), "edge_b_in": ops.nodeDisp(b, dof)}
    return edges


def _member_forces(inventory):
    forces = {}
    for member in inventory["columns"] + inventory["beams"]:
        f = list(ops.eleResponse(member["tag"], "localForce"))
        if len(f) != 12 or not all(math.isfinite(v) for v in f):
            raise RuntimeError(f"member {member['tag']}: local force vector is not 12 finite values")
        forces[member["tag"]] = f
    return forces


def _column_rows(inventory, forces, global_forces, direction, gravity_forces=None, gravity_global=None):
    """Per-column record: local end forces, face moments, and the story shear carried (global frame).

    The story shear is taken from the GLOBAL end forces, which carry the
    P-Delta contribution the local (chord-frame) shear lacks; the local
    value is kept beside it with the difference. The moment diagram is
    linear in the chord frame, so the face moments come from the local
    forces.
    """
    a_bottom = sp.H_BEAM / 2.0
    dof = DIRECTION_DOF[direction] - 1
    rows = []
    for c in inventory["columns"]:
        f, fg = forces[c["tag"]], global_forces[c["tag"]]
        length = c["length_in"]
        x_bottom_face = 0.0 if c["story"] == 1 else a_bottom
        x_top_face = length - sp.H_BEAM / 2.0
        moment = COLUMN_MOMENT_ALONG[direction]
        m_i, m_j = COLUMN_END_MOMENT_INDEX[direction]
        row = {"tag": c["tag"], "story": c["story"], "grid_i": c["grid_i"], "grid_j": c["grid_j"],
               "local_force": f, "global_force": fg, "axial_i_kip": f[0], "axial_j_kip": -f[6],
               "story_shear_kip": -fg[dof], "story_shear_local_kip": COLUMN_STORY_SHEAR[direction](f),
               "p_delta_shear_difference_kip": -fg[dof] - COLUMN_STORY_SHEAR[direction](f),
               "moment_center_i_kip_in": f[m_i], "moment_center_j_kip_in": f[m_j],
               "internal_moment_at_i_kip_in": moment(f, 0.0), "internal_moment_at_j_kip_in": moment(f, length),
               "moment_face_bottom_kip_in": moment(f, x_bottom_face), "moment_face_top_kip_in": moment(f, x_top_face),
               "face_offsets_in": [x_bottom_face, length - x_top_face],
               "linearity_residual_kip_in": moment(f, length) - f[m_j],
               "shear_end_residual_kip": f[COLUMN_SHEAR_INDEX[direction][0]] + f[COLUMN_SHEAR_INDEX[direction][1]]}
        if gravity_forces is not None:
            g, gg = gravity_forces[c["tag"]], gravity_global[c["tag"]]
            d = [a - b for a, b in zip(f, g)]
            dg = [a - b for a, b in zip(fg, gg)]
            row["increment"] = {"story_shear_kip": -dg[dof], "story_shear_local_kip": COLUMN_STORY_SHEAR[direction](d),
                                "moment_center_i_kip_in": d[m_i], "moment_center_j_kip_in": d[m_j],
                                "moment_face_bottom_kip_in": moment(d, x_bottom_face), "moment_face_top_kip_in": moment(d, x_top_face),
                                "axial_i_kip": d[0], "axial_j_kip": -d[6]}
        rows.append(row)
    return rows


def _beam_rows(inventory, forces, gravity_forces=None):
    rows = []
    for b in inventory["beams"]:
        f = forces[b["tag"]]
        row = {"tag": b["tag"], "kind": b["kind"], "floor": b["floor"], "line": b["line"], "span": b["span"],
               "local_force": f, "moment_y_i_kip_in": f[4], "moment_y_j_kip_in": f[10], "shear_z_i_kip": f[2], "torsion_i_kip_in": f[3]}
        if gravity_forces is not None:
            g = gravity_forces[b["tag"]]
            d = [a - c for a, c in zip(f, g)]
            # The lateral increment carries no span load: its moment is linear and its face values follow.
            depth = sp.H_COL if b["kind"] == "beam_x" else sp.B_COL
            row["increment"] = {"moment_y_i_kip_in": d[4], "moment_y_j_kip_in": d[10], "torsion_i_kip_in": d[3],
                                "moment_face_i_kip_in": -d[4] - d[2] * (depth / 2.0),
                                "moment_face_j_kip_in": -d[4] - d[2] * (b["length_in"] - depth / 2.0)}
        rows.append(row)
    return rows


def _line_story_shears(column_rows, direction, use_increment):
    """Column-to-line story shear sums: lines are the frames parallel to the push direction."""
    lines = {}
    for row in column_rows:
        line = row["grid_j"] if direction == "x" else row["grid_i"]
        value = row["increment"]["story_shear_kip"] if use_increment else row["story_shear_kip"]
        lines.setdefault(str(row["story"]), {}).setdefault(str(line), 0.0)
        lines[str(row["story"])][str(line)] += value
    return {story: {"by_line_kip": per, "total_kip": sum(per.values())} for story, per in lines.items()}


def _joint_balances(inventory, global_forces, gravity_global, local_forces, gravity_local, direction, tolerance):
    """Moment balance of the lateral increment about the loaded bending axis at every joint.

    The balance sums the members' GLOBAL end moments about the loaded axis
    (global Y for an X push, global X for a Y push), read from the element's
    global force response. The local-frame mapping of the module docstring
    (GLOBAL_MOMENT_TERMS) is evaluated beside it and its largest disagreement
    with the global value is reported as ``mapping_max_difference_kip_in``.
    """
    terms = GLOBAL_MOMENT_TERMS[direction]
    axis_index = 4 if direction == "x" else 3
    balances = []
    worst, mapping_worst = 0.0, 0.0
    for joint in inventory["joints"]:
        contributions = []
        for m in joint["members"]:
            offset = 0 if m["end"] == "i" else 6
            dg = [a - b for a, b in zip(global_forces[m["tag"]], gravity_global[m["tag"]])]
            dl = [a - b for a, b in zip(local_forces[m["tag"]], gravity_local[m["tag"]])]
            value = dg[offset + axis_index]
            mapped = sum(factor * dl[offset + index] for index, factor in terms[m["kind"]])
            mapping_worst = max(mapping_worst, abs(value - mapped))
            contributions.append({"tag": m["tag"], "kind": m["kind"], "end": m["end"], "role": m["role"],
                                  "global_moment_kip_in": value, "mapped_from_local_kip_in": mapped})
        total = sum(c["global_moment_kip_in"] for c in contributions)
        scale = max([abs(c["global_moment_kip_in"]) for c in contributions] + [1.0])
        residual = total / scale
        worst = max(worst, abs(residual))
        column_below = sum(c["global_moment_kip_in"] for c in contributions if c["role"] == "column_below")
        column_above = sum(c["global_moment_kip_in"] for c in contributions if c["role"] == "column_above")
        beams_in = sum(c["global_moment_kip_in"] for c in contributions if c["kind"] == ("beam_x" if direction == "x" else "beam_y"))
        transverse = sum(c["global_moment_kip_in"] for c in contributions if c["kind"] == ("beam_y" if direction == "x" else "beam_x"))
        denominator = abs(column_below) + abs(column_above)
        share_below = abs(column_below) / denominator if denominator > 1e-6 * scale else None
        balances.append({"joint": joint["key"], "floor": joint["floor"], "grid_i": joint["grid_i"], "grid_j": joint["grid_j"],
                         "contributions": contributions, "sum_kip_in": total, "relative_residual": residual,
                         "column_below_kip_in": column_below, "column_above_kip_in": column_above,
                         "beams_in_direction_kip_in": beams_in, "transverse_beam_torsion_kip_in": transverse,
                         "share_to_column_below": share_below,
                         "share_basis": ("|M column below| / (|M below| + |M above|) of the lateral increment at the joint "
                                         "centre; None where the column moments are within 1e-6 of the joint's largest "
                                         "member moment (undefined, not zero)")})
    return balances, worst, mapping_worst


def _hinge_state(hinges):
    """Spring deformations and moments of each zero-length hinge.

    ``basicForce`` is the spring force on the retained (joint) node side:
    for every member kind and both ends it is the negative of the physical
    element's end moment projected on the spring axis (probed on the
    fixture frame, 2026-09-20). It is negated here so the recorded spring
    moment shares the element-end convention of ``localForce`` and the
    cross-check against the element is a plain equality.
    """
    state = {}
    for h in hinges:
        deformation = list(ops.eleResponse(h["hinge_tag"], "deformation"))
        basic = list(ops.eleResponse(h["hinge_tag"], "basicForce"))
        moment = [-v for v in (basic + [0.0, 0.0])[:2]]
        state[h["hinge_tag"]] = {"deformation": (deformation + [0.0, 0.0])[:2], "moment": moment, "basic_force": basic}
    return state


def _hinge_rows(hinges, hinge_state, forces, direction, events, step, control_disp, base_shear, tolerance):
    rows = []
    for h in hinges:
        deformation = hinge_state[h["hinge_tag"]]["deformation"]
        m_y_spring, m_z_spring = hinge_state[h["hinge_tag"]]["moment"]
        offset = 0 if h["end"] == "i" else 6
        f = forces[h["member_tag"]]
        kind = h["member_type"]
        # Cross-check: the physical element's end moment about the spring axis must equal the spring moment.
        if kind == "column":
            element_moments = (f[offset + 5], -f[offset + 4])           # about global X (local Mz), about global Y (-local My)
        else:
            element_moments = (f[offset + 4], f[offset + 5])            # vertical bending, in-plane bending
        active = HINGE_ACTIVE_INDEX[direction].get(kind)
        row = {"hinge_tag": h["hinge_tag"], "member_tag": h["member_tag"], "member_type": kind, "end": h["end"],
               "deformation": deformation, "moment_y_kip_in": m_y_spring, "moment_z_kip_in": m_z_spring,
               "element_end_moments_kip_in": list(element_moments),
               "spring_element_moment_max_difference_kip_in": max(abs(m_y_spring - element_moments[0]), abs(m_z_spring - element_moments[1])),
               "active_spring_index": active, "state": {}}
        for name, spring in h["springs"].items():
            theta = deformation[spring["index"]]
            moment = m_y_spring if name == "y" else m_z_spring
            fy = spring["fy_positive_kip_in"] if theta >= 0 else spring["fy_negative_kip_in"]
            theta_y = spring["theta_y_positive"] if theta >= 0 else spring["theta_y_negative"]
            yielded = abs(moment) >= (1.0 - tolerance) * fy if fy else False
            capped = theta_y is not None and abs(theta) >= theta_y + h["theta_p"]
            ultimate = abs(theta) >= h["theta_u"]
            row["state"][name] = {"yielded": yielded, "capped": capped, "ultimate": ultimate, "fy_sign_kip_in": fy,
                                  "theta_y_sign": theta_y, "moment_over_fy": abs(moment) / fy if fy else None}
            key = (h["hinge_tag"], name)
            seen = events.setdefault(key, {"yielded": None, "capped": None, "ultimate": None})
            for flag, value in (("yielded", yielded), ("capped", capped), ("ultimate", ultimate)):
                if value and seen[flag] is None:
                    seen[flag] = {"step": step, "control_displacement_in": control_disp, "base_shear_kip": base_shear,
                                  "member_tag": h["member_tag"], "member_type": kind, "end": h["end"], "spring": name,
                                  "rotation": theta, "moment_kip_in": moment, "fy_kip_in": fy, "beam_family": h.get("beam_family")}
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Gravity bookkeeping
# ---------------------------------------------------------------------------

def expected_gravity_total_kip(settings):
    """Total vertical load the declared gravity state applies, from the load inventory (None if unknown)."""
    try:
        floors = sp.NUM_FLOOR
        self_weight = settings.gravity_self_weight_factor * (
            sp.col_self_weight_kip_per_in() * sp.STORY_H * floors * (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1)
            + sp.beam_self_weight_kip_per_in("x") * sp.BAY_X * sp.NUM_BAY_X * (sp.NUM_BAY_Y + 1) * floors
            + sp.beam_self_weight_kip_per_in("y") * sp.BAY_Y * sp.NUM_BAY_Y * (sp.NUM_BAY_X + 1) * floors)
        mode = sp.effective_gravity_load_model()
        if mode == "slab_transfer":
            transfer = sp.FLOOR_TRANSFER
            live_case = "live" if settings.live_pattern in (None, "all") else f"live_pattern_{settings.live_pattern}"
            total = 0.0
            for name, factor in (("dead", settings.gravity_dead_factor), (live_case, settings.gravity_live_factor)):
                case = transfer["unit_cases"].get(name)
                if case is None or factor == 0.0:
                    continue
                per_floor = sum(load for beam in case["beams"] for _, load in beam["node_loads"])
                per_floor += sum(column["direct_load_kip"] for column in case["columns"])
                total += factor * per_floor * floors
            return total + self_weight, "slab transfer unit cases (beam node loads + column direct loads) x factors x floors + member self-weight"
        floor_dead = sp.floor_dead_load_ksf()
        total = floor_dead + sp.FLOOR_LIVE_LOAD_KSF
        floor_factor = (settings.gravity_dead_factor * floor_dead + settings.gravity_live_factor * sp.FLOOR_LIVE_LOAD_KSF) / total
        return floor_factor * sp.total_floor_gravity_load() * floors + self_weight, f"{mode} tributary floor load x factor x floors + member self-weight"
    except Exception as exc:                        # noqa: BLE001 -- unknown stays unknown
        return None, f"unavailable ({type(exc).__name__}: {exc})"


def apply_declared_gravity(settings):
    floor_dead = sp.floor_dead_load_ksf()
    total = floor_dead + sp.FLOOR_LIVE_LOAD_KSF
    floor_factor = (settings.gravity_dead_factor * floor_dead + settings.gravity_live_factor * sp.FLOOR_LIVE_LOAD_KSF) / total
    pattern = settings.live_pattern if sp.effective_gravity_load_model() == "slab_transfer" else "all"
    apply_gravity_loads(floor_factor=floor_factor, self_weight_factor=settings.gravity_self_weight_factor,
                        dead_factor=settings.gravity_dead_factor, live_factor=settings.gravity_live_factor, live_pattern=pattern)
    return {"floor_factor_for_tributary_modes": floor_factor, "dead_factor": settings.gravity_dead_factor,
            "live_factor": settings.gravity_live_factor, "self_weight_factor": settings.gravity_self_weight_factor,
            "live_pattern": pattern, "load_model": sp.effective_gravity_load_model()}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _analysis_setup(settings, control_node, dof):
    ops.wipeAnalysis()
    ops.system("BandGeneral")
    apply_analysis_constraints()
    ops.numberer("RCM")
    ops.test("NormDispIncr", settings.tolerance, settings.max_iterations)
    ops.algorithm("Newton")
    ops.integrator("DisplacementControl", control_node, dof, settings.sign * settings.du_in)
    ops.analysis("Static")


def _recover(settings, control_node, dof):
    for rule in settings.recovery:
        rule = rule if isinstance(rule, RecoveryRule) else RecoveryRule(**rule)
        ops.test("NormDispIncr", rule.tolerance, rule.max_iterations)
        ops.algorithm(*rule.algorithm)
        ops.integrator("DisplacementControl", control_node, dof, settings.sign * settings.du_in * rule.du_factor)
        ok = ops.analyze(1)
        ops.test("NormDispIncr", settings.tolerance, settings.max_iterations)
        ops.algorithm("Newton")
        ops.integrator("DisplacementControl", control_node, dof, settings.sign * settings.du_in)
        if ok == 0:
            return 0, rule.name
    return ok, None


def run_diagnostic(settings, output_dir, build=True, verbose=False):
    """Run one direction-explicit diagnostic on the current Structure_Parameters state; write its files.

    ``build`` rebuilds the IMK model (Model.Build_Model.build_model) and
    applies the declared gravity state; pass False only when the caller has
    done both. Returns the summary (also written to ``summary.json``); the
    per-step detail goes to ``steps.jsonl``.
    """
    settings.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if build:
        ops.wipe()
        build_model()
        gravity_settings = apply_declared_gravity(settings)
        gravity = run_gravity_analysis()
    else:
        gravity_settings, gravity = None, None
    inventory = element_inventory()
    hinges = hinge_inventory(inventory)
    audit = model_audit(inventory, hinges)
    control_node = roof_master_node()
    dof = DIRECTION_DOF[settings.direction]
    direction = settings.direction
    # Gravity state (step 0).
    gravity_forces = _member_forces(inventory)
    gravity_global = _global_forces(inventory)
    reactions, totals, nodal = _reactions(inventory, gravity_global)
    expected_total, expected_basis = expected_gravity_total_kip(settings)
    gravity_vertical_residual = (totals[2] - expected_total) if expected_total is not None else None
    gravity_columns = _column_rows(inventory, gravity_forces, gravity_global, direction)
    gravity_state = {"step": 0, "load_factor": 0.0, "control_displacement_in": ops.nodeDisp(control_node, dof),
                     "floor_displacements": _floor_displacements(), "edge_displacements": _edge_displacements(direction),
                     "reactions": reactions, "reaction_totals": totals, "nodal_reaction_totals": nodal,
                     "reaction_basis": "story-1 column global end-i forces; nodal_reaction_totals is ops.reactions, blind to "
                                       "the penalty-tied base translations of the hinge model",
                     "expected_vertical_load_kip": expected_total, "expected_vertical_basis": expected_basis,
                     "vertical_equilibrium_residual_kip": gravity_vertical_residual,
                     "columns": gravity_columns, "beams": _beam_rows(inventory, gravity_forces),
                     "hinges": [{"hinge_tag": tag, **state} for tag, state in _hinge_state(hinges).items()],
                     "line_story_shears": _line_story_shears(gravity_columns, direction, False)}
    weights, pattern_basis = pattern_weights(settings)
    apply_pattern(settings, weights)
    _analysis_setup(settings, control_node, dof)
    target = settings.target_roof_drift_ratio * sp.NUM_FLOOR * sp.STORY_H
    events = {}
    steps_path = output_dir / "steps.jsonl"
    scalars = []
    checks = {"base_shear_max_residual_kip": 0.0, "story_shear_max_residual_kip": 0.0, "joint_balance_max_relative_residual": 0.0,
              "column_linearity_max_residual_kip_in": 0.0, "column_shear_end_max_residual_kip": 0.0,
              "local_global_moment_mapping_max_difference_kip_in": 0.0, "spring_element_moment_max_difference_kip_in": 0.0,
              "p_delta_story_shear_max_difference_kip": 0.0,
              "sign_check": None, "control_direction_check": None, "all_pass": None}
    stop_reason, failed_step, algorithm_used = "max_steps", None, "Newton"
    peak_lambda = 0.0
    with steps_path.open("w", encoding="utf-8") as steps_file:
        steps_file.write(json.dumps({"kind": "gravity_state", **gravity_state}, allow_nan=False) + "\n")
        for step in range(1, settings.max_steps + 1):
            ok = ops.analyze(1)
            algorithm_used = "Newton"
            if ok != 0:
                ok, algorithm_used = _recover(settings, control_node, dof)
            if ok != 0:
                stop_reason, failed_step = "nonconvergence", step
                break
            lam = ops.getLoadFactor(PATTERN_TAG)
            control = ops.nodeDisp(control_node, dof)
            forces = _member_forces(inventory)
            global_forces = _global_forces(inventory)
            reactions, totals, nodal = _reactions(inventory, global_forces)
            applied = [settings.sign * lam * w for w in weights]
            base_shear = sum(applied)
            # Reactions are the forces on the columns at the base; their increment from the gravity state
            # balances the applied lateral load.
            base_residual = (totals[dof - 1] - gravity_state["reaction_totals"][dof - 1]) + base_shear
            column_rows = _column_rows(inventory, forces, global_forces, direction, gravity_forces, gravity_global)
            line_shears = _line_story_shears(column_rows, direction, True)
            story_residual = 0.0
            for k in range(1, sp.NUM_FLOOR + 1):
                expected = sum(applied[k - 1:])
                story_residual = max(story_residual, abs(line_shears[str(k)]["total_kip"] - expected))
            balances, joint_worst, mapping_worst = _joint_balances(inventory, global_forces, gravity_global, forces, gravity_forces,
                                                                   direction, settings.joint_balance_tolerance)
            linearity = max(abs(r["linearity_residual_kip_in"]) for r in column_rows)
            shear_end = max(abs(r["shear_end_residual_kip"]) for r in column_rows)
            p_delta = max(abs(r["p_delta_shear_difference_kip"]) for r in column_rows)
            hinge_rows = _hinge_rows(hinges, _hinge_state(hinges), forces, direction, events, step, control, base_shear,
                                     settings.hinge_yield_tolerance)
            spring_diff = max((r["spring_element_moment_max_difference_kip_in"] for r in hinge_rows), default=0.0)
            if checks["sign_check"] is None and abs(base_shear) > 0:
                story_1 = line_shears["1"]["total_kip"]
                checks["sign_check"] = {"story_1_column_sum_kip": story_1, "applied_base_shear_kip": base_shear,
                                        "same_sign": (story_1 > 0) == (base_shear > 0),
                                        "reaction_sum_kip": totals[dof - 1], "reaction_opposes_load": (totals[dof - 1] > 0) != (base_shear > 0)}
                checks["control_direction_check"] = {"control_displacement_in": control, "sign": settings.sign,
                                                     "moves_with_sign": (control > 0) == (settings.sign > 0),
                                                     "other_direction_in": ops.nodeDisp(control_node, 3 - dof)}
            checks["base_shear_max_residual_kip"] = max(checks["base_shear_max_residual_kip"], abs(base_residual))
            checks["story_shear_max_residual_kip"] = max(checks["story_shear_max_residual_kip"], story_residual)
            checks["joint_balance_max_relative_residual"] = max(checks["joint_balance_max_relative_residual"], joint_worst)
            checks["column_linearity_max_residual_kip_in"] = max(checks["column_linearity_max_residual_kip_in"], linearity)
            checks["column_shear_end_max_residual_kip"] = max(checks["column_shear_end_max_residual_kip"], shear_end)
            checks["local_global_moment_mapping_max_difference_kip_in"] = max(checks["local_global_moment_mapping_max_difference_kip_in"], mapping_worst)
            checks["spring_element_moment_max_difference_kip_in"] = max(checks["spring_element_moment_max_difference_kip_in"], spring_diff)
            checks["p_delta_story_shear_max_difference_kip"] = max(checks["p_delta_story_shear_max_difference_kip"], p_delta)
            scalar = {"step": step, "algorithm": algorithm_used, "load_factor": lam, "base_shear_kip": base_shear,
                      "control_displacement_in": control, "roof_drift_ratio": control / (sp.NUM_FLOOR * sp.STORY_H),
                      "base_shear_residual_kip": base_residual, "story_shear_residual_kip": story_residual,
                      "joint_balance_relative_residual": joint_worst, "column_linearity_residual_kip_in": linearity,
                      "yielded_hinges": sum(1 for r in hinge_rows for s in r["state"].values() if s["yielded"]),
                      "capped_hinges": sum(1 for r in hinge_rows for s in r["state"].values() if s["capped"])}
            scalars.append(scalar)
            if step % settings.record_every == 0:
                steps_file.write(json.dumps({"kind": "step", **scalar, "applied_loads_kip": applied,
                                             "floor_displacements": _floor_displacements(), "edge_displacements": _edge_displacements(direction),
                                             "reactions": reactions, "reaction_totals": totals, "nodal_reaction_totals": nodal,
                                             "columns": column_rows, "beams": _beam_rows(inventory, forces, gravity_forces),
                                             "line_story_shears": line_shears, "joint_balances": balances, "hinges": hinge_rows},
                                            allow_nan=False) + "\n")
            if verbose and step % 20 == 0:
                print(f"step {step:4d} lambda {lam:10.4f} u {control:9.4f} in V {base_shear:10.3f} kip "
                      f"yielded {scalar['yielded_hinges']} residual {base_residual:.2e}", flush=True)
            peak_lambda = max(peak_lambda, lam)
            if abs(control) >= target:
                stop_reason = "target_displacement_reached"
                break
            if settings.stop_on_strength_loss_fraction > 0 and peak_lambda > 0 and lam <= settings.stop_on_strength_loss_fraction * peak_lambda:
                stop_reason = "strength_loss_stop"
                break
    checks["all_pass"] = bool(
        checks["base_shear_max_residual_kip"] <= settings.equilibrium_tolerance_kip
        and checks["story_shear_max_residual_kip"] <= settings.equilibrium_tolerance_kip
        and checks["joint_balance_max_relative_residual"] <= settings.joint_balance_tolerance
        and checks["column_linearity_max_residual_kip_in"] <= settings.equilibrium_tolerance_kip * max(1.0, sp.STORY_H)
        and checks["column_shear_end_max_residual_kip"] <= settings.equilibrium_tolerance_kip
        and checks["local_global_moment_mapping_max_difference_kip_in"] <= settings.equilibrium_tolerance_kip * max(1.0, sp.STORY_H)
        and checks["spring_element_moment_max_difference_kip_in"] <= settings.equilibrium_tolerance_kip * max(1.0, sp.STORY_H)
        and (checks["sign_check"] or {}).get("same_sign") is True
        and (checks["sign_check"] or {}).get("reaction_opposes_load") is True
        and (checks["control_direction_check"] or {}).get("moves_with_sign") is True
        and (gravity_vertical_residual is None or abs(gravity_vertical_residual) <= 1e-6 * max(1.0, abs(expected_total))))
    first_events = sorted((e for seen in events.values() for e in seen.values() if e), key=lambda e: e["step"])
    summary = {"diagnostic_version": DIAGNOSTIC_VERSION, "settings": settings.to_dict(),
               "pattern": {"weights_kip": weights, "basis": pattern_basis, "reference_base_shear_kip": settings.reference_base_shear_kip,
                           "load_factor_meaning": "lambda x reference base shear = applied base shear (kip)"},
               "gravity": {"settings": gravity_settings, "analysis": gravity, "expected_vertical_load_kip": expected_total,
                           "expected_vertical_basis": expected_basis, "vertical_reaction_kip": gravity_state["reaction_totals"][2],
                           "vertical_equilibrium_residual_kip": gravity_vertical_residual,
                           "reaction_basis": gravity_state["reaction_basis"]},
               "model_audit": audit, "inventory_counts": {"columns": len(inventory["columns"]), "beams": len(inventory["beams"]),
                                                          "joints": len(inventory["joints"]), "hinges": len(hinges)},
               "control": {"node": control_node, "dof": dof, "target_displacement_in": target, "du_in": settings.sign * settings.du_in},
               "stop_reason": stop_reason, "failed_step": failed_step, "completed_steps": len(scalars),
               "peak_load_factor": peak_lambda, "peak_base_shear_kip": peak_lambda * settings.reference_base_shear_kip,
               "checks": checks, "events_first": first_events[:200], "event_count": len(first_events),
               "scalars": scalars, "files": {"steps": str(steps_path)}, "elapsed_sec": time.time() - started,
               "sign_conventions": __doc__.split("Sign conventions")[1] if "Sign conventions" in __doc__ else None}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False), encoding="utf-8")
    return summary
