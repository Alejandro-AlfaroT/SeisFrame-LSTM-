"""Two systems in one frame: the perimeter special moment frames (the seismic force-resisting system) and the
interior gravity frames, on the grouped-design engine (user decision 2026-10-16).

The elastic design model has two states (Structure_Parameters.LATERAL_DESIGN_MODEL). "all_frames" is the
continuous frame: every beam framing into every column, the model of the gravity load path and of the
response-history analyses. "perimeter_sfrs" releases both ends of every interior (gravity) beam in bending,
so the lateral load of the design analyses flows to the perimeter frames and the interior columns lean on
the diaphragms: the structure is proportioned for the seismic forces with the members not designated as
part of the SFRS left out of the lateral analysis (ASCE 7-22 12.2.3.3, 12.12.5; ACI 318-19 18.14).

Per strength combination (decision D1 of the plan):

* a gravity-only combination is solved on the continuous model and every member is read from it;
* a seismic combination gives the SFRS members their actions by superposition,

      continuous(gravity) + [released(gravity + E) - released(gravity)],

  the gravity part of the continuous frame (the interior beams' gravity end moments on the SFRS columns
  included) plus the lateral increment of the released frame, which carries the second-order effect of the
  gravity load it was solved with; the gravity members are read from the continuous model under the same
  gravity factors (no E action is attributed to them; their seismic demand is the compatibility run below).
  Every piece is a linear elastic solve, so the member local forces add; the beam span envelopes are
  recovered from the summed end forces and the element loads of the continuous solve
  (Design.SMRF_Beam_Actions.recover_beam_bending, the same equilibrium the saved-record verifier recomputes);
* the period, the accidental-torsion assessment and the drift screen are those of the released model.

Compatibility run (decision D2; ACI 318-19 18.14.2.1, ASCE 7-22 12.12.5): the design displacement
delta_u = Cd delta_e / Ie of the released model's rho = 1 ELF runs is imposed, level by level, on the diaphragm
masters of the continuous model (ux, uy and the diaphragm rotation, so every interior member sees its own
drift, accidental torsion included) together with the gravity load combinations of 5.3 that carry the
vertical ground motion, (1.2 + 0.2 SDS)D + 1.0L and (0.9 - 0.2 SDS)D, each axis, both signs. The induced
moments and shears of the gravity members, compared with their design strengths by the group strength
routines, decide the 18.14.3 branch of every gravity group (Design.Gravity_Frame_Members).

Under the band policy (Model.Member_Groups.POLICY_ID) every function here reduces to the single-model path
of the existing grouped route, which is unchanged.
"""
from __future__ import annotations

import contextlib
import copy
import math

import openseespy.opensees as ops

import Structure_Parameters as sp
from Analysis.Gravity import run_gravity_analysis
from Design import Design_Driver as driver
from Design import Gravity_Frame_Members as gravity_rules
from Design import Group_Checks as checks
from Design.SMRF_Beam_Actions import applied_element_loads, recover_beam_bending
from Model import Member_Groups as mg
from Model.Build_Model import floor_master_node

CONTINUOUS_MODEL = "all_frames"
RELEASED_MODEL = "perimeter_sfrs"
# The imposed-displacement pattern of the compatibility run (gravity is pattern 1, the ELF patterns 21/22).
COMPATIBILITY_SERIES_TAG = 31
COMPATIBILITY_PATTERN_TAG = 31
# ACI 318-19 18.14.2.1: the gravity load combinations of 5.3 that include the vertical ground motion, with E
# replaced by the design displacement: (5.3.1e) (1.2 + 0.2 SDS) D + 1.0 L and (5.3.1g) (0.9 - 0.2 SDS) D. The
# live load factor is kept at 1.0 (the 0.5 reduction of 5.3.3 is not asserted for the archetype's occupancy).
COMPATIBILITY_FAMILIES = (("high_gravity", 1.2, 1.0), ("low_gravity", 0.9, 0.0))
SUPERPOSED_FIELDS = ("centerline_axial_i_kip", "centerline_axial_j_kip", "axial_i_kip", "axial_j_kip")
MODEL_SFRS = "continuous gravity + released lateral increment"
MODEL_GRAVITY = "continuous gravity only"


# ---- which model ------------------------------------------------------------------------------------------------
def two_system(state=None):
    """Whether the installed (or given) grouped design is under the two-system policy."""
    state = mg.active() if state is None else state
    return state is not None and mg.is_system_policy(state.policy_id)


def design_model_for(state=None):
    """The lateral design model of a grouped design: released under the system policy, continuous otherwise."""
    return RELEASED_MODEL if two_system(state) else CONTINUOUS_MODEL


@contextlib.contextmanager
def lateral_model(model):
    """Set Structure_Parameters.LATERAL_DESIGN_MODEL for the solves inside; always restored."""
    if model not in sp.LATERAL_DESIGN_MODELS:
        raise ValueError(f"lateral design model {model!r} is not one of {sp.LATERAL_DESIGN_MODELS}.")
    saved = sp.LATERAL_DESIGN_MODEL
    sp.LATERAL_DESIGN_MODEL = model
    try:
        yield model
    finally:
        sp.LATERAL_DESIGN_MODEL = saved


def member_systems():
    """{member tag (str): system} of every physical member of the current frame."""
    nx, ny = sp.NUM_BAY_X, sp.NUM_BAY_Y
    return {str(m.member_tag): mg.member_system(m.member_type, m.grid_i, m.grid_j, nx, ny) for m in mg.all_members()}


# ---- the lateral solves of the design route --------------------------------------------------------------------
def model_period(state=None):
    """First valid period of the design model (the released frame under the system policy)."""
    with lateral_model(design_model_for(state)):
        return driver._model_period()


def torsion_assessment(period, cfg, strength=None, state=None):
    """ASCE 7-22 12.3.2.1.1 / 12.8.4.3 on the design model (Design_Driver._torsion_assessment)."""
    model = design_model_for(state)
    with lateral_model(model):
        result = driver._torsion_assessment(period, cfg, strength)
    result["lateral_design_model"] = model
    return result


def drift_screen(period, torsion, cfg, state=None):
    """ASCE 7-22 12.8.6 / 12.8.7 drift and stability on the design model (Design_Driver._drift_screen)."""
    model = design_model_for(state)
    with lateral_model(model):
        result = driver._drift_screen(period, torsion, cfg)
    result["assumptions"]["lateral_design_model"] = model
    if model == RELEASED_MODEL:
        result["assumptions"]["model"] += ("; interior (gravity) beams released in bending at both ends, the perimeter "
                                           "frames carry the lateral load (ASCE 7-22 12.12.5; ACI 318-19 18.14)")
    return result


class CombinationSolver:
    """Solves the strength combinations of one candidate; the gravity-only solves that superposition shares are
    cached per (model, dead factor, live factor, live pattern)."""

    def __init__(self, period, torsion=None, state=None):
        self.period, self.torsion = period, torsion
        self.state = mg.active() if state is None else state
        self.two_system = two_system(self.state)
        self.solves = 0
        self._cache = {}
        self._systems = None

    def systems(self):
        if self._systems is None:
            self._systems = member_systems()
        return self._systems

    def _solve(self, combination, model, torsion):
        with lateral_model(model):
            elf = driver._analyze_combination(combination, self.period, torsion)
        self.solves += 1
        return elf

    def _gravity_only(self, combination, model):
        """The combination's gravity part solved on ``model``: captured member actions, the element loads and the
        beam lengths of that solve (what the superposition of the SFRS members needs)."""
        dead, live, pattern = combination["dead"], combination["live"], combination.get("live_pattern", "all")
        key = (model, dead, live, pattern)
        if key not in self._cache:
            gravity = {**combination, "id": f"{model}_gravity_{dead:.4f}_{live:.4f}_{pattern}", "ex": 0.0, "ey": 0.0}
            self._solve(gravity, model, None)
            self._cache[key] = {"members": checks.capture_member_actions(dead), "element_loads": applied_element_loads(),
                                "lengths": beam_lengths()}
        return self._cache[key]

    def analyze(self, combination):
        """One strength combination -> {"elf", "members", "models"}; ``members`` is what
        Design.Group_Checks.capture_member_actions returns, one row per physical member."""
        seismic = bool(combination["ex"] or combination["ey"])
        if not self.two_system:
            elf = self._solve(combination, CONTINUOUS_MODEL, self.torsion)
            return {"elf": elf, "members": checks.capture_member_actions(combination["dead"]), "models": {"all": CONTINUOUS_MODEL}}
        continuous = self._gravity_only(combination, CONTINUOUS_MODEL)
        if not seismic:
            return {"elf": None, "members": copy.deepcopy(continuous["members"]), "models": {"all": CONTINUOUS_MODEL}}
        released_gravity = self._gravity_only(combination, RELEASED_MODEL)
        elf = self._solve(combination, RELEASED_MODEL, self.torsion)
        released = checks.capture_member_actions(combination["dead"])
        members = superpose_by_system(continuous, released_gravity["members"], released, self.systems())
        return {"elf": elf, "members": members, "models": {mg.SYSTEM_SFRS: MODEL_SFRS, mg.SYSTEM_GRAVITY: MODEL_GRAVITY}}


def beam_lengths():
    """{beam tag (str): centerline length (in)} from the live domain."""
    return {str(m.member_tag): math.dist(ops.nodeCoord(m.node_i), ops.nodeCoord(m.node_j))
            for m in mg.all_members() if not m.is_column}


def superpose_by_system(continuous, released_gravity, released, systems):
    """Member actions of a seismic combination by system: gravity members from the continuous gravity solve;
    SFRS members as continuous gravity plus the released frame's lateral increment, with the beam span
    envelopes recovered from the summed end forces and the continuous solve's element loads."""
    members = {}
    for tag, base in continuous["members"].items():
        if systems[tag] == mg.SYSTEM_GRAVITY:
            members[tag] = {**copy.deepcopy(base), "analysis_model": MODEL_GRAVITY}
            continue
        gravity, lateral = released_gravity[tag], released[tag]
        row = {**base, "analysis_model": MODEL_SFRS}
        row["local_force_kip_kipin"] = [c + (e - g) for c, e, g in zip(base["local_force_kip_kipin"], lateral["local_force_kip_kipin"],
                                                                        gravity["local_force_kip_kipin"])]
        for key in SUPERPOSED_FIELDS:
            row[key] = base[key] + (lateral[key] - gravity[key])
        row["joint_face_offsets_in"] = list(base["joint_face_offsets_in"])
        if base["member_type"] != "column":
            loads = continuous["element_loads"].get(int(tag), {"uniform_z_kip_per_in": 0.0, "point_z_loads": []})
            row["span_bending"] = recover_beam_bending(continuous["lengths"][tag], row["local_force_kip_kipin"],
                                                       loads["uniform_z_kip_per_in"], loads["point_z_loads"])
        members[tag] = row
    return members


# ---- deformation compatibility (ACI 318-19 18.14.2.1; ASCE 7-22 12.12.5) --------------------------------------
def design_displacements(period, torsion, cfg, state=None):
    """delta_e at every diaphragm master of the design model under the rho = 1 ELF of each axis (the drift
    screen's runs, accidental torsion as the screen applies it) and delta_u = Cd delta_e / Ie (ASCE 7-22 12.8.6)."""
    risk = cfg.demands.risk_category if cfg is not None else sp.ASCE_RISK_CATEGORY
    ie = sp.seismic_design_basis(risk_category=risk)["importance_factor"]
    cd = float(getattr(sp, "ASCE_CD", 5.5))
    model = design_model_for(state)
    result = {"cd": cd, "importance_factor": ie, "lateral_design_model": model, "axes": {}, "solves": 0,
              "basis": ("ASCE 7-22 12.8.6 delta_u = Cd delta_e / Ie at the rigid-diaphragm master of every level (translations "
                        "and the diaphragm rotation) under the rho = 1 ELF of each axis on the lateral design model, Cd of "
                        "Table 12.2-1 for the special reinforced concrete moment frame")}
    with lateral_model(model):
        for axis in ("x", "y"):
            combination = {"id": f"compatibility_elf_{axis}", "dead": 1.0, "live": 1.0, "ex": float(axis == "x"),
                           "ey": float(axis == "y"), "live_pattern": "all"}
            elf = driver._analyze_combination(combination, period, torsion)
            result["solves"] += 1
            levels = []
            for k in range(1, sp.NUM_FLOOR + 1):
                master = floor_master_node(k)
                elastic = {"ux": ops.nodeDisp(master, 1), "uy": ops.nodeDisp(master, 2), "rz_rad": ops.nodeDisp(master, 6)}
                levels.append({"floor": k, "master_node": master, "elastic_in": elastic,
                               "design_in": {key: cd * value / ie for key, value in elastic.items()}})
            result["axes"][axis] = {"levels": levels, "base_shear_kip": elf["base_shear_kip"],
                                    "roof_design_displacement_in": levels[-1]["design_in"]["u" + axis]}
    return result


def _impose_design_displacements(levels, sign):
    """The imposed-displacement pattern: delta_u of every level at its master, signed, ramped with the gravity load."""
    ops.timeSeries("Linear", COMPATIBILITY_SERIES_TAG)
    ops.pattern("Plain", COMPATIBILITY_PATTERN_TAG, COMPATIBILITY_SERIES_TAG)
    for level in levels:
        design = level["design_in"]
        for dof, value in ((1, design["ux"]), (2, design["uy"]), (6, design["rz_rad"])):
            ops.sp(level["master_node"], dof, sign * value)


def _achieved_displacements(levels, sign):
    rows = []
    for level in levels:
        design = level["design_in"]
        achieved = {"ux": ops.nodeDisp(level["master_node"], 1), "uy": ops.nodeDisp(level["master_node"], 2),
                    "rz_rad": ops.nodeDisp(level["master_node"], 6)}
        target = {key: sign * value for key, value in design.items()}
        rows.append({"floor": level["floor"], "target_in": target, "achieved_in": achieved,
                     "residual_in": {key: achieved[key] - target[key] for key in target}})
    return rows


def compatibility_run(period, torsion, cfg, state=None, displacements=None):
    """The gravity combinations of 18.14.2.1 acting with the design displacement on the continuous frame.

    Returns None under the band policy. Otherwise {"design_displacements", "cases", "solves", "basis"}; every case
    carries the combination fields of a strength combination plus "axis", "sign" and "imposed" (target and
    achieved master displacements), and ``members`` as capture_member_actions reports them, so the group
    strength routines read the induced actions directly.
    """
    state = mg.active() if state is None else state
    if not two_system(state):
        return None
    displacements = design_displacements(period, torsion, cfg, state) if displacements is None else displacements
    cases, solves, sds = [], displacements["solves"], float(sp.ASCE_SDS)
    for family, dead_base, live in COMPATIBILITY_FAMILIES:
        dead = dead_base + (0.2 * sds if dead_base > 1.0 else -0.2 * sds)
        for axis in ("x", "y"):
            levels = displacements["axes"][axis]["levels"]
            for sign in (1.0, -1.0):
                case_id = f"compatibility_{family}_{axis}{'+' if sign > 0 else '-'}"
                combination = {"id": case_id, "family": f"compatibility_{family}", "dead": dead, "live": live,
                               "ex": 0.0, "ey": 0.0, "live_pattern": "all"}
                with lateral_model(CONTINUOUS_MODEL):
                    driver._apply_combination_loads(combination, period, None)
                    _impose_design_displacements(levels, sign)
                    with driver._quiet():
                        run_gravity_analysis()
                    solves += 1
                    cases.append({**combination, "axis": axis, "sign": sign, "analysis_succeeded": True,
                                  "axial_reference": "joint_faces", "imposed": _achieved_displacements(levels, sign),
                                  "analysis_models": {"all": CONTINUOUS_MODEL + " with imposed design displacement"},
                                  "members": checks.capture_member_actions(dead)})
    return {"design_displacements": displacements, "cases": cases, "solves": solves,
            "basis": ("ACI 318-19 18.14.2.1: gravity load combinations of 5.3 including the vertical ground motion, "
                      "(1.2 + 0.2 SDS) D + 1.0 L and (0.9 - 0.2 SDS) D, acting simultaneously with the design displacement "
                      "delta_u imposed at the diaphragm masters of the continuous frame (ASCE 7-22 12.12.5), each axis, "
                      "both signs; every member's induced actions are read as signed concurrent centerline actions")}


def gravity_group_induced(compatibility, cfg, state=None):
    """The induced-to-design-strength ratios and the 18.14.3 branch of every gravity group.

    The compatibility cases go through the group strength routines exactly as strength combinations do, so a
    column's moment ratio is its P-M interaction ratio at the concurrent induced axial load, a beam's the
    larger of its sagging and hogging ratios, and the shear ratio Vu / phi Vn; a group's ratio is its worst
    member and case, so its branch is the worst branch of its members (18.14.3.1).
    """
    state = mg.active() if state is None else state
    if compatibility is None or state is None:
        return {}
    strengths = checks.strength_checks(compatibility["cases"], cfg, state)
    result = {}
    for gid, row in strengths["groups"].items():
        if state.system_of(gid) != mg.SYSTEM_GRAVITY:
            continue
        if row["member_type"] == "column":
            moment_ratio, shear_ratio = row["pm"]["dcr"], row["shear"]["dcr"]
            governing = {"pm": {k: row["pm"].get(k) for k in ("member_tag", "combination", "axial_kip", "my_kip_in", "mz_kip_in",
                                                               "phi_mn_y_kip_in", "phi_mn_z_kip_in", "dcr", "basis")},
                         "shear": {k: row["shear"].get(k) for k in ("member_tag", "combination", "vu_kip", "phi_vn_kip", "dcr")}}
        else:
            positive, negative = row["flexure_positive"], row["flexure_negative"]
            moment_ratio, shear_ratio = max(positive["dcr"], negative["dcr"]), row["shear"]["dcr"]
            governing = {"flexure_positive": positive, "flexure_negative": negative,
                         "shear": {k: row["shear"].get(k) for k in ("member_tag", "combination", "vu_kip", "phi_vn_kip", "dcr")}}
        branch = gravity_rules.branch_from_ratios(moment_ratio, shear_ratio)
        result[gid] = {"member_type": row["member_type"], "tie": state.tie_of(gid), "moment_ratio": moment_ratio,
                       "shear_ratio": shear_ratio, "branch": branch["branch"], "reason": branch["reason"], "governing": governing,
                       "cases": len(compatibility["cases"])}
    return result


def gravity_frame_evidence(compatibility, cfg, state, beams, columns, actions):
    """ACI 318-19 18.14 evidence of every gravity group: its branch from the compatibility run and the checks of
    that branch on the hoops the capacity design selected (``beams``/``columns`` of
    Design.Group_Capacity.build_group_capacity_design). Without a compatibility run the effects of delta_u are
    not calculated and every gravity group is detailed to 18.14.3.3 (18.14.3.1)."""
    induced = gravity_group_induced(compatibility, cfg, state)
    groups, checks_out = {}, []
    for gid, group in sorted(state.groups.items()):
        if state.system_of(gid) != mg.SYSTEM_GRAVITY:
            continue
        design, tags = state.designs[gid], group["member_tags"]
        row = induced.get(gid)
        branch_input = ({"moment_ratio": row["moment_ratio"], "shear_ratio": row["shear_ratio"]} if row is not None
                        else {"moment_ratio": None, "shear_ratio": None, "calculated": False})
        db = sp.rebar_diameter(design.bar_size)
        if group["member_type"] == "column":
            data = columns[gid]
            hoops, confinement, governing = data.get("hoops") or {}, data.get("confinement") or {}, data.get("governing") or {}
            section = {"ag_in2": design.gross_area_in2, "ast_in2": design.longitudinal_area_in2, "fc_ksi": design.fc_ksi,
                       "fy_ksi": sp.FY_KSI, "pu_kip": max(governing.get("axial_max_kip", 0.0) or 0.0, 0.0)}
            detailing = {"hoop_spacing_in": hoops.get("spacing_in", design.stirrup_spacing_in), "smallest_bar_db_in": db,
                         "ash_ratio_provided": (hoops["ash_provided_per_in"] / confinement["bc_in"]
                                                if hoops.get("ash_provided_per_in") and confinement.get("bc_in") else None),
                         "ash_ratio_table_18_7_5_4": confinement.get("ash_ratio_required"),
                         "configuration_18_7_5_2_ok": bool(hoops) and bool(confinement.get("hx_within_8in_when_required"))}
            strengths = {"ve_kip": governing.get("ve_kip"), "phi_vn_capacity_kip": hoops.get("phi_vn_kip")}
            checks_of = gravity_rules.evaluate_gravity_column(gid, branch_input, strengths, section, detailing)
        else:
            data = beams[gid]
            hoops = data.get("hoops") or {}
            ab = sp.rebar_area(design.bar_size)
            pu = max((abs(float(a["members"][str(tag)]["local_force_kip_kipin"][0])) for a in actions for tag in tags), default=0.0)
            section = {"b_in": design.b_in, "d_in": data["effective_depth_in"], "ag_in2": design.gross_area_in2,
                       "fc_ksi": design.fc_ksi, "fy_ksi": sp.FY_KSI, "as_top_in2": design.top_bars * ab,
                       "as_bot_in2": design.bot_bars * ab, "pu_kip": pu}
            detailing = {"stirrup_spacing_in": hoops.get("spacing_in", design.stirrup_spacing_in), "smallest_bar_db_in": db,
                         "hoops": bool(hoops)}
            strengths = {"ve_kip": max((m["governing_ve_kip"] for m in data["members"]), default=None),
                         "phi_vn_capacity_kip": hoops.get("phi_vn_kip")}
            checks_of = gravity_rules.evaluate_gravity_beam(gid, branch_input, strengths, section, detailing)
        branch = next((c["details"] for c in checks_of if c["id"].endswith("compatibility_branch") and c.get("details")), {})
        groups[gid] = {"member_type": group["member_type"], "tie": state.tie_of(gid), "system": mg.SYSTEM_GRAVITY,
                       "induced": row if row is not None else {"member_type": group["member_type"], "tie": state.tie_of(gid),
                                                               "moment_ratio": None, "shear_ratio": None,
                                                               "branch": gravity_rules.BRANCH_YIELDING,
                                                               "reason": "effects of the design displacement not calculated (18.14.3.1)",
                                                               "governing": None, "cases": 0},
                       "branch": branch.get("branch"), "requirements": branch.get("requirements"), "checks": checks_of,
                       "section": section, "detailing": detailing, "strengths": strengths}
        checks_out.extend(checks_of)
    return {"groups": groups, "checks": checks_out, "compatibility_calculated": compatibility is not None,
            "basis": ("ACI 318-19 18.14: each gravity group's branch (18.14.3.2 within the design strengths, 18.14.3.3 otherwise) "
                      "from the induced actions of the compatibility run through the group strength routines; the checks of "
                      "that branch are read on the hoops the capacity design selected (the special-frame hoops, kept for the "
                      "gravity members in this revision)")}


def gravity_induced_from_capacity(capacity):
    """{gravity group id: induced row} as the capacity design recorded them (empty under the band policy)."""
    gravity = (capacity or {}).get("gravity_frame") or {}
    return {gid: row["induced"] for gid, row in gravity.get("groups", {}).items()}


def compatibility_stage(compatibility, induced):
    """The JSON-safe stage summary of the compatibility run for the evaluation record."""
    if compatibility is None:
        return {"evaluated": False, "basis": "band policy: every member is part of the seismic force-resisting system"}
    displacements = compatibility["design_displacements"]
    return {"evaluated": True, "cases": len(compatibility["cases"]), "solves": compatibility["solves"],
            "cd": displacements["cd"], "importance_factor": displacements["importance_factor"],
            "roof_design_displacement_in": {axis: displacements["axes"][axis]["roof_design_displacement_in"] for axis in ("x", "y")},
            "largest_residual_in": max((abs(v) for case in compatibility["cases"] for level in case["imposed"]
                                        for v in level["residual_in"].values()), default=0.0),
            "groups": {gid: {"tie": row["tie"], "member_type": row["member_type"], "branch": row["branch"],
                             "moment_ratio": row["moment_ratio"], "shear_ratio": row["shear_ratio"]}
                       for gid, row in induced.items()}}
