"""The slab as the seismic diaphragm: ACI 318-19 18.12 on the ASCE 7-22 12.10.1.1 diaphragm forces (2026-10-14).

The slab design of this project (SMRF_Slab, SMRF_Slab_Reinforcement) sizes the two-way slab and its mats
for gravity under Chapter 8 and feeds the mats into the beam strengths and the column-strength ratio of
Chapter 18. What it did not do before this module is treat the slab as the diaphragm of a special moment
frame building in SDC C-D, which Chapter 18.12 requires of cast-in-place slabs in SDC D-F (18.12.1.1) and
which the research population applies in SDC C as well. This module adds, as explicit checks that can
fail a design:

* The diaphragm design force at every level, Fpx of ASCE 7-22 Eq. (12.10-1) with the floor of
  0.2 SDS Ie wpx and the ceiling of 0.4 SDS Ie wpx (12.10.1.1), from the ELF story forces of the final
  frame; rho = 1.0 for these forces (12.3.4.1). With a moment frame on every grid line there are no
  transfer forces and no collectors (12.10.2 does not call for any), which the record states.
* The in-plane shear strength (18.12.9.1): Vn = Acv (2 lambda sqrt(f'c) + rho_t fy), not more than
  8 Acv sqrt(f'c) (18.12.9.2), with phi = 0.75 (21.2.4.2, the shear phi of the columns). The demand is
  the upper bound of a diaphragm spanning between the two exterior frame lines parallel to the force,
  Vu = Fpx / 2 on the section through the slab parallel to the force (Acv = h times the plan dimension
  parallel to the force); interior frame lines carry their share directly and only lower the shear, so
  the bound is conservative and is declared as the basis. rho_t is the ratio of the distributed mats
  parallel to the force (both faces).
* Chord reinforcement for the diaphragm flexure of the same spanning model (18.12.8, 18.12.7.3): the
  chord force Mu / d with Mu = Fpx L / 8 and d the distance between chord centroids, carried by chord
  bars in the slab beside the perimeter beam, developed and spliced for fy (Class B laps of 1.3 ld,
  25.4.2.4 for the slab-bar conditions), with the compressive-stress screen of 18.12.7.5 (0.2 f'c, above
  which confining transverse reinforcement would be required and is not provided here).
* The minimum distributed reinforcement of 18.12.7.1 (24.4.3.2: 0.0018 for Grade 60) and the 18 in
  spacing limit of 18.12.7.1 on the slab mats, the 2 in minimum thickness of 18.12.6.1.

Everything is computed from the saved inputs (the final ELF story forces and weights, the slab, the mats,
the plan) and is audited by exact recomputation at qualification (evaluate_slab_diaphragm), like the slab
strip evidence. The diaphragm is a check of the selected design, not a lever of the search: a failure is
reported as a failed check. Units: kip, inch, ksi.
"""
from __future__ import annotations

import copy
import json
import math

from Design.SMRF_Common import make_check, not_evaluated, summarize_checks

METHOD_VERSION = "aci318_19_18_12_cast_in_place_slab_diaphragm_v1"
_BARS = {4: (0.5, 0.20), 5: (0.625, 0.31), 6: (0.75, 0.44)}
MIN_THICKNESS_IN = 2.0                      # 18.12.6.1
MIN_REINFORCEMENT_RATIO = 0.0018            # 18.12.7.1 via 24.4.3.2, Grade 60
MAX_BAR_SPACING_IN = 18.0                   # 18.12.7.1
PHI_SHEAR = 0.75                            # 21.2.4.2: not more than the shear phi of the vertical elements
PHI_TENSION = 0.90                          # 21.2.2, tension-controlled chord
COMPRESSION_CONFINEMENT_FRACTION = 0.2      # 18.12.7.5
DEFAULT_POLICY = {"chord_bar_sizes": [4, 5, 6], "chord_strip_width_factor": 2.0, "chord_bars_minimum": 2,
                  "chord_bar_pitch_min_factor": 3.0, "chord_bar_pitch_min_in": 2.0,
                  "span_model": "exterior_frame_lines"}


# ---- inputs -----------------------------------------------------------------------------------------------
def diaphragm_inputs_from_state(elf, risk_category=None, policy=None):
    """The diaphragm design inputs of the installed state: plan, slab and mats, hazard, the ELF story forces.

    ``elf`` is Loads.Seismic_ELF.elf_story_forces(final period) of the selected frame.
    """
    import Structure_Parameters as sp
    if sp.SLAB_THICKNESS_IN is None:
        raise ValueError("A diaphragm design needs the slab-aware model (no slab thickness is selected).")
    layout = (sp.SLAB_REINFORCEMENT or {}).get("layout")
    if not layout:
        raise ValueError("A diaphragm design needs the selected slab mats (no slab layout is installed).")
    basis = sp.seismic_design_basis(risk_category=risk_category)
    slab_inputs = (sp.SLAB_REINFORCEMENT.get("inputs") or {}).get("slab") or {}
    fc = float(slab_inputs.get("fc_ksi", sp.FC_BEAM_KSI))
    layers = {}
    for name in ("x_top", "x_bottom", "y_top", "y_bottom"):
        layer = layout["layers"][name]
        layers[name] = {"bar_size": int(layer["bar_size"]), "bar_diameter_in": float(layer["bar_diameter_in"]),
                        "spacing_in": float(layer["spacing_in"]), "area_in2_per_ft": float(layer["area_in2_per_ft"]),
                        "axis": layer["axis"], "face": layer.get("face", name.split("_")[1])}
    return {
        "code": "ACI 318-19 18.12 with ASCE 7-22 12.10.1.1 diaphragm forces",
        "geometry": {"num_bay_x": int(sp.NUM_BAY_X), "num_bay_y": int(sp.NUM_BAY_Y), "num_floor": int(sp.NUM_FLOOR),
                     "bay_x_in": float(sp.BAY_X), "bay_y_in": float(sp.BAY_Y), "story_h_in": float(sp.STORY_H)},
        "sections": {"b_beam_in": float(sp.B_BEAM), "h_beam_in": float(sp.H_BEAM),
                     "b_col_in": float(sp.B_COL), "h_col_in": float(sp.H_COL)},
        "slab": {"thickness_in": float(sp.SLAB_THICKNESS_IN), "fc_ksi": fc, "fy_ksi": float(sp.FY_KSI),
                 "max_aggregate_size_in": float(sp.AGGREGATE_MAX_SIZE_IN),
                 "normalweight": sp.CONCRETE_UNIT_WEIGHT_KCF == 0.150, "layers": layers},
        "seismic": {"sds": float(sp.ASCE_SDS), "importance_factor": float(basis["importance_factor"]),
                    "risk_category": basis["risk_category"], "r": float(sp.ASCE_R),
                    "site_label": getattr(sp, "SEISMIC_SITE_LABEL", None)},
        "elf": {"design_period_sec": float(elf["design_period_sec"]), "base_shear_kip": float(elf["base_shear_kip"]),
                "story_forces_kip": [float(v) for v in elf["story_forces_kip"]],
                "story_weights_kip": [float(v) for v in elf["story_weights_kip"]],
                "story_heights_ft": [float(v) for v in elf["story_heights_ft"]],
                "vertical_distribution_k": float(elf["vertical_distribution_k"]), "cs": float(elf["cs"]),
                "basis": "ASCE 7-22 12.8 ELF story forces of the final model period (Loads.Seismic_ELF)"},
        "policy": {**DEFAULT_POLICY, **(policy or {})},
    }


def _number(value, name, positive=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive.")
    return float(value)


def _validate(inputs):
    if not isinstance(inputs, dict):
        raise ValueError("Diaphragm inputs are required.")
    g, s, slab, seismic, elf = (inputs.get(key) for key in ("geometry", "sections", "slab", "seismic", "elf"))
    for block, name in ((g, "geometry"), (s, "sections"), (slab, "slab"), (seismic, "seismic"), (elf, "elf")):
        if not isinstance(block, dict):
            raise ValueError(f"Diaphragm inputs lack {name}.")
    for key in ("num_bay_x", "num_bay_y", "num_floor"):
        if type(g.get(key)) is not int or g[key] < 1:
            raise ValueError(f"geometry.{key} must be a positive integer.")
    for key in ("bay_x_in", "bay_y_in", "story_h_in"):
        _number(g.get(key), f"geometry.{key}")
    for key in ("b_beam_in", "h_beam_in", "b_col_in", "h_col_in"):
        _number(s.get(key), f"sections.{key}")
    for key in ("thickness_in", "fc_ksi", "fy_ksi", "max_aggregate_size_in"):
        _number(slab.get(key), f"slab.{key}")
    if slab.get("normalweight") is not True:
        raise ValueError("Only normalweight slab concrete is in scope (lambda = 1.0).")
    layers = slab.get("layers")
    if not isinstance(layers, dict) or set(layers) != {"x_top", "x_bottom", "y_top", "y_bottom"}:
        raise ValueError("slab.layers must hold the four mats x_top, x_bottom, y_top, y_bottom.")
    for name, layer in layers.items():
        if int(layer.get("bar_size", 0)) not in _BARS:
            raise ValueError(f"slab.layers.{name}.bar_size must be 4, 5 or 6.")
        for key in ("bar_diameter_in", "spacing_in", "area_in2_per_ft"):
            _number(layer.get(key), f"slab.layers.{name}.{key}")
        if layer.get("axis") != name.split("_")[0]:
            raise ValueError(f"slab.layers.{name}.axis must be {name.split('_')[0]!r}.")
    _number(seismic.get("sds"), "seismic.sds")
    _number(seismic.get("importance_factor"), "seismic.importance_factor")
    forces, weights, heights = (elf.get(key) for key in ("story_forces_kip", "story_weights_kip", "story_heights_ft"))
    n = g["num_floor"]
    for name, values in (("story_forces_kip", forces), ("story_weights_kip", weights), ("story_heights_ft", heights)):
        if not isinstance(values, list) or len(values) != n:
            raise ValueError(f"elf.{name} must list one value per elevated floor ({n}).")
        for k, v in enumerate(values):
            _number(v, f"elf.{name}[{k}]", positive=(name != "story_forces_kip"))
    if any(v < 0 for v in forces):
        raise ValueError("ELF story forces must be nonnegative.")
    base = _number(elf.get("base_shear_kip"), "elf.base_shear_kip")
    if not math.isclose(base, sum(forces), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("The ELF story forces do not sum to the base shear.")
    policy = {**DEFAULT_POLICY, **(inputs.get("policy") or {})}
    if policy["span_model"] != "exterior_frame_lines":
        raise ValueError(f"Unknown diaphragm span model {policy['span_model']!r}.")
    if any(int(b) not in _BARS for b in policy["chord_bar_sizes"]) or not policy["chord_bar_sizes"]:
        raise ValueError("chord_bar_sizes must be a nonempty subset of 4, 5, 6.")
    return policy


# ---- forces -----------------------------------------------------------------------------------------------
def diaphragm_forces(inputs):
    """ASCE 7-22 12.10.1.1: Fpx at every level from the ELF story forces and weights, with its floor and ceiling."""
    seismic, elf = inputs["seismic"], inputs["elf"]
    sds, ie = float(seismic["sds"]), float(seismic["importance_factor"])
    forces, weights = elf["story_forces_kip"], elf["story_weights_kip"]
    n = len(weights)
    levels = []
    for x in range(n):
        sum_f, sum_w = sum(forces[x:]), sum(weights[x:])
        wpx = float(weights[x])
        unbounded = sum_f / sum_w * wpx
        low, high = 0.2 * sds * ie * wpx, 0.4 * sds * ie * wpx
        fpx = min(max(unbounded, low), high)
        bound = "minimum 0.2 SDS Ie wpx" if unbounded < low else ("maximum 0.4 SDS Ie wpx" if unbounded > high else "Eq. (12.10-1)")
        levels.append({"level": x + 1, "wpx_kip": wpx, "sum_forces_above_kip": sum_f, "sum_weights_above_kip": sum_w,
                       "fpx_eq_12_10_1_kip": unbounded, "fpx_min_kip": low, "fpx_max_kip": high, "fpx_kip": fpx,
                       "governing": bound})
    governing = max(levels, key=lambda lv: (lv["fpx_kip"], lv["level"]))
    return {"levels": levels, "governing_level": governing["level"], "fpx_governing_kip": governing["fpx_kip"],
            "redundancy_factor": 1.0,
            "basis": ("Fpx = (sum Fi / sum wi) wpx over the levels x to n, within 0.2 SDS Ie wpx and 0.4 SDS Ie wpx "
                      "(ASCE 7-22 12.10.1.1); rho = 1.0 for diaphragm forces from Eq. (12.10-1) (12.3.4.1); no transfer "
                      "forces: the vertical elements are continuous and identical at every level")}


# ---- strength ---------------------------------------------------------------------------------------------
def _development_length_in(bar, fc_ksi, fy_ksi):
    """25.4.2.4 straight development of a #6 or smaller slab bar (SMRF_Slab_Reinforcement.development_length_in)."""
    from Design.SMRF_Slab_Reinforcement import development_length_in
    return development_length_in(int(bar), fc_ksi, fy_ksi)


def _direction(inputs, axis, fpx_kip, policy):
    """Shear and chord design of the diaphragm for a force along ``axis``."""
    g, s, slab = inputs["geometry"], inputs["sections"], inputs["slab"]
    h, fc, fy = float(slab["thickness_in"]), float(slab["fc_ksi"]), float(slab["fy_ksi"])
    lx, ly = g["num_bay_x"] * float(g["bay_x_in"]), g["num_bay_y"] * float(g["bay_y_in"])
    # The diaphragm spans between the two exterior frame lines parallel to the force; its depth in plan is
    # the dimension parallel to the force, its chords run perpendicular to the force at the far edges.
    parallel, span = (lx, ly) if axis == "x" else (ly, lx)
    chord_axis = "y" if axis == "x" else "x"
    perimeter_beam_width = float(s["b_beam_in"])
    bay_along_chord = float(g["bay_y_in"]) if axis == "x" else float(g["bay_x_in"])
    # 18.12.9.1 shear: rho_t = mats parallel to the force, both faces, over the gross section per unit width.
    mats = [slab["layers"][f"{axis}_top"], slab["layers"][f"{axis}_bottom"]]
    as_parallel = sum(float(m["area_in2_per_ft"]) for m in mats)
    rho_t = as_parallel / (12.0 * h)
    acv = h * parallel
    sqrt_fc_psi = math.sqrt(fc * 1000.0)
    vn_kip = acv * (2.0 * 1.0 * sqrt_fc_psi + rho_t * fy * 1000.0) / 1000.0
    vn_cap_kip = 8.0 * acv * sqrt_fc_psi / 1000.0
    vn_used = min(vn_kip, vn_cap_kip)
    vu_kip = fpx_kip / 2.0
    # Chord: Mu = Fpx L / 8 of the simply supported span, resisted by chord bars in the slab beside the
    # perimeter beam (a strip of chord_strip_width_factor times the thickness, inside the beam web).
    mu_kip_in = fpx_kip * span / 8.0
    strip = policy["chord_strip_width_factor"] * h
    edge_to_centroid = perimeter_beam_width / 2.0 + strip / 2.0
    d_chord = parallel - 2.0 * edge_to_centroid
    if d_chord <= 0.0:
        raise ValueError("The chord strips do not fit inside the plan.")
    tu_kip = mu_kip_in / d_chord
    as_required = tu_kip / (PHI_TENSION * fy)
    chord = None
    trials = []
    for bar in policy["chord_bar_sizes"]:
        db, ab = _BARS[int(bar)]
        count = max(int(policy["chord_bars_minimum"]), math.ceil(as_required / ab - 1e-9))
        pitch_min = max(policy["chord_bar_pitch_min_factor"] * db, policy["chord_bar_pitch_min_in"])
        available = strip - 2.0 * (0.75 + db / 2.0)             # 0.75 in slab cover at the strip edges
        fits = count == 1 or (count - 1) * pitch_min <= available + 1e-9
        pitch = available / (count - 1) if count > 1 else None
        trials.append({"bar_size": int(bar), "count": count, "fits": fits, "pitch_in": pitch})
        if fits:
            ld = _development_length_in(bar, fc, fy)
            chord = {"runs_along": chord_axis, "at_edges": f"{axis} = 0 and {axis} = {parallel:g} in",
                     "placement": (f"in the slab at mid-depth, in a {strip:g} in strip beside the web of the {chord_axis} "
                                   "perimeter beam, continuous along the edge"),
                     "bar_size": int(bar), "count": count, "area_provided_in2": count * ab,
                     "pitch_in": pitch, "pitch_min_in": pitch_min, "strip_width_in": strip,
                     "class_b_lap_in": 1.3 * ld, "development_length_in": ld,
                     "bay_along_chord_in": bay_along_chord,
                     "compressive_stress_ksi": tu_kip / (h * strip),
                     "compressive_stress_limit_ksi": COMPRESSION_CONFINEMENT_FRACTION * fc}
            break
    return {"axis": axis, "fpx_kip": fpx_kip, "plan_parallel_in": parallel, "span_between_exterior_lines_in": span,
            "shear": {"vu_kip": vu_kip, "acv_in2": acv, "rho_t": rho_t, "mats_parallel_in2_per_ft": as_parallel,
                      "vn_18_12_9_1_kip": vn_kip, "vn_cap_18_12_9_2_kip": vn_cap_kip, "vn_used_kip": vn_used,
                      "phi": PHI_SHEAR, "phi_vn_kip": PHI_SHEAR * vn_used,
                      "dcr": vu_kip / (PHI_SHEAR * vn_used) if vn_used > 0 else float("inf"),
                      "basis": ("Vu = Fpx / 2: the diaphragm spanning between the two exterior frame lines parallel to the "
                                "force (upper bound; interior frame lines only lower it); Acv = h times the plan dimension "
                                "parallel to the force; rho_t = both mats parallel to the force")},
            "chord": {"mu_kip_in": mu_kip_in, "lever_arm_in": d_chord, "edge_to_chord_centroid_in": edge_to_centroid,
                      "tu_kip": tu_kip, "phi": PHI_TENSION, "area_required_in2": as_required, "trials": trials,
                      "selected": chord,
                      "basis": "Mu = Fpx L / 8 of the same span; chord force Mu / d, d between the chord centroids"}}


def design_diaphragm(inputs, policy=None):
    """The diaphragm record of one design: forces, both directions, checks and their summary."""
    record = {"method_version": METHOD_VERSION, "stage": "diaphragm_checks_of_the_selected_design",
              "screen_passed": False, "forces": None, "directions": None}
    try:
        saved = json.loads(json.dumps(inputs, allow_nan=False))
        if policy:
            saved["policy"] = {**DEFAULT_POLICY, **(saved.get("policy") or {}), **policy}
        resolved = _validate(saved)
        record["inputs"] = saved
    except (ValueError, TypeError, OverflowError) as exc:
        checks = [not_evaluated("diaphragm.inputs", "ACI 318-19 18.12 / ASCE 7-22 12.10", str(exc))]
        return {**record, "inputs": copy.deepcopy(inputs) if isinstance(inputs, dict) else None,
                "checks": checks, "summary": summarize_checks(checks)}
    g, slab = saved["geometry"], saved["slab"]
    h, fy = float(slab["thickness_in"]), float(slab["fy_ksi"])
    forces = diaphragm_forces(saved)
    fpx = forces["fpx_governing_kip"]
    directions = {axis: _direction(saved, axis, fpx, resolved) for axis in ("x", "y")}
    record["forces"], record["directions"] = forces, directions
    checks = [
        make_check("diaphragm.minimum_thickness", "ACI 318-19 18.12.6.1", h, MIN_THICKNESS_IN, ">=", "in"),
        make_check("diaphragm.design_force_basis", "ASCE 7-22 12.10.1.1 (Eq. 12.10-1 with its floor and ceiling); 12.3.4.1 rho = 1.0",
                   1, 1, "==", details={"governing_level": forces["governing_level"], "fpx_governing_kip": fpx,
                                        "levels": [{k: v for k, v in lv.items()} for lv in forces["levels"]]}),
        make_check("diaphragm.collector_scope", "ASCE 7-22 12.10.2 / ACI 318-19 18.12.7.5", 1, 1, "==",
                   details={"collectors": "none: a special moment frame on every grid line receives its tributary inertia "
                                         "directly; no transfer forces (identical vertical elements at every level)"}),
    ]
    for axis in ("x", "y"):
        mats = [slab["layers"][f"{axis}_top"], slab["layers"][f"{axis}_bottom"]]
        ratio = sum(float(m["area_in2_per_ft"]) for m in mats) / (12.0 * h)
        checks.append(make_check("diaphragm.minimum_reinforcement", "ACI 318-19 18.12.7.1 via 24.4.3.2 (Grade 60)",
                                 ratio, MIN_REINFORCEMENT_RATIO, ">=", "ratio", location=f"mats along {axis}"))
        for face in ("top", "bottom"):
            checks.append(make_check("diaphragm.maximum_bar_spacing", "ACI 318-19 18.12.7.1",
                                     float(slab["layers"][f"{axis}_{face}"]["spacing_in"]), MAX_BAR_SPACING_IN, "<=", "in",
                                     location=f"{axis}_{face}"))
    for axis, d in directions.items():
        sh, ch = d["shear"], d["chord"]
        checks.append(make_check("diaphragm.in_plane_shear", "ACI 318-19 18.12.9.1, 18.12.9.2; 21.2.4.2 phi = 0.75",
                                 sh["vu_kip"], sh["phi_vn_kip"], "<=", "kip", location=f"force along {axis}",
                                 details={"acv_in2": sh["acv_in2"], "rho_t": sh["rho_t"], "vn_used_kip": sh["vn_used_kip"],
                                          "vn_cap_kip": sh["vn_cap_18_12_9_2_kip"], "basis": sh["basis"]}))
        selected = ch["selected"]
        if selected is None:
            checks.append(make_check("diaphragm.chord_reinforcement", "ACI 318-19 18.12.8 / 18.12.7.3", ch["area_required_in2"], 0.0,
                                     "<=", "in2", location=f"force along {axis}",
                                     details={"reason": "no chord bar size of the policy fits the chord strip", "trials": ch["trials"]}))
            continue
        loc = f"force along {axis}, chords along {selected['runs_along']}"
        checks.append(make_check("diaphragm.chord_reinforcement", "ACI 318-19 18.12.8 / 18.12.7.3; 21.2.2 phi = 0.9",
                                 ch["area_required_in2"], selected["area_provided_in2"], "<=", "in2", location=loc,
                                 details={"tu_kip": ch["tu_kip"], "bars": f"{selected['count']} #{selected['bar_size']}",
                                          "placement": selected["placement"]}))
        if selected["count"] > 1:
            checks.append(make_check("diaphragm.chord_bar_spacing", "ACI 318-19 25.2.1 (policy: at least 3 db, 2 in)",
                                     selected["pitch_in"], selected["pitch_min_in"], ">=", "in", location=loc))
        checks.append(make_check("diaphragm.chord_development", "ACI 318-19 18.12.7.3; 25.5.2.1 Class B lap of 25.4.2.4 ld",
                                 selected["class_b_lap_in"], selected["bay_along_chord_in"], "<=", "in", location=loc,
                                 details={"ld_in": selected["development_length_in"],
                                          "basis": "continuous chord bars spliced for fy; one Class B lap fits within a bay"}))
        checks.append(make_check("diaphragm.chord_compression", "ACI 318-19 18.12.7.5 (0.2 f'c screen, applied to the chord)",
                                 selected["compressive_stress_ksi"], selected["compressive_stress_limit_ksi"], "<=", "ksi", location=loc,
                                 details={"basis": "chord compression over the chord strip; above the screen confining hoops "
                                                   "per 18.7.5.2 would be required and are not provided"}))
    record["screen_passed"] = all(c["status"] == "pass" for c in checks)
    return {**record, "checks": checks, "summary": summarize_checks(checks)}


# ---- qualification ----------------------------------------------------------------------------------------
def _input_consistency_problems(record, saved):
    """Where the saved diaphragm inputs disagree with the record they sit in."""
    problems = []
    g, slab_rec = record.get("geometry") or {}, record.get("slab") or {}
    inputs_g = saved.get("geometry") or {}
    for key in ("num_bay_x", "num_bay_y", "num_floor", "bay_x_in", "bay_y_in", "story_h_in"):
        if inputs_g.get(key) != g.get(key):
            problems.append(f"geometry.{key}")
    sections = record.get("sections") or {}
    for key in ("b_beam_in", "h_beam_in", "b_col_in", "h_col_in"):
        if (saved.get("sections") or {}).get(key) != sections.get(key):
            problems.append(f"sections.{key}")
    slab_in = saved.get("slab") or {}
    if slab_in.get("thickness_in") != slab_rec.get("thickness_in"):
        problems.append("slab.thickness_in")
    layout = ((record.get("slab_reinforcement") or {}).get("layout") or {}).get("layers") or {}
    for name in ("x_top", "x_bottom", "y_top", "y_bottom"):
        live, mine = layout.get(name) or {}, (slab_in.get("layers") or {}).get(name) or {}
        for key in ("bar_size", "spacing_in", "area_in2_per_ft"):
            if live.get(key) != mine.get(key):
                problems.append(f"slab.layers.{name}.{key}")
    seismic = record.get("seismic") or {}
    for key in ("sds", "importance_factor"):
        if (saved.get("seismic") or {}).get(key) != seismic.get(key):
            problems.append(f"seismic.{key}")
    elf = saved.get("elf") or {}
    base = (record.get("demand") or {}).get("base_shear_kip")
    if base is None or not math.isclose(float(base), float(elf.get("base_shear_kip", float("nan"))), rel_tol=1e-9, abs_tol=1e-9):
        problems.append("elf.base_shear_kip vs demand.base_shear_kip")
    loads = record.get("floor_loads") or {}
    per_floor, roof_extra = loads.get("total_floor_seismic_weight_kip"), loads.get("roof_column_extension_seismic_weight_kip", 0.0)
    weights = elf.get("story_weights_kip") or []
    if per_floor is not None and weights:
        expected = [float(per_floor)] * len(weights)
        expected[-1] += float(roof_extra or 0.0)
        if any(not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) for a, b in zip(weights, expected)):
            problems.append("elf.story_weights_kip vs floor_loads")
    return problems


def evaluate_slab_diaphragm(record):
    """Qualification: the saved diaphragm design reproduces from its inputs, its inputs are the record's, and its checks."""
    layout = (record.get("slab_reinforcement") or {}).get("layout")
    if record.get("slab") is None or not layout:
        return [not_evaluated("diaphragm.scope", "ACI 318-19 18.12", "no slab layout: the diaphragm is not designed on this route")]
    saved = record.get("slab_diaphragm")
    if not isinstance(saved, dict) or not isinstance(saved.get("inputs"), dict):
        return [not_evaluated("diaphragm.saved_evidence", "ACI 318-19 18.12 / ASCE 7-22 12.10.1.1",
                              "the record carries no diaphragm design (designed before the 2026-10-14 rule)")]
    expected = design_diaphragm(saved["inputs"])
    match = saved == expected
    problems = _input_consistency_problems(record, saved["inputs"])
    return [make_check("diaphragm.saved_evidence", "Deterministic diaphragm evidence audit", int(match), 1, "=="),
            make_check("diaphragm.frame_input_consistency", "Diaphragm inputs / the record's frame, slab, hazard and ELF",
                       int(not problems), 1, "==", details={"disagreements": problems}),
            *expected["checks"]]
