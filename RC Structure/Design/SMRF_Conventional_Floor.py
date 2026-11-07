"""Floor actions, torsion and diaphragm checks of the frame-plus-transfer production route (2026-11-07).

The frame receives the floor loads from a separate plate analysis, so the actions the plate carries itself have to be
brought back to the beams and checked here:

  slab moment in the flange   the T-beam section counts the developed mats in its strength, so its demand is the frame
                              moment plus the slab moment carried across the same flange (each bar's force is counted once,
                              in one section equilibrium). Slab moments are the envelope of the slab cases, conservative
                              for concurrency.
  torsion                     end torque = frame torque + the plate's edge couples on the span (9.5.4, 22.7). Below
                              phi Tth it is neglected (22.7.1.1). Above it the beam gets closed hoops and longitudinal
                              steel for Tu (22.7.6, 9.6.4, 9.7.5, 9.7.6.3). Tu above phi Tcr may drop to phi Tcr
                              (22.7.3.2) where the adjoining slab can carry the panel with that edge restraint released
                              (22.7.3.3), shown by a lower-bound strip check on the installed mats.
  diaphragm                   18.12 and ASCE 7-22 12.10: Fpx by level, concrete in-plane shear, in-plane flexure on the mats
                              outside the edge-beam flanges with the slab's own use of those mats reserved, shear transfer
                              into the frame lines, and the collector axial force in the frame-line beams.
  slab acceptance             the plate strip demands are accepted on computed conditions: complete floors, balanced cases,
                              valid face recovery and a passed mesh refinement.

Units: kip, inch, ksi.
"""
from __future__ import annotations

import math

import Structure_Parameters as sp
from Design.SMRF_Common import make_check, not_evaluated

SLAB_EVIDENCE_COMPUTED = "computed_preconditions_v1"
SLAB_EVIDENCE_ASSERTED = "engineering_assertions_v0"
ROW_TORSION = "beam.torsion_concurrent"
TORSION_CLAUSE = "ACI 318-19 9.5.4; 9.6.4; 9.7.5; 9.7.6.3; 22.7"
PHI_TORSION = 0.75
PHI_TENSION = 0.90
PHI_DIAPHRAGM_SHEAR = 0.60                  # 21.2.4.1 and 21.2.4.2: the shear-controlled value, so flexure control need not be shown
OMEGA_0 = 3.0                               # ASCE 7-22 Table 12.2-1, special reinforced concrete moment frame
SHEAR_FRICTION_MU = 1.4                     # 22.9.4.2, monolithic concrete, normalweight
SPAN_DEPTH_RIGID_LIMIT = 3.0                # ASCE 7-22 12.3.1.2
SIDE_BAR_SIZES = (4, 5, 6, 7, 8, 9)
HOOP_DIAMETER_BOUND_IN = 0.625              # the largest beam hoop of the ladder (No. 5), so the core is a lower bound


def computed_slab_basis(cfg):
    """Whether this candidate's slab demands are accepted on computed conditions instead of named assertions."""
    analysis = getattr(cfg, "floor_analysis", None)
    return (getattr(analysis, "slab_evidence_basis", SLAB_EVIDENCE_ASSERTED) == SLAB_EVIDENCE_COMPUTED
            and getattr(analysis, "slab_procedure", None) is None
            and getattr(analysis, "common_action_procedure", None) is None)


def slab_acceptance_problems(evidence):
    """What keeps the computed strip demands from being accepted; empty when they are usable for strip design."""
    problems = []
    if evidence.get("method") != "plate_finite_element":
        problems.append("the strip demands do not come from the plate analysis")
    preconditions = evidence.get("numerical_preconditions") or {}
    for key, text in (("all_floors_covered_once", "not every floor is covered once"),
                      ("same_slab_inputs_and_cases", "the floors do not share slab inputs and cases"),
                      ("physical_recovery_valid", "the face recovery is not valid on every floor"),
                      ("mesh_refinement_verified", "the mesh refinement did not pass on every floor")):
        if preconditions.get(key) is not True:
            problems.append(text)
    equilibrium = evidence.get("equilibrium") or []
    if not equilibrium or not all(row.get("numerical_balance_passed") is True for row in equilibrium):
        problems.append("a slab load case does not balance")
    if not evidence.get("cases") or not evidence.get("pattern_rule"):
        problems.append("the slab load cases or their pattern rule are missing")
    return problems


# ---- demands the plate carries --------------------------------------------------------------------------------
def _layer_demand(layout, axis, face):
    layer = layout["layers"][f"{axis}_{face}"]
    return float(layer["demand_envelope"]["mu_kip_in_per_ft"])


def flange_slab_moments(member, layout):
    """Slab moment across the member's flange overhangs, hogging and sagging, kip-in (0 without a layout)."""
    from Model import Member_Properties as mp
    if layout is None or sp.SLAB_THICKNESS_IN is None:
        return 0.0, 0.0
    flange = mp.effective_flange(member)
    overhang = max(0.0, float(flange["flange_width_in"]) - float(member.design.b_in))
    axis = mp.beam_axis(member)
    return _layer_demand(layout, axis, "top") * overhang / 12.0, _layer_demand(layout, axis, "bottom") * overhang / 12.0


def _transfer_of(transfers, floor):
    if not transfers:
        return None
    return transfers.get(floor, transfers.get(str(floor)))


def _unit_end_torques(transfer, key):
    """{unit case: (end i, end j)}: the plate's edge couples on one span as fixed-end torques."""
    result = {}
    for name, case in (transfer.get("unit_cases") or {}).items():
        if not case:
            continue
        entry = next((b for b in case["beams"] if (b["axis"], b["line_index"], b["span_index"]) == key), None)
        if entry is None:
            continue
        couples = entry.get("node_couples", [])
        result[name] = (sum(row[1] * (1.0 - row[0]) for row in couples), sum(row[1] * row[0] for row in couples))
    return result


def member_torque(member_record, action, unit_torques):
    """End torques of one member in one combination: frame element torque plus the factored plate couples (magnitudes)."""
    force = member_record.get("local_force_kip_kipin") or [0.0] * 12
    pattern = action.get("live_pattern")
    live_case = "live" if pattern in (None, "all") else f"live_pattern_{pattern}"
    dead = unit_torques.get("dead", (0.0, 0.0))
    live = unit_torques.get(live_case, (0.0, 0.0)) if action.get("live") else (0.0, 0.0)
    gamma_d, gamma_l = float(action.get("dead") or 0.0), float(action.get("live") or 0.0)
    return (abs(float(force[3])) + abs(gamma_d * dead[0] + gamma_l * live[0]),
            abs(float(force[9])) + abs(gamma_d * dead[1] + gamma_l * live[1]))


# ---- torsion -------------------------------------------------------------------------------------------------
def torsion_geometry(b_in, h_in):
    """Closed-hoop centerline geometry (22.7.6.1) with the largest hoop of the ladder, a lower bound of the core."""
    cover = sp.BEAM_CLEAR_COVER_IN
    x1 = b_in - 2.0 * cover - HOOP_DIAMETER_BOUND_IN
    y1 = h_in - 2.0 * cover - HOOP_DIAMETER_BOUND_IN
    aoh = x1 * y1
    return {"x1_in": x1, "y1_in": y1, "aoh_in2": aoh, "ao_in2": 0.85 * aoh, "ph_in": 2.0 * (x1 + y1)}


def edge_release_margin(member, layout):
    """Lower-bound strip check of the panels beside a beam with the edge restraint released (22.7.3.3).

    With zero moment at the released edge, a strip's static moment w l^2 / 8 is carried by the mid-span steel plus half
    of the support steel. A load share between the two directions is free in the strip method, so a share exists when
    the two ratios sum to at least 1. Returns that sum and its inputs."""
    if layout is None:
        return None
    dead = sp.floor_dead_load_ksf()
    wu = (1.2 * dead + 1.6 * sp.FLOOR_LIVE_LOAD_KSF) / 144.0                 # kip per sq in
    per_ft = wu * 12.0
    faces = float(member.design.b_in)
    ln = {"x": sp.BAY_X - faces, "y": sp.BAY_Y - faces}
    ratios = {}
    for axis, span in ln.items():
        bottom = layout["layers"][f"{axis}_bottom"]["flexure"]["phi_mn_kip_in_per_ft"]
        top = layout["layers"][f"{axis}_top"]["flexure"]["phi_mn_kip_in_per_ft"]
        ratios[axis] = (float(bottom) + 0.5 * float(top)) / (per_ft * span * span / 8.0)
    return {"sum_of_ratios": ratios["x"] + ratios["y"], "ratios": ratios, "wu_ksi": wu,
            "basis": "strip method lower bound with the released edge at zero moment"}


def torsion_design(member, tu_kip_in, n_tension_kip, layout):
    """The torsion branch and steel of one beam design for its governing Tu. Pure function of section, Tu and N."""
    from Design.SMRF_Beam_Torsion import threshold
    from Design.SMRF_Beam_Torsion_Adapter import actual_member_torsion_section
    design = member.design
    slab_fc = float(((sp.SLAB_REINFORCEMENT or {}).get("inputs") or {}).get("slab", {}).get("fc_ksi", design.fc_ksi))
    section = actual_member_torsion_section(member, slab_fc)
    limit = threshold(section, tu_kip_in, n_tension_kip)
    result = {"tu_kip_in": tu_kip_in, "n_tension_kip": n_tension_kip, "member_tag": member.member_tag,
              "section": {k: section.get(k) for k in ("Acp_in2", "pcp_in", "fc_ksi", "flanges_omitted")},
              "threshold": limit, "branch": "not_evaluated", "at_per_in2": 0.0, "al_total_in2": 0.0, "al_face_in2": 0.0,
              "side_bars": None, "face_moment_kip_in": 0.0, "tu_design_kip_in": 0.0}
    if limit.get("status") != "pass":
        result["reason"] = limit.get("reason", "the torsion threshold could not be formed")
        return result
    if limit["below_threshold"]:
        result["branch"] = "below_threshold"
        return result
    tu = tu_kip_in
    phi_tcr = PHI_TORSION * limit["Tcr_kip_in"]
    if tu > phi_tcr:
        release = edge_release_margin(member, layout)
        result["edge_release"] = release
        if release is not None and release["sum_of_ratios"] >= 1.0:
            tu = phi_tcr
            result["branch"] = "reduced_to_phi_tcr"
        else:
            result["branch"] = "design_for_tu"
    else:
        result["branch"] = "design_for_tu"
    geometry = torsion_geometry(design.b_in, design.h_in)
    fyt, fy = min(60.0, sp.FYT_KSI), min(60.0, sp.FY_KSI)
    at = tu / (PHI_TORSION * 2.0 * geometry["ao_in2"] * fyt)                    # one leg, in2 per in, 45 degrees
    al = at * geometry["ph_in"] * fyt / fy
    root = math.sqrt(section["fc_ksi"] * 1000.0)
    minimum_base = 5.0 * root * section["Acp_in2"] / (fy * 1000.0)
    al_minimum = max(0.0, min(minimum_base - at * geometry["ph_in"] * fyt / fy,
                              minimum_base - 25.0 * design.b_in / (fyt * 1000.0) * geometry["ph_in"] * fyt / fy))
    required = max(al, al_minimum)
    side_share = required * geometry["y1_in"] / geometry["ph_in"]
    face_share = required * geometry["x1_in"] / geometry["ph_in"]
    count = max(0, math.ceil(geometry["y1_in"] / 12.0) - 1)                     # 9.7.5.1: 12 in around the perimeter
    side_bars = None
    if count:
        minimum_db = max(0.375, 0.042 * min(geometry["ph_in"] / 8.0, 12.0))     # 9.7.5.2
        for bar in SIDE_BAR_SIZES:
            if sp.rebar_diameter(bar) >= minimum_db and count * sp.rebar_area(bar) >= side_share - 1e-9:
                side_bars = {"bar_size": bar, "count_per_side": count, "area_per_side_in2": count * sp.rebar_area(bar)}
                break
        if side_bars is None:
            result["side_bar_problem"] = "no side bar of the ladder carries the side share"
    else:
        face_share += side_share
    d = design.h_in - sp.first_bar_offset_in(sp.BEAM_CLEAR_COVER_IN, HOOP_DIAMETER_BOUND_IN, sp.rebar_diameter(design.bar_size))
    result.update(tu_design_kip_in=tu, at_per_in2=at, al_total_in2=required, al_face_in2=face_share,
                  al_minimum_in2=al_minimum, side_bars=side_bars, geometry=geometry, fyt_used_ksi=fyt, fy_used_ksi=fy,
                  face_moment_kip_in=PHI_TENSION * face_share * fy * 0.9 * d, spacing_limit_in=min(geometry["ph_in"] / 8.0, 12.0))
    return result


def side_bar_strength_increment(member, fy_factor=1.0):
    """Upper bound of the flexural strength the torsion side bars add at full depth lever, for capacity demands only."""
    design = (getattr(sp, "BEAM_TORSION_DESIGN", None) or {}).get(member.group_id) or {}
    bars = design.get("side_bars")
    if not bars:
        return 0.0
    return 2.0 * bars["area_per_side_in2"] * fy_factor * sp.FY_KSI * member.design.h_in / 2.0


def group_torsion_designs(state, actions):
    """{beam group: torsion design at its governing member}, from the torques saved on the actions."""
    designs = {}
    for gid, group in sorted(state.groups.items()):
        if group["member_type"] == "column":
            continue
        worst = None
        for action in actions:
            members = action["members"]
            for tag in group["member_tags"]:
                record = members.get(str(tag)) or {}
                floor = record.get("floor_actions") or {}
                tu = float(floor.get("tu_kip_in", 0.0))
                if worst is None or tu > worst[0]:
                    tension = max(0.0, -min(float(record.get("axial_i_kip", 0.0)), float(record.get("axial_j_kip", 0.0))))
                    worst = (tu, tag, tension + float(floor.get("n_diaphragm_kip", 0.0)))
        if worst is None:
            continue
        layout = (sp.SLAB_REINFORCEMENT or {}).get("layout")
        designs[gid] = torsion_design(state.member(worst[1]), worst[0], worst[2], layout)
    return designs


def diaphragm_force_table(elf, risk_category=None):
    """Fpx by level (ASCE 7-22 12.10.1.1) from the ELF forces."""
    from Design.SMRF_Diaphragm import diaphragm_forces
    basis = sp.seismic_design_basis(risk_category=risk_category)
    return diaphragm_forces({"seismic": {"sds": float(sp.ASCE_SDS), "importance_factor": float(basis["importance_factor"])},
                             "elf": {"story_forces_kip": [float(v) for v in elf["story_forces_kip"]],
                                     "story_weights_kip": [float(v) for v in elf["story_weights_kip"]]}})


def collector_force_kip(level, story_force_kip=None):
    """The collector force of one level (ASCE 7-22 12.10.2.1): overstrength on the largest of the ELF story force, Eq. 12.10-1
    within its limit of Eq. 12.10-3, and Eq. 12.10-2. ``fpx_kip`` already holds those limits. No further cap applies to the
    maximum; the capacity-limited effect of 12.4.3.1 is not taken."""
    return OMEGA_0 * max(float(level["fpx_kip"]), float(story_force_kip or 0.0))


def collector_axial_kip(line_force_kip, bays):
    """Axial force at a frame-line beam face: shear flow over half a bay, with column shears by tributary length."""
    return line_force_kip / (2.0 * max(1, int(bays)))


def attach_floor_actions(actions, state, transfers, elf, cfg):
    """Write ``floor_actions`` on every beam record of every combination and install the group torsion designs.

    Each combination gets its own member dictionaries, so a record shared between combinations is not overwritten."""
    from Model import Member_Groups as mg
    from Model import Member_Properties as mp
    layout = (sp.SLAB_REINFORCEMENT or {}).get("layout")
    table = elf if elf is not None else None
    forces = [float(v) for v in elf["story_forces_kip"]] if elf is not None else None
    static = {}
    with mg.installed(state):
        for gid, group in state.groups.items():
            if group["member_type"] == "column":
                continue
            sfrs = not mg.is_system_policy(state.policy_id) or state.system_of(gid) == mg.SYSTEM_SFRS
            for tag in group["member_tags"]:
                member = state.member(tag)
                axis = mp.beam_axis(member)
                key = (axis, member.grid_j, member.grid_i) if axis == "x" else (axis, member.grid_i, member.grid_j)
                transfer = _transfer_of(transfers, member.story_or_floor)
                torques = _unit_end_torques(transfer, key) if transfer else {}
                hogging, sagging = flange_slab_moments(member, layout)
                axial = 0.0
                if table is not None and sfrs:
                    # In the strength combinations the rigid-diaphragm frame omits the story force entering the frame line. The
                    # overstrength collector force is a separate check of its own combination (floor_rows).
                    bays = sp.NUM_BAY_X if axis == "x" else sp.NUM_BAY_Y
                    axial = collector_axial_kip(forces[member.story_or_floor - 1] / 2.0, bays)
                static[str(tag)] = {"torques": torques, "hogging": hogging, "sagging": sagging, "axial": axial}
    for action in actions:
        members = dict(action["members"])
        seismic = bool(action.get("ex") or action.get("ey"))
        for tag, data in static.items():
            record = members.get(tag)
            if record is None:
                continue
            ti, tj = member_torque(record, action, data["torques"])
            members[tag] = {**record, "floor_actions": {
                "mu_slab_hogging_kip_in": data["hogging"], "mu_slab_sagging_kip_in": data["sagging"],
                "n_diaphragm_kip": data["axial"] if seismic else 0.0, "tu_end_i_kip_in": ti, "tu_end_j_kip_in": tj,
                "tu_kip_in": max(ti, tj), "torsion_face_moment_kip_in": 0.0}}
        action["members"] = members
    with mg.installed(state):
        designs = group_torsion_designs(state, actions)
    sp.BEAM_TORSION_DESIGN = designs
    for action in actions:
        for gid, design in designs.items():
            moment = float(design.get("face_moment_kip_in") or 0.0)
            if not moment:
                continue
            for tag in state.groups[gid]["member_tags"]:
                record = action["members"].get(str(tag))
                if record is not None:
                    record["floor_actions"]["torsion_face_moment_kip_in"] = moment
    return designs


def torsion_rows(state, beams, designs):
    """One row per beam group: the torsion branch and the installed hoops against it."""
    rows = []
    for gid, data in sorted(beams.items()):
        design = designs.get(gid)
        if design is None:
            rows.append(not_evaluated(ROW_TORSION, TORSION_CLAUSE, "No torsion design was formed for this group", gid))
            continue
        if design["branch"] == "not_evaluated":
            rows.append(not_evaluated(ROW_TORSION, TORSION_CLAUSE, design.get("reason", "torsion inputs are unavailable"), gid))
            continue
        if design["branch"] == "below_threshold":
            limit = design["threshold"]
            rows.append(make_check(ROW_TORSION, "ACI 318-19 9.5.4.1; 22.7.1.1", design["tu_kip_in"], limit["phi_Tth_kip_in"],
                                   "<=", "kip-in", gid, details={"branch": "below_threshold", "member_tag": design["member_tag"]}))
            continue
        hoops = data.get("hoops")
        if not hoops:
            rows.append(not_evaluated(ROW_TORSION, TORSION_CLAUSE, "No hoops are selected for this group", gid))
            continue
        ratios, details = {}, {"branch": design["branch"], "tu_kip_in": design["tu_kip_in"],
                               "tu_design_kip_in": design["tu_design_kip_in"], "member_tag": design["member_tag"],
                               "al_total_in2": design["al_total_in2"], "al_face_in2": design["al_face_in2"],
                               "side_bars": design["side_bars"], "edge_release": design.get("edge_release")}
        geometry = design["geometry"]
        area = sp.rebar_area(hoops["bar_size"])
        spacing = hoops["spacing_in"]
        vs = float(data.get("vs_required_kip", 0.0))
        d = float(data["effective_depth_in"])
        fyt = design["fyt_used_ksi"]
        shear_per_leg = vs / (min(60.0, sp.FYT_KSI) * d * hoops["legs"]) if vs > 0 else 0.0
        ratios["hoop_leg_area"] = (design["at_per_in2"] + shear_per_leg) * spacing / area
        ratios["hoop_spacing"] = spacing / design["spacing_limit_in"]
        design_b = state.designs[gid].b_in
        minimum = max(0.75 * math.sqrt(state.designs[gid].fc_ksi * 1000.0), 50.0) * design_b / (fyt * 1000.0)
        provided = hoops["legs"] * area / spacing
        ratios["minimum_transverse"] = minimum / max(provided, 1e-9)
        root = math.sqrt(state.designs[gid].fc_ksi * 1000.0)
        vu_stress = float(data.get("ve_kip", 0.0)) * 1000.0 / (design_b * d)
        tu_stress = design["tu_design_kip_in"] * 1000.0 * geometry["ph_in"] / (1.7 * geometry["aoh_in2"] ** 2)
        limit = PHI_TORSION * (float(data.get("vc_kip", 0.0)) * 1000.0 / (design_b * d) + 8.0 * root)
        ratios["section_limit_22_7_7_1"] = math.hypot(vu_stress, tu_stress) / limit
        count = math.ceil(geometry["y1_in"] / 12.0) - 1
        ratios["side_bars_provided"] = 0.0 if (count <= 0 or design["side_bars"] is not None) else 2.0
        details["ratios"] = ratios
        rows.append(make_check(ROW_TORSION, TORSION_CLAUSE, max(ratios.values()), 1.0, "<=", "ratio", gid, details=details))
    return rows


# ---- diaphragm and floor rows ----------------------------------------------------------------------------------
FLOOR_ROW_IDS = ("floor.gravity_load_balance", "diaphragm.rigid_idealization", "diaphragm.minimum_thickness")
DIRECTIONAL_ROW_IDS = ("diaphragm.minimum_reinforcement", "diaphragm.in_plane_shear", "diaphragm.in_plane_flexure",
                       "diaphragm.interface_shear_transfer", "diaphragm.collector_tension", "diaphragm.collector_compression")
COLUMN_POSITIONS = {"interior": (4, 1.0, 40.0), "edge": (3, 0.5, 30.0), "corner": (2, 0.25, 20.0)}    # faces, tributary share, alpha_s


def floor_row_inventory():
    """The (family, id, location) rows the production route requires of the floor and diaphragm."""
    rows = {("floor", name, "") for name in FLOOR_ROW_IDS}
    rows.update(("floor", name, axis) for name in DIRECTIONAL_ROW_IDS for axis in ("x", "y"))
    rows.update(("floor", "diaphragm.maximum_bar_spacing", f"{axis}_{face}") for axis in ("x", "y") for face in ("top", "bottom"))
    rows.update(("floor", "slab.column_local_shear_stress", position) for position in COLUMN_POSITIONS)
    return rows


def column_local_shear_rows(state, layout, thickness):
    """ACI 318-19 8.6.1.2 at the column positions of the floor. The stress is a bound: the factored load of the whole tributary
    area over the section at the column faces, with the beam depth over the beam width and the slab depth elsewhere, using
    the smallest column, the smallest beam and the shallowest slab layer. Below its trigger no local steel is required;
    above it the mats are compared with the required area."""
    from Model import Member_Groups as mg
    rows = []
    clause = "ACI 318-19 8.6.1.2; 22.6.4; 22.5.5.1.3"
    d_slab = min(float(layer["effective_depth_in"]) for layer in layout["layers"].values())
    with mg.installed(state):
        columns = [m.design for m in mg.all_members("column")]
        beams = [m.design for axis in ("x", "y") for m in mg.all_members(f"beam_{axis}")]
    side = min(min(c.b_in, c.h_in) for c in columns)
    width = min(b.b_in for b in beams)
    depth = min(b.h_in for b in beams) - sp.first_bar_offset_in(sp.BEAM_CLEAR_COVER_IN, HOOP_DIAMETER_BOUND_IN, 1.128)
    fc = float(((sp.SLAB_REINFORCEMENT or {}).get("inputs") or {}).get("slab", {}).get("fc_ksi", 5.0))
    root = min(100.0, math.sqrt(fc * 1000.0))
    size_factor = min(1.0, math.sqrt(2.0 / (1.0 + d_slab / 10.0)))
    limit = 0.75 * 2.0 * size_factor * root
    wu = (1.2 * sp.floor_dead_load_ksf() + 1.6 * sp.FLOOR_LIVE_LOAD_KSF) / 144.0
    area = sp.BAY_X * sp.BAY_Y
    provided = min(sum(float(layout["layers"][f"{axis}_{face}"]["area_in2_per_ft"]) for face in ("top", "bottom")) / 12.0 for axis in ("x", "y"))
    for position, (faces, share, alpha) in COLUMN_POSITIONS.items():
        section = faces * (min(width, side) * depth + max(0.0, side - width) * d_slab)
        stress = wu * area * share * 1000.0 / section
        details = {"tributary_share": share, "faces": faces, "section_area_in2": section, "trigger_psi": limit, "stress_psi": stress}
        if stress <= limit:
            rows.append(make_check("slab.column_local_shear_stress", clause, stress, limit, "<=", "psi", position, details=details))
        else:
            required = 5.0 * stress * (faces * side) / (0.75 * alpha * float(sp.SLAB_FY_KSI) * 1000.0)
            rows.append(make_check("slab.column_local_shear_stress", clause, required, provided, "<=", "in2/in", position, details=details))
    return rows


def _reserve(layout, axis, gravity_ratio):
    """The smallest share of the two mats along ``axis`` left after the slab's own moment under the seismic gravity."""
    shares = []
    for face in ("top", "bottom"):
        layer = layout["layers"][f"{axis}_{face}"]
        shares.append(1.0 - gravity_ratio * float(layer["demand_envelope"]["mu_kip_in_per_ft"])
                      / float(layer["flexure"]["phi_mn_kip_in_per_ft"]))
    return max(0.0, min(shares))


def floor_rows(state, slab, elf, cfg, transfers):
    """Floor load balance and the diaphragm rows of the installed candidate."""
    from Loads.Perimeter_Strip import plate_loads_kip
    from Model import Member_Groups as mg
    from Model import Member_Properties as mp
    rows = []
    layout = (slab.get("reinforcement") or {}).get("layout")
    thickness = slab["slab"]["thickness_in"]
    plan = {"x": sp.NUM_BAY_X * sp.BAY_X, "y": sp.NUM_BAY_Y * sp.BAY_Y}
    # Floor load balance: the transfer of each floor against the plate load, and the plate plus strip against the floor.
    plate_dead, plate_live = plate_loads_kip(sp.floor_dead_load_ksf(), sp.FLOOR_LIVE_LOAD_KSF)
    worst = 0.0
    for floor, transfer in (transfers or {}).items():
        for name, target in (("dead", plate_dead), ("live", plate_live)):
            case = (transfer.get("unit_cases") or {}).get(name)
            if case:
                worst = max(worst, abs(float(case["beam_kip"]) + float(case["column_direct_kip"]) - target) / max(target, 1e-9))
    balance = make_check("floor.gravity_load_balance", "Load path: beam and column transfer equals the plate load", worst, 1.0e-6, "<=",
                         "relative error", details={"floors": len(transfers or {}), "plate_dead_kip": plate_dead,
                                                    "plate_live_kip": plate_live})
    rows.append(balance if transfers else not_evaluated("floor.gravity_load_balance", "Load path", "No floor transfer is available"))
    ratio = max(plan.values()) / min(plan.values())
    rows.append(make_check("diaphragm.rigid_idealization", "ASCE 7-22 12.3.1.2 (span to depth 3 or less)", ratio, SPAN_DEPTH_RIGID_LIMIT,
                           "<=", "span/depth", details={"plan_in": plan, "system": "cast-in-place slab, no horizontal irregularity by "
                                                                                    "the frame layout"}))
    rows.append(make_check("diaphragm.minimum_thickness", "ACI 318-19 18.12.6.1", 2.0, thickness, "<=", "in"))
    missing = None
    if layout is None:
        missing = "no slab layout is installed"
    elif elf is None:
        missing = "no ELF forces are available"
    if missing:
        for family, name, location in sorted(floor_row_inventory()):
            if name not in FLOOR_ROW_IDS:
                rows.append(not_evaluated(name, "ACI 318-19 18.12; ASCE 7-22 12.10", missing, location))
        return rows
    table = diaphragm_force_table(elf, cfg.demands.risk_category)
    fpx = float(table["fpx_governing_kip"])
    slab_fc = float(((sp.SLAB_REINFORCEMENT or {}).get("inputs") or {}).get("slab", {}).get("fc_ksi", 5.0))
    root = min(100.0, math.sqrt(slab_fc * 1000.0))                                  # 12.5.3.3
    fy = float(sp.SLAB_FY_KSI)
    dead, live = sp.floor_dead_load_ksf(), sp.FLOOR_LIVE_LOAD_KSF
    gravity_ratio = ((1.2 + 0.2 * float(sp.ASCE_SDS)) * dead + 0.5 * live) / (1.2 * dead + 1.6 * live)
    with mg.installed(state):
        edge_flange = {}
        for axis in ("x", "y"):
            members = [m for m in mg.all_members(f"beam_{axis}") if m.location_class == "edge"]
            edge_flange[axis] = max(float(mp.effective_flange(m)["flange_width_in"]) for m in members)
            edge_flange[f"{axis}_steel"] = min(state.designs[m.group_id].longitudinal_area_in2 for m in members)
            edge_flange[f"{axis}_area"] = min(state.designs[m.group_id].b_in * state.designs[m.group_id].h_in for m in members)
            edge_flange[f"{axis}_fc"] = min(state.designs[m.group_id].fc_ksi for m in members)
    for axis in ("x", "y"):
        other = "y" if axis == "x" else "x"
        mats = [layout["layers"][f"{axis}_{face}"] for face in ("top", "bottom")]
        per_ft = sum(float(m["area_in2_per_ft"]) for m in mats)
        rows.append(make_check("diaphragm.minimum_reinforcement", "ACI 318-19 18.12.7.1; 24.4.3.2", 0.0018, per_ft / (12.0 * thickness),
                               "<=", "ratio", axis))
        for face, mat in zip(("top", "bottom"), mats):
            rows.append(make_check("diaphragm.maximum_bar_spacing", "ACI 318-19 18.12.7.1", float(mat["spacing_in"]), 18.0, "<=", "in",
                                   f"{axis}_{face}"))
        # Force along ``axis``: the section through the slab runs along ``axis`` and carries half of Fpx.
        vu = fpx / 2.0
        acv = thickness * plan[axis]
        reserve_along = _reserve(layout, axis, gravity_ratio)
        vn_concrete = acv * 2.0 * root / 1000.0
        vn_cap = acv * 8.0 * root / 1000.0
        steel = 0.0 if vu <= PHI_DIAPHRAGM_SHEAR * vn_concrete else acv * (per_ft / (12.0 * thickness)) * reserve_along * fy
        phi_vn = PHI_DIAPHRAGM_SHEAR * min(vn_concrete + steel, vn_cap)
        rows.append(make_check("diaphragm.in_plane_shear", "ACI 318-19 18.12.9.1; 18.12.9.2; 21.2.4.2", vu, phi_vn, "<=", "kip", axis,
                               details={"fpx_kip": fpx, "acv_in2": acv, "vn_concrete_kip": vn_concrete, "steel_counted_kip": steel,
                                        "basis": "concrete term alone unless it falls short; then only the mat share left after slab flexure"}))
        # In-plane flexure for a force along ``other``: the diaphragm spans along ``axis`` between the frames at its ends,
        # its depth is the plan dimension along ``other`` and the mats along ``axis`` take the tension.
        moment = fpx * plan[axis] / 8.0
        depth = plan[other]
        width = max(0.0, depth / 4.0 - edge_flange[axis])                        # 12.5.2.3: within h/4 of the tension edge
        steel_area = (per_ft / 12.0) * width * reserve_along
        centroid = edge_flange[axis] + width / 2.0
        available_force = PHI_TENSION * fy * steel_area
        block = available_force / (0.85 * slab_fc * thickness)
        lever = max(1e-9, depth - centroid - block / 2.0)
        tension = moment / lever
        rows.append(make_check("diaphragm.in_plane_flexure", "ACI 318-19 12.5.1.3(a); 12.5.2.3; 18.12.8; 12.6.3", tension,
                               available_force, "<=", "kip", axis,
                               details={"mu_kip_in": moment, "lever_arm_in": lever, "mat_width_in": width, "mat_reserve": reserve_along,
                                        "edge_flange_excluded_in": edge_flange[axis],
                                        "basis": "beam idealization: tension on the mats within h/4 of the tension edge and outside the "
                                                 "edge-beam flange, which the T-beam section already uses; only the share left after "
                                                 "slab flexure at the seismic gravity load counts (12.6.3)"}))
        # Transfer into the frame lines parallel to ``other``: mats along ``axis`` cross that interface.
        flow = (fpx / 2.0) / plan[other]
        avf = (per_ft / 12.0) * _reserve(layout, axis, gravity_ratio)
        capacity = PHI_DIAPHRAGM_SHEAR * min(SHEAR_FRICTION_MU * avf * fy, min(0.2 * slab_fc, 0.48 + 0.08 * slab_fc, 1.6) * thickness)
        rows.append(make_check("diaphragm.interface_shear_transfer", "ACI 318-19 22.9.4.2; 22.9.4.4; 12.5.3", flow, capacity, "<=", "kip/in", axis,
                               details={"force_along": other, "avf_in2_per_in": avf, "mu": SHEAR_FRICTION_MU}))
        # The frame-line beams along ``axis`` collect a force along ``axis``.
        bays = sp.NUM_BAY_X if axis == "x" else sp.NUM_BAY_Y
        level = max(table["levels"], key=lambda lv: lv["fpx_kip"])
        story = [float(v) for v in elf["story_forces_kip"]][level["level"] - 1]
        force = collector_force_kip(level, story)
        axial = collector_axial_kip(force / 2.0, bays)
        steel_in2 = edge_flange[f"{axis}_steel"]
        details = {"collector_force_kip": force, "fpx_kip": level["fpx_kip"], "story_force_kip": story, "bays": bays,
                   "omega_0": OMEGA_0, "basis": "the frame runs the full edge: axial force is the shear flow over half a bay at the "
                                                "overstrength force; the story force without overstrength enters the beam N-M rows"}
        rows.append(make_check("diaphragm.collector_tension", "ASCE 7-22 12.10.2.1; ACI 318-19 18.12.7.5", axial,
                               PHI_TENSION * min(60.0, float(sp.FY_KSI)) * steel_in2, "<=", "kip", axis, details=details))
        rows.append(make_check("diaphragm.collector_compression", "ACI 318-19 18.12.7.6 (0.5 f'c with overstrength forces)",
                               axial / edge_flange[f"{axis}_area"], 0.5 * edge_flange[f"{axis}_fc"], "<=", "ksi", axis,
                               details={**details, "note": "above the limit the collector needs the 18.12.7.6 confinement; the "
                                                           "frame beam's own hoops are checked by 18.6.4"}))
    rows.extend(column_local_shear_rows(state, layout, thickness))
    return rows
