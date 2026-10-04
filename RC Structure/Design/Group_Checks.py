"""Solved member actions and member strength checks for a grouped design (2026-10-02).

The uniform checks (Design.ACI_Checks) read one column and one beam from cfg and Structure_Parameters.
Here every member is checked on its own group's section and cage, against its own concurrent actions in
every solved combination: a column's P, My and Mz are those of one combination at that column (the
biaxial load-contour check of ACI_Checks.check_column_pm, unchanged), a beam's sagging and hogging
moments those of its own span envelope. A group's result is the governing member and combination; no
action is assembled from unrelated extrema.

The formulas are the uniform ones with the section passed in:
  columns  phi P-M about each axis from ACI_Checks.pm_diagram_for (the sweep build_pm_diagram uses), the
           axial cap, and the load-contour interaction; analysis shear with ACI_Checks.check_shear;
  beams    phi Mn = 0.9 As fy (d - a/2) per face with d from the group's own bar rows
           (Model.Member_Bar_Layers), analysis shear with check_shear on the smaller depth.
"""
from __future__ import annotations

import math

import openseespy.opensees as ops

import Structure_Parameters as sp
from Design.ACI_Checks import (DOMAIN_FAILURE_DCR, PHI_FLEX, _interpolate_pm_capacity, axial_demand_ratio, axial_limits,
                               check_shear, pm_diagram_for)
from Model import Member_Bar_Layers as bar_layers
from Model import Member_Groups as mg
from Model import Member_Properties as mp
from RC_Design_Check import _design_col_steel_layers

_DIAGRAMS = {}


# ---- solved actions ---------------------------------------------------------------------------------------
def joint_beam_depth_in(level, i, j):
    """Depth of the deepest beam framing into the joint at ``level`` (0 at the base: no beams there)."""
    if level < 1:
        return 0.0
    joint = mg.joint_members(level, i, j)
    return max(joint[key].design.h_in for key in ("beam_x_minus", "beam_x_plus", "beam_y_minus", "beam_y_plus")
               if joint[key] is not None)


def capture_member_actions(dead_factor=1.0):
    """Signed concurrent actions of every physical member in the solved domain (grouped design).

    The member-resolved form of Design_Driver._capture_element_actions: a column's joint-face axial loads
    are taken half the depth of the deepest beam at each of its own joints from the centerline nodes, and
    the axial line load is that column's own weight. Keys and meanings are the uniform ones.
    """
    from Design.SMRF_Beam_Actions import current_beam_bending
    members = mg.all_members()
    spans = current_beam_bending([m.member_tag for m in members if not m.is_column])
    result = {}
    for member in members:
        tag = member.member_tag
        forces = list(ops.eleResponse(tag, "localForce"))
        if len(forces) != 12 or not all(math.isfinite(value) for value in forces):
            raise RuntimeError(f"Invalid local force vector for design member {tag}.")
        offset_i = offset_j = axial_line_load = 0.0
        if member.is_column:
            k, i, j = member.story_or_floor, member.grid_i, member.grid_j
            offset_i = 0.5 * joint_beam_depth_in(k - 1, i, j)
            offset_j = 0.5 * joint_beam_depth_in(k, i, j)
            axial_line_load = dead_factor * mp.column_self_weight_kip_per_in(member)
        result[str(tag)] = {
            "member_type": member.member_type, "group_id": member.group_id, "local_force_kip_kipin": forces,
            "centerline_axial_i_kip": forces[0], "centerline_axial_j_kip": -forces[6],
            "axial_i_kip": forces[0] - axial_line_load * offset_i,
            "axial_j_kip": -forces[6] + axial_line_load * offset_j,
            "joint_face_offsets_in": [offset_i, offset_j],
            "axial_line_load_kip_per_in": axial_line_load,
        }
        if not member.is_column:
            result[str(tag)]["span_bending"] = spans[tag]
    return result


# ---- columns ------------------------------------------------------------------------------------------------
def column_pm_diagrams(design):
    """{"y": phi P-M bending through h, "z": through b} of one column design (ACI_Checks.build_pm_diagrams)."""
    layers_y = _design_col_steel_layers(design, about="h")
    layers_z = _design_col_steel_layers(design, about="b")
    key = (design.b_in, design.h_in, design.fc_ksi, tuple(layers_y), tuple(layers_z), sp.FY_KSI, sp.ES_KSI)
    if key not in _DIAGRAMS:
        if len(_DIAGRAMS) > 2048:
            _DIAGRAMS.clear()
        _DIAGRAMS[key] = {"y": pm_diagram_for(design.b_in, design.h_in, design.fc_ksi, sp.FY_KSI, sp.ES_KSI, layers_y),
                          "z": pm_diagram_for(design.h_in, design.b_in, design.fc_ksi, sp.FY_KSI, sp.ES_KSI, layers_z)}
    return _DIAGRAMS[key]


def column_force_components(local_force):
    """(P, My, Mz, Vy, Vz) of one column in one combination: the end-i axial load with the larger end moment and
    shear of each local axis (RC_Design_Check._extract_forces, the uniform check's reading)."""
    f = [float(v) for v in local_force]
    return f[0], max(abs(f[4]), abs(f[10])), max(abs(f[5]), abs(f[11])), max(abs(f[1]), abs(f[7])), max(abs(f[2]), abs(f[8]))


def column_pm_dcr(diagrams, axial_kip, my_kip_in, mz_kip_in, alpha=1.5):
    """Biaxial load-contour DCR of ACI_Checks.check_column_pm for explicit diagrams.

    Returns {"dcr", "basis", "domain_failure", "phi_mn_y_kip_in", "phi_mn_z_kip_in", "axial_dcr"}. The axial
    load is placed against BOTH ends of the surface before anything else (ACI_Checks.axial_limits,
    axial_demand_ratio): above the phi Pn,max cap it is an axial compression failure whatever the moment;
    below the pure-tension end it is an axial tension failure whatever the moment. Both give the
    non-negative ratio of the load to the capacity of its own sign (above 1) and name the failure under
    ``domain_failure``. Inside the surface a load with no moment gives that same ratio (at most 1); a
    moment at an axial load where the surface has no flexural strength left (the tension end) is the
    named failure ``moment_at_axial_end`` with the reserved value DOMAIN_FAILURE_DCR.
    """
    diagram_y, diagram_z = diagrams["y"], diagrams["z"]
    compression, tension = axial_limits(diagram_y, diagram_z)
    axial_dcr = axial_demand_ratio(axial_kip, compression, tension)
    result = {"phi_mn_y_kip_in": None, "phi_mn_z_kip_in": None, "axial_dcr": axial_dcr, "domain_failure": None,
              "axial_capacity_compression_kip": compression, "axial_capacity_tension_kip": tension}
    if axial_kip > max(compression, 1e-9):
        return {**result, "dcr": axial_dcr, "basis": "axial_compression_above_cap", "domain_failure": "compression"}
    if axial_kip < tension:
        return {**result, "dcr": axial_dcr, "basis": "axial_tension_outside_surface", "domain_failure": "tension"}
    if abs(my_kip_in) < 1e-4 and abs(mz_kip_in) < 1e-4:
        return {**result, "dcr": axial_dcr, "basis": "axial"}
    phi_y = _interpolate_pm_capacity(axial_kip, diagram_y)
    phi_z = _interpolate_pm_capacity(axial_kip, diagram_z)
    if phi_y is None or phi_y < 1e-6 or phi_z is None or phi_z < 1e-6:
        return {**result, "dcr": DOMAIN_FAILURE_DCR, "basis": "no_flexural_strength_at_axial_load",
                "domain_failure": "moment_at_axial_end", "phi_mn_y_kip_in": phi_y, "phi_mn_z_kip_in": phi_z}
    contour = ((abs(my_kip_in) / phi_y) ** alpha + (abs(mz_kip_in) / phi_z) ** alpha) ** (1.0 / alpha)
    return {**result, "dcr": max(contour, axial_dcr), "basis": "biaxial_load_contour", "phi_mn_y_kip_in": phi_y,
            "phi_mn_z_kip_in": phi_z}


def column_pm_dcr_max(diagrams, rows, alpha=1.5):
    """Largest column_pm_dcr over demand rows [(tag, combination, P, My, Mz, ...)]: (dcr, row index)."""
    envelope = column_pm_envelope(diagrams, rows, alpha)
    return envelope["dcr"], envelope["index"]


def column_pm_envelope(diagrams, rows, alpha=1.5):
    """column_pm_dcr over demand rows [(tag, combination, P, My, Mz, ...)], evaluated together.

    The same interaction and the same domain rules as column_pm_dcr (the tests hold the two equal row by
    row), vectorised so a candidate cage can be priced on every member and combination of its group.
    Returns {"dcr": largest, "index": its row, "values": every row's DCR, "domain_failures": {"compression",
    "tension", "moment_at_axial_end": row counts}, "outside_domain": True when any row is a domain failure}.
    """
    import numpy as np
    p = np.array([row[2] for row in rows], dtype=float)
    my = np.abs(np.array([row[3] for row in rows], dtype=float))
    mz = np.abs(np.array([row[4] for row in rows], dtype=float))

    def capacity(diagram):
        points = sorted(diagram, key=lambda point: -point[0])
        best = np.full(p.shape, -np.inf)
        for (p1, m1), (p2, m2) in zip(points, points[1:]):
            inside = (p >= p2) & (p <= p1)
            if not inside.any():
                continue
            span = p1 - p2
            t = (p - p2) / span if span > 1e-9 else np.zeros_like(p)
            best = np.where(inside, np.maximum(best, m2 + t * (m1 - m2)), best)
        return best

    phi_y, phi_z = capacity(diagrams["y"]), capacity(diagrams["z"])
    compression, tension = axial_limits(diagrams["y"], diagrams["z"])
    cap = compression if compression > 1e-6 else 1e-9
    # The ratio of each load to the capacity of its own sign (axial_demand_ratio, row by row).
    in_tension = p < 0.0
    tension_ratio = p / tension if tension < -1e-6 else np.full(p.shape, DOMAIN_FAILURE_DCR)
    axial = np.where(in_tension, tension_ratio, p / cap)
    above = p > max(compression, 1e-9)
    below = p < tension
    no_moment = (my < 1e-4) & (mz < 1e-4)
    axial_only = above | below | no_moment
    no_flexure = ~axial_only & (~np.isfinite(phi_y) | (phi_y < 1e-6) | ~np.isfinite(phi_z) | (phi_z < 1e-6))
    safe_y = np.where(np.isfinite(phi_y) & (phi_y >= 1e-6), phi_y, 1.0)
    safe_z = np.where(np.isfinite(phi_z) & (phi_z >= 1e-6), phi_z, 1.0)
    contour = ((my / safe_y) ** alpha + (mz / safe_z) ** alpha) ** (1.0 / alpha)
    dcr = np.where(axial_only, axial, np.where(no_flexure, DOMAIN_FAILURE_DCR, np.maximum(contour, axial)))
    index = int(np.argmax(dcr))
    failures = {"compression": int(above.sum()), "tension": int(below.sum()), "moment_at_axial_end": int(no_flexure.sum())}
    return {"dcr": float(dcr[index]), "index": index, "values": [float(v) for v in dcr], "domain_failures": failures,
            "outside_domain": any(failures.values())}


def column_demand_rows(tags, actions):
    """[(tag, combination id, P, My, Mz, Vy, Vz)] of the given columns in every solved combination."""
    rows = []
    for action in actions:
        members = action["members"]
        for tag in tags:
            rows.append((tag, action["id"], *column_force_components(members[str(tag)]["local_force_kip_kipin"])))
    return rows


def column_group_strength(design, rows, alpha=1.5):
    """Governing P-M and analysis-shear DCR of one column design over its group's demand rows."""
    diagrams = column_pm_diagrams(design)
    cover = sp.longitudinal_cover_in("column", design.bar_size, design.stirrup_bar_size)
    av = design.stirrup_legs * sp.rebar_area(design.stirrup_bar_size)
    best_pm, best_shear = None, None
    for tag, combination, p, my, mz, vy, vz in rows:
        pm = column_pm_dcr(diagrams, p, my, mz, alpha)
        if best_pm is None or pm["dcr"] > best_pm["dcr"]:
            best_pm = {**pm, "member_tag": tag, "combination": combination, "axial_kip": p, "my_kip_in": my, "mz_kip_in": mz}
        vu = math.hypot(vy, vz)
        shear = check_shear(vu, p, bw=design.b_in, d=design.h_in - cover, fc_ksi=design.fc_ksi, Av=av,
                            s=design.stirrup_spacing_in, fy_ksi=sp.FY_KSI, h=design.h_in)
        if best_shear is None or shear.dcr > best_shear["dcr"]:
            best_shear = {"dcr": shear.dcr, "vu_kip": vu, "phi_vn_kip": shear.capacity, "member_tag": tag,
                          "combination": combination, "axial_kip": p}
    ast = design.longitudinal_area_in2
    return {"pm": best_pm, "shear": best_shear, "rho": ast / (design.b_in * design.h_in), "ast_in2": ast,
            "observations": len(rows)}


# ---- beams --------------------------------------------------------------------------------------------------
def beam_demand_rows(tags, actions):
    """[(tag, combination id, Mu+, Mu-, Vu)] of the given beams: each span's own sagging and hogging envelope."""
    rows = []
    for action in actions:
        members = action["members"]
        for tag in tags:
            member = members[str(tag)]
            envelope = member["span_bending"]["full_span"]
            f = member["local_force_kip_kipin"]
            vu = math.hypot(max(abs(f[1]), abs(f[7])), max(abs(f[2]), abs(f[8])))
            rows.append((tag, action["id"], envelope["mu_positive_kip_in"], envelope["mu_negative_kip_in"], vu))
    return rows


def beam_flexural_strengths(design, rows):
    """phi Mn (sagging, hogging) and the effective depths of one beam design on its bar rows."""
    ab, fy, fc, b = sp.rebar_area(design.bar_size), sp.FY_KSI, design.fc_ksi, design.b_in
    result = {}
    for face, count, name in (("bottom", design.bot_bars, "positive"), ("top", design.top_bars, "negative")):
        d = design.h_in - rows[face]["centroid_in"]
        steel = count * ab
        a = steel * fy / (0.85 * fc * b)
        fc_psi, fy_psi = fc * 1000.0, fy * 1000.0
        as_min = max(3.0 * math.sqrt(fc_psi) / fy_psi, 200.0 / fy_psi) * b * d
        result[name] = {"phi_mn_kip_in": max(PHI_FLEX * steel * fy * (d - a / 2.0), 1e-9), "d_in": d, "as_in2": steel,
                        "as_min_in2": as_min, "min_controlled": steel <= as_min * 1.01}
    return result


def beam_group_strength(design, rows, bar_rows):
    """Governing flexure (each sign) and analysis-shear DCR of one beam design over its group's demand rows."""
    strengths = beam_flexural_strengths(design, bar_rows)
    d = min(strengths["positive"]["d_in"], strengths["negative"]["d_in"])
    av = design.stirrup_legs * sp.rebar_area(design.stirrup_bar_size)
    best = {"positive": None, "negative": None, "shear": None}
    for tag, combination, mu_pos, mu_neg, vu in rows:
        for sign, mu in (("positive", mu_pos), ("negative", mu_neg)):
            dcr = mu / strengths[sign]["phi_mn_kip_in"] if mu > 0.0 else 0.0
            if best[sign] is None or dcr > best[sign]["dcr"]:
                best[sign] = {"dcr": dcr, "mu_kip_in": mu, "phi_mn_kip_in": strengths[sign]["phi_mn_kip_in"],
                              "member_tag": tag, "combination": combination}
        shear = check_shear(vu, 0.0, bw=design.b_in, d=d, fc_ksi=design.fc_ksi, Av=av, s=design.stirrup_spacing_in,
                            fy_ksi=sp.FY_KSI)
        if best["shear"] is None or shear.dcr > best["shear"]["dcr"]:
            best["shear"] = {"dcr": shear.dcr, "vu_kip": vu, "phi_vn_kip": shear.capacity, "member_tag": tag,
                             "combination": combination}
    return {"flexure_positive": best["positive"], "flexure_negative": best["negative"], "shear": best["shear"],
            "strengths": strengths, "observations": len(rows)}


# ---- every group of the installed design ----------------------------------------------------------------------
def group_tags(state=None):
    state = state or mg.active()
    return {gid: list(group["member_tags"]) for gid, group in state.groups.items()}


def strength_checks(actions, cfg, state=None):
    """Member strength results of every group of the installed design, from solved combination actions.

    Returns {"groups": {gid: {...}}, "worst": {"column", "beam"}, "column_flexure_axial", ...}. A group's
    ``dcr`` values are those of its governing member and combination, which are named.
    """
    state = state or mg.active()
    if state is None:
        raise mg.GroupedStateError("Group strength checks need an installed grouped design.")
    alpha = getattr(cfg.dcr, "biaxial_contour_exponent", 1.5)
    arrangement = bar_layers.arrangement()
    groups, worst = {}, {"column": 0.0, "beam": 0.0}
    for gid, tags in group_tags(state).items():
        design = state.designs[gid]
        if design.is_column:
            result = column_group_strength(design, column_demand_rows(tags, actions), alpha)
            worst["column"] = max(worst["column"], result["pm"]["dcr"], result["shear"]["dcr"])
            groups[gid] = {"member_type": "column", **result}
        else:
            result = beam_group_strength(design, beam_demand_rows(tags, actions), arrangement[gid])
            worst["beam"] = max(worst["beam"], result["flexure_positive"]["dcr"], result["flexure_negative"]["dcr"],
                                result["shear"]["dcr"])
            groups[gid] = {"member_type": design.member_type, **result}
    return {"groups": groups, "worst": worst,
            "basis": ("every member on its own group's section and cage; columns: concurrent P, My, Mz of each combination "
                      "with the biaxial load-contour interaction; beams: each span's own sagging and hogging envelope; a "
                      "group's value is its governing member and combination")}
