"""Solved member actions and member strength checks for a grouped design (2026-10-02).

The uniform checks (Design.ACI_Checks) read one column and one beam from cfg and Structure_Parameters.
Here every member is checked on its own group's section and cage in every solved
combination. Column checks conservatively retain the componentwise end moment
and shear envelope at BOTH centerline axial ends; this is not a concurrent
joint-face P-M tuple. Beam moments use their own span envelope. A group's result
names its governing physical member, combination and axial end.

The grouped slab-aware checks use:
  columns  rotated strain compatibility with actual bars and biaxial tension-strain
           phi; the tied-column axial cap; analysis shear with ACI_Checks.check_shear.
           Legacy no-slab adapters retain the empirical load contour.
  beams    member-resolved T/L section strengths, actual bar rows and developed slab mats;
           undeveloped exterior mats are excluded and the weakest end/span section is used
           against the full-span envelope; analysis shear uses check_shear on the smaller depth.
"""
from __future__ import annotations

import math

import openseespy.opensees as ops

import Structure_Parameters as sp
from Design.ACI_Checks import (DOMAIN_FAILURE_DCR, PHI_FLEX, _interpolate_pm_capacity, axial_demand_ratio, axial_limits,
                               check_shear, pm_diagram_for)
from Design.SMRF_Beam_Steel import beam_face_steel_limits
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
    """Uniaxial reporting curves and the actual-cage biaxial section used for checks.

    In slab/actual-cage mode selection and final checking use rotated strain
    compatibility, including phi at the actual biaxial strain state. The
    legacy no-slab adapter retains its historical load-contour method.
    """
    layers_y = _design_col_steel_layers(design, about="h")
    layers_z = _design_col_steel_layers(design, about="b")
    key = (design.b_in, design.h_in, design.fc_ksi, tuple(layers_y), tuple(layers_z), sp.FY_KSI, sp.ES_KSI,
           sp.SLAB_THICKNESS_IN is not None)
    if key not in _DIAGRAMS:
        if len(_DIAGRAMS) > 2048:
            _DIAGRAMS.clear()
        _DIAGRAMS[key] = {"y": pm_diagram_for(design.b_in, design.h_in, design.fc_ksi, sp.FY_KSI, sp.ES_KSI, layers_y),
                          "z": pm_diagram_for(design.h_in, design.b_in, design.fc_ksi, sp.FY_KSI, sp.ES_KSI, layers_z)}
        if sp.SLAB_THICKNESS_IN is not None:
            from Design.SMRF_Biaxial_Section import BiaxialSection
            from Design.SMRF_Transitions import bar_coordinates
            bars = [(p["x_in"], p["y_in"], design.bar_area_in2)
                    for p in bar_coordinates(design, sp.COL_CLEAR_COVER_IN)]
            _DIAGRAMS[key]["biaxial_section"] = BiaxialSection(
                design.b_in, design.h_in, design.fc_ksi, sp.FY_KSI, bars, sp.ES_KSI)
    return _DIAGRAMS[key]


def column_force_components(local_force):
    """(P, My, Mz, Vy, Vz) of one column in one combination: the end-i axial load with the larger end moment and
    shear of each local axis (RC_Design_Check._extract_forces, the uniform check's reading)."""
    f = [float(v) for v in local_force]
    return f[0], max(abs(f[4]), abs(f[10])), max(abs(f[5]), abs(f[11])), max(abs(f[1]), abs(f[7])), max(abs(f[2]), abs(f[8]))


def column_pm_dcr(diagrams, axial_kip, my_kip_in, mz_kip_in, alpha=1.5):
    """Actual-cage strain-compatible DCR, or the legacy no-slab load contour.

    Returns {"dcr", "basis", "domain_failure", "phi_mn_y_kip_in", "phi_mn_z_kip_in", "axial_dcr"}. The axial
    load is placed against BOTH ends of the surface before anything else (ACI_Checks.axial_limits,
    axial_demand_ratio): above the phi Pn,max cap it is an axial compression failure whatever the moment;
    below the pure-tension end it is an axial tension failure whatever the moment. Both give the
    non-negative ratio of the load to the capacity of its own sign (above 1) and name the failure under
    ``domain_failure``. Inside the surface a load with no moment gives that same ratio (at most 1); a
    moment at an axial load where the surface has no flexural strength left (the tension end) is the
    named failure ``moment_at_axial_end`` with the reserved value DOMAIN_FAILURE_DCR.
    """
    section = diagrams.get("biaxial_section")
    if section is not None:
        values = _biaxial_envelope(section, [(0, "scalar", axial_kip, my_kip_in, mz_kip_in, 0., 0.)])
        result = {"dcr": values["dcr"], "basis": values["basis"], "axial_dcr": values["axial_values"][0],
                  "axial_capacity_compression_kip": section.compression_cap,
                  "axial_capacity_tension_kip": section.tension_cap, "domain_failure": values["row_domain_failures"][0],
                  "phi_mn_y_kip_in": None, "phi_mn_z_kip_in": None}
        result["section_method"] = values["basis"]
        if result["domain_failure"] is not None:
            result["basis"] = {"compression": "axial_compression_above_cap",
                               "tension": "axial_tension_outside_surface",
                               "moment_at_axial_end": "no_flexural_strength_at_axial_load"}[result["domain_failure"]]
        if result["domain_failure"] is None and abs(my_kip_in) + abs(mz_kip_in) >= 1e-4:
            result["phi_mn_y_kip_in"] = float(section.at_axial(axial_kip, math.pi / 2)["mh"])
            result["phi_mn_z_kip_in"] = float(section.at_axial(axial_kip, 0.)["mb"])
            result["phi_mn_demand_ray_kip_in"] = values["ray_capacities"][0]
        return result
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
    if diagrams.get("biaxial_section") is not None:
        return _biaxial_envelope(diagrams["biaxial_section"], rows)
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


def _biaxial_envelope(section, rows):
    """Vectorized actual strain-compatible capacities, used identically by selection and final checks."""
    import numpy as np
    from Design.SMRF_Biaxial_Section import METHOD
    p=np.asarray([row[2] for row in rows],dtype=float)
    mh=np.abs([row[3] for row in rows]);mb=np.abs([row[4] for row in rows])
    axial=np.where(p<0.,p/section.tension_cap,p/section.compression_cap)
    above=p>section.compression_cap;below=p<section.tension_cap
    moment=np.hypot(mh,mb)
    zero=moment<1e-4
    tension_end=(p<=section.tension_cap+1e-8)&~zero&~below
    valid=~above&~below&~zero&~tension_end
    dcr=axial.copy();capacities=np.zeros(p.shape)
    dcr[tension_end]=DOMAIN_FAILURE_DCR
    if valid.any():
        result=section.ray_capacity(p[valid],mh[valid],mb[valid])
        capacities[valid]=result["capacity"]
        dcr[valid]=np.maximum(axial[valid],result["dcr"])
    index=int(np.argmax(dcr))
    failures={"compression":int(above.sum()),"tension":int(below.sum()),"moment_at_axial_end":int(tension_end.sum())}
    return {"dcr":float(dcr[index]),"index":index,"values":dcr.tolist(),"domain_failures":failures,
            "outside_domain":any(failures.values()),"basis":METHOD,"axial_values":axial.tolist(),
            "ray_capacities":capacities.tolist(),"row_domain_failures":[
                "compression" if a else "tension" if b else "moment_at_axial_end" if t else None
                for a,b,t in zip(above,below,tension_end)]}


def column_pm_selection_screen(diagrams, rows, limit, alpha=1.5):
    """Reject on an exact failing witness; check every row before permitting a cage.

    The old contour only orders probes. It cannot accept a cage, supply a
    reported final DCR, or omit a row from a passing result. Most undersized
    cages can be rejected after a few exact section solves.
    """
    section = diagrams.get("biaxial_section")
    if section is None or len(rows) <= 6:
        result = column_pm_envelope(diagrams, rows, alpha)
        return {**result, "all_rows_checked": True}
    approximate = column_pm_envelope({axis: diagrams[axis] for axis in ("y", "z")}, rows, alpha)
    order = sorted(range(len(rows)), key=lambda i: approximate["values"][i], reverse=True)[:4]
    order += [min(range(len(rows)), key=lambda i: rows[i][2]), max(range(len(rows)), key=lambda i: rows[i][2])]
    order = list(dict.fromkeys(order))
    probe = _biaxial_envelope(section, [rows[i] for i in order])
    if probe["outside_domain"] or probe["dcr"] > limit:
        return {**probe, "index": order[probe["index"]], "all_rows_checked": False,
                "checked_row_indices": order, "scope": "exact failing witness for rejection only; not a final envelope"}
    result = _biaxial_envelope(section, rows)
    return {**result, "all_rows_checked": True}


def column_demand_rows(tags, actions):
    """[(tag, combination id, P, My, Mz, Vy, Vz, axial_end)] for BOTH column ends.

    Compression is f[0] at i and -f[6] at j. Column self-weight makes them
    different; greater compression is not always more adverse on a P-M curve.
    Retain the conservative componentwise end-moment/shear envelope at each P,
    so the old bottom check remains and the omitted top check is added. These
    are centerline action envelopes, not a claim of concurrent face actions.
    SCWB retains its separate joint-face transport and axial-range calculation.
    """
    rows = []
    for action in actions:
        members = action["members"]
        for tag in tags:
            forces = members[str(tag)]["local_force_kip_kipin"]
            _p, *envelope = column_force_components(forces)
            for end, axial in (("i", float(forces[0])), ("j", -float(forces[6]))):
                rows.append((tag, action["id"], axial, *envelope, end))
    return rows


def column_group_strength(design, rows, alpha=1.5):
    """Governing P-M and analysis-shear DCR of one column design over its group's demand rows."""
    diagrams = column_pm_diagrams(design)
    cover = sp.longitudinal_cover_in("column", design.bar_size, design.stirrup_bar_size)
    av = design.stirrup_legs * sp.rebar_area(design.stirrup_bar_size)
    best_pm, best_shear = None, None
    envelope = column_pm_envelope(diagrams, rows, alpha)
    for index, (tag, combination, p, my, mz, vy, vz, *end_context) in enumerate(rows):
        axial_end = end_context[0] if end_context else None
        if index == envelope["index"]:
            pm = column_pm_dcr(diagrams, p, my, mz, alpha)
            best_pm = {**pm, "member_tag": tag, "combination": combination, "axial_kip": p, "my_kip_in": my, "mz_kip_in": mz,
                       "axial_end": axial_end, "action_basis": "centerline_component_envelope_at_both_axial_ends_v2"}
        vu = math.hypot(vy, vz)
        shear = check_shear(vu, p, bw=design.b_in, d=design.h_in - cover, fc_ksi=design.fc_ksi, Av=av,
                            s=design.stirrup_spacing_in, fy_ksi=sp.FYT_KSI, h=design.h_in)
        if best_shear is None or shear.dcr > best_shear["dcr"]:
            best_shear = {"dcr": shear.dcr, "vu_kip": vu, "phi_vn_kip": shear.capacity, "member_tag": tag,
                          "combination": combination, "axial_kip": p, "axial_end": axial_end}
    ast = design.longitudinal_area_in2
    return {"pm": best_pm, "shear": best_shear, "rho": ast / (design.b_in * design.h_in), "ast_in2": ast,
            "observations": len(rows)}


# ---- beams --------------------------------------------------------------------------------------------------
def beam_demand_rows(tags, actions):
    """[(tag, id, Mu+, Mu-, Vu, Nmin, Nmax)] for each member/combination.

    Compression-positive external end N is paired with that combination's
    moment envelope. The internal web/flange flexural couple is not added.
    """
    rows = []
    for action in actions:
        members = action["members"]
        for tag in tags:
            member = members[str(tag)]
            envelope = member["span_bending"]["full_span"]
            f = member["local_force_kip_kipin"]
            if str(member['span_bending'].get('method_version','')).startswith('physical_piecewise_section_force_extrema_'):
                from Design.SMRF_Physical_Member_Actions import validated_member_span
                envelope=validated_member_span(member)['full_span']
                vu=math.hypot(envelope['maximum_abs_inplane_shear_kip'],envelope['maximum_abs_vertical_shear_kip'])
                axial=(envelope['axial_min_kip'],envelope['axial_max_kip'])
            else:
                vu = math.hypot(max(abs(f[1]), abs(f[7])), max(abs(f[2]), abs(f[8])))
                axial = (float(f[0]), -float(f[6]))
            # What the plate carries across the flange, the torsion steel taken from the faces and the collector force
            # enter the section equilibrium here (Design.SMRF_Conventional_Floor, 2026-11-07).
            floor = member.get("floor_actions") or {}
            face = float(floor.get("torsion_face_moment_kip_in", 0.0))
            extra = float(floor.get("n_diaphragm_kip", 0.0))
            rows.append((tag, action["id"], envelope["mu_positive_kip_in"] + float(floor.get("mu_slab_sagging_kip_in", 0.0)) + face,
                         envelope["mu_negative_kip_in"] + float(floor.get("mu_slab_hogging_kip_in", 0.0)) + face, vu,
                         min(axial) - extra, max(axial) + extra))
    return rows


def beam_flexural_strengths(design, rows):
    """Web-only reference and web-steel minimums; production slab-aware checks use the resolved helper below."""
    ab, fy, fc, b = sp.rebar_area(design.bar_size), sp.FY_KSI, design.fc_ksi, design.b_in
    result = {}
    for face, count, name in (("bottom", design.bot_bars, "positive"), ("top", design.top_bars, "negative")):
        limits = beam_face_steel_limits(b_in=b, h_in=design.h_in, fc_ksi=fc, fy_ksi=fy,
                                       bar_area_in2=ab, count=count, rows=rows[face])
        d, steel, as_min = limits["d_in"], limits["as_in2"], limits["as_min_in2"]
        a = steel * fy / (0.85 * fc * b)
        result[name] = {"phi_mn_kip_in": max(PHI_FLEX * steel * fy * (d - a / 2.0), 1e-9), "d_in": d, "as_in2": steel,
                        "as_min_in2": as_min, "min_controlled": steel <= as_min * 1.01}
    return result


BEAM_STRENGTH_BASIS = "member_composite_t_l_actual_rows_developed_mats_v3_net_concrete"


def beam_section_context(member):
    """Geometry/anchorage equivalence key within an installed group's fixed cage and slab layout.

    Used only in local call caches, never across changed designs or slab layouts. Member tags are
    deliberately absent: an exterior end, a different adjacent web, or a different joint core changes
    the key even where the members belong to the same design group.
    """
    if sp.SLAB_THICKNESS_IN is None:
        return ("no_slab",)
    flange = mp.effective_flange(member)
    edge = mp.perimeter_beam(member).design
    # Each exterior end's own perimeter beam, joint depth and corner hook decide its datums (2026-11-01).
    hooks = tuple((end, tuple((key, tuple(sorted(value.items())) if isinstance(value, dict) else value)
                              for key, value in sorted(inputs.items())))
                  for end, inputs in sorted(mp.exterior_end_anchorage_inputs(member).items()))
    from Design import SMRF_Conventional_Detail as conventional
    inventory=None
    if (sp.SLAB_REINFORCEMENT or {}).get("layout") and not conventional.active():
        from Design.SMRF_Joint_Mat_Physical import member_slab_inventory
        inventory=member_slab_inventory(member,flange)
    physical_key=(inventory["status"],tuple((face,layer["count"],layer["depth_in"]) for face,layer in inventory["layers"].items())) if inventory else None
    return (flange["flange_width_in"], flange["clear_span_in"], flange["slab_sides"],
            tuple((o["overhang_in"], o["clear_to_adjacent_web_in"]) for o in flange["overhangs"]),
            tuple(mp.exterior_ends(member).items()), edge.b_in, edge.fc_ksi, edge.stirrup_bar_size, hooks,physical_key)


def beam_group_representatives(state, gid):
    """One member tag per distinct flange/anchorage context, for pricing candidate cages efficiently."""
    representatives = {}
    for tag in state.groups[gid]["member_tags"]:
        representatives.setdefault(beam_section_context(state.member(tag)), tag)
    return tuple(representatives.values())


def resolved_beam_flexural_strengths(member, rows):
    """Design strength of the same section used by capacity design, at each end and within the span.

    The available action record contains full-span maxima, not moments paired with their x coordinate.
    Applying the weakest available section to that envelope is conservative; exterior anchorage is
    never inferred from an interior section. Web minimum/maximum-steel rules remain on the web bars.
    An exterior end's mats count only where its hook develops at both datums (the support face and the
    joint face; S, 2026-11-01). For the 18.6.3.2 checks the negative strength at each face on the
    strong-column basis (either datum, the larger) is given beside it (``nominal_negative_faces_18_6_3_2``).
    """
    design = member.design
    result = beam_flexural_strengths(design, rows)
    if sp.SLAB_THICKNESS_IN is None:
        return result
    section = mp.beam_strengths(member)
    family = section["family"]
    by_end = section["basis"]["exterior_anchorage_by_end"]
    nominal_sections = {}
    for where in ("i", "j", "span"):
        undeveloped = where in by_end and not by_end[where]["developed_both"]
        nominal_sections[where] = family["undeveloped" if undeveloped else "composite"]
    for sign in ("positive", "negative"):
        choices = []
        for where, sections in nominal_sections.items():
            value = sections[sign]
            # Table 21.2.2, nonprestressed tied section: do not grant the tension-controlled
            # phi to a composite section solely because its web-only steel ratio was small.
            strain = value["tension_strain"]
            transition = min(1.0, max(0.0, (strain - sp.FY_KSI / sp.ES_KSI) / 0.003))
            phi = 0.65 + (PHI_FLEX - 0.65) * transition
            choices.append({"section": where, "phi": phi, "mn_kip_in": value["mn_kip_in"],
                            "phi_mn_kip_in": phi * value["mn_kip_in"], "tension_strain": strain,
                            "tension_controlled": value["tension_controlled"]})
        governing = min(choices, key=lambda c: c["phi_mn_kip_in"])
        result[sign].update({**governing, "basis": BEAM_STRENGTH_BASIS,
                             "tension_controlled": all(c["tension_controlled"] for c in choices),
                             "effective_flange_width_in": family["effective_flange_width_in"],
                             "slab_sides": family["slab_sides"], "section_capacities": choices})
    result["nominal_sections"] = {where: {sign: values[sign]["mn_kip_in"] for sign in ("positive", "negative")}
                                  for where, values in nominal_sections.items()}
    result["nominal_negative_faces_18_6_3_2"] = {
        where: family["undeveloped" if (where in by_end and not by_end[where]["developed_either"]) else "composite"]
        ["negative"]["mn_kip_in"] for where in ("i", "j")}
    result["basis"] = section["basis"]
    return result


def beam_group_strength(design, rows, bar_rows, state=None):
    """Governing flexure (each sign) and analysis-shear DCR of one beam design over its group's demand rows."""
    from Design.SMRF_Beam_Axial import installed_member_sections, capacity_at_axial, METHOD as AXIAL_METHOD
    import numpy as np
    strengths = beam_flexural_strengths(design, bar_rows)
    cache, by_member = {}, {}
    # Solve all external axial ordinates of each physical section family in
    # one vector call. The legacy five-field helper rows imply N=0; production
    # capture always supplies both axial ends and uses the exact section path.
    axial_by_context, axial_capacity, member_context = {}, {}, {}
    if state is not None:
        for tag, _combination, _mu_pos, _mu_neg, _vu, *axial in rows:
            if not axial:
                continue
            member = state.member(tag)
            key = member_context.setdefault(tag, beam_section_context(member))
            entry = axial_by_context.setdefault(key, {'member': member, 'values': set()})
            entry['values'].update(axial)
        for key, entry in axial_by_context.items():
            ps = sorted(entry['values'])
            values = capacity_at_axial(installed_member_sections(entry['member'], bar_rows), ps)
            axial_capacity[key] = {p: {
                'positive': float(values['positive_kip_in'][i]), 'negative': float(values['negative_kip_in'][i]),
                'domain_failure': bool(values['domain_failure'][i]),
                'tension_controlled': bool(values['tension_controlled'][i]),
                'section': {s: str(values['section'][s][i]) for s in ('positive', 'negative')},
                'phi': {s: float(values['phi'][s][i]) for s in ('positive', 'negative')},
            } for i, p in enumerate(ps)}
    av = design.stirrup_legs * sp.rebar_area(design.stirrup_bar_size)
    best = {"positive": None, "negative": None, "shear": None}
    for tag, combination, mu_pos, mu_neg, vu, *axial in rows:
        if state is not None and tag not in by_member:
            member = state.member(tag)
            key = beam_section_context(member)
            if key not in cache:
                cache[key] = resolved_beam_flexural_strengths(member, bar_rows)
            by_member[tag] = cache[key]
        current = by_member[tag] if state is not None else strengths
        d = min(current["positive"]["d_in"], current["negative"]["d_in"])
        for sign, mu in (("positive", mu_pos), ("negative", mu_neg)):
            cap = current[sign]['phi_mn_kip_in']
            controlled = current[sign].get('tension_controlled', True)
            axial_governor, axial_result = None, None
            domain = False
            if state is not None and axial:
                table = axial_capacity[member_context[tag]]
                axial_governor = min(axial, key=lambda p: -float('inf') if table[p]['domain_failure']
                                     or not table[p]['tension_controlled'] else table[p][sign])
                axial_result = table[axial_governor]
                domain = axial_result['domain_failure']
                cap = axial_result[sign] if np.isfinite(axial_result[sign]) and not domain else 0.
                controlled = axial_result['tension_controlled']
            dcr = mu / max(cap, 1e-9) if mu > 0.0 else 0.0
            if domain or not controlled:
                # The zero-axial-force beam section must satisfy 9.3.3.1. Reducing phi alone
                # cannot make a non-tension-controlled composite beam an accepted design.
                dcr = max(dcr, DOMAIN_FAILURE_DCR)
            if best[sign] is None or dcr > best[sign]["dcr"]:
                best[sign] = {"dcr": dcr, "mu_kip_in": mu, "phi_mn_kip_in": cap,
                              "member_tag": tag, "combination": combination,
                              "strength_basis": AXIAL_METHOD if axial_result else current[sign].get("basis", "rectangular_no_slab"),
                              "tension_controlled": controlled, 'axial_kip': axial_governor,
                              'axial_domain_failure': domain,
                              "section": axial_result['section'][sign] if axial_result else current[sign].get("section")}
        # Minimum compression / greatest tension is the adverse axial term.
        # Do not take compression credit from a composite area in a web shear check.
        shear_axial = min(0., *axial) if axial else 0.
        shear = check_shear(vu, shear_axial, bw=design.b_in, d=d, fc_ksi=design.fc_ksi, Av=av, s=design.stirrup_spacing_in,
                            fy_ksi=sp.FYT_KSI, h=design.h_in)
        if best["shear"] is None or shear.dcr > best["shear"]["dcr"]:
            best["shear"] = {"dcr": shear.dcr, "vu_kip": vu, "phi_vn_kip": shear.capacity, "member_tag": tag,
                             "combination": combination, 'axial_kip': shear_axial}
    return {"flexure_positive": best["positive"], "flexure_negative": best["negative"], "shear": best["shear"],
            "strengths": ({sign: by_member[best[sign]["member_tag"]][sign] for sign in ("positive", "negative")}
                          if by_member else strengths), "observations": len(rows),
            "strength_basis": BEAM_STRENGTH_BASIS if state is not None and sp.SLAB_THICKNESS_IN is not None else "rectangular_no_slab"}


def beam_detailing_strengths(member, rows):
    """Nominal (not phi-reduced) end/span strengths for the 18.6.3.2 detailing checks.

    Each side of a comparison on its more demanding basis (2026-11-01): the negative strength at a face is the
    larger of the sizing and strong-column values, the positive face and along-span strengths the sizing ones.
    """
    values = resolved_beam_flexural_strengths(member, rows)
    sections = values.get("nominal_sections") or {
        end: {sign: values[sign]["phi_mn_kip_in"] / PHI_FLEX for sign in ("positive", "negative")}
        for end in ("i", "j", "span")}
    faces = values.get("nominal_negative_faces_18_6_3_2") or {}
    return {**{f"mn_{sign}_{label}_kipin": (max(sections[end][sign], faces.get(end, 0.0)) if sign == "negative"
                                             else sections[end][sign])
               for end, label in (("i", "left"), ("j", "right")) for sign in ("positive", "negative")},
            **{f"mn_{sign}_min_along_kipin": min(s[sign] for s in sections.values())
               for sign in ("positive", "negative")}}


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
            result = beam_group_strength(design, beam_demand_rows(tags, actions), arrangement[gid], state)
            worst["beam"] = max(worst["beam"], result["flexure_positive"]["dcr"], result["flexure_negative"]["dcr"],
                                result["shear"]["dcr"])
            groups[gid] = {"member_type": design.member_type, **result}
    return {"groups": groups, "worst": worst,
            "basis": ("every member on its own group's section and cage; columns: each combination's componentwise "
                      "centerline moment/shear envelope checked at both end axial forces with rotated biaxial strain "
                      "compatibility in actual-cage mode (legacy load contour without a slab); beams: each span's own sagging and hogging envelope; a "
                      "group's value is its governing member and combination")}
