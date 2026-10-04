"""Physical SMRF joint inventory and solved-axial SCWB evidence adapter.

This is deliberately independent of OpenSees and mutable model globals.
``build_joint_evidence(record, combination_actions,
expected_combination_ids=...)`` consumes the saved geometry/section/rebar
record and every final-pass factored load case. Each action entry is
``{'id': str, 'analysis_succeeded': True, 'axial_reference': 'joint_faces', 'members': {tag:
{'axial_i_kip': P_i, 'axial_j_kip': P_j}}}``; axial compression is positive
at BOTH joint faces, after the caller's distributed-load cut correction.
``local_force_axial_actions`` converts an explicit OpenSees
12-component *localForce* response (never an unlabelled global eleForce).

The returned ``joints`` and ``through_bar_anchorage`` groups can be passed
to SMRF_Joints.evaluate_joints. Also retain ``inventory`` and
``completeness_checks``. The latter are mandatory evidence, not an optional
display: missing/failed load cases cannot silently become an axial envelope.

Section capacities use actual equally spaced perimeter bars, including the
intermediate side bars, separately in both frame planes and both signs.
ACI 318-19 22.2 strain compatibility uses ecu=.003, elastic-perfectly-plastic
steel and the equivalent rectangular concrete block, with displaced concrete
removed at steel layers. No phi or artificial .8P0 zero-moment point is put
into this nominal moment curve; no demand is clamped to its domain.
This uniaxial nominal capacity is NOT a biaxial member-strength check.

Slab strength is NEVER inferred to be zero. Optional
``record['beam_slab_strengths']['<tag>/<end>/<positive|negative>']`` entries
must explicitly contain slab_basis, slab_mn_kip_in and
section_compatibility_verified=True. The increment must come from compatible
beam-plus-slab versus beam-only sections, not As*fy times a guessed arm.
Positive/negative here denote sagging/hogging beam flexure, not sway.

All joints include the actual beams and columns, including the SINGLE roof
column. No roof exemption, development, column capacity shear, probable joint
shear, beam capacity shear or complete model qualification is inferred.

Primary basis: ACI 318-19 18.7.3.2, 18.8.2.3, 22.2.1--22.2.3:
https://www.ocf.berkeley.edu/~chiep/wp-content/uploads/2024/01/CE-123-ACI-318-19.pdf
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from Design.SMRF_Common import make_check


def _number(value, name, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean.")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite{' and positive' if positive else ''}.")
    return value


def _integer(value, name, minimum=0):
    result = _number(value, name)
    if result != int(result) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(result)


def local_force_axial_actions(forces):
    """Local resisting forces: compression = +N_i at i, -N_j at j.

    For a vertical element created bottom-to-top with a 10 kip downward
    joint load and no member load, localForce[0]=+10, [6]=-10. Do not take
    abs(): uplift/tension must remain negative. Under distributed selfweight
    P_i and P_j legitimately differ.
    """
    if not isinstance(forces, (list, tuple)) or len(forces) != 12:
        raise ValueError("An explicit 12-component localForce response is required.")
    values = [_number(v, "localForce component") for v in forces]
    return {"axial_i_kip": values[0], "axial_j_kip": -values[6]}


def physical_joint_inventory(geometry):
    """Exact all-grid-line topology, matching SMRF_Elastic.physical_members.

    Elevated floor-grid joints only: base support development is a separate
    check. Node/member tags follow the existing builder's creation order.
    Column section h lies in global X; b lies in global Y.
    """
    nx = _integer(geometry.get("num_bay_x"), "num_bay_x", 1)
    ny = _integer(geometry.get("num_bay_y"), "num_bay_y", 1)
    floors = _integer(geometry.get("num_floor"), "num_floor", 1)
    def node(k, i, j):
        return k * (nx + 1) * (ny + 1) + j * (nx + 1) + i + 1
    joints = {}
    for k in range(1, floors + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                tag = node(k, i, j)
                edges = int(i in (0, nx)) + int(j in (0, ny))
                joints[tag] = {"id": f"joint_{tag}", "node_tag": tag,
                               "floor": k, "grid_i": i, "grid_j": j,
                               "is_roof": k == floors,
                               "plan_type": ("interior", "edge", "corner")[edges],
                               "columns": [], "beams_x": [], "beams_y": []}
    members = []
    def add(ni, nj, kind):
        tag = len(members) + 1
        members.append({"tag": tag, "node_i": ni, "node_j": nj, "kind": kind})
        group = "columns" if kind == "column" else f"beams_{kind[-1]}"
        for n, end in ((ni, "i"), (nj, "j")):
            if n in joints:
                joints[n][group].append({"tag": tag, "end": end,
                                        "position": ("above" if end == "i" else "below")
                                        if kind == "column" else
                                        ("positive_side" if end == "i" else "negative_side")})
    for k in range(floors):
        for j in range(ny + 1):
            for i in range(nx + 1):
                add(node(k, i, j), node(k + 1, i, j), "column")
    for k in range(1, floors + 1):
        for j in range(ny + 1):
            for i in range(nx):
                add(node(k, i, j), node(k, i + 1, j), "beam_x")
    for k in range(1, floors + 1):
        for j in range(ny):
            for i in range(nx + 1):
                add(node(k, i, j), node(k, i, j + 1), "beam_y")
    return {"joints": list(joints.values()), "members": members,
            "elevated_joint_count": len(joints),
            "column_count": floors * (nx + 1) * (ny + 1),
            "beam_count": floors * (nx * (ny + 1) + ny * (nx + 1)),
            "base_connections_included": False,
            "basis": "all grid lines, fixed bases, no fictitious roof column"}


def section_bar_coordinates(record, member, axis=None):
    """Actual (area, local_y, local_z) perimeter bars; corners counted once.

    A beam whose record carries the staggered bar elevations
    (reinforcement.beam_bar_stacking.layers, 2026-09-27: the orthogonal cages
    stacked at the joints and, where the column lanes force it, a second layer)
    places that direction's rows at those elevations when ``axis`` is given, so
    the joint's beam capacities agree with the record's own strengths; a record
    from before the rule, or a call without an axis, keeps the nominal row.
    """
    if member not in ("column", "beam"):
        raise ValueError("member must be column or beam.")
    prefix = "col" if member == "column" else "beam"
    section, rebar = record["sections"], record["reinforcement"]
    b = _number(section[f"b_{prefix}_in"], "section width", positive=True)
    h = _number(section[f"h_{prefix}_in"], "section depth", positive=True)
    offset = rebar.get(f"{prefix}_longitudinal_centroid_offset_in",
                       rebar.get("longitudinal_centroid_offset_in"))
    cover = _number(offset, f"{prefix} longitudinal centroid offset", positive=True)
    if 2 * cover >= min(b, h):
        raise ValueError("Bar centroids must lie inside the section.")
    area = _number(rebar.get(f"{prefix}_bar_area_in2"), "bar area", positive=True)
    nt = _integer(rebar.get(f"{prefix}_top_bars"), "top bar count", 2)
    nb = _integer(rebar.get(f"{prefix}_bot_bars"), "bottom bar count", 2)
    ns = _integer(rebar.get(f"{prefix}_side_bars"), "side bar count", 0)
    y, z = b / 2 - cover, h / 2 - cover

    def spread(count, level):
        if count == 1:
            return [(area, 0.0, level)]
        return [(area, -y + 2 * y * n / (count - 1), level) for n in range(count)]

    staggered = ((rebar.get("beam_bar_stacking") or {}).get("layers") or {}).get(axis) if member == "beam" else None
    if staggered:
        from Design.SMRF_Beam_Slab_Strength import validate_bar_rows
        bars = []
        for face, sign, count in (("top", 1, nt), ("bottom", -1, nb)):
            # validated row by row: lengths, counts, declared layers and centroid, elevations inside the section
            for n, elevation in validate_bar_rows(staggered[face], count, f"{face} bars", h):
                bars += spread(n, sign * (h / 2 - elevation))
    else:
        bars = [bar for count, level in ((nt, z), (nb, -z)) for bar in spread(count, level)]
    bars += [(area, side, -z + 2 * z * n / (ns + 1))
             for side in (-y, y) for n in range(1, ns + 1)]
    return bars


def nominal_rectangular_capacity(*, width_in, depth_in, fc_ksi, fy_ksi,
                                 es_ksi, layers, axial_kip):
    """Strict nominal uniaxial strain-compatible section capacity.

    ``layers`` holds (As, distance from the selected compression face).
    Solves equilibrium within each continuous interval of the lumped-bar
    concrete-replacement model. Crossing a bar with the Whitney block makes
    a small force discontinuity, so a global bisection alone is unsafe.
    All valid roots are inspected; the smallest absolute Mn is retained.
    Demands outside the physical tension/compression domain, or in an
    unresolved lumped-layer discontinuity, raise rather than clamp/pass.
    """
    b = _number(width_in, "width", positive=True)
    h = _number(depth_in, "depth", positive=True)
    fc = _number(fc_ksi, "fc", positive=True)
    fy = _number(fy_ksi, "fy", positive=True)
    es = _number(es_ksi, "Es", positive=True)
    p = _number(axial_kip, "factored axial force")
    if fc < 2.5 or fy >= .003 * es:
        raise ValueError("Supported concrete fc >=2.5 ksi and steel fy < .003Es are required.")
    if not layers:
        raise ValueError("Explicit longitudinal bar layers are required.")
    steel = [(_number(a, "bar area", positive=True), _number(d, "bar depth"))
             for a, d in layers]
    if any(not 0 < d < h for _, d in steel):
        raise ValueError("Every bar centroid must lie strictly inside section depth.")
    ast = sum(a for a, _ in steel)
    if ast >= b * h:
        raise ValueError("Steel area must be below gross concrete area.")
    pmin, pmax = -fy * ast, .85 * fc * (b * h - ast) + fy * ast
    tolerance = 1e-10 * max(1, abs(pmin), abs(pmax))
    if p < pmin - tolerance or p > pmax + tolerance:
        from Design.SMRF_Common import SectionAxialDomainError
        raise SectionAxialDomainError(f"Axial force {p:g} outside nominal section domain [{pmin:g}, {pmax:g}].")
    beta = max(.65, min(.85, .85 - .05 * (fc - 4)))
    def state(c, included):
        a = min(beta * c, h)
        concrete = .85 * fc * b * a
        n, moment = concrete, concrete * (h / 2 - a / 2)
        for index, (area, d) in enumerate(steel):
            stress = max(-fy, min(fy, es * .003 * (1 - d / c)))
            force = area * (stress - (.85 * fc if index in included else 0))
            n += force
            moment += force * (h / 2 - d)
        return n, abs(moment)
    tiny = h * 1e-12
    upper = max(h / beta, max(d for _, d in steel) / (1 - fy / (.003 * es))) * 1.01
    boundaries = sorted({tiny, h / beta, upper, *(d / beta for _, d in steel)})
    roots = []
    for lower, higher in zip(boundaries, boundaries[1:]):
        included = {index for index, (_, d) in enumerate(steel)
                    if d < beta * ((lower + higher) / 2)}
        low_p, low_m = state(lower, included)
        high_p, high_m = state(higher, included)
        if p < low_p - tolerance or p > high_p + tolerance:
            continue
        if abs(p - low_p) <= tolerance:
            roots.append((low_m, lower, low_p))
            continue
        if abs(p - high_p) <= tolerance:
            roots.append((high_m, higher, high_p))
            continue
        lo, hi = lower, higher
        for _ in range(70):
            mid = (lo + hi) / 2
            mid_p, mid_m = state(mid, included)
            if abs(mid_p - p) <= tolerance:
                break
            if mid_p < p:
                lo = mid
            else:
                hi = mid
        if abs(mid_p - p) <= tolerance:
            roots.append((mid_m, mid, mid_p))
    if not roots:
        raise ValueError("No equilibrium root: unresolved lumped-bar stress-block discontinuity.")
    moment, neutral_axis, solved_p = min(roots)
    return {"mn_kip_in": moment, "factored_axial_kip": p,
            "neutral_axis_in": neutral_axis, "equilibrium_residual_kip": solved_p - p,
            "physical_axial_domain_kip": [pmin, pmax],
            "equilibrium_roots": len(roots), "nominal_strengths": True,
            "basis": "ACI318-19_22.2_rectangular_strain_compatibility_uniaxial_no_phi"}


def record_section_capacity(record, member, axis, sign, axial_kip):
    """Capacity callback API, using actual record bar positions and materials.

    For columns, X-frame bending stresses vary through local z (h); Y-frame
    stresses vary through local y (b). Beam flexure is about its horizontal
    major axis in either frame. Positive beam flexure has top compression.
    Column sign is a compression-face choice; all signs are evaluated.
    """
    if axis not in ("x", "y") or sign not in ("positive", "negative"):
        raise ValueError("Explicit x/y axis and positive/negative sign are required.")
    prefix = "col" if member == "column" else "beam"
    sections, materials = record["sections"], record["materials"]
    b, h = sections[f"b_{prefix}_in"], sections[f"h_{prefix}_in"]
    bars = section_bar_coordinates(record, member, axis if member == "beam" else None)
    if member == "column" and axis == "y":
        width, depth, coordinates = h, b, [(area, y) for area, y, _ in bars]
    else:
        width, depth, coordinates = b, h, [(area, z) for area, _, z in bars]
    direction = 1 if sign == "positive" else -1
    layers = [(area, depth / 2 - direction * position) for area, position in coordinates]
    return nominal_rectangular_capacity(width_in=width, depth_in=depth,
                                        fc_ksi=sections[f"fc_{prefix}_ksi"],
                                        fy_ksi=materials.get("fy_ksi"),
                                        es_ksi=materials.get("es_ksi"), layers=layers,
                                        axial_kip=axial_kip)


def build_joint_evidence(record, combination_actions, *, expected_combination_ids,
                         section_capacity_evaluator=None):
    """Produce exact geometry + axial-envelope evidence without optimistic gaps.

    All supplied factored combinations and BOTH column compression faces are
    conservatively enveloped for each sway (including opposite-direction
    cases); this can reduce the provided
    SCWB strength but cannot exaggerate it. It is not a substitute for missing
    earthquake/torsion/load-pattern cases in the overall design scope.
    A custom section_capacity_evaluator has the same API as
    record_section_capacity and must return its nominal-strength evidence.
    """
    inventory = physical_joint_inventory(record["geometry"])
    evaluate = section_capacity_evaluator or record_section_capacity
    expected = list(expected_combination_ids)
    if not expected or any(not isinstance(v, str) or not v for v in expected) or len(set(expected)) != len(expected):
        raise ValueError("Unique nonempty expected combination IDs are mandatory.")
    if not isinstance(combination_actions, (list, tuple)):
        raise ValueError("combination_actions must be a list.")
    columns = {m["tag"] for m in inventory["members"] if m["kind"] == "column"}
    actions, issues = {}, []
    for item in combination_actions:
        if not isinstance(item, Mapping) or item.get("id") not in expected:
            issues.append("Unexpected or malformed combination.")
            continue
        cid = item["id"]
        if cid in actions:
            issues.append(f"Duplicate combination {cid}.")
            continue
        actions[cid] = item
        if item.get("analysis_succeeded") is not True:
            issues.append(f"Combination {cid} did not confirm successful analysis.")
        members = item.get("members", {})
        if not isinstance(members, Mapping):
            issues.append(f"Combination {cid} has no member actions.")
            continue
        for tag in columns:
            member = members.get(tag, members.get(str(tag)))
            try:
                if not isinstance(member, Mapping):
                    raise ValueError("Missing column.")
                for end in ("i", "j"):
                    _number(member.get(f"axial_{end}_kip"), "column axial force")
            except ValueError:
                issues.append(f"Combination {cid}, column {tag}: missing/nonfinite end axial force.")
    for cid in expected:
        if cid not in actions:
            issues.append(f"Missing combination {cid}.")
    reference_issues = [f"Combination {cid} must identify axial_reference='joint_faces'; centerline forces are not joint-face forces."
                        for cid, item in actions.items() if item.get("axial_reference") != "joint_faces"]
    complete = not issues
    completeness = [make_check("joint.physical_inventory", "Model connectivity inventory",
                               len(inventory["joints"]), inventory["elevated_joint_count"],
                               comparison="==", details={"base_connections_included": False})]
    completeness.append(make_check("joint.factored_action_inventory", "ACI 318-19 18.7.3.2",
                                    len(issues), 0, details={"issues": issues,
                                    "expected_combination_ids": expected,
                                    "observed_combination_ids": list(actions)}))
    completeness.append(make_check("joint.axial_force_reference", "ACI 318-19 18.7.3.2",
                                    len(reference_issues), 0, details={"issues": reference_issues,
                                    "required_reference": "joint_faces"}))
    result = {"inventory": inventory, "completeness_checks": completeness,
              "expected_combination_ids": expected, "action_inventory_complete": complete,
              "joint_face_reference_complete": complete and not reference_issues,
              "joints": [], "through_bar_anchorage": [],
              "axial_envelope_basis": "minimum nominal strength across every supplied factored case and both column compression faces; no extrapolation",
              "uniaxial_only": True,
              "roof_exemption_applied": "per connection: ACI 318-19 18.7.3.1 (SMRF_Joints.scwb_check)"}
    cache = {}
    def capacity(member, axis, sign, p):
        key = (member, axis, sign, p)
        if key not in cache:
            value = evaluate(record, member, axis, sign, p)
            if (not isinstance(value, Mapping) or value.get("nominal_strengths") is not True
                    or not value.get("basis")):
                raise ValueError("Section evaluator must return explicit nominal-strength basis.")
            mn = _number(value.get("mn_kip_in"), "nominal moment")
            if mn < 0:
                raise ValueError("Nominal moment magnitude cannot be negative.")
            cache[key] = dict(value)
        return cache[key]
    slab_data = record.get("beam_slab_strengths", {})
    slab_data = slab_data if isinstance(slab_data, Mapping) else {}
    for joint in inventory["joints"]:
        entry = {**joint, "directions": {}}
        column_section = record.get("sections", {})
        # Evidence for the ACI 318-19 18.7.3.1 exception (SMRF_Joints.scwb_check): whether a column continues
        # above this connection, and the gross area and concrete strength its axial limit is taken on.
        connection = {"column_discontinuous_above": not any(c["position"] == "above" for c in joint["columns"]),
                      "column_gross_area_in2": (column_section.get("b_col_in") or 0.0) * (column_section.get("h_col_in") or 0.0),
                      "column_fc_ksi": column_section.get("fc_col_ksi")}
        for axis in ("x", "y"):
            beams = joint[f"beams_{axis}"]
            states = {}
            for sign in ("positive", "negative"):
                state = {"nominal_strengths": True, "column_capacities": [], "beam_capacities": [],
                         "connection": dict(connection)}
                for column in joint["columns"]:
                    cap = {**column, "axial_envelope_checked": False}
                    try:
                        if not complete:
                            raise ValueError("Final factored-action inventory is incomplete.")
                        observations = []
                        for cid in expected:
                            members = actions[cid]["members"]
                            p = _number(members.get(column["tag"], members.get(str(column["tag"])))[f"axial_{column['end']}_kip"], "axial force")
                            alternatives = [(face, capacity("column", axis, face, p))
                                            for face in ("positive", "negative")]
                            face, solved = min(alternatives, key=lambda pair: pair[1]["mn_kip_in"])
                            observations.append({"combination_id": cid, "factored_axial_kip": p,
                                                 "mn_kip_in": solved["mn_kip_in"],
                                                 "governing_compression_face": face})
                        governing = min(observations, key=lambda v: v["mn_kip_in"])
                        cap.update(governing)
                        cap.update(axial_envelope_checked=not reference_issues,
                                   axial_min_kip=min(o["factored_axial_kip"] for o in observations),
                                   axial_max_kip=max(o["factored_axial_kip"] for o in observations),
                                   combination_count=len(observations), observations=observations,
                                   compression_face_basis="minimum of both faces; conservative uniaxial sway envelope",
                                   axial_reference="joint_faces" if not reference_issues else "unverified",
                                   capacity_basis=capacity("column", axis, governing["governing_compression_face"], governing["factored_axial_kip"])["basis"])
                        if reference_issues:
                            cap["reason"] = "Joint-face axial reference is unverified; computed end-force capacities are preliminary."
                    except (ValueError, TypeError, KeyError) as exc:
                        cap["reason"] = str(exc)
                    state["column_capacities"].append(cap)
                for beam in beams:
                    # Positive sway: the negative-side beam's right end is
                    # sagging, positive-side beam's left end is hogging.
                    flexure = sign if beam["end"] == "j" else ("negative" if sign == "positive" else "positive")
                    cap = {**beam, "flexure_sign": flexure}
                    try:
                        solved = capacity("beam", axis, flexure, 0.)
                        cap.update(mn_kip_in=solved["mn_kip_in"], capacity_basis=solved["basis"])
                    except (ValueError, TypeError, KeyError) as exc:
                        cap["reason"] = str(exc)
                    supplied = slab_data.get(f"{beam['tag']}/{beam['end']}/{flexure}")
                    if isinstance(supplied, Mapping) and supplied.get("section_compatibility_verified") is True:
                        cap.update({key: supplied.get(key) for key in ("slab_basis", "slab_mn_kip_in")})
                    else:
                        cap["slab_reason"] = "Developed beam-plus-slab section compatibility evidence is missing."
                    state["beam_capacities"].append(cap)
                states[sign] = state
            entry["directions"][axis] = states
            sections, rebar, materials = record.get("sections", {}), record.get("reinforcement", {}), record.get("materials", {})
            result["through_bar_anchorage"].append({
                "id": f"{joint['id']}/{axis}", "bars_pass_through": len(beams) == 2,
                "joint_depth_in": sections.get("h_col_in" if axis == "x" else "b_col_in"),
                "beam_depths_in": [sections.get("h_beam_in") for _ in beams],
                "concrete_type": "normalweight" if materials.get("normalweight") is True else
                                 "lightweight" if materials.get("normalweight") is False else None,
                "bars": [{"grade_ksi": materials.get("fy_ksi"),
                          "diameter_in": rebar.get("beam_bar_diameter_in")}],
                "beam_ends": beams, "scope": "joint depth only; terminating-bar development remains separate"})
        result["joints"].append(entry)
    return result
