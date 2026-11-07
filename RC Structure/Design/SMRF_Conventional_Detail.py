"""Conventional reinforcement detailing of the frame-plus-transfer production route (2026-11-07).

Bar passage, laps, couplers, hooks and roof anchorage are checked with closed-form clearance and development rules at the
sections where a detailer checks them. On this route these rules take the place of the 3D bar-collision inventory
(SMRF_Transition_Physical, SMRF_Transition_Context, SMRF_Splice_Physical, SMRF_Actual_Hoop_Selection), which stays in the
tree for the computed common-action route (Structure_Parameters.DETAIL_BASIS).

Code requirements checked here:
  lap groups          25.5.1.2 sends the clear spacing around a contact lap to 25.2.1; 25.5.1.3 limits a non-contact lap
  couplers            18.2.7 and 25.5.7 by the product actually used; clear spacing and cover to the device envelope
  lateral support     25.7.2.3(b) for bars that stop or lap inside a column cage
  offset bends        10.7.4.1 inside the joint depth with the declared Table 25.3.1 radii
  roof anchorage      18.8.5.1 hook or 18.8.5.3 and 18.8.5.4 straight bar, 25.4.9 in compression, No. 3 through No. 11

Construction allowances, declared here because they are choices and not code clauses:
  plan shift          beam bars may shift sideways between joints, keeping their order; they thread each joint's lanes
  crossing bars       orthogonal bars may touch; the 25.2 clear spacing rules govern parallel bars
  opposing hooks      top and bottom hook tails that would meet in one plane are turned toward the gap between lanes
  hook tails          a tail may run past the beam depth into the column above or below, inside the column core
  coupler stations    hoop stations may shift locally around a coupler body, keeping the number of sets per length

Units: kip, inch, ksi.
"""
from __future__ import annotations

import math

import Structure_Parameters as sp

BASIS = "conventional_detailing_v1"
PHYSICAL_BASIS = "physical_3d_v2"
METHOD = "conventional_closed_form_detailing_v1"
TOL = 1.0e-6
HOOK_TAIL_LATERAL_FACTOR = 1.0     # one bar diameter of clear gap lets a tail pass between two lanes


def active():
    """Whether the installed run uses the conventional detailing basis."""
    return getattr(sp, "DETAIL_BASIS", BASIS) == BASIS


def _check(checks, rule, passes, detail, clause):
    checks.append({"rule": rule, "status": "not_evaluated" if passes is None else "pass" if passes else "fail",
                   "passes": passes, "detail": detail, "clause": clause})


def _finish(out):
    states = [c["status"] for c in out["checks"]]
    out["status"] = "fail" if "fail" in states else "not_evaluated" if "not_evaluated" in states or not states else "pass"
    out["failure_rules"] = [c["rule"] for c in out["checks"] if c["status"] == "fail"]
    out["unknown_rules"] = [c["rule"] for c in out["checks"] if c["status"] == "not_evaluated"]
    return out


def clear_spacing_25_2_1(db_in, aggregate_in):
    """Clear spacing between parallel bars in a layer (25.2.1); 25.5.1.2 applies it around contact laps."""
    return max(1.0, db_in, 4.0 * aggregate_in / 3.0)


def clear_spacing_25_2_3(db_in, aggregate_in):
    """Clear spacing between column bars away from a lap (25.2.3)."""
    return max(1.5, 1.5 * db_in, 4.0 * aggregate_in / 3.0)


def lap_spacing(paths, lower_db, upper_db, aggregate_in=0.75):
    """Clear spacing around the lap groups of a transition (25.5.1.2 to 25.2.1).

    The two partners of one lap are not tested against each other: their transverse distance is the 25.5.1.3 item of
    the transition. Every other pair of bars present in the lap zone keeps the 25.2.1 clear spacing."""
    bars = []
    for index, path in enumerate(paths):
        if path["path"] in ("lap_spliced", "terminated"):
            bars.append((index, "lower", path["target_xy_in"], lower_db))
        if path["upper_xy_in"] is not None:
            bars.append((index, "upper", path["upper_xy_in"], upper_db))
    failures, smallest = [], None
    for position, first in enumerate(bars):
        for second in bars[position + 1:]:
            if first[0] == second[0]:
                continue
            clear = math.dist(first[2], second[2]) - (first[3] + second[3]) / 2.0
            limit = clear_spacing_25_2_1(max(first[3], second[3]), aggregate_in)
            entry = {"bars": [f"{first[1]}_{first[0]}", f"{second[1]}_{second[0]}"], "clear_in": clear, "required_in": limit}
            if smallest is None or clear - limit < smallest["clear_in"] - smallest["required_in"]:
                smallest = entry
            if clear < limit - TOL:
                failures.append(entry)
    return {"passes": not failures, "minimum_margin_pair": smallest, "failures": failures[:24],
            "basis": "25.5.1.2: 25.2.1 spacing around a lap; lap partners are limited by 25.5.1.3"}


def coupler_detail(lower_size, upper_size, *, b_in, h_in, points, fy_ksi, specification, clear_cover_in, spacing_in,
                   hoop_size, outside_limit_in, aggregate_in=0.75, member="column", protected_region=False,
                   local_gap=None):
    """The Type 2 coupler a detail uses: the published product, its cover, clear spacing and room between hoop sets.

    ``local_gap`` ({limit_in, zone_in, ...}): the device lies in a length of ``zone_in`` beyond l0 where a local hoop
    gap up to ``limit_in`` is permitted (18.7.5.5 with the 18.7.6 shear spacing; Group_Capacity passes it for a column's
    own splice). None keeps the rules of the hoop spacing in force."""
    from Design.SMRF_Coupler_Catalog import lookup_coupler
    product = lookup_coupler(lower_size, upper_size, fy_ksi, specification, protected_region)
    out = {"method": METHOD, "status": product["status"], "product": product, "checks": [], "bodies": [],
           "additional_transverse_steel_volume_in3": 0.0, "net_envelope_steel_volume_in3": 0.0,
           "device_count": len(points)}
    if product["status"] != "pass":
        out["reason"] = product.get("reason")
        return out
    envelope, length = product["envelope_diameter_in"], product["length_in"]
    db = max(sp.rebar_diameter(lower_size), sp.rebar_diameter(upper_size))
    dt = sp.rebar_diameter(hoop_size)
    cover = min(min(b_in / 2.0 - abs(x), h_in / 2.0 - abs(y)) for x, y in points) - envelope / 2.0
    _check(out["checks"], "splice_device_cover", cover >= clear_cover_in - TOL,
           {"provided_in": cover, "required_in": clear_cover_in}, "ER-0129 3.2; Table 20.5.1.3.1")
    limit = clear_spacing_25_2_3(db, aggregate_in) if member == "column" else clear_spacing_25_2_1(db, aggregate_in)
    centers = [math.dist(a, b) for i, a in enumerate(points) for b in points[i + 1:]]
    same_level = min(centers) - envelope if centers else float("inf")
    staggered = min(centers) - (envelope + db) / 2.0 if centers else float("inf")
    _check(out["checks"], "splice_device_clear_spacing", max(same_level, staggered) >= limit - TOL,
           {"same_level_clear_in": same_level, "staggered_clear_in": staggered, "required_in": limit,
            "staggered": same_level < limit - TOL}, "25.2 applied to the device envelope; adjacent devices staggered where needed")
    gap = length + dt
    second_gap = 2.0 * spacing_in - gap
    inside = cover - (clear_cover_in + dt)                  # clear distance from the device to the hoop's inner face
    beyond_l0 = None
    if local_gap is not None:
        # Beyond l0 the hoop stations may open one local gap around the device within the 18.7.5.5 and shear limits
        # (2026-11-07); staggered devices need two such gaps in the length.
        needed = gap * (2.0 if same_level < limit - TOL else 1.0)
        beyond_l0 = {**local_gap, "zone_needed_in": needed,
                     "applies": local_gap["zone_in"] >= needed - TOL and gap <= local_gap["limit_in"] + TOL}
    fits = (inside >= -TOL or gap <= spacing_in + TOL
            or (gap <= outside_limit_in + TOL and second_gap >= dt + 1.0 - TOL)
            or bool(beyond_l0 and beyond_l0["applies"]))
    _check(out["checks"], "splice_device_between_hoop_sets", fits,
           {"device_inside_hoop_line_in": inside, "device_plus_hoop_in": gap, "hoop_spacing_in": spacing_in,
            "local_spacing_limit_in": outside_limit_in, "remaining_gap_in": second_gap, "beyond_l0": beyond_l0,
            "basis": "the hoops pass outside a device inside their line; otherwise the device sits between two hoop sets"},
           "18.7.5.5 or 18.6.4.6 outside the hinge zones; hoop stations shift locally and keep the sets per length")
    return _finish(out)


def transition_detail(result, lower, upper, *, fy_ksi, clear_cover_in, upper_clear_height_in, joint_depth_in,
                      lower_hoop_spacing_in=None, physical_context=None):
    """The closed-form detail checks of the bar paths a column transition declares (SMRF_Transitions.column_transition)."""
    from Design import SMRF_Transition_Physical as physical
    context = physical_context or {}
    out = {"method": METHOD, "status": "not_evaluated", "checks": [], "coordinate_basis": "hoop_corner_containment_v1",
           "tie_sets": [], "couplers": [], "additional_transverse_steel_volume_in3": 0.0,
           "longitudinal_quantity_adjustment_in3": 0.0, "mechanical_device_net_steel_volume_in3": 0.0}
    checks = out["checks"]
    if not result.get("geometry_supported", result["supported"]):
        _check(checks, "preliminary_transition_rules", False, "At least one declared path, scope or anchorage rule fails.",
               "declared transition scope")
    paths = result["paths"]
    if not paths:
        return _finish(out)
    aggregate = float(context.get("aggregate_max_size_in", sp.AGGREGATE_MAX_SIZE_IN))
    lower_db, upper_db = sp.rebar_diameter(lower.bar_size), sp.rebar_diameter(upper.bar_size)
    largest = max(path["offset_in"] for path in paths)
    height = physical.offset_bend_geometry(largest, lower.bar_size)["height_in"] if largest > TOL else 0.0
    _check(checks, "finite_radius_bends_fit", height <= joint_depth_in + TOL,
           {"required_height_in": height, "available_in": joint_depth_in}, "10.7.4.1; Table 25.3.1 radii")
    laps = lap_spacing(paths, lower_db, upper_db, aggregate)
    _check(checks, "lap_group_clear_spacing", laps["passes"], laps, "25.5.1.2; 25.2.1")
    extra_in_upper = [path for path in paths if path["path"] in ("lap_spliced", "terminated")]
    if extra_in_upper:
        legs = upper.stirrup_legs_by_direction or (upper.stirrup_legs,) * 2
        offset = physical_offset(upper, clear_cover_in)
        spans = [(upper.b_in - 2.0 * offset) / max(1, legs[0] - 1), (upper.h_in - 2.0 * offset) / max(1, legs[1] - 1)]
        reach = max(spans) / 2.0 - lower_db
        _check(checks, "bars_in_lap_zone_supported_25_7_2_3_b", reach <= 6.0 + TOL,
               {"largest_clear_to_supported_bar_in": reach, "limit_in": 6.0,
                "basis": "a lap partner or stopping bar lies between two legs of the upper cage"},
               "25.7.2.3(b)")
    mechanical = [path for path in paths if path["path"] == "mechanically_spliced"]
    if mechanical:
        spacing = float(lower_hoop_spacing_in or lower.stirrup_spacing_in)
        points = [tuple(path["upper_xy_in"]) for path in mechanical]
        detail = coupler_detail(lower.bar_size, upper.bar_size, b_in=upper.b_in, h_in=upper.h_in, points=points,
                                fy_ksi=fy_ksi, specification=context.get("longitudinal_specification", sp.REINFORCEMENT_SPECIFICATION),
                                clear_cover_in=clear_cover_in, spacing_in=spacing, hoop_size=upper.stirrup_bar_size,
                                outside_limit_in=min(6.0, (5.0 if fy_ksi >= 80.0 else 6.0) * min(lower_db, upper_db)),
                                aggregate_in=aggregate, member="column")
        out["mechanical_splice_detail"] = detail
        out["product"] = detail.get("product")
        _check(checks, "mechanical_splice_product", True if detail["product"]["status"] == "pass"
               else (None if detail["product"]["status"] == "not_evaluated" else False), detail["product"],
               "18.2.7; 25.5.7; product actually used")
        if detail["product"]["status"] == "pass":
            _check(checks, "splice_devices_and_transverse_detail", detail["status"] == "pass", detail["checks"],
                   "18.2.7; 25.2; 18.7.5.5")
    return _finish(out)


def physical_offset(design, clear_cover_in):
    from Design.SMRF_Transitions import centroid_cover_in
    return centroid_cover_in(design, clear_cover_in)


def roof_detail(design, *, extension_in, joint_depth_in, fy_ksi, clear_cover_in, spacing_in, normalweight=True):
    """Anchorage of a column's bars at the roof, with the hoops continued through the declared extension above the roof."""
    from Design.SMRF_Roof_Anchorage import straight_roof_development_bound
    from Design.SMRF_Transitions import bar_coordinates
    out = {"method": "conventional_roof_extension_anchorage_v1", "status": "not_evaluated", "checks": [],
           "extension_in": extension_in, "tie_sets": [], "net_head_steel_volume_in3": 0.0,
           "additional_transverse_steel_volume_in3": 0.0, "termination_type": None}
    checks = out["checks"]
    db = sp.rebar_diameter(design.bar_size)
    bars = bar_coordinates(design, clear_cover_in)
    tip = extension_in - clear_cover_in - db / 2.0
    length = joint_depth_in + max(0.0, tip)
    _check(checks, "roof_extension_is_full_column_depth", extension_in >= max(design.b_in, design.h_in) - TOL,
           {"declared_extension_in": extension_in, "required_for_15_2_6_in": max(design.b_in, design.h_in)},
           "15.2.6(b); Table 18.8.4.3")
    if int(design.bar_size) > 11:
        _check(checks, "roof_bar_anchorage", None,
               {"reason": "18.8.5.1 and 18.8.5.3 cover No. 3 through No. 11; no supported termination is declared for a larger bar"},
               "18.8.5")
    else:
        root = min(math.sqrt(design.fc_ksi * 1000.0), 100.0)
        lam = 1.0 if normalweight else 0.75
        ldh = max(fy_ksi * 1000.0 * db / (65.0 * lam * root), (8.0 if normalweight else 10.0) * db, 6.0 if normalweight else 7.5)
        straight = 3.25 * ldh
        compression = max(0.02 * fy_ksi * 1000.0 / root * db, 0.0003 * fy_ksi * 1000.0 * db, 8.0)
        general = straight_roof_development_bound(design, extension_in=extension_in, joint_depth_in=joint_depth_in,
                                                  fy_ksi=fy_ksi, clear_cover_in=clear_cover_in, normalweight=normalweight)
        straight_ok = (length >= straight - TOL and length >= compression - TOL and general.get("status") == "pass")
        hook_ok = length >= ldh - TOL and length >= compression - TOL
        out["termination_type"] = ("straight_bar_at_declared_extension_top" if straight_ok
                                   else "standard_hook_at_declared_extension_top" if hook_ok else None)
        _check(checks, "roof_bar_anchorage", straight_ok or hook_ok,
               {"available_in": length, "straight_required_in": straight, "hook_required_in": ldh,
                "compression_required_in": compression, "general_straight": general.get("ld_in"),
                "termination": out["termination_type"],
                "basis": "hoops continue to the extension top, so the whole length is confined (18.8.5.4 factor not needed)"},
               "18.8.5.1; 18.8.5.3(b); 18.8.5.4; 25.4.9")
    area = sp.rebar_area(design.bar_size)
    sets = math.ceil(extension_in / max(spacing_in, 1e-9))
    out["extension_concrete_in3"] = design.b_in * design.h_in * extension_in
    out["extension_longitudinal_steel_in3"] = len(bars) * area * max(0.0, tip)
    out["extension_transverse_steel_in3"] = sets * hoop_set_length_in(design) * sp.rebar_area(design.stirrup_bar_size)
    return _finish(out)


def hoop_set_length_in(design):
    """Cut length of one hoop set with its crossties (the plain rectangular form)."""
    cover = sp.COL_CLEAR_COVER_IN if design.is_column else sp.BEAM_CLEAR_COVER_IN
    db = sp.rebar_diameter(design.stirrup_bar_size)
    width, depth = design.b_in - 2.0 * cover, design.h_in - 2.0 * cover
    hook = max(6.0 * db, 3.0)
    if design.is_column:
        legs_b, legs_h = design.stirrup_legs_by_direction or (design.stirrup_legs, design.stirrup_legs)
        ties = (legs_b - 2) * (depth + 2.0 * hook) + (legs_h - 2) * (width + 2.0 * hook)
    else:
        ties = (design.stirrup_legs - 2) * (depth + 2.0 * hook)
    return 2.0 * (width + depth) + 2.0 * hook + ties


def beam_joint_detail(entry):
    """The hook arrangement and plan path of the beam bars at one joint type (the lanes are beam.bars_thread_column)."""
    from Design.SMRF_Cage_Geometry import same_lane_hook_separation_in
    from Model import Member_Bar_Layers as bar_layers
    from Model import Member_Properties as mp
    out = {"method": METHOD, "status": "not_evaluated", "checks": [], "beam_segments": [], "beam_geometry_complete": True,
           "beam_geometry_checks": [], "beam_hook_families": [], "beam_longitudinal_adjustment_in3": 0.0,
           "placement_obstacle_segments": []}
    checks = out["checks"]
    for axis, members in entry["beams"].items():
        if not members:
            continue
        mixed = bool((entry.get("mixed") or {}).get(axis))
        terminating = len(members) < 2 or mixed
        for beam in {member.group_id: member for member in members}.values():
            design = beam.design
            rows = bar_layers.group_rows(beam.group_id)
            db = sp.rebar_diameter(design.bar_size)
            hoop = sp.rebar_diameter(design.stirrup_bar_size)
            first = sp.first_bar_offset_in(sp.BEAM_CLEAR_COVER_IN, hoop, db)
            count = max(n for face in ("top", "bottom") for n in rows[face]["per_layer"])
            spacing = (design.b_in - 2.0 * first) / max(1, count - 1)
            clear = spacing - db
            span = mp.beam_clear_span_in(beam)
            _check(checks, "beam_bars_keep_order_between_joints", (design.b_in - 2.0 * first) <= span / 6.0 + TOL,
                   {"largest_plan_shift_in": design.b_in - 2.0 * first, "clear_span_in": span, "slope_limit": "1 in 6"},
                   "construction allowance: plan shift along the span; spacing is smallest at the joints checked")
            if terminating:
                vertical = design.h_in - rows["top"]["offsets_in"][0] - rows["bottom"]["offsets_in"][0]
                meet = vertical < same_lane_hook_separation_in(design.bar_size) - TOL
                _check(checks, "opposing_hook_tails", (not meet) or count < 2 or clear >= HOOK_TAIL_LATERAL_FACTOR * db - TOL,
                       {"rows_apart_in": vertical, "tails_meet_in_plane": meet, "lane_clear_in": clear, "bar_diameter_in": db,
                        "group": beam.group_id, "axis": axis},
                       "construction allowance: tails turned toward the gap between lanes; development is 18.8.5.1")
    return _finish(out)


def lap_or_device(design, *, lap_in, zone_in, rows=None, clear_cover_in, spacing_in, fy_ksi, specification, aggregate_in,
                  outside_limit_in, protected_region=False, local_gap=None):
    """A member's own splice: a Class B lap where the lap groups keep their spacing, otherwise a Type 2 coupler."""
    db = sp.rebar_diameter(design.bar_size)
    from Design.SMRF_Transitions import bar_coordinates
    if design.is_column:
        bars = bar_coordinates(design, clear_cover_in)
        points = [(bar["x_in"], bar["y_in"]) for bar in bars]
        centers = min(math.dist(a, b) for i, a in enumerate(points) for b in points[i + 1:])
    else:
        hoop = sp.rebar_diameter(design.stirrup_bar_size)
        first = sp.first_bar_offset_in(sp.BEAM_CLEAR_COVER_IN, hoop, db)
        count = max(n for face in ("top", "bottom") for n in rows[face]["per_layer"])
        centers = (design.b_in - 2.0 * first) / max(1, count - 1) if count > 1 else design.b_in
        spread = [first - design.b_in / 2.0 + k * centers for k in range(count)] if count > 1 else [0.0]
        points = [(x, design.h_in / 2.0 - rows["top"]["offsets_in"][0]) for x in spread]
    limit = clear_spacing_25_2_1(db, aggregate_in)
    same_level = centers - 2.0 * db
    staggered = centers - 1.5 * db
    lap_permitted = int(design.bar_size) <= 11 and lap_in is not None and lap_in <= zone_in + TOL
    fits_same = lap_permitted and same_level >= limit - TOL
    fits_staggered = lap_permitted and staggered >= limit - TOL and 2.0 * lap_in <= zone_in + TOL
    if fits_same or fits_staggered:
        return {"method": METHOD, "status": "pass", "splice": "class_b_lap", "stagger_laps": not fits_same,
                "checks": [{"rule": "lap_group_clear_spacing", "status": "pass", "passes": True,
                            "detail": {"same_level_clear_in": same_level, "staggered_clear_in": staggered, "required_in": limit,
                                       "lap_in": lap_in, "zone_in": zone_in}, "clause": "25.5.1.2; 25.2.1"}],
                "bodies": [], "additional_transverse_steel_volume_in3": 0.0, "net_envelope_steel_volume_in3": 0.0}
    detail = coupler_detail(design.bar_size, design.bar_size, b_in=design.b_in, h_in=design.h_in, points=points, fy_ksi=fy_ksi,
                            specification=specification, clear_cover_in=clear_cover_in, spacing_in=spacing_in,
                            hoop_size=design.stirrup_bar_size, outside_limit_in=outside_limit_in, aggregate_in=aggregate_in,
                            member="column" if design.is_column else "beam", protected_region=protected_region,
                            local_gap=local_gap)
    detail["splice"] = "type_2_mechanical"
    return detail


def slab_steel_volume_in3(layout, floor_areas_in2):
    """Slab mat steel from the layout's area per foot over the floor areas (uniform mats, no global bar inventory)."""
    per_inch = sum(float(layer["area_in2_per_ft"]) for layer in layout["layers"].values()) / 12.0
    return per_inch * sum(floor_areas_in2)
