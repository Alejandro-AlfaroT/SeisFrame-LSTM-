"""Column transition paths and physical closure, ACI 318-19.

The fast path assigns straight, finite-offset, lap, mechanical or terminated
longitudinal paths and preserves all proposal catalogs. Identical cages are
straight. Intersecting lap cylinders are refused; they are never silently
moved. ``geometry_supported`` is preliminary search admissibility, not final
physical acceptance. With ``physical_context``, SMRF_Transition_Physical
checks the actual beam segments, rounded hoop/crosstie geometry, bend radii,
clear spacing, endpoint ownership and published coupler envelopes. Final
acceptance requires ``physically_verified``; missing evidence is distinct
from computed failure. Without context the preliminary path remains usable
by the search but physical status is explicitly not_evaluated.

Coordinates use shared hoop_corner_containment_v1. Tie-force demand uses
longitudinal fy / transverse fyt, including Grade80/Grade60 combinations.
A reduced section always uses the conservative Other column-continuity
branch of Table 18.8.4.3; no larger lower hoop continuation is invented.
Development/lap lengths use 25.4.2.3/25.5.2.2 with no excess-steel reduction.
"""
from __future__ import annotations

import math

RULES_ID = "column_transition_rules_v4_physical_details"
OFFSET_BEND_FACE_STEP_LIMIT_IN = 3.0       # ACI 318-19 10.7.4.2
OFFSET_BEND_SLOPE = 6.0                    # 10.7.4.1: 1 in 6
OFFSET_TIE_FORCE_FACTOR = 1.5              # 10.7.6.4.1
OFFSET_TIE_DISTANCE_IN = 6.0               # 10.7.6.4.2
OFFSET_TIE_MIN_SPACING_IN = 3.0            # declared search restriction (the hoop ladder's minimum spacing); not an ACI minimum
NON_CONTACT_LAP_MAX_IN = 6.0               # 25.5.1.3
LAP_MAX_BAR_SIZE = 11                      # 25.5.1.1
COORDINATE_TOLERANCE_IN = 1.0e-6           # two declared coordinates are the same position within this distance
KINDS = ("same_cage", "bar_size_reduction", "bar_count_reduction", "offset_bends", "unsupported")
PATHS = ("straight", "offset", "lap_spliced", "mechanically_spliced", "terminated")
_BAR = {3: (0.375, 0.11), 4: (0.5, 0.20), 5: (0.625, 0.31), 6: (0.75, 0.44), 7: (0.875, 0.60), 8: (1.0, 0.79),
        9: (1.128, 1.00), 10: (1.27, 1.27), 11: (1.41, 1.56), 14: (1.693, 2.25), 18: (2.257, 4.00)}
_GRADE_FACTOR = {60.0: 1.0, 80.0: 1.15, 100.0: 1.3}    # psi_g, Table 25.4.2.5


# ---- development and lap lengths ------------------------------------------------------------------------------
def development_length_in(bar_size, fc_ksi, fy_ksi, favourable=True):
    """Straight tension development length by ACI 318-19 Table 25.4.2.3 (the simplified expressions).

    ``favourable`` selects the first row (clear spacing at least db, clear cover at least db and ties not
    less than the Code minimum along ld): fy psi_g / (25 sqrt(f'c)) db for No. 6 and smaller, / 20 for
    No. 7 and larger. Otherwise the "other cases" row: 3 fy psi_g / (50 sqrt(f'c)) db and / 40. The
    caller establishes which row applies (``development_case``). psi_t = psi_e = 1, lambda = 1,
    sqrt(f'c) limited to 100 psi (25.4.1.4), ld at least 12 in (25.4.2.1).
    """
    db = _BAR[int(bar_size)][0]
    if float(fy_ksi) not in _GRADE_FACTOR:
        raise ValueError(f"No reinforcement grade factor is declared for fy = {fy_ksi} ksi.")
    root = min(math.sqrt(fc_ksi * 1000.0), 100.0)
    small = int(bar_size) <= 6
    base = fy_ksi * 1000.0 * _GRADE_FACTOR[float(fy_ksi)] / root * db
    length = base / (25.0 if small else 20.0) if favourable else 3.0 * base / (50.0 if small else 40.0)
    return max(12.0, length)


def lap_length_in(lower_bar, upper_bar, fc_ksi, fy_ksi, favourable=True):
    """Class B tension lap of two bars of (possibly) different size: max(ld larger, 1.3 ld smaller) (25.5.2.2)."""
    larger, smaller = max(lower_bar, upper_bar), min(lower_bar, upper_bar)
    return max(development_length_in(larger, fc_ksi, fy_ksi, favourable),
               1.3 * development_length_in(smaller, fc_ksi, fy_ksi, favourable))


# ---- bar coordinates ------------------------------------------------------------------------------------------
def centroid_cover_in(design, clear_cover_in):
    """Face to the centroid of a first longitudinal bar: the bar position grid step at or beyond clear cover +
    hoop diameter + half the bar diameter (Structure_Parameters.first_bar_offset_in)."""
    import Structure_Parameters as sp
    return sp.first_bar_offset_in(clear_cover_in, _BAR[int(design.stirrup_bar_size)][0], _BAR[int(design.bar_size)][0])


def bar_coordinates(design, clear_cover_in):
    """Every longitudinal bar of a column about its centerline: [{"face", "index", "x_in", "y_in"}].

    x runs along b and y along h. The top and bottom rows (y = +/- (h/2 - c)) hold ``top_bars`` and
    ``bot_bars`` bars including the corners, evenly spaced; each side face (x = +/- (b/2 - c)) holds
    ``side_bars`` bars between the rows, every bar on the bar position grid
    (Structure_Parameters.face_bar_positions_in). c is ``centroid_cover_in``.
    """
    import Structure_Parameters as sp
    c = centroid_cover_in(design, clear_cover_in)
    half_b, half_h = 0.5 * design.b_in - c, 0.5 * design.h_in - c
    if half_b <= 0.0 or half_h <= 0.0:
        raise ValueError("The bar centroids do not fit inside the section.")
    bars = []
    for face, y, count in (("top", half_h, design.top_bars), ("bottom", -half_h, design.bot_bars)):
        for i, x in enumerate(sp.face_bar_positions_in(design.b_in, c, int(count))):
            bars.append({"face": face, "index": i, "x_in": x - 0.5 * design.b_in, "y_in": y})
    side = sp.face_bar_positions_in(design.h_in, c, int(design.side_bars) + 2)[1:-1]
    for face, x in (("left", -half_b), ("right", half_b)):
        for k, y in enumerate(side):
            bars.append({"face": face, "index": k, "x_in": x, "y_in": y - 0.5 * design.h_in})
    return bars


def minimum_clear_spacing_in(design, clear_cover_in):
    """Smallest clear distance between two longitudinal bars of the cage."""
    bars = bar_coordinates(design, clear_cover_in)
    db = _BAR[int(design.bar_size)][0]
    return min(math.hypot(a["x_in"] - b["x_in"], a["y_in"] - b["y_in"]) for i, a in enumerate(bars) for b in bars[i + 1:]) - db


def development_case(design, clear_cover_in):
    """Which row of Table 25.4.2.3 the bars of one column cage are in, with the quantities that decide it."""
    db = _BAR[int(design.bar_size)][0]
    spacing = minimum_clear_spacing_in(design, clear_cover_in)
    # clear cover to the bar as built: the first bar in the hoop corner on the grid, less half the bar
    # (rounded to the nanoinch so an even-spacing cage returns the exact specified cover plus hoop)
    cover = round(centroid_cover_in(design, clear_cover_in) - 0.5 * db, 9)
    favourable = spacing >= db - 1e-9 and cover >= db - 1e-9
    return {"table": "ACI 318-19 Table 25.4.2.3", "favourable": favourable,
            "row": ("clear spacing at least db, clear cover at least db, ties not less than the Code minimum" if favourable
                    else "other cases"),
            "bar_diameter_in": db, "clear_spacing_in": spacing, "clear_cover_to_bar_in": cover,
            "ties": "the column's hoops and crossties along the length", "psi_t": 1.0, "psi_e": 1.0, "lambda": 1.0}


def _match_along_face(lower, upper):
    """Order-preserving one-to-one pairing of upper positions to lower positions of least total distance.

    ``lower`` and ``upper`` are coordinates along one face, sorted; len(upper) <= len(lower). Returns the
    lower index paired with each upper bar. In one dimension the best pairing never crosses, so a pass
    over the two sorted lists finds it.
    """
    n, m = len(lower), len(upper)
    infinity = float("inf")
    cost = [[infinity] * (n + 1) for _ in range(m + 1)]
    take = [[False] * (n + 1) for _ in range(m + 1)]
    for j in range(n + 1):
        cost[0][j] = 0.0
    for i in range(1, m + 1):
        for j in range(i, n + 1):
            skip = cost[i][j - 1]
            use = cost[i - 1][j - 1] + abs(upper[i - 1] - lower[j - 1])
            if use <= skip:
                cost[i][j], take[i][j] = use, True
            else:
                cost[i][j] = skip
    pairs, i, j = [], m, n
    while i > 0:
        if take[i][j]:
            pairs.append(j - 1)
            i, j = i - 1, j - 1
        else:
            j -= 1
    return list(reversed(pairs))


def _item(rule, clause, passes, detail):
    return {"rule": rule, "clause": clause, "passes": bool(passes), "detail": detail}


def transition_bar_clearance(paths, lower_diameter_in, upper_diameter_in):
    """Check the volumes of the declared longitudinal bars, without inventing new paths.

    A lap contains two distinct bars simultaneously. Mechanical splices and a
    continuous bar have one longitudinal centerline at a given elevation, so
    they are not treated as two overlapping lap bars. Terminated lower tails
    remain present through the lower half of the declared lap region.

    For offset bends, all bars use the common inclined run. At each horizontal
    section the inclined cylinder fits within a circle of radius r/cos(theta).
    The minimum distance of each pair of linear paths is found analytically.
    This conservative non-intersection test does not establish code minimum
    clear spacing, coupler size, bend radius, or engagement by transverse bars.
    """
    run = OFFSET_BEND_SLOPE * max((p["offset_in"] for p in paths), default=0.0)
    joint, lap = [], []
    for i, path in enumerate(paths):
        radius = lower_diameter_in / 2.0
        projected = radius * math.sqrt(1.0 + (path["offset_in"] / run) ** 2) if run else radius
        joint.append((f"lower_{i}", path["lower_xy_in"], path["target_xy_in"], projected))
        if path["path"] in ("lap_spliced", "terminated"):
            lap.append((f"lower_tail_{i}", path["target_xy_in"], radius))
        if path["upper_xy_in"] is not None:
            lap.append((f"upper_{i}", path["upper_xy_in"], upper_diameter_in / 2.0))
    minima, collisions = {}, []
    for region, bars in (("joint", joint), ("upper_overlap_zone", lap)):
        minimum = None
        for i, first in enumerate(bars):
            for second in bars[i + 1:]:
                if region == "joint":
                    name_a, start_a, end_a, radius_a = first
                    name_b, start_b, end_b, radius_b = second
                    delta = [start_a[k] - start_b[k] for k in (0, 1)]
                    slope = [(end_a[k] - start_a[k]) - (end_b[k] - start_b[k]) for k in (0, 1)]
                    square = sum(v*v for v in slope)
                    u = max(0.0, min(1.0, -sum(a*b for a, b in zip(delta, slope)) / square)) if square else 0.0
                    distance = math.hypot(*(delta[k] + u * slope[k] for k in (0, 1)))
                else:
                    name_a, xy_a, radius_a = first
                    name_b, xy_b, radius_b = second
                    distance = math.dist(xy_a, xy_b)
                    u = None
                clearance = distance - radius_a - radius_b
                entry = {"region": region, "bars": [name_a, name_b], "clearance_in": clearance,
                         "fraction_through_inclined_run": u}
                if minimum is None or clearance < minimum["clearance_in"]:
                    minimum = entry
                if clearance < -COORDINATE_TOLERANCE_IN:
                    collisions.append(entry)
        minima[region] = minimum
    return {"method": "declared_bar_cylinder_clearance_v1", "passes": not collisions,
            "minimum_by_region": minima, "collisions": collisions,
            "limits": "longitudinal bar non-intersection only; code spacing, couplers, bend radii and transverse engagement remain separate"}


# ---- the transition -------------------------------------------------------------------------------------------
def column_transition(lower, upper, *, fy_ksi, clear_cover_in, upper_clear_height_in, joint_depth_in,
                      lower_hoop_spacing_in=None, physical_context=None, lap_allowed=True):
    """Place, pair and check the bars of the transition from ``lower`` (story k) to ``upper`` (story k + 1).

    ``lower`` and ``upper`` are Model.Member_Groups.MemberDesign columns. ``upper_clear_height_in`` is the
    clear height of the upper story (splices sit in its center half); ``joint_depth_in`` the depth of the
    shallowest beam framing into the joint (the height available to an offset bend); ``lower_hoop_spacing_in``
    the spacing of the lower column's hoops, which continue through the joint.

    Returns {"rules", "kind", "supported", "column_reinforcement_continuous", "items", "paths", "path_counts",
    "splice", "offset_bend", "confinement", "terminated_lower_bars", "extra_bar_length_in",
    "additional_hoop_sets_per_joint", "not_established", "detail", ...}. ``supported`` is True only when
    every item of the rule set passes.
    """
    for name, design in (("lower", lower), ("upper", upper)):
        if getattr(design, "member_type", None) != "column":
            raise ValueError(f"The {name} member of a column transition must be a column design.")
    items = []
    fc = lower.fc_ksi
    same_section = abs(lower.h_in - upper.h_in) < 1e-9 and abs(lower.b_in - upper.b_in) < 1e-9
    result = {"rules": RULES_ID, "kind": "unsupported", "supported": False, "column_reinforcement_continuous": False,
              "items": items, "paths": [], "path_counts": {path: 0 for path in PATHS}, "splice": None, "offset_bend": None,
              "terminated_lower_bars": 0, "extra_bar_length_in": 0.0, "additional_hoop_sets_per_joint": 0,
              "lower": {"b_in": lower.b_in, "h_in": lower.h_in, "bar_size": lower.bar_size, "top_bars": lower.top_bars,
                        "bot_bars": lower.bot_bars, "side_bars": lower.side_bars, "stirrup_bar_size": lower.stirrup_bar_size},
              "upper": {"b_in": upper.b_in, "h_in": upper.h_in, "bar_size": upper.bar_size, "top_bars": upper.top_bars,
                        "bot_bars": upper.bot_bars, "side_bars": upper.side_bars, "stirrup_bar_size": upper.stirrup_bar_size},
              "continuity_branch": "continuing_column",
              "continuity_basis": ("Table 18.8.4.3: an unchanged physical section may continue; reduced sections use Other. "
                                   "No extension under 15.2.6 is inferred from a passing longitudinal path."),
              "confinement": {
                  "joint": "the hoops of the column below continue through the joint depth (18.8.3.1)",
                  "above_joint": ("the hoops of the column below continue for one column depth above the joint (priced as "
                                  "continued confinement of a continuing column; not an extension under 15.2.6)"
                                  if same_section else
                                  "the hoops of the column below cannot continue in the smaller section; the upper column's own "
                                  "end-zone hoops start at the joint face"),
                  "lower_hoops_continue_above_joint_in": max(upper.b_in, upper.h_in) if same_section else None,
                  "established_15_2_6_b": bool(same_section)},
              "lower_hoops_continue_above_joint_in": max(upper.b_in, upper.h_in) if same_section else None,
              "joint_core": "the column below (Model.Member_Properties joint-core convention)",
              "not_established": ["which hoop legs engage which offset bar, and their anchorage at the bend",
                                  "the placement of the bend ties among the beam bars inside the joint",
                                  "congestion of the lap zone (two bars side by side at every splice)"]
                                 + ([] if same_section else ["the drawn detail that makes the column above a reduced column an "
                                                             "actual continuing column (Table 18.8.4.3 continuous-column branch, "
                                                             "the declared reading); the extension branch 15.2.6(b) is not "
                                                             "claimed and not tested"])}

    def finish(kind, detail):
        result["kind"] = kind
        result["supported"] = kind != "unsupported" and all(item["passes"] for item in items)
        result["geometry_supported"] = result["supported"]
        # A smaller upper section cannot contain the larger lower hoops.
        # Use the conservative Other table branch, independently of whether
        # individual longitudinal paths can be detailed successfully.
        result["column_reinforcement_continuous"] = result["supported"] and same_section
        result["continuity_branch"] = "continuing_column" if same_section else "other_conservative_reduced_column"
        result["coordinate_basis"] = "hoop_corner_containment_v1"
        result["detail"] = detail
        if not result["supported"] and kind != "unsupported":
            result["detail"] += "; not every rule of this transition is met: " + "; ".join(
                f"{item['rule']} ({item['detail']})" for item in items if not item["passes"])
        if physical_context is not None:
            if physical_context.get("conventional"):
                from Design.SMRF_Conventional_Detail import transition_detail as verify_transition
            else:
                from Design.SMRF_Transition_Physical import verify_transition
            result["physical_detail"] = verify_transition(
                result,lower,upper,fy_ksi=fy_ksi,clear_cover_in=clear_cover_in,
                upper_clear_height_in=upper_clear_height_in,joint_depth_in=joint_depth_in,
                lower_hoop_spacing_in=lower_hoop_spacing_in,physical_context=physical_context)
            result["physically_verified"] = result["physical_detail"]["status"] == "pass"
            if result["terminated_lower_bars"] and "terminated_tail_length_above_joint_in" in result["physical_detail"]:
                result["extra_bar_length_in"] = result["terminated_lower_bars"]*result["physical_detail"]["terminated_tail_length_above_joint_in"]
            result["not_established"] = result["physical_detail"]["unknown_rules"]
            if result["physical_detail"]["status"] == "fail":
                result["supported"] = False
                result["column_reinforcement_continuous"] = False
        else:
            result["physically_verified"] = False
            result["physical_detail"] = {"method":"column_transition_physical_v1","status":"not_evaluated",
                                         "checks":[],"unknown_rules":["actual beam-bar and transverse placement context not supplied"]}
            result["not_established"] = ["actual beam-bar and transverse placement context not supplied"]
        return result

    # ---- the declared scope ----
    items.append(_item("one column concrete grade", "project scope (group-specific grades are a later extension)",
                       upper.fc_ksi == lower.fc_ksi, f"lower {lower.fc_ksi:g} ksi, upper {upper.fc_ksi:g} ksi"))
    if upper.fc_ksi != lower.fc_ksi:
        return finish("unsupported", "the two columns use different concrete grades; no joint material rule is declared")
    step_h, step_b = 0.5 * (lower.h_in - upper.h_in), 0.5 * (lower.b_in - upper.b_in)
    if step_h < -1e-9 or step_b < -1e-9:
        items.append(_item("the column above is no larger than the column below", "declared scope", False,
                           f"face steps {step_h:g} in (h) and {step_b:g} in (b)"))
        return finish("unsupported", "a larger column above a smaller one is not a declared transition")
    if upper.bar_size > lower.bar_size:
        items.append(_item("bars above are no larger than bars below", "declared scope", False,
                           f"No. {upper.bar_size} above No. {lower.bar_size}"))
        return finish("unsupported", "larger bars above than below have no declared splice or anchorage")
    fewer = (upper.top_bars <= lower.top_bars and upper.bot_bars <= lower.bot_bars and upper.side_bars <= lower.side_bars)
    if not fewer:
        items.append(_item("no face carries more bars above than below", "declared scope", False,
                           f"lower {lower.top_bars}/{lower.bot_bars}/{lower.side_bars}, upper "
                           f"{upper.top_bars}/{upper.bot_bars}/{upper.side_bars} (top/bottom/side per face)"))
        return finish("unsupported", "bars that start at the floor without a bar below need dowels; none are declared")
    step = max(step_h, step_b)
    if not same_section:
        items.append(_item("each face steps in by less than 3 in", "ACI 318-19 10.7.4.2", step < OFFSET_BEND_FACE_STEP_LIMIT_IN - 1e-9,
                           f"face steps {step_h:g} in (h) and {step_b:g} in (b)"))
        if step >= OFFSET_BEND_FACE_STEP_LIMIT_IN - 1e-9:
            return finish("unsupported", (f"a face step of {step:g} in needs separate dowels lap spliced beside the offset face "
                                          "(10.7.4.2); a lap there is in the end region of a special-frame column (18.7.4.4) and "
                                          "no dowel detail is declared"))

    # ---- splice availability (decides the path of a pair that is not one continuous bar) ----
    lower_case, upper_case = development_case(lower, clear_cover_in), development_case(upper, clear_cover_in)
    favourable = lower_case["favourable"] and upper_case["favourable"]
    lap = lap_length_in(lower.bar_size, upper.bar_size, fc, fy_ksi, favourable)
    center_half = 0.5 * upper_clear_height_in
    lap_permitted = max(lower.bar_size, upper.bar_size) <= LAP_MAX_BAR_SIZE
    lap_fits = lap_permitted and lap <= center_half and lap_allowed
    non_contact = min(lap / 5.0, NON_CONTACT_LAP_MAX_IN)
    splice = {"class_b_lap_in": lap, "center_half_clear_height_in": center_half,
              "lap_splice_permitted_25_5_1_1": lap_permitted, "lap_splice_feasible": lap_fits,
              "non_contact_distance_limit_in": non_contact,
              "splice_type": "class_B_lap_center_half" if lap_fits else "type_2_mechanical_18.2.7",
              "development": {"lower": lower_case, "upper": upper_case,
                              "row_used": "first row" if favourable else "other cases",
                              "ld_larger_bar_in": development_length_in(max(lower.bar_size, upper.bar_size), fc, fy_ksi, favourable),
                              "ld_smaller_bar_in": development_length_in(min(lower.bar_size, upper.bar_size), fc, fy_ksi, favourable)},
              "basis": "18.7.4.4 (center half, tension lap, enclosed by hoops); 25.5.2.2 (bars of different size); "
                       "25.5.1.1 (no laps above No. 11); 25.5.1.3 (non-contact lap); Table 25.4.2.3 (ld); 18.2.7 Type 2 "
                       "mechanical otherwise"}

    # ---- coordinates, pairing and one path per lower bar ----
    lower_bars, upper_bars = bar_coordinates(lower, clear_cover_in), bar_coordinates(upper, clear_cover_in)
    lower_db, upper_db = _BAR[int(lower.bar_size)][0], _BAR[int(upper.bar_size)][0]
    # A lower bar that runs straight into the upper story must lie inside the upper column's hoops.
    inside = clear_cover_in + _BAR[int(upper.stirrup_bar_size)][0] + 0.5 * lower_db
    limit_x, limit_y = 0.5 * upper.b_in - inside, 0.5 * upper.h_in - inside

    def clipped(bar):
        return (max(-limit_x, min(limit_x, bar["x_in"])), max(-limit_y, min(limit_y, bar["y_in"])))

    paths, problems = [], []
    for face in ("top", "bottom", "left", "right"):
        along = "x_in" if face in ("top", "bottom") else "y_in"
        below = sorted((b for b in lower_bars if b["face"] == face), key=lambda b: b[along])
        above = sorted((b for b in upper_bars if b["face"] == face), key=lambda b: b[along])
        partner = _match_along_face([b[along] for b in below], [b[along] for b in above])
        paired = dict(zip(partner, above))
        for index, bar in enumerate(below):
            origin = (bar["x_in"], bar["y_in"])
            entry = {"face": face, "lower_index": bar["index"], "lower_xy_in": list(origin), "upper_index": None,
                     "upper_xy_in": None, "splice_distance_in": None}
            mate = paired.get(index)
            if mate is None:
                target, path = clipped(bar), "terminated"
            else:
                goal = (mate["x_in"], mate["y_in"])
                entry.update(upper_index=mate["index"], upper_xy_in=list(goal))
                distance = math.hypot(goal[0] - origin[0], goal[1] - origin[1])
                same_bar = upper.bar_size == lower.bar_size
                lap_target = clipped(bar)
                lap_distance = math.dist(goal, lap_target)
                lap_geometry_fits = lap_distance >= 0.5 * (lower_db + upper_db) - COORDINATE_TOLERANCE_IN
                usable_lap = lap_fits and lap_geometry_fits
                if lap_fits and not lap_geometry_fits and not (same_bar and distance <= COORDINATE_TOLERANCE_IN):
                    entry["lap_geometry_refusal"] = {
                        "center_separation_in": lap_distance,
                        "minimum_nonintersecting_center_separation_in": 0.5 * (lower_db + upper_db),
                        "reason": "the distinct lap bars would occupy overlapping volumes"}
                # A bar keeps its place in the cage only where the face carries the same number of bars above and
                # below; otherwise the upper bar is another bar and is spliced to this one.
                same_place = len(below) == len(above)
                if same_bar and distance <= COORDINATE_TOLERANCE_IN:
                    target, path = goal, "straight"
                elif same_bar and (same_place or not usable_lap):
                    target, path = goal, "offset"                    # one continuous bar, bent to its upper position
                elif usable_lap:
                    target, path = lap_target, "lap_spliced"
                    entry["splice_distance_in"] = math.hypot(goal[0] - target[0], goal[1] - target[1])
                else:
                    target, path = goal, "mechanically_spliced"      # coaxial: the lower bar goes to the upper position
            offset = (target[0] - origin[0], target[1] - origin[1])
            magnitude = math.hypot(*offset)
            entry.update(path=path, target_xy_in=list(target), offset_xy_in=list(offset), offset_in=magnitude,
                         bent=magnitude > COORDINATE_TOLERANCE_IN)
            if path == "terminated" and entry["bent"]:
                problems.append(f"{face} bar {bar['index']} stops above the joint but lies {magnitude:.3f} in outside the upper cage")
            paths.append(entry)
    counts = {path: sum(1 for p in paths if p["path"] == path) for path in PATHS}
    bent = [p for p in paths if p["bent"]]
    result.update(paths=paths, path_counts=counts, bars_bent=len(bent))
    physical = transition_bar_clearance(paths, lower_db, upper_db)
    result["physical_bar_clearance"] = physical
    items.append(_item("declared longitudinal bar paths do not occupy overlapping volumes",
                       "physical geometry of the declared transition", physical["passes"],
                       (f"{len(physical['collisions'])} intersecting bar pair(s); no undeclared relocation or splice substituted"
                        if not physical["passes"] else "the declared longitudinal bar cylinders do not intersect")))
    spliced = counts["lap_spliced"] + counts["mechanically_spliced"]
    splice["splice_type"] = ("mixed_class_B_lap_and_type_2_mechanical" if counts["lap_spliced"] and counts["mechanically_spliced"]
                             else "type_2_mechanical_18.2.7" if counts["mechanically_spliced"]
                             else "class_B_lap_center_half" if counts["lap_spliced"] else "none")
    splice["lap_geometry_refusals"] = sum("lap_geometry_refusal" in p for p in paths)
    if spliced or counts["terminated"]:
        result["splice"] = splice
    if counts["lap_spliced"]:
        worst = max(p["splice_distance_in"] for p in paths if p["path"] == "lap_spliced")
        items.append(_item("every lap spliced upper bar is within the non-contact lap distance of its lower bar",
                           "ACI 318-19 25.5.1.3", worst <= non_contact + 1e-9,
                           f"largest transverse distance {worst:.2f} in; allowed min(lap / 5, 6 in) = {non_contact:.2f} in"))
    if counts["terminated"]:
        result["terminated_lower_bars"] = counts["terminated"]
        # They run from the top of the joint to the middle of the upper story and stop there.
        result["extra_bar_length_in"] = counts["terminated"] * 0.5 * upper_clear_height_in
        ld = development_length_in(lower.bar_size, fc, fy_ksi, lower_case["favourable"])
        items.append(_item("terminated lower bars stay inside the upper cage without an offset", "declared scope",
                           not problems, "; ".join(problems[:3]) or "every terminated bar lies inside the upper column's hoops"))
        items.append(_item("terminated lower bars are developed above the joint", "ACI 318-19 Table 25.4.2.3; 15.2.6(a)",
                           0.5 * upper_clear_height_in >= max(ld, max(upper.b_in, upper.h_in)),
                           f"they run {0.5 * upper_clear_height_in:.1f} in above the joint; ld = {ld:.1f} in "
                           f"({'first row' if lower_case['favourable'] else 'other cases'}), column depth "
                           f"{max(upper.b_in, upper.h_in):g} in"))

    # ---- offset bends: geometry, ties and their stations ----
    if bent:
        largest = max(p["offset_in"] for p in bent)
        run = OFFSET_BEND_SLOPE * largest
        items.append(_item("the inclined part of the offset bends fits in the joint depth at a slope of at most 1 in 6",
                           "ACI 318-19 10.7.4.1", run <= joint_depth_in + 1e-9,
                           f"largest bar offset {largest:.3f} in needs {run:.2f} in; joint depth {joint_depth_in:g} in"))
        bar_area = _BAR[int(lower.bar_size)][1]
        hoop_area = _BAR[int(lower.stirrup_bar_size)][1]
        legs = lower.stirrup_legs_by_direction or (lower.stirrup_legs,) * 2
        spacing = lower_hoop_spacing_in if lower_hoop_spacing_in is not None else lower.stirrup_spacing_in
        # Per direction: the legs that cross the section in that direction carry 1.5 times the horizontal
        # components on the more heavily loaded side. Legs across the b faces run along h (direction y).
        required, per_set = {}, {}
        for axis, component, coordinate, leg_count in (("x", 0, "x_in", legs[1]), ("y", 1, "y_in", legs[0])):
            sides = [sum(bar_area * abs(p["offset_xy_in"][component]) for p in bent if sign * p["lower_xy_in"][component] > 0.0)
                     for sign in (1.0, -1.0)]
            import Structure_Parameters as sp
            transverse_fy = float((physical_context or {}).get("transverse_fy_ksi",sp.FYT_KSI))
            required[axis] = OFFSET_TIE_FORCE_FACTOR * max(sides) / run * fy_ksi / transverse_fy
            per_set[axis] = leg_count * hoop_area
        sets_required = max(1, max(int(math.ceil(required[a] / per_set[a] - 1e-9)) for a in required))
        sets_at_spacing = 1 + int(math.floor(OFFSET_TIE_DISTANCE_IN / spacing + 1e-9))
        sets_that_fit = 1 + int(math.floor(OFFSET_TIE_DISTANCE_IN / min(spacing, OFFSET_TIE_MIN_SPACING_IN) + 1e-9))
        sets = max(sets_at_spacing, min(sets_required, sets_that_fit))
        station_spacing = spacing if sets <= sets_at_spacing else OFFSET_TIE_DISTANCE_IN / (sets - 1)
        items.append(_item("hoop sets at each bend carry 1.5 times the horizontal component of the offset bars",
                           "ACI 318-19 10.7.6.4", sets_required <= sets_that_fit,
                           f"tie area needed {required['x']:.2f} in2 (x) and {required['y']:.2f} in2 (y): {sets_required} hoop "
                           f"set(s) of the lower column within 6 in of each bend; {sets_at_spacing} are there at its "
                           f"{spacing:g} in spacing and {sets_that_fit} fit at {OFFSET_TIE_MIN_SPACING_IN:g} in"))
        additional = 2 * max(0, min(sets_required, sets_that_fit) - sets_at_spacing)   # two bend points per joint
        result["additional_hoop_sets_per_joint"] = additional
        result["offset_bend"] = {
            "face_step_in": [step_h, step_b], "bars_bent": len(bent), "bar_offset_in": largest,
            "smallest_bar_offset_in": min(p["offset_in"] for p in bent), "inclined_length_in": run,
            "slope_of_steepest_bar": f"1 in {OFFSET_BEND_SLOPE:g}",
            "bend_points_below_joint_top_in": [0.0, run],
            "tie_area_required_in2": required, "tie_area_per_hoop_set_in2": per_set,
            "longitudinal_fy_ksi": fy_ksi, "transverse_fy_ksi": transverse_fy,
            "tie_area_provided_in2": {axis: sets * per_set[axis] for axis in per_set},
            "hoop_sets_required": sets_required, "hoop_sets_at_hoop_spacing": sets_at_spacing,
            "hoop_sets_that_fit": sets_that_fit, "hoop_sets_within_6in": sets,
            "additional_hoop_sets_at_each_bend": additional // 2,
            "tie_stations_from_each_bend_point_in": [k * station_spacing for k in range(sets)],
            "tie_set": {"bar_size": lower.stirrup_bar_size, "legs_across_b_face": legs[0], "legs_across_h_face": legs[1],
                        "basis": "the lower column's hoop set, which continues through the joint"},
            "force_path": ("each offset bar's horizontal component, Ab fy x offset / inclined length, is taken by the hoop "
                           "legs crossing the section in that direction at the two bend points"),
            "minimum_station_spacing_in": OFFSET_TIE_MIN_SPACING_IN,
            "minimum_station_spacing_basis": "declared search restriction, not an ACI minimum"}

    if counts["terminated"]:
        kind = "bar_count_reduction"
    elif not same_section:
        kind = "offset_bends"
    elif spliced:
        kind = "bar_size_reduction"
    elif bent:
        kind = "offset_bends"
    else:
        kind = "same_cage"
    described = ", ".join(f"{counts[path]} {path.replace('_', ' ')}" for path in PATHS if counts[path])
    finished = finish(kind, f"{len(paths)} lower bars: {described}"
                      + (f"; {len(bent)} offset bent inside the joint" if bent and not counts["offset"] == len(bent) else "")
                      + (f"; splices in the center half of the upper story ({splice['splice_type']})" if spliced else ""))
    if (lap_allowed and counts["lap_spliced"] and (physical_context or {}).get("conventional")
            and finished["physical_detail"]["status"] != "pass"):
        # Laps that do not keep their spacing: the same pairs with Type 2 couplers, when that detail is supported.
        alternative = column_transition(lower, upper, fy_ksi=fy_ksi, clear_cover_in=clear_cover_in,
                                        upper_clear_height_in=upper_clear_height_in, joint_depth_in=joint_depth_in,
                                        lower_hoop_spacing_in=lower_hoop_spacing_in, physical_context=physical_context,
                                        lap_allowed=False)
        if alternative["physical_detail"]["status"] == "pass":
            return alternative
    return finished
