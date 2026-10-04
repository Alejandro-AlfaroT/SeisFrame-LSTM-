"""Cage geometry beyond the tie arrangement: hooks and bends, bar clearances, and the assembled 3D joint.

SMRF_Cage_Layout decides which longitudinal bars the hoops and crossties support. This module adds what
the review of 2026-09-27 asked for and the design record still calls "scalar only":

1. Hook and bend geometry (ACI 318-19 Tables 25.3.1 and 25.3.2, seismic hook and crosstie definitions of
   2.3, the end-for-end alternation of crosstie 90-degree hooks of 18.7.5.2(c) and 18.6.4.3): every hoop
   and crosstie with its inside bend diameter, hook extensions and out-to-out dimensions, and the 90-degree
   hook orientation alternating from one tie set to the next.
2. Bar clearances (25.2.1 beams, 25.2.3 columns, 25.2.2 layers) against the declared aggregate.
3. The assembled joint: the beam bars of each direction threaded between the column bars (lanes), the two
   orthogonal top and bottom layers stacked at the joint (the lower one loses effective depth), the slab
   mats against the beam bars, and the hook tails of terminating beam bars inside the joint.

Everything here is pure geometry from the design record and the ACI tables. It reports what fits and what
does not; it changes no strength. The conventions it declares (which direction's bars sit on top, the
clearance used between a beam bar and a column bar it passes) are recorded in the output. Units: inches.
"""
from __future__ import annotations

import math

from Design.SMRF_Cage_Layout import beam_cage, column_cage, face_bar_positions

BAR_DIAMETER = {3: 0.375, 4: 0.5, 5: 0.625, 6: 0.75, 7: 0.875, 8: 1.0, 9: 1.128, 10: 1.27, 11: 1.41, 14: 1.693, 18: 2.257}
SEISMIC_HOOK_MIN_EXTENSION_IN = 3.0          # ACI 318-19 2.3, seismic hook: 6db and 3 in
INTERLAYER_CLEAR_MIN_IN = 1.0                # 25.2.2
# Retain the earlier 1-in displacement proposals as diagnostics. No bar path or
# revised elevation was installed by that convention, so an overlap still fails.
SLAB_BOTTOM_MAT_DISPLACEMENT_MAX_IN = 1.0
STACKING_CONVENTIONS = ("x_over_y", "y_over_x")


def bar_diameter(size):
    try:
        return BAR_DIAMETER[int(size)]
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Unknown bar size {size!r}") from exc


# ---- hooks and bends ------------------------------------------------------------------------------
def hook_geometry(bar_size, kind):
    """Inside bend diameter and straight extension of one hook, with the provision it comes from.

    kind: "seismic_135" (hoop ends and the seismic end of a crosstie), "crosstie_90" (the other end),
          "development_90" and "development_180" (standard hooks of longitudinal bars, Table 25.3.1).
    """
    size = int(bar_size)
    db = bar_diameter(size)
    if kind in ("seismic_135", "crosstie_90", "tie_180"):
        if size > 8:
            raise ValueError("Table 25.3.2 covers stirrup, tie and hoop bars up to No. 8")
        bend = (4.0 if size <= 5 else 6.0) * db
        if kind == "seismic_135":
            extension, provision = max(6.0 * db, SEISMIC_HOOK_MIN_EXTENSION_IN), "2.3 seismic hook; Table 25.3.2 (135-degree)"
        elif kind == "crosstie_90":
            extension, provision = (6.0 if size <= 5 else 12.0) * db, "2.3 crosstie; Table 25.3.2 (90-degree)"
        else:
            extension, provision = max(4.0 * db, 2.5), "Table 25.3.2 (180-degree)"
        return {"kind": kind, "bar_size": size, "db_in": db, "inside_bend_diameter_in": bend,
                "extension_in": extension, "provision": provision}
    if kind in ("development_90", "development_180"):
        bend = (6.0 if size <= 8 else 8.0 if size <= 11 else 10.0) * db
        if kind == "development_90":
            extension = 12.0 * db
        else:
            extension = max(4.0 * db, 2.5)
        return {"kind": kind, "bar_size": size, "db_in": db, "inside_bend_diameter_in": bend,
                "extension_in": extension, "provision": "Table 25.3.1"}
    raise ValueError(f"Unknown hook kind {kind!r}")


def tie_set(width_in, depth_in, clear_cover_in, tie_size, crosstie_spans, set_index=0):
    """One hoop set: the perimeter hoop and its crossties, dimensioned, with the 90-degree ends alternated.

    ``crosstie_spans`` is a list of (axis, position_in, length_in): a crosstie along ``axis`` ("x" runs
    across the width, "y" across the depth) at the given lateral position, engaging bars ``length_in``
    apart centre to centre. ``set_index`` alternates the side of the 90-degree hook from one set to the
    next (18.7.5.2(c) for columns, 18.6.4.3 for beams).
    """
    db = bar_diameter(tie_size)
    seismic = hook_geometry(tie_size, "seismic_135")
    ninety = hook_geometry(tie_size, "crosstie_90")
    hoop = {"element": "perimeter_hoop", "bar_size": int(tie_size),
            "out_to_out_in": [width_in - 2.0 * clear_cover_in, depth_in - 2.0 * clear_cover_in],
            "inside_bend_diameter_in": seismic["inside_bend_diameter_in"],
            "hooks": ["seismic_135", "seismic_135"], "hook_extension_in": seismic["extension_in"],
            "hook_location": "one corner, both ends with seismic hooks engaging the corner bar (25.7.4.2)"}
    ties = []
    for n, (axis, position, length) in enumerate(crosstie_spans):
        ninety_end = ("far" if (set_index + n) % 2 == 0 else "near")
        ties.append({"element": "crosstie", "bar_size": int(tie_size), "axis": axis, "position_in": position,
                     "engaged_bar_spacing_in": length,
                     "cut_length_in": length + db + seismic["extension_in"] + ninety["extension_in"]
                     + math.pi / 2.0 * (seismic["inside_bend_diameter_in"] + db) * 0.5 * (3.0 / 4.0 + 1.0 / 2.0),
                     "hooks": {"near": "crosstie_90" if ninety_end == "near" else "seismic_135",
                               "far": "crosstie_90" if ninety_end == "far" else "seismic_135"},
                     "ninety_degree_end": ninety_end,
                     "alternation": "90-degree hook alternates end for end in successive sets (18.7.5.2(c) / 18.6.4.3)"})
    return {"set_index": set_index, "hoop": hoop, "crossties": ties, "hook_geometry": {"seismic_135": seismic, "crosstie_90": ninety}}


# ---- clearances -----------------------------------------------------------------------------------
def clear_spacing_limit(member, bar_db_in, aggregate_in):
    """Minimum clear spacing between parallel bars in a layer (25.2.1 beams, 25.2.3 columns)."""
    if member == "column":
        return max(1.5, 1.5 * bar_db_in, 4.0 / 3.0 * aggregate_in)
    return max(1.0, bar_db_in, 4.0 / 3.0 * aggregate_in)


def layer_clearances(positions, bar_db_in):
    return [b - a - bar_db_in for a, b in zip(positions, positions[1:])]


# ---- the assembled joint --------------------------------------------------------------------------
def thread_bars(n, bar_db, lo, hi, obstacles, obstacle_db, clear_min):
    """Place ``n`` bars of diameter ``bar_db`` between ``lo`` and ``hi`` (centre limits), keeping
    ``clear_min`` clear from each other and from the obstacle bars (column bars the beam bars pass).
    Greedy from the left. Returns the positions placed (fewer than n when they do not fit)."""
    blocked = [(o - (obstacle_db + bar_db) / 2.0 - clear_min, o + (obstacle_db + bar_db) / 2.0 + clear_min) for o in obstacles]
    placed, x = [], lo
    while len(placed) < n and x <= hi + 1e-9:
        for a, b in blocked:
            if a < x < b:
                x = b
        if x > hi + 1e-9:
            break
        placed.append(x)
        x += bar_db + clear_min
    return placed


def column_bars_plan(b_in, h_in, clear_cover_in, hoop_db_in, bar_db_in, top_bars, side_bars):
    """Plan positions (x, y) of every column bar, from the cage layout SMRF_Cage_Layout uses.

    The faces of width b (perpendicular to X, at x = c and x = h - c) carry ``top_bars`` bars along Y;
    the faces of depth h (at y = c and y = b - c) carry ``side_bars`` interior bars along X plus the corners.
    """
    cage = column_cage(b_in, h_in, clear_cover_in, hoop_db_in, bar_db_in, top_bars, side_bars)
    c = clear_cover_in + hoop_db_in + bar_db_in / 2.0
    bars = []
    for y in cage["faces"]["b_face"]:
        bars += [(c, y), (h_in - c, y)]
    for x in cage["faces"]["h_face"][1:-1]:
        bars += [(x, c), (x, b_in - c)]
    return sorted(set((round(x, 6), round(y, 6)) for x, y in bars))


def beam_bars_threading(b_col_in, h_col_in, b_beam_in, beam_cover_in, beam_hoop_db_in, beam_db_in, bars_per_face,
                        column_bars, col_db_in, clear_in):
    """Per direction: how many beam bars pass between the column bars in one layer, and where.

    An X-beam runs along X: its bars at lateral positions y must avoid every column bar whose y lies in
    the beam band; a Y-beam likewise against the column bars' x positions.
    """
    directions = {}
    for axis, width_axis_extent in (("x", b_col_in), ("y", h_col_in)):
        offset = (width_axis_extent - b_beam_in) / 2.0
        inner_lo = offset + beam_cover_in + beam_hoop_db_in + beam_db_in / 2.0
        inner_hi = offset + b_beam_in - beam_cover_in - beam_hoop_db_in - beam_db_in / 2.0
        lateral = [y if axis == "x" else x for x, y in column_bars]
        obstacles = sorted(set(p for p in lateral if offset - col_db_in < p < offset + b_beam_in + col_db_in))
        nominal = [offset + p for p in face_bar_positions(b_beam_in, beam_cover_in, beam_hoop_db_in, beam_db_in, bars_per_face)]
        conflicts = [p for p in nominal if any(abs(p - o) < (col_db_in + beam_db_in) / 2.0 + clear_in - 1e-9 for o in obstacles)]
        placed = thread_bars(bars_per_face, beam_db_in, inner_lo, inner_hi, obstacles, col_db_in, clear_in)
        per_layer = len(placed)
        directions[axis] = {
            "beam_band_in": [offset, offset + b_beam_in], "bar_centre_limits_in": [inner_lo, inner_hi],
            "column_bar_lanes_blocked_in": obstacles, "nominal_positions_in": nominal,
            "nominal_positions_in_conflict": conflicts,
            "threaded_positions_in": placed, "bars_per_layer_that_fit": per_layer,
            "layers_needed": math.ceil(bars_per_face / per_layer) if per_layer else None,
            "fits_in_one_layer": per_layer >= bars_per_face,
            "clearance_used_in": clear_in,
        }
    return directions


def column_cage_admits_beam_bars(b_col_in, h_col_in, col_cover_in, col_hoop_db_in, col_db_in, top_bars, side_bars,
                                 b_beam_in, beam_cover_in, beam_hoop_db_in, beam_db_in, bars_per_face, aggregate_in,
                                 max_layers=1):
    """The column cage rule (2026-09-27): do the beam bars of both directions pass this cage in at most
    ``max_layers`` layers (one until the two-layer decision of the same day)?

    The same geometry as the capacity check beam.bars_thread_column, so a cage the rule offers is one
    the check accepts. Returns {"passes", "bars_per_layer": {axis: n}, "layers_needed": {axis: n}, ...}.
    """
    column_bars = column_bars_plan(b_col_in, h_col_in, col_cover_in, col_hoop_db_in, col_db_in, top_bars, side_bars)
    clear = clear_spacing_limit("beam", beam_db_in, aggregate_in)
    directions = beam_bars_threading(b_col_in, h_col_in, b_beam_in, beam_cover_in, beam_hoop_db_in, beam_db_in,
                                     bars_per_face, column_bars, col_db_in, clear)
    return {"passes": all(d["layers_needed"] is not None and d["layers_needed"] <= max_layers for d in directions.values()),
            "max_layers": max_layers,
            "bars_per_layer": {axis: d["bars_per_layer_that_fit"] for axis, d in directions.items()},
            "layers_needed": {axis: d["layers_needed"] for axis, d in directions.items()},
            "directions": directions}


def layer_slots(n_near, n_far, order):
    """Mirrors Structure_Parameters.beam_bar_layer_slots: slot indices (0 nearest the face, one pitch apart)
    of the near and far directions' layers under the "blocked" or "interleaved" order."""
    if order == "blocked":
        return list(range(n_near)), list(range(n_near, n_near + n_far))
    if order != "interleaved":
        raise ValueError(f"layer order must be 'blocked' or 'interleaved', not {order!r}")
    near, far, slot = [], [], 0
    for i in range(max(n_near, n_far)):
        if i < n_near:
            near.append(slot)
            slot += 1
        if i < n_far:
            far.append(slot)
            slot += 1
    return near, far


def joint_assembly(record, stacking="x_over_y", pass_clearance_in=None, max_layers=None, layer_order=None):
    """Fit of the beam bars, column bars and slab mats at a joint, and what the stacking costs in depth.

    ``stacking`` names the direction whose beam bars sit on top at the joint (top layers; bottom layers
    stack in the mirrored order so each direction gives up depth at one face only). ``max_layers`` is
    how many layers a direction's bars may take before the fit fails (default: the record's
    reinforcement.beam_bar_stacking.max_layers, else one). ``layer_order`` says how the two
    directions' layers follow one another at a face: "blocked" (the upper cage, then the lower) or
    "interleaved" (upper, lower, upper, lower, ...); default the record's, else blocked.
    """
    if stacking not in STACKING_CONVENTIONS:
        raise ValueError(f"stacking must be one of {STACKING_CONVENTIONS}")
    s, r = record["sections"], record["reinforcement"]
    if max_layers is None:
        max_layers = int((r.get("beam_bar_stacking") or {}).get("max_layers", 1))
    if layer_order is None:
        layer_order = (r.get("beam_bar_stacking") or {}).get("layer_order", "blocked")
    slab = record.get("slab") or {}
    layout = ((record.get("slab_reinforcement") or {}).get("layout") or {}).get("layers") or {}
    aggregate = float((record.get("materials") or {}).get("aggregate_size_in", 0.75))
    bc, hc, bw, hb = s["b_col_in"], s["h_col_in"], s["b_beam_in"], s["h_beam_in"]
    col_db, col_tie_db = bar_diameter(r["col_bar_size"]), bar_diameter(r["col_stirrup_bar_size"])
    beam_db, beam_tie_db = bar_diameter(r["beam_bar_size"]), bar_diameter(r["beam_stirrup_bar_size"])
    col_cover, beam_cover = r["col_clear_cover_in"], r["beam_clear_cover_in"]
    clear = pass_clearance_in if pass_clearance_in is not None else clear_spacing_limit("beam", beam_db, aggregate)

    # Column bar plan positions and the beam bars threading them (shared with the column cage rule).
    column_bars = column_bars_plan(bc, hc, col_cover, col_tie_db, col_db, r["col_top_bars"], r["col_side_bars"])
    directions = beam_bars_threading(bc, hc, bw, beam_cover, beam_tie_db, beam_db, r["beam_top_bars"],
                                     column_bars, col_db, clear)

    # Vertical stacking of the two orthogonal cages at the joint (top and, mirrored, bottom). Where the
    # lanes force a second layer (max_layers > 1) a direction's bars split into layers one pitch apart,
    # and the lower direction's first layer sits below the upper direction's last.
    upper, lower = ("x", "y") if stacking == "x_over_y" else ("y", "x")
    centroid = r["beam_longitudinal_centroid_offset_in"]
    inter = max(INTERLAYER_CLEAR_MIN_IN, beam_db)
    pitch = beam_db + inter
    n_bars = r["beam_top_bars"]

    def split(n, fit):
        # Mirrors Structure_Parameters._split_into_layers: one elevation when everything fits, when no
        # lane exists, or when more layers than the limit would be needed (the fit check fails then).
        if max_layers <= 1 or not fit or fit >= n:
            return [n]
        layers, remaining = [], n
        while remaining > 0:
            layers.append(min(fit, remaining))
            remaining -= layers[-1]
        return layers if len(layers) <= max_layers else [n]

    layers_by = {axis: split(n_bars, directions[axis]["bars_per_layer_that_fit"]) for axis in ("x", "y")}
    upper_slots, lower_slots = layer_slots(len(layers_by[upper]), len(layers_by[lower]), layer_order)
    upper_offsets = [centroid + k * pitch for k in upper_slots]
    lower_offsets = [centroid + k * pitch for k in lower_slots]

    def group(layers, offsets):
        return sum(c * o for c, o in zip(layers, offsets)) / sum(layers)

    upper_centroid, lower_centroid = group(layers_by[upper], upper_offsets), group(layers_by[lower], lower_offsets)
    d_nominal = hb - centroid
    d_lower = hb - lower_centroid
    as_ = n_bars * r["beam_bar_area_in2"]
    fy, fc = (record.get("materials") or {}).get("fy_ksi", 60.0), s["fc_beam_ksi"]
    a = as_ * fy / (0.85 * fc * bw)
    lever_ratio = (d_lower - a / 2.0) / (d_nominal - a / 2.0)
    stacking_result = {
        "convention": stacking, "upper_direction": upper, "lower_direction": lower, "max_layers": max_layers,
        "layer_order": layer_order,
        "layers": {upper: layers_by[upper], lower: layers_by[lower]},
        "layer_offsets_from_face_in": {upper: upper_offsets, lower: lower_offsets},
        "upper_layer_centroid_from_face_in": upper_centroid, "lower_layer_centroid_from_face_in": lower_centroid,
        "interlayer_clear_in": inter, "layer_pitch_in": pitch,
        "effective_depth_nominal_in": d_nominal, "effective_depth_lower_in": d_lower,
        "lower_direction_lever_arm_ratio": lever_ratio,
        "strength_basis_note": (f"the record's strengths use d = {d_nominal:g} in for both directions; the lower "
                                f"direction's face bars at the joint sit at {lower_centroid:g} in, a lever-arm ratio of "
                                f"{lever_ratio:.3f} (rectangular-block estimate) at the faces where its layer is the lower one"),
        "bottom_layers": f"mirrored: {lower} bars nearer the bottom face, {upper} bars above them",
    }

    # Slab mats against the top layers (elevations from the beam top face; the slab is flush with it).
    # Only the mats that CROSS a beam meet its bars: a mat parallel to the beam is omitted within the
    # beam width, the beam's own bars taking its place (declared convention). Crossing bars may touch;
    # a physical overlap fails, a crossing with less than 1 in clear is flagged as tight but placeable.
    mats = []
    mats_placed = True
    for name, layer in layout.items():
        if not name.endswith("top") and not name.endswith("bottom"):
            continue
        if any(key not in layer for key in ("bar_size", "clear_cover_outer_mat_in", "layer", "axis")):
            # A layout that carries areas and spacings only (older or reduced records) places no mats;
            # the beam-bar stacking is still assembled and the result says the mats were not placed.
            mats_placed = False
            continue
        mat_db = bar_diameter(layer["bar_size"])
        if name.endswith("top"):
            depth = layer["clear_cover_outer_mat_in"] + mat_db / 2.0 + (0.0 if layer["layer"] == "outer" else mat_db)
        else:
            depth = slab.get("thickness_in", 0.0) - layer["clear_cover_outer_mat_in"] - mat_db / 2.0 - (0.0 if layer["layer"] == "outer" else mat_db)
        mats.append({"mat": name, "db_in": mat_db, "centroid_from_top_in": depth, "runs_along": layer["axis"]})
    beam_layers = [{"layer": f"{axis} beam top bars" + (f" layer {k + 1}" if len(offsets) > 1 else ""),
                    "direction": axis, "centroid_from_top_in": offset, "db_in": beam_db}
                   for axis, offsets in ((upper, upper_offsets), (lower, lower_offsets))
                   for k, offset in enumerate(offsets)]
    clashes, tight, displaced = [], [], []
    for mat in mats:
        for bl in beam_layers:
            if mat["runs_along"] == bl["direction"]:
                continue                                   # parallel mat: omitted within the beam width
            gap = abs(mat["centroid_from_top_in"] - bl["centroid_from_top_in"]) - (mat["db_in"] + bl["db_in"]) / 2.0
            if gap < 0.0 and mat["mat"].endswith("bottom") and -gap <= SLAB_BOTTOM_MAT_DISPLACEMENT_MAX_IN:
                proposal = {"between": [mat["mat"], bl["layer"]], "displacement_in": -gap,
                            "status": "unresolved", "applied": False,
                            "note": "a displacement magnitude is not a resolved bar path; the stored mat elevation "
                                    "still overlaps the beam bar and remains a failed geometry check"}
                displaced.append(proposal)
                clashes.append({"between": proposal["between"], "overlap_in": -gap,
                                "note": proposal["note"]})
            elif gap < 0.0:
                clashes.append({"between": [mat["mat"], bl["layer"]], "overlap_in": -gap,
                                "note": "crossing slab bar and beam bar occupy the same elevation"})
            elif gap < INTERLAYER_CLEAR_MIN_IN:
                tight.append({"between": [mat["mat"], bl["layer"]], "clear_in": gap,
                              "note": "crossing bars with less than 1 in clear: placeable, bars may touch"})

    # Hook tails of terminating beam bars inside an exterior joint (Table 25.3.1 90-degree hook).
    hook = hook_geometry(r["beam_bar_size"], "development_90")
    deepest_top_layer = max(upper_offsets + lower_offsets)
    tail_bottom = deepest_top_layer + hook["inside_bend_diameter_in"] / 2.0 + beam_db + hook["extension_in"]
    embedment = hc - col_cover - col_tie_db - beam_db / 2.0
    anchorage = ((record.get("capacity_design") or {}).get("anchorage") or {}).get("directions") or {}
    hooks = {"hook": hook, "tail_reaches_from_top_face_in": tail_bottom, "joint_depth_in": hb,
             "tail_within_joint_depth": tail_bottom <= hb - beam_cover,
             "straight_embedment_to_hook_in": embedment,
             "ldh_required_in": {k: v.get("ldh_required_in") for k, v in anchorage.items()},
             "note": "the tail turns down inside the column core past the far-face bars; its lane is the bar's own lane"}

    # A second layer, where the lanes force one, moves the centroid of the bars down by half a layer.
    for axis, d in directions.items():
        if d["layers_needed"] and d["layers_needed"] > 1:
            second = centroid + beam_db + inter
            group_centroid = (centroid * d["bars_per_layer_that_fit"] + second * (r["beam_top_bars"] - d["bars_per_layer_that_fit"])) / r["beam_top_bars"]
            d["two_layer_centroid_from_face_in"] = group_centroid
            d["two_layer_lever_arm_ratio"] = (hb - group_centroid - a / 2.0) / (d_nominal - a / 2.0)

    checks = []
    for axis, d in directions.items():
        d["fits_within_layer_limit"] = d["layers_needed"] is not None and d["layers_needed"] <= max_layers
        checks.append({"rule": f"fit: beam bars pass between the column bars in at most {max_layers} layer(s) (clearance per 25.2.1)",
                       "direction": axis,
                       "passes": d["fits_within_layer_limit"],
                       "detail": f"{r['beam_top_bars']} #{r['beam_bar_size']} bars, {d['bars_per_layer_that_fit']} fit per layer "
                                 f"between column bar lanes at {[round(o, 1) for o in d['column_bar_lanes_blocked_in']]} in; "
                                 f"layers needed {d['layers_needed']}, placed as {layers_by[axis]}; "
                                 f"nominal positions in conflict: {len(d['nominal_positions_in_conflict'])}"
                                 + (f"; two layers would put the group centroid {d['two_layer_centroid_from_face_in']:.2f} in from the face "
                                    f"(lever-arm ratio {d['two_layer_lever_arm_ratio']:.3f})" if d.get("two_layer_centroid_from_face_in") else "")})
    checks.append({"rule": "stacking of orthogonal layers at the joint", "passes": True,
                   "detail": f"{lower} bars {lower_centroid:g} in from the face at the joint (lever-arm ratio {lever_ratio:.3f}); "
                             "not reflected in the record's strengths", "requires_strength_update": lever_ratio < 0.999})
    for cl in clashes:
        checks.append({"rule": "crossing slab mats against beam top bars", "passes": False, "detail": cl})
    for item in displaced:
        checks.append({"rule": "crossing slab bottom mat displacement unresolved", "passes": False, "detail": item})
    if not clashes:
        checks.append({"rule": "crossing slab mats against beam top bars", "passes": True,
                       "detail": f"no overlap; {len(tight)} tight crossing(s) under 1 in clear" if tight else "no overlap, 1 in clear kept"})
    checks.append({"rule": "Table 25.3.1 hook tail inside the joint", "passes": hooks["tail_within_joint_depth"],
                   "detail": f"tail reaches {tail_bottom:.1f} in from the top face; joint depth {hb:g} in"})
    return {"column_bars_plan_in": column_bars, "directions": directions, "stacking": stacking_result,
            "slab_mats": mats, "slab_mats_placed": mats_placed, "beam_top_layers": beam_layers,
            "slab_clashes": clashes, "slab_tight_crossings": tight, "slab_bottom_mat_displacements": displaced,
            "exterior_hooks": hooks, "checks": checks, "passes": all(c["passes"] for c in checks)}


# ---- full evaluation ------------------------------------------------------------------------------
def column_cage_geometry(record):
    s, r = record["sections"], record["reinforcement"]
    aggregate = float((record.get("materials") or {}).get("aggregate_size_in", 0.75))
    bc, hc = s["b_col_in"], s["h_col_in"]
    db, tie_db = bar_diameter(r["col_bar_size"]), bar_diameter(r["col_stirrup_bar_size"])
    legs = r["col_stirrup_legs_by_direction"]
    cage = column_cage(bc, hc, r["col_clear_cover_in"], tie_db, db, r["col_top_bars"], r["col_side_bars"], legs=legs)
    c = r["col_clear_cover_in"] + tie_db + db / 2.0
    spans = []
    if cage.get("arrangement"):
        # crossties across the b faces engage b-face bars: they run along X between x = c and x = h - c
        for pos, sup in zip(cage["arrangement"]["b_face"]["positions_in"], cage["arrangement"]["b_face"]["supported"]):
            if sup and pos not in (cage["arrangement"]["b_face"]["positions_in"][0], cage["arrangement"]["b_face"]["positions_in"][-1]):
                spans.append(("x", pos, hc - 2.0 * c))
        for pos, sup in zip(cage["arrangement"]["h_face"]["positions_in"], cage["arrangement"]["h_face"]["supported"]):
            if sup and pos not in (cage["arrangement"]["h_face"]["positions_in"][0], cage["arrangement"]["h_face"]["positions_in"][-1]):
                spans.append(("y", pos, bc - 2.0 * c))
    sets = [tie_set(hc, bc, r["col_clear_cover_in"], r["col_stirrup_bar_size"], spans, set_index=k) for k in range(2)]
    limit = clear_spacing_limit("column", db, aggregate)
    clearances = {face: layer_clearances(pos, db) for face, pos in cage["faces"].items()}
    clear_ok = all(min(v) >= limit - 1e-9 for v in clearances.values() if v)
    checks = list(cage.get("checks", []))
    checks.append({"rule": "25.2.3 column bar clear spacing", "passes": clear_ok,
                   "detail": f"min clear {min(min(v) for v in clearances.values() if v):.2f} in vs limit {limit:.2f} in"})
    return {"arrangement": cage, "tie_sets": sets, "clearances_in": clearances, "clear_limit_in": limit,
            "hoop_spacing_in": r["col_stirrup_spacing_in"], "checks": checks,
            "passes": bool(cage.get("constructible")) and all(c["passes"] for c in checks)}


def beam_cage_geometry(record):
    s, r = record["sections"], record["reinforcement"]
    aggregate = float((record.get("materials") or {}).get("aggregate_size_in", 0.75))
    bw, hb = s["b_beam_in"], s["h_beam_in"]
    db, tie_db = bar_diameter(r["beam_bar_size"]), bar_diameter(r["beam_stirrup_bar_size"])
    cage = beam_cage(bw, r["beam_clear_cover_in"], tie_db, db, r["beam_top_bars"], r["beam_bot_bars"], legs=r["beam_stirrup_legs"])
    c = r["beam_clear_cover_in"] + tie_db + db / 2.0
    spans = []
    if cage.get("arrangement"):
        top = cage["arrangement"]["top"]
        for pos, sup in zip(top["positions_in"], top["supported"]):
            if sup and pos not in (top["positions_in"][0], top["positions_in"][-1]):
                spans.append(("y", pos, hb - 2.0 * c))
    sets = [tie_set(bw, hb, r["beam_clear_cover_in"], r["beam_stirrup_bar_size"], spans, set_index=k) for k in range(2)]
    limit = clear_spacing_limit("beam", db, aggregate)
    clearances = {face: layer_clearances(pos, db) for face, pos in cage["faces"].items()}
    clear_ok = all(min(v) >= limit - 1e-9 for v in clearances.values() if v)
    checks = list(cage.get("checks", []))
    checks.append({"rule": "25.2.1 beam bar clear spacing", "passes": clear_ok,
                   "detail": f"min clear {min(min(v) for v in clearances.values() if v):.2f} in vs limit {limit:.2f} in"})
    return {"arrangement": cage, "tie_sets": sets, "clearances_in": clearances, "clear_limit_in": limit,
            "hoop_spacing_in": r["beam_stirrup_spacing_in"], "checks": checks,
            "passes": bool(cage.get("constructible")) and all(c["passes"] for c in checks)}


def evaluate_cage_geometry(record, stacking=None):
    """Items 1 and 2 of the 2026-09-27 detailing review for one design record.

    Both stacking conventions are evaluated; ``stacking`` picks the one reported as ``joint``, else the
    convention with the fewer failed checks (ties to x_over_y).
    """
    column = column_cage_geometry(record)
    beam = beam_cage_geometry(record)
    joints = {name: joint_assembly(record, stacking=name) for name in STACKING_CONVENTIONS}
    if stacking is None:
        # Fewer failed checks first, then fewer displaced bottom-mat crossings (a mat left where it is drawn
        # is preferred to one moved by the placement convention), then the declared order.
        stacking = min(STACKING_CONVENTIONS, key=lambda n: (sum(1 for c in joints[n]["checks"] if not c["passes"]),
                                                           len(joints[n]["slab_bottom_mat_displacements"]),
                                                           STACKING_CONVENTIONS.index(n)))
    joint = joints[stacking]
    return {"schema_version": "smrf_cage_geometry_v1", "column": column, "beam": beam, "joint": joint,
            "joint_by_stacking": joints, "stacking_selected": stacking,
            "summary": {"column_cage_passes": column["passes"], "beam_cage_passes": beam["passes"],
                        "joint_assembly_passes": joint["passes"],
                        "failed_checks": [c for part in (column, beam, joint) for c in part["checks"] if not c["passes"]]},
            "scope": "geometry of the record's scalar quantities against ACI 318-19 Chapter 25 and 18.6/18.7; "
                     "declared conventions: stacking order, beam-bar-to-column-bar clearance = 25.2.1 limit, slab "
                     "mats parallel to a beam omitted within its width; no strength is recomputed here"}
