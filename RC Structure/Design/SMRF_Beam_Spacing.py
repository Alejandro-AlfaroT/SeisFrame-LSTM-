"""Validate saved beam rows using independently recomputed column-cage lanes.

Spacing evidence only: hooks, crosstie engagement and fabrication are separate.
"""
import math

from Design.SMRF_Beam_Slab_Strength import validate_bar_rows
from Design.SMRF_Cage_Geometry import beam_bars_threading, clear_spacing_limit, column_bars_plan, layer_slots
from Design.SMRF_Common import make_check, not_evaluated

GEOMETRY_EPS_IN = 1e-9


def _number(data, key, minimum=0.0, positive=True):
    value = data.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric, not boolean")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Missing or invalid {key}") from exc
    if not math.isfinite(value) or value < minimum or (positive and value <= minimum):
        raise ValueError(f"Invalid {key}")
    return value


def _count(data, key, minimum=1):
    value = _number(data, key, minimum, positive=False)
    if not value.is_integer():
        raise ValueError(f"{key} must be an integer")
    return int(value)


def _check(name, actual, required, location, details):
    """Preserve raw values, using only a local absolute geometry tolerance."""
    result = make_check(name, "ACI 318-19 25.2.1; 25.2.2", actual, required, ">=", "in", location,
                        {**details, "numerical_tolerance_in": GEOMETRY_EPS_IN})
    if result["status"] == "fail" and actual >= required - GEOMETRY_EPS_IN:
        result["status"] = "pass"
    return result


def evaluate_layered_beam_spacing(beam, column):
    """Return checks; malformed or inconsistent declared rows raise ValueError.

    Scalar SMRF_Detailing inputs plus beam.bar_stacking, copied directly from
    reinforcement.beam_bar_stacking. Saved lane coordinates are never trusted.
    """
    stack = beam["bar_stacking"]
    if not isinstance(stack, dict):
        raise ValueError("bar_stacking must be a dictionary")
    limit = _count(stack, "max_layers")
    convention, order = stack.get("convention"), stack.get("layer_order")
    if convention not in ("x_over_y", "y_over_x", "none") or order not in ("blocked", "interleaved"):
        raise ValueError("Missing or invalid stacking convention or layer_order")
    layers = stack.get("layers")
    if not isinstance(layers, dict) or not all(isinstance(layers.get(a), dict) for a in ("x", "y")):
        raise ValueError("Both directional beam row declarations are required")
    b, h, db = (_number(beam, k) for k in ("b_in", "h_in", "bar_db_in"))
    tie, cover = _number(beam, "stirrup_db_in"), _number(beam, "clear_cover_in", positive=False)
    base = cover + tie + db / 2
    if 2 * base >= min(b, h):
        raise ValueError("Beam bar centers must fit inside the section")
    clear = clear_spacing_limit("beam", db, _number(beam, "aggregate_size_in"))
    totals = {"top": _count(beam, "n_top", 2), "bottom": _count(beam, "n_bottom", 2)}
    rows = {"x": {}, "y": {}}
    for axis in rows:
        for face, total in totals.items():
            declaration = layers[axis].get(face)
            if not isinstance(declaration, dict):
                raise ValueError(f"Missing {axis}/{face} rows")
            count = _count(declaration, "layers")
            _number(declaration, "centroid_in")
            offsets = declaration.get("offsets_in")
            if not isinstance(offsets, (list, tuple)) or any(isinstance(v, bool) for v in offsets):
                raise ValueError(f"Invalid {axis}/{face} offsets")
            rows[axis][face] = validate_bar_rows(declaration, total, f"{axis}/{face}", h)
            if count != len(rows[axis][face]) or count > limit:
                raise ValueError(f"{axis}/{face} declared layers contradict the rows or exceed the limit")
    # Match the same convention/elevations consumed by saved strengths. A
    # different layout must not silently replace the one being qualified.
    pitch = db + max(1.0, db)
    upper, lower = ("x", "y") if convention == "x_over_y" else ("y", "x")
    for face in totals:
        near, far = (upper, lower) if face == "top" else (lower, upper)
        if convention == "none":
            if any(len(rows[a][face]) != 1 for a in rows):
                raise ValueError("Stacking none requires a single row per face")
            slots = {"x": [0], "y": [0]}
        else:
            ns, fs = layer_slots(len(rows[near][face]), len(rows[far][face]), order)
            slots = {near: ns, far: fs}
        for axis in rows:
            if any(not math.isclose(offset, base + slot * pitch, rel_tol=0, abs_tol=GEOMETRY_EPS_IN)
                   for (_, offset), slot in zip(rows[axis][face], slots[axis])):
                raise ValueError(f"{axis}/{face} offsets contradict the declared stacking geometry")
    directions = None
    if convention != "none":
        cb, ch, cdb, ctie = (_number(column, k) for k in ("b_in", "h_in", "bar_db_in", "stirrup_db_in"))
        cc = _number(column, "clear_cover_in", positive=False)
        nt, nb = _count(column, "n_top", 2), _count(column, "n_bottom", 2)
        side = _count(column, "n_side_per_face", 0)
        if nt != nb or 2 * (cc + ctie + cdb / 2) >= min(cb, ch):
            raise ValueError("Column threading requires valid symmetric top/bottom face geometry")
        bars = column_bars_plan(cb, ch, cc, ctie, cdb, nt, side)
        directions = beam_bars_threading(cb, ch, b, cover, tie, db, max(totals.values()), bars, cdb, clear)
    checks = []
    for face in totals:
        evidence, spacings, unplaced, cover_margins = {}, [], [], []
        for axis in rows:
            evidence[axis] = []
            for index, (count, offset) in enumerate(rows[axis][face]):
                positions = (directions[axis]["threaded_positions_in"][:count] if directions is not None else
                             [base + i * (b - 2 * base) / (count - 1) for i in range(count)])
                if any(isinstance(p, bool) or not math.isfinite(p) for p in positions):
                    raise ValueError(f"Nonfinite or invalid {axis}/{face} lane coordinate")
                if len(positions) != count:
                    unplaced.append(f"{axis}/{face}/row:{index + 1}")
                band_offset = ((cb if axis == "x" else ch) - b) / 2 if directions is not None else 0.0
                lo, hi = band_offset + base, band_offset + b - base
                cover_margins.extend(min(p - lo, hi - p) for p in positions)
                gaps = [right - left - db for left, right in zip(positions, positions[1:])]
                spacings.extend(gaps)
                evidence[axis].append({"bar_count": count, "offset_in": offset,
                                       "positions_in": positions, "clear_spacings_in": gaps})
        details = {"basis": "minimum across all declared rows in both directions", "rows": evidence,
                   "coordinate_basis": "column plan for threaded rows; beam-local for stacking none"}
        if unplaced:
            checks.append(make_check("beam.bar_layer_capacity", "Declared beam rows / column-bar lanes", 0, 1, "==",
                                     location=face, details={**details, "unplaced_rows": unplaced}))
            checks.append(not_evaluated("beam.bar_clear_spacing", "ACI 318-19 25.2.1",
                                        "Not all declared bars fit the recomputed lanes", face))
        elif spacings:
            checks.append(_check("beam.bar_clear_spacing", min(spacings), clear, face, details))
        else:
            checks.append(make_check("beam.bar_clear_spacing", "ACI 318-19 25.2.1", 1, 1, "==", location=face,
                                     details={**details, "applicability": "one bar per row; no adjacent pair"}))
        if cover_margins:
            checks.append(_check("beam.bar_row_cover", min(cover_margins), 0.0, face,
                                 {"basis": "bar center inside clear cover plus hoop and half-bar bounds"}))
        for axis in rows:
            offsets = [o for _, o in rows[axis][face]]
            if len(offsets) > 1:
                gap = min(b - a - db for a, b in zip(offsets, offsets[1:]))
                checks.append(_check("beam.bar_interlayer_spacing", gap, 1.0, f"{axis}/{face}", {"offsets_in": offsets}))
    for axis in rows:
        gap = h - rows[axis]["top"][-1][1] - rows[axis]["bottom"][-1][1] - db
        checks.append(_check("beam.opposing_rows_clear_spacing", gap, 1.0, axis, {}))
        if directions is not None:
            positions = directions[axis]["threaded_positions_in"]
            obstacles = directions[axis]["column_bar_lanes_blocked_in"]
            if positions and obstacles:
                gap = min(abs(p - o) - (db + cdb) / 2 for p in positions for o in obstacles)
                checks.append(_check("beam.column_bar_clear_spacing", gap, clear, axis,
                                     {"positions_in": positions, "column_bar_positions_in": obstacles}))
    return checks
