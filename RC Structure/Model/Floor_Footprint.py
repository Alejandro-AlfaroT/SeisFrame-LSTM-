"""One physical slab boundary: the actual outside faces of perimeter beams.

Grid coordinates remain column/beam centerlines. A floor is rectangular only
when each of its four perimeter sides has one constant beam width. Unequal
opposite widths are supported; stepped sides are explicitly outside this
adapter. Geometry and integrations are pure, with no structural solve.
"""
from __future__ import annotations

import math
from functools import lru_cache

METHOD = "actual_perimeter_beam_outer_faces_v1"
NODAL_METHOD = "integrated_piecewise_linear_grid_shapes_v1"


def geometry():
    import Structure_Parameters as sp
    return {"nx": sp.NUM_BAY_X, "ny": sp.NUM_BAY_Y,
            "lx": float(sp.BAY_X), "ly": float(sp.BAY_Y)}


def _geometry(g):
    nx, ny = g["nx"], g["ny"]
    lx, ly = float(g["lx"]), float(g["ly"])
    if any(type(v) is not int or v < 1 for v in (nx, ny)):
        raise ValueError("Physical floor requires positive integer bay counts")
    if any(not math.isfinite(v) or v <= 0 for v in (lx, ly)):
        raise ValueError("Physical floor requires finite positive bay lengths")
    return nx, ny, lx, ly


def rectangle(x_min, x_max, y_min, y_max):
    values = tuple(float(v) for v in (x_min, x_max, y_min, y_max))
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Finite physical rectangle coordinates required")
    x0, x1, y0, y1 = values
    if x1 < x0 or y1 < y0:
        raise ValueError("Physical rectangle bounds are reversed")
    area = (x1 - x0) * (y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return {"x_min_in": x0, "x_max_in": x1, "y_min_in": y0, "y_max_in": y1,
            "area_in2": area, "centroid_in": [cx, cy],
            "first_moment_x_in3": area * cx, "first_moment_y_in3": area * cy}


def from_perimeter_beams(g, beams):
    """Freeze bounds from ``axis,line,start,end,b_in`` beam descriptions.

    ``lx,ly`` are individual bay lengths, ``nx,ny`` their counts. The input
    may contain interior beams; complete, unique, exact perimeter spans are
    required. A changing width on a side is rejected rather than enclosed
    in a larger fictitious slab rectangle.
    """
    nx, ny, lx, ly = _geometry(g)
    sides = {key: [] for key in ("x_minus", "x_plus", "y_minus", "y_plus")}
    for b in beams:
        axis = b["axis"]
        if axis not in ("x", "y"):
            raise ValueError("Physical perimeter beam axis must be x or y")
        line = float(b["line"])
        if axis == "y" and line == 0.: side = "x_minus"
        elif axis == "y" and line == nx * lx: side = "x_plus"
        elif axis == "x" and line == 0.: side = "y_minus"
        elif axis == "x" and line == ny * ly: side = "y_plus"
        else: continue
        width = float(b["b_in"])
        if not math.isfinite(width) or width <= 0:
            raise ValueError("Physical perimeter beam width must be finite and positive")
        sides[side].append((float(b["start"]), float(b["end"]), width))
    widths = {}
    for side, spans in sides.items():
        count, length = (ny, ly) if side.startswith("x_") else (nx, lx)
        expected = [(i * length, (i + 1) * length) for i in range(count)]
        if sorted((a, b) for a, b, _ in spans) != expected:
            raise ValueError(f"Perimeter side {side} must cover each actual bay exactly once")
        values = {w for _, _, w in spans}
        if len(values) != 1:
            raise ValueError(f"Perimeter side {side} has varying widths; a stepped physical slab is unsupported")
        widths[side] = values.pop()
    result = rectangle(-widths["x_minus"] / 2, nx * lx + widths["x_plus"] / 2,
                       -widths["y_minus"] / 2, ny * ly + widths["y_plus"] / 2)
    result.update(method=METHOD, perimeter_beam_widths_in=widths)
    return result


def uniform_floor():
    """Uniform adapter; grouped section sentinels deliberately cannot pass."""
    import Structure_Parameters as sp
    g = geometry(); nx, ny, lx, ly = _geometry(g)
    beams = [{"axis": "x", "line": j * ly, "start": i * lx, "end": (i + 1) * lx, "b_in": sp.B_BEAM}
             for j in (0, ny) for i in range(nx)]
    beams += [{"axis": "y", "line": i * lx, "start": j * ly, "end": (j + 1) * ly, "b_in": sp.B_BEAM}
              for i in (0, nx) for j in range(ny)]
    return from_perimeter_beams(g, beams)


def physical_floor(floor, state=None):
    """Actual per-floor footprint from a grouped state or the uniform adapter."""
    import Structure_Parameters as sp
    from Model import Member_Groups as mg
    if type(floor) is not int or not 1 <= floor <= sp.NUM_FLOOR:
        raise ValueError("Physical floor index must name an actual elevated floor")
    state = mg.active() if state is None else state
    if state is None:
        result = uniform_floor()
    else:
        beams = []
        for member in state.members():
            if member.is_column or member.story_or_floor != floor:
                continue
            axis = "x" if member.member_type == "beam_x" else "y"
            start = member.grid_i * sp.BAY_X if axis == "x" else member.grid_j * sp.BAY_Y
            beams.append({"axis": axis,
                          "line": member.grid_j * sp.BAY_Y if axis == "x" else member.grid_i * sp.BAY_X,
                          "start": start, "end": start + (sp.BAY_X if axis == "x" else sp.BAY_Y),
                          "b_in": member.design.b_in})
        result = from_perimeter_beams(geometry(), beams)
    return dict(result, floor=floor)


def column_overlap(footprint, x, y, h_x, b_y):
    """Exact column/slab intersection rectangle, including its first moments."""
    if min(h_x, b_y) <= 0 or not all(math.isfinite(v) for v in (x, y, h_x, b_y)):
        raise ValueError("Finite positive column plan dimensions required")
    x0, x1 = max(x - h_x / 2, footprint["x_min_in"]), min(x + h_x / 2, footprint["x_max_in"])
    y0, y1 = max(y - b_y / 2, footprint["y_min_in"]), min(y + b_y / 2, footprint["y_max_in"])
    return rectangle(x0, max(x0, x1), y0, max(y0, y1))


def column_overlap_area_in2(footprint, x, y, h_x, b_y):
    return column_overlap(footprint, x, y, h_x, b_y)["area_in2"]


def tributary_cell(footprint, g, i, j):
    """Physical midpoint-bounded cell, for geometric reporting, not nodal mass."""
    nx, ny, lx, ly = _geometry(g)
    if not (type(i) is int and type(j) is int and 0 <= i <= nx and 0 <= j <= ny):
        raise ValueError("Tributary node must lie on the actual floor grid")
    return rectangle(footprint["x_min_in"] if i == 0 else (i - .5) * lx,
                     footprint["x_max_in"] if i == nx else (i + .5) * lx,
                     footprint["y_min_in"] if j == 0 else (j - .5) * ly,
                     footprint["y_max_in"] if j == ny else (j + .5) * ly)


def _line_integrals(start, end, count, length):
    """Exact integrals of linear grid hats, linearly extrapolated at edges.

    The two active functions sum to one and reproduce x, including outside
    the centerline plan. Thus the 2-D tensor product preserves volume and
    both physical first moments, unlike lumping an extended half-bay cell
    at its centerline node. Individual overhang contributions can be
    negative; the assembled physical floor masses must remain positive.
    """
    if end < start:
        raise ValueError("Integration interval bounds reversed")
    result = [0.] * (count + 1)
    cuts = [start] + [i * length for i in range(count + 1) if start < i * length < end] + [end]
    for a, b in zip(cuts, cuts[1:]):
        index = min(count - 1, max(0, int(math.floor((a + b) / (2 * length)))))
        right = (b - a) * ((a + b) / 2 - index * length) / length
        result[index] += b - a - right
        result[index + 1] += right
    return result


def rectangle_nodal_areas(rect, g):
    """Unrestricted signed exact shape-function integral of any rectangle."""
    nx, ny, lx, ly = _geometry(g)
    wx = _line_integrals(rect["x_min_in"], rect["x_max_in"], nx, lx)
    wy = _line_integrals(rect["y_min_in"], rect["y_max_in"], ny, ly)
    return {(i, j): wx[i] * wy[j] for j in range(ny + 1) for i in range(nx + 1)}


def consistent_nodal_areas(footprint, g):
    result = rectangle_nodal_areas(footprint, g)
    if min(result.values()) <= 0:
        raise ValueError("Physical overhang is outside positive lumped grid-mass applicability")
    return result


def sum_rectangular_volumes(g, pieces):
    """Integrate signed ``(rectangle,height)`` disjoint-volume expressions.

    A full column prism minus its slab intersection is supplied as two
    signed pieces. Beam drops use their actual clear length and full width.
    The caller must check positivity after adding floor-area mass.
    """
    nx, ny, _, _ = _geometry(g)
    nodal = {(i, j): 0. for j in range(ny + 1) for i in range(nx + 1)}
    volume = mx = my = 0.
    for rect, height in pieces:
        if not math.isfinite(height):
            raise ValueError("Finite signed prism height required")
        volume += rect["area_in2"] * height
        mx += rect["first_moment_x_in3"] * height
        my += rect["first_moment_y_in3"] * height
        for key, area in rectangle_nodal_areas(rect, g).items():
            nodal[key] += area * height
    return {"volume_in3": volume, "first_moment_x_in4": mx,
            "first_moment_y_in4": my, "nodal_volumes_in3": nodal,
            "nodal_method": NODAL_METHOD}


def member_volume_ledger(g, footprint, columns, beams, slab_t, story_h):
    """Actual non-slab concrete, with exact consistent nodal volume weights.

    Columns: ``x,y,h_x,b_y`` and optional ``roof_extension_in``. Beams:
    ``axis,line,start,end,b_in,h_in`` where start/end are the joint faces.
    Returns a fresh value; numeric integrations are cached by all geometry.
    """
    gkey = _geometry(g)
    bounds = tuple(footprint[k] for k in ("x_min_in", "x_max_in", "y_min_in", "y_max_in"))
    ckey = tuple((c["x"], c["y"], c["h_x"], c["b_y"], c.get("roof_extension_in", 0.)) for c in columns)
    bkey = tuple((b["axis"], b["line"], b["start"], b["end"], b["b_in"], b["h_in"]) for b in beams)
    result = _member_volume_cached(gkey, bounds, ckey, bkey, float(slab_t), float(story_h))
    return dict(result, nodal_volumes_in3=dict(result["nodal_volumes_in3"]))


@lru_cache(maxsize=64)
def _member_volume_cached(gkey, bounds, columns, beams, slab_t, story_h):
    nx, ny, lx, ly = gkey
    g = dict(nx=nx, ny=ny, lx=lx, ly=ly)
    footprint = rectangle(*bounds)
    if not 0 <= slab_t < story_h:
        raise ValueError("Physical slab thickness must be nonnegative and less than story height")
    pieces = []
    for x, y, hx, by, extension in columns:
        if extension < 0:
            raise ValueError("Physical column extension must be nonnegative")
        pieces.append((rectangle(x-hx/2, x+hx/2, y-by/2, y+by/2), story_h+extension))
        pieces.append((column_overlap(footprint, x, y, hx, by), -slab_t))
    for axis, line, start, end, width, depth in beams:
        if axis not in ("x", "y") or min(end-start, width, depth-slab_t) <= 0:
            raise ValueError("Physical beam drop requires a positive clear span, width and depth")
        r = rectangle(start, end, line-width/2, line+width/2) if axis == "x" else rectangle(line-width/2, line+width/2, start, end)
        pieces.append((r, depth-slab_t))
    return sum_rectangular_volumes(g, pieces)
