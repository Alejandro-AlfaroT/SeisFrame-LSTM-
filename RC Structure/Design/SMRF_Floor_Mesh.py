"""Bounded rectangular floor grids with explicit coordinate provenance.

Explicit offsets repeat within each bay. They are a numerical mesh request,
not a claim of convergence or engineering applicability. Uniform defaults
retain their existing limit; the larger explicit-grid budget is opt-in.
"""
from __future__ import annotations

import hashlib
import json
import math

MAX_EXPLICIT_SHELLS = 45000


def _offsets(values, length, name):
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        raise ValueError(f"{name} must contain at least three bay coordinates")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError(f"{name} must contain finite numeric coordinates")
    values = [float(v) for v in values]
    if values[0] != 0.0 or values[-1] != length:
        raise ValueError(f"{name} must start at zero and end at its bay length")
    if any(b - a <= 1e-8 for a, b in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing, with cells wider than 1e-8 in")
    return values


def floor_mesh(nx, ny, lx, ly, mesh_per_bay, *, mesh_spec=None, uniform_shell_limit=8192):
    if mesh_spec is None:
        count = mesh_per_bay
        if (isinstance(count, bool) or not isinstance(count, (int, float))
                or not math.isfinite(count) or int(count) != count or not 2 <= count <= 24):
            raise ValueError("Uniform floor mesh requires 2-24 integer subdivisions per bay")
        count = int(count)
        ox = [i * lx / count for i in range(count + 1)]
        oy = [i * ly / count for i in range(count + 1)]
        budget, kind, solver = uniform_shell_limit, "uniform", "BandGeneral"
    else:
        if not isinstance(mesh_spec, dict) or set(mesh_spec) != {"x_offsets_in", "y_offsets_in", "max_shells"}:
            raise ValueError("Explicit mesh requires x_offsets_in, y_offsets_in and max_shells only")
        budget = mesh_spec["max_shells"]
        if isinstance(budget, bool) or not isinstance(budget, int) or not 4 <= budget <= MAX_EXPLICIT_SHELLS:
            raise ValueError(f"Explicit mesh max_shells must be an integer from 4 to {MAX_EXPLICIT_SHELLS}")
        ox = _offsets(mesh_spec["x_offsets_in"], lx, "x_offsets_in")
        oy = _offsets(mesh_spec["y_offsets_in"], ly, "y_offsets_in")
        kind, solver = "explicit_rectangular", "UmfPack"
    mx, my = len(ox) - 1, len(oy) - 1
    count = nx * ny * mx * my
    if count > budget:
        raise ValueError(f"Requested floor mesh has {count} shells, exceeding explicit budget {budget}")
    xs = [i * lx + offset for i in range(nx) for offset in ox[:-1]] + [nx * lx]
    ys = [j * ly + offset for j in range(ny) for offset in oy[:-1]] + [ny * ly]
    dx = [b - a for a, b in zip(xs, xs[1:])]
    dy = [b - a for a, b in zip(ys, ys[1:])]
    coordinates = {"x_coordinates_in": xs, "y_coordinates_in": ys}
    digest = hashlib.sha256(json.dumps(coordinates, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    return {"kind": kind, **coordinates, "x_offsets_in": ox, "y_offsets_in": oy,
            "coordinate_sha256": digest, "shell_count": count, "node_count": len(xs) * len(ys),
            "subdivisions_per_bay": mx if mx == my else None,
            "subdivisions_x_per_bay": mx, "subdivisions_y_per_bay": my,
            "dx_in": lx / mx if mesh_spec is None else None,
            "dy_in": ly / my if mesh_spec is None else None,
            "minimum_cell_width_in": min(*dx, *dy),
            "maximum_cell_aspect_ratio": max(max(dx) / min(dy), max(dy) / min(dx)),
            "shell_budget": budget, "solver": solver}


def coupled_floor_mesh(nx, ny, lx, ly, mesh_per_bay, *, beam_width_in,
                       slab_perimeter='centerlines', mesh_spec=None, uniform_shell_limit=8192):
    """Add slab-only strips to the beam faces, preserving every interior node.

    Integer indices of the original grid remain unchanged. A face extension
    adds nodes at -1 and n+1 and cells at -1 and n, without extending beams or
    changing supports. Every added cell inherits its nearest panel's pressure,
    including that panel's dead/live factors and pattern. Corner cells occur
    once in the rectangular grid. Counts include the extension before solving.
    """
    if slab_perimeter not in ('centerlines', 'beam_outer_faces'):
        raise ValueError('slab_perimeter must be centerlines or beam_outer_faces')
    if (isinstance(beam_width_in, bool) or not isinstance(beam_width_in, (int, float))
            or not math.isfinite(beam_width_in) or not 0 < beam_width_in < min(lx, ly)):
        raise ValueError('Beam width must be finite, positive and less than either bay')
    grid = floor_mesh(nx, ny, lx, ly, mesh_per_bay, mesh_spec=mesh_spec,
                      uniform_shell_limit=uniform_shell_limit)
    extension = beam_width_in/2 if slab_perimeter == 'beam_outer_faces' else 0.
    xs, ys = grid['x_coordinates_in'], grid['y_coordinates_in']
    if extension:
        xs, ys = [-extension]+xs+[nx*lx+extension], [-extension]+ys+[ny*ly+extension]
    count = (len(xs)-1)*(len(ys)-1)
    if count > grid['shell_budget']:
        raise ValueError(f'Perimeter-inclusive floor has {count} shells, exceeding budget {grid["shell_budget"]}')
    coordinates = dict(x_coordinates_in=xs, y_coordinates_in=ys)
    dx, dy = [b-a for a,b in zip(xs,xs[1:])], [b-a for a,b in zip(ys,ys[1:])]
    area = (xs[-1]-xs[0])*(ys[-1]-ys[0])
    return dict(grid, **coordinates, slab_perimeter=slab_perimeter,
                coordinate_sha256=hashlib.sha256(json.dumps(coordinates, sort_keys=True, separators=(',', ':'),
                                                           allow_nan=False).encode()).hexdigest(),
                shell_count=count, node_count=len(xs)*len(ys), index_start=-1 if extension else 0,
                perimeter_extension_in=extension, represented_area_in2=area,
                centerline_area_in2=nx*lx*ny*ly, added_area_in2=area-nx*lx*ny*ly,
                minimum_cell_width_in=min(*dx,*dy),
                maximum_cell_aspect_ratio=max(max(dx)/min(dy),max(dy)/min(dx)),
                perimeter_load_basis='Nearest adjacent panel pressure, including factored dead/live pattern; corner area once')


# Graded face recipe, version 1. Half-bay node pattern from the column line
# to midspan, mirrored: two nodes between the column line and the beam face
# at 2/7 and 4/7 of the face offset, the face itself, then nine nodes over
# the clear half-span at these fractions of it (113 = 120 - 7, the clear
# half-span of the 240-in / 14-in benchmark this reproduces exactly). Level
# 0 keeps every other interior node (12 cells per bay), level 1 is the full
# pattern (24), level 2 adds every midpoint (48), level 3 adds the midpoints
# of the cells between the second near-face node and the second node beyond
# the face (60 on the benchmark). Every level retains the previous nodes.
GRADED_FACE_V1 = {
    "near_face_fractions_of_face": (2.0 / 7.0, 4.0 / 7.0),
    "beyond_face_units_of_113": (3, 8, 15, 25, 38, 53, 73, 93, 113),
    "coarse_half_indices": (0, 3, 5, 7, 9, 11, 12),
    "fine_band_half_indices": (2, 5),
    "basis": "2026-09-24 fixed-candidate benchmark graded meshes (24/48/60 per 240-in bay, 14-in beam), "
             "resolved for the current bay length and beam face; not a validated recipe for every geometry",
}
RECIPES = {"graded_face_v1": GRADED_FACE_V1}
RECIPE_LEVELS = 4


def graded_face_levels(bay_length_in, beam_width_in, levels=RECIPE_LEVELS, recipe=GRADED_FACE_V1):
    """Nested bay offset lists of the graded face recipe, coarsest first."""
    if isinstance(levels, bool) or not isinstance(levels, int) or not 2 <= levels <= RECIPE_LEVELS:
        raise ValueError(f"A recipe plan needs 2 to {RECIPE_LEVELS} levels")
    for name, value in (("bay length", bay_length_in), ("beam width", beam_width_in)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Recipe {name} must be a finite positive number")
    face = beam_width_in / 2.0
    half_span = bay_length_in / 2.0
    if face >= half_span / 2.0:
        raise ValueError("Beam face must lie in the first quarter of the bay for the graded face recipe")
    clear = half_span - face
    half = [0.0] + [face * f for f in recipe["near_face_fractions_of_face"]] + [face] + [
        face + u * (clear / recipe["beyond_face_units_of_113"][-1]) for u in recipe["beyond_face_units_of_113"]]
    half[-1] = half_span

    def mirror(values):
        return sorted(set(values) | {bay_length_in - v for v in values})

    def midpoints(values, lo=None, hi=None):
        extra = [(a + b) / 2.0 for a, b in zip(values, values[1:])
                 if lo is None or lo <= (a + b) / 2.0 <= hi or bay_length_in - hi <= (a + b) / 2.0 <= bay_length_in - lo]
        return sorted(set(values) | set(extra))

    base = mirror(half)
    coarse = mirror([half[i] for i in recipe["coarse_half_indices"]])
    fine = midpoints(base)
    lo, hi = (half[i] for i in recipe["fine_band_half_indices"])
    finest = midpoints(fine, lo, hi)
    return [coarse, base, fine, finest][:levels]


def resolve_recipe_plan(geometry, sections, policy):
    """Resolve a named recipe for this geometry and beam width against the shell budget.

    Levels are nested and increasing, so the affordable ones form a prefix.
    Fewer than two affordable levels is an explicit ``unresolved_budget``
    result: nothing is coarsened and nothing passes.
    """
    name = policy["recipe"]
    if name not in RECIPES:
        raise ValueError(f"Unknown slab refinement recipe {name!r}")
    budget = policy["max_shells"]
    if isinstance(budget, bool) or not isinstance(budget, int) or not 4 <= budget <= MAX_EXPLICIT_SHELLS:
        raise ValueError(f"Recipe max_shells must be an integer from 4 to {MAX_EXPLICIT_SHELLS}")
    nx, ny = int(geometry["num_bay_x"]), int(geometry["num_bay_y"])
    lx, ly = float(geometry["bay_x_in"]), float(geometry["bay_y_in"])
    width = float(sections["b_beam_in"])
    if nx < 1 or ny < 1:
        raise ValueError("Recipe geometry needs at least one bay in each direction")
    xs = graded_face_levels(lx, width, policy["levels"], RECIPES[name])
    ys = graded_face_levels(ly, width, policy["levels"], RECIPES[name])
    resolved, dropped = [], []
    for level, (ox, oy) in enumerate(zip(xs, ys)):
        count = nx * ny * (len(ox) - 1) * (len(oy) - 1)
        entry = {"level": level, "shell_count": count, "cells_per_bay": [len(ox) - 1, len(oy) - 1]}
        if count <= budget and not dropped:
            resolved.append(dict(entry, mesh={"x_offsets_in": ox, "y_offsets_in": oy, "max_shells": budget}))
        else:
            dropped.append(dict(entry, reason=f"{count} shells exceed the declared budget {budget}"))
    return {"recipe": name, "recipe_definition": RECIPES[name],
            "inputs": {"num_bay_x": nx, "num_bay_y": ny, "bay_x_in": lx, "bay_y_in": ly, "beam_width_in": width,
                       "levels": policy["levels"], "max_shells": budget},
            "meshes": [r["mesh"] for r in resolved], "resolved_levels": resolved, "dropped_levels": dropped,
            "status": "resolved" if len(resolved) >= 2 else "unresolved_budget",
            "detail": (f"{len(resolved)} of {len(xs)} levels fit the {budget}-shell budget"
                       + ("" if len(resolved) >= 2 else "; a refinement comparison needs two"))}


def nested_refinement(coarse, fine):
    """Require retained nodes and a genuinely finer grid in at least one axis."""
    increased = False
    for key in ("x_coordinates_in", "y_coordinates_in"):
        a, b = coarse[key], fine[key]
        if not set(a).issubset(b):
            raise ValueError("Refinement meshes must retain every previous coordinate")
        increased |= len(b) > len(a)
    if not increased:
        raise ValueError("Refinement must add coordinates; repeated grids are not refinement")
