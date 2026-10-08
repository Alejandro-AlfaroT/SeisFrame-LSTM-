"""Bounded rectangular floor grids with explicit coordinate provenance.

Explicit offsets repeat within each bay, or are given bay by bay (one list per
bay, all of one length) when the beams bounding the bays have different
widths, so every bay keeps the same number of cells and the cell-index
arithmetic of the floor models holds. They are a numerical mesh request,
not a claim of convergence or engineering applicability. Uniform defaults
retain their existing limit; the larger explicit-grid budget is opt-in.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

MAX_EXPLICIT_SHELLS = 130000


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


def _bay_offsets(values, length, bays, name):
    """(offsets of the first bay, per-bay offsets or None) from one list or one list per bay."""
    if isinstance(values, (list, tuple)) and values and isinstance(values[0], (list, tuple)):
        if len(values) != bays:
            raise ValueError(f"{name} given bay by bay must have one list for each of the {bays} bays")
        rows = [_offsets(row, length, name) for row in values]
        if any(len(row) != len(rows[0]) for row in rows):
            raise ValueError(f"{name} given bay by bay must use the same number of cells in every bay")
        return rows[0], rows
    return _offsets(values, length, name), None


def span_offsets(grid, axis, span_index):
    """Bay offsets of one span of a grid from ``floor_mesh`` (the same list for every span unless given bay by bay)."""
    by_bay = grid.get(f"{axis}_offsets_by_bay_in")
    return by_bay[span_index] if by_bay is not None else grid[f"{axis}_offsets_in"]


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
        ox, ox_by_bay = _bay_offsets(mesh_spec["x_offsets_in"], lx, nx, "x_offsets_in")
        oy, oy_by_bay = _bay_offsets(mesh_spec["y_offsets_in"], ly, ny, "y_offsets_in")
        # SuperLU avoids the observed fine-grid UMFPACK numeric allocation failure.
        # Mesh geometry, shell budget and refinement acceptance are unchanged.
        kind, solver = "explicit_rectangular", "SuperLU"
    if mesh_spec is None:
        ox_by_bay = oy_by_bay = None
    mx, my = len(ox) - 1, len(oy) - 1
    count = nx * ny * mx * my
    if count > budget:
        raise ValueError(f"Requested floor mesh has {count} shells, exceeding explicit budget {budget}")
    xs = [i * lx + offset for i in range(nx) for offset in (ox if ox_by_bay is None else ox_by_bay[i])[:-1]] + [nx * lx]
    ys = [j * ly + offset for j in range(ny) for offset in (oy if oy_by_bay is None else oy_by_bay[j])[:-1]] + [ny * ly]
    dx = [b - a for a, b in zip(xs, xs[1:])]
    dy = [b - a for a, b in zip(ys, ys[1:])]
    coordinates = {"x_coordinates_in": xs, "y_coordinates_in": ys}
    digest = hashlib.sha256(json.dumps(coordinates, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    by_bay = {}
    if ox_by_bay is not None or oy_by_bay is not None:
        # Offsets given bay by bay: the single lists are withheld so nothing reads one bay's offsets for another.
        by_bay = {"x_offsets_by_bay_in": ox_by_bay if ox_by_bay is not None else [ox] * nx,
                  "y_offsets_by_bay_in": oy_by_bay if oy_by_bay is not None else [oy] * ny}
        ox = oy = None
        kind = "explicit_rectangular_by_bay"
    return {"kind": kind, **coordinates, "x_offsets_in": ox, "y_offsets_in": oy, **by_bay,
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


def graded_face_levels_between(bay_length_in, left_width_in, right_width_in, levels=RECIPE_LEVELS, recipe=GRADED_FACE_V1):
    """The graded face recipe for a bay whose two bounding beams have different widths.

    The left half of the bay is graded toward the left beam's face and the right half toward the right
    beam's, meeting at midspan; the number of nodes is that of the symmetric recipe, so every bay of a
    floor keeps the same cell count. Equal widths give exactly ``graded_face_levels``.
    """
    if left_width_in == right_width_in:
        return graded_face_levels(bay_length_in, left_width_in, levels, recipe)
    if isinstance(levels, bool) or not isinstance(levels, int) or not 2 <= levels <= RECIPE_LEVELS:
        raise ValueError(f"A recipe plan needs 2 to {RECIPE_LEVELS} levels")
    for name, value in (("bay length", bay_length_in), ("beam width", left_width_in), ("beam width", right_width_in)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Recipe {name} must be a finite positive number")
    half_span = bay_length_in / 2.0

    def half_pattern(width):
        face = width / 2.0
        if face >= half_span / 2.0:
            raise ValueError("Beam face must lie in the first quarter of the bay for the graded face recipe")
        clear = half_span - face
        half = [0.0] + [face * f for f in recipe["near_face_fractions_of_face"]] + [face] + [
            face + u * (clear / recipe["beyond_face_units_of_113"][-1]) for u in recipe["beyond_face_units_of_113"]]
        half[-1] = half_span
        return half

    left, right = half_pattern(left_width_in), half_pattern(right_width_in)

    def join(left_values, right_values):
        return sorted(set(left_values) | {bay_length_in - v for v in right_values})

    def midpoints(values, bands=None):
        extra = [(a + b) / 2.0 for a, b in zip(values, values[1:])
                 if bands is None or any(lo <= (a + b) / 2.0 <= hi for lo, hi in bands)]
        return sorted(set(values) | set(extra))

    base = join(left, right)
    coarse = join([left[i] for i in recipe["coarse_half_indices"]], [right[i] for i in recipe["coarse_half_indices"]])
    fine = midpoints(base)
    lo_index, hi_index = recipe["fine_band_half_indices"]
    finest = midpoints(fine, [(left[lo_index], left[hi_index]),
                              (bay_length_in - right[hi_index], bay_length_in - right[lo_index])])
    return [coarse, base, fine, finest][:levels]


def _by_bay_levels(length, widths, levels, recipe):
    """Per level, the offsets of every bay along one axis; ``widths`` are the beams at the bay boundaries."""
    bays = [graded_face_levels_between(length, widths[k], widths[k + 1], levels, recipe) for k in range(len(widths) - 1)]
    return [[bay[level] for bay in bays] for level in range(len(bays[0]))]


def resolve_recipe_plan(geometry, sections, policy):
    """Resolve a named recipe for this geometry and beam width against the shell budget.

    Levels are nested and increasing, so the affordable ones form a prefix.
    Fewer than two affordable levels is an explicit ``unresolved_budget``
    result: nothing is coarsened and nothing passes.

    ``sections`` is the one-beam dictionary (``b_beam_in``), a floor described line by line
    (Design.SMRF_Floor_Sections), or the ``face_widths_in`` recorded by an earlier resolution: with
    unlike beam lines every bay is graded toward the faces of its own two bounding beams.
    """
    if "face_widths_in" in sections or sections.get("schema") is not None:
        return _resolve_recipe_plan_by_line(geometry, sections, policy)
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


def _resolve_recipe_plan_by_line(geometry, sections, policy):
    name = policy["recipe"]
    if name not in RECIPES:
        raise ValueError(f"Unknown slab refinement recipe {name!r}")
    budget = policy["max_shells"]
    if isinstance(budget, bool) or not isinstance(budget, int) or not 4 <= budget <= MAX_EXPLICIT_SHELLS:
        raise ValueError(f"Recipe max_shells must be an integer from 4 to {MAX_EXPLICIT_SHELLS}")
    nx, ny = int(geometry["num_bay_x"]), int(geometry["num_bay_y"])
    lx, ly = float(geometry["bay_x_in"]), float(geometry["bay_y_in"])
    if nx < 1 or ny < 1:
        raise ValueError("Recipe geometry needs at least one bay in each direction")
    if "face_widths_in" in sections:
        widths = sections["face_widths_in"]
    else:
        from Design.SMRF_Floor_Sections import face_widths
        widths = face_widths(sections, nx, ny)
    widths = {axis: [float(w) for w in widths[axis]] for axis in ("x", "y")}
    if len(widths["x"]) != nx + 1 or len(widths["y"]) != ny + 1:
        raise ValueError("Recipe face widths need one beam width for every beam line")
    xs = _by_bay_levels(lx, widths["x"], policy["levels"], RECIPES[name])
    ys = _by_bay_levels(ly, widths["y"], policy["levels"], RECIPES[name])
    resolved, dropped = [], []
    for level, (ox, oy) in enumerate(zip(xs, ys)):
        count = nx * ny * (len(ox[0]) - 1) * (len(oy[0]) - 1)
        entry = {"level": level, "shell_count": count, "cells_per_bay": [len(ox[0]) - 1, len(oy[0]) - 1]}
        if count <= budget and not dropped:
            resolved.append(dict(entry, mesh={"x_offsets_in": ox, "y_offsets_in": oy, "max_shells": budget}))
        else:
            dropped.append(dict(entry, reason=f"{count} shells exceed the declared budget {budget}"))
    return {"recipe": name, "recipe_definition": RECIPES[name],
            "inputs": {"num_bay_x": nx, "num_bay_y": ny, "bay_x_in": lx, "bay_y_in": ly, "face_widths_in": widths,
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


LOCAL_EXTENSION_METHOD = "failed_witness_face_band_v1"


def witness_face_band_extension(geometry, recipe_inputs, coarse_actions, fine_actions,
                                comparison, previous_grid, used_axes=()):
    """Propose one nested face-band bisection from failed, located strip witnesses.

    All faces along the selected axis are refined, using each bay's own two
    bounding beam widths. Each axis is eligible only once. Ranking uses the
    smallest normalized distance to a physical face, rounded to 12 decimals
    for reflection-stable ties; ties choose X before Y. This is a bounded
    numerical proposal, never an acceptance result or a geometry change.
    """
    result = {"method": LOCAL_EXTENSION_METHOD, "used_axes": list(used_axes)}

    def refused(status, detail):
        return {**result, "status": status, "detail": detail}

    try:
        if any(axis not in ("x", "y") for axis in used_axes) or len(set(used_axes)) != len(used_axes):
            raise ValueError("Previously extended axes must be distinct X/Y axes")
        nx, ny = (int(geometry[k]) for k in ("num_bay_x", "num_bay_y"))
        lengths = {a: float(geometry[f"bay_{a}_in"]) for a in ("x", "y")}
        bays = {"x": nx, "y": ny}
        widths = recipe_inputs.get("face_widths_in")
        if widths is None:
            widths = {a: [recipe_inputs["beam_width_in"]] * (bays[a] + 1) for a in ("x", "y")}
        widths = {a: [float(w) for w in widths[a]] for a in ("x", "y")}
        bands = {}
        for axis in ("x", "y"):
            length = lengths[axis]
            if (len(widths[axis]) != bays[axis] + 1 or not math.isfinite(length) or length <= 0
                    or any(not math.isfinite(w) or not 0 < w < length / 2 for w in widths[axis])):
                raise ValueError("Local face refinement requires finite, supported bay and beam-face geometry")
            bands[axis] = []
            for left, right in zip(widths[axis], widths[axis][1:]):
                def near(width):
                    face = width / 2.0
                    return (face * (4.0 / 7.0), face + 8.0 * ((length / 2.0 - face) / 113.0))
                left_band, right_band = near(left), near(right)
                bands[axis].append((left_band, (length - right_band[1], length - right_band[0])))
        failed = [row for row in comparison["comparisons"] if row["within_tolerance"] is not True]
        if not failed or comparison["all_within_tolerance"] is True:
            raise ValueError("An extension needs a completed failed comparison")
        actions = {"coarse": coarse_actions, "fine": fine_actions}
        indexed = {}
        for side, action in actions.items():
            strips = action["strips"]
            indexed[side] = {(s["panel_id"], s["axis"], s["face"]): s for s in strips}
            if len(indexed[side]) != len(strips):
                raise ValueError("Duplicate strip identities cannot select a refinement witness")
        candidates = []
        for row in failed:
            match = re.fullmatch(r"panel_x(\d+)_y(\d+)", row["panel_id"])
            if match is None:
                raise ValueError("Failed strip has no supported panel-grid identity")
            panel = dict(zip(("x", "y"), (int(v) - 1 for v in match.groups())))
            if any(not 0 <= panel[a] < bays[a] for a in ("x", "y")):
                raise ValueError("Failed strip lies outside the declared floor")
            if row["metric"] not in ("mu_kip_in_per_ft", "vu_kip_per_ft"):
                raise ValueError("Unsupported failed strip metric")
            location = "moment_location" if row["metric"] == "mu_kip_in_per_ft" else "shear_location"
            for side in ("coarse", "fine"):
                strip = indexed[side][(row["panel_id"], row["axis"], row["face"])]
                witness = strip[location]
                local = {}
                for axis in ("x", "y"):
                    value = witness[f"{axis}_in"]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError("Failed strip witness coordinates must be finite")
                    local[axis] = value - panel[axis] * lengths[axis]
                    if not -1e-8 <= local[axis] <= lengths[axis] + 1e-8:
                        raise ValueError("Failed strip witness is outside its declared panel")
                for axis in ("x", "y"):
                    if axis in used_axes:
                        continue
                    pos, length, k = local[axis], lengths[axis], panel[axis]
                    near_bands = bands[axis][k]
                    if not any(lo - 1e-8 <= pos <= hi + 1e-8 for lo, hi in near_bands):
                        continue
                    faces = (widths[axis][k] / 2.0, length - widths[axis][k + 1] / 2.0)
                    score = min(abs(pos - face) / (length / 2.0 - width / 2.0)
                                for face, width in zip(faces, widths[axis][k:k + 2]))
                    candidates.append({"axis": axis, "score": score, "rank_score": round(score, 12),
                                       "panel_id": row["panel_id"], "strip_axis": row["axis"],
                                       "face": row["face"], "metric": row["metric"], "side": side,
                                       "witness_x_in": witness["x_in"], "witness_y_in": witness["y_in"]})
        if not candidates:
            return refused("no_supported_witness", "No failed witness lies in an unused axis's supported face band")
        selected = min(candidates, key=lambda c: (c["rank_score"], c["axis"], c["panel_id"],
                                                   c["strip_axis"], c["face"], c["metric"], c["side"]))
        axis = selected["axis"]
        result.update(selected_axis=axis, selected_witness=selected, candidate_count=len(candidates),
                      ranking="nearest_normalized_face_distance_rounded_12_decimals_then_x_before_y")
        spec = {"max_shells": previous_grid["shell_budget"]}
        for a in ("x", "y"):
            by_bay = previous_grid.get(f"{a}_offsets_by_bay_in")
            values = by_bay if by_bay is not None else [previous_grid[f"{a}_offsets_in"]] * bays[a]
            refined = []
            for k, offsets in enumerate(values):
                extra = ([(lo + hi) / 2.0 for lo, hi in zip(offsets, offsets[1:])
                          if any(start <= (lo + hi) / 2.0 <= end for start, end in bands[a][k])]
                         if a == axis else [])
                refined.append(sorted(set(offsets) | set(extra)))
            if len({len(v) for v in refined}) != 1:
                raise ValueError("Local extension must retain equal cell counts across bays")
            spec[f"{a}_offsets_in"] = refined if by_bay is not None else refined[0]
        mx = len(spec["x_offsets_in"][0] if isinstance(spec["x_offsets_in"][0], list) else spec["x_offsets_in"]) - 1
        my = len(spec["y_offsets_in"][0] if isinstance(spec["y_offsets_in"][0], list) else spec["y_offsets_in"]) - 1
        result.update(mesh_spec=spec, cells_per_bay=[mx, my], shell_count=nx * ny * mx * my)
        if result["shell_count"] > min(spec["max_shells"], MAX_EXPLICIT_SHELLS):
            return refused("budget_exceeded", f"Local extension needs {result['shell_count']} shells; budget is {spec['max_shells']}")
        grid = floor_mesh(nx, ny, lengths["x"], lengths["y"], 4, mesh_spec=spec)
        nested_refinement(previous_grid, grid)
        result.update(status="proposed", coordinate_sha256=grid["coordinate_sha256"])
        return result
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        return refused("unsupported_evidence", str(exc))
