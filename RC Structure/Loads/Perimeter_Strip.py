"""Perimeter strip of the physical floor on the slab-transfer load route (2026-11-05).

The floor transfer (Design.SMRF_Floor_Transfer) is solved for the centreline plate: the panels between the
grid lines. Weights and masses use the physical footprint (Model.Floor_Footprint), which also holds the slab
layer over the outer half of every perimeter beam and the four corner squares. On the transfer route that
strip, with its superimposed dead and live load, was in no gravity load (load audit of 2026-11-05). This
module is the one description of it, read by the frame loads (Loads.Gravity_Loads), by the beam gravity
shears taken straight from a transfer (Design.Group_Capacity, Design.SMRF_Capacity_Design) and by the load
totals given to the transfer validator:

  beams    each perimeter beam carries the strip of its own side as a uniform line load over its centreline
           span: pressure times the strip width (the grid line to the footprint's outer face), with the
           couple of the strip's eccentricity about the beam axis, half at each end joint (a single frame
           element has no interior node; the same end-joint rule as the transfer's own torsion couples);
  corners  each corner square lies over its corner column: a vertical load at that joint with the couple
           of its centroid offset. It is on no beam span and enters no beam shear;
  live     a unit live case loads a strip piece when it loads the panel beside it (the corner panel for a
           corner square), so the strip of a live pattern follows that pattern's panels.

The pressures are the transfer's own unit pressures, so the strip carries exactly the dead and live factors
of the transfer loads it is added to. The pieces of a floor sum to footprint area minus plate area, and
their first moments to the footprint's minus the plate's. Geometry and arithmetic only: no structural solve.
"""
from __future__ import annotations

import math

import Structure_Parameters as sp
from Model import Floor_Footprint as fp

METHOD = "perimeter_strip_beam_line_loads_and_corner_joint_loads_v1"
# What a beam gravity shear basis says when the strip contributed to it, with the method it was read by (2026-11-06).
SHEAR_BASIS_CLAUSE = f"; plus the perimeter strip of the physical floor on its clear span ({METHOD})"


def pieces(physical, nx, ny, lx, ly):
    """Strip pieces between a footprint rectangle and the centreline plate.

    ``beams``   {(axis, line_index, span_index): width_in, offset_in, length_in, panel}: the transfer's beam
                keys; ``offset_in`` is the signed distance from the beam axis to the strip's centroid (along
                Y for an X beam, along X for a Y beam) and ``panel`` the panel beside the beam;
    ``corners`` {(grid_i, grid_j): area_in2, offset_in (dx, dy), panel}.
    """
    out = {"x_minus": -physical["x_min_in"], "x_plus": physical["x_max_in"] - nx * lx,
           "y_minus": -physical["y_min_in"], "y_plus": physical["y_max_in"] - ny * ly}
    if any(not math.isfinite(v) or v < 0.0 for v in out.values()):
        raise ValueError("The perimeter strip needs a footprint that contains the centreline plate.")
    beams, corners = {}, {}
    for line, side, sign in ((0, "y_minus", -1.0), (ny, "y_plus", 1.0)):
        for span in range(nx):
            beams[("x", line, span)] = {"width_in": out[side], "offset_in": sign * out[side] / 2.0,
                                        "length_in": lx, "panel": (span, min(line, ny - 1))}
    for line, side, sign in ((0, "x_minus", -1.0), (nx, "x_plus", 1.0)):
        for span in range(ny):
            beams[("y", line, span)] = {"width_in": out[side], "offset_in": sign * out[side] / 2.0,
                                        "length_in": ly, "panel": (min(line, nx - 1), span)}
    for i, side_x, sign_x in ((0, "x_minus", -1.0), (nx, "x_plus", 1.0)):
        for j, side_y, sign_y in ((0, "y_minus", -1.0), (ny, "y_plus", 1.0)):
            corners[(i, j)] = {"area_in2": out[side_x] * out[side_y],
                               "offset_in": (sign_x * out[side_x] / 2.0, sign_y * out[side_y] / 2.0),
                               "panel": (min(i, nx - 1), min(j, ny - 1))}
    area = (sum(piece["width_in"] * piece["length_in"] for piece in beams.values())
            + sum(piece["area_in2"] for piece in corners.values()))
    plate = nx * lx * ny * ly
    if not math.isclose(plate + area, physical["area_in2"], rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"Centreline plate plus perimeter strip is {plate + area:.12g} in2; the footprint is "
                         f"{physical['area_in2']:.12g} in2.")
    return {"method": METHOD, "beams": beams, "corners": corners, "outstand_in": out, "area_in2": area,
            "plate_area_in2": plate, "footprint_area_in2": physical["area_in2"]}


def floor_pieces(floor):
    """Strip pieces of one elevated floor, from the footprint the weights use (Model.Floor_Footprint.physical_floor).
    None without a slab, where the weights use the centreline plan itself."""
    if sp.SLAB_THICKNESS_IN is None:
        return None
    return pieces(fp.physical_floor(floor), sp.NUM_BAY_X, sp.NUM_BAY_Y, float(sp.BAY_X), float(sp.BAY_Y))


def uniform_pieces(geometry, b_beam_in):
    """Strip pieces of a uniform design described by a record: its geometry and its one beam width, through the
    same perimeter-beam footprint as Model.Floor_Footprint.uniform_floor (which reads the installed parameters)."""
    nx, ny = geometry["num_bay_x"], geometry["num_bay_y"]
    lx, ly = float(geometry["bay_x_in"]), float(geometry["bay_y_in"])
    beams = [{"axis": "x", "line": j * ly, "start": i * lx, "end": (i + 1) * lx, "b_in": b_beam_in}
             for j in (0, ny) for i in range(nx)]
    beams += [{"axis": "y", "line": i * lx, "start": j * ly, "end": (j + 1) * ly, "b_in": b_beam_in}
              for i in (0, nx) for j in range(ny)]
    return pieces(fp.from_perimeter_beams({"nx": nx, "ny": ny, "lx": lx, "ly": ly}, beams), nx, ny, lx, ly)


def case_pressure_ksi(transfer, name, panel):
    """Pressure (kip/in2) a unit case of ``transfer`` puts on the strip piece beside ``panel``: the dead pressure
    everywhere, the live pressure where the case loads that panel (every panel for ``live``)."""
    if name == "dead":
        return transfer["dead_pressure_ksf"] / 144.0
    if name != "live":
        loaded = next((pattern["panels"] for pattern in transfer.get("live_patterns", [])
                       if f"live_pattern_{pattern['id']}" == name), None)
        if loaded is None:
            raise ValueError(f"Floor transfer has no live pattern for unit case {name!r}.")
        if tuple(panel) not in {tuple(p) for p in loaded}:
            return 0.0
    return transfer["live_pressure_ksf"] / 144.0


def line_load_kip_per_in(strip, transfer, key, name):
    """Line load (kip/in, downward) of unit case ``name`` on beam ``key`` = (axis, line_index, span_index) from
    the strip pieces ``strip``; zero for a beam off the perimeter and without a strip."""
    piece = None if strip is None else strip["beams"].get(tuple(key))
    if piece is None:
        return 0.0
    return case_pressure_ksi(transfer, name, piece["panel"]) * piece["width_in"]


def beam_floor_pieces(floor, key, store=None):
    """Strip pieces of the installed design's floor ``floor`` for the line loads of beam ``key``; None for a beam
    off the perimeter and without a slab, for which no footprint is built. ``store`` is the caller's
    {floor: pieces}: a floor's footprint is built on its first perimeter beam and reused for as long as the caller
    keeps the store, which must not outlive the installed design (2026-11-06)."""
    axis, line = key[0], key[1]
    if sp.SLAB_THICKNESS_IN is None or line not in (0, sp.NUM_BAY_Y if axis == "x" else sp.NUM_BAY_X):
        return None
    if store is None:
        return floor_pieces(floor)
    if floor not in store:
        store[floor] = floor_pieces(floor)
    return store[floor]


def floor_line_load_kip_per_in(floor, transfer, key, name):
    """``line_load_kip_per_in`` for a beam of the installed design's floor ``floor``."""
    return line_load_kip_per_in(beam_floor_pieces(floor, key), transfer, key, name)


def floor_loads(floor, transfer, dead_factor=1.0, live_factor=1.0, live_pattern="all"):
    """Factored strip loads of floor ``floor`` for the frame, with the factors and the live case of the transfer
    loads they join.

    ``beams``  {(axis, line_index, span_index): line load, kip/in downward, over the centreline span};
    ``joints`` {(grid_i, grid_j): [vertical load (kip, downward), global Mx, global My (kip-in)]}: the corner
               squares and, at each end joint of a perimeter beam, half the couple of its strip about its axis.
    A downward load P at (dx, dy) from a joint is P at the joint with Mx = -P dy and My = P dx.
    """
    result = {"method": METHOD, "beams": {}, "joints": {}, "total_kip": 0.0}
    strip = floor_pieces(floor)
    if strip is None:
        return result
    live_case = "live" if live_pattern in (None, "all") else f"live_pattern_{live_pattern}"

    def add(joint, load, mx, my):
        current = result["joints"].setdefault(joint, [0.0, 0.0, 0.0])
        current[0] += load
        current[1] += mx
        current[2] += my

    for name, factor in (("dead", dead_factor), (live_case, live_factor)):
        if factor == 0.0:
            continue
        for key, piece in strip["beams"].items():
            w = factor * line_load_kip_per_in(strip, transfer, key, name)
            if w == 0.0:
                continue
            axis, line, span = key
            result["beams"][key] = result["beams"].get(key, 0.0) + w
            total = w * piece["length_in"]
            result["total_kip"] += total
            couple = (-total * piece["offset_in"], 0.0) if axis == "x" else (0.0, total * piece["offset_in"])
            for joint in (((span, line), (span + 1, line)) if axis == "x" else ((line, span), (line, span + 1))):
                add(joint, 0.0, 0.5 * couple[0], 0.5 * couple[1])
        for joint, piece in strip["corners"].items():
            load = factor * case_pressure_ksi(transfer, name, piece["panel"]) * piece["area_in2"]
            if load == 0.0:
                continue
            dx, dy = piece["offset_in"]
            add(joint, load, -load * dy, load * dx)
            result["total_kip"] += load
    return result


def plate_loads_kip(dead_ksf, live_ksf):
    """(dead, live) load in kip of the centreline plate a floor transfer is solved for: the totals its validator
    is given. The plate is the same on every floor. Every elevated floor's plate and strip must make up the
    floor area the weights use (Model.Member_Properties.floor_area_in2); a floor where they do not is refused,
    so the transfer and the strip together are the footprint load or nothing is validated."""
    from Model import Member_Properties as mp
    plate = sp.NUM_BAY_X * sp.BAY_X * sp.NUM_BAY_Y * sp.BAY_Y
    for floor in range(1, sp.NUM_FLOOR + 1):
        strip = floor_pieces(floor)
        total = plate + (0.0 if strip is None else strip["area_in2"])
        if not math.isclose(total, mp.floor_area_in2(floor), rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(f"Floor {floor}: the centreline plate and the perimeter strip make {total:.12g} in2; "
                             f"the weights use {mp.floor_area_in2(floor):.12g} in2.")
    return dead_ksf * plate / 144.0, live_ksf * plate / 144.0
