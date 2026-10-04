"""Slab-to-frame gravity load transfer for the SMRF design and NTHA models.

The frame models (elastic design model and IMK response model) are bare
centerline frames. Floor gravity used to reach them as tributary-area nodal
loads, which leaves every beam with no gravity moment at all. This module
runs the flexible-beam floor model (SMRF_Floor_Analysis, ``flexible_beams``)
once for unit dead pressure and once for unit live pressure and packs the
result as the discrete node loads each frame beam receives and the footprint
share each column takes directly. Loads/Gravity_Loads applies that record,
scaled by the dead and live factors of the combination at hand, to every
floor of the frame in place of the tributary nodal loads.

Accounting, so nothing is counted twice:
- floor pressure (slab self-weight + SDL, and live) is applied to the floor
  model only, and reaches the frame solely through this transfer;
- beam drop and column self-weight stay on the frame as element loads
  (Gravity_Loads._apply_element_self_weight) and are never floor pressure;
- the seismic mass keeps its tributary-area distribution; only the gravity
  load path changes.

The two unit cases are linear, so a combination's floor load is
``dead_factor * dead + live_factor * live``. Additional unit live cases
carry explicit panel patterns. A case whose exported forces or first moments
do not balance is rejected even if the original shell reactions balance.
"""
from __future__ import annotations

import copy
import math

from Design import SMRF_Floor_Sections as floor_sections
from Design.SMRF_Floor_Analysis import (METHOD_VERSION_FLEXIBLE, analyze_floor,
                                        transfer_mesh_per_bay)

TRANSFER_SCHEMA = "smrf_floor_transfer_v3_force_and_couple_export"
# One transfer per mechanically distinct floor of a grouped design (2026-10-02): the container names, for
# every elevated floor, the signature of its by-line description and holds one v3 transfer per signature.
BY_FLOOR_SCHEMA = "smrf_floor_transfer_by_floor_v1"
LOAD_MODEL_NAME = "slab_transfer"


def _pack_case(result):
    beams = [{"axis": item["axis"], "line_index": item["line_index"],
              "span_index": item["span_index"],
              "node_loads": [[load["x_fraction"], load["load_kip"]] for load in item["node_loads"]],
              "node_couples": [[c["x_fraction"], c["local_x_kip_in"], c["local_y_kip_in"]] for c in item["node_couples"]],
              "total_kip": item["total_kip"],
              "equivalent_uniform_kip_per_in": item["equivalent_uniform_kip_per_in"]}
             for item in result["beam_transfer"]]
    columns = [{"grid_i": item["grid_i"], "grid_j": item["grid_j"],
                "direct_load_kip": item["direct_load_kip"],
                "couple_global_mx_kip_in": item["couple_global_mx_kip_in"],
                "couple_global_my_kip_in": item["couple_global_my_kip_in"],
                "footprint_share_kip_informational": item["footprint_share_kip_informational"]}
               for item in result["column_direct_loads"]]
    return {"loadcase": copy.deepcopy(result["loadcase"]),
            "applied_kip": result["transfer_equilibrium"]["applied_downward_kip"],
            "beam_kip": result["transfer_equilibrium"]["beam_node_loads_kip"],
            "column_direct_kip": result["transfer_equilibrium"]["column_direct_loads_kip"],
            "column_direct_fraction": result["transfer_equilibrium"]["column_direct_fraction"],
            "equilibrium": copy.deepcopy(result["transfer_equilibrium"]),
            "beams": beams, "columns": columns}


def global_couple(axis, local_x, local_y):
    """Beam-local (torsion, bending) couple to global (Mx, My): x beams map directly, y beams rotate."""
    return (local_x, local_y) if axis == "x" else (-local_y, local_x)


def build_floor_transfer(slab_record, geometry, sections, live_load_ksf, mesh_per_bay=None,
                         live_patterns=()):
    """Unit dead and unit live slab-to-frame transfers for one common floor.

    ``sections`` needs b/h/fc of the beam and b/h of the column (footprint), or
    is a floor described line by line (Design.SMRF_Floor_Sections), whose
    signature the transfer then carries.
    ``live_patterns`` (SMRF_Demands.live_load_patterns) adds one unit live
    case per arrangement, keyed ``live_pattern_<id>``, with live pressure on
    that arrangement's panels only. Raises on any solve or equilibrium
    failure: a missing transfer must never silently fall back to tributary
    loads.
    """
    mesh = mesh_per_bay or transfer_mesh_per_bay(geometry["num_bay_x"], geometry["num_bay_y"])
    cases = {
        "dead": {"id": "unit_dead", "dead_factor": 1.0, "live_factor": 0.0,
                 "live_load_ksf": live_load_ksf, "live_pattern": "none"},
        "live": {"id": "unit_live", "dead_factor": 0.0, "live_factor": 1.0,
                 "live_load_ksf": live_load_ksf, "live_pattern": "all"},
    }
    for pattern in live_patterns:
        cases[f"live_pattern_{pattern['id']}"] = {
            "id": f"unit_live_{pattern['id']}", "dead_factor": 0.0, "live_factor": 1.0,
            "live_load_ksf": live_load_ksf, "live_pattern": [list(p) for p in pattern["panels"]]}
    unit_cases = {}
    beam_model = None
    for name, case in cases.items():
        if name.startswith("live") and live_load_ksf <= 0:
            unit_cases[name] = None
            continue
        result = analyze_floor(slab_record, geometry, sections, case, mesh_per_bay=mesh,
                               support_model="flexible_beams")
        if result["status"] != "transfer_complete":
            audit = result.get("transfer_equilibrium", {})
            raise RuntimeError(
                f"Floor transfer {case['id']} failed: status {result['status']}; "
                f"exported load errors (force, X first moment, Y first moment) = "
                f"{audit.get('relative_error')}, {audit.get('x_first_moment_relative_error')}, "
                f"{audit.get('y_first_moment_relative_error')}. "
                "Resolve the mechanical transfer; do not bypass with stored pass flags.")
        unit_cases[name] = _pack_case(result)
        beam_model = result["beam_model"]
    return {
        "schema": TRANSFER_SCHEMA, "method_version": METHOD_VERSION_FLEXIBLE,
        "load_model": LOAD_MODEL_NAME, "mesh_per_bay": mesh,
        "geometry": {key: geometry[key] for key in ("num_bay_x", "num_bay_y", "bay_x_in", "bay_y_in")},
        "sections": (copy.deepcopy(sections) if floor_sections.is_by_line(sections) else
                     {key: sections[key] for key in ("b_beam_in", "h_beam_in", "fc_beam_ksi", "b_col_in", "h_col_in")}),
        **({"floor_sections_sha256": floor_sections.signature(sections)} if floor_sections.is_by_line(sections) else {}),
        "slab_thickness_in": slab_record["thickness_in"],
        "dead_pressure_ksf": slab_record["concrete_unit_weight_kcf"] * slab_record["thickness_in"] / 12.0
        + slab_record["superimposed_dead_load_ksf"],
        "live_pressure_ksf": live_load_ksf,
        "live_patterns": [{"id": p["id"], "rule": p.get("rule"), "panels": [list(x) for x in p["panels"]]}
                          for p in live_patterns],
        "beam_model": beam_model,
        "unit_cases": unit_cases,
        "basis": ("Flexible ACI 8.4.1.8 T-beam floor model; beam-line node forces and couples (torsion, bending) "
                  "from the solved segments, column-node forces and couples from reactions and beam ends; nothing "
                  "redistributed. Couples reach the bare frame as statically equivalent nodal moments. "
                  "Beam drop and column self-weight are separate frame element loads."),
    }


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Floor transfer {name} must be a finite number.")
    return value


def _same(value, expected, name):
    value = _finite(value, name)
    if not math.isclose(value, expected, rel_tol=1e-8, abs_tol=1e-9):
        raise ValueError(f"Floor transfer {name} carries {value:.9g}, expected {expected:.9g}.")


def _index(value, limit, name):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < limit:
        raise ValueError(f"Floor transfer {name} must be an integer within the floor grid.")
    return value


def validate_floor_transfer(transfer, geometry, sections, slab_thickness_in, expected_dead_kip, expected_live_kip):
    """Recompute inventory, force and first moments from actual saved loads.

    Saved totals/pass flags are not evidence of what Gravity_Loads will apply.
    Signed beam/column loads are permitted: patterned elastic floors can have
    uplift. Every beam and column must appear exactly once in every unit case.
    """
    if not isinstance(transfer, dict) or transfer.get("schema") != TRANSFER_SCHEMA:
        raise ValueError("Floor transfer record is missing or has an unknown schema.")
    if transfer.get("method_version") != METHOD_VERSION_FLEXIBLE or transfer.get("load_model") != LOAD_MODEL_NAME:
        raise ValueError("Floor transfer method or load model is stale or unknown.")
    for key in ("num_bay_x", "num_bay_y", "bay_x_in", "bay_y_in"):
        if transfer["geometry"].get(key) != geometry[key]:
            raise ValueError(f"Floor transfer geometry.{key} disagrees with the current model.")
    if floor_sections.is_by_line(sections) or floor_sections.is_by_line(transfer.get("sections")):
        # A floor described line by line: the saved description must be exactly the current floor's.
        if transfer.get("sections") != sections:
            raise ValueError("Floor transfer was solved for other beam lines or supports than the current floor's.")
        if transfer.get("floor_sections_sha256") != floor_sections.signature(sections):
            raise ValueError("Floor transfer signature does not match its beam lines and supports.")
    else:
        for key in ("b_beam_in", "h_beam_in", "fc_beam_ksi", "b_col_in", "h_col_in"):
            if transfer["sections"].get(key) != sections[key]:
                raise ValueError(f"Floor transfer sections.{key} disagrees with the current model.")
    if transfer.get("slab_thickness_in") != slab_thickness_in:
        raise ValueError("Floor transfer slab thickness disagrees with the current model.")
    nx, ny = geometry["num_bay_x"], geometry["num_bay_y"]
    lx, ly = geometry["bay_x_in"], geometry["bay_y_in"]
    panel_area = lx * ly / 144.0
    area = nx * ny * panel_area
    for value, name in ((expected_dead_kip, "expected dead"), (expected_live_kip, "expected live")):
        if _finite(value, name) < 0:
            raise ValueError(f"Floor transfer {name} must not be negative.")
    _same(transfer.get("dead_pressure_ksf"), expected_dead_kip / area, "dead pressure")
    _same(transfer.get("live_pressure_ksf"), expected_live_kip / area, "live pressure")
    all_panels = {(i, j) for i in range(nx) for j in range(ny)}
    case_panels = {"dead": all_panels, "live": all_panels}
    for pattern in transfer.get("live_patterns", []):
        pid = pattern.get("id")
        if not isinstance(pid, str) or not pid.strip() or f"live_pattern_{pid}" in case_panels:
            raise ValueError("Floor transfer live pattern IDs must be nonempty and unique.")
        panels = set()
        for panel in pattern.get("panels", []):
            if not isinstance(panel, (list, tuple)) or len(panel) != 2:
                raise ValueError("Floor transfer pattern must contain panel index pairs.")
            i, j = _index(panel[0], nx, "pattern i"), _index(panel[1], ny, "pattern j")
            if (i, j) in panels:
                raise ValueError("Floor transfer pattern contains duplicate panels.")
            panels.add((i, j))
        if not panels:
            raise ValueError("Floor transfer live patterns must contain at least one panel.")
        case_panels[f"live_pattern_{pid}"] = panels
    cases = transfer.get("unit_cases", {})
    if set(cases) != set(case_panels):
        raise ValueError("Floor transfer unit case inventory does not match its dead/live/pattern inventory.")
    expected_beams = {("x", j, i) for j in range(ny + 1) for i in range(nx)} | {
        ("y", i, j) for i in range(nx + 1) for j in range(ny)}
    expected_columns = {(i, j) for i in range(nx + 1) for j in range(ny + 1)}
    for name, panels in case_panels.items():
        pressure = (expected_dead_kip if name == "dead" else expected_live_kip) / area
        expected = pressure * panel_area * len(panels)
        case = cases[name]
        if case is None:
            if expected > 0:
                raise ValueError(f"Floor transfer lacks the {name} unit case.")
            continue
        descriptor = case.get("loadcase", {})
        _same(descriptor.get("dead_factor"), 1.0 if name == "dead" else 0.0, f"{name} dead factor")
        _same(descriptor.get("live_factor"), 0.0 if name == "dead" else 1.0, f"{name} live factor")
        _same(descriptor.get("live_load_ksf"), expected_live_kip / area, f"{name} live pressure")
        pattern = descriptor.get("live_pattern")
        if name in ("dead", "live"):
            if pattern != ("none" if name == "dead" else "all"):
                raise ValueError(f"Floor transfer {name} has the wrong load pattern.")
        elif (not isinstance(pattern, list) or len(pattern) != len(panels)
              or sorted(pattern) != [list(p) for p in sorted(panels)]):
            raise ValueError(f"Floor transfer {name} has the wrong load pattern.")
        if case.get("equilibrium", {}).get("numerical_balance_passed") is not True:
            raise ValueError(f"Floor transfer {name} case did not balance.")
        beam_total = column_total = mx = my = couple_x = couple_y = 0.0
        beam_keys, column_keys = set(), set()
        for beam in case["beams"]:
            axis, line, span = beam.get("axis"), beam.get("line_index"), beam.get("span_index")
            if axis not in ("x", "y"):
                raise ValueError("Floor transfer beam axis must be x or y.")
            _index(line, ny + 1 if axis == "x" else nx + 1, "beam line")
            _index(span, nx if axis == "x" else ny, "beam span")
            key = (axis, line, span)
            if key in beam_keys:
                raise ValueError("Floor transfer has a duplicate beam.")
            beam_keys.add(key)
            total, fractions = 0.0, set()
            for x, p in beam["node_loads"]:
                _finite(x, "beam load position")
                _finite(p, "beam load")
                if not 0 < x < 1 or x in fractions:
                    raise ValueError("Floor transfer beam loads must have unique positions strictly inside the span.")
                fractions.add(x)
                total += p
                mx += p * ((span + x) * lx if axis == "x" else line * lx)
                my += p * (line * ly if axis == "x" else (span + x) * ly)
            if not fractions:
                raise ValueError("Floor transfer beam is missing its load inventory.")
            couple_positions = set()
            for row in beam.get("node_couples", []):
                if not isinstance(row, (list, tuple)) or len(row) != 3:
                    raise ValueError("Floor transfer beam couples must be [position, torsion, bending] rows.")
                x, cx, cy = row
                _finite(x, "beam couple position")
                if not 0 < x < 1 or x in couple_positions:
                    raise ValueError("Floor transfer beam couples must have unique positions strictly inside the span.")
                couple_positions.add(x)
                gx, gy = global_couple(axis, _finite(cx, "beam torsion couple"), _finite(cy, "beam bending couple"))
                couple_x += gx
                couple_y += gy
            if couple_positions != fractions:
                raise ValueError("Floor transfer beam couples must sit at the beam's load positions.")
            _same(beam.get("total_kip"), total, f"{name} beam total")
            _same(beam.get("equivalent_uniform_kip_per_in"), total / (lx if axis == "x" else ly), f"{name} beam uniform summary")
            beam_total += total
        for column in case["columns"]:
            i = _index(column.get("grid_i"), nx + 1, "column i")
            j = _index(column.get("grid_j"), ny + 1, "column j")
            if (i, j) in column_keys:
                raise ValueError("Floor transfer has a duplicate column.")
            column_keys.add((i, j))
            p = _finite(column.get("direct_load_kip"), "column load")
            column_total += p
            mx += p * i * lx
            my += p * j * ly
            couple_x += _finite(column.get("couple_global_mx_kip_in"), "column couple Mx")
            couple_y += _finite(column.get("couple_global_my_kip_in"), "column couple My")
        if beam_keys != expected_beams or column_keys != expected_columns:
            raise ValueError(f"Floor transfer {name} has an incomplete beam/column inventory.")
        _same(case.get("beam_kip"), beam_total, f"{name} beam summary")
        _same(case.get("column_direct_kip"), column_total, f"{name} column summary")
        _same(case.get("applied_kip"), expected, f"{name} applied summary")
        _same(beam_total + column_total, expected, f"{name} actual force")
        expected_x = sum(pressure * panel_area * (i + .5) * lx for i, j in panels)
        expected_y = sum(pressure * panel_area * (j + .5) * ly for i, j in panels)
        # Downward P at (x, y): external Mx = -P*y + couple_x, My = P*x + couple_y.
        for actual, target, length, axis in ((mx + couple_y, expected_x, lx, "X"), (my - couple_x, expected_y, ly, "Y")):
            if abs(actual - target) > 1e-8 * max(abs(target), expected * length, 1e-9):
                raise ValueError(f"Floor transfer {name} {axis} moment balance carries {actual:.9g}, expected {target:.9g} kip-in.")
    return transfer


def is_by_floor(record):
    return isinstance(record, dict) and record.get("schema") == BY_FLOOR_SCHEMA


def build_floor_transfers_by_floor(slab_record, geometry, description, live_load_ksf, mesh_per_bay=None,
                                   live_patterns=(), reuse=None):
    """One transfer per mechanically distinct floor of a grouped design.

    ``description`` is Design.SMRF_Floor_Sections.by_floor(): every elevated floor with the signature of
    its beam lines and supports. Floors with one signature share one solution; nothing is shared by band
    or by position. ``reuse`` is an earlier container: a transfer whose signature, slab, loads, patterns
    and mesh are all unchanged is kept (validated by the caller as usual), anything else is solved again.
    A failed solve raises, as for the uniform transfer: there is no tributary fallback.
    """
    floor_sections.validate_by_floor(description, geometry["num_bay_x"], geometry["num_bay_y"], geometry["num_floor"])
    mesh = mesh_per_bay or transfer_mesh_per_bay(geometry["num_bay_x"], geometry["num_bay_y"])
    patterns = [{"id": p["id"], "rule": p.get("rule"), "panels": [list(x) for x in p["panels"]]} for p in live_patterns]
    kept = (reuse or {}).get("transfers", {}) if is_by_floor(reuse) else {}
    transfers, reused = {}, []
    for sha, sections, _floors in floor_sections.distinct_floors(description):
        old = kept.get(sha)
        if (old is not None and old.get("sections") == sections and old.get("mesh_per_bay") == mesh
                and old.get("slab_thickness_in") == slab_record["thickness_in"]
                and old.get("live_pressure_ksf") == live_load_ksf and old.get("live_patterns") == patterns
                and old.get("geometry") == {key: geometry[key] for key in ("num_bay_x", "num_bay_y", "bay_x_in", "bay_y_in")}
                and old.get("slab_record_sha256") == _slab_identity(slab_record)):
            transfers[sha] = old
            reused.append(sha)
            continue
        transfer = build_floor_transfer(slab_record, geometry, sections, live_load_ksf, mesh_per_bay=mesh,
                                        live_patterns=live_patterns)
        transfer["slab_record_sha256"] = _slab_identity(slab_record)
        transfers[sha] = transfer
    return {"schema": BY_FLOOR_SCHEMA, "method_version": METHOD_VERSION_FLEXIBLE, "load_model": LOAD_MODEL_NAME,
            "mesh_per_bay": mesh, "floors": dict(description["floors"]), "transfers": transfers,
            "distinct_floor_count": len(transfers), "reused_signatures": reused,
            "basis": ("One flexible-beam floor solution for every mechanically distinct floor (beam lines, supports, slab, "
                      "loads, patterns and mesh); floors are matched by the signature of those inputs, never by band or "
                      "position. Each transfer is the v3 force-and-couple export, validated on its own.")}


def _slab_identity(slab_record):
    """What of the slab record the floor solution depends on."""
    import hashlib
    import json
    payload = {key: slab_record[key] for key in ("thickness_in", "concrete_fc_ksi", "concrete_unit_weight_kcf",
                                                 "superimposed_dead_load_ksf")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def validate_floor_transfers_by_floor(record, geometry, description, slab_thickness_in, expected_dead_kip, expected_live_kip):
    """Validate a by-floor container against the CURRENT floors and return {floor: transfer}.

    ``description`` is the by-floor description of the model as it stands (built from the installed
    design, not read from the record). Every floor must be served by a transfer solved for exactly its
    own beam lines and supports; a container left from another design, a floor without a transfer, an
    unused transfer and a uniform (one-floor) transfer are all refused. Each transfer is then put through
    the full force, couple and first-moment validation of ``validate_floor_transfer``.
    """
    if not is_by_floor(record):
        raise ValueError("A grouped design needs a floor transfer solved floor by floor; this record is not one "
                         f"(schema {None if not isinstance(record, dict) else record.get('schema')!r}).")
    if record.get("method_version") != METHOD_VERSION_FLEXIBLE or record.get("load_model") != LOAD_MODEL_NAME:
        raise ValueError("Floor transfer method or load model is stale or unknown.")
    floor_sections.validate_by_floor(description, geometry["num_bay_x"], geometry["num_bay_y"], geometry["num_floor"])
    if record.get("floors") != description["floors"]:
        stale = sorted(k for k in description["floors"] if (record.get("floors") or {}).get(k) != description["floors"][k])
        raise ValueError(f"Floor transfer was solved for other floors than the current design's (floors {stale} differ).")
    transfers = record.get("transfers")
    if not isinstance(transfers, dict) or set(transfers) != set(description["distinct"]):
        raise ValueError("Floor transfer does not hold exactly one solution per distinct floor of the current design.")
    result = {}
    for sha, sections, floors in floor_sections.distinct_floors(description):
        transfer = validate_floor_transfer(transfers[sha], geometry, sections, slab_thickness_in,
                                           expected_dead_kip, expected_live_kip)
        if transfer.get("mesh_per_bay") != record.get("mesh_per_bay"):
            raise ValueError("Floor transfers of one design must share one mesh.")
        for floor in floors:
            result[floor] = transfer
    return result


def transfer_totals(transfer, dead_factor, live_factor):
    """Total floor load (kip) the transfer applies for the given factors."""
    total = 0.0
    for name, factor in (("dead", dead_factor), ("live", live_factor)):
        case = transfer["unit_cases"].get(name)
        if case is not None:
            total += factor * (case["beam_kip"] + case["column_direct_kip"])
    return total
