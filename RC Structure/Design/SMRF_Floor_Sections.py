"""The beam lines and supports of one floor, for the floor models (grouped iterative design, 2026-10-02).

The floor and slab modules used to receive ``sections`` as one beam and one column for the whole
building. With design groups a floor has its own beam on every line and its own column under every
support, and floors of different story bands differ. This module defines the by-line description the
floor modules accept instead, builds it for a floor of the installed design, and gives it an identity:

    {"schema": SCHEMA,
     "beam_lines": {"x": [beam of X line j = 0..ny], "y": [beam of Y line i = 0..nx]},   # b_in, h_in, fc_ksi
     "supports": [[column below the floor at (i, j) for i = 0..nx] for j = 0..ny],        # b_in, h_in
     "fc_beam_ksi": the one beam concrete grade (also the slab's)}

A beam line carries one section over all its spans (the grouping policy gives every span of a line at a
floor the same group; a line that does not is refused here rather than averaged). The support entry is
the column whose section the joint core takes (Model.Member_Properties: the column below). Nothing in
the description is bookkeeping: group names, floor numbers and bands are left out, so two floors with the
same mechanical inputs have the same signature and may share one solution, and two floors that differ in
any of them never do.

The one-beam, one-column dictionary of the uniform code is still accepted everywhere; the accessors below
read either form. ``require_uniform`` is the refusal used by the diagnostic-only floor modules that are
not wired for unlike lines.
"""
from __future__ import annotations

import hashlib
import json

SCHEMA = "smrf_floor_sections_by_line_v1"
BY_FLOOR_SCHEMA = "smrf_floor_sections_by_floor_v1"
_UNIFORM_KEYS = ("b_beam_in", "h_beam_in", "fc_beam_ksi", "b_col_in", "h_col_in")
_PASS_THROUGH = ("slab_poisson_ratio", "beam_stiffness_modifier", "beam_torsion_modifier")


def is_by_line(sections):
    return isinstance(sections, dict) and sections.get("schema") == SCHEMA


def is_by_floor(sections):
    return isinstance(sections, dict) and sections.get("schema") == BY_FLOOR_SCHEMA


def require_uniform(sections, who):
    """Refuse a by-line or by-floor description in a module that models one beam and one column."""
    if isinstance(sections, dict) and sections.get("schema") is not None:
        raise ValueError(f"{who} models one beam section and one column section and is not wired for a grouped design; "
                         "it cannot take a floor described line by line.")
    return sections


def beam_line(sections, axis, line_index):
    """{"b_in", "h_in", "fc_ksi"} of the beam on X line j or Y line i."""
    if is_by_line(sections):
        return sections["beam_lines"][axis][line_index]
    return {"b_in": sections["b_beam_in"], "h_in": sections["h_beam_in"], "fc_ksi": sections["fc_beam_ksi"]}


def beam_width(sections, axis, line_index):
    """Width of the beam on X line j or Y line i (the only beam property the strip and mesh code needs)."""
    if is_by_line(sections):
        return sections["beam_lines"][axis][line_index]["b_in"]
    return sections["b_beam_in"]


def beam_lines(sections, axis, count):
    return [beam_line(sections, axis, index) for index in range(count)]


def support(sections, i, j):
    """{"b_in", "h_in"} of the column under support (i, j), or None when the description has no column."""
    if is_by_line(sections):
        return sections["supports"][j][i]
    if "b_col_in" in sections or "h_col_in" in sections:
        return {"b_in": sections["b_col_in"], "h_in": sections["h_col_in"]}
    return None


def beam_concrete_ksi(sections):
    return sections["fc_beam_ksi"]


def panel_face_widths(sections, panel_i, panel_j):
    """Widths of the four beams bounding panel (i, j): {"x": (lower, upper), "y": (lower, upper)}.

    "x" names the faces normal to X, which belong to the Y-direction beams on lines i and i + 1; "y" the
    faces normal to Y, the X-direction beams on lines j and j + 1.
    """
    return {"x": (beam_width(sections, "y", panel_i), beam_width(sections, "y", panel_i + 1)),
            "y": (beam_width(sections, "x", panel_j), beam_width(sections, "x", panel_j + 1))}


def face_widths(sections, nx, ny):
    """Beam widths at the faces the mesh recipe grades toward: {"x": [Y line i widths], "y": [X line j widths]}."""
    return {"x": [beam_width(sections, "y", i) for i in range(nx + 1)],
            "y": [beam_width(sections, "x", j) for j in range(ny + 1)]}


def _canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def signature(sections):
    """sha256 of a by-line description (its mechanical content only)."""
    if not is_by_line(sections):
        raise ValueError("A floor signature is defined for a by-line floor description.")
    return hashlib.sha256(_canonical(sections).encode("utf-8")).hexdigest()


def validate(sections, nx, ny):
    """Shape and values of a by-line description; raises on anything missing or non-physical."""
    if not is_by_line(sections):
        raise ValueError("Not a by-line floor description.")
    lines = sections.get("beam_lines") or {}
    if set(lines) != {"x", "y"} or len(lines["x"]) != ny + 1 or len(lines["y"]) != nx + 1:
        raise ValueError("A by-line floor needs one beam for every X line and every Y line.")
    for axis in ("x", "y"):
        for beam in lines[axis]:
            if set(beam) != {"b_in", "h_in", "fc_ksi"} or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0 for v in beam.values()):
                raise ValueError("Every beam line needs positive b_in, h_in and fc_ksi.")
            if beam["fc_ksi"] != sections.get("fc_beam_ksi"):
                raise ValueError("Beam lines must share the one beam concrete grade (fc_beam_ksi) in this revision.")
    rows = sections.get("supports")
    if not isinstance(rows, list) or len(rows) != ny + 1 or any(len(row) != nx + 1 for row in rows):
        raise ValueError("A by-line floor needs one support column for every grid intersection.")
    for row in rows:
        for column in row:
            if set(column) != {"b_in", "h_in"} or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0 for v in column.values()):
                raise ValueError("Every support needs positive b_in and h_in.")
    unknown = set(sections) - {"schema", "beam_lines", "supports", "fc_beam_ksi", *_PASS_THROUGH}
    if unknown:
        raise ValueError(f"Unknown fields in a by-line floor description: {sorted(unknown)}.")
    return sections


def uniform_equivalent(sections):
    """The one-beam, one-column dictionary when every line and support is the same; None otherwise."""
    if not is_by_line(sections):
        return dict(sections)
    beams = [beam for axis in ("x", "y") for beam in sections["beam_lines"][axis]]
    columns = [column for row in sections["supports"] for column in row]
    if any(beam != beams[0] for beam in beams) or any(column != columns[0] for column in columns):
        return None
    return {"b_beam_in": beams[0]["b_in"], "h_beam_in": beams[0]["h_in"], "fc_beam_ksi": beams[0]["fc_ksi"],
            "b_col_in": columns[0]["b_in"], "h_col_in": columns[0]["h_in"]}


# ---- from the installed design ---------------------------------------------------------------------------
def for_floor(floor):
    """By-line description of floor ``floor`` of the installed grouped design."""
    import Structure_Parameters as sp
    from Model import Member_Groups as mg
    from Model import Member_Properties as mp
    if not mg.is_grouped():
        raise mg.GroupedStateError("A by-line floor description is built from an installed grouped design.")
    nx, ny = sp.NUM_BAY_X, sp.NUM_BAY_Y

    def line(kind, index):
        spans = ([mg.beam_at("beam_x", floor, i, index) for i in range(nx)] if kind == "beam_x"
                 else [mg.beam_at("beam_y", floor, index, j) for j in range(ny)])
        designs = {(m.design.b_in, m.design.h_in, m.design.fc_ksi) for m in spans}
        if len(designs) != 1:
            raise mg.GroupedStateError(f"Floor {floor}: the spans of {kind} line {index} have different sections "
                                       f"({sorted(designs)}); a beam line carries one section in the floor models.")
        b, h, fc = designs.pop()
        return {"b_in": b, "h_in": h, "fc_ksi": fc}

    lines = {"x": [line("beam_x", j) for j in range(ny + 1)], "y": [line("beam_y", i) for i in range(nx + 1)]}
    grades = {beam["fc_ksi"] for axis in ("x", "y") for beam in lines[axis]}
    if len(grades) != 1:
        raise mg.GroupedStateError(f"Floor {floor}: beam groups use different concrete grades {sorted(grades)}; this revision "
                                   "keeps one beam grade (group-specific grades wait for joint material-transition rules).")
    supports = []
    for j in range(ny + 1):
        row = []
        for i in range(nx + 1):
            design = mp.joint_core_column(floor, i, j).design
            row.append({"b_in": design.b_in, "h_in": design.h_in})
        supports.append(row)
    return validate({"schema": SCHEMA, "beam_lines": lines, "supports": supports, "fc_beam_ksi": grades.pop()}, nx, ny)


def by_floor():
    """{"schema", "floors": {"1": signature, ...}, "distinct": {signature: {"sections", "floors": [k, ...]}}}
    for the installed grouped design: every floor described, floors with one signature sharing one entry."""
    import Structure_Parameters as sp
    floors, distinct = {}, {}
    for floor in range(1, sp.NUM_FLOOR + 1):
        sections = for_floor(floor)
        sha = signature(sections)
        floors[str(floor)] = sha
        distinct.setdefault(sha, {"sections": sections, "floors": []})["floors"].append(floor)
    return {"schema": BY_FLOOR_SCHEMA, "floors": floors, "distinct": distinct}


def validate_by_floor(description, nx, ny, num_floor):
    if not is_by_floor(description):
        raise ValueError("Not a by-floor description.")
    floors, distinct = description.get("floors"), description.get("distinct")
    if not isinstance(floors, dict) or sorted(floors) != sorted(str(k) for k in range(1, num_floor + 1)):
        raise ValueError("A by-floor description names every elevated floor exactly once.")
    if not isinstance(distinct, dict) or set(floors.values()) != set(distinct):
        raise ValueError("Floors and distinct floor signatures do not correspond.")
    for sha, entry in distinct.items():
        validate(entry["sections"], nx, ny)
        if signature(entry["sections"]) != sha:
            raise ValueError("A floor description does not match its signature.")
        if sorted(entry["floors"]) != sorted(int(k) for k, v in floors.items() if v == sha):
            raise ValueError("A distinct floor lists other floors than those that carry its signature.")
    return description


def distinct_floors(description):
    """[(signature, sections, [floors])] in ascending order of the first floor each serves."""
    return sorted(((sha, entry["sections"], list(entry["floors"])) for sha, entry in description["distinct"].items()),
                  key=lambda item: item[2][0])
