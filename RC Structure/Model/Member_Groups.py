"""Design groups and the canonical per-member resolver (grouped iterative design, 2026-10-02).

User decision (grouped design, 2026-10-02): member design may vary by
story band and by framing location while mirrored positions keep one design. This module is the one
place that says which physical member belongs to which design group and what that group selected. Design,
both analysis models, the gravity and mass ledgers, the floor models, the joint and section checks, the
diagnostics and the exports all resolve a member through it.

Membership is bookkeeping from integer grid indices only: no coordinate, no tolerance.

  story bands        pairs of stories from the base; a last single story is merged into the band below
                     (four stories: 1-2, 3-4; five: 1-2, 3-5; nine: 1-2, 3-4, 5-6, 7-9). A research
                     search policy, not a code rule.
  column story k     the segment between level k-1 and level k; it belongs to the band containing k.
  beam at floor k    belongs to the band containing k.
  column locations   corner (i on an X boundary and j on a Y boundary), x_boundary_edge (i = 0 or nx,
                     j strictly interior), y_boundary_edge (j = 0 or ny, i strictly interior), interior.
  beam locations     edge or interior line, separately for beam_x and beam_y (X and Y spans can differ).

A group is (band, member type, location): four column groups and four beam groups per band. The reference case (2 x 6 bays, six stories) has
24 groups over 318 members. Groups carry no dimensions or reinforcement by themselves: those are
selected by the iterative design and stored as ``designs`` of a ``GroupedDesign``.

Two modes, never mixed:
  * uniform (legacy): no grouped design is installed; ``resolve`` reads the single column and beam
    design held in Structure_Parameters. This is the explicit uniform adapter the existing uniform
    search, its records and the historical diagnostics use.
  * grouped: ``install`` puts a ``GroupedDesign`` in force and replaces the uniform section and cage
    values of Structure_Parameters by a sentinel that raises on any use. A consumer that has not been
    made member-aware therefore fails loudly instead of installing one stale section everywhere, and
    a grouped record can never fall back to a global property.

Physical element tags and nodes are those of Model.Build_Model / Design.SMRF_Elastic (columns by story,
then X beams, then Y beams); spring tags and local axes are identified separately (Model.IMK_Hinges).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace

import Structure_Parameters as sp

SCHEMA = "seisframe_member_groups_v1"
POLICY_ID = "story_pairs_from_base_orphan_merged__location_class_v1"
STORIES_PER_BAND = 2
MEMBER_TYPES = ("column", "beam_x", "beam_y")
COLUMN_LOCATIONS = ("corner", "x_boundary_edge", "y_boundary_edge", "interior")
BEAM_LOCATIONS = ("edge", "interior")
POLICY = {
    "id": POLICY_ID,
    "stories_per_band": STORIES_PER_BAND,
    "story_bands": "pairs of stories from the base; a last single story is merged into the band below",
    "column_story": "segment from level k-1 to level k belongs to story k",
    "beam_floor": "a beam at floor k belongs to the band containing k",
    "column_locations": {"corner": "i on an X boundary and j on a Y boundary",
                         "x_boundary_edge": "i = 0 or i = nx, with j strictly interior",
                         "y_boundary_edge": "j = 0 or j = ny, with i strictly interior",
                         "interior": "i and j strictly interior"},
    "beam_locations": {"edge": "beam_x on j = 0 or j = ny; beam_y on i = 0 or i = nx", "interior": "every other line"},
    "symmetry": "mirrored positions share a group; earthquake signs and load patterns are still enveloped over its members",
    "basis": "research search policy (user decision 2026-10-02); not a code rule; membership from integer grid indices only",
    "selection": "iterative design; a group prescribes no dimension and no reinforcement",
}

# The Structure_Parameters names that hold ONE section or cage for every column or every beam.
UNIFORM_SECTION_KEYS = (
    "B_COL", "H_COL", "FC_COL_KSI", "B_BEAM", "H_BEAM", "FC_BEAM_KSI",
    "COL_BAR_SIZE", "COL_TOP_BARS", "COL_BOT_BARS", "COL_SIDE_BARS", "COL_BAR_AREA",
    "BEAM_BAR_SIZE", "BEAM_TOP_BARS", "BEAM_BOT_BARS", "BEAM_SIDE_BARS", "BEAM_BAR_AREA",
    "COL_STIRRUP_SPACING", "BEAM_STIRRUP_SPACING", "COL_STIRRUP_BAR_SIZE", "COL_STIRRUP_LEGS",
    "COL_STIRRUP_LEGS_BY_DIRECTION", "BEAM_STIRRUP_BAR_SIZE", "BEAM_STIRRUP_LEGS",
)


class GroupedStateError(RuntimeError):
    """A uniform (one-section-for-all) value was used, or a member could not be resolved, under a grouped design."""


class _UniformValueWithdrawn:
    """Stands in for a uniform section value while a grouped design is installed: any use raises."""

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def _refuse(self, *args, **kwargs):
        raise GroupedStateError(
            f"Structure_Parameters.{self.name} is not defined under a grouped design: every member has its own section and "
            "cage. Resolve the member (Model.Member_Groups.resolve) instead of reading one value for all members.")

    __float__ = __int__ = __index__ = __bool__ = __round__ = __abs__ = __neg__ = __pos__ = _refuse
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _refuse
    __floordiv__ = __rfloordiv__ = __mod__ = __rmod__ = __pow__ = __rpow__ = __divmod__ = _refuse
    __lt__ = __le__ = __gt__ = __ge__ = __eq__ = __ne__ = __hash__ = _refuse
    __format__ = __str__ = __iter__ = __len__ = __getitem__ = __contains__ = _refuse

    def __repr__(self):
        return f"<uniform {self.name} withdrawn: grouped design installed>"


# ---- membership ---------------------------------------------------------------------------------------
def story_bands(num_floor, stories_per_band=STORIES_PER_BAND):
    """Story bands from the base; a last band shorter than the others is merged into the one below."""
    if type(num_floor) is not int or num_floor < 1:
        raise ValueError(f"num_floor must be a positive integer, got {num_floor!r}.")
    bands = [list(range(start, min(start + stories_per_band, num_floor + 1)))
             for start in range(1, num_floor + 1, stories_per_band)]
    if len(bands) > 1 and len(bands[-1]) < stories_per_band:
        bands[-2].extend(bands.pop())
    return bands


def band_label(band):
    return f"s{band[0]:02d}_{band[-1]:02d}"


def column_location(i, j, nx, ny):
    on_x, on_y = i in (0, nx), j in (0, ny)
    return "corner" if on_x and on_y else "x_boundary_edge" if on_x else "y_boundary_edge" if on_y else "interior"


def beam_location(member_type, i, j, nx, ny):
    if member_type == "beam_x":
        return "edge" if j in (0, ny) else "interior"
    if member_type == "beam_y":
        return "edge" if i in (0, nx) else "interior"
    raise ValueError(f"{member_type!r} is not a beam type.")


def group_id(band, member_type, location):
    return f"{band_label(band)}__{member_type}__{location}"


def member_inventory(nx, ny, nz):
    """Every physical member in tag order with its integer grid position, end nodes and group.

    ``story_or_floor`` is the story of a column (segment k-1 to k) or the floor of a beam. Node tags
    are Model.Build_Model.node_tag for these counts; element tags are the builders' own order.
    """
    for name, value in (("num_bay_x", nx), ("num_bay_y", ny), ("num_floor", nz)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    per_level = (nx + 1) * (ny + 1)
    bands = story_bands(nz)
    band_of = {k: band for band in bands for k in band}

    def node(k, i, j):
        return k * per_level + j * (nx + 1) + i + 1

    rows = []

    def add(k, i, j, member_type, node_i, node_j, location):
        rows.append({"member_tag": len(rows) + 1, "member_type": member_type, "story_or_floor": k, "grid_i": i, "grid_j": j,
                     "node_i": node_i, "node_j": node_j, "group_id": group_id(band_of[k], member_type, location),
                     "location_class": location, "band": band_label(band_of[k])})

    for k in range(1, nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                add(k, i, j, "column", node(k - 1, i, j), node(k, i, j), column_location(i, j, nx, ny))
    for k in range(1, nz + 1):
        for j in range(ny + 1):
            for i in range(nx):
                add(k, i, j, "beam_x", node(k, i, j), node(k, i + 1, j), beam_location("beam_x", i, j, nx, ny))
    for k in range(1, nz + 1):
        for j in range(ny):
            for i in range(nx + 1):
                add(k, i, j, "beam_y", node(k, i, j), node(k, i, j + 1), beam_location("beam_y", i, j, nx, ny))
    return rows


def _canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def build_assignments(nx, ny, nz):
    """The complete, deterministic membership map for these counts, with its identity."""
    members = member_inventory(nx, ny, nz)
    groups = {}
    for row in members:
        entry = groups.setdefault(row["group_id"], {"group_id": row["group_id"], "member_type": row["member_type"],
                                                    "band": row["band"], "stories_or_floors": set(),
                                                    "location_class": row["location_class"], "member_tags": []})
        entry["stories_or_floors"].add(row["story_or_floor"])
        entry["member_tags"].append(row["member_tag"])
    group_rows = []
    for gid in sorted(groups):
        entry = groups[gid]
        group_rows.append({**entry, "stories_or_floors": sorted(entry["stories_or_floors"]), "member_count": len(entry["member_tags"])})
    payload = {"schema": SCHEMA, "policy": POLICY, "counts": {"num_bay_x": nx, "num_bay_y": ny, "num_floor": nz},
               "story_bands": story_bands(nz), "groups": group_rows, "members": members}
    return {**payload, "sha256": hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()}


def assignment_problems(assignments, nx, ny, nz):
    """Why a membership map is not THE map of these counts; empty when it is.

    Every physical member has exactly one assignment, of its own type, at its own grid position, to
    the group its position defines. A missing, duplicate, wrong-type or geometrically inconsistent
    assignment is a problem; nothing is repaired.
    """
    problems = []
    if not isinstance(assignments, dict):
        return ["assignments is not an object"]
    if assignments.get("schema") != SCHEMA:
        problems.append(f"schema is {assignments.get('schema')!r}, expected {SCHEMA!r}")
    if (assignments.get("policy") or {}).get("id") != POLICY_ID:
        problems.append(f"grouping policy is {(assignments.get('policy') or {}).get('id')!r}, expected {POLICY_ID!r}")
    expected = build_assignments(nx, ny, nz)
    if assignments.get("counts") != expected["counts"]:
        problems.append(f"counts are {assignments.get('counts')}, the model has {expected['counts']}")
    rows = assignments.get("members")
    if not isinstance(rows, list):
        return problems + ["members is missing"]
    by_tag, duplicates = {}, set()
    for row in rows:
        tag = row.get("member_tag") if isinstance(row, dict) else None
        if type(tag) is not int:
            problems.append(f"a member row has no integer member_tag: {row!r}"[:200])
            continue
        if tag in by_tag:
            duplicates.add(tag)
        by_tag[tag] = row
    if duplicates:
        problems.append(f"duplicate assignments for member tags {sorted(duplicates)[:12]}")
    wanted = {row["member_tag"]: row for row in expected["members"]}
    missing = sorted(set(wanted) - set(by_tag))
    extra = sorted(set(by_tag) - set(wanted))
    if missing:
        problems.append(f"{len(missing)} members have no assignment, e.g. {missing[:12]}")
    if extra:
        problems.append(f"{len(extra)} assignments name members the model does not have, e.g. {extra[:12]}")
    wrong = []
    for tag in sorted(set(wanted) & set(by_tag)):
        for key in ("member_type", "story_or_floor", "grid_i", "grid_j", "node_i", "node_j", "group_id", "location_class"):
            if by_tag[tag].get(key) != wanted[tag][key]:
                wrong.append(f"member {tag}: {key} is {by_tag[tag].get(key)!r}, its position gives {wanted[tag][key]!r}")
                break
    if wrong:
        problems.append(f"{len(wrong)} assignments disagree with the member's type or position, e.g. " + "; ".join(wrong[:4]))
    listed = {g.get("group_id"): g for g in assignments.get("groups") or [] if isinstance(g, dict)}
    for group in expected["groups"]:
        got = listed.get(group["group_id"])
        if got is None:
            problems.append(f"group {group['group_id']} is missing")
        elif sorted(got.get("member_tags") or []) != group["member_tags"] or got.get("member_type") != group["member_type"]:
            problems.append(f"group {group['group_id']} lists other members or another member type")
    for gid in sorted(set(listed) - {g["group_id"] for g in expected["groups"]}):
        problems.append(f"group {gid!r} is not a group of these counts")
    if not problems and assignments.get("sha256") != expected["sha256"]:
        problems.append("the assignment digest does not match its content")
    return problems


# ---- what a group selected ----------------------------------------------------------------------------
@dataclass(frozen=True)
class MemberDesign:
    """The section, concrete strength and cage one design group selected (kip, inch, ksi)."""
    member_type: str                 # "column", "beam_x" or "beam_y"
    b_in: float
    h_in: float
    fc_ksi: float
    bar_size: int
    top_bars: int
    bot_bars: int
    side_bars: int
    stirrup_bar_size: int
    stirrup_legs: int
    stirrup_spacing_in: float
    stirrup_legs_by_direction: tuple | None = None       # columns: hoop legs (across_b_face, across_h_face)

    @property
    def is_column(self):
        return self.member_type == "column"

    @property
    def bar_area_in2(self):
        return sp.rebar_area(self.bar_size)

    @property
    def gross_area_in2(self):
        return self.b_in * self.h_in

    @property
    def legs_by_direction(self):
        """Column hoop legs as the design record writes them, or None when no per-direction count was selected."""
        if self.stirrup_legs_by_direction is None:
            return None
        return dict(zip(LEG_DIRECTIONS, self.stirrup_legs_by_direction))

    @property
    def longitudinal_bars(self):
        """Total longitudinal bars: columns count each side face once per face pair."""
        return self.top_bars + self.bot_bars + 2 * self.side_bars

    @property
    def longitudinal_area_in2(self):
        return self.longitudinal_bars * self.bar_area_in2

    def section_key(self):
        """What makes two members share one section and material definition in a model."""
        return (self.member_type if self.is_column else "beam", self.b_in, self.h_in, self.fc_ksi, self.bar_size, self.top_bars,
                self.bot_bars, self.side_bars, self.stirrup_bar_size, self.stirrup_legs, self.stirrup_spacing_in,
                self.stirrup_legs_by_direction)

    def form_key(self):
        """Formwork size: the concrete outline only."""
        return ("column" if self.is_column else "beam", self.b_in, self.h_in)


LEG_DIRECTIONS = ("across_b_face", "across_h_face")
_DESIGN_FIELDS = ("b_in", "h_in", "fc_ksi", "bar_size", "top_bars", "bot_bars", "side_bars", "stirrup_bar_size", "stirrup_legs",
                  "stirrup_spacing_in")


def design_problems(design, member_type):
    problems = []
    if not isinstance(design, MemberDesign):
        return [f"not a MemberDesign: {design!r}"]
    if design.member_type != member_type:
        problems.append(f"design is for {design.member_type!r}, the group holds {member_type!r} members")
    for name in ("b_in", "h_in", "fc_ksi", "stirrup_spacing_in"):
        value = getattr(design, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            problems.append(f"{name} = {value!r} is not a finite positive number")
    for name, lower in (("bar_size", 3), ("top_bars", 2), ("bot_bars", 2), ("side_bars", 0), ("stirrup_bar_size", 3), ("stirrup_legs", 2)):
        value = getattr(design, name)
        if type(value) is not int or value < lower:
            problems.append(f"{name} = {value!r} must be an integer of at least {lower}")
    if design.is_column:
        legs = design.stirrup_legs_by_direction
        if legs is not None and (len(legs) != 2 or any(type(v) is not int or v < 2 for v in legs)):
            problems.append(f"stirrup_legs_by_direction = {legs!r} must be two integers of at least 2")
    elif design.stirrup_legs_by_direction is not None:
        problems.append("stirrup_legs_by_direction applies to columns only")
    return problems


def uniform_member_design(member_type):
    """The explicit uniform adapter: the one column or beam design held in Structure_Parameters."""
    if member_type == "column":
        legs = sp.COL_STIRRUP_LEGS_BY_DIRECTION
        return MemberDesign("column", sp.B_COL, sp.H_COL, sp.FC_COL_KSI, sp.COL_BAR_SIZE, sp.COL_TOP_BARS, sp.COL_BOT_BARS,
                            sp.COL_SIDE_BARS, sp.COL_STIRRUP_BAR_SIZE, sp.COL_STIRRUP_LEGS, sp.COL_STIRRUP_SPACING,
                            None if legs is None else _legs_tuple(legs))
    if member_type in ("beam_x", "beam_y"):
        return MemberDesign(member_type, sp.B_BEAM, sp.H_BEAM, sp.FC_BEAM_KSI, sp.BEAM_BAR_SIZE, sp.BEAM_TOP_BARS, sp.BEAM_BOT_BARS,
                            sp.BEAM_SIDE_BARS, sp.BEAM_STIRRUP_BAR_SIZE, sp.BEAM_STIRRUP_LEGS, sp.BEAM_STIRRUP_SPACING)
    raise ValueError(f"Unknown member type {member_type!r}.")


@dataclass(frozen=True)
class ResolvedMember:
    """One physical member with its position, its group and the design that group selected."""
    member_tag: int
    member_type: str
    story_or_floor: int
    grid_i: int
    grid_j: int
    node_i: int
    node_j: int
    group_id: str
    location_class: str
    band: str
    design: MemberDesign

    @property
    def is_column(self):
        return self.member_type == "column"


class GroupedDesign:
    """A membership map with the design each group selected. Immutable in use: ``with_designs`` returns a new one."""

    def __init__(self, assignments, designs):
        counts = (assignments or {}).get("counts") or {}
        problems = assignment_problems(assignments, counts.get("num_bay_x"), counts.get("num_bay_y"), counts.get("num_floor")) \
            if all(type(counts.get(k)) is int for k in ("num_bay_x", "num_bay_y", "num_floor")) else ["counts are missing"]
        if problems:
            raise GroupedStateError("Member assignments are not valid: " + "; ".join(problems[:6]))
        self.assignments = assignments
        self.groups = {g["group_id"]: g for g in assignments["groups"]}
        missing = sorted(set(self.groups) - set(designs))
        extra = sorted(set(designs) - set(self.groups))
        if missing or extra:
            raise GroupedStateError(f"Group designs do not cover the groups exactly: missing {missing[:6]}, unknown {extra[:6]}")
        bad = []
        for gid, group in self.groups.items():
            bad += [f"{gid}: {p}" for p in design_problems(designs[gid], group["member_type"])]
        if bad:
            raise GroupedStateError("Group designs are not valid: " + "; ".join(bad[:6]))
        self.designs = dict(designs)
        self._members = {row["member_tag"]: row for row in assignments["members"]}

    @property
    def counts(self):
        return self.assignments["counts"]

    def member(self, member_tag):
        row = self._members.get(int(member_tag))
        if row is None:
            raise GroupedStateError(f"Member {member_tag!r} has no assignment in the installed grouped design.")
        return ResolvedMember(design=self.designs[row["group_id"]], **{k: row[k] for k in (
            "member_tag", "member_type", "story_or_floor", "grid_i", "grid_j", "node_i", "node_j", "group_id", "location_class", "band")})

    def members(self, member_type=None):
        return [self.member(tag) for tag, row in self._members.items() if member_type is None or row["member_type"] == member_type]

    def with_designs(self, updates):
        """A new grouped design with some groups changed; the receiver is untouched (trial and rollback)."""
        unknown = sorted(set(updates) - set(self.groups))
        if unknown:
            raise GroupedStateError(f"Unknown groups {unknown[:6]}")
        return GroupedDesign(self.assignments, {**self.designs, **updates})

    def is_uniform(self):
        """True when every column group and every beam group selected the same design (the equivalence fixture)."""
        keys = {}
        for gid, group in self.groups.items():
            keys.setdefault("column" if group["member_type"] == "column" else "beam", set()).add(
                replace(self.designs[gid], member_type="x").section_key())
        return all(len(v) == 1 for v in keys.values())

    def identity(self):
        payload = {"assignments_sha256": self.assignments["sha256"], "policy": POLICY_ID,
                   "designs": {gid: _design_dict(self.designs[gid]) for gid in sorted(self.designs)}}
        return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()

    def to_record(self):
        """The grouped block of a design record: policy, explicit assignments, group designs, per-member geometry."""
        return {"schema": SCHEMA, "policy": POLICY, "assignments": self.assignments,
                "designs": {gid: _design_dict(self.designs[gid]) for gid in sorted(self.designs)},
                "members": [{"member_tag": m.member_tag, "member_type": m.member_type, "story_or_floor": m.story_or_floor,
                             "grid_i": m.grid_i, "grid_j": m.grid_j, "node_i": m.node_i, "node_j": m.node_j,
                             "group_id": m.group_id, **_design_dict(m.design)} for m in self.members()],
                "distinct_form_sizes": self.distinct_form_sizes(), "uniform": self.is_uniform(), "sha256": self.identity()}

    def distinct_form_sizes(self):
        forms = {"column": set(), "beam": set()}
        for design in self.designs.values():
            forms[design.form_key()[0]].add(design.form_key()[1:])
        return {kind: sorted(list(v) for v in values) for kind, values in forms.items()}

    @classmethod
    def from_record(cls, block):
        if not isinstance(block, dict) or block.get("schema") != SCHEMA:
            raise GroupedStateError(f"Not a grouped design block (schema {None if not isinstance(block, dict) else block.get('schema')!r}).")
        designs = {gid: _design_from_dict(values) for gid, values in (block.get("designs") or {}).items()}
        state = cls(block.get("assignments"), designs)
        if block.get("sha256") != state.identity():
            raise GroupedStateError("The grouped design block does not match its digest; it was edited after it was written.")
        saved = {row.get("member_tag"): row for row in block.get("members") or []}
        for member in state.members():
            row = saved.get(member.member_tag)
            if row is None or any(row.get(k) != v for k, v in {"group_id": member.group_id, **_design_dict(member.design)}.items()):
                raise GroupedStateError(f"The saved per-member row of member {member.member_tag} disagrees with its group's design.")
        return state

    @classmethod
    def uniform(cls, nx=None, ny=None, nz=None, column=None, beam=None):
        """Every group with the same column and beam design: the uniform record expanded into groups.

        An equivalence fixture and the seed of a grouped search; it prescribes nothing for that search.
        Defaults are the counts and the single designs of Structure_Parameters (read before any install).
        """
        nx = sp.NUM_BAY_X if nx is None else nx
        ny = sp.NUM_BAY_Y if ny is None else ny
        nz = sp.NUM_FLOOR if nz is None else nz
        column = column or uniform_member_design("column")
        beam = beam or uniform_member_design("beam_x")
        assignments = build_assignments(nx, ny, nz)
        designs = {g["group_id"]: (column if g["member_type"] == "column" else replace(beam, member_type=g["member_type"]))
                   for g in assignments["groups"]}
        return cls(assignments, designs)


def _legs_tuple(legs):
    """(across_b_face, across_h_face) from the record's dict form."""
    if not isinstance(legs, dict) or set(legs) != set(LEG_DIRECTIONS):
        raise GroupedStateError(f"Column hoop legs by direction must name {LEG_DIRECTIONS}, got {legs!r}.")
    return tuple(legs[direction] for direction in LEG_DIRECTIONS)


def _design_dict(design):
    values = asdict(design)
    values["stirrup_legs_by_direction"] = design.legs_by_direction
    return values


def design_record(design):
    """A MemberDesign as the plain mapping records, registries and exports carry."""
    return _design_dict(design)


def _design_from_dict(values):
    values = dict(values)
    legs = values.get("stirrup_legs_by_direction")
    values["stirrup_legs_by_direction"] = None if legs is None else _legs_tuple(legs)
    return MemberDesign(**values)


# ---- the installed state and the canonical resolver ---------------------------------------------------
_ACTIVE = {"state": None, "withdrawn": None}


def active():
    """The installed grouped design, or None in the uniform mode."""
    return _ACTIVE["state"]


def is_grouped():
    return _ACTIVE["state"] is not None


def install(state):
    """Put a grouped design in force. The uniform section and cage values of Structure_Parameters are
    withdrawn (any use raises) until ``clear``; the model counts must be the assignments' counts."""
    if not isinstance(state, GroupedDesign):
        raise GroupedStateError("install() takes a GroupedDesign.")
    counts = {"num_bay_x": sp.NUM_BAY_X, "num_bay_y": sp.NUM_BAY_Y, "num_floor": sp.NUM_FLOOR}
    if state.counts != counts:
        raise GroupedStateError(f"The grouped design is for counts {state.counts}; the model has {counts}.")
    if _ACTIVE["state"] is None:
        _ACTIVE["withdrawn"] = {key: getattr(sp, key) for key in UNIFORM_SECTION_KEYS}
        for key in UNIFORM_SECTION_KEYS:
            setattr(sp, key, _UniformValueWithdrawn(key))
    sp.UNIFORM_SECTIONS_WITHDRAWN = True
    _ACTIVE["state"] = state
    return state


def clear():
    """Leave the grouped mode and give the uniform values back as they were before ``install``."""
    if _ACTIVE["state"] is not None:
        for key, value in (_ACTIVE["withdrawn"] or {}).items():
            setattr(sp, key, value)
    sp.UNIFORM_SECTIONS_WITHDRAWN = False
    _ACTIVE["state"], _ACTIVE["withdrawn"] = None, None


@contextlib.contextmanager
def installed(state):
    """Put ``state`` in force for a block and restore what was installed before (a trial design, or none)."""
    previous = _ACTIVE["state"]
    install(state)
    try:
        yield state
    finally:
        if previous is None:
            clear()
        else:
            install(previous)


def member_position(member_tag):
    """Type and integer grid position of a physical member from the model counts (both modes)."""
    nx, ny, nz = sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR
    tag = int(member_tag)
    n_col, n_bx, n_by = nz * (nx + 1) * (ny + 1), nz * nx * (ny + 1), nz * (nx + 1) * ny
    if not 1 <= tag <= n_col + n_bx + n_by:
        raise GroupedStateError(f"Member tag {member_tag!r} is not a physical member of a {nx} x {ny} x {nz} frame.")
    if tag <= n_col:
        k, rest = divmod(tag - 1, (nx + 1) * (ny + 1))
        j, i = divmod(rest, nx + 1)
        return "column", k + 1, i, j
    if tag <= n_col + n_bx:
        k, rest = divmod(tag - n_col - 1, nx * (ny + 1))
        j, i = divmod(rest, nx)
        return "beam_x", k + 1, i, j
    k, rest = divmod(tag - n_col - n_bx - 1, (nx + 1) * ny)
    j, i = divmod(rest, nx + 1)
    return "beam_y", k + 1, i, j


def resolve(member_tag):
    """THE resolver: physical member -> group -> selected section, material and reinforcement.

    Grouped mode: from the installed design; an unassigned member raises. Uniform mode: the member's
    position and group are those the policy defines, and its design is the single column or beam
    design of Structure_Parameters (the explicit uniform adapter).
    """
    state = _ACTIVE["state"]
    if state is not None:
        return state.member(member_tag)
    member_type, k, i, j = member_position(member_tag)
    nx, ny, nz = sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR
    per_level = (nx + 1) * (ny + 1)
    node = lambda level, a, b: level * per_level + b * (nx + 1) + a + 1          # noqa: E731
    band = next(b for b in story_bands(nz) if k in b)
    if member_type == "column":
        location, node_i, node_j = column_location(i, j, nx, ny), node(k - 1, i, j), node(k, i, j)
    elif member_type == "beam_x":
        location, node_i, node_j = beam_location("beam_x", i, j, nx, ny), node(k, i, j), node(k, i + 1, j)
    else:
        location, node_i, node_j = beam_location("beam_y", i, j, nx, ny), node(k, i, j), node(k, i, j + 1)
    return ResolvedMember(int(member_tag), member_type, k, i, j, node_i, node_j, group_id(band, member_type, location), location,
                          band_label(band), uniform_member_design(member_type))


def design_of(member_tag):
    return resolve(member_tag).design


def all_members(member_type=None):
    """Every physical member resolved, in tag order (both modes)."""
    nx, ny, nz = sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR
    total = nz * ((nx + 1) * (ny + 1) + nx * (ny + 1) + (nx + 1) * ny)
    members = [resolve(tag) for tag in range(1, total + 1)]
    return members if member_type is None else [m for m in members if m.member_type == member_type]


def column_at(story, i, j):
    """The column of story ``story`` (segment story-1 to story) at grid (i, j)."""
    nx, ny, nz = sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR
    if not (1 <= story <= nz and 0 <= i <= nx and 0 <= j <= ny):
        raise GroupedStateError(f"No column at story {story}, grid ({i}, {j}).")
    return resolve((story - 1) * (nx + 1) * (ny + 1) + j * (nx + 1) + i + 1)


def beam_at(member_type, floor, i, j):
    """The beam of ``member_type`` at floor ``floor`` starting at grid (i, j)."""
    nx, ny, nz = sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR
    n_col = nz * (nx + 1) * (ny + 1)
    if member_type == "beam_x":
        if not (1 <= floor <= nz and 0 <= i < nx and 0 <= j <= ny):
            raise GroupedStateError(f"No X beam at floor {floor}, grid ({i}, {j}).")
        return resolve(n_col + (floor - 1) * nx * (ny + 1) + j * nx + i + 1)
    if member_type == "beam_y":
        if not (1 <= floor <= nz and 0 <= i <= nx and 0 <= j < ny):
            raise GroupedStateError(f"No Y beam at floor {floor}, grid ({i}, {j}).")
        return resolve(n_col + nz * nx * (ny + 1) + (floor - 1) * (nx + 1) * ny + j * (nx + 1) + i + 1)
    raise ValueError(f"{member_type!r} is not a beam type.")


def joint_members(level, i, j):
    """The members that actually frame into the joint at ``level`` (0 is the base), grid (i, j).

    Returns columns below and above and the X and Y beams on each side, each a ResolvedMember or None.
    At a band boundary the column below and the column above belong to different groups; at the roof
    there is no column above; at the base there are no beams.
    """
    nx, ny, nz = sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR
    if not (0 <= level <= nz and 0 <= i <= nx and 0 <= j <= ny):
        raise GroupedStateError(f"No joint at level {level}, grid ({i}, {j}).")
    joint = {"level": level, "grid_i": i, "grid_j": j,
             "column_below": column_at(level, i, j) if level >= 1 else None,
             "column_above": column_at(level + 1, i, j) if level < nz else None,
             "beam_x_minus": None, "beam_x_plus": None, "beam_y_minus": None, "beam_y_plus": None}
    if level >= 1:
        if i > 0:
            joint["beam_x_minus"] = beam_at("beam_x", level, i - 1, j)
        if i < nx:
            joint["beam_x_plus"] = beam_at("beam_x", level, i, j)
        if j > 0:
            joint["beam_y_minus"] = beam_at("beam_y", level, i, j - 1)
        if j < ny:
            joint["beam_y_plus"] = beam_at("beam_y", level, i, j)
    return joint
