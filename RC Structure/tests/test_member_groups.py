"""Design groups and the canonical member resolver (Model/Member_Groups, grouped iterative design 2026-10-02).

Stage A of the grouped design: membership, identity, the two modes, and the uniform design expanded
into groups. Bookkeeping only: nothing here designs or analyses a member.
"""
import contextlib
import copy
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

RC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_DIR))

import Structure_Parameters as sp                                              # noqa: E402
from Model import Member_Groups as mg                                           # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "reference_group_assignments.json"
ROW_KEYS = ("member_tag", "member_type", "story_or_floor", "grid_i", "grid_j", "node_i", "node_j", "group_id", "location_class")


@contextlib.contextmanager
def counts(nx, ny, nz):
    """The model counts for a block, with every Structure_Parameters name restored and the grouped mode left."""
    snapshot = dict(vars(sp))
    try:
        sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR = nx, ny, nz
        yield
    finally:
        mg.clear()
        for key in set(vars(sp)) - set(snapshot):
            delattr(sp, key)
        vars(sp).update(snapshot)


class Membership(unittest.TestCase):
    def test_reference_membership_matches_the_fixture_member_for_member(self):
        reference = json.loads(FIXTURE.read_text(encoding="utf-8"))
        ours = mg.build_assignments(2, 6, 6)
        self.assertEqual(len(ours["members"]), 318)
        self.assertEqual(len(ours["groups"]), 24)
        self.assertEqual(ours["story_bands"], reference["story_bands"])
        for mine, theirs in zip(ours["members"], reference["members"]):
            self.assertEqual({k: mine[k] for k in ROW_KEYS}, {k: theirs[k] for k in ROW_KEYS})
        keys = ("group_id", "member_type", "stories_or_floors", "location_class", "member_count", "member_tags")
        self.assertEqual([{k: g[k] for k in keys} for g in ours["groups"]], [{k: g[k] for k in keys} for g in reference["groups"]])
        self.assertEqual({g["group_id"]: g["member_count"] for g in ours["groups"]}["s01_02__column__x_boundary_edge"], 20)

    def test_story_bands_pair_from_the_base_and_merge_an_orphan_story(self):
        self.assertEqual(mg.story_bands(4), [[1, 2], [3, 4]])
        self.assertEqual(mg.story_bands(5), [[1, 2], [3, 4, 5]])
        self.assertEqual(mg.story_bands(6), [[1, 2], [3, 4], [5, 6]])
        self.assertEqual(mg.story_bands(9), [[1, 2], [3, 4], [5, 6], [7, 8, 9]])
        self.assertEqual(mg.story_bands(1), [[1]])
        self.assertEqual(mg.band_label([3, 4, 5]), "s03_05")

    def test_every_population_count_is_covered_exactly_once_with_mirrored_groups(self):
        for nx in range(2, 7):
            for ny in range(2, 7):
                for nz in range(4, 10):
                    rows = mg.member_inventory(nx, ny, nz)
                    expected = nz * ((nx + 1) * (ny + 1) + nx * (ny + 1) + (nx + 1) * ny)
                    self.assertEqual([r["member_tag"] for r in rows], list(range(1, expected + 1)))
                    lookup = {(r["member_type"], r["story_or_floor"], r["grid_i"], r["grid_j"]): r["group_id"] for r in rows}
                    self.assertEqual(len(lookup), expected)
                    for r in rows:
                        kind, k, i, j = r["member_type"], r["story_or_floor"], r["grid_i"], r["grid_j"]
                        self.assertEqual(lookup[kind, k, nx - i - (kind == "beam_x"), j], r["group_id"])      # mirror in X
                        self.assertEqual(lookup[kind, k, i, ny - j - (kind == "beam_y")], r["group_id"])      # mirror in Y
                    bands = mg.story_bands(nz)
                    self.assertEqual(len({r["group_id"] for r in rows}), 8 * len(bands))          # four column and four beam groups per band
                    self.assertEqual(sorted(k for band in bands for k in band), list(range(1, nz + 1)))

    def test_column_story_is_the_segment_below_its_level_and_locations_use_integer_indices(self):
        rows = {r["member_tag"]: r for r in mg.member_inventory(2, 6, 6)}
        first, per_level = rows[1], 3 * 7
        self.assertEqual((first["member_type"], first["story_or_floor"], first["node_i"], first["node_j"]), ("column", 1, 1, 1 + per_level))
        self.assertEqual(rows[2]["location_class"], "y_boundary_edge")                 # i interior, j = 0
        self.assertEqual(rows[4]["location_class"], "x_boundary_edge")                 # i = 0, j interior
        self.assertEqual(rows[5]["location_class"], "interior")
        third_story = [r for r in rows.values() if r["member_type"] == "column" and r["story_or_floor"] == 3]
        self.assertEqual({r["band"] for r in third_story}, {"s03_04"})
        self.assertEqual(mg.column_location(0, 0, 2, 6), "corner")
        self.assertEqual(mg.beam_location("beam_x", 0, 6, 2, 6), "edge")
        self.assertEqual(mg.beam_location("beam_y", 1, 0, 2, 6), "interior")
        for bad in ((0, 2, 4), (2, 2.0, 4), (2, 2, True)):
            with self.assertRaises(ValueError):
                mg.member_inventory(*bad)

    def test_missing_duplicate_wrong_type_and_inconsistent_assignments_are_refused(self):
        good = mg.build_assignments(2, 2, 4)
        self.assertEqual(mg.assignment_problems(good, 2, 2, 4), [])

        def problems(change, counts=(2, 2, 4)):
            bad = copy.deepcopy(good)
            change(bad)
            return mg.assignment_problems(bad, *counts)

        self.assertTrue(any("no assignment" in p for p in problems(lambda a: a["members"].pop(7))))
        self.assertTrue(any("duplicate" in p for p in problems(lambda a: a["members"].append(dict(a["members"][3])))))
        self.assertTrue(any("disagree" in p for p in problems(lambda a: a["members"][0].update(member_type="beam_x"))))
        self.assertTrue(any("disagree" in p for p in problems(lambda a: a["members"][0].update(group_id="s01_02__column__interior"))))
        self.assertTrue(any("disagree" in p for p in problems(lambda a: a["members"][10].update(grid_i=2))))
        self.assertTrue(any("disagree" in p for p in problems(lambda a: a["members"][10].update(node_j=999))))
        self.assertTrue(any("does not have" in p for p in problems(lambda a: a["members"].append({**a["members"][0], "member_tag": 9999}))))
        self.assertTrue(any("lists other members" in p for p in problems(lambda a: a["groups"][0]["member_tags"].pop())))
        self.assertTrue(any("is missing" in p for p in problems(lambda a: a["groups"].pop())))
        self.assertTrue(any("policy" in p for p in problems(lambda a: a["policy"].update(id="other"))))
        self.assertTrue(any("schema" in p for p in problems(lambda a: a.update(schema="v0"))))
        self.assertTrue(any("digest" in p for p in problems(lambda a: a.update(sha256="0" * 64))))
        self.assertTrue(mg.assignment_problems(good, 2, 3, 4))                                 # another building's map
        self.assertTrue(mg.assignment_problems(None, 2, 2, 4))


class Designs(unittest.TestCase):
    COLUMN = mg.MemberDesign("column", 30.0, 30.0, 5.0, 9, 4, 4, 2, 4, 4, 4.0, (4, 4))
    BEAM = mg.MemberDesign("beam_x", 20.0, 28.0, 4.0, 8, 4, 3, 0, 4, 2, 5.0)

    def test_uniform_expansion_puts_one_design_in_every_group_and_prescribes_nothing_else(self):
        state = mg.GroupedDesign.uniform(2, 6, 6, column=self.COLUMN, beam=self.BEAM)
        self.assertTrue(state.is_uniform())
        self.assertEqual(len(state.designs), 24)
        for member in state.members():
            wanted = self.COLUMN if member.is_column else replace(self.BEAM, member_type=member.member_type)
            self.assertEqual(member.design, wanted)
        self.assertEqual(state.distinct_form_sizes(), {"column": [[30.0, 30.0]], "beam": [[20.0, 28.0]]})
        changed = state.with_designs({"s05_06__column__interior": replace(self.COLUMN, b_in=24.0, h_in=24.0)})
        self.assertFalse(changed.is_uniform())
        self.assertTrue(state.is_uniform())                                                    # the seed is untouched
        self.assertNotEqual(changed.identity(), state.identity())
        self.assertEqual(changed.distinct_form_sizes()["column"], [[24.0, 24.0], [30.0, 30.0]])
        tags = {m.member_tag for m in changed.members() if m.design.b_in == 24.0}
        self.assertEqual(tags, set(changed.groups["s05_06__column__interior"]["member_tags"]))  # only that group's members
        self.assertEqual(len(tags), 10)

    def test_group_designs_must_cover_the_groups_exactly_and_be_of_the_right_kind(self):
        state = mg.GroupedDesign.uniform(2, 2, 4, column=self.COLUMN, beam=self.BEAM)
        designs = dict(state.designs)
        gid = "s01_02__beam_y__edge"
        for label, broken in (("missing", {k: v for k, v in designs.items() if k != gid}),
                              ("unknown", {**designs, "s09_10__column__corner": self.COLUMN}),
                              ("wrong kind", {**designs, gid: self.COLUMN}),
                              ("wrong axis", {**designs, gid: self.BEAM}),
                              ("zero width", {**designs, gid: replace(self.BEAM, member_type="beam_y", b_in=0.0)}),
                              ("one bar", {**designs, gid: replace(self.BEAM, member_type="beam_y", top_bars=1)}),
                              ("float count", {**designs, gid: replace(self.BEAM, member_type="beam_y", bot_bars=3.0)}),
                              ("beam with column legs", {**designs, gid: replace(self.BEAM, member_type="beam_y", stirrup_legs_by_direction=(2, 2))})):
            with self.assertRaises(mg.GroupedStateError, msg=label):
                mg.GroupedDesign(state.assignments, broken)
        with self.assertRaises(mg.GroupedStateError):
            state.with_designs({"no_such_group": self.COLUMN})
        with self.assertRaises(mg.GroupedStateError):
            mg.GroupedDesign({**state.assignments, "members": state.assignments["members"][:-1]}, designs)

    def test_record_block_round_trips_and_an_edited_block_is_refused(self):
        state = mg.GroupedDesign.uniform(2, 2, 4, column=self.COLUMN, beam=self.BEAM).with_designs(
            {"s03_04__column__corner": replace(self.COLUMN, b_in=26.0, h_in=26.0, bar_size=8)})
        block = json.loads(json.dumps(state.to_record(), allow_nan=False))
        self.assertEqual(block["schema"], mg.SCHEMA)
        self.assertEqual(block["policy"]["id"], mg.POLICY_ID)
        self.assertEqual(len(block["members"]), len(state.assignments["members"]))
        self.assertFalse(block["uniform"])
        row = next(r for r in block["members"] if r["group_id"] == "s03_04__column__corner")
        self.assertEqual((row["b_in"], row["bar_size"], row["stirrup_legs_by_direction"]), (26.0, 8, {"across_b_face": 4, "across_h_face": 4}))
        restored = mg.GroupedDesign.from_record(block)
        self.assertEqual(restored.identity(), state.identity())
        self.assertEqual(restored.designs, state.designs)
        for label, change in (("design edited", lambda b: b["designs"]["s01_02__column__corner"].update(b_in=40.0)),
                              ("member row edited", lambda b: b["members"][0].update(b_in=40.0)),
                              ("schema relabelled", lambda b: b.update(schema="other")),
                              ("member row dropped", lambda b: b["members"].pop())):
            edited = copy.deepcopy(block)
            change(edited)
            with self.assertRaises(mg.GroupedStateError, msg=label):
                mg.GroupedDesign.from_record(edited)
        with self.assertRaises(mg.GroupedStateError):
            mg.GroupedDesign.from_record({"b_col_in": 30.0})                                    # a uniform block is not a grouped one


class Modes(unittest.TestCase):
    def test_uniform_mode_resolves_every_member_to_the_single_design_and_its_policy_group(self):
        with counts(2, 6, 6):
            self.assertFalse(mg.is_grouped())
            reference = {r["member_tag"]: r for r in mg.member_inventory(2, 6, 6)}
            members = mg.all_members()
            self.assertEqual(len(members), 318)
            for member in members:
                row = reference[member.member_tag]
                self.assertEqual({k: getattr(member, k) for k in ROW_KEYS}, {k: row[k] for k in ROW_KEYS})
                self.assertEqual(member.design, mg.uniform_member_design(member.member_type))
            self.assertEqual(mg.resolve(1).design.b_in, sp.B_COL)
            self.assertEqual(mg.resolve(318).design.h_in, sp.H_BEAM)
            self.assertEqual(mg.member_position(127), ("beam_x", 1, 0, 0))
            for bad in (0, 319, -1):
                with self.assertRaises(mg.GroupedStateError):
                    mg.resolve(bad)

    def test_grouped_mode_withdraws_the_uniform_values_and_restores_them_on_clear(self):
        with counts(2, 2, 4):
            before = {key: getattr(sp, key) for key in mg.UNIFORM_SECTION_KEYS}
            state = mg.GroupedDesign.uniform().with_designs(
                {"s03_04__column__interior": replace(mg.uniform_member_design("column"), b_in=22.0, h_in=22.0)})
            mg.install(state)
            self.assertTrue(mg.is_grouped())
            self.assertIs(mg.active(), state)
            for use in (lambda: sp.B_COL * 2, lambda: float(sp.H_BEAM), lambda: sp.FC_COL_KSI > 4, lambda: f"{sp.B_BEAM:g}",
                        lambda: sp.COL_TOP_BARS + 1, lambda: sp.rect_area(sp.B_COL, sp.H_COL), lambda: int(sp.BEAM_BAR_SIZE),
                        lambda: bool(sp.COL_SIDE_BARS), lambda: sp.B_COL == 22.0, lambda: json.dumps({"b": float(sp.B_COL)})):
                with self.assertRaises(mg.GroupedStateError):
                    use()
            with self.assertRaises(mg.GroupedStateError):
                mg.uniform_member_design("column").b_in * 1.0                                   # the uniform adapter has nothing to adapt
            interior_upper = mg.column_at(3, 1, 1)
            self.assertEqual((interior_upper.group_id, interior_upper.design.b_in), ("s03_04__column__interior", 22.0))
            self.assertEqual(mg.column_at(2, 1, 1).design.b_in, before["B_COL"])
            self.assertEqual(mg.column_at(3, 0, 1).design.b_in, before["B_COL"])                # another location, same band
            mg.clear()
            self.assertFalse(mg.is_grouped())
            self.assertEqual({key: getattr(sp, key) for key in mg.UNIFORM_SECTION_KEYS}, before)
            self.assertEqual(mg.column_at(3, 1, 1).design.b_in, before["B_COL"])

    def test_a_grouped_design_of_other_counts_is_not_installed(self):
        with counts(2, 2, 4):
            other = mg.GroupedDesign.uniform(2, 3, 4)
            with self.assertRaises(mg.GroupedStateError):
                mg.install(other)
            self.assertFalse(mg.is_grouped())
            self.assertEqual(sp.B_COL * 1.0, sp.B_COL)                                          # nothing was withdrawn
            with self.assertRaises(mg.GroupedStateError):
                mg.install({"designs": {}})

    def test_joint_members_are_the_actual_incident_members(self):
        with counts(2, 6, 6):
            mg.install(mg.GroupedDesign.uniform())
            boundary = mg.joint_members(2, 1, 3)                                                # top of band 1-2, interior
            self.assertEqual((boundary["column_below"].group_id, boundary["column_above"].group_id),
                             ("s01_02__column__interior", "s03_04__column__interior"))
            self.assertEqual((boundary["column_below"].story_or_floor, boundary["column_above"].story_or_floor), (2, 3))
            self.assertEqual({k: boundary[k].group_id for k in ("beam_x_minus", "beam_x_plus", "beam_y_minus", "beam_y_plus")},
                             {"beam_x_minus": "s01_02__beam_x__interior", "beam_x_plus": "s01_02__beam_x__interior",
                              "beam_y_minus": "s01_02__beam_y__interior", "beam_y_plus": "s01_02__beam_y__interior"})
            self.assertEqual(boundary["beam_x_minus"].node_j, boundary["column_below"].node_j)   # they meet at the joint node
            self.assertEqual(boundary["beam_x_plus"].node_i, boundary["column_above"].node_i)
            roof = mg.joint_members(6, 0, 0)                                                    # roof corner
            self.assertIsNone(roof["column_above"])
            self.assertEqual(roof["column_below"].group_id, "s05_06__column__corner")
            self.assertIsNone(roof["beam_x_minus"]); self.assertIsNone(roof["beam_y_minus"])
            self.assertEqual((roof["beam_x_plus"].group_id, roof["beam_y_plus"].group_id), ("s05_06__beam_x__edge", "s05_06__beam_y__edge"))
            edge = mg.joint_members(4, 0, 3)                                                    # an X-boundary edge joint at a band top
            self.assertEqual((edge["column_below"].group_id, edge["column_above"].group_id),
                             ("s03_04__column__x_boundary_edge", "s05_06__column__x_boundary_edge"))
            self.assertEqual((edge["beam_x_plus"].group_id, edge["beam_y_plus"].group_id), ("s03_04__beam_x__interior", "s03_04__beam_y__edge"))
            base = mg.joint_members(0, 1, 1)
            self.assertIsNone(base["column_below"]); self.assertIsNone(base["beam_x_plus"])
            self.assertEqual(base["column_above"].story_or_floor, 1)
            with self.assertRaises(mg.GroupedStateError):
                mg.joint_members(7, 0, 0)
            # every member end appears at exactly one joint
            seen = {}
            for level in range(0, 7):
                for j in range(7):
                    for i in range(3):
                        joint = mg.joint_members(level, i, j)
                        for key, member in joint.items():
                            if hasattr(member, "member_tag"):
                                seen.setdefault(member.member_tag, []).append(key)
            self.assertEqual(sorted(seen), list(range(1, 319)))
            self.assertTrue(all(len(v) == 2 for v in seen.values()))


if __name__ == "__main__":
    unittest.main()
