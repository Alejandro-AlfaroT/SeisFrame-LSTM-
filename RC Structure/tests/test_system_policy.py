"""The system_uniform_v1 grouping policy (2026-10-16): one band, the location groups, four design ties."""
import unittest
from dataclasses import replace

import Structure_Parameters as sp
from Model import Member_Groups as mg

COLUMN = mg.MemberDesign("column", 24.0, 24.0, 6.0, 8, 4, 4, 2, 4, 4, 4.0, (4, 4))
BEAM = mg.MemberDesign("beam_x", 14.0, 24.0, 5.0, 7, 3, 3, 0, 4, 2, 5.0)
LIGHT_COLUMN = replace(COLUMN, b_in=18.0, h_in=18.0, bar_size=7, side_bars=1)
LIGHT_BEAM = replace(BEAM, b_in=12.0, h_in=20.0, top_bars=2, bot_bars=2)


class Assignments(unittest.TestCase):
    def test_one_band_eight_groups_with_systems_and_ties(self):
        assignments = mg.build_assignments(3, 4, 5, mg.SYSTEM_POLICY_ID)
        self.assertEqual(assignments["policy"]["id"], mg.SYSTEM_POLICY_ID)
        self.assertEqual(assignments["story_bands"], [[1, 2, 3, 4, 5]])
        groups = {g["group_id"]: g for g in assignments["groups"]}
        self.assertEqual(len(groups), 8)
        self.assertTrue(all(gid.startswith("s01_05__") for gid in groups))
        self.assertEqual(groups["s01_05__column__corner"]["design_tie"], "sfrs_column")
        self.assertEqual(groups["s01_05__column__x_boundary_edge"]["design_tie"], "sfrs_column")
        self.assertEqual(groups["s01_05__column__interior"]["design_tie"], "gravity_column")
        self.assertEqual(groups["s01_05__beam_x__edge"]["design_tie"], "sfrs_beam")
        self.assertEqual(groups["s01_05__beam_y__interior"]["design_tie"], "gravity_beam")
        self.assertEqual(groups["s01_05__column__interior"]["system"], mg.SYSTEM_GRAVITY)
        self.assertEqual(groups["s01_05__beam_y__edge"]["system"], mg.SYSTEM_SFRS)
        # every member carries its system; the counts are those of the plan
        members = assignments["members"]
        self.assertEqual(sum(1 for m in members if m["member_type"] == "column" and m["system"] == mg.SYSTEM_GRAVITY), 5 * 6)
        self.assertEqual(sum(1 for m in members if m["member_type"] == "beam_x" and m["system"] == mg.SYSTEM_SFRS), 5 * 6)
        self.assertEqual(mg.assignment_problems(assignments, 3, 4, 5), [])
        # the band policy is unchanged and distinct
        band = mg.build_assignments(3, 4, 5)
        self.assertEqual(band["policy"]["id"], mg.POLICY_ID)
        self.assertEqual(band["story_bands"], [[1, 2], [3, 4, 5]])
        self.assertNotEqual(band["sha256"], assignments["sha256"])
        self.assertTrue(all("design_tie" not in g for g in band["groups"]))
        self.assertEqual(mg.assignment_problems(band, 3, 4, 5), [])

    def test_a_policy_mismatch_is_a_problem(self):
        assignments = mg.build_assignments(3, 3, 4, mg.SYSTEM_POLICY_ID)
        edited = {**assignments, "policy": {**assignments["policy"], "id": "something_else"}}
        self.assertTrue(any("grouping policy" in p for p in mg.assignment_problems(edited, 3, 3, 4)))


class TiedDesigns(unittest.TestCase):
    def setUp(self):
        self.saved = {k: getattr(sp, k) for k in ("NUM_BAY_X", "NUM_BAY_Y", "NUM_FLOOR")}
        sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR = 3, 3, 4

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(sp, k, v)

    def test_uniform_seed_and_tie_updates(self):
        state = mg.GroupedDesign.uniform(column=COLUMN, beam=BEAM, policy_id=mg.SYSTEM_POLICY_ID)
        self.assertEqual(state.policy_id, mg.SYSTEM_POLICY_ID)
        self.assertEqual(sorted(state.ties()), ["gravity_beam", "gravity_column", "sfrs_beam", "sfrs_column"])
        self.assertEqual(len(state.ties()["sfrs_column"]), 3)
        self.assertEqual(len(state.ties()["sfrs_beam"]), 2)
        self.assertEqual(state.system_of("s01_04__beam_x__interior"), mg.SYSTEM_GRAVITY)
        self.assertEqual(state.tie_of("s01_04__column__y_boundary_edge"), "sfrs_column")
        lighter = state.with_tied_designs({"gravity_column": LIGHT_COLUMN, "gravity_beam": LIGHT_BEAM})
        self.assertEqual(lighter.designs["s01_04__column__interior"].b_in, 18.0)
        self.assertEqual(lighter.designs["s01_04__column__corner"].b_in, 24.0)
        self.assertEqual(lighter.designs["s01_04__beam_y__interior"].member_type, "beam_y")
        self.assertEqual(lighter.designs["s01_04__beam_y__interior"].h_in, 20.0)
        self.assertEqual(lighter.designs["s01_04__beam_x__edge"].h_in, 24.0)
        designs = lighter.tie_designs()
        self.assertEqual(designs["gravity_beam"].h_in, 20.0)
        self.assertNotEqual(state.identity(), lighter.identity())
        self.assertFalse(lighter.is_uniform())
        self.assertTrue(state.is_uniform())
        # the record round trip keeps the policy
        block = lighter.to_record()
        self.assertEqual(block["policy"]["id"], mg.SYSTEM_POLICY_ID)
        self.assertEqual(mg.GroupedDesign.from_record(block).identity(), lighter.identity())

    def test_a_tie_holding_two_designs_is_refused(self):
        state = mg.GroupedDesign.uniform(column=COLUMN, beam=BEAM, policy_id=mg.SYSTEM_POLICY_ID)
        with self.assertRaisesRegex(mg.GroupedStateError, "design tie"):
            state.with_designs({"s01_04__column__corner": LIGHT_COLUMN})
        with self.assertRaises(mg.GroupedStateError):
            state.with_tied_designs({"no_such_tie": LIGHT_COLUMN})

    def test_the_band_policy_has_no_ties(self):
        state = mg.GroupedDesign.uniform(column=COLUMN, beam=BEAM)
        self.assertEqual(state.ties(), {})
        self.assertIsNone(state.tie_of("s01_02__column__corner"))
        changed = state.with_designs({"s01_02__column__corner": LIGHT_COLUMN})       # bands may differ freely
        self.assertEqual(changed.designs["s01_02__column__corner"].b_in, 18.0)


if __name__ == "__main__":
    unittest.main()
