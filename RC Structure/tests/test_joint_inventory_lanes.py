"""Beam lanes from the joint's bar inventory at a column transition (2026-10-31; since the review of the same day,
joint_inventory_lanes_v2__swept_bend_intervals: each moved bar blocks its whole bend interval, test_cage_review_corrections).

  * at every joint without a moved bar the lanes are the core cage's, exactly, and the bar-row basis keeps its form;
  * a 2 in face step bends the column bars through the lanes: a 32 in core of twenty #10 passes three #9 per layer on
    its own and none through the joint's inventory, while a 1 in step keeps all three;
  * the bar rows, the selection evidence and the authoritative threading row (beam.bars_thread_column) read the same
    lanes, and the bar-row basis records the rule where it lowers a lane;
  * the column chain counts and cuts the cage pairs whose band-boundary joint closes the beam lanes, and the chain it
    selects from a pool of threading cages keeps them open.

Synthetic demands on the fixture frame; geometry and selection only.
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402
import test_beam_threading_evidence as tb                                        # noqa: E402

import Structure_Parameters as sp                                               # noqa: E402
from Design import Group_Capacity as gc                                          # noqa: E402
from Design import Group_Selection as selection                                  # noqa: E402
from Design import SMRF_Transitions as transitions                               # noqa: E402
from Design.Config import DesignConfig                                           # noqa: E402
from Model import Member_Bar_Layers as bar_layers                                # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402
from Model import Member_Properties as mp                                        # noqa: E402

LOWER = replace(gf.COLUMN, b_in=32.0, h_in=32.0, fc_ksi=8.0, bar_size=10, top_bars=6, bot_bars=6, side_bars=4)
BEAM = replace(gf.BEAM, b_in=22.0, h_in=36.0, fc_ksi=5.0, bar_size=9, top_bars=6, bot_bars=6)
SIX = {"NUM_FLOOR": 6, **tb.RULES_SEISMIC}
POLICY = "system_three_story_v1"


def two_bands(beam=BEAM, upper_size=28.0):
    """Six stories, bands of three: 32 in columns below, ``upper_size`` columns above with the same cage and grade."""
    base = mg.GroupedDesign.uniform(gf.NX, gf.NY, 6, LOWER, beam, policy_id=POLICY)
    return base.with_designs({g: replace(base.designs[g], b_in=upper_size, h_in=upper_size)
                              for g in base.groups if g.startswith("s04_06__column__")})


def every_joint_beam(state):
    for level in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                joint = mg.joint_members(level, i, j)
                for axis in ("x", "y"):
                    for side in ("minus", "plus"):
                        beam = joint[f"beam_{axis}_{side}"]
                        if beam is not None:
                            yield (level, i, j), axis, beam


class JointInventoryLanes(unittest.TestCase):
    def test_lanes_are_the_core_cages_wherever_no_bar_moves(self):
        for frame, factory in (({}, gf.changed_state), (SIX, lambda: mg.GroupedDesign.uniform(gf.NX, gf.NY, 6, LOWER, BEAM,
                                                                                                  policy_id=POLICY))):
            with gf.frame(**frame):
                state = mg.install(factory())
                checked = 0
                for joint, axis, beam in every_joint_beam(state):
                    if bar_layers.joint_inventory(*joint) is None:
                        core = mp.joint_core_column(*joint).design
                        self.assertEqual(bar_layers.bars_per_layer_at_joint(*joint, beam.design, axis),
                                         bar_layers.bars_per_layer(core, beam.design, axis))
                        checked += 1
                self.assertGreater(checked, 0)
        with gf.frame(**SIX):
            mg.install(mg.GroupedDesign.uniform(gf.NX, gf.NY, 6, LOWER, BEAM, policy_id=POLICY))
            self.assertNotIn("joint_inventory", bar_layers.arrangement()["_basis"])     # the earlier form of the rows

    def test_a_2in_face_step_closes_the_lanes_of_a_32in_core(self):
        with gf.frame():
            for size, expected in ((30.0, 3), (28.0, 0)):
                upper = replace(LOWER, b_in=size, h_in=size)
                result = transitions.column_transition(LOWER, upper, fy_ksi=60.0, clear_cover_in=sp.COL_CLEAR_COVER_IN,
                                                       upper_clear_height_in=153.0, joint_depth_in=36.0)
                self.assertTrue(result["supported"])                                # a declared offset-bend transition
                bars = bar_layers.transition_inventory(LOWER, upper, result)
                self.assertIsNotNone(bars)
                for axis in ("x", "y"):
                    self.assertEqual(bar_layers.bars_per_layer(LOWER, BEAM, axis), 3)
                    self.assertEqual(bar_layers.inventory_lanes(bars, LOWER, BEAM, axis), expected, (size, axis))
            same = transitions.column_transition(LOWER, LOWER, fy_ksi=60.0, clear_cover_in=sp.COL_CLEAR_COVER_IN,
                                                 upper_clear_height_in=153.0, joint_depth_in=36.0)
            self.assertIsNone(bar_layers.transition_inventory(LOWER, LOWER, same))   # nothing moves: the core cage

    def test_rows_evidence_and_the_capacity_row_read_the_same_lanes(self):
        with gf.frame(**SIX):
            state = mg.install(two_bands())
            gid = "s01_03__beam_x__edge"
            rows = bar_layers.arrangement()
            lowest = min(bar_layers.bars_per_layer_at_joint(*joint, beam.design, axis)
                         for joint, axis, beam in every_joint_beam(state) if beam.group_id == gid)
            self.assertEqual(lowest, 0)
            self.assertEqual(rows["_basis"]["bars_per_layer_governing"][gid], lowest)
            recorded = rows["_basis"]["joint_inventory"]
            self.assertEqual(recorded["rule"], bar_layers.JOINT_INVENTORY_RULE_ID)
            self.assertTrue(all(entry["joint"][0] == 3 and entry["joint_lanes"] < entry["core_cage_lanes"]
                                for entry in recorded["joints_with_fewer_lanes"]))
            evidence = selection._lane_governors(state, ("s01_03__beam_x__edge", "s01_03__beam_y__edge"), state.designs[gid])
            self.assertEqual(evidence[gid]["bars_per_layer"], 0)
            self.assertFalse(evidence[gid]["threads"])
            kinds = [kind["representative"] for kind in gc.joint_inventory(state)["types"].values()]
            boundary = next(e for e in kinds if e["level"] == 3 and e["core"].group_id == "s01_03__column__corner")
            for axis in ("x", "y"):
                row = gc.joint_threading(boundary, axis)
                self.assertEqual((row["bars_per_layer_that_fit"], row["bars_per_layer_core_cage"], row["passes"]), (0, 3, False))
                self.assertIn("inventory", row["lane_basis"])
            below = next(e for e in kinds if e["level"] < 3 and e["core"].group_id == "s01_03__column__corner")
            self.assertEqual(gc.joint_threading(below, "x")["lane_basis"], "the core cage")

    def test_the_chain_cuts_cage_pairs_whose_boundary_joint_closes_the_lanes(self):
        with gf.frame(**SIX):
            state = mg.install(two_bands(beam=replace(BEAM, top_bars=4, bot_bars=4)))
            actions = tb.synthetic_actions(state)
            locations = dict(selection._column_units(state))["gravity_column"]
            selected, report = selection.select_column_chain(state, locations, actions, DesignConfig())
            self.assertEqual(report["status"], "selected")
            self.assertIn(report["cage_pool"], ("preferred_beam_layers", "cages_the_beam_bars_thread"))
            (counts,) = report["links"].values()
            self.assertGreater(counts["beam_lanes"], 0)
            self.assertEqual(counts["pairs"] - counts["transition"] - counts["strong_column"] - counts["beam_lanes"] > 0, True)
            chosen = mg.install(state.with_designs(selected))
            joint = (3, 1, 1)                                                          # the interior line's boundary joint
            self.assertEqual(mp.joint_core_column(*joint).group_id, "s01_03__column__interior")
            for axis in ("x", "y"):
                beams = [m for m in (mg.joint_members(*joint)[f"beam_{axis}_{s}"] for s in ("minus", "plus")) if m is not None]
                for beam in beams:
                    fit = bar_layers.bars_per_layer_at_joint(*joint, beam.design, axis)
                    self.assertTrue(selection._lanes_admit(beam.design, fit), (axis, fit))
            self.assertIs(mg.active(), chosen)


if __name__ == "__main__":
    unittest.main()
