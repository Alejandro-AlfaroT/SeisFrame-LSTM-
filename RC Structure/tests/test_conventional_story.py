"""Hand-value tests of the story mechanism strength (Design.SMRF_Conventional_Story). No solve."""
import unittest
from unittest import mock

import Structure_Parameters as sp
from Design import SMRF_Conventional_Story as story


class Member:
    def __init__(self, tag, level, i, j, design="c"):
        self.member_tag, self.story_or_floor, self.grid_i, self.grid_j, self.design = tag, level, i, j, design


class State:
    policy_id = "x"

    def __init__(self, columns):
        self._columns = columns

    def members(self, kind):
        return self._columns if kind == "column" else []


def joint(level, beams):
    record = {"beam_capacities": [{"tag": 1, "mn_kip_in": beams, "slab_mn_kip_in": 0.0}]}
    return {"floor": level, "grid_i": 0, "grid_j": 0, "directions": {axis: {"positive": record, "negative": record}
                                                                         for axis in ("x", "y")}}


class TwoStoryColumnLine(unittest.TestCase):
    def setUp(self):
        self.patches = [mock.patch.multiple(sp, NUM_BAY_X=1, NUM_BAY_Y=1, NUM_FLOOR=2)]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()

    def _calculate(self, own, beams):
        """One column line at grid (0, 0), two stories; every column end has strength ``own``."""
        columns = [Member(11, 1, 0, 0), Member(12, 2, 0, 0)]
        actions = [{"id": "E", "ex": 1.0}]
        gravity = [{"id": "g1", "represented_design_combination_ids": ["E"],
                    "members": {"11": {"axial_i_kip": 100.0, "axial_j_kip": 90.0}, "12": {"axial_i_kip": 40.0, "axial_j_kip": 30.0}}}]
        capacity = {"scwb": {"joints": [joint(1, beams[0]), joint(2, beams[1])]}}
        with mock.patch("Design.Group_Capacity.column_nominal_moment", lambda design, axis, p: (own, "positive", None)), \
                mock.patch("Design.Group_Capacity.column_clear_heights", lambda member: {"physical": 100.0}), \
                mock.patch("Model.Member_Groups.layout_of", lambda policy: "perimeter"), \
                mock.patch("Model.Member_Groups.column_sfrs_directions", lambda i, j, nx, ny, layout: {"x", "y"}):
            return story.calculate(State(columns), actions, capacity, gravity)

    def test_strong_beams_leave_the_columns_in_control(self):
        result = self._calculate(own=1000.0, beams=(5000.0, 5000.0))
        first = result["stories"][0]["by_direction"]["x"]["story_strength_kip"]
        self.assertAlmostEqual(first, (1000.0 + 1000.0) / 100.0)

    def test_weak_beams_limit_the_column_ends(self):
        # Level 1: 600 shared equally by two columns of 1000: 300 each. Roof: 400 all to the column below.
        result = self._calculate(own=1000.0, beams=(600.0, 400.0))
        rows = {row["story"]: row["by_direction"]["x"]["story_strength_kip"] for row in result["stories"]}
        self.assertAlmostEqual(rows[1], (300.0 + 1000.0) / 100.0)          # base: own strength
        self.assertAlmostEqual(rows[2], (400.0 + 300.0) / 100.0)
        self.assertTrue(story.model_valid(result))

    def test_no_gravity_state_of_the_seismic_combinations_is_refused(self):
        with self.assertRaises(ValueError):
            story._gravity_members([{"id": "E", "ex": 1.0}], [{"id": "g", "represented_design_combination_ids": ["G"], "members": {}}])

    def test_a_beam_without_strength_is_refused(self):
        with self.assertRaises(ValueError):
            story._beam_sum({"beam_capacities": [{"tag": 3}]})

    def test_model_valid_needs_every_story(self):
        self.assertFalse(story.model_valid({"method": story.METHOD, "stories": []}))
        self.assertFalse(story.model_valid({"method": "other", "stories": []}))


if __name__ == "__main__":
    unittest.main()
