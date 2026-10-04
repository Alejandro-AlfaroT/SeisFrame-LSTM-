"""Column design shear with the beam delivery divided between the columns by analysis (2026-10-04).

ACI 318-19 18.7.6.1.1 lets the column shear be limited by the joint strengths based on the beams' Mpr, and its
commentary asks for the beam moments to be distributed to the columns above and below by analysis. These tests
check the shares against hand values from made-up combination actions (both sway senses, an opposite-sense
joint, a gravity-size joint that must be ignored), then Ve story by story: the base end, the floor joints, the
roof joint, the clear heights, the per-end column cap and the analysis-shear floor.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Design import SMRF_Capacity_Design as capacity                              # noqa: E402
from Design.Config import DesignConfig                                           # noqa: E402
from Design.SMRF_Capacity_Design import (COLUMN_SHEAR_METHOD_ANALYSIS_SPLIT, COLUMN_SHEAR_METHOD_JOINT_LIMITED,  # noqa: E402
                                         CLEAR_HEIGHT_PHYSICAL, column_action_envelopes, design_column_shear)
from test_smrf_column_shear_alternative import hoop_demanding_state, strengths_for   # noqa: E402

# Column end moments (kip-in) by story for one column line: (end i at the bottom, end j at the top).
SWAY = {1: (900.0, 600.0), 2: (400.0, 300.0), 3: (700.0, 500.0), 4: (-100.0, 250.0)}
EVEN = {1: (500.0, 350.0), 2: (350.0, 200.0), 3: (200.0, 100.0), 4: (100.0, 80.0)}
SMALL = {1: (5.0, 1.0), 2: (99.0, 2.0), 3: (3.0, 1.0), 4: (1.0, 1.0)}          # gravity-size: must not set a share


def action(name, moments, sign=1.0, shear=20.0):
    members = {}
    for story, (m_i, m_j) in moments.items():
        forces = [0.0] * 12
        forces[4], forces[10] = sign * m_i, sign * m_j                            # x frame: local My at i, j
        forces[5], forces[11] = sign * m_i, sign * m_j                            # y frame: local Mz at i, j
        forces[1] = forces[2] = shear
        members[str(story)] = {"member_type": "column", "local_force_kip_kipin": forces,
                               "axial_i_kip": 100.0 * (5 - story), "axial_j_kip": 95.0 * (5 - story)}
    return {"id": name, "analysis_succeeded": True, "members": members}


def envelopes(*actions):
    return column_action_envelopes(list(actions), 1)


class AnalysisShares(unittest.TestCase):
    def setUp(self):
        self.detail = envelopes(action("sway+", SWAY), action("sway-", SWAY, sign=-1.0), action("even", EVEN),
                                action("gravity", SMALL))["detail"]

    def share(self, story, end, axis="x"):
        return self.detail[story]["beam_moment_share"][axis][end]

    def test_shares_by_hand(self):
        # joint above story 1: sway 600 below, 400 above -> 0.6 / 0.4; even 350 / 350 -> 0.5 / 0.5
        self.assertAlmostEqual(self.share(1, "top")["share"], 0.6)
        self.assertAlmostEqual(self.share(2, "bottom")["share"], 0.5)
        # joint above story 2: sway 300 / 700 -> 0.3 / 0.7; even 200 / 200 -> 0.5 / 0.5
        self.assertAlmostEqual(self.share(2, "top")["share"], 0.5)
        self.assertAlmostEqual(self.share(3, "bottom")["share"], 0.7)
        # joint above story 3: sway 500 below and -100 above (opposite sense): the column below takes it all
        self.assertEqual(self.share(3, "top")["share"], 1.0)
        self.assertFalse(self.share(3, "top")["source"]["same_sense"])
        self.assertAlmostEqual(self.share(4, "bottom")["share"], 0.5)             # from the even case; the sway case gives 0
        self.assertIsNone(self.share(1, "bottom"))                                # the base
        self.assertEqual(self.share(4, "top")["share"], 1.0)                      # no column above the roof joint
        self.assertEqual(self.share(1, "top", "y")["share"], self.share(1, "top", "x")["share"])

    def test_both_sway_senses_agree_and_a_gravity_size_joint_is_ignored(self):
        one_sense = envelopes(action("sway+", SWAY))["detail"]
        other = envelopes(action("sway-", SWAY, sign=-1.0))["detail"]
        for story in (1, 2, 3):
            self.assertEqual(one_sense[story]["beam_moment_share"]["x"]["top"]["share"],
                             other[story]["beam_moment_share"]["x"]["top"]["share"])
        # "gravity" alone would give 1 / (1 + 99) at the first joint and 0.99 to the column above; it is below a
        # quarter of the largest joint moment there (1000) and is not used.
        top = self.share(1, "top")
        self.assertEqual((top["observations"], top["observations_used"]), (4, 3))
        self.assertAlmostEqual(self.share(2, "bottom")["share"], 0.5)
        self.assertEqual(top["largest_joint_moment_kip_in"], 1000.0)
        alone = envelopes(action("gravity", SMALL))["detail"]                     # when it is all there is, it is used
        self.assertAlmostEqual(alone[2]["beam_moment_share"]["x"]["bottom"]["share"], 0.99)


class AnalysisSplitShear(unittest.TestCase):
    def setUp(self):
        self.state = hoop_demanding_state()
        made = envelopes(action("sway+", SWAY), action("sway-", SWAY, sign=-1.0), action("even", EVEN))
        self.state["column_action_envelopes"] = made["detail"]
        self.state["column_axial_envelope"] = made["axial"]
        self.state["column_shear_demand"] = made["shear"]
        self.state["column_clear_height_convention"] = CLEAR_HEIGHT_PHYSICAL
        self.strengths = strengths_for(self.state)
        self.result = design_column_shear(self.state, self.strengths, method=COLUMN_SHEAR_METHOD_ANALYSIS_SPLIT)

    def delivery(self, axis):
        interior, edge = self.strengths[f"{axis}_interior"], self.strengths[f"{axis}_edge"]
        both = lambda s: s["mpr_negative_kip_in"] + s["mpr_positive_kip_in"]     # noqa: E731
        return max(both(interior), both(edge), interior["mpr_negative_kip_in"], edge["mpr_negative_kip_in"])

    def test_ve_story_by_story_by_hand(self):
        shares = {1: (0.6, None), 2: (0.5, 0.5), 3: (1.0, 0.7), 4: (1.0, 0.5)}    # (top, bottom)
        for axis in ("x", "y"):
            delivery = self.delivery(axis)
            for story, (top, bottom) in shares.items():
                entry = self.result["stories"][axis][story]
                own = {end: max(v["mpr_kip_in"] for v in entry["column_own"]["mpr_by_end_and_sense"][end].values()) for end in ("i", "j")}
                m_top = min(own["j"], top * delivery)
                m_bottom = own["i"] if bottom is None else min(own["i"], bottom * delivery)
                height = 120.0 - 10.0 if story == 1 else 120.0 - 20.0             # physical base: story_h - h_beam / 2
                self.assertEqual(entry["clear_height_in"], height)
                self.assertAlmostEqual(entry["ve_analysis_split_kip"], (m_top + m_bottom) / height, places=9)
                self.assertAlmostEqual(entry["ve_kip"], max((m_top + m_bottom) / height, 20.0), places=9)
                self.assertEqual(entry["analysis_split"]["beam_delivery_kip_in"], delivery)
                self.assertEqual(entry["method"], COLUMN_SHEAR_METHOD_ANALYSIS_SPLIT)
        split = self.result["stories"]["x"][1]["analysis_split"]
        self.assertEqual(split["bottom"]["limited_by"], "column (base)")          # a fixed base is not limited by beams
        self.assertEqual(split["top"]["limited_by"], "beams")                     # weak beams under a strong column
        self.assertEqual(self.result["stories"]["x"][4]["analysis_split"]["top"]["basis"], "roof joint: no column above, share 1")
        self.assertEqual(self.result["column_shear_method"], COLUMN_SHEAR_METHOD_ANALYSIS_SPLIT)
        self.assertEqual(self.result["clear_height_convention"], CLEAR_HEIGHT_PHYSICAL)

    def test_it_is_not_the_equal_split_and_never_exceeds_the_column_own_value(self):
        for axis in ("x", "y"):
            for story in (1, 2, 3, 4):
                entry = self.result["stories"][axis][story]
                self.assertLessEqual(entry["ve_analysis_split_kip"], entry["ve_own_envelope_kip"] * (1.0 + 1e-12) + 1e-9
                                     if story > 1 else float("inf"))
            # story 3 carries the whole delivery at its top and 0.7 of it at its bottom: 1.7 against the equal split's 1.0
            entry = self.result["stories"][axis][3]
            delivery = self.delivery(axis)
            if 1.0 * delivery < entry["analysis_split"]["top"]["column_mpr_kip_in"]:
                self.assertAlmostEqual(entry["ve_analysis_split_kip"] * 100.0, 1.7 * delivery, places=6)
                self.assertGreater(entry["ve_analysis_split_kip"], entry["ve_joint_limited_kip"] * 1.6)

    def test_each_end_is_capped_by_the_column_and_the_analysis_shear_is_a_floor(self):
        strong = {key: {**value, "mpr_negative_kip_in": 1.0e7, "mpr_positive_kip_in": 1.0e7} for key, value in self.strengths.items()}
        capped = design_column_shear(self.state, strong, method=COLUMN_SHEAR_METHOD_ANALYSIS_SPLIT)
        for story in (1, 2, 3, 4):
            entry = capped["stories"]["x"][story]
            self.assertEqual(entry["analysis_split"]["top"]["limited_by"], "column")
            own = {end: max(v["mpr_kip_in"] for v in entry["column_own"]["mpr_by_end_and_sense"][end].values()) for end in ("i", "j")}
            self.assertAlmostEqual(entry["ve_analysis_split_kip"], (own["i"] + own["j"]) / entry["clear_height_in"], places=9)
        state = {**self.state, "column_action_envelopes": envelopes(action("sway+", SWAY, shear=5000.0))["detail"]}
        floored = design_column_shear(state, self.strengths, method=COLUMN_SHEAR_METHOD_ANALYSIS_SPLIT)
        self.assertEqual(floored["stories"]["x"][2]["ve_kip"], 5000.0)

    def test_missing_analysis_moments_are_an_error_and_the_other_methods_still_report_the_split(self):
        bare = {**self.state, "column_action_envelopes": {}}
        with self.assertRaisesRegex(ValueError, "analysis split"):
            design_column_shear(bare, self.strengths, method=COLUMN_SHEAR_METHOD_ANALYSIS_SPLIT)
        legacy = design_column_shear(self.state, self.strengths, method=COLUMN_SHEAR_METHOD_JOINT_LIMITED)
        self.assertEqual(legacy["stories"]["x"][2]["ve_analysis_split_kip"], self.result["stories"]["x"][2]["ve_analysis_split_kip"])
        self.assertEqual(legacy["column_shear_method"], COLUMN_SHEAR_METHOD_JOINT_LIMITED)

    def test_the_uniform_design_reads_the_analysis_split(self):
        self.assertEqual(DesignConfig().capacity.uniform_column_shear_method, COLUMN_SHEAR_METHOD_ANALYSIS_SPLIT)
        self.assertIn(COLUMN_SHEAR_METHOD_ANALYSIS_SPLIT, capacity.COLUMN_SHEAR_METHODS)


if __name__ == "__main__":
    unittest.main()
