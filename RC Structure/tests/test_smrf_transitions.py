"""Column transitions at a band boundary (Design.SMRF_Transitions, declared rules column_transition_rules_v2).

Every longitudinal bar is placed at its declared coordinates in both columns and given one path through
the joint. The tests work the coordinates by hand: what is supported, what a supported transition asks of
the bend, the splice and the hoops, what stays unsupported, and the two reviewed counterexamples (a hoop
size change moves the bars; an offset is measured between bar coordinates, not between column faces).
A supported transition means the declared rule set is met; the rule set itself awaits engineering review.
"""
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Design import SMRF_Transitions as tr                                      # noqa: E402
from Model.Member_Groups import MemberDesign                                   # noqa: E402

LOWER = MemberDesign("column", 30.0, 30.0, 5.0, 10, 4, 4, 2, 4, 4, 4.0, (4, 4))  # No. 10, 4/4/2, No. 4 hoops 4 legs at 4 in
KWARGS = dict(fy_ksi=60.0, clear_cover_in=1.5, upper_clear_height_in=140.0, joint_depth_in=28.0)
COVER = 1.5 + 0.5 + 0.5 * 1.27                                                    # clear + No. 4 hoop + half a No. 10
BENDS = "the inclined part of the offset bends fits in the joint depth at a slope of at most 1 in 6"
TIES = "hoop sets at each bend carry 1.5 times the horizontal component of the offset bars"


def transition(upper, lower=LOWER, **overrides):
    return tr.column_transition(lower, upper, **{**KWARGS, **overrides})


def failed(result):
    return [item["rule"] for item in result["items"] if not item["passes"]]


class DevelopmentAndLap(unittest.TestCase):
    def test_development_length_is_the_table_25_4_2_3_expression(self):
        # first row, No. 7 and larger: fy / (20 sqrt(fc)) db
        self.assertAlmostEqual(tr.development_length_in(10, 5.0, 60.0), 60000.0 / (20.0 * math.sqrt(5000.0)) * 1.27, places=9)
        # first row, No. 6 and smaller: 25; the 12 in floor governs small bars in strong concrete
        self.assertAlmostEqual(tr.development_length_in(6, 5.0, 60.0), 60000.0 / (25.0 * math.sqrt(5000.0)) * 0.75, places=9)
        self.assertEqual(tr.development_length_in(3, 8.0, 60.0), 12.0)
        # other cases: 3 fy / (40 sqrt(fc)) db and 3 fy / (50 sqrt(fc)) db, one and a half times the first row
        self.assertAlmostEqual(tr.development_length_in(10, 5.0, 60.0, favourable=False), 1.5 * tr.development_length_in(10, 5.0, 60.0), places=9)
        self.assertAlmostEqual(tr.development_length_in(6, 5.0, 60.0, favourable=False), 3.0 * 60000.0 / (50.0 * math.sqrt(5000.0)) * 0.75, places=9)
        # Grade 80 carries psi_g = 1.15; an undeclared grade is refused
        self.assertAlmostEqual(tr.development_length_in(10, 5.0, 80.0), 80000.0 * 1.15 / (20.0 * math.sqrt(5000.0)) * 1.27, places=9)
        with self.assertRaises(ValueError):
            tr.development_length_in(10, 5.0, 75.0)

    def test_the_table_row_is_decided_from_the_cage(self):
        case = tr.development_case(LOWER, 1.5)
        self.assertTrue(case["favourable"])
        self.assertEqual(case["clear_cover_to_bar_in"], 2.0)                               # 1.5 in clear + No. 4 hoop
        self.assertAlmostEqual(case["clear_spacing_in"], (30.0 - 2 * COVER) / 3.0 - 1.27, places=12)   # four bars across the face
        crowded = replace(LOWER, b_in=14.0, h_in=14.0, bar_size=11, top_bars=5, bot_bars=5)
        self.assertLess(tr.development_case(crowded, 1.5)["clear_spacing_in"], 1.41)       # closer than one bar diameter
        self.assertFalse(tr.development_case(crowded, 1.5)["favourable"])
        thick = replace(LOWER, bar_size=18, stirrup_bar_size=4)                             # 2.257 in bar under 2.0 in of cover
        self.assertFalse(tr.development_case(thick, 1.5)["favourable"])
        self.assertEqual(tr.development_case(thick, 1.5)["row"], "other cases")

    def test_lap_of_unlike_bars_is_the_larger_of_ld_large_and_1_3_ld_small(self):
        large, small = tr.development_length_in(10, 5.0, 60.0), tr.development_length_in(8, 5.0, 60.0)
        self.assertAlmostEqual(tr.lap_length_in(10, 8, 5.0, 60.0), max(large, 1.3 * small), places=9)
        self.assertEqual(tr.lap_length_in(8, 10, 5.0, 60.0), tr.lap_length_in(10, 8, 5.0, 60.0))
        self.assertAlmostEqual(tr.lap_length_in(10, 8, 5.0, 60.0, favourable=False), 1.5 * tr.lap_length_in(10, 8, 5.0, 60.0), places=9)


class BarCoordinates(unittest.TestCase):
    def test_every_bar_is_placed_about_the_centerline(self):
        bars = tr.bar_coordinates(LOWER, 1.5)
        self.assertEqual(len(bars), 4 + 4 + 2 * 2)
        half = 15.0 - COVER
        top = sorted(b["x_in"] for b in bars if b["face"] == "top")
        self.assertEqual({b["y_in"] for b in bars if b["face"] == "top"}, {half})
        for got, expected in zip(top, (-half, -half / 3.0, half / 3.0, half)):
            self.assertAlmostEqual(got, expected, places=12)
        left = sorted(b["y_in"] for b in bars if b["face"] == "left")
        for got, expected in zip(left, (-half / 3.0, half / 3.0)):
            self.assertAlmostEqual(got, expected, places=12)
        self.assertEqual({b["x_in"] for b in bars if b["face"] == "left"}, {-half})

    def test_the_hoop_and_the_bar_size_move_the_bars(self):
        self.assertAlmostEqual(tr.centroid_cover_in(LOWER, 1.5), COVER, places=12)
        self.assertAlmostEqual(tr.centroid_cover_in(replace(LOWER, stirrup_bar_size=5), 1.5) - COVER, 0.125, places=12)
        self.assertAlmostEqual(tr.centroid_cover_in(replace(LOWER, bar_size=8), 1.5) - COVER, -0.135, places=12)

    def test_pairing_along_a_face_is_one_to_one_in_order_and_of_least_distance(self):
        self.assertEqual(tr._match_along_face([-9.0, -3.0, 3.0, 9.0], [-9.0, -3.0, 3.0, 9.0]), [0, 1, 2, 3])
        self.assertEqual(tr._match_along_face([-9.0, -3.0, 3.0, 9.0], [-9.0, 9.0]), [0, 3])
        middle = tr._match_along_face([-9.0, -3.0, 3.0, 9.0], [-9.0, 0.0, 9.0])
        self.assertEqual((middle[0], middle[2]), (0, 3))
        self.assertIn(middle[1], (1, 2))                                                  # either neighbour, never a corner
        self.assertEqual(tr._match_along_face([-9.0, 9.0], []), [])


class Paths(unittest.TestCase):
    def test_identical_columns_run_every_bar_straight(self):
        result = transition(replace(LOWER, stirrup_spacing_in=6.0))                       # another hoop spacing moves no bar
        self.assertEqual((result["kind"], result["supported"], result["column_reinforcement_continuous"]), ("same_cage", True, True))
        self.assertEqual(result["path_counts"], {"straight": 12, "offset": 0, "lap_spliced": 0, "mechanically_spliced": 0, "terminated": 0})
        self.assertIsNone(result["splice"])
        self.assertIsNone(result["offset_bend"])
        self.assertEqual((result["extra_bar_length_in"], result["additional_hoop_sets_per_joint"]), (0.0, 0))

    def test_a_larger_hoop_above_moves_every_bar_and_is_not_a_straight_path(self):
        # Reviewed counterexample: No. 4 hoops below, No. 5 above, one section and cage. Every centroid moves in
        # by 0.125 in on each axis it is tied to; a corner bar by 0.125 sqrt(2).
        result = transition(replace(LOWER, stirrup_bar_size=5))
        self.assertEqual((result["kind"], result["supported"]), ("offset_bends", True))
        self.assertEqual(result["path_counts"]["offset"], 12)
        self.assertEqual(result["path_counts"]["straight"], 0)
        bend = result["offset_bend"]
        self.assertAlmostEqual(bend["bar_offset_in"], 0.125 * math.sqrt(2.0), places=12)
        self.assertAlmostEqual(bend["inclined_length_in"], 6.0 * 0.125 * math.sqrt(2.0), places=12)
        corner = next(p for p in result["paths"] if p["face"] == "top" and p["lower_index"] == 0)
        self.assertAlmostEqual(corner["offset_xy_in"][0], 0.125, places=12)
        self.assertAlmostEqual(corner["offset_xy_in"][1], -0.125, places=12)
        # a smaller hoop above moves them out by the same amount
        self.assertAlmostEqual(transition(LOWER, lower=replace(LOWER, stirrup_bar_size=5))["offset_bend"]["bar_offset_in"],
                               0.125 * math.sqrt(2.0), places=12)

    def test_an_offset_is_measured_between_bar_coordinates_not_between_faces(self):
        # Reviewed counterexample: 30 to 26 in, No. 10 bars, No. 4 hoops below and No. 5 above, an 18 in joint.
        # The faces step 2 in; the bars move 2.125 in per axis, 3.0052 in at a corner, and need 18.031 in at 1 in 6.
        upper = replace(LOWER, b_in=26.0, h_in=26.0, stirrup_bar_size=5)
        result = transition(upper, joint_depth_in=18.0)
        bend = result["offset_bend"]
        self.assertAlmostEqual(bend["bar_offset_in"], math.hypot(2.125, 2.125), places=12)
        self.assertAlmostEqual(bend["inclined_length_in"], 6.0 * math.hypot(2.125, 2.125), places=12)
        self.assertGreater(bend["inclined_length_in"], 18.0)
        self.assertEqual((result["kind"], result["supported"], result["column_reinforcement_continuous"]), ("offset_bends", False, False))
        self.assertEqual(failed(result), [BENDS])
        # with the same hoops the bars move exactly as the faces do, and 16.97 in fits
        same_hoops = transition(replace(LOWER, b_in=26.0, h_in=26.0), joint_depth_in=18.0)
        self.assertAlmostEqual(same_hoops["offset_bend"]["inclined_length_in"], 12.0 * math.sqrt(2.0), places=12)
        self.assertTrue(same_hoops["supported"])
        self.assertFalse(transition(replace(LOWER, b_in=26.0, h_in=26.0), joint_depth_in=12.0)["supported"])

    def test_the_inclined_length_is_set_by_the_bar_that_moves_most(self):
        result = transition(replace(LOWER, b_in=26.0, h_in=26.0))
        offsets = sorted({round(p["offset_in"], 9) for p in result["paths"]})
        self.assertAlmostEqual(offsets[-1], 2.0 * math.sqrt(2.0), places=9)               # the corners
        self.assertLess(offsets[0], offsets[-1])                                          # a side bar near mid-height moves less
        self.assertEqual(result["offset_bend"]["bend_points_below_joint_top_in"], [0.0, result["offset_bend"]["inclined_length_in"]])
        self.assertEqual(result["offset_bend"]["smallest_bar_offset_in"], min(p["offset_in"] for p in result["paths"]))

    def test_smaller_bars_above_are_lapped_in_the_center_half_beside_their_lower_bars(self):
        result = transition(replace(LOWER, bar_size=8))
        self.assertEqual((result["kind"], result["supported"]), ("bar_size_reduction", True))
        self.assertEqual(result["path_counts"]["lap_spliced"], 12)
        splice = result["splice"]
        self.assertAlmostEqual(splice["class_b_lap_in"], tr.lap_length_in(10, 8, 5.0, 60.0), places=9)
        self.assertEqual((splice["center_half_clear_height_in"], splice["splice_type"]), (70.0, "class_B_lap_center_half"))
        self.assertEqual(splice["development"]["row_used"], "first row")
        # the lower bars run straight (they lie inside the upper hoops); a corner pair sits 0.135 sqrt(2) apart
        self.assertEqual(result["bars_bent"], 0)
        corner = next(p for p in result["paths"] if p["face"] == "top" and p["lower_index"] == 0)
        self.assertAlmostEqual(corner["splice_distance_in"], 0.135 * math.sqrt(2.0), places=9)
        self.assertLessEqual(corner["splice_distance_in"], splice["non_contact_distance_limit_in"])

    def test_a_lap_that_does_not_fit_or_is_not_permitted_is_a_coaxial_mechanical_splice(self):
        short = transition(replace(LOWER, bar_size=8), upper_clear_height_in=80.0)        # center half 40 in < lap
        self.assertEqual(short["splice"]["splice_type"], "type_2_mechanical_18.2.7")
        self.assertEqual(short["path_counts"]["mechanically_spliced"], 12)
        self.assertEqual(short["bars_bent"], 12)                                          # each lower bar goes to its upper bar's axis
        self.assertAlmostEqual(short["offset_bend"]["bar_offset_in"], 0.135 * math.sqrt(2.0), places=9)
        self.assertTrue(short["supported"])
        big = transition(replace(LOWER, bar_size=11), lower=replace(LOWER, bar_size=14))  # No. 14: no lap (25.5.1.1)
        self.assertFalse(big["splice"]["lap_splice_permitted_25_5_1_1"])
        self.assertEqual(big["path_counts"]["mechanically_spliced"], 12)

    def test_fewer_bars_above_terminates_the_unpaired_bars_and_counts_their_length(self):
        # 4/4/2 below, 4/4/1 above: on each side face one lower bar is paired with the single upper bar and one stops
        result = transition(replace(LOWER, side_bars=1))
        self.assertEqual(result["kind"], "bar_count_reduction")
        self.assertEqual(result["path_counts"]["terminated"], 2)
        self.assertEqual(result["path_counts"]["straight"], 8)                            # the top and bottom rows are unchanged
        self.assertEqual(result["terminated_lower_bars"], 2)
        self.assertEqual(result["extra_bar_length_in"], 2 * 0.5 * 140.0)
        paired = [p for p in result["paths"] if p["face"] in ("left", "right") and p["path"] != "terminated"]
        self.assertEqual(len(paired), 2)
        # the upper side bar sits at mid-height, a third of the half-height from either lower bar
        distance = (15.0 - COVER) / 3.0
        for path in paired:
            self.assertIn(path["path"], ("lap_spliced", "offset"))
            self.assertAlmostEqual(path["splice_distance_in"] if path["path"] == "lap_spliced" else path["offset_in"], distance, places=9)
        self.assertEqual(result["supported"], not failed(result))

    def test_a_reduced_column_with_smaller_bars_offsets_the_lower_bars_into_the_upper_cage_and_laps_them(self):
        result = transition(replace(LOWER, b_in=28.0, h_in=28.0, bar_size=8))
        self.assertEqual((result["kind"], result["supported"]), ("offset_bends", True))
        self.assertEqual(result["path_counts"]["lap_spliced"], 12)
        self.assertEqual(result["bars_bent"], 12)
        self.assertAlmostEqual(result["offset_bend"]["bar_offset_in"], math.sqrt(2.0), places=9)   # 1 in per axis: inside the upper hoops
        self.assertIn("continuous-column branch", " ".join(result["not_established"]))
        self.assertEqual(result["continuity_branch"], "continuing_column")
        self.assertIsNone(result["confinement"]["lower_hoops_continue_above_joint_in"])
        self.assertFalse(result["confinement"]["established_15_2_6_b"])


class BendTies(unittest.TestCase):
    def test_tie_demand_is_the_sum_of_the_horizontal_components_on_the_heavier_side(self):
        # 30 to 26 in: every top-row bar moves 2 in toward the centre in y; the two side bars above mid-height move 2/3 in.
        result = transition(replace(LOWER, b_in=26.0, h_in=26.0))
        bend = result["offset_bend"]
        run = 12.0 * math.sqrt(2.0)
        expected = 1.5 * 1.27 * (4 * 2.0 + 2 * 2.0 / 3.0) / run
        self.assertAlmostEqual(bend["tie_area_required_in2"]["y"], expected, places=9)
        self.assertAlmostEqual(bend["tie_area_required_in2"]["x"], expected, places=9)    # a square cage with the same rows
        self.assertEqual(bend["tie_area_per_hoop_set_in2"], {"x": 4 * 0.20, "y": 4 * 0.20})
        # two sets are needed and the 4 in hoop spacing already puts two within 6 in of each bend
        self.assertEqual((bend["hoop_sets_required"], bend["hoop_sets_at_hoop_spacing"], bend["hoop_sets_that_fit"]), (2, 2, 3))
        self.assertEqual(bend["tie_stations_from_each_bend_point_in"], [0.0, 4.0])
        self.assertEqual((bend["additional_hoop_sets_at_each_bend"], result["additional_hoop_sets_per_joint"]), (0, 0))
        self.assertEqual(bend["minimum_station_spacing_basis"], "declared search restriction, not an ACI minimum")

    def test_too_few_sets_at_the_hoop_spacing_are_made_up_by_additional_sets_at_both_bends(self):
        # 2 legs of No. 4 give 0.4 in2 per set: three sets are needed, the 4 in spacing gives two, a third fits at 3 in
        lower = replace(LOWER, stirrup_legs=2, stirrup_legs_by_direction=(2, 2))
        result = transition(replace(lower, b_in=26.0, h_in=26.0), lower=lower)
        bend = result["offset_bend"]
        self.assertEqual((bend["hoop_sets_required"], bend["hoop_sets_at_hoop_spacing"], bend["hoop_sets_within_6in"]), (3, 2, 3))
        self.assertEqual(bend["tie_stations_from_each_bend_point_in"], [0.0, 3.0, 6.0])
        self.assertEqual((bend["additional_hoop_sets_at_each_bend"], result["additional_hoop_sets_per_joint"]), (1, 2))
        self.assertTrue(result["supported"])
        self.assertAlmostEqual(bend["tie_area_provided_in2"]["y"], 3 * 2 * 0.20, places=12)

    def test_ties_that_cannot_fit_within_six_inches_fail_the_rule(self):
        # six No. 14 bars a face with 2 legs of No. 3: far more than the three sets that fit
        light = replace(LOWER, bar_size=14, top_bars=6, bot_bars=6, side_bars=4, stirrup_bar_size=3, stirrup_legs=2,
                        stirrup_legs_by_direction=(2, 2))
        result = transition(replace(light, b_in=28.0, h_in=28.0), lower=light)
        self.assertEqual(result["kind"], "offset_bends")
        self.assertIn(TIES, failed(result))
        self.assertFalse(result["supported"])
        self.assertEqual(result["offset_bend"]["hoop_sets_that_fit"], 3)
        self.assertGreater(result["offset_bend"]["hoop_sets_required"], 3)

    def test_the_lower_hoop_spacing_in_force_is_the_one_supplied(self):
        tight = transition(replace(LOWER, b_in=26.0, h_in=26.0), lower_hoop_spacing_in=3.0)
        self.assertEqual(tight["offset_bend"]["hoop_sets_at_hoop_spacing"], 3)
        self.assertEqual(tight["offset_bend"]["hoop_sets_required"], 2)                      # two suffice; three are there
        self.assertEqual(tight["offset_bend"]["tie_stations_from_each_bend_point_in"], [0.0, 3.0, 6.0])
        self.assertEqual(tight["additional_hoop_sets_per_joint"], 0)


class OutsideTheDeclaredScope(unittest.TestCase):
    def assert_unsupported(self, result, rule=None):
        self.assertEqual((result["kind"], result["supported"], result["column_reinforcement_continuous"]), ("unsupported", False, False))
        if rule is not None:
            self.assertIn(rule, failed(result))

    def test_a_larger_column_above(self):
        self.assert_unsupported(transition(replace(LOWER, b_in=32.0, h_in=32.0)),
                                "the column above is no larger than the column below")

    def test_a_face_step_of_three_inches_or_more(self):
        self.assert_unsupported(transition(replace(LOWER, b_in=24.0, h_in=24.0)), "each face steps in by less than 3 in")

    def test_larger_bars_or_more_bars_above(self):
        self.assert_unsupported(transition(replace(LOWER, bar_size=11)), "bars above are no larger than bars below")
        self.assert_unsupported(transition(replace(LOWER, top_bars=5, bot_bars=5)), "no face carries more bars above than below")

    def test_different_concrete_grades(self):
        self.assert_unsupported(transition(replace(LOWER, fc_ksi=4.0)), "one column concrete grade")

    def test_bars_that_stop_where_the_column_steps_in_are_not_supported(self):
        # 30 to 28 with one side bar fewer: the bar that stops lies outside the upper cage and would need a bend or a hook
        result = transition(replace(LOWER, b_in=28.0, h_in=28.0, side_bars=1))
        self.assertEqual((result["kind"], result["supported"], result["column_reinforcement_continuous"]), ("bar_count_reduction", False, False))
        self.assertIn("terminated lower bars stay inside the upper cage without an offset", failed(result))

    def test_a_beam_design_is_refused(self):
        beam = MemberDesign("beam_x", 18.0, 28.0, 4.0, 8, 4, 3, 0, 4, 2, 5.0)
        with self.assertRaises(ValueError):
            transition(beam)

    def test_every_result_names_the_rule_set_and_what_it_does_not_establish(self):
        for upper in (LOWER, replace(LOWER, bar_size=8), replace(LOWER, b_in=26.0, h_in=26.0), replace(LOWER, b_in=24.0, h_in=24.0)):
            result = transition(upper)
            self.assertEqual(result["rules"], "column_transition_rules_v2")
            self.assertIn(result["kind"], tr.KINDS)
            self.assertTrue(result["not_established"])
            self.assertEqual(sum(result["path_counts"].values()), len(result["paths"]))
        same = transition(LOWER)
        self.assertEqual(same["confinement"]["lower_hoops_continue_above_joint_in"], 30.0)
        self.assertTrue(same["confinement"]["established_15_2_6_b"])


if __name__ == "__main__":
    unittest.main()
