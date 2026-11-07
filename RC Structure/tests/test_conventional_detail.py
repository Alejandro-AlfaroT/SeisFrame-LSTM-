"""Hand-value tests of the conventional detailing rules (Design.SMRF_Conventional_Detail). No solve."""
import math
import unittest

import Structure_Parameters as sp
from Design import SMRF_Conventional_Detail as conventional
from Design.SMRF_Transitions import column_transition
from Model.Member_Groups import MemberDesign


def column(b=30.0, h=30.0, bar=9, top=4, bot=4, side=2, fc=6.0, hoop=4, legs=3, spacing=4.0):
    return MemberDesign("column", b, h, fc, bar, top, bot, side, hoop, legs, spacing, (legs, legs))


class Spacing(unittest.TestCase):
    def test_25_2_1_takes_the_largest_of_three(self):
        self.assertAlmostEqual(conventional.clear_spacing_25_2_1(1.128, 0.75), 1.128)
        self.assertAlmostEqual(conventional.clear_spacing_25_2_1(0.5, 0.75), 1.0)
        self.assertAlmostEqual(conventional.clear_spacing_25_2_1(0.5, 1.5), 2.0)

    def test_25_2_3_is_the_column_rule(self):
        self.assertAlmostEqual(conventional.clear_spacing_25_2_3(1.128, 0.75), 1.692)
        self.assertAlmostEqual(conventional.clear_spacing_25_2_3(0.75, 0.75), 1.5)

    def _path(self, kind, lower, upper):
        return {"path": kind, "target_xy_in": list(lower), "upper_xy_in": None if upper is None else list(upper)}

    def test_lap_partners_are_not_tested_against_each_other(self):
        paths = [self._path("lap_spliced", (0.0, 0.0), (0.2, 0.0))]
        self.assertTrue(conventional.lap_spacing(paths, 1.128, 1.0)["passes"])

    def test_another_bar_close_to_a_lap_fails(self):
        # Centers 2.0 in apart: clear 2.0 - (1.128 + 1.128) / 2 = 0.872 in against 1.128 in.
        paths = [self._path("lap_spliced", (0.0, 0.0), (0.0, 0.0)), self._path("terminated", (2.0, 0.0), None)]
        result = conventional.lap_spacing(paths, 1.128, 1.128)
        self.assertFalse(result["passes"])
        self.assertAlmostEqual(result["failures"][0]["clear_in"], 0.872)
        self.assertAlmostEqual(result["failures"][0]["required_in"], 1.128)

    def test_exact_limit_passes(self):
        paths = [self._path("lap_spliced", (0.0, 0.0), (0.0, 0.0)), self._path("terminated", (2.256, 0.0), None)]
        self.assertTrue(conventional.lap_spacing(paths, 1.128, 1.128)["passes"])


class Couplers(unittest.TestCase):
    POINTS = [(-11.0, 11.0), (-3.5, 11.0), (3.5, 11.0), (11.0, 11.0)]

    def _detail(self, lower=9, upper=9, spacing=6.0, grade=60.0):
        return conventional.coupler_detail(lower, upper, b_in=30.0, h_in=30.0, points=self.POINTS, fy_ksi=grade,
                                           specification=f"ASTM A706 Grade {int(grade)}", clear_cover_in=1.5,
                                           spacing_in=spacing, hoop_size=4, outside_limit_in=6.0)

    def test_published_same_size_passes_with_room(self):
        detail = self._detail(spacing=6.0)
        self.assertEqual(detail["status"], "pass")
        self.assertEqual(detail["product"]["part_number"], "EL28A2")

    def test_hoop_stations_may_shift_around_the_body(self):
        # No. 9 body 3 19/32 in, hoop 0.5 in: gap 4.094 in. At 3 in spacing the next gap is 6 - 4.094 = 1.906 >= 1.5.
        detail = self._detail(spacing=3.0)
        row = next(c for c in detail["checks"] if c["rule"] == "splice_device_between_hoop_sets")
        self.assertTrue(row["passes"])
        self.assertAlmostEqual(row["detail"]["remaining_gap_in"], 1.90625)

    def test_too_tight_hoops_leave_no_room(self):
        # Devices reaching past the hoop line (cover 15 - 12.6 - 0.75 = 1.65 in, hoop line at 1.5 + 0.5 = 2.0 in): at 2.5 in
        # the next gap is 5 - 4.094 = 0.906 < 1.5, so the body has no room between the sets.
        points = [(-12.6, 12.6), (-4.2, 12.6), (4.2, 12.6), (12.6, 12.6)]
        detail = conventional.coupler_detail(9, 9, b_in=30.0, h_in=30.0, points=points, fy_ksi=60.0,
                                             specification="ASTM A706 Grade 60", clear_cover_in=1.5, spacing_in=2.5,
                                             hoop_size=4, outside_limit_in=6.0)
        self.assertEqual(detail["status"], "fail")
        self.assertIn("splice_device_between_hoop_sets", detail["failure_rules"])

    def test_a_device_inside_the_hoop_line_needs_no_gap(self):
        # The default points sit 3.25 in from the faces: the device clears the hoop line by 1.25 in at any spacing.
        detail = self._detail(spacing=2.5)
        row = next(c for c in detail["checks"] if c["rule"] == "splice_device_between_hoop_sets")
        self.assertTrue(row["passes"])
        self.assertAlmostEqual(row["detail"]["device_inside_hoop_line_in"], 1.25)

    def _no11_beyond_l0(self, local_gap):
        # G03 o0002 (2026-11-07): 32 x 32 in, No. 11 on the 0.5 in grid 3.0 in from the faces, No. 5 hoops at 3.0 in.
        # EL36A2 body 3.96875 in: gap 3.96875 + 0.625 = 4.59375 in; the device reaches 1/16 in past the hoop line.
        points = [(-13.0, 13.0), (0.0, 13.0), (13.0, 13.0), (-13.0, 0.0), (13.0, 0.0), (-13.0, -13.0), (0.0, -13.0),
                  (13.0, -13.0)]
        detail = conventional.coupler_detail(11, 11, b_in=32.0, h_in=32.0, points=points, fy_ksi=60.0,
                                             specification="ASTM A706 Grade 60", clear_cover_in=1.5, spacing_in=3.0,
                                             hoop_size=5, outside_limit_in=6.0, local_gap=local_gap)
        return detail, next(c for c in detail["checks"] if c["rule"] == "splice_device_between_hoop_sets")

    def test_l0_spacing_leaves_no_room_for_a_no11_body(self):
        detail, row = self._no11_beyond_l0(None)
        self.assertAlmostEqual(row["detail"]["device_inside_hoop_line_in"], -0.0625)
        self.assertAlmostEqual(row["detail"]["device_plus_hoop_in"], 4.59375)
        self.assertAlmostEqual(row["detail"]["remaining_gap_in"], 6.0 - 4.59375)
        self.assertFalse(row["passes"])

    def test_beyond_l0_a_local_gap_within_18_7_5_5_and_shear_passes(self):
        detail, row = self._no11_beyond_l0({"limit_in": 6.0, "zone_in": 68.0})
        self.assertTrue(row["passes"])
        self.assertTrue(row["detail"]["beyond_l0"]["applies"])
        self.assertEqual(detail["status"], "pass")

    def test_a_shear_spacing_below_the_gap_keeps_the_rule(self):
        _detail, row = self._no11_beyond_l0({"limit_in": 4.5, "zone_in": 68.0})
        self.assertFalse(row["detail"]["beyond_l0"]["applies"])
        self.assertFalse(row["passes"])

    def test_no_length_beyond_l0_keeps_the_rule(self):
        _detail, row = self._no11_beyond_l0({"limit_in": 6.0, "zone_in": 4.0})
        self.assertFalse(row["detail"]["beyond_l0"]["applies"])
        self.assertFalse(row["passes"])

    def test_unpublished_pair_is_unknown_not_failed(self):
        detail = self._detail(lower=11, upper=9)
        self.assertEqual(detail["status"], "not_evaluated")
        self.assertEqual(detail["product"]["status"], "not_evaluated")

    def test_cover_is_measured_to_the_device(self):
        points = [(-13.5, 13.5), (13.5, 13.5)]
        detail = conventional.coupler_detail(9, 9, b_in=30.0, h_in=30.0, points=points, fy_ksi=60.0,
                                             specification="ASTM A706 Grade 60", clear_cover_in=1.5, spacing_in=6.0,
                                             hoop_size=4, outside_limit_in=6.0)
        row = next(c for c in detail["checks"] if c["rule"] == "splice_device_cover")
        # 15 - 13.5 - 1.5 / 2 = 0.75 in against 1.5 in.
        self.assertAlmostEqual(row["detail"]["provided_in"], 0.75)
        self.assertFalse(row["passes"])


class Roof(unittest.TestCase):
    def _detail(self, bar=9, fy=60.0, fc=6.0):
        return conventional.roof_detail(column(bar=bar, fc=fc), extension_in=30.0, joint_depth_in=30.0, fy_ksi=fy,
                                        clear_cover_in=1.5, spacing_in=4.0)

    def test_straight_bar_grade_60(self):
        detail = self._detail()
        row = next(c for c in detail["checks"] if c["rule"] == "roof_bar_anchorage")
        ldh = 60000.0 * 1.128 / (65.0 * math.sqrt(6000.0))
        self.assertAlmostEqual(row["detail"]["hook_required_in"], ldh)
        self.assertAlmostEqual(row["detail"]["straight_required_in"], 3.25 * ldh)
        self.assertEqual(detail["status"], "pass")
        self.assertEqual(detail["termination_type"], "straight_bar_at_declared_extension_top")

    def test_grade_80_no_11_needs_the_hook(self):
        # ldh = 80000 x 1.41 / (65 x 77.46) = 22.4 in; straight 72.8 in exceeds 57.8 in; the hook fits.
        detail = self._detail(bar=11, fy=80.0)
        self.assertEqual(detail["status"], "pass")
        self.assertEqual(detail["termination_type"], "standard_hook_at_declared_extension_top")

    def test_no_14_has_no_supported_termination(self):
        detail = self._detail(bar=14)
        self.assertEqual(detail["status"], "not_evaluated")
        self.assertIn("roof_bar_anchorage", detail["unknown_rules"])

    def test_short_extension_fails(self):
        detail = conventional.roof_detail(column(), extension_in=10.0, joint_depth_in=30.0, fy_ksi=60.0, clear_cover_in=1.5,
                                          spacing_in=4.0)
        self.assertIn("roof_extension_is_full_column_depth", detail["failure_rules"])


class Transitions(unittest.TestCase):
    def _transition(self, lower, upper, **context):
        return column_transition(lower, upper, fy_ksi=60.0, clear_cover_in=1.5, upper_clear_height_in=156.0, joint_depth_in=30.0,
                                 physical_context={"conventional": True, **context})

    def test_same_cage_is_straight_and_passes(self):
        result = self._transition(column(), column())
        self.assertEqual(result["kind"], "same_cage")
        self.assertEqual(result["physical_detail"]["status"], "pass")
        self.assertTrue(result["physically_verified"])

    def test_detail_method_is_conventional(self):
        result = self._transition(column(), column())
        self.assertEqual(result["physical_detail"]["method"], conventional.METHOD)

    def test_unsupported_pair_stays_unverified(self):
        result = self._transition(column(bar=14, top=5, bot=5, side=3), column(bar=9, top=5, bot=5, side=3))
        self.assertFalse(result["physically_verified"])
        self.assertNotEqual(result["physical_detail"]["status"], "pass")


class Basis(unittest.TestCase):
    def test_default_is_conventional(self):
        self.assertEqual(sp.DETAIL_BASIS, conventional.BASIS)
        self.assertTrue(conventional.active())

    def test_the_computed_route_value_turns_it_off(self):
        previous = sp.DETAIL_BASIS
        try:
            sp.DETAIL_BASIS = conventional.PHYSICAL_BASIS
            self.assertFalse(conventional.active())
        finally:
            sp.DETAIL_BASIS = previous


class Quantities(unittest.TestCase):
    def test_slab_steel_uses_area_per_foot_over_floor_area(self):
        layout = {"layers": {name: {"area_in2_per_ft": 0.3} for name in ("x_top", "x_bottom", "y_top", "y_bottom")}}
        self.assertAlmostEqual(conventional.slab_steel_volume_in3(layout, [1000.0, 1000.0]), 1.2 / 12.0 * 2000.0)


if __name__ == "__main__":
    unittest.main()
