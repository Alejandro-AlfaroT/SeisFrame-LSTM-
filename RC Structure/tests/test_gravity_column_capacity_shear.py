"""Gravity column capacity shear per direction (ACI 318-19 22.5.1.1 with 18.7.6.1.1; 18.14.3.2(b), 18.14.3.3(c)).

Regression of 2026-10-31: the row compared the governing direction's Ve (largest required Vs) with the lighter
direction's hoop phi Vn, so a column whose cage has unequal face bar counts, and so unequal hoop legs, failed although
each direction passed against its own hoops. The case runs the real column shear design and hoop selection
(Design.SMRF_Capacity_Design.design_column_shear, column-own method) on a 26 x 26 in, f'c 5 ksi column with #11
bars, 3 on each b face and 4 per side face, at a 168 in story; then the producer's per-direction pairing
(Design.Perimeter_Systems.gravity_column_capacity_shear) and the rule (Design.Gravity_Frame_Members).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))                        # sibling fixture module

from Design import Gravity_Frame_Members as gravity_rules                        # noqa: E402
from Design import SMRF_Capacity_Design as cd                                    # noqa: E402
from Design.Perimeter_Systems import gravity_column_capacity_shear               # noqa: E402
from test_smrf_capacity_design import state                                     # noqa: E402

AB, DB = 1.56, 1.41                                                              # #11
SIZE, TOP, SIDE = 26.0, 3, 4
COVER = 1.5 + 0.625 + DB / 2.0


def _layers(top, side):
    """Bending through h: the b-face bars in the extreme layers, one bar per side face at each interior level."""
    return ([(top * AB, COVER)] + [(2 * AB, COVER + (SIZE - 2 * COVER) * k / (side + 1)) for k in range(1, side + 1)]
            + [(top * AB, SIZE - COVER)])


def _layers_about_z(top, side):
    """Bending through b: corners plus side bars in the outer layers, the interior b-face bars between."""
    outer = (2 + side) * AB
    layers = [(outer, COVER), (outer, SIZE - COVER)]
    for _face in range(2):
        layers += [(AB, COVER + (SIZE - 2 * COVER) * k / (top - 1)) for k in range(1, top - 1)]
    return sorted(layers, key=lambda layer: layer[1])


def asymmetric_column_shear():
    st = state(geometry={"story_h_in": 168.0, "num_floor": 4},
               sections={"b_col_in": SIZE, "h_col_in": SIZE, "fc_col_ksi": 5.0},
               column={"bar_size": 11, "top_bars": TOP, "bot_bars": TOP, "side_bars": SIDE, "centroid_offset_in": COVER,
                       "stirrup_bar_size": 4, "layers": _layers(TOP, SIDE), "layers_about_z": _layers_about_z(TOP, SIDE)},
               column_axial_envelope={s: (10.0 * (5 - s), 50.0 * (5 - s)) for s in range(1, 5)},
               column_shear_demand={s: 5.0 for s in range(1, 5)})
    return cd.design_column_shear(st, None, method=cd.COLUMN_SHEAR_METHOD_COLUMN_OWN)


def gravity_row(strengths):
    section = {"ag_in2": SIZE * SIZE, "ast_in2": (2 * TOP + 2 * SIDE) * AB, "fc_ksi": 5.0, "fy_ksi": 60.0, "pu_kip": 200.0}
    detailing = {"hoop_spacing_in": 3.0, "smallest_bar_db_in": DB, "ash_ratio_provided": None,
                 "ash_ratio_table_18_7_5_4": None, "configuration_18_7_5_2_ok": True}
    checks = gravity_rules.evaluate_gravity_column("gravity", {"moment_ratio": 0.5, "shear_ratio": 0.5}, strengths,
                                                   section, detailing)
    return next(c for c in checks if c["id"] == "gravity.column_capacity_shear")


class GravityColumnCapacityShear(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = asymmetric_column_shear()

    def test_the_asymmetric_cage_passes_in_each_direction_but_not_across_directions(self):
        hoops, governing, worst = self.data["hoops"], self.data["governing"], self.data["worst_by_direction"]
        self.assertEqual(hoops["legs"], {"across_b_face": 3, "across_h_face": 4})
        self.assertEqual(governing["axis"], "y")
        # The lighter-direction scalar is x's: against the governing (y) Ve it fails.
        self.assertEqual(hoops["phi_vn_kip"], hoops["by_direction"]["x"]["phi_vn_kip"])
        self.assertGreater(governing["ve_kip"], hoops["phi_vn_kip"])
        # Each direction against its own hoops passes (s <= Av fyt d / Vs,req per direction).
        for axis in ("x", "y"):
            self.assertLessEqual(worst[axis]["ve_kip"], hoops["by_direction"][axis]["phi_vn_kip"], axis)

    def test_the_row_compares_one_direction_and_reports_the_larger_ratio(self):
        worst, own = self.data["worst_by_direction"], self.data["hoops"]["by_direction"]
        ratios = {a: worst[a]["ve_kip"] / own[a]["phi_vn_kip"] for a in ("x", "y")}
        strengths = gravity_column_capacity_shear(self.data)
        self.assertGreater(ratios["x"], ratios["y"])
        self.assertEqual(strengths["capacity_shear_axis"], "x")                 # not the governing y
        self.assertEqual((strengths["ve_kip"], strengths["phi_vn_capacity_kip"]), (worst["x"]["ve_kip"], own["x"]["phi_vn_kip"]))
        self.assertEqual(set(strengths["capacity_shear_by_direction"]), {"x", "y"})
        for axis in ("x", "y"):
            self.assertAlmostEqual(strengths["capacity_shear_by_direction"][axis]["ratio"], ratios[axis])
        row = gravity_row(strengths)
        self.assertEqual(row["status"], "pass")
        self.assertEqual((row["demand"], row["capacity"]), (worst["x"]["ve_kip"], own["x"]["phi_vn_kip"]))
        self.assertEqual(row["details"]["axis"], "x")
        self.assertEqual(row["details"]["by_direction"], strengths["capacity_shear_by_direction"])

    def test_without_every_direction_hoop_record_the_row_is_not_evaluated(self):
        strengths = gravity_column_capacity_shear({**self.data, "hoops": None})
        self.assertEqual(strengths["ve_kip"], self.data["governing"]["ve_kip"])
        self.assertIsNone(strengths["phi_vn_capacity_kip"])
        self.assertIsNone(strengths["capacity_shear_axis"])
        self.assertEqual(gravity_row(strengths)["status"], "not_evaluated")
        # One direction's record missing: the lighter-direction scalar is not used in its place.
        hoops = self.data["hoops"]
        partial = {**self.data, "hoops": {**hoops, "by_direction": {"x": hoops["by_direction"]["x"]}}}
        self.assertIsNone(gravity_column_capacity_shear(partial)["phi_vn_capacity_kip"])

    def test_a_tie_reports_the_governing_direction(self):
        tied = {"governing": {"axis": "y", "ve_kip": 200.0},
                "worst_by_direction": {"x": {"ve_kip": 100.0}, "y": {"ve_kip": 200.0}},
                "hoops": {"phi_vn_kip": 150.0, "by_direction": {"x": {"phi_vn_kip": 150.0}, "y": {"phi_vn_kip": 300.0}}}}
        self.assertEqual(gravity_column_capacity_shear(tied)["capacity_shear_axis"], "y")


if __name__ == "__main__":
    unittest.main()
