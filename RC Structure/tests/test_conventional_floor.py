"""Hand-value tests of the floor actions, torsion and diaphragm rules (Design.SMRF_Conventional_Floor). No solve."""
import math
import unittest

import Structure_Parameters as sp
from Design import Group_Checks as checks
from Design import SMRF_Capacity_Design as cd
from Design import SMRF_Conventional_Floor as floor


class SlabAcceptance(unittest.TestCase):
    GOOD = {"method": "plate_finite_element", "cases": [{"id": "1.4D"}], "pattern_rule": "6.4.3.2",
            "equilibrium": [{"numerical_balance_passed": True}],
            "numerical_preconditions": {"all_floors_covered_once": True, "same_slab_inputs_and_cases": True,
                                        "physical_recovery_valid": True, "mesh_refinement_verified": True}}

    def test_complete_evidence_has_no_problem(self):
        self.assertEqual(floor.slab_acceptance_problems(self.GOOD), [])

    def test_each_missing_condition_is_named(self):
        for key in ("all_floors_covered_once", "same_slab_inputs_and_cases", "physical_recovery_valid", "mesh_refinement_verified"):
            evidence = {**self.GOOD, "numerical_preconditions": {**self.GOOD["numerical_preconditions"], key: False}}
            self.assertEqual(len(floor.slab_acceptance_problems(evidence)), 1, key)

    def test_unbalanced_case_blocks(self):
        evidence = {**self.GOOD, "equilibrium": [{"numerical_balance_passed": False}]}
        self.assertTrue(floor.slab_acceptance_problems(evidence))

    def test_no_case_blocks(self):
        self.assertTrue(floor.slab_acceptance_problems({**self.GOOD, "cases": []}))


class Torques(unittest.TestCase):
    TRANSFER = {"unit_cases": {"dead": {"beams": [{"axis": "x", "line_index": 0, "span_index": 1,
                                                   "node_couples": [[0.25, 8.0, 0.0], [0.75, 4.0, 0.0]]}]},
                               "live": {"beams": [{"axis": "x", "line_index": 0, "span_index": 1,
                                                   "node_couples": [[0.5, 2.0, 0.0]]}]}}}

    def test_couples_split_by_position(self):
        unit = floor._unit_end_torques(self.TRANSFER, ("x", 0, 1))
        self.assertAlmostEqual(unit["dead"][0], 8.0 * 0.75 + 4.0 * 0.25)
        self.assertAlmostEqual(unit["dead"][1], 8.0 * 0.25 + 4.0 * 0.75)
        self.assertAlmostEqual(unit["live"][0], 1.0)

    def test_member_torque_adds_factored_plate_couples_to_the_frame(self):
        unit = floor._unit_end_torques(self.TRANSFER, ("x", 0, 1))
        record = {"local_force_kip_kipin": [0, 0, 0, 10.0, 0, 0, 0, 0, 0, -3.0, 0, 0]}
        ti, tj = floor.member_torque(record, {"dead": 1.2, "live": 1.6, "live_pattern": "all"}, unit)
        self.assertAlmostEqual(ti, 10.0 + 1.2 * 7.0 + 1.6 * 1.0)
        self.assertAlmostEqual(tj, 3.0 + 1.2 * 5.0 + 1.6 * 1.0)

    def test_no_live_factor_ignores_live(self):
        unit = floor._unit_end_torques(self.TRANSFER, ("x", 0, 1))
        ti, _ = floor.member_torque({"local_force_kip_kipin": [0.0] * 12}, {"dead": 1.4, "live": 0.0}, unit)
        self.assertAlmostEqual(ti, 1.4 * 7.0)


class TorsionGeometry(unittest.TestCase):
    def test_hoop_centerline_core(self):
        geometry = floor.torsion_geometry(16.0, 30.0)
        cover = sp.BEAM_CLEAR_COVER_IN
        self.assertAlmostEqual(geometry["x1_in"], 16.0 - 2.0 * cover - 0.625)
        self.assertAlmostEqual(geometry["y1_in"], 30.0 - 2.0 * cover - 0.625)
        self.assertAlmostEqual(geometry["ao_in2"], 0.85 * geometry["aoh_in2"])
        self.assertAlmostEqual(geometry["ph_in"], 2.0 * (geometry["x1_in"] + geometry["y1_in"]))


class EdgeRelease(unittest.TestCase):
    LAYOUT = {"layers": {f"{axis}_{face}": {"flexure": {"phi_mn_kip_in_per_ft": 100.0}} for axis in ("x", "y") for face in ("top", "bottom")}}

    def _member(self, width=16.0):
        class Design:
            b_in = width

        class Member:
            design = Design()
        return Member()

    def test_strip_method_sum(self):
        saved = sp.BAY_X, sp.BAY_Y, sp.FLOOR_LIVE_LOAD_KSF
        try:
            sp.BAY_X, sp.BAY_Y, sp.FLOOR_LIVE_LOAD_KSF = 300.0, 300.0, 0.05
            result = floor.edge_release_margin(self._member(), self.LAYOUT)
            wu = (1.2 * sp.floor_dead_load_ksf() + 1.6 * 0.05) / 144.0
            ln = 300.0 - 16.0
            one = 150.0 / (wu * 12.0 * ln * ln / 8.0)
            self.assertAlmostEqual(result["ratios"]["x"], one)
            self.assertAlmostEqual(result["sum_of_ratios"], 2.0 * one)
        finally:
            sp.BAY_X, sp.BAY_Y, sp.FLOOR_LIVE_LOAD_KSF = saved

    def test_no_layout_gives_no_margin(self):
        self.assertIsNone(floor.edge_release_margin(self._member(), None))


class Collector(unittest.TestCase):
    LEVEL = {"fpx_kip": 300.0, "fpx_max_kip": 800.0}

    def test_overstrength_on_the_larger_of_fx_and_fpx_without_a_cap(self):
        # ASCE 7-22 12.10.2.1 as printed has no Eq. 12.10-3 cap on the maximum (the limit sits inside Fpx).
        self.assertAlmostEqual(floor.collector_force_kip(self.LEVEL, 100.0), 900.0)
        self.assertAlmostEqual(floor.collector_force_kip({"fpx_kip": 200.0, "fpx_max_kip": 800.0}, 350.0), 1050.0)

    def test_axial_is_half_a_bay_of_shear_flow(self):
        # 400 kip on a three-bay line: 400 / (3 a) x a / 2 = 66.7 kip.
        self.assertAlmostEqual(floor.collector_axial_kip(400.0, 3), 400.0 / 6.0)


class Demands(unittest.TestCase):
    def _action(self, floor_actions=None):
        member = {"local_force_kip_kipin": [5.0, 2.0, 3.0, 0, 0, 0, -4.0, 1.0, 1.0, 0, 0, 0],
                  "span_bending": {"full_span": {"mu_positive_kip_in": 100.0, "mu_negative_kip_in": 200.0}}}
        if floor_actions:
            member["floor_actions"] = floor_actions
        return {"id": "c1", "members": {"7": member}}

    def test_floor_actions_add_to_the_section_demand(self):
        rows = checks.beam_demand_rows([7], [self._action({"mu_slab_hogging_kip_in": 50.0, "mu_slab_sagging_kip_in": 20.0,
                                                           "torsion_face_moment_kip_in": 5.0, "n_diaphragm_kip": 10.0})])
        tag, cid, mu_pos, mu_neg, vu, n_min, n_max = rows[0]
        self.assertEqual((tag, cid), (7, "c1"))
        self.assertAlmostEqual(mu_pos, 125.0)
        self.assertAlmostEqual(mu_neg, 255.0)
        self.assertAlmostEqual(n_min, 4.0 - 10.0)
        self.assertAlmostEqual(n_max, 5.0 + 10.0)
        self.assertAlmostEqual(vu, math.hypot(2.0, 3.0))

    def test_without_floor_actions_the_row_is_unchanged(self):
        rows = checks.beam_demand_rows([7], [self._action()])
        self.assertAlmostEqual(rows[0][2], 100.0)
        self.assertAlmostEqual(rows[0][3], 200.0)
        self.assertAlmostEqual(rows[0][5], 4.0)


class Hoops(unittest.TestCase):
    def _select(self, **torsion):
        return cd.select_beam_hoops(16.0, 26.0, 5.0, 60.0, 9, 1.5, 3, 3, 100.0, 80.0, grid_in=None, **torsion)

    def test_zero_torsion_changes_nothing(self):
        self.assertEqual(self._select(), self._select(torsion_at_per_in=0.0))

    def test_torsion_steel_reduces_the_spacing(self):
        plain = self._select()["hoops"]
        with_torsion = self._select(torsion_at_per_in=0.05)["hoops"]
        self.assertTrue(with_torsion["spacing_in"] < plain["spacing_in"] or
                        (with_torsion["bar_size"], with_torsion["legs"]) != (plain["bar_size"], plain["legs"]))
        vs = 100.0 / cd.PHI_SHEAR
        per_leg = vs / (60.0 * 26.0 * with_torsion["legs"])
        limit = sp.rebar_area(with_torsion["bar_size"]) / (0.05 + per_leg)
        self.assertLessEqual(with_torsion["spacing_in"], limit + 1e-9)

    def test_torsion_spacing_limit_applies(self):
        limited = self._select(torsion_at_per_in=1e-6, torsion_spacing_limit_in=3.0)["hoops"]
        self.assertLessEqual(limited["spacing_in"], 3.0)


class Inventory(unittest.TestCase):
    def test_rows_of_the_floor_family(self):
        rows = floor.floor_row_inventory()
        self.assertEqual(len(rows), 3 + 6 * 2 + 4 + 3)
        self.assertTrue(all(family == "floor" for family, _name, _location in rows))
        self.assertIn(("floor", "diaphragm.in_plane_shear", "x"), rows)
        self.assertIn(("floor", "diaphragm.maximum_bar_spacing", "y_bottom"), rows)


if __name__ == "__main__":
    unittest.main()
