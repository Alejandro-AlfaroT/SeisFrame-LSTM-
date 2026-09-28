"""Slab-aware gravity/mass bookkeeping; no response-history analysis required."""
import contextlib
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
from Geometry_Overrides import apply_geometry_overrides
from Loads.Seismic_ELF import seismic_weight_per_floor
from Design.SMRF_Elastic import gravity_weight_per_story


class SlabLoadTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        values = {
            "NUM_BAY_X": 2, "NUM_BAY_Y": 3, "NUM_FLOOR": 4,
            "BAY_X": 240.0, "BAY_Y": 300.0, "STORY_H": 144.0,
            "B_COL": 24.0, "H_COL": 30.0, "B_BEAM": 18.0, "H_BEAM": 28.0,
            "SLAB_THICKNESS_IN": 8.0, "FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF": 0.05,
            "FLOOR_DEAD_LOAD_KSF": 0.15, "FLOOR_LIVE_LOAD_KSF": 0.05,
            "SEISMIC_LIVE_LOAD_FRACTION": 0.0, "CONCRETE_UNIT_WEIGHT_KCF": 0.150,
            "CONCRETE_UNIT_WEIGHT_KCI": 0.150 / 1728.0,
            "NUM_MODES": sp.NUM_MODES, "GEOMETRY_VARIANT_NAME": getattr(sp, "GEOMETRY_VARIANT_NAME", "baseline"),
        }
        for name, value in values.items():
            self.stack.enter_context(mock.patch.object(sp, name, value, create=True))

    def tearDown(self):
        self.stack.close()

    def nodes(self):
        return [(i, j) for i in range(sp.NUM_BAY_X + 1) for j in range(sp.NUM_BAY_Y + 1)]

    def test_dead_load_replaces_bundled_allowance_instead_of_adding(self):
        sp.SLAB_THICKNESS_IN = 6.0
        self.assertAlmostEqual(sp.slab_self_weight_ksf(), 0.075)
        self.assertAlmostEqual(sp.floor_dead_load_ksf(), 0.125)
        self.assertAlmostEqual(sp.floor_load_ksi(), 0.175 / 144)
        self.assertAlmostEqual(sp.seismic_combination_floor_factor(), (1.2 * 0.125 + 0.5 * 0.05) / 0.175)
        sp.FLOOR_DEAD_LOAD_KSF = 99.0
        self.assertAlmostEqual(sp.floor_dead_load_ksf(), 0.125)

    def test_hand_computed_drop_clearspan_and_column_weights(self):
        unit_weight = 0.150 / 1728
        column = unit_weight * 24 * 30 * (144 - 8)
        beam_x = unit_weight * 18 * (28 - 8) * (240 - 30)
        beam_y = unit_weight * 18 * (28 - 8) * (300 - 24)
        self.assertAlmostEqual(sp.col_self_weight_kip_per_in() * 144, column)
        self.assertAlmostEqual(sp.beam_self_weight_kip_per_in("x") * 240, beam_x)
        self.assertAlmostEqual(sp.beam_self_weight_kip_per_in("y") * 300, beam_y)
        self.assertAlmostEqual(sp.total_structural_self_weight_per_floor(), 12 * column + 8 * beam_x + 9 * beam_y)
        self.assertAlmostEqual(sp.node_structural_self_weight_kip(0, 0), column + (beam_x + beam_y) / 2)
        self.assertAlmostEqual(sp.node_structural_self_weight_kip(1, 1), column + beam_x + beam_y)

    def test_member_mass_and_elf_conservation(self):
        sp.SEISMIC_LIVE_LOAD_FRACTION = 0.25
        self.assertAlmostEqual(sum(sp.node_structural_self_weight_kip(i, j) for i, j in self.nodes()),
                               sp.total_structural_self_weight_per_floor())
        nodal_weight = sum(sp.node_seismic_mass(i, j) * sp.G for i, j in self.nodes())
        self.assertAlmostEqual(nodal_weight, sp.total_floor_seismic_weight())
        self.assertEqual(seismic_weight_per_floor(), [sp.total_floor_seismic_weight()] * 4)
        area_sqft = 240 * 2 * 300 * 3 / 144
        self.assertAlmostEqual(nodal_weight, (0.15 + 0.25 * 0.05) * area_sqft
                               + sp.total_structural_self_weight_per_floor())

    def test_gravity_does_not_include_member_weight_twice(self):
        area_load = sum(sp.node_gravity_load_kip(i, j) for i, j in self.nodes())
        self.assertAlmostEqual(area_load, sp.total_floor_gravity_load())
        self.assertAlmostEqual(gravity_weight_per_story(), area_load + sp.total_structural_self_weight_per_floor())
        x_floor = sum(-sp.beam_gravity_wz_kip_per_in("x", j) / 2 * sp.BAY_X * sp.NUM_BAY_X
                      for j in range(sp.NUM_BAY_Y + 1))
        y_floor = sum(-sp.beam_gravity_wz_kip_per_in("y", i) / 2 * sp.BAY_Y * sp.NUM_BAY_Y
                      for i in range(sp.NUM_BAY_X + 1))
        self.assertAlmostEqual(x_floor + y_floor, area_load)

    def test_applied_gravity_totals_in_both_distribution_modes(self):
        from Loads import Gravity_Loads as loads
        n_col = sp.NUM_FLOOR * (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1)
        n_bx = sp.NUM_FLOOR * sp.NUM_BAY_X * (sp.NUM_BAY_Y + 1)
        floor_factor, dead_factor = 1.3, 1.2
        expected = sp.NUM_FLOOR * (floor_factor * sp.total_floor_gravity_load()
                                   + dead_factor * sp.total_structural_self_weight_per_floor())
        for mode in ("nodal", "beam_uniform"):
            with self.subTest(mode=mode), mock.patch.object(sp, "GRAVITY_LOAD_MODEL", mode), \
                 mock.patch.object(loads.ops, "timeSeries"), mock.patch.object(loads.ops, "pattern"), \
                 mock.patch.object(loads.ops, "load") as nodal, mock.patch.object(loads.ops, "eleLoad") as element:
                loads.apply_gravity_loads(floor_factor, dead_factor)
                weight = sum(-call.args[3] for call in nodal.call_args_list)
                for call in element.call_args_list:
                    args = call.args
                    tag = args[1]
                    length = sp.STORY_H if tag <= n_col else sp.BAY_X if tag <= n_col + n_bx else sp.BAY_Y
                    weight -= (args[5] + args[6]) * length
                self.assertAlmostEqual(weight, expected)

    def test_thickness_updates_load_and_mass_without_changing_sections(self):
        before = sp.total_floor_seismic_weight()
        sp.SLAB_THICKNESS_IN = 9.0
        self.assertGreater(sp.total_floor_seismic_weight(), before)
        self.assertEqual(sp.H_BEAM, 28.0)
        self.assertEqual(sp.H_COL, 30.0)
        self.assertAlmostEqual(sum(sp.node_seismic_mass(i, j) * sp.G for i, j in self.nodes()),
                               sp.total_floor_seismic_weight())

    def test_legacy_load_and_mass_values_are_retained(self):
        sp.SLAB_THICKNESS_IN = None
        sp.FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF = float("nan")
        sp.SEISMIC_LIVE_LOAD_FRACTION = float("nan")
        self.assertEqual(sp.floor_dead_load_ksf(), 0.15)
        self.assertIsNone(sp.slab_self_weight_ksf())
        self.assertAlmostEqual(sp.floor_load_ksi(), 0.20 / 144)
        self.assertAlmostEqual(sp.beam_self_weight_kip_per_in(), sp.CONCRETE_UNIT_WEIGHT_KCI * 18 * 28)
        self.assertAlmostEqual(sp.col_self_weight_kip_per_in(), sp.CONCRETE_UNIT_WEIGHT_KCI * 24 * 30)
        for i, j in self.nodes():
            self.assertAlmostEqual(sp.node_seismic_mass(i, j) * sp.G, sp.node_gravity_load_kip(i, j))
        self.assertEqual(sp.total_floor_seismic_weight(), sp.total_floor_gravity_load())
        self.assertEqual(sp.floor_load_metadata()["seismic_live_load_fraction"], 1.0)

    def test_unequal_y_span_in_member_total_is_not_replaced_by_x_span(self):
        sp.SLAB_THICKNESS_IN = None
        self.assertAlmostEqual(sp.total_structural_self_weight_per_floor(),
                               12 * 144 * sp.col_self_weight_kip_per_in()
                               + (8 * 240 + 9 * 300) * sp.beam_self_weight_kip_per_in())

    def test_new_mode_requires_an_explicit_beam_axis(self):
        for axis in (None, "z", "X"):
            with self.subTest(axis=axis), self.assertRaisesRegex(ValueError, "beam_axis"):
                sp.beam_self_weight_kip_per_in(axis)

    def test_invalid_selected_state_is_rejected(self):
        invalid = {
            "SLAB_THICKNESS_IN": [0, -1, 28, 29, float("nan"), float("inf")],
            "FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF": [-0.1, float("nan")],
            "SEISMIC_LIVE_LOAD_FRACTION": [-0.1, 1.1, float("nan")],
            "FLOOR_LIVE_LOAD_KSF": [-1, float("inf")],
            "H_BEAM": [144, 145, 7], "H_COL": [240], "B_COL": [300],
            "CONCRETE_UNIT_WEIGHT_KCF": [0, float("nan")],
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value), mock.patch.object(sp, name, value):
                    with self.assertRaises(ValueError):
                        sp.floor_dead_load_ksf()
        for fraction in (0.0, 1.0):
            with mock.patch.object(sp, "SEISMIC_LIVE_LOAD_FRACTION", fraction):
                self.assertTrue(math.isfinite(sp.total_floor_seismic_weight()))

    def test_geometry_change_invalidates_slab_not_policy(self):
        apply_geometry_overrides({"BAY_X": sp.BAY_X}, emit=False)
        self.assertEqual(sp.SLAB_THICKNESS_IN, 8.0)
        apply_geometry_overrides({"BAY_X": 264.0}, emit=False)
        self.assertIsNone(sp.SLAB_THICKNESS_IN)
        self.assertEqual(sp.FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF, 0.05)
        self.assertEqual(sp.SEISMIC_LIVE_LOAD_FRACTION, 0.0)

    def test_metadata_records_accounting_basis_and_all_weight_parts(self):
        result = sp.floor_load_metadata()
        self.assertEqual(result["load_model"], "slab_aware_v1")
        self.assertEqual(result["slab_thickness_in"], 8.0)
        self.assertAlmostEqual(result["floor_dead_load_ksf"], 0.15)
        self.assertAlmostEqual(result["member_self_weight_per_floor_kip"],
                               result["column_self_weight_per_floor_kip"]
                               + result["beam_x_self_weight_per_floor_kip"]
                               + result["beam_y_self_weight_per_floor_kip"])
        self.assertIn("not a physical quantity takeoff", result["load_accounting_basis"])
        self.assertEqual(result["total_floor_seismic_weight_kip"], sp.total_floor_seismic_weight())

    def test_elastic_model_base_reactions_balance_slab_aware_gravity(self):
        from Design import Design_Driver as driver
        from Design.SMRF_Elastic import build_design_model
        from Loads.Gravity_Loads import apply_gravity_loads
        from Analysis.Gravity import run_gravity_analysis
        from Model.Build_Model import node_tag
        sp.NUM_FLOOR = 1
        expected = sp.total_floor_gravity_load() + sp.total_structural_self_weight_per_floor()
        for mode in ("nodal", "beam_uniform"):
            with self.subTest(mode=mode), mock.patch.object(sp, "GRAVITY_LOAD_MODEL", mode):
                try:
                    with driver._quiet():
                        build_design_model()
                        apply_gravity_loads()
                        run_gravity_analysis()
                        driver.ops.reactions()
                    upward = sum(driver.ops.nodeReaction(node_tag(0, i, j), 3) for i, j in self.nodes())
                    self.assertAlmostEqual(upward, expected, places=6)
                finally:
                    driver.ops.wipe()


if __name__ == "__main__":
    unittest.main()
