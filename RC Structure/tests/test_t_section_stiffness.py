"""Frame beams bend on the T/L section of their line (ACI 318-19 R6.6.3.1.1, 6.3.2 flange).

Hand values are case_0144 of the 2026-09-16 verification set (dv150_v9):
6 x 2 bays at 168 x 132 in, 24x24 columns, 10x18 beams, 5-in slab. The
rectangular web was 5.45% off the monolithic slab/frame reference on column
vertical reactions; the 6.3.2 T/L section brought it to 1.16%.
"""
import contextlib
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import openseespy.opensees as ops  # noqa: E402
import Structure_Parameters as sp  # noqa: E402
from Design.SMRF_Beam_Slab_Strength import effective_flange_width  # noqa: E402
from Design.SMRF_Elastic import beam_line_family, build_design_model  # noqa: E402
from Model import IMK_Hinges  # noqa: E402
from Model.Build_Model import node_tag  # noqa: E402

CASE_0144 = {"NUM_BAY_X": 6, "NUM_BAY_Y": 2, "NUM_FLOOR": 8, "BAY_X": 168.0, "BAY_Y": 132.0, "STORY_H": 120.0,
             "B_COL": 24.0, "H_COL": 24.0, "B_BEAM": 10.0, "H_BEAM": 18.0, "FC_COL_KSI": 4.0, "FC_BEAM_KSI": 6.0,
             "SLAB_THICKNESS_IN": 5.0}


def rect_iy(b, h):
    return b * h ** 3 / 12.0


class SectionTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        for name, value in CASE_0144.items():
            self.stack.enter_context(mock.patch.object(sp, name, value, create=True))

    def tearDown(self):
        self.stack.close()

    def test_table_6_3_2_1_flange_widths(self):
        # x beams: clear span 168 - 24 = 144 -> ln/8 = 18 governs interior, ln/12 = 12 the edge
        self.assertEqual(sp.effective_flange_width_in(10.0, 5.0, 144.0, 122.0, 2), (46.0, 18.0))
        self.assertEqual(sp.effective_flange_width_in(10.0, 5.0, 144.0, 122.0, 1), (22.0, 12.0))
        # y beams: clear span 132 - 24 = 108 -> 13.5 and 9
        self.assertEqual(sp.effective_flange_width_in(10.0, 5.0, 108.0, 158.0, 2), (37.0, 13.5))
        self.assertEqual(sp.effective_flange_width_in(10.0, 5.0, 108.0, 158.0, 1), (19.0, 9.0))
        # 8h and half the clear web are the other two bounds
        self.assertEqual(sp.effective_flange_width_in(10.0, 5.0, 1000.0, 1000.0, 2), (90.0, 40.0))
        self.assertEqual(sp.effective_flange_width_in(10.0, 5.0, 1000.0, 30.0, 2), (40.0, 15.0))
        with self.assertRaises(ValueError):
            sp.effective_flange_width_in(10.0, 5.0, 144.0, 122.0, 3)
        # The strength module's flange is the same function.
        self.assertEqual(effective_flange_width(10.0, 5.0, 144.0, 122.0, 2), (46.0, 18.0))

    def test_t_section_inertia(self):
        # bf = bw is the rectangular section
        self.assertAlmostEqual(sp.t_section_inertia_in4(10.0, 18.0, 5.0, 10.0), rect_iy(10.0, 18.0))
        # Hand calculation for the 46-in interior flange: areas 230 and 130 in2,
        # centroid 2.5 and 11.5 in from the top -> 5.75 in; parallel-axis sum.
        flange, web = 46.0 * 5.0, 10.0 * 13.0
        centroid = (flange * 2.5 + web * 11.5) / (flange + web)
        by_hand = (46.0 * 125.0 / 12.0 + flange * (2.5 - centroid) ** 2
                   + 10.0 * 13.0 ** 3 / 12.0 + web * (11.5 - centroid) ** 2)
        self.assertAlmostEqual(sp.t_section_inertia_in4(10.0, 18.0, 5.0, 46.0), by_hand)
        self.assertAlmostEqual(by_hand, 9038.0, delta=1.0)
        self.assertGreater(by_hand / rect_iy(10.0, 18.0), 1.8)            # R6.6.3.1.1: about 2x the web
        with self.assertRaises(ValueError):
            sp.t_section_inertia_in4(10.0, 5.0, 5.0, 46.0)
        with self.assertRaises(ValueError):
            sp.t_section_inertia_in4(10.0, 18.0, 5.0, 8.0)

    def test_beam_family_sections(self):
        expected = {("x", "edge"): 22.0, ("x", "interior"): 46.0, ("y", "edge"): 19.0, ("y", "interior"): 37.0}
        for (axis, position), bf in expected.items():
            section = sp.beam_flexural_section(axis, position)
            self.assertEqual(section["flange_width_in"], bf)
            self.assertAlmostEqual(section["iy_in4"], sp.t_section_inertia_in4(10.0, 18.0, 5.0, bf))
            self.assertIn("6.3.2", section["basis"])
        self.assertAlmostEqual(sp.beam_flexural_inertia_in4("x", "interior"), 9038.0, delta=1.0)
        self.assertAlmostEqual(sp.beam_flexural_inertia_in4("y", "edge"), 6475.0, delta=1.0)
        with self.assertRaises(ValueError):
            sp.beam_flexural_section("z", "edge")
        with self.assertRaises(ValueError):
            sp.beam_flexural_section("x", "middle")

    def test_no_slab_is_the_rectangular_web(self):
        with mock.patch.object(sp, "SLAB_THICKNESS_IN", None):
            section = sp.beam_flexural_section("x", "interior")
        self.assertAlmostEqual(section["iy_in4"], rect_iy(10.0, 18.0))
        self.assertEqual(section["flange_width_in"], 10.0)
        self.assertIn("rectangular", section["basis"])


class FrameTests(unittest.TestCase):
    SMALL = {"NUM_BAY_X": 2, "NUM_BAY_Y": 2, "NUM_FLOOR": 2, "BAY_X": 180.0, "BAY_Y": 150.0, "STORY_H": 144.0,
             "B_COL": 20.0, "H_COL": 20.0, "B_BEAM": 12.0, "H_BEAM": 20.0, "FC_COL_KSI": 4.0, "FC_BEAM_KSI": 4.0,
             "SLAB_THICKNESS_IN": 5.0, "FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF": 0.05, "FLOOR_LIVE_LOAD_KSF": 0.05,
             "SEISMIC_LIVE_LOAD_FRACTION": 0.0, "GRAVITY_LOAD_MODEL": "nodal", "FLOOR_TRANSFER": None}

    def setUp(self):
        self.stack = contextlib.ExitStack()
        for name, value in self.SMALL.items():
            self.stack.enter_context(mock.patch.object(sp, name, value, create=True))

    def tearDown(self):
        self.stack.close()
        ops.wipe()

    def test_beam_line_family_from_the_node_grid(self):
        per_story = 9
        self.assertEqual(beam_line_family(node_tag(1, 0, 0), "beam_x"), ("x", "edge"))
        self.assertEqual(beam_line_family(node_tag(1, 0, 1), "beam_x"), ("x", "interior"))
        self.assertEqual(beam_line_family(node_tag(2, 1, 2), "beam_x"), ("x", "edge"))
        self.assertEqual(beam_line_family(node_tag(1, 0, 0), "beam_y"), ("y", "edge"))
        self.assertEqual(beam_line_family(node_tag(1, 1, 0), "beam_y"), ("y", "interior"))
        self.assertEqual(beam_line_family(node_tag(2, 2, 1), "beam_y"), ("y", "edge"))
        self.assertEqual(node_tag(2, 0, 0), 2 * per_story + 1)
        with self.assertRaises(ValueError):
            beam_line_family(node_tag(1, 0, 0), "column")

    def _first_period(self):
        with contextlib.redirect_stdout(None):
            build_design_model()
        lam = ops.eigen("-genBandArpack", 1)
        ops.wipe()
        return 2.0 * math.pi / math.sqrt(lam[0])

    def test_design_frame_is_stiffer_with_the_slab_flange(self):
        """The same sections with and without a slab: the T/L frame has the shorter period."""
        with_slab = self._first_period()
        with mock.patch.object(sp, "SLAB_THICKNESS_IN", None):
            rectangular = self._first_period()
        self.assertLess(with_slab, rectangular)
        self.assertLess(with_slab / rectangular, 0.95)                      # 12-17% shorter in the plan cases; 8.6% on this small frame
        self.assertGreater(with_slab / rectangular, 0.75)

    def test_imk_elastic_member_carries_the_line_section(self):
        modifier = sp.section_stiffness_modifier("beam")
        interior = IMK_Hinges._member_properties("beam_x", family=("x", "interior"))
        edge = IMK_Hinges._member_properties("beam_x", family=("x", "edge"))
        self.assertAlmostEqual(interior["iy"], modifier * sp.beam_flexural_inertia_in4("x", "interior"))
        self.assertAlmostEqual(edge["iy"], modifier * sp.beam_flexural_inertia_in4("x", "edge"))
        self.assertGreater(interior["iy"], edge["iy"])
        self.assertGreater(edge["iy"], modifier * rect_iy(12.0, 20.0))
        self.assertEqual(interior["family"], "x_interior")
        self.assertIn("6.3.2", interior["iy_basis"])
        # Without a family the interior line is assumed; iz stays the web.
        self.assertAlmostEqual(IMK_Hinges._member_properties("beam_y")["iy"], modifier * sp.beam_flexural_inertia_in4("y", "interior"))
        self.assertAlmostEqual(interior["iz"], modifier * sp.rect_iz(12.0, 20.0))
        column = IMK_Hinges._member_properties("column", axial_kip=0.0)
        self.assertAlmostEqual(column["iy"], sp.section_stiffness_modifier("column") * rect_iy(20.0, 20.0))


class ExportTests(unittest.TestCase):
    def test_sap_beam_modifiers_carry_the_t_section(self):
        from Design.Export_SAP2000 import Frame, _beam_stiffness_ratios
        record = {"geometry": {"num_bay_x": 6, "num_bay_y": 2, "num_floor": 8, "bay_x_in": 168.0, "bay_y_in": 132.0,
                               "story_h_in": 120.0},
                  "sections": {"b_beam_in": 10.0, "h_beam_in": 18.0, "b_col_in": 24.0, "h_col_in": 24.0},
                  "slab": {"thickness_in": 5.0}}
        ratios = _beam_stiffness_ratios(record, Frame(record))
        self.assertAlmostEqual(ratios[("x", "interior")], 9038.0 / rect_iy(10.0, 18.0), delta=0.001)
        self.assertAlmostEqual(ratios[("y", "edge")], 6475.0 / rect_iy(10.0, 18.0), delta=0.001)
        self.assertTrue(all(r > 1.0 for r in ratios.values()))
        record["slab"] = {"thickness_in": None}
        self.assertEqual(set(_beam_stiffness_ratios(record, Frame(record)).values()), {1.0})


if __name__ == "__main__":
    unittest.main()
