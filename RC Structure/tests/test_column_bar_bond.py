"""ACI 318-19 18.7.4.3 for special-frame column bars (design-basis rule of 2026-10-02, from the code-book review):
1.25 ld <= lu / 2 with ld from Eq. (25.4.2.4a) at the cage's own cb and Ktr = 0. Pinned here: the equation against
the independent screen of the reference record, the cb convention, the caps and minimums, the cage filter, the
capacity check, the planner lever, the grouped selection, and the V2 profile handed through the generation chain.
"""
import argparse
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Structure_Parameters as sp                                              # noqa: E402
import Redesign as rd                                                          # noqa: E402
from Design import ACI_Checks as aci                                           # noqa: E402
from Design import Design_Driver as driver                                     # noqa: E402
from Design import Section_Design                                              # noqa: E402
from Design import SMRF_Capacity_Design as cd                                  # noqa: E402
from Design.Config import DesignConfig                                         # noqa: E402

V2 = "v2_nonlinear_flexure_screening_v1"


def frame(story_h=168.0, beam_h=30.0, fc=5.0, hoop=5):
    """A 42-in column with the given hoops over a story of the given height and beam depth (slab-aware path)."""
    return mock.patch.multiple(
        sp, B_COL=42.0, H_COL=42.0, FC_COL_KSI=fc, FY_KSI=60.0, COL_CLEAR_COVER_IN=1.5, COL_STIRRUP_BAR_SIZE=hoop,
        B_BEAM=24.0, H_BEAM=beam_h, STORY_H=story_h, BEAM_CLEAR_COVER_IN=1.5, BEAM_STIRRUP_BAR_SIZE=4,
        BEAM_BAR_SIZE=9, BEAM_TOP_BARS=4, BEAM_BOT_BARS=4, AGGREGATE_MAX_SIZE_IN=0.75, SLAB_THICKNESS_IN=6.0,
        BEAM_BAR_MAX_LAYERS=2)


class Equation(unittest.TestCase):
    def test_reproduces_the_independent_screen_of_the_reference_record(self):
        """Reference record of 2 October 2026: No. 10 bars, No. 5 hoops, 1.5 in cover, f'c 5 ksi, Grade 60:
        cb 2.760 in, (cb + Ktr)/db 2.17323, ld 37.18997 in, 1.25 ld 46.48747 in against lu/2 = 66 in."""
        cb = aci.column_bar_cb_in(40.0, 40.0, 1.5, 5, 10, 4, 2)
        self.assertAlmostEqual(cb["cb_in"], 2.760, places=9)
        self.assertEqual(cb["governed_by"], "edge distance")
        bond = aci.column_bar_bond_18_7_4_3(10, 5.0, 60.0, cb["cb_in"], 132.0)
        self.assertAlmostEqual(bond["confinement_ratio"], 2.17323, places=5)
        self.assertAlmostEqual(bond["ld_in"], 37.18997, places=4)
        self.assertAlmostEqual(bond["factored_ld_in"], 46.48747, places=4)
        self.assertAlmostEqual(bond["half_clear_height_in"], 66.0, places=12)
        self.assertTrue(bond["passes"])
        self.assertAlmostEqual(bond["dcr"], 46.48747 / 66.0, places=5)
        self.assertEqual((bond["psi_t"], bond["psi_e"], bond["psi_s"], bond["psi_g"], bond["lambda"]), (1.0, 1.0, 1.0, 1.0, 1.0))
        self.assertFalse(bond["confinement_ratio_capped"])
        # The Table 25.4.2.3 expression the transitions module uses is the conservative method, not this one.
        self.assertAlmostEqual(1.25 * cd.straight_development_length(10, 5.0, 60.0), 67.35192, places=4)

    def test_half_the_bar_spacing_governs_a_crowded_face(self):
        cb = aci.column_bar_cb_in(20.0, 20.0, 1.5, 4, 8, 6, 4)             # pitch (20 - 2 x 2.5) / 5 = 3.0 in
        self.assertEqual(cb["governed_by"], "half the bar spacing")
        self.assertAlmostEqual(cb["cb_in"], 1.5, places=12)

    def test_caps_minimum_and_factors(self):
        capped = aci.column_bar_bond_18_7_4_3(8, 5.0, 60.0, 4.0, 120.0)             # 4.0 / 1.0 = 4 > 2.5
        self.assertTrue(capped["confinement_ratio_capped"])
        self.assertEqual(capped["confinement_ratio"], 2.5)
        self.assertAlmostEqual(capped["ld_in"], 0.075 * 60000.0 / (70.71067811865476 * 2.5) * 1.0, places=9)
        strong = aci.column_bar_bond_18_7_4_3(8, 12.0, 60.0, 4.0, 120.0)            # sqrt(12000) = 109.5 > 100 psi
        self.assertTrue(strong["sqrt_fc_capped"])
        self.assertEqual(strong["sqrt_fc_psi"], 100.0)
        tiny = aci.column_bar_bond_18_7_4_3(3, 8.0, 60.0, 3.0, 120.0)
        self.assertTrue(tiny["minimum_ld_applied"])
        self.assertEqual(tiny["ld_in"], 12.0)
        self.assertEqual(aci.column_bar_bond_18_7_4_3(6, 5.0, 60.0, 2.0, 120.0)["psi_s"], 0.8)
        self.assertEqual(aci.column_bar_bond_18_7_4_3(7, 5.0, 60.0, 2.0, 120.0)["psi_s"], 1.0)
        self.assertEqual(aci.column_bar_bond_18_7_4_3(8, 5.0, 80.0, 2.0, 120.0)["psi_g"], 1.15)
        with self.assertRaisesRegex(ValueError, "psi_g"):
            aci.column_bar_bond_18_7_4_3(8, 5.0, 75.0, 2.0, 120.0)
        failing = aci.column_bar_bond_18_7_4_3(18, 4.0, 60.0, 3.0, 120.0)           # No. 18 at 4 ksi in a 10-ft clear story
        self.assertFalse(failing["passes"])
        self.assertGreater(failing["dcr"], 1.0)


class CageSelection(unittest.TestCase):
    def test_a_bar_that_cannot_develop_in_half_the_clear_height_is_never_offered(self):
        cfg = DesignConfig()
        with frame(story_h=144.0, beam_h=30.0, fc=4.0):
            self.assertTrue(rd._column_cage_bond_ok(9, 4, 2))
            self.assertFalse(rd._column_cage_bond_ok(14, 4, 2), "1.25 ld of a No. 14 at 4 ksi exceeds 57 in")
            self.assertFalse(rd._column_cage_bond_ok(18, 4, 2))
            ag = 42.0 * 42.0
            offered = rd._col_candidates(0.01 * ag, 0.06 * ag, cfg)
            self.assertTrue(offered)
            self.assertFalse({c[0] for c in offered} & {14, 18})
            self.assertTrue(rd.column_cages_that_bond_exist(cfg))
        with frame(story_h=144.0, beam_h=30.0, fc=4.0), mock.patch.object(sp, "SLAB_THICKNESS_IN", None):
            self.assertTrue(rd.column_cages_that_bond_exist(cfg), "the legacy path is untouched")

    def test_no_cage_at_all_when_the_story_is_too_short_for_every_declared_bar(self):
        cfg = DesignConfig()
        with frame(story_h=96.0, beam_h=36.0, fc=4.0):                           # 60 in clear: half is 30 in
            self.assertTrue(rd.column_cages_that_bond_exist(cfg), "No. 7 and smaller still develop in 30 in")
            self.assertFalse({c[0] for c in rd._col_candidates(0.01 * 42.0 ** 2, 0.06 * 42.0 ** 2, cfg)} & {11, 14})
        with frame(story_h=60.0, beam_h=36.0, fc=4.0):                           # 24 in clear: half is 12 in, below 1.25 x 12
            self.assertFalse(rd.column_cages_that_bond_exist(cfg))
            self.assertEqual(rd._col_candidates(0.01 * 42.0 ** 2, 0.06 * 42.0 ** 2, cfg), [])


class CapacityCheckAndPlanner(unittest.TestCase):
    def state(self, story_h=168.0, beam_h=30.0, clear_heights=None):
        state = {"geometry": {"story_h_in": story_h, "num_floor": 6},
                 "sections": {"b_col_in": 42.0, "h_col_in": 42.0, "fc_col_ksi": 5.0, "b_beam_in": 24.0, "h_beam_in": beam_h, "fc_beam_ksi": 4.0},
                 "materials": {"fy_ksi": 60.0, "es_ksi": 29000.0, "normalweight": True},
                 "column": {"bar_size": 10, "top_bars": 4, "bot_bars": 4, "side_bars": 2, "clear_cover_in": 1.5, "stirrup_bar_size": 5}}
        if clear_heights is not None:
            state["clear_heights_in"] = clear_heights
        return state

    def test_the_uniform_evidence_uses_the_typical_story_and_the_grouped_evidence_the_least_member(self):
        bond = cd.design_column_bar_bond(self.state())
        self.assertAlmostEqual(bond["clear_height_in"], 138.0, places=12)
        self.assertAlmostEqual(bond["factored_ld_in"], 46.48747, places=4)
        self.assertTrue(bond["passes"])
        self.assertIn("typical story", bond["clear_height_basis"])
        self.assertEqual(bond["ktr_in"], 0.0)
        self.assertIn("318-25", bond["edition_note"])
        grouped = cd.design_column_bar_bond(self.state(clear_heights={1: {"face": 138.0, "physical": 153.0}, 2: {"face": 135.0, "physical": 135.0}}))
        self.assertAlmostEqual(grouped["clear_height_in"], 135.0, places=12)
        short = cd.design_column_bar_bond(self.state(story_h=120.0, beam_h=36.0))        # 84 in clear: half is 42 in
        self.assertFalse(short["passes"])

    def test_the_planner_steps_the_column_and_the_reason_is_declared(self):
        columns = Section_Design.column_ladder()
        beams = Section_Design.beam_ladder(span_in=240.0, story_height_in=168.0)
        ci = next(i for i, r in enumerate(columns) if r == (32.0, 32.0, 5.0))
        bi = next(i for i, r in enumerate(beams) if r[1] == 24.0)
        flags = {"drift_ok": True, "scwb_ok": True, "joint_scwb_failed": False, "capacity_accepted": False,
                 "beam_section_adequate": True, "beam_hoops_selected": True, "beam_bars_thread": True,
                 "column_section_adequate": True, "column_hoops_selected": True, "joints_all_pass": True,
                 "anchorage_all_pass": True, "joint_shear_ratio": 0.5, "column_bar_bond_ok": False}
        next_column, next_beam, reasons = driver._plan_next_rungs(columns, beams, ci, bi, {"beam": 0.8, "column": 0.4},
                                                                  0.85, 1.0, ci, flags)
        self.assertIn("column_bar_bond", reasons)
        self.assertIn("column_bar_bond", driver.COLUMN_STEP_REASONS)
        self.assertGreater(next_column, ci)
        self.assertEqual(next_beam, bi)
        _c, _b, quiet = driver._plan_next_rungs(columns, beams, ci, bi, {"beam": 0.8, "column": 0.4}, 0.85, 1.0, ci,
                                                {**flags, "column_bar_bond_ok": True, "capacity_accepted": True})
        self.assertNotIn("column_bar_bond", quiet)


class ProfileThroughGeneration(unittest.TestCase):
    def test_the_parent_hands_the_profile_to_the_child_and_the_child_to_every_run(self):
        from Data_Generation import Generate_Hybrid_Dataset as child
        from Data_Generation import Generate_Parameterized_Dataset as parent
        case = {"geometry_name": "g", "num_bay_x": 2, "num_bay_y": 2, "num_floor": 4, "bay_x_in": 240.0, "bay_y_in": 240.0,
                "story_height_in": 168.0, "seismic_site": "sdc_d_low", "result_ids": [11],
                "runs": [{"run_index": 1, "result_id": 11, "scale_factor": 1.0, "run_name": "peer_11__sf_1p0"}]}
        paths = {"ntha": Path("root/ntha"), "dataset": Path("root/dataset")}
        with_profile = types.SimpleNamespace(python_exe="python", set_name="peer_strong_63", max_npts=15000, profile=V2)
        command = parent.command_for(with_profile, case, paths)
        self.assertEqual(command[command.index("--profile") + 1], V2)
        without = types.SimpleNamespace(python_exe="python", set_name="peer_strong_63", max_npts=15000)
        self.assertNotIn("--profile", parent.command_for(without, case, paths))
        parser = argparse.ArgumentParser()
        parser.add_argument("--profile", default=None)
        self.assertEqual(parser.parse_args(["--profile", V2]).profile, V2)
        source = Path(child.__file__).read_text(encoding="utf-8")
        self.assertIn('command.extend(["--profile", args.profile])', source)
        import Ground_Motion_Main as gm
        with mock.patch.object(sys, "argv", ["Ground_Motion_Main.py", "--profile", V2, "--design-only"]):
            self.assertEqual(gm.parse_args().profile, V2)
        self.assertIn("analysis_profile_id", gm.OUTPUT_IDENTITY_KEYS)

    def test_the_plan_records_the_profile_and_refuses_another_one(self):
        import tempfile
        from Data_Generation import Generate_Parameterized_Dataset as parent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = parent.load_plan(root, 1, [11, 23], 7, 15000, set_name="peer_strong_63", profile=V2)
            self.assertEqual(len(cases), 1)
            import json
            plan = json.loads((root / "parameter_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["analysis_profile"], V2)
            parent.load_plan(root, 1, [11, 23], 7, 15000, set_name="peer_strong_63", profile=V2)       # same plan resumes
            with self.assertRaisesRegex(RuntimeError, "analysis_profile"):
                parent.load_plan(root, 1, [11, 23], 7, 15000, set_name="peer_strong_63", profile=None)


if __name__ == "__main__":
    unittest.main()
