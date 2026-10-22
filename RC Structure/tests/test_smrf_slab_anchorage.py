"""Slab bars are credited at exterior beam ends only where the perimeter hook develops them."""
import contextlib
import io
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os as _os                                                                    # noqa: E402
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from slab_model import composite_slab                                              # noqa: E402  (slab-model tests, NoSlab branch)
import openseespy.opensees as ops  # noqa: E402
import Structure_Parameters as sp  # noqa: E402
from Design.SMRF_Beam_Slab_Strength import (hook_development_length_in, perimeter_slab_bar_anchorage,  # noqa: E402
                                            beam_slab_strengths, exterior_ends)
from Model import IMK_Hinges  # noqa: E402


def layout(bar=4, area=0.20, spacing=10.0):
    return {"bar_size": bar, "layers": {f"{axis}_{face}": {"bar_size": bar, "bar_area_in2": area, "spacing_in": spacing,
                                                          "effective_depth_in": 4.0 if axis == "x" else 3.5}
                                        for axis in ("x", "y") for face in ("top", "bottom")}}


class HookDevelopmentTests(unittest.TestCase):
    def test_25_4_3_1_hand_values(self):
        # #4, 4 ksi, spacing 10 >= 6db: fy psi_c / (55 sqrt(fc)) db^1.5 = 60000 * 0.8667 / 3478.5 * 0.3536 = 5.29 -> 6 in minimum
        ldh, factors = hook_development_length_in(4, 4.0, 60.0, 10.0)
        self.assertAlmostEqual(factors["psi_c"], 4000 / 15000 + 0.6, places=9)
        self.assertAlmostEqual(ldh, 6.0, places=9)
        # #6, 4 ksi: above both minima
        ldh6, _ = hook_development_length_in(6, 4.0, 60.0, 10.0)
        self.assertAlmostEqual(ldh6, 60000 * (4000 / 15000 + 0.6) / (55 * math.sqrt(4000)) * 0.75 ** 1.5, places=6)
        # closely spaced hooks without ties: psi_r = 1.6
        tight, f = hook_development_length_in(6, 4.0, 60.0, 4.0)
        self.assertEqual(f["psi_r"], 1.6)
        self.assertAlmostEqual(tight, 1.6 * ldh6, places=6)
        # above 6 ksi psi_c = 1
        self.assertEqual(hook_development_length_in(4, 8.0, 60.0, 10.0)[1]["psi_c"], 1.0)

    def test_perimeter_anchorage_against_the_edge_beam(self):
        # #4 mats hook into a 10-in perimeter beam: 6.0 <= 10 - 1.5 - 0.5 = 8.0
        ok = perimeter_slab_bar_anchorage(layout(4), "x", 10.0, 1.5, 0.5, 4.0, 60.0)
        self.assertTrue(ok["developed"])
        self.assertAlmostEqual(ok["embedment_available_in"], 8.0)
        # #6 mats do not: 9.71 > 8.0
        bad = perimeter_slab_bar_anchorage(layout(6, 0.44), "x", 10.0, 1.5, 0.5, 4.0, 60.0)
        self.assertFalse(bad["developed"])
        self.assertIsNone(perimeter_slab_bar_anchorage(None, "x", 10.0, 1.5, 0.5, 4.0, 60.0))

    def test_exterior_ends(self):
        self.assertEqual(exterior_ends("beam_x", 0, 3, 3), {"i": True, "j": False})
        self.assertEqual(exterior_ends("beam_x", 2, 3, 3), {"i": False, "j": True})
        self.assertEqual(exterior_ends("beam_x", 1, 3, 3), {"i": False, "j": False})
        self.assertEqual(exterior_ends("beam_y", 0, 3, 1), {"i": True, "j": True})


def record(bar=4, area=0.20, beam_width=10.0):
    return {"geometry": {"num_bay_x": 3, "num_bay_y": 3, "num_floor": 1, "bay_x_in": 120.0, "bay_y_in": 120.0},
            "sections": {"b_beam_in": beam_width, "h_beam_in": 18.0, "fc_beam_ksi": 4.0, "h_col_in": 18.0, "b_col_in": 18.0},
            "reinforcement": {"beam_bar_size": 7, "beam_top_bars": 2, "beam_bot_bars": 2,
                              "beam_longitudinal_centroid_offset_in": 2.4375, "beam_clear_cover_in": 1.5,
                              "beam_stirrup_diameter_in": 0.5},
            "materials": {"fy_ksi": 60.0}, "slab": {"thickness_in": 5.0},
            "slab_reinforcement": {"layout": layout(bar, area)}}


@composite_slab
class ExteriorEndCreditTests(unittest.TestCase):
    def test_developed_mats_are_credited_everywhere_with_the_hook_recorded(self):
        entries, families = beam_slab_strengths(record(4))
        self.assertTrue(families["x_edge"]["exterior_anchorage"]["developed"])
        self.assertTrue(families["y_edge"]["exterior_anchorage"]["developed"])
        # first x-beam (tag 17 for one floor: 16 columns), end i exterior
        ext = entries["17/i/negative"]
        interior = entries["17/j/negative"]
        self.assertTrue(ext["exterior_end"])
        self.assertFalse(interior["exterior_end"])
        self.assertEqual(ext["slab_basis"], "developed_effective_width")
        self.assertAlmostEqual(ext["mn_composite_kip_in"], interior["mn_composite_kip_in"])
        self.assertIsNotNone(ext["exterior_anchorage"])
        self.assertIsNone(interior["exterior_anchorage"])

    def test_undeveloped_mats_are_not_credited_at_exterior_hogging_ends(self):
        entries, families = beam_slab_strengths(record(6, 0.44))
        self.assertFalse(families["x_edge"]["exterior_anchorage"]["developed"])
        ext_neg, ext_pos = entries["17/i/negative"], entries["17/i/positive"]
        int_neg = entries["17/j/negative"]
        self.assertEqual(ext_neg["slab_basis"], "terminated_undeveloped")
        self.assertEqual(ext_neg["slab_mn_kip_in"], 0.0)
        self.assertAlmostEqual(ext_neg["mn_composite_kip_in"], ext_neg["mn_rectangular_kip_in"])
        self.assertLess(ext_neg["mn_composite_kip_in"], int_neg["mn_composite_kip_in"])
        # Sagging at that end keeps the flange concrete but neither mat: the
        # bottom mat would otherwise add undeveloped tension steel (fourth
        # cross-check: 1242.61 with the mats vs 1110.56 without).
        family = families["x_edge"]
        self.assertEqual(ext_pos["slab_basis"], "flange_concrete_undeveloped_bars")
        self.assertAlmostEqual(ext_pos["mn_composite_kip_in"], family["mn_undeveloped_positive_kip_in"])
        self.assertLess(ext_pos["mn_composite_kip_in"], family["mn_positive_kip_in"])
        self.assertGreater(ext_pos["mn_composite_kip_in"], ext_pos["mn_rectangular_kip_in"])
        self.assertAlmostEqual(ext_pos["slab_mn_kip_in"], ext_pos["mn_composite_kip_in"] - ext_pos["mn_rectangular_kip_in"])
        self.assertEqual(ext_pos["slab_steel_in_flange_in2"], 0.0)
        self.assertAlmostEqual(family["mn_positive_kip_in"], 1242.61, places=1)
        self.assertAlmostEqual(family["mn_undeveloped_positive_kip_in"], 1110.56, places=1)
        # Interior ends and the interior-end sagging keep the composite strengths.
        self.assertEqual(entries["17/j/positive"]["slab_basis"], "developed_effective_width")
        # The joint rule accepts both bases: zero slab term for hogging, the flange-concrete term for sagging.
        from Design.SMRF_Joints import scwb_check
        column = {"mn_kip_in": 5000.0, "factored_axial_kip": 10.0, "axial_envelope_checked": True}
        state = {"nominal_strengths": True, "column_capacities": [column],
                 "beam_capacities": [{"mn_kip_in": ext_neg["mn_composite_kip_in"], "slab_basis": "terminated_undeveloped",
                                      "slab_mn_kip_in": 0.0}]}
        self.assertEqual(scwb_check(state)["status"], "pass")
        state["beam_capacities"] = [{"mn_kip_in": ext_pos["mn_rectangular_kip_in"], "slab_basis": "flange_concrete_undeveloped_bars",
                                     "slab_mn_kip_in": ext_pos["slab_mn_kip_in"]}]
        self.assertEqual(scwb_check(state)["status"], "pass")
        self.assertAlmostEqual(scwb_check(state)["demand"], 1.2 * ext_pos["mn_composite_kip_in"])
        state["beam_capacities"][0]["slab_basis"] = "terminated_undeveloped"          # a nonzero term needs a strength basis
        self.assertEqual(scwb_check(state)["status"], "not_evaluated")

    def test_hinges_take_the_rectangular_strength_at_an_undeveloped_exterior_end(self):
        values = {"NUM_BAY_X": 3, "NUM_BAY_Y": 3, "BAY_X": 120.0, "BAY_Y": 120.0,
                  "B_COL": 18.0, "H_COL": 18.0, "B_BEAM": 10.0, "H_BEAM": 18.0, "FC_BEAM_KSI": 4.0,
                  "FY_KSI": 60.0, "BEAM_BAR_SIZE": 7, "BEAM_TOP_BARS": 2, "BEAM_BOT_BARS": 2,
                  "BEAM_BAR_AREA": 0.60, "SLAB_THICKNESS_IN": 5.0, "BEAM_CLEAR_COVER_IN": 1.5, "BEAM_STIRRUP_BAR_SIZE": 4,
                  "SLAB_REINFORCEMENT": {"layout": layout(6, 0.44)}}
        with contextlib.ExitStack() as stack:
            for name, value in values.items():
                stack.enter_context(mock.patch.object(sp, name, value, create=True))
            ops.wipe()
            ops.model("basic", "-ndm", 3, "-ndf", 6)
            ops.node(1, 0.0, 120.0, 0.0)
            ops.node(2, 120.0, 120.0, 0.0)        # first span of an interior x line
            ops.node(3, 120.0, 120.0, 0.0)
            ops.node(4, 240.0, 120.0, 0.0)        # middle span
            hog, sag, basis = IMK_Hinges.beam_yield_moments("beam_x", 1, 2)
            hog_mid, sag_mid, basis_mid = IMK_Hinges.beam_yield_moments("beam_x", 3, 4)
            ops.wipe()
        self.assertFalse(basis["exterior_anchorage"]["developed"])
        self.assertEqual(basis["exterior_ends"], {"i": True, "j": False})
        self.assertLess(basis["hogging_i_kip_in"], hog)                 # rectangular at the exterior end
        self.assertAlmostEqual(basis["hogging_j_kip_in"], hog)          # composite at the interior end
        self.assertLess(basis["sagging_i_kip_in"], sag)                 # flange concrete, no mats, at the exterior end
        self.assertAlmostEqual(basis["sagging_j_kip_in"], sag)
        self.assertAlmostEqual(basis_mid["hogging_i_kip_in"], hog_mid)
        self.assertAlmostEqual(basis_mid["hogging_j_kip_in"], hog_mid)
        self.assertAlmostEqual(basis_mid["sagging_i_kip_in"], sag_mid)
        self.assertAlmostEqual(basis_mid["sagging_j_kip_in"], sag_mid)
        # The per-end strengths reach the springs and the registry.
        with contextlib.ExitStack() as stack:
            for name, value in values.items():
                stack.enter_context(mock.patch.object(sp, name, value, create=True))
            ops.wipe()
            ops.model("basic", "-ndm", 3, "-ndf", 6)
            ops.node(1, 0.0, 120.0, 0.0)
            ops.node(2, 120.0, 120.0, 0.0)
            ops.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
            IMK_Hinges.reset_hinge_registry()
            with contextlib.redirect_stdout(io.StringIO()):
                IMK_Hinges.create_imk_member(901, 1, 2, "beam_x", 1)
            entry = IMK_Hinges.hinge_registry()[901]
            ke = IMK_Hinges.imk_hinge_stiffness("beam_x", "rot_y", 120.0, family=IMK_Hinges.beam_line_family("beam_x", 1))
            ops.wipe()
        self.assertAlmostEqual(entry["yield_moment_y_hogging_i_kip_in"], basis["hogging_i_kip_in"])
        self.assertAlmostEqual(entry["yield_moment_y_sagging_i_kip_in"], basis["sagging_i_kip_in"])
        self.assertAlmostEqual(entry["yield_moment_y_sagging_j_kip_in"], sag)
        self.assertLess(entry["theta_y_spring_y_i"], entry["theta_y_spring_y_j"])
        self.assertAlmostEqual(entry["theta_y_spring_y_i"] * ke, min(basis["hogging_i_kip_in"], basis["sagging_i_kip_in"]))
        self.assertAlmostEqual(entry["theta_y_spring_y_j"] * ke, min(hog, sag))
        # With #4 mats the hook fits and both ends keep the composite strength.
        values["SLAB_REINFORCEMENT"] = {"layout": layout(4)}
        with contextlib.ExitStack() as stack:
            for name, value in values.items():
                stack.enter_context(mock.patch.object(sp, name, value, create=True))
            ops.wipe()
            ops.model("basic", "-ndm", 3, "-ndf", 6)
            ops.node(1, 0.0, 120.0, 0.0)
            ops.node(2, 120.0, 120.0, 0.0)
            hog4, _s4, basis4 = IMK_Hinges.beam_yield_moments("beam_x", 1, 2)
            ops.wipe()
        self.assertTrue(basis4["exterior_anchorage"]["developed"])
        self.assertAlmostEqual(basis4["hogging_i_kip_in"], hog4)
        self.assertAlmostEqual(basis4["sagging_i_kip_in"], _s4)


class LadderTests(unittest.TestCase):
    def test_ladder_refuses_a_bar_whose_hook_does_not_fit_the_perimeter_beam(self):
        from test_smrf_slab_reinforcement import inputs, evidence
        from Design.SMRF_Slab_Reinforcement import design_slab_reinforcement
        context = {"clear_span_x_in": 102.0, "clear_span_y_in": 102.0, "beam_width_in": 10.0,
                   "alpha_f_min": 5.6, "alpha_f_l2_l1_min": 5.6, "thickness_screen_passed": True,
                   "column_core_width_in": 13.0,
                   "two_way_shear_path_assessed": True, "columns_at_beam_intersections": True,
                   "beam_clear_cover_in": 1.5, "beam_hoop_diameter_in": 0.5, "fc_beam_ksi": 4.0}
        result = design_slab_reinforcement(inputs(), evidence(), policy={"bar_sizes": [6]}, context=context)
        self.assertIsNone(result["layout"])
        self.assertIn("Hook development", result["trial_history"][0]["reason"])
        result = design_slab_reinforcement(inputs(), evidence(), policy={"bar_sizes": [4, 5, 6]}, context=context)
        self.assertIn(result["layout"]["bar_size"], (4, 5))
        anchorage = [c for c in result["checks"] if c["id"] == "slab_perimeter_bar_anchorage"]
        self.assertEqual(len(anchorage), 2)
        self.assertTrue(all(c["status"] == "pass" for c in anchorage))

    def test_hook_is_checked_at_each_candidate_spacing_not_the_tightest_offered(self):
        """Fourth cross-check: spacings 3-12 rejected every #5 layout because the hook was priced at 3 in (psi_r = 1.6)."""
        from test_smrf_slab_reinforcement import inputs, evidence
        from Design.SMRF_Slab_Reinforcement import design_slab_reinforcement
        context = {"clear_span_x_in": 102.0, "clear_span_y_in": 102.0, "beam_width_in": 10.0,
                   "alpha_f_min": 5.6, "alpha_f_l2_l1_min": 5.6, "thickness_screen_passed": True,
                   "column_core_width_in": 13.0,
                   "two_way_shear_path_assessed": True, "columns_at_beam_intersections": True,
                   "beam_clear_cover_in": 1.5, "beam_hoop_diameter_in": 0.5, "fc_beam_ksi": 4.0}
        spacings = [float(s) for s in range(3, 13)]
        result = design_slab_reinforcement(inputs(), evidence(), policy={"bar_sizes": [5], "spacing_options_in": spacings},
                                           context=context)
        self.assertIsNotNone(result["layout"])
        self.assertEqual(result["layout"]["bar_size"], 5)
        self.assertEqual({layer["spacing_in"] for layer in result["layout"]["layers"].values()}, {12.0})
        self.assertEqual(result["summary"]["counts"]["fail"], 0)
        trial = result["trial_history"][0]
        self.assertTrue(trial["passed"])
        self.assertEqual(trial["spacings_rejected_by_hook_in"], [3.0])      # #5 hooks at 3 in < 6 db: 11.8 > 8.0
        without_three = design_slab_reinforcement(inputs(), evidence(), policy={"bar_sizes": [5], "spacing_options_in": spacings[1:]},
                                                  context=context)
        self.assertEqual(without_three["layout"]["layers"], result["layout"]["layers"])
        # A bar with no fitting spacing at all is still not offered.
        six = design_slab_reinforcement(inputs(), evidence(), policy={"bar_sizes": [6], "spacing_options_in": spacings},
                                        context=context)
        self.assertIsNone(six["layout"])
        self.assertIn("at every spacing offered", six["trial_history"][0]["reason"])


if __name__ == "__main__":
    unittest.main()

