"""The three V6 IMK audit corrections (2026-10-23): Haselton et al. (2008) Eq. 3.10 with the total longitudinal steel,
Eq. 3.20 with s/d, and a member reference energy that does not depend on the springs' local coordinates.

Hand-calculated equation examples (symmetric and asymmetric cages, side bars, a non-square section, stacked beam rows,
both bending axes), native material sign reversal, and member end / axis / connectivity checks. Synthetic values: these
are software checks of the declared equations and mapping, not an experimental calibration.
"""
import contextlib
import io
import math
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import openseespy.opensees as ops

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import Structure_Parameters as sp
from Model import Analysis_Profile as ap
from Model import IMK_Calibration as cal
from Model import Member_Groups as mg
from Model.IMK_Materials import (CANONICAL_BASELINE_STATUS, CANONICAL_BASELINE_VERSION, MATERIAL_TYPES,
                                 PROVISIONAL_ANCHOR_VERSION, active_energy_modes, define_canonical_rotational_imk,
                                 define_rotational_imk)
from test_imk_energy_anchor import BASE_LAMDA, PROVENANCE, base_cyclic, member
from test_imk_energy_mapping import NEG, POS, analyze_displacements, cyclic, history

MPA = 6.894757
SLAB_MODE = {"SLAB_THICKNESS_IN": 8.0, "IMK_DETERIORATION_MODE": "haselton_2008", "FY_KSI": 60.0,
             "IMK_COLUMN_THETA_Y": 0.004, "IMK_BEAM_THETA_Y": 0.005}
# 24 x 24 in column, #9 bars: 4 top, 4 bottom, 2 on EACH side face (12 bars), #4 hoops with 4 legs at 4 in.
COLUMN = mg.MemberDesign("column", 24.0, 24.0, 8.0, 9, 4, 4, 2, 4, 4, 4.0)
# 20 x 28 in column (b x h), #8 bars, 2 side bars per side face, #4 hoops with 4 legs at 5 in.
WIDE = mg.MemberDesign("column", 20.0, 28.0, 7.0, 8, 4, 4, 2, 4, 4, 5.0)
# 18 x 30 in beam, #8 bars: 5 top, 3 bottom, 1 side bar per side face; #4 stirrups, 2 legs at 6 in.
BEAM = mg.MemberDesign("beam_x", 18.0, 30.0, 6.0, 8, 5, 3, 1, 4, 2, 6.0)
# Two top layers (centroid 3.75 in below the top face) and one bottom layer at the 3.0 in first-layer offset.
ROWS = {"top": {"centroid_in": 3.75}, "bottom": {"centroid_in": 3.0}}


def theta_p_by_hand(design, nu, rho):
    rho_sh = design.stirrup_legs * sp.rebar_area(design.stirrup_bar_size) / (design.b_in * design.stirrup_spacing_in)
    s_n = design.stirrup_spacing_in / sp.rebar_diameter(design.bar_size) * math.sqrt(60.0 * MPA / 100.0)
    return (0.12 * 1.55 * 0.16 ** nu * (0.02 + 40.0 * rho_sh) ** 0.43 * 0.54 ** (0.01 * design.fc_ksi * MPA)
            * 0.66 ** (0.1 * s_n) * 2.27 ** (10.0 * rho))


class SlabState(unittest.TestCase):
    def setUp(self):
        self.patch = mock.patch.multiple(sp, **SLAB_MODE)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        # The inputs the hand values below rest on: bars on the 0.5 in grid at cover 1.5 + layout hoop #5 + db/2.
        self.assertEqual((sp.COL_CLEAR_COVER_IN, sp.BEAM_CLEAR_COVER_IN), (1.5, 1.5))
        self.assertEqual(sp.longitudinal_cover_in("column", 9, 4), 3.0)        # 1.5 + 0.625 + 0.564 -> 3.0
        self.assertEqual(sp.longitudinal_cover_in("column", 8, 4), 3.0)        # 1.5 + 0.625 + 0.500 -> 3.0
        self.assertEqual(sp.longitudinal_cover_in("beam", 8, 4), 3.0)
        self.assertEqual((sp.rebar_area(9), sp.rebar_area(8)), (1.0, 0.79))


class Eq310TotalLongitudinalSteel(SlabState):
    def test_square_column_counts_both_faces_and_both_side_faces_once(self):
        self.assertEqual(cal.total_longitudinal_bars("column", COLUMN), 12)        # 4 + 4 + 2 x 2
        rho = 12 * 1.0 / (24.0 * (24.0 - 3.0))                                     # 0.023810
        record = cal.longitudinal_steel_record("column", COLUMN)
        self.assertAlmostEqual(record["rho"], rho, places=12)
        self.assertEqual((record["b_in"], record["d_in"], record["total_bars"]), (24.0, 21.0, 12))
        self.assertEqual(record["definition"], cal.RHO_DEFINITION)
        self.assertFalse(record["beam_extrapolation"])
        theta = cal.haselton_theta_p("column", 0.1, design=COLUMN)
        self.assertAlmostEqual(theta, theta_p_by_hand(COLUMN, 0.1, rho), places=12)
        # Against the old numerator (the larger face only, 4 bars): theta_p rises by 2.27^(10 * delta rho)
        old = theta_p_by_hand(COLUMN, 0.1, 4 * 1.0 / (24.0 * 21.0))
        self.assertAlmostEqual(theta / old, 2.27 ** (10.0 * (rho - 4.0 / 504.0)), places=12)
        self.assertAlmostEqual(theta / old, 1.1390, places=4)                         # e^(0.15873 ln 2.27)

    def test_non_square_column_uses_b_and_the_depth_across_h(self):
        rho = 12 * 0.79 / (20.0 * (28.0 - 3.0))                                     # 0.018960
        self.assertAlmostEqual(cal.longitudinal_steel_ratio("column", WIDE), rho, places=12)

    def test_asymmetric_beam_with_side_bars_and_stacked_rows(self):
        self.assertEqual(cal.total_longitudinal_bars("beam_x", BEAM), 10)         # 5 + 3 + 2 x 1
        depths = cal.effective_depths("beam_x", BEAM, ROWS)
        self.assertEqual(depths["y"]["d_in"], 26.25)                               # hogging: 30 - 3.75 (two top layers)
        self.assertEqual(depths["beam_vertical_bending"]["sagging_d_in"], 27.0)    # recorded, not used
        record = cal.longitudinal_steel_record("beam_x", BEAM, depths)
        self.assertAlmostEqual(record["rho"], 10 * 0.79 / (18.0 * 26.25), places=12)
        self.assertTrue(record["beam_extrapolation"])
        self.assertIn("no slab steel", record["scope"])

    def test_a_beam_priced_without_rows_says_so(self):
        depths = cal.effective_depths("beam_x", BEAM)
        self.assertEqual(depths["y"]["d_in"], 27.0)
        self.assertIn("no bar rows supplied", depths["rows_source"])

    def test_the_uniform_globals_count_the_same_way(self):
        with mock.patch.multiple(sp, COL_TOP_BARS=3, COL_BOT_BARS=3, COL_SIDE_BARS=1, BEAM_TOP_BARS=4, BEAM_BOT_BARS=2,
                                 BEAM_SIDE_BARS=0):
            self.assertEqual(cal.total_longitudinal_bars("column"), 8)
            self.assertEqual(cal.total_longitudinal_bars("beam_x"), 6)


class Eq320EffectiveDepth(SlabState):
    def test_square_column_both_axes(self):
        nu = 0.2
        lam = 170.7 * 0.27 ** nu * 0.10 ** (4.0 / 21.0)
        d = cal.deterioration_for_member("column", nu, COLUMN)
        self.assertEqual(d["deterioration_equation_version"], cal.EQ_3_20_VERSION)
        for axis in ("y", "z"):
            self.assertAlmostEqual(d["deterioration_by_axis"][axis]["lambda_haselton_dimensionless"], lam, places=12)
            self.assertAlmostEqual(d["deterioration_by_axis"][axis]["spacing_effective_depth_ratio"], 4.0 / 21.0, places=12)
            self.assertAlmostEqual(d["deterioration_by_axis"][axis]["lambda_opensees_by_mode_rad"]["S"], lam * 0.004, places=12)
            self.assertEqual(d["deterioration_by_axis"][axis]["lambda_opensees_by_mode_rad"]["A"], 1e12)
        self.assertAlmostEqual(d["lambda_haselton_dimensionless"], lam, places=12)
        # s/h made the energy 10^(s/d - s/h) times too large with every other input fixed
        self.assertAlmostEqual(170.7 * 0.27 ** nu * 0.10 ** (4.0 / 24.0) / lam, 10 ** (4.0 / 21.0 - 4.0 / 24.0), places=12)

    def test_non_square_column_takes_each_axis_own_depth(self):
        d = cal.deterioration_for_member("column", 0.0, WIDE)["deterioration_by_axis"]
        self.assertAlmostEqual(d["y"]["lambda_haselton_dimensionless"], 170.7 * 0.10 ** (5.0 / 25.0), places=12)  # across h
        self.assertAlmostEqual(d["z"]["lambda_haselton_dimensionless"], 170.7 * 0.10 ** (5.0 / 17.0), places=12)  # across b
        self.assertLess(d["z"]["lambda_haselton_dimensionless"], d["y"]["lambda_haselton_dimensionless"])

    def test_beam_vertical_bending_uses_its_rows_and_lateral_bending_its_width(self):
        depths = cal.effective_depths("beam_x", BEAM, ROWS)
        d = cal.deterioration_for_member("beam_x", 0.0, BEAM, depths)
        self.assertAlmostEqual(d["deterioration_by_axis"]["y"]["lambda_haselton_dimensionless"], 170.7 * 0.10 ** (6.0 / 26.25), places=12)
        self.assertAlmostEqual(d["deterioration_by_axis"]["z"]["lambda_haselton_dimensionless"], 170.7 * 0.10 ** (6.0 / 15.0), places=12)
        self.assertAlmostEqual(d["deterioration_by_axis"]["y"]["lambda_opensees_rad"],
                               170.7 * 0.10 ** (6.0 / 26.25) * 0.005, places=12)
        self.assertTrue(d["deterioration_beam_extrapolation"])
        self.assertEqual(len(d["deterioration_unsupported_extensions"]), 3)

    def test_the_backbone_carries_depths_rho_and_both_axes(self):
        backbone = cal.backbone_for_member("beam_x", 0.0, design=BEAM, rows=ROWS)
        self.assertEqual(backbone["theta_p_longitudinal_steel"]["d_in"], 26.25)
        self.assertEqual(backbone["deterioration_effective_depths"]["y"]["d_in"], 26.25)
        self.assertEqual(set(backbone["deterioration_by_axis"]), {"y", "z"})
        # unchanged by these corrections: theta_pc and its cap, theta_u = theta_p + theta_pc + nominal theta_y
        self.assertLessEqual(backbone["theta_pc"], cal.THETA_PC_CAP)
        self.assertAlmostEqual(backbone["theta_u"], backbone["theta_p"] + backbone["theta_pc"] + 0.005, places=12)


def spring(kind, reverse=False, legacy=False, positive=POS, negative=NEG):
    """One zero-length spring driven through the same PHYSICAL history in either coordinate orientation."""
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    ops.node(1, 0., 0., 0.)
    ops.node(2, 0., 0., 0.)
    ops.fix(1, 1, 1, 1, 1, 1, 1)
    ops.equalDOF(1, 2, 1, 2, 3, 4, 6)
    if legacy:                                          # what the unmapped path installs: Lamda on the input branch
        pos, neg = (negative, positive) if reverse else (positive, negative)
        swapped = replace(base_cyclic(kind), d_pos=cyclic(kind).d_neg, d_neg=cyclic(kind).d_pos) if reverse else base_cyclic(kind)
        md = define_rotational_imk(kind, 11, 10000., pos, neg, swapped, provenance=PROVENANCE)
    else:
        md = define_canonical_rotational_imk(kind, 11, 10000., positive, negative, base_cyclic(kind), reverse=reverse,
                                             physical_directions=("hogging", "sagging"), provenance=PROVENANCE)
    ops.element("zeroLength", 21, 1, 2, "-mat", 11, "-dir", 5)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 0., 0., 0., 0., 1., 0.)
    rows = [[0., 0.]]
    for _ in analyze_displacements(2, 5, history(.025) * (-1 if reverse else 1)):
        rows.append(ops.eleResponse(21, "material", 1, "stressStrain"))
    ops.wipe()
    return np.array(rows), md


class CanonicalMaterial(unittest.TestCase):
    def tearDown(self):
        ops.wipe()

    def test_sign_reversal_gives_the_same_physical_response(self):
        for kind in MATERIAL_TYPES:
            with self.subTest(material=kind):
                reference, md = spring(kind)
                reversed_result, reverse_md = spring(kind, reverse=True)
                self.assertLess(float(np.max(abs(reference + reversed_result))), 1e-7)
                self.assertGreater(float(np.max(abs(reference[:, 0]))), POS.fy)                         # it yielded
                self.assertGreater(float(np.max(abs(reference[1:401, 0] - reference[801:1201, 0]))), 1.)  # and deteriorated
                for mode in active_energy_modes(kind):
                    expected = BASE_LAMDA[mode] * POS.fy                         # the canonical (first) branch's My
                    for record in (md, reverse_md):
                        self.assertAlmostEqual(record["reference_energies_kip_in_rad"][mode], expected, places=9)
                    self.assertAlmostEqual(md["cyclic"][f"lamda_{mode.lower()}"], BASE_LAMDA[mode], places=12)
                    self.assertAlmostEqual(reverse_md["cyclic"][f"lamda_{mode.lower()}"],
                                           BASE_LAMDA[mode] * POS.fy / NEG.fy, places=12)
                mapping = md["provenance"]["energy_mapping"]
                self.assertEqual((mapping["version"], mapping["status"]), (CANONICAL_BASELINE_VERSION, CANONICAL_BASELINE_STATUS))
                self.assertEqual(mapping["canonical_physical_direction"], "hogging")
                self.assertIn("not an experimental calibration", mapping["limits"])

    def test_the_legacy_mapping_is_not_invariant_which_is_the_corrected_defect(self):
        reference, _ = spring("IMKPeakOriented", legacy=True)
        reversed_result, _ = spring("IMKPeakOriented", reverse=True, legacy=True)
        self.assertGreater(float(np.max(abs(reference + reversed_result))) / NEG.fy, 0.01)

    def test_suppressed_modes_stay_suppressed(self):
        cyclic_ak = replace(base_cyclic("IMKPeakOriented"), lamda_a=1e12, lamda_k=1e12)
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        for reverse in (False, True):
            md = define_canonical_rotational_imk("IMKPeakOriented", 31 + reverse, 10000., POS, NEG, cyclic_ak,
                                                 reverse=reverse, physical_directions=("hogging", "sagging"),
                                                 provenance=PROVENANCE)
            self.assertGreaterEqual(min(md["cyclic"]["lamda_a"], md["cyclic"]["lamda_k"]), 1e12 * POS.fy / NEG.fy)
            self.assertEqual(md["reference_energies_kip_in_rad"]["A"], 1e12 * POS.fy)


def subdividing_displacements(node, dof, values, tolerance=1e-10):
    """The shared fixture's displacement driver, with a failed target retried in 4, 16, ... equal sub-steps.

    The cantilever's IMKPeakOriented hinge does not converge in ONE Newton step to some targets of the history (the
    shared driver stops there, in every orientation and under the legacy mapping alike); smaller steps follow the same
    physical history to the same targets."""
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.test("NormDispIncr", tolerance, 80)
    ops.algorithm("Newton")
    ops.integrator("DisplacementControl", node, dof, 1e-6)
    ops.analysis("Static")
    for target in values[1:]:
        parts = 1
        while True:
            # a failed analyze reverts to the last committed step, so the remainder is measured from here
            step = float((target - ops.nodeDisp(node, dof)) / parts)
            failed = False
            for _ in range(parts):
                ops.integrator("DisplacementControl", node, dof, step)
                if ops.analyze(1) != 0:
                    failed = True
                    break
            if not failed:
                break
            parts *= 4
            if parts > 256:
                raise AssertionError(f"Synthetic cyclic step failed at {target} even in {parts // 4} sub-steps")
        yield


class CanonicalMember(unittest.TestCase):
    MODE = CANONICAL_BASELINE_VERSION

    def setUp(self):
        import test_imk_energy_anchor as anchor_fixture
        self.patch = mock.patch.object(anchor_fixture, "analyze_displacements", subdividing_displacements)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def tearDown(self):
        ops.wipe()

    def test_both_axes_ends_and_swapped_connectivity_give_one_physical_response(self):
        for kind in ("IMKBilin", "IMKPeakOriented"):
            reference, registry, _ = member(kind, "x", mode=self.MODE)
            self.assertEqual(registry["energy_mapping_mode"], self.MODE)
            self.assertGreater(float(np.max(abs(reference[:, 0]))), 1000.)
            for axis in ("x", "y"):
                for reverse in (False, True):
                    for high in (False, True):
                        with self.subTest(material=kind, axis=axis, reversed_nodes=reverse, fixed_high=high):
                            actual, _, _ = member(kind, axis, reverse, high, mode=self.MODE)
                            np.testing.assert_allclose(actual, reference, rtol=2e-7, atol=2e-6)

    def test_the_legacy_ends_differ_and_the_canonical_start_end_is_the_legacy_one(self):
        canonical, _, _ = member("IMKPeakOriented", "x", mode=self.MODE)
        start, _, _ = member("IMKPeakOriented", "x", mode="legacy_unmapped")
        end, _, _ = member("IMKPeakOriented", "x", fixed_high=True, mode="legacy_unmapped")
        np.testing.assert_allclose(start, canonical, rtol=1e-9, atol=1e-7)     # the baseline is preserved at end i
        self.assertGreater(float(np.max(abs(start - end))), 100.)              # kip-in: the coordinate defect

    def test_end_energies_and_installed_lamda(self):
        hogging, sagging = 2000., 1000.
        for reverse in (False, True):
            with self.subTest(reversed_nodes=reverse):
                _, registry, table = member("IMKPeakOriented", "x", reverse, mode=self.MODE, strengths=(hogging, sagging),
                                            drive=False)
                for end in ("i", "j"):
                    y = registry["installed_materials"][end]["y"]
                    mapping = y["provenance"]["energy_mapping"]
                    self.assertEqual(mapping["reference_moment_kip_in"], hogging)
                    self.assertEqual(mapping["canonical_physical_direction"], "hogging")
                    positive_input = y["positive"]["fy"]
                    # end i of a canonically connected member puts hogging on the positive input; end j sagging
                    canonical_start = (end == "i") != reverse
                    self.assertEqual(positive_input, hogging if canonical_start else sagging)
                    for mode in "SCAK":
                        self.assertAlmostEqual(y["reference_energies_kip_in_rad"][mode],
                                               mapping["base_lamda_rad"][mode] * hogging, places=6)
                    z = registry["installed_materials"][end]["z"]
                    self.assertEqual(z["positive"]["fy"], z["negative"]["fy"])          # symmetric: Lamda unchanged
                    self.assertAlmostEqual(z["cyclic"]["lamda_s"], z["provenance"]["energy_mapping"]["base_lamda_rad"]["S"])
                self.assertTrue(all(row["energy_mapping_status"] == self.MODE for row in table))

    def test_ends_of_different_strength_keep_their_own_strengths_and_energies(self):
        from Model import IMK_Hinges as hinges
        own = {"i": (2000., 1000.), "j": (1500., 800.)}               # (hogging, sagging) of each member end
        basis = {"basis": "synthetic per-end strengths", "hogging_i_kip_in": own["i"][0], "sagging_i_kip_in": own["i"][1],
                 "hogging_j_kip_in": own["j"][0], "sagging_j_kip_in": own["j"][1]}
        backbone = {"theta_p": .01, "theta_pc": .05, "theta_u": .1, "source": "synthetic"}
        for reverse in (False, True):
            with self.subTest(reversed_nodes=reverse):
                ops.wipe()
                ops.model("basic", "-ndm", 3, "-ndf", 6)
                hinges.reset_hinge_registry()
                ops.node(1, 0., 0., 0.)
                ops.node(2, 240., 0., 0.)
                ops.geomTransf("Linear", 9, 0, 0, 1)
                ni, nj = (2, 1) if reverse else (1, 2)
                with mock.patch.multiple(sp, IMK_MATERIAL_TYPE="IMKPeakOriented", IMK_ENERGY_MAPPING_MODE=self.MODE), \
                        mock.patch.object(hinges, "backbone_for_member", return_value=backbone), \
                        mock.patch.object(hinges, "beam_yield_moments", return_value=(*own["i"], basis)), \
                        contextlib.redirect_stdout(io.StringIO()):
                    hinges.create_imk_member(1, ni, nj, "beam_x", 9)
                    registry = hinges.hinge_registry()[1]
                for end, (hogging, sagging) in own.items():
                    y = registry["installed_materials"][end]["y"]
                    # the end's own two strengths, on whichever input carries each; never the other end's
                    self.assertEqual(sorted((y["positive"]["fy"], y["negative"]["fy"])), sorted((hogging, sagging)))
                    base = y["provenance"]["energy_mapping"]["base_lamda_rad"]
                    for mode in "SCAK":
                        self.assertAlmostEqual(y["reference_energies_kip_in_rad"][mode], base[mode] * hogging, places=6)
                energies = [registry["installed_materials"][e]["y"]["reference_energies_kip_in_rad"]["S"] for e in "ij"]
                self.assertAlmostEqual(energies[0] / energies[1], own["i"][0] / own["j"][0])   # not forced equal
        ops.wipe()


class DefaultsAndProfiles(unittest.TestCase):
    def test_the_canonical_baseline_is_the_default_and_the_screening_profile_requires_it(self):
        self.assertEqual(sp.IMK_ENERGY_MAPPING_MODE, CANONICAL_BASELINE_VERSION)
        self.assertEqual(ap.PROFILES[ap.V2_FLEXURE]["required_state"]["IMK_ENERGY_MAPPING_MODE"], CANONICAL_BASELINE_VERSION)
        self.assertEqual(ap.PROFILES[ap.V2_FLEXURE_ANCHORED]["settings"]["IMK_ENERGY_MAPPING_MODE"], PROVISIONAL_ANCHOR_VERSION)
        self.assertEqual(sp.IMK_CYCLIC_CALIBRATION_STATUS, "provisional_not_experimentally_calibrated")
        self.assertIn("not an experimental calibration", ap.PROFILES[ap.V2_FLEXURE]["declarations"]["member_energy"])

    def test_an_output_of_other_backbone_definitions_is_refused(self):
        import json
        import tempfile
        from Data_Generation.Graph_Exporter import collect_global_parameters
        from Ground_Motion_Main import OUTPUT_IDENTITY_KEYS, validate_ntha_output_compatibility
        self.assertIn("imk_backbone_definitions", OUTPUT_IDENTITY_KEYS)
        self.assertEqual(collect_global_parameters()["imk_backbone_definitions"],
                         {"eq_3_10_rho": cal.RHO_DEFINITION, "effective_depth": cal.EFFECTIVE_DEPTH_DEFINITION,
                          "eq_3_20": cal.EQ_3_20_VERSION})
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "global_parameters.json").write_text(json.dumps(collect_global_parameters()), encoding="utf-8")
            validate_ntha_output_compatibility(output_dir)                 # the same definitions: accepted
            for name in ("RHO_DEFINITION", "EFFECTIVE_DEPTH_DEFINITION"):
                with self.subTest(definition=name), mock.patch.object(cal, name, "an_earlier_definition"), \
                        self.assertRaisesRegex(RuntimeError, "imk_backbone_definitions"):
                    validate_ntha_output_compatibility(output_dir)


if __name__ == "__main__":
    unittest.main()
