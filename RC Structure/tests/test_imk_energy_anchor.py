"""Provisional minimum-yield energy anchor of the member hinges (pre-generation review, 2026-10-04).

These tests establish NUMERICAL INVARIANCE of the anchor: the same physical spring gives the same physical
response whichever of its directions is the positive input coordinate, at the material, at an assembled
member and under swapped connectivity. They use synthetic values and do not calibrate RC.
"""
from dataclasses import replace
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import openseespy.opensees as ops

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import Structure_Parameters as sp
from Analysis import Hinge_Moment_Rotation as hmr
from Model import Analysis_Profile as ap
from Model import IMK_Hinges as hinges
from Model.IMK_Materials import (MAPPING_VERSION, MATERIAL_TYPES, PROVISIONAL_ANCHOR_STATUS, PROVISIONAL_ANCHOR_VERSION,
                                 active_energy_modes, define_anchored_rotational_imk, define_rotational_imk)
from test_imk_energy_mapping import NEG, POS, analyze_displacements, cyclic, history, profile

PROVENANCE = {"calibration_id": "synthetic", "status": "verification_only", "deformation_scope": "isolated_3d_spring"}
BASE_LAMDA = {"S": 0.5, "C": 0.65, "A": 0.8, "K": 0.95}                # rad; E_ref = Lamda * min(My) = 50..95 kip-in rad


def base_cyclic(kind):
    modes = active_energy_modes(kind)
    return replace(cyclic(kind), **{f"lamda_{m.lower()}": BASE_LAMDA[m] for m in modes})


def spring(kind, reverse=False, legacy=False, positive=POS, negative=NEG):
    """One zero-length spring driven through the same PHYSICAL history in either coordinate orientation."""
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    ops.node(1, 0., 0., 0.)
    ops.node(2, 0., 0., 0.)
    ops.fix(1, 1, 1, 1, 1, 1, 1)
    ops.equalDOF(1, 2, 1, 2, 3, 4, 6)
    if legacy:                                           # what the unmapped path installs: Lamda on the input branch
        pos, neg = (negative, positive) if reverse else (positive, negative)
        swapped = replace(base_cyclic(kind), d_pos=cyclic(kind).d_neg, d_neg=cyclic(kind).d_pos) if reverse else base_cyclic(kind)
        md = define_rotational_imk(kind, 11, 10000., pos, neg, swapped, provenance=PROVENANCE)
    else:
        md = define_anchored_rotational_imk(kind, 11, 10000., positive, negative, base_cyclic(kind), reverse=reverse,
                                            physical_directions=("physical_a", "physical_b"), provenance=PROVENANCE)
    ops.element("zeroLength", 21, 1, 2, "-mat", 11, "-dir", 5)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 0., 0., 0., 0., 1., 0.)
    rows = [[0., 0.]]
    for _ in analyze_displacements(2, 5, history(.025) * (-1 if reverse else 1)):
        rows.append(ops.eleResponse(21, "material", 1, "stressStrain"))
    ops.wipe()
    return np.array(rows), md


def member(kind, axis, reversed_nodes=False, fixed_high=False, mode=PROVISIONAL_ANCHOR_VERSION, strengths=(2000., 1000.),
           drive=True):
    """A cantilever beam with asymmetric end strengths; returns the fixed-end hinge's moment-rotation and registry."""
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    hinges.reset_hinge_registry()
    ops.node(1, 0., 0., 0.)
    ops.node(2, *((240., 0., 0.) if axis == "x" else (0., 240., 0.)))
    ops.geomTransf("Linear", 9, 0, 0, 1)
    fixed, free = (2, 1) if fixed_high else (1, 2)
    ops.fix(fixed, 1, 1, 1, 1, 1, 1)
    ni, nj = (2, 1) if reversed_nodes else (1, 2)
    values = {"IMK_MATERIAL_TYPE": kind, "IMK_ENERGY_MAPPING_MODE": mode, "IMK_D_POS": .65, "IMK_D_NEG": .90,
              "IMK_FMAXFY_POS": 1.10, "IMK_FMAXFY_NEG": 1.12, "IMK_FRESFY_POS": .10, "IMK_FRESFY_NEG": .15}
    # Base Lamda such that Lamda * 1000 kip-in equals the synthetic explicit profile of this material.
    values.update({f"IMK_LAMBDA_{m}": e / 1000. for m, e in profile(kind, 20.)["energies_kip_in_rad"].items()})
    bb = {"theta_p": .01, "theta_pc": .05, "theta_u": .1, "theta_p_neg": .014, "theta_pc_neg": .055, "theta_u_neg": .11,
          "source": "synthetic directionally asymmetric regression"}
    fixture = None
    if mode == MAPPING_VERSION:                          # explicit energies equal to Lamda * min(My): the cross-check
        fixture = {e: {a: profile(kind, 20.) for a in ("y", "z")} for e in ("i", "j")}
    with mock.patch.multiple(sp, **values), mock.patch.object(hinges, "backbone_for_member", return_value=bb), \
         mock.patch.object(hinges, "beam_yield_moments", return_value=(*strengths, {"basis": "synthetic"})), \
         contextlib.redirect_stdout(io.StringIO()):
        hinges.create_imk_member(1, ni, nj, "beam_" + axis, 9, _verification_calibrations=fixture)
        registry = hinges.hinge_registry()[1]
        table = hmr.spring_table([hinges.hinge_element_tag(1, 1), hinges.hinge_element_tag(1, 2)])
    if not drive:
        ops.wipe()
        return None, registry, table
    tag = hinges.hinge_element_tag(1, 1 if fixed == ni else 2)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    load = [0.] * 6
    load[2] = 1.
    ops.load(free, *load)
    rows = [[0., 0.]]
    for _ in analyze_displacements(free, 3, history(8., 80)):
        rows.append(np.array(ops.eleResponse(tag, "material", 1, "stressStrain")) * (-1 if fixed_high else 1))
    ops.wipe()
    return np.array(rows), registry, table


class AnchorMaterialTests(unittest.TestCase):
    def tearDown(self):
        ops.wipe()

    def test_same_physical_history_is_invariant_to_the_input_coordinate(self):
        for kind in MATERIAL_TYPES:
            with self.subTest(material=kind):
                reference, md = spring(kind)
                reversed_result, reverse_md = spring(kind, reverse=True)
                self.assertLess(float(np.max(abs(reference + reversed_result))), 1e-7)
                self.assertGreater(float(np.max(abs(reference[:, 0]))), POS.fy)                         # it yielded
                self.assertGreater(float(np.max(abs(reference[1:401, 0] - reference[801:1201, 0]))), 1.)  # and deteriorated
                self.assertEqual(md["positive"], reverse_md["negative"])
                self.assertEqual(md["negative"], reverse_md["positive"])
                self.assertEqual(md["cyclic"]["d_pos"], reverse_md["cyclic"]["d_neg"])
                for mode in active_energy_modes(kind):
                    expected = BASE_LAMDA[mode] * min(POS.fy, NEG.fy)
                    for record in (md, reverse_md):
                        self.assertAlmostEqual(record["reference_energies_kip_in_rad"][mode], expected, places=9)
                        mapping = record["provenance"]["energy_mapping"]
                        self.assertAlmostEqual(mapping["reference_energies_kip_in_rad"][mode], expected, places=9)
                        self.assertAlmostEqual(mapping["installed_lamda_rad"][mode] * record["positive"]["fy"], expected, places=9)
                self.assertEqual(md["provenance"]["energy_mapping"]["status"], PROVISIONAL_ANCHOR_STATUS)
                self.assertEqual(md["provenance"]["energy_mapping"]["reference_moment_kip_in"], 100.)
                self.assertTrue(reverse_md["provenance"]["energy_mapping"]["coordinate_reversed"])

    def test_the_unmapped_path_is_not_invariant_which_is_the_defect_the_anchor_removes(self):
        kind = "IMKPeakOriented"
        forward, md = spring(kind, legacy=True)
        backward, reverse_md = spring(kind, reverse=True, legacy=True)
        self.assertGreater(float(np.max(abs(forward + backward))), 1.)
        self.assertAlmostEqual(reverse_md["reference_energies_kip_in_rad"]["S"] / md["reference_energies_kip_in_rad"]["S"],
                               NEG.fy / POS.fy)

    def test_equal_strengths_install_the_base_lamda_unchanged(self):
        for kind in MATERIAL_TYPES:
            symmetric = replace(NEG, fy=POS.fy, dp=POS.dp, dpc=POS.dpc, du=POS.du, fmax_fy=POS.fmax_fy, fres_fy=POS.fres_fy)
            for reverse in (False, True):
                with self.subTest(material=kind, reverse=reverse):
                    _, anchored = spring(kind, reverse=reverse, negative=symmetric)
                    _, legacy = spring(kind, reverse=reverse, legacy=True, negative=symmetric)
                    self.assertEqual(anchored["command"], legacy["command"])            # bit-identical OpenSees command

    def test_the_anchor_is_never_recorded_as_an_experimental_calibration(self):
        _, md = spring("IMKPeakOriented")
        self.assertNotIn("energy_calibration", md["provenance"])
        self.assertIn("not an experimental calibration", md["provenance"]["energy_mapping"]["limits"])
        self.assertEqual(md["provenance"]["energy_mapping_status"], PROVISIONAL_ANCHOR_VERSION)


class AnchorMemberTests(unittest.TestCase):
    def tearDown(self):
        ops.wipe()
        hinges.reset_hinge_registry()

    def test_assembled_beams_both_axes_ends_and_swapped_connectivity(self):
        for kind in ("IMKBilin", "IMKPeakOriented"):
            reference, registry, _ = member(kind, "x")
            self.assertEqual(registry["energy_mapping_mode"], PROVISIONAL_ANCHOR_VERSION)
            self.assertFalse(registry["verification_only"])
            self.assertGreater(float(np.max(abs(reference[:, 0]))), 1000.)
            for axis in ("x", "y"):
                for reverse in (False, True):
                    for high in (False, True):
                        with self.subTest(material=kind, axis=axis, reversed_nodes=reverse, fixed_high=high):
                            actual, _, _ = member(kind, axis, reverse, high)
                            np.testing.assert_allclose(actual, reference, rtol=2e-7, atol=2e-6)

    def test_anchored_member_equals_the_explicit_energy_route_given_the_same_energies(self):
        # The fixture's base Lamda on min(My) = 1000 kip-in is exactly the synthetic explicit profile.
        for kind in ("IMKBilin", "IMKPeakOriented"):
            with self.subTest(material=kind):
                anchored, _, _ = member(kind, "x")
                explicit, _, _ = member(kind, "x", mode=MAPPING_VERSION)
                np.testing.assert_allclose(anchored, explicit, rtol=1e-9, atol=1e-7)

    def test_both_ends_of_one_beam_carry_the_same_reference_energy_and_the_legacy_ends_do_not(self):
        _, registry, table = member("IMKPeakOriented", "x", drive=False)
        strong = [row for row in table if row["local_axis"] == "y"]
        self.assertEqual([row["energy_anchor_policy"] for row in strong], [PROVISIONAL_ANCHOR_VERSION] * 2)
        self.assertEqual([row["energy_calibration_status"] for row in strong], [PROVISIONAL_ANCHOR_STATUS] * 2)
        self.assertEqual([row["energy_reference_moment_kip_in"] for row in strong], [1000., 1000.])
        self.assertEqual(strong[0]["reference_energy_by_mode_kip_in_rad"], strong[1]["reference_energy_by_mode_kip_in_rad"])
        self.assertEqual((strong[0]["yield_moment_positive_kip_in"], strong[0]["yield_moment_negative_kip_in"]), (2000., 1000.))
        self.assertEqual((strong[1]["yield_moment_positive_kip_in"], strong[1]["yield_moment_negative_kip_in"]), (1000., 2000.))
        installed = {end: registry["installed_materials"][end]["y"] for end in ("i", "j")}
        base = sp_lamda_s = profile("IMKPeakOriented", 20.)["energies_kip_in_rad"]["S"] / 1000.
        self.assertAlmostEqual(installed["i"]["cyclic"]["lamda_s"], base / 2)     # hogging (2000) is the positive input at end i
        self.assertAlmostEqual(installed["j"]["cyclic"]["lamda_s"], base)         # sagging (1000) is the positive input at end j
        _, legacy_registry, legacy_table = member("IMKPeakOriented", "x", mode="legacy_unmapped", drive=False)
        legacy = [row for row in legacy_table if row["local_axis"] == "y"]
        self.assertEqual([row["energy_reference_moment_kip_in"] for row in legacy], [2000., 1000.])
        self.assertNotEqual(legacy[0]["reference_energy_by_mode_kip_in_rad"], legacy[1]["reference_energy_by_mode_kip_in_rad"])

    def test_symmetric_member_installs_the_legacy_commands(self):
        _, anchored, _ = member("IMKPeakOriented", "x", strengths=(1500., 1500.), drive=False)
        _, legacy, _ = member("IMKPeakOriented", "x", mode="legacy_unmapped", strengths=(1500., 1500.), drive=False)
        for end in ("i", "j"):
            for axis in ("y", "z"):
                self.assertEqual(anchored["installed_materials"][end][axis]["reference_energies_kip_in_rad"],
                                 legacy["installed_materials"][end][axis]["reference_energies_kip_in_rad"])
                self.assertEqual(anchored["installed_materials"][end][axis]["cyclic"]["lamda_s"],
                                 legacy["installed_materials"][end][axis]["cyclic"]["lamda_s"])
        # End i is not reversed, so its commands are bit-identical to the legacy ones.
        for axis in ("y", "z"):
            self.assertEqual(anchored["installed_materials"]["i"][axis]["command"], legacy["installed_materials"]["i"][axis]["command"])


class AnchoredProfileTests(unittest.TestCase):
    def test_the_research_profile_sets_the_anchor_and_the_screening_profile_still_refuses_it(self):
        saved = {key: getattr(sp, key) for key in dir(sp) if key.isupper()}
        try:
            before = ap.profile_identity(ap.V2_FLEXURE)["sha256"] if not ap.configuration_problems(ap.V2_FLEXURE) else None
            identity = ap.apply_profile(ap.V2_FLEXURE_ANCHORED)
            self.assertEqual(sp.IMK_ENERGY_MAPPING_MODE, PROVISIONAL_ANCHOR_VERSION)
            self.assertEqual(identity["state"]["IMK_ENERGY_MAPPING_MODE"], PROVISIONAL_ANCHOR_VERSION)
            self.assertIn("not an experimental calibration", identity["declarations"]["member_energy"])
            self.assertIn("P-M-M", identity["declarations"]["columns"])
            with self.assertRaises(ValueError):
                ap.apply_profile(ap.V2_FLEXURE)                    # the screening profile requires the legacy mapping
            sp.IMK_ENERGY_MAPPING_MODE = "legacy_unmapped"
            screening = ap.apply_profile(ap.V2_FLEXURE)
            self.assertNotEqual(screening["sha256"], identity["sha256"])
            if before is not None:
                self.assertEqual(screening["sha256"], before)
        finally:
            for key, value in saved.items():
                setattr(sp, key, value)


if __name__ == "__main__":
    unittest.main()
