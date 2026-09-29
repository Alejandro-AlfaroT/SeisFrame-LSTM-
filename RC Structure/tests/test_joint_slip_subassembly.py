"""Orientation regression for the diagnostic slip interface (Analysis/Joint_Slip_Subassembly, 2026-09-27).

Codex's slip-calibration resolution: the interface stiffness is a property of the physical
calibration and invariant under node-order/axis reversal; complete branches, directional D and
reference energies are mapped together; the original positive-selected Ke is the negative control
and must be caught. Acceptance is a relative moment discrepancy below 1e-9 between equivalent
orientations. Nothing here is a physical calibration: the bond law is provisional and the finite
energy case uses a synthetic test value.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

RC_DIR = Path(__file__).resolve().parents[1]
if str(RC_DIR) not in sys.path:
    sys.path.insert(0, str(RC_DIR))

import openseespy.opensees as ops  # noqa: E402
from Analysis import Joint_Slip_Subassembly as jss  # noqa: E402
from Model.IMK_Materials import (CyclicParameters, RotationalBackbone, define_mapped_rotational_imk,  # noqa: E402
                                 define_rotational_imk)

EC_KSI = 57.0 * (8000.0 ** 0.5)     # the review candidate's 8 ksi beam concrete


def calibrations():
    """The review candidate's two beam faces: No. 11 bars anchored in 4 ksi column concrete, the actual
    row elevations and cracked neutral axes of case_0002 (physical, not synthetic)."""
    hog = {**jss.slip_calibration(11, 60.0, 4.0, 22.475, 6.347, 4, EC_KSI), "my": 10538.4, "sign": "hogging"}
    sag = {**jss.slip_calibration(11, 60.0, 4.0, 25.295, 7.255, 4, EC_KSI), "my": 10729.9, "sign": "sagging"}
    return hog, sag


def protocol():
    targets = [0.0]
    for amp in (1e-4, 1e-3, 3e-3, 6e-3, 1.2e-2):
        targets += [amp, -amp, amp, -amp, 0.0]
    return np.concatenate([np.linspace(a, b, max(3, int(abs(b - a) / 1e-5) + 1))[1:] for a, b in zip(targets, targets[1:])])


def drive(mat_tag, theta, reverse):
    """Moment history of one material under a rotation history, in the physical sign for either orientation."""
    ops.testUniaxialMaterial(mat_tag)
    out = []
    for value in theta:
        ops.setStrain(float(-value if reverse else value))
        out.append(-ops.getStress() if reverse else ops.getStress())
    return np.asarray(out)


def legacy_positive_selected(mat_tag, cal_hog, cal_sag, reverse):
    """The original prototype's mapping: Ke from whichever face the element calls positive, shared by both."""
    pos, neg = (cal_sag, cal_hog) if reverse else (cal_hog, cal_sag)
    ke = pos["my"] / pos["theta_slip_y"]
    branches = []
    for cal in (pos, neg):
        dp = cal["theta_slip_u"] - cal["theta_slip_y"]
        branches.append(RotationalBackbone(dp, 10.0 * dp, cal["theta_slip_y"] + 11.0 * dp, cal["my"], 1.25, 0.2))
    cyclic = CyclicParameters(1e12, 1e12, 1e12, 1.0, 1.0, 1.0, 1.0, 1.0, lamda_a=1e12, c_a=1.0, kappa_f=0.25, kappa_d=0.25)
    define_rotational_imk("IMKPinching", mat_tag, ke, *branches, cyclic,
                          provenance={"calibration_id": "legacy_positive_selected_ke_negative_control", "status": "verification_only",
                                      "deformation_scope": "negative control"})
    return ke


class InterfaceStiffness(unittest.TestCase):
    def test_flexibility_mean_is_invariant_to_face_order_and_records_the_branch_mismatch(self):
        hog, sag = calibrations()
        ke, rule = jss.interface_stiffness(hog, sag)
        ke_swapped, _ = jss.interface_stiffness(sag, hog)
        self.assertEqual(ke, ke_swapped)
        secants = rule["face_secants_kip_in_per_rad"]
        self.assertAlmostEqual(ke, 2.0 / (1.0 / secants["hogging"] + 1.0 / secants["sagging"]))
        self.assertTrue(secants["sagging"] > ke > secants["hogging"])
        mismatch = rule["branch_yield_rotation_mismatch"]
        self.assertLess(mismatch["hogging"], 0.0)
        self.assertGreater(mismatch["sagging"], 0.0)
        self.assertLess(max(abs(v) for v in mismatch.values()), 0.10)
        with self.assertRaises(ValueError):
            jss.interface_stiffness(hog, sag, rule="positive_selected")

    def test_incompatible_capping_target_is_refused(self):
        hog, sag = calibrations()
        ops.wipe()
        too_short = {**sag, "theta_slip_u": sag["my"] / jss.interface_stiffness(hog, sag)[0] * 0.5}
        with self.assertRaisesRegex(ValueError, "incompatible slip target"):
            jss._slip_material(1, hog, too_short, 0.25, 0.25, 1e12, reverse=False)
        ops.wipe()


class OrientationRegression(unittest.TestCase):
    def tearDown(self):
        ops.wipe()

    def test_repaired_interface_is_orientation_invariant_and_the_legacy_mapping_is_caught(self):
        hog, sag = calibrations()
        theta = protocol()
        ops.wipe()
        ops.model("basic", "-ndm", 1, "-ndf", 1)
        results = {}
        for reverse in (False, True):
            tag = 10 + int(reverse)
            ke, installed, _, rule, targets = jss._slip_material(tag, hog, sag, 0.25, 0.25, 1e12, reverse=reverse)
            results[reverse] = (drive(tag, theta, reverse), ke, installed)
        forward, backward = results[False][0], results[True][0]
        discrepancy = float(np.max(np.abs(forward - backward)) / np.max(np.abs(forward)))
        self.assertEqual(results[False][1], results[True][1], "Ke is a property of the calibration, not of the orientation")
        self.assertLess(discrepancy, jss.ORIENTATION_TOLERANCE)
        mapping = results[True][2]["provenance"]["energy_mapping"]
        self.assertTrue(mapping["coordinate_reversed"])
        self.assertEqual((mapping["positive_input_direction"], mapping["negative_input_direction"]), ("sagging", "hogging"))
        self.assertEqual(results[False][2]["positive"], results[True][2]["negative"])
        for mode in ("S", "C", "A", "K"):
            self.assertAlmostEqual(results[False][2]["reference_energies_kip_in_rad"][mode],
                                   results[True][2]["reference_energies_kip_in_rad"][mode])
        # negative control: the positive-selected Ke of the first prototype depends on the orientation
        legacy = {}
        for reverse in (False, True):
            tag = 20 + int(reverse)
            ke = legacy_positive_selected(tag, hog, sag, reverse)
            legacy[reverse] = (drive(tag, theta, reverse), ke)
        self.assertNotEqual(legacy[False][1], legacy[True][1])
        self.assertGreater(abs(legacy[False][1] - legacy[True][1]) / legacy[False][1], 0.10)
        control = float(np.max(np.abs(legacy[False][0] - legacy[True][0])) / np.max(np.abs(legacy[False][0])))
        self.assertGreater(control, 1e-2, "the negative control must expose the orientation error")

    def test_finite_synthetic_energy_is_mapped_invariantly(self):
        """A finite deterioration energy (a synthetic test value, not a calibration) must be carried
        through the reversal by the mapping, so the orientation stays invariant while deterioration acts."""
        hog, sag = calibrations()
        ke, _ = jss.interface_stiffness(hog, sag)
        theta = protocol()
        ops.wipe()
        ops.model("basic", "-ndm", 1, "-ndf", 1)
        branches = {}
        for sign, cal in (("hogging", hog), ("sagging", sag)):
            dp = cal["theta_slip_u"] - cal["my"] / ke
            branches[sign] = RotationalBackbone(dp, 10.0 * dp, cal["my"] / ke + 11.0 * dp, cal["my"], 1.25, 0.2)
        cyclic = CyclicParameters(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, lamda_a=1.0, c_a=1.0, kappa_f=0.25, kappa_d=0.25)
        profile = {"calibration_id": "synthetic_orientation_fixture", "status": "verification_only", "material_type": "IMKPinching",
                   "units": "kip-in*rad", "deformation_scope": "synthetic face rotation",
                   "derivation": "prescribed synthetic energy for the coordinate-invariance test; not experimental",
                   "applicability_basis": "software verification only", "source_refs": ["tests/test_joint_slip_subassembly.py"],
                   "specimen_ids": ["synthetic"], "energies_kip_in_rad": {m: 2000.0 for m in ("S", "C", "A", "K")}}
        results = {}
        for reverse in (False, True):
            tag = 30 + int(reverse)
            md = define_mapped_rotational_imk("IMKPinching", tag, ke, branches["hogging"], branches["sagging"], cyclic,
                                              calibration=profile, reverse=reverse, physical_directions=("hogging", "sagging"),
                                              verification_only=True,
                                              provenance={"calibration_id": "synthetic", "status": "verification_only",
                                                          "deformation_scope": "synthetic"})
            results[reverse] = (drive(tag, theta, reverse), md)
        discrepancy = float(np.max(np.abs(results[False][0] - results[True][0])) / np.max(np.abs(results[False][0])))
        self.assertLess(discrepancy, jss.ORIENTATION_TOLERANCE)
        self.assertEqual(results[False][1]["cyclic"]["d_pos"], results[True][1]["cyclic"]["d_neg"])
        # deterioration did act (the finite energy is small): the last cycle's peak is below the first cycle's at the same amplitude
        forward = results[False][0]
        amp = np.where(np.isclose(theta, 1.2e-2))[0]
        self.assertGreater(len(amp), 1)
        self.assertLess(forward[amp[-1]], forward[amp[0]])


if __name__ == "__main__":
    unittest.main()
