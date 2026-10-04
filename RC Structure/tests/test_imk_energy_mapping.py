"""Synthetic mechanics verification; these values do not calibrate RC."""
from dataclasses import replace
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
import openseespy.opensees as ops

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
from Model import IMK_Hinges as hinges
from Model.IMK_Materials import (MAPPING_VERSION, MATERIAL_TYPES, RotationalBackbone,
    CyclicParameters, active_energy_modes, define_mapped_rotational_imk, validate_energy_calibration)

POS = RotationalBackbone(.01, .05, .10, 100., 1.10, .10)
NEG = RotationalBackbone(.014, .055, .11, 160., 1.12, .15)
RESULTS, HISTORIES = {}, {}


def profile(kind, scale=1.):
    return {"calibration_id": "synthetic_energy_mapping_fixture", "status": "verification_only",
        "material_type": kind, "units": "kip-in*rad", "deformation_scope": "synthetic_member_end_spring",
        "source_refs": ["tests/test_imk_energy_mapping.py; invented verification values"],
        "specimen_ids": ["synthetic; no experimental specimen"],
        "derivation": "Independently chosen synthetic energies; not inferred from either branch",
        "applicability_basis": "Software verification only; forbidden for structural runs",
        "energies_kip_in_rad": {m: e * scale for m, e in zip(active_energy_modes(kind), (50., 65., 80., 95.))}}


def cyclic(kind):
    return CyclicParameters(.5, .5, .5, 1., 1., 1., .65, .90,
        lamda_a=None if kind == "IMKBilin" else .5, c_a=None if kind == "IMKBilin" else 1.,
        kappa_f=.4 if kind == "IMKPinching" else None, kappa_d=.6 if kind == "IMKPinching" else None)


def history(amplitude, intervals=100):
    values = [0.]
    for target in [amplitude, 0., -amplitude, 0.] * 3:
        values.extend(np.linspace(values[-1], target, intervals + 1)[1:])
    return np.array(values)


def analyze_displacements(node, dof, values, tolerance=1e-10):
    ops.system("BandGeneral")
    ops.numberer("RCM")
    # All fixture constraints are homogeneous SP/identity equalDOF constraints;
    # eliminate them exactly rather than introducing a stiff penalty scale.
    ops.constraints("Plain")
    ops.test("NormDispIncr", tolerance, 80)
    ops.algorithm("Newton")
    ops.integrator("DisplacementControl", node, dof, 1e-6)
    ops.analysis("Static")
    for target in values[1:]:
        ops.integrator("DisplacementControl", node, dof, float(target - ops.nodeDisp(node, dof)))
        if ops.analyze(1) != 0:
            raise AssertionError(f"Synthetic cyclic step failed at {target}")
        yield


def spring(kind, reverse=False, scale=1.):
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    ops.node(1, 0., 0., 0.)
    ops.node(2, 0., 0., 0.)
    ops.fix(1, 1, 1, 1, 1, 1, 1)
    ops.equalDOF(1, 2, 1, 2, 3, 4, 6)
    md = define_mapped_rotational_imk(kind, 11, 10000. * scale,
        replace(POS, fy=POS.fy * scale), replace(NEG, fy=NEG.fy * scale), cyclic(kind),
        calibration=profile(kind, scale), reverse=reverse, verification_only=True,
        physical_directions=("physical_a", "physical_b"),
        provenance={"calibration_id": "synthetic", "status": "verification_only",
                    "deformation_scope": "isolated_3d_spring"})
    ops.element("zeroLength", 21, 1, 2, "-mat", 11, "-dir", 5)
    assert ops.nodeCoord(1) == ops.nodeCoord(2) and ops.eleNodes(21) == [1, 2]
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 0., 0., 0., 0., 1., 0.)
    rows = [[0., 0.]]
    values = history(.025) * (-1 if reverse else 1)
    for _ in analyze_displacements(2, 5, values):
        pair = ops.eleResponse(21, "material", 1, "stressStrain")
        np.testing.assert_allclose(pair, [ops.eleResponse(21, "basicForce")[0],
                                       ops.eleResponse(21, "deformation")[0]], atol=1e-10)
        rows.append(pair)
    result = np.array(rows)
    np.testing.assert_allclose(result[:, 1], values, atol=1e-10)
    ops.wipe()
    return result, md


def member(kind, axis, reversed_nodes=False, fixed_high=False):
    column = axis.startswith("column_")
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    hinges.reset_hinge_registry()
    ops.node(1, 0., 0., 0.)
    ops.node(2, *((0., 0., 240.) if column else (240., 0., 0.) if axis == "x" else (0., 240., 0.)))
    ops.geomTransf("Linear", 9, *((0, 1, 0) if column else (0, 0, 1)))
    fixed, free = (2, 1) if fixed_high else (1, 2)
    ops.fix(fixed, 1, 1, 1, 1, 1, 1)
    ni, nj = (2, 1) if reversed_nodes else (1, 2)
    # Columns in the default fixture have larger moments. These independent
    # synthetic energies keep the orientation check away from loss of all
    # lateral resistance; this is not an experimental calibration rule.
    fixture = {e: {a: profile(kind, 200. if column else 20.) for a in ("y", "z")} for e in ("i", "j")}
    values = {"IMK_MATERIAL_TYPE": kind, "IMK_ENERGY_MAPPING_MODE": MAPPING_VERSION,
              "IMK_D_POS": .65, "IMK_D_NEG": .90, "IMK_FMAXFY_POS": 1.10,
              "IMK_FMAXFY_NEG": 1.12, "IMK_FRESFY_POS": .10, "IMK_FRESFY_NEG": .15}
    bb = {"theta_p": .01, "theta_pc": .05, "theta_u": .1,
          "theta_p_neg": .014, "theta_pc_neg": .055, "theta_u_neg": .11,
          "source": "synthetic directionally asymmetric regression"}
    with mock.patch.multiple(sp, **values), mock.patch.object(hinges, "backbone_for_member", return_value=bb), \
         mock.patch.object(hinges, "beam_yield_moments", return_value=(2000., 1000., {"basis": "synthetic"})), \
         contextlib.redirect_stdout(io.StringIO()):
        hinges.create_imk_member(1, ni, nj, "column" if column else "beam_" + axis, 9,
                                 _verification_calibrations=fixture)
    registry = hinges.hinge_registry()[1]
    for end_id, node in ((1, ni), (2, nj)):
        assert hinges.hinge_node_tag(1, end_id) != node
        assert ops.nodeCoord(node) == ops.nodeCoord(hinges.hinge_node_tag(1, end_id))
    tag = hinges.hinge_element_tag(1, 1 if fixed == ni else 2)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    dof = (1 if axis.endswith("x") else 2) if column else 3
    load = [0.] * 6
    load[dof - 1] = 1.
    ops.load(free, *load)
    rows = [[0., 0.]]
    material_index = 2 if column and dof == 1 else 1
    # A column reversal can oscillate at a ~5e-9 increment in this synthetic
    # assembly. Use 1e-8 and independently bound equilibrium at every step;
    # these fixture tolerances do not change any production solver settings.
    max_equilibrium_error = 0.
    for _ in analyze_displacements(free, dof, history(8., 80), 1e-8 if column else 1e-10):
        pair = np.array(ops.eleResponse(tag, "material", material_index, "stressStrain"))
        moment_scale = max(registry["yield_moment_y_kip_in"], registry["yield_moment_z_kip_in"])
        error = abs(abs(pair[0]) - abs(ops.getLoadFactor(1) * 240.)) / moment_scale
        max_equilibrium_error = max(max_equilibrium_error, error)
        assert error < 1e-5, f"Cantilever moment equilibrium error {error}"
        rows.append(pair * (-1 if fixed_high else 1))
    registry["fixture_equilibrium_relative_error"] = max_equilibrium_error
    ops.wipe()
    return np.array(rows), registry


class EnergyMappingTests(unittest.TestCase):
    def tearDown(self):
        ops.wipe()

    def test_native_3d_coordinate_invariance_and_moment_rescaling(self):
        for kind in MATERIAL_TYPES:
            with self.subTest(material=kind):
                reference, md = spring(kind)
                reversed_result, reverse_md = spring(kind, reverse=True)
                factor = .112984829027
                scaled, _ = spring(kind, scale=factor)
                error = float(np.max(abs(reference + reversed_result)))
                scale_error = float(np.max(abs(reference[:, 0] - scaled[:, 0] / factor)))
                self.assertLess(error, 1e-7)
                self.assertLess(scale_error, 1e-7)
                self.assertGreater(float(np.max(abs(reference[:, 0]))), POS.fy)
                self.assertGreater(float(np.max(abs(reference[1:401, 0] - reference[801:1201, 0]))), 1.)
                self.assertEqual(md["positive"], reverse_md["negative"])
                self.assertEqual(md["negative"], reverse_md["positive"])
                self.assertEqual(md["cyclic"]["d_pos"], reverse_md["cyclic"]["d_neg"])
                for mode, energy in profile(kind)["energies_kip_in_rad"].items():
                    self.assertAlmostEqual(md["reference_energies_kip_in_rad"][mode], energy)
                    self.assertAlmostEqual(reverse_md["reference_energies_kip_in_rad"][mode], energy)
                RESULTS[kind] = {"points_per_history": len(reference), "failed_steps": 0,
                    "coordinate_reversal_error": error, "moment_rescaling_error": scale_error}
                HISTORIES[kind + "_reference"] = reference
                HISTORIES[kind + "_reversed_back"] = -reversed_result

    def test_assembled_beams_both_axes_ends_and_connectivities(self):
        for kind in ("IMKBilin", "IMKPeakOriented"):
            reference, _ = member(kind, "x")
            errors = []
            for axis in ("x", "y"):
                for reverse in (False, True):
                    for high in (False, True):
                        with self.subTest(material=kind, axis=axis, reverse=reverse, fixed_high=high):
                            actual, registry = member(kind, axis, reverse, high)
                            np.testing.assert_allclose(actual, reference, rtol=2e-7, atol=2e-6)
                            self.assertTrue(registry["verification_only"])
                            errors.append(float(np.max(abs(actual - reference))))
            RESULTS[kind + "_assembled"] = {"variants": 8, "points_per_history": len(reference),
                "max_moment_rotation_difference": max(errors), "failed_steps": 0}

    def test_missing_or_unreviewed_profiles_fail_before_domain_mutation(self):
        for bad in (None, {}, {"i": {}, "j": {}}):
            with self.subTest(profile=bad), mock.patch.multiple(sp, IMK_ENERGY_MAPPING_MODE=MAPPING_VERSION,
                    IMK_MEMBER_ENERGY_CALIBRATIONS={"1": bad}), mock.patch.object(hinges, "_create_hinge_node") as mutate:
                with self.assertRaises(ValueError):
                    hinges.create_imk_member(1, 1, 2, "beam_x", 9)
                mutate.assert_not_called()
        synthetic = {e: {a: profile("IMKPeakOriented") for a in ("y", "z")} for e in ("i", "j")}
        with mock.patch.multiple(sp, IMK_MATERIAL_TYPE="IMKPeakOriented", IMK_ENERGY_MAPPING_MODE=MAPPING_VERSION,
                                 IMK_MEMBER_ENERGY_CALIBRATIONS={"1": synthetic}):
            with self.assertRaisesRegex(ValueError, "experimentally_supported"):
                hinges.create_imk_member(1, 1, 2, "beam_x", 9)

    def test_assembled_columns_both_bending_axes_and_connectivities(self):
        for kind in ("IMKBilin", "IMKPeakOriented"):
            errors, equilibrium_errors = [], []
            for axis in ("column_x", "column_y"):
                reference, _ = member(kind, axis)
                for reverse in (False, True):
                    for high in (False, True):
                        with self.subTest(material=kind, axis=axis, reverse=reverse, fixed_high=high):
                            actual, registry = member(kind, axis, reverse, high)
                            np.testing.assert_allclose(actual, reference, rtol=2e-7, atol=2e-6)
                            errors.append(float(np.max(abs(actual - reference))))
                            equilibrium_errors.append(registry["fixture_equilibrium_relative_error"])
            RESULTS[kind + "_columns"] = {"variants": 8, "points_per_history": len(reference),
                "max_moment_rotation_difference": max(errors), "failed_steps": 0,
                "max_relative_moment_equilibrium_error": max(equilibrium_errors),
                "displacement_increment_tolerance": 1e-8}

    def test_output_identity_rejects_changed_mapping_or_profile(self):
        from Data_Generation.Graph_Exporter import collect_global_parameters
        from Ground_Motion_Main import validate_ntha_output_compatibility
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "global_parameters.json").write_text(json.dumps(collect_global_parameters()))
            for key, value, field in (("IMK_ENERGY_MAPPING_MODE", MAPPING_VERSION, "imk_energy_mapping_mode"),
                    ("IMK_MEMBER_ENERGY_CALIBRATIONS", {"synthetic_identity_probe": {}}, "imk_member_energy_calibrations")):
                with self.subTest(field=field), mock.patch.object(sp, key, value):
                    with self.assertRaisesRegex(RuntimeError, field):
                        validate_ntha_output_compatibility(path)

    def test_profile_contract_rejects_invalid_energy_and_missing_review(self):
        kind = "IMKPeakOriented"
        for value in (0., -1., float("nan"), float("inf"), True):
            p = profile(kind)
            p["energies_kip_in_rad"]["S"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_energy_calibration(p, kind, verification_only=True)
        p = profile(kind)
        p["status"] = "experimentally_supported"
        with self.assertRaisesRegex(ValueError, "reviewed_by"):
            validate_energy_calibration(p, kind)
        p = profile(kind)
        snapshot = validate_energy_calibration(p, kind, verification_only=True)
        p["energies_kip_in_rad"]["S"] = 999.
        self.assertEqual(snapshot["energies_kip_in_rad"]["S"], 50.)


# These hand totals describe the framed members; the column extension above the roof is verified on its own
# (tests/test_roof_extension.py), so this module runs with the terminating roof column.
def setUpModule():
    global _ROOF_PATCH
    from unittest import mock as _mock
    import Structure_Parameters as _sp
    _ROOF_PATCH = _mock.patch.object(_sp, "ROOF_COLUMN_EXTENSION", False)
    _ROOF_PATCH.start()


def tearDownModule():
    _ROOF_PATCH.stop()


if __name__ == "__main__":
    unittest.main()
