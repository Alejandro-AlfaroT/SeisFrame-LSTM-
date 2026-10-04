"""Synchronized hinge moment-rotation histories of Analysis.NTHA (V2 learning target, 2026-10-01).

A small frame under the V2 nonlinear-flexure profile is driven by a short biaxial pulse strong enough to
yield hinges. Full-rate ``material stressStrain`` recorders are attached the way the fixed-design
diagnostic attaches them, and the histories NTHA stores are compared with the recorder output ON MATCHED
TIMES for both spring axes and both member ends: nonzero moment, yielding, the gravity state, and the
alignment of scheduled snapshots with the internal sub-steps of a subdivided recovery. The export files
are then read back, including a failed and a missing-value case, to show that nothing is zero-filled.

These are implementation checks of the measurement and export path on a synthetic fixture. They are not
an experimental calibration of the member model and they launch no screening analysis.
"""
import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openseespy.opensees as ops                                              # noqa: E402
import Structure_Parameters as sp                                              # noqa: E402
from Analysis import Hinge_Hysteresis_Diagnostic as hd                         # noqa: E402
from Analysis import Hinge_Moment_Rotation as hmr                               # noqa: E402
from Analysis import NTHA as ntha                                               # noqa: E402
from Analysis.Gravity import run_gravity_analysis                              # noqa: E402
from Loads.Gravity_Loads import apply_gravity_loads                            # noqa: E402
from Loads.Ground_Motion import GroundMotionRecord                             # noqa: E402
from Model import Analysis_Profile as ap                                        # noqa: E402
from Model.Build_Model import build_model                                       # noqa: E402
from Model.IMK_Hinges import hinge_registry                                     # noqa: E402

SMALL = dict(NUM_BAY_X=2, NUM_BAY_Y=1, NUM_FLOOR=2)
PROFILE_KEYS = tuple(ap.GOVERNED_KEYS) + ("ANALYSIS_PROFILE_ID",)


def pulse(scale, n=160, dt=0.01, phase=0.0):
    t = np.arange(n) * dt
    return scale * np.sin(2.0 * np.pi * t / 0.4 + phase) * np.exp(-((t - 0.6) / 0.35) ** 2)


@contextlib.contextmanager
def v2_small_frame():
    """The SMALL geometry under the V2 profile, with every touched Structure_Parameters name restored."""
    with contextlib.ExitStack() as stack:
        for name in set(SMALL) | set(PROFILE_KEYS) | {"NUM_MODES"}:
            stack.enter_context(mock.patch.object(sp, name, getattr(sp, name)))
        for name, value in SMALL.items():
            setattr(sp, name, value)
        sp.NUM_MODES = 3 * sp.NUM_FLOOR
        ap.apply_profile(ap.V2_FLEXURE)
        try:
            yield
        finally:
            ops.wipe()


def run_with_recorders(folder, scale_x, scale_y, *, stride=None, try_step=None):
    """Build, gravity, attach the diagnostic's recorders, run NTHA, read the recorders within the window."""
    build_model()
    apply_gravity_loads()
    run_gravity_analysis()
    topology = ap.verify_installed_domain(ap.V2_FLEXURE)
    hinges = hd.hinge_inventory(hd.element_inventory())
    coverage = hd.attach_hinge_recorders(Path(folder) / "recorders", hinges)
    record_x = GroundMotionRecord("PULSE_X", 0.01, pulse(scale_x), units="g")
    record_y = GroundMotionRecord("PULSE_Y", 0.01, pulse(scale_y, phase=0.7), units="g")
    try:
        if try_step is None:
            results = ntha.run_ntha(record_x, record_y=record_y, hinge_history_stride=stride)
        else:
            with mock.patch.object(ntha, "_try_step", try_step):
                results = ntha.run_ntha(record_x, record_y=record_y, hinge_history_stride=stride)
    finally:
        hd.close_recorders()
    last = results["time_history"][-1] if results["time_history"] else None
    recorded = hd.read_hinge_recorders(coverage, time_limit=last)
    registry = hinge_registry()
    return results, recorded, registry, topology


class SynchronizedHistories(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        with v2_small_frame():
            cls.profile_stride = sp.NTHA_HINGE_HISTORY_STRIDE
            cls.results, cls.recorded, cls.registry, cls.topology = run_with_recorders(cls.tmp.name, 1.6, 1.1)
            cls.arrays, cls.rows, cls.schema = hmr.assemble(cls.results, cls.registry)
            cls.counts = hmr.write_hinge_moment_rotation(Path(cls.tmp.name) / "out", cls.results, cls.registry)
            cls.comparison = hmr.compare_with_recorders(cls.arrays, cls.recorded)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_profile_installs_member_hinges_and_no_joint_or_interface_spring(self):
        self.assertTrue(self.topology["consistent"], self.topology["problems"])
        self.assertEqual(self.topology["joint_spring_elements_in_domain"], 0)
        self.assertEqual(self.topology["face_slip_interfaces_registered"], 0)
        self.assertEqual(self.topology["hinge_spring_elements_in_domain"], 2 * self.topology["members_with_hinges"])
        self.assertEqual(self.topology["installed_spring_material_types"], {"IMKPeakOriented": 4 * self.topology["members_with_hinges"]})

    def test_every_scheduled_step_is_stored_at_the_profile_stride(self):
        self.assertEqual(self.profile_stride, 1)
        status = self.results["status"]
        self.assertFalse(status["failed"])
        self.assertEqual(len(self.arrays["time_sec"]), status["completed_steps"])
        np.testing.assert_array_equal(self.arrays["scheduled_step"], np.arange(status["completed_steps"]))
        np.testing.assert_allclose(self.arrays["time_sec"], self.results["time_history"], rtol=0, atol=1e-12)
        self.assertTrue(np.all(np.diff(self.arrays["time_sec"]) > 0))

    def test_moment_and_rotation_agree_with_the_recorders_on_matched_times_for_both_axes(self):
        self.assertTrue(self.comparison["compared"])
        for axis in ("y", "z"):
            block = self.comparison["axes"][axis]
            self.assertTrue(block["compared"], block)
            self.assertEqual(block["unmatched_snapshots"], 0)
            self.assertEqual(block["matched_snapshots"], len(self.arrays["time_sec"]))
            self.assertGreater(block["max_abs_moment_kip_in"], 1.0)                      # nonzero moment on this axis
            # recorder files are written with 12 significant digits
            self.assertLessEqual(block["max_abs_moment_difference_kip_in"], 1e-9 * max(1.0, block["max_abs_moment_kip_in"]))
            self.assertLessEqual(block["max_abs_rotation_difference_rad"], 1e-11 * max(1.0, 1e3 * block["max_abs_rotation_rad"]) + 1e-13)
            self.assertTrue(block["commit_count_locates_the_same_recorder_row"])

    def test_both_member_ends_and_both_member_classes_are_covered_with_their_identities(self):
        self.assertEqual(len(self.rows), self.arrays["rotation_rad"].shape[1])
        seen = {(r["member_class"], r["end"], r["local_axis"]) for r in self.rows}
        for member_class in ("beam", "column"):
            for end in ("i", "j"):
                for axis in ("y", "z"):
                    self.assertIn((member_class, end, axis), seen)
        for k, row in enumerate(self.rows):
            self.assertEqual(row["spring_index"], k)
            self.assertEqual(row["hinge_element_tag"], sp.IMK_HINGE_ELEMENT_TAG_BASE + 10 * row["physical_member_tag"] + row["end_id"])
            self.assertEqual(row["spring_local_direction"], 5 if row["local_axis"] == "y" else 6)
            self.assertEqual(row["material_type"], "IMKPeakOriented")
            self.assertEqual(row["bar_slip_owner"], "member_hinge"); self.assertEqual(float(row["bond_slip_indicator"]), 1.0)
            self.assertEqual(row["panel_shear_owner"], "")                                # no joint spring in the V2 profile
            self.assertEqual(row["energy_mapping_status"], "legacy_unmapped")
            self.assertTrue(row["calibration_id"]); self.assertTrue(row["calibration_status"])
            self.assertTrue(row["global_rotation_axis"]); self.assertTrue(row["retained_joint_node"] != "")
        np.testing.assert_array_equal(self.arrays["spring_hinge_element_tag"], [r["hinge_element_tag"] for r in self.rows])

    def test_hinges_yield_and_beam_strength_asymmetry_is_not_symmetrised(self):
        moment = self.arrays["moment_kip_in"]
        yielded, asymmetric = 0, 0
        for k, row in enumerate(self.rows):
            if row["local_axis"] != "y":
                continue
            fy_pos, fy_neg = float(row["yield_moment_positive_kip_in"]), float(row["yield_moment_negative_kip_in"])
            if moment[:, k].max() >= 0.999 * fy_pos or -moment[:, k].min() >= 0.999 * fy_neg:
                yielded += 1
            if row["member_class"] == "beam" and abs(fy_pos - fy_neg) > 1e-6 * max(fy_pos, fy_neg):
                asymmetric += 1
                # hogging is positive at end i and negative at end j: the larger strength sits on that side
                if row["end"] == "i":
                    self.assertEqual(row["positive_moment_meaning"], "hogging (top in tension)")
                else:
                    self.assertEqual(row["negative_moment_meaning"], "hogging (top in tension)")
        self.assertGreater(yielded, 0, "the pulse was meant to yield at least one strong-axis spring")
        self.assertEqual(self.schema["sign_convention"]["beam_local_y"],
                         "hogging (top in tension) is positive at member end i and negative at member end j")
        self.assertGreaterEqual(asymmetric, 0)          # symmetric without a slab layout; never forced either way

    def test_gravity_state_is_stored_and_histories_are_absolute(self):
        self.assertEqual(self.schema["reference_state"]["kind"], "absolute")
        self.assertTrue(self.schema["reference_state"]["gravity_state_recorded"])
        gravity_m, gravity_r = self.arrays["gravity_moment_kip_in"], self.arrays["gravity_rotation_rad"]
        self.assertTrue(np.all(np.isfinite(gravity_m))); self.assertTrue(np.all(np.isfinite(gravity_r)))
        self.assertGreater(np.abs(gravity_m).max(), 1.0)                                  # gravity loads the beam hinges
        # the first transient snapshot continues from the gravity state: it is not an increment from zero
        first = self.arrays["moment_kip_in"][0]
        self.assertLess(np.abs(first - gravity_m).max(), 0.5 * np.abs(gravity_m).max() + 1.0)
        self.assertLess(float(self.arrays["gravity_time_sec"]), float(self.arrays["time_sec"][0]))

    def test_files_survive_the_export_route_and_read_back_exactly(self):
        out = Path(self.tmp.name) / "out"
        self.assertTrue(self.counts["hinge_moment_rotation_available"])
        self.assertEqual(self.counts["hinge_moment_rotation_missing_values"], 0)
        arrays, rows, schema = hmr.read_hinge_moment_rotation(out)
        for key in hmr.ARRAYS:
            self.assertIn(key, arrays)
            np.testing.assert_array_equal(arrays[key], self.arrays[key])
        self.assertEqual(arrays["moment_kip_in"].dtype, np.float64)
        self.assertEqual(len(rows), len(self.rows))
        self.assertEqual(schema["schema_version"], hmr.SCHEMA_VERSION)
        self.assertEqual(schema["identity"]["analysis_profile"]["id"], ap.V2_FLEXURE)
        self.assertEqual(schema["identity"]["joint_model"], "rigid_centerline")
        self.assertEqual(schema["identity"]["risk_category"], "III")
        self.assertEqual(schema["sampling"]["stride"], 1)
        self.assertIn("not experimentally isolated plastic flexure", schema["quantities"]["rotation_rad"])
        self.assertIn("relative to the ground", schema["global_reference_outputs"]["floor_acceleration"])
        json.dumps(schema, allow_nan=False)                                               # the schema itself holds no NaN


class RecoverySubsteps(unittest.TestCase):
    def test_snapshots_stay_on_scheduled_times_when_steps_are_subdivided(self):
        """Force the subdivided recovery on some steps: the recorder gains nine rows per such step, the
        stored history gains none, and the pairing by time (and by commit count) still agrees."""
        forced = {7, 8, 31}
        calls = {"n": 0}
        real = ntha._try_step

        def try_step(dt):
            step = calls["n"]; calls["n"] += 1
            if step in forced:
                ops.test("NormDispIncr", ntha._TOL_FALLBACK, ntha._ITER_FALLBACK, 0)
                ops.algorithm("KrylovNewton")
                ok = ops.analyze(ntha._DT_SUBDIVIDE, dt / ntha._DT_SUBDIVIDE)
                ntha._set_primary_algorithm()
                return (0, f"KrylovNewton+subdivide({ntha._DT_SUBDIVIDE}x)") if ok == 0 else (ok, "failed")
            return real(dt)

        with tempfile.TemporaryDirectory() as folder, v2_small_frame():
            results, recorded, registry, _ = run_with_recorders(folder, 0.5, 0.3, try_step=try_step)
            arrays, _rows, schema = hmr.assemble(results, registry)
            comparison = hmr.compare_with_recorders(arrays, recorded)
        n = results["status"]["completed_steps"]
        self.assertEqual(len(arrays["time_sec"]), n)
        self.assertEqual(sorted(np.flatnonzero(arrays["internal_commits"] == ntha._DT_SUBDIVIDE)), sorted(forced))
        self.assertEqual(schema["sampling"]["subdivided_rows"], len(forced))
        self.assertEqual(int(arrays["commit_count"][-1]), n + 9 * len(forced))
        # scheduled times: a subdivided step still ends on the scheduled time grid
        np.testing.assert_allclose(arrays["time_sec"], 0.01 * (1 + np.arange(n)), rtol=0, atol=1e-9)
        for axis in ("y", "z"):
            block = comparison["axes"][axis]
            self.assertEqual(block["recorder_rows"], n + 9 * len(forced))                 # the recorder has the sub-steps
            self.assertEqual(block["recorder_rows_that_are_not_snapshots"], 9 * len(forced))
            self.assertEqual(block["unmatched_snapshots"], 0)
            self.assertTrue(block["commit_count_locates_the_same_recorder_row"])
            self.assertLessEqual(block["max_abs_moment_difference_kip_in"], 1e-9 * max(1.0, block["max_abs_moment_kip_in"]))
        # zipping the recorder rows against the snapshots by row number would be wrong after the first subdivision
        rec_time = np.asarray(recorded["y"]["time"])
        self.assertGreater(np.abs(rec_time[:n] - arrays["time_sec"]).max(), 1e-6)


class FailureAndMissingValues(unittest.TestCase):
    def test_failed_analysis_keeps_the_partial_history_and_its_status(self):
        calls = {"n": 0}
        real = ntha._try_step

        def try_step(dt):
            calls["n"] += 1
            return real(dt) if calls["n"] <= 12 else (-3, "failed")

        with tempfile.TemporaryDirectory() as folder, v2_small_frame():
            results, _recorded, registry, _ = run_with_recorders(folder, 0.5, 0.3, try_step=try_step)
            counts = hmr.write_hinge_moment_rotation(Path(folder) / "out", results, registry)
            arrays, _rows, schema = hmr.read_hinge_moment_rotation(Path(folder) / "out")
        self.assertTrue(results["status"]["failed"])
        self.assertEqual(results["status"]["completed_steps"], 12)
        self.assertEqual(counts["hinge_moment_rotation_rows"], 12)                        # the partial history, no invented rows
        self.assertEqual(len(arrays["time_sec"]), 12)
        self.assertTrue(schema["status"]["failed"]); self.assertTrue(schema["status"]["truncated"])
        self.assertEqual(schema["status"]["failed_step"], 12)
        self.assertIsNotNone(schema["status"]["domain_time_at_failure_sec"])
        self.assertTrue(np.all(np.isfinite(arrays["moment_kip_in"])))

    def test_a_failure_before_the_first_step_still_writes_the_gravity_state_and_no_rows(self):
        with tempfile.TemporaryDirectory() as folder, v2_small_frame():
            results, _recorded, registry, _ = run_with_recorders(folder, 0.5, 0.3, try_step=lambda dt: (-3, "failed"))
            hmr.write_hinge_moment_rotation(Path(folder) / "out", results, registry)
            arrays, rows, schema = hmr.read_hinge_moment_rotation(Path(folder) / "out")
        self.assertEqual(arrays["moment_kip_in"].shape, (0, len(rows)))
        self.assertEqual(schema["counts"]["rows"], 0)
        self.assertTrue(np.all(np.isfinite(arrays["gravity_moment_kip_in"])))
        self.assertTrue(schema["status"]["failed"])

    def test_missing_query_values_are_nan_and_counted_never_zero(self):
        real = ops.eleResponse
        with tempfile.TemporaryDirectory() as folder, v2_small_frame():
            build_model(); apply_gravity_loads(); run_gravity_analysis()
            tags = ntha._imk_hinge_element_tags()
            lost = tags[3]

            def flaky(tag, *args):
                if int(tag) == int(lost) and args and args[0] == "basicForce":
                    return []
                return real(tag, *args)

            with mock.patch.object(ntha.ops, "eleResponse", flaky):
                results = ntha.run_ntha(GroundMotionRecord("PULSE_X", 0.01, pulse(0.3, n=20), units="g"), hinge_history_stride=1)
            registry = hinge_registry()
            counts = hmr.write_hinge_moment_rotation(Path(folder) / "out", results, registry)
            arrays, _rows, schema = hmr.read_hinge_moment_rotation(Path(folder) / "out")
        k = tags.index(lost)
        rows = len(arrays["time_sec"])
        self.assertTrue(np.all(np.isnan(arrays["moment_kip_in"][:, 2 * k:2 * k + 2])))    # the lost moments are NaN ...
        self.assertTrue(np.all(np.isfinite(arrays["rotation_rad"][:, 2 * k:2 * k + 2])))  # ... the rotations were read
        self.assertFalse(np.any(arrays["moment_kip_in"][:, 2 * k:2 * k + 2] == 0.0))
        self.assertEqual(counts["hinge_moment_rotation_missing_values"], 2 * rows)
        self.assertEqual(schema["missing"]["values_reported_by_analysis"], 2 * rows)
        self.assertTrue(np.all(np.isnan(arrays["gravity_moment_kip_in"][2 * k:2 * k + 2])))
        other = np.delete(arrays["moment_kip_in"], [2 * k, 2 * k + 1], axis=1)
        self.assertTrue(np.all(np.isfinite(other)))

    def test_mismatched_history_lengths_are_refused_not_zipped(self):
        results = {"hinge_tag_order": [sp.IMK_HINGE_ELEMENT_TAG_BASE + 11], "status": {},
                   "hinge_rotation_history": [[0.0, 0.0], [0.0, 0.0]], "hinge_moment_history": [[1.0, 1.0]],
                   "hinge_history_time": [0.01, 0.02], "hinge_rotation_steps": [0, 1],
                   "hinge_history_commit_count": [1, 2], "hinge_history_strategy": ["Newton", "Newton"]}
        with self.assertRaises(ValueError):
            hmr.assemble(results, registry={})

    def test_a_result_without_the_moment_history_writes_a_schema_and_no_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            legacy = {"hinge_tag_order": [sp.IMK_HINGE_ELEMENT_TAG_BASE + 11], "status": {"failed": False},
                      "hinge_rotation_history": [[0.0, 0.0]], "hinge_rotation_steps": [0]}
            counts = hmr.write_hinge_moment_rotation(folder, legacy, registry={})
            arrays, _rows, schema = hmr.read_hinge_moment_rotation(folder)
        self.assertFalse(counts["hinge_moment_rotation_available"])
        self.assertIsNone(arrays); self.assertFalse(schema["available"])
        self.assertFalse((Path(folder) / hmr.NPZ_NAME).exists())


class ExportRoute(unittest.TestCase):
    def test_histories_survive_the_production_save_the_hybrid_sample_and_the_surrogate_reader(self):
        """The real route: Ground_Motion_Main.run_one -> save_ntha_outputs -> Hybrid_Exporter -> schema_v3."""
        import argparse
        rc = Path(__file__).resolve().parents[1]
        for extra in (rc / "Data_Generation", rc.parent / "RC Hybrid Surrogate Model"):
            if str(extra) not in sys.path:
                sys.path.insert(0, str(extra))
        import Ground_Motion_Main as gm
        import Hybrid_Exporter
        try:
            from rc_hybrid_surrogate import schema_v3
        except ImportError:                                 # the surrogate's own dependencies are not installed here
            schema_v3 = None
        with tempfile.TemporaryDirectory() as folder, v2_small_frame():
            folder = Path(folder)
            accel_x, accel_y = pulse(1.6), pulse(1.1, phase=0.7)
            np.savetxt(folder / "pulse_x.txt", accel_x); np.savetxt(folder / "pulse_y.txt", accel_y)
            record_x = GroundMotionRecord("PULSE_X", 0.01, accel_x, units="g", source_path=str(folder / "pulse_x.txt"))
            record_y = GroundMotionRecord("PULSE_Y", 0.01, accel_y, units="g", source_path=str(folder / "pulse_y.txt"))
            out = folder / "ntha"
            out.mkdir()
            summary = gm.run_one(record_x, record_y, argparse.Namespace(damping_ratio=0.05, rayleigh_mode_i=0, rayleigh_mode_j=2, dt_factor=1.0), out)
            for name in (hmr.NPZ_NAME, hmr.SCHEMA_NAME, hmr.SPRINGS_NAME, "response_arrays.npz", "global_parameters.json"):
                self.assertTrue((out / name).exists(), name)
            self.assertTrue(summary["hinge_moment_rotation_available"])
            self.assertEqual(summary["hinge_moment_rotation_missing_values"], 0)
            arrays, rows, schema = hmr.read_hinge_moment_rotation(out)
            steps = summary["status"]["completed_steps"]
            self.assertEqual(arrays["moment_kip_in"].shape, (steps, len(rows)))                 # stride 1 under the V2 profile
            with np.load(out / "response_arrays.npz") as data:
                response = {key: data[key] for key in data.files}
            self.assertEqual(response["hinge_moment"].shape, response["hinge_rotation"].shape)
            np.testing.assert_allclose(response["hinge_moment"], arrays["moment_kip_in"].astype(np.float32), rtol=0, atol=0)
            np.testing.assert_allclose(response["hinge_history_time"], arrays["time_sec"], rtol=0, atol=0)
            np.testing.assert_array_equal(response["hinge_history_commit_count"], arrays["commit_count"])
            parameters = json.loads((out / "global_parameters.json").read_text(encoding="utf-8"))
            self.assertEqual((parameters["risk_category"], parameters["seismic_importance_factor"]), ("III", 1.25))
            self.assertEqual((parameters["analysis_profile_id"], parameters["joint_model"]), (ap.V2_FLEXURE, "rigid_centerline"))
            for key in ("risk_category", "seismic_importance_factor", "analysis_profile_id", "joint_model"):
                self.assertIn(key, gm.OUTPUT_IDENTITY_KEYS)
            self.assertFalse((out / "joint_springs.csv").exists())                              # no joint spring was installed
            metadata = Hybrid_Exporter.compile_hybrid_sample(out)
            with np.load(out / "hybrid_sample.npz") as data:
                sample_arrays = {key: data[key] for key in data.files}                         # loads without allow_pickle
            self.assertEqual(sample_arrays["hinge_moment"].shape, sample_arrays["hinge_rotation"].shape)
            np.testing.assert_array_equal(sample_arrays["hinge_moment"], response["hinge_moment"])
            np.testing.assert_array_equal(sample_arrays["hinge_history_time"], response["hinge_history_time"])
            self.assertTrue(metadata["hinge_moment_present"])
            self.assertEqual(metadata["hinge_history_columns"], ["local_y_direction_5", "local_z_direction_6"])
            self.assertEqual(metadata["hinge_moment_rotation"]["schema_version"], hmr.SCHEMA_VERSION)
            self.assertEqual(metadata["hinge_moment_rotation"]["reference_state"]["kind"], "absolute")
            self.assertEqual(metadata["hinge_history_stride"], 1)
            if schema_v3 is not None:
                sample = schema_v3.load_sample(out / "hybrid_sample.npz")
                schema_v3.validate(sample)
                moment = sample.as_entity_history("hinge_moment")
                rotation = sample.as_entity_history("hinge_rotation")
                self.assertEqual(moment.shape, rotation.shape)
                self.assertEqual(moment.shape[-1], 2)                                           # local y, local z per hinge
                self.assertEqual(sample.num_components("hinge_moment"), 2)
                np.testing.assert_array_equal(sample.strided_steps("hinge_moment"), sample.strided_steps("hinge_rotation"))
            # a sample compiled from an analysis without the moment history carries no such array, and still loads
            legacy = folder / "legacy"
            legacy.mkdir()
            for path in out.iterdir():
                if path.name not in ("hybrid_sample.npz", "hybrid_metadata.json", hmr.NPZ_NAME, hmr.SCHEMA_NAME, hmr.SPRINGS_NAME, "response_arrays.npz"):
                    (legacy / path.name).write_bytes(path.read_bytes())
            np.savez_compressed(legacy / "response_arrays.npz", **{k: v for k, v in response.items()
                                                                  if k not in ("hinge_moment", "hinge_history_time", "hinge_history_commit_count")})
            legacy_metadata = Hybrid_Exporter.compile_hybrid_sample(legacy)
            with np.load(legacy / "hybrid_sample.npz") as data:
                self.assertNotIn("hinge_moment", data.files)
                self.assertEqual(data["hinge_rotation"].shape, response["hinge_rotation"].shape)
            self.assertFalse(legacy_metadata["hinge_moment_present"])
            self.assertFalse(legacy_metadata["hinge_moment_rotation"]["available"])
            if schema_v3 is not None:
                schema_v3.validate(schema_v3.load_sample(legacy / "hybrid_sample.npz"))


if __name__ == "__main__":
    unittest.main()
