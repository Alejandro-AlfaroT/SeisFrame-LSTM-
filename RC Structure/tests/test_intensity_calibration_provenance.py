"""Intensity calibration: the drift-model fit and its inversion, the pilot reader's exclusions, the
plan's target drift, and the model identity that ties an artifact to the design route and hinge model
it was fitted on (added 2026-09-26 after the 2026-09-08 artifacts were found to come from another
design route, another hinge material and another story-height range).
"""
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RC_DIR = os.path.dirname(HERE)
sys.path.insert(0, RC_DIR)
sys.path.insert(0, os.path.join(RC_DIR, "Data_Generation"))

import Calibrate_Intensity as ci  # noqa: E402
import Structure_Parameters as sp  # noqa: E402
from Analysis.Response_Spectrum import SPECTRUM_PERIODS_SEC  # noqa: E402
from Generate_Parameterized_Dataset import build_runs  # noqa: E402

COEFFICIENTS = {"a": -3.5, "b": 0.9, "c": 1.2}
SA_LEVEL = {"r0": 0.3, "r1": 0.6, "r2": 0.9}          # flat spectra, g


def flat_spectra(levels=SA_LEVEL):
    return {record: np.full(len(SPECTRUM_PERIODS_SEC), level) for record, level in levels.items()}


def drift_model(coefficients, sa, period):
    return math.exp(coefficients["a"] + coefficients["b"] * math.log(sa) + coefficients["c"] * math.log(period))


def synthetic_observations(count=12, coefficients=COEFFICIENTS, model=None):
    observations = []
    for i in range(count):
        period = 0.5 + 0.1 * i
        scale = 0.5 + 0.25 * (i % 4)
        record = f"r{i % 3}"
        observations.append({
            "case_id": f"case_{i:04d}", "run_name": "peer_1", "period_sec": period,
            "record_x": record, "record_y": record, "scale_factor": scale,
            "peak_drift_ratio": drift_model(coefficients, scale * SA_LEVEL[record], period),
            "num_floor": 4 + i % 3, "story_h_in": 144.0 + 12.0 * (i % 2),
            "model": dict(model or {}),
        })
    return observations


class DriftModelFit(unittest.TestCase):
    def test_fit_recovers_exact_coefficients_from_noise_free_pilot(self):
        coefficients, report = ci.fit_drift_model(synthetic_observations(), flat_spectra())
        for key in ("a", "b", "c"):
            self.assertAlmostEqual(coefficients[key], COEFFICIENTS[key], places=6)
        self.assertTrue(report["fitted"])
        self.assertEqual(report["sample_count"], 12)
        self.assertAlmostEqual(report["r_squared"], 1.0, places=9)

    def test_scale_factor_inverts_the_drift_model(self):
        period, unscaled, target = 0.8, 0.5, 0.015
        factor = ci.scale_factor_for(target, period, unscaled, COEFFICIENTS)
        self.assertGreater(factor, ci.SCALE_FACTOR_MIN)
        self.assertLess(factor, ci.SCALE_FACTOR_MAX)
        self.assertAlmostEqual(drift_model(COEFFICIENTS, factor * unscaled, period), target, places=9)
        self.assertEqual(ci.scale_factor_for(0.5, period, unscaled, COEFFICIENTS), ci.SCALE_FACTOR_MAX)
        self.assertEqual(ci.scale_factor_for(1e-6, period, unscaled, COEFFICIENTS), ci.SCALE_FACTOR_MIN)

    def test_small_pilot_is_not_fitted(self):
        coefficients, report = ci.fit_drift_model(synthetic_observations(5), flat_spectra())
        self.assertFalse(report["fitted"])
        self.assertEqual(coefficients, ci.DEFAULT_COEFFICIENTS)


def write_run(root, case_id, run_name, drift, failed=False, period=0.8, record="r1", scale=1.5, globals_=None):
    run_dir = Path(root) / "cases" / case_id / "ntha" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": {"failed": failed},
        "max_story_drift_resultant": {"peak_drift_resultant_ratio": drift},
        "elastic_reference_period_mode_1_sec": period,
        "record_summary_x": {"record_id": record, "scale_factor": scale},
        "record_summary_y": {"record_id": record, "scale_factor": scale},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "global_parameters.json").write_text(json.dumps(globals_ or {}), encoding="utf-8")
    return run_dir


class PilotReader(unittest.TestCase):
    def test_exclusions_and_model_identity_are_read_from_the_run(self):
        current = ci.current_model_identity()
        globals_ = {key: value for key, value in current.items() if key != "design_schema_version"}
        globals_.update({"num_floor": 5, "story_h_in": 156.0})
        with tempfile.TemporaryDirectory() as temp_dir:
            write_run(temp_dir, "case_0001", "kept", 0.02, globals_=globals_)
            write_run(temp_dir, "case_0001", "died_early", 0.005, failed=True, globals_=globals_)
            write_run(temp_dir, "case_0001", "diverged", 0.5, globals_=globals_)
            write_run(temp_dir, "case_0001", "collapsed_but_physical", 0.06, failed=True, globals_=globals_)
            (Path(temp_dir) / "cases" / "case_0001" / "design.json").write_text(
                json.dumps({"schema_version": current["design_schema_version"]}), encoding="utf-8")
            observations = ci.collect_pilot_observations([temp_dir])
        self.assertEqual(sorted(item["run_name"] for item in observations), ["collapsed_but_physical", "kept"])
        kept = next(item for item in observations if item["run_name"] == "kept")
        self.assertEqual(kept["num_floor"], 5)
        self.assertEqual(kept["story_h_in"], 156.0)
        self.assertEqual(kept["model"], current)

    def test_pilot_identity_reports_agreement_and_mixture(self):
        base = {"imk_material_type": "IMKPeakOriented", "design_schema_version": "s1"}
        observations = synthetic_observations(4, model=base)
        identity = ci.pilot_model_identity(observations)
        self.assertEqual(identity["values"]["imk_material_type"], "IMKPeakOriented")
        self.assertEqual(identity["values"]["imk_deterioration_mode"], None)
        self.assertEqual(identity["mixed_keys"], [])
        observations[0]["model"]["imk_material_type"] = "IMKBilin"
        identity = ci.pilot_model_identity(observations)
        self.assertEqual(identity["mixed_keys"], ["imk_material_type"])
        self.assertEqual(sorted(identity["values"]["imk_material_type"]), ["IMKBilin", "IMKPeakOriented"])


class ArtifactModelIdentity(unittest.TestCase):
    def build(self, temp_dir, model):
        observations = synthetic_observations(12, model=model)
        with mock.patch.object(ci, "collect_pilot_observations", return_value=observations), \
                mock.patch.object(ci, "record_spectra", return_value=flat_spectra()):
            return ci.build_calibration([temp_dir], Path(temp_dir) / "calibration.json", seed=3)

    def test_artifact_records_the_pilot_model_and_loads_only_against_it(self):
        current = ci.current_model_identity()
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = self.build(temp_dir, current)
            path = Path(temp_dir) / "calibration.json"
            recorded = payload["model_identity"]
            self.assertEqual(recorded["keys"], list(ci.MODEL_IDENTITY_KEYS))
            self.assertEqual(recorded["pilot"]["values"], current)
            self.assertEqual(recorded["current_at_fit"], current)
            self.assertEqual(recorded["num_floor_range"], [4, 6])
            self.assertEqual(recorded["story_h_in_range"], [144.0, 156.0])
            self.assertEqual(ci.calibration_model_mismatch(payload, current), {})
            self.assertEqual(ci.load_calibration(path, require_model=True)["coefficients"], payload["coefficients"])

            other = dict(current, imk_material_type="IMKBilin", imk_deterioration_mode="direct")
            mismatch = ci.calibration_model_mismatch(payload, other)
            self.assertEqual(sorted(mismatch), ["imk_deterioration_mode", "imk_material_type"])
            with self.assertRaisesRegex(ValueError, "imk_material_type"):
                ci.load_calibration(path, require_model=other)

            # An artifact from before 2026-09-26 carries no identity: it mismatches on every key.
            stripped = dict(payload)
            stripped.pop("model_identity")
            self.assertEqual(sorted(ci.calibration_model_mismatch(stripped, current)), sorted(ci.MODEL_IDENTITY_KEYS))
            path.write_text(json.dumps(stripped), encoding="utf-8")
            with self.assertRaises(ValueError):
                ci.load_calibration(path, require_model=True)
            self.assertEqual(ci.load_calibration(path)["coefficients"], payload["coefficients"],
                             "without require_model the artifact still loads, for tools and audits")

    def test_pilot_values_win_over_the_fit_time_code_state(self):
        current = ci.current_model_identity()
        pilot_model = dict(current, imk_material_type="IMKBilin")
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = self.build(temp_dir, pilot_model)
        self.assertEqual(payload["model_identity"]["pilot"]["values"]["imk_material_type"], "IMKBilin")
        self.assertEqual(payload["model_identity"]["current_at_fit"]["imk_material_type"], current["imk_material_type"])
        mismatch = ci.calibration_model_mismatch(payload, current)
        self.assertEqual(list(mismatch), ["imk_material_type"] if current["imk_material_type"] != "IMKBilin" else [])


class CurrentModelIdentity(unittest.TestCase):
    def test_reads_the_design_schema_without_importing_the_driver(self):
        identity = ci.current_model_identity()
        from Design.Design_Driver import DESIGN_SCHEMA_VERSION
        self.assertEqual(identity["design_schema_version"], DESIGN_SCHEMA_VERSION)
        self.assertEqual(identity["imk_material_type"], sp.IMK_MATERIAL_TYPE)
        self.assertEqual(identity["imk_deterioration_mode"], sp.IMK_DETERIORATION_MODE)
        self.assertEqual(identity["element_formulation"], sp.ELEMENT_FORMULATION)
        self.assertEqual(set(identity), set(ci.MODEL_IDENTITY_KEYS))


class PlanTargets(unittest.TestCase):
    def test_runs_keep_the_target_drift_they_were_scaled_for(self):
        runs = build_runs([11, 12], [1.0], scale_factors=[1.4, 0.8], targets=[0.012, 0.03])
        self.assertEqual([run["target_drift_ratio"] for run in runs], [0.012, 0.03])
        self.assertEqual([run["scale_factor"] for run in runs], [1.4, 0.8])
        ladder = build_runs([11], [1.0, 2.0])
        self.assertTrue(all("target_drift_ratio" not in run for run in ladder))


if __name__ == "__main__":
    unittest.main()
