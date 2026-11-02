import contextlib
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
from Data_Generation.Graph_Exporter import collect_global_parameters, collect_reinforcement_geometry
from Data_Generation import Hybrid_Exporter as hybrid
from Ground_Motion_Main import validate_ntha_output_compatibility

# Numeric global tensor width of the hybrid_gnn_lstm_v4 sample contract (frozen IMK V2 calibration, 2026-10-29):
# 43 in v3, less the nominal imk_beam_theta_y / imk_column_theta_y and the global imk_theta_p_pos / imk_theta_pc_pos /
# imk_theta_u_pos constants. The sidecars must still not widen it.
GLOBAL_FEATURE_COUNT = 38


class ReinforcementExportTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        settings = {"SLAB_THICKNESS_IN": 6., "B_COL": 24., "H_COL": 30.,
                    "B_BEAM": 16., "H_BEAM": 24., "BEAM_CLEAR_COVER_IN": 1.5,
                    "COL_CLEAR_COVER_IN": 1.75, "COL_BAR_SIZE": 8,
                    "BEAM_BAR_SIZE": 6, "COL_STIRRUP_BAR_SIZE": 4,
                    "BEAM_STIRRUP_BAR_SIZE": 3, "AGGREGATE_MAX_SIZE_IN": .75,
                    "REINFORCEMENT_SPECIFICATION": "ASTM A706 Grade 60",
                    "MATERIAL_EXPOSURE": "sheltered_interior"}
        for name, value in settings.items():
            self.stack.enter_context(mock.patch.object(sp, name, value, create=True))

    def tearDown(self):
        self.stack.close()

    def test_new_sidecar_matches_actual_cover_and_core_geometry(self):
        sidecar = collect_global_parameters()["reinforcement_geometry"]
        self.assertEqual(sidecar["schema_version"], "smrf_reinforcement_geometry_v1")
        # bars on the 0.5 in grid, laid out for the #5 hoop of the ladders (2026-10-14): the #8 column bar at
        # 1.75 + 0.625 + 0.5 = 2.875 -> 3.0 in, the #6 beam bar at 1.5 + 0.625 + 0.375 = 2.5 in; the hoops follow
        # the bars, so the core is the cover as built (3.0 - 0.5 - 0.5 = 2.0 in; 2.5 - 0.375 - 0.375 = 1.75 in)
        self.assertEqual(sidecar["column"]["longitudinal_centroid_offset_in"], 3.0)
        self.assertEqual(sidecar["beam"]["longitudinal_centroid_offset_in"], 2.5)
        self.assertEqual(sidecar["column"]["fiber_core_width_in"], 24 - 2 * 2.0)
        self.assertEqual(sidecar["column"]["fiber_core_depth_in"], 30 - 2 * 2.0)
        self.assertEqual(sidecar["column"]["hoop_centerline_width_in"], 24 - 2 * 2.0 - .5)
        self.assertEqual(sidecar["beam"]["fiber_core_area_in2"], (16 - 3.5) * (24 - 3.5))
        self.assertEqual(sidecar["aggregate_max_size_in"], .75)
        self.assertEqual(sidecar["materials"]["reinforcement_specification"], "ASTM A706 Grade 60")
        self.assertEqual(sidecar["materials"]["exposure"], "sheltered_interior")
        self.assertEqual(sidecar["materials"]["es_ksi"], sp.ES_KSI)
        self.assertEqual(json.loads(json.dumps(sidecar)), sidecar)

    def test_numeric_global_tensor_stays_43_values(self):
        # The name records the v3 width; the v4 contract is GLOBAL_FEATURE_COUNT (see above).
        params = collect_global_parameters()
        with_sidecar = hybrid._json_to_feature_array(params, hybrid.GLOBAL_FEATURE_KEYS)
        params.pop("reinforcement_geometry")
        without_sidecar = hybrid._json_to_feature_array(params, hybrid.GLOBAL_FEATURE_KEYS)
        self.assertEqual(len(hybrid.GLOBAL_FEATURE_KEYS), GLOBAL_FEATURE_COUNT)
        self.assertEqual(with_sidecar.shape, (GLOBAL_FEATURE_COUNT,))
        self.assertTrue((with_sidecar == without_sidecar).all())

    def test_legacy_has_no_invented_cover_or_material_provenance(self):
        sp.SLAB_THICKNESS_IN = None
        self.assertIsNone(collect_reinforcement_geometry())
        params = collect_global_parameters()
        params.pop("reinforcement_geometry")
        params.pop("floor_loads")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "global_parameters.json").write_text(json.dumps(params), encoding="utf-8")
            validate_ntha_output_compatibility(path)

    def test_output_identity_detects_cover_aggregate_and_material_changes(self):
        params = collect_global_parameters()
        for field, value in (("BEAM_CLEAR_COVER_IN", 2.),
                             ("AGGREGATE_MAX_SIZE_IN", 1.),
                             ("REINFORCEMENT_SPECIFICATION", "different specification"),
                             ("MATERIAL_EXPOSURE", "different exposure")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "global_parameters.json").write_text(json.dumps(params), encoding="utf-8")
                validate_ntha_output_compatibility(path)
                with mock.patch.object(sp, field, value):
                    with self.assertRaisesRegex(RuntimeError, "reinforcement_geometry"):
                        validate_ntha_output_compatibility(path)

    def test_new_output_missing_provenance_is_not_accepted_as_compatible(self):
        params = collect_global_parameters()
        params.pop("reinforcement_geometry")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "global_parameters.json").write_text(json.dumps(params), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "reinforcement_geometry"):
                validate_ntha_output_compatibility(path)

    def test_invalid_new_geometry_is_rejected(self):
        sp.B_BEAM = 4.
        with self.assertRaisesRegex(ValueError, "bar centroids"):
            collect_reinforcement_geometry()

    def test_compiler_copies_saved_sidecar_not_live_model(self):
        source_parameters = collect_global_parameters()
        saved_sidecar = copy.deepcopy(source_parameters["reinforcement_geometry"])
        sp.BEAM_CLEAR_COVER_IN = 2.
        # Minimal in-memory exporter fixtures; no structural simulation.
        csv_data = {
            "nodes.csv": [{"node_tag": "1"}],
            "edges.csv": [{"source": "1", "target": "1", "ele_tag": "1"}],
            "elements.csv": [{"ele_tag": "1", "element_type": "column"}],
            "time_history.csv": [{"time_sec": "0.01"}],
            "story_drift_peaks.csv": [{"story": "1"}],
            "node_envelope.csv": [{"node_tag": "1"}],
            "element_end_force_envelope.csv": [{"ele_tag": "1", "node_tag": "1", "end": "i"}],
            # The compiler reads only a hinge_backbone v2 table (frozen IMK V2 calibration, 2026-10-29): one unreleased
            # end (end_released 0, an installed spring) with zero demand, of the calibration the saved global
            # parameters declare.
            "hinge_backbone.csv": [{"hinge_backbone_version": hybrid.HINGE_BACKBONE_VERSION,
                                    "hinge_ele_tag": str(sp.IMK_HINGE_ELEMENT_TAG_BASE + 11), "ele_tag": "1",
                                    "end_id": "1", "member_type": "column", "material_type": "IMKPeakOriented",
                                    "member_spring_calibration": source_parameters["imk_member_spring_calibration"],
                                    "end_released": "0", "capping_ratio": "0.0", "damage_ratio": "0.0",
                                    "yielded": "0", "past_capping": "0", "past_native_cutoff": "0"}],
        }
        json_data = {"global_parameters.json": source_parameters,
                     "status.json": {"failed": False, "completed_steps": 1, "npts_requested": 1}}
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(hybrid, "required_files_present", return_value=[]), \
                mock.patch.object(hybrid, "_read_csv", side_effect=lambda p: csv_data.get(Path(p).name, [])), \
                mock.patch.object(hybrid, "_read_json", side_effect=lambda p, default=None: json_data.get(Path(p).name, default)), \
                mock.patch.object(hybrid.np, "load", return_value={}), \
                mock.patch.object(hybrid.np, "savez_compressed") as save:
            (Path(directory) / "status.json").write_text(json.dumps(json_data["status.json"]))
            metadata = hybrid.compile_hybrid_sample(directory)
            self.assertEqual(metadata["reinforcement_geometry"], saved_sidecar)
            self.assertEqual(save.call_args.kwargs["global_features"].shape, (GLOBAL_FEATURE_COUNT,))
            self.assertNotIn("reinforcement_geometry", save.call_args.kwargs)

    def test_existing_tensor_reuse_does_not_invent_or_replace_provenance(self):
        # A kept tensor is reused only when its metadata names the current sample contract (hybrid_gnn_lstm_v4, frozen
        # IMK V2 calibration, 2026-10-29); one without it is refused, never returned as current, and left untouched.
        for existing in ({}, {"reinforcement_geometry": {"recorded": "original"}}):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "status.json").write_text("{}")
                (path / "hybrid_sample.npz").write_bytes(b"existing tensor bytes")
                (path / "hybrid_metadata.json").write_text(
                    json.dumps({"schema_version": hybrid.SCHEMA_VERSION, **existing}), encoding="utf-8")
                (path / "global_parameters.json").write_text(json.dumps(collect_global_parameters()), encoding="utf-8")
                metadata = hybrid.compile_hybrid_sample(path)
                self.assertEqual(metadata["reinforcement_geometry"], existing.get("reinforcement_geometry"))
                self.assertEqual((path / "hybrid_sample.npz").read_bytes(), b"existing tensor bytes")
                (path / "hybrid_metadata.json").write_text(json.dumps(existing), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "is schema None"):
                    hybrid.compile_hybrid_sample(path)
                self.assertEqual((path / "hybrid_sample.npz").read_bytes(), b"existing tensor bytes")
                self.assertEqual(json.loads((path / "hybrid_metadata.json").read_text(encoding="utf-8")), existing)


if __name__ == "__main__":
    unittest.main()
