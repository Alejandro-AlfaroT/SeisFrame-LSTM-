"""ACI 318-19 18.12 diaphragm checks on the ASCE 7-22 12.10.1.1 forces (2026-10-14)."""
import copy
import json
import unittest

from Design.SMRF_Diaphragm import design_diaphragm, diaphragm_forces, evaluate_slab_diaphragm


def layer(axis, face, bar=4, spacing=12.0):
    diameter, area = {4: (0.5, 0.20), 5: (0.625, 0.31), 6: (0.75, 0.44)}[bar]
    return {"bar_size": bar, "bar_diameter_in": diameter, "spacing_in": spacing, "area_in2_per_ft": area * 12.0 / spacing,
            "axis": axis, "face": face}


def inputs(**overrides):
    data = {"geometry": {"num_bay_x": 3, "num_bay_y": 4, "num_floor": 5, "bay_x_in": 240.0, "bay_y_in": 216.0, "story_h_in": 156.0},
            "sections": {"b_beam_in": 14.0, "h_beam_in": 28.0, "b_col_in": 22.0, "h_col_in": 22.0},
            "slab": {"thickness_in": 6.0, "fc_ksi": 5.0, "fy_ksi": 60.0, "max_aggregate_size_in": 0.75, "normalweight": True,
                     "layers": {"x_top": layer("x", "top"), "x_bottom": layer("x", "bottom"),
                                "y_top": layer("y", "top"), "y_bottom": layer("y", "bottom")}},
            "seismic": {"sds": 1.0, "importance_factor": 1.25, "risk_category": "III", "r": 8.0, "site_label": "sdc_d_high"},
            "elf": {"design_period_sec": 0.9, "base_shear_kip": 300.0, "story_forces_kip": [20.0, 40.0, 60.0, 80.0, 100.0],
                    "story_weights_kip": [360.0] * 5, "story_heights_ft": [13.0 * k for k in range(1, 6)],
                    "vertical_distribution_k": 1.2, "cs": 300.0 / 1800.0}}
    for key, value in overrides.items():
        block, _, field = key.partition(".")
        data[block][field] = value
    return data


class Forces(unittest.TestCase):
    def test_fpx_bounds_and_governing_level(self):
        forces = diaphragm_forces(inputs())
        levels = forces["levels"]
        # top level: Fpx = Fn wn / wn = 100 kip (within 90 and 180)
        self.assertAlmostEqual(levels[-1]["fpx_kip"], 100.0)
        self.assertEqual(levels[-1]["governing"], "Eq. (12.10-1)")
        # first level: sum F / sum w * w1 = 300/1800*360 = 60 < 0.2 SDS Ie wpx = 90 -> floor
        self.assertAlmostEqual(levels[0]["fpx_eq_12_10_1_kip"], 60.0)
        self.assertAlmostEqual(levels[0]["fpx_kip"], 90.0)
        self.assertEqual(levels[0]["governing"], "minimum 0.2 SDS Ie wpx")
        self.assertEqual(forces["governing_level"], 5)
        self.assertEqual(forces["redundancy_factor"], 1.0)
        # a top-heavy force hits the ceiling 0.4 SDS Ie wpx
        heavy = inputs(**{"elf.story_forces_kip": [10.0, 10.0, 10.0, 10.0, 260.0]})
        self.assertAlmostEqual(diaphragm_forces(heavy)["levels"][-1]["fpx_kip"], 0.4 * 1.0 * 1.25 * 360.0)


class Design(unittest.TestCase):
    def test_passing_design_and_its_audit(self):
        record = design_diaphragm(inputs())
        self.assertTrue(record["screen_passed"])
        self.assertEqual(record["summary"]["counts"]["fail"], 0)
        ids = {c["id"] for c in record["checks"]}
        for required in ("diaphragm.minimum_thickness", "diaphragm.design_force_basis", "diaphragm.minimum_reinforcement",
                         "diaphragm.maximum_bar_spacing", "diaphragm.in_plane_shear", "diaphragm.chord_reinforcement",
                         "diaphragm.chord_development", "diaphragm.chord_compression", "diaphragm.collector_scope"):
            self.assertIn(required, ids)
        x = record["directions"]["x"]
        # force along x: the diaphragm spans the y dimension (4 x 216) between the exterior x-frame lines, its
        # depth is the x dimension (3 x 240); chords along y at the x edges
        self.assertEqual(x["plan_parallel_in"], 720.0)
        self.assertEqual(x["span_between_exterior_lines_in"], 864.0)
        self.assertEqual(x["chord"]["selected"]["runs_along"], "y")
        self.assertAlmostEqual(x["shear"]["vu_kip"], 50.0)
        self.assertAlmostEqual(x["shear"]["acv_in2"], 6.0 * 720.0)
        self.assertAlmostEqual(x["shear"]["rho_t"], 2 * 0.2 / (12.0 * 6.0))
        self.assertAlmostEqual(x["chord"]["mu_kip_in"], 100.0 * 864.0 / 8.0)
        self.assertAlmostEqual(x["chord"]["lever_arm_in"], 720.0 - 2.0 * (7.0 + 6.0))
        # exact recomputation after a JSON round trip (the qualification audit)
        copied = json.loads(json.dumps(record))
        self.assertEqual(design_diaphragm(copied["inputs"]), copied)

    def test_thin_slab_with_sparse_mats_fails_minimum_reinforcement_and_spacing(self):
        sparse = inputs()
        for name in sparse["slab"]["layers"]:
            sparse["slab"]["layers"][name] = layer(name.split("_")[0], name.split("_")[1], bar=4, spacing=20.0)
        sparse["slab"]["thickness_in"] = 14.0
        record = design_diaphragm(sparse)
        failed = {(c["id"], c["location"]) for c in record["checks"] if c["status"] == "fail"}
        self.assertIn(("diaphragm.maximum_bar_spacing", "x_top"), failed)
        self.assertIn(("diaphragm.minimum_reinforcement", "mats along x"), failed)
        self.assertFalse(record["screen_passed"])

    def test_chord_bars_step_up_in_size_when_the_strip_is_full(self):
        heavy = inputs(**{"elf.story_forces_kip": [10.0, 10.0, 10.0, 10.0, 2960.0], "elf.base_shear_kip": 3000.0,
                          "seismic.sds": 2.5, "slab.thickness_in": 5.0})
        record = design_diaphragm(heavy)
        selected = record["directions"]["x"]["chord"]["selected"]
        self.assertIsNotNone(selected)
        self.assertGreater(selected["bar_size"], 4)
        self.assertGreaterEqual(selected["pitch_in"], selected["pitch_min_in"] - 1e-9)

    def test_invalid_inputs_are_not_evaluated_not_passed(self):
        bad = inputs()
        bad["elf"]["base_shear_kip"] = 1.0
        record = design_diaphragm(bad)
        self.assertEqual([c["status"] for c in record["checks"]], ["not_evaluated"])
        self.assertFalse(record["screen_passed"])
        self.assertEqual([c["status"] for c in design_diaphragm(None)["checks"]], ["not_evaluated"])


class Qualification(unittest.TestCase):
    def record(self):
        data = inputs()
        diaphragm = design_diaphragm(data)
        layers = {name: {"bar_size": lay["bar_size"], "spacing_in": lay["spacing_in"], "area_in2_per_ft": lay["area_in2_per_ft"]}
                  for name, lay in data["slab"]["layers"].items()}
        return {"geometry": data["geometry"], "sections": data["sections"], "slab": {"thickness_in": 6.0},
                "slab_reinforcement": {"layout": {"layers": layers}},
                "seismic": {"sds": 1.0, "importance_factor": 1.25}, "demand": {"base_shear_kip": 300.0},
                "floor_loads": {"total_floor_seismic_weight_kip": 360.0, "roof_column_extension_seismic_weight_kip": 0.0},
                "slab_diaphragm": diaphragm}

    def test_saved_evidence_and_consistency(self):
        checks = evaluate_slab_diaphragm(self.record())
        by_id = {c["id"]: c for c in checks if c["id"] in ("diaphragm.saved_evidence", "diaphragm.frame_input_consistency")}
        self.assertEqual(by_id["diaphragm.saved_evidence"]["status"], "pass")
        self.assertEqual(by_id["diaphragm.frame_input_consistency"]["status"], "pass")
        self.assertTrue(all(c["status"] == "pass" for c in checks))

    def test_tampered_evidence_and_foreign_inputs_fail(self):
        record = self.record()
        record["slab_diaphragm"]["directions"]["x"]["shear"]["phi_vn_kip"] *= 2.0
        checks = {c["id"]: c for c in evaluate_slab_diaphragm(record)}
        self.assertEqual(checks["diaphragm.saved_evidence"]["status"], "fail")
        record = self.record()
        record["demand"]["base_shear_kip"] = 250.0
        record["slab"]["thickness_in"] = 7.0
        checks = {c["id"]: c for c in evaluate_slab_diaphragm(record)}
        self.assertEqual(checks["diaphragm.frame_input_consistency"]["status"], "fail")
        self.assertIn("slab.thickness_in", checks["diaphragm.frame_input_consistency"]["details"]["disagreements"])
        self.assertIn("elf.base_shear_kip vs demand.base_shear_kip", checks["diaphragm.frame_input_consistency"]["details"]["disagreements"])

    def test_records_without_a_diaphragm_stay_open(self):
        record = self.record()
        del record["slab_diaphragm"]
        checks = evaluate_slab_diaphragm(record)
        self.assertEqual([c["id"] for c in checks], ["diaphragm.saved_evidence"])
        self.assertEqual(checks[0]["status"], "not_evaluated")
        self.assertEqual(evaluate_slab_diaphragm({"slab": None})[0]["id"], "diaphragm.scope")


if __name__ == "__main__":
    unittest.main()
