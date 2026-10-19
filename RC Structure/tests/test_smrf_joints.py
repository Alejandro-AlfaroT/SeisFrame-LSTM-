"""Independent numeric examples and missing-evidence regressions."""
import copy
import json
import sys
import unittest
from pathlib import Path

RC_DIR = Path(__file__).resolve().parents[1]
if str(RC_DIR) not in sys.path:
    sys.path.insert(0, str(RC_DIR))

from Design.SMRF_Joints import (
    MPR_BASIS, beam_capacity_shear_checks, beam_capacity_shear_envelope,
    evaluate_joints, joint_shear_check, rectangular_joint_area,
    scwb_check, through_bar_anchorage_checks,
)


def nominal_state():
    return {"nominal_strengths": True,
            "column_capacities": [
                {"mn_kip_in": 1200, "factored_axial_kip": 100, "axial_envelope_checked": True},
                {"mn_kip_in": 1200, "factored_axial_kip": 150, "axial_envelope_checked": True}],
            "beam_capacities": [
                {"mn_kip_in": 1000, "slab_mn_kip_in": 0, "slab_basis": "no_slab"},
                {"mn_kip_in": 1000, "slab_mn_kip_in": 0, "slab_basis": "no_slab"}]}


def shear_data():
    return {"clear_span_in": 100, "mpr_left_negative_kip_in": 2000,
            "mpr_right_positive_kip_in": 1000, "mpr_left_positive_kip_in": 1200,
            "mpr_right_negative_kip_in": 800, "gravity_reactions_kip": [10, 15],
            "mpr_basis": MPR_BASIS, "gravity_basis": "factored_zero_end_moment_reactions",
            "phi_vn_left_kip": 40, "phi_vn_right_kip": 35,
            "shear_capacity_requirements_checked": True}


def anchorage_data():
    return {"bars_pass_through": True, "joint_depth_in": 20,
            "concrete_type": "normalweight", "beam_depths_in": [30, 32],
            "bars": [{"grade_ksi": 60, "diameter_in": 1.0}]}


class SCWBTests(unittest.TestCase):
    def test_nominal_sum_boundary_no_phi(self):
        check = scwb_check(nominal_state())
        self.assertEqual(check["status"], "pass")
        self.assertEqual(check["demand"], 2400)
        self.assertEqual(check["capacity"], 2400)

    def test_roof_does_not_duplicate_missing_upper_column(self):
        data = nominal_state()
        data["column_capacities"].pop()
        data["is_roof"] = True
        self.assertEqual(scwb_check(data)["status"], "fail")

    def test_developed_slab_can_change_result(self):
        data = nominal_state()
        data["beam_capacities"][0].update(slab_mn_kip_in=200, slab_basis="developed_effective_width")
        check = scwb_check(data)
        self.assertEqual(check["status"], "fail")
        self.assertEqual(check["demand"], 2640)

    def test_missing_slab_not_silently_zero(self):
        data = nominal_state()
        del data["beam_capacities"][0]["slab_mn_kip_in"]
        self.assertEqual(scwb_check(data)["status"], "not_evaluated")

    def test_service_gravity_only_is_not_factored_envelope(self):
        data = nominal_state()
        data["column_capacities"][0]["axial_envelope_checked"] = False
        self.assertEqual(scwb_check(data)["status"], "not_evaluated")

    def test_phi_factored_input_not_accepted(self):
        data = nominal_state()
        del data["nominal_strengths"]
        self.assertEqual(scwb_check(data)["status"], "not_evaluated")

    def test_both_axes_and_sway_signs_are_separate(self):
        joint = {"id": "j1", "directions": {
            axis: {sign: nominal_state() for sign in ("positive", "negative")}
            for axis in ("x", "y")}}
        joint["directions"]["y"]["negative"]["column_capacities"][0]["mn_kip_in"] = 1100
        checks = [c for c in evaluate_joints({"joints": [joint]}) if c["id"] == "scwb"]
        self.assertEqual(len(checks), 4)
        self.assertEqual([c["location"] for c in checks if c["status"] == "fail"], ["j1/y/negative"])

    def test_missing_axis_never_cloned(self):
        checks = evaluate_joints({"joints": [{"id": "j1", "directions": {"x": {"positive": nominal_state()}}}]})
        scwb = [c for c in checks if c["id"] == "scwb"]
        self.assertEqual(sum(c["status"] == "not_evaluated" for c in scwb), 3)

    def test_nonfinite_capacity_is_json_safe_unknown(self):
        data = nominal_state()
        data["column_capacities"][0]["mn_kip_in"] = float("nan")
        result = scwb_check(data)
        self.assertEqual(result["status"], "not_evaluated")
        json.dumps(result, allow_nan=False)


class CapacityShearTests(unittest.TestCase):
    def test_asymmetric_gravity_and_moments_at_both_faces(self):
        # Mechanism shears 30 and 20 kip; each signed pair balances 25 kip gravity.
        result = beam_capacity_shear_envelope(shear_data())
        self.assertEqual(result["face_shears_kip"], {"positive": [40, -15], "negative": [-10, 35]})
        self.assertEqual(result["left_required_kip"], 40)
        self.assertEqual(result["right_required_kip"], 35)

    def test_capacity_check_boundary_and_failure(self):
        data = shear_data()
        checks = beam_capacity_shear_checks(data)
        self.assertEqual([c["status"] for c in checks], ["pass", "pass"])
        data["phi_vn_right_kip"] = 34.9
        self.assertEqual(beam_capacity_shear_checks(data)[1]["status"], "fail")

    def test_missing_capacity_retains_calculated_demand_without_passing(self):
        data = shear_data()
        del data["phi_vn_left_kip"]
        result = beam_capacity_shear_checks(data)[0]
        self.assertEqual(result["status"], "not_evaluated")
        self.assertEqual(result["demand"], 40)

    def test_no_silent_probable_strength_assumption(self):
        data = shear_data()
        data["mpr_basis"] = "nominal"
        self.assertEqual(beam_capacity_shear_checks(data)[0]["status"], "not_evaluated")

    def test_invalid_geometry_rejected(self):
        data = shear_data()
        data["clear_span_in"] = 0
        with self.assertRaises(ValueError):
            beam_capacity_shear_envelope(data)


class AnchorageAndJointShearTests(unittest.TestCase):
    def test_grade60_and_grade80_depth(self):
        data = anchorage_data()
        self.assertEqual(through_bar_anchorage_checks(data)[0]["status"], "pass")
        data["bars"][0]["grade_ksi"] = 80
        result = through_bar_anchorage_checks(data)
        self.assertEqual(result[0]["demand"], 26)
        self.assertEqual(result[0]["status"], "fail")
        self.assertEqual(result[1]["status"], "pass")

    def test_half_beam_depth_can_govern(self):
        data = anchorage_data()
        data["beam_depths_in"] = [48]
        result = through_bar_anchorage_checks(data)[0]
        self.assertEqual(result["demand"], 24)
        self.assertEqual(result["status"], "fail")

    def test_lightweight_grade60_and_grade80_restriction(self):
        data = anchorage_data()
        data["concrete_type"] = "lightweight"
        self.assertAlmostEqual(through_bar_anchorage_checks(data)[0]["demand"], 20 / .75)
        data["bars"][0]["grade_ksi"] = 80
        result = through_bar_anchorage_checks(data)
        self.assertEqual(result[1]["status"], "fail")

    def test_terminating_bars_need_different_anchorage_check(self):
        data = anchorage_data()
        data["bars_pass_through"] = False
        self.assertEqual(through_bar_anchorage_checks(data)[0]["status"], "not_evaluated")

    def test_eccentric_rectangular_joint_area(self):
        # 12in beam in 24in column, offset 3in -> nearest extension 3in,
        # effective width18in; joint depth30in -> 540in2.
        self.assertEqual(rectangular_joint_area(column_depth_in=30, column_width_in=24,
                                               beam_width_in=12, beam_center_offset_in=3), 540)
        with self.assertRaises(ValueError):
            rectangular_joint_area(column_depth_in=30, column_width_in=24,
                                   beam_width_in=12, beam_center_offset_in=7)

    def test_joint_strength_uses_085_phi_and_signed_equilibrium(self):
        data = {"probable_face_forces_complete": True,
                "column_shear_consistent_with_mpr": True,
                "beam_face_forces_kip": [120, 100], "column_shear_kip": 50,
                "nominal_vn_kip": 200, "capacity_basis": "ACI318-19_Table18.8.4.3",
                "capacity_topology_and_confinement_checked": True}
        result = joint_shear_check(data)
        self.assertEqual(result["demand"], 170)
        self.assertEqual(result["capacity"], 170)
        self.assertEqual(result["status"], "pass")
        data["column_shear_kip"] = -50
        self.assertEqual(joint_shear_check(data)["demand"], 270)
        self.assertEqual(joint_shear_check(data)["status"], "fail")

    def test_missing_joint_topology_is_not_evaluated(self):
        data = {"probable_face_forces_complete": True,
                "column_shear_consistent_with_mpr": True,
                "beam_face_forces_kip": [120, 100], "column_shear_kip": 50}
        result = joint_shear_check(data)
        self.assertEqual(result["demand"], 170)
        self.assertEqual(result["status"], "not_evaluated")

    def test_gravity_joint_uses_075_not_smrf_phi(self):
        data = {"rule": "chapter_15", "steel_stress_factor": 1.0, "beam_moment_basis": "nominal (fy)",
                "probable_face_forces_complete": True, "column_shear_consistent_with_mpr": True,
                "beam_face_forces_kip": [120, 100], "column_shear_kip": 70, "nominal_vn_kip": 200,
                "capacity_basis": "ACI318-19_Table15.4.2.3 (Table 18.8.4.3 row, conservative)",
                "capacity_topology_and_confinement_checked": True, "evidence_complete": True,
                "unevaluated_evidence": []}
        before = copy.deepcopy(data)
        check = joint_shear_check(data)
        self.assertEqual(data, before)
        self.assertEqual(check["status"], "pass")
        self.assertEqual(check["capacity"], 150)
        self.assertEqual(check["details"]["phi"], 0.75)
        self.assertIn("15.4.1.1(b)", check["clause"])
        # 160 kip would pass at .85; it must fail at the Chapter 15 value.
        data["column_shear_kip"] = 60
        self.assertEqual(joint_shear_check(data)["status"], "fail")
        for change in ({"rule": "18.8"}, {"rule": "unknown"}, {"rule": []},
                       {"steel_stress_factor": 1.25}, {"beam_moment_basis": None},
                       {"capacity_topology_and_confinement_checked": False},
                       {"evidence_complete": False}, {"unevaluated_evidence": ["column continuity"]},
                       {"probable_face_forces_complete": False}, {"column_shear_consistent_with_mpr": False},
                       {"nominal_vn_kip": float("nan")}, {"capacity_basis": "unsupported"}):
            with self.subTest(change=change):
                result = joint_shear_check({**before, **change})
                self.assertEqual(result["status"], "not_evaluated")
                json.dumps(result, allow_nan=False)
        missing_rule = {k: v for k, v in before.items() if k != "rule"}
        self.assertEqual(joint_shear_check(missing_rule)["status"], "not_evaluated")

    def test_missing_groups_do_not_make_empty_passing_checklist(self):
        checks = evaluate_joints({})
        self.assertEqual(len(checks), 4)
        self.assertTrue(all(c["status"] == "not_evaluated" for c in checks))

    def test_inputs_are_not_mutated(self):
        data = nominal_state()
        before = copy.deepcopy(data)
        scwb_check(data)
        self.assertEqual(data, before)


if __name__ == "__main__":
    unittest.main()
