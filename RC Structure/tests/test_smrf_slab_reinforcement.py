import copy
import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Design.SMRF_Beam_Slab_Strength import HOOK_CORNER_CONVENTION
# Building-corner hooks that keep psi_o = 1.0 and need no 25.4.3.4 ties (covers of 7 in, inside the column core).
CORNERS = {axis: {"side_cover_in": 7.0, "in_plane_cover_in": 7.0, "in_column_core": True,
                  "convention": HOOK_CORNER_CONVENTION} for axis in ("x", "y")}
from Design.SMRF_Slab_Reinforcement import (
    design_slab_reinforcement, evaluate_slab_reinforcement,
    one_way_shear_strength, rectangular_strip_strength, slab_input_signature,
)


def inputs(**changes):
    data = dict(thickness_in=8.0, fc_ksi=4.0, fy_ksi=60.0,
                max_aggregate_size_in=.75, exposure="sheltered_interior",
                steel_specification="ASTM A706", concrete_type="normalweight",
                panel_ids=["panel_x1_y1", "panel_x2_y1"], num_floor=4)
    return {**data, **changes}


def evidence(data=None, mu=80., vu=2.):
    data = inputs() if data is None else data
    return dict(verified=True, analysis_applicability_verified=True, all_floors_enveloped=True,
                load_pattern_envelope_verified=True, spatial_envelope_per_unit_width=True,
                twisting_moment_resolution_verified=True, zero_membrane_force_verified=True,
                method="plate_finite_element", source="Unit-test fixture: analytical actions, not a real design",
                load_combination_basis="1.4D and 1.2D+1.6L, all live-load patterns",
                analysis_model_sha256="a" * 64, slab_input_sha256=slab_input_signature(data),
                shear_envelope_basis="support_face_maximum",
                load_scope="uniform_area_gravity_no_concentrated_loads",
                strips=[dict(panel_id=panel, axis=axis, face=face, mu_kip_in_per_ft=mu,
                             vu_kip_per_ft=vu, demand_location="Source section 1, factored envelope",
                             tension_face_at_shear_verified=True)
                        for panel in data["panel_ids"] for axis in ("x", "y") for face in ("top", "bottom")])


class SlabReinforcementTests(unittest.TestCase):
    def test_flexural_strength_matches_hand_calculation(self):
        # A_s=.4 in2/ft, fc=4 ksi, d=7 in, fy=60 ksi.
        # a=As*fy/(.85*fc*b)=24/40.8; Mn=24*(7-a/2).
        result = rectangular_strip_strength(4, 60, 7, .4)
        a = 24. / 40.8
        self.assertAlmostEqual(result["stress_block_in"], a)
        self.assertAlmostEqual(result["neutral_axis_in"], a / .85)
        self.assertAlmostEqual(result["mn_kip_in_per_ft"], 24 * (7 - a/2))
        self.assertAlmostEqual(result["phi_mn_kip_in_per_ft"], .9 * 24 * (7-a/2))
        self.assertTrue(result["tension_controlled"])
        self.assertAlmostEqual(result["minimum_tension_controlled_strain"], 60/29000 + .003)

    def test_overreinforced_section_does_not_assume_yield_or_tension_control(self):
        result = rectangular_strip_strength(4, 60, 3, 10)
        self.assertFalse(result["tension_controlled"])
        self.assertLess(result["steel_stress_ksi"], 60)
        self.assertLess(result["phi"], .9)

    def test_beta1_changes_with_concrete_strength(self):
        self.assertEqual(rectangular_strip_strength(4, 60, 7, .4)["beta1"], .85)
        self.assertAlmostEqual(rectangular_strip_strength(6, 60, 7, .4)["beta1"], .75)
        self.assertEqual(rectangular_strip_strength(8, 60, 7, .4)["beta1"], .65)

    def test_shear_size_and_reinforcement_effect_matches_hand_calculation(self):
        result = one_way_shear_strength(4, 20, .8)
        rho = .8/(12*20)
        size = math.sqrt(2/3)
        nominal = 8 * size * rho**(1/3) * math.sqrt(4000) * 12 * 20 / 1000
        self.assertAlmostEqual(result["lambda_s"], size)
        self.assertAlmostEqual(result["rho_longitudinal"], rho)
        self.assertAlmostEqual(result["vc_kip_per_ft"], nominal)
        self.assertAlmostEqual(result["phi_vc_kip_per_ft"], .75*nominal)
        # No obsolete lower bound of 2sqrt(fc)*b*d.
        self.assertLess(result["vc_kip_per_ft"], 2*math.sqrt(4000)*12*20/1000)
        self.assertGreater(one_way_shear_strength(4, 20, 1.6)["vc_kip_per_ft"], result["vc_kip_per_ft"])
        self.assertEqual(one_way_shear_strength(4, 7, .4)["lambda_s"], 1)

    def test_shear_upper_bound_is_applied(self):
        result = one_way_shear_strength(4, 7, 100)
        self.assertAlmostEqual(result["vc_psi"], 5*math.sqrt(4000))

    def test_valid_fixture_sizes_four_layers_and_never_accepts_whole_slab(self):
        result = design_slab_reinforcement(inputs(), evidence())
        self.assertTrue(result["screen_passed"])
        self.assertFalse(result["accepted"])
        self.assertFalse(result["summary"]["accepted"])
        self.assertEqual(set(result["layout"]["layers"]), {"x_top", "x_bottom", "y_top", "y_bottom"})
        self.assertEqual(result["summary"]["counts"]["fail"], 0)
        # Without floor context the completion checks stay open, plus the two
        # provisions that are open by scope.
        self.assertEqual(result["summary"]["counts"]["not_evaluated"], 3)
        self.assertEqual(len({layer["bar_size"] for layer in result["layout"]["layers"].values()}), 1)
        self.assertEqual(result["layout"]["layers"]["x_top"]["clear_cover_outer_mat_in"], .75)
        self.assertIn("slab_completion_context", result["summary"]["not_evaluated"])
        self.assertIn("slab_fire_resistance", result["summary"]["not_evaluated"])

    def test_floor_context_settles_development_shear_path_and_corner_checks(self):
        context = {"clear_span_x_in": 102.0, "clear_span_y_in": 102.0, "beam_width_in": 10.0,
                   "alpha_f_min": 5.6, "alpha_f_l2_l1_min": 5.6, "thickness_screen_passed": True,
                   "column_core_width_in": 13.0,
                   "two_way_shear_path_assessed": True, "slab_column_local_steel_assessed": False,
                   "columns_at_beam_intersections": True,
                   "beam_clear_cover_in": 1.5, "beam_hoop_diameter_in": 0.5, "fc_beam_ksi": 4.0,
                   "perimeter_hook_corners": CORNERS}
        result = design_slab_reinforcement(inputs(), evidence(), context=context)
        self.assertTrue(result["screen_passed"])
        ids = {c["id"]: c for c in result["checks"] if c["location"] in ("", "x_top", "x")}
        for check_id in ("slab_bar_development", "slab_perimeter_bar_anchorage", "slab_continuity_and_extensions",
                         "slab_crack_control_spacing", "slab_two_way_shear_applicability", "slab_deflection_control",
                         "slab_integrity_bottom_bars_through_column", "slab_corner_reinforcement"):
            self.assertEqual(ids[check_id]["status"], "pass", check_id)
        self.assertEqual(set(result["summary"]["not_evaluated"]), {"slab_column_local_minimum_steel", "slab_fire_resistance"})
        self.assertEqual(result["inputs"]["context"]["alpha_f_min"], 5.6)
        # Saved evidence audit reproduces the context-bearing record exactly.
        self.assertEqual(evaluate_slab_reinforcement(result)[0]["status"], "pass")
        # A weak-beam floor leaves the two-way shear path open instead of asserting it.
        weak = design_slab_reinforcement(inputs(), evidence(), context={**context, "alpha_f_min": 0.5})
        self.assertIn("slab_two_way_shear_applicability", weak["summary"]["not_evaluated"])
        # So does a floor whose beams do not take the whole panel shear: the Table 8.10.8.1
        # criterion is alpha_f1 l2/l1, not alpha_f alone (a stiff beam on a long, narrow panel).
        partial = design_slab_reinforcement(inputs(), evidence(), context={**context, "alpha_f_l2_l1_min": 0.8})
        self.assertIn("slab_two_way_shear_applicability", partial["summary"]["not_evaluated"])
        self.assertIn("alpha_f1 l2/l1 < 1.0", next(c for c in partial["checks"] if c["id"] == "slab_two_way_shear_applicability")["details"]["reason"])
        # So does a stiff-beam floor whose shear path the engineer has not assessed.
        unassessed = design_slab_reinforcement(inputs(), evidence(), context={**context, "two_way_shear_path_assessed": False})
        # And a column bearing on the slab directly is a slab-column connection: not covered, whatever is asserted.
        direct = design_slab_reinforcement(inputs(), evidence(), context={**context, "columns_at_beam_intersections": False})
        self.assertIn("slab_two_way_shear_applicability", direct["summary"]["not_evaluated"])
        applicability = next(c for c in result["checks"] if c["id"] == "slab_two_way_shear_applicability")
        # 318-19: 8.4.3 is one-way shear, 8.4.4 two-way (8.4.4.1.1 at columns, loads and reaction areas);
        # the beam-share criterion is 318-14 Table 8.10.8.1, which the 2019 body text does not carry.
        self.assertIn("8.4.3.1", applicability["clause"])
        self.assertIn("8.4.4.1.1", applicability["clause"])
        self.assertNotIn("8.4.4.2.1", applicability["clause"])
        self.assertIn("ACI 318-14 Table 8.10.8.1", applicability["clause"])
        self.assertNotIn("318-19 8.10.8", applicability["clause"])
        self.assertEqual(applicability["units"], "alpha_f1 l2/l1")
        self.assertEqual(applicability["details"]["beam_share_of_panel_shear"], "100% (Table 8.10.8.1 at alpha_f1 l2/l1 >= 1.0)")
        self.assertIn("slab_two_way_shear_applicability", unassessed["summary"]["not_evaluated"])
        self.assertTrue(unassessed["screen_passed"])
        malformed = design_slab_reinforcement(inputs(), evidence(), context={"clear_span_x_in": 1})
        self.assertIsNone(malformed["layout"])
        self.assertIn("context must contain", malformed["checks"][0]["details"]["reason"])

    def test_crossing_layers_have_different_effective_depths(self):
        result = design_slab_reinforcement(inputs(), evidence())
        layers = result["layout"]["layers"]
        diameter = layers["x_top"]["bar_diameter_in"]
        self.assertAlmostEqual(layers["x_top"]["effective_depth_in"] - layers["y_top"]["effective_depth_in"], diameter)
        flipped = design_slab_reinforcement(inputs(), evidence(), {"outer_axis": "y"})
        layers = flipped["layout"]["layers"]
        self.assertGreater(layers["y_top"]["effective_depth_in"], layers["x_top"]["effective_depth_in"])

    def test_governing_panel_and_face_control_uniform_layout(self):
        demand = evidence(mu=20)
        for row in demand["strips"]:
            if row["panel_id"] == "panel_x2_y1" and row["axis"] == "x" and row["face"] == "top":
                row["mu_kip_in_per_ft"] = 180
        result = design_slab_reinforcement(inputs(), demand)
        self.assertTrue(result["screen_passed"])
        self.assertEqual(result["layout"]["layers"]["x_top"]["demand_envelope"]["mu_kip_in_per_ft"], 180)
        layers = result["layout"]["layers"]
        self.assertGreater(layers["x_top"]["area_in2_per_ft"], layers["x_bottom"]["area_in2_per_ft"])
        flexure = [c for c in result["checks"] if c["id"] == "slab_strip_flexure"]
        self.assertEqual(len(flexure), 8)

    def test_zero_demands_still_require_minimum_steel_both_faces(self):
        result = design_slab_reinforcement(inputs(), evidence(mu=0, vu=0))
        self.assertTrue(result["screen_passed"])
        for layer in result["layout"]["layers"].values():
            self.assertGreaterEqual(layer["area_in2_per_ft"], .0018*12*8)
            self.assertLessEqual(layer["spacing_in"], min(2*8, 18))

    def test_failures_do_not_return_a_fallback_layout(self):
        for data, demand, policy in (
            (inputs(), evidence(mu=1e6), None),
            (inputs(), evidence(vu=1e6), None),
            (inputs(thickness_in=2), evidence(inputs(thickness_in=2)), None),
            (inputs(), evidence(), {"spacing_options_in": [40]}),
            (inputs(), evidence(), {"spacing_options_in": [.6]}),
        ):
            with self.subTest(data=data, policy=policy):
                result = design_slab_reinforcement(data, demand, policy)
                self.assertFalse(result["screen_passed"])
                self.assertIsNone(result["layout"])
                self.assertEqual(result["summary"]["counts"]["fail"], 1)

    def test_missing_or_unverified_demands_do_not_select_bars(self):
        values = [None, {}, evidence()]
        values[-1]["verified"] = 1  # Equality to True is insufficient.
        for demand in values:
            result = design_slab_reinforcement(inputs(), demand)
            self.assertFalse(result["screen_passed"])
            self.assertIsNone(result["layout"])
            self.assertTrue(all(c["status"] == "not_evaluated" for c in result["checks"]))

    def test_analysis_provenance_and_all_scope_flags_are_required(self):
        fields = ("verified", "analysis_applicability_verified", "all_floors_enveloped",
                  "load_pattern_envelope_verified", "spatial_envelope_per_unit_width",
                  "twisting_moment_resolution_verified", "zero_membrane_force_verified",
                  "source", "method", "load_combination_basis", "analysis_model_sha256")
        for field in fields:
            with self.subTest(field=field):
                demand = evidence()
                del demand[field]
                result = design_slab_reinforcement(inputs(), demand)
                self.assertFalse(result["screen_passed"])
                self.assertEqual(result["checks"][0]["status"], "not_evaluated")

    def test_unverified_attempt_keeps_inputs_and_its_actual_rejection_reason(self):
        data, demand = inputs(), evidence()
        demand["verified"] = False
        result = design_slab_reinforcement(data, demand)
        self.assertIsNone(result["layout"])
        self.assertFalse(result["accepted"])
        self.assertFalse(result["screen_passed"])
        initial_reason = result["checks"][0]["details"]["reason"]
        self.assertIn("verification", initial_reason.lower())
        # JSON round trip and caller mutations must preserve the rejected attempt.
        data["thickness_in"] = 99.
        demand["verified"] = True
        result = json.loads(json.dumps(result, allow_nan=False))
        self.assertEqual(result["inputs"]["slab"]["thickness_in"], 8.)
        self.assertIs(result["inputs"]["demand_evidence"]["verified"], False)
        audit = evaluate_slab_reinforcement(result)
        self.assertEqual(audit[0]["id"], "slab_strip_saved_evidence")
        self.assertEqual(audit[0]["status"], "pass")
        self.assertEqual(audit[1]["details"]["reason"], initial_reason)
        self.assertEqual(audit[1]["status"], "not_evaluated")
        result["checks"][0]["details"]["reason"] = "fabricated reason"
        self.assertEqual(evaluate_slab_reinforcement(result)[0]["status"], "fail")

    def test_rigid_diaphragm_or_depth_dependent_shear_is_rejected(self):
        for key, value in (("method", "rigid_diaphragm_frame"),
                           ("shear_envelope_basis", "d_from_support"),
                           ("load_scope", "concentrated_load"),
                           ("analysis_model_sha256", "unknown")):
            demand = evidence()
            demand[key] = value
            self.assertFalse(design_slab_reinforcement(inputs(), demand)["screen_passed"])

    def test_panel_axis_face_inventory_must_be_exact(self):
        for modification in ("missing", "duplicate", "wrong_panel", "unhashable", "wrong_axis", "missing_tension_face"):
            with self.subTest(modification=modification):
                demand = evidence()
                if modification == "missing":
                    demand["strips"].pop()
                elif modification == "duplicate":
                    demand["strips"].append(copy.deepcopy(demand["strips"][0]))
                elif modification == "wrong_panel":
                    demand["strips"][0]["panel_id"] = "unmodeled_panel"
                elif modification == "unhashable":
                    demand["strips"][0]["panel_id"] = []
                elif modification == "wrong_axis":
                    demand["strips"][0]["axis"] = "z"
                else:
                    demand["strips"][0]["tension_face_at_shear_verified"] = False
                self.assertFalse(design_slab_reinforcement(inputs(), demand)["screen_passed"])

    def test_stale_inputs_cannot_reuse_demands(self):
        self.assertFalse(design_slab_reinforcement(inputs(thickness_in=9), evidence())["screen_passed"])
        self.assertFalse(design_slab_reinforcement(inputs(fc_ksi=5), evidence())["screen_passed"])
        self.assertFalse(design_slab_reinforcement(inputs(num_floor=5), evidence())["screen_passed"])
        reverse = inputs(panel_ids=list(reversed(inputs()["panel_ids"])))
        self.assertEqual(slab_input_signature(inputs()), slab_input_signature(reverse))

    def test_bad_numbers_or_unsupported_materials_fail_closed(self):
        for key, value in (("thickness_in", float("nan")), ("thickness_in", True),
                           ("fc_ksi", 12), ("fy_ksi", 80), ("num_floor", 1.5),
                           ("panel_ids", []), ("panel_ids", ["x", "x"]),
                           ("exposure", "exterior"), ("concrete_type", "lightweight")):
            with self.subTest(key=key, value=value):
                result = design_slab_reinforcement(inputs(**{key: value}), evidence())
                self.assertFalse(result["screen_passed"])
                self.assertEqual(result["checks"][0]["status"], "not_evaluated")
        for bad in (True, float("inf"), -1, None):
            demand = evidence()
            demand["strips"][0]["mu_kip_in_per_ft"] = bad
            self.assertFalse(design_slab_reinforcement(inputs(), demand)["screen_passed"])

    def test_bad_policy_is_not_accepted(self):
        for policy in ({"clear_cover_in": .5}, {"bar_sizes": [True]}, {"bar_sizes": [7]},
                       {"bar_sizes": []}, {"outer_axis": "z"}, {"spacing_options_in": [float("nan")]},
                       {"unsupported_option": 1}):
            self.assertFalse(design_slab_reinforcement(inputs(), evidence(), policy)["screen_passed"])

    def test_saved_evidence_is_recomputed_and_json_stable(self):
        result = design_slab_reinforcement(inputs(), evidence())
        result = json.loads(json.dumps(result, allow_nan=False))
        self.assertEqual(evaluate_slab_reinforcement(result)[0]["status"], "pass")
        result["layout"]["layers"]["x_top"]["flexure"]["phi_mn_kip_in_per_ft"] *= 100
        self.assertEqual(evaluate_slab_reinforcement(result)[0]["status"], "fail")
        self.assertFalse(any(c["status"] == "pass" for c in evaluate_slab_reinforcement(None)))

    def test_provenance_is_snapshotted_and_json_safe(self):
        demand = evidence()
        result = design_slab_reinforcement(inputs(), demand)
        demand["strips"][0]["mu_kip_in_per_ft"] = 1e6
        self.assertEqual(result["inputs"]["demand_evidence"]["strips"][0]["mu_kip_in_per_ft"], 80)
        self.assertEqual(evaluate_slab_reinforcement(result)[0]["status"], "pass")
        for value in (object(), float("nan")):
            demand = evidence()
            demand["additional_metadata"] = value
            self.assertFalse(design_slab_reinforcement(inputs(), demand)["screen_passed"])


if __name__ == "__main__":
    unittest.main()
