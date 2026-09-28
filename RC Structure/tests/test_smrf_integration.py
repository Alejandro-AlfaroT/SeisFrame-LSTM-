import contextlib
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
from Design import Design_Driver as driver
from Design.Config import DesignConfig
from Design.SMRF_Qualification import (qualify_design, require_accepted_design,
                                        ensure_generation_release_ready)
from Design.SMRF_Elastic import physical_members, build_design_model
from RC_Design_Check import get_element_tags


def artifact():
    from Design.SMRF_Slab import choose_slab
    saved = {"schema_version": driver.DESIGN_SCHEMA_VERSION,
            "geometry": driver._slab_geometry(),
            "sections": {"b_col_in": sp.B_COL, "h_col_in": sp.H_COL, "fc_col_ksi": sp.FC_COL_KSI,
                         "b_beam_in": sp.B_BEAM, "h_beam_in": sp.H_BEAM, "fc_beam_ksi": sp.FC_BEAM_KSI},
            "reinforcement": {"col_bar_size": sp.COL_BAR_SIZE, "col_top_bars": sp.COL_TOP_BARS,
                              "col_bot_bars": sp.COL_BOT_BARS, "col_side_bars": sp.COL_SIDE_BARS,
                              "beam_bar_size": sp.BEAM_BAR_SIZE, "beam_top_bars": sp.BEAM_TOP_BARS,
                              "beam_bot_bars": sp.BEAM_BOT_BARS, "beam_side_bars": sp.BEAM_SIDE_BARS,
                              "col_stirrup_bar_size": sp.COL_STIRRUP_BAR_SIZE, "col_stirrup_legs": sp.COL_STIRRUP_LEGS,
                              "col_stirrup_legs_by_direction": {"across_b_face": sp.COL_STIRRUP_LEGS,
                                                                "across_h_face": sp.COL_STIRRUP_LEGS},
                              "beam_stirrup_bar_size": sp.BEAM_STIRRUP_BAR_SIZE, "beam_stirrup_legs": sp.BEAM_STIRRUP_LEGS,
                              "col_stirrup_spacing_in": sp.COL_STIRRUP_SPACING,
                              "beam_stirrup_spacing_in": sp.BEAM_STIRRUP_SPACING,
                              "legacy_centroid_offset_in": sp.COVER,
                              "beam_clear_cover_in": sp.BEAM_CLEAR_COVER_IN,
                              "col_clear_cover_in": sp.COL_CLEAR_COVER_IN,
                              "beam_longitudinal_centroid_offset_in": sp.BEAM_CLEAR_COVER_IN + sp.rebar_diameter(sp.BEAM_STIRRUP_BAR_SIZE) + sp.rebar_diameter(sp.BEAM_BAR_SIZE)/2,
                              "col_longitudinal_centroid_offset_in": sp.COL_CLEAR_COVER_IN + sp.rebar_diameter(sp.COL_STIRRUP_BAR_SIZE) + sp.rebar_diameter(sp.COL_BAR_SIZE)/2},
            "materials": {"aggregate_size_in": sp.AGGREGATE_MAX_SIZE_IN,
                          "fy_ksi": sp.FY_KSI, "fyt_ksi": sp.FY_KSI, "es_ksi": sp.ES_KSI,
                          "normalweight": True, "reinforcement_specification": sp.REINFORCEMENT_SPECIFICATION,
                          "exposure": sp.MATERIAL_EXPOSURE},
            "dcr": {"beam": 0.2, "column": 0.4, "accepted": True}}
    saved["slab"] = choose_slab(saved["geometry"], saved["sections"], {})
    for prefix in ("beam", "col"):
        bars = saved["reinforcement"]
        bars[f"{prefix}_bar_area_in2"] = sp.rebar_area(bars[f"{prefix}_bar_size"])
        bars[f"{prefix}_bar_diameter_in"] = sp.rebar_diameter(bars[f"{prefix}_bar_size"])
        bars[f"{prefix}_stirrup_diameter_in"] = sp.rebar_diameter(bars[f"{prefix}_stirrup_bar_size"])
    return saved


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        for name in set(driver._STATE_KEYS) | {"COVER", "NUM_BAY_X", "NUM_BAY_Y", "NUM_FLOOR", "NUM_MODES", "ASCE_SDS"}:
            self.stack.enter_context(mock.patch.object(sp, name, getattr(sp, name)))

    def tearDown(self):
        self.stack.close()

    def test_soft_utilization_is_not_a_code_failure_but_missing_checks_are(self):
        result = qualify_design(artifact())
        checks = {c["id"]: c for c in result["checks"]}
        self.assertEqual(checks["candidate.beam_strength_screen"]["status"], "pass")
        self.assertFalse(result["accepted"])
        with self.assertRaisesRegex(RuntimeError, "not qualified"):
            require_accepted_design(artifact())
        with self.assertRaisesRegex(RuntimeError, "not release-ready"):
            ensure_generation_release_ready()

    def test_spacing_cache_roundtrip_and_config_sync(self):
        saved = artifact()
        saved["reinforcement"].update(col_stirrup_spacing_in=3, beam_stirrup_spacing_in=4)
        driver.apply_design(saved)
        cfg = DesignConfig()
        cfg.rebar.stirrup_spacing_col_in = 9
        driver._sync_cfg_to_sp(cfg)
        self.assertEqual(sp.COL_STIRRUP_SPACING, 3)
        self.assertEqual(cfg.rebar.stirrup_spacing_col_in, 3)
        self.assertEqual(cfg.rebar.stirrup_spacing_beam_in, 4)

    def test_incomplete_cache_rejected_before_mutation(self):
        saved = artifact()
        saved["sections"]["b_col_in"] = 50
        saved["reinforcement"].pop("col_stirrup_spacing_in")
        before = sp.B_COL
        with self.assertRaises(ValueError):
            driver.apply_design(saved)
        self.assertEqual(sp.B_COL, before)

    def test_identity_changes_with_geometry_and_not_with_selected_tie_spacing(self):
        original = driver.design_request_identity()["sha256"]
        sp.COL_STIRRUP_SPACING = 3
        self.assertEqual(driver.design_request_identity()["sha256"], original)
        sp.NUM_FLOOR += 1
        self.assertNotEqual(driver.design_request_identity()["sha256"], original)

    def test_cache_reuse_and_mismatched_geometry_preserve_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "design.json"
            with mock.patch.object(driver, "design_structure", return_value=artifact()) as design:
                record, created = driver.load_or_create_design(path, verbose=False)
                before = path.read_bytes()
                reused, recreated = driver.load_or_create_design(path, verbose=False)
            self.assertTrue(created)
            self.assertFalse(recreated)
            design.assert_called_once()
            self.assertEqual(record["request_identity"], reused["request_identity"])
            sp.NUM_FLOOR += 1
            with self.assertRaisesRegex(RuntimeError, "different inputs"):
                driver.load_or_create_design(path, verbose=False)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(list(Path(directory).glob("*.lock")))

    def test_existing_lock_prevents_double_design(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "design.json"
            lock = path.with_name(".design.json.lock")
            lock.write_text("reserved", encoding="utf-8")
            with mock.patch.object(driver, "design_structure") as design:
                with self.assertRaisesRegex(RuntimeError, "reserved"):
                    driver.load_or_create_design(path)
            design.assert_not_called()
            self.assertEqual(lock.read_text(), "reserved")

    def test_strength_pass_solves_each_combination_once_and_checks_captured_actions(self):
        cfg = DesignConfig()
        captured = {"1": {"axial_i_kip": 12., "axial_j_kip": 10.}}
        joints = {"evaluated": True, "all_pass": True, "counts": {"pass": 4, "fail": 0, "not_evaluated": 0},
                  "min_ratio_provided": 1.5, "ratio_required": 1.2, "basis": "test",
                  "steel_raised": False, "steel_exhausted": False}
        with mock.patch.object(driver, "_analyze_combination", return_value={"base_shear_kip": 100}) as analyze, \
             mock.patch.object(driver, "_governing_dcrs", return_value=(0.4, 0.7, {1: "member"})) as checks, \
             mock.patch.object(driver, "_capture_element_actions", return_value=captured), \
             mock.patch.object(driver, "redesign_steel", return_value=({"x": 1}, None, False, [])) as resize, \
             mock.patch.object(driver, "_capacity_design", return_value=None), \
             mock.patch.object(driver, "_scwb_steel_floor", return_value=joints) as floor, \
             mock.patch.object(driver, "apply_updates") as update:
            worst, _, actions, joint_scwb = driver._steel_pass(cfg, 1.0, max_steel_iter=3)
        # The elastic frame does not depend on the bars: one solve per combination.
        self.assertEqual(analyze.call_count, 18)
        # Every check ran on the captured actions, never on the live domain.
        self.assertEqual(checks.call_count, 18)
        self.assertTrue(all(call.kwargs["member_actions"] is captured for call in checks.call_args_list))
        self.assertEqual(len(resize.call_args.args[0]), 18)
        self.assertEqual(update.call_count, 1)
        self.assertEqual(floor.call_count, 1)
        self.assertEqual(worst, {"column": 0.4, "beam": 0.7})
        self.assertEqual(len(actions), 18)
        self.assertTrue(all(item["axial_reference"] == "joint_faces" for item in actions))
        self.assertIs(joint_scwb, joints)

    def test_strength_pass_iterates_until_the_installed_cage_stops_changing(self):
        cfg = DesignConfig()
        sp.COL_BAR_SIZE, sp.COL_TOP_BARS, sp.COL_BOT_BARS, sp.COL_SIDE_BARS = 8, 3, 3, 2
        joints = {"evaluated": True, "all_pass": True, "counts": {}, "min_ratio_provided": 1.3,
                  "ratio_required": 1.2, "basis": "test", "steel_raised": False, "steel_exhausted": False}
        proposals = iter([({"bar_size": 8, "n_top": 4, "n_bot": 4, "n_side": 2}, None, False, []),
                          ({"bar_size": 8, "n_top": 4, "n_bot": 4, "n_side": 2}, None, False, [])])

        def update(col_update, beam_update, cfg=None):
            if col_update:
                sp.COL_TOP_BARS = sp.COL_BOT_BARS = col_update["n_top"]

        with mock.patch.object(driver, "_analyze_combination", return_value=None), \
             mock.patch.object(driver, "_governing_dcrs", return_value=(0.4, 0.7, {1: "member"})) as checks, \
             mock.patch.object(driver, "_capture_element_actions", return_value={}), \
             mock.patch.object(driver, "redesign_steel", side_effect=lambda *_a, **_k: next(proposals)) as resize, \
             mock.patch.object(driver, "_capacity_design", return_value=None), \
             mock.patch.object(driver, "_scwb_steel_floor", return_value=joints), \
             mock.patch.object(driver, "apply_updates", side_effect=update):
            driver._steel_pass(cfg, 1.0, max_steel_iter=6)
        # Iteration 1 changes the cage (3 -> 4 bars); iteration 2 proposes the
        # same cage, nothing changes, and the pass stops without using the cap.
        self.assertEqual(resize.call_count, 2)
        self.assertEqual(checks.call_count, 18 * 2)

    def test_design_model_tag_mapping_matches_physical_model(self):
        sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR = 2, 3, 2
        columns, bx, by = get_element_tags()
        members = list(physical_members())
        self.assertEqual([m[0] for m in members if m[3] == "column"], columns)
        self.assertEqual([m[0] for m in members if m[3] == "beam_x"], bx)
        self.assertEqual([m[0] for m in members if m[3] == "beam_y"], by)

    def test_small_elastic_design_model_and_drift_adapter_smoke(self):
        # In-memory static verification only: no records, NTHA or output folder.
        sp.NUM_BAY_X = sp.NUM_BAY_Y = sp.NUM_FLOOR = 1
        original_formulation = sp.ELEMENT_FORMULATION
        try:
            build_design_model()
            self.assertEqual(len(driver.ops.getEleTags()), 8)
            result = driver._drift_screen(0.3)
            self.assertEqual(len(result["stories"]), 2)
            self.assertFalse(result["malformed_checks"])
            self.assertFalse(result["duplicate_check_ids"])
            self.assertEqual(sp.ELEMENT_FORMULATION, original_formulation)
        finally:
            driver.ops.wipe()

    def test_generation_preflight_stops_before_design_or_records(self):
        import Ground_Motion_Main as main
        args = SimpleNamespace(design_only=False, skip_design=False, catalog_summary=False)
        with mock.patch.object(main, "parse_args", return_value=args), \
             mock.patch.object(main, "apply_geometry_from_args", return_value=None), \
             mock.patch.object(main, "select_records") as select, \
             mock.patch.object(main, "load_or_create_design") as design, \
             mock.patch.object(main, "run_one") as run:
            with self.assertRaisesRegex(RuntimeError, "not release-ready"):
                main.main()
        select.assert_not_called()
        design.assert_not_called()
        run.assert_not_called()

    def test_tampered_saved_evidence_is_detected_and_not_assertable(self):
        """Reviewer reproductions: stale summaries and altered hoops must fail, whatever is asserted."""
        sp.NUM_BAY_X = sp.NUM_BAY_Y = sp.NUM_FLOOR = 1
        sp.NUM_MODES = 3
        # The toy 10-in beam meets an 18-in column with four #8 per face: one bar passes per layer, so its
        # two bars take two layers (allowed since 2026-09-27) and the layered depth leaves a one-iteration
        # search no capacity-shear section; start at the 18-in variant, where the bars pass in one layer.
        sp.B_BEAM = 18.0
        try:
            record = driver.design_structure(max_section_iter=1, max_steel_iter=1, verbose=False)
        finally:
            driver.ops.wipe()
        # As read back from disk: nested objects no longer alias each other.
        record = json.loads(json.dumps(record, allow_nan=False))
        probe = {"floor_hand_check_verified": True, "strength_model_verified": True,
                 "detailing_model_consistency_verified": True, "slab_column_local_steel_assessed": True,
                 "fire_resistance_scope_accepted": True, "congestion_and_placement_accepted": True,
                 "floor_frame_compatibility_reviewed": True,
                 "asserted_by": "probe", "assertion_date": "2026-09-13", "assertion_basis": "test"}
        # Assertions ride in the request identity (demand_basis is inside the input signature).
        record.setdefault("request_identity", {}).setdefault("policy", {})["verification"] = probe

        def statuses(rec):
            return {c["id"]: c["status"] for c in qualify_design(rec)["checks"]}

        # 1. A saved column reaction multiplied by ten: the summary is recomputed from the rows.
        tampered = copy.deepcopy(record)
        tampered["coupled_comparison"]["columns"][0]["frame_vertical_kip"] *= 10.0
        result = statuses(tampered)
        self.assertEqual(result["floor.coupled_frame_compatibility"], "fail")
        self.assertEqual(result["floor.coupled_summary_consistent"], "fail")
        # 2. Saved beam hoop spacing multiplied by 100: the capacity design no longer recomputes.
        tampered = copy.deepcopy(record)
        tampered["capacity_design"]["transverse"]["beam"]["spacing_in"] *= 100.0
        result = statuses(tampered)
        self.assertEqual(result["qualification.capacity_evidence_recomputed"], "fail")
        self.assertEqual(result["qualification.hoops_match_design"], "pass")     # judged on the recomputed hoops
        self.assertEqual(result["qualification.joint_capacity_completion"], "not_evaluated")
        # 2b. A nested saved value nobody compares directly (the reviewer's second
        #     reproduction): downstream reads only the recomputed object, so the
        #     tamper is simply never consumed and the record still qualifies the same.
        tampered = copy.deepcopy(record)
        tampered["capacity_design"]["beams"]["hoops"]["bar_size"] = 11
        result = statuses(tampered)
        self.assertEqual(result["qualification.capacity_evidence_recomputed"], "pass")
        hoop_check = next(c for c in qualify_design(tampered)["checks"] if c["id"] == "beam.hoop_layout_and_axial_applicability")
        self.assertNotEqual(hoop_check["details"]["hoops"]["bar_size"], 11)
        # 2c. Saved slab strengths altered: recomputed, compared, not consumed.
        tampered = copy.deepcopy(record)
        key = next(iter(tampered["beam_slab_strengths"]))
        tampered["beam_slab_strengths"][key]["slab_basis"] = "developed_effective_width"
        tampered["beam_slab_strengths"][key]["slab_mn_kip_in"] = 1e6
        self.assertEqual(statuses(tampered)["qualification.slab_strength_evidence_recomputed"], "fail")
        # 3. The model's hoops changed after the design: the capacity design still
        #    recomputes (it selects hoops, it does not read them) but the model no
        #    longer carries what was designed -- a failure the
        #    detailing_model_consistency assertion cannot paper over, and the saved
        #    capacity evidence is not consumed.
        tampered = copy.deepcopy(record)
        tampered["reinforcement"]["beam_stirrup_spacing_in"] *= 100.0
        result = statuses(tampered)
        self.assertEqual(result["qualification.hoops_match_design"], "fail")
        self.assertEqual(result["qualification.capacity_evidence_recomputed"], "pass")
        self.assertEqual(result["qualification.joint_capacity_completion"], "not_evaluated")
        self.assertEqual(result["qualification.detailing_model_consistency"], "pass")     # its own item, still asserted
        self.assertFalse(qualify_design(tampered)["accepted"])
        # 3b. Every asymmetric frame reaction scaled by 1.04 (the follow-up reproduction):
        #     the per-column check passes by construction, the pattern total does not.
        tampered = copy.deepcopy(record)
        for row in tampered["coupled_comparison"]["asymmetric"]["columns"]:
            row["frame_vertical_kip"] = 1.04 * row["coupled_vertical_kip"]
        result = statuses(tampered)
        self.assertEqual(result["floor.coupled_frame_compatibility_pattern"], "pass")
        self.assertEqual(result["floor.coupled_total_load_pattern"], "fail")
        # 3c. Duplicated or missing column identities are not evidence.
        tampered = copy.deepcopy(record)
        rows = tampered["coupled_comparison"]["asymmetric"]["columns"]
        rows[1] = copy.deepcopy(rows[0])
        result = statuses(tampered)
        self.assertEqual(result["floor.coupled_frame_compatibility_pattern"], "not_evaluated")
        self.assertEqual(result["floor.coupled_total_load_pattern"], "not_evaluated")
        tampered = copy.deepcopy(record)
        tampered["coupled_comparison"]["columns"].pop()
        self.assertEqual(statuses(tampered)["floor.coupled_frame_compatibility"], "not_evaluated")
        # 4. A comparison from other inputs is not evidence.
        tampered = copy.deepcopy(record)
        tampered["coupled_comparison"]["analysis_input_sha256"] = "0" * 64
        self.assertEqual(statuses(tampered)["floor.coupled_frame_compatibility"], "not_evaluated")
        # Untampered: the integrity checks pass.
        result = statuses(copy.deepcopy(record))
        self.assertEqual(result["qualification.capacity_evidence_recomputed"], "pass")
        self.assertEqual(result["qualification.hoops_match_design"], "pass")
        self.assertEqual(result["floor.coupled_summary_consistent"], "pass")

    def test_in_memory_candidate_pipeline_reports_incomplete(self):
        # Bounded design-only smoke check. No plan, case files or GM analysis.
        sp.NUM_BAY_X = sp.NUM_BAY_Y = sp.NUM_FLOOR = 1
        sp.NUM_MODES = 3
        # The toy 10-in beam meets an 18-in column with four #8 per face: one bar passes per layer, so its
        # two bars take two layers (allowed since 2026-09-27) and the layered depth leaves a one-iteration
        # search no capacity-shear section; start at the 18-in variant, where the bars pass in one layer.
        sp.B_BEAM = 18.0
        try:
            result = driver.design_structure(max_section_iter=1, max_steel_iter=1, verbose=False)
            self.assertEqual(result["schema_version"], driver.DESIGN_SCHEMA_VERSION)
            self.assertEqual(result["iterations"], 1)
            self.assertFalse(result["dcr"]["accepted"])
            self.assertGreater(result["qualification"]["counts"]["not_evaluated"], 0)
            self.assertEqual(len(result["drift_screen"]["stories"]), 2)
            self.assertIn("col_stirrup_spacing_in", result["reinforcement"])
            self.assertTrue(result["slab"]["thickness_screen_passed"])
            self.assertEqual(result["slab"]["thickness_in"], sp.SLAB_THICKNESS_IN)
            self.assertEqual(result["floor_loads"]["floor_dead_load_ksf"], sp.floor_dead_load_ksf())
            self.assertEqual(result["history"][0]["slab"], result["slab"])
            from Design.SMRF_Demands import live_load_patterns
            patterns = live_load_patterns(sp.NUM_BAY_X, sp.NUM_BAY_Y)
            self.assertEqual(len(result["design_actions"]["combinations"]), 18 + len(patterns))
            # No story_strength_model_verified assertion: the strength criterion is provisional and a low
            # TIR cannot certify the absence of Type 1 (review item M1).
            self.assertEqual(result["demand_basis"]["torsion"]["torsional_irregularity"], "unresolved")
            self.assertEqual(result["demand_basis"]["regularity"]["lateral_strength_distribution"]["applicability"]["status"],
                             "provisional")
            self.assertEqual([p["id"] for p in result["demand_basis"]["live_load_patterns"]], [p["id"] for p in patterns])
            floor = result["floor_analysis"]
            self.assertEqual(floor["status"], "diagnostic_only")
            self.assertEqual(floor["errors"], [])
            self.assertEqual(len(floor["cases"]), 5)
            for case in [*floor["cases"], floor["refined_full_live_case"]]:
                self.assertEqual(case["status"], "diagnostic_complete")
                self.assertTrue(case["equilibrium"]["numerical_balance_passed"])
                self.assertFalse(case["verified"])
            self.assertTrue(floor["transferred_to_frame_design"])
            self.assertEqual(result["gravity_load_model"], "slab_transfer")
            transfer = result["floor_transfer"]
            self.assertEqual(transfer["slab_thickness_in"], result["slab"]["thickness_in"])
            area = sp.BAY_X * sp.NUM_BAY_X * sp.BAY_Y * sp.NUM_BAY_Y / 144.0
            for name, pressure in (("dead", sp.floor_dead_load_ksf()), ("live", sp.FLOOR_LIVE_LOAD_KSF)):
                case = transfer["unit_cases"][name]
                self.assertAlmostEqual(case["beam_kip"] + case["column_direct_kip"], pressure * area, places=9)
            self.assertFalse(result["detailing"]["full_cage_verified"])
            self.assertEqual(result["detailing"]["beam"]["hoop_spacing_in"], sp.BEAM_STIRRUP_SPACING)
            self.assertEqual(result["detailing"]["column"]["hoop_spacing_in"], sp.COL_STIRRUP_SPACING)
            self.assertLessEqual(sp.BEAM_STIRRUP_SPACING,
                                 min((sp.H_BEAM - sp.longitudinal_cover_in("beam"))/4,
                                     6*sp.rebar_diameter(sp.BEAM_BAR_SIZE), 6))
            # Default config carries no engineering assertions about the slab
            # actions: evidence is computed, nothing is certified, no layout.
            strength = result["slab_reinforcement"]
            self.assertIsNone(strength["layout"])
            # Rejected attempts retain their exact inputs for review.
            self.assertEqual(strength["inputs"]["demand_evidence"], result["slab_actions"])
            computed = result["slab_actions"]              # but it is saved for review
            self.assertFalse(computed["verified"])
            self.assertEqual(len(computed["strips"]), 4)   # 1 panel x 2 axes x 2 faces
            self.assertTrue(all(item["numerical_balance_passed"] for item in computed["equilibrium"]))
            sample = next(iter(result["beam_slab_strengths"].values()))
            self.assertEqual(sample["slab_basis"], "unknown")
            self.assertFalse(sample["section_compatibility_verified"])
            statuses = {c["id"]: c["status"] for c in result["qualification"]["checks"]}
            self.assertEqual(statuses["beam.interior_flexure_envelope"], "pass")
            from Design.SMRF_Beam_Actions import evaluate_saved_beam_bending
            first_case = result["design_actions"]["combinations"][0]
            first_beam = next(m for m in first_case["members"].values() if m["member_type"] == "beam_x")
            first_beam["span_bending"]["full_span"]["mu_positive_kip_in"] += 1.
            self.assertEqual(evaluate_saved_beam_bending(result)["status"], "fail")
            first_beam["span_bending"]["full_span"]["mu_positive_kip_in"] -= 1.
            self.assertEqual(evaluate_saved_beam_bending(result)["status"], "pass")
            recovered = first_beam.pop("span_bending")
            self.assertEqual(evaluate_saved_beam_bending(result)["status"], "not_evaluated")
            first_beam["span_bending"] = recovered
            self.assertGreaterEqual(recovered["full_span"]["mu_positive_kip_in"], 0.)
            self.assertEqual(statuses["qualification.slab_contribution"], "not_evaluated")
            self.assertEqual(statuses["floor.qualified_slab_actions"], "not_evaluated")
            self.assertEqual(statuses["scwb"] if "scwb" in statuses else "not_evaluated", "not_evaluated")
            capacity = result["capacity_design"]
            self.assertIn("beam_capacity_shear", result["joint_evidence"])
            self.assertEqual(len(capacity["joints"]["joints"]), 4)     # a one-bay frame has corner joints only
            # The saved capacity evidence is recomputed from the record itself at qualification.
            self.assertEqual(statuses["qualification.capacity_evidence_recomputed"], "pass")
            self.assertEqual(statuses["qualification.hoops_match_design"], "pass")
            self.assertEqual(statuses["floor.coupled_summary_consistent"], "pass")
            self.assertEqual(statuses["detailing.cage_layout"], "pass")      # generated arrangement, checked
            self.assertEqual(result["reinforcement"]["beam_stirrup_spacing_in"],
                             capacity["transverse"]["beam"]["spacing_in"])
            self.assertLessEqual(result["reinforcement"]["col_stirrup_spacing_in"], 4.0)
            self.assertEqual(statuses["qualification.detailing_model_consistency"], "not_evaluated")
            self.assertTrue(capacity["joints"]["joint_transverse"]["designed"])
            self.assertFalse(result["qualification"]["duplicate_check_ids"])
            from Design.SMRF_Design_Evidence import joint_evidence
            self.assertTrue(joint_evidence(result)["joints"])

            # With the engineering assertions made (after review), the same
            # pipeline selects a layout and the slab-dependent checks evaluate.
            from Design.Config import DesignConfig, SlabActionAssertions
            asserted = DesignConfig(slab_actions=SlabActionAssertions(
                analysis_applicability_verified=True, all_floors_enveloped=True,
                load_pattern_envelope_verified=True, spatial_envelope_per_unit_width=True,
                twisting_moment_resolution_verified=True, zero_membrane_force_verified=True,
                verified=True, two_way_shear_path_assessed=True,
                asserted_by="test", assertion_date="2026-09-13", assertion_basis="unit test fixture"))
            # Asserted flags without a refinement plan are refused before any
            # solve, naming the missing field (the single-mesh evidence from
            # the unasserted run above stays unverified).
            with self.assertRaisesRegex(ValueError, "FloorAnalysisConfig.slab_refinement"):
                driver._update_slab_reinforcement(asserted, result["slab"])
            self.assertFalse(sp.SLAB_ACTIONS["verified"])
            # A numerical comparison is required even for this signed fixture.
            # These normalized offsets reproduce the existing bounded graded
            # support refinement, scaled to this test's bay dimensions.
            half = [0., 2., 4., 7., 10., 15., 22., 32., 45., 60., 80., 100., 120.]
            base = sorted(set(half + [240. - x for x in half]))
            coarse = sorted(base + [(a + b) / 2 for a, b in zip(base, base[1:])])
            fine = sorted(coarse + [(a + b) / 2 for a, b in zip(coarse, coarse[1:])
                                    if 4 < (a + b) / 2 < 15 or 225 < (a + b) / 2 < 236])
            asserted.floor_analysis.slab_refinement = {
                "meshes": [{"x_offsets_in": [x * sp.BAY_X / 240 for x in offsets],
                            "y_offsets_in": [x * sp.BAY_Y / 240 for x in offsets], "max_shells": 45000}
                           for offsets in (coarse, fine)],
                "moment_tolerance": .05, "shear_tolerance": .05,
                "tolerance_basis": "Test fixture: inherited 5% local-demand investigation screen; not engineering approval"}
            verified = driver.design_structure(cfg=asserted, max_section_iter=1, max_steel_iter=1, verbose=False)
            layout = verified["slab_reinforcement"]["layout"]
            self.assertEqual(set(layout["layers"]), {"x_top", "x_bottom", "y_top", "y_bottom"})
            self.assertTrue(verified["slab_reinforcement"]["screen_passed"])
            evidence = verified["slab_reinforcement"]["inputs"]["demand_evidence"]
            self.assertTrue(evidence["verified"])
            self.assertEqual(evidence["engineering_assertions"]["asserted_by"], "test")
            self.assertEqual(evidence["shear_envelope_basis"], "support_face_maximum")
            families = verified["beam_slab_families"]
            self.assertGreater(families["x_interior"]["mn_negative_kip_in"],
                               families["x_interior"]["rectangular"]["negative"]["mn_kip_in"])
            sample = next(iter(verified["beam_slab_strengths"].values()))
            self.assertEqual(sample["slab_basis"], "developed_effective_width")
            # Governing joint: one roof column against Mnb- + Mnb+ of the strongest family.
            self.assertAlmostEqual(verified["scwb"]["beam_nominal_moment_kip_in"],
                                   max(f["mn_negative_kip_in"] + f["mn_positive_kip_in"] for f in families.values()))
            statuses = {c["id"]: c["status"] for c in verified["qualification"]["checks"]}
            self.assertEqual(statuses["qualification.slab_contribution"], "pass")
            self.assertEqual(statuses["floor.qualified_slab_actions"], "pass")
            self.assertEqual(statuses["slab_bar_development"], "pass")
            self.assertEqual(statuses["slab_two_way_shear_applicability"], "pass")
            self.assertNotEqual(verified["request_identity"]["sha256"] if "request_identity" in verified else 1,
                                result["request_identity"]["sha256"] if "request_identity" in result else 2)
            changed = copy.deepcopy(result)
            changed["sections"]["b_col_in"] += 2
            with self.assertRaisesRegex(ValueError, "Solved member actions"):
                joint_evidence(changed)
            checks = {c["id"]: c for c in qualify_design(changed)["checks"]}
            self.assertEqual(checks["joint.reproducible_actions"]["status"], "not_evaluated")
            json.dumps(result, allow_nan=False)
        finally:
            driver.ops.wipe()

    def test_slab_cache_roundtrip_and_tampering_rejection(self):
        saved = artifact()
        driver.apply_design(saved)
        self.assertEqual(sp.SLAB_THICKNESS_IN, saved["slab"]["thickness_in"])
        self.assertAlmostEqual(sp.floor_dead_load_ksf(), saved["slab"]["total_dead_load_ksf"])
        before = driver._capture_state()
        tampered = copy.deepcopy(saved)
        tampered["slab"]["thickness_in"] += 0.5
        with self.assertRaisesRegex(ValueError, "slab thickness evidence"):
            driver.apply_design(tampered)
        self.assertEqual(driver._capture_state(), before)
        tampered = copy.deepcopy(saved)
        tampered["sections"]["h_beam_in"] += 2
        with self.assertRaisesRegex(ValueError, "disagrees"):
            driver.apply_design(tampered)
        self.assertEqual(driver._capture_state(), before)

    def test_slab_policy_identity_is_not_selected_slab_state(self):
        cfg = DesignConfig()
        original = driver.design_request_identity(cfg)["sha256"]
        driver._select_slab(cfg)
        self.assertEqual(driver.design_request_identity(cfg)["sha256"], original)
        cfg.slab.superimposed_dead_load_ksf += .01
        self.assertNotEqual(driver.design_request_identity(cfg)["sha256"], original)
        cfg.slab.superimposed_dead_load_ksf -= .01
        cfg.slab.minimum_thickness_in = 8.0
        self.assertNotEqual(driver.design_request_identity(cfg)["sha256"], original)

    def test_reselects_slab_after_beam_rung_change(self):
        cfg = DesignConfig()
        with mock.patch.object(sp, "BAY_X", 240), mock.patch.object(sp, "BAY_Y", 240):
            driver._apply_rung((12, 14, 4), "beam")
            first = driver._select_slab(cfg)
            driver._apply_rung((18, 24, 4), "beam")
            second = driver._select_slab(cfg)
        self.assertGreater(first["thickness_in"], second["thickness_in"])
        self.assertEqual(sp.SLAB_THICKNESS_IN, second["thickness_in"])

    def test_slab_sizing_exhaustion_retries_deeper_compatible_beam(self):
        cfg = DesignConfig()
        cfg.slab.maximum_thickness_in = 6.0
        with mock.patch.object(sp, "BAY_X", 240), mock.patch.object(sp, "BAY_Y", 240):
            index, slab, attempts = driver._fit_slab_and_beam(
                cfg, [(12., 14., 4.), (12., 14., 5.), (18., 24., 4.)], 0)
        self.assertEqual(index, 2)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(slab["thickness_in"], 5.5)
        self.assertEqual(cfg.sections.h_beam_in, 24)

    def test_unsupported_slab_geometry_is_not_a_sizing_retry(self):
        from Design.SMRF_Slab import SlabSizingError
        with mock.patch.object(driver, "_select_slab", side_effect=ValueError("unsupported")) as select:
            with self.assertRaisesRegex(ValueError, "unsupported"):
                driver._fit_slab_and_beam(DesignConfig(), [(12., 14., 4.), (18., 24., 4.)], 0)
        self.assertEqual(select.call_count, 1)
        with mock.patch.object(driver, "_select_slab", side_effect=SlabSizingError("exhausted")) as select:
            with self.assertRaisesRegex(SlabSizingError, "No compatible beam/slab"):
                driver._fit_slab_and_beam(DesignConfig(), [(12., 14., 4.)], 0)
        self.assertEqual(select.call_count, 1)

    def test_export_contract_and_output_compatibility_include_slab_loads(self):
        from Data_Generation.Graph_Exporter import collect_global_parameters
        from Data_Generation.Hybrid_Exporter import GLOBAL_FEATURE_KEYS, _json_to_feature_array
        from Ground_Motion_Main import validate_ntha_output_compatibility
        driver._select_slab(DesignConfig())
        params = collect_global_parameters()
        self.assertEqual(params["floor_dead_load_ksf"], sp.floor_dead_load_ksf())
        self.assertEqual(params["floor_loads"]["slab_thickness_in"], sp.SLAB_THICKNESS_IN)
        self.assertEqual(len(GLOBAL_FEATURE_KEYS), 43)
        self.assertEqual(len(_json_to_feature_array(params, GLOBAL_FEATURE_KEYS)), 43)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "global_parameters.json").write_text(json.dumps(params), encoding="utf-8")
            validate_ntha_output_compatibility(path)
            sp.SLAB_THICKNESS_IN += .5
            with self.assertRaisesRegex(RuntimeError, "different model configurations"):
                validate_ntha_output_compatibility(path)

    def test_legacy_output_without_slab_metadata_can_still_be_read(self):
        from Data_Generation.Graph_Exporter import collect_global_parameters
        from Ground_Motion_Main import validate_ntha_output_compatibility
        sp.SLAB_THICKNESS_IN = None
        params = collect_global_parameters()
        params.pop("floor_loads")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "global_parameters.json").write_text(json.dumps(params), encoding="utf-8")
            validate_ntha_output_compatibility(path)

    def test_factored_analysis_uses_selected_slab_dead_load(self):
        driver._select_slab(DesignConfig())
        with mock.patch.object(driver, "build_model"), \
             mock.patch.object(driver, "apply_gravity_loads") as apply, \
             mock.patch.object(driver, "run_gravity_analysis"):
            driver._analyze_combination({"dead": 1.2, "live": 1.6, "ex": 0, "ey": 0}, .3)
        dead, live = sp.floor_dead_load_ksf(), sp.FLOOR_LIVE_LOAD_KSF
        self.assertAlmostEqual(apply.call_args.kwargs["floor_factor"], (1.2*dead + 1.6*live)/(dead+live))
        self.assertEqual(apply.call_args.kwargs["self_weight_factor"], 1.2)

    def test_saved_bar_geometry_and_material_tampering_rejected_before_apply(self):
        saved = artifact()
        before = driver._capture_state()
        for key in ("beam_bar_area_in2", "col_bar_diameter_in", "beam_stirrup_diameter_in",
                    "col_longitudinal_centroid_offset_in"):
            tampered = copy.deepcopy(saved)
            tampered["reinforcement"][key] += 0.1
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                driver.apply_design(tampered)
            self.assertEqual(driver._capture_state(), before)
        saved["materials"]["reinforcement_specification"] = "ASTM A615 Grade 60"
        with self.assertRaises(ValueError):
            driver.apply_design(saved)
        self.assertEqual(driver._capture_state(), before)

    def test_incomplete_scalar_restore_is_atomic(self):
        before = driver._capture_state()
        for group, key in (("sections", "fc_beam_ksi"), ("reinforcement", "col_top_bars"),
                           ("reinforcement", "beam_bot_bars"), ("geometry", "story_h_in")):
            saved = artifact()
            saved[group].pop(key)
            with self.assertRaises(ValueError):
                driver.apply_design(saved)
            self.assertEqual(driver._capture_state(), before)

    def test_unapproved_config_rejected_before_design_mutation(self):
        before = driver._capture_state()
        for key, value in (("beam_clear_cover_in", 1.25), ("col_clear_cover_in", float("nan")),
                           ("aggregate_max_size_in", 0)):
            cfg = DesignConfig()
            setattr(cfg.rebar, key, value)
            with self.assertRaises(ValueError):
                driver.design_structure(cfg=cfg)
            self.assertEqual(driver._capture_state(), before)
        cfg = DesignConfig()
        cfg.materials.fy_ksi = 80
        with self.assertRaises(ValueError):
            driver.design_structure(cfg=cfg)
        self.assertEqual(driver._capture_state(), before)

    def test_detailing_cannot_override_frame_section_or_cover(self):
        from Design.SMRF_Qualification import detailing_inputs
        saved = artifact()
        saved["detailing"] = {"beam": {"b_in": 99, "clear_cover_in": 0,
                                          "end_zone_length_in": 40}}
        values = detailing_inputs(saved)["beam"]
        self.assertEqual(values["b_in"], saved["sections"]["b_beam_in"])
        self.assertEqual(values["clear_cover_in"], sp.BEAM_CLEAR_COVER_IN)
        self.assertEqual(values["end_zone_length_in"], 40)
        checks = {c["id"]: c for c in qualify_design(saved)["checks"]}
        self.assertEqual(checks["detailing.beam_authoritative_inputs"]["status"], "fail")

    def test_floor_solver_returned_failures_are_reported_without_fabricating_bars(self):
        from Design.SMRF_Design_Evidence import floor_diagnostics
        saved = artifact()
        saved["floor_loads"] = {"floor_live_load_ksf": .05}
        for status in ("analysis_failed", "equilibrium_failed"):
            bad = {"status": status, "analysis_return_code": -1, "verified": False}
            with mock.patch("Design.SMRF_Floor_Analysis.analyze_floor", return_value=bad):
                result = floor_diagnostics(saved, DesignConfig().floor_analysis)
            self.assertEqual(result["status"], "diagnostic_errors")
            self.assertEqual(len(result["errors"]), 6)
            self.assertFalse(result["verified"])

    def test_slab_strength_metadata_must_match_current_frame(self):
        from Design.SMRF_Design_Evidence import slab_strength_inputs
        from Design.SMRF_Slab_Reinforcement import design_slab_reinforcement
        from test_smrf_slab_reinforcement import evidence
        saved = artifact()
        inputs = slab_strength_inputs(saved)
        saved["slab_reinforcement_inputs"] = inputs
        saved["slab_reinforcement"] = design_slab_reinforcement(inputs, evidence(inputs, mu=1, vu=.1))
        checks = {c["id"]: c for c in qualify_design(saved)["checks"]}
        self.assertEqual(checks["slab_strip.frame_input_consistency"]["status"], "pass")
        saved["slab"]["thickness_in"] += .5
        checks = {c["id"]: c for c in qualify_design(saved)["checks"]}
        self.assertEqual(checks["slab_strip.frame_input_consistency"]["status"], "fail")
        self.assertNotIn("slab_strip_flexure", checks)
        saved["materials"]["reinforcement_specification"] = "unknown"
        with self.assertRaisesRegex(ValueError, "cannot be inferred"):
            slab_strength_inputs(saved)

    def test_column_face_axial_transport_uses_actual_factored_self_weight(self):
        from Design.SMRF_Elastic import physical_members
        sp.NUM_BAY_X = sp.NUM_BAY_Y = 1
        sp.NUM_FLOOR = 2
        driver._select_slab(DesignConfig())
        dead = 1.4
        try:
            driver._analyze_combination({"dead": dead, "live": 0, "ex": 0, "ey": 0}, .3)
            actions = driver._capture_element_actions(dead)
            weight = dead * sp.col_self_weight_kip_per_in()
            for tag, ni, _nj, kind in physical_members():
                result = actions[str(tag)]
                if kind == "column":
                    offset_i = 0.0 if driver.ops.nodeCoord(ni, 3) == 0 else sp.H_BEAM / 2
                    self.assertAlmostEqual(result["centerline_axial_i_kip"] - result["centerline_axial_j_kip"],
                                           weight * sp.STORY_H, places=8)
                    self.assertAlmostEqual(result["axial_i_kip"], result["centerline_axial_i_kip"] - weight*offset_i)
                    self.assertAlmostEqual(result["axial_j_kip"], result["centerline_axial_j_kip"] + weight*sp.H_BEAM/2)
                else:
                    self.assertEqual(result["joint_face_offsets_in"], [0., 0.])
        finally:
            driver.ops.wipe()

    def test_transverse_limits_reapplied_before_checks_after_bar_updates(self):
        cfg = DesignConfig()
        sp.COL_BAR_SIZE = 8
        sp.B_COL = sp.H_COL = 18
        sp.COL_STIRRUP_SPACING = sp.BEAM_STIRRUP_SPACING = 6
        seen = []
        joints = {"evaluated": False, "all_pass": True, "counts": {}, "min_ratio_provided": None,
                  "ratio_required": 1.2, "basis": "test", "steel_raised": False, "steel_exhausted": False}

        def checks(*_args, **_kwargs):
            seen.append((sp.COL_STIRRUP_SPACING, cfg.rebar.stirrup_spacing_col_in))
            return .4, .7, {1: "member"}

        def update(*_args):
            sp.COL_BAR_SIZE = 5
            sp.COL_BAR_AREA = sp.rebar_area(5)
            sp.COL_STIRRUP_SPACING = 6  # Simulate a looser shear-only candidate.

        with mock.patch.object(driver, "_analyze_combination", return_value=None), \
             mock.patch.object(driver, "_governing_dcrs", side_effect=checks), \
             mock.patch.object(driver, "_capture_element_actions", return_value={}), \
             mock.patch.object(driver, "redesign_steel", return_value=({"x": 1}, None, False, [])), \
             mock.patch.object(driver, "_capacity_design", return_value=None), \
             mock.patch.object(driver, "_scwb_steel_floor", return_value=joints), \
             mock.patch.object(driver, "apply_updates", side_effect=update):
            driver._steel_pass(cfg, 1., max_steel_iter=2)
        self.assertEqual(seen[:18], [(4., 4.)] * 18)
        self.assertEqual(seen[18:], [(3., 3.)] * 18)


if __name__ == "__main__":
    unittest.main()
