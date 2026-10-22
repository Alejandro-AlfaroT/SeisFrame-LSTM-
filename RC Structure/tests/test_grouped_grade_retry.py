"""Grade 80 fallback decisions, fresh searches, provenance and atomic restoration."""
import copy
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf
import Structure_Parameters as sp
from Design import Grouped_Record as gr, Grouped_Search as gs, Grouped_Runner as runner
from Design.Config import DesignConfig


def evaluation(**constraints):
    return {"status": "evaluated", "feasible": False, "constraints": constraints, "dcr": {"beam": 1.2, "column": .8}}


def search_result(e=None, found=False, grade=60.):
    return {"version": gs.SEARCH_VERSION, "policy": asdict(gs.SearchPolicy()),
            "final": {"token": "complete Grade 80 evaluation"} if found else None,
            "stop": {"reason": "reduction_budget_exhausted" if found else "feasibility_no_growth_move"},
            "log": [{"index": 1, "phase": "feasibility", "longitudinal_grade_ksi": grade,
                     "decision": "feasible" if found else "not_feasible", "evaluation": e or evaluation()}],
            "counts": {"feasibility_candidates": 1, "reduction_candidates": 0, "skipped_moves": 0,
                       "accepted_reductions": 0, "tradeoffs": 0}, "tradeoffs": [], "elapsed_seconds": 1.}


class GradeRetryTests(unittest.TestCase):
    def setUp(self):
        self.frame = gf.frame(FY_KSI=60., REINFORCEMENT_SPECIFICATION=sp.REINFORCEMENT_SPECIFICATION_BY_GRADE[60.])
        self.frame.__enter__()
        self.cfg = DesignConfig.from_structure_parameters()
        self.seed = gf.uniform_state()

    def tearDown(self):
        self.frame.__exit__(None, None, None)

    def test_strength_and_named_congestion_are_retry_reasons(self):
        cases = [evaluation(strength_within_ceiling=False),
                 evaluation(lane_limited_beam_groups=["s01_03__beam_x__edge"]),
                 evaluation(exhausted_groups=["s01_03__beam_x__edge"]),
                 evaluation(capacity_design_failed_checks=[{"id": "column.cage_layout", "status": "fail"}]),
                 evaluation(exhausted_column_lines={"corner": {
                     "blocked_by": "no admissible cage of this band meets its own strength and strong-column demands"}}),
                 {"status": "infeasible", "failed_stage": "bar_rows", "failure_evidence": {"group_ids": ["beam"]}},
                 {"status": "infeasible", "failed_stage": "capacity_design", "failure_evidence": {
                     "exception": "SectionAxialDomainError", "group_id": "column"}}]
        for case in cases:
            with self.subTest(case=case):
                self.assertTrue(gs.grade80_retry_reasons(case))

    def test_drift_solver_transition_and_open_checks_do_not_trigger(self):
        cases = [evaluation(drift_accepted=False, strength_within_ceiling=True),
                 {"status": "error", "reason": "solver error"},
                 {"status": "infeasible", "failed_stage": "gravity_analysis"},
                 {"status": "infeasible", "failed_stage": "slab_thickness"},
                 evaluation(exhausted_groups=["s01_03__column__corner"], exhausted_column_lines={
                     "corner": {"blocked_by": "no cage of this band continues a feasible chain from the band below"}}),
                 evaluation(capacity_design_failed_checks=[{"id": "column.cage_layout", "status": "not_evaluated"}]),
                 evaluation(capacity_design_failed_checks=[{"id": "joint.through_bar_depth", "status": "fail"}])]
        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(gs.grade80_retry_reasons(case), [])

    def test_retry_restarts_original_seed_updates_config_and_keeps_both_attempts(self):
        primary = search_result(evaluation(strength_within_ceiling=False))
        secondary = search_result(found=True, grade=80.)
        calls = []
        def run(seed, cfg, policy, **kwargs):
            calls.append((seed, cfg, kwargs, sp.FY_KSI, sp.FYT_KSI, sp.SLAB_FY_KSI))
            if len(calls) == 2:
                self.assertIsNone(sp.FLOOR_TRANSFER)
                self.assertIsNone(sp.SLAB_REINFORCEMENT)
                self.assertNotIn("context", kwargs)
            return copy.deepcopy(primary if len(calls) == 1 else secondary)
        with tempfile.TemporaryDirectory() as directory, patch.object(gs, "search", side_effect=run), \
                patch.object(gr.gd, "install_evaluation") as install:
            path = Path(directory) / "grouped_candidates.jsonl"
            result, cfg = gr._search_with_grade_retry(self.cfg, self.seed, gs.SearchPolicy(), path, False)
            self.assertEqual(json.loads((Path(directory) / "grouped_steel_grade_trial.json").read_text())[
                "selected_grade_ksi"], 80.)
            self.assertEqual(calls[1][2]["log_path"].name, "grouped_candidates_grade80.jsonl")
            install.assert_called_once_with(secondary["final"])
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(c[0] is self.seed for c in calls))
        self.assertEqual([c[3:] for c in calls], [(60., 60., 60.), (80., 60., 60.)])
        self.assertEqual(cfg.materials.fy_ksi, 80.)
        self.assertEqual(cfg.materials.reinforcement_specification, sp.REINFORCEMENT_SPECIFICATION_BY_GRADE[80.])
        self.assertEqual(self.cfg.materials.fy_ksi, 60.)
        self.assertEqual(result["counts"]["feasibility_candidates"], 2)
        self.assertEqual([e["index"] for e in result["log"]], [1, 2])
        self.assertEqual([e["longitudinal_grade_ksi"] for e in result["log"]], [60., 80.])
        self.assertEqual(sp.FY_KSI, 80.)

    def test_existing_feasible_grade60_and_initial_grade80_do_not_retry(self):
        for initial, found, policy in ((60., True, gs.SearchPolicy()), (80., False, gs.SearchPolicy()),
                                        (60., False, gs.SearchPolicy(grade80_retry_on_limits=False))):
            with self.subTest(initial=initial, found=found):
                sp.apply_reinforcement_grade(initial)
                cfg = DesignConfig.from_structure_parameters()
                result = search_result(evaluation(strength_within_ceiling=False), found=found, grade=initial)
                with patch.object(gs, "search", return_value=result) as run:
                    search, selected = gr._search_with_grade_retry(cfg, self.seed, policy, None, False)
                self.assertEqual(run.call_count, 1)
                self.assertFalse(search["steel_grade_trial"]["attempted"])
                self.assertIs(selected, cfg)

    def test_only_terminal_failure_triggers_retry(self):
        result = search_result(evaluation(drift_accepted=False, strength_within_ceiling=True))
        result["log"].insert(0, {"index": 0, "phase": "feasibility",
                                  "evaluation": evaluation(strength_within_ceiling=False)})
        with patch.object(gs, "search", return_value=result) as run:
            result, cfg = gr._search_with_grade_retry(self.cfg, self.seed, gs.SearchPolicy(), None, False)
        self.assertEqual(run.call_count, 1)
        self.assertFalse(result["steel_grade_trial"]["attempted"])

    def test_failed_and_error_trials_restore_grade_and_derived_state(self):
        for error in (False, True):
            with self.subTest(error=error):
                old_layout = {"sentinel": "Grade 60 slab"}
                sp.SLAB_REINFORCEMENT = old_layout
                calls = []
                def run(*a, **kw):
                    calls.append(sp.FY_KSI)
                    if len(calls) == 2:
                        sp.SLAB_REINFORCEMENT = {"sentinel": "Grade 80 slab"}
                        if error:
                            raise RuntimeError("solver error during retry")
                    return search_result(evaluation(strength_within_ceiling=False), grade=sp.FY_KSI)
                with patch.object(gs, "search", side_effect=run):
                    if error:
                        with self.assertRaisesRegex(RuntimeError, "solver error"):
                            gr._search_with_grade_retry(self.cfg, self.seed, gs.SearchPolicy(), None, False)
                    else:
                        result, cfg = gr._search_with_grade_retry(self.cfg, self.seed, gs.SearchPolicy(), None, False)
                        self.assertIsNone(result["final"])
                        self.assertEqual(result["steel_grade_trial"]["selected_grade_ksi"], 60.)
                self.assertEqual(calls, [60., 80.])
                self.assertEqual(sp.FY_KSI, 60.)
                self.assertIs(sp.SLAB_REINFORCEMENT, old_layout)

    def retry_record(self):
        primary = search_result(evaluation(strength_within_ceiling=False))
        retry = search_result(found=True, grade=80.)
        with patch.object(gs, "search", side_effect=[primary, retry]), patch.object(gr.gd, "install_evaluation"):
            search, _ = gr._search_with_grade_retry(self.cfg, self.seed, gs.SearchPolicy(), None, False)
        sp.apply_reinforcement_grade(60.)
        return {"materials": {"fy_ksi": 80.}, "search": {"policy": search["policy"],
                "steel_grade_trial": search["steel_grade_trial"], "candidates": search["log"]}}

    def test_saved_grade80_retry_can_be_loaded_from_geometry_grade60(self):
        record = self.retry_record()
        self.assertEqual(gr._validated_record_grade(record), 80.)
        self.assertEqual(sp.FY_KSI, 60.)  # validation itself does not mutate state

    def test_material_edit_or_missing_retry_evidence_is_rejected(self):
        record = self.retry_record()
        for mutate in (lambda r: r["search"].pop("steel_grade_trial"),
                       lambda r: r["search"]["steel_grade_trial"].update(triggers=[]),
                       lambda r: r["search"]["candidates"][0].update(evaluation=evaluation(drift_accepted=False)),
                       lambda r: r["search"]["policy"].update(grade80_retry_on_limits=False),
                       lambda r: r["search"]["candidates"][1].update(decision="not_feasible")):
            altered = copy.deepcopy(record)
            mutate(altered)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                gr._validated_record_grade(altered)

    def test_failed_install_restores_grade60(self):
        record = self.retry_record()
        record.update(schema_version=gr.SCHEMA, design_mode="grouped", geometry=gr.driver._slab_geometry(),
                      member_groups=self.seed.to_record(), floor_model=sp.floor_model_record())
        record["materials"].update(es_ksi=sp.ES_KSI, fyt_ksi=60., aggregate_size_in=.75,
                                   beam_clear_cover_in=1.5, col_clear_cover_in=1.5,
                                   reinforcement_specification=sp.REINFORCEMENT_SPECIFICATION_BY_GRADE[80.])
        def fail(_record):
            self.assertEqual(sp.FY_KSI, 80.)
            raise ValueError("invalid slab")
        with patch.object(gr, "_validated_slab", side_effect=fail), self.assertRaisesRegex(ValueError, "invalid slab"):
            gr.apply_grouped_design(record)
        self.assertEqual(sp.FY_KSI, 60.)
        self.assertEqual(sp.REINFORCEMENT_SPECIFICATION, sp.REINFORCEMENT_SPECIFICATION_BY_GRADE[60.])

    def test_policy_is_in_identity_and_cli_can_disable_it(self):
        seed = {"grouping_policy": self.seed.policy_id}
        a = gr.grouped_request_identity(self.cfg, seed, gs.SearchPolicy())
        b = gr.grouped_request_identity(self.cfg, seed, gs.SearchPolicy(grade80_retry_on_limits=False))
        self.assertNotEqual(a["sha256"], b["sha256"])
        self.assertEqual(a["search"]["steel_grade_retry_rule"], gs.STEEL_GRADE_RETRY_RULE)
        args = runner.build_parser().parse_args(["--max-feasibility-trials", "16", "--max-reduction-trials", "6",
                                                  "--no-grade80-retry"])
        self.assertTrue(args.no_grade80_retry)
        self.assertTrue(gs.SearchPolicy(grade80_retry_on_limits="yes").problems())


if __name__ == "__main__":
    unittest.main()
