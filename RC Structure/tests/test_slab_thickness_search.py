"""Screening repair items 1 and 3 (case_0003 of the 2 October screens): when the converged slab analysis admits no
reinforcement layout, the slab thickness steps up the declared ladder while the screen supports it, then the beam steps
to the next deeper compatible rung; every trial is kept with what refused it, exhaustion is explicit and carries that
evidence to the run folder; a refinement comparison failure is not retried."""
import contextlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Structure_Parameters as sp                                              # noqa: E402
from Design import Design_Driver as driver                                     # noqa: E402
from Design import Verify_Designs as verify                                    # noqa: E402

PROBE_DATE = "2026-10-02"


@contextlib.contextmanager
def tiny_frame():
    snapshot = dict(vars(sp))
    try:
        sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR = 1, 1, 2
        sp.BAY_X = sp.BAY_Y = 180.0
        sp.STORY_H = 144.0
        sp.NUM_MODES = 3
        sp.B_COL = sp.H_COL = 20.0
        sp.B_BEAM, sp.H_BEAM = 14.0, 24.0
        sp.apply_seismic_site("sdc_d_low")
        yield
    finally:
        for key in set(vars(sp)) - set(snapshot):
            delattr(sp, key)
        vars(sp).update(snapshot)


class ThicknessLadder(unittest.TestCase):
    def test_the_screen_can_start_from_a_raised_floor_and_records_it(self):
        with tiny_frame():
            cfg = verify.probe_config(PROBE_DATE)
            base = driver._select_slab(cfg)
            raised = driver._select_slab(cfg, 7.0)
            self.assertLess(base["thickness_in"], 7.0)
            self.assertEqual(raised["thickness_in"], 7.0)
            self.assertEqual(raised["inputs"]["policy"]["minimum_thickness_in"], 7.0)
            self.assertEqual(sp.SLAB_THICKNESS_IN, 7.0)
            thicker, unavailable = driver._thicker_slab(cfg, raised)
            self.assertEqual(thicker["thickness_in"], 7.0 + cfg.slab.thickness_increment_in)
            self.assertIsNone(unavailable)
            top = dict(raised, thickness_in=cfg.slab.maximum_thickness_in)
            none, unavailable = driver._thicker_slab(cfg, top)
            self.assertIsNone(none)
            self.assertIn("tops out at 14 in", unavailable)

    def test_a_screen_that_supports_no_thicker_slab_is_named_as_the_reason(self):
        """case_0003: 14 x 20 beams on 30 ft bays pass the screen at 12 in only; above it alpha_fm leaves the
        beam-supported scope, which is the reason the thickness ladder ends, not its declared maximum."""
        from Design.SMRF_Slab import SlabSizingError
        with tiny_frame():
            cfg = verify.probe_config(PROBE_DATE)
            slab = driver._select_slab(cfg)
            with mock.patch.object(driver, "_select_slab", side_effect=SlabSizingError("7/19 unsupported trials")):
                none, unavailable = driver._thicker_slab(cfg, slab)
            self.assertIsNone(none)
            self.assertIn("screen supports no slab from 5.5 to 14 in on beam 14 x 24 in", unavailable)
            self.assertIn("unsupported trials", unavailable)

    def test_the_search_steps_until_a_layout_passes_and_keeps_every_trial(self):
        with tiny_frame():
            cfg = verify.probe_config(PROBE_DATE)
            slab = driver._select_slab(cfg)

            def reinforcement(cfg_, slab_):
                if slab_["thickness_in"] < 6.5:
                    raise driver.SlabLayoutError("Slab reinforcement not selected (slab refinement passed): No allowed four-layer "
                                                 "layout passes; increase slab thickness or revisit demands/policy.", slab_["thickness_in"])
                return {"layout": {"bar_size": 4}}

            with mock.patch.object(driver, "_update_slab_reinforcement", side_effect=reinforcement), \
                 mock.patch.object(driver, "_update_floor_transfer") as transfer:
                chosen, search = driver._slab_thickness_search(cfg, slab)
            self.assertEqual(chosen["thickness_in"], 6.5)
            self.assertEqual(search["thickness_in"], 6.5)
            self.assertEqual([t["thickness_in"] for t in search["trials"]], [5.0, 5.5, 6.0])
            self.assertEqual(search["steps"], 3)
            self.assertIn("four-layer", search["trials"][0]["reason"])
            self.assertEqual(transfer.call_count, 3)                                # rebuilt at every new thickness
            self.assertEqual(sp.SLAB_THICKNESS_IN, 6.5)

    def test_exhaustion_is_explicit_and_assigns_nothing(self):
        with tiny_frame():
            cfg = verify.probe_config(PROBE_DATE)
            slab = driver._select_slab(cfg)

            def never(cfg_, slab_):
                raise driver.SlabLayoutError("no layout", slab_["thickness_in"])

            with mock.patch.object(driver, "_update_slab_reinforcement", side_effect=never), \
                 mock.patch.object(driver, "_update_floor_transfer"):
                with self.assertRaises(driver.SlabThicknessExhausted) as caught:
                    driver._slab_thickness_search(cfg, slab)
            self.assertRegex(str(caught.exception), r"at any thickness from 5 to 14 in on beam 14 x 24 in \(19 thickness "
                                                    r"trial\(s\); next thickness unavailable: the declared ladder tops out at 14 in\)")
            self.assertEqual(len(caught.exception.trials), 19)
            self.assertEqual(caught.exception.evidence["next_thickness_unavailable"], caught.exception.reason)

    def test_a_refinement_comparison_failure_is_not_a_thickness_matter(self):
        """_update_slab_reinforcement with its solvers mocked: layout None after a passed refinement is the search's
        cue (SlabLayoutError); after a failed comparison it stays the plain error."""
        with tiny_frame():
            cfg = verify.probe_config(PROBE_DATE)
            slab = driver._select_slab(cfg)
            checks = [{"id": "slab_strip_reinforcement_ladder", "status": "fail",
                       "details": {"reason": "No allowed four-layer layout passes; increase slab thickness or revisit demands/policy."}}]

            def run(status):
                evidence = {"refinement": {"status": status, "comparisons": [{"comparisons": [
                    {"metric": "vu_kip_per_ft", "relative_change": 0.3, "within_tolerance": status == "passed"}]}],
                    "levels": [{"requested_mesh": {"subdivisions_per_bay": 24}}, {"requested_mesh": {"subdivisions_per_bay": 48}}]}}
                with mock.patch("Design.SMRF_Slab_Refinement.build_refined_slab_action_evidence", return_value=evidence), \
                     mock.patch("Design.SMRF_Slab_Reinforcement.design_slab_reinforcement", return_value={"layout": None, "checks": checks}), \
                     mock.patch.object(driver, "_slab_strength_inputs_from_state", return_value={}), \
                     mock.patch.object(driver, "_slab_completion_context", return_value={}):
                    driver._update_slab_reinforcement(cfg, slab)

            self.assertTrue(cfg.slab_actions.all_asserted())
            with self.assertRaises(driver.SlabLayoutError) as caught:
                run("passed")
            self.assertEqual(caught.exception.thickness_in, slab["thickness_in"])
            self.assertIn("four-layer", str(caught.exception))
            # Only a passed refinement makes a missing layout a thickness matter: unconverged levels, a level
            # that did not solve (an unclosed sagging-shear bound among them) and an unresolved shell budget
            # all stay errors the thickness search does not step on.
            for status in ("comparison_failed", "analysis_failed", "unresolved_budget"):
                with self.assertRaises(RuntimeError) as plain:
                    run(status)
                self.assertNotIsInstance(plain.exception, driver.SlabLayoutError, status)
                self.assertIsInstance(plain.exception, driver.SlabDesignError, status)
                self.assertIn(status, str(plain.exception))


LADDER = [(14.0, 24.0, 4.0), (14.0, 24.0, 5.0), (14.0, 26.0, 4.0), (14.0, 28.0, 4.0)]


class BeamRungStep(unittest.TestCase):
    def _exhausted(self, thickness):
        trials = [{"thickness_in": thickness, "reason": "no layout", "evidence": {"unsized_layers": {"x_top": {}}}}]
        return driver.SlabThicknessExhausted("exhausted", trials, "the screen supports no thicker slab",
                                             {"thickness_trials": trials})

    def test_an_exhausted_thickness_ladder_steps_to_the_next_deeper_rung(self):
        with tiny_frame():
            cfg = verify.probe_config(PROBE_DATE)

            def thickness_search(cfg_, slab_):
                if sp.H_BEAM < 28.0:
                    raise self._exhausted(slab_["thickness_in"])
                return slab_, {"thickness_in": slab_["thickness_in"], "steps": 0, "trials": []}

            with mock.patch.object(driver, "_slab_thickness_search", side_effect=thickness_search), \
                 mock.patch.object(driver, "_update_floor_transfer"):
                index, slab, attempts, search = driver._slab_search(cfg, LADDER, 0)
            self.assertEqual(index, 3)                                               # same-depth f'c rung skipped
            self.assertEqual((sp.B_BEAM, sp.H_BEAM), (14.0, 28.0))
            self.assertEqual(search["first_fitted_beam_index"], 0)
            self.assertEqual([step["beam_section"] for step in search["beam_rung_steps"]],
                             [[14.0, 24.0, 4.0], [14.0, 26.0, 4.0]])
            self.assertEqual(search["beam_rung_steps"][0]["next_thickness_unavailable"], "the screen supports no thicker slab")
            self.assertEqual(attempts, [])

    def test_a_rung_that_passes_records_no_step(self):
        with tiny_frame():
            cfg = verify.probe_config(PROBE_DATE)
            with mock.patch.object(driver, "_slab_thickness_search",
                                   side_effect=lambda c, s: (s, {"thickness_in": s["thickness_in"], "steps": 0, "trials": []})), \
                 mock.patch.object(driver, "_update_floor_transfer"):
                index, _slab, _attempts, search = driver._slab_search(cfg, LADDER, 0)
            self.assertEqual(index, 0)
            self.assertEqual(search["beam_rung_steps"], [])

    def test_exhaustion_of_every_rung_is_explicit_and_keeps_every_trial(self):
        with tiny_frame():
            cfg = verify.probe_config(PROBE_DATE)
            with mock.patch.object(driver, "_slab_thickness_search",
                                   side_effect=lambda c, s: (_ for _ in ()).throw(self._exhausted(s["thickness_in"]))), \
                 mock.patch.object(driver, "_update_floor_transfer"):
                with self.assertRaises(driver.SlabSearchExhausted) as caught:
                    driver._slab_search(cfg, LADDER, 0)
            self.assertRegex(str(caught.exception), r"from \(14\.0, 24\.0, 4\.0\) to \(14\.0, 28\.0, 4\.0\) \(3 beam rung\(s\), "
                                                    r"3 thickness trial\(s\); no deeper compatible beam rung remains\)")
            steps = caught.exception.evidence["beam_rung_steps"]
            self.assertEqual(len(steps), 3)
            self.assertIn("unsized_layers", steps[0]["thickness_trials"][0]["evidence"])


class FailureEvidence(unittest.TestCase):
    def test_the_explanation_names_the_check_no_offered_layer_can_pass(self):
        """A shear demand above every candidate's phi Vc: the layer is unsized and one-way shear is what remains."""
        from Design.SMRF_Slab_Reinforcement import design_slab_reinforcement, explain_unsized_layers
        from tests.test_smrf_slab_reinforcement import evidence, inputs
        record = design_slab_reinforcement(inputs(), evidence(mu=80.0, vu=40.0))
        self.assertIsNone(record["layout"])
        explanation = explain_unsized_layers(record)
        self.assertEqual(set(explanation), {"x_top", "x_bottom", "y_top", "y_bottom"})
        entry = explanation["x_top"]
        self.assertEqual(entry["demand_envelope"]["vu_kip_per_ft"], 40.0)
        self.assertEqual([c["id"] for c in entry["failed_checks"]], ["slab_strip_one_way_shear"])
        self.assertLess(entry["failed_checks"][0]["capacity"], 40.0)
        self.assertEqual(entry["closest_offered_candidate"]["phi_vc_kip_per_ft"], entry["failed_checks"][0]["capacity"])
        passing = design_slab_reinforcement(inputs(), evidence())
        self.assertEqual(explain_unsized_layers(passing), {})
        self.assertEqual(record, design_slab_reinforcement(inputs(), evidence(mu=80.0, vu=40.0)))   # record untouched

    def test_a_layout_refusal_carries_the_bar_trials_and_the_explanation(self):
        with tiny_frame():
            cfg = verify.probe_config(PROBE_DATE)
            slab = driver._select_slab(cfg)
            checks = [{"id": "slab_strip_reinforcement_ladder", "status": "fail", "details": {"reason": "No allowed four-layer layout passes"}}]
            evidence = {"refinement": {"status": "passed", "comparisons": [{"comparisons": [
                {"metric": "vu_kip_per_ft", "relative_change": 0.01, "within_tolerance": True}]}],
                "levels": [{"requested_mesh": {"subdivisions_per_bay": 24}}, {"requested_mesh": {"subdivisions_per_bay": 48}}]}}
            record = {"layout": None, "checks": checks, "trial_history": [{"bar_size": 4, "passed": False, "layers_sized": []}]}
            with mock.patch("Design.SMRF_Slab_Refinement.build_refined_slab_action_evidence", return_value=evidence), \
                 mock.patch("Design.SMRF_Slab_Reinforcement.design_slab_reinforcement", return_value=record), \
                 mock.patch.object(driver, "_slab_strength_inputs_from_state", return_value={}), \
                 mock.patch.object(driver, "_slab_completion_context", return_value={}):
                with self.assertRaises(driver.SlabLayoutError) as caught:
                    driver._update_slab_reinforcement(cfg, slab)
            refusal = caught.exception.evidence
            self.assertEqual(refusal["thickness_in"], slab["thickness_in"])
            self.assertEqual(refusal["beam_section"], [14.0, 24.0, sp.FC_BEAM_KSI])
            self.assertEqual(refusal["bar_trials"], record["trial_history"])
            self.assertEqual(refusal["failed_checks"], checks)
            self.assertEqual(len(refusal["refinement"]["final_strip_comparisons"]), 1)
            self.assertIn("unsized_layers", refusal)

    def test_the_worker_writes_the_evidence_beside_the_result(self):
        case = {"case_id": "case_9001"}
        evidence = {"beam_rung_steps": [{"beam_section": [14.0, 24.0, 4.0], "thickness_trials": []}]}
        with tempfile.TemporaryDirectory() as folder, tiny_frame():
            with mock.patch.object(verify, "configure_case", return_value=({}, None)), \
                 mock.patch("Design.Design_Driver.load_or_create_design",
                            side_effect=driver.SlabSearchExhausted("no slab at any rung", evidence)):
                result = verify._run_worker(case, folder, True, PROBE_DATE)
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["failure_evidence"], "failure_evidence.json")
            with open(os.path.join(folder, "failure_evidence.json"), encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), evidence)
            with mock.patch.object(verify, "configure_case", return_value=({}, None)), \
                 mock.patch("Design.Design_Driver.load_or_create_design", side_effect=RuntimeError("plain")):
                plain = verify._run_worker(case, os.path.join(folder, "plain"), True, PROBE_DATE)
            self.assertNotIn("failure_evidence", plain)
            self.assertFalse(os.path.exists(os.path.join(folder, "plain", "failure_evidence.json")))


if __name__ == "__main__":
    unittest.main()
