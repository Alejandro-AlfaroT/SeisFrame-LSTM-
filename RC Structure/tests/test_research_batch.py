"""The diagnostic research batch runner: its scaling rule, its plan and its run bookkeeping (2026-10-04).

No response-history analysis is run here; the pieces that decide WHAT is run and how it is scaled are checked
against hand values, and the claim / resume / shard bookkeeping against the file system.
"""
import json
import math
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

RC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_DIR))
sys.path.insert(0, str(RC_DIR / "Data_Generation"))
import Run_Research_Batch as batch                                               # noqa: E402

MANIFEST = RC_DIR / "pilots" / "v2DesignPilot100" / "screening_plan.json"


def mode(number, period, ratio_x, ratio_y, valid=True):
    return {"mode": number, "valid": valid, "period": period,
            "participation": None if not valid else {"modal_mass_ratio_x": ratio_x, "modal_mass_ratio_y": ratio_y}}


class ScalingRule(unittest.TestCase):
    def test_design_spectrum_by_hand(self):
        sds, sd1, tl = 1.0, 0.6, 8.0                                  # Ts = 0.6 s, T0 = 0.12 s
        self.assertAlmostEqual(batch.design_spectrum_sa_g(0.06, sds, sd1, tl), 0.7)
        self.assertAlmostEqual(batch.design_spectrum_sa_g(0.12, sds, sd1, tl), 1.0)
        self.assertAlmostEqual(batch.design_spectrum_sa_g(0.6, sds, sd1, tl), 1.0)
        self.assertAlmostEqual(batch.design_spectrum_sa_g(1.2, sds, sd1, tl), 0.5)
        self.assertAlmostEqual(batch.design_spectrum_sa_g(10.0, sds, sd1, tl), 0.6 * 8.0 / 100.0)
        with self.assertRaises(ValueError):
            batch.design_spectrum_sa_g(0.0, sds, sd1, tl)

    def test_pseudo_acceleration_against_the_resonant_steady_state_and_linearity(self):
        period, dt, amplitude = 1.0, 0.005, 0.1
        time = np.arange(0.0, 60.0, dt)
        ground = amplitude * np.sin(2.0 * np.pi * time / period)
        sa = batch.pseudo_acceleration_g(ground, dt, period)
        self.assertAlmostEqual(sa / amplitude, 1.0 / (2.0 * batch.DAMPING), delta=0.15)      # 10 at 5 % damping
        self.assertAlmostEqual(batch.pseudo_acceleration_g(3.0 * ground, dt, period), 3.0 * sa, places=9)
        self.assertLess(batch.pseudo_acceleration_g(ground, dt, 4.0), sa / 5.0)               # far from resonance

    def test_reference_period_is_chosen_by_participation_not_by_mode_number(self):
        modes = [mode(1, 1.60, 1e-9, 2e-9),                           # torsional: first, longest, no translation
                 mode(2, 1.20, 0.02, 0.80), mode(3, 1.05, 0.81, 0.01), mode(4, 0.40, 0.10, 0.00), mode(5, None, 0, 0, valid=False)]
        periods = batch.directional_periods(modes)
        self.assertEqual((periods["x"]["mode"], periods["y"]["mode"]), (3, 2))
        self.assertEqual(periods["reference_period_sec"], 1.20)
        self.assertEqual(periods["reference_direction"], "y")
        self.assertFalse(periods["same_mode_in_both_directions"])
        with self.assertRaises(RuntimeError):
            batch.directional_periods([mode(1, 1.0, 0.8, 0.0)])       # nothing moves in Y
        with self.assertRaises(RuntimeError):
            batch.directional_periods([mode(1, None, 0, 0, valid=False)])

    def test_one_common_multiplier_meets_the_geometric_mean_target(self):
        factor, geometric_mean = batch.scale_factor(1.5, 0.4, 0.9, 0.1)
        self.assertAlmostEqual(geometric_mean, 0.3)
        self.assertAlmostEqual(factor, 1.5 * 0.4 / 0.3)
        self.assertAlmostEqual(math.sqrt((factor * 0.9) * (factor * 0.1)), 1.5 * 0.4)        # the target, and the imbalance is kept
        self.assertAlmostEqual((factor * 0.9) / (factor * 0.1), 9.0)
        with self.assertRaises(RuntimeError):
            batch.scale_factor(1.0, 0.4, 0.0, 0.1)


class Assignment(unittest.TestCase):
    def cases(self, count=100):
        sites = ("sdc_c", "sdc_d_low", "sdc_d_high", "sdc_e", "sdc_e_near")
        return [{"case_id": f"case_{k + 1:04d}", "seismic_site": sites[(k * 7) % 5], "num_floor": 4 + (k * 3) % 6} for k in range(count)]

    def test_alpha_groups_are_33_34_33_balanced_and_seeded(self):
        cases = self.cases()
        alphas = batch.assign_alphas(cases, 11)
        counts = {a: sum(1 for v in alphas.values() if v == a) for a in (0.5, 1.0, 1.5)}
        self.assertEqual(counts, {0.5: 33, 1.0: 34, 1.5: 33})
        for site in {c["seismic_site"] for c in cases}:
            here = [alphas[c["case_id"]] for c in cases if c["seismic_site"] == site]
            for alpha in (0.5, 1.0, 1.5):
                self.assertLessEqual(abs(here.count(alpha) - len(here) / 3.0), 1.0, (site, alpha))
        self.assertEqual(alphas, batch.assign_alphas(list(reversed(cases)), 11))             # order of the input does not matter
        self.assertNotEqual(alphas, batch.assign_alphas(cases, 12))

    def test_records_are_not_reused_until_the_set_is_exhausted(self):
        pairs = [{"pair_key": f"p{k}"} for k in range(7)]
        cases = self.cases(10)
        chosen = batch.assign_records(cases, pairs, 5)
        keys = [chosen[c["case_id"]]["pair_key"] for c in cases]
        self.assertEqual(len(set(keys[:7])), 7)
        self.assertEqual(keys[7:], keys[:3])
        self.assertEqual(chosen, batch.assign_records(cases, pairs, 5))
        with self.assertRaises(RuntimeError):
            batch.assign_records(cases, [], 5)


class PlanAndBookkeeping(unittest.TestCase):
    def test_plan_is_reproducible_pins_designs_and_refuses_an_edit_or_a_changed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root, designs = Path(directory) / "out", Path(directory) / "designs"
            (designs / "case_0001").mkdir(parents=True)
            (designs / "case_0001" / "design.json").write_text("{}", encoding="utf-8")
            (designs / "case_0001" / "result.json").write_text(json.dumps({
                "status": "designed", "design_sha256": "abc", "profile_id": "p", "profile_sha256": "q",
                "fail_ids": [], "not_evaluated_ids": ["demands.torsional_irregularity"]}), encoding="utf-8")
            args = Namespace(manifest=str(MANIFEST), design_roots=[str(designs)], record_set=batch.DEFAULT_RECORD_SET, seed=3)
            plan = batch.load_or_write_plan(root, args)
            self.assertEqual(len(plan["cases"]), 100)
            self.assertEqual(plan["scaling"]["alpha_counts"], {"0.5": 33, "1.0": 34, "1.5": 33})
            self.assertEqual(plan["designs_pinned"], 1)
            first = plan["cases"][0]
            self.assertEqual({k: first['design'][k] for k in
                              ('status','design_sha256','profile_id','profile_sha256','failed_checks','open_checks')},
                             {"status": "designed", "design_sha256": "abc", "profile_id": "p", "profile_sha256": "q",
                              "failed_checks": 0, "open_checks": 1})
            self.assertEqual(first['design']['result_sha256'], batch._sha256_file(designs/'case_0001/result.json'))
            self.assertIsNone(first['design']['m1_addendum_sha256'])
            self.assertIsNone(plan["cases"][1]["design"])
            self.assertEqual(len(first["record"]["x"]["sha256"]), 64)
            self.assertNotIn(str(designs), json.dumps(plan))                                  # no machine path in the plan
            self.assertEqual(batch.load_or_write_plan(root, args)["plan_sha256"], plan["plan_sha256"])
            with self.assertRaisesRegex(RuntimeError, "different plan"):
                batch.load_or_write_plan(root, Namespace(**{**vars(args), "seed": 4}))
            edited = json.loads((root / batch.PLAN_NAME).read_text(encoding="utf-8"))
            edited["cases"][0]["alpha"] = 9.0
            (root / batch.PLAN_NAME).write_text(json.dumps(edited), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "edited"):
                batch.load_or_write_plan(root, Namespace(manifest=None, design_roots=None, record_set=None, seed=None))
            # shards partition the plan: every case in exactly one
            shards = [batch.select_cases(plan, Namespace(case_ids=None, shard=k, of=4)) for k in (1, 2, 3, 4)]
            self.assertEqual(sorted(sum(shards, [])), sorted(row["case"]["case_id"] for row in plan["cases"]))
            self.assertEqual([len(s) for s in shards], [25, 25, 25, 25])

    def test_a_case_is_claimed_once_and_its_state_is_read_from_its_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(batch.case_state(root, "case_0001"), "open")
            self.assertTrue(batch.claim(root / "case_0001"))
            self.assertFalse(batch.claim(root / "case_0001"))                                 # a second launch, here or elsewhere
            self.assertEqual(batch.case_state(root, "case_0001"), "claimed")                 # running or interrupted: not rerun
            (root / "case_0001" / batch.STATUS_NAME).write_text(json.dumps({"state": "analysis_failed"}), encoding="utf-8")
            self.assertEqual(batch.case_state(root, "case_0001"), "final:analysis_failed")

    def test_design_use_policy(self):
        open_only = batch.design_state({"status": "designed", "fail_ids": [], "not_evaluated_ids": ["demands.torsional_irregularity"]})
        self.assertEqual(batch.design_usable(open_only), (True, ""))                          # open checks are recorded, not asserted
        self.assertEqual(open_only["open_check_ids"], ["demands.torsional_irregularity"])
        failed = batch.design_state({"status": "designed", "fail_ids": ["joint.shear_screen"], "not_evaluated_ids": []})
        self.assertFalse(batch.design_usable(failed)[0])
        self.assertTrue(batch.design_usable(failed, include_failed=True)[0])
        self.assertFalse(batch.design_usable(batch.design_state({"status": "failed"}))[0])


if __name__ == "__main__":
    unittest.main()
