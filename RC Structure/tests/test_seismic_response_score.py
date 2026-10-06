"""Hand-calculated calibration scores; no structural analyses are executed."""
import math
from pathlib import Path
import sys
import unittest


RC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_DIR))
from Data_Generation.Seismic_Response_Score import (  # noqa: E402
    METRIC_NAMES, default_policy, score_response, validate_policy,
)


def response(yielded=0.10, damage=0.25, interstory=0.0125, roof=0.015):
    return dict(zip(METRIC_NAMES, (yielded, damage, interstory, roof)))


class ResponseScore(unittest.TestCase):
    def test_all_references_reached_is_exactly_target(self):
        result = score_response(response())
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["target_reached"])
        self.assertTrue(result["all_references_reached"])
        self.assertEqual([row["contribution"] for row in result["components"].values()],
                         [0.45, 0.35, 0.15, 0.05])

    def test_half_references_do_not_reach_target(self):
        result = score_response(response(0.05, 0.125, 0.00625, 0.0075))
        self.assertAlmostEqual(result["score"], 0.5)
        self.assertFalse(result["target_reached"])

    def test_yielding_can_compensate_for_other_metrics_below_reference(self):
        # Ratios [2, .2, .4, .2]: .90 + .07 + .06 + .01 = 1.04.
        result = score_response(response(0.20, 0.05, 0.005, 0.003))
        self.assertAlmostEqual(result["score"], 1.04)
        self.assertTrue(result["target_reached"])
        self.assertFalse(result["all_references_reached"])
        self.assertFalse(result["components"]["roof_drift_ratio"]["reference_reached"])

    def test_drift_references_are_not_caps(self):
        # Ratios [.5, .5, 2, 3]: .225 + .175 + .30 + .15 = .85.
        result = score_response(response(0.05, 0.125, 0.025, 0.045))
        self.assertAlmostEqual(result["score"], 0.85)
        self.assertAlmostEqual(result["components"]["roof_drift_ratio"]["ratio"], 3.0)
        self.assertAlmostEqual(result["components"]["peak_interstory_drift_ratio"]["contribution"], 0.30)

    def test_uncapped_policy_makes_single_metric_dominance_visible(self):
        # This is the declared tradeoff, not an implicit local-distribution gate.
        result = score_response(response(0.0, 1.0, 0.0, 0.0))
        self.assertAlmostEqual(result["score"], 1.4)
        self.assertTrue(result["target_reached"])
        self.assertFalse(result["all_references_reached"])

    def test_zero_response_is_valid_but_not_target(self):
        result = score_response(response(0, 0, 0, 0))
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["target_reached"])

    def test_missing_and_extra_metrics_are_rejected(self):
        missing = response()
        del missing["roof_drift_ratio"]
        for metrics in (None, missing, {**response(), "unknown": 1}):
            with self.subTest(metrics=metrics), self.assertRaises(ValueError):
                score_response(metrics)

    def test_invalid_metric_values_are_rejected(self):
        for name in METRIC_NAMES:
            for invalid in (None, math.nan, math.inf, -math.inf, -0.1, True, "0.1"):
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    score_response({**response(), name: invalid})
        with self.assertRaises(ValueError):
            score_response(response(yielded=1.0001))

    def test_invalid_policy_values_and_keys_are_rejected(self):
        bad_policies = []
        for invalid in (-0.1, math.nan, math.inf, None, True):
            policy = default_policy()
            policy["weights"]["roof_drift_ratio"] = invalid
            bad_policies.append(policy)
        for invalid in (0, -0.01, math.nan, math.inf, None, True):
            policy = default_policy()
            policy["references"]["roof_drift_ratio"] = invalid
            bad_policies.append(policy)
        for field, invalid in (("target_score", 0), ("target_score", math.inf),
                               ("schema_version", "unknown"), ("aggregation", "capped")):
            policy = default_policy()
            policy[field] = invalid
            bad_policies.append(policy)
        policy = default_policy()
        policy["weights"]["roof_drift_ratio"] = 0.10  # Sum 1.05; do not renormalize.
        bad_policies.append(policy)
        for field in ("weights", "references"):
            policy = default_policy()
            del policy[field]["roof_drift_ratio"]
            bad_policies.append(policy)
        for policy in bad_policies:
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                validate_policy(policy)

    def test_valid_custom_policy_is_saved_without_mutating_caller(self):
        policy = default_policy()
        policy["weights"] = dict(zip(METRIC_NAMES, (0.5, 0.5, 0.0, 0.0)))
        policy["target_score"] = 0.75
        result = score_response(response(0.10, 0.125, 0, 0), policy)
        self.assertAlmostEqual(result["score"], 0.75)
        self.assertTrue(result["target_reached"])
        result["policy"]["weights"]["fraction_hinges_yielded"] = 99
        self.assertEqual(policy["weights"]["fraction_hinges_yielded"], 0.5)
        policy["references"]["roof_drift_ratio"] = 99
        self.assertEqual(default_policy()["references"]["roof_drift_ratio"], 0.015)

    def test_nonfinite_normalized_result_is_rejected(self):
        policy = default_policy()
        policy["references"]["peak_hinge_damage_ratio"] = 1e-300
        with self.assertRaises(ValueError):
            score_response(response(damage=1e300), policy)


if __name__ == "__main__":
    unittest.main()
