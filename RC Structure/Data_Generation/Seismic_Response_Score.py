"""Pure response scoring for seismic intensity calibration.

The drift and hinge values are references, not individual acceptance gates or
upper limits. Ratios are deliberately uncapped: excess response in one metric
can compensate for another below its reference. Consequently this score does
not require all four references to be attained and a sufficiently large single
metric can dominate it. Raw components remain visible for diagnostic review.

This objective does not establish analysis completion, design qualification,
training eligibility, cycle counts, frequency coverage, or yielding at every
story. Those checks belong to the caller. It does not replace ``is_inelastic``.
No OpenSees or numerical-analysis dependencies are needed to evaluate it.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real


# v2 (frozen IMK V2 calibration, 2026-10-29): both hinge metrics measure each axis and native sign on the end's own
# spring (Data_Generation/Hybrid_Exporter.DAMAGE_METRIC_DEFINITIONS). peak_hinge_damage_ratio is the native capping
# ratio, plastic native spring rotation of the governing branch over its native dp; fraction_hinges_yielded is the
# fraction of hinge ends with some native branch past its own My/Kh. v1 took max |rotation| of both springs less the
# larger branch-minimum yield, over the member theta_p. Weights, references and aggregation are unchanged, but the
# values differ from v1 even under the legacy replay (native dp = theta_p there, so only the denominator agrees), and
# a v1 policy or score is not read as a v2 one.
POLICY_SCHEMA_VERSION = "seisframe_seismic_response_score_policy_v2"
SCORE_SCHEMA_VERSION = "seisframe_seismic_response_score_v2"
AGGREGATION = "weighted_reference_ratios_uncapped"
METRIC_NAMES = (
    "fraction_hinges_yielded",
    "peak_hinge_damage_ratio",
    "peak_interstory_drift_ratio",
    "roof_drift_ratio",
)

DEFAULT_POLICY = {
    "schema_version": POLICY_SCHEMA_VERSION,
    "aggregation": AGGREGATION,
    "target_score": 1.0,
    "weights": dict(zip(METRIC_NAMES, (0.45, 0.35, 0.15, 0.05))),
    "references": dict(zip(METRIC_NAMES, (0.10, 0.25, 0.0125, 0.015))),
}


def default_policy():
    """Return an independent copy of the declared default calibration policy."""
    return deepcopy(DEFAULT_POLICY)


def _exact_keys(value, expected, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly {tuple(expected)!r}")


def _finite_number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    if number < 0 or (positive and number == 0):
        condition = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {condition}")
    return number


def validate_policy(policy):
    """Validate and copy a policy without silently changing its weights.

    Weights must sum to one within floating-point roundoff. Custom weights,
    references and targets are supported but must be saved with their scores;
    changing the schema or aggregation is not supported by this evaluator.
    """
    _exact_keys(policy, DEFAULT_POLICY, "policy")
    if policy["schema_version"] != POLICY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported score policy schema: {policy['schema_version']!r}")
    if policy["aggregation"] != AGGREGATION:
        raise ValueError(f"Unsupported score aggregation: {policy['aggregation']!r}")
    _exact_keys(policy["weights"], METRIC_NAMES, "weights")
    _exact_keys(policy["references"], METRIC_NAMES, "references")
    weights = {name: _finite_number(policy["weights"][name], f"weight {name}")
               for name in METRIC_NAMES}
    if any(value > 1 for value in weights.values()) or not math.isclose(
            math.fsum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Score weights must sum to 1.0")
    references = {
        name: _finite_number(policy["references"][name], f"reference {name}", positive=True)
        for name in METRIC_NAMES
    }
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "aggregation": AGGREGATION,
        "target_score": _finite_number(policy["target_score"], "target_score", positive=True),
        "weights": weights,
        "references": references,
    }


def score_response(metrics, policy=None):
    """Return an uncapped weighted score and its complete component accounting.

    ``metrics`` must contain exactly ``METRIC_NAMES``. Drift ratios and yielded
    fraction are fractions (0.0125 means 1.25%); the yielded fraction counts hinge
    ends with some native branch past its own My/Kh, and damage ratio is the
    dimensionless native capping ratio (1 at the native capping point).
    Missing, nonfinite or negative data are rejected, never treated as zero.
    Completion/convergence must be checked independently before accepting a run.
    """
    _exact_keys(metrics, METRIC_NAMES, "metrics")
    selected = validate_policy(default_policy() if policy is None else policy)
    components = {}
    for name in METRIC_NAMES:
        value = _finite_number(metrics[name], name)
        if name == "fraction_hinges_yielded" and value > 1:
            raise ValueError("fraction_hinges_yielded must lie in [0, 1]")
        reference = selected["references"][name]
        weight = selected["weights"][name]
        ratio = value / reference
        if not math.isfinite(ratio):
            raise ValueError(f"Normalized ratio for {name} is not finite")
        components[name] = {
            "value": value,
            "reference": reference,
            "weight": weight,
            "ratio": ratio,
            "contribution": weight * ratio,
            "reference_reached": value >= reference,
        }
    score = math.fsum(component["contribution"] for component in components.values())
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "policy": selected,
        "score": score,
        "target_reached": score >= selected["target_score"],
        "all_references_reached": all(component["reference_reached"]
                                      for component in components.values()),
        "components": components,
    }
