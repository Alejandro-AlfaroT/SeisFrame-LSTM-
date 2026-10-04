"""JSON-safe, fail-closed evidence records for the SMRF design review.

These records describe checks within an explicitly declared research scope;
they are not a building-code certification. Missing evidence is not success.
"""
from __future__ import annotations

import math
from datetime import date


class SectionAxialDomainError(ValueError):
    """An axial load (or range) lies outside what a section can carry: the section has no strength there.

    A ValueError, as before, so nothing that caught it changes. The grouped design search catches this
    class by name and records the candidate that produced it as infeasible; it never clamps the load and
    never treats another exception as this one.
    """


def assertion_provenance_valid(policy):
    """Require a named, dated basis; this does not authenticate/review it."""
    if not isinstance(policy, dict):
        return False
    if any(not isinstance(policy.get(key), str) or not policy[key].strip()
           for key in ("asserted_by", "assertion_date", "assertion_basis")):
        return False
    try:
        return date.fromisoformat(policy["assertion_date"]).isoformat() == policy["assertion_date"]
    except ValueError:
        return False


def _finite(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def not_evaluated(check_id, clause, reason, location=""):
    return {"id": check_id, "clause": clause, "status": "not_evaluated",
            "demand": None, "capacity": None, "units": "", "location": location,
            "details": {"reason": reason}}


def make_check(check_id, clause, demand, capacity, comparison="<=", units="",
               location="", details=None):
    if comparison not in ("<=", ">=", "=="):
        raise ValueError(f"Unsupported comparison: {comparison}")
    a, b = _finite(demand), _finite(capacity)
    if a is None or b is None:
        return not_evaluated(check_id, clause, "Finite numeric inputs are required.", location)
    passed = {"<=": a <= b, ">=": a >= b, "==": a == b}[comparison]
    return {"id": check_id, "clause": clause, "status": "pass" if passed else "fail",
            "demand": a, "capacity": b, "comparison": comparison, "units": units,
            "location": location, "details": details or {}}


def summarize_checks(checks):
    """An empty, duplicate, missing, or malformed checklist cannot be accepted."""
    checks = list(checks)
    keys = [(c.get("id"), c.get("location", "")) for c in checks]
    duplicate = len(keys) != len(set(keys))
    counts = {state: sum(c.get("status") == state for c in checks)
              for state in ("pass", "fail", "not_evaluated")}
    malformed = any(not c.get("id") or not c.get("clause") or
                    c.get("status") not in counts for c in checks)
    return {"accepted": bool(checks) and not duplicate and not malformed and
            counts["pass"] == len(checks), "counts": counts,
            "duplicate_check_ids": duplicate, "malformed_checks": malformed,
            "failed": [c.get("id") for c in checks if c.get("status") == "fail"],
            "not_evaluated": [c.get("id") for c in checks
                              if c.get("status") == "not_evaluated"]}
