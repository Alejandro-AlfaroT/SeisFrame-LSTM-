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


def awaiting_review(check_id, clause, reason, item, location=""):
    """A not_evaluated row that waits on an engineering review (``item``), not on a missing calculation.

    The distinction is carried as ``details.review_dependency`` so a consumer can keep review-dependent items
    apart from unknown computed evidence; it never turns the row into a pass.
    """
    row = not_evaluated(check_id, clause, reason, location)
    row["details"]["review_dependency"] = {"item": item, "kind": "engineering_review"}
    return row


# Items the research scope excludes by a declared decision: their rows are neither checked nor passed. Each is keyed
# by the row id it is reported under (at the empty location); consumers read this table, never a row's own label.
SCOPE_EXCLUSIONS = {
    "fire_resistance": ("fire-resistance design (ACI 318-19 4.11 and the applicable building code: the required ratings "
                        "and the cover and thickness they need) is excluded from the capstone's seismic research scope "
                        "by the user's decision of 2026-11-01; it is not checked and not passed"),
}
SCOPE_EXCLUDED_ROWS = {"slab_fire_resistance": "fire_resistance"}


def scope_exclusion(check_id, clause, reason, item, location=""):
    """A not_evaluated row of an item the declared scope excludes (SCOPE_EXCLUSIONS); never a pass.

    The label (``details.scope_exclusion``) only names the item. A consumer honours it where its own declared
    scope lists that row (Design.Code_Check_Contract.expected_inventory, split_scope_exclusions); elsewhere the
    row is an ordinary unknown.
    """
    if item not in SCOPE_EXCLUSIONS:
        raise ValueError(f"{item!r} is not a declared scope exclusion")
    row = not_evaluated(check_id, clause, reason, location)
    row["details"]["scope_exclusion"] = {"item": item, "status": "excluded: not checked, not passed"}
    return row


def split_scope_exclusions(checks):
    """(kept, excluded) of a qualification checklist.

    A row of a declared excluded item (SCOPE_EXCLUDED_ROWS) that is not_evaluated, waits on no review and is
    labelled with that item moves to ``excluded`` with the declared basis. A pass, fail or review row for such an
    item is replaced by a not_evaluated row: the item is not checked within this scope, so the row blocks
    acceptance instead of standing as evidence. Every other row is kept as given.
    """
    kept, excluded = [], []
    for check in checks:
        item = SCOPE_EXCLUDED_ROWS.get(check.get("id")) if isinstance(check, dict) and not check.get("location") else None
        if item is None:
            kept.append(check)
            continue
        details = check.get("details") if isinstance(check.get("details"), dict) else {}
        label = details.get("scope_exclusion")
        if (check.get("status") == "not_evaluated" and not details.get("review_dependency")
                and isinstance(label, dict) and label.get("item") == item):
            excluded.append({"id": check["id"], "item": item, "clause": check.get("clause"),
                             "status": "excluded: not checked, not passed", "basis": SCOPE_EXCLUSIONS[item]})
        else:
            kept.append(not_evaluated(check.get("id"), check.get("clause", ""),
                                      f"{item} is excluded from the declared scope: a {check.get('status')} row for it "
                                      "cannot stand as evidence (not checked, not passed)"))
    return kept, excluded


def _input_kind(value):
    """What an input that is not a finite number actually was (kept, so missing and invalid stay distinguishable)."""
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "boolean"
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "non_numeric"
    return "nan" if math.isnan(number) else "infinite"


def make_check(check_id, clause, demand, capacity, comparison="<=", units="",
               location="", details=None):
    if comparison not in ("<=", ">=", "=="):
        raise ValueError(f"Unsupported comparison: {comparison}")
    a, b = _finite(demand), _finite(capacity)
    if a is None or b is None:
        row = not_evaluated(check_id, clause, "Finite numeric inputs are required.", location)
        row["details"]["input_problems"] = {name: _input_kind(value) for name, value, parsed in
                                            (("demand", demand, a), ("capacity", capacity, b)) if parsed is None}
        return row
    passed = {"<=": a <= b, ">=": a >= b, "==": a == b}[comparison]
    return {"id": check_id, "clause": clause, "status": "pass" if passed else "fail",
            "demand": a, "capacity": b, "comparison": comparison, "units": units,
            "location": location, "details": details or {}}


TOLERANCE_KEYS = ("numerical_tolerance_in", "numerical_tolerance")


def check_row_problems(check):
    """Why one checklist row cannot be read as the evidence it claims; empty when it can.

    A pass or fail row must carry finite demand and capacity, a supported comparison, and a status
    that follows from them (a tolerance the producer declared in ``details`` under TOLERANCE_KEYS is
    honoured, as SMRF_Beam_Spacing applies one). A not_evaluated row carries no comparison outcome.
    This reads one row; whether the row belongs to a required inventory is a separate question
    (Design.Code_Check_Contract).
    """
    if not isinstance(check, dict):
        return ["not_a_record"]
    problems = []
    if not isinstance(check.get("id"), str) or not check["id"]:
        problems.append("missing_id")
    if not isinstance(check.get("clause"), str) or not check["clause"]:
        problems.append("missing_clause")
    if not isinstance(check.get("location", ""), str):
        problems.append("location_not_text")
    status = check.get("status")
    if status not in ("pass", "fail", "not_evaluated"):
        return problems + ["unknown_status"]
    if status == "not_evaluated":
        return problems
    demand, capacity, comparison = _finite(check.get("demand")), _finite(check.get("capacity")), check.get("comparison")
    if demand is None or capacity is None:
        return problems + ["status_without_finite_numbers"]
    if comparison not in ("<=", ">=", "=="):
        return problems + ["unsupported_comparison"]
    details = check.get("details") if isinstance(check.get("details"), dict) else {}
    tolerance = next((details[key] for key in TOLERANCE_KEYS if key in details), 0.0)
    tolerance = _finite(tolerance)
    if tolerance is None or tolerance < 0.0:
        return problems + ["invalid_declared_tolerance"]
    strict = {"<=": demand <= capacity, ">=": demand >= capacity, "==": demand == capacity}[comparison]
    relaxed = {"<=": demand <= capacity + tolerance, ">=": demand >= capacity - tolerance,
               "==": abs(demand - capacity) <= tolerance}[comparison]
    if (status == "pass" and not relaxed) or (status == "fail" and strict):
        problems.append("status_contradicts_numbers")
    return problems


def action_inventory_problems(actions, expected_ids, member_tags=(), keys=("axial_i_kip", "axial_j_kip"),
                              axial_reference="joint_faces"):
    """({id: action}, problems) of a solved factored-action inventory, read the way SMRF_Joint_Adapter reads it.

    Every expected combination id exactly once, nothing unexpected, ``analysis_succeeded`` True, the declared
    ``axial_reference``, and a finite value under each of ``keys`` for every member in ``member_tags``. A
    duplicate id is reported, never overwritten; an action that is not accepted is left out of the mapping.
    """
    expected = list(expected_ids or [])
    problems, by_id = [], {}
    if not expected or len(set(expected)) != len(expected) or any(not isinstance(e, str) or not e for e in expected):
        problems.append("the expected combination ids are not a unique nonempty list")
    if not isinstance(actions, (list, tuple)):
        return {}, problems + ["the solved actions are not a list"]
    for item in actions:
        cid = item.get("id") if isinstance(item, dict) else None
        if cid not in expected:
            problems.append(f"unexpected or malformed combination {cid!r}")
            continue
        if cid in by_id:
            problems.append(f"duplicate combination {cid}")
            continue
        issues = []
        if item.get("analysis_succeeded") is not True:
            issues.append(f"combination {cid} did not confirm a successful analysis")
        if axial_reference is not None and item.get("axial_reference") != axial_reference:
            issues.append(f"combination {cid} does not identify axial_reference={axial_reference!r}")
        members = item.get("members")
        if not isinstance(members, dict):
            issues.append(f"combination {cid} has no member actions")
        else:
            for tag in member_tags:
                member = members.get(str(tag))
                if not isinstance(member, dict) or any(_finite(member.get(key)) is None for key in keys):
                    issues.append(f"combination {cid}, member {tag}: missing or nonfinite {'/'.join(keys)}")
        problems.extend(issues)
        if not issues:
            by_id[cid] = item
    problems.extend(f"missing combination {cid}" for cid in expected if cid not in by_id
                    and not any(p.startswith(f"combination {cid}") for p in problems))
    return by_id, problems


def summarize_checks(checks):
    """Summarize the rows supplied; a list that is empty, has a duplicate (id, location), or holds a malformed or
    self-contradictory row is not accepted.

    This only reads the rows it is given. It cannot know that a required check was left out of a nonempty
    list: required coverage is checked against an inventory derived from the design itself
    (Design.Code_Check_Contract), never from the list under review.
    """
    checks = list(checks)
    keys = [(c.get("id"), c.get("location", "")) if isinstance(c, dict) else (None, None) for c in checks]
    duplicate = len(keys) != len(set(keys))
    counts = {state: sum(isinstance(c, dict) and c.get("status") == state for c in checks)
              for state in ("pass", "fail", "not_evaluated")}
    problems = {index: check_row_problems(c) for index, c in enumerate(checks)}
    malformed = any(problems.values())
    contradictory = [c.get("id") for index, c in enumerate(checks) if "status_contradicts_numbers" in problems[index]]
    return {"accepted": bool(checks) and not duplicate and not malformed and
            counts["pass"] == len(checks), "counts": counts,
            "duplicate_check_ids": duplicate, "malformed_checks": malformed,
            "contradictory_checks": contradictory,
            "failed": [c.get("id") for c in checks if isinstance(c, dict) and c.get("status") == "fail"],
            "not_evaluated": [c.get("id") for c in checks
                              if isinstance(c, dict) and c.get("status") == "not_evaluated"]}
