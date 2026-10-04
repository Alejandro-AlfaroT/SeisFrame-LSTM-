"""Bounded, non-monotone search bookkeeping for uniform frame designs.

All proposals require a fresh full-frame evaluation. Quantities are modeled
material quantities, not costs.

Selection since 2026-10-03 (user decision: strength before geometry, large sections the last resort): among
passing designs the smallest column wins, then the smallest beam, then the lower concrete grades. The earlier
rule adopted a candidate only on material dominance with no concrete grade raised, so a smaller column at a
higher grade could never replace a larger one at a lower grade (the 3 October screens kept 40-in columns at
4 ksi beside passing 34-in columns at 8 ksi). ``dominates`` is kept and reported for each feasible trial.
"""
from __future__ import annotations
import math

POLICY = "uniform_size_first_v2"
METRICS = ("concrete_total_in3", "total_steel_in3", "column_fc_ksi", "beam_fc_ksi",
           "distinct_form_size_count")


def feasibility_priority(worst, hard_max, drift_accepted, failed_checks, scwb_accepted):
    """Explore the most promising evaluated parent first; ties preserve proposal order.

    Joint alternatives from a beam at DCR 5 must not starve the demand-directed
    path once it has reached a beam near DCR 1. This orders work only, never
    accepts an unevaluated proposal or prunes a parent on a strength estimate.
    """
    return (max(0.0, max(worst["beam"], worst["column"]) - hard_max),
            int(not drift_accepted), len(failed_checks) + int(not scwb_accepted))


def targeted_alternatives(columns, beams, ci, bi, reasons, failed_checks=()):
    """Independent moves from the failed parent, not successive irreversible growth."""
    c, b = columns[ci], beams[bi]
    reasons = set(reasons)
    checks = {item["id"] for item in failed_checks}
    joint = bool(reasons & {"joint_shear_or_anchorage", "joint_scwb", "scwb_screen"})
    geometry = any("bar_stacking" in k or "thread" in k for k in checks)
    alternatives = []
    if joint:
        stronger = sorted((i for i, r in enumerate(columns) if r[:2] == c[:2] and r[2] > c[2]),
                          key=lambda i: columns[i][2])
        if stronger:
            alternatives.append((stronger[0], bi, "same column size, next concrete grade"))
    if joint or geometry:
        wider = sorted((i for i, r in enumerate(beams) if r[1:] == b[1:] and r[0] > b[0]),
                       key=lambda i: beams[i][0])
        if wider:
            alternatives.append((ci, wider[0], "same column and beam depth, wider beam for cage/confinement"))
    if geometry:
        deeper = sorted((i for i, r in enumerate(beams) if r[0] >= b[0] and r[1] > b[1] and r[2] == b[2]),
                        key=lambda i: (beams[i][1], beams[i][0]))
        if deeper:
            alternatives.append((ci, deeper[0], "deeper beam for unresolved bar stacking"))
    return alternatives


def dominates(candidate, incumbent):
    """None means quantities are unavailable, not zero."""
    if any(candidate.get(k) is None or incumbent.get(k) is None
           or not math.isfinite(candidate[k]) or not math.isfinite(incumbent[k]) for k in METRICS):
        return None
    pairs = [(candidate[k], incumbent[k]) for k in METRICS]
    return all(a <= b + 1e-9 for a, b in pairs) and any(a < b - 1e-9 for a, b in pairs)


SIZE_FAILURES_TO_STOP = 3
CONFINING_WIDTH_RATIO = 0.75                 # ACI 318-19 15.2.8(a): transverse beams at least 3/4 of the column face


def confining_beam_retry(columns, beams, column_index, beam_index, constraints):
    """The wider beam to retry a failed size trial with, or None.

    Applies when joint shear is what the trial failed, alone or with the beam bar layering check (strength,
    drift and the strong-column check hold, no other capacity check failed), and the beam is narrower than
    three quarters of the column: the first
    rung of the same depth and grade at least that wide confines the joint (Table 18.8.4.3, gamma up a third).
    """
    if not isinstance(constraints, dict):
        return None
    failed = {item.get("id") for item in constraints.get("capacity_design_failed_checks") or []}
    # Joint shear, alone or with the beam bar layering against the slab mats: a wider beam gives the bars more
    # lanes per layer as well as confining the joint (r5 trials at 34 in failed both together).
    if "joint.shear_screen" not in failed or not failed <= {"joint.shear_screen", "beam.bar_stacking_clear_of_slab_mats"}:
        return None
    if not (constraints.get("strength_within_ceiling") and constraints.get("drift_accepted")
            and (constraints.get("joint_scwb") or {}).get("all_pass") is not False):
        return None
    b, h, fc = beams[beam_index]
    target = CONFINING_WIDTH_RATIO * columns[column_index][0]
    if b >= target - 1e-9:
        return None
    wider = sorted((i for i, r in enumerate(beams) if r[1] == h and r[2] == fc and r[0] >= target - 1e-9),
                   key=lambda i: beams[i][0])
    return wider[0] if wider else None


def preference(column, beam):
    """Sort key of a passing design; smaller is preferred. Column area, beam area, then the concrete grades."""
    return (column[0] * column[1], beam[0] * beam[1], column[2], beam[2])


def _plan(search, columns):
    """Queue the next trial (at most one is pending), or none when the descent is complete.

    Size phase: the column sizes below the first passing one, nearest first, each at the top concrete grade
    with the beam held. A failure prunes nothing (feasibility is not monotone in size: a smaller island can
    pass where a larger size failed), but SIZE_FAILURES_TO_STOP consecutive failures end the phase. Grade
    phase: at the smallest passing size, the grades below the one it passed at, highest first, until one fails
    (every capacity falls with the grade, so the first failure ends it).
    """
    search["pending"] = []
    if search.get("retry") is not None:
        search["pending"] = [search["retry"]]
        return
    if search["phase"] == "size":
        if search["sizes"] and search["size_failures"] < SIZE_FAILURES_TO_STOP:
            size = search["sizes"][0]
            search["pending"] = [(columns.index((size, size, search["grades"][-1])), search["beam_held"])]
            return
        search["phase"] = "grade"
        search["grade_queue"] = sorted((g for g in search["grades"] if g < search["best"][1]), reverse=True)
    if search["grade_queue"]:
        size = search["best"][0]
        search["pending"] = [(columns.index((size, size, search["grade_queue"][0])), search["beam_held"])]


def start(columns, beams, column_index, beam_index, snapshot, budget):
    if type(budget) is not int or budget < 0:
        raise ValueError("uniform_reduction_trials must be a nonnegative integer")
    col = columns[column_index]
    search = {"budget": budget, "beam": beam_index, "beam_held": beam_index, "accepted": snapshot,
              "first_passing_column": list(col), "first_passing_beam": list(beams[beam_index]), "trials": [],
              "phase": "size", "grades": sorted({c[2] for c in columns}),
              "sizes": sorted({c[0] for c in columns if c[0] < col[0]}, reverse=True),
              "size_failures": 0, "best": (col[0], col[2]), "grade_queue": [], "retry": None,
              "feasible": [{"column": list(col), "beam": list(beams[beam_index]),
                            "quantities": snapshot["entry"].get("quantities", {}), "decision": "incumbent"}]}
    _plan(search, columns)
    return search


def next_pair(search):
    if len(search["trials"]) >= search["budget"] or not search["pending"]:
        return None
    column, search["beam"] = search["pending"][0]
    return column


def advance(search, columns, beams, column_index, accepted, note=None, snapshot=None,
            constraints=None, outcome=None, actual_beam_index=None):
    requested = search["pending"].pop(0)
    beam = search["beam"] if actual_beam_index is None else actual_beam_index
    column = columns[column_index]
    trial = {"column": list(column), "beam": list(beams[beam]), "phase": search["phase"],
             "requested_pair": list(requested), "accepted": bool(accepted), "note": note,
             "outcome": outcome or ("feasible" if accepted else "infeasible"), "constraints": constraints}
    if accepted:
        if snapshot is None:
            raise ValueError("A feasible trial requires its complete installed-state snapshot")
        q = snapshot["entry"].get("quantities", {})
        incumbent = search["accepted"]["entry"]
        preferred = preference(column, beams[beam]) < preference(incumbent["column_section"], incumbent["beam_section"])
        trial.update(quantities=q, material_dominance=dominates(q, incumbent.get("quantities", {})),
                     decision="adopted" if preferred else "retained_alternative")
        if preferred:
            search["accepted"] = snapshot
        search["feasible"].append({k: trial[k] for k in ("column", "beam", "quantities", "decision")})
    if search["phase"] == "size":
        was_retry = search["retry"] is not None
        search["retry"] = None
        wider = None if accepted or was_retry else confining_beam_retry(columns, beams, column_index, beam, constraints)
        if wider is not None:
            # Joint shear alone, under a beam too narrow to confine the joint: the same column once more with
            # the beam at three quarters of it, before the size counts as failed.
            trial["retry_with_wider_beam"] = list(beams[wider])
            search["retry"] = (column_index, wider)
        else:
            search["sizes"].pop(0)
            if accepted:
                search["size_failures"], search["best"] = 0, (column[0], column[2])
                if was_retry:
                    search["beam_held"] = beam            # the confining width is kept for the smaller sizes
            else:
                search["size_failures"] += 1
    else:
        search["grade_queue"].pop(0)
        if accepted:
            search["best"] = (column[0], column[2])
        else:
            search["grade_queue"] = []                     # a lower grade only lowers every capacity
    search["trials"].append(trial)
    _plan(search, columns)
    return next_pair(search)


def summary(search, columns, beams):
    held = list(beams[search["beam_held"]])
    untested = ([{"column": [size, size, search["grades"][-1]], "beam": held} for size in search["sizes"]]
                if search["phase"] == "size" else
                [{"column": [search["best"][0], search["best"][0], grade], "beam": held} for grade in search["grade_queue"]])
    budget_hit = len(search["trials"]) >= search["budget"] and bool(search["pending"])
    return {"policy": POLICY, "first_passing_column": search["first_passing_column"],
            "first_passing_beam": search["first_passing_beam"],
            "selected_column": search["accepted"]["entry"]["column_section"],
            "selected_beam": search["accepted"]["entry"]["beam_section"],
            "trials": search["trials"], "feasible_alternatives": search["feasible"],
            "trial_budget": search["budget"], "untested_count": len(untested), "untested": untested,
            "stop_reason": ("trial_budget_exhausted" if budget_hit else
                            "size_descent_stopped_after_consecutive_failures" if untested else "declared_descent_complete"),
            "basis": "full evaluations, beam held: column sizes below the first passing one at the top concrete grade, "
                     f"nearest first, ended by {SIZE_FAILURES_TO_STOP} consecutive failures (no monotonicity assumed "
                     "before that); a size that fails joint shear alone under a beam narrower than 3/4 of the column is "
                     "retried once with the beam widened to that width; then lower grades at the smallest passing size "
                     "until one fails; the smallest column is selected, then the lowest passing grade; no global-minimum claim"}
