"""Contract-valid evaluation records for search tests (Design.Code_Check_Contract, 2026-10-30).

A stub evaluation must meet the same contract a real one does: its code-check rows cover the inventory derived
from the design itself, every completion flag is an explicit boolean, and the rows are bound to the final state.
``rows_for`` writes one row per required check (passing unless told otherwise) and ``evaluation_record`` wraps
them in the record Grouped_Design.evaluate_candidate returns. The rows are synthetic: they exercise the search's
handling of the contract, not any structural check.
"""
import Structure_Parameters as sp
from Design import Code_Check_Contract as contract
from Design import Grouped_Design as gd
from Design.SMRF_Common import make_check, not_evaluated
from Model import Member_Groups as mg

COMPLETION = {flag: True for flag in contract.COMPLETION_FLAGS}
SLAB_RECORD = {"layout": {"bar_size": 4, "layers": {}}, "inputs": {"slab": {"thickness_in": 7.0}, "context": {"fixture": True}}}
SLAB_THICKNESS_IN = 7.0


def _row(family, name, location, status="pass", reason_code=None):
    if status == "pass":
        check = make_check(name, "fixture clause", 1, 1, "==", location=location)
    elif status == "fail":
        check = make_check(name, "fixture clause", 2.0, 1.0, "<=", location=location)
    else:
        check = not_evaluated(name, "fixture clause", f"fixture: {reason_code or 'unknown'}", location)
        if reason_code:
            check["details"]["reason_code"] = reason_code
    return contract.compact_row(family, check)


def rows_for(state, cfg=None, statuses=None, drop=(), extra=()):
    """One row per required check of ``state``: pass unless ``statuses`` {(family, id, location): status or
    (status, reason_code)} says otherwise; ``drop`` removes keys, ``extra`` appends compact rows."""
    statuses = statuses or {}
    expected = contract.expected_inventory(state, cfg)
    keys = set(expected["exact"])
    for rule in expected["conditional"]:
        keys |= {tuple(key) for key in rule["requires"]}
    rows = []
    for key in sorted(keys):
        if key in drop:
            continue
        status = statuses.get(key, "pass")
        status, code = status if isinstance(status, tuple) else (status, None)
        rows.append(_row(*key, status=status, reason_code=code))
    for pattern in expected["patterns"]:
        if "group" in pattern:
            if not any(r["family"] == pattern["family"] and (r["location"] == pattern["group"]
                                                              or r["location"].startswith(pattern["group"] + "/"))
                       for r in rows):
                rows.append(_row(pattern["family"], "fixture.detailing_rule", f"{pattern['group']}/fixture"))
        else:
            gid = pattern["prefix"].split("/member_")[0]
            tag = state.groups[gid]["member_tags"][0]
            for name in pattern["id_in"]:
                rows.append(_row(pattern["family"], name, f"{pattern['prefix']}{tag}{pattern['suffix']}"))
    return rows + list(extra)


def evaluation_record(state, cfg=None, rows=None, quantities=None, constraints=None, screen=True, label="fixture",
                      seed_identity=None):
    """The evaluation record of ``state`` with contract rows: ``feasible`` follows from the rows, the completion
    flags in ``constraints`` (all True by default) and the objectives, exactly as the evaluator computes it."""
    rows = rows_for(state, cfg) if rows is None else rows
    expected = contract.expected_inventory(state, cfg)
    result = contract.code_check_result(rows, expected)
    flags = {**COMPLETION, **(constraints or {})}
    quantities = quantities if quantities is not None else {"concrete_total_in3": 1000.0, "total_steel_in3": 100.0,
                                                            "frame_steel_in3": 90.0, "distinct_form_size_count": 4}
    identity = state.identity()
    completion = contract.numerical_completion(flags, identity)[2]
    feasible = contract.accepted_for_code_design(result, completion, contract.objective_availability(quantities))
    return {"version": gd.EVALUATION_VERSION, "label": label, "sections": gd._section_table(state), "stages": {},
            "solves": {}, "elapsed_seconds": 0.0, "status": "evaluated", "seed_identity": seed_identity or identity,
            "state": state, "result_identity": identity, "candidate_screen_passed": bool(screen),
            "constraints": {**flags, "candidate_screen_passed": bool(screen)}, "quantities": quantities,
            "margins": {}, "dcr": {"column": 0.5, "beam": 0.5}, "feasible": feasible, "code_check": result,
            "code_checks": {"contract_version": contract.CONTRACT_VERSION, "scope": contract.SCOPE["id"],
                            "state_identity": identity,
                            "realized_identity": contract.realized_identity(identity, SLAB_THICKNESS_IN, SLAB_RECORD),
                            "rows": rows},
            "nonlinear_performance": dict(contract.NONLINEAR_STATUS),
            "evidence": {"slab": {"slab": {"thickness_in": SLAB_THICKNESS_IN}, "reinforcement": SLAB_RECORD},
                         "capacity": {"joint_types": {}}}}


def installed_geometry_matches(state):
    """The fixture frame's geometry is the design's (expected_inventory refuses a mismatch)."""
    counts = state.counts
    return (sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR) == (counts["num_bay_x"], counts["num_bay_y"], counts["num_floor"])


def scwb_keys(state, cfg=None):
    """The required strong-column keys of ``state``."""
    return [key for key in sorted(contract.expected_inventory(state, cfg)["exact"]) if key[1] == "scwb"]


def any_gid(state, member_type="column", system=None):
    return next(gid for gid in sorted(state.groups) if (state.designs[gid].is_column == (member_type == "column"))
                and (system is None or state.system_of(gid) == system or not mg.is_system_policy(state.policy_id)))
