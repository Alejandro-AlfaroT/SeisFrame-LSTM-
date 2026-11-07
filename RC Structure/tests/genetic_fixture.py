"""Synthetic, schema-faithful evaluation records for the genetic-search tests (2026-10-30).

Contract-valid code-check rows (contract_fixture) with the evidence blocks the diagnostic adapter reads: capacity
(joint types and strong-column joint records from the production topology helper Group_Capacity.joint_inventory on
the fixture frame), member strengths, factored-action inventory and the demand basis. Every number is synthetic:
these records exercise the adapter, the operators, the selection and the controller, not any structural check.
"""
import contract_fixture as cf
from Design import Code_Check_Contract as contract
from Design import Group_Capacity as gc
from Design import Grouped_Record as gr
from Design import Grouped_Search as gs
from Design.SMRF_Common import make_check, not_evaluated
from Model import Member_Groups as mg

POLICY = "system_three_story_v1"
COMBINATIONS = ("fixture_c1", "fixture_c2")
COLUMN_MN, BEAM_MN = 4000.0, 1000.0


def seed_state():
    common = set(gs.beam_rungs("beam_x")) & set(gs.beam_rungs("beam_y"))
    beam = min(common, key=lambda r: abs(r[0] - 16) + abs(r[1] - 30))
    state, _ = gr.seed_from_sections((26, 26, 6), (*beam, 5), policy_id=POLICY)
    return state


def capacity_evidence(state):
    """Joint types and strong-column joint records shaped as Group_Capacity writes them (synthetic strengths)."""
    with mg.installed(state):
        inventory = gc.joint_inventory(state)
        types, joints = {}, []
        for type_id, kind in sorted(inventory["types"].items()):
            entry = kind["representative"]
            axes = [a for a in ("x", "y") if entry["beams"][a]]
            types[type_id] = {
                "type_id": type_id, "joints": list(kind["joints"]), "core_group": entry["core"].group_id,
                "column_above_group": None if entry["above"] is None else entry["above"].group_id,
                "anchorage": {a: {"through_bars_present": len(entry["beams"][a]) == 2 and not entry["mixed"][a]} for a in axes},
                "threading": {a: {"beam_group": sorted({b.group_id for b in entry["beams"][a]})[0],
                                  "core_group": entry["core"].group_id} for a in axes},
                "shear": {a: {"vj_kip": 100.0, "phi_vn_kip": 200.0, "capacity_topology_and_confinement_checked": True}
                          for a in axes},
                "slab_mats": {"clashes": [], "tight_crossings": [], "passes": True, "slab_mats_placed": True}}
        for entry in inventory["joints"]:
            columns = [entry["below"]] + ([entry["above"]] if entry["above"] is not None else [])
            directions = {}
            for axis in ("x", "y"):
                caps = [{"group_id": c.group_id, "tag": c.member_tag, "mn_kip_in": COLUMN_MN, "axial_envelope_checked": True,
                         "combination_count": len(COMBINATIONS), "factored_axial_kip": 100.0} for c in columns]
                beams = [{"group_id": b.group_id, "tag": b.member_tag, "mn_kip_in": BEAM_MN, "slab_basis": "no_slab",
                          "slab_mn_kip_in": 0.0}
                         for b in (entry["joint"][f"beam_{axis}_minus"], entry["joint"][f"beam_{axis}_plus"]) if b is not None]
                directions[axis] = {"required": gc.joint_rule(entry, axis, state) == gc.JOINT_RULE_SFRS,
                                    "positive": {"nominal_strengths": True, "column_capacities": caps, "beam_capacities": beams},
                                    "negative": {"nominal_strengths": True, "column_capacities": [dict(c) for c in caps],
                                                 "beam_capacities": [dict(b) for b in beams]}}
            joints.append({"id": entry["id"], "floor": entry["level"], "type_id": entry["type_id"],
                           "is_roof": entry["is_roof"], "directions": directions})
    beams = {gid: {"vs_required_kip": 50.0, "vs_limit_kip": 100.0, "governing_members": [{"governing_ve_kip": 60.0}]}
             for gid in sorted(state.groups) if not state.designs[gid].is_column}
    columns = {gid: {"governing": {"axis": "x", "ve_kip": 60.0, "vs_required_kip": 40.0, "ve_own_envelope_kip": 60.0,
                                   "vu_analysis_kip": 40.0, "method": "column_own_probable_envelope_v3"},
                     "vs_limit_kip": 100.0,
                     "hoops": {"phi_vn_kip": 90.0, "by_direction": {"x": {"phi_vn_kip": 120.0}, "y": {"phi_vn_kip": 90.0}}}}
               for gid in sorted(state.groups) if state.designs[gid].is_column}
    return {"joint_types": types, "scwb": {"joints": joints, "expected_combination_ids": list(COMBINATIONS),
                                           "action_inventory_complete": True}, "beams": beams, "columns": columns}


def strengths(state, overrides=None):
    """Member strengths of every group (passing unless ``overrides`` {gid: {component: dict}} says otherwise)."""
    groups = {}
    for gid, group in sorted(state.groups.items()):
        n = len(group["member_tags"]) * len(COMBINATIONS)
        tag = group["member_tags"][0]
        if state.designs[gid].is_column:
            result = {"member_type": "column",
                      "pm": {"dcr": 0.5, "basis": "biaxial_load_contour", "domain_failure": None, "member_tag": tag,
                             "combination": COMBINATIONS[0]},
                      "shear": {"dcr": 0.4, "vu_kip": 40.0, "phi_vn_kip": 100.0, "member_tag": tag,
                                "combination": COMBINATIONS[0]}, "observations": n}
        else:
            flexure = {"dcr": 0.5, "mu_kip_in": 500.0, "phi_mn_kip_in": 1000.0, "tension_controlled": True,
                       "member_tag": tag, "combination": COMBINATIONS[0]}
            result = {"member_type": group["member_type"], "flexure_positive": dict(flexure),
                      "flexure_negative": dict(flexure),
                      "shear": {"dcr": 0.4, "vu_kip": 40.0, "phi_vn_kip": 100.0, "member_tag": tag,
                                "combination": COMBINATIONS[0]}, "observations": n}
        for component, values in (overrides or {}).get(gid, {}).items():
            result[component] = {**result[component], **values}
        groups[gid] = result
    return {"groups": groups}


def actions():
    return [{"id": cid, "analysis_succeeded": True, "axial_reference": "joint_faces", "members": {}} for cid in COMBINATIONS]


def row(family, check_id, location, demand, capacity, comparison="<=", units="", details=None):
    return contract.compact_row(family, make_check(check_id, "fixture clause", demand, capacity, comparison, units,
                                                   location, details=details))


def unknown_row(family, check_id, location, reason_code):
    check = not_evaluated(check_id, "fixture clause", f"fixture: {reason_code}", location)
    check["details"]["reason_codes"] = [reason_code]
    return contract.compact_row(family, check)


def strength_row(gid, groups):
    result = groups[gid]
    names = ("pm", "shear") if result["member_type"] == "column" else ("flexure_positive", "flexure_negative", "shear")
    return row("strength", "member.strength", gid, max(result[n]["dcr"] for n in names), 1.0, "<=", "DCR")


def scwb_failure(evidence, joint_id, axis, sign, beam_mn=20000.0):
    """Raise the beam strengths of one joint direction so 1.2 sum Mnb exceeds sum Mnc; returns the matching row."""
    record = next(j for j in evidence["scwb"]["joints"] if j["id"] == joint_id)
    state = record["directions"][axis][sign]
    for beam in state["beam_capacities"]:
        beam["mn_kip_in"] = beam_mn
    columns = sum(c["mn_kip_in"] for c in state["column_capacities"])
    beams = sum(b["mn_kip_in"] + b["slab_mn_kip_in"] for b in state["beam_capacities"])
    return row("joints", "scwb", f"{joint_id}/{axis}/{sign}", 1.2 * beams, columns, "<=", "kip-in",
               details={"ratio_provided": columns / beams, "ratio_required": 1.2})


def record(state, cfg=None, *, replace=None, strength_overrides=None, evidence=None, constraints=None, quantities=None,
           seed_identity=None, demand_basis=None, drop=None):
    """A contract-valid evaluation record of ``state``: every required row passes unless ``replace`` {(family, id,
    location): compact row} supplies it; member.strength rows follow the strength evidence; ``drop(row)`` True
    leaves a row out (a required row the evaluator did not produce)."""
    groups = strengths(state, strength_overrides)["groups"]
    rows = cf.rows_for(state, cfg)
    replace = dict(replace or {})
    for gid in groups:
        replace.setdefault(("strength", "member.strength", gid), strength_row(gid, groups))
    rows = [replace.pop((r["family"], r["id"], r["location"]), r) for r in rows] + list(replace.values())
    if drop is not None:
        rows = [r for r in rows if not drop(r)]
    result = cf.evaluation_record(state, cfg, rows=rows, quantities=quantities, constraints=constraints,
                                  seed_identity=seed_identity, screen=True)
    result["evidence"].update(capacity=evidence or capacity_evidence(state), strengths={"groups": groups},
                              actions=actions(), expected_ids=list(COMBINATIONS),
                              demand_basis=demand_basis if demand_basis is not None else {"drift": {"limit_ratio": 0.02}})
    return result
