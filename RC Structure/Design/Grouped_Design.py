"""Evaluation of one grouped design candidate, its quantities, and its design record (2026-10-02).

User decision (grouped design, 2026-10-02): member design may vary by story
band and framing location. Model.Member_Groups holds what each group selected and resolves every member;
this module evaluates a complete candidate (a GroupedDesign) the way Design_Driver.design_structure
evaluates one uniform rung pair, with every step on the installed members:

  slab        one thickness passing every panel of every mechanically distinct floor (SMRF_Slab), one
              slab-to-frame transfer per distinct floor (SMRF_Floor_Transfer), strip demands of every
              distinct floor and one common layout designed on their envelope (SMRF_Slab_Refinement,
              SMRF_Slab_Reinforcement); a layout the strip ladder cannot supply moves the slab thickness
              up, never to a predefined slab;
  demands     the elastic design frame built from the resolver, the torsion assessment, every strength
              combination solved once and kept as signed concurrent member actions;
  bars        each group's own cage on its own demands (Group_Selection), hoops from the group capacity
              design (Group_Capacity), repeated until the installed cage stops changing;
  checks      member strength (Group_Checks), capacity design with joints and transitions, the joint
              evaluation (SMRF_Joints), story drift and stability, the demand basis with story-by-story
              strength, stiffness and mass evidence.

A candidate that cannot be evaluated for a named design reason is ``infeasible`` with the stage and the
evidence that says why: no slab thickness, a floor solve that does not balance, a gravity analysis that
does not converge, an axial load outside a section's strength domain (SectionAxialDomainError: the load
is never clamped), no admissible cage. Anything else is an error in the code or the inputs and is raised.

A candidate whose final shared cage arrangement leaves beam bar rows that do not fit, and whose bounded closure
(Group_Selection.close_terminal_arrangement) does not repair them, stops at stage ``bar_rows`` with the structured
incomplete outcome (2026-11-01): the rejected and the last geometrically valid state apart, neither carrying a strength
result, the failing call site and, with ``keep_actions``, the solved actions and the selection's inputs and report, so
the selector can be replayed without a solve. No capacity calculation is ever made on rows known not to fit.

An evaluated candidate reports three things apart (code-check audit of the genetic search, 2026-10-30): the
preliminary screen ``candidate_screen_passed`` (no recorded failure in the families it reads, a settled pass;
an unevaluated check is not a failure there), the authoritative ``code_check`` (Design.Code_Check_Contract:
every check the final design requires, derived from its own topology, present once and passing; known
failures, unknown or missing checks and review dependencies listed apart) and ``feasible`` (that code check
passed, the evaluation completed and the objectives are available). The strong-column beams credit the slab
mats only as far as the slab record establishes them on the final floor context.

Nothing here asserts a verification. The story-strength model is provisional (review item M1), the
transition rules are declared rules awaiting review, and Design.SMRF_Qualification
.GENERATION_RELEASE_READY is untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
import traceback
from dataclasses import asdict

import openseespy.opensees as ops

import Structure_Parameters as sp
from Design import Code_Check_Contract as contract
from Design import Design_Driver as driver
from Design import Group_Capacity as capacity_design
from Design import Group_Checks as checks
from Design import Group_Selection as selection
from Design import Perimeter_Systems as systems
from Design import SMRF_Floor_Sections as floor_sections
from Design.SMRF_Beam_Slab_Strength import BeamBarRowsError, CageArrangementError, validate_bar_rows
from Design.SMRF_Common import SectionAxialDomainError, awaiting_review, make_check, not_evaluated, summarize_checks
from Model import Member_Bar_Layers as bar_layers
from Model import Member_Groups as mg
from Model import Member_Properties as mp

SCHEMA = "rc_smrf_grouped_candidate_v1"
# v2 (2026-10-30, code-check audit of the genetic search): ``feasible`` is the authoritative code-check acceptance
# (Design.Code_Check_Contract); the former expression is ``candidate_screen_passed``, a preliminary screen only.
# v3 (2026-11-01): the terminal cage arrangement is closed or reported incomplete before any capacity calculation
# (Group_Selection.close_terminal_arrangement); the record keeps the closure and an incomplete outcome its evidence.
# v4 (2026-11-01, closure review): a proven reinforcement cycle is resolved only on verification (resolve_cycle) and
# recorded either way (constraints.reinforcement_resolution); the seed arrangement outcome attributes the slab mats by
# an audit with the mats masked and keeps its declared call site.
EVALUATION_VERSION = "grouped_candidate_evaluation_v4_cycle_resolution"
CYCLE_RESOLUTION_ID = "reinforcement_cycle_verified_resolution_v1"
CYCLE_RESOLUTION_MAX_CANDIDATES = 4   # distinct states of a cycle tried, in _cycle_candidates order
STEEL_UNIT_WEIGHT_LB_PER_IN3 = 490.0 / 1728.0
SLAB_CAGE_PASSES = 2          # re-pricings of the slab layout on the selected cages before it is reported as not settled
INFEASIBLE_STAGES = ("slab_thickness", "floor_transfer", "slab_reinforcement", "gravity_analysis", "member_actions",
                     "reinforcement_selection", "capacity_design")


class CandidateInfeasible(Exception):
    """A named design reason one candidate cannot be evaluated further (never a programming error)."""

    def __init__(self, stage, reason, evidence=None):
        super().__init__(f"{stage}: {reason}")
        self.stage, self.reason, self.evidence = stage, reason, evidence or {}


# ---- slab and floors --------------------------------------------------------------------------------------
def _slab_policy(cfg, minimum_thickness_in=None):
    policy = {**asdict(cfg.slab), "fy_ksi": sp.SLAB_FY_KSI, "concrete_unit_weight_kcf": sp.CONCRETE_UNIT_WEIGHT_KCF}
    if minimum_thickness_in is not None:
        policy["minimum_thickness_in"] = minimum_thickness_in
    return policy


def slab_completion_context(slab, cfg, state=None):
    """Floor context of the slab completion checks, as the least favourable installed member gives it.

    The uniform driver reads one beam and one column; here the shortest beam clear span in each
    direction, the narrowest perimeter beam with the largest hoop bar on a perimeter line (hook
    embedment), the smallest column core and the least beam-to-slab stiffness ratios over every floor.
    """
    state = state or mg.active()
    panels = [panel for entry in slab.get("floor_panels", [{"panels": slab["panels"]}]) for panel in entry["panels"]]
    edges = [edge for panel in panels for edge in panel["edges"]]
    beams = mg.all_members("beam_x") + mg.all_members("beam_y")
    perimeter = [m for m in beams if m.location_class == "edge"]
    columns = mg.all_members("column")
    return {"clear_span_x_in": min(mp.beam_clear_span_in(m) for m in beams if m.member_type == "beam_x"),
            "clear_span_y_in": min(mp.beam_clear_span_in(m) for m in beams if m.member_type == "beam_y"),
            "beam_width_in": min(m.design.b_in for m in perimeter),
            "alpha_f_min": min(edge["alpha_f"] for edge in edges),
            "alpha_f_l2_l1_min": min(driver._beam_shear_share_criterion(edge) for edge in edges),
            "thickness_screen_passed": slab["thickness_screen_passed"] is True,
            "column_core_width_in": min(min(m.design.b_in, m.design.h_in) - 2.0 * mp.longitudinal_cover_in(m.design)
                                        for m in columns),
            "two_way_shear_path_assessed": _two_way_shear_path_assessed(cfg),
            "slab_column_local_steel_assessed": _slab_column_local_steel_assessed(cfg),
            "columns_at_beam_intersections": True,
            "beam_clear_cover_in": sp.BEAM_CLEAR_COVER_IN,
            "beam_hoop_diameter_in": max(sp.rebar_diameter(m.design.stirrup_bar_size) for m in perimeter),
            "fc_beam_ksi": floor_sections.for_floor(1)["fc_beam_ksi"],
            "perimeter_hook_corners": _perimeter_hook_corners()}


def _perimeter_hook_corners():
    """The least favourable building-corner hook of each axis over every floor (Member_Properties.perimeter_hook_corner):
    the smallest side cover, the smallest in-plane cover, and inside the column core only where every corner is."""
    from Design.SMRF_Beam_Slab_Strength import HOOK_CORNER_CONVENTION
    result = {}
    for axis in ("x", "y"):
        corners = [mp.perimeter_hook_corner(level, i, j, axis) for level in range(1, sp.NUM_FLOOR + 1)
                   for i in (0, sp.NUM_BAY_X) for j in (0, sp.NUM_BAY_Y)]
        result[axis] = {"side_cover_in": min(c["side_cover_in"] for c in corners),
                        "in_plane_cover_in": min(c["in_plane_cover_in"] for c in corners),
                        "in_column_core": all(c["in_column_core"] for c in corners), "convention": HOOK_CORNER_CONVENTION}
    return result


def slab_assertion_scope(cfg, description):
    """Whether the run's slab-action assertions apply to this candidate (2026-11-01): they are asserted
    (SlabActionAssertions.all_asserted) and every distinct floor section set of the candidate is one the run's
    design-evidence profile covers (FloorAnalysisConfig.slab_assertion_floor_sections; None: no restriction, the earlier
    rule), at the slab thickness in force, which must be the reviewed one (FloorAnalysisConfig
    .slab_assertion_thickness_in; None: no restriction). A candidate with a floor outside the covered sets, or a slab of
    another thickness (another candidate's governing floor), is evaluated unasserted, never on another configuration's
    review; the slab stage does not step past a reviewed thickness. ``in_scope`` decides which assertions a candidate's
    evidence records (a partial review included), ``effective`` whether they are complete there, which is what selects a
    layout. A configuration without these fields (one that predates them, such as a saved record's policy) carries no
    restriction.

    Under a slab procedure (FloorAnalysisConfig.slab_procedure, Design.SMRF_Slab_Procedure) the procedure decides
    instead, and fails closed: every distinct floor must lie in its domain, the slab thickness on its ladder, and the
    procedure must be active; the per-floor domain result is recorded."""
    analysis = getattr(cfg, "floor_analysis", None)
    procedure = getattr(analysis, "slab_procedure", None)
    if procedure is not None:
        from Design import SMRF_Slab_Procedure as procedure_module
        asserted = cfg.slab_actions.all_asserted()
        rows = [{"floor_sections_sha256": sha, "floors": served,
                 "problems": procedure_module.floor_domain_problems(procedure, sections, None, check_thickness=False)}
                for sha, sections, served in floor_sections.distinct_floors(description)]
        try:
            thickness_in = procedure_module.on_ladder(sp.SLAB_THICKNESS_IN,
                                                      procedure["procedure"]["domain"]["thickness_ladder_in"])
        except (KeyError, TypeError, ValueError):
            thickness_in = False
        floors_in = all(not row["problems"] for row in rows)
        active = procedure.get("status") in procedure_module.ACTIVE_STATUSES
        in_scope = bool(floors_in and thickness_in and active)
        return {"asserted": asserted, "covered_floor_sections": None,
                "candidate_floor_sections": sorted(row["floor_sections_sha256"] for row in rows),
                "covered_slab_thickness_in": None, "candidate_slab_thickness_in": sp.SLAB_THICKNESS_IN,
                "floors_in_scope": floors_in, "thickness_in_scope": thickness_in, "in_scope": in_scope,
                "effective": bool(asserted and in_scope),
                "procedure": {"identity": procedure.get("identity"), "status": procedure.get("status"), "floors": rows}}
    covered = getattr(analysis, "slab_assertion_floor_sections", None)
    thickness = getattr(analysis, "slab_assertion_thickness_in", None)
    floors = sorted(sha for sha, _sections, _served in floor_sections.distinct_floors(description))
    asserted = cfg.slab_actions.all_asserted()
    floors_in = covered is None or set(floors) <= set(covered)
    thickness_in = thickness is None or (sp.SLAB_THICKNESS_IN is not None
                                         and abs(float(sp.SLAB_THICKNESS_IN) - float(thickness)) <= 1e-9)
    return {"asserted": asserted, "covered_floor_sections": None if covered is None else sorted(covered),
            "candidate_floor_sections": floors, "covered_slab_thickness_in": thickness,
            "candidate_slab_thickness_in": sp.SLAB_THICKNESS_IN, "floors_in_scope": floors_in,
            "thickness_in_scope": thickness_in, "in_scope": bool(floors_in and thickness_in),
            "effective": bool(asserted and floors_in and thickness_in)}


def _effective_assertions(cfg, description):
    """The slab-action assertions this candidate is evaluated under: the run's within their scope (as given, a partial
    review included: its flags and provenance stay in the evidence, and only a complete set selects a layout), none
    outside it."""
    from Design.Config import SlabActionAssertions
    return cfg.slab_actions if slab_assertion_scope(cfg, description)["in_scope"] else SlabActionAssertions()


def _two_way_shear_path_assessed(cfg):
    """The two-way shear path assessment of the installed candidate's floors, under the assertions in force there."""
    assertions = _effective_assertions(cfg, floor_sections.by_floor())
    return assertions.all_asserted() and assertions.two_way_shear_path_assessed is True


def _slab_column_local_steel_assessed(cfg):
    """The 8.6.1.2 applicability assessment of the installed candidate's floors, under the assertions in force there."""
    assertions = _effective_assertions(cfg, floor_sections.by_floor())
    return assertions.all_asserted() and assertions.slab_column_local_steel_assessed is True


def _slab_evidence(cfg, slab, description, inputs, context):
    """Strip demand evidence of every distinct floor (refined when a plan is declared), cached by floor, under the
    assertions in force for this candidate (slab_assertion_scope).

    A plan that does not resolve on a floor's own beam faces is recorded as that floor's unverified evidence
    (SMRF_Slab_Refinement.unresolvable_plan_evidence), never raised (2026-11-01). Under a slab procedure every
    distinct floor also earns its instance record (Design.SMRF_Slab_Procedure.instance_record); a floor outside the
    domain or failing a check, or any floor under a procedure that is not active, enters the combination with its
    flags masked, and the records are replayed from the combined evidence before they are kept."""
    from Design import SMRF_Slab_Procedure as procedure_module
    from Design.SMRF_Slab_Actions import _FLAGS, build_slab_action_evidence
    from Design.SMRF_Slab_Refinement import (build_refined_slab_action_evidence, combine_floor_evidence,
                                             unresolvable_plan_evidence)
    plan = cfg.floor_analysis.slab_refinement
    procedure = getattr(cfg.floor_analysis, "slab_procedure", None)
    assertions = _effective_assertions(cfg, description)
    if plan is None and assertions.all_asserted():
        raise ValueError("Asserted slab actions require FloorAnalysisConfig.slab_refinement (an explicit mesh plan or a "
                         "named recipe such as Design.Config.PROBE_SLAB_REFINEMENT): single-mesh slab demands cannot be "
                         "verified, so no slab reinforcement could be selected.")
    cache = context.setdefault("slab_evidence", {})
    parts = [{k: slab[k] for k in ("thickness_in", "concrete_fc_ksi", "concrete_unit_weight_kcf",
                                   "superimposed_dead_load_ksf")}, inputs, plan, asdict(assertions), sp.FLOOR_LIVE_LOAD_KSF,
             cfg.floor_analysis.transfer_mesh_per_bay]
    if procedure is not None:
        parts.append([procedure.get("identity"), procedure.get("status")])
    slab_key = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    floors, instances = [], []
    for sha, sections, served in floor_sections.distinct_floors(description):
        key = (sha, slab_key)
        if key not in cache:
            ops.wipe()
            if plan is None:
                cache[key] = build_slab_action_evidence(slab, driver._slab_geometry(), sections, sp.FLOOR_LIVE_LOAD_KSF, inputs,
                                                        mesh_per_bay=cfg.floor_analysis.transfer_mesh_per_bay,
                                                        uniform_all_floors=False, assertions=asdict(assertions))
                context["floor_solves"] = context.get("floor_solves", 0) + 1
            else:
                try:
                    cache[key] = build_refined_slab_action_evidence(slab, driver._slab_geometry(), sections,
                                                                    sp.FLOOR_LIVE_LOAD_KSF, inputs, plan,
                                                                    assertions=asdict(assertions))
                    context["floor_solves"] = context.get("floor_solves", 0) + 1
                except ValueError as exc:
                    # Raised before any solve: the plan does not resolve on this floor's own beam faces.
                    cache[key] = unresolvable_plan_evidence(driver._slab_geometry(), plan, exc, asdict(assertions))
        evidence = cache[key]
        if procedure is not None:
            instance = procedure_module.instance_record(procedure, sha, served, evidence, sections, slab,
                                                        sp.FLOOR_LIVE_LOAD_KSF, sp.NUM_BAY_X, sp.NUM_BAY_Y)
            instances.append(instance)
            if not instance["passed"]:
                evidence = procedure_module.masked(evidence, "outside the procedure's domain or failing an instance check")
            elif procedure.get("status") not in procedure_module.ACTIVE_STATUSES:
                evidence = procedure_module.masked(evidence, f"procedure {procedure.get('status')}")
        floors.append({"floor_sections_sha256": sha, "floors": served, "evidence": evidence})
    combined = combine_floor_evidence(floors, sp.NUM_FLOOR)
    if procedure is not None:
        combined["procedure_instances"] = instances
        combined["procedure"] = {"identity": procedure.get("identity"), "status": procedure.get("status")}
        problems = procedure_module.replay_problems(procedure, combined, description, slab, sp.FLOOR_LIVE_LOAD_KSF,
                                                    sp.NUM_BAY_X, sp.NUM_BAY_Y)
        if problems:
            for flag in _FLAGS:
                combined[flag] = False
            combined["procedure"]["replay_problems"] = problems
    return combined


def _refinement_summary(evidence):
    """What a refused slab says about its refinement, floor by floor, with the strips outside tolerance."""
    rows = []
    for entry in evidence.get("by_floor", []):
        refinement = entry["evidence"].get("refinement") or {}
        item = {"floors": entry["floors"], "floor_sections_sha256": entry["floor_sections_sha256"],
                "status": refinement.get("status"), "status_detail": refinement.get("status_detail"),
                "levels": [{"index": level.get("index"), "status": level.get("status"), "error": level.get("error"),
                            "cells_per_bay": [(level.get("requested_mesh") or {}).get("subdivisions_x_per_bay"),
                                              (level.get("requested_mesh") or {}).get("subdivisions_y_per_bay")],
                            "elapsed_seconds": level.get("elapsed_seconds")} for level in refinement.get("levels", [])]}
        if refinement.get("comparisons"):
            last = refinement["comparisons"][-1].get("comparisons") or []
            failed = [row for row in last if not row.get("within_tolerance")]
            item["final_comparison"] = {"strips_compared": len(last), "strips_outside_tolerance": len(failed),
                                        "outside_tolerance": sorted(failed, key=lambda r: -(r.get("relative_change") or 0.0))[:24]}
        rows.append(item)
    return rows


REVIEW_BUNDLE_VERSION = "slab_action_review_bundle_v1"


def slab_review_bundle(evidence):
    """The slab action evidence of a candidate laid out for its engineering review (2026-10-30).

    What was computed (load cases and patterns, every mechanically distinct floor, the mesh refinement comparison,
    the equilibrium of every case, the physical-face shear recovery, the membrane check and the numerical
    preconditions) is kept apart from what an engineer asserts (Design.Config.SlabActionAssertions, with author,
    date and basis). ``blocking`` lists what keeps the strip demands from being accepted. Nothing here asserts
    anything; the independent reference calculation of a representative floor stays a review item.
    """
    from Design.SMRF_Slab_Actions import _FLAGS
    if not isinstance(evidence, dict):
        return {"version": REVIEW_BUNDLE_VERSION, "status": "no_slab_action_evidence",
                "blocking": ["no slab action evidence was computed (the floor transfer to the frame is disabled)"]}
    preconditions = evidence.get("numerical_preconditions") or {}
    assertions = evidence.get("engineering_assertions") or {}
    equilibrium = evidence.get("equilibrium") or []
    recovery = evidence.get("shear_recovery") or {}
    computed = {"distinct_floors": [{"floors": entry.get("floors"), "floor_sections_sha256": entry.get("floor_sections_sha256")}
                                    for entry in evidence.get("by_floor", [])],
                "refinement": _refinement_summary(evidence),
                "equilibrium": {"cases": len(equilibrium),
                                "all_balanced": bool(equilibrium) and all(row.get("numerical_balance_passed") for row in equilibrium)},
                "face_shear_recovery": {"method": recovery.get("method"),
                                        "top_rows_recovered_at_face": recovery.get("top_rows_recovered_at_face")},
                "max_abs_membrane_kip_per_in": evidence.get("max_abs_membrane_kip_per_in"),
                "numerical_preconditions": preconditions, "numerical_basis": evidence.get("numerical_basis")}
    review = {"assertions": {flag: assertions.get(flag) for flag in _FLAGS},
              "asserted_by": assertions.get("asserted_by"), "assertion_date": assertions.get("assertion_date"),
              "assertion_basis": assertions.get("assertion_basis"),
              "provenance_valid": evidence.get("assertion_provenance_valid"),
              "independent_reference_calculation": ("not provided: hand-check a representative floor's strip moments, shears "
                                                    "and beam transfer against the model (floor.independent_hand_verification)")}
    blocking = ([f"computed precondition not met: {name}" for name, value in preconditions.items() if value is not True]
                + [f"not asserted: {flag}" for flag in _FLAGS if assertions.get(flag) is not True]
                + ([] if evidence.get("assertion_provenance_valid") else ["no named, dated assertion basis"])
                + ([] if computed["equilibrium"]["all_balanced"] else ["a slab case does not balance"]))
    return {"version": REVIEW_BUNDLE_VERSION, "status": "ready_for_review" if not blocking else "blocked",
            "method": evidence.get("method"), "source": evidence.get("source"),
            "load_combination_basis": evidence.get("load_combination_basis"),
            "cases": [case.get("id") for case in evidence.get("cases", [])], "pattern_rule": evidence.get("pattern_rule"),
            "computed": computed, "engineering_review": review, "blocking": blocking}


def slab_stage(cfg, context):
    """Slab thickness, floor transfers, strip demands and the common layout for the installed design.

    Returns {"slab", "description", "transfer", "evidence", "reinforcement", "attempts"} and leaves the
    result installed in Structure_Parameters. Raises CandidateInfeasible with what was tried.
    """
    from Design.SMRF_Demands import live_load_patterns
    from Design.SMRF_Floor_Transfer import build_floor_transfers_by_floor
    from Design.SMRF_Slab import SlabSizingError, choose_slab
    from Design.SMRF_Slab_Reinforcement import design_slab_reinforcement
    description = floor_sections.by_floor()
    geometry = driver._slab_geometry()
    patterns = live_load_patterns(sp.NUM_BAY_X, sp.NUM_BAY_Y) if cfg.demands.live_load_patterning else ()
    attempts, minimum = [], None
    while True:
        try:
            slab = choose_slab(geometry, description, _slab_policy(cfg, minimum))
        except SlabSizingError as exc:
            raise CandidateInfeasible("slab_thickness", str(exc), {"attempts": attempts}) from exc
        driver._apply_slab(slab)
        attempt = {"thickness_in": slab["thickness_in"], "required_thickness_in": slab["required_thickness_in"],
                   "governing_floors": slab.get("governing_floors")}
        attempts.append(attempt)
        if not cfg.floor_analysis.transfer_to_frame:
            sp.FLOOR_TRANSFER = sp.SLAB_REINFORCEMENT = sp.SLAB_ACTIONS = None
            attempt["status"] = "transfer_disabled"
            return {"slab": slab, "description": description, "transfer": None, "evidence": None, "reinforcement": None,
                    "attempts": attempts}
        ops.wipe()
        try:
            transfer = build_floor_transfers_by_floor(slab, geometry, description, sp.FLOOR_LIVE_LOAD_KSF,
                                                      mesh_per_bay=cfg.floor_analysis.transfer_mesh_per_bay,
                                                      live_patterns=patterns, reuse=context.get("transfer"))
        except RuntimeError as exc:
            if "Floor transfer" not in str(exc):
                raise
            attempt["status"] = "floor_transfer_failed"
            raise CandidateInfeasible("floor_transfer", str(exc), {"attempts": attempts}) from exc
        context["transfer"] = transfer
        context["floor_solves"] = context.get("floor_solves", 0) + len(transfer["transfers"]) - len(transfer["reused_signatures"])
        sp.FLOOR_TRANSFER = transfer
        inputs = driver._slab_strength_inputs_from_state(slab, cfg)
        evidence = _slab_evidence(cfg, slab, description, inputs, context)
        sp.SLAB_ACTIONS = evidence
        record = design_slab_reinforcement(inputs, evidence, None, slab_completion_context(slab, cfg))
        sp.SLAB_REINFORCEMENT = record
        attempt["layout_selected"] = record["layout"] is not None
        scope = slab_assertion_scope(cfg, description)
        attempt["assertion_scope"] = {k: scope[k] for k in ("asserted", "in_scope", "effective")}
        if record["layout"] is not None or not scope["effective"]:
            attempt["status"] = ("layout_selected" if record["layout"] is not None else
                                 "actions_not_asserted_no_layout" if not scope["asserted"] else
                                 "assertions_out_of_scope_no_layout")
            return {"slab": slab, "description": description, "transfer": transfer, "evidence": evidence,
                    "reinforcement": record, "attempts": attempts, "assertion_scope": scope}
        refinement = evidence.get("refinement") or {}
        reasons = [c["details"].get("reason", c["id"]) for c in record["checks"] if c["status"] != "pass"]
        if refinement.get("status") != "passed":
            # The numerical screen of the strip demands did not pass on some floor: more slab does not answer that.
            attempt["status"] = f"slab_refinement_{refinement.get('status')}"
            raise CandidateInfeasible(
                "slab_reinforcement", f"slab refinement {refinement.get('status')}: {refinement.get('status_detail')}",
                {"attempts": attempts, "refinement": _refinement_summary(evidence), "reasons": reasons})
        if any(c["id"] == "slab_verified_strip_action_inputs" for c in record["checks"]):
            # The strip routine did not accept the demand evidence itself (a flag, a signature, a recovery): that is
            # not a thickness question either.
            attempt["status"] = "strip_action_evidence_not_accepted"
            raise CandidateInfeasible("slab_reinforcement", "the strip action evidence was not accepted: " + "; ".join(reasons[:3]),
                                      {"attempts": attempts, "refinement": _refinement_summary(evidence), "reasons": reasons})
        # Demands verified but no admissible layout at this thickness: the next thickness, never a predefined slab.
        attempt.update(status="no_admissible_layout", reasons=reasons[:6],
                       trial_history=[t for t in record.get("trial_history", [])][:40])
        procedure = getattr(cfg.floor_analysis, "slab_procedure", None)
        if procedure is not None:
            from Design import SMRF_Slab_Procedure as procedure_module
            following = slab["thickness_in"] + cfg.slab.thickness_increment_in
            if not procedure_module.on_ladder(following, procedure["procedure"]["domain"]["thickness_ladder_in"]):
                # A thicker slab than the procedure's domain would be evaluated unasserted: the ladder stops here.
                raise CandidateInfeasible("slab_reinforcement",
                                          "no admissible slab reinforcement layout within the procedure's slab thickness "
                                          f"domain (the next thickness, {following:g} in, is outside it)",
                                          {"attempts": attempts, "reasons": reasons, "assertion_scope": scope})
        reviewed = getattr(cfg.floor_analysis, "slab_assertion_thickness_in", None)
        if reviewed is not None:
            # The assertions cover the reviewed thickness only: a thicker slab would be evaluated unasserted, where no
            # layout can be selected, so the ladder stops here with the reviewed thickness's reasons.
            raise CandidateInfeasible("slab_reinforcement",
                                      f"no admissible slab reinforcement layout at the reviewed slab thickness ({reviewed:g} "
                                      "in); the slab-action assertions cover that thickness only",
                                      {"attempts": attempts, "reasons": reasons, "assertion_scope": scope})
        minimum = slab["thickness_in"] + cfg.slab.thickness_increment_in
        if minimum > cfg.slab.maximum_thickness_in + 1e-9:
            raise CandidateInfeasible("slab_reinforcement",
                                      "no admissible slab reinforcement layout at any thickness of the declared range",
                                      {"attempts": attempts, "reasons": reasons})


# ---- story-by-story regularity evidence ---------------------------------------------------------------------
def story_strength_distribution(cfg):
    """Frame-line story strengths of every story from the beams actually installed there (provisional model).

    Each beam adds its member-wide composite strength (every flange mat, both signs). Beam sizing drops the mats at
    an exterior end whose hook does not develop at both datums; this story strength still counts them, a declared
    approximation of the provisional model (2026-11-01, decision D5).

    The uniform model prices one set of four beam families and declares every story alike. Here each
    story's lines are summed from that floor's own beams (Model.Member_Properties.beam_strengths), so the
    one-sided fraction of ASCE 7-22 Table 12.3-1 is a value per story and direction, and the story
    strengths are available for the vertical comparison. The model is the same beam-sway approximation,
    with the same open questions (column, joint and shear limits, base and roof mechanisms): review item
    M1 stays open and nothing here asserts it.
    """
    from Design.SMRF_Demands import one_sided_strength_fraction
    # Under the system policy only the perimeter frames are the seismic force-resisting system: the interior lines
    # are gravity frames and carry no story strength here (ASCE 7-22 Table 12.3-1 reads the SFRS; decision D7).
    perimeter_only = systems.two_system()
    layout = mg.active_layout()
    stories, worst = [], 0.0
    for floor in range(1, sp.NUM_FLOOR + 1):
        entry = {"story": floor, "by_direction": {}}
        for direction, kind, lines, spans, spacing in (("x", "beam_x", sp.NUM_BAY_Y + 1, sp.NUM_BAY_X, sp.BAY_Y),
                                                       ("y", "beam_y", sp.NUM_BAY_X + 1, sp.NUM_BAY_Y, sp.BAY_X)):
            positions, strengths, groups, line_systems = [], [], [], []
            for line in range(lines):
                total, group = 0.0, None
                counted = not perimeter_only
                for span in range(spans):
                    beam = mg.beam_at(kind, floor, span, line) if kind == "beam_x" else mg.beam_at(kind, floor, line, span)
                    # A span counts when its beam is part of the SFRS: every span of a perimeter line, the ring spans of
                    # a core line under the perimeter-core layout, none of a gravity line.
                    if perimeter_only and mg.member_system(kind, beam.grid_i, beam.grid_j, sp.NUM_BAY_X, sp.NUM_BAY_Y,
                                                           layout) != mg.SYSTEM_SFRS:
                        continue
                    counted = True
                    values = mp.beam_strengths(beam)
                    total += (values["hogging"] + values["sagging"]) / sp.STORY_H
                    group = beam.group_id
                positions.append(line * spacing)
                strengths.append(total if counted else 0.0)
                groups.append(group)
                line_systems.append(mg.SYSTEM_SFRS if counted else mg.SYSTEM_GRAVITY)
            fraction, detail = one_sided_strength_fraction(positions, strengths, 0.5 * positions[-1])
            entry["by_direction"][direction] = {"line_positions_in": positions, "line_story_strength_kip": strengths,
                                                "line_groups": groups, "line_systems": line_systems, "story_strength_kip": sum(strengths),
                                                "one_side_fraction": fraction, **detail}
            worst = max(worst, fraction)
        stories.append(entry)
    return {"one_side_fraction": worst, "stories": stories,
            "lines_counted": (("the seismic force-resisting lines only: "
                              + ("the perimeter frames and the core ring" if layout == mg.SFRS_LAYOUT_PERIMETER_CORE else "the perimeter frames")
                              + " (the other lines are gravity frames, ACI 318-19 18.14)") if perimeter_only
                              else "every frame line"),
            "applicability": driver._strength_model_applicability(cfg),
            "uniform_over_height": {"claim": False,
                                    "basis": "grouped design: sections and cages vary by story band and plan location, so "
                                             "every story is priced on its own beams"},
            "model": ("beam-sway mechanism story strength per frame line, story by story: the sum over the line's spans of "
                      "(Mn- + Mn+) / story height of the beams installed at that floor, composite with the developed slab "
                      "where established; center of mass at the plan center. A declared approximation (review item M1): "
                      "column-limited, joint-limited and shear-limited mechanisms, the base and the roof are not priced"),
            "basis": ("ASCE 7-22 Table 12.3-1 Type 1 strength criterion per story and direction; the largest story value "
                      "is reported. The story strengths also feed the weak-story comparison (Table 12.3-2 Types 4a, 4b), "
                      "which stays open with the model")}


# ASCE 7-22 Table 12.3-2 as the vertical evidence uses it. The 2022 edition has no weight (mass) irregularity
# (the former Type 2 was deleted); vertical geometric irregularity is Type 2, in-plane discontinuity Type 3,
# and the weak-story types are 4a and 4b (formerly 5a and 5b). Source: the standard itself, ASCE/SEI 7-22
# p. 120: Table 12.3-2 and Sections 12.3.2.2 (exceptions 1 and 2), 12.3.3.1, 12.3.3.2 and 12.3.3.3.
VERTICAL_THRESHOLDS = {
    "edition": "ASCE 7-22",
    "source": "ASCE/SEI 7-22 Table 12.3-2; Sections 12.3.2.2, 12.3.3.1, 12.3.3.2, 12.3.3.3 (p. 120 of the standard)",
    "soft_story_1a": {"ratio_to_story_above": 0.70, "ratio_to_average_of_three_above": 0.80},
    "extreme_soft_story_1b": {"ratio_to_story_above": 0.60, "ratio_to_average_of_three_above": 0.70},
    "drift_ratio_exception": {"limit": 1.30, "applies_to": ["1a", "1b"], "section": "12.3.2.2 exception 1",
                              "scope": "no story drift ratio above 130% of that of the next story above; the relationship "
                                       "of the top two stories is not evaluated; torsional effects need not be considered; "
                                       "with no relationship left to evaluate the exception is not shown"},
    "low_rise_exception": {"applies_to": ["1a", "1b"], "section": "12.3.2.2 exception 2",
                           "one_story_sdc": "any", "two_story_sdc": ["B", "C", "D"],
                           "scope": "Types 1a and 1b are not required to be considered for one-story buildings in any SDC "
                                    "or for two-story buildings in SDC B, C or D"},
    "weak_story_4a": {"ratio_to_story_above": 1.00,
                      "sdc_e_f_permitted_at_or_above": 0.80,
                      "reading": ("Table 12.3-2: story lateral strength less than that in the story above; the exception of "
                                  "12.3.3.1 permits Type 4a in SDC E and F where the strength is not less than 80% of that "
                                  "in the story above")},
    "extreme_weak_story_4b": {"ratio_to_story_above": 0.65},
    "prohibitions": {"12.3.3.1": "SDC E and F: Types 1b, 4a (below 80%) and 4b not permitted",
                     "12.3.3.2": "SDC D: Type 4b not permitted",
                     "12.3.3.3": "SDC B and C: a structure with Type 4b is limited to two stories or 30 ft in height, "
                                 "unless the weak story resists Omega0 times the design force"},
    "not_in_this_edition": "weight (mass) irregularity: story weights and their ratios are kept as a description only",
}


def _finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0.0


def story_evidence(drift_screen, elf, num_floor):
    """The drift rows and ELF story shears the stiffness ratios are formed from, validated.

    Returns (rows {(story, direction): row}, shear {story: kip}, problems). The evidence is usable only
    when ``problems`` is empty: one row for every story and direction, no duplicates, finite positive
    elastic drifts and drift ratios, one finite non-negative ELF force per level whose sum is positive and
    equals the recorded base shear, and drift runs made at load factor 1.0. Nothing is filled in.
    """
    problems, rows = [], {}
    forces = (elf or {}).get("story_forces_kip")
    shear = {}
    if (not isinstance(forces, (list, tuple)) or len(forces) != num_floor
            or any(isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(f) or f < 0.0 for f in forces)):
        problems.append(f"the ELF story forces are not {num_floor} finite non-negative values")
    else:
        shear = {k: sum(forces[k - 1:]) for k in range(1, num_floor + 1)}
        if not shear[1] > 0.0:
            problems.append("the ELF base shear is not positive")
        base = (elf or {}).get("base_shear_kip")
        if base is not None and not (_finite_positive(base) and math.isclose(base, shear[1], rel_tol=1e-9, abs_tol=1e-9)):
            problems.append(f"the ELF story forces sum to {shear[1]!r}, the recorded base shear is {base!r}")
        if any(not shear[k] > 0.0 for k in shear):
            problems.append("a story carries no ELF shear")
    load_factor = ((drift_screen or {}).get("assumptions") or {}).get("rho_for_drift_load")
    if load_factor != 1.0:
        problems.append(f"the drift runs' load basis is not the unit ELF load (rho_for_drift_load = {load_factor!r})")
    for row in (drift_screen or {}).get("stories") or []:
        # location "story:<k>/Q<axis>/<axis>"
        try:
            label, axis = row["location"].split(":", 1)[1].rsplit("/", 1)
            story = int(label.split("/")[0])
        except (KeyError, ValueError, IndexError, AttributeError, TypeError):
            problems.append(f"a drift row has no readable story and direction ({(row or {}).get('location')!r})"
                            if isinstance(row, dict) else "a drift row is not a record")
            continue
        if axis not in ("x", "y") or not 1 <= story <= num_floor:
            problems.append(f"a drift row names story {story} direction {axis!r}, outside the frame")
            continue
        if (story, axis) in rows:
            problems.append(f"story {story} direction {axis} has more than one drift row")
            continue
        if not _finite_positive(row.get("elastic_drift_in")) or not _finite_positive(row.get("drift_ratio")):
            problems.append(f"story {story} direction {axis}: the elastic drift or the drift ratio is not a finite positive number")
        rows[(story, axis)] = row
    missing = [f"{k}/{axis}" for axis in ("x", "y") for k in range(1, num_floor + 1) if (k, axis) not in rows]
    if missing:
        problems.append(f"no drift row for story/direction {', '.join(missing[:12])}" + (" ..." if len(missing) > 12 else ""))
    return rows, shear, problems


def vertical_regularity(drift_screen, elf, strength, ledger):
    """Story stiffness and strength ratios of the installed design against ASCE 7-22 Table 12.3-2, with the
    story weights as a description, and what each can and cannot establish.

    Stiffness of a story and direction is the ELF story shear over the largest elastic story drift of
    the drift runs (accidental torsion included, so it is a screen, not a pure translational stiffness).
    The ratios are formed only from complete evidence (``story_evidence``); with a row missing, duplicated
    or unusable no ratio is formed and the stiffness check stays not evaluated. Strength is the
    provisional story-strength model (review item M1). ASCE 7-22 has no weight (mass) irregularity: the
    member-resolved story weights and their ratios are recorded without a threshold or a check.
    """
    nf = sp.NUM_FLOOR
    rows, shear, problems = story_evidence(drift_screen, elf, nf)
    limits = VERTICAL_THRESHOLDS
    stiffness, exception_rows, soft, extreme_soft = [], [], [], []
    if not problems:
        for axis in ("x", "y"):
            values = {k: shear[k] / rows[(k, axis)]["elastic_drift_in"] for k in range(1, nf + 1)}
            for k in range(1, nf):
                three = [values[m] for m in range(k + 1, k + 4) if m <= nf]
                average = sum(three) / 3.0 if len(three) == 3 else None
                entry = {"story": k, "direction": axis, "stiffness_kip_per_in": values[k],
                         "ratio_to_story_above": values[k] / values[k + 1],
                         "ratio_to_average_of_three_above": None if average is None else values[k] / average,
                         "drift_ratio_over_story_above": rows[(k, axis)]["drift_ratio"] / rows[(k + 1, axis)]["drift_ratio"]}
                above, three_above = entry["ratio_to_story_above"], entry["ratio_to_average_of_three_above"]
                if (above < limits["extreme_soft_story_1b"]["ratio_to_story_above"]
                        or (three_above is not None and three_above < limits["extreme_soft_story_1b"]["ratio_to_average_of_three_above"])):
                    entry["indicated_type"] = "1b"
                    extreme_soft.append(entry)
                elif (above < limits["soft_story_1a"]["ratio_to_story_above"]
                      or (three_above is not None and three_above < limits["soft_story_1a"]["ratio_to_average_of_three_above"])):
                    entry["indicated_type"] = "1a"
                    soft.append(entry)
                else:
                    entry["indicated_type"] = None
                stiffness.append(entry)
                if k <= nf - 2:                                     # the top two stories' relationship is not evaluated
                    exception_rows.append({"story": k, "direction": axis, "drift_ratio_over_story_above": entry["drift_ratio_over_story_above"]})
    # Exception 1 holds on the relationships it names. With none left to evaluate (two stories or fewer) it is
    # not shown: an empty list is not a passing one.
    exception = (None if problems else
                 bool(exception_rows)
                 and all(r["drift_ratio_over_story_above"] <= limits["drift_ratio_exception"]["limit"] for r in exception_rows))
    floors = {row["floor"]: row["seismic_weight_kip"] for row in ledger["floors"]}
    weights = [{"story": k, "seismic_weight_kip": floors.get(k),
                "ratio_to_story_above": floors[k] / floors[k + 1] if k < nf and floors.get(k + 1) else None,
                "ratio_to_story_below": floors[k] / floors[k - 1] if k > 1 and floors.get(k - 1) else None}
               for k in range(1, nf + 1) if floors.get(k) is not None]
    by_story = {entry.get("story"): entry for entry in (strength or {}).get("stories", []) if isinstance(entry, dict)}
    strengths, strength_problems = [], []
    for axis in ("x", "y"):
        for k in range(1, nf):
            try:
                here = by_story[k]["by_direction"][axis]["story_strength_kip"]
                above = by_story[k + 1]["by_direction"][axis]["story_strength_kip"]
            except (KeyError, TypeError):
                strength_problems.append(f"no story strength for story {k} or {k + 1}, direction {axis}")
                continue
            if not _finite_positive(here) or not _finite_positive(above):
                strength_problems.append(f"story {k} direction {axis}: a story strength is not a finite positive number")
                continue
            ratio = here / above
            strengths.append({"story": k, "direction": axis, "story_strength_kip": here, "ratio_to_story_above": ratio,
                              "indicated_type": ("4b" if ratio < limits["extreme_weak_story_4b"]["ratio_to_story_above"] else
                                                 "4a" if ratio < limits["weak_story_4a"]["ratio_to_story_above"] else None),
                              "at_or_above_80_percent": ratio >= limits["weak_story_4a"]["sdc_e_f_permitted_at_or_above"]})
    applicability = ((strength or {}).get("applicability") or {}).get("status")
    return {"num_stories": nf, "stiffness": stiffness, "strength": strengths,
            "story_weights": {"rows": weights,
                              "status": ("description only: ASCE 7-22 Table 12.3-2 has no weight (mass) irregularity; no threshold "
                                         "and no check is applied to these ratios")},
            "thresholds": limits,
            "evidence": {"stiffness_complete": not problems, "stiffness_problems": problems,
                         "strength_complete": not strength_problems, "strength_problems": strength_problems,
                         "strength_model_status": applicability},
            "indicated": {"soft_story_1a": None if problems else soft, "extreme_soft_story_1b": None if problems else extreme_soft,
                          "drift_ratio_exception_holds": exception, "drift_ratio_exception_rows": exception_rows,
                          "drift_ratio_exception_evaluable": None if problems else bool(exception_rows),
                          "weak_story_4a_provisional": [r for r in strengths if r["indicated_type"] == "4a"],
                          "extreme_weak_story_4b_provisional": [r for r in strengths if r["indicated_type"] == "4b"]},
            "basis": ("stiffness from the ELF story shear over the largest elastic story drift of the drift runs (accidental "
                      "torsion included); strength from the provisional story-strength model (review item M1); story weights "
                      "from the member-resolved seismic weight ledger, as a description. Vertical geometric irregularity "
                      "(Type 2) and in-plane discontinuity (Type 3) do not arise: every frame line runs the full height on "
                      "aligned centerlines with the same plan dimensions")}


def grouped_regularity(strength, vertical, state=None):
    """The regularity block of the demand basis for a grouped design: nothing by construction over height. The frame
    lines named are the SFRS of the installed policy (2026-10-30): every grid line under the band policy, the
    perimeter frames (and a core ring or core lines) under a system policy, never the band policy's premise."""
    state = state or mg.active()
    policy_id = state.policy_id if state is not None else mg.POLICY_ID
    if mg.is_system_policy(policy_id):
        layout = mg.layout_of(policy_id)
        frames = ("moment frames on the perimeter"
                  + {mg.SFRS_LAYOUT_PERIMETER_CORE: " and the core ring",
                     mg.SFRS_LAYOUT_PERIMETER_CORE_LINES: " and the core lines"}.get(layout, "")
                  + " (the other lines are gravity frames, ACI 318-19 18.14)")
    else:
        frames = "moment frames on every grid line"
    return {"regular": None,
            "horizontal": f"rectangular plan, {frames}, rigid diaphragm without openings: no Type 2-5 "
                          "horizontal irregularities by construction; Type 1 (ASCE 7-22 Table 12.3-1) from the torsion "
                          "assessment and the story-by-story strength distribution, whose model is provisional",
            "vertical": "sections and reinforcement vary by story band: stiffness (ASCE 7-22 Table 12.3-2 Types 1a, 1b) and "
                        "strength (Types 4a, 4b) are evaluated story by story (vertical_evidence); Types 2 and 3 do not arise "
                        "by construction; no vertical regularity is claimed by construction",
            "vertical_evidence": vertical,
            "lateral_strength_distribution": strength}


def regularity_checks(vertical, sdc=None, strength_model_verified=False):
    """Checks of the vertical evidence: what is established is evaluated, what is missing or rests on M1 stays open.

    ``sdc`` is the seismic design category of the record ("B".."F"), which decides the consequences that
    12.3.3.1 and 12.3.3.2 attach to the extreme types and whether exception 2 of 12.3.2.2 covers a two-story
    frame (SDC B, C and D; a one-story frame in any SDC). ``strength_model_verified`` is whether the record
    carries a valid assertion of the story-strength model (review item M1); without it the strength check
    stays open whatever the block says about itself. A stiffness check passes only on complete story
    evidence. There is no weight (mass) check: ASCE 7-22 has no such irregularity.
    """
    indicated, evidence, limits = vertical["indicated"], vertical["evidence"], vertical["thresholds"]
    result = []
    # The open outcomes that wait on an engineering judgement rather than on missing numbers (2026-10-30): the
    # provisional story-strength model (review item M1) and the consequences of an indicated irregularity.
    review_items = {"awaiting review item M1": "story_strength_model_verified",
                    "stiffness irregularity indicated": "vertical_irregularity_consequences_reviewed",
                    "strength irregularity indicated": "vertical_irregularity_consequences_reviewed"}

    def left_open(name, clause, message, **details):
        """A not-evaluated check that keeps its reason and carries the evidence beside it."""
        item = review_items.get(details.get("category"))
        check = awaiting_review(name, clause, message, item) if item else not_evaluated(name, clause, message)
        check["details"] = {**(check.get("details") or {}), **details}
        return check
    clause = "ASCE 7-22 Table 12.3-2 Types 1a, 1b; 12.3.2.2; 12.3.3.1"
    name = "demands.vertical_stiffness_regularity"
    if not evidence["stiffness_complete"]:
        result.append(left_open(name, clause, "the story drift and ELF shear evidence is incomplete or unusable: "
                                + "; ".join(evidence["stiffness_problems"][:4]),
                                category="incomplete story evidence", problems=evidence["stiffness_problems"]))
    else:
        soft, extreme, exception = indicated["soft_story_1a"], indicated["extreme_soft_story_1b"], indicated["drift_ratio_exception_holds"]
        stories, low_rise = vertical.get("num_stories"), limits["low_rise_exception"]
        evaluable = indicated.get("drift_ratio_exception_evaluable", True)
        details = {"soft_story_1a": soft, "extreme_soft_story_1b": extreme, "drift_ratio_exception_holds": exception,
                   "drift_ratio_exception_evaluable": evaluable,
                   "drift_ratio_exception_rows": indicated["drift_ratio_exception_rows"], "seismic_design_category": sdc,
                   "num_stories": stories,
                   "thresholds": {k: limits[k] for k in ("soft_story_1a", "extreme_soft_story_1b", "drift_ratio_exception",
                                                         "low_rise_exception")}}
        unshown = ("the 12.3.2.2 drift-ratio exception does not hold" if evaluable else
                   "the 12.3.2.2 drift-ratio exception has no story relationship to evaluate (the top two stories are left out)")
        if stories == 1 or (stories == 2 and sdc in low_rise["two_story_sdc"]):
            result.append(make_check(name, clause, 0, 0, "==",
                                     details={**details, "basis": "12.3.2.2 exception 2: Types 1a and 1b are not required to be "
                                                                  "considered for a one-story building in any SDC or a two-story "
                                                                  "building in SDC B, C or D; any ratio is recorded without a "
                                                                  "consequence"}))
        elif not soft and not extreme:
            result.append(make_check(name, clause, 0, 0, "==", details={**details, "basis": "no story is below the Type 1a or 1b ratios"}))
        elif exception:
            result.append(make_check(name, clause, 0, 0, "==",
                                     details={**details, "basis": "ratios below Type 1a or 1b are indicated, and the 12.3.2.2 "
                                                                  "drift-ratio exception holds, so Types 1a and 1b do not apply"}))
        elif extreme and sdc in ("E", "F"):
            result.append(make_check(name, "ASCE 7-22 12.3.3.1 (Type 1b not permitted in SDC E and F)", len(extreme), 0, "==",
                                     details={**details, "basis": f"an extreme soft story (Type 1b) is indicated and {unshown}"}))
        else:
            kinds = ("extreme soft story (Type 1b)" if extreme else "soft story (Type 1a)")
            result.append(left_open(name, clause, f"{len(soft) + len(extreme)} story/direction row(s) indicate a {kinds} and "
                                    f"{unshown}; the consequences for this frame "
                                    "need engineering review", category="stiffness irregularity indicated", **details))
    weak, extreme_weak = indicated["weak_story_4a_provisional"], indicated["extreme_weak_story_4b_provisional"]
    name, clause = "demands.vertical_strength_regularity", "ASCE 7-22 Table 12.3-2 Types 4a, 4b; 12.3.3.1; 12.3.3.2; 12.3.3.3"
    details = {"weak_story_4a_rows": weak, "extreme_weak_story_4b_rows": extreme_weak, "seismic_design_category": sdc,
               "thresholds": {k: limits[k] for k in ("weak_story_4a", "extreme_weak_story_4b", "prohibitions")},
               "strength_model_status": evidence["strength_model_status"]}
    if not evidence["strength_complete"]:
        result.append(left_open(name, clause, "the story strength evidence is incomplete or unusable: "
                                + "; ".join(evidence["strength_problems"][:4]),
                                category="incomplete story evidence", problems=evidence["strength_problems"], **details))
    elif not strength_model_verified:
        result.append(left_open(name, clause,
                                "story strengths come from the provisional beam-sway model (review item M1: column, joint "
                                "and shear limits, base and roof mechanisms not priced); a weak story is neither "
                                "established nor excluded"
                                + (f"; the provisional model indicates {len(weak)} Type 4a and {len(extreme_weak)} Type 4b "
                                   "row(s)" if weak or extreme_weak else ""),
                                category="awaiting review item M1", **details))
    else:
        prohibited = ([r for r in extreme_weak] if sdc in ("D", "E", "F") else []) + \
                     ([r for r in weak if not r["at_or_above_80_percent"]] if sdc in ("E", "F") else [])
        if not weak and not extreme_weak:
            result.append(make_check(name, clause, 0, 0, "==", details={**details, "basis": "no story is weaker than the story above"}))
        elif prohibited:
            result.append(make_check(name, clause, len(prohibited), 0, "==",
                                     details={**details, "prohibited_rows": prohibited,
                                              "basis": "a weak-story type that 12.3.3.1 or 12.3.3.2 does not permit in this SDC"}))
        else:
            result.append(left_open(name, clause, f"{len(weak)} Type 4a and {len(extreme_weak)} Type 4b row(s) are indicated; "
                                    "the consequences for this frame need engineering review"
                                    + ("; 12.3.3.3 limits a structure with Type 4b in SDC B or C to two stories or 30 ft unless "
                                       "the weak story resists Omega0 times the design force"
                                       if extreme_weak and sdc in ("B", "C") else ""),
                                    category="strength irregularity indicated", **details))
    return result


# ---- quantities ----------------------------------------------------------------------------------------------
def _hoop_set_length_in(design):
    """Bar length of one hoop set: the perimeter hoop and its crossties, with seismic hook extensions."""
    cover = sp.COL_CLEAR_COVER_IN if design.is_column else sp.BEAM_CLEAR_COVER_IN
    db = sp.rebar_diameter(design.stirrup_bar_size)
    width, depth = design.b_in - 2.0 * cover, design.h_in - 2.0 * cover
    hook = max(6.0 * db, 3.0)
    if design.is_column:
        legs_b, legs_h = design.stirrup_legs_by_direction or (design.stirrup_legs, design.stirrup_legs)
        # legs across the b faces run along h; legs across the h faces run along b
        ties = (legs_b - 2) * (depth + 2.0 * hook) + (legs_h - 2) * (width + 2.0 * hook)
    else:
        ties = (design.stirrup_legs - 2) * (depth + 2.0 * hook)
    return 2.0 * (width + depth) + 2.0 * hook + ties


def quantities(state=None, slab=None, slab_reinforcement=None, capacity=None):
    """Modeled concrete and reinforcement of the installed design, by group and in total.

    Concrete: columns over the story height less the slab, beam drops between the faces of their own end
    joints, and the slab over the centerline footprint (the weight ledger's volumes, so nothing is counted
    twice). Longitudinal steel: every bar over the member's centerline length, plus the lower bars that run
    on past a transition. Transverse steel: one hoop set (perimeter hoop and crossties, seismic hooks) at
    the installed spacing along the whole member, as the model takes it. Slab mats: the common layout over
    the floor area, when established; unknown (None), never zero, when it is not.
    """
    state = state or mg.active()
    slab_t = sp.SLAB_THICKNESS_IN or 0.0
    groups, concrete, longitudinal, transverse = {}, 0.0, 0.0, 0.0
    for gid, group in sorted(state.groups.items()):
        design = state.designs[gid]
        volume = steel = hoops = 0.0
        for tag in group["member_tags"]:
            member = state.member(tag)
            if member.is_column:
                volume += design.b_in * design.h_in * (sp.STORY_H - slab_t)
                length = sp.STORY_H
            else:
                volume += design.b_in * (design.h_in - slab_t) * mp.beam_clear_span_in(member)
                length = mp.beam_span_in(member)
            steel += design.longitudinal_area_in2 * length
            hoops += (sp.rebar_area(design.stirrup_bar_size) * _hoop_set_length_in(design)
                      * length / design.stirrup_spacing_in)
        groups[gid] = {"member_type": design.member_type, "members": len(group["member_tags"]),
                       "section_in": [design.b_in, design.h_in], "fc_ksi": design.fc_ksi,
                       "concrete_in3": volume, "longitudinal_steel_in3": steel, "transverse_steel_in3": hoops,
                       "longitudinal_ratio": design.longitudinal_area_in2 / design.gross_area_in2}
        concrete, longitudinal, transverse = concrete + volume, longitudinal + steel, transverse + hoops
    transition_steel, transition_hoops, unpriced = 0.0, 0.0, []
    if capacity is not None:
        for key, transition in capacity.get("transitions", {}).items():
            joints = len(transition.get("joints", []))
            lower_bar = transition["lower"]["bar_size"]
            transition_steel += transition.get("extra_bar_length_in", 0.0) * sp.rebar_area(lower_bar) * joints
            # The detail-dependent hoops of the transition: additional sets at the offset bends, and the lower
            # column's hoops continued one column depth above the joint where they are closer than the upper's.
            lower_gid, upper_gid = key.split(">", 1)
            lower, upper = state.designs[lower_gid], state.designs[upper_gid]
            lower_set = sp.rebar_area(lower.stirrup_bar_size) * _hoop_set_length_in(lower)
            transition_hoops += transition.get("additional_hoop_sets_per_joint", 0) * lower_set * joints
            depth = (transition.get("confinement") or {}).get("lower_hoops_continue_above_joint_in")
            if depth is None:
                if transition.get("kind") != "same_cage":
                    unpriced.append(key)                 # a reduced column: no continuing-hoop detail is declared to price
            else:
                upper_set = sp.rebar_area(upper.stirrup_bar_size) * _hoop_set_length_in(upper)
                transition_hoops += max(0.0, depth / lower.stirrup_spacing_in * lower_set
                                        - depth / upper.stirrup_spacing_in * upper_set) * joints
    slab_volume = slab_t * mp.floor_area_in2() * sp.NUM_FLOOR
    layout = (slab_reinforcement or {}).get("layout") if slab_reinforcement else None
    slab_steel = None
    if layout is not None:
        area_in2 = mp.floor_area_in2()
        slab_steel = 0.0
        for layer in layout["layers"].values():
            # bars at ``spacing_in`` across the floor, each running the floor's length in its own direction
            slab_steel += layer["bar_area_in2"] / layer["spacing_in"] * area_in2
        slab_steel *= sp.NUM_FLOOR
    frame_steel = longitudinal + transverse + transition_steel + transition_hoops
    return {"groups": groups,
            "concrete_frame_in3": concrete, "concrete_slab_in3": slab_volume, "concrete_total_in3": concrete + slab_volume,
            "concrete_total_yd3": (concrete + slab_volume) / 46656.0,
            "longitudinal_steel_in3": longitudinal, "transverse_steel_in3": transverse,
            "transition_extra_steel_in3": transition_steel, "transition_extra_transverse_steel_in3": transition_hoops,
            "transitions_without_priced_confinement": sorted(unpriced), "frame_steel_in3": frame_steel,
            "frame_steel_lb": frame_steel * STEEL_UNIT_WEIGHT_LB_PER_IN3,
            "slab_steel_in3": slab_steel,
            "slab_steel_lb": None if slab_steel is None else slab_steel * STEEL_UNIT_WEIGHT_LB_PER_IN3,
            "total_steel_in3": None if slab_steel is None else frame_steel + slab_steel,
            "distinct_form_sizes": state.distinct_form_sizes(),
            "distinct_form_size_count": sum(len(v) for v in state.distinct_form_sizes().values()),
            "basis": ("modeled quantities for comparing candidates, not a takeoff: no laps, hooks of longitudinal bars or "
                      "waste; joint hoops at the member spacing. At column transitions the terminated bars, the additional "
                      "hoop sets at offset bends and the lower hoops continued above a joint of unchanged section are "
                      "priced; above a reduced column no continuing-hoop detail is declared, so none is priced "
                      "(transitions_without_priced_confinement); steel at 490 lb/ft3")}


# ---- the reinforcement pass ---------------------------------------------------------------------------------
def joint_check_inputs(capacity, state):
    """What Design.SMRF_Joints.evaluate_joints reads, from a group capacity design (design time and qualification)."""
    through = [{"id": f"{type_id}/{axis}", "bars_pass_through": True,
                "joint_depth_in": item["through_bar_depth_available_in"],
                "beam_depths_in": [state.designs[item["beam_group"]].h_in],
                "concrete_type": "normalweight" if sp.CONCRETE_UNIT_WEIGHT_KCF == 0.150 else None,
                "bars": [{"grade_ksi": sp.FY_KSI, "diameter_in": item["beam_bar_diameter_in"]}],
                "scope": "joint depth only; terminating-bar development is joint.terminating_bar_hook"}
               for type_id, kind in capacity["joint_types"].items() for axis, item in kind["anchorage"].items()
               if item["through_bars_present"]]
    # The joint directions where 18.7.3.2 is not required (Group_Capacity.scwb_evidence: the column is not part of the
    # SFRS in that direction) keep their evidence and their flag; SMRF_Joints.evaluate_joints skips them.
    not_required = [f"{joint['id']}/{axis}" for joint in capacity["scwb"]["joints"]
                    for axis, states in joint["directions"].items() if not states.get("required", True)]
    return {"joints": capacity["scwb"]["joints"], "scwb_not_required": not_required, **capacity["joint_evidence"],
            "through_bar_anchorage": through}


def scwb_constraint(joint_checks, capacity):
    """The strong-column summary of a candidate: Code_Check_Contract.scwb_summary over the joint directions the
    capacity design marks required. ``all_pass`` needs every required row present once and passing;
    ``no_known_failure`` is the weaker statement the reinforcement pass reacts to."""
    joints = (capacity.get("scwb") or {}).get("joints")
    # Without the capacity design's joint evidence no inventory is known, so nothing can be complete.
    required = None if joints is None else [f"{joint['id']}/{axis}/{sign}" for joint in joints
                                            for axis, states in joint["directions"].items() if states.get("required", True)
                                            for sign in ("positive", "negative")]
    summary = contract.scwb_summary(joint_checks, required)
    if required is None:
        summary.update(all_pass=False, complete=False)
    return {**summary, "ratio_required": sp.SCWB_RATIO_MIN}


def _cycle_candidates(window):
    """The distinct states of a proven cycle in a fixed order: the most column steel first (the strong-column side of
    the alternation), then the most steel, then the state identity. The order favours strength, not the selection
    objective: a state adopted on it is not the selection's optimum."""
    def steel(state, columns_only):
        return sum(design.longitudinal_area_in2 * len(state.groups[gid]["member_tags"])
                   for gid, design in state.designs.items() if design.is_column or not columns_only)
    distinct = {}
    for state in window:
        distinct.setdefault(state.identity(), state)
    return sorted(distinct.values(), key=lambda s: (-steel(s, True), -steel(s, False), s.identity()))


def resolve_cycle(window, actions, expected_ids, cfg, transfers, scwb_margin, reserve=None, compatibility=None):
    """A proven cycle of the reinforcement passes, resolved only on verification (CYCLE_RESOLUTION_ID, 2026-11-01
    closure review). The states the passes alternate between are proposals, never a convergence. They are tried in the
    fixed order of _cycle_candidates (at most CYCLE_RESOLUTION_MAX_CANDIDATES), and one is adopted only when all of
    these hold on it: the complete arrangement fits and every installed cage meets the selection's own rules on the
    installed design (Group_Selection.verify_installed_selection); the capacity design built on it asks for no other
    hoops (its hoops are a fixed point); and that capacity design is accepted. The first such state is adopted with
    that capacity design, which describes it. Otherwise nothing is adopted and the passes stay unsettled: the honest
    incomplete outcome, with every candidate's unmet rules recorded. A proposal whose verification or capacity design
    raises a named model or arrangement failure (ValueError or RuntimeError: an axial load outside a section's domain,
    rows that do not fit, a section model without an equilibrium root, layers that do not settle) is recorded with it
    and not adopted, so the passes stay unsettled as they would without it; a programming error propagates.

    Returns (record, adopted state or None, its capacity design or None, its verification or None)."""
    candidates = _cycle_candidates(window)
    record = {"rules": CYCLE_RESOLUTION_ID, "verification_rules": selection.VERIFICATION_ID,
              "distinct_states": len(candidates), "not_tried": max(0, len(candidates) - CYCLE_RESOLUTION_MAX_CANDIDATES),
              "candidates": [], "adopted_state_identity": None}
    for state in candidates[:CYCLE_RESOLUTION_MAX_CANDIDATES]:
        mg.install(state)
        row = {"state_identity": state.identity()}
        record["candidates"].append(row)
        try:
            verification = selection.verify_installed_selection(state, actions, cfg, scwb_margin, reserve)
        except (ValueError, RuntimeError) as exc:
            row.update(selection_rules_met=False, verification_error=f"{type(exc).__name__}: {str(exc)[:500]}")
            continue
        row.update(selection_rules_met=verification["verified"], unmet_rules=verification["failing"])
        if not verification["verified"]:
            continue
        try:
            capacity = capacity_design.build_group_capacity_design(actions, expected_ids, cfg, transfers, state,
                                                                   compatibility=compatibility)
        except (ValueError, RuntimeError) as exc:
            row["capacity_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
            continue
        updates = capacity_design.hoop_updates(capacity, state)
        row.update(hoops_fixed=not updates, hoops_asked=sorted(updates), capacity_accepted=bool(capacity["accepted"]),
                   capacity_not_passing=sorted({c["id"] for c in capacity["checks"] if c["status"] != "pass"}))
        if not updates and capacity["accepted"]:
            record["adopted_state_identity"] = state.identity()
            return record, state, capacity, verification
    record["reason"] = ("no state of the cycle meets the selection's rules with fixed hoops and an accepted capacity "
                        "design; the passes stay unsettled")
    return record, None, None, None


def reinforcement_pass(state, actions, expected_ids, cfg, transfers, max_passes=10, scwb_margins=(1.0, 1.03, 1.06),
                       compatibility=None):
    """Cages and hoops of every group until neither changes any more. Returns the evidence of the final design.

    Each pass selects the longitudinal cages on the solved actions, runs the group capacity design on
    them and installs the hoops it selects; a hoop bar moves the longitudinal bars (cover + hoop + db/2),
    so the selection is repeated on the new hoops. The pass is settled only when BOTH hold in the same
    pass: the longitudinal selection reports that no cage changed any more, and the capacity design asks
    for no other hoops than those installed. A selection that did not settle is never covered by hoops
    that happened to stand still. When the passes run out, the design is reported as not settled (a search
    outcome, not an error) and the capacity design is rebuilt on the design that is left installed, so the
    evidence returned always describes that design. When the exact joint evaluation finds a strong-column
    failure that the selection estimate did not, the selection is repeated asking for a larger margin.

    A pass is a function of its input (the state, the margin and the reserved cores), so a pass whose input repeats an
    earlier pass's input at the same margin proves a cycle (2026-11-01): the passes stop there, and the result says how
    they ended (``stop``: settled, cycle or pass_budget; ``cycle``: the passes that repeat). Each pass's log names the
    terminal-arrangement closures of its selection. A cycle is resolved only on verification (resolve_cycle): a state
    it alternates between is adopted when it meets the selection's rules with fixed hoops and an accepted capacity
    design, and is described by its own verification and capacity design; ``settled`` and ``selection_settled`` stay
    False either way (``resolution``: settled, verified_cycle_resolution, cycle_unresolved or pass_budget).
    """
    from Design.SMRF_Joints import evaluate_joints
    log, result = [], None
    # The column lanes an adopted cage coordination opened stay reserved for the rest of this pass (Group_Selection
    # select_reinforcement ``reserve``): otherwise the next selection could close them again.
    reserve = []
    for margin in scwb_margins:
        settled, evidence_rebuilt, stop, cycle, inputs = False, False, "pass_budget", None, {}
        visited, resolved = [], False                 # the input of every pass, and whether a cycle was resolved
        for index in range(max_passes):
            key = (state.identity(), tuple(sorted(reserve)))
            if key in inputs:
                stop, cycle = "cycle", {"scwb_margin": margin, "pass": index + 1, "repeats_pass": inputs[key],
                                        "detail": "the reinforcement pass would start again from the input of an "
                                                  "earlier pass: the passes alternate and cannot settle"}
                # The inputs of the passes the cycle repeats are proposals only: adopted on verification, never as a
                # convergence (resolve_cycle).
                cycle["resolution"], adopted, adopted_capacity, verification = resolve_cycle(
                    visited[inputs[key] - 1:], actions, expected_ids, cfg, transfers, margin, reserve, compatibility)
                if adopted is not None:
                    state, capacity, updates, resolved = adopted, adopted_capacity, {}, True
                    report = selection.verified_selection_report(report, verification, reserve)
                mg.install(state)
                break
            inputs[key] = index + 1
            visited.append(state)
            state, report = selection.select_reinforcement(state, actions, cfg, scwb_margin=margin,
                                                           **({"reserve": reserve} if reserve else {}))
            reserve = list(report.get("reserved_cores") or reserve)
            mg.install(state)
            capacity = capacity_design.build_group_capacity_design(actions, expected_ids, cfg, transfers, state,
                                                                   compatibility=compatibility)
            updates = capacity_design.hoop_updates(capacity, state)
            log.append({"scwb_margin": margin, "pass": index + 1, "selection_settled": bool(report["settled"]),
                        "hoops_settled": not updates, "exhausted_groups": report["exhausted_groups"],
                        "hoops_changed": sorted(updates),
                        "terminal_arrangements": [t.get("status") for t in report.get("terminal_arrangements") or []]})
            if not updates and report["settled"]:
                settled, stop = True, "settled"
                break
            if updates:
                state = state.with_designs(updates)
                mg.install(state)
        if not settled and updates:
            # The last capacity design was made before these hoops went in. A hoop moves the longitudinal bars (cover +
            # hoop + db/2), so the complete arrangement is closed again on them (Group_Selection
            # .close_terminal_arrangement: unchanged when the rows fit, the incomplete outcome raised when no route
            # closes it) before the capacity design is rebuilt on the installed design.
            state = selection.close_terminal_arrangement(state, actions, cfg, margin, report, reserve,
                                                         inputs={"call": "reinforcement_pass_after_hoops",
                                                                 "input_state": state, "scwb_margin": margin,
                                                                 "reserve": sorted(reserve)})
            mg.install(state)
            capacity = capacity_design.build_group_capacity_design(actions, expected_ids, cfg, transfers, state,
                                                                   compatibility=compatibility)
            evidence_rebuilt = True
        joint_checks = evaluate_joints(joint_check_inputs(capacity, state))
        failed = [c for c in joint_checks if c["id"] == "scwb" and c["status"] == "fail"]
        result = {"state": state, "capacity": capacity, "joint_checks": joint_checks, "selection": report,
                  "settled": settled, "selection_settled": bool(report["settled"]),
                  # An adopted state's hoops are the fixed point its own capacity design was checked for.
                  "hoops_settled": resolved or (bool(log) and log[-1]["hoops_settled"] and not evidence_rebuilt),
                  "evidence_rebuilt_on_final_design": evidence_rebuilt, "scwb_margin": margin, "log": log,
                  "stop": stop, "cycle": cycle, "scwb": scwb_constraint(joint_checks, capacity),
                  "resolution": ("settled" if settled else contract.VERIFIED_CYCLE_RESOLUTION if resolved
                                 else "cycle_unresolved" if stop == "cycle" else "pass_budget")}
        if not failed or report["exhausted_groups"]:
            break
    return result


# ---- one candidate --------------------------------------------------------------------------------------------
def invalid_beam_row_groups(state):
    """Beam groups whose installed bar rows do not fit their depth beside the beams they cross
    (Design.SMRF_Beam_Slab_Strength.validate_bar_rows on the arrangement of Model.Member_Bar_Layers)."""
    if state is None:
        return []
    bad = []
    with mg.installed(state):
        try:
            rows = bar_layers.arrangement()
        except mg.GroupedStateError:
            return sorted(gid for gid, group in state.groups.items() if group["member_type"] != "column")
        for gid, group in state.groups.items():
            if group["member_type"] == "column":
                continue
            design = state.designs[gid]
            try:
                validate_bar_rows(rows[gid]["top"], design.top_bars, "top bars", design.h_in)
                validate_bar_rows(rows[gid]["bottom"], design.bot_bars, "bottom bars", design.h_in)
            except ValueError:
                bad.append(gid)
    return sorted(bad)


TORSION_CLAUSE = "ASCE 7-22 12.8.4.3 amplification with the Table 12.3-1 Type 1 classification"


def torsional_amplification_check(demand_basis):
    """Was the 12.8.4.3 amplification the analyses applied enough for the final Type 1 classification?

    Type 1 established: the applied Ax must reach the 12.8.4.3 envelope. Type 1 excluded (TIR <= 1.2 and a verified
    story-strength model): nothing is required. Unresolved (the TIR does not establish Type 1 and the provisional
    strength model cannot exclude it): an applied amplification that already reaches the envelope covers either
    outcome; otherwise the row waits on review item M1, never a pass.
    """
    name = "demands.torsional_amplification"
    torsion = (demand_basis or {}).get("torsion")
    if not torsion:
        return not_evaluated(name, TORSION_CLAUSE, "no torsion assessment was made")
    label = (torsion.get("classification") or {}).get("label")
    applied, envelope = torsion.get("amplification"), torsion.get("amplification_envelope_12_8_4_3")
    details = {"classification": label, "applied_amplification": applied, "envelope": envelope, "tir": torsion.get("tir")}
    if label == "none":
        return make_check(name, TORSION_CLAUSE, 1.0, applied, "<=", details={**details, "basis": "Type 1 is excluded"})
    if label == "type_1":
        return make_check(name, TORSION_CLAUSE, envelope, applied, "<=", details={**details, "basis": "Type 1 is established"})
    if isinstance(applied, (int, float)) and isinstance(envelope, (int, float)) and applied >= envelope:
        return make_check(name, TORSION_CLAUSE, envelope, applied, "<=",
                          details={**details, "basis": "Type 1 unresolved; the applied amplification covers it either way"})
    return awaiting_review(name, TORSION_CLAUSE, "Type 1 is neither established nor excluded: the story-strength model is "
                           "provisional (review item M1) and the applied amplification does not reach the 12.8.4.3 envelope",
                           "story_strength_model_verified")


def torsional_classification_review(demand_basis):
    """The Table 12.3-1 Type 1 classification as a review row when it is unresolved: its consequences reach past the
    12.8.4.3 amplification (ASCE 7-22 Table 12.3-1 lists the sections it triggers), so an unresolved classification
    stays a listed dependency even when the applied amplification covers Type 1."""
    label = (((demand_basis or {}).get("torsion") or {}).get("classification") or {}).get("label")
    if label != "unresolved":
        return []
    return [awaiting_review("demands.torsional_irregularity_classification", "ASCE 7-22 Table 12.3-1 Type 1",
                            "the TIR does not establish Type 1 and the provisional story-strength model (review item M1) "
                            "cannot exclude it; the consequences of Type 1 other than the 12.8.4.3 amplification follow "
                            "the classification", "story_strength_model_verified")]


def slab_code_checks(slab, eligibility, cfg):
    """The slab rows of the candidate: the layout's design state, the currency of its evidence, the record's own strip
    and completion checks (its review items marked as such) and, when the slab actions await their engineering
    assertion, that review as its own row."""
    record = slab.get("reinforcement")
    if record is None:
        return [not_evaluated("slab.reinforcement_designed", "ACI 318-19 Chapters 8, 20, 25",
                              "no slab reinforcement record: the floor transfer to the frame is disabled")]
    clause = "ACI 318-19 Chapters 8, 20, 25 (slab strip strength and completion)"
    code = eligibility.get("reason_code")
    if record.get("layout") is not None:
        designed = make_check("slab.reinforcement_designed", clause, 1, 1, "==",
                              details={"layout_selected": True, "slab_checks_passed": eligibility.get("slab_checks_passed")})
    elif code == "no_admissible_slab_layout":
        designed = make_check("slab.reinforcement_designed", clause, 0, 1, "==",
                              details={"reason": eligibility.get("reason"), "reason_code": code})
    else:
        designed = not_evaluated("slab.reinforcement_designed", clause, str(eligibility.get("reason")))
        designed["details"]["reason_code"] = code
    rows = [designed,
            make_check("slab.evidence_current", "Evidence integrity: the slab priced on the final floor context",
                       int(eligibility.get("context_current") is not False), 1, "==",
                       details={"context_current": eligibility.get("context_current"), "reason_code": code}),
            *record.get("checks", [])]
    if code in ("slab_action_review_missing", "slab_action_refinement_incomplete", "slab_action_evidence_not_accepted"):
        rows.append(awaiting_review("slab.action_evidence_review", "ACI 318-19 Chapters 6 and 8 (Design.Config."
                                    "SlabActionAssertions)", "the computed slab action evidence awaits its named, dated "
                                    "engineering assertion after review (and a completed mesh refinement); until then no "
                                    "slab layout is selected and the strong-column checks stay unknown",
                                    "slab_action_assertions"))
    return rows


def candidate_code_checks(state, cfg, *, strengths, capacity, joint_checks, drift_screen, vertical, strength,
                          demand_basis, slab, eligibility, exhausted):
    """Every check row of the final candidate, tagged with its family, as Design.Code_Check_Contract reads them.

    member strength (one row per group at its governing member), the capacity design's checks, the joint checks,
    the 18.6/18.7 group detailing and column confinement rows (Grouped_Record's evaluators on the final design),
    drift and stability, vertical regularity, torsional amplification, the slab rows and the reinforcement
    selection. Nothing is filtered by status.
    """
    from Design import Grouped_Record as records
    from Design.SMRF_Common import assertion_provenance_valid
    from Design.SMRF_Demands import seismic_design_category
    rows = []

    def add(family, items):
        rows.extend(contract.compact_row(family, item) for item in items)
    strength_clause = "ACI 318-19 9.5, 10.5, 21, 22: design strength at least the required strength (DCR <= 1.0)"
    for gid, result in sorted(strengths["groups"].items()):
        if result["member_type"] == "column":
            demand = max(result["pm"]["dcr"], result["shear"]["dcr"])
            details = {"pm": {k: result["pm"].get(k) for k in ("dcr", "member_tag", "combination")},
                       "shear": {k: result["shear"].get(k) for k in ("dcr", "member_tag", "combination")}}
        else:
            demand = max(result["flexure_positive"]["dcr"], result["flexure_negative"]["dcr"], result["shear"]["dcr"])
            details = {key: {k: result[key].get(k) for k in ("dcr", "member_tag", "combination")}
                       for key in ("flexure_positive", "flexure_negative", "shear")}
        add("strength", [make_check("member.strength", strength_clause, demand, 1.0, "<=", "DCR", gid, details=details)])
    add("capacity", capacity["checks"])
    add("joints", joint_checks)
    view = {"materials": records._materials(cfg, state), "detailing": records.group_detailing(state, capacity),
            "geometry": driver._slab_geometry()}
    add("detailing", records.group_detailing_checks(view, state, capacity))
    add("detailing", [c for c in records._capacity_completion_checks(capacity, state)
                      if c["id"] in ("column.confinement_area_and_support", "detailing.mechanical_splice_product_qualification")])
    add("drift", drift_screen["checks"])
    try:
        sdc = seismic_design_category(sp.ASCE_SDS, sp.ASCE_SD1, sp.ASCE_S1, cfg.demands.risk_category)
    except (ValueError, TypeError):
        sdc = None
    verification = asdict(cfg.verification)
    asserted = (assertion_provenance_valid(verification) and verification.get("story_strength_model_verified") is True
                and ((strength or {}).get("applicability") or {}).get("status") == "verified")
    add("regularity", regularity_checks(vertical, sdc, strength_model_verified=asserted))
    if cfg.demands.accidental_torsion_ratio:
        add("torsion", [torsional_amplification_check(demand_basis), *torsional_classification_review(demand_basis)])
    if cfg.floor_analysis.transfer_to_frame:
        add("slab", slab_code_checks(slab, eligibility, cfg))
    add("reinforcement", [make_check("reinforcement.selected_for_every_group",
                                     "ACI 318-19 18.6.3, 18.7.4 (an admissible cage for every group)", len(exhausted), 0, "==",
                                     details={"exhausted_groups": list(exhausted)})])
    return rows


def evaluate_candidate(state, cfg, context=None, label="candidate", keep_actions=True):
    """Evaluate one grouped design completely and return its evaluation record.

    ``context`` carries what can be reused between candidates (floor transfers and strip evidence of
    unchanged floors) and counts the solves. The returned record has ``status`` ``evaluated`` or
    ``infeasible``. An evaluated candidate carries three separate results: ``candidate_screen_passed`` (the
    preliminary screen: no recorded failure in the screened families, a settled pass), ``code_check`` (the
    authoritative result of Design.Code_Check_Contract over ``code_checks``: every required check of the final
    design present once and passing, with known failures, unknown and missing checks and review dependencies
    listed apart) and ``feasible`` (that code check passed, the evaluation completed and the material objectives
    are available). Nonlinear performance is reported as not requested. The grouped design that results (cages
    and hoops selected for these sections) is under ``state``; it stays installed on return.
    """
    context = {} if context is None else context
    started = time.perf_counter()
    record = {"version": EVALUATION_VERSION, "label": label, "sections": _section_table(state),
              "seed_identity": state.identity(), "status": "evaluated", "feasible": False,
              "candidate_screen_passed": False, "stages": {}}
    solves = {"frame": 0, "floor_before": context.get("floor_solves", 0)}
    kept = {}                       # the inputs of the reinforcement pass, kept for an incomplete outcome (keep_actions)
    try:
        mg.install(state)
        _require_family_grades(state)
        slab = slab_stage(cfg, context)
        kept["slab"] = slab
        record["stages"]["slab"] = {"thickness_in": slab["slab"]["thickness_in"], "attempts": slab["attempts"],
                                    "distinct_floors": len(slab["description"]["distinct"]),
                                    "layout_selected": bool((slab["reinforcement"] or {}).get("layout")),
                                    "review_bundle": slab_review_bundle(slab["evidence"]),
                                    "assertion_scope": slab.get("assertion_scope"),
                                    # Under a slab procedure, the evidence each distinct floor earned (P1, 2026-11-01).
                                    "procedure": ((slab.get("evidence") or {}).get("procedure"),
                                                  (slab.get("evidence") or {}).get("procedure_instances"))}
        # The seed's complete arrangement under the slab in force (2026-11-01): mats the slab stage selected can move
        # beam rows, and the design model (model_period builds every beam's hinges from its rows, gravity beams
        # included), the pre-selection story strengths and everything after them read those rows. No beam can be
        # selected again before the frame is solved, so rows that do not fit stop the candidate here with the
        # structured outcome; nothing is priced on them. An arrangement that does not settle, or a malformed row record,
        # is the error it was.
        seed_audit = bar_layers.row_audit(state)
        bar_layers.raise_arrangement_errors(seed_audit)
        record["stages"]["slab"]["seed_rows_fit"] = seed_audit["valid"]
        if not seed_audit["valid"]:
            raise CageArrangementError(*_seed_arrangement_outcome(state, seed_audit, slab))
        transfers = None
        if slab["transfer"] is not None:
            from Design.SMRF_Floor_Transfer import validate_floor_transfers_by_floor
            area = mp.floor_area_in2() / 144.0
            transfers = validate_floor_transfers_by_floor(slab["transfer"], driver._slab_geometry(), slab["description"],
                                                          sp.SLAB_THICKNESS_IN, sp.floor_dead_load_ksf() * area,
                                                          sp.FLOOR_LIVE_LOAD_KSF * area)
        kept["transfers"] = transfers
        from Design.SMRF_Demands import REDUNDANCY_FACTOR_STRENGTH, live_load_patterns, strength_load_combinations
        try:
            # The period is the entire structure's under every policy (the continuous frame). Torsion assessment,
            # strength combinations and drift on the design model of the installed policy: the continuous frame
            # under the band policy, the released (perimeter-SFRS) frame under the system policy, where the
            # seismic combinations superpose by member system (Design.Perimeter_Systems).
            period = systems.model_period(state)
            solves["frame"] += 1
            strength = story_strength_distribution(cfg)
            torsion = systems.torsion_assessment(period, cfg, strength, state) if cfg.demands.accidental_torsion_ratio else None
            solves["frame"] += 4 if torsion is not None else 0
            patterns = (live_load_patterns(sp.NUM_BAY_X, sp.NUM_BAY_Y)
                        if cfg.demands.live_load_patterning and sp.FLOOR_TRANSFER is not None else ())
            combinations = strength_load_combinations(sp.ASCE_SDS, redundancy_factor=REDUNDANCY_FACTOR_STRENGTH,
                                                      live_patterns=patterns)
            expected_ids = [c["id"] for c in combinations]
            actions, elf_used = [], None
            solver = systems.CombinationSolver(period, torsion, state)
            for combination in combinations:
                solved = solver.analyze(combination)
                elf = solved["elf"]
                elf_used = elf if elf is not None else elf_used
                actions.append({**combination, "analysis_succeeded": True, "axial_reference": "joint_faces",
                                "analysis_models": solved["models"], "members": solved["members"]})
            solves["frame"] += solver.solves
            # Deformation compatibility of the gravity members (ACI 318-19 18.14.2.1, ASCE 7-22 12.12.5): the design
            # displacement of the released frame imposed on the continuous frame with the 5.3 gravity combinations
            # that carry the vertical ground motion. It depends on the sections only, so it is solved once here and
            # the cages are selected on it; None under the band policy (every member is in the SFRS).
            compatibility = systems.compatibility_run(period, torsion, cfg, state)
            solves["frame"] += compatibility["solves"] if compatibility else 0
            kept.update(period=period, torsion=torsion, elf=elf_used, actions=actions, expected_ids=expected_ids,
                        compatibility=compatibility)
        except RuntimeError as exc:
            if "gravity analysis failed" not in str(exc).lower():
                raise
            raise CandidateInfeasible("gravity_analysis", str(exc)) from exc
        record["stages"]["demands"] = {"model_period_sec": period, "combinations": len(actions),
                                       "base_shear_kip": (elf_used or {}).get("base_shear_kip")}
        try:
            settled = reinforcement_pass(state, actions, expected_ids, cfg, transfers, compatibility=compatibility)
            # The slab completion context reads the installed perimeter hoops and column cores: re-price the
            # layout on the selected cages and, if it changed, the cages on the layout (bounded). The slab and
            # the cages are consistent only when a re-pricing on the final cages leaves the layout unchanged.
            layout_settled = slab["reinforcement"] is None
            for _ in range(SLAB_CAGE_PASSES):
                if slab["reinforcement"] is None:
                    break
                from Design.SMRF_Slab_Reinforcement import design_slab_reinforcement
                inputs = driver._slab_strength_inputs_from_state(slab["slab"], cfg)
                repriced = design_slab_reinforcement(inputs, slab["evidence"], None,
                                                     slab_completion_context(slab["slab"], cfg, settled["state"]))
                changed = repriced.get("layout") != slab["reinforcement"].get("layout")
                slab["reinforcement"] = repriced
                sp.SLAB_REINFORCEMENT = repriced
                if not changed:
                    layout_settled = True
                    break
                settled = reinforcement_pass(settled["state"], actions, expected_ids, cfg, transfers, compatibility=compatibility)
        except SectionAxialDomainError as exc:
            raise CandidateInfeasible("capacity_design", f"a factored axial load is outside a section's strength domain: {exc}",
                                      {"exception": "SectionAxialDomainError", "group_id": getattr(exc, "group_id", None)}) from exc
        state = settled["state"]
        mg.install(state)
        # A layout that was selected before the cages and is lost on the final cages leaves the slab without one.
        layout_kept = not record["stages"]["slab"]["layout_selected"] or bool((slab["reinforcement"] or {}).get("layout"))
        # The slab evidence on the final cages (2026-10-30): the record in force must have been priced on the final
        # floor context, and the strong-column beams read what that record establishes
        # (SMRF_Slab_Reinforcement.slab_strength_eligibility). When that differs from what the last capacity design
        # read, the capacity design is rebuilt once on the final state and its joints evaluated again.
        from Design.SMRF_Joints import evaluate_joints
        from Design.SMRF_Slab_Reinforcement import slab_strength_eligibility
        context_now = slab_completion_context(slab["slab"], cfg, state) if slab["reinforcement"] is not None else None
        eligibility = (capacity_design.default_slab_eligibility() if slab["reinforcement"] is None
                       else slab_strength_eligibility(slab["reinforcement"], context_now))
        capacity, joint_checks, capacity_current = settled["capacity"], settled["joint_checks"], True
        if ((capacity.get("slab_eligibility_signature"), capacity.get("slab_eligibility_reason_code"))
                != (eligibility.get("signature"), eligibility.get("reason_code"))):
            capacity = capacity_design.build_group_capacity_design(actions, expected_ids, cfg, transfers, state,
                                                                   compatibility=compatibility, slab_eligibility=eligibility)
            joint_checks = evaluate_joints(joint_check_inputs(capacity, state))
            capacity_current = not capacity_design.hoop_updates(capacity, state)
            record["stages"]["slab"]["capacity_rebuilt_on_final_slab_evidence"] = True
        slab_current = eligibility.get("context_current") is not False
        strengths = checks.strength_checks(actions, cfg, state)
        drift_screen = systems.drift_screen(period, torsion, cfg, state)
        solves["frame"] += 2
        induced = systems.gravity_induced_from_capacity(capacity)      # the gravity groups' branches on the final cages
        record["stages"]["compatibility"] = systems.compatibility_stage(compatibility, induced)
        strength = story_strength_distribution(cfg)                # on the selected cages
        ledger = mp.weight_ledger()
        vertical = vertical_regularity(drift_screen, elf_used, strength, ledger)
        demand_basis = driver._demand_basis(cfg, torsion, elf_used, drift_screen, strength,
                                            grouped_regularity(strength, vertical))
        scwb = scwb_constraint(joint_checks, capacity)
        exhausted = settled["selection"]["exhausted_groups"]
        joint_failed = [c for c in joint_checks if c["status"] == "fail"]
        worst = strengths["worst"]
        hard_max = cfg.dcr.dcr_hard_max
        constraints = {
            "strength_within_ceiling": max(worst["beam"], worst["column"]) <= hard_max,
            "reinforcement_selected_for_every_group": not exhausted,
            # Settled means the longitudinal selection and the hoops both stopped changing in the same pass.
            "reinforcement_settled": bool(settled["settled"]),
            "longitudinal_selection_settled": bool(settled["selection_settled"]),
            "hoops_settled": bool(settled["hoops_settled"]),
            # The slab layout was re-priced on the final cages and did not change (true when no layout is designed).
            "slab_layout_settled": bool(layout_settled),
            "slab_layout_kept": layout_kept,
            "capacity_design_accepted": capacity["accepted"],
            "capacity_design_failed_checks": [{"id": c["id"], "location": c.get("location", ""), "status": c["status"],
                                               "demand": c.get("demand"), "capacity": c.get("capacity"), "units": c.get("units", "")}
                                              for c in capacity["checks"] if c["status"] != "pass"],
            "joint_checks_failed": [{"id": c["id"], "location": c.get("location", ""), "demand": c.get("demand"),
                                     "capacity": c.get("capacity")} for c in joint_failed][:60],
            "joint_scwb": {k: scwb[k] for k in ("evaluated", "all_pass", "no_known_failure", "complete", "counts",
                                                "required", "missing_count", "duplicates", "unexpected",
                                                "exceptions_18_7_3_1", "min_ratio_provided", "unknown_reason_codes",
                                                "semantics")},
            # The slab record was priced on the final floor context, and the capacity design read the slab evidence of
            # the final state (rebuilt once when it had not).
            "slab_evidence_current": bool(slab_current),
            "capacity_evidence_current": bool(capacity_current),
            "slab": {key: eligibility.get(key) for key in ("layout_selected", "slab_checks_complete", "slab_checks_passed",
                                                          "interior_development", "perimeter_anchorage", "credit",
                                                          "reason_code", "reason", "failed_checks", "unknown_checks",
                                                          "review_dependencies")},
            "transitions_supported": all(t["supported"] for t in capacity["transitions"].values()),
            "gravity_frame_branches": {gid: row["branch"] for gid, row in induced.items()},
            "gravity_frame_failed_checks": [c["id"] + (f"@{c['location']}" if c.get("location") else "")
                                            for c in ((capacity.get("gravity_frame") or {}).get("checks") or []) if c["status"] == "fail"],
            "drift_accepted": drift_screen["accepted"],
            "drift_failed_checks": [c["id"] + (f"@{c['location']}" if c.get("location") else "")
                                    for c in drift_screen["checks"] if c["status"] == "fail"],
            "exhausted_groups": exhausted,
            # Beams whose cages that reach the demand do not thread their columns: room between the column bars, not
            # strength, so the search moves the columns the beam threads.
            "lane_limited_beam_groups": sorted({gid for item in settled["selection"]["passes"][-1]["beams"].values()
                                                if item.get("lane_limited")
                                                for gid in (item.get("groups") or [item.get("group_id")]) if gid}),
            # Beams whose strength threading constrains with a flexural deficit left (Group_Selection threading evidence,
            # authenticated): a stronger admissible cage does not thread the cores the evidence names, and the strongest
            # threading cage is short. lane_limited above keeps its own meaning (a cage that reaches the demand exists).
            "threading_constrained_beam_groups": sorted({gid for item in settled["selection"]["passes"][-1]["beams"].values()
                                                         if (item.get("threading") or {}).get("threading_constrained_deficit") is True
                                                         and item["threading"].get("authenticated") is True
                                                         for gid in (item.get("groups") or [item.get("group_id")]) if gid}),
            "beam_threading_evidence": selection.threading_constraint_summary(settled["selection"]),
            # The installed cages that do not thread a joint, whatever the selector reported: their lanes, what blocks
            # them (a core cage, or a column transition by both groups) and the look-ahead, for the threading rows.
            "beam_lane_evidence": selection.installed_lane_evidence(state, cfg),
            # The joint-aware cage coordination of the final selection (what was tried, adopted or why it failed).
            "cage_coordination": selection.coordination_summary(settled["selection"]),
            # The terminal cage arrangement of the final selection: valid as selected, or closed by which route.
            "terminal_arrangement": selection.terminal_arrangement_summary(settled["selection"]),
            # How the reinforcement passes ended: settled, a proven cycle, or the pass budget (2026-11-01), and how a
            # cycle was resolved: only a verified resolution of this final state completes the selection flags
            # (Code_Check_Contract.numerical_completion); the flags above stay as the passes ended.
            "reinforcement_stop": settled.get("stop"), "reinforcement_cycle": settled.get("cycle"),
            "reinforcement_resolution": {"kind": settled.get("resolution"),
                                         "state_identity": ((settled.get("cycle") or {}).get("resolution") or {})
                                         .get("adopted_state_identity")},
            # For a column line without a feasible cage chain: the band at which the chain breaks.
            "exhausted_column_lines": {location: {"blocked_at": item.get("blocked_at"), "blocked_by": item.get("blocked_by")}
                                       for location, item in settled["selection"]["passes"][-1]["columns"].items()
                                       if item.get("exhausted")},
        }
        # The preliminary screen (the former ``feasible``): no recorded failure in the families it reads and a settled
        # pass. It is NOT the code check: an unevaluated check is not a failure here, so it can pass with required
        # evidence missing. The search's growth and reduction steer on it; acceptance is ``feasible`` below.
        # Its settle term follows the contract: a settled pass, or a verified cycle resolution of this same final state
        # (Code_Check_Contract.numerical_completion), so the screen and the code check read the reinforcement alike.
        selection_complete = constraints["reinforcement_settled"] or contract.numerical_completion(constraints,
                                                                                                    state.identity())[3]
        screen = bool(constraints["strength_within_ceiling"] and not exhausted and selection_complete
                      and constraints["slab_layout_settled"] and layout_kept and slab_current and capacity_current
                      and capacity["accepted"] and not joint_failed and drift_screen["accepted"])
        constraints["candidate_screen_passed"] = screen
        amounts = quantities(state, slab["slab"], slab["reinforcement"], capacity)
        # The authoritative code check (Design.Code_Check_Contract): every check the final design requires, from its
        # own topology, present once and passing; the evaluation completed; the material objectives available.
        rows = candidate_code_checks(state, cfg, strengths=strengths, capacity=capacity, joint_checks=joint_checks,
                                     drift_screen=drift_screen, vertical=vertical, strength=strength,
                                     demand_basis=demand_basis, slab=slab, eligibility=eligibility, exhausted=exhausted)
        expected = contract.expected_inventory(state, cfg)
        code_check = contract.code_check_result(rows, expected)
        complete = contract.numerical_completion(constraints, state.identity())[2]
        feasible = contract.accepted_for_code_design(code_check, complete, contract.objective_availability(amounts))
        constraints["code_check_status"] = code_check["status"]
        record.update({
            "feasible": feasible, "candidate_screen_passed": screen, "constraints": constraints, "quantities": amounts,
            "code_check": code_check,
            "code_checks": {"contract_version": contract.CONTRACT_VERSION, "scope": contract.SCOPE["id"],
                            "state_identity": state.identity(),
                            "realized_identity": contract.realized_identity(state.identity(), slab["slab"]["thickness_in"],
                                                                            slab["reinforcement"]),
                            "expected_counts": expected["counts"], "scwb_not_required": expected["scwb_not_required"],
                            "rows": rows},
            "nonlinear_performance": dict(contract.NONLINEAR_STATUS),
            "margins": _margins(strengths, capacity, joint_checks, drift_screen),
            "dcr": {"column": worst["column"], "beam": worst["beam"]},
            "result_identity": state.identity(),
            "sections": _section_table(state),
        })
        record["stages"]["reinforcement"] = {"log": settled["log"], "scwb_margin": settled["scwb_margin"],
                                             "settled": settled["settled"], "selection_settled": settled["selection_settled"],
                                             "hoops_settled": settled["hoops_settled"],
                                             "evidence_rebuilt_on_final_design": settled["evidence_rebuilt_on_final_design"],
                                             "stop": settled.get("stop"), "cycle": settled.get("cycle"),
                                             "resolution": settled.get("resolution"),
                                             "slab_layout_settled": bool(layout_settled), "slab_layout_kept": layout_kept}
        if keep_actions:
            record["evidence"] = {"state": state, "slab": slab, "transfers": transfers, "period": period, "torsion": torsion,
                                  "elf": elf_used, "actions": actions, "expected_ids": expected_ids, "strengths": strengths,
                                  "capacity": capacity, "joint_checks": joint_checks, "selection": settled["selection"],
                                  "drift_screen": drift_screen, "demand_basis": demand_basis, "ledger": ledger,
                                  "vertical": vertical, "compatibility": compatibility, "gravity_induced": induced}
        record["state"] = state
    except BeamBarRowsError as exc:
        bar_rows_outcome(record, exc, state, kept, keep_actions)
    except CandidateInfeasible as exc:
        record.update(status="infeasible", feasible=False, failed_stage=exc.stage, reason=exc.reason,
                      failure_evidence=_json_safe(exc.evidence))
        record["state"] = mg.active()
    record["elapsed_seconds"] = time.perf_counter() - started
    record["solves"] = {"frame": solves["frame"], "floor": context.get("floor_solves", 0) - solves["floor_before"]}
    return record


def _rows_fit_without_mats(state):
    """(fits, problem) of ``state``'s complete arrangement with the selected slab mats masked (the same outer axis, no
    mat layers): fits is True or False, or None when no layout is in force or the masked audit gives no answer (its
    layers do not settle, or it raises; ``problem`` says which). A mat can push a crossing layer that then pushes the
    failing row, so the failing rows' own controls do not show every row a mat moved. Advisory only: it labels an
    outcome and never stops one."""
    record = sp.SLAB_REINFORCEMENT
    layout = (record or {}).get("layout")
    if not layout:
        return None, None
    sp.SLAB_REINFORCEMENT = {**record, "layout": {**layout, "layers": {}}}
    try:
        with mg.installed(state):
            audit = bar_layers.row_audit(state)
    except Exception as exc:                          # noqa: BLE001 -- an advisory label never replaces the outcome
        return None, f"the masked audit raised {type(exc).__name__}: {str(exc)[:300]}"
    finally:
        sp.SLAB_REINFORCEMENT = record
    if audit.get("settle_error"):
        return None, f"the masked layers do not settle: {audit['settle_error']}"
    return bool(audit["valid"]), None


def _seed_arrangement_outcome(state, audit, slab):
    """(message, evidence, detail) of a seed whose complete arrangement does not fit under the slab in force. The
    mats are named as the cause only when the arrangement fits with them masked (_rows_fit_without_mats); a mat named
    in a failing row's controls of an arrangement that does not fit without the mats either is a contributing mat, not
    the cause; otherwise the rows do not fit beside the crossing beams."""
    groups = sorted({item["group"] for item in audit["invalid"]})
    named = any(control and control.get("by") == "slab_mat" for item in audit["invalid"]
                for control in item.get("controls") or [])
    layout = bool((slab.get("reinforcement") or {}).get("layout"))
    fits_masked, masked_problem = _rows_fit_without_mats(state) if layout else (None, None)
    by_mats = layout and fits_masked is True
    contributing = layout and not by_mats and named
    classification = ("seed_rows_do_not_fit_the_selected_slab_mats" if by_mats else
                      "seed_rows_do_not_fit_with_or_without_the_slab_mats" if contributing else "seed_rows_do_not_fit")
    beside = (" and the selected slab mats" if by_mats else
              " (a selected slab mat moved them, and they do not fit without the mats either)" if contributing else "")
    reason = (f"the seed's beam bar rows do not fit beside the crossing beams{beside} "
              f"({', '.join(groups[:6])}); no beam is selected before the frame is solved, so the candidate stops before "
              "any strength is priced on them: an incomplete outcome, not proof that no cage of the declared domain fits")
    evidence = {"outcome": "seed_arrangement_invalid", "classification": classification, "reason": reason,
                "group_ids": groups, "slab_layout_selected": layout,
                "mats_named_by_failing_rows": named, "rows_fit_with_mats_masked": fits_masked,
                **({"masked_audit_problem": masked_problem} if masked_problem else {}),
                "rejected_state": {"state_identity": audit["state_identity"], "invalid_rows": audit["invalid"],
                                   "role": "the submitted state under the slab in force: no strength result belongs to it"},
                "last_geometrically_valid_state": None,
                "scope": ("the design model (its beam hinges are calibrated from every beam's rows, gravity beams "
                          "included) and the pre-selection story strengths need the seed's rows; the search's move is a "
                          "deeper or wider beam or another seed cage"),
                "call_site": "Design.Grouped_Design.evaluate_candidate (seed arrangement after the slab stage)"}
    return reason, evidence, {"rejected_state": state, "selection_input_state": state}


def bar_rows_outcome(record, exc, state, kept, keep_actions):
    """The evaluation record of a candidate stopped by bar rows that do not fit (BeamBarRowsError): a candidate outcome,
    not an error; the search steps the beams named. On the reinforcement pass it is the incomplete outcome of the
    terminal-arrangement closure (CageArrangementError: the rejected and the last geometrically valid state apart,
    neither with a strength result); on the seed's own rows the groups named. The failing call site is kept, and with
    ``keep_actions`` what replays the selector without a solve: ``kept`` (the solved actions and the reinforcement pass's
    other inputs), the failing selection's input state and full report, the rejected state and the last valid designs."""
    detail = getattr(exc, "detail", None) or {}
    rejected = detail.get("rejected_state") or mg.active() or state
    evidence = getattr(exc, "evidence", None)
    failure = ({"exception": type(exc).__name__, **evidence} if evidence else
               {"exception": type(exc).__name__, "group_ids": invalid_beam_row_groups(rejected)})
    # Where it was raised, beside the call site an outcome declares for itself (kept, not replaced).
    declared = failure.get("call_site") if isinstance(failure.get("call_site"), str) else None
    failure["call_site"] = {**_call_site(exc), **({"declared": declared} if declared else {})}
    record.update(status="infeasible", feasible=False, failed_stage="bar_rows",
                  reason=f"beam bar rows do not fit beside the crossing beams: {exc}",
                  failure_evidence=_json_safe(failure))
    record["state"] = rejected
    if keep_actions and kept:
        record["evidence"] = {"incomplete": True, **kept, "selection_input_state": detail.get("selection_input_state"),
                              "rejected_state": rejected, "selection": detail.get("selection_report"),
                              "last_geometrically_valid_designs": detail.get("last_geometrically_valid_designs")}
    return record


def _call_site(exc):
    """Where an exception was raised in the package code: the innermost Design/Model frame and the chain to it
    (module.function:line), from its traceback."""
    rows = [f"{os.path.splitext(os.path.basename(frame.filename))[0]}.{frame.name}:{frame.lineno}"
            for frame in traceback.extract_tb(exc.__traceback__)
            if os.sep + "Design" + os.sep in frame.filename or os.sep + "Model" + os.sep in frame.filename]
    return {"raised_at": rows[-1] if rows else None, "chain": rows[-8:]}


def install_evaluation(evaluation):
    """Put an evaluated candidate back in force exactly: its member groups, slab, floor transfer, strip evidence
    and slab layout. The rollback of a rejected trial: a trial leaves its own slab state in Structure_Parameters,
    and the member groups alone would then sit on another design's floor transfer."""
    evidence = evaluation.get("evidence")
    if evaluation.get("status") != "evaluated" or evidence is None:
        raise mg.GroupedStateError("Only an evaluated candidate that kept its evidence can be reinstalled.")
    slab = evidence["slab"]
    mg.install(evaluation["state"])
    driver._apply_slab(slab["slab"])
    sp.FLOOR_TRANSFER = slab["transfer"]
    sp.SLAB_ACTIONS = slab["evidence"]
    sp.SLAB_REINFORCEMENT = slab["reinforcement"]
    return evaluation


def _require_family_grades(state, policy=None):
    """One grade per grade family over the full height, the gravity families on their own ladder
    (Design.Concrete_Grades; a saved record names the policy it was written under, the legacy rule when none)."""
    from Design import Concrete_Grades as cg
    cg.require(state, cg.POLICY_ID if policy is None else policy)


def _section_table(state):
    return {gid: {"b_in": d.b_in, "h_in": d.h_in, "fc_ksi": d.fc_ksi, "bar_size": d.bar_size, "top_bars": d.top_bars,
                  "bot_bars": d.bot_bars, "side_bars": d.side_bars, "stirrup_bar_size": d.stirrup_bar_size,
                  "stirrup_legs": d.stirrup_legs, "stirrup_spacing_in": d.stirrup_spacing_in,
                  "stirrup_legs_by_direction": d.legs_by_direction}
            for gid, d in sorted(state.designs.items())}


def _margins(strengths, capacity, joint_checks, drift_screen):
    """Least margin (1 - demand / capacity) of each check family, with where it occurs."""
    def least(rows):
        rows = [r for r in rows if r[0] is not None]
        if not rows:
            return None
        ratio, where = max(rows, key=lambda r: r[0])
        return {"demand_over_capacity": ratio, "margin": 1.0 - ratio, "at": where}

    def ratio(check):
        demand, cap = check.get("demand"), check.get("capacity")
        if demand is None or not cap or check.get("comparison") != "<=":
            return None
        return demand / cap

    groups = strengths["groups"]
    result = {
        "column_pm": least([(g["pm"]["dcr"], gid) for gid, g in groups.items() if g["member_type"] == "column"]),
        "beam_flexure": least([(max(g["flexure_positive"]["dcr"], g["flexure_negative"]["dcr"]), gid)
                               for gid, g in groups.items() if g["member_type"] != "column"]),
        "beam_capacity_shear_section": least([(ratio(c), c.get("location")) for c in capacity["checks"]
                                              if c["id"] == "beam.capacity_shear_section"]),
        "column_capacity_shear_section": least([(ratio(c), c.get("location")) for c in capacity["checks"]
                                                if c["id"] == "column.capacity_shear_section"]),
        "joint_shear": least([(ratio(c), c.get("location")) for c in capacity["checks"] if c["id"] == "joint.shear_screen"]),
        # An 18.7.3.1 exception row compares Pu with Ag f'c / 10, another quantity: not a strong-column margin.
        "scwb": least([(ratio(c), c.get("location")) for c in joint_checks if c["id"] == "scwb" and c["status"] != "not_evaluated"
                       and not (c.get("details") or {}).get("roof_exemption_applied")]),
        "story_drift": least([(ratio(c), c.get("location")) for c in drift_screen["checks"] if c["id"] == "demands.story_drift"]),
        "stability": least([(ratio(c), c.get("location")) for c in drift_screen["checks"] if c["id"] == "demands.stability"]),
    }
    return result


def _json_safe(value):
    """A JSON-serialisable copy: member objects, designs and states become their records; nothing is dropped silently."""
    if isinstance(value, mg.GroupedDesign):
        return value.to_record()
    if isinstance(value, mg.MemberDesign):
        return mg.design_record(value)
    if isinstance(value, mg.ResolvedMember):
        return {"member_tag": value.member_tag, "group_id": value.group_id}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def public_evaluation(evaluation):
    """The evaluation without its in-memory evidence: what a candidate log keeps for every candidate.

    The code-check result is kept whole; of its rows the log keeps every row that did not pass (failures, unknown,
    review and malformed rows, with their reasons) and counts the passing ones it leaves out. The complete rows stay
    on the in-memory evaluation, which is what a search validates (Code_Check_Contract.assess_evaluation) and what the
    design record saves (code_check_rows)."""
    public = {k: v for k, v in evaluation.items() if k not in ("evidence", "state")}
    block = public.get("code_checks")
    if isinstance(block, dict) and isinstance(block.get("rows"), list):
        rows = block["rows"]
        public["code_checks"] = {**{k: v for k, v in block.items() if k != "rows"},
                                 "rows_not_passing": [r for r in rows if r.get("status") != "pass"],
                                 "passing_rows_omitted_from_log": sum(1 for r in rows if r.get("status") == "pass")}
    return _json_safe(public)
