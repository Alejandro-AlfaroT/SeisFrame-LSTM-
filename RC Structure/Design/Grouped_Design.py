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

Nothing here asserts a verification. The story-strength model is provisional (review item M1), the
transition rules are declared rules awaiting review, and Design.SMRF_Qualification
.GENERATION_RELEASE_READY is untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict

import openseespy.opensees as ops

import Structure_Parameters as sp
from Design import Design_Driver as driver
from Design import Group_Capacity as capacity_design
from Design import Group_Checks as checks
from Design import Group_Selection as selection
from Design import SMRF_Floor_Sections as floor_sections
from Design.SMRF_Common import SectionAxialDomainError, make_check, not_evaluated, summarize_checks
from Model import Member_Bar_Layers as bar_layers
from Model import Member_Groups as mg
from Model import Member_Properties as mp

SCHEMA = "rc_smrf_grouped_candidate_v1"
EVALUATION_VERSION = "grouped_candidate_evaluation_v1"
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
    policy = {**asdict(cfg.slab), "fy_ksi": sp.FY_KSI, "concrete_unit_weight_kcf": sp.CONCRETE_UNIT_WEIGHT_KCF}
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
            "two_way_shear_path_assessed": (cfg.slab_actions.all_asserted()
                                             and cfg.slab_actions.two_way_shear_path_assessed is True),
            "columns_at_beam_intersections": True,
            "beam_clear_cover_in": sp.BEAM_CLEAR_COVER_IN,
            "beam_hoop_diameter_in": max(sp.rebar_diameter(m.design.stirrup_bar_size) for m in perimeter),
            "fc_beam_ksi": floor_sections.for_floor(1)["fc_beam_ksi"]}


def _slab_evidence(cfg, slab, description, inputs, context):
    """Strip demand evidence of every distinct floor (refined when a plan is declared), cached by floor."""
    from Design.SMRF_Slab_Actions import build_slab_action_evidence
    from Design.SMRF_Slab_Refinement import build_refined_slab_action_evidence, combine_floor_evidence
    plan = cfg.floor_analysis.slab_refinement
    if plan is None and cfg.slab_actions.all_asserted():
        raise ValueError("Asserted slab actions require FloorAnalysisConfig.slab_refinement (an explicit mesh plan or a "
                         "named recipe such as Design.Config.PROBE_SLAB_REFINEMENT): single-mesh slab demands cannot be "
                         "verified, so no slab reinforcement could be selected.")
    cache = context.setdefault("slab_evidence", {})
    slab_key = hashlib.sha256(json.dumps([{k: slab[k] for k in ("thickness_in", "concrete_fc_ksi", "concrete_unit_weight_kcf",
                                                                "superimposed_dead_load_ksf")}, inputs, plan,
                                          asdict(cfg.slab_actions), sp.FLOOR_LIVE_LOAD_KSF,
                                          cfg.floor_analysis.transfer_mesh_per_bay],
                                         sort_keys=True, default=str).encode("utf-8")).hexdigest()
    floors = []
    for sha, sections, served in floor_sections.distinct_floors(description):
        key = (sha, slab_key)
        if key not in cache:
            ops.wipe()
            if plan is None:
                cache[key] = build_slab_action_evidence(slab, driver._slab_geometry(), sections, sp.FLOOR_LIVE_LOAD_KSF, inputs,
                                                        mesh_per_bay=cfg.floor_analysis.transfer_mesh_per_bay,
                                                        uniform_all_floors=False, assertions=asdict(cfg.slab_actions))
            else:
                cache[key] = build_refined_slab_action_evidence(slab, driver._slab_geometry(), sections, sp.FLOOR_LIVE_LOAD_KSF,
                                                                inputs, plan, assertions=asdict(cfg.slab_actions))
            context["floor_solves"] = context.get("floor_solves", 0) + 1
        floors.append({"floor_sections_sha256": sha, "floors": served, "evidence": cache[key]})
    return combine_floor_evidence(floors, sp.NUM_FLOOR)


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
        if record["layout"] is not None or not cfg.slab_actions.all_asserted():
            attempt["status"] = "layout_selected" if record["layout"] is not None else "actions_not_asserted_no_layout"
            return {"slab": slab, "description": description, "transfer": transfer, "evidence": evidence,
                    "reinforcement": record, "attempts": attempts}
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
        minimum = slab["thickness_in"] + cfg.slab.thickness_increment_in
        if minimum > cfg.slab.maximum_thickness_in + 1e-9:
            raise CandidateInfeasible("slab_reinforcement",
                                      "no admissible slab reinforcement layout at any thickness of the declared range",
                                      {"attempts": attempts, "reasons": reasons})


# ---- story-by-story regularity evidence ---------------------------------------------------------------------
def story_strength_distribution(cfg):
    """Frame-line story strengths of every story from the beams actually installed there (provisional model).

    The uniform model prices one set of four beam families and declares every story alike. Here each
    story's lines are summed from that floor's own beams (Model.Member_Properties.beam_strengths), so the
    one-sided fraction of ASCE 7-22 Table 12.3-1 is a value per story and direction, and the story
    strengths are available for the vertical comparison. The model is the same beam-sway approximation,
    with the same open questions (column, joint and shear limits, base and roof mechanisms): review item
    M1 stays open and nothing here asserts it.
    """
    from Design.SMRF_Demands import one_sided_strength_fraction
    stories, worst = [], 0.0
    for floor in range(1, sp.NUM_FLOOR + 1):
        entry = {"story": floor, "by_direction": {}}
        for direction, kind, lines, spans, spacing in (("x", "beam_x", sp.NUM_BAY_Y + 1, sp.NUM_BAY_X, sp.BAY_Y),
                                                       ("y", "beam_y", sp.NUM_BAY_X + 1, sp.NUM_BAY_Y, sp.BAY_X)):
            positions, strengths, groups = [], [], []
            for line in range(lines):
                total, group = 0.0, None
                for span in range(spans):
                    beam = mg.beam_at(kind, floor, span, line) if kind == "beam_x" else mg.beam_at(kind, floor, line, span)
                    values = mp.beam_strengths(beam)
                    total += (values["hogging"] + values["sagging"]) / sp.STORY_H
                    group = beam.group_id
                positions.append(line * spacing)
                strengths.append(total)
                groups.append(group)
            fraction, detail = one_sided_strength_fraction(positions, strengths, 0.5 * positions[-1])
            entry["by_direction"][direction] = {"line_positions_in": positions, "line_story_strength_kip": strengths,
                                                "line_groups": groups, "story_strength_kip": sum(strengths),
                                                "one_side_fraction": fraction, **detail}
            worst = max(worst, fraction)
        stories.append(entry)
    return {"one_side_fraction": worst, "stories": stories,
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


def grouped_regularity(strength, vertical):
    """The regularity block of the demand basis for a grouped design: nothing by construction over height."""
    return {"regular": None,
            "horizontal": "rectangular plan, frames on every grid line, rigid diaphragm without openings: no Type 2-5 "
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

    def left_open(name, clause, message, **details):
        """A not-evaluated check that keeps its reason and carries the evidence beside it."""
        check = not_evaluated(name, clause, message)
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
    return {"joints": capacity["scwb"]["joints"], **capacity["joint_evidence"], "through_bar_anchorage": through}


def reinforcement_pass(state, actions, expected_ids, cfg, transfers, max_passes=6, scwb_margins=(1.0, 1.03, 1.06)):
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
    """
    from Design.SMRF_Joints import evaluate_joints
    log, result = [], None
    for margin in scwb_margins:
        settled, evidence_rebuilt = False, False
        for index in range(max_passes):
            state, report = selection.select_reinforcement(state, actions, cfg, scwb_margin=margin)
            mg.install(state)
            capacity = capacity_design.build_group_capacity_design(actions, expected_ids, cfg, transfers, state)
            updates = capacity_design.hoop_updates(capacity, state)
            log.append({"scwb_margin": margin, "pass": index + 1, "selection_settled": bool(report["settled"]),
                        "hoops_settled": not updates, "exhausted_groups": report["exhausted_groups"],
                        "hoops_changed": sorted(updates)})
            if not updates and report["settled"]:
                settled = True
                break
            if updates:
                state = state.with_designs(updates)
                mg.install(state)
        if not settled and updates:
            # The last capacity design was made before these hoops went in: rebuild it on the installed design.
            capacity = capacity_design.build_group_capacity_design(actions, expected_ids, cfg, transfers, state)
            evidence_rebuilt = True
        joint_checks = evaluate_joints(joint_check_inputs(capacity, state))
        scwb = [c for c in joint_checks if c["id"] == "scwb"]
        failed = [c for c in scwb if c["status"] == "fail"]
        result = {"state": state, "capacity": capacity, "joint_checks": joint_checks, "selection": report,
                  "settled": settled, "selection_settled": bool(report["settled"]),
                  "hoops_settled": bool(log) and log[-1]["hoops_settled"] and not evidence_rebuilt,
                  "evidence_rebuilt_on_final_design": evidence_rebuilt, "scwb_margin": margin, "log": log,
                  "scwb": {"evaluated": any(c["status"] != "not_evaluated" for c in scwb), "all_pass": not failed,
                           "counts": {status: sum(1 for c in scwb if c["status"] == status)
                                      for status in ("pass", "fail", "not_evaluated")},
                           "min_ratio_provided": min((c["details"]["ratio_provided"] for c in scwb
                                                      if c["status"] != "not_evaluated"), default=None),
                           "ratio_required": sp.SCWB_RATIO_MIN}}
        if not failed or report["exhausted_groups"]:
            break
    return result


# ---- one candidate --------------------------------------------------------------------------------------------
def evaluate_candidate(state, cfg, context=None, label="candidate", keep_actions=True):
    """Evaluate one grouped design completely and return its evaluation record.

    ``context`` carries what can be reused between candidates (floor transfers and strip evidence of
    unchanged floors) and counts the solves. The returned record has ``status`` ``evaluated`` or
    ``infeasible``; an evaluated candidate carries ``feasible`` (every acceptance constraint met) and its
    ``constraints``, ``quantities`` and ``margins``. The grouped design that results (cages and hoops
    selected for these sections) is under ``state``; it stays installed on return.
    """
    context = {} if context is None else context
    started = time.perf_counter()
    record = {"version": EVALUATION_VERSION, "label": label, "sections": _section_table(state),
              "seed_identity": state.identity(), "status": "evaluated", "feasible": False, "stages": {}}
    solves = {"frame": 0, "floor_before": context.get("floor_solves", 0)}
    try:
        mg.install(state)
        _require_common_grades(state)
        slab = slab_stage(cfg, context)
        record["stages"]["slab"] = {"thickness_in": slab["slab"]["thickness_in"], "attempts": slab["attempts"],
                                    "distinct_floors": len(slab["description"]["distinct"]),
                                    "layout_selected": bool((slab["reinforcement"] or {}).get("layout"))}
        transfers = None
        if slab["transfer"] is not None:
            from Design.SMRF_Floor_Transfer import validate_floor_transfers_by_floor
            area = mp.floor_area_in2() / 144.0
            transfers = validate_floor_transfers_by_floor(slab["transfer"], driver._slab_geometry(), slab["description"],
                                                          sp.SLAB_THICKNESS_IN, sp.floor_dead_load_ksf() * area,
                                                          sp.FLOOR_LIVE_LOAD_KSF * area)
        from Design.SMRF_Demands import REDUNDANCY_FACTOR_STRENGTH, live_load_patterns, strength_load_combinations
        try:
            period = driver._model_period()
            solves["frame"] += 1
            strength = story_strength_distribution(cfg)
            torsion = driver._torsion_assessment(period, cfg, strength) if cfg.demands.accidental_torsion_ratio else None
            solves["frame"] += 4 if torsion is not None else 0
            patterns = (live_load_patterns(sp.NUM_BAY_X, sp.NUM_BAY_Y)
                        if cfg.demands.live_load_patterning and sp.FLOOR_TRANSFER is not None else ())
            combinations = strength_load_combinations(sp.ASCE_SDS, redundancy_factor=REDUNDANCY_FACTOR_STRENGTH,
                                                      live_patterns=patterns)
            expected_ids = [c["id"] for c in combinations]
            actions, elf_used = [], None
            for combination in combinations:
                elf = driver._analyze_combination(combination, period, torsion)
                solves["frame"] += 1
                elf_used = elf if elf is not None else elf_used
                actions.append({**combination, "analysis_succeeded": True, "axial_reference": "joint_faces",
                                "members": checks.capture_member_actions(combination["dead"])})
        except RuntimeError as exc:
            if "gravity analysis failed" not in str(exc).lower():
                raise
            raise CandidateInfeasible("gravity_analysis", str(exc)) from exc
        record["stages"]["demands"] = {"model_period_sec": period, "combinations": len(actions),
                                       "base_shear_kip": (elf_used or {}).get("base_shear_kip")}
        try:
            settled = reinforcement_pass(state, actions, expected_ids, cfg, transfers)
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
                settled = reinforcement_pass(settled["state"], actions, expected_ids, cfg, transfers)
        except SectionAxialDomainError as exc:
            raise CandidateInfeasible("capacity_design", f"a factored axial load is outside a section's strength domain: {exc}",
                                      {"exception": "SectionAxialDomainError", "group_id": getattr(exc, "group_id", None)}) from exc
        state = settled["state"]
        mg.install(state)
        # A layout that was selected before the cages and is lost on the final cages leaves the slab without one.
        layout_kept = not record["stages"]["slab"]["layout_selected"] or bool((slab["reinforcement"] or {}).get("layout"))
        strengths = checks.strength_checks(actions, cfg, state)
        capacity = settled["capacity"]
        drift_screen = driver._drift_screen(period, torsion, cfg)
        solves["frame"] += 2
        strength = story_strength_distribution(cfg)                # on the selected cages
        ledger = mp.weight_ledger()
        vertical = vertical_regularity(drift_screen, elf_used, strength, ledger)
        demand_basis = driver._demand_basis(cfg, torsion, elf_used, drift_screen, strength,
                                            grouped_regularity(strength, vertical))
        scwb = settled["scwb"]
        exhausted = settled["selection"]["exhausted_groups"]
        joint_failed = [c for c in settled["joint_checks"] if c["status"] == "fail"]
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
            "joint_scwb": {k: scwb[k] for k in ("evaluated", "all_pass", "counts", "min_ratio_provided")},
            "transitions_supported": all(t["supported"] for t in capacity["transitions"].values()),
            "drift_accepted": drift_screen["accepted"],
            "drift_failed_checks": [c["id"] + (f"@{c['location']}" if c.get("location") else "")
                                    for c in drift_screen["checks"] if c["status"] == "fail"],
            "exhausted_groups": exhausted,
            # For a column line without a feasible cage chain: the band at which the chain breaks.
            "exhausted_column_lines": {location: {"blocked_at": item.get("blocked_at"), "blocked_by": item.get("blocked_by")}
                                       for location, item in settled["selection"]["passes"][-1]["columns"].items()
                                       if item.get("exhausted")},
        }
        feasible = (constraints["strength_within_ceiling"] and not exhausted and constraints["reinforcement_settled"]
                    and constraints["slab_layout_settled"] and layout_kept
                    and capacity["accepted"] and not joint_failed and drift_screen["accepted"])
        constraints["candidate_screen_passed"] = feasible
        amounts = quantities(state, slab["slab"], slab["reinforcement"], capacity)
        record.update({
            "feasible": feasible, "constraints": constraints, "quantities": amounts,
            "margins": _margins(strengths, capacity, settled["joint_checks"], drift_screen),
            "dcr": {"column": worst["column"], "beam": worst["beam"]},
            "result_identity": state.identity(),
            "sections": _section_table(state),
        })
        record["stages"]["reinforcement"] = {"log": settled["log"], "scwb_margin": settled["scwb_margin"],
                                             "settled": settled["settled"], "selection_settled": settled["selection_settled"],
                                             "hoops_settled": settled["hoops_settled"],
                                             "evidence_rebuilt_on_final_design": settled["evidence_rebuilt_on_final_design"],
                                             "slab_layout_settled": bool(layout_settled), "slab_layout_kept": layout_kept}
        if keep_actions:
            record["evidence"] = {"state": state, "slab": slab, "transfers": transfers, "period": period, "torsion": torsion,
                                  "elf": elf_used, "actions": actions, "expected_ids": expected_ids, "strengths": strengths,
                                  "capacity": capacity, "joint_checks": settled["joint_checks"], "selection": settled["selection"],
                                  "drift_screen": drift_screen, "demand_basis": demand_basis, "ledger": ledger,
                                  "vertical": vertical}
        record["state"] = state
    except CandidateInfeasible as exc:
        record.update(status="infeasible", feasible=False, failed_stage=exc.stage, reason=exc.reason,
                      failure_evidence=_json_safe(exc.evidence))
        record["state"] = mg.active()
    record["elapsed_seconds"] = time.perf_counter() - started
    record["solves"] = {"frame": solves["frame"], "floor": context.get("floor_solves", 0) - solves["floor_before"]}
    return record


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


def _require_common_grades(state):
    """One column grade and one beam grade in this revision (group-specific grades wait for joint material rules)."""
    for kind, groups in (("column", [g for g in state.groups.values() if g["member_type"] == "column"]),
                         ("beam", [g for g in state.groups.values() if g["member_type"] != "column"])):
        grades = {state.designs[g["group_id"]].fc_ksi for g in groups}
        if len(grades) != 1:
            raise mg.GroupedStateError(f"{kind} groups use different concrete grades {sorted(grades)}; this revision keeps one "
                                       f"{kind} grade for the whole frame.")


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
        "scwb": least([(ratio(c), c.get("location")) for c in joint_checks if c["id"] == "scwb" and c["status"] != "not_evaluated"]),
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
    """The evaluation without its in-memory evidence: what a candidate log keeps for every candidate."""
    return _json_safe({k: v for k, v in evaluation.items() if k not in ("evidence", "state")})
