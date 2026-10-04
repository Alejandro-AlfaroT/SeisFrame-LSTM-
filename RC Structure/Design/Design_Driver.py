"""
Design/Design_Driver.py
=======================

Searches for a frame design candidate and writes traceable SMRF qualification
evidence. This is NOT a complete building-code design or certification.

Why this exists
---------------
Redesign.run_iterative_redesign resizes longitudinal bars against a
gravity-only demand, and it was only ever wired into Main.py. The dataset
generator calls Ground_Motion_Main, which had no design step at all, so every
generated structure kept the Structure_Parameters defaults regardless of its
height or span. This driver closes both gaps:

  * demand comes from the ASCE 7-22 seismic combination (1.2D + 0.5L + E),
    not gravity alone, so member sizes respond to building height
  * the search covers section dimensions and concrete strength as well as
    reinforcement, via the ladders in Design/Section_Design.py

Design depends only on geometry and loads, never on the ground motion, so the
result is cached per structure and reused by every record and intensity run of
that case. Building it once instead of once per analysis is the difference
between designing a thousand structures and designing several thousand.

Unit system: kip, inch, ksi.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid

import openseespy.opensees as ops

import Structure_Parameters as sp
from Analysis.Gravity import run_gravity_analysis
from Analysis.Modal import run_modal_analysis
from Design.ACI_Checks import run_checks_phase1
from Model.IMK_Calibration import (
    column_gravity_axial,
    column_moment_at_axial,
    column_pm_nominal_for,
)
from RC_Design_Check import _col_steel_layers
from Design.Config import DesignConfig
from Design.Section_Design import (
    beam_ladder,
    column_ladder,
    nearest_rung_index,
    suggest_rung_index,
    suggest_beam_rung_for_moment,
    validate_rung,
)
from Loads.Gravity_Loads import apply_gravity_loads
from Loads.Seismic_ELF import apply_elf_loads
from Design.SMRF_Elastic import build_design_model as build_model
from RC_Design_Check import get_element_tags
from Redesign import apply_updates, column_cages_that_bond_exist, column_cages_that_thread_exist, redesign_steel


DESIGN_ARTIFACT_NAME = "design.json"
# v11 (2026-10-01): the record carries its risk basis (category, Ie, drift criterion) and the request
# identity carries the risk category and the nonlinear model profile, so a Risk II or other-profile
# artifact is refused, never relabelled.
DESIGN_SCHEMA_VERSION = "rc_smrf_candidate_v11_risk_basis_profile"

_STATE_KEYS = (
    "B_COL", "H_COL", "FC_COL_KSI", "B_BEAM", "H_BEAM", "FC_BEAM_KSI",
    "COL_BAR_SIZE", "COL_TOP_BARS", "COL_BOT_BARS", "COL_SIDE_BARS", "COL_BAR_AREA",
    "BEAM_BAR_SIZE", "BEAM_TOP_BARS", "BEAM_BOT_BARS", "BEAM_SIDE_BARS", "BEAM_BAR_AREA",
    "COL_STIRRUP_SPACING", "BEAM_STIRRUP_SPACING",
    "COL_STIRRUP_BAR_SIZE", "COL_STIRRUP_LEGS", "COL_STIRRUP_LEGS_BY_DIRECTION",
    "BEAM_STIRRUP_BAR_SIZE", "BEAM_STIRRUP_LEGS",
    "SLAB_THICKNESS_IN", "FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF", "SEISMIC_LIVE_LOAD_FRACTION",
    "BEAM_CLEAR_COVER_IN", "COL_CLEAR_COVER_IN", "AGGREGATE_MAX_SIZE_IN",
    "REINFORCEMENT_SPECIFICATION", "MATERIAL_EXPOSURE", "FLOOR_TRANSFER",
    "SLAB_REINFORCEMENT", "SLAB_ACTIONS", "BEAM_BAR_STACKING", "BEAM_BAR_MAX_LAYERS", "BEAM_BAR_LAYER_ORDER",
)


@contextlib.contextmanager
def _quiet():
    """Suppress OpenSees C-level banners emitted during repeated model builds."""
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = (os.dup(1), os.dup(2))
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for handle in saved:
            os.close(handle)
        os.close(devnull)


def _capture_state():
    return copy.deepcopy({key: getattr(sp, key) for key in _STATE_KEYS})


def _restore_state(state):
    for key, value in copy.deepcopy(state).items():
        setattr(sp, key, value)


def _apply_rung(rung, member_type):
    """Write one ladder rung into Structure_Parameters."""
    b, h, fc = rung
    if member_type == "column":
        sp.B_COL, sp.H_COL, sp.FC_COL_KSI = b, h, fc
    else:
        sp.B_BEAM, sp.H_BEAM, sp.FC_BEAM_KSI = b, h, fc


def _slab_geometry():
    return {"num_bay_x": sp.NUM_BAY_X, "num_bay_y": sp.NUM_BAY_Y,
            "num_floor": sp.NUM_FLOOR, "bay_x_in": sp.BAY_X,
            "bay_y_in": sp.BAY_Y, "story_h_in": sp.STORY_H}


class SlabDesignError(RuntimeError):
    """A slab design refusal that keeps its evidence (trials, failing checks, strip comparisons) for the run record."""

    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}


class SlabLayoutError(SlabDesignError):
    """The slab analysis converged but no reinforcement layout passes at this thickness (the thickness search's cue)."""

    def __init__(self, message, thickness_in, evidence=None):
        super().__init__(message, evidence)
        self.thickness_in = thickness_in


class SlabThicknessExhausted(SlabDesignError):
    """No supported thickness of the current beam rung admits a layout (the slab search's cue to step the beam)."""

    def __init__(self, message, trials, reason, evidence=None):
        super().__init__(message, evidence)
        self.trials = trials
        self.reason = reason


class SlabSearchExhausted(SlabDesignError):
    """No supported thickness of any deeper compatible beam rung admits a layout; nothing was assigned."""


def _select_slab(cfg, minimum_thickness_in=None):
    """Recompute one thickness for the current beam rung before any analysis.

    ``minimum_thickness_in`` raises the floor of the declared thickness ladder (the slab thickness search,
    screening repair item 3, 2026-10-02); the policy saved with the slab records the floor it was chosen from.
    """
    from Design.SMRF_Slab import choose_slab
    policy = {**asdict(cfg.slab), "fy_ksi": sp.FY_KSI, "concrete_unit_weight_kcf": sp.CONCRETE_UNIT_WEIGHT_KCF}
    if minimum_thickness_in is not None:
        policy["minimum_thickness_in"] = float(minimum_thickness_in)
    slab = choose_slab(
        _slab_geometry(),
        {"b_beam_in": sp.B_BEAM, "h_beam_in": sp.H_BEAM, "fc_beam_ksi": sp.FC_BEAM_KSI},
        policy)
    _apply_slab(slab)
    return slab


def _apply_slab(slab):
    sp.SLAB_THICKNESS_IN = slab["thickness_in"]
    sp.FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF = slab["superimposed_dead_load_ksf"]
    sp.SEISMIC_LIVE_LOAD_FRACTION = slab["live_load_mass_fraction"]


def _fit_slab_and_beam(cfg, beams, preferred, minimum_thickness_in=None):
    """Bounded coupled retry; never promote a failed slab screen to a pass.

    Only sizing exhaustion is retried. Invalid policy/unsupported system or
    aspect ratio remains an explicit error. Concrete strength alone cannot
    improve alpha when slab and beam have the same concrete.
    """
    from Design.SMRF_Slab import SlabSizingError
    attempts, attempted_dimensions = [], set()
    initial = beams[preferred]
    for index in range(preferred, len(beams)):
        rung = beams[index]
        dimensions = rung[:2]
        if dimensions in attempted_dimensions:
            continue
        if index != preferred and (rung[1] <= initial[1] or rung[0] < initial[0]):
            continue
        if _compatible_beam_index(beams, index) != index:
            continue
        attempted_dimensions.add(dimensions)
        _apply_rung(rung, "beam")
        try:
            slab = _select_slab(cfg, minimum_thickness_in)
        except SlabSizingError as exc:
            attempts.append({"beam_section": list(rung), "reason": str(exc)})
            continue
        _sync_cfg_to_sp(cfg)
        return index, slab, attempts
    raise SlabSizingError(
        f"No compatible beam/slab combination after {len(attempts)} bounded sizing attempts. "
        "Revise geometry or the explicit section/thickness bounds; no fallback was assigned.")


def _update_floor_transfer(cfg, slab):
    """Recompute the slab-to-frame transfer for the current slab and sections."""
    if not cfg.floor_analysis.transfer_to_frame:
        sp.FLOOR_TRANSFER = None
        return None
    from Design.SMRF_Floor_Transfer import build_floor_transfer
    # The floor model needs an empty domain. Every frame analysis rebuilds
    # from scratch, so nothing in the current domain is still needed.
    ops.wipe()
    from Design.SMRF_Demands import live_load_patterns
    patterns = live_load_patterns(sp.NUM_BAY_X, sp.NUM_BAY_Y) if cfg.demands.live_load_patterning else ()
    sp.FLOOR_TRANSFER = build_floor_transfer(
        slab, _slab_geometry(),
        {"b_beam_in": sp.B_BEAM, "h_beam_in": sp.H_BEAM, "fc_beam_ksi": sp.FC_BEAM_KSI,
         "b_col_in": sp.B_COL, "h_col_in": sp.H_COL},
        sp.FLOOR_LIVE_LOAD_KSF, mesh_per_bay=cfg.floor_analysis.transfer_mesh_per_bay,
        live_patterns=patterns)
    return sp.FLOOR_TRANSFER


def _slab_strength_inputs_from_state(slab, cfg):
    """The strip routine's inputs for the current state, via the record builder."""
    from Design.SMRF_Design_Evidence import slab_strength_inputs
    return slab_strength_inputs({
        "slab": slab,
        "materials": {"fy_ksi": sp.FY_KSI, "aggregate_size_in": sp.AGGREGATE_MAX_SIZE_IN,
                      "exposure": cfg.materials.exposure,
                      "reinforcement_specification": cfg.materials.reinforcement_specification},
        "geometry": {"num_floor": sp.NUM_FLOOR}})


def _beam_shear_share_criterion(edge):
    """ACI 318-14 Table 8.10.8.1: alpha_f1 times transverse/beam span."""
    span = sp.BAY_X if edge["beam_axis"] == "x" else sp.BAY_Y
    return edge["alpha_f"] * edge["slab_strip_width_in"] / span


def _slab_completion_context(slab, cfg):
    return {"clear_span_x_in": sp.BAY_X - sp.H_COL, "clear_span_y_in": sp.BAY_Y - sp.B_COL,
            "beam_width_in": sp.B_BEAM, "alpha_f_min": min(edge["alpha_f"] for panel in slab["panels"]
                                                           for edge in panel["edges"]),
            "alpha_f_l2_l1_min": min(_beam_shear_share_criterion(edge) for panel in slab["panels"]
                                     for edge in panel["edges"]),
            "thickness_screen_passed": slab["thickness_screen_passed"] is True,
            "column_core_width_in": min(sp.B_COL, sp.H_COL) - 2.0 * sp.longitudinal_cover_in("column"),
            "two_way_shear_path_assessed": (cfg.slab_actions.all_asserted()
                                             and cfg.slab_actions.two_way_shear_path_assessed is True),
            # This builder places beams in both directions at each column.
            # Acceptance of the local shear path remains a separate assertion.
            "columns_at_beam_intersections": True,
            "beam_clear_cover_in": sp.BEAM_CLEAR_COVER_IN,
            "beam_hoop_diameter_in": sp.rebar_diameter(sp.BEAM_STIRRUP_BAR_SIZE),
            "fc_beam_ksi": sp.FC_BEAM_KSI}


def _update_slab_reinforcement(cfg, slab):
    """Slab strip actions and reinforcement for the current slab and sections.

    Runs inside the section loop because the slab mats feed the beam
    strengths the SCWB screen compares columns against. Without the
    engineering assertions in cfg.slab_actions the actions are computed but
    unverified, no reinforcement is selected, and the record says so; the
    frame is then sized on rectangular beam strengths as a proxy and the
    qualification keeps the slab contribution and SCWB not_evaluated. A slab
    whose ladder is exhausted under asserted evidence is a design failure.
    """
    if not cfg.floor_analysis.transfer_to_frame:
        sp.SLAB_REINFORCEMENT = None
        sp.SLAB_ACTIONS = None
        return None
    plan = cfg.floor_analysis.slab_refinement
    if plan is None and cfg.slab_actions.all_asserted():
        # Say what is missing before any solve: single-mesh demands are never
        # verified, so an asserted design could not select reinforcement.
        raise ValueError("Asserted slab actions require FloorAnalysisConfig.slab_refinement (an explicit mesh "
                         "plan or a named recipe such as Design.Config.PROBE_SLAB_REFINEMENT): single-mesh slab "
                         "demands cannot be verified, so no slab reinforcement could be selected.")
    from Design.SMRF_Slab_Actions import build_slab_action_evidence
    from Design.SMRF_Slab_Refinement import build_refined_slab_action_evidence
    from Design.SMRF_Slab_Reinforcement import design_slab_reinforcement
    inputs = _slab_strength_inputs_from_state(slab, cfg)
    ops.wipe()
    sections = {"b_beam_in": sp.B_BEAM, "h_beam_in": sp.H_BEAM, "fc_beam_ksi": sp.FC_BEAM_KSI,
                "b_col_in": sp.B_COL, "h_col_in": sp.H_COL}
    if plan is None:
        evidence = build_slab_action_evidence(
            slab, _slab_geometry(), sections, sp.FLOOR_LIVE_LOAD_KSF, inputs,
            mesh_per_bay=cfg.floor_analysis.transfer_mesh_per_bay, assertions=asdict(cfg.slab_actions))
    else:
        # Resolved here, for these sections: a beam rung change re-resolves the recipe.
        evidence = build_refined_slab_action_evidence(
            slab, _slab_geometry(), sections, sp.FLOOR_LIVE_LOAD_KSF, inputs,
            plan, assertions=asdict(cfg.slab_actions))
    sp.SLAB_ACTIONS = evidence
    record = design_slab_reinforcement(inputs, evidence, None, _slab_completion_context(slab, cfg))
    if record["layout"] is None and cfg.slab_actions.all_asserted():
        refinement = evidence.get("refinement") or {}
        status = refinement.get("status", "not_requested")
        detail = refinement.get("status_detail") or ""
        if refinement.get("comparisons"):
            # The evidence is not saved when the design refuses, so the
            # comparison that refused it is summarized in the message.
            last = refinement["comparisons"][-1]
            rows = last.get("comparisons") or []
            failed = [r for r in rows if not r.get("within_tolerance")]
            worst = {metric: max((r.get("relative_change") or 0.0) for r in rows if r.get("metric") == metric)
                     for metric in ("mu_kip_in_per_ft", "vu_kip_per_ft") if any(r.get("metric") == metric for r in rows)}
            levels = [lvl.get("requested_mesh", {}).get("subdivisions_per_bay") for lvl in refinement.get("levels", [])]
            detail = (f"final pair of levels {levels[-2:]} cells per bay: {len(failed)} of {len(rows)} strip "
                      f"comparisons outside tolerance, largest relative change "
                      + ", ".join(f"{k} {v:.4f}" for k, v in worst.items())
                      + (f"; {detail}" if detail else ""))
        message = (f"Slab reinforcement not selected (slab refinement {status}"
                   + (f": {detail}" if detail else "") + "): " + "; ".join(
                       c["details"].get("reason", c["id"]) for c in record["checks"] if c["status"] != "pass"))
        # The refused attempt is not saved in a design record, so what refused it travels with the error
        # (screening repair item 1, 2026-10-02): the run writes it beside the result.
        from Design.SMRF_Slab_Reinforcement import explain_unsized_layers
        comparisons = refinement.get("comparisons") or []
        refusal = {"thickness_in": slab["thickness_in"],
                   "beam_section": [sp.B_BEAM, sp.H_BEAM, sp.FC_BEAM_KSI],
                   "column_section": [sp.B_COL, sp.H_COL, sp.FC_COL_KSI],
                   "refinement": {"status": status, "detail": detail,
                                  "final_strip_comparisons": (comparisons[-1].get("comparisons") if comparisons else None)},
                   "bar_trials": record.get("trial_history", []),
                   "failed_checks": [c for c in record["checks"] if c["status"] != "pass"],
                   "unsized_layers": explain_unsized_layers(record)}
        if status != "passed":
            # The strip actions are not verified: the levels did not converge (comparison_failed), a level
            # did not solve or its sagging-shear bound did not close (analysis_failed), or the shell budget
            # left fewer than two levels (unresolved_budget). None of these says the slab is too thin, so the
            # thickness search does not step on them; they stay an error with their evidence (screening
            # repair item 4; widened from comparison_failed alone on 2026-10-03).
            raise SlabDesignError(message, refusal)
        # The actions are verified and no layout of the declared bars and spacings carries them at this
        # thickness: the slab thickness search steps the ladder (screening repair item 3, 2026-10-02).
        raise SlabLayoutError(message, slab["thickness_in"], refusal)
    sp.SLAB_REINFORCEMENT = record
    return record


def _thicker_slab(cfg, slab):
    """The next thickness of the declared ladder that passes the screen above ``slab``, applied.

    Returns (slab, None), or (None, reason) when there is none: the ladder's top, or a screen that supports no
    thicker slab on this beam rung (a thicker slab lowers alpha_fm; at or below 0.2 the beam-supported screen
    does not apply).
    """
    from Design.SMRF_Slab import SlabSizingError
    floor = slab["thickness_in"] + cfg.slab.thickness_increment_in
    if floor > cfg.slab.maximum_thickness_in + 1e-9:
        return None, f"the declared ladder tops out at {cfg.slab.maximum_thickness_in:g} in"
    try:
        return _select_slab(cfg, floor), None
    except SlabSizingError as exc:
        return None, (f"the thickness screen supports no slab from {floor:g} to {cfg.slab.maximum_thickness_in:g} in "
                      f"on beam {sp.B_BEAM:g} x {sp.H_BEAM:g} in ({exc})")


def _slab_thickness_search(cfg, slab):
    """Strip actions and reinforcement at the current thickness; when the converged analysis admits no layout,
    the thickness steps up the declared ladder (policy increment, while the screen supports the thickness), the
    transfer and the actions are rebuilt at each step, and every trial is kept with what refused it. Exhaustion
    raises SlabThicknessExhausted naming why the next thickness is unavailable; no thickness is assigned and no
    refinement failure is retried (screening repair items 1 and 3, 2026-10-02).

    Returns (slab, search) with search = {"thickness_in", "steps", "trials": [{"thickness_in", "reason", "evidence"}]}.
    """
    trials = []
    while True:
        try:
            _update_slab_reinforcement(cfg, slab)
        except SlabLayoutError as exc:
            trials.append({"thickness_in": slab["thickness_in"], "reason": str(exc), "evidence": exc.evidence})
            thicker, unavailable = _thicker_slab(cfg, slab)
            if thicker is None:
                raise SlabThicknessExhausted(
                    f"Slab reinforcement not selected at any thickness from {trials[0]['thickness_in']:g} to "
                    f"{slab['thickness_in']:g} in on beam {sp.B_BEAM:g} x {sp.H_BEAM:g} in ({len(trials)} thickness "
                    f"trial(s); next thickness unavailable: {unavailable}); last: {exc}",
                    trials, unavailable, {"thickness_trials": trials, "next_thickness_unavailable": unavailable}) from exc
            slab = thicker
            _update_floor_transfer(cfg, slab)
            continue
        return slab, {"thickness_in": slab["thickness_in"], "steps": len(trials), "trials": trials,
                      "basis": ("thickness stepped up the declared ladder until a four-layer layout passes; "
                                "a refinement comparison failure is not retried")}


def _slab_search(cfg, beams, beam_index, minimum_thickness_in=None):
    """The slab for a beam rung: thickness fit, transfer, strip actions and reinforcement.

    When no supported thickness of the rung admits a layout, the beam steps to the next deeper compatible rung
    and the slab is fitted and searched again there, from that rung's own screen: beam depth raises alpha_f, so
    the beams take the panel shear the slab could not, and the screen admits a thinner slab. Bounded by the beam
    ladder; every rung's thickness trials are kept, and SlabSearchExhausted carries all of them (screening
    repair items 1 and 3, 2026-10-02).

    Returns (beam_index, slab, fitting_attempts, search); search adds "first_fitted_beam_index" and
    "beam_rung_steps" to the thickness search of the rung that passed.
    """
    rung_steps, first_fitted = [], None
    while True:
        beam_index, slab, attempts = _fit_slab_and_beam(cfg, beams, beam_index, minimum_thickness_in)
        if first_fitted is None:
            first_fitted, first_attempts = beam_index, attempts
        _update_floor_transfer(cfg, slab)
        try:
            slab, search = _slab_thickness_search(cfg, slab)
        except SlabThicknessExhausted as exc:
            rung_steps.append({"beam_section": list(beams[beam_index]), "thickness_trials": exc.trials,
                               "next_thickness_unavailable": exc.reason})
            deeper = _next_deeper_beam_index(beams, beam_index)
            if deeper is not None:
                compatible = _compatible_beam_index(beams, deeper)
                deeper = compatible if compatible >= deeper else None
            if deeper is None or len(rung_steps) > len(beams):
                trial_count = sum(len(step["thickness_trials"]) for step in rung_steps)
                raise SlabSearchExhausted(
                    f"Slab reinforcement not selected at any supported thickness of any beam rung from "
                    f"{tuple(rung_steps[0]['beam_section'])} to {tuple(beams[beam_index])} ({len(rung_steps)} beam "
                    f"rung(s), {trial_count} thickness trial(s); no deeper compatible beam rung remains); last: {exc}",
                    {"beam_rung_steps": rung_steps}) from exc
            beam_index = deeper
            continue
        search["first_fitted_beam_index"] = first_fitted
        search["beam_rung_steps"] = rung_steps
        return beam_index, slab, first_attempts, search


def _beam_strength_families():
    """Composite (beam + developed slab) or rectangular Mn per beam family."""
    from Design.SMRF_Beam_Slab_Strength import composite_beam_strengths
    layout = (sp.SLAB_REINFORCEMENT or {}).get("layout") if sp.SLAB_THICKNESS_IN is not None else None
    beam = {"b_in": sp.B_BEAM, "h_in": sp.H_BEAM, "fc_ksi": sp.FC_BEAM_KSI, "fy_ksi": sp.FY_KSI,
            "bar_size": sp.BEAM_BAR_SIZE, "top_bars": sp.BEAM_TOP_BARS, "bot_bars": sp.BEAM_BOT_BARS,
            "centroid_offset_in": sp.longitudinal_cover_in("beam")}
    offsets = sp.beam_bar_stacking_offsets_in()          # the stacked orthogonal layers, per direction
    rows = sp.beam_bar_layers()                           # and the actual rows each direction's bars occupy
    slab = {"thickness_in": sp.SLAB_THICKNESS_IN if layout is not None else 0.0}
    geometry = {"bay_x_in": sp.BAY_X, "bay_y_in": sp.BAY_Y, "h_col_in": sp.H_COL, "b_col_in": sp.B_COL}
    return {f"{axis}_{position}": composite_beam_strengths({**beam, "centroid_offset_in": offsets[axis], "layers": rows[axis]},
                                                           slab, layout, geometry, axis, position)
            for axis in ("x", "y") for position in ("edge", "interior")}


def _governing_beam_nominal_moment():
    """Largest beam Mn over families and signs; slab-inclusive when a layout exists."""
    from Design.SMRF_Beam_Slab_Strength import governing_beam_nominal_moment
    return governing_beam_nominal_moment(_beam_strength_families())


def _validate_cached_slab(record):
    """Reject inconsistent slab evidence before mutating the live model."""
    from Design.SMRF_Slab import evaluate_slab
    slab = record.get("slab")
    checks = evaluate_slab(slab)
    required = {"slab_thickness_evidence", "slab_thickness_screen"}
    if {c["id"] for c in checks if c["status"] == "pass"} != required:
        raise ValueError("Cached design lacks reproducible slab thickness evidence.")
    inputs = slab["inputs"]
    for group in ("geometry", "sections"):
        for key, value in inputs[group].items():
            if record.get(group, {}).get(key) != value:
                raise ValueError(f"Cached slab {group}.{key} disagrees with its frame design.")
    if (slab["concrete_unit_weight_kcf"] != sp.CONCRETE_UNIT_WEIGHT_KCF
            or inputs["policy"]["fy_ksi"] != sp.FY_KSI):
        raise ValueError("Cached slab material assumptions disagree with the current model.")
    return slab


def _sync_cfg_to_sp(cfg):
    """Mirror the current Structure_Parameters section state into cfg.

    The ACI checks read sections and concrete strengths from cfg --
    check_column_pm takes Ag from cfg.sections, and build_pm_diagram builds
    the entire interaction surface from it -- while the ladder writes them to
    sp. cfg was constructed once at entry (DesignConfig.from_structure_
    parameters) and never refreshed, so after any escalation the checks
    measured the new, larger, heavier model against the capacity of the
    section the design STARTED with: demand growing, capacity frozen.

    That is why column DCR rose with section size instead of falling
    (case_0013: 1.104 at 26x26 up to 2.005 at 36x36) and why any case needing
    a column escalation could not converge. Cases that accept on the first
    iteration were unaffected, because cfg still matched sp -- which is why
    this stayed hidden.
    """
    if cfg is None:
        return
    cfg.sections.b_col_in = sp.B_COL
    cfg.sections.h_col_in = sp.H_COL
    cfg.sections.b_beam_in = sp.B_BEAM
    cfg.sections.h_beam_in = sp.H_BEAM
    cfg.materials.fc_col_ksi = sp.FC_COL_KSI
    cfg.materials.fc_beam_ksi = sp.FC_BEAM_KSI
    cfg.rebar.stirrup_spacing_col_in = sp.COL_STIRRUP_SPACING
    cfg.rebar.stirrup_spacing_beam_in = sp.BEAM_STIRRUP_SPACING
    cfg.rebar.col_stirrup_bar_size = sp.COL_STIRRUP_BAR_SIZE
    cfg.rebar.beam_stirrup_bar_size = sp.BEAM_STIRRUP_BAR_SIZE
    cfg.rebar.col_stirrup_legs = sp.COL_STIRRUP_LEGS
    cfg.rebar.beam_stirrup_legs = sp.BEAM_STIRRUP_LEGS
    cfg.rebar.beam_clear_cover_in = sp.BEAM_CLEAR_COVER_IN
    cfg.rebar.col_clear_cover_in = sp.COL_CLEAR_COVER_IN
    cfg.rebar.aggregate_max_size_in = sp.AGGREGATE_MAX_SIZE_IN


def _model_period():
    """First valid elastic period of the current model, for the ELF period cap."""
    with _quiet():
        ops.wipe()
        build_model()
        modes = run_modal_analysis()
    for mode in modes:
        if mode.get("valid") and mode.get("period"):
            return float(mode["period"])
    return None


def _analyze_combination(combination, model_period_sec, torsion=None):
    """Analyze explicit signed gravity/seismic factors; preserve simultaneous actions.

    ``torsion`` = {"ratio", "amplification"} applies ASCE 7-22 12.8.4.2
    accidental torsion with each seismic force, signed with that force; an
    optional ``sign`` (+1/-1) flips the eccentricity side for the
    12.3.2.1.1 assessment cases and is never set for a strength combination.
    """
    with _quiet():
        ops.wipe()
        build_model()
    dead, live = combination["dead"], combination["live"]
    floor_dead = sp.floor_dead_load_ksf()
    total = floor_dead + sp.FLOOR_LIVE_LOAD_KSF
    if total <= 0:
        raise ValueError("Floor gravity load must be positive for load factoring.")
    floor_factor = (dead * floor_dead + live * sp.FLOOR_LIVE_LOAD_KSF) / total
    pattern = combination.get("live_pattern", "all")
    if sp.effective_gravity_load_model() != "slab_transfer":
        pattern = "all"
    apply_gravity_loads(floor_factor=floor_factor, self_weight_factor=dead,
                        dead_factor=dead, live_factor=live, live_pattern=pattern)
    elf = None
    for direction, factor in (("x", combination["ex"]), ("y", combination["ey"])):
        if factor:
            elf = apply_elf_loads(direction, model_period_sec=model_period_sec, load_factor=factor,
                                  accidental_torsion_ratio=(torsion or {}).get("ratio", 0.0),
                                  torsion_amplification=(torsion or {}).get("amplification", 1.0),
                                  torsion_sign=(torsion or {}).get("sign", 1.0) * (1.0 if factor > 0 else -1.0))
    with _quiet():
        run_gravity_analysis()
    return elf


def _torsion_assessment(model_period_sec, cfg, strength=None):
    """ASCE 7-22 12.3.2.1.1 TIR and Table 12.3-1 Type 1, from ELF drift runs with accidental torsion.

    For each direction and each accidental torsion case (the 5% eccentricity
    on either side of the center of mass, Ax = 1.0, ELF forces of 12.8 at
    the design period) the story drifts at the two edge frames of the loaded
    direction are read from the rigid-diaphragm model; delta_avg is their
    average and the TIR is the largest delta_max / delta_avg over every
    story, direction and case. Both eccentricity signs are run rather than
    argued equivalent by symmetry: every case has its rows, and the residual
    between the two signs is reported so the symmetry the archetype relies
    on elsewhere is measured here. The strength-distribution criterion of
    Type 1 comes from the frame's construction (_regularity_by_construction).
    The amplification returned is what the strength combinations and drift
    runs then apply (12.8.4.3).
    """
    from Design.SMRF_Elastic import floor_xy_displacements
    from Design.SMRF_Demands import (story_node_deltas, classify_torsional_irregularity, story_drift_ratio,
                                     amplification_from_level_displacements, TORSION_CASES)
    ratio = cfg.demands.accidental_torsion_ratio
    worst, rows, by_case, ax_by_level = 0.0, [], {}, {}
    for axis in ("x", "y"):
        for sign in (1.0, -1.0):
            case = f"{axis}{'+' if sign > 0 else '-'}"
            combination = {"id": f"torsion_{case}", "dead": 1.0, "live": 1.0,
                           "ex": float(axis == "x"), "ey": float(axis == "y"), "live_pattern": "all"}
            _analyze_combination(combination, model_period_sec, {"ratio": ratio, "amplification": 1.0, "sign": sign})
            case_worst = 0.0
            for k in range(1, sp.NUM_FLOOR + 1):
                upper = floor_xy_displacements(k)
                deltas = story_node_deltas(upper, floor_xy_displacements(k - 1))
                if axis == "x":
                    line_a = [f"{i},0" for i in range(sp.NUM_BAY_X + 1)]
                    line_b = [f"{i},{sp.NUM_BAY_Y}" for i in range(sp.NUM_BAY_X + 1)]
                else:
                    line_a = [f"0,{j}" for j in range(sp.NUM_BAY_Y + 1)]
                    line_b = [f"{sp.NUM_BAY_X},{j}" for j in range(sp.NUM_BAY_Y + 1)]
                # Story drifts at the two edge frames (Table 12.3-1 / 12.3.2.1.1) and
                # the signed level displacements at the same edges (12.8.4.3).
                end_a = max(abs(deltas[node][axis]) for node in line_a)
                end_b = max(abs(deltas[node][axis]) for node in line_b)
                level_a = max((upper[node][axis] for node in line_a), key=abs)
                level_b = max((upper[node][axis] for node in line_b), key=abs)
                drift_ratio = story_drift_ratio(end_a, end_b)
                level = amplification_from_level_displacements(level_a, level_b)
                rows.append({"story": k, "direction": axis, "case": case, "eccentricity_sign": sign,
                             "delta_end_a_in": end_a, "delta_end_b_in": end_b, "delta_max_over_avg": drift_ratio,
                             "delta_level_a_in": level_a, "delta_level_b_in": level_b,
                             "level_max_over_avg": level["ratio"], "ax_level": level["ax"]})
                case_worst = max(case_worst, drift_ratio)
                ax_by_level[k] = max(ax_by_level.get(k, 0.0), level["ax"])
            by_case[case] = case_worst
            worst = max(worst, case_worst)
    residual = 0.0
    for axis in ("x", "y"):
        plus = {r["story"]: r["delta_max_over_avg"] for r in rows if r["case"] == f"{axis}+"}
        minus = {r["story"]: r["delta_max_over_avg"] for r in rows if r["case"] == f"{axis}-"}
        residual = max(residual, max(abs(plus[k] - minus[k]) for k in plus))
    # A grouped design supplies its own story-by-story strength distribution (Design.Grouped_Design).
    strength = _lateral_strength_distribution(cfg) if strength is None else strength
    classification = classify_torsional_irregularity(worst, strength["one_side_fraction"],
                                                     strength_model_verified=_strength_model_verified(cfg))
    envelope = max(ax_by_level.values())
    required = envelope if classification["type_1"] else 1.0
    return {"ratio": ratio, "assessment_amplification": 1.0, "base": "fixed",
            "amplification": required, "amplification_required": required,
            "amplification_by_level": ax_by_level, "amplification_envelope_12_8_4_3": envelope,
            "amplification_basis": ("12.8.4.3 Ax = (delta_max / 1.2 delta_avg)^2, 1 <= Ax <= 3, from the edge level "
                                    "displacements of the Ax = 1 runs, per level, direction and eccentricity case; the "
                                    "largest per-level value is applied at every level of every seismic strength "
                                    "combination and drift run when Type 1 is established (a conservative envelope: Mta "
                                    "at each level is scaled by at least its own Ax), 1.0 otherwise -- including the "
                                    "'unresolved' outcome, where the TIR does not establish Type 1 and the provisional "
                                    "strength model cannot exclude it; qualification keeps that item open"),
            "max_drift_ratio": worst, "tir": worst, "tir_by_case": by_case,
            "torsional_irregularity": classification["label"], "classification": classification,
            "strength_distribution": strength,
            "cases": list(TORSION_CASES), "stories": rows, "sign_symmetry_residual_max": residual,
            "basis": ("ASCE 7-22 12.3.2.1.1: TIR = delta_max / delta_avg of the story drifts at the two edge frames of "
                      "the loaded direction, ELF forces of 12.8 with 5% accidental torsion (12.8.4.2) and Ax = 1.0, "
                      "rigid diaphragm, every story, both directions, both eccentricity signs; Table 12.3-1 Type 1 when "
                      "TIR > 1.2 or more than 75% of a story's strength lies at or on one side of the center of mass; "
                      "Ax per 12.8.4.3 from the level displacements of the same runs")}


def _strength_model_verified(cfg):
    """Has a person asserted the story-strength model (IndependentVerification.story_strength_model_verified)?"""
    from Design.SMRF_Common import assertion_provenance_valid
    verification = getattr(cfg, "verification", None)
    if verification is None:
        return False
    policy = asdict(verification)
    return bool(policy.get("story_strength_model_verified")) and assertion_provenance_valid(policy)


def _strength_model_applicability(cfg):
    """The applicability record of the story-strength model: its review status, never implied by arithmetic."""
    from Design.SMRF_Demands import STRENGTH_MODEL_REVIEW_ITEM
    verified = _strength_model_verified(cfg)
    policy = asdict(cfg.verification) if cfg is not None and getattr(cfg, "verification", None) is not None else {}
    return {"status": "verified" if verified else "provisional",
            "review_item": STRENGTH_MODEL_REVIEW_ITEM,
            "asserted_by": policy.get("asserted_by") if verified else None,
            "assertion_date": policy.get("assertion_date") if verified else None,
            "assertion_basis": policy.get("assertion_basis") if verified else None,
            "consequence": ("a verified model resolves the Table 12.3-1 strength criterion both ways" if verified else
                            "a provisional model can establish Type 1 (more than 75% on one side) but cannot certify its "
                            "absence; with TIR <= 1.2 the classification is 'unresolved' and qualification keeps "
                            "demands.torsional_irregularity open"),
            "basis": ("the frame-line story-strength model (bays x (Mn- + Mn+) / h per line) is a declared approximation "
                      "whose scientific applicability -- base, roof and column-limited mechanisms, gravity transfer, axial "
                      "redistribution, shear limits -- is the reviewers' decision, asserted through "
                      "IndependentVerification.story_strength_model_verified with author, date and basis")}


def _lateral_strength_distribution(cfg=None):
    """Story lateral strength by frame line and the Table 12.3-1 one-sided fraction, per direction.

    Model: a frame line's story lateral strength is its beam-sway mechanism
    strength, bays x (Mn- + Mn+) / story height of the line's beam family
    (edge family on the two perimeter lines, interior family elsewhere;
    composite with the developed slab where the layout is established, the
    rectangular proxy otherwise); columns are identical on every line and
    members are uniform over height, so the fraction is the same at every
    story. The center of mass is the plan center (uniform floor mass). A
    line through the center counts on both sides ("at or on one side"), so
    three identical lines give 2/3, not 1/2; weaker perimeter families push
    the fraction of a three-line direction above that. A failure to price
    the families leaves the fraction unknown, never 0.5.
    """
    from Design.SMRF_Demands import one_sided_strength_fraction, line_story_strengths, STRENGTH_FAMILIES
    from Design.SMRF_Beam_Slab_Strength import beam_slab_strengths
    try:
        # The same rebuild qualification performs from the record (sections,
        # cage, chosen slab thickness, established layout or none), so the
        # recorded family strengths reproduce from the saved final cage.
        state = _state_record_core()
        state["slab"] = {"thickness_in": sp.SLAB_THICKNESS_IN if sp.SLAB_THICKNESS_IN is not None else 0.0}
        state["slab_reinforcement"] = sp.SLAB_REINFORCEMENT if sp.SLAB_THICKNESS_IN is not None else None
        families = beam_slab_strengths(state)[1]
    except Exception as exc:                            # noqa: BLE001 -- unknown is the honest answer
        return {"one_side_fraction": None, "model": None, "applicability": _strength_model_applicability(cfg),
                "basis": f"not evaluated: beam strength families unavailable ({type(exc).__name__}: {exc})"}
    inputs = {"story_h_in": sp.STORY_H, "bays": {"x": sp.NUM_BAY_X, "y": sp.NUM_BAY_Y},
              "families": {name: {"mn_negative_kip_in": families[name]["mn_negative_kip_in"],
                                  "mn_positive_kip_in": families[name]["mn_positive_kip_in"]} for name in STRENGTH_FAMILIES},
              "source": ("Design.SMRF_Beam_Slab_Strength.beam_slab_strengths on the installed sections, longitudinal cage and "
                         "slab (composite with the established layout, flange concrete only without one); the rebuild "
                         "qualification repeats from the record")}
    geometry = _slab_geometry()
    by_direction, worst = {}, 0.0
    for direction in ("x", "y"):
        positions, strengths, names = line_story_strengths(direction, geometry, inputs["families"], sp.STORY_H)
        fraction, detail = one_sided_strength_fraction(positions, strengths, 0.5 * positions[-1])
        by_direction[direction] = {"line_positions_in": positions, "line_story_strength_kip": strengths,
                                   "line_families": names, "one_side_fraction": fraction, **detail}
        worst = max(worst, fraction)
    return {"one_side_fraction": worst, "by_direction": by_direction, "strength_inputs": inputs,
            "applicability": _strength_model_applicability(cfg),
            "uniform_over_height": {"claim": True,
                                    "basis": ("one section and one longitudinal cage per member type over the full height "
                                              "(record.sections, record.reinforcement), so every story has the same line "
                                              "strengths; the archetype's construction, checked by the record's single "
                                              "section and reinforcement blocks")},
            "model": ("beam-sway mechanism story strength per frame line: bays x (Mn- + Mn+) / story height of the line's "
                      "beam family (edge family on the two perimeter lines, interior family elsewhere), composite with the "
                      "developed slab where established, flange-concrete section otherwise; identical columns on every "
                      "line; uniform over height; center of mass at the plan center; priced on the installed cage. A "
                      "declared approximation awaiting the scientific review's hand calculation (column base, roof and "
                      "column-limited mechanisms, shear limits): not a verified resistance model"),
            "basis": ("ASCE 7-22 Table 12.3-1 Type 1 strength criterion: more than 75% of a story's lateral strength at or "
                      "on one side of the center of mass; a line through the center counts on both sides")}


def _regularity_by_construction(strength_distribution=None):
    """Horizontal/vertical regularity of the archetype: rectangular grid, uniform stories and sections."""
    return {"regular": True,
            "horizontal": "rectangular plan, frames on every grid line, rigid diaphragm without openings: "
                          "no Type 2-5 horizontal irregularities by construction; Type 1 (ASCE 7-22 Table 12.3-1) from "
                          "the torsion assessment (12.3.2.1.1 TIR) and the strength-distribution criterion below, whose "
                          "model is provisional until asserted (lateral_strength_distribution.applicability)",
            "vertical": "uniform story height, sections and reinforcement over height: no Type 1-4 vertical "
                        "irregularities (ASCE 7-22 Table 12.3-2, which has no weight (mass) irregularity) by construction",
            "lateral_strength_distribution": (strength_distribution if strength_distribution is not None
                                              else _lateral_strength_distribution())}


def _demand_basis(cfg, torsion, elf, drift_screen, strength=None, regularity=None):
    """The saved demand basis of the iteration's final state.

    The torsion assessment ran at the start of the iteration; the strength
    distribution is re-priced on the cage installed at the end and the
    classification re-derived on it, so the saved regularity block and the
    saved classification describe the same evidence. ``amplification`` stays
    the value the analyses applied; ``amplification_required`` is re-derived
    on the final classification, so a Type 1 that appears only on the final
    cage shows up as an unmet requirement rather than a silent pass.
    """
    from Design.SMRF_Demands import (live_load_patterns, classify_torsional_irregularity, CODE_EDITION,
                                     REDUNDANCY_FACTOR_STRENGTH)
    ts = sp.ASCE_SD1 / sp.ASCE_SDS if sp.ASCE_SDS > 0 else 0.0
    # ``strength`` and ``regularity`` are supplied by a grouped design, whose stories are not alike.
    strength = _lateral_strength_distribution(cfg) if strength is None else strength
    if torsion is not None:
        classification = classify_torsional_irregularity(torsion["tir"], strength["one_side_fraction"],
                                                         strength_model_verified=_strength_model_verified(cfg))
        torsion = {**torsion, "classification": classification, "torsional_irregularity": classification["label"],
                   "strength_distribution": strength,
                   "amplification_required": (torsion["amplification_envelope_12_8_4_3"] if classification["type_1"] else 1.0)}
    return {
        "policy": asdict(cfg.demands),
        "verification": asdict(cfg.verification),
        "code_edition": CODE_EDITION,
        "analysis_procedure": {"procedure": "equivalent lateral force (ASCE 7-22 12.8)",
                               "permitted_by": "ASCE 7-22 12.6(a): permitted for any structure",
                               "model": "three-dimensional elastic frame, rigid diaphragms, cracked section stiffness, P-Delta"},
        "redundancy_factor": REDUNDANCY_FACTOR_STRENGTH,
        "torsion": torsion,
        "regularity": _regularity_by_construction(strength) if regularity is None else regularity,
        "design_period_sec": (elf or {}).get("design_period_sec"),
        "ts_sec": ts,
        "period_basis": {"model_period_sec": (elf or {}).get("model_period_sec"),
                         "asce_ta_sec": (elf or {}).get("asce_ta_sec"),
                         "capped_at_cu_ta": (elf or {}).get("period_capped_at_cu_ta")},
        "live_load_patterns": (live_load_patterns(sp.NUM_BAY_X, sp.NUM_BAY_Y)
                               if cfg.demands.live_load_patterning and sp.FLOOR_TRANSFER is not None else []),
        "patterns_in_strength_envelope": bool(cfg.demands.live_load_patterning and sp.FLOOR_TRANSFER is not None),
        "drift": {**drift_screen.get("assumptions", {}),
                  "beam_stiffness_modifier": sp.section_stiffness_modifier("beam"),
                  "column_stiffness_modifier": sp.section_stiffness_modifier("column"),
                  "second_order_included": True},
    }


def _governing_dcrs(cfg, member_actions=None):
    """Worst PM/flexure/shear DCR for columns and for beams in the current state.

    ``member_actions`` (from _capture_element_actions) lets the checks run on
    solved actions instead of the live domain; see _steel_pass.
    """
    col_tags, beam_x_tags, beam_y_tags = get_element_tags()
    results = run_checks_phase1(col_tags, beam_x_tags + beam_y_tags, cfg, member_actions=member_actions)

    column_dcr = 0.0
    beam_dcr = 0.0
    for result in results.values():
        if result.member_type == "column":
            column_dcr = max(column_dcr, result.dcr("PM"), result.dcr("shear"))
        else:
            beam_dcr = max(
                beam_dcr,
                result.dcr("flexure_pos"),
                result.dcr("flexure_neg"),
                result.dcr("shear"),
            )
    return column_dcr, beam_dcr, results


def _beam_flexure_demand(results):
    """The largest factored beam moment (either sign, kip-in) among one combination's check results."""
    demand = 0.0
    for result in results.values():
        limit_states = getattr(result, "limit_states", None)
        if limit_states is None or getattr(result, "member_type", "column") == "column":
            continue
        for key in ("flexure_neg", "flexure_pos"):
            state = limit_states.get(key)
            if state is not None:
                demand = max(demand, abs(state.demand))
    return demand


def _capture_element_actions(dead_factor=1.0):
    """Keep signed concurrent centerline actions, not independent absolute maxima."""
    from Design.SMRF_Elastic import physical_members
    from Design.SMRF_Beam_Actions import current_beam_bending
    physical = list(physical_members())
    spans = current_beam_bending([tag for tag, ni, nj, kind in physical if kind != "column"])
    members = {}
    for tag, ni, _nj, kind in physical:
        forces = list(ops.eleResponse(tag, "localForce"))
        if len(forces) != 12 or not all(math.isfinite(value) for value in forces):
            raise RuntimeError(f"Invalid local force vector for design member {tag}.")
        offset_i = offset_j = axial_line_load = 0.0
        if kind == "column":
            # Column local x is upward. P(x)=P_i-w*x for the uniform axial
            # dead load actually applied to this centerline model. At a base
            # support there is no lower beam joint to trim.
            offset_i = sp.H_BEAM / 2 if ops.nodeCoord(ni, 3) > 0 else 0.0
            offset_j = sp.H_BEAM / 2
            axial_line_load = dead_factor * sp.col_self_weight_kip_per_in()
        members[str(tag)] = {
            "member_type": kind, "local_force_kip_kipin": forces,
            "centerline_axial_i_kip": forces[0], "centerline_axial_j_kip": -forces[6],
            "axial_i_kip": forces[0] - axial_line_load * offset_i,
            "axial_j_kip": -forces[6] + axial_line_load * offset_j,
            "joint_face_offsets_in": [offset_i, offset_j],
            "axial_line_load_kip_per_in": axial_line_load,
        }
        if kind != "column":
            members[str(tag)]["span_bending"] = spans[tag]
    return members


def _col_steel_layers_about_z():
    """Column steel layers for bending through b: compression on an h face.

    Mirror of RC_Design_Check._col_steel_layers for the y frame. The corner
    bars and the side-face bars sit in the two outermost layers; each interior
    top/bottom bar position holds one top bar and one bottom bar.
    """
    cover = sp.longitudinal_cover_in("column")
    b, ab = sp.B_COL, sp.COL_BAR_AREA
    n_top, n_bot = max(2, sp.COL_TOP_BARS), max(2, sp.COL_BOT_BARS)
    outer = (2 + sp.COL_SIDE_BARS) * ab
    layers = [(outer, cover)]
    for count in (n_top, n_bot):
        if count > 2:
            layers += [(ab, cover + (b - 2.0 * cover) * k / (count - 1)) for k in range(1, count - 1)]
    layers.append((outer, b - cover))
    return sorted(layers, key=lambda layer: layer[1])


def _column_envelopes(combination_actions):
    """Per-story column (min, max) joint-face axial and max |V| over every final case (legacy pair)."""
    envelopes = _column_action_envelopes(combination_actions)
    return envelopes["axial"], envelopes["shear"]


def _column_action_envelopes(combination_actions):
    """The full per-story column envelopes: per-end axial ranges with sources, per-direction shears."""
    from Design.SMRF_Capacity_Design import column_action_envelopes
    per_story = (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1)
    return column_action_envelopes(combination_actions, per_story)


def _capacity_state(cfg, combination_actions):
    envelopes = _column_action_envelopes(combination_actions)
    axial, shear = envelopes["axial"], envelopes["shear"]
    capacity_policy = getattr(cfg, "capacity", None)
    return {
        "geometry": _slab_geometry(),
        "sections": {"b_col_in": sp.B_COL, "h_col_in": sp.H_COL, "fc_col_ksi": sp.FC_COL_KSI,
                     "b_beam_in": sp.B_BEAM, "h_beam_in": sp.H_BEAM, "fc_beam_ksi": sp.FC_BEAM_KSI},
        "materials": {"fy_ksi": sp.FY_KSI, "es_ksi": sp.ES_KSI,
                      "normalweight": sp.CONCRETE_UNIT_WEIGHT_KCF == 0.150,
                      "aggregate_size_in": sp.AGGREGATE_MAX_SIZE_IN},
        "beam": {"bar_size": sp.BEAM_BAR_SIZE, "top_bars": sp.BEAM_TOP_BARS, "bot_bars": sp.BEAM_BOT_BARS,
                 "centroid_offset_in": sp.longitudinal_cover_in("beam"), "clear_cover_in": sp.BEAM_CLEAR_COVER_IN,
                 "stirrup_bar_size": sp.BEAM_STIRRUP_BAR_SIZE,
                 # Orthogonal layers stacked at the joints: each direction's top/bottom offsets, the
                 # worst single value for shear depth, and the convention (design-basis rule 2026-09-27).
                 "centroid_offsets_by_axis_in": sp.beam_bar_stacking_offsets_in(),
                 "worst_centroid_offset_in": sp.beam_worst_centroid_offset_in(),
                 "stacking_convention": sp.beam_bar_stacking_convention(),
                 "max_layers": sp.BEAM_BAR_MAX_LAYERS, "layer_order": sp.BEAM_BAR_LAYER_ORDER, "layers": sp.beam_bar_layers(),
                 # The smeared line weight the frame element carries over its full
                 # centerline length (drop weight spread over L, see
                 # Structure_Parameters.beam_self_weight_kip_per_in) and the
                 # physical drop weight per inch that acts between the joint faces.
                 "self_weight_kip_per_in": {"x": sp.beam_self_weight_kip_per_in("x"),
                                            "y": sp.beam_self_weight_kip_per_in("y")},
                 "drop_weight_kip_per_in": sp.beam_drop_weight_kip_per_in()},
        "column": {"bar_size": sp.COL_BAR_SIZE, "top_bars": sp.COL_TOP_BARS, "bot_bars": sp.COL_BOT_BARS,
                   "side_bars": sp.COL_SIDE_BARS, "centroid_offset_in": sp.longitudinal_cover_in("column"),
                   "clear_cover_in": sp.COL_CLEAR_COVER_IN, "stirrup_bar_size": sp.COL_STIRRUP_BAR_SIZE,
                   # Bending through h (x frame) and through b (y frame); the
                   # capacity design takes Mpr, Ve and the hoop legs per direction.
                   "layers": _col_steel_layers(),
                   "layers_about_z": _col_steel_layers_about_z()},
        "slab": {"thickness_in": sp.SLAB_THICKNESS_IN,
                 "layout": (sp.SLAB_REINFORCEMENT or {}).get("layout") if sp.SLAB_THICKNESS_IN is not None else None},
        "transfer": sp.FLOOR_TRANSFER, "sds": sp.ASCE_SDS,
        "column_axial_envelope": axial, "column_shear_demand": shear,
        # Per-end axial ranges with their source combination / column tag and
        # the per-direction shear maxima (SMRF_Capacity_Design.column_action_envelopes).
        "column_action_envelopes": envelopes["detail"],
        "combination_actions_used": envelopes["combinations_used"],
        # The 18.7.6.1.1 Ve rule is a request-identity policy (Design.Config.CapacityPolicy).
        "column_shear_method": (getattr(capacity_policy, "uniform_column_shear_method", None)
                                or getattr(capacity_policy, "column_shear_method", None)),
        "column_clear_height_convention": getattr(capacity_policy, "column_clear_height_convention", None),
        "joint_continuity": _joint_continuity_declaration(),
    }



def _joint_continuity_declaration():
    """The reinforcement continuity the generated detailing provides at every joint.

    ACI 318-19 Table 18.8.4.3 reads the column and the beam in the direction
    of Vj as continuous when the member on the far side of the joint carries
    its longitudinal and transverse reinforcement through the joint
    (15.2.6(b), 15.2.7(b)); 15.2.8(c) asks the transverse beams for two
    continuous top and bottom bars. This design uses one beam family per
    direction with the same top and bottom bars in every span, laid through
    interior joints with laps only between the 2h hinge zones or Type 2
    mechanical splices (design_splices), and one column cage over the
    height lapped in the center half of the clear height (18.7.4.4). Those
    are design intents of the generated detailing, declared here so the
    joint classification cites them instead of inferring continuity from a
    face count; whether the drawn cage realises them stays with
    qualification.detailing_model_consistency (IndependentVerification),
    never with this declaration.
    """
    declaration = {"beam_reinforcement_continuous_through_interior_joints": True,
                   "column_reinforcement_continuous_through_floor_joints": True,
                   "basis": ("one beam family per direction with the same top/bottom bars in every span, laps outside the 2h "
                             "hinge zones or Type 2 mechanical (18.6.3.3, 18.2.7); one column cage over the height with laps in "
                             "the center half (18.7.4.4); splice placement per capacity_design.splices")}
    if sp.ROOF_COLUMN_EXTENSION:
        # User decision 2026-10-03: every column is carried one column depth above the roof joint, with its
        # longitudinal bars and hoops continued through the extension (ACI 318-19 15.2.6). The roof joints were
        # the governing joints for shear at Table 18.8.4.3's "Other" column row; the extension is a common detail
        # (it sits within the parapet height) and costs no section size. Its weight (one column depth of column
        # per column line) is carried as load and mass at the roof nodes (Model/Roof_Extension, 2026-10-04).
        declaration.update({
            "roof_column_extension_in": max(sp.B_COL, sp.H_COL),
            "column_reinforcement_continued_through_roof_extension": True,
            "roof_extension_basis": ("column stub one column depth above the roof joint, bars and hoops of the column below "
                                     "continued through it (ACI 318-19 15.2.6(a), (b)); its weight is a vertical load "
                                     "and a mass at each roof node (Model/Roof_Extension)")})
    return declaration


def _capacity_design(cfg, combination_actions):
    """Probable-strength shear, joint shear and anchorage; installs the hoops it selects.

    The hoops feed the hinge backbones (rho_sh in Model/IMK_Calibration), so
    they are part of the design state, not a report appended afterwards.
    """
    from Design.SMRF_Capacity_Design import build_capacity_design
    # A selected hoop diameter moves the longitudinal bars, changing the
    # probable moments which generated its own shear demand. Return evidence
    # for a stable installed cage, not the input to the last hoop selection.
    seen = set()
    for _ in range(8):
        before = _cage_signature()
        if before in seen:
            raise RuntimeError("Capacity hoop selection cycled; installed-cage closure was not established.")
        seen.add(before)
        capacity = build_capacity_design(_capacity_state(cfg, combination_actions))
        for member, prefix in (("beam", "BEAM"), ("column", "COL")):
            hoops = capacity["transverse"][member]
            if hoops is not None:
                setattr(sp, f"{prefix}_STIRRUP_BAR_SIZE", hoops["bar_size"])
                setattr(sp, f"{prefix}_STIRRUP_SPACING", hoops["spacing_in"])
                if member == "column":
                    # The scalar used by legacy checks is the lighter direction.
                    sp.COL_STIRRUP_LEGS = hoops["legs_model"]
                    sp.COL_STIRRUP_LEGS_BY_DIRECTION = dict(hoops["legs"])
                else:
                    setattr(sp, f"{prefix}_STIRRUP_LEGS", hoops["legs"])
        _sync_cfg_to_sp(cfg)
        if _cage_signature() == before:
            return capacity
    raise RuntimeError("Capacity hoop selection exceeded 8 updates; installed-cage closure was not established.")


def _transverse_geometry(cfg):
    """Scalar zone/spacing selection, not a complete confinement cage design."""
    from Design.SMRF_Detailing import select_transverse_geometry
    inputs = {"material": {"fy_ksi": sp.FY_KSI,
                           "normalweight": sp.CONCRETE_UNIT_WEIGHT_KCF == .150}}
    for member, prefix in (("beam", "BEAM"), ("column", "COL")):
        inputs[member] = {
            "b_in": getattr(sp, f"B_{prefix}"), "h_in": getattr(sp, f"H_{prefix}"),
            "clear_cover_in": getattr(sp, f"{prefix}_CLEAR_COVER_IN"),
            "bar_db_in": sp.rebar_diameter(getattr(sp, f"{prefix}_BAR_SIZE")),
            "stirrup_db_in": sp.rebar_diameter(getattr(sp, f"{prefix}_STIRRUP_BAR_SIZE")),
            "hoop_spacing_in": getattr(sp, f"{prefix}_STIRRUP_SPACING"),
        }
    inputs["column"]["clear_height_in"] = sp.STORY_H - sp.H_BEAM
    result = select_transverse_geometry(inputs,
                                       minimum_spacing_in=cfg.rebar.stirrup_spacing_min_in,
                                       spacing_step_in=cfg.rebar.stirrup_spacing_step_in)
    result["analysis_application"] = (
        "Selected conservative spacing is uniform over each full member; end-zone lengths are "
        "recorded geometry only. Zoned confinement properties and the hoop/crosstie cage remain unverified.")
    return result


def _apply_transverse_geometry(cfg):
    # Complete the pure search before mutating either member family.
    geometry = _transverse_geometry(cfg)
    sp.BEAM_STIRRUP_SPACING = geometry["beam"]["hoop_spacing_in"]
    sp.COL_STIRRUP_SPACING = geometry["column"]["hoop_spacing_in"]
    _sync_cfg_to_sp(cfg)
    return geometry


def _cage_signature():
    """Every installed cage quantity that can change strength or detailing."""
    return (sp.COL_BAR_SIZE, sp.COL_TOP_BARS, sp.COL_BOT_BARS, sp.COL_SIDE_BARS,
            sp.BEAM_BAR_SIZE, sp.BEAM_TOP_BARS, sp.BEAM_BOT_BARS, sp.BEAM_SIDE_BARS,
            sp.COL_STIRRUP_BAR_SIZE, sp.COL_STIRRUP_SPACING, sp.COL_STIRRUP_LEGS,
            tuple(sorted((sp.COL_STIRRUP_LEGS_BY_DIRECTION or {}).items())),
            sp.BEAM_STIRRUP_BAR_SIZE, sp.BEAM_STIRRUP_SPACING, sp.BEAM_STIRRUP_LEGS,
            sp.BEAM_BAR_STACKING, sp.BEAM_BAR_MAX_LAYERS, sp.BEAM_BAR_LAYER_ORDER)


def _state_record_core():
    """Geometry, sections, reinforcement and materials of the live state.

    One source for the design record and for the in-loop joint check, so the
    joint adapter prices exactly the cage, cover and materials that are
    installed (and later written out).
    """
    return {
        "geometry": {
            "num_bay_x": sp.NUM_BAY_X,
            "num_bay_y": sp.NUM_BAY_Y,
            "num_floor": sp.NUM_FLOOR,
            "bay_x_in": sp.BAY_X,
            "bay_y_in": sp.BAY_Y,
            "story_h_in": sp.STORY_H,
        },
        "sections": {
            "b_col_in": sp.B_COL,
            "h_col_in": sp.H_COL,
            "fc_col_ksi": sp.FC_COL_KSI,
            "b_beam_in": sp.B_BEAM,
            "h_beam_in": sp.H_BEAM,
            "fc_beam_ksi": sp.FC_BEAM_KSI,
        },
        "reinforcement": {
            "col_bar_size": sp.COL_BAR_SIZE,
            "col_top_bars": sp.COL_TOP_BARS,
            "col_bot_bars": sp.COL_BOT_BARS,
            "col_side_bars": sp.COL_SIDE_BARS,
            "beam_bar_size": sp.BEAM_BAR_SIZE,
            "beam_top_bars": sp.BEAM_TOP_BARS,
            "beam_bot_bars": sp.BEAM_BOT_BARS,
            "beam_side_bars": sp.BEAM_SIDE_BARS,
            "col_stirrup_bar_size": sp.COL_STIRRUP_BAR_SIZE,
            "col_stirrup_legs": sp.COL_STIRRUP_LEGS,
            "col_stirrup_legs_by_direction": (dict(sp.COL_STIRRUP_LEGS_BY_DIRECTION)
                                              if sp.COL_STIRRUP_LEGS_BY_DIRECTION is not None
                                              else {"across_b_face": sp.COL_STIRRUP_LEGS, "across_h_face": sp.COL_STIRRUP_LEGS}),
            "col_stirrup_spacing_in": sp.COL_STIRRUP_SPACING,
            "beam_stirrup_bar_size": sp.BEAM_STIRRUP_BAR_SIZE,
            "beam_stirrup_legs": sp.BEAM_STIRRUP_LEGS,
            "beam_stirrup_spacing_in": sp.BEAM_STIRRUP_SPACING,
            "legacy_centroid_offset_in": sp.COVER,
            "cover_basis": "clear_cover_outside_hoops",
            "beam_clear_cover_in": sp.BEAM_CLEAR_COVER_IN,
            "col_clear_cover_in": sp.COL_CLEAR_COVER_IN,
            "beam_longitudinal_centroid_offset_in": sp.longitudinal_cover_in("beam"),
            # The orthogonal top and bottom layers stacked at the joints: the convention and each
            # direction's face offsets; the nominal offset above is the upper layer's.
            "beam_bar_stacking": {"mode": sp.BEAM_BAR_STACKING, "convention": sp.beam_bar_stacking_convention(),
                                  "offsets_in": sp.beam_bar_stacking_offsets_in(),
                                  "max_layers": sp.BEAM_BAR_MAX_LAYERS, "layer_order": sp.BEAM_BAR_LAYER_ORDER,
                                  "layers": sp.beam_bar_layers(), "layer_pitch_in": sp.beam_bar_layer_pitch_in(),
                                  "interlayer_clear_min_in": sp.BEAM_BAR_STACKING_INTERLAYER_CLEAR_MIN_IN,
                                  "basis": ("orthogonal beam bars cannot share an elevation where they cross at a joint; "
                                            "the lower direction's cage runs straight below the upper direction's last "
                                            "layer, so each direction gives up depth at one face; bars the column lanes "
                                            "do not admit in one layer take a second layer directly below (25.2.2), "
                                            "up to max_layers (design-basis rules 2026-09-27)")},
            "col_longitudinal_centroid_offset_in": sp.longitudinal_cover_in("column"),
            "col_bar_diameter_in": sp.rebar_diameter(sp.COL_BAR_SIZE),
            "beam_bar_diameter_in": sp.rebar_diameter(sp.BEAM_BAR_SIZE),
            "col_bar_area_in2": sp.COL_BAR_AREA,
            "beam_bar_area_in2": sp.BEAM_BAR_AREA,
            "col_stirrup_diameter_in": sp.rebar_diameter(sp.COL_STIRRUP_BAR_SIZE),
            "beam_stirrup_diameter_in": sp.rebar_diameter(sp.BEAM_STIRRUP_BAR_SIZE),
        },
        "materials": {"fy_ksi": sp.FY_KSI, "fyt_ksi": sp.FY_KSI,
                      "normalweight": sp.CONCRETE_UNIT_WEIGHT_KCF == 0.150,
                      "aggregate_size_in": sp.AGGREGATE_MAX_SIZE_IN, "es_ksi": sp.ES_KSI},
    }


def _joint_scwb_state(actions, expected_ids):
    """ACI 318-19 18.7.3.2 at every joint of the live state, from solved actions.

    Runs the same adapter and evaluator qualification runs (SMRF_Joint_Adapter
    on the installed cage; SMRF_Joints.scwb_check with the factored axial
    envelope and both compression faces), so the search loop accepts what
    qualification will accept. Checks that cannot be evaluated (no
    established slab strength) stay unevaluated here as they do there; only
    a failed check drives the search.

    Returns a summary plus, under ``_failing`` and ``_state``, the failed
    checks with their sway states and the priced record for the steel
    selection; the underscore keys are not written to the history.
    """
    from Design.SMRF_Beam_Slab_Strength import beam_slab_strengths
    from Design.SMRF_Joint_Adapter import build_joint_evidence
    from Design.SMRF_Joints import scwb_check
    state = _state_record_core()
    state["slab"] = {"thickness_in": sp.SLAB_THICKNESS_IN if sp.SLAB_THICKNESS_IN is not None else 0.0}
    state["slab_reinforcement"] = sp.SLAB_REINFORCEMENT if sp.SLAB_THICKNESS_IN is not None else None
    state["beam_slab_strengths"] = beam_slab_strengths(state)[0]
    evidence = build_joint_evidence(state, actions, expected_combination_ids=list(expected_ids))
    failing, counts, ratios, excepted = [], {"pass": 0, "fail": 0, "not_evaluated": 0}, [], 0
    for joint in evidence["joints"]:
        for axis in ("x", "y"):
            for sign in ("positive", "negative"):
                sway = joint["directions"][axis][sign]
                check = scwb_check(sway, location=f"{joint['id']}/{axis}/{sign}")
                counts[check["status"]] += 1
                if check["status"] == "not_evaluated":
                    continue
                if check["details"].get("roof_exemption_applied") is True:
                    # 18.7.3.1: the rule is not required at this connection; its ratio is not a margin.
                    excepted += 1
                    continue
                ratios.append(check["details"]["ratio_provided"])
                if check["status"] == "fail":
                    failing.append((check, sway, axis))
    return {
        "evaluated": counts["pass"] + counts["fail"] > 0,
        "all_pass": counts["fail"] == 0,
        "counts": counts,
        "min_ratio_provided": min(ratios) if ratios else None,
        "ratio_required": sp.SCWB_RATIO_MIN,
        "excepted_18_7_3_1": excepted,
        "basis": ("per-joint nominal joint-face strengths on the installed cage; column Mn enveloped over "
                  "every factored combination and both compression faces; beam Mn with the developed slab "
                  "where established; the 18.7.3.1 exception where the column is discontinuous above and "
                  "Pu < Ag f'c / 10 (min_ratio_provided is over the connections the rule applies to)"),
        "failed_checks": [{key: check.get(key) for key in ("id", "location", "status", "demand", "capacity", "units")}
                          for check, _sway, _axis in failing],
        "_failing": failing,
        "_state": state,
    }


def _scwb_column_steel(cfg, joint_scwb):
    """Least column cage for which every failed joint satisfies 18.7.3.2.

    Candidate cages come from the generator the strength pick uses (bar
    spacing and the 18.7.5.2(f) hx limit applied), between the ACI minimum
    and cfg.rebar.rho_col_practical_max; each is priced with the joint
    adapter's own section capacity at the ends of every failed column's
    factored axial envelope, both compression faces. The pick is confirmed
    on the next steel iteration by the full per-joint evaluation. Returns
    (update or None, exhausted).
    """
    from Redesign import col_candidates_for_beam_bars, scwb_cage_order
    from Design.SMRF_Joint_Adapter import record_section_capacity
    failing = joint_scwb["_failing"]
    if not failing:
        return None, False
    state = joint_scwb["_state"]
    ag = sp.B_COL * sp.H_COL
    # The column cage rule (2026-09-27): the escalation draws from the cages whose lanes admit the
    # beam bars; when those run out within the practical ratio the column grows rather than the cage
    # closing the lanes. Only when no cage at this column threads does it draw from every cage. With
    # the column-side one-layer preference the cages the beam bars pass in fewer layers come first,
    # then the least steel: the first cage that satisfies every failed joint is installed.
    pool, _threaded = col_candidates_for_beam_bars(cfg.rebar.rho_col_min * ag, cfg.rebar.rho_col_practical_max * ag, cfg)
    candidates = sorted(pool, key=lambda c: scwb_cage_order(c, cfg))
    ordered = sorted(failing, key=lambda item: item[0]["details"]["ratio_provided"])

    def trial_record(candidate):
        bar_size, n_top, n_bot, n_side, _ast = candidate
        rebar = {**state["reinforcement"],
                 "col_bar_size": bar_size, "col_top_bars": n_top, "col_bot_bars": n_bot, "col_side_bars": n_side,
                 "col_bar_area_in2": sp.rebar_area(bar_size), "col_bar_diameter_in": sp.rebar_diameter(bar_size),
                 "col_longitudinal_centroid_offset_in": sp.longitudinal_cover_in("column", bar_size)}
        return {**state, "reinforcement": rebar}

    def provided(record, sway, axis):
        total = 0.0
        for column in sway["column_capacities"]:
            axials = {column.get("factored_axial_kip"), column.get("axial_min_kip"), column.get("axial_max_kip")}
            axials.discard(None)
            if not axials:
                raise ValueError("Failed SCWB check without a column axial envelope.")
            total += min(record_section_capacity(record, "column", axis, face, p)["mn_kip_in"]
                         for p in axials for face in ("positive", "negative"))
        return total

    from Design.SMRF_Common import SectionAxialDomainError
    rejected_domain = []

    def satisfies(candidate):
        record = trial_record(candidate)
        try:
            return all(provided(record, sway, axis) >= check["demand"] for check, sway, axis in ordered)
        except SectionAxialDomainError as exc:
            # The factored axial load of a failed column leaves this cage's strength domain: the cage is
            # infeasible for the load, by name, and a heavier cage (a larger domain) is tried next. The demand
            # and the domain stay in the message (2026-10-02, screening repair item 2; the grouped path
            # rejects the same way in Group_Selection). Nothing is clamped or extrapolated.
            rejected_domain.append({"cage": list(candidate[:4]), "ast_in2": candidate[4], "error": str(exc)})
            return False

    for candidate in candidates:
        if satisfies(candidate):
            bar_size, n_top, n_bot, n_side, _ast = candidate
            joint_scwb["axial_domain"] = _axial_domain_summary(rejected_domain, candidates)
            return {"bar_size": bar_size, "n_top": n_top, "n_bot": n_bot, "n_side": n_side}, False
    joint_scwb["axial_domain"] = _axial_domain_summary(rejected_domain, candidates)
    rejected_keys = {tuple(item["cage"]) for item in rejected_domain}
    priced = [c for c in candidates if tuple(c[:4]) not in rejected_keys]
    if not priced:
        # Every cage of this section is outside its domain for the load, or none was offered: the section
        # itself is too small; the search steps the column (design_structure, axial-domain escalation).
        return None, True
    # Exhausted within the practical ratio: install the cage that comes closest
    # on the worst joint so the shortfall is measured, and let the section grow.
    check, sway, axis = ordered[0]
    best = max(priced, key=lambda c: provided(trial_record(c), sway, axis))
    bar_size, n_top, n_bot, n_side, _ast = best
    return {"bar_size": bar_size, "n_top": n_top, "n_bot": n_bot, "n_side": n_side}, True


def _axial_domain_summary(rejected, candidates):
    """What the strong-column escalation rejected for the axial domain, kept in the history (public key)."""
    return {"rejected_cages": len(rejected), "cages_offered": len(candidates),
            "all_cages_rejected": bool(candidates) and len(rejected) == len(candidates),
            "first_rejection": rejected[0]["error"] if rejected else None,
            "rejected": rejected[:8]}


def _public(joint_scwb):
    return {key: value for key, value in joint_scwb.items() if not key.startswith("_")}


def _column_cage_rule_summary(cfg, capacity):
    """The column cage rule's state for one evaluated candidate (design basis 2026-09-27).

    ``threading_cages_available`` says whether any constructible cage at this column admits the beam
    bars in one layer; ``installed_cage_threads`` whether the cage the steel pass installed does. The
    planner reads the first when the second is False: cages exist, so the column grows; none exist, so
    the beam widens.
    """
    threading = capacity.get("bar_threading") or {}
    layers_needed = [d.get("layers_needed") for d in (threading.get("by_direction") or {}).values()]
    return {"basis": ("column cages are offered only where their lanes admit the beam bars of both directions within "
                      "the layer limit (Redesign.col_candidates_for_beam_bars: the strength pick and the strong-column "
                      "escalation); among those, cages the beam bars pass in fewer layers come first when "
                      "col_prefer_fewest_beam_layers is on; when no such cage exists at the column a strength cage "
                      "is installed and the beam widens; when they exist but strength or the strong-column rule "
                      "needed more steel, the column grows"),
            "threading_cages_available": column_cages_that_thread_exist(cfg),
            # ACI 318-19 18.7.4.3 (2026-10-02): cages whose bars cannot develop in half the clear height at
            # Ktr = 0 are never offered; when none can, the column steps (planner reason column_bar_bond).
            "bond_cages_available": column_cages_that_bond_exist(cfg),
            "installed_cage_bond": (capacity.get("column_bar_bond") or {}).get("passes"),
            "one_layer_preference": bool(cfg.rebar.col_prefer_fewest_beam_layers),
            "installed_cage_beam_layers": max((99 if n is None else n for n in layers_needed), default=None),
            "installed_cage_threads": threading.get("passes"),
            "bars_per_layer_installed": {axis: d.get("bars_per_layer_that_fit")
                                         for axis, d in (threading.get("by_direction") or {}).items()}}


def _scwb_steel_floor(cfg, actions, expected_ids):
    """Raise the installed column cage to the strong-column requirement.

    Evaluates the joints on the cage that is installed now and, when any
    joint fails, replaces the cage with _scwb_column_steel's pick. Returns
    the history summary with ``steel_raised`` and ``steel_exhausted``.
    """
    joint_scwb = _joint_scwb_state(actions, expected_ids)
    update, exhausted = (None, False)
    if not joint_scwb["all_pass"]:
        update, exhausted = _scwb_column_steel(cfg, joint_scwb)
        if update is not None:
            apply_updates(update, None, cfg)
    summary = _public(joint_scwb)
    summary["steel_raised"] = update is not None
    summary["steel_exhausted"] = exhausted
    return summary


def _steel_pass(cfg, model_period_sec, max_steel_iter, torsion=None):
    """Resize reinforcement at the current sections until the spec stops changing.

    The elastic design frame (gross section properties with constant
    stiffness modifiers) does not depend on the reinforcement, so every
    combination is solved once per section and the reinforcement iterations
    re-run only the checks on the captured actions. Keep every
    member/combination result: choosing one case by a summed DCR loses other
    governing actions and can under-design the opposite direction.

    Each iteration selects the bars for strength (Redesign.redesign_steel)
    and then raises the column cage to the per-joint strong-column
    requirement (_scwb_steel_floor); the pass ends when the installed bars
    stop changing. Returns worst DCRs, the ELF record, the actions and the
    joint SCWB summary for the cage that is installed on return.
    """
    from Design.SMRF_Demands import strength_load_combinations, live_load_patterns, REDUNDANCY_FACTOR_STRENGTH
    patterns = (live_load_patterns(sp.NUM_BAY_X, sp.NUM_BAY_Y)
                if cfg.demands.live_load_patterning and sp.FLOOR_TRANSFER is not None else ())
    combinations = strength_load_combinations(sp.ASCE_SDS, redundancy_factor=REDUNDANCY_FACTOR_STRENGTH,
                                              live_patterns=patterns)
    expected_ids = [combination["id"] for combination in combinations]
    if max_steel_iter < 1:
        raise ValueError("max_steel_iter must be positive.")

    elf_used = None
    actions = []
    for combination in combinations:
        elf = _analyze_combination(combination, model_period_sec, torsion)
        if elf is not None:
            elf_used = elf
        actions.append({**combination, "analysis_succeeded": True,
                        "axial_reference": "joint_faces",
                        "members": _capture_element_actions(combination["dead"])})

    worst = {"column": 0.0, "beam": 0.0}
    joint_scwb = None
    for steel_iteration in range(max_steel_iter):
        # Candidate bar diameter and effective depth change these limits.
        # Apply them before every check, including after a longitudinal update.
        _apply_transverse_geometry(cfg)
        worst = {"column": 0.0, "beam": 0.0, "beam_flexure_demand_kip_in": 0.0}
        governing_results = {}
        for action in actions:
            column_dcr, beam_dcr, results = _governing_dcrs(cfg, member_actions=action["members"])
            worst["column"] = max(worst["column"], column_dcr)
            worst["beam"] = max(worst["beam"], beam_dcr)
            worst["beam_flexure_demand_kip_in"] = max(worst["beam_flexure_demand_kip_in"], _beam_flexure_demand(results))
            governing_results.update({(action["id"], tag): result for tag, result in results.items()})
        checked_cage = _cage_signature()

        # Never return demands from the state before the last steel update.
        if steel_iteration == max_steel_iter - 1:
            break

        before = _cage_signature()
        column_update, beam_update, converged, _penalties = redesign_steel(governing_results, cfg)
        if not converged and (column_update or beam_update):
            apply_updates(column_update, beam_update, cfg)
        # The joint rule is priced on bar positions, and those sit inside the
        # hoops the capacity design selects for this cage (cover + hoop
        # diameter + db/2). Install those hoops first so the joint check here
        # sees the same section qualification will see; a #4 -> #5 hoop moves
        # the bars 1/16 in and is worth ~0.4% of column Mn, enough to turn
        # 1.204 in the loop into 1.1995 in qualification.
        _capacity_design(cfg, actions)
        joint_scwb = _scwb_steel_floor(cfg, actions, expected_ids)
        if _cage_signature() == before:
            break

    if joint_scwb is None or joint_scwb["steel_raised"]:
        # The cage changed after its last evaluation (or was never evaluated):
        # report the joints on the bars that are actually installed, inside
        # the hoops designed for them.
        _capacity_design(cfg, actions)
        joint_scwb = {**_public(_joint_scwb_state(actions, expected_ids)),
                      "steel_raised": False, "steel_exhausted": (joint_scwb or {}).get("steel_exhausted", False)}
    if _cage_signature() != checked_cage:
        # The bounded finalization can select hoops after the last strength
        # evaluation. Hoops affect both effective depth and shear resistance;
        # recheck the installed cage even when the steel budget is exhausted.
        worst = {"column": 0.0, "beam": 0.0, "beam_flexure_demand_kip_in": 0.0}
        for action in actions:
            column_dcr, beam_dcr, results = _governing_dcrs(cfg, member_actions=action["members"])
            worst["column"] = max(worst["column"], column_dcr)
            worst["beam"] = max(worst["beam"], beam_dcr)
            worst["beam_flexure_demand_kip_in"] = max(worst["beam_flexure_demand_kip_in"], _beam_flexure_demand(results))
    return worst, elf_used, actions, joint_scwb


def _repair_bar_stacking(cfg, actions, worst, joint_scwb, capacity):
    """Try actual supported beam row arrangements before changing concrete geometry.

    Only a candidate otherwise passing strength, joints and capacity design enters
    this repair. Every trial starts from the same installed cage. Changed rows are
    installed in SP, so strengths, joint checks and later IMK hinges share them.
    Slab mats stay at their designed elevations; no displacement is inferred.
    """
    from Design.SMRF_Capacity_Design import design_bar_threading
    from Design.SMRF_Beam_Slab_Strength import BeamBarRowsError
    report = {"status": "not_needed", "trials": []}
    if not capacity.get("bar_threading", {}).get("slab_clashes"):
        return worst, joint_scwb, capacity, report
    failed = [c["id"] for c in capacity["checks"] if c["status"] != "pass"]
    if (set(failed) != {"beam.bar_stacking_clear_of_slab_mats"}
            or max(worst["beam"], worst["column"]) > cfg.dcr.dcr_hard_max
            or not joint_scwb.get("evaluated") or not joint_scwb.get("all_pass")
            or (joint_scwb.get("counts") or {}).get("not_evaluated")):
        report.update(status="other_constraints_fail", failed_checks=failed)
        return worst, joint_scwb, capacity, report

    seed = _capture_state()
    original = (sp.beam_bar_stacking_convention(), sp.BEAM_BAR_LAYER_ORDER)
    arrangements = [(axis, order) for order in (original[1], "blocked", "interleaved")
                    for axis in (original[0], "x_over_y", "y_over_x")]
    seen = {original}
    selected = False
    try:
        for axis, order in arrangements:
            if (axis, order) in seen:
                continue
            seen.add((axis, order))
            _restore_state(seed)
            _sync_cfg_to_sp(cfg)
            sp.BEAM_BAR_STACKING, sp.BEAM_BAR_LAYER_ORDER = axis, order
            trial = {"convention": axis, "layer_order": order}
            report["trials"].append(trial)
            try:
                # Cheap geometric rejection before the full capacity closure.
                threading = design_bar_threading(_capacity_state(cfg, actions))
                if threading["slab_clashes"] or not threading["passes"]:
                    trial.update(status="geometry_failed", slab_clashes=threading["slab_clashes"])
                    continue
                candidate_capacity = _capacity_design(cfg, actions)
                candidate_worst = {"column": 0., "beam": 0., "beam_flexure_demand_kip_in": 0.}
                for action in actions:
                    col, beam, results = _governing_dcrs(cfg, member_actions=action["members"])
                    candidate_worst["column"] = max(candidate_worst["column"], col)
                    candidate_worst["beam"] = max(candidate_worst["beam"], beam)
                    candidate_worst["beam_flexure_demand_kip_in"] = max(
                        candidate_worst["beam_flexure_demand_kip_in"], _beam_flexure_demand(results))
                candidate_joint = {**_public(_joint_scwb_state(actions, [a["id"] for a in actions])),
                                   "steel_raised": False, "steel_exhausted": False}
                passed = (candidate_capacity["accepted"]
                          and max(candidate_worst["beam"], candidate_worst["column"]) <= cfg.dcr.dcr_hard_max
                          and candidate_joint["evaluated"] and candidate_joint["all_pass"]
                          and not (candidate_joint.get("counts") or {}).get("not_evaluated"))
                trial.update(status="accepted" if passed else "checks_failed", dcr=candidate_worst,
                             joint_scwb=candidate_joint,
                             failed_checks=[c["id"] for c in candidate_capacity["checks"] if c["status"] != "pass"],
                             layers=sp.beam_bar_layers())
                if passed:
                    selected = True
                    report["status"] = "resolved"
                    return candidate_worst, candidate_joint, candidate_capacity, report
            except BeamBarRowsError as error:
                trial.update(status="geometry_failed", error=str(error))
        report["status"] = "arrangements_exhausted"
        return worst, joint_scwb, capacity, report
    finally:
        if not selected:
            _restore_state(seed)
            _sync_cfg_to_sp(cfg)


def _analyze_bar_layout(cfg, slab, slab_search, max_steel_iter):
    """Analyze fixed member sections, trying stacking before the slab thickness ladder.

    A thicker slab changes mass, floor transfer and strip actions: rebuild all of
    them and rerun frame demands and reinforcement, not just the geometry check.
    Each thickness starts from the same requested cage. The policy's finite slab
    ladder bounds retries; numerical failures propagate as unresolved analyses.
    """
    cage_seed = {key: copy.deepcopy(getattr(sp, key)) for key in _STATE_KEYS
                 if key.startswith(("BEAM_", "COL_"))}
    layout_trials = []
    while True:
        for key, value in cage_seed.items():
            setattr(sp, key, copy.deepcopy(value))
        _sync_cfg_to_sp(cfg)
        period = _model_period()
        torsion = _torsion_assessment(period, cfg) if cfg.demands.accidental_torsion_ratio else None
        worst, elf, actions, joint = _steel_pass(cfg, period, max_steel_iter, torsion)
        capacity = _capacity_design(cfg, actions)
        worst, joint, capacity, repair = _repair_bar_stacking(cfg, actions, worst, joint, capacity)
        layout_trials.append({"thickness_in": slab["thickness_in"], **repair})
        if repair["status"] != "arrangements_exhausted":
            break
        thicker, unavailable = _thicker_slab(cfg, slab)
        if thicker is None:
            layout_trials[-1]["next_thickness_unavailable"] = unavailable
            break
        slab = thicker
        _update_floor_transfer(cfg, slab)
        slab, next_search = _slab_thickness_search(cfg, slab)
        slab_search["steps"] += 1 + next_search["steps"]
        slab_search["trials"].extend(next_search["trials"])
        slab_search["thickness_in"] = slab["thickness_in"]
    slab_search["bar_layout_trials"] = layout_trials
    return slab, period, torsion, worst, elf, actions, joint, capacity


def _feasible_beam_indices(beams, column_b_in, column_h_in, span_x_in=None, span_y_in=None, story_h_in=None):
    """Beam rungs that fit BOTH clear spans under the given column dimensions.

    ACI 318-19 18.6.2.1(a) needs ln >= 4d at every span: the x span clears
    the column depth h, the y span its width b. The spans and story height
    default to the live geometry; the beam bar centroid offset comes from
    the current bar and hoop sizes. Sorted ladder indices.
    """
    span_x = sp.BAY_X if span_x_in is None else span_x_in
    span_y = sp.BAY_Y if span_y_in is None else span_y_in
    story = sp.STORY_H if story_h_in is None else story_h_in
    feasible = []
    for index, rung in enumerate(beams):
        try:
            for span, column_depth in ((span_x, column_h_in), (span_y, column_b_in)):
                validate_rung(rung, "beam", span_in=span, story_height_in=story,
                              column_depth_in=column_depth,
                              effective_depth_in=rung[1] - sp.longitudinal_cover_in("beam"))
        except ValueError:
            continue
        feasible.append(index)
    return feasible


def _compatible_beam_index(beams, preferred, column=None):
    """One shared beam section must fit BOTH directions and the given (default: current) columns.

    Returns the first feasible rung at or above ``preferred`` (the ladder is
    ordered by capacity), so an escalation whose target rung the clear span
    forbids (18.6.2.1(a)) lands on the next rung that fits rather than on a
    lighter one; only when nothing above fits does the heaviest feasible
    rung below apply. ``column`` = (b, h, fc) plans against a column rung
    that is not installed yet.
    """
    b_col, h_col = (sp.B_COL, sp.H_COL) if column is None else (column[0], column[1])
    feasible = _feasible_beam_indices(beams, b_col, h_col)
    if not feasible:
        raise ValueError("No common beam section fits both clear spans with these columns.")
    above = [index for index in feasible if index >= preferred]
    return min(above) if above else max(feasible)


def _drift_screen(model_period_sec, torsion=None, cfg=None):
    """Separate rho=1 QEx/QEy runs; envelope all structural floor nodes."""
    from Design.SMRF_Elastic import floor_xy_displacements, gravity_weight_above_story
    from Design.SMRF_Demands import story_node_deltas, evaluate_drift_and_stability, seismic_design_category
    from Design.SMRF_Common import not_evaluated, summarize_checks
    checks, rows = [], []
    # One risk basis for the category, Ie and the drift limit: a DemandPolicy that names another
    # category, or an ASCE_IE that is not that category's, raises here instead of being screened.
    risk = cfg.demands.risk_category if cfg else sp.ASCE_RISK_CATEGORY
    sp.seismic_design_basis(risk_category=risk)
    try:
        sdc = seismic_design_category(sp.ASCE_SDS, sp.ASCE_SD1, sp.ASCE_S1, risk)
    except ValueError:
        sdc = "D"
    limit_sdc = sdc if sdc in ("D", "E", "F") else "D"
    basis = sp.seismic_design_basis(risk_category=risk, redundancy_factor=1.3, seismic_design_category=limit_sdc)
    for axis in ("x", "y"):
        combination = {"id": f"drift_{axis}", "dead": 1.0, "live": 1.0,
                       "ex": float(axis == "x"), "ey": float(axis == "y"), "live_pattern": "all"}
        try:
            elf = _analyze_combination(combination, model_period_sec, torsion)
            stories = []
            for k in range(1, sp.NUM_FLOOR + 1):
                deltas = story_node_deltas(floor_xy_displacements(k), floor_xy_displacements(k - 1))
                stories.append({"id": f"{k}/Q{axis}", "height_in": sp.STORY_H,
                                "node_deltas_in": deltas, "expected_node_ids": [
                                    f"{i},{j}" for j in range(sp.NUM_BAY_Y + 1)
                                    for i in range(sp.NUM_BAY_X + 1)],
                                "directions": [axis], "story_shear_kip": {
                                    axis: sum(elf["story_forces_kip"][k - 1:])},
                                "gravity_above_kip": gravity_weight_above_story(k)})
            result = evaluate_drift_and_stability(stories, importance_factor=basis["importance_factor"],
                                                  risk_category=basis["risk_category"],
                                                  drift_limit_ratio=basis["allowable_story_drift_ratio"],
                                                  redundancy_factor=basis["redundancy_factor"],
                                                  seismic_design_category=limit_sdc,
                                                  second_order_included=True)
            checks.extend(result["checks"])
            rows.extend(result["stories"])
        except RuntimeError as exc:
            checks.append(not_evaluated(f"demands.drift_analysis_{axis}", "ASCE 7-22 12.8.6--12.8.7", str(exc)))
    return {"checks": checks, "stories": rows,
            "assumptions": {"cd": 5.5, "rho_for_drift_load": 1.0, "rho_for_limit": basis["redundancy_factor"],
                            "limit_scope": (f"Risk Category {basis['risk_category']} solely moment frame; Table 12.12-1 limit "
                                            "divided by rho as for SDC D/E/F (a lower SDC is screened as D, conservatively); "
                                            "no low-rise allowance"),
                            "risk_category": basis["risk_category"], "importance_factor": basis["importance_factor"],
                            "base_drift_limit_ratio": basis["allowable_story_drift_ratio"],
                            "effective_drift_limit_ratio": basis["effective_story_drift_ratio"],
                            "low_rise_allowance_asserted": basis["low_rise_allowance_asserted"],
                            "limit_seismic_design_category": limit_sdc,
                            "seismic_design_category": sdc,
                            "accidental_torsion_included": bool(torsion),
                            "gravity": "uniform full D+L plus member weight",
                            "model": "elastic cracked stiffness; PDelta columns; centerline joints"},
            **summarize_checks(checks)}


def _scwb_required_column_moment():
    """Column nominal moment needed to satisfy ACI 318-19 18.7.3.2 at every joint.

    sum(Mnc) >= 1.2 sum(Mnb) at each joint, with no roof exemption (the
    joint qualification applies none). Members are uniform over height, so
    the governing joint is an interior roof joint: one column below against
    a hogging beam on one side and a sagging beam on the other,
    Mnc >= 1.2 (Mnb- + Mnb+), evaluated at the low roof axial load. Floor
    joints, with two columns, need only 0.6 (Mnb- + Mnb+) and never govern.
    Slab mats within the effective flange count toward Mnb (18.7.3.2); the
    composite values are the same ones the beam hinges yield at.
    """
    families = _beam_strength_families()
    total = sp.SCWB_RATIO_MIN * max(f["mn_negative_kip_in"] + f["mn_positive_kip_in"]
                                    for f in families.values())
    if not _scwb_roof_exception_applies():
        return total
    # 18.7.3.1 (user decision 2026-10-02): the roof connection is excepted, so the governing joint is the
    # floor below it, where two columns share the beam sum; a one-story frame has no joint the rule applies to.
    return 0.5 * total if sp.NUM_FLOOR >= 2 else 0.0


def _scwb_roof_exception_applies():
    """Sizing proxy of the ACI 318-19 18.7.3.1 exception at the roof connections.

    The largest roof-column tributary gravity axial load, times 1.5 as a bound on the factored combinations,
    against Ag f'c / 10 of the installed column. A proxy for the sizing screen only: the per-joint check
    (SMRF_Joints.scwb_check) decides on the solved factored axial envelope.
    """
    from Design.SMRF_Joints import SCWB_DISCONTINUOUS_COLUMN_EXCEPTION
    if not SCWB_DISCONTINUOUS_COLUMN_EXCEPTION:
        return False
    axial = max(column_gravity_axial(max(1, sp.NUM_FLOOR), i, j)
                for i in range(sp.NUM_BAY_X + 1) for j in range(sp.NUM_BAY_Y + 1))
    return 1.5 * axial < sp.B_COL * sp.H_COL * sp.FC_COL_KSI / 10.0


def _scwb_governing_axial():
    """Legacy top-corner gravity estimate used ONLY as a sizing proxy.

    This is not the factored axial envelope required at each joint. Both
    low/tensile and high compression forces can govern the actual PM check.
    Full SCWB qualification remains open until that envelope is implemented.
    """
    return column_gravity_axial(max(1, sp.NUM_FLOOR), 0, 0)


def _column_nominal_moment(section=None):
    """Column Mn from the nominal P-M surface at the governing axial load.

    sp.column_nominal_moment_y() is a singly-reinforced BEAM formula: it
    counts only max(top, bottom) bars -- ignoring the side steel and the
    opposite face -- and assumes zero axial load. For a column that
    understates Mn by roughly 2x at zero axial and 4x under service
    compression, which drove this search to demand columns several times
    larger than ACI actually requires, and made SCWB unsatisfiable for
    frames that in fact comply. Model/IMK_Hinges.py already takes hinge
    capacity off the P-M surface for exactly this reason; using it here
    makes the design side agree with the model that gets analysed.

    section is (b, h, fc); defaults to the section currently installed. A
    candidate section is priced with at least the ACI 318-19 18.7.4.1
    minimum of 1% longitudinal steel (the current cage scaled up in place):
    pricing a larger rung with the smaller section's bars, which it cannot
    legally keep, made the ladder overshoot by a rung or two.
    """
    installed = section is None
    if installed:
        section = (sp.B_COL, sp.H_COL, sp.FC_COL_KSI)
    b, h, fc = section
    layers = _col_steel_layers(h=h)
    if not installed:
        total = sum(area for area, _ in layers)
        minimum = 0.01 * b * h
        if 0 < total < minimum:
            layers = [(area * minimum / total, depth) for area, depth in layers]
    diagram = column_pm_nominal_for(b, h, fc, layers)
    return column_moment_at_axial(_scwb_governing_axial(), diagram)


def _next_larger_column_same_grade(ladder, index):
    """The next column size at the concrete grade of ladder[index]; None at the largest size."""
    b, _h, fc = ladder[index]
    larger = sorted({rung[0] for rung in ladder if rung[0] > b})
    return ladder.index((larger[0], larger[0], fc)) if larger else None


def _column_strength_first_steps(ladder, index):
    """(same size at the top concrete grade, next larger size at the top grade); None where there is none.

    The first is None when ladder[index] is already at the top grade of its size.
    """
    b, h, fc = ladder[index]
    same = [i for i, rung in enumerate(ladder) if rung[:2] == (b, h) and rung[2] > fc]
    top_same = max(same, key=lambda i: ladder[i][2]) if same else None
    larger_sizes = sorted({rung[0] for rung in ladder if rung[0] > b})
    larger_top = None
    if larger_sizes:
        candidates = [i for i, rung in enumerate(ladder) if rung[0] == larger_sizes[0]]
        larger_top = max(candidates, key=lambda i: ladder[i][2])
    return top_same, larger_top


def _next_larger_column_index(ladder, index):
    """First rung with a strictly larger cross-section than ladder[index].

    The ladder interleaves concrete strengths within each size (18x18 at 4, 5,
    6, 8 ksi, then 20x20 at 4, ...), so stepping one rung usually only raises
    f'c. That is the wrong lever for a gravity stability failure: flexural
    stiffness goes as E*I, and E rises with sqrt(f'c), so 5 -> 8 ksi buys about
    26% while 18 -> 20 in buys (20/18)^4 = 1.52x. Skip to the next real size.

    Returns None when the ladder has no larger section.
    """
    b, h, _fc = ladder[index]
    for candidate in range(index + 1, len(ladder)):
        cb, ch, _ = ladder[candidate]
        if cb > b or ch > h:
            return candidate
    return None


def _next_deeper_beam_index(ladder, index):
    """First rung above ladder[index] with a strictly deeper section.

    The beam ladder is ordered by capacity and mixes f'c and width steps
    with depth steps. Story drift answers to depth, so that escalation takes
    the next deeper rung. None at the top.
    """
    _b, h, _fc = ladder[index]
    for candidate in range(index + 1, len(ladder)):
        if ladder[candidate][1] > h:
            return candidate
    return None


def _next_wider_beam_index(ladder, index):
    """First rung above ladder[index] with the same depth and a wider web.

    The capacity-shear section limit (22.5.1.2 with 18.6.5.1 Ve) answers to
    bw d; on short clear spans 18.6.2.1(a) caps d, so width is the lever.
    None when the depth has no wider variant above the current rung.
    """
    b, h, _fc = ladder[index]
    for candidate in range(index + 1, len(ladder)):
        if ladder[candidate][1] == h and ladder[candidate][0] > b:
            return candidate
    return None


def _next_stronger_index(ladder, index):
    """First rung above ladder[index] with the same dimensions and a higher f'c.

    The capacity-shear section limit 8 sqrt(f'c) bw d (22.5.1.2), the joint
    strength gamma sqrt(f'c) Aj (18.8.4.1) and the elastic stiffness
    E = 57000 sqrt(f'c) all rise with concrete strength at fixed
    dimensions, so a rung with no larger dimension left still has material
    candidates: case_0013 stopped at a 20x32 fc-4 beam with the fc-5/6/8
    rungs unvisited, case_0073 at a 36x36 fc-6 column with fc-8 unvisited.
    None when the dimensions are already at their top strength. Works on
    either ladder: same dimensions at higher f'c always sit above (the
    column ladder interleaves f'c within a size, the beam ladder is
    proxy-ordered and the proxy grows with sqrt(f'c)).
    """
    b, h, fc = ladder[index]
    for candidate in range(index + 1, len(ladder)):
        cb, ch, cfc = ladder[candidate]
        if (cb, ch) == (b, h) and cfc > fc:
            return candidate
    return None


def _column_index_for_joint_shear(ladder, index, ratio):
    """First rung whose joint area covers the worst joint-shear ratio.

    ACI 318-19 18.8.4.1 gives Vn = gamma sqrt(f'c) Aj, so at fixed gamma a
    joint short by ``ratio`` needs b*h*sqrt(f'c) scaled by it. gamma itself
    can fall when a wider column loses a confined face (18.8.4.2), so this
    is the sizing jump; the next evaluation decides.
    """
    b, h, fc = ladder[index]
    required = b * h * math.sqrt(fc) * ratio
    for candidate in range(index + 1, len(ladder)):
        cb, ch, cfc = ladder[candidate]
        if cb * ch * math.sqrt(cfc) >= required:
            return candidate
    return len(ladder) - 1


def _plan_next_rungs(columns, beams, column_index, beam_index, worst, target, hard_max,
                     scwb_index, flags):
    """Decide the next (column, beam) rungs from this iteration's evaluation.

    Pure: the search loop supplies the ladders, the current rungs, the worst
    DCRs, the strength-screen SCWB rung and the evaluation ``flags``
    (drift_ok, scwb_ok, joint_scwb_failed, capacity_accepted,
    beam_section_adequate, beam_hoops_selected, column_section_adequate,
    column_hoops_selected, joints_all_pass, anchorage_all_pass, and
    joint_shear_ratio: the worst Vj / phi Vn, or None).

    Strength sizes both members toward their targets; every requirement that
    the strength screen does not see moves the member it depends on: drift
    takes the next beam depth, the beam capacity-shear section takes the
    wider variant of the current depth (then the next depth), a joint
    that fails 18.7.3.2 after the steel pass exhausted the column cage takes
    the next column size, joint shear, anchorage and column shear take the
    next column size. When a member has no larger dimension left, the same
    dimensions at the next concrete strength are the step (the shear
    section limit, joint strength and stiffness all grow with sqrt(f'c));
    only a beam with neither dimension nor strength left moves the column.
    While any requirement is unmet neither member steps down, so the search
    cannot trade one failure for another and cycle. Returns
    (next_column, next_beam, reasons); feasibility against the clear span
    and the visited set are applied by _plan_next_candidate.
    """
    reasons = []
    next_beam = suggest_rung_index(beams, beam_index, max(worst["beam"], 1e-6), target)
    moment = worst.get("beam_flexure_demand_kip_in")
    if moment and next_beam > beam_index:
        # Gentler beam growth (user decision 2026-10-02). The proxy jump scales the section by DCR / target,
        # and a beam the steel pass could not reinforce reports its DCR on the seed bars (5 to 12 in the
        # 2 October screens), which sent a 12 x 24 beam straight to 30 x 36 at 8 ksi and left columns the
        # ladder cannot supply. The governing moment at a moderate steel ratio names the rung the demand
        # asks for; the proxy jump is kept as the upper bound and the next evaluation decides.
        next_beam = min(next_beam, suggest_beam_rung_for_moment(beams, beam_index, moment, sp.FY_KSI))
    stalled_beam = bool(moment) and worst["beam"] > hard_max and next_beam == beam_index
    strength_index = suggest_rung_index(columns, column_index, max(worst["column"], 1e-6), hard_max)
    next_column = max(strength_index, scwb_index)
    if worst["beam"] > hard_max:
        reasons.append("beam_strength")
    if worst["column"] > hard_max:
        reasons.append("column_strength")
    if not flags["scwb_ok"] and scwb_index > column_index:
        reasons.append("scwb_screen")
    larger_column = _next_larger_column_index(columns, column_index)
    stronger_column = _next_stronger_index(columns, column_index)
    top_grade_column, larger_at_top_grade = _column_strength_first_steps(columns, column_index)
    deeper_beam = _next_deeper_beam_index(beams, beam_index)
    wider_beam = _next_wider_beam_index(beams, beam_index)
    stronger_beam = _next_stronger_index(beams, beam_index)

    # The dimension a beam step pulls on (depth for drift, width for the shear section and the bar
    # threading) is held as a floor: the ladder is proxy-ordered, so a wider, shallower rung can sit
    # above the step and the index alone would not keep the lever (2026-09-27, four width variants).
    floors = {"depth": 0.0, "width": 0.0}

    def grow_beam(prefer_width=False):
        nonlocal next_beam, next_column
        order = ([wider_beam, deeper_beam] if prefer_width else [deeper_beam, wider_beam]) + [stronger_beam]
        step = next((candidate for candidate in order if candidate is not None), None)
        if step is not None:
            next_beam = max(next_beam, step)
            floors["depth"] = max(floors["depth"], beams[step][1])
            if step != deeper_beam:
                floors["width"] = max(floors["width"], beams[step][0])
        elif larger_column is not None:
            next_column = max(next_column, larger_column)
        elif stronger_column is not None:
            next_column = max(next_column, stronger_column)

    def grow_column():
        # Strength before geometry (user decision 2026-10-03): a failed column requirement takes the same
        # size at the top concrete grade first; only a column already at the top grade takes the next
        # size, and takes it at the top grade. The step-down pass lowers the grade again where it can.
        nonlocal next_column
        step = top_grade_column if top_grade_column is not None else larger_at_top_grade
        if step is not None:
            next_column = max(next_column, step)

    if stalled_beam:
        # The moment fits this rung at the growth steel ratio, yet no bar layout carries it (the lanes, the
        # 20 db joint depth or the layer limit): the section is not short of depth but of room for bars.
        # A beam narrower than three quarters of the column widens (r2 screen, case_0002: three evaluations
        # at 10 x 20 with DCR 2.5 on the seed bars). A beam already that wide, or with no wider variant, has
        # the column as its limit: the lanes between the column bars and the 20 db joint depth (18.8.2.3) are
        # geometry, so the column takes the next SIZE at its grade; a higher grade makes no room (r4 screen,
        # case_0010, 2026-10-03: sixteen evaluations at a 22-in column with the beam on its seed bars).
        if wider_beam is None or beams[beam_index][0] >= 0.75 * columns[column_index][0]:
            roomier = _next_larger_column_same_grade(columns, column_index)
            if roomier is not None:
                reasons.append("beam_bars_need_column_room")
                next_column = max(next_column, roomier)
            else:
                grow_beam(prefer_width=True)
        else:
            grow_beam(prefer_width=True)
    if not flags["drift_ok"]:
        reasons.append("drift")
        grow_beam()
    if not flags["beam_section_adequate"] or not flags["beam_hoops_selected"]:
        reasons.append("beam_capacity_shear")
        grow_beam(prefer_width=True)
    if not flags.get("beam_bars_thread", True):
        # The beam bars do not pass between the installed column bars in one layer.
        reasons.append("beam_bar_threading")
        if flags.get("column_cage_can_thread", False):
            # Cages that admit the bars exist at this column (the column cage rule offered them) but
            # strength or the strong-column rule needed more steel than they hold: a larger column
            # keeps the lanes and needs less steel, so the column is the lever.
            reasons.append("column_cage_threading")
            grow_column()
        else:
            # No cage at this column admits the bars: the beam band is too narrow, width is the lever.
            grow_beam(prefer_width=True)
    if not flags.get("column_bar_bond_ok", True):
        # ACI 318-19 18.7.4.3: the installed bars do not develop in half the clear height at Ktr = 0. ld
        # falls with sqrt(f'c) and a larger section admits smaller bars, so the column steps.
        reasons.append("column_bar_bond")
        grow_column()
    if flags["joint_scwb_failed"]:
        reasons.append("joint_scwb")
        grow_column()
    if not flags["column_section_adequate"] or not flags["column_hoops_selected"]:
        reasons.append("column_capacity_shear")
        grow_column()
    if not flags["joints_all_pass"] or not flags["anchorage_all_pass"]:
        reasons.append("joint_shear_or_anchorage")
        # Table 18.8.4.3: a joint confined by transverse beams at least 3/4 of the column face (15.2.8) gains a
        # quarter in gamma (12 to 15 at a roof, 15 to 20 at a floor). Where the failing joints lack only that
        # width and the current depth has a variant that wide, the beam widens first; a larger column would
        # lower the width ratio further. The next evaluation decides, and a joint that still fails then moves
        # the column (r2 screen, 2026-10-02: roof joints at gamma 12 held 0004 at 30 in and 0005 at 40 in).
        width_target = flags.get("joint_confinement_width_in")
        widened = None
        if width_target and flags["anchorage_all_pass"]:
            widened = next((i for i in range(beam_index + 1, len(beams))
                            if beams[i][1] == beams[beam_index][1] and beams[i][2] == beams[beam_index][2]
                            and beams[i][0] >= width_target), None)
        if widened is not None:
            reasons.append("joint_confinement_width")
            next_beam = max(next_beam, widened)
            floors["depth"] = max(floors["depth"], beams[widened][1])
            floors["width"] = max(floors["width"], beams[widened][0])
        else:
            grow_column()
            ratio = flags.get("joint_shear_ratio")
            if ratio is not None and ratio > 1.0:
                next_column = max(next_column, _column_index_for_joint_shear(columns, column_index, ratio))
    if floors["depth"] or floors["width"]:
        held = next((i for i in range(next_beam, len(beams))
                     if beams[i][1] >= floors["depth"] and beams[i][0] >= floors["width"]), None)
        if held is not None:
            next_beam = held
    unmet = (reasons or not flags["scwb_ok"] or not flags["capacity_accepted"]
             or not flags["drift_ok"])
    if unmet:
        next_column = max(next_column, column_index)
        next_beam = max(next_beam, beam_index)
    return next_column, next_beam, reasons


# Which member a step reason moves; drift moves the beam first and falls to the column.
BEAM_STEP_REASONS = ("beam_strength", "drift", "beam_capacity_shear", "beam_bar_threading", "joint_confinement_width")
COLUMN_STEP_REASONS = ("column_strength", "scwb_screen", "joint_scwb", "column_capacity_shear",
                       "joint_shear_or_anchorage", "drift", "column_cage_threading", "column_bar_bond",
                       "beam_bars_need_column_room")
STOP_REASONS = ("candidate_screen_passed", "iteration_budget_exhausted", "no_candidate_under_strategy",
                "candidate_set_exhausted", "repeated_candidate_after_substitution")


def _plan_next_candidate(columns, beams, column_index, beam_index, worst, target, hard_max, scwb_index, flags,
                         visited, feasible_beams):
    """The next unvisited, feasible candidate pair from the planner's proposal, or an explicit stop.

    ``feasible_beams(column_index)`` returns the sorted beam indices that
    fit both clear spans under that column rung; ``visited`` holds the
    (column, beam) pairs already evaluated, the current one included. The
    proposal is kept as proposed. A substitution -- the clear-span rule
    under the proposed columns (18.6.2.1(a)), or a pair already evaluated
    -- is recorded with its reason, and the candidate becomes the first
    unvisited feasible pair at or above the proposal, advanced along the
    member the step reasons point at (beam reasons move the beam, column
    reasons the column, drift either). When no such pair exists the plan
    carries a stop reason: ``no_candidate_under_strategy`` when unvisited
    feasible pairs remain at or above the current pair that this strategy
    does not reach, ``candidate_set_exhausted`` when none remain. Neither
    says anything about the geometry: the domain is the declared ladder
    pair under the never-step-down rule, and the iteration budget ends the
    search elsewhere.
    """
    proposed_column, proposed_beam, reasons = _plan_next_rungs(columns, beams, column_index, beam_index, worst,
                                                               target, hard_max, scwb_index, flags)
    plan = {"reasons": reasons, "proposed": {"column": proposed_column, "beam": proposed_beam},
            "substitutions": [], "candidate": None, "stop_reason": None, "stop_detail": None}
    beam_moves = any(reason in BEAM_STEP_REASONS for reason in reasons)
    column_moves = any(reason in COLUMN_STEP_REASONS for reason in reasons)
    if not beam_moves and not column_moves:
        beam_moves = column_moves = True      # unmet without a named reason (a screen flag alone): try both

    def settle(column, preferred):
        options = feasible_beams(column)
        if not options:
            return None
        above = [j for j in options if j >= preferred]
        return min(above) if above else max(options)

    def unvisited_at_or_above(column_floor, beam_floor):
        return [(c, j) for c in range(column_floor, len(columns))
                for j in feasible_beams(c) if j >= beam_floor and (c, j) not in visited]

    column, beam = proposed_column, settle(proposed_column, proposed_beam)
    if beam is None:
        plan.update(stop_reason="candidate_set_exhausted",
                    stop_detail=f"no beam rung fits both clear spans under column rung {list(columns[proposed_column])}")
        return plan
    if beam != proposed_beam:
        plan["substitutions"].append({"stage": "clear_span_compatibility", "requested_beam": proposed_beam, "beam": beam,
                                      "reason": ("ACI 318-19 18.6.2.1(a) ln >= 4d under the proposed columns: the requested "
                                                 "rung does not fit; next feasible rung at or above it, else the heaviest "
                                                 "feasible rung")})
    if (column, beam) in visited or (column, beam) == (column_index, beam_index):
        requested = (column, beam)
        found = None
        if beam_moves:
            found = next(((column, j) for j in feasible_beams(column) if j > beam and (column, j) not in visited), None)
        if found is None and column_moves:
            for c in range(column + 1, len(columns)):
                j = settle(c, beam)
                if j is not None and (c, j) not in visited:
                    found = (c, j)
                    break
        if found is None:
            remaining = unvisited_at_or_above(column_index, beam_index)
            if remaining:
                plan.update(stop_reason="no_candidate_under_strategy",
                            stop_detail=(f"{len(remaining)} unvisited feasible ladder pair(s) remain at or above the current "
                                         f"pair, none reachable by the step the failed requirements call for "
                                         f"({', '.join(reasons) or 'unnamed screen'})"))
            else:
                plan.update(stop_reason="candidate_set_exhausted",
                            stop_detail=("no unvisited feasible ladder pair remains at or above the current pair "
                                         "(never-step-down strategy over the declared column and beam ladders)"))
            return plan
        plan["substitutions"].append({"stage": "visited_pair", "requested": list(requested), "candidate": list(found),
                                      "reason": ("the proposed pair was already evaluated; advanced to the next unvisited "
                                                 "feasible pair along the member the failed requirements move")})
        column, beam = found
    plan["candidate"] = {"column": column, "beam": beam}
    return plan


def _column_size_exception(history, columns):
    """A selected column past the last-resort size (Section_Design.COLUMN_SIZE_LAST_RESORT_IN), with what the
    top-grade columns at or below that size failed; None when the selected column is within it."""
    from Design.Section_Design import COLUMN_SIZE_LAST_RESORT_IN
    if max(sp.B_COL, sp.H_COL) <= COLUMN_SIZE_LAST_RESORT_IN:
        return None
    top = max(rung[2] for rung in columns)
    tried = []
    for entry in history:
        size, _h, fc = entry["column_section"]
        constraints = entry.get("constraints") or {}
        if size <= COLUMN_SIZE_LAST_RESORT_IN and fc == top and not constraints.get("candidate_screen_passed"):
            tried.append({"column": entry["column_section"], "beam": entry["beam_section"],
                          "failed_capacity_checks": sorted({c["id"] for c in constraints.get("capacity_design_failed_checks", [])}),
                          "drift_accepted": constraints.get("drift_accepted"),
                          "strength_within_ceiling": constraints.get("strength_within_ceiling"),
                          "joint_scwb_all_pass": (constraints.get("joint_scwb") or {}).get("all_pass")})
    return {"selected_column": [sp.B_COL, sp.H_COL, sp.FC_COL_KSI], "last_resort_size_in": COLUMN_SIZE_LAST_RESORT_IN,
            "top_grade_ksi": top, "top_grade_trials_at_or_below_the_size": tried,
            "top_grade_tried_at_or_below_the_size": bool(tried),
            "note": "large geometry is the last resort (user decision 2026-10-03); review what the listed trials failed"}


def _candidate_quantities():
    """Use the existing material ledger on the uniform frame, including hoops and slab mats."""
    from Design.Grouped_Design import quantities
    from Model.Member_Groups import GroupedDesign
    result = quantities(state=GroupedDesign.uniform(), slab_reinforcement=sp.SLAB_REINFORCEMENT)
    return {key: result[key] for key in ("concrete_total_in3", "total_steel_in3", "distinct_form_size_count")} | {
        "column_fc_ksi": sp.FC_COL_KSI, "beam_fc_ksi": sp.FC_BEAM_KSI,
        "basis": result["basis"]}


def _column_reduction_start(columns, column_index, beam_index, snapshot, beams, budget):
    from Design.Uniform_Search import start
    search = start(columns, beams, column_index, beam_index, snapshot, budget)
    search["beams"] = beams
    return search


def _column_reduction_next(reduction, columns):
    from Design.Uniform_Search import next_pair
    return next_pair(reduction)


def _column_reduction_advance(reduction, columns, column_index, accepted, note=None, snapshot=None,
                              constraints=None, outcome=None, actual_beam_index=None):
    from Design.Uniform_Search import advance
    return advance(reduction, columns, reduction["beams"], column_index, accepted, note, snapshot,
                   constraints, outcome, actual_beam_index)


def _constraint_summary(worst, hard_max, scwb_screen_ok, joint_scwb, capacity, drift_screen, accepted):
    """Every acceptance constraint of one evaluated candidate, with its failures spelled out.

    Kept per history entry so the selected iteration and the last iteration
    each carry their own record: a candidate the objective prefers may fail
    one constraint while the last candidate evaluated fails another
    (case_0138: beam shear at the saved iteration, SCWB at the last).
    """
    failing = [{"id": c["id"], "location": c.get("location", ""), "status": c["status"],
                "demand": c.get("demand"), "capacity": c.get("capacity"), "units": c.get("units", "")}
               for c in capacity["checks"] if c["status"] != "pass"]
    return {"candidate_screen_passed": accepted,
            "strength_within_ceiling": max(worst["beam"], worst["column"]) <= hard_max,
            "scwb_screen_satisfied": scwb_screen_ok,
            "joint_scwb": {key: joint_scwb.get(key) for key in ("evaluated", "all_pass", "counts", "min_ratio_provided",
                                                                  "steel_exhausted", "failed_checks")},
            "capacity_design_accepted": capacity["accepted"], "capacity_design_failed_checks": failing,
            "drift_accepted": drift_screen["accepted"],
            "drift_failed_checks": [c["id"] + (f"@{c['location']}" if c.get("location") else "")
                                    for c in drift_screen["checks"] if c["status"] == "fail"]}


def _smallest_scwb_column_index(ladder):
    """First column rung satisfying strong-column/weak-beam.

    Returns (index, satisfied). When no rung can satisfy the rule the largest
    is returned with satisfied=False, so an exhausted ladder is distinguishable
    from a real match -- previously both came back as a bare index and running
    out of column looked identical to succeeding on the last one.
    """
    required = _scwb_required_column_moment()
    for index, section in enumerate(ladder):
        if _column_nominal_moment(section) >= required:
            return index, True
    return len(ladder) - 1, False


def design_structure(cfg=None, max_section_iter=10, max_steel_iter=6, verbose=True, keep_history=True):
    """Search for feasibility, then compare modeled material quantities within a fixed budget.

    Returns a JSON-serializable record of the final sections, reinforcement,
    governing DCRs, and the ELF demand they were designed against.
    ``keep_history`` False drops the per-iteration entries from the record
    (data generation keeps only the final design, user decision 2026-10-02);
    the search summary, the counts and the selected candidate's evidence stay.
    """
    cfg = cfg or DesignConfig.from_structure_parameters()
    # Category, Ie and the drift criterion come from one basis; a policy that names another
    # category, or a stale ASCE_IE, stops the design before anything is analysed.
    sp.seismic_design_basis(risk_category=cfg.demands.risk_category, site_class=cfg.demands.site_class)
    if (cfg.materials.reinforcement_specification != "ASTM A706 Grade 60"
            or cfg.materials.exposure != "sheltered_interior" or sp.FY_KSI != 60.0
            or cfg.materials.fy_ksi != sp.FY_KSI or cfg.materials.es_ksi != sp.ES_KSI):
        raise ValueError("The current research design scope requires sheltered interior A706 Grade 60 reinforcement.")
    for value in (cfg.rebar.beam_clear_cover_in, cfg.rebar.col_clear_cover_in):
        if isinstance(value, bool) or not math.isfinite(value) or value < 1.5:
            raise ValueError("Frame clear cover must be finite and at least 1.5 in outside hoops.")
    problems = cfg.demands.problems()
    provenance = (cfg.demands.declared_by, cfg.demands.declaration_date, cfg.demands.declaration_basis)
    blank_fields = {"declared_by is blank", "declaration_date is blank", "declaration_basis is blank"}
    if problems and (any(str(v).strip() for v in provenance) or set(problems) - blank_fields):
        raise ValueError("DemandPolicy is not a valid declaration: " + "; ".join(problems))
    aggregate = cfg.rebar.aggregate_max_size_in
    if isinstance(aggregate, bool) or not math.isfinite(aggregate) or aggregate <= 0:
        raise ValueError("Maximum aggregate size must be finite and positive.")
    sp.BEAM_CLEAR_COVER_IN = cfg.rebar.beam_clear_cover_in
    sp.COL_CLEAR_COVER_IN = cfg.rebar.col_clear_cover_in
    sp.AGGREGATE_MAX_SIZE_IN = cfg.rebar.aggregate_max_size_in
    sp.REINFORCEMENT_SPECIFICATION = cfg.materials.reinforcement_specification
    sp.MATERIAL_EXPOSURE = cfg.materials.exposure
    if type(cfg.iteration.uniform_reduction_trials) is not int or cfg.iteration.uniform_reduction_trials < 0:
        raise ValueError("uniform_reduction_trials must be a nonnegative integer")
    target = cfg.dcr.dcr_target
    band_lo, band_hi = cfg.dcr.dcr_band_lo, cfg.dcr.dcr_band_hi

    span = max(sp.BAY_X, sp.BAY_Y)
    columns = column_ladder()
    beams = beam_ladder(span_in=span, story_height_in=sp.STORY_H)

    column_index = nearest_rung_index(columns, sp.B_COL, sp.H_COL, sp.FC_COL_KSI)
    beam_index = nearest_rung_index(beams, sp.B_BEAM, sp.H_BEAM, sp.FC_BEAM_KSI)

    from Design.SMRF_Common import SectionAxialDomainError
    from Design.SMRF_Beam_Slab_Strength import BeamBarRowsError
    history = []
    visited = set()
    visited_order = []
    feasibility_frontier = []
    frontier_priorities = {}
    scheduled = set()
    best = None
    gravity_failures = []
    # A factored axial load outside the installed section's strength domain (steel pass or capacity design):
    # the section cannot carry the load with any cage, so the column steps without spending a design
    # iteration, like a gravity failure (2026-10-02, screening repair item 2). Each is kept with its demand.
    axial_domain_failures = []
    axial_domain_escalations = 0
    # A beam too shallow for its layered cage (a bar row past mid-depth): the beam steps to the next deeper
    # rung without spending a design iteration (r2 screen, case_0002, 2026-10-02). Each is kept.
    beam_layering_failures = []
    # The slab floor is local to a candidate; it is reset for each independent trial.
    slab_floor_in = None
    stop_reason, stop_detail = "iteration_budget_exhausted", None
    # The column step-down pass, opened by the first passing candidate (None until then). Its trials are
    # bounded by an explicit evaluation budget, not by max_section_iter.
    reduction = None

    # Section-domain and layout retries are bounded separately by their finite ladders.
    # Numerical nonconvergence is unresolved evidence, never a section-growth instruction.
    iteration = 0
    candidate_seed = _capture_state()
    initial_transverse = {key: getattr(sp, key) for key in (
        "BEAM_STIRRUP_BAR_SIZE", "BEAM_STIRRUP_LEGS", "BEAM_STIRRUP_SPACING",
        "COL_STIRRUP_BAR_SIZE", "COL_STIRRUP_LEGS", "COL_STIRRUP_LEGS_BY_DIRECTION", "COL_STIRRUP_SPACING")}
    while iteration < max_section_iter or reduction is not None:
        # Every pair starts from the same reinforcement and slab request. No failed trial
        # can ratchet the next trial's cage, hoop diameter or slab thickness upward.
        _restore_state(candidate_seed)
        slab_floor_in = None
        # Hoops are selected per section by the capacity design below; the
        # detailing selector never increases a spacing, so start each rung
        # from the requested values rather than the previous rung's hoops.
        for key, value in initial_transverse.items():
            setattr(sp, key, value)
        _apply_rung(columns[column_index], "column")
        requested_beam = beam_index
        substitutions = []
        beam_index = _compatible_beam_index(beams, beam_index)
        if beam_index != requested_beam:
            substitutions.append({"stage": "clear_span_compatibility", "requested_beam": list(beams[requested_beam]),
                                  "beam": list(beams[beam_index]),
                                  "reason": "ACI 318-19 18.6.2.1(a) ln >= 4d with the installed columns"})
        _apply_rung(beams[beam_index], "beam")
        _sync_cfg_to_sp(cfg)
        validate_rung(columns[column_index], "column")
        validate_rung(beams[beam_index], "beam", span_in=span, story_height_in=sp.STORY_H)

        before_slab_fit = beam_index
        # The transfer depends on the slab and on the beam/column sections, so
        # it is rebuilt whenever the ladder moves (inside the slab search); the
        # restored best state carries its own copy (FLOOR_TRANSFER is in _STATE_KEYS).
        try:
            beam_index, slab, slab_retries, slab_search = _slab_search(cfg, beams, beam_index, slab_floor_in)
        except SlabDesignError as error:
            if reduction is None:
                raise
            # A step-down trial whose slab is refused is a failed trial; the passing design stands.
            column_index = _column_reduction_advance(
                reduction, columns, column_index, False, str(error),
                constraints=getattr(error, "evidence", None),
                outcome="infeasible" if isinstance(error, (SlabLayoutError, SlabThicknessExhausted, SlabSearchExhausted))
                else "analysis_unresolved",
                actual_beam_index=beam_index)
            if column_index is None:
                break
            beam_index = reduction["beam"]
            continue
        fitted = slab_search["first_fitted_beam_index"]
        if fitted != before_slab_fit:
            substitutions.append({"stage": "slab_fitting", "requested_beam": list(beams[before_slab_fit]),
                                  "beam": list(beams[fitted]),
                                  "reason": "no slab thickness fits the requested beam (SMRF_Slab.choose_slab); "
                                            "next deeper/wider compatible rung that one does"})
        if slab_search["beam_rung_steps"]:
            substitutions.append({"stage": "slab_layout", "requested_beam": list(beams[fitted]),
                                  "beam": list(beams[beam_index]),
                                  "reason": "no four-layer slab layout at any supported thickness of the requested beam "
                                            "(slab search); next deeper compatible rung whose slab carries its strip actions"})
        if slab_search["steps"]:
            slab_floor_in = slab["thickness_in"]
        try:
            slab, period, torsion, worst, elf, combination_actions, joint_scwb, capacity = _analyze_bar_layout(
                cfg, slab, slab_search, max_steel_iter)
        except SlabDesignError as error:
            if reduction is None:
                raise
            column_index = _column_reduction_advance(
                reduction, columns, column_index, False, str(error),
                constraints=getattr(error, "evidence", None),
                outcome="infeasible" if isinstance(error, (SlabLayoutError, SlabThicknessExhausted, SlabSearchExhausted))
                else "analysis_unresolved", actual_beam_index=beam_index)
            if column_index is None:
                break
            beam_index = reduction["beam"]
            continue
        except SectionAxialDomainError as error:
            if reduction is not None:
                column_index = _column_reduction_advance(reduction, columns, column_index, False, str(error),
                                                          actual_beam_index=beam_index)
                if column_index is None:
                    break
                beam_index = reduction["beam"]
                continue
            axial_domain_failures.append({
                "escalation": axial_domain_escalations + 1,
                "design_iterations_used": iteration,
                "column_section": list(columns[column_index]),
                "beam_section": list(beams[beam_index]),
                "column_bars": [sp.COL_BAR_SIZE, sp.COL_TOP_BARS, sp.COL_BOT_BARS, sp.COL_SIDE_BARS],
                "error": str(error),
                "basis": "the factored axial load leaves the section's nominal P-M domain (SectionAxialDomainError); "
                         "the same size at the top concrete grade first, then the next size at the top grade "
                         "(strength before geometry, 2026-10-03)",
            })
            step, larger_top = _column_strength_first_steps(columns, column_index)
            if step is None:
                step = larger_top
            if step is None:
                raise RuntimeError(
                    f"{error} No column in the ladder carries this factored axial load with any cage "
                    f"({sp.NUM_FLOOR} stories, {sp.NUM_FLOOR * sp.STORY_H:.0f} in tall); the ladder tops out at "
                    f"{tuple(columns[-1])}."
                ) from error
            column_index = step
            axial_domain_escalations += 1
            if axial_domain_escalations > len(columns):
                raise RuntimeError("Axial-domain escalation did not terminate; the column ladder is inconsistent.") from error
            continue
        except BeamBarRowsError as error:
            if reduction is not None:
                column_index = _column_reduction_advance(reduction, columns, column_index, False, str(error),
                                                          actual_beam_index=beam_index)
                if column_index is None:
                    break
                beam_index = reduction["beam"]
                continue
            beam_layering_failures.append({"escalation": len(beam_layering_failures) + 1, "design_iterations_used": iteration,
                                           "column_section": list(columns[column_index]), "beam_section": list(beams[beam_index]),
                                           "beam_bars": [sp.BEAM_BAR_SIZE, sp.BEAM_TOP_BARS, sp.BEAM_BOT_BARS], "error": str(error),
                                           "basis": "the beam's bar rows reach past mid-depth; next deeper compatible rung"})
            deeper = _next_deeper_beam_index(beams, beam_index)
            if deeper is not None:
                compatible = _compatible_beam_index(beams, deeper)
                deeper = compatible if compatible >= deeper else None
            if deeper is None or len(beam_layering_failures) > len(beams):
                raise RuntimeError(f"{error} No deeper compatible beam rung remains in the ladder.") from error
            beam_index = deeper
            continue
        except RuntimeError as error:
            numerical = ("gravity analysis failed", "capacity hoop selection cycled",
                         "capacity hoop selection exceeded")
            if not any(message in str(error).lower() for message in numerical):
                raise
            if reduction is not None:
                column_index = _column_reduction_advance(
                    reduction, columns, column_index, False, str(error),
                    outcome="analysis_unresolved", actual_beam_index=beam_index)
                if column_index is None:
                    break
                beam_index = reduction["beam"]
                continue
            raise RuntimeError(
                f"Uniform candidate analysis unresolved for column {columns[column_index]} / "
                f"beam {beams[beam_index]}: {error}. No strength failure inferred and no automatic size growth."
            ) from error

        iteration += 1
        # The sizing screen (uniform-member proxy) and the per-joint rule the
        # steel pass closed on; a joint that still fails is a section matter.
        scwb_screen_ok = _column_nominal_moment() >= _scwb_required_column_moment()
        joint_scwb_failed = joint_scwb["evaluated"] and not joint_scwb["all_pass"]
        # The sizing screen is a uniform-member proxy at a tributary axial load. Where every joint was
        # evaluated on its factored axial envelope, that exact check is the requirement and the proxy only
        # sizes the first jump (r2 screen, 2026-10-02: the proxy pushed case_0004's column from 26 in upward
        # while every joint passed). With any joint unevaluated the proxy still has to hold.
        joint_scwb_complete = bool(joint_scwb["evaluated"]) and not (joint_scwb.get("counts") or {}).get("not_evaluated")
        scwb_ok = (not joint_scwb_failed) if joint_scwb_complete else (scwb_screen_ok and not joint_scwb_failed)
        drift_screen = _drift_screen(period, torsion, cfg)
        demand_basis = _demand_basis(cfg, torsion, elf, drift_screen)

        entry = {
            "iteration": iteration,
            "column_section": list(columns[column_index]),
            "beam_section": list(beams[beam_index]),
            "slab": slab,
            "slab_sizing_retries": slab_retries,
            "slab_thickness_search": slab_search,
            "column_dcr": worst["column"],
            "beam_dcr": worst["beam"],
            "beam_flexure_demand_kip_in": worst.get("beam_flexure_demand_kip_in"),
            "model_period_sec": period,
            "base_shear_kip": (elf or {}).get("base_shear_kip"),
            "column_bars": [sp.COL_BAR_SIZE, sp.COL_TOP_BARS, sp.COL_BOT_BARS, sp.COL_SIDE_BARS],
            "beam_bars": [sp.BEAM_BAR_SIZE, sp.BEAM_TOP_BARS, sp.BEAM_BOT_BARS],
            "scwb_satisfied": scwb_ok,
            "scwb_screen_satisfied": scwb_screen_ok,
            "scwb_joint": joint_scwb,
            "capacity_design_accepted": capacity["accepted"],
            "torsional_irregularity": (torsion or {}).get("torsional_irregularity"),
            "joint_shear_satisfied": capacity["joints"]["all_pass"],
            "anchorage_satisfied": capacity["anchorage"]["all_pass"],
            "hoops": {"beam": capacity["transverse"]["beam"], "column": capacity["transverse"]["column"]},
            "drift_screen": drift_screen,
            # Must come off the same P-M capacity as scwb_ok above. Leaving
            # this on sp.column_nominal_moment_y() recorded the old beam-formula
            # ratio beside the new pass/fail flag, so a history row could read
            # "1.171, satisfied" against a reported 2.971 for the same section.
            "scwb_ratio": (
                _column_nominal_moment() / (_scwb_required_column_moment() / sp.SCWB_RATIO_MIN)
                if _scwb_required_column_moment() > 0 else None
            ),
            "beam_at_ladder_floor": beam_index == 0,
            "column_at_ladder_floor": column_index == 0,
            "requested_beam_section": list(beams[requested_beam]),
            "substitutions_before_evaluation": substitutions,
            # The column cage rule (2026-09-27): whether cages that admit the beam bars exist at this
            # column, and whether the installed one does; the planner's lever when threading fails.
            "column_cage_rule": _column_cage_rule_summary(cfg, capacity),
            "quantities": _candidate_quantities(),
        }
        history.append(entry)
        if verbose:
            print(
                "  [design] iter {}: col {:.0f}x{:.0f} fc{:.0f} DCR={:.3f} | "
                "beam {:.0f}x{:.0f} fc{:.0f} DCR={:.3f} | T1={:.3f}s | slab={:.1f}in".format(
                    iteration,
                    columns[column_index][0], columns[column_index][1], columns[column_index][2],
                    worst["column"],
                    beams[beam_index][0], beams[beam_index][1], beams[beam_index][2],
                    worst["beam"],
                    period or 0.0,
                    slab["thickness_in"],
                )
            )

        # Beams are the intended yielding elements, so they carry the DCR
        # band. Columns are capacity-protected by ACI 318-19 18.7.3: forcing
        # them into the same band would require them to be weaker than the
        # beams they are required to out-strength, which is precisely the
        # soft-story behaviour the model exists to study. They get a strength
        # ceiling and the SCWB rule instead.
        deviation = abs(worst["beam"] - target)
        if max(worst["beam"], worst["column"]) > cfg.dcr.dcr_hard_max:
            deviation += 10.0
        if not scwb_ok:
            deviation += 5.0
        if not capacity["accepted"]:
            deviation += 5.0
        if not drift_screen["accepted"]:
            deviation += 20.0
        if best is None or deviation < best["deviation"]:
            best = {"deviation": deviation, "entry": entry, "state": _capture_state(),
                    "combination_actions": combination_actions, "capacity": capacity,
                    "demand_basis": demand_basis}

        # Candidate screening only. The preferred utilization band is not a
        # code requirement, and this legacy SCWB proxy is not joint acceptance.
        accepted = (
            worst["beam"] <= cfg.dcr.dcr_hard_max
            and worst["column"] <= cfg.dcr.dcr_hard_max
            and scwb_ok
            and capacity["accepted"]
            and drift_screen["accepted"]
        )
        entry["constraints"] = _constraint_summary(worst, cfg.dcr.dcr_hard_max, scwb_screen_ok, joint_scwb,
                                                   capacity, drift_screen, accepted)
        if accepted or reduction is not None:
            snapshot = ({"deviation": deviation, "entry": entry, "state": _capture_state(),
                         "combination_actions": combination_actions, "capacity": capacity,
                         "demand_basis": demand_basis} if accepted else None)
            if reduction is None:
                stop_reason, stop_detail = "candidate_screen_passed", "every evaluated requirement met at this rung pair"
                reduction = _column_reduction_start(columns, column_index, beam_index, snapshot, beams,
                                                    cfg.iteration.uniform_reduction_trials)
                next_column = _column_reduction_next(reduction, columns)
                entry["column_reduction"] = {"role": "first passing candidate"}
            else:
                next_column = _column_reduction_advance(reduction, columns, column_index, accepted, None, snapshot,
                                                        entry["constraints"], actual_beam_index=beam_index)
                entry["column_reduction"] = {"role": "step-down trial", "accepted": bool(accepted)}
            if next_column is None:
                break
            column_index, beam_index = next_column, reduction["beam"]
            continue

        key = (column_index, beam_index)
        if key in visited:
            # The planner never proposes an evaluated pair; only a substitution
            # at the top of this iteration (slab fitting) can land on one.
            entry["duplicate_after_substitution"] = True
            feasibility_frontier = [pair for pair in feasibility_frontier if pair not in visited]
            if feasibility_frontier:
                feasibility_frontier.sort(key=lambda pair: frontier_priorities[pair])
                column_index, beam_index = feasibility_frontier.pop(0)
                continue
            stop_reason = "repeated_candidate_after_substitution"
            stop_detail = (f"rung pair column {list(columns[column_index])} / beam {list(beams[beam_index])} was "
                           f"evaluated twice and no queued alternative remains; substitutions: {substitutions}")
            break
        visited.add(key)
        visited_order.append(key)

        scwb_index, _scwb_reachable = _smallest_scwb_column_index(columns)
        if joint_scwb_complete and not joint_scwb_failed:
            scwb_index = min(scwb_index, column_index)       # every joint passes: the proxy asks for no step
        failing_joints = [joint for joint in capacity["joints"]["joints"].values() if joint.get("passes") is False]
        confinement = [(joint.get("classification") or {}).get("confinement") or {} for joint in failing_joints]
        # Every failing joint has two transverse beams that miss 15.2.8 on width alone: beam width is the lever.
        width_only = bool(confinement) and all(
            c.get("two_transverse_beams") and c.get("width_ok") is False and c.get("extends_beyond_joint")
            and c.get("continuous_top_and_bottom_bars") and c.get("stirrups_ok") for c in confinement)
        plan = _plan_next_candidate(
            columns, beams, column_index, beam_index, worst, target, cfg.dcr.dcr_hard_max, scwb_index,
            {"drift_ok": drift_screen["accepted"], "scwb_ok": scwb_ok, "joint_scwb_failed": joint_scwb_failed,
             "capacity_accepted": capacity["accepted"],
             "beam_section_adequate": capacity["beams"]["section_adequate"],
             "beam_hoops_selected": capacity["transverse"]["beam"] is not None,
             "beam_bars_thread": (capacity.get("bar_threading") or {}).get("passes", True),
             "column_cage_can_thread": entry["column_cage_rule"]["threading_cages_available"],
             "column_bar_bond_ok": (capacity.get("column_bar_bond") or {}).get("passes", True),
             "column_section_adequate": capacity["columns"]["section_adequate"],
             "column_hoops_selected": capacity["transverse"]["column"] is not None,
             "joints_all_pass": capacity["joints"]["all_pass"],
             "anchorage_all_pass": capacity["anchorage"]["all_pass"],
             "joint_confinement_width_in": 0.75 * max(sp.B_COL, sp.H_COL) if width_only else None,
             "joint_shear_ratio": max((joint["vj_kip"] / joint["phi_vn_kip"]
                                       for joint in capacity["joints"]["joints"].values()
                                       if joint.get("phi_vn_kip")), default=None)},
            visited, lambda c: _feasible_beam_indices(beams, columns[c][0], columns[c][1]))
        entry["step_reasons"] = plan["reasons"]
        candidate = plan["candidate"]
        entry["proposal"] = {
            "proposed": {"column": list(columns[plan["proposed"]["column"]]), "beam": list(beams[plan["proposed"]["beam"]])},
            "substitutions": plan["substitutions"],
            "candidate": ({"column": list(columns[candidate["column"]]), "beam": list(beams[candidate["beam"]])}
                          if candidate else None),
            "stop_reason": plan["stop_reason"], "stop_detail": plan["stop_detail"]}
        entry["next_rungs"] = entry["proposal"]["candidate"]

        from Design.Uniform_Search import targeted_alternatives, feasibility_priority
        alternatives = targeted_alternatives(columns, beams, column_index, beam_index, plan["reasons"],
                                             entry["constraints"]["capacity_design_failed_checks"])
        proposals = ([(candidate["column"], candidate["beam"], "existing demand-directed proposal")]
                     if candidate is not None else []) + alternatives
        entry["targeted_alternatives"] = []
        priority = feasibility_priority(worst, cfg.dcr.dcr_hard_max, drift_screen["accepted"],
                                        entry["constraints"]["capacity_design_failed_checks"], scwb_ok)
        for c, b, why in proposals:
            if (c, b) in visited or (c, b) in scheduled:
                continue
            if b not in _feasible_beam_indices(beams, columns[c][0], columns[c][1]):
                continue
            scheduled.add((c, b))
            frontier_priorities[(c, b)] = priority
            feasibility_frontier.append((c, b))
            entry["targeted_alternatives"].append({"column": list(columns[c]), "beam": list(beams[b]), "reason": why})
        feasibility_frontier = [pair for pair in feasibility_frontier if pair not in visited]
        if not feasibility_frontier:
            stop_reason, stop_detail = plan["stop_reason"] or "candidate_set_exhausted", plan["stop_detail"]
            break
        feasibility_frontier.sort(key=lambda pair: frontier_priorities[pair])
        column_index, beam_index = feasibility_frontier.pop(0)
        entry["next_rungs"] = {"column": list(columns[column_index]), "beam": list(beams[beam_index])}
    else:
        stop_reason = "iteration_budget_exhausted"
        stop_detail = (f"max_section_iter = {max_section_iter} evaluations used; the last plan still proposed "
                       f"{history[-1].get('next_rungs') if history else None}")

    column_reduction = None
    if reduction is not None:
        from Design.Uniform_Search import summary as reduction_summary
        best = reduction["accepted"]
        column_reduction = reduction_summary(reduction, columns, beams)
        stop_detail = (f"smallest passing column selected after {len(reduction['trials'])} full step-down trial(s) "
                       f"({column_reduction['first_passing_column']} to {column_reduction['selected_column']}); "
                       f"{column_reduction['untested_count']} declared alternatives untested; size first, then the "
                       "lowest passing grade at that size; not a global-minimum claim")
    if best is None:
        raise RuntimeError(
            f"No usable design found in {max_section_iter} section iterations; "
            f"{len(gravity_failures)} unresolved gravity analyses and {len(axial_domain_failures)} trials left the "
            "section's axial domain. Raise max_section_iter or extend the column ladder."
        )

    _restore_state(best["state"])
    _sync_cfg_to_sp(cfg)
    final = best["entry"]
    governing = max(final["column_dcr"], final["beam_dcr"])
    # Load-path check of the selected frame: the bare frame with its transfer
    # against the monolithic coupled model, same slab, sections and loads.
    coupled_comparison = None
    if sp.FLOOR_TRANSFER is not None:
        from Design.SMRF_Coupled_Comparison import compare_transfer_to_coupled
        coupled_comparison = compare_transfer_to_coupled(final["slab"],
                                                         combination_actions=best["combination_actions"])

    # Evaluated against the restored (final) section, so the reported figures
    # describe the design that is actually written out.
    column_mn = _column_nominal_moment()
    # The beam moment one column of the governing joint must match: Mnb- + Mnb+ of the strongest family at a
    # single-column roof joint, half of it at the floor below when the roof connection is excepted (18.7.3.1).
    beam_mn = _scwb_required_column_moment() / sp.SCWB_RATIO_MIN
    roof_excepted = _scwb_roof_exception_applies()
    governing_joint = ("the floor joint below the roof, two columns sharing the beam sum (roof connection excepted by "
                       "ACI 318-19 18.7.3.1)" if roof_excepted else "an interior roof joint with a single column")
    scwb_screen_satisfied = column_mn >= _scwb_required_column_moment()
    joint_scwb = final["scwb_joint"]
    joint_scwb_failed = joint_scwb["evaluated"] and not joint_scwb["all_pass"]
    joint_scwb_complete = bool(joint_scwb["evaluated"]) and not (joint_scwb.get("counts") or {}).get("not_evaluated")
    scwb_satisfied = (not joint_scwb_failed) if joint_scwb_complete else (scwb_screen_satisfied and not joint_scwb_failed)
    drift_screen = final["drift_screen"]
    core = _state_record_core()

    # What the beam rung answers to. In band it is the DCR target; below the
    # band it is whichever requirement moved the beam off the strength rung
    # (recorded in the previous iteration's step reasons) or the ladder floor.
    final_index = next(i for i, item in enumerate(history) if item is final)
    previous_reasons = history[final_index - 1].get("step_reasons", []) if final_index > 0 else []
    if band_lo <= final["beam_dcr"] <= band_hi:
        governed_by = "demand"
    elif final.get("beam_at_ladder_floor") and final["beam_dcr"] < band_lo:
        governed_by = "minimum_section"
    elif "drift" in previous_reasons:
        governed_by = "drift"
    elif "beam_capacity_shear" in previous_reasons:
        governed_by = "capacity_design"
    elif any(reason in previous_reasons for reason in ("joint_scwb", "scwb_screen")):
        governed_by = "scwb"
    else:
        governed_by = "search_limit"

    record = {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "slab": final["slab"],
        "floor_loads": sp.floor_load_metadata(),
        "gravity_load_model": sp.effective_gravity_load_model(),
        "floor_transfer": sp.FLOOR_TRANSFER,
        "slab_reinforcement": sp.SLAB_REINFORCEMENT,
        "slab_actions": sp.SLAB_ACTIONS,
        "geometry": core["geometry"],
        "sections": core["sections"],
        "reinforcement": core["reinforcement"],
        "materials": {**core["materials"],
                      "reinforcement_specification": cfg.materials.reinforcement_specification,
                      "exposure": cfg.materials.exposure},
        "dcr": {
            "column": final["column_dcr"],
            "beam": final["beam_dcr"],
            "governing": governing,
            "band_lo": band_lo,
            "band_hi": band_hi,
            "beam_in_band": band_lo <= final["beam_dcr"] <= band_hi,
            "column_within_ceiling": final["column_dcr"] <= cfg.dcr.dcr_hard_max,
            # SCWB belongs here. The search loop already refused to stop
            # without it, but this published flag left it out, so a design
            # that fell back after exhausting the ladder with columns weaker
            # than their beams was still reported as accepted -- and anything
            # downstream filtering on this flag believed it.
            "candidate_strength_screen_passed": (
                final["beam_dcr"] <= cfg.dcr.dcr_hard_max
                and final["column_dcr"] <= cfg.dcr.dcr_hard_max
                and scwb_satisfied
            ),
            "exceeds_capacity": governing > cfg.dcr.dcr_hard_max,
            "target_basis": ("beam flexure carries the DCR band; columns are capacity-protected; drift, "
                             "capacity shear and the joint rule can hold the beam below the band"),
            "governed_by": governed_by,
        },
        "scwb": {
            "ratio_min": sp.SCWB_RATIO_MIN,
            "column_nominal_moment_kip_in": column_mn,
            "beam_nominal_moment_kip_in": beam_mn,
            "column_axial_kip": _scwb_governing_axial(),
            "column_moment_basis": "preliminary nominal P-M surface at top-corner service gravity; NOT joint qualification",
            "screen_satisfied": scwb_screen_satisfied,
            "joint_check": joint_scwb,
            "beam_moment_basis": (("Mnb- + Mnb+ of the strongest beam family, composite beam-plus-developed-slab (ACI 18.7.3.2 / 6.3.2); "
                                   if (sp.SLAB_REINFORCEMENT or {}).get("layout") else
                                   "Mnb- + Mnb+ of the strongest beam family, rectangular beam; ")
                                  + "per column of the governing joint, which is " + governing_joint),
            "roof_exception_18_7_3_1_screen": roof_excepted,
            "qualification_check": False,
            "ratio_provided": (column_mn / beam_mn if beam_mn > 0 else None),
            "satisfied": scwb_satisfied,
        },
        "seismic": {
            "sds": sp.ASCE_SDS,
            "sd1": sp.ASCE_SD1,
            "s1": sp.ASCE_S1,
            "r": sp.ASCE_R,
            "site_label": getattr(sp, "SEISMIC_SITE_LABEL", None),
            # The risk basis the design was made under (one source: Structure_Parameters).
            "risk_category": sp.ASCE_RISK_CATEGORY,
            "importance_factor": sp.ASCE_IE,
            "design_basis": sp.seismic_design_basis(risk_category=cfg.demands.risk_category),
        },
        "demand": {
            "basis": "Signed gravity/seismic combinations, Ev and 100/30 effects; scope limitations in qualification",
            "model_period_sec": final["model_period_sec"],
            "base_shear_kip": final["base_shear_kip"],
        },
        "iterations": len(history),
        "design_actions": {
            "basis": "Signed simultaneous centerline elastic member actions, compression-positive axial forces; joint-face moment transport is separate",
            "expected_combination_ids": [item["id"] for item in best["combination_actions"]],
            "combinations": best["combination_actions"],
        },
        "drift_screen": drift_screen,
        "capacity_design": best["capacity"],
        "demand_basis": best["demand_basis"],
        "coupled_comparison": coupled_comparison,
        "gravity_failures": gravity_failures,
        "axial_domain_escalations": axial_domain_failures,
        "beam_layering_escalations": beam_layering_failures,
        "slab_thickness_search": final.get("slab_thickness_search"),
        "history": history if keep_history else [],
        "search": {
            "stop_reason": stop_reason,
            "stop_detail": stop_detail,
            "selected_iteration": final["iteration"],
            "last_iteration": history[-1]["iteration"],
            "iterations_evaluated": len(history),
            "history_retained": bool(keep_history),
            "history_note": ("every evaluated iteration is recorded under history" if keep_history else
                             "data generation keeps only the final design: the per-iteration entries are not written; "
                             "a verification run of the same request identity holds them"),
            "max_section_iter": max_section_iter,
            "visited_pairs": [{"column": list(columns[c]), "beam": list(beams[b])} for c, b in visited_order],
            "domain": {"column_rungs": len(columns), "beam_rungs": len(beams),
                       "feasible_beam_rungs_under_selected_columns": len(_feasible_beam_indices(beams, sp.B_COL, sp.H_COL)),
                       "strategy": ("bounded frontier of demand-directed proposals and independent joint/cage alternatives; "
                                    "each pair starts from the same requested reinforcement/slab state and is fully evaluated")},
            "column_reduction": column_reduction,
            "column_size_exception": _column_size_exception(history, columns),
            "untested_feasibility_proposals": [{"column": list(columns[c]), "beam": list(beams[b])}
                                                for c, b in feasibility_frontier],
            "feasibility_trials": [{key: item.get(key) for key in (
                "iteration", "column_section", "beam_section", "constraints", "quantities", "targeted_alternatives")}
                for item in history if not item.get("column_reduction") or
                item["column_reduction"].get("role") == "first passing candidate"],
            "selected_constraints": final.get("constraints"),
            "last_constraints": history[-1].get("constraints"),
            "selection_objective": ("feasibility first; after the first passing candidate, modeled material dominance "
                                    "without increasing concrete grades or form count; feasible tradeoffs retained. "
                                    "DCR penalties rank diagnostic fallback only when no candidate passes"),
            "interpretation": ("the stop reason describes how this bounded search ended; none of them is evidence "
                               "that no code-compliant frame exists for the geometry"),
        },
        "detailing": {**_transverse_geometry(cfg), "joint_continuity": _joint_continuity_declaration(),
                      # The column cage rule's state at the selected candidate (design basis 2026-09-27).
                      "column_cage_rule": best["entry"].get("column_cage_rule"),
                      # ACI 318-19 15.5: column f'c above 1.4 times the floor's (declared placement, 2026-10-03).
                      "column_concrete_through_floor": ("column-strength concrete is placed in the floor at the column "
                                                        "(ACI 318-19 15.5(a)); declared, not checked on a pour drawing")},
    }
    # The selected frame's forces are now copied into the artifact. Release
    # that scratch domain before entering the independent floor diagnostic.
    ops.wipe()
    from Design.SMRF_Design_Evidence import attach_design_evidence, analysis_input_signature
    record["design_actions"]["analysis_input_sha256"] = analysis_input_signature(record)
    record["drift_screen"]["analysis_input_sha256"] = analysis_input_signature(record)
    if record.get("coupled_comparison"):
        record["coupled_comparison"]["analysis_input_sha256"] = analysis_input_signature(record)
    attach_design_evidence(record, cfg)
    from Design.SMRF_Qualification import qualify_design
    record["qualification"] = qualify_design(record)
    record["dcr"]["accepted"] = record["qualification"]["accepted"]
    return record


def apply_design(record):
    """Apply a stored design artifact to Structure_Parameters."""
    sections = record["sections"]
    rebar = record["reinforcement"]
    required = ("col_stirrup_bar_size", "col_stirrup_legs", "col_stirrup_legs_by_direction", "col_stirrup_spacing_in",
                "beam_stirrup_bar_size", "beam_stirrup_legs", "beam_stirrup_spacing_in",
                "beam_side_bars", "legacy_centroid_offset_in", "beam_clear_cover_in",
                "col_clear_cover_in", "beam_longitudinal_centroid_offset_in",
                "col_longitudinal_centroid_offset_in")
    missing = [key for key in required if key not in rebar]
    if missing:
        raise ValueError(f"Cached design lacks {missing}; cannot reproduce its reinforcement.")
    # Validate the whole state before the first assignment. A missing field
    # halfway through restoration must not leave a partially changed model.
    for key in ("b_col_in", "h_col_in", "fc_col_ksi", "b_beam_in", "h_beam_in", "fc_beam_ksi"):
        value = sections.get(key)
        if (not isinstance(value, (float, int)) or isinstance(value, bool)
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"Cached section {key} must be finite and positive.")
    if any(record.get("geometry", {}).get(key) != value for key, value in _slab_geometry().items()):
        raise ValueError("Cached design geometry disagrees with the current model.")
    for prefix in ("beam", "col"):
        for suffix, lower in (("top_bars", 2), ("bot_bars", 2), ("side_bars", 0), ("stirrup_legs", 2)):
            value = rebar.get(f"{prefix}_{suffix}")
            if type(value) is not int or value < lower:
                raise ValueError(f"Cached {prefix}_{suffix} is not a valid bar count.")
        spacing = rebar.get(f"{prefix}_stirrup_spacing_in")
        if (not isinstance(spacing, (float, int)) or isinstance(spacing, bool)
                or not math.isfinite(spacing) or spacing <= 0):
            raise ValueError("Cached transverse spacing must be finite and positive.")
    legs_by_direction = rebar.get("col_stirrup_legs_by_direction")
    if (not isinstance(legs_by_direction, dict) or set(legs_by_direction) != {"across_b_face", "across_h_face"}
            or any(type(v) is not int or v < 2 for v in legs_by_direction.values())
            or min(legs_by_direction.values()) != rebar.get("col_stirrup_legs")):
        raise ValueError("Cached col_stirrup_legs_by_direction must give both directions, at least 2 legs each, "
                         "with col_stirrup_legs the lighter of them.")
    legacy = rebar["legacy_centroid_offset_in"]
    if (not isinstance(legacy, (float, int)) or isinstance(legacy, bool)
            or not math.isfinite(legacy) or legacy <= 0):
        raise ValueError("Cached legacy cover must be finite and positive.")
    from Design.SMRF_Qualification import reinforcement_consistency_checks
    consistency = reinforcement_consistency_checks(record)
    if any(item["status"] != "pass" for item in consistency):
        raise ValueError("Cached member cover/bar positions or material metadata are inconsistent.")
    material = record.get("materials", {})
    if material.get("fy_ksi") != sp.FY_KSI or material.get("es_ksi") != sp.ES_KSI:
        raise ValueError("Cached steel material properties disagree with the current model.")
    slab = _validate_cached_slab(record)
    aggregate = record.get("materials", {}).get("aggregate_size_in")
    if aggregate is None or not math.isfinite(aggregate) or aggregate <= 0:
        raise ValueError("Cached design lacks a valid aggregate size.")
    for prefix in ("beam", "col"):
        clear = rebar[f"{prefix}_clear_cover_in"]
        expected = clear + sp.rebar_diameter(rebar[f"{prefix}_stirrup_bar_size"]) + sp.rebar_diameter(rebar[f"{prefix}_bar_size"]) / 2
        if (not math.isfinite(clear) or clear < 1.5
                or not math.isclose(expected, rebar[f"{prefix}_longitudinal_centroid_offset_in"], abs_tol=1e-12)):
            raise ValueError("Cached member cover/bar positions are inconsistent.")
    sp.B_COL = sections["b_col_in"]
    sp.H_COL = sections["h_col_in"]
    sp.FC_COL_KSI = sections["fc_col_ksi"]
    sp.B_BEAM = sections["b_beam_in"]
    sp.H_BEAM = sections["h_beam_in"]
    sp.FC_BEAM_KSI = sections["fc_beam_ksi"]
    sp.COL_BAR_SIZE = rebar["col_bar_size"]
    sp.COL_TOP_BARS = rebar["col_top_bars"]
    sp.COL_BOT_BARS = rebar["col_bot_bars"]
    sp.COL_SIDE_BARS = rebar["col_side_bars"]
    sp.COL_BAR_AREA = sp.rebar_area(sp.COL_BAR_SIZE)
    sp.BEAM_BAR_SIZE = rebar["beam_bar_size"]
    sp.BEAM_TOP_BARS = rebar["beam_top_bars"]
    sp.BEAM_BOT_BARS = rebar["beam_bot_bars"]
    sp.BEAM_BAR_AREA = sp.rebar_area(sp.BEAM_BAR_SIZE)
    for key, name in (("col_stirrup_bar_size", "COL_STIRRUP_BAR_SIZE"),
                      ("col_stirrup_legs", "COL_STIRRUP_LEGS"),
                      ("col_stirrup_spacing_in", "COL_STIRRUP_SPACING"),
                      ("beam_stirrup_bar_size", "BEAM_STIRRUP_BAR_SIZE"),
                      ("beam_stirrup_legs", "BEAM_STIRRUP_LEGS"),
                      ("beam_stirrup_spacing_in", "BEAM_STIRRUP_SPACING"),
                      ("beam_side_bars", "BEAM_SIDE_BARS"),
                      ("legacy_centroid_offset_in", "COVER"),
                      ("beam_clear_cover_in", "BEAM_CLEAR_COVER_IN"),
                      ("col_clear_cover_in", "COL_CLEAR_COVER_IN")):
        if key not in rebar:
            raise ValueError(f"Cached design lacks {key}; cannot reproduce its reinforcement.")
        setattr(sp, name, rebar[key])
    sp.COL_STIRRUP_LEGS_BY_DIRECTION = dict(rebar["col_stirrup_legs_by_direction"])
    _apply_slab(slab)
    sp.AGGREGATE_MAX_SIZE_IN = aggregate
    sp.REINFORCEMENT_SPECIFICATION = material["reinforcement_specification"]
    sp.MATERIAL_EXPOSURE = material["exposure"]
    transfer = record.get("floor_transfer")
    if record.get("gravity_load_model") == "slab_transfer":
        from Design.SMRF_Floor_Transfer import validate_floor_transfer
        area = sp.BAY_X * sp.NUM_BAY_X * sp.BAY_Y * sp.NUM_BAY_Y / 144.0
        validate_floor_transfer(
            transfer, _slab_geometry(),
            {"b_beam_in": sp.B_BEAM, "h_beam_in": sp.H_BEAM, "fc_beam_ksi": sp.FC_BEAM_KSI,
             "b_col_in": sp.B_COL, "h_col_in": sp.H_COL},
            sp.SLAB_THICKNESS_IN, sp.floor_dead_load_ksf() * area, sp.FLOOR_LIVE_LOAD_KSF * area)
        sp.FLOOR_TRANSFER = transfer
    else:
        if transfer is not None:
            raise ValueError("Cached design carries a floor transfer but does not declare the slab_transfer load model.")
        sp.FLOOR_TRANSFER = None
    strength = record.get("slab_reinforcement")
    if strength is not None and strength.get("inputs"):
        from Design.SMRF_Slab_Reinforcement import evaluate_slab_reinforcement
        audit = evaluate_slab_reinforcement(strength)
        if audit[0]["status"] != "pass":
            raise ValueError("Cached slab reinforcement does not reproduce.")
        if strength["inputs"]["slab"]["thickness_in"] != sp.SLAB_THICKNESS_IN:
            raise ValueError("Cached slab reinforcement belongs to a different slab thickness.")
    sp.SLAB_REINFORCEMENT = strength
    # The saved joint-shear categories (gamma, Aj, Vn per level/kind/axis) feed the joint springs
    # of the nonlinear frame (Model/Joint_Springs); a record without them gets the framing fallback.
    sp.JOINT_SHEAR_CATEGORIES = ((record.get("capacity_design") or {}).get("joints") or {}).get("joints")
    # Orthogonal beam bar stacking: a record from before the rule keeps the single elevation it was
    # designed with; a record that carries the rule must reproduce its offsets under the live slab.
    stacking = rebar.get("beam_bar_stacking")
    sp.BEAM_BAR_STACKING = stacking["mode"] if stacking else "none"
    if stacking:
        # The layer limit and layer order the record was designed under (one layer, blocked, before
        # the same day's later decisions).
        sp.BEAM_BAR_MAX_LAYERS = int(stacking.get("max_layers", 1))
        sp.BEAM_BAR_LAYER_ORDER = stacking.get("layer_order", "blocked")
        expected = sp.beam_bar_stacking_offsets_in()
        recorded = stacking.get("offsets_in") or {}
        if (stacking.get("convention") != sp.beam_bar_stacking_convention()
                or any(not math.isclose(expected[axis][face], float((recorded.get(axis) or {}).get(face, float("nan"))),
                                        abs_tol=1e-9)
                       for axis in ("x", "y") for face in ("top", "bottom"))):
            raise ValueError("Cached beam bar stacking does not reproduce.")
        # The actual rows (2026-09-27): a record that carries them must reproduce them, row by row,
        # and they must be well formed (Design.SMRF_Beam_Slab_Strength.validate_bar_rows).
        recorded_rows = stacking.get("layers")
        if recorded_rows is not None:
            from Design.SMRF_Beam_Slab_Strength import validate_bar_rows
            live_rows = sp.beam_bar_layers()
            for axis in ("x", "y"):
                for face, count in (("top", sp.BEAM_TOP_BARS), ("bottom", sp.BEAM_BOT_BARS)):
                    saved = validate_bar_rows(recorded_rows[axis][face], count, f"{axis} {face} bars", sp.H_BEAM)
                    live = [(n, o) for n, o in zip(live_rows[axis][face]["per_layer"], live_rows[axis][face]["offsets_in"])]
                    if len(saved) != len(live) or any(n != m or not math.isclose(a, b, abs_tol=1e-9)
                                                     for (n, a), (m, b) in zip(saved, live)):
                        raise ValueError("Cached beam bar rows do not reproduce.")
    return record


def source_sha256():
    """{relative path: SHA-256} of every source file a design depends on (independent of the installed design)."""
    source_root = Path(__file__).resolve().parents[1]
    files = [source_root / name for name in ("Structure_Parameters.py", "RC_Design_Check.py", "Redesign.py")]
    for folder in ("Design", "Model", "Loads", "Analysis"):
        files.extend((source_root / folder).rglob("*.py"))
    return {path.relative_to(source_root).as_posix(): hashlib.sha256(
        path.read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest()
        for path in sorted(files)}


def design_request_identity(cfg=None):
    """Portable identity of design inputs and source, excluding GM/intensity."""
    cfg = cfg or DesignConfig.from_structure_parameters()
    # An identity is refused, not built, for a contradictory risk basis (policy category vs the
    # configured category vs ASCE_IE): there is no consistent request to identify.
    sp.seismic_design_basis(risk_category=cfg.demands.risk_category)
    keys = ("NUM_BAY_X", "NUM_BAY_Y", "NUM_FLOOR", "BAY_X", "BAY_Y", "STORY_H",
            "FLOOR_DEAD_LOAD_KSF", "FLOOR_LIVE_LOAD_KSF", "CONCRETE_UNIT_WEIGHT_KCF",
            "GRAVITY_LOAD_MODEL", "ASCE_SDS", "ASCE_SD1", "ASCE_S1", "ASCE_R", "ASCE_RISK_CATEGORY", "ASCE_IE",
            "ASCE_CU", "ASCE_TL", "FY_KSI", "ES_KSI", "COVER")
    inputs = {key: getattr(sp, key) for key in keys}
    policy = asdict(cfg)
    # The nonlinear-model profile the design is made for (Model/Analysis_Profile): part of the
    # policy so the methodology hash and the cache check both refuse a design made under another
    # profile, even though the elastic design model itself does not read these settings.
    from Model.Analysis_Profile import profile_identity
    policy["model_profile"] = profile_identity()
    # Selected sections/reinforcement are outputs, not request parameters.
    policy.pop("sections", None)
    for key in ("fc_col_ksi", "fc_beam_ksi"):
        policy["materials"].pop(key, None)
    for key in list(policy.get("rebar", {})):
        if key.startswith("stirrup_spacing_") and key not in ("stirrup_spacing_min_in", "stirrup_spacing_step_in"):
            policy["rebar"].pop(key)
    hashes = source_sha256()
    payload = {"schema": DESIGN_SCHEMA_VERSION, "inputs": inputs, "policy": policy,
               "source_sha256": hashes}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(), **json.loads(canonical)}


def load_or_create_design(design_path, cfg=None, verbose=True, max_section_iter=None, keep_history=True):
    """Read a cached design artifact, or run the design and write it.

    ``max_section_iter`` overrides the search's section-iteration budget (the driver's default when
    None); a run that asks for a larger bound records it in its plan (Design/Verify_Designs).
    ``keep_history`` False (data generation) writes the record without its per-iteration entries.

    Returns (record, created). The artifact is written atomically so a
    concurrent generation worker never reads a half-written design.
    """
    design_path = Path(design_path)
    identity = design_request_identity(cfg)
    if design_path.exists():
        record = json.loads(design_path.read_text(encoding="utf-8"))
        if record.get("schema_version") != DESIGN_SCHEMA_VERSION or record.get("request_identity", {}).get("sha256") != identity["sha256"]:
            raise RuntimeError("Existing design uses different inputs or methodology. Preserve it and choose a new output root.")
        apply_design(record)
        return record, False

    design_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = design_path.with_name(f".{design_path.name}.lock")
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"Design is already reserved, or an interrupted lock needs review: {lock_path}") from exc
    temporary = design_path.with_name(f".{design_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with lock:
            lock.write(json.dumps({"pid": os.getpid(), "request_sha256": identity["sha256"]}))
        # Another writer may have completed between the initial read and lock.
        if design_path.exists():
            record = json.loads(design_path.read_text(encoding="utf-8"))
            if record.get("request_identity", {}).get("sha256") != identity["sha256"] or record.get("schema_version") != DESIGN_SCHEMA_VERSION:
                raise RuntimeError("Another writer created a different design; existing artifact preserved.")
            apply_design(record)
            return record, False
        record = design_structure(cfg=cfg, verbose=verbose, keep_history=keep_history,
                                  **({} if max_section_iter is None else {"max_section_iter": int(max_section_iter)}))
        record["request_identity"] = identity
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # No-clobber rename on Windows; writers in this workflow share the lock.
        if design_path.exists():
            raise RuntimeError("Design destination appeared during analysis; existing artifact preserved.")
        temporary.rename(design_path)
        return record, True
    finally:
        if temporary.exists():
            temporary.unlink()
        lock_path.unlink()
