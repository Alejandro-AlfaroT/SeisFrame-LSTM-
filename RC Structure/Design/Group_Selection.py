"""Reinforcement selection for a grouped design: one cage per group, on the group's own demands (2026-10-02).

The uniform design picks one column cage from the building's worst column and one beam cage from the
largest beam moments. Here every group selects its own cage from the same candidate domain
(Design.Config.RebarConfig: bar sizes, counts, steel ratios) and the same constructibility rules
(Redesign._col_candidates and _beam_candidates), with each rule evaluated on the members that actually
adjoin:

  beams     candidates are priced on the group's own sagging and hogging envelopes, at the bar depth the
            orthogonal layers give them where they cross the other beams (Model.Member_Bar_Layers), and
            must pass the lanes of every column cage they thread and 20 db through every joint core
            they pass (18.8.2.3);
  columns   the groups of one plan location form a chain over the height. Each band's candidates are
            priced on its members' concurrent P, My, Mz in every combination; adjacent bands must form a
            supported transition (Design.SMRF_Transitions); every joint of the chain must satisfy
            18.7.3.2 with the two columns actually framing in (one at the roof, where the 18.7.3.1
            exception applies as the evaluator decides it), each at its own factored axial range. A
            column whose axial range leaves a cage's P-M surface has no
            strength there: that cage is not a candidate for it (no clamp, no exception). The least
            cost chain is found exactly by a pass over the bands.

Objective: Design.Config.DCRTargets.objective, as in the uniform pick. ``min_deviation``: the cage whose
governing DCR is closest to the target (within the convergence tolerance), then the least steel;
``min_volume``: the least steel with DCR within the ceiling. The target is a research preference: a
column is never refused for being below it. The strong-column estimate used during selection is the
nominal P-M sweep at the ends of each column's axial range; the exact joint evaluation of the capacity
design is the authority, and ``scwb_margin`` lets the caller ask for more when that evaluation fails.

Beam cage evidence and joint-aware coordination (2026-10-31, the lower-perimeter cage audit):

  pricing   a tied cage is priced in each group's place on that group's own members: each member context's own
            sagging and hogging envelope against that context's own strength, the pairing of
            Group_Checks.beam_group_strength. The union of the tie's demands against the weakest strength is kept as
            ``dcr_union_bound``: a sufficient flexural screen when it is at most 1, never a member DCR.
  refusals  every refused cage names its first reason by code and class (BEAM_REFUSALS): ``hard_section`` (a code
            limit of the section), ``construction_policy`` (a declared detailing or proportioning rule of this basis)
            or ``missing_evidence`` (no rows to price). Later checks are not reached after the first refusal.
  threading ``lane_limited`` keeps its meaning (a cage that reaches the demand exists but does not thread). The
            separate ``threading`` block says whether threading constrains the strength at all (a stronger
            admissible cage does not thread) and whether a flexural deficit remains, with the best threading and
            nonthreading cages, their DCRs, lanes, the cores that block them, the installed rows with the joints that
            set them, and a geometry-only look-ahead of the legal larger column sizes. A refused or nonthreading cage
            is never made admissible to produce it; the lanes rest on the detailing policy
            (SMRF_Cage_Geometry.passage_policy), which nothing here relaxes.
  repair    a threading-constrained deficit is offered to coordinate_threading: a stronger beam cage is installed
            prospectively, the column lines it threads are selected again requiring that it thread them (strength,
            axial domain, strong-column estimate, bar bond, cage rules and transitions unchanged), and the whole
            selection is run again from there. The changed set is kept only when the beam is stronger, the selection
            settles and no group is newly exhausted; repeated states and the work limits stop it, and every outcome is
            logged. Kept or not, nothing is accepted here: the capacity design, joints and code check follow.
  coupling  the bands directly above and below the blocking cores' band in the same column lines meet it at the
            band-boundary joints (strong-column sums and transitions). Where their cages thread their own beams today,
            a trial reserves them too, so the coordinated chain may not close their lanes to open the blocked band's
            (2026-10-31: on o0013 the uncoupled trial gave the s01_03 beams lanes by taking them from s04_06, and the two
            bands then alternated). The chain report counts, per band boundary, the pairs cut by a transition and by the
            strong-column rule, so a failed trial says which.
  lanes at  where the column above has another cage whose declared transition moves bars inside the joint, the beam
  a boundary bars there pass the joint's bar inventory (Model.Member_Bar_Layers.bars_per_layer_at_joint, 2026-10-31):
            the rows and the evidence read it, and the column chain checks it on every cage pair at a band boundary
            (the joint inventory of that pair), as a requirement where the band's pool is one of cages its beam bars
            thread and as the counted cut ``beam_lanes`` of the chain report. The evidence keeps a core cage's own
            lanes apart from a transition joint's and names such a joint by both column groups (the core below and the
            column above): its look-ahead resizes the column above, the core below or both at that joint's own
            inventory (transition_lookahead), and a core cage's look-ahead is labelled core-only, never a resolution of a
            transition joint (2026-10-31 review).
  installed the same lanes, blocking and look-ahead for every beam group whose INSTALLED cage does not thread a joint
  lanes     it passes (installed_lane_evidence, the record's constraints.beam_lane_evidence), whatever the selector's
            status: a failed beam.bars_thread_column row carries its own attribution without a flexural shortfall or a
            strongest-threading-cage record (2026-10-31 follow-up review).

Terminal cage arrangement (2026-11-01, the four bar-row rejections of the 15-minute search run):

  audit     a column cage chosen in a pass changes the lanes of the beams that thread it, and with them the layers of
            those beams and of every beam they cross: a pass that ends on a column change hands on beam rows that no beam
            selection priced. Every state a pass reaches (after its beams, after its columns) is audited on the complete
            shared arrangement (Model.Member_Bar_Layers.row_audit, report["geometry"]); the last one whose rows fit is kept
            with its designs, where it arose and the column-unit reports of its column cages
            (report["last_geometrically_valid"]).
  closure   a selection whose terminal arrangement does not fit is closed by a bounded, deterministic sequence
            (close_terminal_arrangement, TERMINAL_RULES_ID): (1) every beam unit selected again on the final columns,
            each priced in place on the final shared arrangement; (2) when that leaves rows that do not fit, the column
            lines whose cages set them (the core and the column above at the joints where the rows were raised, the
            cores whose lanes split the bars of the beam and of the beams it crosses) and whose longitudinal cage changed
            since the last geometrically valid state get those longitudinal cages back as whole lines (their installed
            hoops are kept: a hoop the capacity design required is never reverted), then every beam unit selected again
            on them. A route is adopted only when its beam units stopped changing and the complete arrangement it ends
            on fits; its closure row carries the column-unit reports in force. Nothing is relaxed: the beam
            pricing (strength, rows, practical ratio, through depth, tension control, balance, threading preference) and
            the column cages (strength, axial domain, bar bond, transitions, as selected) are the selector's own; the
            column selection's strong-column estimate on the final beams is recorded beside an adopted route
            (column_rules_on_final_state), the exact joint evaluation of the capacity design remains the authority, and
            the selection stays unsettled (so no code-design acceptance follows from a closure alone).
  incomplete when no route closes it, CageArrangementError carries the structured outcome: its classification (a beam
            unit with no fitting cage, beam units that keep alternating, a repeated state, rows still invalid), the
            rejected terminal state and the last geometrically valid state kept apart (neither carries a strength,
            capacity or acceptance result), the invalid rows with what set them, the dependencies, every route's log,
            the pass history and what replays the call without a solve. It is an incomplete coordination outcome within
            these bounds, not proof that no cage of the declared domain fits.

Verification of an installed state (2026-11-01 closure review, VERIFICATION_ID): verify_installed_selection checks the
installed cages against the selection's own rules without selecting anything: the complete arrangement fits, every beam
unit's installed cage is admissible in every group's place, threads and is within dcr_hard_max (select_beam_groups with
``installed_only``), and every column line's installed chain meets the column rules on the installed beams
(select_column_chain with ``installed_only``). The reinforcement passes use it on the states a proven cycle alternates
between (Design.Grouped_Design.resolve_cycle); verified says the rules admit the state, never that the selection
converged to it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import types
from dataclasses import asdict, is_dataclass, replace

import Structure_Parameters as sp
from Design import Group_Checks as checks
from Design.SMRF_Beam_Slab_Strength import BeamBarRowsError, CageArrangementError
from Design.SMRF_Cage_Geometry import passage_policy
from Design.SMRF_Joints import discontinuous_column_exempt
from Design.SMRF_Transitions import column_transition
from Model import Member_Bar_Layers as bar_layers
from Model import Member_Groups as mg
from Model import Member_Properties as mp
from Model.IMK_Calibration import column_moment_at_axial

RULES_ID = "group_reinforcement_selection_v2_roof_exception"
# Under the system policy (Model.Member_Groups.SYSTEM_POLICY_ID, 2026-10-16) the groups of a design tie select one
# cage together, and the strong-column rule applies only in the directions in which a column is part of a
# perimeter frame (ACI 318-19 18.7.3.2 is a special-moment-frame rule; an interior column and the gravity
# direction of an edge column are members not designated as part of the SFRS, 18.14).
SYSTEM_RULES_ID = "system_reinforcement_selection_v2_roof_exception"
BEAM_UNIT_SWEEPS = 3                      # beam units re-selected within a pass until none changes (crossing beams share joints)
COLUMN_SUPPORTED_BAR_HX_MAX_IN = 14.0
# Beam cage refusals by code: (class, the first rule the cage fails). The classes keep the code limits of a section apart
# from the declared detailing and proportioning rules of this basis and from evidence that could not be established.
BEAM_REFUSALS = {
    "through_joint_depth": ("hard_section", "18.8.2.3: the joint depth parallel to a through bar is below 20 db (26 db, Grade 80)"),
    "bar_layers_unsettled": ("missing_evidence", "the straight-bar layering did not settle: there are no rows to price"),
    "rows_invalid": ("missing_evidence", "the layering returned rows that are not a valid arrangement"),
    "width_fit": ("hard_section", "25.2.1: without lanes the bars stay in one row, which does not keep the clear spacing"),
    "row_past_mid_depth": ("construction_policy", "a straight bar row reaches past mid-depth beside the crossing beams"),
    "tension_control": ("hard_section", "9.3.3.1: the composite section is not tension controlled"),
    "steel_ratio_code": ("hard_section", "18.6.3.1: a face ratio above the code limit"),
    "steel_ratio_practical": ("construction_policy", "a face ratio above the practical beam ceiling "
                                                     "(RebarConfig.rho_beam_practical_max, 2026-10-17)"),
    "minimum_steel": ("hard_section", "9.6.1.2: below the minimum flexural steel"),
    "strength_balance": ("hard_section", "18.6.3.2: end reversal or along-span strength balance"),
}
# An admissible cage that does not thread the column cages, by cause. Both are construction policy: the bar grid and the
# passage clearance give the lanes, BEAM_BAR_MAX_LAYERS caps the layers (SMRF_Cage_Geometry.passage_policy).
THREADING_BLOCKS = {"no_lane": "no bar of this size passes between the column bars on the bar grid",
                    "layer_cap": "the lanes would need more layers than Structure_Parameters.BEAM_BAR_MAX_LAYERS"}
THREADING_EVIDENCE_ID = "beam_threading_evidence_v2"   # v2: core-cage and transition-joint blockage and look-ahead apart
THREADING_PROSPECTS_MAX = 6               # stronger nonthreading cages carried as repair prospects, per beam unit
INSTALLED_LANES_ID = "installed_cage_lane_evidence_v1"
# The joint-aware repair of a threading-constrained beam deficit (coordinate_threading) and its work limits.
COORDINATION_ID = "joint_aware_cage_coordination_v1"
COORDINATION_MAX_UNITS = 2                # threading-constrained beam units coordinated per selection
COORDINATION_MAX_PROSPECTS = 3            # prospective stronger beam cages tried per unit
COORDINATION_MAX_TRIALS = 4               # trial selections per call (each a full set of selection passes)
COORDINATION_MEMO_MAX = 16                # remembered failed coordinations (the same inputs, by content, are not retried)
_COORDINATION_MEMO = {}


def _trial_state(state, gid, design):
    """A trial with one group's design changed; under the system policy the whole design tie moves with it."""
    tie = state.tie_of(gid)
    return state.with_tied_designs({tie: design}) if tie else state.with_designs({gid: design})


def _scwb_axes(state, member):
    """The directions in which 18.7.3.2 applies at this column's joints: both under the band policy (every frame is
    part of the SFRS); under the system policy the perimeter-frame directions of the column
    (Model.Member_Groups.column_sfrs_directions), none for an interior column."""
    if not mg.is_system_policy(state.policy_id):
        return ("x", "y")
    directions = mg.column_sfrs_directions(member.grid_i, member.grid_j, sp.NUM_BAY_X, sp.NUM_BAY_Y,
                                           layout=mg.layout_of(state.policy_id))
    return tuple(axis for axis in ("x", "y") if axis in directions)


def _beam_units(state):
    """[(label, group ids)] selected together: each beam group alone under the band policy, each beam tie under
    the system policy."""
    ties = state.ties()
    if ties:
        return [(tie, tuple(gids)) for tie, gids in sorted(ties.items()) if mg.tie_spec(tie)["member_type"] == "beam"]
    return [(gid, (gid,)) for gid in sorted(g for g, group in state.groups.items() if group["member_type"] != "column")]


def _column_units(state):
    """[(label, plan locations)] whose column groups form one chain: each location under the band policy, the
    locations of each column tie under the system policy."""
    if state.ties():
        return [(tie, tuple(spec["locations"])) for tie, spec in mg.SYSTEM_DESIGN_TIES.items()
                if spec["member_type"] == "column" and any(mg.tie_kind(t) == tie for t in state.ties())]
    return [(location, (location,)) for location in mg.COLUMN_LOCATIONS]


def _objective(dcr, steel, cfg):
    if cfg.dcr.objective == "min_volume":
        return (0.0, steel)
    tol = max(cfg.iteration.convergence_tol, 1e-12)
    return (round(abs(dcr - cfg.dcr.dcr_target) / tol) * tol, steel)


# ---- candidate cages --------------------------------------------------------------------------------------
def column_cage_candidates(design, cfg, rho_max=None, layout=None):
    """Column cages of the declared domain that one section can hold, as MemberDesigns (hoops unchanged).

    The static rules of Redesign._col_candidates on the slab-aware path: ACI ratio limits, clear spacing
    on both faces, supported bars within 14 in (18.7.5.2(f)), no more face bars than the hoop ladder can
    tie, and a layout the largest hoop can confine at the minimum spacing. ``rho_max`` defaults to the
    practical ceiling the uniform strong-column escalation uses (RebarConfig.rho_col_practical_max), lifted to
    the ACI 318-19 18.7.4.1 limit (RebarConfig.rho_col_max) for a column at its layout's size cap: at the cap the
    search exhausts the reinforcement before the layout ladder adds frames (user direction 2026-10-18).
    """
    from Design.SMRF_Capacity_Design import column_layout_confinable
    from Design.Section_Design import column_size_cap_in
    from Redesign import COLUMN_MAX_TIED_FACE_BARS
    b, h, ag = design.b_in, design.h_in, design.b_in * design.h_in
    low = max(0.01, cfg.rebar.rho_col_min)
    if rho_max is None and max(b, h) >= column_size_cap_in(layout) - 1e-9:
        rho_max = cfg.rebar.rho_col_max                                   # the last lever at the cap
    high = min(0.06, cfg.rebar.rho_col_max, cfg.rebar.rho_col_practical_max if rho_max is None else rho_max)
    result = []
    for bar_size in cfg.rebar.bar_sizes_col:
        ab, db = sp.rebar_area(bar_size), sp.rebar_diameter(bar_size)
        clear_min = sp.longitudinal_clear_spacing_in("column", bar_size)
        cover = sp.longitudinal_cover_in("column", bar_size, design.stirrup_bar_size)
        for n_top in cfg.rebar.col_n_top_iter():
            for n_side in cfg.rebar.col_n_side_options:
                if n_top < 2 or n_side < 0:
                    continue
                rho = (2 * n_top + 2 * n_side) * ab / ag
                if not low <= rho <= high + 1e-12:
                    continue
                pitch_b, pitch_h = (b - 2 * cover) / (n_top - 1), (h - 2 * cover) / (n_side + 1)
                if pitch_b - db < clear_min or pitch_h - db < clear_min:
                    continue
                if pitch_b > COLUMN_SUPPORTED_BAR_HX_MAX_IN or pitch_h > COLUMN_SUPPORTED_BAR_HX_MAX_IN:
                    continue
                if max(n_top, n_side + 2) > COLUMN_MAX_TIED_FACE_BARS:
                    continue
                if not column_layout_confinable(b, h, design.fc_ksi, sp.FYT_KSI, sp.COL_CLEAR_COVER_IN, n_top, n_side):
                    continue
                result.append(replace(design, bar_size=bar_size, top_bars=n_top, bot_bars=n_top, side_bars=n_side))
    return result


def beam_cage_candidates(design, cfg, through_depth_in=None):
    """Beam cages of the declared domain for one section (bar size and counts; hoops unchanged)."""
    result = []
    for bar_size in cfg.rebar.bar_sizes_beam:
        if through_depth_in is not None and sp.through_joint_bar_depth_factor() * sp.rebar_diameter(bar_size) > through_depth_in:
            continue                                                    # 18.8.2.3: 20 db (Grade 60) or 26 db (Grade 80)
        for n_top in cfg.rebar.beam_n_iter():
            for n_bot in ((n_top,) if cfg.rebar.grouped_beam_symmetric else cfg.rebar.beam_n_iter()):
                if n_top < 2 or n_bot < 2:
                    continue
                result.append(replace(design, bar_size=bar_size, top_bars=n_top, bot_bars=n_bot))
    return result


# ---- beams ------------------------------------------------------------------------------------------------
def _beam_through_depth(state, gid):
    """Smallest joint-core depth a through bar of this group passes (None when every joint is an end)."""
    depths = []
    for tag in state.groups[gid]["member_tags"]:
        member = state.member(tag)
        k, i, j = member.story_or_floor, member.grid_i, member.grid_j
        ends = ((i, j), (i + 1, j)) if member.member_type == "beam_x" else ((i, j), (i, j + 1))
        for gi, gj in ends:
            joint = mg.joint_members(k, gi, gj)
            axis = mp.beam_axis(member)
            if joint[f"beam_{axis}_minus"] is not None and joint[f"beam_{axis}_plus"] is not None:
                core = mp.joint_core_dimensions(k, gi, gj)
                depths.append(core[0] if axis == "x" else core[1])
    return min(depths) if depths else None


def _refused(code, detail):
    """A refused beam cage: its first reason, by code and class (BEAM_REFUSALS)."""
    return {"admissible": False, "reason": detail, "reason_code": code, "reason_class": BEAM_REFUSALS[code][0]}


def _lanes_admit(design, fit):
    """Whether ``fit`` lanes per layer carry the cage's bars within the layer cap."""
    n = max(design.top_bars, design.bot_bars)
    return bool(fit) and math.ceil(n / fit) <= sp.BEAM_BAR_MAX_LAYERS


def evaluate_beam_candidate(state, gid, candidate, demand, cfg, representatives=None, paired_demand=None):
    """Price one beam cage in its place: rows from the layering, code limits, governing flexural DCR.

    ``demand`` = (largest sagging, largest hogging moment over the group's members and combinations): against the
    weakest member strength it gives ``dcr_union_bound``. ``paired_demand`` = {representative tag: (sagging, hogging)}
    of each member context (_paired_beam_demand): each against that context's own strength gives the member-paired
    ``dcr`` the final check reports; without it ``dcr`` is the union bound (the earlier rule).
    Returns {"admissible", "threads", "layers", "lanes", "dcr", "dcr_basis", "dcr_union_bound", "steel_in2", "reason",
    "reason_code", "reason_class"}; a refusal names only its first reason.
    """
    from Design.SMRF_Beam_Slab_Strength import validate_bar_rows
    design = candidate
    trial = _trial_state(state, gid, design)
    with mg.installed(trial):
        try:
            arrangement = bar_layers.arrangement()
        except mg.GroupedStateError as exc:
            return _refused("bar_layers_unsettled", f"bar layers do not settle: {exc}")
    rows = arrangement[gid]
    n = max(design.top_bars, design.bot_bars)
    fit = arrangement["_basis"].get("bars_per_layer_governing", {}).get(gid)
    stacked = arrangement["_basis"].get("stacked", False)
    threads = (not stacked) or (fit is not None and fit > 0 and math.ceil(n / fit) <= sp.BEAM_BAR_MAX_LAYERS)
    lanes = {"bars_per_layer": fit if stacked else None, "layers_needed": math.ceil(n / fit) if stacked and fit else None,
             "max_layers": sp.BEAM_BAR_MAX_LAYERS, "stacked": bool(stacked),
             "blocked_by": None if threads else ("no_lane" if not fit else "layer_cap")}
    db = sp.rebar_diameter(design.bar_size)
    if not threads:
        # No lane arrangement: the bars are kept in one row (the threading check fails on it); they must
        # at least fit across the width.
        cover = sp.longitudinal_cover_in("beam", design.bar_size, design.stirrup_bar_size)
        if (design.b_in - 2 * cover) / (n - 1) - db < sp.longitudinal_clear_spacing_in("beam", design.bar_size):
            return _refused("width_fit", "bars do not fit across the width")
    try:
        validate_bar_rows(rows["top"], design.top_bars, "top bars", design.h_in)
        validate_bar_rows(rows["bottom"], design.bot_bars, "bottom bars", design.h_in)
    except BeamBarRowsError as exc:
        return _refused("row_past_mid_depth", str(exc))
    except ValueError as exc:
        return _refused("rows_invalid", str(exc))
    strengths = checks.beam_flexural_strengths(design, rows)
    with mg.installed(trial):
        tags = tuple(representatives if representatives is not None else checks.beam_group_representatives(trial, gid))
        resolved = [checks.resolved_beam_flexural_strengths(trial.member(tag), rows) for tag in tags]
    if any(not value[sign].get("tension_controlled", True) for value in resolved for sign in ("positive", "negative")):
        return _refused("tension_control", "composite beam is not tension controlled (9.3.3.1)")
    for sign in ("positive", "negative"):
        strengths[sign]["phi_mn_kip_in"] = min(value[sign]["phi_mn_kip_in"] for value in resolved)
    d = min(strengths["positive"]["d_in"], strengths["negative"]["d_in"])
    ab = sp.rebar_area(design.bar_size)
    as_top, as_bot = design.top_bars * ab, design.bot_bars * ab
    as_min = max(strengths["positive"]["as_min_in2"], strengths["negative"]["as_min_in2"])
    code_cap = sp.beam_rho_max_code()
    rho_cap = min(code_cap, getattr(cfg.rebar, "rho_beam_practical_max", 0.025))
    if d <= 0 or max(as_top, as_bot) > rho_cap * design.b_in * d:
        # Above the code limit, or above the practical ceiling only (a proportioning policy of this basis).
        over_code = d <= 0 or max(as_top, as_bot) > code_cap * design.b_in * d
        return _refused("steel_ratio_code" if over_code else "steel_ratio_practical",
                        f"steel ratio above {rho_cap:g} (18.6.3.1 at {code_cap:g} for Grade {sp.FY_KSI:g}; the selection "
                        "ceiling of 2026-10-17)")
    if min(as_top, as_bot) < as_min:
        return _refused("minimum_steel", "below minimum steel (9.6.1.2)")

    def nominal(area):
        a = area * sp.FY_KSI / (0.85 * design.fc_ksi * design.b_in)
        return area * sp.FY_KSI * (d - a / 2.0)

    for value in resolved:
        sections = value.get("nominal_sections") or {end: {"positive": nominal(as_bot), "negative": nominal(as_top)}
                                                    for end in ("i", "j", "span")}
        # 18.6.3.2 on each side's more demanding basis (Group_Checks.beam_detailing_strengths): the face negative
        # strength on the strong-column basis where it is larger.
        faces = value.get("nominal_negative_faces_18_6_3_2") or {}
        negative = {end: max(section["negative"], faces.get(end, 0.0)) for end, section in sections.items() if end != "span"}
        maximum = max(max(section["positive"] for end, section in sections.items() if end != "span"),
                      max(negative.values()))
        if (any(section["positive"] < 0.5 * negative[end] for end, section in sections.items() if end != "span")
                or min(moment for section in sections.values() for moment in section.values()) <= 0
                or min(moment for section in sections.values() for moment in section.values()) < 0.25 * maximum):
            return _refused("strength_balance", "end reversal or along-span strength balance (18.6.3.2)")
    mu_pos, mu_neg = demand
    union = max(mu_pos / strengths["positive"]["phi_mn_kip_in"], mu_neg / strengths["negative"]["phi_mn_kip_in"])
    dcr, basis = union, "union_of_demands"
    if paired_demand is not None:
        missing = [tag for tag in tags if tag not in paired_demand]
        if missing:
            raise ValueError(f"the paired demand of {gid} has no entry for the member contexts {missing[:4]}")
        # Each member context's own envelope against its own strength (the pairing of Group_Checks.beam_group_strength).
        dcr = max(max(paired_demand[tag][0] / value["positive"]["phi_mn_kip_in"] if paired_demand[tag][0] > 0.0 else 0.0,
                      paired_demand[tag][1] / value["negative"]["phi_mn_kip_in"] if paired_demand[tag][1] > 0.0 else 0.0)
                  for tag, value in zip(tags, resolved))
        basis = "member_paired"
    return {"admissible": True, "threads": threads, "layers": max(rows["top"]["layers"], rows["bottom"]["layers"]),
            "lanes": lanes, "dcr": dcr, "dcr_basis": basis, "dcr_union_bound": union, "steel_in2": as_top + as_bot,
            "reason": None, "reason_code": None, "reason_class": None,
            "strength_basis": checks.BEAM_STRENGTH_BASIS if sp.SLAB_THICKNESS_IN is not None else "rectangular_no_slab"}


def _paired_beam_demand(state, gid, rows):
    """(representative tags, {representative tag: (largest sagging, largest hogging)}) of one beam group, the state
    installed: each member's own envelope gathered by the flange/anchorage context its strength is computed on
    (Group_Checks.beam_section_context; one representative per context, as beam_group_representatives picks it), so a
    cage is priced as the final check prices it. A context without demand rows carries (0, 0)."""
    representative, of_tag = {}, {}
    for tag in state.groups[gid]["member_tags"]:
        of_tag[tag] = representative.setdefault(checks.beam_section_context(state.member(tag)), tag)
    demand = {tag: (0.0, 0.0) for tag in representative.values()}
    for tag, _combination, mu_pos, mu_neg, _vu in rows:
        rep = of_tag[tag]
        demand[rep] = (max(demand[rep][0], mu_pos), max(demand[rep][1], mu_neg))
    return tuple(representative.values()), demand


# ---- threading evidence -------------------------------------------------------------------------------------------
def _cage(design):
    return {"bar_size": design.bar_size, "top_bars": design.top_bars, "bot_bars": design.bot_bars}


def _priced_entry(gids, candidate, result):
    return {**_cage(candidate), "dcr": result["dcr"], "dcr_union_bound": result["dcr_union_bound"],
            "steel_in2": result["steel_in2"], "threads": result["threads"], "layers": result["layers"],
            "by_group": dict(zip(gids, result["by_group"]))}


def _lane_governors(state, gids, design):
    """{gid: lanes} of the cage ``design`` in each group's place: the lanes per layer at every joint core the group's
    bars pass (Model.Member_Bar_Layers.bars_per_layer_at_joint, the rule the layering uses: the joint's bar inventory
    at a transition), the cores that give the governing (smallest) count and the joints where they do. The core cages'
    own lanes are kept apart (``core_cage_lanes``), and every band-boundary joint whose bar inventory gives fewer lanes
    than its core cage is listed under ``transitions`` as "<core>><above>", by both column groups (the core below and
    the column above), with its joint lanes, the core cage's lanes and the joints' geometry: there the transition of
    the two cages is the bottleneck, not the core alone. Geometry only."""
    trial = _trial_state(state, gids[0], replace(design, member_type=state.groups[gids[0]]["member_type"]))
    result = {}
    with mg.installed(trial):
        kinds = bar_layers.joint_types()
        for gid in gids:
            axis = state.groups[gid]["member_type"][len("beam_"):]
            cores, own, joints, transitions = {}, {}, {}, {}
            for kind in kinds:
                beam = kind[axis]
                if beam is None or beam.group_id != gid:
                    continue
                core = kind["column"].group_id
                cage = bar_layers.bars_per_layer(kind["column"].design, beam.design, axis)
                fit = cage
                for joint in kind["joints"]:
                    at = bar_layers.bars_per_layer_at_joint(*joint, beam.design, axis)
                    fit = min(fit, at)
                    if at < cage:
                        inventory = bar_layers.joint_inventory(*joint)
                        entry = transitions.setdefault(f"{inventory['core_group']}>{inventory['above_group']}", {
                            "core_group": inventory["core_group"], "above_group": inventory["above_group"],
                            "joint_lanes": at, "core_cage_lanes": cage, "established": True,
                            "transition": inventory["transition"], "joints": [], "geometry": set()})
                        entry["joint_lanes"] = min(entry["joint_lanes"], at)
                        entry["established"] = entry["established"] and inventory["established"]
                        entry["joints"].append(list(joint))
                        entry["geometry"].add((inventory["joint_depth_in"], inventory["upper_clear_height_in"]))
                own[core] = cage if core not in own else min(own[core], cage)
                cores[core] = fit if core not in cores else min(cores[core], fit)
                joints.setdefault(core, []).extend(kind["joints"])
            low = min(cores.values()) if cores else None
            governing = sorted(c for c, v in cores.items() if v == low)
            for entry in transitions.values():
                entry["joints"] = sorted(entry["joints"])[:6]
                entry["geometry"] = [list(g) for g in sorted(entry["geometry"])]
            result[gid] = {"axis": axis, "bars_per_layer": low,
                           "layers_needed": math.ceil(max(design.top_bars, design.bot_bars) / low) if low else None,
                           "max_layers": sp.BEAM_BAR_MAX_LAYERS, "threads": _lanes_admit(design, low),
                           "cores": dict(sorted(cores.items())), "governing_cores": governing,
                           "joints": {c: [list(j) for j in sorted(joints[c])[:6]] for c in governing},
                           "core_cage_lanes": dict(sorted(own.items())), "transitions": dict(sorted(transitions.items())),
                           "governing_transitions": sorted(n for n, e in transitions.items() if e["joint_lanes"] == low)}
    return result


def _blocking(lanes, gids, candidate):
    """What keeps ``candidate`` from threading, from its _lane_governors: (the cores whose joints block it, whatever the
    cause; the cores whose own cage blocks it; the band-boundary transitions whose joint inventory blocks it, each as
    {"core_group", "above_group"})."""
    cores = sorted({c for gid in gids for c, fit in lanes[gid]["cores"].items() if not _lanes_admit(candidate, fit)})
    cages = sorted({c for gid in gids for c, fit in lanes[gid]["core_cage_lanes"].items() if not _lanes_admit(candidate, fit)})
    pairs = {name: {"core_group": e["core_group"], "above_group": e["above_group"]} for gid in gids
             for name, e in lanes[gid]["transitions"].items() if not _lanes_admit(candidate, e["joint_lanes"])}
    return cores, cages, [pairs[name] for name in sorted(pairs)]


def _row_evidence(state, gids, design):
    """The rows of the cage ``design`` in each group's place (Model.Member_Bar_Layers) with what set each offset
    (row_controls: the crossing layer or mat and the joint type). None when the layering does not settle."""
    trial = _trial_state(state, gids[0], replace(design, member_type=state.groups[gids[0]]["member_type"]))
    with mg.installed(trial):
        try:
            rows, controls = bar_layers.arrangement(), bar_layers.row_controls()
        except mg.GroupedStateError:
            return None
    return {gid: {face: {"per_layer": rows[gid][face]["per_layer"], "offsets_in": rows[gid][face]["offsets_in"],
                         "set_by": controls[gid][face]} for face in ("top", "bottom")} for gid in gids}


def lane_lookahead(column, beam, axis, sizes, cfg, layout=None):
    """Geometry only: the lanes per layer ``beam``'s bars would find in ``column``'s place at each square size of
    ``sizes``, with the same column cage and the most any admissible cage of that size gives (column_cage_candidates:
    ratios, spacing, supported bars, confinable layout). Not strength, the strong-column rule, bar bond, transitions,
    joints or an analysis: it informs a size proposal and is never a design."""
    result = []
    for size in sizes:
        sized = replace(column, b_in=float(size), h_in=float(size))
        cages = column_cage_candidates(sized, cfg, layout=layout)
        same = any((c.bar_size, c.top_bars, c.bot_bars, c.side_bars)
                   == (column.bar_size, column.top_bars, column.bot_bars, column.side_bars) for c in cages)
        result.append({"size_in": float(size), "same_cage_lanes": bar_layers.bars_per_layer(sized, beam, axis),
                       "same_cage_admissible": same,
                       "most_lanes_any_cage": max((bar_layers.bars_per_layer(c, beam, axis) for c in cages), default=None),
                       "admissible_cages": len(cages)})
    return result


def transition_lookahead(lower, upper, beam, axis, sizes, cfg, geometry, needed, layout=None):
    """Geometry only, at the band-boundary joints of one transition (``geometry``: their distinct [joint depth, upper
    clear height]): the lanes per layer ``beam``'s bars find through the joint's own bar inventory
    (Model.Member_Bar_Layers, swept bend intervals) when the column above takes each square size of ``sizes`` up to the
    column below's (``upper_band``: a larger column above is not a declared transition), when the column below takes
    each larger size (``lower_band``), or when both take it together (``coupled``), each band keeping its cage; with
    whether the declared transition is supported at every one of those joints and whether both cages are admissible at
    their sizes (column_cage_candidates). The first size of each family that gives ``needed`` lanes per layer with a
    supported transition and admissible cages is reported; a size is never said to open the joint from a core cage
    alone. Only this joint: the resized band's other joints, strength, the strong-column rule, bar bond and joint shear
    are not evaluated, and nothing is analysed: it informs a size proposal and is never a design. ``coupled`` changes
    two bands and is evidence only (a search proposal changes one gene)."""
    def key(c):
        return c.bar_size, c.top_bars, c.bot_bars, c.side_bars

    admissible = {}

    def cage_ok(design):
        token = (design.b_in, design.h_in, key(design))
        if token not in admissible:
            admissible[token] = any(key(c) == key(design) for c in column_cage_candidates(design, cfg, layout=layout))
        return admissible[token]

    def row(size, lo, up):
        supported, lanes, kinds = True, None, set()
        core = bar_layers.bars_per_layer(lo, beam, axis)
        for depth, clear in geometry:
            transition = column_transition(lo, up, fy_ksi=sp.FY_KSI, clear_cover_in=sp.COL_CLEAR_COVER_IN,
                                           upper_clear_height_in=clear, joint_depth_in=depth)
            bars = None if lo == up else bar_layers.transition_inventory(lo, up, transition)
            at = core if bars is None else min(core, bar_layers.inventory_lanes(bars, lo, beam, axis))
            supported = supported and bool(transition["supported"])
            lanes = at if lanes is None else min(lanes, at)
            kinds.add(transition["kind"])
        return {"size_in": float(size), "transition_supported": supported, "transition_kinds": sorted(kinds),
                "joint_lanes": lanes, "core_cage_lanes": core, "cages_admissible": cage_ok(lo) and cage_ok(up)}

    square = abs(lower.b_in - lower.h_in) < 1e-9 and abs(upper.b_in - upper.h_in) < 1e-9
    families = {"upper_band": [], "lower_band": [], "coupled": []}
    if square and geometry:
        for size in sizes:
            if upper.b_in + 1e-9 < size <= lower.b_in + 1e-9:
                families["upper_band"].append(row(size, lower, replace(upper, b_in=float(size), h_in=float(size))))
            if size > lower.b_in + 1e-9:
                grown = replace(lower, b_in=float(size), h_in=float(size))
                families["lower_band"].append(row(size, grown, upper))
                families["coupled"].append(row(size, grown, replace(upper, b_in=float(size), h_in=float(size))))

    def first(rows):
        return next((r["size_in"] for r in rows
                     if r["transition_supported"] and r["cages_admissible"] and r["joint_lanes"] >= needed), None)

    return {"current_size_in": lower.b_in, "current_above_size_in": upper.b_in, "square_sections": square, **families,
            "first_upper_band_size": first(families["upper_band"]), "first_lower_band_size": first(families["lower_band"]),
            "first_coupled_size": first(families["coupled"]),
            "basis": ("geometry only, at this transition's band-boundary joints: the declared transition of the two cages "
                      "at each size and the joint's bar inventory with swept bend intervals; the resized band's other "
                      "joints, strength, strong column, bar bond, joint shear and re-analysis are not evaluated; not a "
                      "design")}


def _lane_lookahead(state, gids, design, cfg):
    """{key: look-ahead} for what blocks the cage ``design``, by scope:

      "<core>/<axis>", scope core_cage: a core whose own cage blocks it. lane_lookahead over the core's square sizes
          from its own to the layout's cap, and the first size whose core-cage lanes carry the cage within the layer
          cap. It is core-only: where the core is also the lower band of a blocking transition (``transition_partners``,
          the groups above), no size here is shown to open that joint; only the transition entry evaluates it.
      "<core>><above>/<axis>", scope transition_joint: a band-boundary joint whose bar inventory blocks it, named by
          both column groups. transition_lookahead at that joint's own inventory: the column above, the core below or
          both resized, and the first size of each that opens the joint's lanes.

    Geometry only; never a design."""
    from Design.Section_Design import column_sizes_in
    layout = mg.layout_of(state.policy_id)
    sizes = list(column_sizes_in(layout))
    needed = math.ceil(max(design.top_bars, design.bot_bars) / max(1, sp.BEAM_BAR_MAX_LAYERS))
    result = {}
    for gid, lanes in _lane_governors(state, gids, design).items():
        axis = lanes["axis"]
        beam = replace(state.designs[gid], bar_size=design.bar_size, top_bars=design.top_bars, bot_bars=design.bot_bars)
        blocked = {name: e for name, e in lanes["transitions"].items() if not _lanes_admit(design, e["joint_lanes"])}
        for core, fit in lanes["core_cage_lanes"].items():
            column = state.designs[core]
            if _lanes_admit(design, fit) or column.b_in != column.h_in:
                continue
            rows = lane_lookahead(column, beam, axis, [s for s in sizes if s >= column.b_in - 1e-9], cfg, layout)
            result[f"{core}/{axis}"] = {
                "scope": "core_cage", "core_group": core, "axis": axis, "beam_group": gid, "current_size_in": column.b_in,
                "lanes_needed_per_layer": needed, "sizes": rows,
                "first_size_same_cage": next((r["size_in"] for r in rows
                                              if r["same_cage_admissible"] and r["same_cage_lanes"] >= needed), None),
                "first_size_any_cage": next((r["size_in"] for r in rows if (r["most_lanes_any_cage"] or 0) >= needed), None),
                "transition_partners": sorted(e["above_group"] for e in blocked.values() if e["core_group"] == core),
                "basis": ("geometry only: the core cage's own lanes of the declared detailing policy at each size; not a "
                          "transition joint's inventory, strength, strong column, bar bond, joint check or re-analysis; "
                          "not a design")}
        for name, entry in blocked.items():
            result[f"{name}/{axis}"] = {
                "scope": "transition_joint", "core_group": entry["core_group"], "above_group": entry["above_group"],
                "axis": axis, "beam_group": gid, "joints": entry["joints"], "lanes_needed_per_layer": needed,
                "joint_lanes": entry["joint_lanes"], "core_cage_lanes": entry["core_cage_lanes"],
                "inventory_established": entry["established"],
                **transition_lookahead(state.designs[entry["core_group"]], state.designs[entry["above_group"]], beam, axis,
                                       sizes, cfg, entry["geometry"], needed, layout)}
    return result


def _threading_evidence(state, gids, installed, priced, refusals, domain_size, cfg, deficit):
    """The ``threading`` block of a beam unit's report (THREADING_EVIDENCE_ID).

    ``constrains_strength``: an admissible cage stronger than every threading cage does not thread. ``deficit_remains``:
    the installed cage (the strongest that threads) is above the ceiling. ``threading_constrained_deficit`` is both.
    ``authenticated``: every cage of the domain reached a definite outcome (no missing evidence) and, with a deficit,
    the installed cage is the strongest threading one. With a constrained deficit the block also carries the repair
    prospects (stronger nonthreading cages), the lanes and blocking cores of the installed and the strongest
    nonthreading cage, the installed rows with what set them and a geometry-only look-ahead of larger columns. A
    refused or nonthreading cage stays what it is: nothing here admits or accepts anything."""
    classes = {gid: {} for gid in gids}
    for gid in gids:
        for code, count in refusals[gid].items():
            kind = BEAM_REFUSALS[code][0]
            classes[gid][kind] = classes[gid].get(kind, 0) + count
    threading = [item for item in priced if item[1]["threads"]]
    blocked = [item for item in priced if not item[1]["threads"]]
    best_t = min(threading, key=lambda item: item[1]["dcr"]) if threading else None
    best_n = min(blocked, key=lambda item: item[1]["dcr"]) if blocked else None
    constrains = best_n is not None and (best_t is None or best_n[1]["dcr"] < best_t[1]["dcr"] * (1.0 - 1e-9))
    missing = sum(classes[gid].get("missing_evidence", 0) for gid in gids)
    causes = {}
    for _candidate, result in blocked:
        cause = "no_lane" if any((row.get("lanes") or {}).get("blocked_by") == "no_lane" for row in result["by_group"]) \
            else "layer_cap"
        causes[cause] = causes.get(cause, 0) + 1
    entry = next((_priced_entry(gids, c, r) for c, r in priced if _cage(c) == _cage(installed)), None)
    is_best = best_t is not None and _cage(best_t[0]) == _cage(installed)
    evidence = {
        "basis": THREADING_EVIDENCE_ID,
        "authenticated": missing == 0 and (not deficit or best_t is None or is_best),
        "constrains_strength": bool(constrains), "deficit_remains": bool(deficit),
        "threading_constrained_deficit": bool(constrains and deficit),
        "installed": entry if entry is not None else {**_cage(installed), "priced": False},
        "best_threading": None if best_t is None else _priced_entry(gids, *best_t),
        "best_nonthreading": None if best_n is None else _priced_entry(gids, *best_n),
        "first_refusals": {gid: dict(sorted(refusals[gid].items())) for gid in gids},
        "refusal_classes": {gid: dict(sorted(classes[gid].items())) for gid in gids},
        "domain_cages": domain_size, "admissible_cages": len(priced), "threading_cages": len(threading),
        "nonthreading_by_cause": dict(sorted(causes.items())),
        "dcr_basis": "member_paired; dcr_union_bound is a sufficient flexural screen only, never a member DCR",
        "policy": passage_policy(sp.rebar_diameter(installed.bar_size)),
    }
    if not (constrains and deficit):
        return evidence
    ceiling = cfg.dcr.dcr_hard_max
    limit = best_t[1]["dcr"] if best_t is not None else math.inf
    stronger = [item for item in blocked if item[1]["dcr"] < limit * (1.0 - 1e-9)]
    # Repair prospects: the cages that reach the demand, least steel first, then the others, strongest first.
    stronger.sort(key=lambda item: (item[1]["dcr"] > ceiling,
                                    item[1]["steel_in2"] if item[1]["dcr"] <= ceiling else item[1]["dcr"], item[1]["steel_in2"]))
    prospects = []
    for candidate, result in stronger[:THREADING_PROSPECTS_MAX]:
        lanes = _lane_governors(state, gids, candidate)
        cores, cages, transitions = _blocking(lanes, gids, candidate)
        prospects.append({**_cage(candidate), "dcr": result["dcr"], "dcr_union_bound": result["dcr_union_bound"],
                          "steel_in2": result["steel_in2"], "blocking_cores": cores, "blocking_core_cages": cages,
                          "blocking_transitions": transitions, "lanes": lanes})
    pairs = {f"{t['core_group']}>{t['above_group']}": t for p in prospects for t in p["blocking_transitions"]}
    evidence.update(
        prospects=prospects,
        lanes={"installed": _lane_governors(state, gids, installed),
               "strongest_nonthreading": _lane_governors(state, gids, best_n[0])},
        blocking_cores=sorted({core for p in prospects for core in p["blocking_cores"]}),
        blocking_core_cages=sorted({core for p in prospects for core in p["blocking_core_cages"]}),
        blocking_transitions=[pairs[name] for name in sorted(pairs)],
        rows=_row_evidence(state, gids, installed),
        lookahead=_lane_lookahead(state, gids, best_n[0], cfg))
    evidence["core_groups"] = sorted({core for gid in gids for core in evidence["lanes"]["installed"][gid]["cores"]})
    return evidence


def select_beam_group(state, gid, actions, cfg):
    """The cage one beam group selects, given every other group as it stands. Returns (design, report)."""
    return select_beam_groups(state, (gid,), actions, cfg)


def _merge_beam_results(results):
    """One cage priced in several groups' places: it threads only if it threads everywhere, its layers, DCR and union
    bound are the worst, its steel the sum."""
    return {"admissible": True, "threads": all(r["threads"] for r in results), "layers": max(r["layers"] for r in results),
            "dcr": max(r["dcr"] for r in results),
            "dcr_union_bound": max(r.get("dcr_union_bound", r["dcr"]) for r in results),
            "steel_in2": sum(r["steel_in2"] for r in results), "reason": None,
            "by_group": [{k: r.get(k) for k in ("threads", "layers", "dcr", "dcr_union_bound", "lanes")} for r in results]}


def select_beam_groups(state, gids, actions, cfg, installed_only=False):
    """The cage the beam groups ``gids`` select together: one group under the band policy, the groups of a design
    tie under the system policy. Priced in each group's place on its own members' demands, each member context's
    envelope against its own strength (the pairing of the final check; the union of the tie's demands against the
    weakest strength is kept as ``dcr_union_bound``), admissible only in every group's own place (its own layering,
    joint cores and column lanes). Returns (design typed as the first group, report); the report's ``threading``
    block is the threading evidence of the module docstring. ``installed_only``: the installed cage is the only
    candidate, under every rule below (verify_installed_selection); status "selected" then says it is admissible in
    every group's place and within dcr_hard_max."""
    gids = tuple(gids)
    current = state.designs[gids[0]]
    rows_by = {gid: checks.beam_demand_rows(state.groups[gid]["member_tags"], actions) for gid in gids}
    rows = [row for gid in gids for row in rows_by[gid]]
    demand = (max(r[2] for r in rows), max(r[3] for r in rows))
    priced, refusals = [], {gid: {} for gid in gids}
    with mg.installed(state):
        depths = [_beam_through_depth(state, gid) for gid in gids]
        paired = {gid: _paired_beam_demand(state, gid, rows_by[gid]) for gid in gids}
    through = min((d for d in depths if d is not None), default=None)
    domain = beam_cage_candidates(current, cfg)
    candidates = beam_cage_candidates(current, cfg, through)
    if installed_only:
        def installed(candidate):
            return (candidate.bar_size, candidate.top_bars, candidate.bot_bars) == (current.bar_size, current.top_bars,
                                                                                     current.bot_bars)
        domain, candidates = [c for c in domain if installed(c)], [c for c in candidates if installed(c)]
    for gid in gids:
        if len(domain) > len(candidates):
            refusals[gid]["through_joint_depth"] = len(domain) - len(candidates)
    for candidate in candidates:
        results = [evaluate_beam_candidate(state, gid, replace(candidate, member_type=state.groups[gid]["member_type"]), demand, cfg,
                                           paired[gid][0], paired_demand=paired[gid][1])
                   for gid in gids]
        for gid, result in zip(gids, results):
            if not result["admissible"]:
                refusals[gid][result["reason_code"]] = refusals[gid].get(result["reason_code"], 0) + 1
        if all(r["admissible"] for r in results):
            priced.append((candidate, _merge_beam_results(results)))
    report = {"group_id": gids[0] if len(gids) == 1 else None, "groups": list(gids),
              "demand_mu_positive_kip_in": demand[0], "demand_mu_negative_kip_in": demand[1],
              "candidates_admissible": len(priced), "through_bar_depth_in": through,
              "strength_basis": checks.BEAM_STRENGTH_BASIS if sp.SLAB_THICKNESS_IN is not None else "rectangular_no_slab",
              "dcr_basis": "member_paired", **({"installed_only": True} if installed_only else {})}

    def evidence(installed, deficit):
        return _threading_evidence(state, gids, installed, priced, refusals, len(domain), cfg, deficit)

    if not priced:
        return current, {**report, "status": "no_admissible_cage", "exhausted": True,
                         "detail": "no cage of the declared domain satisfies the code limits in this section",
                         "threading": evidence(current, True)}
    pool = [item for item in priced if item[1]["threads"]] or priced
    report["lanes_available"] = any(item[1]["threads"] for item in priced)
    feasible = [item for item in pool if item[1]["dcr"] <= cfg.dcr.dcr_hard_max]
    if not feasible:
        candidate, result = min(pool, key=lambda item: item[1]["dcr"])
        # Lane-limited: a cage that reaches the demand exists but does not thread the column bars (room, not strength).
        # Its meaning is unchanged; whether threading constrains the strength at all is the threading block's.
        lane_limited = any(item[1]["dcr"] <= cfg.dcr.dcr_hard_max for item in priced if not item[1]["threads"])
        return candidate, {**report, "status": "strength_short", "exhausted": True, "dcr": result["dcr"],
                           "dcr_union_bound": result["dcr_union_bound"],
                           "lane_limited": lane_limited, "strongest_dcr_any_cage": min(item[1]["dcr"] for item in priced),
                           "detail": ("the cages that reach the demand do not thread the column cage in enough lanes; the strongest "
                                      "threading cage is installed so the shortfall is measured" if lane_limited else
                                      "no admissible cage reaches the demand; the strongest is installed so the shortfall is measured"),
                           "threading": evidence(candidate, True)}
    if cfg.rebar.beam_prefer_fewest_layers:
        fewest = min(item[1]["layers"] for item in feasible)
        feasible = [item for item in feasible if item[1]["layers"] == fewest]
    candidate, result = min(feasible, key=lambda item: _objective(item[1]["dcr"], item[1]["steel_in2"], cfg))
    return candidate, {**report, "status": "selected", "exhausted": False, "dcr": result["dcr"],
                       "dcr_union_bound": result["dcr_union_bound"], "layers": result["layers"],
                       "threads": result["threads"], "threading": evidence(candidate, False)}


# ---- columns ----------------------------------------------------------------------------------------------
def _column_chain(state, location):
    """Per band from the base, the column group ids of one plan location (a tuple of one), or of several locations
    selected together (the locations of a design tie under the system policy)."""
    locations = (location,) if isinstance(location, str) else tuple(location)
    groups = [g for g in state.groups.values() if g["member_type"] == "column" and g["location_class"] in locations]
    bands = {}
    for group in groups:
        bands.setdefault(min(group["stories_or_floors"]), []).append(group["group_id"])
    return [tuple(sorted(bands[start])) for start in sorted(bands)]


def _axial_ranges(tags, actions):
    """{tag: {"i": (min, max), "j": (min, max)}} joint-face axial range of each column over the combinations."""
    result = {}
    for tag in tags:
        ends = {}
        for end in ("i", "j"):
            values = [float(a["members"][str(tag)][f"axial_{end}_kip"]) for a in actions]
            ends[end] = (min(values), max(values))
        result[tag] = ends
    return result


def _column_mn_estimate(design, axis, axial_range):
    """Least nominal moment over the ends of an axial range (0 when the range leaves the P-M surface)."""
    diagram = mp.column_pm_diagram(design, "h" if axis == "x" else "b")
    return min(column_moment_at_axial(p, diagram) for p in axial_range)


def _joint_requirements(state, established):
    """{(level, i, j): {axis: required sum of column Mn}}: 1.2 times the larger beam sum of the two sways."""
    if not established:
        return {}
    requirements = {}
    for level in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                joint = mg.joint_members(level, i, j)
                entry = {}
                for axis in ("x", "y"):
                    sums = {"positive": 0.0, "negative": 0.0}
                    for side, end in (("minus", "j"), ("plus", "i")):
                        beam = joint[f"beam_{axis}_{side}"]
                        if beam is None:
                            continue
                        basis = mp.beam_strengths(beam)["basis"]
                        for sign in ("positive", "negative"):
                            flexure = sign if end == "j" else ("negative" if sign == "positive" else "positive")
                            # The strong-column basis the evaluator reads (either datum, Group_Capacity.scwb_evidence).
                            sums[sign] += basis[f"scwb_{'sagging' if flexure == 'positive' else 'hogging'}_{end}_kip_in"]
                    entry[axis] = sp.SCWB_RATIO_MIN * max(sums.values())
                requirements[(level, i, j)] = entry
    return requirements


def _core_beams(state, gid):
    """[(beam design, axis)] of the beams whose bars thread this column group's cage (it is their joint core)."""
    pairs = {}
    for tag in state.groups[gid]["member_tags"]:
        member = state.member(tag)
        joint = mg.joint_members(member.story_or_floor, member.grid_i, member.grid_j)
        for axis in ("x", "y"):
            for side in ("minus", "plus"):
                beam = joint[f"beam_{axis}_{side}"]
                if beam is not None:
                    pairs[(beam.group_id, axis)] = (beam.design, axis)
    return list(pairs.values())


def _column_clear_height_face(member):
    from Design.Group_Capacity import column_clear_heights
    return column_clear_heights(member)["face"]


def _cage_develops_its_bars(candidate, clear_height_in):
    """ACI 318-19 18.7.4.3 with Eq. (25.4.2.4a) at the cage's own cb and Ktr = 0 (Design/ACI_Checks)."""
    from Design.ACI_Checks import column_bar_bond_18_7_4_3, column_bar_cb_in
    cb = column_bar_cb_in(candidate.b_in, candidate.h_in, sp.COL_CLEAR_COVER_IN, candidate.stirrup_bar_size,
                          candidate.bar_size, candidate.top_bars, candidate.side_bars)["cb_in"]
    return column_bar_bond_18_7_4_3(candidate.bar_size, candidate.fc_ksi, sp.FY_KSI, cb, clear_height_in)["passes"]


def select_column_chain(state, location, actions, cfg, scwb_margin=1.0, requirements=None, threading_required=False,
                        installed_only=False):
    """Cages of the column groups of one plan location, chosen together. Returns ({gid: design}, report).

    ``requirements`` are the strong-column sums of _joint_requirements built once per selection pass on the
    beams just selected (their rows validated against the cages of that moment); built here when not given.
    ``threading_required`` (a coordinated trial, coordinate_threading): True, or the column group ids whose bands it
    applies to. Such a band's cages must let the bars of every beam framing into its joints thread them within the
    layer cap; its nonthreading pools are not used. The other bands keep their pools. ``installed_only``: each band's
    installed cage is its only candidate, under every rule below (a verification of the installed chain on the beams
    installed now, close_terminal_arrangement); status "selected" then says the installed chain meets them.

    A pass over the bands from the base: each band's candidates carry the least total cost of a feasible
    chain below them. Node feasibility: strength on the band's own concurrent demands, and 18.7.3.2 at
    the joints whose two columns are both in the band and at the roof, except at a roof joint whose
    column's largest joint-face axial compression is below Ag f'c / 10 of its section (ACI 318-19
    18.7.3.1, SMRF_Joints.discontinuous_column_exempt, as the evaluator decides it; the cages share the
    section and grade, so the exemption is the band's, independent of the cage and the margin). Link
    feasibility: a supported transition, and 18.7.3.2 at the band-boundary joints with the lower band's
    cage below and the upper band's above.
    """
    chain = _column_chain(state, location)
    label = location if isinstance(location, str) else "+".join(location)
    if not chain:
        return {}, {"location": label, "status": "absent"}
    # The strong-column estimate screens the cages whenever the beams have end strengths to sum (a slab is
    # installed). Without an established slab layout the beam sums carry no slab mats, so the estimate is the
    # grouped form of the uniform proxy screen and the exact joint evaluation stays not_evaluated.
    established = sp.SLAB_THICKNESS_IN is not None
    layout_established = established and (sp.SLAB_REINFORCEMENT or {}).get("layout") is not None
    alpha = getattr(cfg.dcr, "biaxial_contour_exponent", 1.5)
    with mg.installed(state):
        if requirements is None:
            requirements = _joint_requirements(state, established)
        core_beams = {gids: [pair for gid in gids for pair in _core_beams(state, gid)] for gids in chain}
    report = {"location": label, "chain": [list(gids) for gids in chain], "scwb_margin": scwb_margin,
              **({"threading_required": True if threading_required is True else sorted(threading_required)}
                 if threading_required else {}),
              "scwb_in_selection": established,
              "scwb_screen": ("beam plus developed slab mats" if layout_established else
                              "proxy: beams without slab mats (no slab layout established)" if established else "none"),
              "bands": {}}
    nodes = []                                                      # per band: [(candidate, cost, info)]
    for gids in chain:
        key_label, current = "+".join(gids), state.designs[gids[0]]
        tags = [tag for gid in gids for tag in state.groups[gid]["member_tags"]]
        rows = checks.column_demand_rows(tags, actions)
        ranges = _axial_ranges(tags, actions)
        members = {tag: state.member(tag) for tag in tags}
        stories = set().union(*(set(state.groups[gid]["stories_or_floors"]) for gid in gids))
        priced, rejected = [], {"strength": 0, "axial_domain": 0, "scwb": 0, "bar_bond": 0}
        clear_height = min(_column_clear_height_face(member) for member in members.values())
        roof_exempt = sorted(tag for tag, member in members.items() if member.story_or_floor + 1 > sp.NUM_FLOOR
                             and discontinuous_column_exempt(ranges[tag]["j"][1], member.design.b_in * member.design.h_in,
                                                             member.design.fc_ksi)[0])
        domain = column_cage_candidates(current, cfg, layout=mg.layout_of(state.policy_id))
        if installed_only:
            domain = [c for c in domain if (c.bar_size, c.top_bars, c.bot_bars, c.side_bars)
                      == (current.bar_size, current.top_bars, current.bot_bars, current.side_bars)]
            if not domain:
                rejected["not_in_domain"] = 1
        for candidate in domain:
            if not _cage_develops_its_bars(candidate, clear_height):
                # ACI 318-19 18.7.4.3 (2026-10-02): 1.25 ld at Ktr = 0 must fit in half the clear height.
                rejected["bar_bond"] += 1
                continue
            envelope = checks.column_pm_envelope(checks.column_pm_diagrams(candidate), rows, alpha)
            dcr = envelope["dcr"]
            if envelope["outside_domain"]:
                # A demand outside the cage's axial domain, in compression or in tension, rejects the cage by name.
                rejected["axial_domain"] += 1
                continue
            if dcr > cfg.dcr.dcr_hard_max:
                rejected["strength"] += 1
                continue
            needed = [math.ceil(max(b.top_bars, b.bot_bars) / fit) if fit and fit > 0 else 99
                      for b, axis in core_beams[gids] for fit in (bar_layers.bars_per_layer(candidate, b, axis),)]
            layers = max(needed, default=1)
            # Strong column at the joints that involve only this band (both columns in it, or the roof).
            mn = {(tag, end, axis): _column_mn_estimate(candidate, axis, ranges[tag][end])
                  for tag in tags for end in ("i", "j") for axis in ("x", "y")}
            ok = True
            for tag, member in members.items():
                level = member.story_or_floor                          # the joint at this column's top
                key = (level, member.grid_i, member.grid_j)
                if key not in requirements:
                    continue
                above_story = level + 1
                if tag in roof_exempt:
                    continue                                        # 18.7.3.1 at this roof connection
                for axis in _scwb_axes(state, member):
                    required = scwb_margin * requirements[key][axis]
                    if above_story > sp.NUM_FLOOR:
                        provided = mn[(tag, "j", axis)]
                    elif above_story in stories:
                        above = mg_column_tag(above_story, member.grid_i, member.grid_j)
                        provided = mn[(tag, "j", axis)] + mn[(above, "i", axis)]
                    else:
                        continue                                    # a band boundary: decided on the link
                    if provided < required:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                rejected["scwb"] += 1
                continue
            steel = candidate.longitudinal_area_in2 * len(tags) * sp.STORY_H
            priced.append({"design": candidate, "dcr": dcr, "layers": layers, "mn": mn, "steel": steel,
                           "threads": layers <= sp.BEAM_BAR_MAX_LAYERS})
        threading = [item for item in priced if item["threads"]]
        if threading_required is True or (threading_required and set(gids) & set(threading_required)):
            # A coordinated trial: the beam bars must thread this band's cage; the nonthreading pools are not used.
            rejected["beam_lanes"] = len(priced) - len(threading)
            priced = threading
        preferred = threading or priced
        if cfg.rebar.col_prefer_fewest_beam_layers and preferred:
            fewest = min(item["layers"] for item in preferred)
            preferred = [item for item in preferred if item["layers"] == fewest]
        report["bands"][key_label] = {"candidates_admissible": len(priced), "lanes_available": bool(threading),
                                      "rejected": rejected, "members": len(tags), "groups": list(gids),
                                      "roof_exempt_18_7_3_1": roof_exempt}
        # The beam-layer preferences narrow a band's cages; they are preferences, so the chain is solved on the
        # preferred cages first and on the wider pools only when no chain exists among them.
        nodes.append({"gids": gids, "key": key_label, "pool": preferred, "pools": [preferred, threading or priced, priced],
                      "members": members, "ranges": ranges, "stories": stories})

    def boundary_ok(lower_node, lower, upper_node, upper):
        """18.7.3.2 at the joints between two bands, lower cage below and upper cage above."""
        top_story = max(lower_node["stories"])
        for tag, member in lower_node["members"].items():
            if member.story_or_floor != top_story:
                continue
            key = (top_story, member.grid_i, member.grid_j)
            if key not in requirements:
                continue
            above = mg_column_tag(top_story + 1, member.grid_i, member.grid_j)
            for axis in _scwb_axes(state, member):
                if lower["mn"][(tag, "j", axis)] + upper["mn"][(above, "i", axis)] < scwb_margin * requirements[key][axis]:
                    return False
        return True

    # Per band boundary: the clear height of the upper story and the joint depth an offset bend has to fit in.
    boundary, boundary_beams = {}, {}
    with mg.installed(state):
        from Design.Group_Capacity import column_clear_heights
        for k in range(1, len(nodes)):
            member = min(nodes[k]["members"].values(), key=lambda m: m.member_tag)      # a column of the band's first story
            joint = mg.joint_members(member.story_or_floor - 1, member.grid_i, member.grid_j)
            depths = [joint[key].design.h_in for key in ("beam_x_minus", "beam_x_plus", "beam_y_minus", "beam_y_plus")
                      if joint[key] is not None]
            boundary[k] = (column_clear_heights(member)["physical"], min(depths))
            # The beams at the band-boundary joints (the floor below the band's first story), each design once.
            first = min(nodes[k]["stories"])
            seen = {}
            for upper_member in nodes[k]["members"].values():
                if upper_member.story_or_floor != first:
                    continue
                at = mg.joint_members(first - 1, upper_member.grid_i, upper_member.grid_j)
                for axis in ("x", "y"):
                    for side in ("minus", "plus"):
                        beam = at[f"beam_{axis}_{side}"]
                        if beam is not None:
                            seen[(beam.design, axis)] = (beam.design, axis)
            boundary_beams[k] = list(seen.values())
    transitions, transition_details, inventories = {}, {}, {}

    def transition_ok(k, lower, upper):
        key = (k, lower["design"], upper["design"])
        if key not in transitions:
            clear, depth = boundary[k]
            transition = column_transition(lower["design"], upper["design"], fy_ksi=sp.FY_KSI,
                                           clear_cover_in=sp.COL_CLEAR_COVER_IN, upper_clear_height_in=clear,
                                           joint_depth_in=depth)
            transitions[key] = (transition["supported"],
                                transition.get("extra_bar_length_in", 0.0) * sp.rebar_area(lower["design"].bar_size))
            transition_details[key] = transition.get("detail")
            inventories[key] = bar_layers.transition_inventory(lower["design"], upper["design"], transition)
        return transitions[key]

    def lanes_ok(k, lower, upper):
        """The beam bars at the band-boundary joints pass the joint's bar inventory of this cage pair in the layer cap."""
        bars = inventories.get((k, lower["design"], upper["design"]))
        return bars is None or all(_lanes_admit(beam, bar_layers.inventory_lanes(bars, lower["design"], beam, axis))
                                   for beam, axis in boundary_beams[k])

    def cut(counts, rule, k, lower, upper):
        """Count a cage pair a boundary rule cut, keeping the first few as named examples."""
        counts[rule] += 1
        if len(counts["examples"]) < 3:
            counts["examples"].append({
                "cut_by": rule, "lower": [lower["design"].bar_size, lower["design"].top_bars, lower["design"].side_bars],
                "upper": [upper["design"].bar_size, upper["design"].top_bars, upper["design"].side_bars],
                "detail": ((transition_details.get((k, lower["design"], upper["design"])) or "")[:240] if rule == "transition"
                           else "18.7.3.2 estimate at the band-boundary joints" if rule == "strong_column"
                           else "the beam bars at the band-boundary joints do not pass the joint's bar inventory (the "
                                "transition moves column bars inside the joint) in the layer cap")})

    def forward():
        """best[k][n] = (cost of the cheapest feasible chain ending in candidate n of band k, predecessor). Records in
        report["links"], per band boundary, the cage pairs tried and those an unsupported transition, the strong-column
        rule at the boundary joints, or (where the lower band's pool is of cages its beam bars thread) the beam lanes
        through the joint's bar inventory cut."""
        best, links = [], {}
        report["links"] = links
        for k, node in enumerate(nodes):
            column = []
            for item in node["pool"]:
                own = _objective(item["dcr"], item["steel"], cfg)
                if k == 0:
                    column.append((own, None))
                    continue
                options = []
                for index, previous in enumerate(nodes[k - 1]["pool"]):
                    if best[k - 1][index] is None:
                        continue
                    supported, extra = transition_ok(k, previous, item)
                    counts = links.setdefault(f"{nodes[k - 1]['key']}>{node['key']}",
                                              {"pairs": 0, "transition": 0, "strong_column": 0, "beam_lanes": 0,
                                               "examples": []})
                    counts["pairs"] += 1
                    if not supported:
                        cut(counts, "transition", k, previous, item)
                        continue
                    if not boundary_ok(nodes[k - 1], previous, node, item):
                        cut(counts, "strong_column", k, previous, item)
                        continue
                    if nodes[k - 1]["lanes_required"] and not lanes_ok(k, previous, item):
                        cut(counts, "beam_lanes", k, previous, item)
                        continue
                    total = tuple(a + b for a, b in zip(best[k - 1][index][0], own))
                    lines = len(node["members"]) // max(1, len(node["stories"]))        # column lines of this location
                    options.append(((total[0], total[1] + extra * lines), index))
                column.append(min(options, key=lambda o: o[0]) if options else None)
            best.append(column)
        return best, ([(entry[0], index) for index, entry in enumerate(best[-1]) if entry is not None] if best and best[-1] else [])

    tiers = ("preferred_beam_layers", "cages_the_beam_bars_thread", "all_admissible_cages")
    for tier, name in enumerate(tiers):
        if tier and all(len(node["pools"][tier]) == len(node["pools"][tier - 1]) for node in nodes):
            continue                                                    # the wider pools add nothing here
        for node in nodes:
            node["pool"] = node["pools"][tier]
            # A pool of cages the beam bars thread is a preference the band-boundary lanes must keep too.
            node["lanes_required"] = bool(node["pool"]) and all(item["threads"] for item in node["pool"])
        best, last = forward()
        report["cage_pool"] = name
        if last:
            break
    if not last:
        # No feasible chain: say where it breaks and install, per band, the cage that band would select on its own
        # demands (its own objective, the chain rules set aside), so the checks measure the shortfall on a cage of
        # the size the band needs. The heaviest cage would raise the probable moments and with them the capacity
        # shears of this line, and the failure would be read as a shear problem. Nothing is clamped or passed.
        blocked_index = next((k for k, column in enumerate(best) if not any(e is not None for e in column)), 0)
        blocked = nodes[blocked_index]["key"]
        selected = {}
        for node in nodes:
            current = state.designs[node["gids"][0]]
            if node["pool"]:
                design = min(node["pool"], key=lambda item: _objective(item["dcr"], item["steel"], cfg))["design"]
            else:
                everything = column_cage_candidates(current, cfg, layout=mg.layout_of(state.policy_id))
                design = max(everything, key=lambda c: c.longitudinal_area_in2) if everything else current
            selected.update({gid: design for gid in node["gids"]})
        report.update(status="no_feasible_chain", exhausted=True, blocked_at=blocked,
                      blocked_by=("no admissible cage of this band meets its own strength and strong-column demands"
                                  if not nodes[blocked_index]["pool"] else
                                  "no cage of this band continues a feasible chain from the band below (transition or the "
                                  "strong-column rule at the boundary joints)"),
                      detail=("no combination of admissible cages along this column line satisfies strength, the "
                              "strong-column rule and a supported transition at every band; each band carries the cage it "
                              "would select on its own demands so the checks measure the shortfall"))
        return selected, report
    _cost, index = min(last, key=lambda item: item[0])
    selected = {}
    for k in range(len(nodes) - 1, -1, -1):
        item = nodes[k]["pool"][index]
        selected.update({gid: item["design"] for gid in nodes[k]["gids"]})
        report["bands"][nodes[k]["key"]].update(dcr=item["dcr"], layers=item["layers"], threads=item["threads"],
                                                rho=item["design"].longitudinal_area_in2 / item["design"].gross_area_in2)
        index = best[k][index][1]
    report.update(status="selected", exhausted=False)
    return selected, report


def mg_column_tag(story, i, j):
    return (story - 1) * (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1) + j * (sp.NUM_BAY_X + 1) + i + 1


# ---- every group ------------------------------------------------------------------------------------------
def select_reinforcement(state, actions, cfg, scwb_margin=1.0, max_passes=4, coordinate=True, reserve=None):
    """Longitudinal cages of every group of ``state`` from the solved actions. Returns (state, report).

    Beams first (their strengths set what the columns must exceed), then each column line; repeated
    until no cage changes, because the layers of a beam depend on the cages it threads and crosses. The
    frame actions do not depend on the bars, so no re-analysis is needed between passes. With ``coordinate``
    a threading-constrained beam deficit is then offered to coordinate_threading (bounded; report["coordination"]).
    ``reserve`` are column group ids whose bands must keep the bars of the beams at their joints threading
    (select_column_chain ``threading_required``): the lanes an adopted coordination opened, which the caller carries
    from one selection to the next (report["reserved_cores"]) so that the next selection does not close them again.
    The state returned has a complete arrangement whose rows fit: a terminal arrangement that does not is closed
    (close_terminal_arrangement), or CageArrangementError carries the incomplete outcome (2026-11-01).
    """
    tied = bool(state.ties())
    report = {"rules": SYSTEM_RULES_ID if tied else RULES_ID, "objective": cfg.dcr.objective, "scwb_margin": scwb_margin,
              "passes": [], "settled": False, "policy": state.policy_id,
              "units": {"beams": [label for label, _ in _beam_units(state)], "columns": [label for label, _ in _column_units(state)]},
              "reserved_cores": sorted(reserve or [])}
    selection_input = state
    state = _selection_passes(state, actions, cfg, scwb_margin, max_passes, report, reserve)
    if coordinate:
        state = coordinate_threading(state, actions, cfg, scwb_margin, max_passes, report, reserve)
    state = close_terminal_arrangement(state, actions, cfg, scwb_margin, report, report.get("reserved_cores"),
                                       inputs={"call": "select_reinforcement", "input_state": selection_input,
                                               "scwb_margin": scwb_margin, "max_passes": max_passes,
                                               "coordinate": coordinate, "reserve": sorted(reserve or [])})
    report["exhausted_groups"] = _exhausted_groups(report)
    return state, report


def _exhausted_groups(report):
    final = report["passes"][-1]
    return sorted([gid for item in final["beams"].values() if item.get("exhausted") for gid in item["groups"]]
                  + [gid for item in final["columns"].values() if item.get("exhausted")
                     for band in item.get("chain", []) for gid in band])


def _beam_sweeps(state, actions, cfg, changes, beams):
    """The beam units selected on the installed columns until none changes (at most BEAM_UNIT_SWEEPS sweeps): a beam's
    layers at a joint depend on the beams it crosses, so a unit selected before another changed is priced again on the
    final crossing beams. ``changes`` and ``beams`` receive the changed designs and the unit reports. Returns (state,
    whether the last sweep changed nothing)."""
    tied = bool(state.ties())
    for _sweep in range(BEAM_UNIT_SWEEPS):
        moved = False
        for label, gids in _beam_units(state):
            design, beams[label] = select_beam_groups(state, gids, actions, cfg)
            wanted = {gid: replace(design, member_type=state.groups[gid]["member_type"]) for gid in gids}
            updates = {gid: d for gid, d in wanted.items() if d != state.designs[gid]}
            if updates:
                changes.update(updates)
                moved = True
                state = state.with_tied_designs({label: design}) if tied else state.with_designs(updates)
        if not moved:
            return state, True
    return state, False


def _selection_passes(state, actions, cfg, scwb_margin, max_passes, report, reserve=None):
    """The selection passes of select_reinforcement from ``state``: each appended to report["passes"], and
    report["settled"] set when a pass changes no cage. ``reserve``: select_reinforcement. A pass is a function of the
    state it starts from, so a pass that ends on a state an earlier pass ended on proves a cycle: it is recorded in
    report["cycle"] and the passes stop there. The input state and every state a pass reaches (after its beams, after
    its columns) are audited on the complete arrangement (_note_geometry). Returns the state."""
    report["settled"] = False
    ends = {}
    columns_now = None                  # the column-unit reports the installed column cages came from (None: the input's)
    _note_geometry(state, report, {"pass": 0, "phase": "input"}, columns_now)
    report["pass_budget"] = {"max_passes": max_passes, "used": 0, "exhausted": False}
    for index in range(max_passes):
        report["pass_budget"]["used"] = index + 1
        changes, beams, columns = {}, {}, {}
        state, _converged = _beam_sweeps(state, actions, cfg, changes, beams)
        _note_geometry(state, report, {"pass": index + 1, "phase": "beams"}, columns_now)
        # The strong-column sums once per pass, on the beams just selected: a column unit that takes a heavier cage
        # changes the lanes of the beams that thread it, and their rows are re-laid only by the next pass's beam
        # selection, so the requirements are not rebuilt between column units.
        try:
            with mg.installed(state):
                requirements = _joint_requirements(state, sp.SLAB_THICKNESS_IN is not None)
        except BeamBarRowsError as exc:
            # The beams as selected do not stack at a joint (a layer past mid-depth): no column is priced on them this
            # pass; the next pass re-selects the beams, and a design that never settles is reported, not solved around.
            report["passes"].append({"pass": index + 1, "changed_groups": sorted(changes), "beams": beams, "columns": {},
                                     "beam_rows_problem": str(exc)})
            continue
        tied = bool(state.ties())
        for label, locations in _column_units(state):
            selected, columns[label] = select_column_chain(state, locations if tied else locations[0], actions, cfg, scwb_margin,
                                                           requirements=requirements,
                                                           threading_required=sorted(reserve) if reserve else False)
            updates = {gid: design for gid, design in selected.items() if design != state.designs[gid]}
            if updates:
                changes.update(updates)
                state = state.with_designs(updates)
        columns_now = columns
        report["columns_in_force"] = _brief_columns(columns_now)
        _note_geometry(state, report, {"pass": index + 1, "phase": "columns"}, columns_now)
        report["passes"].append({"pass": index + 1, "changed_groups": sorted(changes), "beams": beams, "columns": columns})
        if not changes:
            report["settled"] = True
            break
        identity = state.identity()
        if identity in ends:
            report["cycle"] = {"pass": index + 1, "repeats_pass": ends[identity],
                               "detail": "the selection returned to the state an earlier pass ended on: it alternates"}
            break
        ends[identity] = index + 1
    report["pass_budget"]["exhausted"] = not report["settled"] and "cycle" not in report
    return state


# ---- joint-aware coordination of a threading-constrained beam deficit --------------------------------------------
def _constrained_units(report):
    """[(label, unit report)] of the final pass's beam units with an authenticated threading-constrained deficit and
    prospects to try."""
    final = report["passes"][-1]
    return [(label, item) for label, item in sorted(final["beams"].items())
            if (item.get("threading") or {}).get("threading_constrained_deficit") is True
            and item["threading"].get("authenticated") is True and item["threading"].get("prospects")]


def _brief_prospect(prospect):
    return {k: prospect.get(k) for k in ("bar_size", "top_bars", "bot_bars", "dcr", "dcr_union_bound", "steel_in2",
                                         "blocking_cores", "blocking_core_cages", "blocking_transitions")}


LOOKAHEAD_BRIEF_KEYS = ("scope", "core_group", "above_group", "axis", "current_size_in", "current_above_size_in",
                        "lanes_needed_per_layer", "joint_lanes", "core_cage_lanes", "first_size_same_cage",
                        "first_size_any_cage", "transition_partners", "first_upper_band_size", "first_lower_band_size",
                        "first_coupled_size")


def _brief_lookahead(entry):
    return {k: entry[k] for k in LOOKAHEAD_BRIEF_KEYS if k in entry}


def _lookahead_for(lookahead, prospect):
    """The look-ahead entries that bear on what blocks ``prospect``: the core-cage entries of the cores whose own cage
    blocks it and the transition entries of the transitions whose joint blocks it."""
    cages = set(prospect.get("blocking_core_cages") or [])
    pairs = {(t["core_group"], t["above_group"]) for t in prospect.get("blocking_transitions") or []}
    return {key: _brief_lookahead(value) for key, value in sorted(lookahead.items())
            if (value.get("scope") == "core_cage" and value.get("core_group") in cages)
            or (value.get("scope") == "transition_joint" and (value.get("core_group"), value.get("above_group")) in pairs)}


def _content_fingerprint(value):
    """sha256 of the canonical JSON of ``value`` (dataclasses by their fields, sets sorted, a bare sentinel object as a
    marker, a uniform value withdrawn by an installed grouped design by its name: a configuration built while one is
    installed holds those), or None when some part of it has no canonical content form."""
    def encode(item):
        if is_dataclass(item) and not isinstance(item, type):
            return asdict(item)
        if isinstance(item, (set, frozenset)):
            return sorted(item, key=repr)
        if type(item) is object:
            return "<sentinel>"
        if isinstance(item, mg._UniformValueWithdrawn):
            return {"withdrawn_uniform_value": item.name}
        raise TypeError(f"no canonical form for {type(item).__name__}")
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=encode)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _policy_fingerprint():
    """One content fingerprint of the geometry and material policy the selection and the coordination read: every
    public setting of Structure_Parameters (covers, aggregate, the crossing gap, the bar grid and layout hoop, the layer
    cap and order, fy, fyt, the slab and its mats, the frame, ...) and the declared concrete-grade policy. The uniform
    section values a grouped design withdraws are left out: the state identity carries every group's section. None
    when a setting has no canonical content form."""
    from Design import Concrete_Grades as grades
    settings = {name: value for name, value in vars(sp).items()
                if name.isupper() and not name.startswith("_") and not callable(value)
                and not isinstance(value, (types.ModuleType, mg._UniformValueWithdrawn))}
    return _content_fingerprint({"structure_parameters": settings, "concrete_grades": grades.identity_block()})


def _coordination_key(state, report, actions, cfg, scwb_margin, max_passes, reserve):
    """What a coordination outcome depends on, by content: the state the selection ended on (cages and grades) with
    whether it settled and what it exhausted, the actions and the design configuration (content fingerprints, so a
    changed nested setting or action is another key whichever object carries it), the margin, passes and reservation,
    the geometry and material policy (_policy_fingerprint) and the work limits. None when an input has no canonical
    content form: the memo is then neither read nor written."""
    fingerprints = (_content_fingerprint(actions), _content_fingerprint(cfg), _policy_fingerprint())
    if None in fingerprints:
        return None
    return (state.identity(), bool(report["settled"]), tuple(_exhausted_groups(report)), scwb_margin, max_passes,
            tuple(sorted(reserve or ())), *fingerprints,
            (COORDINATION_MAX_UNITS, COORDINATION_MAX_PROSPECTS, COORDINATION_MAX_TRIALS, THREADING_PROSPECTS_MAX,
             BEAM_UNIT_SWEEPS))


def coordinate_threading(state, actions, cfg, scwb_margin, max_passes, report, reserve=None):
    """The joint-aware repair of a threading-constrained beam deficit (COORDINATION_ID; see the module docstring).

    For each such beam unit (at most COORDINATION_MAX_UNITS) its prospects are tried in order (at most
    COORDINATION_MAX_PROSPECTS each, COORDINATION_MAX_TRIALS in all): _coordinated_trial. The first trial that makes
    the beam stronger with a settled selection and no newly exhausted group is adopted: its state is returned, its
    passes join report["passes"] and report["settled"] is its own. Otherwise ``state`` is returned unchanged. Every
    trial is logged in report["coordination"] with its outcome and, when it failed, why; an adoption adds the cores it
    reserved to report["reserved_cores"]. A coordination that adopted nothing is remembered under the content of its
    inputs (_coordination_key) and reported, not run again, for the same inputs; any changed input is a new key, and an
    input without a canonical content form bypasses the memo. Selection only: the cages of the declared domain at the
    installed sizes; column-size alternatives are the evidence's look-ahead, not changed here; nothing is accepted here."""
    log = {"rules": COORDINATION_ID, "status": "not_needed", "trials": [], "adopted": [],
           "limits": {"units": COORDINATION_MAX_UNITS, "prospects_per_unit": COORDINATION_MAX_PROSPECTS,
                      "trials": COORDINATION_MAX_TRIALS, "passes_per_trial": max_passes},
           "scope": ("selection only: cages of the declared domain at the installed sizes; column sizes are proposed by "
                     "the search from the look-ahead, never changed here; an adopted change is evaluated in full after")}
    report["coordination"] = log
    if not report["passes"]:
        return state
    basis = report
    if not report["settled"]:
        # The last pass priced its beams on the columns it started from, which then changed: the beam units that ended
        # short are priced again on the final state, so a trial starts from evidence of the state it changes.
        final = report["passes"][-1]
        fresh = {label: select_beam_groups(state, tuple(item["groups"]), actions, cfg)[1]
                 for label, item in sorted(final["beams"].items()) if item.get("status") == "strength_short"}
        basis = {"passes": [{**final, "beams": {**final["beams"], **fresh}}]}
        log["basis"] = "beam units priced again on the final state of an unsettled selection" + (
            " (a cycle)" if report.get("cycle") else "")
    if not _constrained_units(basis):
        return state
    key = _coordination_key(state, report, actions, cfg, scwb_margin, max_passes, reserve)
    if key is None:
        log["memo"] = "not used: an input has no canonical content form"
    elif key in _COORDINATION_MEMO:
        log.update(copy.deepcopy(_COORDINATION_MEMO[key]), reused=True)
        return state
    visited, attempted, trials = {state.identity()}, [], 0
    while len(attempted) < COORDINATION_MAX_UNITS and not log.get("work_limit_reached"):
        pending = [(label, item) for label, item in _constrained_units(basis) if label not in attempted]
        if not pending:
            break
        label, item = pending[0]
        attempted.append(label)
        before = set(_exhausted_groups(report))
        for prospect in item["threading"]["prospects"][:COORDINATION_MAX_PROSPECTS]:
            if trials >= COORDINATION_MAX_TRIALS:
                log["work_limit_reached"] = True
                break
            trials += 1
            outcome = _coordinated_trial(state, label, item, prospect, actions, cfg, scwb_margin, max_passes, visited, before,
                                         reserve)
            log["trials"].append(outcome["log"])
            if outcome["adopted"]:
                state = outcome["state"]
                report["passes"].extend({**row, "coordination_trial": len(log["trials"])} for row in outcome["report"]["passes"])
                report["geometry"] = list(report.get("geometry") or []) + [
                    {**row, "coordination_trial": len(log["trials"])} for row in outcome["report"].get("geometry") or []]
                if outcome["report"].get("last_geometrically_valid") is not None:
                    report["last_geometrically_valid"] = {**outcome["report"]["last_geometrically_valid"],
                                                          "coordination_trial": len(log["trials"])}
                if outcome["report"].get("columns_in_force") is not None:
                    report["columns_in_force"] = outcome["report"]["columns_in_force"]
                report["settled"] = outcome["report"]["settled"]
                report.pop("cycle", None)
                basis = report
                report["reserved_cores"] = outcome["reserved"]
                reserve = outcome["reserved"]
                log["adopted"].append(len(log["trials"]))
                break
    log["status"] = ("adopted" if log["adopted"] else "work_limit_reached" if log.get("work_limit_reached")
                     else "no_coordinated_change")
    log["work"] = {"trials": trials, "units": attempted}
    if not log["adopted"] and key is not None:
        if len(_COORDINATION_MEMO) >= COORDINATION_MEMO_MAX:
            _COORDINATION_MEMO.clear()
        _COORDINATION_MEMO[key] = copy.deepcopy({**{k: log[k] for k in ("status", "trials", "work") if k in log},
                                                 "work_limit_reached": bool(log.get("work_limit_reached"))})
    return state


def _adjacent_band_cores(state, cores):
    """(adjacent, open): the column groups of the bands directly above and below each band holding one of ``cores``, in
    the same column lines (they meet it at the band-boundary joints), and of those the ones whose cage the bars of every
    beam at their joints thread today within the layer cap (the lanes a coupled trial keeps open)."""
    tied = bool(state.ties())
    adjacent = set()
    for _unit, locations in _column_units(state):
        chain = _column_chain(state, locations if tied else locations[0])
        for index, band in enumerate(chain):
            if set(band) & set(cores):
                for other in (index - 1, index + 1):
                    if 0 <= other < len(chain):
                        adjacent.update(chain[other])
    adjacent -= set(cores)
    open_now = []
    with mg.installed(state):
        for gid in sorted(adjacent):
            # Every joint the group's columns are the core of, with the joint's own lanes (its bar inventory where the
            # column above has another cage).
            joints = {(m.story_or_floor, m.grid_i, m.grid_j) for m in (state.member(t) for t in state.groups[gid]["member_tags"])}
            if all(_lanes_admit(beam.design, bar_layers.bars_per_layer_at_joint(*joint, beam.design, axis))
                   for joint in joints for axis in ("x", "y")
                   for beam in (mg.joint_members(*joint)[f"beam_{axis}_{side}"] for side in ("minus", "plus")) if beam is not None):
                open_now.append(gid)
    return sorted(adjacent), open_now


def _coordinated_trial(state, label, item, prospect, actions, cfg, scwb_margin, max_passes, visited, exhausted_before,
                       reserve=None):
    """One coordinated trial: the prospective cage installed in the beam unit, each column line holding a core that
    blocks it selected again with threading required at those cores' bands (select_column_chain: strength, axial
    domain, strong-column estimate, bar bond, cage rules, transitions), then the whole selection from there with the
    same lanes reserved (otherwise the column pass, which prefers the cages the CURRENT beam bars thread, can close them
    again and the two selections alternate). Returns {"adopted", "log"} and, when adopted, the settled "state", its
    "report" and the "reserved" cores. ``state`` itself is never changed."""
    gids = tuple(item["groups"])
    tied = bool(state.ties())
    entry = {"unit": label, "groups": list(gids), "prospect": _brief_prospect(prospect), "dcr_before": item.get("dcr")}

    def failed(reason, **extra):
        entry.update(status="failed", reason=reason, **extra)
        entry["column_size_lookahead"] = _lookahead_for(item["threading"].get("lookahead") or {}, prospect)
        return {"adopted": False, "log": entry}

    cage = replace(state.designs[gids[0]], bar_size=prospect["bar_size"], top_bars=prospect["top_bars"],
                   bot_bars=prospect["bot_bars"])
    prospective = _trial_state(state, gids[0], cage)
    cores = set(prospect.get("blocking_cores") or [])
    # The bands above and below meet the blocked band at the boundary joints: their open lanes are reserved too.
    adjacent, kept = _adjacent_band_cores(state, cores)
    reserved = sorted(set(reserve or ()) | cores | set(kept))
    entry.update(reserved_cores=reserved, adjacent_band_cores=adjacent, adjacent_lanes_kept=kept,
                 adjacent_not_threading_now=sorted(set(adjacent) - set(kept)))
    lines = [(unit, locations) for unit, locations in _column_units(state)
             if cores & {gid for band in _column_chain(state, locations if tied else locations[0]) for gid in band}]
    if not lines:
        return failed("no column line holds the cores that block the prospective cage")
    updates, entry["column_lines"] = {}, {}
    for unit, locations in lines:
        try:
            selected, chain = select_column_chain(prospective, locations if tied else locations[0], actions, cfg,
                                                  scwb_margin, threading_required=reserved)
        except (BeamBarRowsError, mg.GroupedStateError) as exc:
            return failed(f"the prospective cage cannot be arranged at column line {unit}: {exc}")
        entry["column_lines"][unit] = {**{k: chain.get(k) for k in ("status", "blocked_at", "blocked_by", "cage_pool")},
                                       "bands": {key: {k: band.get(k) for k in ("candidates_admissible", "lanes_available",
                                                                                 "rejected")}
                                                 for key, band in (chain.get("bands") or {}).items()},
                                       "links": chain.get("links")}
        if chain.get("status") != "selected":
            return failed(f"no cage chain of column line {unit} lets the prospective cage thread, keeps the adjacent bands' "
                          "lanes open and keeps strength, the axial domain, the strong-column estimate, bar bond, the cage "
                          "rules and supported transitions")
        updates.update({gid: design for gid, design in selected.items() if design != prospective.designs[gid]})
    trial = prospective.with_designs(updates) if updates else prospective
    if trial.identity() in visited:
        return failed("repeated_state: the coordinated cages repeat a state already tried")
    visited.add(trial.identity())
    sub = {"passes": [], "settled": False}
    try:
        settled = _selection_passes(trial, actions, cfg, scwb_margin, max_passes, sub, reserved)
    except (BeamBarRowsError, mg.GroupedStateError) as exc:
        return failed(f"the selection from the coordinated cages failed: {exc}")
    entry["state_identity"] = settled.identity()
    entry["passes"] = [{"changed_groups": row["changed_groups"][:12]} for row in sub["passes"]]
    if sub.get("cycle"):
        entry["cycle"] = sub["cycle"]
        return failed("the selection from the coordinated cages alternates (a cycle)")
    if not sub["settled"]:
        return failed(f"the selection from the coordinated cages did not settle in {max_passes} passes")
    if settled.identity() != trial.identity() and settled.identity() in visited:
        return failed("repeated_state: the selection from the coordinated cages returned to a state already tried")
    visited.add(settled.identity())
    final = sub["passes"][-1]
    after = (final["beams"].get(label) or {}).get("dcr")
    entry.update(dcr_after=after, installed_after=_cage(settled.designs[gids[0]]),
                 column_lines_after={unit: (final["columns"].get(unit) or {}).get("status") for unit, _ in lines})
    new = sorted(set(_exhausted_groups(sub)) - exhausted_before)
    if new:
        return failed(f"the coordinated change exhausts groups that were not exhausted: {new[:6]}")
    before = entry["dcr_before"]
    if not (isinstance(after, (int, float)) and isinstance(before, (int, float)) and after < before * (1.0 - 1e-9)):
        return failed("the selection from the coordinated cages did not make the beam stronger")
    entry.update(status="adopted", reason=("the beam is stronger, the selection settled and no group is newly exhausted; "
                                           "the capacity design, joint checks and code check of the evaluation follow"))
    return {"adopted": True, "state": settled, "report": sub, "log": entry, "reserved": reserved}


# ---- the terminal cage arrangement (2026-11-01) ---------------------------------------------------------------------
TERMINAL_RULES_ID = "terminal_arrangement_closure_v1"
VERIFICATION_ID = "installed_selection_verification_v1"
TERMINAL_ROUTES = ("beams_on_final_columns", "responsible_columns_restored")
CAGE_FIELDS = ("bar_size", "top_bars", "bot_bars", "side_bars")     # the longitudinal cage a restoration gives back


def _cage_of(record):
    return {k: record[k] for k in CAGE_FIELDS}


def _audit_brief(audit):
    """A row audit without its row controls (what the pass history keeps)."""
    return {"state_identity": audit["state_identity"], "valid": audit["valid"], "settle_error": audit.get("settle_error"),
            "invalid": [{k: item.get(k) for k in ("group", "face", "error", "reason")} for item in audit["invalid"]]}


def _brief_columns(columns):
    """The column-unit reports a restoration carries forward (what the exhausted groups and the record read)."""
    keys = ("location", "status", "exhausted", "chain", "blocked_at", "blocked_by", "cage_pool", "scwb_margin", "restored",
            "restored_from")
    return {label: {k: item[k] for k in keys if k in item} for label, item in (columns or {}).items()}


def _note_geometry(state, report, where, columns):
    """Audit the complete arrangement of ``state`` (Model.Member_Bar_Layers.row_audit), record it in report["geometry"]
    with ``where`` it arose, and keep it as report["last_geometrically_valid"] (its designs, where it arose and the
    column-unit reports its column cages came from, ``columns``; None: the selection's input cages) when its rows fit.
    Returns the audit."""
    audit = bar_layers.row_audit(state)
    report.setdefault("geometry", []).append({**where, **_audit_brief(audit)})
    if audit["valid"]:
        report["last_geometrically_valid"] = {
            "state_identity": audit["state_identity"], "where": dict(where),
            "designs": {gid: mg.design_record(design) for gid, design in sorted(state.designs.items())},
            "columns": None if columns is None else _brief_columns(columns)}
    return audit


def _row_dependencies(state, audit, reference):
    """What the invalid rows of ``audit`` depend on in the complete arrangement of ``state``, per beam group: the joints
    where one of its rows was raised (its row controls) with the core column group there and the column above where that
    has another cage (the joint's bar inventory), the crossing beam groups whose layers raised it, and the cores (and
    transition columns above) whose lanes split its bars and the crossing groups' bars into layers (_lane_governors).
    ``reference`` is report["last_geometrically_valid"] (or None): the column groups among those dependencies whose
    longitudinal cage (CAGE_FIELDS) differs from it are ``responsible`` (a hoop change alone moves no lane a restoration
    could give back, and the installed hoops are kept); ``column_lines`` are the column units holding them and
    ``restore`` the reference's longitudinal cages of those whole lines where they differ."""
    groups, involved = {}, set()
    with mg.installed(state):
        for item in audit["invalid"]:
            gid = item["group"]
            entry = groups.setdefault(gid, {"faces": set(), "joints": set(), "core_groups": set(), "above_groups": set(),
                                            "crossing_groups": set()})
            entry["faces"].add(item["face"])
            for control in item.get("controls") or []:
                if not control:
                    continue
                kind = control.get("joint_type") or {}
                if kind.get("column"):
                    entry["core_groups"].add(kind["column"])
                entry["crossing_groups"].update(kind[axis] for axis in ("x", "y") if kind.get(axis) and kind[axis] != gid)
                if control.get("by") == "crossing_layer" and control.get("group"):
                    entry["crossing_groups"].add(control["group"])
                for joint in control.get("joints") or []:
                    entry["joints"].add(tuple(joint))
                    above = mg.joint_members(*joint)["column_above"]
                    if above is not None and above.group_id != mp.joint_core_column(*joint).group_id:
                        entry["above_groups"].add(above.group_id)
    for gid, entry in groups.items():
        lanes = {}
        for beam in sorted({gid} | entry["crossing_groups"]):
            governors = _lane_governors(state, (beam,), state.designs[beam])[beam]
            lanes[beam] = {"cores": sorted(governors["cores"]),
                           "transitions_above": sorted({e["above_group"] for e in governors["transitions"].values()})}
        entry["lanes"] = lanes
        involved |= entry["core_groups"] | entry["above_groups"]
        involved |= {c for value in lanes.values() for c in value["cores"] + value["transitions_above"]}
    designs = (reference or {}).get("designs") or {}

    def changed(gid):
        return gid in designs and _cage_of(mg.design_record(state.designs[gid])) != _cage_of(designs[gid])

    responsible = sorted(gid for gid in involved if changed(gid))
    hoops_only = sorted(gid for gid in involved if gid in designs and not changed(gid)
                        and mg.design_record(state.designs[gid]) != designs[gid])
    tied = bool(state.ties())
    lines, restore = [], {}
    for unit, locations in _column_units(state):
        chain = sorted(g for band in _column_chain(state, locations if tied else locations[0]) for g in band)
        if set(chain) & set(responsible):
            lines.append(unit)
            restore.update({g: _cage_of(designs[g]) for g in chain if changed(g)})
    return {"groups": {gid: {"faces": sorted(e["faces"]), "joints": [list(j) for j in sorted(e["joints"])[:12]],
                             "joint_count": len(e["joints"]), "core_groups": sorted(e["core_groups"]),
                             "above_groups": sorted(e["above_groups"]), "crossing_groups": sorted(e["crossing_groups"]),
                             "lanes": e["lanes"]} for gid, e in sorted(groups.items())},
            "column_groups": sorted(involved), "responsible_column_groups": responsible,
            "column_groups_with_only_other_changes": hoops_only, "column_lines": lines,
            "restore": restore, "reference_state_identity": (reference or {}).get("state_identity")}


def _column_rules(state, actions, cfg, scwb_margin, reserve=None):
    """Evidence beside an adopted closure: the installed column cages against the column selection's own rules on the
    final beams (select_column_chain with each band's installed cage only: strength, axial domain, bar bond, the
    strong-column estimate on the final beams within and between bands, supported transitions). Not a gate: the exact
    joint evaluation of the capacity design remains the authority, and the selection stays unsettled."""
    tied = bool(state.ties())
    try:
        with mg.installed(state):
            requirements = _joint_requirements(state, sp.SLAB_THICKNESS_IN is not None)
    except BeamBarRowsError as exc:
        return {"status": "not_evaluated", "reason": str(exc)}
    result = {}
    for label, locations in _column_units(state):
        _selected, chain = select_column_chain(state, locations if tied else locations[0], actions, cfg, scwb_margin,
                                               requirements=requirements, installed_only=True,
                                               threading_required=sorted(reserve) if reserve else False)
        result[label] = {"status": chain.get("status"), "blocked_at": chain.get("blocked_at"),
                         "blocked_by": chain.get("blocked_by"),
                         "rejected": {key: band.get("rejected") for key, band in (chain.get("bands") or {}).items()},
                         "links": {key: {k: v for k, v in counts.items() if k != "examples"}
                                   for key, counts in (chain.get("links") or {}).items()}}
    return result


def verify_installed_selection(state, actions, cfg, scwb_margin=1.0, reserve=None):
    """The installed cages of ``state`` against the selection's own rules on the installed design, selecting nothing
    (VERIFICATION_ID; the module docstring). In order: the complete arrangement fits (Model.Member_Bar_Layers
    .row_audit); every beam unit's installed cage is admissible in every group's place, threads its column cages and is
    within dcr_hard_max (select_beam_groups installed_only: status selected and threading); every column line's
    installed chain meets the column rules on the strong-column sums of the installed beams (select_column_chain
    installed_only: status selected; ``reserve`` as select_reinforcement). The unit reports have the shape of a
    selection pass, so verified_selection_report can describe a state adopted on this verification. Threading is
    required of every beam unit: stricter than the selector, which falls back to a cage that does not thread when none
    does (a construction check of the capacity design, beam.bars_thread_column, would fail on it anyway).

    Returns {"rules", "state_identity", "verified", "rows_fit", "invalid_rows", "failing", "beams", "columns"};
    ``failing`` names each rule that is not met (rows, rows_unsettled or rows_malformed for the arrangement, a beam
    unit and its status, a column line and its status); an arrangement whose layers do not settle keeps its
    ``settle_error``."""
    with mg.installed(state):
        audit = bar_layers.row_audit(state)
    result = {"rules": VERIFICATION_ID, "state_identity": state.identity(), "verified": False,
              "rows_fit": bool(audit["valid"]),
              "invalid_rows": [{k: item.get(k) for k in ("group", "face", "error")} for item in audit["invalid"]],
              "failing": [], "beams": {}, "columns": {}}
    if not audit["valid"]:
        if audit.get("settle_error"):
            result.update(failing=["rows_unsettled"], settle_error=audit["settle_error"])
        elif any(item.get("error") == "rows_invalid" for item in audit["invalid"]):
            result["failing"].append("rows_malformed")
        else:
            result["failing"].append("rows")
        return result
    for label, gids in _beam_units(state):
        _design, unit = select_beam_groups(state, gids, actions, cfg, installed_only=True)
        result["beams"][label] = unit
        if unit.get("status") != "selected":
            result["failing"].append(f"beam:{label}:{unit.get('status')}")
        elif unit.get("threads") is not True:
            result["failing"].append(f"beam:{label}:not_threading")
    try:
        with mg.installed(state):
            requirements = _joint_requirements(state, sp.SLAB_THICKNESS_IN is not None)
    except BeamBarRowsError as exc:
        result["failing"].append("strong_column_sums")
        result["reason"] = str(exc)
        return result
    tied = bool(state.ties())
    for label, locations in _column_units(state):
        _selected, chain = select_column_chain(state, locations if tied else locations[0], actions, cfg, scwb_margin,
                                               requirements=requirements, installed_only=True,
                                               threading_required=sorted(reserve) if reserve else False)
        result["columns"][label] = chain
        if chain.get("status") != "selected":
            result["failing"].append(f"column:{label}:{chain.get('status')}")
    result["verified"] = not result["failing"]
    return result


def verified_selection_report(previous, verification, reserve=None):
    """The selection record of a state adopted on verify_installed_selection: its one pass is the verification (the
    installed cages only, nothing changed), its terminal arrangement the verified one, and no coordination ran on it.
    ``settled`` stays False: the selection did not converge to this state, its rules admit it."""
    report = {"rules": previous.get("rules"), "objective": previous.get("objective"),
              "scwb_margin": previous.get("scwb_margin"), "policy": previous.get("policy"), "units": previous.get("units"),
              "reserved_cores": sorted(reserve or []), "settled": False,
              "passes": [{"pass": "verification", "changed_groups": [], "beams": verification["beams"],
                          "columns": verification["columns"]}],
              "terminal_arrangements": [{"rules": TERMINAL_RULES_ID, "context": "installed_selection_verification",
                                         "status": "valid", "final_state_identity": verification["state_identity"]}],
              "verification": {k: verification[k] for k in ("rules", "state_identity", "verified", "failing")}}
    report["exhausted_groups"] = _exhausted_groups(report)
    return report


def close_terminal_arrangement(state, actions, cfg, scwb_margin, report, reserve=None, inputs=None):
    """The terminal arrangement of a selection, closed (TERMINAL_RULES_ID; the module docstring).

    Returns ``state`` when its complete arrangement fits. Otherwise the routes of TERMINAL_ROUTES are tried in order and
    the state of the first one whose complete arrangement fits is returned: its beam reports are appended to
    report["passes"] as a closure row (with the column-unit reports in force), report["settled"] is False and the column
    selection's own rules on the final beams are recorded beside it (_column_rules, evidence only). When no route closes
    it, CageArrangementError carries the incomplete outcome (_incomplete_outcome). Every closure is logged in
    report["terminal_arrangements"]. ``inputs``: what replays the call that produced ``state`` (its input state, margin,
    passes, reservation), kept with an incomplete outcome. ``reserve`` is recorded, and the column rules beside an
    adopted route read it (threading_required); no route changes a reserved core's cage except by restoring its whole
    line's longitudinal cage to the last geometrically valid state's, with the installed hoops kept. An adopted route
    becomes the column reports in force (report["columns_in_force"]), so a later closure on the same report starts from
    it. An arrangement that does not settle, or a malformed row record, is raised
    (Model.Member_Bar_Layers.raise_arrangement_errors), not closed."""
    audit = bar_layers.row_audit(state)
    bar_layers.raise_arrangement_errors(audit)
    log = {"rules": TERMINAL_RULES_ID, "status": "valid", "context": (inputs or {}).get("call", "select_reinforcement"),
           "initial": _audit_brief(audit), "routes": [], "reserved_cores": sorted(reserve or []),
           "limits": {"routes": list(TERMINAL_ROUTES), "beam_unit_sweeps": BEAM_UNIT_SWEEPS}}
    report.setdefault("terminal_arrangements", []).append(log)
    if audit["valid"]:
        return state
    reference = report.get("last_geometrically_valid")
    dependencies = _row_dependencies(state, audit, reference)
    log.update(status="closing", dependencies={k: v for k, v in dependencies.items() if k != "restore"},
               reference=None if reference is None else {k: reference.get(k) for k in ("state_identity", "where")})
    # The column-unit reports in force (the selection's last column phase; the last pass row of a pass whose beams did
    # not stack carries none).
    in_force = dict(report.get("columns_in_force") or (report["passes"][-1].get("columns") if report["passes"] else None)
                    or {})
    starts = [(TERMINAL_ROUTES[0], state, {})]
    if dependencies["restore"]:
        restored = {gid: replace(state.designs[gid], **cage) for gid, cage in sorted(dependencies["restore"].items())}
        starts.append((TERMINAL_ROUTES[1], state.with_designs(restored), restored))
    else:
        log["routes"].append({"route": TERMINAL_ROUTES[1], "status": "not_available",
                              "reason": ("no geometrically valid state was reached in this selection" if reference is None
                                         else "no longitudinal column cage the invalid rows depend on changed since the "
                                              "last geometrically valid state"
                                              + (" (only hoops or other non-cage fields changed: the installed hoops are "
                                                 "kept)" if dependencies["column_groups_with_only_other_changes"] else ""))})
    seen = {state.identity()}
    for route, start, restored in starts:
        entry = {"route": route, "start_identity": start.identity()}
        if restored:
            entry.update(restored_groups=sorted(restored), column_lines=list(dependencies["column_lines"]))
            if start.identity() in seen:
                log["routes"].append({**entry, "status": "failed",
                                      "reason": "repeated_state: the restored cages repeat a state already tried"})
                continue
            seen.add(start.identity())
        changes, beams = {}, {}
        try:
            trial, converged = _beam_sweeps(start, actions, cfg, changes, beams)
        except (BeamBarRowsError, mg.GroupedStateError) as exc:
            log["routes"].append({**entry, "status": "failed", "reason": f"the beam units could not be priced: {exc}"})
            continue
        no_fit = sorted(label for label, item in beams.items() if item.get("status") == "no_admissible_cage")
        entry.update(changed_groups=sorted(changes), beam_units_converged=converged,
                     beam_status={label: item.get("status") for label, item in sorted(beams.items())},
                     units_without_admissible_cage=no_fit,
                     refusals={label: (beams[label].get("threading") or {}).get("first_refusals") for label in no_fit},
                     end_identity=trial.identity())
        if trial.identity() != start.identity() and trial.identity() in seen:
            log["routes"].append({**entry, "status": "failed",
                                  "reason": "repeated_state: the beam units returned to a state already tried"})
            continue
        seen.add(trial.identity())
        check = bar_layers.row_audit(trial)
        entry["audit"] = _audit_brief(check)
        if not converged or not check["valid"]:
            # A beam step whose units still moved in its last sweep priced some unit on crossing beams that changed
            # after it: its reports do not describe the final state, so it is never adopted, even with rows that fit.
            reason = ("the beam units did not stop changing within the sweep limit" if not converged else
                      "a beam unit has no admissible cage in its place on these columns" if no_fit else
                      f"the trial's arrangement does not settle: {check['settle_error']}" if check.get("settle_error") else
                      "the complete arrangement still leaves rows that do not fit")
            log["routes"].append({**entry, "status": "failed", "reason": reason})
            continue
        rules = _column_rules(trial, actions, cfg, scwb_margin, reserve)
        columns = dict(in_force)
        if restored:
            source = (reference or {}).get("columns") or {}
            for unit in dependencies["column_lines"]:
                before, given = in_force.get(unit) or {}, source.get(unit) or {}
                # A restored line keeps the exhaustion its selection found (in force or where its cages came from): the
                # search's growth signal is not lost; whether the restored chain meets the column rules on the final
                # beams is recorded beside it.
                columns[unit] = {"location": unit, "status": "restored", "restored": True,
                                 "exhausted": bool(before.get("exhausted") or given.get("exhausted")),
                                 "chain": before.get("chain") or given.get("chain") or [],
                                 **{k: (before.get(k) or given.get(k)) for k in ("blocked_at", "blocked_by")
                                    if before.get(k) or given.get(k)},
                                 "restored_from": (reference or {}).get("where"),
                                 "installed_chain_on_final_beams": (rules.get(unit) or {}).get("status")
                                 if isinstance(rules, dict) else None}
        entry.update(status="adopted", column_rules_on_final_state=rules)
        log["routes"].append(entry)
        log.update(status="closed", adopted_route=route, rejected_state_identity=state.identity(),
                   final_state_identity=trial.identity())
        report["passes"].append({"pass": len(report["passes"]) + 1, "closure_route": route,
                                 "changed_groups": sorted(set(changes) | set(restored)), "beams": beams, "columns": columns})
        report["settled"] = False
        report["columns_in_force"] = _brief_columns(columns)
        report["exhausted_groups"] = _exhausted_groups(report)
        _note_geometry(trial, report, {"pass": len(report["passes"]), "phase": "closure"}, columns)
        return trial
    log["status"] = "incomplete"
    raise _incomplete_outcome(state, audit, report, dependencies, reference, log, inputs)


def _incomplete_outcome(state, audit, report, dependencies, reference, log, inputs):
    """CageArrangementError with the structured incomplete outcome of close_terminal_arrangement: the classification, the
    rejected terminal state and the last geometrically valid state apart (neither carries a strength, capacity or
    acceptance result), the invalid rows with their row controls, the dependencies, every route, the pass history with
    each state's audit, and what replays the call (``detail`` holds the objects)."""
    failed = [r for r in log["routes"] if r.get("status") == "failed"]
    if any(r.get("units_without_admissible_cage") for r in failed if r.get("beam_units_converged") is not False):
        classification = "no_fitting_beam_cage"
    elif any(r.get("beam_units_converged") is False for r in failed):
        classification = "beam_units_alternate"
    elif any(str(r.get("reason", "")).startswith("repeated_state") for r in failed):
        classification = "repeated_state"
    else:
        classification = "rows_still_invalid"
    log["classification"] = classification
    groups = sorted({item["group"] for item in audit["invalid"]})
    reason = (f"the final shared cage arrangement leaves beam bar rows that do not fit ({', '.join(groups[:6])}) and the "
              f"bounded closure did not repair them ({classification}): an incomplete coordination outcome within these "
              "bounds, not proof that no cage of the declared domain fits")
    inputs = dict(inputs or {})
    input_state = inputs.pop("input_state", None)
    layout = (sp.SLAB_REINFORCEMENT or {}).get("layout") if sp.SLAB_THICKNESS_IN is not None else None
    coordination = report.get("coordination") or {}
    evidence = {
        "outcome": "terminal_arrangement_incomplete", "rules": TERMINAL_RULES_ID, "classification": classification,
        "reason": reason, "group_ids": groups,
        "scope": ("bounded closure: every beam unit selected again on the final columns, then the responsible column lines "
                  "restored to their last geometrically valid cages with the beam units selected again on them; section "
                  "sizes, grades and the declared cage domain are not changed here"),
        "rejected_state": {"state_identity": state.identity(), "invalid_rows": audit["invalid"],
                           "role": ("the terminal arrangement the selection ended on, rejected: no strength, capacity or "
                                    "acceptance result belongs to it")},
        "last_geometrically_valid_state": (None if reference is None else {
            "state_identity": reference.get("state_identity"), "where": reference.get("where"),
            "role": ("the last state of this selection whose complete arrangement fits; not evaluated here: no strength, "
                     "capacity or acceptance result belongs to it either")}),
        "dependencies": {k: v for k, v in dependencies.items() if k != "restore"},
        "routes": [{k: v for k, v in r.items() if k != "column_rules_on_final_state"} for r in log["routes"]],
        "pass_history": [{"pass": row.get("pass"), "changed_groups": list(row.get("changed_groups") or [])[:24],
                          "beams": {label: item.get("status") for label, item in sorted((row.get("beams") or {}).items())},
                          "columns": {label: item.get("status") for label, item in sorted((row.get("columns") or {}).items())},
                          **{k: row[k] for k in ("beam_rows_problem", "coordination_trial", "closure_route") if k in row}}
                         for row in report["passes"]],
        "geometry": list(report.get("geometry") or []),
        "settled": bool(report.get("settled")), "cycle": report.get("cycle"), "pass_budget": report.get("pass_budget"),
        "coordination": {k: coordination.get(k) for k in ("rules", "status", "adopted", "work", "work_limit_reached")},
        "inputs": {**inputs, "selection_input_identity": None if input_state is None else input_state.identity(),
                   "reserved_cores": list(report.get("reserved_cores") or []), "slab_thickness_in": sp.SLAB_THICKNESS_IN,
                   "slab_layout_present": layout is not None,
                   "slab_layout_sha256": (None if layout is None else hashlib.sha256(
                       json.dumps(layout, sort_keys=True, default=str).encode("utf-8")).hexdigest())},
        "call_site": f"Design.Group_Selection.close_terminal_arrangement ({log['context']})"}
    detail = {"rejected_state": state, "selection_input_state": input_state, "selection_report": report,
              "last_geometrically_valid_designs": None if reference is None else reference.get("designs")}
    return CageArrangementError(reason, evidence, detail)


def terminal_arrangement_summary(report):
    """The terminal-arrangement closures of a selection report, as the evaluation record keeps them
    (constraints.terminal_arrangement)."""
    keys = ("rules", "context", "status", "adopted_route", "rejected_state_identity", "final_state_identity", "classification")
    return [{**{k: log.get(k) for k in keys}, "initial_invalid": (log.get("initial") or {}).get("invalid"),
             "routes": [{k: r.get(k) for k in ("route", "status", "reason", "changed_groups", "restored_groups",
                                                 "units_without_admissible_cage", "column_rules_on_final_state")}
                        for r in log.get("routes") or []]}
            for log in report.get("terminal_arrangements") or []]


# ---- what an evaluation record keeps ------------------------------------------------------------------------------
def _brief(entry):
    if not entry:
        return None
    return {**{k: entry.get(k) for k in ("bar_size", "top_bars", "bot_bars", "dcr", "dcr_union_bound", "steel_in2",
                                         "threads", "layers")},
            "by_group": {gid: {k: (value or {}).get(k) for k in ("dcr", "dcr_union_bound", "threads", "layers")}
                         for gid, value in (entry.get("by_group") or {}).items()}}


def installed_lane_evidence(state, cfg):
    """The lanes of each beam group's installed cage at the joints its bars pass, for the search diagnostics
    (INSTALLED_LANES_ID; the evaluation record's constraints.beam_lane_evidence): every group whose installed cage does
    not thread every one of those joints within the layer cap, whatever the selector reported for it. Per such group:
    its cage, its lanes (the core cages and the transition joints by both column groups, _lane_governors), what blocks
    it (_blocking) and the look-ahead by scope (_lane_lookahead: core_cage and transition_joint), all for that cage.
    Bound to ``state`` by its identity. Geometry only: nothing is selected, admitted or accepted here."""
    result = {"basis": INSTALLED_LANES_ID, "state_identity": state.identity(), "groups": {}}
    for gid in sorted(g for g, group in state.groups.items() if group["member_type"] != "column"):
        design = state.designs[gid]
        lanes = _lane_governors(state, (gid,), design)
        if lanes[gid]["threads"]:
            continue
        cores, cages, transitions = _blocking(lanes, (gid,), design)
        result["groups"][gid] = {
            **_cage(design), "axis": lanes[gid]["axis"], "lanes": lanes[gid], "blocking_cores": cores,
            "blocking_core_cages": cages, "blocking_transitions": transitions,
            "lookahead": {key: _brief_lookahead(value) for key, value in _lane_lookahead(state, (gid,), design, cfg).items()}}
    return result


def threading_constraint_summary(report):
    """{beam group id: compact threading evidence} of the final pass's beam units that ended short of strength: what the
    evaluation record keeps (constraints.beam_threading_evidence) for the search diagnostics."""
    final = (report.get("passes") or [{}])[-1]
    result = {}
    for label, item in sorted((final.get("beams") or {}).items()):
        evidence = item.get("threading")
        if item.get("status") != "strength_short" or not isinstance(evidence, dict):
            continue
        lanes = evidence.get("lanes") or {}
        for gid in item.get("groups") or []:
            result[gid] = {
                "unit": label, "basis": evidence.get("basis"), "authenticated": evidence.get("authenticated") is True,
                "lane_limited": bool(item.get("lane_limited")),
                "constrains_strength": evidence.get("constrains_strength") is True,
                "deficit_remains": evidence.get("deficit_remains") is True,
                "threading_constrained_deficit": evidence.get("threading_constrained_deficit") is True,
                "installed": _brief(evidence.get("installed")), "best_threading": _brief(evidence.get("best_threading")),
                "best_nonthreading": _brief(evidence.get("best_nonthreading")),
                "blocking_cores": list(evidence.get("blocking_cores") or []),
                "blocking_core_cages": list(evidence.get("blocking_core_cages") or []),
                "blocking_transitions": list(evidence.get("blocking_transitions") or []),
                "core_groups": list(evidence.get("core_groups") or []),
                "first_refusals": (evidence.get("first_refusals") or {}).get(gid, {}),
                "refusal_classes": (evidence.get("refusal_classes") or {}).get(gid, {}),
                "nonthreading_by_cause": evidence.get("nonthreading_by_cause") or {},
                "lanes": {name: (value or {}).get(gid) for name, value in lanes.items()},
                "rows": (evidence.get("rows") or {}).get(gid),
                "lookahead": {key: _brief_lookahead(value) for key, value in (evidence.get("lookahead") or {}).items()
                              if value.get("beam_group") == gid},
                "policy": evidence.get("policy")}
    return result


def coordination_summary(report):
    """The coordination log of a selection report, as the evaluation record keeps it (constraints.cage_coordination)."""
    log = report.get("coordination")
    if not isinstance(log, dict):
        return None
    keys = ("unit", "groups", "prospect", "status", "reason", "dcr_before", "dcr_after", "installed_after", "column_lines",
            "column_lines_after", "column_size_lookahead", "reserved_cores", "adjacent_band_cores", "adjacent_lanes_kept",
            "adjacent_not_threading_now", "passes", "cycle")
    return {"rules": log.get("rules"), "status": log.get("status"), "reason": log.get("reason"), "basis": log.get("basis"),
            "cycle": report.get("cycle"),
            "reused": bool(log.get("reused")), "reserved_cores": list(report.get("reserved_cores") or []),
            "adopted": list(log.get("adopted") or []), "work": log.get("work"),
            "work_limit_reached": bool(log.get("work_limit_reached")), "limits": log.get("limits"),
            "trials": [{k: trial.get(k) for k in keys} for trial in log.get("trials") or []]}
