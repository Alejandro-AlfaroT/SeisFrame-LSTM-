"""The authoritative code-check result of one grouped design candidate and the contract its evaluation must meet
(code-check audit of the genetic search, 2026-10-30).

Three results are kept apart and none stands in for another:

  preliminary screen   Grouped_Design's ``candidate_screen_passed``: no recorded failure in the families it reads
                       and a settled reinforcement pass. It names no failure; it is not evidence that a check passed.
  code check           this module: every check the final design requires, derived from that design's own topology
                       and system assignment (never from the list under review), present exactly once, well formed,
                       and passing. Known failures stay visible beside unknown checks; an unknown check is neither a
                       failure nor a pass. Review-dependent items (engineering assessments the code does not make) are
                       listed apart: they neither fail nor pass the computed checks and are never asserted here.
  qualification        Design.Grouped_Record.qualify_grouped_design on the saved record (integrity, recomputation and
                       the reviews), the gate before any time-history analysis. GENERATION_RELEASE_READY is untouched.

The candidate scope (SCOPE) is the ACI 318-19 special-moment-frame strength and detailing checks the evaluator
implements, with the supporting ASCE 7-22 drift, stability, vertical-regularity and torsional-amplification checks;
its exclusions are named. Nonlinear response is not part of it: its status is reported as not requested, never as
a failed code check.

``expected_inventory`` reads only the realized design (member groups, joints, systems, story count) and the
declared configuration. ``code_check_result`` classifies the supplied rows against it. ``assess_evaluation`` is the
boundary a search applies to an evaluation record before ranking or archiving it: schema, completion flags,
identities, the binding of the rows to the final design, and an independent recomputation of the result.

Contract v2 (2026-11-01 closure review): a proven reinforcement cycle resolved on verification
(Grouped_Design.resolve_cycle: the adopted state meets the selection's own rules, its hoops are a fixed point and its
capacity design is accepted) completes reinforcement_settled and longitudinal_selection_settled for that final state
only (numerical_completion). Both flags stay false in the record, so no convergence is claimed; every other flag, and
the code check itself, is unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import copy

from Design.SMRF_Common import SCOPE_EXCLUDED_ROWS, SCOPE_EXCLUSIONS, check_row_problems
from Design import SMRF_Joints as joint_rules
from Design.SMRF_Joints import SCWB_EXCEPTION_CLAUSE

CONTRACT_VERSION = "grouped_code_check_contract_v7_frame_transfer_conventional"
SCOPE = {
    "id": "aci318_19_smrf_candidate_code_checks_v2",
    "basis": ["ACI 318-19 Chapter 18 special moment frames and the chapters it references (as implemented)",
              "ASCE 7-22 12.8.6-12.8.7 drift and stability, Table 12.3-2 vertical regularity, 12.8.4.3 torsional "
              "amplification"],
    "families": {
        "strength": "member strength of every group at its governing member and combination (DCR <= 1.0)",
        "capacity": "capacity-design shear, hoops, cages, column bond, transitions, joint shear screens, anchorage, "
                    "threading, bar stacking, frame-beam axial applicability (18.6.4.7), splice type and location "
                    "(18.2.7.2), gravity-frame members (18.14) under a system policy",
        "joints": "strong-column / weak-beam at every required joint, direction and sway sign (18.7.3.2; at a roof "
                  "connection its 18.7.3.1 exception, recorded with Pu against the column's own Ag f'c / 10), beam "
                  "capacity shear, through-bar depth and joint shear (SMRF_Joints)",
        "detailing": "18.6 / 18.7 dimensional and bar rules per group, beam-to-joint pairs, column confinement area",
        "drift": "ASCE 7-22 story drift, stability and node coverage, every story and direction",
        "regularity": "ASCE 7-22 Table 12.3-2 stiffness and strength (story strength rests on review item M1)",
        "torsion": "ASCE 7-22 12.8.4.3 amplification applied against the Type 1 classification",
        "slab": ("the slab strip and completion checks of the selected layout, its development and currency; the "
                 "two-way shear path and the slab-column local steel (8.6.1.2), each resolved as passing or assessed "
                 "not applicable"),
        "reinforcement": "an admissible cage for every group",
    },
    "excluded": {
        "diaphragm": ("diaphragm, collector and chord design (ACI 318-19 18.12, ASCE 7-22 12.10); the frame model's "
                      "rigid diaphragm also removes diaphragm forces from the beam axial envelope"),
        "foundation": "foundations and the fixed-base idealization",
        "demand_declarations": ("the declared demand policy (risk category, site, live-load and redundancy "
                                "declarations) is checked on the saved record (SMRF_Demands.evaluate_demand_basis)"),
        "record_integrity": "saved-record integrity and recomputation: Grouped_Record.qualify_grouped_design",
        "nonlinear_performance": "nonlinear response-history performance: optional, not a code check",
        "fire_resistance": SCOPE_EXCLUSIONS["fire_resistance"],
    },
}
COMPLETION_FLAGS = ("reinforcement_settled", "longitudinal_selection_settled", "hoops_settled", "slab_layout_settled",
                    "slab_layout_kept", "slab_evidence_current", "capacity_evidence_current")
ROW_STATES = ("pass", "fail", "unknown", "review", "invalid", "not_applicable")
DETAIL_KEYS = ("reason", "reason_code", "reason_codes", "review_dependency", "input_problems", "numerical_tolerance_in",
               "numerical_tolerance", "ratio_provided", "ratio_required", "branch", "category", "scope_exclusion",
               "roof_exemption_applied", "exception")
# The flags a verified cycle resolution completes for its own final state (numerical_completion).
SELECTION_FLAGS = ("reinforcement_settled", "longitudinal_selection_settled")
VERIFIED_CYCLE_RESOLUTION = "verified_cycle_resolution"
SLAB_COMPLETION_REQUIRED = (("slab_strip_cover", ""), ("slab_strip_face_mat_clearance", ""), ("slab_bar_development", ""),
                            ("slab_perimeter_bar_anchorage", "x"), ("slab_perimeter_bar_anchorage", "y"),
                            ("slab_continuity_and_extensions", ""), ("slab_deflection_control", ""),
                            ("slab_integrity_bottom_bars_through_column", ""),
                            # Structural items, each resolved on its own as passing or assessed not applicable (2026-11-01).
                            ("slab_two_way_shear_applicability", ""), ("slab_column_local_minimum_steel", ""))
COMPUTED_SLAB_COMPLETION_REQUIRED = tuple(item for item in SLAB_COMPLETION_REQUIRED
    if item[0] not in ('slab_two_way_shear_applicability','slab_column_local_minimum_steel')) + (
    ('slab_computed_physical_mat_inventory',''),) + tuple((name,layer)
    for layer in ('x_top','x_bottom','y_top','y_bottom') for name in (
        'slab_computed_layout_definition','slab_strip_minimum_flexural_steel','slab_strip_maximum_spacing',
        'slab_strip_horizontal_clear_spacing','slab_crack_control_spacing','slab_computed_corner_reinforcement'))
# The rows of a declared scope exclusion (SMRF_Common.SCOPE_EXCLUSIONS): skipped like a declared not-applicable row when
# not_evaluated and labelled with their item; invalid when they pass, fail or wait on a review.
EXCLUDED_ROWS = {("slab", row_id, ""): item for row_id, item in sorted(SCOPE_EXCLUDED_ROWS.items())}
NONLINEAR_STATUS = {"status": "not_requested",
                    "basis": "nonlinear response-history performance is outside the code-check scope; its absence is "
                             "not a failed code check"}


def scope_for_common_actions(required):
    """The diaphragm leaves the old exclusion only with its mandatory computed rows."""
    scope = copy.deepcopy(SCOPE)
    from Design import SMRF_Conventional_Detail as conventional
    if required and conventional.active():
        scope["id"] = "aci318_19_smrf_candidate_code_checks_v4_frame_transfer"
        scope["excluded"].pop("diaphragm", None)
        scope["families"]["floor"] = ("floor load balance, rigid diaphragm idealization (ASCE 7-22 12.3.1), diaphragm in-plane shear "
                                       "and flexure, shear transfer and collector axial force (ACI 318-19 18.12, ASCE 7-22 12.10)")
    elif required:
        scope["id"] = "aci318_19_smrf_candidate_code_checks_v3_common_actions"
        scope["excluded"].pop("diaphragm", None)
        scope["families"]["common_actions"] = (
            "Actual signed slab/beam/diaphragm actions, complete cases, shared steel and development, "
            "concurrent torsion, diaphragm collectors/chords, and each of the seven method dispositions; "
            "computed missing evidence blocks acceptance independently of any profile review")
    return scope


def numerical_completion(constraints, state_identity):
    """(flags, problems, complete, resolved, malformed) of an evaluation's completion flags. ``flags`` holds each flag
    that is an explicit boolean; ``problems`` names each false flag that is not met; complete needs every flag present
    and true, except that the SELECTION_FLAGS are met by a verified cycle resolution of this same final state
    (``resolved``): the passes stopped on a proven cycle (reinforcement_stop "cycle"), the cycle's own resolution record
    adopted ``state_identity`` (reinforcement_cycle.resolution.adopted_state_identity), the tag
    constraints["reinforcement_resolution"] is of kind VERIFIED_CYCLE_RESOLUTION and names that same identity, and the
    hoops are settled. ``state_identity`` must be a non-empty identity. ``malformed`` maps each flag that is not an
    explicit boolean to its value."""
    resolution = constraints.get("reinforcement_resolution")
    cycle = constraints.get("reinforcement_cycle")
    adopted = ((cycle.get("resolution") if isinstance(cycle, dict) else None) or {}).get("adopted_state_identity")
    resolved = (isinstance(state_identity, str) and bool(state_identity) and isinstance(resolution, dict)
                and resolution.get("kind") == VERIFIED_CYCLE_RESOLUTION and resolution.get("state_identity") == state_identity
                and constraints.get("reinforcement_stop") == "cycle" and adopted == state_identity
                and constraints.get("hoops_settled") is True)
    flags, problems, malformed = {}, [], {}
    for flag in COMPLETION_FLAGS:
        value = constraints.get(flag)
        if type(value) is not bool:
            malformed[flag] = value
            continue
        flags[flag] = value
        if not value and not (resolved and flag in SELECTION_FLAGS):
            problems.append(f"{flag} is false")
    complete = not problems and not malformed and len(flags) == len(COMPLETION_FLAGS)
    return flags, problems, complete, resolved, malformed


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def realized_identity(state_identity, slab_thickness_in=None, slab_record=None):
    """The identity of the realized design the rows describe: member groups with their final cages and hoops, the
    slab thickness and the selected slab layout with the inputs and floor context it was priced on. Changing any
    slab bar, cage, hoop or the slab changes it."""
    inputs = (slab_record or {}).get("inputs") if isinstance(slab_record, dict) else None
    return _sha({"member_groups": state_identity, "slab_thickness_in": slab_thickness_in,
                 "slab_layout": (slab_record or {}).get("layout") if isinstance(slab_record, dict) else None,
                 "slab_inputs": (inputs or {}).get("slab"), "slab_context": (inputs or {}).get("context")})


def compact_row(family, check):
    """One check row as the contract keeps it: identity, status, numbers and the reason codes, with its family."""
    if not isinstance(check, dict):
        return {"family": family, "id": None, "location": "", "status": None, "row_problems": ["not_a_record"]}
    details = check.get("details") if isinstance(check.get("details"), dict) else {}
    kept = {key: details[key] for key in DETAIL_KEYS if key in details}
    if isinstance(details.get("upstream_reasons"), list):
        kept["upstream_reasons"] = details["upstream_reasons"][:6]
    return {"family": family, "id": check.get("id"), "clause": check.get("clause"), "location": check.get("location", ""),
            "status": check.get("status"), "demand": check.get("demand"), "capacity": check.get("capacity"),
            "comparison": check.get("comparison"), "units": check.get("units", ""), "details": kept}


def scwb_summary(checks, required=None):
    """The strong-column / weak-beam rows summarized without vacuous truth.

    ``all_pass`` is True only when the rows are complete (every required (location) present exactly once when
    ``required`` is given, at least one row otherwise) and every row passes; ``no_known_failure`` is the old,
    weaker statement (no row failed) and is named as such. Unknown rows are counted, never read as passes or
    failures, and no ratio is made up for them.
    """
    rows = [c for c in checks if isinstance(c, dict) and c.get("id") == "scwb"]
    counts = {status: sum(1 for c in rows if c.get("status") == status) for status in ("pass", "fail", "not_evaluated")}
    locations = [c.get("location", "") for c in rows]
    duplicates = sorted({loc for loc in locations if locations.count(loc) > 1})
    missing, unexpected = [], []
    if required is not None:
        required = set(required)
        missing = sorted(required - set(locations))
        unexpected = sorted(set(locations) - required)
    complete = bool(rows) and not duplicates and not missing and not unexpected and counts["not_evaluated"] == 0
    exceptions = [c for c in rows if c.get("status") != "not_evaluated" and (c.get("details") or {}).get("roof_exemption_applied")]
    ratios = [(c.get("details") or {}).get("ratio_provided") for c in rows
              if c.get("status") != "not_evaluated" and c not in exceptions]
    codes = sorted({code for c in rows if c.get("status") == "not_evaluated"
                    for code in ((c.get("details") or {}).get("reason_codes") or ["unclassified"])})
    return {"evaluated": counts["pass"] + counts["fail"] > 0, "all_pass": complete and counts["fail"] == 0,
            "no_known_failure": counts["fail"] == 0, "complete": complete, "counts": counts,
            "required": None if required is None else len(required), "missing": missing[:24],
            "missing_count": len(missing), "duplicates": duplicates[:24], "unexpected": unexpected[:24],
            "exceptions_18_7_3_1": len(exceptions),
            "min_ratio_provided": min((r for r in ratios if isinstance(r, (int, float))), default=None),
            "unknown_reason_codes": codes,
            "semantics": ("all_pass: every required joint, direction and sway sign present once and passing; "
                          "no_known_failure: no row failed (says nothing about unknown rows)")}


# ---- the required inventory ----------------------------------------------------------------------------------
def _detailing_inventory(state):
    """Candidate detailing identities, independent of emitted check rows.

    Grouped_Record supplies actual bar_rows (so spacing/steel are per face, not per
    uniform stacking direction), and envelopes the two joint geometry rules per group.
    Reversal/span strength additionally cover each distinct flange/anchorage context.
    Gravity beams retain 18.6.3.1/bar fit, but omit SFRS balance and hinge-zone rules.
    Gravity columns retain the column rules used by the declared grouped scope.
    Qualification-only completion/review rows are not new candidate requirements.
    """
    from Design import Group_Capacity as gc, Group_Checks as checks
    from Model import Member_Groups as mg
    exact = {("detailing", "detailing.material_scope", "")}

    def require(gid, names, locations=("",)):
        exact.update(("detailing", name, f"{gid}/{loc}" if loc else gid)
                     for name in names for loc in locations)

    with mg.installed(state):
        for gid, design in sorted(state.designs.items()):
            if design.is_column:
                require(gid, ("column.minimum_dimension", "column.aspect_ratio", "column.side_bar_clear_spacing",
                              "column.minimum_rho", "column.maximum_rho", "column.end_hoop_spacing_dimension_bar",
                              "column.end_hoop_spacing", "column.end_zone_length", "column.supported_bar_distance_basic",
                              "column.confinement_area_and_support"))
                require(gid, ("column.bar_clear_spacing",), ("top", "bottom"))
            else:
                require(gid, ("beam.minimum_width", "beam.width_to_depth_design_basis", "beam.depth_to_width_design_basis"))
                require(gid, ("beam.bar_clear_spacing", "beam.minimum_steel", "beam.maximum_rho", "beam.continuous_bars"),
                        ("top", "bottom"))
                if gc.is_gravity_beam_group(state, gid):
                    continue
                require(gid, ("beam.clear_span", "beam.column_projection", "beam.end_hoop_spacing",
                              "beam.end_zone_length", "beam.first_hoop"))
                contexts = [gid] + [f"{gid}/member:{tag}" for tag in checks.beam_group_representatives(state, gid)[1:]]
                for context in contexts:
                    require(context, ("beam.reversal_balance",), ("left", "right"))
                    require(context, ("beam.span_strength",), ("positive", "negative"))
    return exact


def _config(cfg):
    from Design.Config import DesignConfig, validate_beam_practical_ceiling
    cfg = cfg if cfg is not None else DesignConfig()
    validate_beam_practical_ceiling(cfg)
    return cfg


def evaluation_inventory(evaluation, cfg=None):
    """Inventory under the final record's realized context, even after a search restores globals.

    Beam context equivalence depends on the realized slab thickness, the joint types on the realized slab layout
    (its mats pass through the joints) and the beam contexts on the detail basis the candidate was evaluated on.
    Read those declared inputs, never the emitted rows; restore the caller's environment on success or error.
    """
    import Structure_Parameters as sp
    realized = ((evaluation.get("evidence") or {}).get("slab") or {})
    slab = realized.get("slab")
    if not isinstance(slab, dict) or "thickness_in" not in slab:
        raise ValueError("the final slab thickness declaration is missing from the evaluation")
    thickness = slab["thickness_in"]
    if thickness is not None and (type(thickness) not in (int, float) or not math.isfinite(thickness) or thickness <= 0):
        raise ValueError("the final slab thickness must be positive and finite, or explicitly None for no slab")
    basis = evaluation.get("detail_basis")
    previous = (sp.SLAB_THICKNESS_IN, sp.SLAB_REINFORCEMENT, sp.DETAIL_BASIS)
    try:
        sp.SLAB_THICKNESS_IN = thickness
        sp.SLAB_REINFORCEMENT = realized.get("reinforcement")
        if basis is not None:
            sp.DETAIL_BASIS = basis
        return expected_inventory(evaluation["state"], cfg)
    finally:
        sp.SLAB_THICKNESS_IN, sp.SLAB_REINFORCEMENT, sp.DETAIL_BASIS = previous


def expected_inventory(state, cfg=None):
    """The checks the realized design ``state`` requires in the candidate scope, from its topology alone.

    Returns {"exact": {(family, id, location)}, "patterns": [...], "conditional": [...], "scwb_required":
    [locations], "scwb_not_required": {location: basis}, "not_applicable": [...], "excluded": [(family, id,
    location, item)], "scwb_exception_limits": {location: Ag f'c / 10}}. Exact rows must appear once; a pattern
    asks for a count of rows matching it; a conditional set applies when its trigger row passes; the excluded rows
    are the declared scope exclusions (EXCLUDED_ROWS); an 18.7.3.1 exception row may stand only at a listed roof
    location, against its listed limit.
    Joints, their types, transitions and anchorage come from Group_Capacity.joint_inventory on the installed
    design; the SFRS directions of each column from Member_Groups.column_sfrs_directions under the state's layout.
    """
    import Structure_Parameters as sp
    from Design import Group_Capacity as gc
    from Model import Member_Groups as mg
    cfg = _config(cfg)
    counts = state.counts
    nx, ny, nf = counts["num_bay_x"], counts["num_bay_y"], counts["num_floor"]
    if (sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR) != (nx, ny, nf):
        raise ValueError(f"the installed frame geometry {(sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR)} is not the "
                         f"design's {(nx, ny, nf)}")
    exact, patterns, conditional, not_applicable = _detailing_inventory(state), [], [], []
    system = mg.is_system_policy(state.policy_id)
    layout = mg.layout_of(state.policy_id)
    gravity_beams, gravity_groups = set(), {}
    for gid, group in sorted(state.groups.items()):
        design = state.designs[gid]
        exact.add(("strength", "member.strength", gid))
        exact.add(("capacity", "detailing.splice_location", gid))
        exact.add(("capacity", "detailing.splice_physical_detail", gid))
        if system and state.system_of(gid) == mg.SYSTEM_GRAVITY:
            gravity_groups[gid] = "column" if design.is_column else "beam"
        if design.is_column:
            for name in ("column.capacity_shear_section", "column.hoops_selected", "column.supported_bar_spacing",
                         "column.cage_layout", "column.bar_bond_development"):
                exact.add(("capacity", name, gid))
        else:
            for name in ("beam.capacity_shear_section", "beam.hoops_selected", "beam.cage_layout","beam.slab_physical_inventory","beam.torsion_concurrent"):
                exact.add(("capacity", name, gid))
            if gc.is_gravity_beam_group(state, gid):
                gravity_beams.add(gid)
            else:
                exact.add(("capacity", "beam.axial_force_applicability", gid))
            for family_name, _dead, _live in gc.GRAVITY_FAMILIES:
                patterns.append({"family": "joints", "id_in": ("beam_capacity_shear_left", "beam_capacity_shear_right"),
                                 "prefix": f"{gid}/member_", "suffix": f"/{family_name}", "exactly": 2,
                                 "description": f"beam capacity shear at both faces of the governing {family_name} member "
                                                f"of {gid}"})
    for gid, kind in gravity_groups.items():
        exact.add(("capacity", f"gravity.{kind}_compatibility_branch", gid))
    exact.add(("capacity", "detailing.splices_designed", ""))
    exact.add(("reinforcement", "reinforcement.selected_for_every_group", ""))
    with mg.installed(state):
        inventory = gc.joint_inventory(state)
        scwb_required, scwb_not_required, exception_limits = [], {}, {}
        for entry in inventory["joints"]:
            for axis in ("x", "y"):
                if gc.joint_rule(entry, axis, state) == gc.JOINT_RULE_SFRS:
                    scwb_required += [f"{entry['id']}/{axis}/{sign}" for sign in ("positive", "negative")]
                    if entry["above"] is None and joint_rules.SCWB_DISCONTINUOUS_COLUMN_EXCEPTION:
                        # The column stops above this connection: the one place an 18.7.3.1 row may stand, at its
                        # own section's Ag f'c / 10 (Group_Capacity.scwb_evidence builds the same limit).
                        below = entry["below"].design
                        for sign in ("positive", "negative"):
                            exception_limits[f"{entry['id']}/{axis}/{sign}"] = below.b_in * below.h_in * below.fc_ksi / 10.0
                else:
                    scwb_not_required[f"{entry['id']}/{axis}"] = ("not required: the column is not part of the seismic "
                                                                  "force-resisting system in this direction (ACI 318-19 "
                                                                  f"18.14; {layout} layout)")
        transitions, through_any = set(), False
        for type_id, kind in inventory["types"].items():
            entry = kind["representative"]
            above = entry["above"]
            exact.add(("capacity","beam.joint_physical_detail",type_id))
            if above is not None:
                exact.add(('capacity', 'column.transition_physical_detail', type_id))
            else:
                exact.add(('capacity','column.roof_anchorage_physical_detail',type_id))
            if above is not None and above.group_id != entry["core"].group_id:
                transitions.add(f"{entry['core'].group_id}>{above.group_id}")
            exact.add(("capacity", "beam.bar_stacking_clear_of_slab_mats", type_id))
            for axis in ("x", "y"):
                beams = entry["beams"][axis]
                if not beams:
                    continue
                exact.add(("capacity", "joint.shear_screen", f"joint_shear/{type_id}/{axis}"))
                exact.add(("capacity", "joint.classification_evidence", f"joint_shear/{type_id}/{axis}"))
                exact.add(("joints", "joint_shear", f"joint_shear/{type_id}/{axis}"))
                exact.add(("capacity", "beam.bars_thread_column", f"{type_id}/{axis}"))
                through = len(beams) == 2 and not bool((entry.get("mixed") or {}).get(axis))
                exact.add(("capacity", "joint.through_bar_depth" if through else "joint.terminating_bar_hook",
                           f"{type_id}/{axis}"))
                if through:
                    through_any = True
                    exact.add(("joints", "joint_through_bar_depth", f"{type_id}/{axis}"))
                    if abs(float(sp.FY_KSI) - 80.0) < 1e-9:
                        exact.add(("joints", "joint_grade80_normalweight", f"{type_id}/{axis}"))
    for key in transitions:
        exact.add(("capacity", "column.transition", key))
    for location in scwb_required:
        exact.add(("joints", "scwb", location))
    if not through_any:
        not_applicable.append(("joints", "through_bar_anchorage", "",
                               "no joint direction has beam bars passing through: 18.8.2.3 has nothing to check"))
    for story in range(1, nf + 1):
        for axis in ("x", "y"):
            exact.add(("drift", "demands.node_coverage", f"story:{story}/Q{axis}"))
            exact.add(("drift", "demands.story_drift", f"story:{story}/Q{axis}/{axis}"))
            exact.add(("drift", "demands.stability", f"story:{story}/Q{axis}/{axis}"))
    exact.add(("regularity", "demands.vertical_stiffness_regularity", ""))
    exact.add(("regularity", "demands.vertical_strength_regularity", ""))
    if cfg.demands.accidental_torsion_ratio:
        exact.add(("torsion", "demands.torsional_amplification", ""))
    if cfg.floor_analysis.transfer_to_frame:
        computed=(getattr(cfg.floor_analysis,'common_action_procedure',None) or {}).get('kind')=='computed_common_actions_v1'
        if computed:
            from Design.SMRF_Common_Action_Contract import required_inventory
            exact.update(required_inventory())
        else:
            from Design.SMRF_Conventional_Floor import floor_row_inventory
            exact.update(floor_row_inventory())
        exact.add(("slab", "slab.reinforcement_designed", ""))
        exact.add(("slab", "slab.evidence_current", ""))
        slab_required=COMPUTED_SLAB_COMPLETION_REQUIRED if computed else SLAB_COMPLETION_REQUIRED
        conditional.append({"trigger": ("slab", "slab.reinforcement_designed", ""),
                            "requires": [("slab", name, location) for name, location in slab_required],
                            "description": "the completion and development checks of a selected slab layout"})
    return {"exact": exact, "patterns": patterns, "conditional": conditional, "scwb_required": sorted(scwb_required),
            "scwb_exception_limits": exception_limits,
            "scwb_not_required": scwb_not_required, "not_applicable": not_applicable,
            "excluded": [(f, n, l, item) for (f, n, l), item in EXCLUDED_ROWS.items()],
            "review_permitted": {key: sorted(items) for key, items in REVIEW_PERMITTED.items()},
            "scope": scope_for_common_actions(cfg.floor_analysis.transfer_to_frame)["id"],
            "scope_definition": scope_for_common_actions(cfg.floor_analysis.transfer_to_frame),
            "counts": {"exact": len(exact), "patterns": len(patterns),
                                             "scwb_required": len(scwb_required)}}


# The required checks that may legitimately wait on an engineering review, and on which review only. Any other required
# row that labels itself review-dependent is invalid: a label cannot turn a computed unknown into a non-blocking item.
REVIEW_PERMITTED = {
    ("regularity", "demands.vertical_strength_regularity"): {"story_strength_model_verified",
                                                             "vertical_irregularity_consequences_reviewed"},
    ("regularity", "demands.vertical_stiffness_regularity"): {"vertical_irregularity_consequences_reviewed"},
    ("torsion", "demands.torsional_amplification"): {"story_strength_model_verified"},
}


# ---- classifying the rows ------------------------------------------------------------------------------------
def classify_row(row):
    """(state, problems) of one compact row: pass, fail, unknown, review or invalid."""
    problems = check_row_problems(row)
    if problems:
        return "invalid", problems
    if row["status"] == "not_evaluated":
        review = (row.get("details") or {}).get("review_dependency")
        if isinstance(review, dict) and review.get("item"):
            return "review", []
        return "unknown", []
    return row["status"], []


def _witness(row, state, problems=None):
    item = {"family": row.get("family"), "id": row.get("id"), "location": row.get("location", ""), "state": state}
    if state in ("pass", "fail"):
        item.update(demand=row.get("demand"), capacity=row.get("capacity"), comparison=row.get("comparison"),
                    units=row.get("units", ""))
    details = row.get("details") or {}
    for key in ("reason", "reason_code", "reason_codes", "branch", "review_dependency", "upstream_reasons"):
        if key in details:
            item[key] = details[key]
    if problems:
        item["problems"] = problems
    return item


def violation(row):
    """The dimensionless exceedance of one failed row (0 for anything else): demand / capacity - 1 for '<=',
    capacity / demand - 1 for '>=', 1 for an unmet equality. No unit-bearing quantities are added together."""
    if row.get("status") != "fail":
        return 0.0
    demand, capacity, comparison = row.get("demand"), row.get("capacity"), row.get("comparison")
    try:
        demand, capacity = float(demand), float(capacity)
    except (TypeError, ValueError):
        return 1.0
    if comparison == "<=" and capacity > 0.0:
        return max(demand / capacity - 1.0, 0.0) or 1e-9
    if comparison == ">=" and demand > 0.0:
        return max(capacity / demand - 1.0, 0.0) or 1e-9
    return 1.0


def _is_scwb_exception(row):
    """Whether a strong-column row claims the 18.7.3.1 exception (its clause, or its recorded flag)."""
    details = row.get("details") if isinstance(row.get("details"), dict) else {}
    return row.get("clause") == SCWB_EXCEPTION_CLAUSE or details.get("roof_exemption_applied") is True


def code_check_result(rows, expected):
    """The authoritative code-check result of compact ``rows`` against the ``expected`` inventory.

    status: ``invalid`` (a malformed or self-contradictory row, a duplicated identity, a row the design does not
    have, or a required row claiming a review it is not permitted to wait on), ``failed`` (a known failure; unknown
    rows may remain beside it), ``incomplete`` (a required row missing or unknown), ``passed`` (every required row
    present once, well formed and passing). Review-dependent rows are listed and do not change the status; only the
    required checks of ``expected["review_permitted"]`` may be review-dependent, and only on the named review. A row
    of a declared scope exclusion (``expected["excluded"]``) is skipped when it is not_evaluated and labelled with its
    item, and invalid otherwise: the item is not checked, so it can neither pass nor fail. Every list keeps the rows'
    own identities and reasons; a failing duplicate stays among the known failures.
    """
    exact = expected["exact"]
    governed = {(family, name) for family, name, _ in exact}
    permitted = {tuple(key.split("|", 1)) if isinstance(key, str) else tuple(key): set(items)
                 for key, items in (expected.get("review_permitted") or {}).items()}
    # Every key that can be required (exactly, or once its conditional trigger passes) is held to the permitted list.
    required_keys = set(exact) | {tuple(key) for rule in expected.get("conditional", []) for key in rule["requires"]}
    seen, duplicates, unexpected, invalid = {}, [], [], []
    duplicate_failures = []
    states = {}
    not_applicable = {(f, n, l): basis for f, n, l, basis in expected.get("not_applicable", [])}
    excluded = {(f, n, l): item for f, n, l, item in expected.get("excluded", [])}
    applicable_rows = []
    for row in rows:
        key = (row.get("family"), row.get("id"), row.get("location", ""))
        if key in not_applicable and row.get("status") == "not_evaluated":
            continue
        if key in seen:
            duplicates.append({"family": key[0], "id": key[1], "location": key[2], "status": row.get("status")})
            if classify_row(row)[0] == "fail":
                duplicate_failures.append(_witness(row, "fail"))
            continue
        seen[key] = row
        state, problems = classify_row(row)
        if key in excluded and state != "invalid":
            label = (row.get("details") or {}).get("scope_exclusion")
            if state == "unknown" and isinstance(label, dict) and label.get("item") == excluded[key]:
                continue                                  # outside the declared scope: not checked, not passed
            state, problems = "invalid", [f"a row of the scope exclusion {excluded[key]!r} must be not_evaluated and "
                                          "labelled with it: the item is not checked, so it cannot pass, fail or wait "
                                          "on a review"]
        if key[:2] == ("joints", "scwb") and state in ("pass", "fail") and _is_scwb_exception(row):
            limit = (expected.get("scwb_exception_limits") or {}).get(key[2])
            details = row.get("details") if isinstance(row.get("details"), dict) else {}
            if (state != "pass" or limit is None or row.get("comparison") != "<="
                    or not float(row["demand"]) < float(row["capacity"])
                    or any(k in details for k in ("numerical_tolerance_in", "numerical_tolerance"))
                    or not math.isclose(float(row["capacity"]), float(limit), rel_tol=1e-9, abs_tol=1e-9)):
                state, problems = "invalid", ["an 18.7.3.1 exception row is valid only at a roof connection of this "
                                              "design, passing, against that column's own Ag f'c / 10"]
        if state == "review" and key in required_keys:
            item = ((row.get("details") or {}).get("review_dependency") or {}).get("item")
            if item not in permitted.get((key[0], key[1]), set()):
                state, problems = "invalid", [f"required check labelled as waiting on review {item!r}, which it may not"]
        if state == "invalid":
            invalid.append(_witness(row, state, problems))
        if (key[0], key[1]) in governed and key not in exact:
            unexpected.append({"family": key[0], "id": key[1], "location": key[2]})
        states[key] = state
        applicable_rows.append(row)
    missing = sorted(key for key in exact if key not in seen)
    for rule in expected.get("conditional", []):
        if states.get(tuple(rule["trigger"])) == "pass":
            missing += sorted(tuple(key) for key in rule["requires"] if tuple(key) not in seen)
    pattern_problems = []
    for pattern in expected.get("patterns", []):
        if "group" in pattern:
            gid = pattern["group"]
            found = [r for r in applicable_rows if r.get("family") == pattern["family"]
                     and (r.get("location") == gid or str(r.get("location", "")).startswith(gid + "/"))]
            if len(found) < pattern["minimum"]:
                pattern_problems.append({"pattern": pattern["description"], "found": len(found)})
        else:
            found = [r for r in applicable_rows if r.get("family") == pattern["family"] and r.get("id") in pattern["id_in"]
                     and str(r.get("location", "")).startswith(pattern["prefix"])
                     and str(r.get("location", "")).endswith(pattern["suffix"])]
            if len(found) != pattern["exactly"]:
                pattern_problems.append({"pattern": pattern["description"], "found": len(found),
                                         "expected": pattern["exactly"]})
    witnesses = {state: [_witness(seen[key], state) for key, s in states.items() if s == state]
                 for state in ("fail", "unknown", "review")}
    witnesses["fail"] += duplicate_failures
    counts = {state: sum(1 for s in states.values() if s == state) for state in ("pass", "fail", "unknown", "review", "invalid")}
    counts.update(missing=len(missing) + len(pattern_problems), duplicates=len(duplicates), unexpected=len(unexpected),
                  rows=len(rows), required=len(exact))
    families = {}
    for (family, _name, _loc), state in states.items():
        families.setdefault(family, {s: 0 for s in ("pass", "fail", "unknown", "review", "invalid")})[state] += 1
    if invalid or duplicates or unexpected:
        status = "invalid"
    elif counts["fail"]:
        status = "failed"
    elif missing or pattern_problems or counts["unknown"]:
        status = "incomplete"
    else:
        status = "passed"
    reviews = {}
    for item in witnesses["review"]:
        name = (item.get("review_dependency") or {}).get("item") or "unnamed_review"
        reviews.setdefault(name, []).append(f"{item['id']}@{item['location']}" if item["location"] else item["id"])
    return {"contract_version": CONTRACT_VERSION, "scope_id": expected.get("scope", SCOPE["id"]), "status": status, "accepted": status == "passed",
            "complete": not (missing or pattern_problems or counts["unknown"] or invalid or duplicates or unexpected),
            "counts": counts, "families": families,
            "known_failures": witnesses["fail"], "unknown": witnesses["unknown"],
            "missing": [{"family": f, "id": n, "location": l} for f, n, l in missing] + pattern_problems,
            "duplicates": duplicates, "unexpected": unexpected, "invalid": invalid,
            "review_dependencies": [{"item": name, "rows": rows_} for name, rows_ in sorted(reviews.items())],
            "unknown_reason_codes": sorted({code for item in witnesses["unknown"]
                                            for code in ([item["reason_code"]] if item.get("reason_code") else [])
                                            + list(item.get("reason_codes") or [])}),
            "violation": {"max": max((violation(r) for r in applicable_rows), default=0.0),
                          "sum": sum(violation(r) for r in applicable_rows),
                          "basis": "dimensionless exceedances of the known failures only"},
            "not_applicable": [{"family": f, "id": n, "location": l, "basis": b} for (f, n, l), b in not_applicable.items()],
            "excluded": [{"family": f, "id": n, "location": l, "item": item, "basis": SCOPE["excluded"].get(item)}
                         for (f, n, l), item in excluded.items()]}


# ---- the boundary a search applies -------------------------------------------------------------------------------
OBJECTIVE_KEYS = ("concrete_total_in3", "total_steel_in3", "frame_steel_in3", "distinct_form_size_count")


def _finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def objective_availability(quantities):
    """{key: {"available", "value"}} of the material objectives; a missing quantity is unavailable, never zero."""
    quantities = quantities if isinstance(quantities, dict) else {}
    return {key: {"available": _finite_nonnegative(quantities.get(key)), "value": quantities.get(key)}
            for key in OBJECTIVE_KEYS}


def accepted_for_code_design(code_check, completion_complete, objectives):
    """The one acceptance a search may archive: the code check passed, the evaluation completed numerically (every
    completion flag True, the two selection flags also met by a verified cycle resolution of the same final state:
    numerical_completion) and every material objective is available. Review dependencies stay listed beside it."""
    return bool(code_check.get("accepted")) and bool(completion_complete) and all(
        item["available"] for item in objectives.values())


def reselection_problems(submitted, final):
    """Why ``final`` is not ``submitted`` with only its reinforcement reselected: the same assignments and policy,
    the same groups, and each group's member type, section and concrete grade unchanged. Bars, hoops and legs may
    differ (the evaluation selects them)."""
    if submitted is None or final is None:
        return []
    problems = []
    if submitted.assignments.get("sha256") != final.assignments.get("sha256") or submitted.policy_id != final.policy_id:
        problems.append("the final design has other group assignments or another grouping policy than the submitted one")
    if set(submitted.designs) != set(final.designs):
        return problems + ["the final design has other groups than the submitted one"]
    for gid in sorted(submitted.designs):
        a, b = submitted.designs[gid], final.designs[gid]
        changed = [name for name in ("member_type", "b_in", "h_in", "fc_ksi") if getattr(a, name) != getattr(b, name)]
        if changed:
            problems.append(f"group {gid}: {', '.join(changed)} differ from the submitted design")
    return problems


def assess_evaluation(evaluation, submitted_identity=None, cfg=None, submitted_state=None):
    """Validate one evaluation record and return its authoritative classification.

    A search calls this before ranking or archiving a candidate. The record must carry the contract version,
    the submitted seed identity (equal to ``submitted_identity`` captured before evaluation), the final state
    whose identity is ``result_identity``, every completion flag as an explicit boolean, and its code-check rows
    bound to that final design (state and realized identities). The required inventory is derived again from the
    final state and the result recomputed from the rows; a record whose own reported result differs is invalid.
    A final design whose cages and hoops differ from the submitted seed is normal (they are reselected); given
    ``submitted_state``, any other difference (groups, assignments, policy, a section or a concrete grade) makes the
    record invalid. A missing final state is not replaced by the seed. A record whose own rows are invalid
    (malformed, duplicated, unexpected, or a required row claiming a review it may not wait on) is invalid.

    Returns {"valid", "problems", "classification" (accepted / check_failed / unresolved / invalid),
    "numerical_completion", "preliminary_screen", "code_check", "known_failures", "review_dependencies",
    "objectives", "nonlinear_performance", "accepted"}.
    """
    problems = []
    result = {"contract_version": CONTRACT_VERSION, "valid": False, "problems": problems, "classification": "invalid",
              "numerical_completion": {"complete": False, "flags": {}, "problems": []},
              "preliminary_screen": None, "code_check": None, "known_failures": [], "review_dependencies": [],
              "objectives": {}, "nonlinear_performance": dict(NONLINEAR_STATUS), "accepted": False}
    if not isinstance(evaluation, dict):
        problems.append("the evaluation is not a record")
        return result
    status = evaluation.get("status")
    if status not in ("evaluated", "infeasible"):
        problems.append(f"unknown evaluation status {status!r}")
        return result
    if submitted_identity is not None and evaluation.get("seed_identity") != submitted_identity:
        problems.append("the evaluation's seed identity is not the identity of the design submitted for evaluation")
    if status == "infeasible":
        result.update(valid=not problems, classification="unresolved" if not problems else "invalid",
                      numerical_completion={"complete": False, "flags": {},
                                            "problems": [f"evaluation stopped at {evaluation.get('failed_stage')}: "
                                                         f"{evaluation.get('reason')}"]})
        return result
    from Design.Grouped_Design import EVALUATION_VERSION
    if evaluation.get("version") != EVALUATION_VERSION:
        problems.append(f"the evaluation is not of version {EVALUATION_VERSION}")
    state = evaluation.get("state")
    if state is None or not hasattr(state, "identity"):
        problems.append("the evaluated record carries no final design state")
        return result
    identity = state.identity()
    if evaluation.get("result_identity") != identity:
        problems.append("result_identity is not the identity of the final design state")
    if submitted_state is not None:
        if submitted_identity is not None and submitted_state.identity() != submitted_identity:
            problems.append("the submitted design state is not the one whose identity was captured")
        problems.extend(reselection_problems(submitted_state, state))
    constraints = evaluation.get("constraints")
    if not isinstance(constraints, dict):
        problems.append("the evaluated record has no constraints block")
        constraints = {}
    flags, completion_problems, complete, resolved, malformed = numerical_completion(constraints, identity)
    for flag, value in malformed.items():
        problems.append(f"completion flag {flag} is {value!r}, not an explicit boolean")
    result["numerical_completion"] = {"complete": complete, "flags": flags, "problems": completion_problems,
                                      **({"selection_flags_met_by": VERIFIED_CYCLE_RESOLUTION} if resolved else {})}
    screen = evaluation.get("candidate_screen_passed")
    if type(screen) is not bool:
        problems.append("candidate_screen_passed is not an explicit boolean")
    result["preliminary_screen"] = {"passed": screen if type(screen) is bool else None,
                                    "meaning": "no recorded failure in the screened families and a settled pass; "
                                               "not evidence that every required check passed"}
    block = evaluation.get("code_checks")
    if not isinstance(block, dict) or block.get("contract_version") != CONTRACT_VERSION:
        problems.append(f"the code-check rows are missing or not of contract {CONTRACT_VERSION}")
        return result
    if block.get("state_identity") != identity:
        problems.append("the code-check rows are bound to another design state")
    slab = ((evaluation.get("evidence") or {}).get("slab") or {})
    if "evidence" not in evaluation or not isinstance(slab, dict):
        problems.append("the realized slab evidence is not in the record, so the rows' realized identity cannot be checked")
    else:
        realized = realized_identity(identity, (slab.get("slab") or {}).get("thickness_in"), slab.get("reinforcement"))
        if block.get("realized_identity") != realized:
            problems.append("the code-check rows are bound to another realized design (slab layout, inputs or context)")
    rows = block.get("rows")
    if not isinstance(rows, list):
        problems.append("the code-check rows are not a list")
        return result
    try:
        expected = evaluation_inventory(evaluation, cfg)
    except Exception as exc:                                  # noqa: BLE001 -- an underivable inventory is a finding
        problems.append(f"the required inventory cannot be derived from the final design: {type(exc).__name__}: {exc}")
        return result
    recomputed = code_check_result(rows, expected)
    if recomputed["status"] == "invalid":
        problems.append(f"the code-check rows are invalid: {len(recomputed['invalid'])} malformed or mislabelled, "
                        f"{len(recomputed['duplicates'])} duplicated, {len(recomputed['unexpected'])} unexpected")
    reported = evaluation.get("code_check")
    if not isinstance(reported, dict) or reported.get("status") != recomputed["status"] \
            or reported.get("counts") != recomputed["counts"]:
        problems.append("the evaluator's reported code-check result does not follow from its rows and the "
                        "topology-derived inventory")
    objectives = objective_availability(evaluation.get("quantities"))
    accepted = accepted_for_code_design(recomputed, result["numerical_completion"]["complete"], objectives)
    if evaluation.get("feasible") is not accepted:
        problems.append(f"feasible is {evaluation.get('feasible')!r}; the code check, completion and objectives give "
                        f"{accepted!r}")
    result.update(code_check=recomputed, known_failures=recomputed["known_failures"],
                  review_dependencies=recomputed["review_dependencies"], objectives=objectives)
    if problems:
        result.update(valid=False, classification="invalid", accepted=False)
        return result
    if accepted:
        classification = "accepted"
    elif (recomputed["status"] == "failed" and not recomputed["counts"]["unknown"] and not recomputed["missing"]
          and result["numerical_completion"]["complete"]):
        classification = "check_failed"
    else:
        classification = "unresolved"
    result.update(valid=True, classification=classification, accepted=accepted)
    return result
