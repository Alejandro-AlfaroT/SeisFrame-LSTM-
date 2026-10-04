"""The grouped design record: built from a search result, applied to the model, and qualified (2026-10-02).

A grouped record (schema ``rc_smrf_grouped_candidate_v1``) is the grouped counterpart of the uniform
design record of Design_Driver. It has no ``sections`` and no ``reinforcement`` block: the frame is its
``member_groups`` block (explicit assignments, each group's selected design, one resolved row per
physical member), and nothing in it can be installed into the uniform section values of
Structure_Parameters. A uniform consumer that is handed one fails on the missing keys instead of
reading a summary.

  build_record              the final candidate of Design.Grouped_Search with its evidence, the candidate
                            log, the retained tradeoffs and the stop reason;
  apply_grouped_design      validates a saved record against the running model and installs it (the grouped
                            design, the slab, the by-floor transfer, the slab layout); all or nothing;
  qualify_grouped_design    recomputes the capacity design, the member strengths, the quantities, the weight
                            ledger and the story-by-story regularity evidence from the record and consumes
                            only the recomputed objects; items that rest on reviews not made stay
                            ``not_evaluated``;
  load_or_create_grouped_design
                            the cached entry point: a record made under other inputs, source, grouping,
                            seed or search policy is refused, never relabelled.

Nothing here asserts a verification or lifts Design.SMRF_Qualification.GENERATION_RELEASE_READY.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import types
import uuid
from dataclasses import asdict
from pathlib import Path

import openseespy.opensees as ops

import Structure_Parameters as sp
from Design import Design_Driver as driver
from Design import Group_Capacity as capacity_design
from Design import Group_Checks as group_checks
from Design import Group_Selection as selection
from Design import Grouped_Design as gd
from Design import Grouped_Search as gs
from Design import SMRF_Floor_Sections as floor_sections
from Design import SMRF_Transitions as transition_rules
from Design.SMRF_Common import make_check, not_evaluated, summarize_checks
from Model import Member_Bar_Layers as bar_layers
from Model import Member_Groups as mg
from Model import Member_Properties as mp

SCHEMA = gd.SCHEMA
ARTIFACT_NAME = "grouped_design.json"
CANDIDATE_LOG_NAME = "grouped_candidates.jsonl"
QUALIFICATION_VERSION = "grouped_qualification_v1"
# Placeholder cages of an explicit-section seed. Every group's reinforcement is selected by the first
# evaluation; these only make the seed a complete MemberDesign.
NOMINAL_COLUMN_CAGE = {"bar_size": 8, "top_bars": 3, "bot_bars": 3, "side_bars": 1, "stirrup_bar_size": 4, "stirrup_legs": 3,
                       "stirrup_spacing_in": 4.0, "stirrup_legs_by_direction": (3, 3)}
NOMINAL_BEAM_CAGE = {"bar_size": 8, "top_bars": 3, "bot_bars": 3, "side_bars": 0, "stirrup_bar_size": 4, "stirrup_legs": 2,
                     "stirrup_spacing_in": 4.0}
# Model state a grouped record installs besides the grouped design itself.
_STATE_KEYS = ("SLAB_THICKNESS_IN", "FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF", "SEISMIC_LIVE_LOAD_FRACTION", "AGGREGATE_MAX_SIZE_IN",
               "REINFORCEMENT_SPECIFICATION", "MATERIAL_EXPOSURE", "FLOOR_TRANSFER", "SLAB_REINFORCEMENT", "SLAB_ACTIONS",
               "BEAM_BAR_STACKING", "BEAM_BAR_MAX_LAYERS", "BEAM_BAR_LAYER_ORDER", "JOINT_SHEAR_CATEGORIES",
               "BEAM_CLEAR_COVER_IN", "COL_CLEAR_COVER_IN")
LIMITATIONS = (
    "Column transitions follow the declared rules of Design.SMRF_Transitions (" + transition_rules.RULES_ID + "): a limited "
    "supported scope, not a statement that no other detail can be built. They await engineering review; a face step of "
    "3 in or more (dowels, ACI 318-19 10.7.4.2) is not a supported transition, and hoop-leg engagement at offset bends, "
    "the lap zone and the confinement above a reduced column are not established.",
    "Concrete grades are those of the seed and are not searched: no size is shown to be a minimum, and no geometry "
    "infeasible, over the full material domain.",
    "One column concrete grade and one beam concrete grade for the whole frame; group-specific grades are not available.",
    "Square columns; opposite sides of the plan share a group; the two-story band rule is a research policy, not a code rule.",
    "The story-strength model is provisional (review item M1); vertical strength regularity is recorded, not established.",
    "Nonlinear model scope: IMK member hinges with rigid centerline joints. Fiber members, scissors joint springs and the "
    "floor compatibility, cut-region, nodal-cut and composite-section diagnostics refuse a grouped design.",
    "Quantities are modeled quantities for comparing candidates (no laps, hooks, waste); no cost data is used.",
)


class GroupedDesignNotFound(RuntimeError):
    """The bounded search ended without a feasible design. ``search`` carries the log and the stop reason."""

    def __init__(self, message, search):
        super().__init__(message)
        self.search = search


def _canonical(value):
    """The JSON form a saved record has (string keys, lists), so in-memory and reloaded evidence compare alike."""
    return json.loads(json.dumps(gd._json_safe(value), allow_nan=False))


def _sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def differences(saved, recomputed, path="", tolerance=1e-9, limit=12, found=None):
    """Paths at which two JSON-like values differ (numbers within a relative and absolute tolerance)."""
    found = [] if found is None else found
    if len(found) >= limit:
        return found
    if isinstance(saved, dict) and isinstance(recomputed, dict):
        for key in sorted(set(saved) | set(recomputed), key=str):
            if key not in saved or key not in recomputed:
                found.append(f"{path}/{key}: only in the {'recomputed' if key not in saved else 'saved'} evidence")
            else:
                differences(saved[key], recomputed[key], f"{path}/{key}", tolerance, limit, found)
            if len(found) >= limit:
                break
    elif isinstance(saved, list) and isinstance(recomputed, list):
        if len(saved) != len(recomputed):
            found.append(f"{path}: {len(saved)} saved entries, {len(recomputed)} recomputed")
        else:
            for index, (a, b) in enumerate(zip(saved, recomputed)):
                differences(a, b, f"{path}[{index}]", tolerance, limit, found)
                if len(found) >= limit:
                    break
    elif (isinstance(saved, (int, float)) and isinstance(recomputed, (int, float))
          and not isinstance(saved, bool) and not isinstance(recomputed, bool)):
        if not math.isclose(saved, recomputed, rel_tol=tolerance, abs_tol=tolerance):
            found.append(f"{path}: {saved!r} saved, {recomputed!r} recomputed")
    elif saved != recomputed:
        found.append(f"{path}: {saved!r} saved, {recomputed!r} recomputed")
    return found


# ---- seeds -----------------------------------------------------------------------------------------------------
def _require_seed_sections(column, beam):
    for name, values in (("column", column), ("beam", beam)):
        if len(values) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                                   for v in values):
            raise ValueError(f"The seed {name} section is (b_in, h_in, fc_ksi), finite and positive; got {values!r}.")
    if column[0] != column[1]:
        raise ValueError(f"The grouped search keeps square columns; the seed column is {column[0]:g} x {column[1]:g}.")


def seed_from_sections(column, beam):
    """(GroupedDesign, description): every group starts on one column and one beam section (b, h, fc)."""
    column, beam = tuple(float(v) for v in column), tuple(float(v) for v in beam)
    _require_seed_sections(column, beam)
    state = mg.GroupedDesign.uniform(sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR,
                                     mg.MemberDesign("column", *column, **NOMINAL_COLUMN_CAGE),
                                     mg.MemberDesign("beam_x", *beam, **NOMINAL_BEAM_CAGE))
    return state, {"kind": "explicit_uniform_sections", "column": list(column), "beam": list(beam),
                   "note": ("where the search starts, not a lower or upper bound on any group; the cages are placeholders "
                            "and every group's reinforcement is selected by the first evaluation")}


def seed_from_uniform_record(record):
    """(GroupedDesign, description): a saved uniform design expanded into its groups (the equivalence fixture)."""
    sections, rebar = record["sections"], record["reinforcement"]
    column = (sections["b_col_in"], sections["h_col_in"], sections["fc_col_ksi"])
    beam = (sections["b_beam_in"], sections["h_beam_in"], sections["fc_beam_ksi"])
    _require_seed_sections(column, beam)
    legs = rebar.get("col_stirrup_legs_by_direction")
    state = mg.GroupedDesign.uniform(
        sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR,
        mg.MemberDesign("column", *column, rebar["col_bar_size"], rebar["col_top_bars"], rebar["col_bot_bars"],
                        rebar["col_side_bars"], rebar["col_stirrup_bar_size"], rebar["col_stirrup_legs"],
                        rebar["col_stirrup_spacing_in"], None if legs is None else mg._legs_tuple(legs)),
        mg.MemberDesign("beam_x", *beam, rebar["beam_bar_size"], rebar["beam_top_bars"], rebar["beam_bot_bars"],
                        rebar["beam_side_bars"], rebar["beam_stirrup_bar_size"], rebar["beam_stirrup_legs"],
                        rebar["beam_stirrup_spacing_in"]))
    return state, {"kind": "uniform_design_record", "column": list(column), "beam": list(beam),
                   "source_schema": record.get("schema_version"),
                   "source_request_sha256": (record.get("request_identity") or {}).get("sha256"),
                   "note": "the saved uniform design expanded into its groups; the search may change any group"}


# ---- identity --------------------------------------------------------------------------------------------------
def grouped_request_identity(cfg=None, seed=None, policy=None):
    """Portable identity of a grouped design request: the uniform request's inputs, policy and source, plus the
    grouping policy and assignments, the seed, the search policy and budget, and the declared rule versions."""
    if mg.is_grouped():
        raise mg.GroupedStateError("A grouped request identity is built before a grouped design is installed.")
    policy = policy or gs.SearchPolicy()
    base = driver.design_request_identity(cfg)
    assignments = mg.build_assignments(sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR)
    payload = {"schema": SCHEMA, "inputs": base["inputs"], "policy": base["policy"], "source_sha256": base["source_sha256"],
               "uniform_request_sha256": base["sha256"],
               "grouping": {"policy_id": mg.POLICY_ID, "policy": mg.POLICY, "assignments_sha256": assignments["sha256"],
                            "group_count": len(assignments["groups"])},
               "seed": seed,
               "search": {"version": gs.SEARCH_VERSION, **asdict(policy),
                          "max_column_size_step_in": gs.MAX_COLUMN_SIZE_STEP_IN,
                          "max_beam_reduction_options": gs.MAX_BEAM_REDUCTION_OPTIONS},
               "rules": {"evaluation": gd.EVALUATION_VERSION, "reinforcement_selection": selection.RULES_ID,
                         "column_transitions": transition_rules.RULES_ID, "bar_layers": bar_layers.RULE_ID,
                         "capacity_design": capacity_design.METHOD_VERSION, "qualification": QUALIFICATION_VERSION}}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(), **json.loads(canonical)}


# ---- the record ------------------------------------------------------------------------------------------------
def _floor_loads():
    """The load inventory of the installed grouped design; member weights come from the member ledger."""
    return mp.floor_load_metadata()


def _materials(cfg, state):
    grades = {kind: sorted({state.designs[g["group_id"]].fc_ksi for g in state.groups.values()
                            if (g["member_type"] == "column") == (kind == "column")}) for kind in ("column", "beam")}
    return {"fy_ksi": sp.FY_KSI, "fyt_ksi": sp.FY_KSI, "es_ksi": sp.ES_KSI,
            "normalweight": sp.CONCRETE_UNIT_WEIGHT_KCF == 0.150, "aggregate_size_in": sp.AGGREGATE_MAX_SIZE_IN,
            "reinforcement_specification": cfg.materials.reinforcement_specification, "exposure": cfg.materials.exposure,
            "fc_col_ksi": grades["column"][0], "fc_beam_ksi": grades["beam"][0],
            "beam_clear_cover_in": sp.BEAM_CLEAR_COVER_IN, "col_clear_cover_in": sp.COL_CLEAR_COVER_IN,
            "cover_basis": "clear_cover_outside_hoops",
            "beam_bar_stacking": {"mode": sp.BEAM_BAR_STACKING, "convention": sp.beam_bar_stacking_convention(),
                                  "max_layers": sp.BEAM_BAR_MAX_LAYERS, "layer_order": sp.BEAM_BAR_LAYER_ORDER,
                                  "rule": bar_layers.RULE_ID}}


def group_detailing(state, capacity):
    """The end-zone geometry declared with each group's hoops (the spacing itself is the capacity design's)."""
    groups = {}
    for gid, group in sorted(state.groups.items()):
        design = state.designs[gid]
        if design.is_column:
            heights = [capacity_design.column_clear_heights(state.member(tag)) for tag in group["member_tags"]]
            clear = max(max(h.values()) for h in heights)
            confinement = (capacity["columns"].get(gid) or {}).get("confinement") or {}
            groups[gid] = {"hoop_spacing_in": design.stirrup_spacing_in, "clear_height_in": clear,
                           "end_zone_length_in": max(design.b_in, design.h_in, clear / 6.0, 18.0),
                           "hx_in": confinement.get("hx_in"), "zone_origin": "joint_face"}
        else:
            groups[gid] = {"hoop_spacing_in": design.stirrup_spacing_in, "end_zone_length_in": 2.0 * design.h_in,
                           "first_hoop_distance_in": 2.0, "zone_origin": "joint_face"}
    return {"groups": groups,
            "analysis_application": ("each group's selected hoop spacing is uniform over the whole member; end-zone lengths "
                                     "are recorded geometry (18.6.4.1, 18.7.5.1), the largest clear height of the group's "
                                     "columns governing lo"),
            "joint_continuity": {
                "beam_reinforcement_continuous_through_interior_joints": True,
                "column_reinforcement_continuous_through_floor_joints": "per transition (capacity_design.transitions)",
                "basis": ("each beam group's top and bottom bars are the same in every span of the group and run through "
                          "interior joints; where two beam groups meet at a joint the bars of each are developed per the "
                          "joint anchorage evidence; a column cage changes only at a band boundary by a declared "
                          f"transition ({transition_rules.RULES_ID})")}}


def beam_end_strengths(state):
    """{member tag: nominal hogging and sagging strength at each end}: what each beam's hinges yield at."""
    result = {}
    for member in state.members():
        if member.is_column:
            continue
        basis = mp.beam_strengths(member)["basis"]
        result[str(member.member_tag)] = {"group_id": member.group_id, "family": basis["family"],
                                          **{f"{sign}_{end}_kip_in": basis[f"{sign}_{end}_kip_in"]
                                             for sign in ("hogging", "sagging") for end in ("i", "j")}}
    return result


def _legacy_summary(state):
    """A labelled digest for readers of the old one-section tables. Not a design input; nothing installs it."""
    columns = [d for d in state.designs.values() if d.is_column]
    beams = [d for d in state.designs.values() if not d.is_column]
    return {"label": "summary only: a grouped design has no one section or cage; read member_groups",
            "column_size_range_in": [min(d.b_in for d in columns), max(d.b_in for d in columns)],
            "beam_depth_range_in": [min(d.h_in for d in beams), max(d.h_in for d in beams)],
            "beam_width_range_in": [min(d.b_in for d in beams), max(d.b_in for d in beams)],
            "distinct_form_sizes": state.distinct_form_sizes()}


def build_record(search, cfg, seed, identity=None, verbose=True):
    """The grouped design record of a finished search (its final feasible candidate). Raises GroupedDesignNotFound."""
    final = search.get("final")
    if final is None:
        raise GroupedDesignNotFound(f"The grouped search found no feasible design: {search['stop']}", search)
    evidence, state = final["evidence"], final["state"]
    mg.install(state)
    slab = evidence["slab"]
    coupled = None
    if sp.FLOOR_TRANSFER is not None:
        from Design.SMRF_Coupled_Comparison import compare_transfer_to_coupled
        if verbose:
            print("[grouped] coupled load-path comparison of the selected design", flush=True)
        coupled = compare_transfer_to_coupled(slab["slab"], combination_actions=evidence["actions"])
    slab_inputs = driver._slab_strength_inputs_from_state(slab["slab"], cfg)
    slab_reinforcement = slab["reinforcement"]
    if slab_reinforcement is None:
        from Design.SMRF_Slab_Reinforcement import design_slab_reinforcement
        slab_reinforcement = design_slab_reinforcement(slab_inputs, None)   # no verified strip actions: stays open
    elf = evidence["elf"] or {}
    capacity = evidence["capacity"]
    arrangement = bar_layers.arrangement()
    record = {
        "schema_version": SCHEMA,
        "design_mode": "grouped",
        "geometry": driver._slab_geometry(),
        "materials": _materials(cfg, state),
        "member_groups": state.to_record(),
        "bar_layers": {"groups": {gid: rows for gid, rows in arrangement.items() if gid != "_basis"},
                       "basis": arrangement["_basis"]},
        "slab": slab["slab"],
        "floor_loads": _floor_loads(),
        "gravity_load_model": sp.effective_gravity_load_model(),
        "floor_sections": slab["description"],
        "floor_transfer": sp.FLOOR_TRANSFER,
        "slab_reinforcement": slab_reinforcement,
        "slab_reinforcement_inputs": slab_inputs,
        "slab_actions": sp.SLAB_ACTIONS,
        "seismic": {"sds": sp.ASCE_SDS, "sd1": sp.ASCE_SD1, "s1": sp.ASCE_S1, "r": sp.ASCE_R,
                    "site_label": getattr(sp, "SEISMIC_SITE_LABEL", None),
                    "risk_category": sp.ASCE_RISK_CATEGORY, "importance_factor": sp.ASCE_IE,
                    "design_basis": sp.seismic_design_basis(risk_category=cfg.demands.risk_category)},
        "demand": {"basis": "Signed gravity/seismic combinations, Ev and 100/30 effects; scope limitations in qualification",
                   "model_period_sec": evidence["period"], "base_shear_kip": elf.get("base_shear_kip"),
                   "story_forces_kip": elf.get("story_forces_kip")},
        "design_actions": {
            "basis": ("Signed simultaneous centerline elastic member actions of every physical member on its own group's "
                      "section, compression-positive axial forces; joint-face moment transport is separate"),
            "expected_combination_ids": evidence["expected_ids"], "combinations": evidence["actions"]},
        "member_strength": evidence["strengths"],
        "beam_end_strengths": beam_end_strengths(state),
        "capacity_design": capacity,
        "joint_checks": evidence["joint_checks"],
        "drift_screen": evidence["drift_screen"],
        "demand_basis": evidence["demand_basis"],
        "coupled_comparison": coupled,
        "dcr": {"column": final["dcr"]["column"], "beam": final["dcr"]["beam"],
                "governing": max(final["dcr"].values()),
                "candidate_screen_passed": bool(final["feasible"]),
                "basis": "the largest member-strength DCR of any column group and of any beam group, each on its own cage"},
        "constraints": final["constraints"],
        "quantities": final["quantities"],
        "margins": final["margins"],
        "stages": final["stages"],
        "detailing": group_detailing(state, capacity),
        "search": {"version": search["version"], "policy": search["policy"], "seed": seed,
                   "stop_reason": (search["stop"] or {}).get("reason"), "stop_detail": (search["stop"] or {}).get("detail"),
                   "counts": search["counts"], "elapsed_seconds": search["elapsed_seconds"],
                   "candidates": search["log"], "tradeoffs": search["tradeoffs"],
                   "interpretation": search["interpretation"]},
        "legacy_uniform_summary": _legacy_summary(state),
        "limitations": list(LIMITATIONS),
    }
    ops.wipe()
    record = _canonical(record)
    from Design.SMRF_Design_Evidence import analysis_input_signature
    signature = analysis_input_signature(record)
    record["design_actions"]["analysis_input_sha256"] = signature
    record["drift_screen"]["analysis_input_sha256"] = signature
    if record.get("coupled_comparison"):
        record["coupled_comparison"]["analysis_input_sha256"] = signature
    if identity is not None:
        record["request_identity"] = identity
    record["qualification"] = _canonical(qualify_grouped_design(record))
    record["dcr"]["accepted"] = record["qualification"]["accepted"]
    apply_grouped_design(record)                     # qualification restores the previous state; leave the design in force
    return record


# ---- applying a record -----------------------------------------------------------------------------------------
@contextlib.contextmanager
def scratch_state():
    """Run a block that installs a grouped record and put back whatever was in force before it."""
    previous = mg.active()
    saved = {key: getattr(sp, key, None) for key in _STATE_KEYS}
    try:
        yield
    finally:
        if previous is None:
            mg.clear()
        else:
            mg.install(previous)
        for key, value in saved.items():
            setattr(sp, key, value)


def _validated_slab(record):
    from Design.SMRF_Slab import evaluate_slab
    slab = record.get("slab")
    passed = {c["id"] for c in evaluate_slab(slab) if c["status"] == "pass"}
    if passed != {"slab_thickness_evidence", "slab_thickness_screen"}:
        raise ValueError("The grouped record lacks reproducible slab thickness evidence.")
    if (slab["concrete_unit_weight_kcf"] != sp.CONCRETE_UNIT_WEIGHT_KCF or slab["inputs"]["policy"]["fy_ksi"] != sp.FY_KSI):
        raise ValueError("The grouped record's slab material assumptions disagree with the current model.")
    for key, value in slab["inputs"]["geometry"].items():
        if record["geometry"].get(key) != value:
            raise ValueError(f"The grouped record's slab geometry.{key} disagrees with its frame.")
    return slab


def apply_grouped_design(record):
    """Install a saved grouped design: the member groups, the slab, the by-floor transfer and the slab layout.

    Everything is validated against the running model; a failure leaves the model as it was. The uniform
    section values of Structure_Parameters are withdrawn while the design is in force
    (Model.Member_Groups.install); nothing of a grouped record is written into them.
    """
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA or record.get("design_mode") != "grouped":
        raise ValueError(f"Not a grouped design record (schema {None if not isinstance(record, dict) else record.get('schema_version')!r}).")
    if any(key in record for key in ("sections", "reinforcement")):
        raise ValueError("A grouped record carries no uniform sections or reinforcement block; this one does and is refused.")
    if any(record.get("geometry", {}).get(key) != value for key, value in driver._slab_geometry().items()):
        raise ValueError("The grouped record's geometry disagrees with the current model.")
    material = record.get("materials") or {}
    if material.get("fy_ksi") != sp.FY_KSI or material.get("es_ksi") != sp.ES_KSI:
        raise ValueError("The grouped record's steel properties disagree with the current model.")
    aggregate = material.get("aggregate_size_in")
    if isinstance(aggregate, bool) or not isinstance(aggregate, (int, float)) or not math.isfinite(aggregate) or aggregate <= 0:
        raise ValueError("The grouped record lacks a valid aggregate size.")
    for key in ("beam_clear_cover_in", "col_clear_cover_in"):
        value = material.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 1.5:
            raise ValueError(f"The grouped record's {key} must be at least 1.5 in.")
    stacking = material.get("beam_bar_stacking") or {}
    state = mg.GroupedDesign.from_record(record.get("member_groups"))          # digest and per-member rows
    previous = mg.active()
    saved = {key: getattr(sp, key, None) for key in _STATE_KEYS}
    try:
        slab = _validated_slab(record)
        mg.install(state)
        gd._require_common_grades(state)
        driver._apply_slab(slab)
        sp.AGGREGATE_MAX_SIZE_IN = aggregate
        sp.REINFORCEMENT_SPECIFICATION = material["reinforcement_specification"]
        sp.MATERIAL_EXPOSURE = material["exposure"]
        sp.BEAM_CLEAR_COVER_IN, sp.COL_CLEAR_COVER_IN = material["beam_clear_cover_in"], material["col_clear_cover_in"]
        sp.BEAM_BAR_STACKING = stacking.get("mode", "none")
        sp.BEAM_BAR_MAX_LAYERS = int(stacking.get("max_layers", 1))
        sp.BEAM_BAR_LAYER_ORDER = stacking.get("layer_order", "blocked")
        if stacking.get("convention") != sp.beam_bar_stacking_convention():
            raise ValueError("The grouped record's beam bar stacking convention does not reproduce.")
        description = floor_sections.by_floor()
        if record.get("floor_sections") != description:
            raise ValueError("The grouped record's floor descriptions are not those of its member groups.")
        transfer = record.get("floor_transfer")
        if record.get("gravity_load_model") == "slab_transfer":
            from Design.SMRF_Floor_Transfer import validate_floor_transfers_by_floor
            area = mp.floor_area_in2() / 144.0
            validate_floor_transfers_by_floor(transfer, record["geometry"], description, sp.SLAB_THICKNESS_IN,
                                              sp.floor_dead_load_ksf() * area, sp.FLOOR_LIVE_LOAD_KSF * area)
            sp.FLOOR_TRANSFER = transfer
        else:
            if transfer is not None:
                raise ValueError("The grouped record carries a floor transfer but does not declare the slab_transfer load model.")
            sp.FLOOR_TRANSFER = None
        strength = record.get("slab_reinforcement")
        if strength is not None and strength.get("inputs"):
            from Design.SMRF_Slab_Reinforcement import evaluate_slab_reinforcement
            if evaluate_slab_reinforcement(strength)[0]["status"] != "pass":
                raise ValueError("The grouped record's slab reinforcement does not reproduce.")
            if strength["inputs"]["slab"]["thickness_in"] != sp.SLAB_THICKNESS_IN:
                raise ValueError("The grouped record's slab reinforcement belongs to a different slab thickness.")
        sp.SLAB_REINFORCEMENT = strength
        sp.SLAB_ACTIONS = record.get("slab_actions")
        sp.JOINT_SHEAR_CATEGORIES = None                 # the scissors joint springs are not wired for groups
        rows = bar_layers.arrangement()
        live = _canonical({"groups": {gid: value for gid, value in rows.items() if gid != "_basis"}, "basis": rows["_basis"]})
        if differences(record.get("bar_layers"), live):
            raise ValueError("The grouped record's beam bar rows do not reproduce: "
                             + "; ".join(differences(record.get("bar_layers"), live)[:3]))
    except Exception:
        if previous is None:
            mg.clear()
        else:
            mg.install(previous)
        for key, value in saved.items():
            setattr(sp, key, value)
        raise
    return record


def is_grouped_record(record):
    return isinstance(record, dict) and record.get("design_mode") == "grouped"


def apply_record(record):
    """Install a saved design of either kind: a grouped record through ``apply_grouped_design``, a uniform one
    through Design_Driver.apply_design (after leaving any grouped design that was in force)."""
    if is_grouped_record(record):
        return apply_grouped_design(record)
    mg.clear()
    return driver.apply_design(record)


# ---- qualification ---------------------------------------------------------------------------------------------
def _cfg_view(record):
    """The two policy blocks the recomputation reads, from the record's own request identity."""
    policy = (record.get("request_identity") or {}).get("policy") or {}
    return types.SimpleNamespace(capacity=types.SimpleNamespace(**(policy.get("capacity") or {})),
                                 dcr=types.SimpleNamespace(**(policy.get("dcr") or {})))


def _rows_max(rows):
    return max(rows["per_layer"]) if rows.get("per_layer") else None


def group_detailing_checks(record, state, capacity):
    """ACI 318-19 18.6 / 18.7 dimensional and bar rules, one group at a time, and the pairs each beam actually meets.

    Design.SMRF_Detailing.evaluate_detailing prices one beam against one column. Here each beam group and
    each column group is priced on its own section, cage, hoops and declared end zones; the two rules
    that involve a beam and the column it frames into (clear span 18.6.2.1(a), projection 18.6.2.1(c))
    are evaluated on every physical beam at its own two joints and reported per beam group at its worst.
    """
    from Design.SMRF_Detailing import evaluate_detailing
    material = record["materials"]
    declared = (record.get("detailing") or {}).get("groups") or {}
    arrangement = bar_layers.arrangement()
    checks, scope_done = [], False
    PAIR_IDS = {"beam.clear_span", "beam.column_projection"}
    OPEN_IDS = {"column.confinement_area_and_support", "beam.hoop_layout_and_axial_applicability",
                "detailing.anchorage_splices_and_cover"}

    def inputs_of(design, gid):
        values = {"b_in": design.b_in, "h_in": design.h_in, "fc_ksi": design.fc_ksi,
                  "bar_db_in": sp.rebar_diameter(design.bar_size), "bar_area_in2": sp.rebar_area(design.bar_size),
                  "stirrup_db_in": sp.rebar_diameter(design.stirrup_bar_size),
                  "n_top": design.top_bars, "n_bottom": design.bot_bars, "n_side_per_face": design.side_bars,
                  "hoop_spacing_in": design.stirrup_spacing_in, "aggregate_size_in": material.get("aggregate_size_in"),
                  "clear_cover_in": material["col_clear_cover_in" if design.is_column else "beam_clear_cover_in"]}
        for key, value in (declared.get(gid) or {}).items():
            if key not in values and value is not None:
                values[key] = value
        return values

    geometry = {"span_x_in": record["geometry"]["bay_x_in"], "span_y_in": record["geometry"]["bay_y_in"]}
    scope = {"fy_ksi": material.get("fy_ksi"), "normalweight": material.get("normalweight")}
    for gid, group in sorted(state.groups.items()):
        design = state.designs[gid]
        member = state.member(group["member_tags"][0])
        if design.is_column:
            joint = mg.joint_members(member.story_or_floor, member.grid_i, member.grid_j)
            partner = next(joint[key] for key in ("beam_x_minus", "beam_x_plus", "beam_y_minus", "beam_y_plus")
                           if joint[key] is not None)
            own = inputs_of(design, gid)
            result = evaluate_detailing({"material": scope, "geometry": geometry, "column": own,
                                         "beam": inputs_of(partner.design, partner.group_id)})
            prefix = "column."
        else:
            core = mp.joint_core_column(member.story_or_floor, member.grid_i, member.grid_j)
            own = inputs_of(design, gid)
            rows = arrangement[gid]
            # The widest row is what has to fit across the width (bars beyond the lanes go to a second layer).
            own["n_top"], own["n_bottom"] = _rows_max(rows["top"]) or design.top_bars, _rows_max(rows["bottom"]) or design.bot_bars
            strengths = group_checks.beam_flexural_strengths(design, rows)
            positive, negative = strengths["positive"]["phi_mn_kip_in"], strengths["negative"]["phi_mn_kip_in"]
            own.update({"mn_positive_left_kipin": positive, "mn_negative_left_kipin": negative,
                        "mn_positive_right_kipin": positive, "mn_negative_right_kipin": negative,
                        "mn_positive_min_along_kipin": positive, "mn_negative_min_along_kipin": negative,
                        "continuous_top_bars": design.top_bars, "continuous_bottom_bars": design.bot_bars})
            result = evaluate_detailing({"material": scope, "geometry": geometry, "beam": own,
                                         "column": inputs_of(core.design, core.group_id)})
            prefix = "beam."
        for check in result:
            name = check["id"]
            if name == "detailing.material_scope":
                if not scope_done:
                    checks.append(check)
                    scope_done = True
                continue
            if name in PAIR_IDS or name in OPEN_IDS or not (name.startswith(prefix) or name == "detailing.geometry_complete"):
                continue
            location = check.get("location") or ""
            checks.append({**check, "location": f"{gid}/{location}" if location else gid})
        if not design.is_column:
            # The members of the group at their own joints.
            d = design.h_in - mp.longitudinal_cover_in(design)
            worst_span, worst_projection = None, None
            for tag in group["member_tags"]:
                beam = state.member(tag)
                axis = mp.beam_axis(beam)
                clear = mp.beam_clear_span_in(beam)
                if worst_span is None or clear < worst_span[0]:
                    worst_span = (clear, tag)
                k, i, j = beam.story_or_floor, beam.grid_i, beam.grid_j
                for gi, gj in (((i, j), (i + 1, j)) if axis == "x" else ((i, j), (i, j + 1))):
                    along_x, along_y = mp.joint_core_dimensions(k, gi, gj)
                    depth, width = (along_x, along_y) if axis == "x" else (along_y, along_x)
                    projection, limit = max(0.0, (design.b_in - width) / 2.0), min(width, 0.75 * depth)
                    if worst_projection is None or projection - limit > worst_projection[0] - worst_projection[1]:
                        worst_projection = (projection, limit, tag)
            checks.append(make_check("beam.clear_span", "ACI 318-19 18.6.2.1(a)", worst_span[0], 4.0 * d, ">=", "in", gid,
                                     details={"member_tag": worst_span[1],
                                              "basis": "the shortest clear span of the group between the faces of its own end joints"}))
            checks.append(make_check("beam.column_projection", "ACI 318-19 18.6.2.1(c)", worst_projection[0],
                                     worst_projection[1], "<=", "in", gid, details={"member_tag": worst_projection[2]}))
    return checks


def _capacity_completion_checks(capacity, state):
    """The cage-level items the detailing evaluator leaves open, settled per group from the recomputed capacity design."""
    checks = []
    for gid, data in sorted((capacity.get("columns") or {}).items()):
        confinement, hoops = data.get("confinement") or {}, data.get("hoops")
        if not hoops or not confinement:
            checks.append(not_evaluated("column.confinement_area_and_support", "ACI 318-19 18.7.5.2--18.7.5.4",
                                        "no hoops are selected for this group", gid))
            continue
        by_direction = hoops.get("by_direction") or {}
        if by_direction:
            tightest = min(by_direction.values(), key=lambda v: v["ash_provided_per_in"] - v["ash_required_per_in"])
            demand, provided = tightest["ash_required_per_in"], tightest["ash_provided_per_in"]
        else:
            demand, provided = confinement["ash_ratio_required"] * confinement["bc_in"], hoops["ash_provided_per_in"]
        checks.append(make_check("column.confinement_area_and_support", "ACI 318-19 18.7.5.2--18.7.5.4", demand, provided,
                                 "<=", "in2/in", gid,
                                 details={"legs": hoops["legs"], "bar_size": hoops["bar_size"], "spacing_in": hoops["spacing_in"],
                                          "high_axial": confinement["high_axial"]}))
    for gid, data in sorted((capacity.get("beams") or {}).items()):
        if data.get("hoops"):
            checks.append(make_check("beam.hoop_layout_and_axial_applicability", "ACI 318-19 18.6.2.1 / 18.6.4 / 18.6.5",
                                     int(bool(data.get("section_adequate"))), 1, "==", location=gid,
                                     details={"hoops": data["hoops"],
                                              "axial_basis": "frame beams carry no factored axial load above Ag fc/10"}))
        else:
            checks.append(not_evaluated("beam.hoop_layout_and_axial_applicability", "ACI 318-19 18.6.4",
                                        "no hoops are selected for this group", gid))
    anchorage = [c for c in capacity.get("checks", []) if c["id"] in ("joint.through_bar_depth", "joint.terminating_bar_hook")]
    checks.append(make_check("detailing.anchorage_splices_and_cover",
                             "ACI 318-19 18.6.3.3 / 18.7.4.4 / 18.8.5 / 20.5 / 25.4--25.5",
                             int(bool(anchorage) and all(c["status"] == "pass" for c in anchorage)), 1, "==",
                             details={"anchorage_items": len(anchorage), "splices": capacity.get("splices"),
                                      "scope": "hooked anchorage, through-bar depth, splice type and location per group and "
                                               "cover; congestion and placement are detailing.congestion_and_placement"}))
    open_items = [c for c in capacity.get("checks", []) if c.get("status") == "not_evaluated"]
    failed = [c for c in capacity.get("checks", []) if c.get("status") == "fail"]
    if open_items and not failed:
        checks.append(not_evaluated("qualification.joint_capacity_completion", "ACI 318-19 18.7.3; 18.8",
                                    "capacity-design items without supporting evidence: " + ", ".join(
                                        f"{c.get('id')}@{c.get('location', '')}" for c in open_items[:8])))
    else:
        checks.append(make_check("qualification.joint_capacity_completion", "ACI 318-19 18.7.3; 18.8",
                                 int(bool(capacity.get("accepted"))), 1, "==",
                                 details={"method_version": capacity.get("method_version"),
                                          "failed": [f"{c['id']}@{c.get('location', '')}" for c in failed[:12]]}))
    columns = capacity.get("columns") or {}
    checks.append(make_check("qualification.column_capacity_shear", "ACI 318-19 18.7.6",
                             int(bool(columns) and all(c.get("hoops") and c.get("section_adequate") for c in columns.values())),
                             1, "==", details={"groups": len(columns)}))
    return checks


def _open_checks(record, capacity):
    """What a grouped design leaves open whatever its numbers: reviews that have not been made."""
    checks = [
        not_evaluated("qualification.strength_model_verification", "ACI 318-19 Chapters 6, 18, 21, 22",
                      "Verify biaxial P-M capacity, force signs and joint-face actions, beam axial effects, strength "
                      "reduction factors and cracked-stiffness assumptions independently."),
        {**not_evaluated("qualification.detailing_model_consistency", "Research model-to-design consistency",
                         "Independent validation: confirm that each group's confinement zones, bar positions, cover and "
                         "hoops propagate consistently to section strength, IMK calibration and exports."),
         "details": {"reason": "awaiting independent validation"}},
        not_evaluated("detailing.congestion_and_placement", "ACI 318-19 25.2 / 26.6; constructability",
                      "Bar-placement drawings, mechanical-splice staggering and congestion at joints are not designed."),
    ]
    layout = (record.get("slab_reinforcement") or {}).get("layout")
    if layout is not None:
        checks.append(make_check("qualification.slab_contribution", "ACI 318-19 18.7.3.2 / 6.3.2", 1, 1, "==",
                                 details={"basis": "slab mats within each beam's own effective flange, developed as continuous "
                                                   "mats at interior ends and by the perimeter hook at exterior ends, in the "
                                                   "strong-column rule, joint shear and the beam hinges"}))
    else:
        checks.append(not_evaluated("qualification.slab_contribution", "ACI 318-19 18.7.3.2 / 6.3.2",
                                    "Slab reinforcement is not established; beam strengths are those of the bare beams with "
                                    "the flange concrete. Do not read this as zero slab strength."))
    changing = sorted(key for key, t in (capacity.get("transitions") or {}).items() if t.get("kind") != "same_cage")
    if changing:
        check = not_evaluated("detailing.column_transition_rules_reviewed",
                              "ACI 318-19 10.7.4, 10.7.6.4, 15.2.6, 18.7.4.4, 25.4, 25.5",
                              f"{len(changing)} band boundary transition(s) move, splice or stop column bars under the declared "
                              f"rules {transition_rules.RULES_ID}, which await engineering review")
        open_items = sorted({item for key in changing for item in (capacity["transitions"][key].get("not_established") or [])})
        check["details"].update({"category": "awaiting review of the declared transition rules", "transitions": changing,
                                 "kinds": {key: capacity["transitions"][key].get("kind") for key in changing},
                                 "not_established": open_items})
        checks.append(check)
    return checks


def qualify_grouped_design(record):
    """The fail-closed checklist of a grouped record. The record is installed for the recomputation and the
    model is put back as it was. A record that cannot be installed is not accepted, with the reason."""
    from Design.SMRF_Qualification import SCOPE, _apply_verification_assertions
    base = {"schema_version": QUALIFICATION_VERSION, "implementation_stage": "partial_not_release_ready",
            "scope": {**SCOPE, "geometry": "regular rectangular grid; sections and reinforcement by story band and plan location"}}
    with scratch_state():
        try:
            apply_grouped_design(record)
        except Exception as exc:                          # noqa: BLE001 -- a record that does not install is a finding
            checks = [make_check("grouped.record_installs", "Evidence integrity", 0, 1, "==",
                                 details={"error": f"{type(exc).__name__}: {exc}"})]
            return {**base, "checks": checks, **summarize_checks(checks)}
        checks = _qualification_checks(record, mg.active())
        ops.wipe()
    checks = _apply_verification_assertions(record, checks)
    return {**base, "checks": checks, **summarize_checks(checks)}


def _qualification_checks(record, state):
    from Design.SMRF_Design_Evidence import analysis_input_signature, slab_strength_inputs, validated_combinations
    from Design.SMRF_Joints import evaluate_joints
    from Design.SMRF_Qualification import reinforcement_consistency_checks
    checks = [make_check("grouped.record_installs", "Evidence integrity", 1, 1, "==",
                         details={"groups": len(state.groups), "members": len(state.members()),
                                  "member_groups_sha256": state.identity(),
                                  "basis": "assignments, group designs and per-member rows match the block's digest; slab, "
                                           "floor descriptions, transfer, slab layout and bar rows reproduce"})]
    cfg = _cfg_view(record)
    geometry, loads = record["geometry"], record["floor_loads"]
    # Declared material scope (the member-by-member bar database is the group designs themselves).
    checks.append(next(c for c in reinforcement_consistency_checks({"materials": record["materials"]})
                       if c["id"] == "reinforcement.declared_material_scope"))

    # 1. Solved actions: bound to this frame, canonical in id and factors.
    actions, expected_ids, transfers = None, None, None
    try:
        actions, expected_ids = validated_combinations(record)
        saved_ids = (record.get("design_actions") or {}).get("expected_combination_ids")
        if [a["id"] for a in actions] != expected_ids or saved_ids != expected_ids:
            raise ValueError("the solved combinations are not exactly the canonical set, in order")
        if record.get("gravity_load_model") == "slab_transfer":
            from Design.SMRF_Floor_Transfer import validate_floor_transfers_by_floor
            area = mp.floor_area_in2() / 144.0
            transfers = validate_floor_transfers_by_floor(record["floor_transfer"], geometry, floor_sections.by_floor(),
                                                          record["slab"]["thickness_in"], loads["floor_dead_load_ksf"] * area,
                                                          loads["floor_live_load_ksf"] * area)
        checks.append(make_check("joint.reproducible_actions", "ACI 318-19 18.7--18.8", 1, 1, "==",
                                 details={"combinations": len(actions)}))
    except (KeyError, ValueError, TypeError) as exc:
        actions = None
        checks.append(not_evaluated("joint.reproducible_actions", "ACI 318-19 18.7--18.8", str(exc)))

    # 2. Capacity design, member strengths and joints: recomputed from the record; only the recomputed objects are read.
    capacity = None
    if actions is not None:
        try:
            recomputed = capacity_design.build_group_capacity_design(actions, expected_ids, cfg, transfers, state)
            found = differences(record.get("capacity_design"), _canonical(recomputed))
        except Exception as exc:                      # noqa: BLE001 -- any failure is a finding, not a pass
            recomputed, found = None, [f"recomputation failed: {type(exc).__name__}: {exc}"]
        checks.append(make_check("qualification.capacity_evidence_recomputed", "Evidence integrity", int(not found), 1, "==",
                                 details={"basis": "group capacity design rebuilt from the record's member groups, slab layout, "
                                                   "transfers and saved combination actions and compared with the saved evidence",
                                          "differences": found}))
        hoops = None if recomputed is None else capacity_design.hoop_updates(recomputed, state)
        checks.append(make_check("qualification.hoops_match_design", "Evidence integrity",
                                 int(hoops is not None and not hoops), 1, "==",
                                 details={"basis": "the hoops every group's members carry (model, IMK calibration, exports) must "
                                                   "be the hoops the recomputed capacity design selects for that group",
                                          "groups_that_differ": sorted(hoops or {})}))
        if recomputed is not None and not found and not hoops:
            capacity = recomputed
        try:
            strengths = group_checks.strength_checks(actions, cfg, state)
            found = differences({"worst": (record.get("member_strength") or {}).get("worst")},
                                _canonical({"worst": strengths["worst"]}))
            checks.append(make_check("qualification.member_strength_recomputed", "Evidence integrity", int(not found), 1, "==",
                                     details={"differences": found}))
            for gid, result in sorted(strengths["groups"].items()):
                if result["member_type"] == "column":
                    checks.append(make_check("candidate.column_strength_screen", "Preliminary current member-strength implementation",
                                             max(result["pm"]["dcr"], result["shear"]["dcr"]), 1.0, units="DCR", location=gid,
                                             details={"pm": result["pm"], "shear": result["shear"]}))
                else:
                    checks.append(make_check("candidate.beam_strength_screen", "Preliminary current member-strength implementation",
                                             max(result["flexure_positive"]["dcr"], result["flexure_negative"]["dcr"],
                                                 result["shear"]["dcr"]), 1.0, units="DCR", location=gid,
                                             details={k: result[k] for k in ("flexure_positive", "flexure_negative", "shear")}))
        except Exception as exc:                      # noqa: BLE001
            checks.append(not_evaluated("qualification.member_strength_recomputed", "Evidence integrity",
                                        f"{type(exc).__name__}: {exc}"))
    if capacity is not None:
        checks.extend(evaluate_joints(gd.joint_check_inputs(capacity, state)))
        checks.extend(capacity["checks"])
        checks.extend(_capacity_completion_checks(capacity, state))
        checks.extend(group_detailing_checks(record, state, capacity))
    else:
        for name, clause in (("qualification.joint_capacity_completion", "ACI 318-19 18.7.3; 18.8"),
                             ("qualification.column_capacity_shear", "ACI 318-19 18.7.6"),
                             ("detailing.group_checks", "ACI 318-19 18.6; 18.7")):
            checks.append(not_evaluated(name, clause, "No usable capacity design: the saved evidence does not reproduce."))

    # 3. Weights, story strengths and the vertical evidence: recomputed and compared, then evaluated.
    ledger = mp.weight_ledger()
    found = differences(loads.get("ledger"), _canonical(ledger))
    checks.append(make_check("grouped.weight_ledger_recomputed", "Evidence integrity", int(not found), 1, "==",
                             details={"differences": found, "total_seismic_weight_kip": ledger["total_seismic_weight_kip"]}))
    regularity = (record.get("demand_basis") or {}).get("regularity") or {}
    saved_strength = regularity.get("lateral_strength_distribution") or {}
    try:
        # Only the numbers are compared; the applicability status of the model is the record's own assertion policy.
        stories = _canonical(gd.story_strength_distribution(None))
        found = differences({k: saved_strength.get(k) for k in ("one_side_fraction", "stories")},
                            {k: stories[k] for k in ("one_side_fraction", "stories")})
        # The applicability status of the strength model is the record's own (its assertion policy), carried over.
        stories["applicability"] = saved_strength.get("applicability")
        vertical = _canonical(gd.vertical_regularity(
            record["drift_screen"], {key: (record.get("demand") or {}).get(key) for key in ("story_forces_kip", "base_shear_kip")},
            stories, ledger))
        found += differences(regularity.get("vertical_evidence"), vertical)
    except Exception as exc:                          # noqa: BLE001
        vertical, found = None, [f"recomputation failed: {type(exc).__name__}: {exc}"]
    checks.append(make_check("demands.story_evidence_recomputed", "Evidence integrity", int(not found), 1, "==",
                             details={"basis": "story-by-story line strengths from the installed beams, story stiffness from the "
                                               "saved drift runs and ELF story shears, story mass from the member ledger",
                                      "differences": found}))
    from Design.SMRF_Demands import evaluate_demand_basis
    checks.extend(evaluate_demand_basis(record))
    if vertical is not None and not found:
        from Design.SMRF_Demands import seismic_design_category
        seismic, policy = record.get("seismic") or {}, ((record.get("request_identity") or {}).get("policy") or {}).get("demands") or {}
        try:
            sdc = seismic_design_category(seismic["sds"], seismic["sd1"], seismic["s1"],
                                          policy.get("risk_category") or seismic.get("risk_category") or "II")
        except (KeyError, ValueError, TypeError):
            sdc = None
        from Design.SMRF_Common import assertion_provenance_valid
        verification = (((record.get("request_identity") or {}).get("policy") or {}).get("verification")
                        or (record.get("demand_basis") or {}).get("verification") or {})
        asserted = (assertion_provenance_valid(verification) and verification.get("story_strength_model_verified") is True
                    and (saved_strength.get("applicability") or {}).get("status") == "verified")
        checks.extend(gd.regularity_checks(vertical, sdc, strength_model_verified=asserted))
    else:
        for name in ("demands.vertical_stiffness_regularity", "demands.vertical_strength_regularity"):
            checks.append(not_evaluated(name, "ASCE 7-22 Table 12.3-2", "the saved story evidence does not reproduce"))

    found = differences(record.get("beam_end_strengths"), _canonical(beam_end_strengths(state)))
    checks.append(make_check("grouped.beam_end_strengths_recomputed", "Evidence integrity", int(not found), 1, "==",
                             details={"basis": "nominal hogging and sagging strength of every physical beam at each end, on its "
                                               "own section, bar rows, flange and the common slab layout: what its hinges yield at",
                                      "differences": found}))

    # 4. Slab, floor load path, beam envelopes.
    from Design.SMRF_Slab import evaluate_slab
    checks.extend(evaluate_slab(record.get("slab")))
    if record.get("gravity_load_model") == "slab_transfer" or record.get("floor_transfer") is not None:
        checks.append(make_check("floor.transfer_load_integrity",
                                 "Recomputed frame load inventory and force/moment balance, floor by floor",
                                 int(transfers is not None), 1, "==",
                                 details={"distinct_floors": len((record.get("floor_sections") or {}).get("distinct") or {})}))
        from Design.SMRF_Coupled_Comparison import evaluate_coupled_comparison
        checks.extend(evaluate_coupled_comparison(record))
    from Design.SMRF_Beam_Actions import evaluate_saved_beam_bending
    checks.append(evaluate_saved_beam_bending(record))
    strength = record.get("slab_reinforcement") or {}
    strength_inputs = strength.get("inputs", {}).get("slab")
    if strength_inputs is not None or record.get("slab_reinforcement_inputs") is not None:
        try:
            from Design.SMRF_Slab_Reinforcement import slab_input_signature
            expected = slab_input_signature(slab_strength_inputs(record))
            saved = [value for value in (strength_inputs, record.get("slab_reinforcement_inputs")) if value is not None]
            matches = all(slab_input_signature(value) == expected for value in saved)
        except (KeyError, ValueError, TypeError, OverflowError):
            matches = False
        checks.append(make_check("slab_strip.frame_input_consistency", "Slab strength / current frame inputs", int(matches), 1, "=="))
        if not matches:
            strength = None
    from Design.SMRF_Slab_Reinforcement import evaluate_slab_reinforcement
    checks.extend(evaluate_slab_reinforcement(strength))
    from Design.SMRF_Slab_Actions import evaluate_slab_actions
    checks.extend(evaluate_slab_actions((strength or {}).get("inputs", {}).get("demand_evidence") or record.get("slab_actions")))

    # 5. Drift and stability results bound to this frame.
    drift = record.get("drift_screen")
    try:
        drift_matches = isinstance(drift, dict) and drift.get("analysis_input_sha256") == analysis_input_signature(record)
    except (KeyError, ValueError, TypeError):
        drift_matches = False
    if drift_matches and drift.get("checks"):
        checks.extend(drift["checks"])
    else:
        checks.append(not_evaluated("demands.drift_and_stability_results", "ASCE 7-22 12.8.6--12.8.7",
                                    "Independent QEx/QEy drift/stability results are not available."))

    # 6. The quantities the search compared candidates on.
    try:
        amounts = _canonical(gd.quantities(state, record.get("slab"), record.get("slab_reinforcement"), capacity))
        found = (differences(record.get("quantities"), amounts) if capacity is not None
                 else ["no usable capacity design to price the transitions"])
    except Exception as exc:                          # noqa: BLE001
        found = [f"recomputation failed: {type(exc).__name__}: {exc}"]
    checks.append(make_check("grouped.quantities_recomputed", "Evidence integrity", int(not found), 1, "==",
                             details={"differences": found}))
    checks.extend(_open_checks(record, capacity or {}))
    return checks


def require_accepted_grouped_design(record):
    """Recompute the checklist; a saved accepted=True is insufficient."""
    qualification = qualify_grouped_design(record)
    if not qualification["accepted"]:
        counts = qualification["counts"]
        raise RuntimeError(f"Grouped SMRF design not qualified: {counts['fail']} failed and {counts['not_evaluated']} "
                           "unevaluated checks. No time-history analysis was launched.")
    return qualification


# ---- design and cache ------------------------------------------------------------------------------------------
def design_grouped_structure(cfg, seed_state, seed, policy=None, log_path=None, identity=None, verbose=True):
    """Run the grouped search from a seed and return its record. Raises GroupedDesignNotFound with the search."""
    search = gs.search(seed_state, cfg, policy, log_path=log_path, verbose=verbose)
    return build_record(search, cfg, seed, identity=identity, verbose=verbose)


def load_or_create_grouped_design(design_path, cfg, seed_state, seed, policy=None, verbose=True):
    """Read a cached grouped record, or run the grouped search and write it. Returns (record, created).

    The candidate log is written beside the record as the search runs (``grouped_candidates.jsonl``), so a
    search that ends without a feasible design, or is interrupted, leaves every candidate it evaluated;
    a log that already exists is never appended to or replaced: a new attempt takes a new directory.
    """
    design_path = Path(design_path)
    identity = grouped_request_identity(cfg, seed, policy)
    if design_path.exists():
        record = json.loads(design_path.read_text(encoding="utf-8"))
        if record.get("schema_version") != SCHEMA or (record.get("request_identity") or {}).get("sha256") != identity["sha256"]:
            raise RuntimeError("The existing grouped design uses different inputs, source, grouping, seed or search policy. "
                               "Preserve it and choose a new output directory.")
        apply_grouped_design(record)
        return record, False
    design_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = design_path.with_name(f".{design_path.name}.lock")
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"The grouped design is already reserved, or an interrupted lock needs review: {lock_path}") from exc
    temporary = design_path.with_name(f".{design_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with lock:
            lock.write(json.dumps({"pid": os.getpid(), "request_sha256": identity["sha256"]}))
        record = design_grouped_structure(cfg, seed_state, seed, policy, log_path=design_path.with_name(CANDIDATE_LOG_NAME),
                                          identity=identity, verbose=verbose)
        temporary.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n", encoding="utf-8")
        if design_path.exists():
            raise RuntimeError("The grouped design destination appeared during the search; the existing artifact is preserved.")
        temporary.rename(design_path)
        return record, True
    finally:
        if temporary.exists():
            temporary.unlink()
        lock_path.unlink()
