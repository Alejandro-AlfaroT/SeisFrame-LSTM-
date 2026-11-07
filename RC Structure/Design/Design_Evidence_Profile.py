"""Per-run design-evidence profile: the explicit, validated slab evidence configuration of one run (2026-11-01).

The default run computes single-mesh slab actions and asserts nothing: no slab layout is selected and the strong-column
checks that need the developed slab stay unknown. A profile changes that only by what it declares, in one JSON file:

  slab_refinement          the numerical mesh refinement plan of the slab actions
                           (Design.SMRF_Slab_Refinement: an explicit mesh plan or a named recipe), loaded apart from
                           the assertions, so numerical evidence can be produced while every assertion stays false. Its
                           tolerance is the declared investigation criterion (``slab_refinement_basis``), not an ACI
                           requirement or a validated error bound.
  slab_action_assertions   the engineering assertions (Design.Config.SlabActionAssertions), every field explicit. A true
                           flag needs a reviewed profile: review_status "reviewed_for_declared_scope", the review block
                           (who, when, on what basis, the decision and the evidence files with their sha256, checked on
                           load), the assertion provenance equal to that review (never filled in here), a refinement
                           plan, and the floor section sets the review covers.
  scope                    the case (id, and the sha256 of the run's --case-file, or of its --manifest file for a
                           manifest run), the analysis profile, the geometry, the slab materials and
                           loads and the slab policy (live load patterning, transfer mesh, thickness ladder) the profile
                           covers, checked against the run before any evaluation; and, for assertions, the floor
                           section signatures (Design.SMRF_Floor_Sections) and the slab thickness reviewed (one
                           thickness of the declared ladder). A candidate
                           with a floor outside them, or another slab thickness, is evaluated unasserted
                           (FloorAnalysisConfig.slab_assertion_floor_sections, slab_assertion_thickness_in).
  provenance, kind,        who prepared the file, when and why; ``kind`` "unreviewed_example", "test_fixture" (loaded
  review_status            only where a caller allows it: the runner refuses it) or "reviewed"; the review status.

Rejected on load: unknown, missing or repeated keys at every level, a boolean given as anything but a JSON boolean,
blank or undated provenance, a malformed plan (key sets, tolerances, the recipe name, its levels and shell budget, an
explicit plan's meshes), an asserted flag without a matching review, a reviewed thickness off the declared ladder, a
review whose evidence files are missing or differ from their recorded hash, a PROBE stamp in the assertions or in any
review block outside a test fixture. A scope that differs from the run is refused by the runner, which also resolves the
plan on the seed's floors before any evaluation.
Nothing is inherited from the PROBE configuration (Design.Verify_Designs.probe_config) and no approver is synthesized.

The profile enters the run before any evaluation: apply() returns the DesignConfig it implies (its slab_actions and
floor_analysis are in the evaluation identity), and summary() is the block the run's request carries (the generation
basis carries its id, kind, review status, asserted flag and ``identity_sha256``, which covers every field and the
hashes of the referenced evidence), so a changed plan, assertion,
review or scope is another identity and invalidates the cache and any checkpoint made under the old one. Nothing here
verifies anything: a reviewed profile records a decision taken elsewhere, for the scope it names.
"""
from __future__ import annotations

import copy
import datetime
import hashlib
import json
import math
import re
from dataclasses import fields, replace
from pathlib import Path

import Structure_Parameters as sp

SCHEMA = "design_evidence_profile_v1"
KINDS = ("unreviewed_example", "test_fixture", "reviewed")
# Schema v2 (2026-11-01): the procedure-based slab evidence path (Design.SMRF_Slab_Procedure). A procedure block
# replaces the floor section list and the single reviewed thickness; the kind "reviewed_procedure" replaces "reviewed".
SCHEMA_V2 = "design_evidence_profile_v2"
SCHEMA_V3 = "design_evidence_profile_v3"
COMPUTED_KIND = "computed_common_actions_v1"
KINDS_V2 = ("unreviewed_example", "test_fixture", "reviewed_procedure")
_FIXTURE_TEXT = re.compile(r"TEST\s+FIXTURE", re.IGNORECASE)
REVIEW_STATUSES = ("unreviewed", "under_review", "reviewed_for_declared_scope")
TOP_KEYS = {"schema", "profile_id", "kind", "review_status", "provenance", "scope", "slab_refinement",
            "slab_refinement_basis", "slab_action_assertions", "review"}
PROVENANCE_KEYS = {"prepared_by", "prepared_date", "purpose"}
DECLARATION_KEYS = {"declared_by", "declaration_date", "declaration_basis"}
SCOPE_KEYS = {"case_id", "case_sha256", "profile", "geometry", "materials", "loads", "slab_policy",
              "floor_sections_sha256", "slab_thickness_in"}
SLAB_POLICY_KEYS = {"live_load_patterning", "transfer_mesh_per_bay", "minimum_thickness_in", "maximum_thickness_in",
                    "thickness_increment_in"}
GEOMETRY_KEYS = {"num_bay_x", "num_bay_y", "num_floor", "bay_x_in", "bay_y_in", "story_h_in"}
MATERIAL_KEYS = {"slab_fy_ksi", "concrete_unit_weight_kcf"}
LOAD_KEYS = {"floor_live_load_ksf", "superimposed_dead_load_ksf"}
REVIEW_KEYS = {"reviewed_by", "review_date", "basis", "decision", "evidence"}
EVIDENCE_KEYS = {"path", "sha256", "description"}
ASSERTION_FLAGS = ("analysis_applicability_verified", "all_floors_enveloped", "load_pattern_envelope_verified",
                   "spatial_envelope_per_unit_width", "twisting_moment_resolution_verified", "zero_membrane_force_verified",
                   "verified", "two_way_shear_path_assessed", "slab_column_local_steel_assessed")
# Flags added after profiles were first written (2026-11-01): a profile that predates one omits it and reads it as
# False. The key is never inserted into the profile's data, so such a profile keeps its identity.
LEGACY_OPTIONAL_FLAGS = ("slab_column_local_steel_assessed",)
ASSERTION_TEXT = ("asserted_by", "assertion_date", "assertion_basis")
DEFAULT_BASIS = ("no design-evidence profile: single-mesh slab actions, no engineering assertions, no slab layout "
                 "selected, the strong-column checks that need the developed slab stay unknown")
_TOLERANCE = 1e-9


class EvidenceProfileError(ValueError):
    """A design-evidence profile that cannot be used, with the first reason."""


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _keys(value, expected, where):
    if not isinstance(value, dict):
        raise EvidenceProfileError(f"{where} must be a JSON object")
    missing, unknown = sorted(expected - set(value)), sorted(set(value) - expected)
    if missing or unknown:
        raise EvidenceProfileError(f"{where}: missing keys {missing}, unknown keys {unknown}")


def _text(value, where):
    if not isinstance(value, str) or not value.strip():
        raise EvidenceProfileError(f"{where} must be a non-blank string")
    return value


def _date(value, where):
    _text(value, where)
    try:
        if datetime.date.fromisoformat(value).isoformat() != value:
            raise ValueError
    except ValueError as exc:
        raise EvidenceProfileError(f"{where} must be an ISO date YYYY-MM-DD, not {value!r}") from exc
    return value


def _number(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise EvidenceProfileError(f"{where} must be a finite number")
    return float(value)


def _integer(value, where):
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceProfileError(f"{where} must be an integer")
    return value


def _sha(value, where):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise EvidenceProfileError(f"{where} must be a 64-character lowercase sha256")
    return value


_PROBE_WORD = re.compile(r"(?<![A-Za-z0-9_])PROBE(?![A-Za-z0-9_])")


def _probe_stamp(*values):
    """Whether a value carries the PROBE stamp (Design.Verify_Designs.PROBE, Design.Config.PROBE_SLAB_REFINEMENT's
    basis): the upper-case word PROBE. A word such as "probed" or a name such as "Probert" is not one."""
    return any(isinstance(v, str) and _PROBE_WORD.search(v) for v in values)


def _strings(value):
    """Every declared text, including nested provenance, evidence and dispositions."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _probe_plan(plan):
    """Whether a refinement plan is the PROBE configuration's (Design.Config.PROBE_SLAB_REFINEMENT): stamped, or the
    same numerical plan apart from its tolerance wording."""
    from Design.Config import PROBE_SLAB_REFINEMENT

    def numbers(value):
        return {k: v for k, v in value.items() if k != "tolerance_basis"}
    return _probe_stamp(plan.get("tolerance_basis")) or numbers(plan) == numbers(PROBE_SLAB_REFINEMENT)


def _validate(profile, base_dir, allow_test_fixture):
    if isinstance(profile, dict) and profile.get("schema") == SCHEMA_V3:
        _keys(profile, TOP_KEYS | {"common_action_procedure"}, "computed profile")
        if profile["kind"] != COMPUTED_KIND or profile["review_status"] != "not_applicable_computed":
            raise EvidenceProfileError("the computed path has its own kind and not_applicable_computed review status")
        if profile["slab_action_assertions"] is not None or profile["review"] is not None:
            raise EvidenceProfileError("computed profiles cannot activate legacy assertions or invent a review")
        _keys(profile["scope"], SCOPE_KEYS, "scope")
        if profile["scope"].get("floor_sections_sha256") is not None or profile["scope"].get("slab_thickness_in") is not None:
            raise EvidenceProfileError("the computed domain replaces single-floor/thickness review scope")
        if not allow_test_fixture and any(_FIXTURE_TEXT.search(t) or _probe_stamp(t) for t in _strings(profile)):
            raise EvidenceProfileError("test/probe text is not a computed run configuration")
        # Reuse the exact legacy scope/plan/provenance checks on an unasserted view;
        # no legacy review rule is relaxed or written into the original profile.
        view = {key: copy.deepcopy(profile[key]) for key in TOP_KEYS}
        view.update(schema=SCHEMA, kind="unreviewed_example", review_status="unreviewed")
        _validate(view, base_dir, allow_test_fixture)
        from Design import SMRF_Common_Action_Procedure as computed
        try:
            computed.validate(profile["common_action_procedure"], base_dir)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            raise EvidenceProfileError("computed common-action procedure: " + str(exc)) from exc
        return False
    v2 = isinstance(profile, dict) and profile.get("schema") == SCHEMA_V2
    declaration = isinstance(profile, dict) and not v2 and "demand_declaration" in profile
    _keys(profile, (TOP_KEYS | {"procedure"}) if v2 else (TOP_KEYS | ({"demand_declaration"} if declaration else set())),
          "the profile")
    if declaration:
        # The run's demand-basis declaration (2026-11-07): who declared the DemandPolicy values, when and on what basis.
        _keys(profile["demand_declaration"], DECLARATION_KEYS, "demand_declaration")
        _text(profile["demand_declaration"]["declared_by"], "demand_declaration.declared_by")
        _date(profile["demand_declaration"]["declaration_date"], "demand_declaration.declaration_date")
        _text(profile["demand_declaration"]["declaration_basis"], "demand_declaration.declaration_basis")
    if profile["schema"] not in (SCHEMA, SCHEMA_V2):
        raise EvidenceProfileError(f"schema must be {SCHEMA!r} or {SCHEMA_V2!r}, not {profile['schema']!r}")
    _text(profile["profile_id"], "profile_id")
    kinds = KINDS_V2 if v2 else KINDS
    if profile["kind"] not in kinds:
        raise EvidenceProfileError(f"kind must be one of {kinds}, not {profile['kind']!r}")
    if profile["kind"] == "test_fixture" and not allow_test_fixture:
        raise EvidenceProfileError("a test-fixture profile is not a run configuration; it is loaded only by tests")
    if profile["review_status"] not in REVIEW_STATUSES:
        raise EvidenceProfileError(f"review_status must be one of {REVIEW_STATUSES}, not {profile['review_status']!r}")
    provenance = profile["provenance"]
    _keys(provenance, PROVENANCE_KEYS, "provenance")
    _text(provenance["prepared_by"], "provenance.prepared_by")
    _date(provenance["prepared_date"], "provenance.prepared_date")
    _text(provenance["purpose"], "provenance.purpose")
    # scope
    scope = profile["scope"]
    _keys(scope, SCOPE_KEYS, "scope")
    _text(scope["case_id"], "scope.case_id")
    _sha(scope["case_sha256"], "scope.case_sha256")
    _text(scope["profile"], "scope.profile")
    _keys(scope["geometry"], GEOMETRY_KEYS, "scope.geometry")
    for key in ("num_bay_x", "num_bay_y", "num_floor"):
        _integer(scope["geometry"][key], f"scope.geometry.{key}")
    for key in ("bay_x_in", "bay_y_in", "story_h_in"):
        _number(scope["geometry"][key], f"scope.geometry.{key}")
    _keys(scope["materials"], MATERIAL_KEYS, "scope.materials")
    _keys(scope["loads"], LOAD_KEYS, "scope.loads")
    for block, keys in (("materials", MATERIAL_KEYS), ("loads", LOAD_KEYS)):
        for key in keys:
            _number(scope[block][key], f"scope.{block}.{key}")
    policy = scope["slab_policy"]
    _keys(policy, SLAB_POLICY_KEYS, "scope.slab_policy")
    if type(policy["live_load_patterning"]) is not bool:
        raise EvidenceProfileError("scope.slab_policy.live_load_patterning must be a JSON boolean")
    if policy["transfer_mesh_per_bay"] is not None and _integer(policy["transfer_mesh_per_bay"],
                                                                "scope.slab_policy.transfer_mesh_per_bay") < 1:
        raise EvidenceProfileError("scope.slab_policy.transfer_mesh_per_bay must be null or a positive integer")
    for key in ("minimum_thickness_in", "maximum_thickness_in", "thickness_increment_in"):
        if _number(policy[key], f"scope.slab_policy.{key}") <= 0:
            raise EvidenceProfileError(f"scope.slab_policy.{key} must be positive")
    thickness = scope["slab_thickness_in"]
    if thickness is not None:
        if _number(thickness, "scope.slab_thickness_in") <= 0:
            raise EvidenceProfileError("scope.slab_thickness_in must be null or a positive number")
        # The slab stage only reaches thicknesses of the declared ladder: one off it could never be in scope.
        low, high, step = (float(policy[k]) for k in ("minimum_thickness_in", "maximum_thickness_in",
                                                       "thickness_increment_in"))
        steps = (float(thickness) - low) / step
        if not low - _TOLERANCE <= float(thickness) <= high + _TOLERANCE or abs(steps - round(steps)) > 1e-9:
            raise EvidenceProfileError(f"scope.slab_thickness_in {thickness} is not on the declared thickness ladder "
                                       f"({low:g} to {high:g} in by {step:g} in): no candidate could be in scope")
    sections = scope["floor_sections_sha256"]
    if v2 and (sections is not None or thickness is not None):
        raise EvidenceProfileError("a v2 profile's procedure domain replaces scope.floor_sections_sha256 and "
                                   "scope.slab_thickness_in: give both as null")
    if sections is not None:
        if not isinstance(sections, list) or not sections or len(set(sections)) != len(sections):
            raise EvidenceProfileError("scope.floor_sections_sha256 must be null or a non-empty list without repeats")
        for index, value in enumerate(sections):
            _sha(value, f"scope.floor_sections_sha256[{index}]")
    # the numerical plan, apart from the assertions
    plan = profile["slab_refinement"]
    if plan is not None:
        from Design.SMRF_Slab_Refinement import _validate_policy
        try:
            _validate_policy(plan)
        except (ValueError, TypeError, KeyError) as exc:
            raise EvidenceProfileError(f"slab_refinement: {exc}") from exc
        _plan_shape(plan)
        _text(profile["slab_refinement_basis"], "slab_refinement_basis")
        if ((_probe_stamp(profile["slab_refinement_basis"]) or _probe_plan(plan))
                and "investigation" not in profile["slab_refinement_basis"].lower()):
            raise EvidenceProfileError("a PROBE-derived plan must say in slab_refinement_basis that its tolerance is an "
                                       "investigation criterion")
    elif profile["slab_refinement_basis"] is not None:
        raise EvidenceProfileError("slab_refinement_basis is given without a slab_refinement plan")
    # the engineering assertions
    assertions = profile["slab_action_assertions"]
    asserted = False
    if assertions is not None:
        from Design.Config import SlabActionAssertions
        names = {f.name for f in fields(SlabActionAssertions)}
        _keys(assertions, names - {flag for flag in LEGACY_OPTIONAL_FLAGS if flag not in assertions},
              "slab_action_assertions")
        for flag in ASSERTION_FLAGS:
            if type(assertions.get(flag, False)) is not bool:
                raise EvidenceProfileError(f"slab_action_assertions.{flag} must be a JSON boolean, not "
                                           f"{assertions.get(flag)!r}")
        for key in ASSERTION_TEXT:
            if not isinstance(assertions[key], str):
                raise EvidenceProfileError(f"slab_action_assertions.{key} must be a string")
        asserted = any(assertions.get(flag, False) for flag in ASSERTION_FLAGS)
        if not asserted and any(assertions[key].strip() for key in ASSERTION_TEXT):
            raise EvidenceProfileError("slab_action_assertions names an approver, date or basis without asserting "
                                       "anything: leave them blank until a review asserts")
    review = profile["review"]
    if review is not None:
        _keys(review, REVIEW_KEYS, "review")
        _text(review["reviewed_by"], "review.reviewed_by")
        _date(review["review_date"], "review.review_date")
        _text(review["basis"], "review.basis")
        _text(review["decision"], "review.decision")
        if profile["kind"] != "test_fixture" and _probe_stamp(
                review["reviewed_by"], review["basis"], review["decision"],
                *(item.get("description") for item in review["evidence"] if isinstance(review["evidence"], list)
                  and isinstance(item, dict))):
            raise EvidenceProfileError("a PROBE stamp is a test configuration, never a run's engineering review")
        if not isinstance(review["evidence"], list) or not review["evidence"]:
            raise EvidenceProfileError("review.evidence must list the reviewed evidence files")
        for index, item in enumerate(review["evidence"]):
            _keys(item, EVIDENCE_KEYS, f"review.evidence[{index}]")
            _text(item["path"], f"review.evidence[{index}].path")
            _sha(item["sha256"], f"review.evidence[{index}].sha256")
            _text(item["description"], f"review.evidence[{index}].description")
            path = Path(item["path"])
            path = path if path.is_absolute() else Path(base_dir) / path
            if not path.is_file():
                raise EvidenceProfileError(f"review.evidence[{index}]: {item['path']} does not exist")
            if _sha256_file(path) != item["sha256"]:
                raise EvidenceProfileError(f"review.evidence[{index}]: {item['path']} differs from its recorded sha256")
    if v2:
        from Design import SMRF_Slab_Procedure as procedure_module
        try:
            procedure_module.validate(profile["procedure"], ASSERTION_FLAGS, base_dir)
        except ValueError as exc:
            raise EvidenceProfileError(str(exc)) from exc
        if plan is None:
            raise EvidenceProfileError("a procedure profile declares its refinement plan (slab_refinement)")
        if profile["procedure"]["protocol"]["plan_sha256"] != procedure_module.plan_sha256(plan):
            raise EvidenceProfileError("procedure.protocol.plan_sha256 is not the sha256 of this profile's slab_refinement "
                                       "plan: the reviewed protocol binds the plan it was reviewed with")
        if asserted:
            unbacked = procedure_module.unbacked_flags(profile["procedure"], assertions)
            if unbacked:
                raise EvidenceProfileError(f"asserted flags {unbacked} rest on dispositions that are not all accepted: a "
                                           "procedure is no permission to set a flag true")
    if profile["kind"] in ("reviewed", "reviewed_procedure"):
        texts = list(_strings(profile))
        if any(_FIXTURE_TEXT.search(t) for t in texts) or _probe_stamp(*texts):
            raise EvidenceProfileError("a reviewed profile cannot carry TEST FIXTURE or PROBE text: those are test "
                                       "configurations, never an engineering review")
    if profile["kind"] in ("reviewed", "reviewed_procedure") and (profile["review_status"] != "reviewed_for_declared_scope"
                                                                  or review is None):
        raise EvidenceProfileError("a reviewed profile needs review_status reviewed_for_declared_scope and its review block")
    if profile["review_status"] == "reviewed_for_declared_scope" and profile["kind"] == "unreviewed_example":
        raise EvidenceProfileError("an unreviewed example cannot carry the status reviewed_for_declared_scope")
    if asserted:
        if profile["kind"] == "unreviewed_example":
            raise EvidenceProfileError("an unreviewed example asserts nothing")
        if profile["review_status"] != "reviewed_for_declared_scope" or review is None:
            raise EvidenceProfileError("asserted slab actions need review_status reviewed_for_declared_scope and the "
                                       "review block")
        if plan is None:
            raise EvidenceProfileError("asserted slab actions need a slab_refinement plan: single-mesh demands cannot "
                                       "be verified")
        if sections is None and not v2:
            raise EvidenceProfileError("asserted slab actions need scope.floor_sections_sha256: the floor section sets "
                                       "the review covers")
        if thickness is None and not v2:
            raise EvidenceProfileError("asserted slab actions need scope.slab_thickness_in: the slab thickness the "
                                       "review covers")
        for key in ASSERTION_TEXT:
            _text(assertions[key], f"slab_action_assertions.{key}")
        _date(assertions["assertion_date"], "slab_action_assertions.assertion_date")
        if (assertions["asserted_by"], assertions["assertion_date"]) != (review["reviewed_by"], review["review_date"]):
            raise EvidenceProfileError("the assertion provenance must be the review's own (reviewed_by, review_date); "
                                       "no approver is filled in here")
        if profile["kind"] != "test_fixture" and _probe_stamp(*(assertions[k] for k in ASSERTION_TEXT),
                                                              review["reviewed_by"], review["basis"]):
            raise EvidenceProfileError("PROBE assertions are a test configuration, never a run's engineering review")
    return asserted


def _plan_shape(plan):
    """What the plan resolver checks later, checked now: the recipe name, its levels and shell budget, or an explicit
    plan's meshes (count, keys, budgets, offsets)."""
    from Design.SMRF_Floor_Mesh import MAX_EXPLICIT_SHELLS, RECIPE_LEVELS, RECIPES

    def bounded(value, low, high, where):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise EvidenceProfileError(f"{where} must be an integer from {low} to {high}")

    if "recipe" in plan:
        if plan["recipe"] not in RECIPES:
            raise EvidenceProfileError(f"slab_refinement.recipe must be one of {sorted(RECIPES)}")
        bounded(plan["levels"], 2, RECIPE_LEVELS, "slab_refinement.levels")
        bounded(plan["max_shells"], 4, MAX_EXPLICIT_SHELLS, "slab_refinement.max_shells")
        return
    meshes = plan["meshes"]
    if not isinstance(meshes, list) or not 2 <= len(meshes) <= 8:
        raise EvidenceProfileError("slab_refinement.meshes must list between two and eight meshes")
    for index, mesh in enumerate(meshes):
        _keys(mesh, {"x_offsets_in", "y_offsets_in", "max_shells"}, f"slab_refinement.meshes[{index}]")
        bounded(mesh["max_shells"], 4, MAX_EXPLICIT_SHELLS, f"slab_refinement.meshes[{index}].max_shells")
        for axis in ("x_offsets_in", "y_offsets_in"):
            if not isinstance(mesh[axis], list) or not mesh[axis]:
                raise EvidenceProfileError(f"slab_refinement.meshes[{index}].{axis} must be a list of bay coordinates "
                                           "from 0 to the bay length (one list for every bay, or one list per bay)")
            rows = mesh[axis] if isinstance(mesh[axis][0], list) else [mesh[axis]]
            for row in rows:
                if not isinstance(row, list) or len(row) < 3:
                    raise EvidenceProfileError(f"slab_refinement.meshes[{index}].{axis} needs at least three offsets")
                for value in row:
                    _number(value, f"slab_refinement.meshes[{index}].{axis}")


def _refuse_constant(name):
    raise EvidenceProfileError(f"{name} is not a JSON number")


def _pairs(pairs):
    """A JSON object that names each key once (a repeated key would hide what runs behind what is read)."""
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise EvidenceProfileError(f"the key {key!r} is repeated in one JSON object")
        seen[key] = value
    return seen


def load(path, *, allow_test_fixture=False):
    """The validated profile of ``path`` as a Profile (EvidenceProfileError on the first problem)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
        profile = json.loads(text, parse_constant=_refuse_constant, object_pairs_hook=_pairs)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceProfileError(f"{path}: {exc}") from exc
    asserted = _validate(profile, path.resolve().parent, allow_test_fixture)
    return Profile(profile, path.resolve(), _sha256_file(path), asserted, procedure_status(profile, path.resolve().parent))


def procedure_status(profile, base_dir=None):
    """The status of a v2 profile's procedure (None for v1): ``inactive_method_changed`` when a reviewed method file
    hashes otherwise now; ``active`` for a reviewed procedure under its review; ``test_fixture_active`` for a test
    fixture (loaded only by tests; never an engineering review); ``inactive_unreviewed`` otherwise."""
    if profile.get("schema") == SCHEMA_V3:
        from Design.SMRF_Common_Action_Procedure import assessment
        return assessment(profile["common_action_procedure"], Path(base_dir or "."))["status"]
    if profile.get("schema") != SCHEMA_V2:
        return None
    from Design import SMRF_Slab_Procedure as procedure_module
    if procedure_module.method_problems(profile["procedure"]):
        return "inactive_method_changed"
    if (profile["kind"] == "reviewed_procedure" and profile["review_status"] == "reviewed_for_declared_scope"
            and profile["review"] is not None):
        return "active"
    if profile["kind"] == "test_fixture":
        return "test_fixture_active"
    return "inactive_unreviewed"


class Profile:
    """A validated design-evidence profile (load())."""

    def __init__(self, data, path, file_sha256, asserted, procedure_status=None):
        self.data, self.path, self.file_sha256, self.asserted = copy.deepcopy(data), path, file_sha256, asserted
        self.procedure_status = procedure_status

    def identity(self):
        """sha256 of the validated profile's canonical content (the evidence files enter by their recorded hashes)."""
        return hashlib.sha256(_canonical({"schema": SCHEMA, "profile": self.data}).encode("utf-8")).hexdigest()

    def scope_problems(self, case, case_sha256, profile, cfg):
        """How the run differs from the declared scope (an empty list when it does not)."""
        scope, problems = self.data["scope"], []
        expected = {"case_id": case.get("case_id"), "case_sha256": case_sha256, "profile": profile}
        problems += [f"scope.{k}: declared {scope[k]!r}, run {v!r}" for k, v in expected.items() if scope[k] != v]
        run = {"geometry": {"num_bay_x": sp.NUM_BAY_X, "num_bay_y": sp.NUM_BAY_Y, "num_floor": sp.NUM_FLOOR,
                            "bay_x_in": sp.BAY_X, "bay_y_in": sp.BAY_Y, "story_h_in": sp.STORY_H},
               "materials": {"slab_fy_ksi": sp.SLAB_FY_KSI, "concrete_unit_weight_kcf": sp.CONCRETE_UNIT_WEIGHT_KCF},
               "loads": {"floor_live_load_ksf": sp.FLOOR_LIVE_LOAD_KSF,
                         "superimposed_dead_load_ksf": cfg.slab.superimposed_dead_load_ksf},
               "slab_policy": {"minimum_thickness_in": cfg.slab.minimum_thickness_in,
                               "maximum_thickness_in": cfg.slab.maximum_thickness_in,
                               "thickness_increment_in": cfg.slab.thickness_increment_in}}
        for block, values in run.items():
            for key, value in values.items():
                declared = scope[block][key]
                if not (isinstance(value, (int, float)) and abs(float(value) - float(declared))
                        <= _TOLERANCE * max(1.0, abs(float(value)))):
                    problems.append(f"scope.{block}.{key}: declared {declared!r}, run {value!r}")
        declared = scope["slab_policy"]
        for key, value in (("live_load_patterning", cfg.demands.live_load_patterning),
                           ("transfer_mesh_per_bay", cfg.floor_analysis.transfer_mesh_per_bay)):
            if declared[key] != value or type(declared[key]) is not type(value):
                problems.append(f"scope.slab_policy.{key}: declared {declared[key]!r}, run {value!r}")
        return problems

    def apply(self, cfg):
        """The configuration with the profile applied and, when the profile carries one, its demand-basis declaration."""
        applied = self._apply(cfg)
        declaration = self.data.get("demand_declaration")
        if declaration is None:
            return applied
        demands = replace(applied.demands, **declaration)
        problems = demands.problems()
        if problems:
            raise EvidenceProfileError("demand_declaration: " + "; ".join(problems[:4]))
        return replace(applied, demands=demands)

    def _apply(self, cfg):
        """The DesignConfig this profile implies: its refinement plan, its assertions (the defaults when none) and the
        floor section sets the assertions cover; ``cfg`` itself is not changed."""
        from Design.Config import SlabActionAssertions
        assertions = self.data["slab_action_assertions"]
        if self.data["schema"] == SCHEMA_V3:
            from Design.SMRF_Common_Action_Procedure import assessment
            data = self.data["common_action_procedure"]
            result = assessment(data, self.path.parent)
            common = {"kind": COMPUTED_KIND, "status": result["status"], "identity": result["identity"],
                      "procedure": copy.deepcopy(data), "evidence_base_dir": str(self.path.parent)}
            return replace(cfg, slab_actions=SlabActionAssertions(),
                           floor_analysis=replace(cfg.floor_analysis, common_action_procedure=common,
                               slab_refinement=copy.deepcopy(self.data["slab_refinement"]), slab_procedure=None,
                               slab_assertion_floor_sections=None, slab_assertion_thickness_in=None))
        if self.data["schema"] == SCHEMA_V2:
            # The procedure is carried whatever its status, so every candidate floor records the evidence it earns;
            # the assertions apply only under an active procedure.
            from Design import SMRF_Slab_Procedure as procedure_module
            active = self.procedure_status in procedure_module.ACTIVE_STATUSES
            procedure = {"status": self.procedure_status, "identity": procedure_module.identity(self.data["procedure"]),
                         "procedure": copy.deepcopy(self.data["procedure"])}
            return replace(cfg, slab_actions=(SlabActionAssertions(**assertions) if active and assertions is not None
                                              else SlabActionAssertions()),
                           floor_analysis=replace(cfg.floor_analysis,
                                                  slab_refinement=copy.deepcopy(self.data["slab_refinement"]),
                                                  slab_assertion_floor_sections=None, slab_assertion_thickness_in=None,
                                                  slab_procedure=procedure, common_action_procedure=None))
        sections = self.data["scope"]["floor_sections_sha256"]
        thickness = self.data["scope"]["slab_thickness_in"]
        return replace(cfg,
                       slab_actions=SlabActionAssertions(**assertions) if assertions is not None else SlabActionAssertions(),
                       floor_analysis=replace(cfg.floor_analysis,
                                              common_action_procedure=None,
                                              slab_refinement=copy.deepcopy(self.data["slab_refinement"]),
                                              slab_assertion_floor_sections=(tuple(sections) if self.asserted and sections
                                                                             else None),
                                              slab_assertion_thickness_in=(float(thickness) if self.asserted
                                                                           and thickness is not None else None)))

    def summary(self):
        """The block a run's request carries: what the profile is, what it declares and its identity (the generation
        basis carries its id, kind, review status, asserted flag and identity_sha256)."""
        from Design.Config import SlabActionAssertions
        data = self.data
        if data["schema"] == SCHEMA_V3:
            from Design.SMRF_Common_Action_Procedure import assessment
            result = assessment(data["common_action_procedure"], self.path.parent)
            return {"schema": SCHEMA_V3, "profile_id": data["profile_id"], "kind": COMPUTED_KIND,
                    "review_status": data["review_status"], "path": str(self.path), "file_sha256": self.file_sha256,
                    "identity_sha256": self.identity(), "provenance": copy.deepcopy(data["provenance"]),
                    "scope": copy.deepcopy(data["scope"]), "slab_refinement": copy.deepcopy(data["slab_refinement"]),
                    "slab_refinement_basis": data["slab_refinement_basis"], "slab_action_assertions": None,
                    "asserted": False, "review": None, "procedure_status": result["status"],
                    "procedure_identity_sha256": result["identity"], "method_files_changed": result["method_problems"],
                    "computed_proof_assessment": result,
                    "effect": "Distinct computed common-action method; no legacy assertions. Every candidate requires its own thirteen computed checks."}
        complete = self.asserted and SlabActionAssertions(**data["slab_action_assertions"]).all_asserted()
        if data["schema"] == SCHEMA_V2:
            from Design import SMRF_Slab_Procedure as procedure_module
            procedure = {"procedure_status": self.procedure_status,
                         "procedure_identity_sha256": procedure_module.identity(data["procedure"]),
                         "method_files_changed": procedure_module.method_problems(data["procedure"]),
                         "effect": ("procedure-based slab evidence: every candidate floor is checked against the "
                                    "procedure's domain and protocol, and the assertions apply to the floors that pass"
                                    if self.procedure_status in procedure_module.ACTIVE_STATUSES else
                                    f"procedure {self.procedure_status}: every candidate floor records the evidence it "
                                    "earns, nothing is asserted, no slab layout is selected")}
            return {"schema": data["schema"], "profile_id": data["profile_id"], "kind": data["kind"],
                    "review_status": data["review_status"], "path": str(self.path), "file_sha256": self.file_sha256,
                    "identity_sha256": self.identity(), "provenance": copy.deepcopy(data["provenance"]),
                    "scope": copy.deepcopy(data["scope"]), "slab_refinement": copy.deepcopy(data["slab_refinement"]),
                    "slab_refinement_basis": data["slab_refinement_basis"],
                    "slab_action_assertions": copy.deepcopy(data["slab_action_assertions"]),
                    "asserted": self.asserted and self.procedure_status in procedure_module.ACTIVE_STATUSES,
                    "review": copy.deepcopy(data["review"]), **procedure}
        return {"schema": SCHEMA, "profile_id": data["profile_id"], "kind": data["kind"],
                "demand_declaration": copy.deepcopy(data.get("demand_declaration")),
                "review_status": data["review_status"], "path": str(self.path), "file_sha256": self.file_sha256,
                "identity_sha256": self.identity(), "provenance": copy.deepcopy(data["provenance"]),
                "scope": copy.deepcopy(data["scope"]), "slab_refinement": copy.deepcopy(data["slab_refinement"]),
                "slab_refinement_basis": data["slab_refinement_basis"],
                "slab_action_assertions": copy.deepcopy(data["slab_action_assertions"]), "asserted": self.asserted,
                "review": copy.deepcopy(data["review"]),
                "effect": (("numerical refinement of the slab actions" if data["slab_refinement"] is not None else
                            "single-mesh slab actions (no refinement plan)")
                           + " with every engineering assertion false: no slab layout is selected and the strong-column "
                             "checks that need the developed slab stay unknown"
                           if not self.asserted else
                           "asserted slab actions for the floor section sets and slab thickness of the declared scope "
                           "only; every other candidate is evaluated unasserted" if complete else
                           "partially asserted slab actions for the floor section sets and slab thickness of the declared "
                           "scope: the asserted flags are recorded in those candidates' evidence, but no slab layout is "
                           "selected until every numerical flag is asserted; every other candidate is evaluated "
                           "unasserted")}


def default_summary():
    """The block a run without a profile carries."""
    return {"schema": SCHEMA, "profile_id": None, "basis": DEFAULT_BASIS}
