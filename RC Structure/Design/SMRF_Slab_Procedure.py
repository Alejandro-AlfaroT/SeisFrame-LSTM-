"""The procedure-based slab evidence path (P1, 2026-11-01): a reviewed slab-action procedure and the evidence each
candidate floor earns under it.

A procedure binds four things, declared in a design-evidence profile of schema v2 (Design.Design_Evidence_Profile)
and carried by FloorAnalysisConfig.slab_procedure:

  method        the reviewed method files and their sha256 (line endings normalized), METHOD_FILES; files read for
                information only (INFORMATION_FILES) are recorded, never compared;
  domain        the beam sections and grades, the support (joint-core column) sections and the slab thickness ladder
                the review covers;
  protocol      the refinement plan (the profile's slab_refinement), the resolution every floor must reach and the
                instance checks every floor must pass (INSTANCE_CHECKS);
  dispositions  one engineering decision per open item of the slab assessment (DISPOSITION_IDS), each with its
                consequence and evidence files, and ``flag_basis``: the dispositions each assertion flag rests on.

The procedure is ``active`` only when the profile is a reviewed procedure, its method files hash as reviewed and every
asserted flag rests on accepted dispositions (Design_Evidence_Profile decides and records ``status``); otherwise it
only records evidence and asserts nothing. Under any status every candidate floor is checked on its own:

  - floor_domain_problems: its beams and supports are in the declared sets; the thickness is on the declared ladder;
  - instance_record: the refinement resolved as the protocol declares and passed, every case balanced, the plate
    carried no membrane force, the face shear recovery was complete, and the load cases are the ACI 318-19 5.3.1 /
    6.4.3 inventory, recomputed here from the slab's own dead and live loads.

A floor outside the domain or failing a check keeps its numbers and has its engineering flags masked (masked());
the combination over the floors then fails closed (SMRF_Slab_Refinement.combine_floor_evidence). Nothing here sets a
flag true, and a procedure is no permission to set one: the flags come from the reviewed assertions alone.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

PROCEDURE_SCHEMA = "slab_action_procedure_v1"
INSTANCE_VERSION = "slab_procedure_instance_v1"
# The reviewed method (decision D7, 2026-11-01): the floor model, its refinement, recovery and actions, the floor
# section description, the slab and this module. The slab reinforcement module (the consumer gate) changes with the
# hook and gate rules, so it is recorded for information only.
METHOD_FILES = ("Design/SMRF_Slab_Actions.py", "Design/SMRF_Slab_Refinement.py", "Design/SMRF_Floor_Mesh.py",
                "Design/SMRF_Floor_Analysis.py", "Design/SMRF_Slab_Recovery.py", "Design/SMRF_Floor_Sections.py",
                "Design/SMRF_Slab.py", "Design/SMRF_Slab_Procedure.py")
INFORMATION_FILES = ("Design/SMRF_Slab_Reinforcement.py",)
INSTANCE_CHECKS = ("resolution_as_declared", "refinement_passed", "static_balance", "membrane_free",
                   "face_recovery_complete", "load_case_inventory")
# The open items of the slab assessment (research/ga_closure_followup_20261101/slab_assessment/SLAB_ASSESSMENT_DRAFT.md
# section 2). Fire resistance is outside the declared scope (SMRF_Common.SCOPE_EXCLUSIONS), not a disposition.
DISPOSITION_IDS = ("floor_idealization", "supports_and_faces", "stiffness_sensitivity", "two_way_shear_path",
                   "local_slab_column_steel", "independent_reference", "tolerance")
DECISIONS = ("open", "accepted", "rejected")
# The dispositions every flag's basis must include (a profile may add more, never fewer): each structural gate rests on
# its own item, and the numerical flags on the items that decide whether the plate model represents the floor.
REQUIRED_DISPOSITIONS = {
    "analysis_applicability_verified": ("floor_idealization", "supports_and_faces", "stiffness_sensitivity",
                                        "independent_reference"),
    "all_floors_enveloped": ("floor_idealization",),
    "load_pattern_envelope_verified": ("independent_reference",),
    "spatial_envelope_per_unit_width": ("supports_and_faces", "independent_reference"),
    "twisting_moment_resolution_verified": ("independent_reference",),
    "zero_membrane_force_verified": ("floor_idealization",),
    "verified": ("floor_idealization", "supports_and_faces", "stiffness_sensitivity", "independent_reference",
                 "tolerance"),
    "two_way_shear_path_assessed": ("two_way_shear_path",),
    "slab_column_local_steel_assessed": ("local_slab_column_steel",),
}
STATUSES = ("active", "test_fixture_active", "inactive_unreviewed", "inactive_method_changed")
TOP_KEYS = {"schema", "method", "domain", "protocol", "dispositions", "flag_basis"}
METHOD_KEYS = {"id", "source_sha256", "information_sha256"}
DOMAIN_KEYS = {"beam_sections_in", "beam_fc_ksi", "column_sections_in", "thickness_ladder_in"}
LADDER_KEYS = {"minimum", "maximum", "increment"}
PROTOCOL_KEYS = {"expected_resolution", "instance_checks", "plan_sha256"}
RESOLUTION_KEYS = {"status", "base_levels"}
DISPOSITION_KEYS = {"decision", "consequence", "evidence"}
EVIDENCE_KEYS = {"path", "sha256"}
RC = Path(__file__).resolve().parents[1]
_TOLERANCE = 1e-9


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def file_sha256(path):
    """sha256 of a source file with its line endings normalized to LF (one review holds across checkouts)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def method_sha256(root=RC):
    """({path: sha256} of METHOD_FILES, {path: sha256} of INFORMATION_FILES) as they are now under ``root``."""
    return ({rel: file_sha256(Path(root) / rel) for rel in METHOD_FILES},
            {rel: file_sha256(Path(root) / rel) for rel in INFORMATION_FILES})


def plan_sha256(plan):
    """sha256 of a refinement plan's canonical content: the protocol's binding of the plan it was reviewed with."""
    return hashlib.sha256(_canonical(plan).encode("utf-8")).hexdigest()


def identity(procedure):
    """sha256 of the procedure block's canonical content."""
    return hashlib.sha256(_canonical(procedure).encode("utf-8")).hexdigest()


def _keys(value, expected, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a JSON object")
    missing, unknown = sorted(expected - set(value)), sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(f"{where}: missing keys {missing}, unknown keys {unknown}")


def _positive(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{where} must be a positive finite number")
    return float(value)


def _sha(value, where):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{where} must be a lowercase sha256 hex digest")


def _pairs(value, where):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a non-empty list of [b_in, h_in] pairs")
    pairs = []
    for index, pair in enumerate(value):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{where}[{index}] must be a [b_in, h_in] pair")
        pairs.append((_positive(pair[0], f"{where}[{index}][0]"), _positive(pair[1], f"{where}[{index}][1]")))
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"{where} repeats a section")
    return pairs


def validate(procedure, flags, base_dir=None):
    """The procedure block checked for shape (ValueError on the first problem). ``flags`` are the assertion flags
    flag_basis must name exactly. Disposition evidence files are resolved against ``base_dir`` and their sha256 is
    checked when a base directory is given."""
    _keys(procedure, TOP_KEYS, "procedure")
    if procedure["schema"] != PROCEDURE_SCHEMA:
        raise ValueError(f"procedure.schema must be {PROCEDURE_SCHEMA!r}")
    method = procedure["method"]
    _keys(method, METHOD_KEYS, "procedure.method")
    if not isinstance(method["id"], str) or not method["id"].strip():
        raise ValueError("procedure.method.id must be a non-blank string")
    for block, files in (("source_sha256", METHOD_FILES), ("information_sha256", INFORMATION_FILES)):
        _keys(method[block], set(files), f"procedure.method.{block}")
        for rel, value in method[block].items():
            _sha(value, f"procedure.method.{block}[{rel}]")
    domain = procedure["domain"]
    _keys(domain, DOMAIN_KEYS, "procedure.domain")
    _pairs(domain["beam_sections_in"], "procedure.domain.beam_sections_in")
    _pairs(domain["column_sections_in"], "procedure.domain.column_sections_in")
    grades = domain["beam_fc_ksi"]
    if not isinstance(grades, list) or not grades or len(set(grades)) != len(grades):
        raise ValueError("procedure.domain.beam_fc_ksi must be a non-empty list without repeats")
    for index, grade in enumerate(grades):
        _positive(grade, f"procedure.domain.beam_fc_ksi[{index}]")
    ladder = domain["thickness_ladder_in"]
    _keys(ladder, LADDER_KEYS, "procedure.domain.thickness_ladder_in")
    low, high, step = (_positive(ladder[k], f"procedure.domain.thickness_ladder_in.{k}") for k in ("minimum", "maximum",
                                                                                                    "increment"))
    if high < low:
        raise ValueError("procedure.domain.thickness_ladder_in: maximum below minimum")
    protocol = procedure["protocol"]
    _keys(protocol, PROTOCOL_KEYS, "procedure.protocol")
    _keys(protocol["expected_resolution"], RESOLUTION_KEYS, "procedure.protocol.expected_resolution")
    if protocol["expected_resolution"]["status"] != "resolved":
        raise ValueError("procedure.protocol.expected_resolution.status must be 'resolved'")
    levels = protocol["expected_resolution"]["base_levels"]
    if isinstance(levels, bool) or not isinstance(levels, int) or not 2 <= levels <= 8:
        raise ValueError("procedure.protocol.expected_resolution.base_levels must be an integer from 2 to 8")
    _sha(protocol["plan_sha256"], "procedure.protocol.plan_sha256")
    checks = protocol["instance_checks"]
    if not isinstance(checks, list) or sorted(checks) != sorted(INSTANCE_CHECKS) or len(set(checks)) != len(checks):
        raise ValueError(f"procedure.protocol.instance_checks must list each of {INSTANCE_CHECKS} once")
    dispositions = procedure["dispositions"]
    _keys(dispositions, set(DISPOSITION_IDS), "procedure.dispositions")
    for name, entry in dispositions.items():
        _keys(entry, DISPOSITION_KEYS, f"procedure.dispositions.{name}")
        if entry["decision"] not in DECISIONS:
            raise ValueError(f"procedure.dispositions.{name}.decision must be one of {DECISIONS}")
        if not isinstance(entry["consequence"], str) or not entry["consequence"].strip():
            raise ValueError(f"procedure.dispositions.{name}.consequence must be a non-blank string")
        if not isinstance(entry["evidence"], list):
            raise ValueError(f"procedure.dispositions.{name}.evidence must be a list")
        if entry["decision"] == "accepted" and not entry["evidence"]:
            raise ValueError(f"procedure.dispositions.{name}: an accepted disposition needs its evidence files")
        for index, item in enumerate(entry["evidence"]):
            _keys(item, EVIDENCE_KEYS, f"procedure.dispositions.{name}.evidence[{index}]")
            _sha(item["sha256"], f"procedure.dispositions.{name}.evidence[{index}].sha256")
            if base_dir is not None:
                path = Path(item["path"])
                path = path if path.is_absolute() else Path(base_dir) / path
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                    raise ValueError(f"procedure.dispositions.{name}.evidence[{index}]: {item['path']} is missing or "
                                     "differs from its recorded sha256")
    basis = procedure["flag_basis"]
    _keys(basis, set(flags), "procedure.flag_basis")
    for flag, names in basis.items():
        if not isinstance(names, list) or not names or any(n not in DISPOSITION_IDS for n in names):
            raise ValueError(f"procedure.flag_basis.{flag} must list disposition ids of {DISPOSITION_IDS}")
        missing = sorted(set(REQUIRED_DISPOSITIONS.get(flag, ())) - set(names))
        if missing:
            raise ValueError(f"procedure.flag_basis.{flag} must include its own dispositions {missing}")
    return procedure


def unbacked_flags(procedure, assertions):
    """The asserted flags whose listed dispositions are not all accepted (a procedure never backs a flag by itself)."""
    accepted = {name for name, entry in procedure["dispositions"].items() if entry["decision"] == "accepted"}
    return sorted(flag for flag, names in procedure["flag_basis"].items()
                  if (assertions or {}).get(flag) is True and not set(names) <= accepted)


def method_problems(procedure, root=RC):
    """The reviewed method files whose sha256 differs now (an empty list when the method is the reviewed one)."""
    current, _information = method_sha256(root)
    return sorted(rel for rel, value in procedure["method"]["source_sha256"].items() if current.get(rel) != value)


def on_ladder(thickness_in, ladder):
    if thickness_in is None:
        return False
    low, high, step = (float(ladder[k]) for k in ("minimum", "maximum", "increment"))
    steps = (float(thickness_in) - low) / step
    return low - _TOLERANCE <= float(thickness_in) <= high + _TOLERANCE and abs(steps - round(steps)) <= 1e-9


def _beams(description):
    for axis in ("x", "y"):
        for row in description["beam_lines"][axis]:
            yield from ([row] if isinstance(row, dict) else row)


ACTIVE_STATUSES = ("active", "test_fixture_active")


def floor_domain_problems(procedure, description, thickness_in, check_thickness=True):
    """Why one floor (SMRF_Floor_Sections.for_floor) or its slab thickness lies outside the procedure's domain; an
    empty list when it does not. Malformed input is outside. ``procedure`` is the configuration's
    FloorAnalysisConfig.slab_procedure ({"status", "identity", "procedure"})."""
    try:
        domain = procedure["procedure"]["domain"]
        beams = {(float(b["b_in"]), float(b["h_in"])) for b in _beams(description)}
        grades = {float(b["fc_ksi"]) for b in _beams(description)} | {float(description["fc_beam_ksi"])}
        supports = {(float(c["b_in"]), float(c["h_in"])) for row in description["supports"] for c in row}
        allowed_beams = {(float(b), float(h)) for b, h in domain["beam_sections_in"]}
        allowed_columns = {(float(b), float(h)) for b, h in domain["column_sections_in"]}
        allowed_grades = {float(g) for g in domain["beam_fc_ksi"]}
        problems = [f"beam section {b:g}x{h:g} in outside the domain" for b, h in sorted(beams - allowed_beams)]
        problems += [f"beam concrete {g:g} ksi outside the domain" for g in sorted(grades - allowed_grades)]
        problems += [f"support section {b:g}x{h:g} in outside the domain" for b, h in sorted(supports - allowed_columns)]
        if check_thickness and not on_ladder(thickness_in, domain["thickness_ladder_in"]):
            problems.append(f"slab thickness {thickness_in} in is not on the domain's ladder")
        return problems
    except (KeyError, TypeError, ValueError) as exc:
        return [f"the floor cannot be read against the domain: {type(exc).__name__}: {exc}"]


def expected_load_cases(slab, live_ksf, nx, ny):
    """The ACI 318-19 load-case inventory of one slab, written here apart from the producer (SMRF_Slab_Actions):
    1.4D and 1.2D + 1.6L on all panels (5.3.1); where the live load is not at most 0.75 D, also 3/4 of the factored
    live load on each checkerboard and on the panels adjacent to each interior support line (6.4.3.3(a), (b))."""
    dead = float(slab["concrete_unit_weight_kcf"]) * float(slab["thickness_in"]) / 12.0 + float(slab["superimposed_dead_load_ksf"])
    expected = {"1.4D": (1.4, 0.0), "1.2D+1.6L_all": (1.2, 1.6)}
    if not float(live_ksf) <= 0.75 * dead:
        for name in ("even", "odd"):
            expected[f"1.2D+1.2L_{name}"] = (1.2, 1.6 * 0.75)
        for line in range(1, ny):
            expected[f"1.2D+1.2L_adjacent_x_line_{line}"] = (1.2, 1.6 * 0.75)
        for line in range(1, nx):
            expected[f"1.2D+1.2L_adjacent_y_line_{line}"] = (1.2, 1.6 * 0.75)
    return expected


def _load_cases_problems(evidence, slab, live_ksf, nx, ny):
    expected = expected_load_cases(slab, live_ksf, nx, ny)
    cases = {case.get("id"): case for case in evidence.get("cases") or []}
    problems = []
    if set(cases) != set(expected):
        problems.append(f"cases {sorted(cases)} differ from the declared inventory {sorted(expected)}")
    for name, (dead, live) in expected.items():
        case = cases.get(name)
        if case is not None and not (math.isclose(case.get("dead_factor", math.nan), dead, abs_tol=1e-12)
                                     and math.isclose(case.get("live_factor", math.nan), live, abs_tol=1e-12)):
            problems.append(f"case {name}: factors {case.get('dead_factor')}/{case.get('live_factor')}, declared {dead}/{live}")
    boards = [cases[n].get("live_pattern") for n in ("1.2D+1.2L_even", "1.2D+1.2L_odd") if n in cases]
    if len(boards) == 2:
        panels = sorted(tuple(p) for board in boards for p in board)
        if panels != sorted((i, j) for i in range(nx) for j in range(ny)):
            problems.append("the two checkerboards do not cover every panel exactly once")
    return problems


def instance_record(procedure, floor_sha256, floors, evidence, description, slab, live_ksf, nx, ny):
    """The evidence one floor earned under the procedure: the domain and every instance check, with ``passed``."""
    block = procedure["procedure"]
    refinement = evidence.get("refinement") or {}
    resolution = refinement.get("resolution") or {}
    preconditions = evidence.get("numerical_preconditions") or {}
    expected = block["protocol"]["expected_resolution"]
    checks = {
        "resolution_as_declared": (resolution.get("status") == expected["status"]
                                   and len(resolution.get("meshes") or []) == expected["base_levels"]),
        "refinement_passed": refinement.get("status") == "passed",
        "static_balance": bool(evidence.get("equilibrium")) and all(
            row.get("numerical_balance_passed") is True for row in evidence.get("equilibrium") or []),
        "membrane_free": preconditions.get("zero_membrane_force_verified") is True,
        "face_recovery_complete": (evidence.get("shear_recovery") or {}).get("top_rows_recovered_at_face") is True,
    }
    try:
        problems = _load_cases_problems(evidence, slab, live_ksf, nx, ny)
    except (KeyError, TypeError, ValueError) as exc:
        problems = [f"{type(exc).__name__}: {exc}"]
    checks["load_case_inventory"] = not problems
    domain = floor_domain_problems(procedure, description, slab.get("thickness_in"))
    return {"version": INSTANCE_VERSION, "procedure_identity": procedure["identity"], "status": procedure["status"],
            "floor_sections_sha256": floor_sha256, "floors": list(floors), "domain_problems": domain,
            "checks": checks, "load_case_problems": problems[:6],
            "refinement_status": refinement.get("status"), "passed": not domain and all(checks.values())}


def masked(evidence, reason):
    """A copy of one floor's evidence with every engineering flag false (its numbers kept): what a floor outside the
    domain, failing an instance check or under an inactive procedure contributes to the combination."""
    from Design.SMRF_Slab_Actions import _FLAGS
    result = copy.deepcopy(evidence)
    for flag in _FLAGS:
        result[flag] = False
    result["procedure_mask"] = reason
    return result


def replay_problems(procedure, combined, description, slab, live_ksf, nx, ny):
    """Each floor's instance record recomputed from its own evidence and compared with the recorded one (a cached or
    edited pass cannot stand); an empty list when they agree."""
    problems = []
    instances = {row.get("floor_sections_sha256"): row for row in combined.get("procedure_instances") or []}
    for entry in combined.get("by_floor") or []:
        sha = entry.get("floor_sections_sha256")
        sections = (description.get("distinct") or {}).get(sha, {}).get("sections")
        recorded = instances.get(sha)
        if sections is None or recorded is None:
            problems.append(f"floor {str(sha)[:12]}: no recorded procedure instance")
            continue
        again = instance_record(procedure, sha, entry.get("floors") or [], entry.get("evidence") or {}, sections, slab,
                                live_ksf, nx, ny)
        if again != recorded:
            problems.append(f"floor {str(sha)[:12]}: the recorded procedure instance does not follow from its evidence")
    return problems
