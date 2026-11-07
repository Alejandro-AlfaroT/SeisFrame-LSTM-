"""Source/domain-bound computed method profiles, independent of legacy review flags.

This is a numerical proof reader, not a substitute reviewer. Every required method
item needs comparisons reconstructed from separately hashed actual/reference files.
Every candidate still has to pass its own thirteen common-action checks.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from Design import SMRF_Common_Action_Contract as contract

SCHEMA = "computed_common_actions_v2_bound_material_basis"
PROOF_SCHEMA = "computed_common_action_method_proof_v2_bound_material_basis"
DOMAIN_KEYS = {"beam_sections_in", "beam_fc_ksi", "column_sections_in", "thickness_ladder_in"}
LIMIT_KEYS = {"mesh_per_bay", "max_shells_per_floor", "max_coupled_dofs", "max_case_count",
              "reference_updates", "reference_wall_seconds", "primary_wall_seconds", "physical_case_batch_size",
              "max_required_cuts", "strength_wall_seconds", "max_strength_evaluations", "max_chunk_cuts"}
PROTOCOL_KEYS = {"required_row_ids", "instance_method", "analysis_limits", "mesh_refinement", "material_action_basis"}
REFINEMENT_KEYS = {"levels", "moment_relative_tolerance", "shear_relative_tolerance",
                   "absolute_tolerances", "tolerance_basis"}
SOURCE_FILES = ("Design/SMRF_Common_Action_Procedure.py", "Design/Design_Evidence_Profile.py", "Design/Config.py")


def _keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(label + " has missing or unexpected fields")


def _number(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("a finite non-boolean numeric value is required")
    return float(value)


def material_basis_from_protocol(protocol):
    """Explicit mechanics/strength choices are method-proof inputs, not flags."""
    from Design.SMRF_Bar_Pitch_Layers import METHOD as PITCH
    from Design.SMRF_Common_Strength_Stream import _local_basis
    basis=protocol['material_action_basis']
    _keys(basis,{'floor_material_mapping','strength_material_action_basis'},'material/action basis')
    if basis['floor_material_mapping'] not in ('clipped_cylinders_v2',PITCH):
        raise ValueError('Explicit supported floor material mapping required')
    _local_basis(basis['strength_material_action_basis'],basis)
    return dict(material_mapping=basis['floor_material_mapping'],strength_material_action_basis=basis['strength_material_action_basis'])


def _hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference(reference, base):
    _keys(reference, {"path", "sha256", "description"}, "evidence reference")
    if not isinstance(reference["description"], str) or not reference["description"].strip():
        raise ValueError("evidence reference needs its concrete purpose")
    path = Path(reference["path"])
    path = (base/path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file() or _hash_file(path) != reference["sha256"]:
        raise ValueError("computed evidence is absent or changed: " + str(path))
    return path


def _value(pointer, base):
    _keys(pointer, {"file", "json_path"}, "numeric evidence pointer")
    path = _reference(pointer["file"], base)
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(pointer["json_path"], list):
        raise ValueError("json_path must be a list of explicit keys/indices")
    for key in pointer["json_path"]:
        if type(key) not in (int, str):
            raise ValueError("invalid numeric evidence index")
        value = value[key]
    return _number(value)


def _proof(reference, base, name, procedure):
    path = _reference(reference, base)
    proof = json.loads(path.read_text(encoding="utf-8-sig"))
    _keys(proof, {"schema", "row_id", "source_identity_sha256", "domain_sha256", "protocol_sha256",
                  "independent_method", "coverage_basis", "comparisons"}, "method proof")
    expected = {"schema": PROOF_SCHEMA, "row_id": name,
                "source_identity_sha256": contract.sha256(procedure["method"]["source_sha256"]),
                "domain_sha256": contract.sha256(procedure["domain"]),
                "protocol_sha256": contract.sha256(procedure["protocol"])}
    if any(proof[k] != value for k, value in expected.items()):
        raise ValueError("method proof is not bound to this exact method/domain/protocol: " + name)
    if not isinstance(proof["independent_method"], str) or not proof["independent_method"].strip():
        raise ValueError("method proof must identify its independent calculation")
    _reference(proof["coverage_basis"], path.parent)
    if not isinstance(proof["comparisons"], list) or not proof["comparisons"]:
        raise ValueError("a method proof needs actual numeric comparisons")
    rows = []
    for item in proof["comparisons"]:
        _keys(item, {"name", "actual", "reference", "comparison", "absolute_tolerance", "relative_tolerance",
                     "criterion_source"}, "method comparison")
        if not isinstance(item["name"], str) or not item["name"].strip():
            raise ValueError("method comparison needs a name")
        if item["actual"]["file"]["sha256"] == item["reference"]["file"]["sha256"]:
            raise ValueError("actual and independent reference cannot be the same file")
        actual, target = _value(item["actual"], path.parent), _value(item["reference"], path.parent)
        absolute, relative = _number(item["absolute_tolerance"]), _number(item["relative_tolerance"])
        if absolute < 0 or relative < 0:
            raise ValueError("negative numerical tolerance")
        _reference(item["criterion_source"], path.parent)
        tolerance = absolute + relative*abs(target)
        operation = item["comparison"]
        if operation not in ("close", "<=", ">="):
            raise ValueError("unsupported method comparison")
        passed = {"close": abs(actual-target) <= tolerance, "<=": actual <= target+tolerance,
                  ">=": actual >= target-tolerance}[operation]
        rows.append({"name": item["name"], "actual": actual, "reference": target,
                     "tolerance": tolerance, "comparison": operation, "passed": passed})
    return {"proof_sha256": reference["sha256"], "comparisons": rows,
            "status": "pass" if all(r["passed"] for r in rows) else "fail"}


def validate(procedure, base_dir):
    """Validate declarations; pending absent proof is allowed, fabricated flags are not."""
    _keys(procedure, {"schema", "method", "domain", "protocol", "proofs"}, "computed procedure")
    if procedure["schema"] != SCHEMA:
        raise ValueError("wrong computed procedure schema")
    _keys(procedure["method"], {"id", "source_sha256"}, "computed method")
    if not isinstance(procedure["method"]["id"], str) or not procedure["method"]["id"].strip():
        raise ValueError("computed method needs its identity")
    source = procedure["method"]["source_sha256"]
    if not isinstance(source, dict) or not source:
        raise ValueError("computed method needs explicit source hashes")
    for name, digest in source.items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("malformed method source identity")
    domain = procedure["domain"]; _keys(domain, DOMAIN_KEYS, "computed method domain")
    for key in ("beam_sections_in", "column_sections_in"):
        if not isinstance(domain[key], list) or not domain[key]: raise ValueError("empty section domain")
        for row in domain[key]:
            if not isinstance(row, list) or len(row) != 2 or any(_number(x) <= 0 for x in row):
                raise ValueError("invalid section domain")
    if not isinstance(domain["beam_fc_ksi"], list) or not domain["beam_fc_ksi"] or any(_number(x) <= 0 for x in domain["beam_fc_ksi"]):
        raise ValueError("invalid grade domain")
    ladder = domain["thickness_ladder_in"]; _keys(ladder, {"minimum", "maximum", "increment"}, "thickness domain")
    if any(_number(x) <= 0 for x in ladder.values()) or ladder["maximum"] < ladder["minimum"]:
        raise ValueError("invalid thickness ladder")
    protocol = procedure["protocol"]; _keys(protocol, PROTOCOL_KEYS, "computed protocol")
    material_basis_from_protocol(protocol)
    if protocol["required_row_ids"] != list(contract.ROW_IDS) or protocol["instance_method"] != contract.METHOD:
        raise ValueError("all thirteen current computed checks are required")
    limits = protocol["analysis_limits"]; _keys(limits, LIMIT_KEYS, "analysis limits")
    for key, value in limits.items():
        if _number(value) <= 0 or (not key.endswith("seconds") and type(value) is not int):
            raise ValueError("analysis limits must be positive explicit values")
    refine = protocol["mesh_refinement"]; _keys(refine, REFINEMENT_KEYS, "mesh refinement")
    levels = refine["levels"]
    if not isinstance(levels, list) or len(levels) < 2 or any(type(n) is not int or n < 1 for n in levels) or sorted(set(levels)) != levels:
        raise ValueError("at least two strictly increasing actual mesh levels are required")
    for key in ("moment_relative_tolerance", "shear_relative_tolerance"):
        if not 0 < _number(refine[key]) < 1: raise ValueError("invalid relative refinement tolerance")
    if refine["absolute_tolerances"] is not None:
        _keys(refine["absolute_tolerances"], {"moment_kip_in", "shear_kip"}, "absolute tolerances")
        if any(_number(v) < 0 for v in refine["absolute_tolerances"].values()): raise ValueError("negative absolute tolerance")
    if refine["tolerance_basis"] is not None: _reference(refine["tolerance_basis"], Path(base_dir))
    _keys(procedure["proofs"], contract.ROW_IDS, "computed proofs")
    for name, refs in procedure["proofs"].items():
        if not isinstance(refs, list): raise ValueError("proofs must be lists of actual evidence references")
        for reference in refs: _proof(reference, Path(base_dir), name, procedure)


def assessment(procedure, base_dir):
    validate(procedure, base_dir)
    sources = procedure["method"]["source_sha256"]
    source_errors = contract.source_problems(sources)
    source_errors += ["missing source: " + p for p in SOURCE_FILES if p not in sources]
    checks = {name: [_proof(ref, Path(base_dir), name, procedure) for ref in refs]
              for name, refs in procedure["proofs"].items()}
    missing = [name for name, rows in checks.items() if not rows]
    failed = [name for name, rows in checks.items() if any(row["status"] != "pass" for row in rows)]
    refine = procedure["protocol"]["mesh_refinement"]
    pending_protocol = refine["absolute_tolerances"] is None or refine["tolerance_basis"] is None
    status = ("pending_method_changed" if source_errors else "computed_failed" if failed else
              "pending" if missing or pending_protocol else "computed_ready")
    return {"status": status, "identity": contract.sha256(procedure), "method_problems": source_errors,
            "missing_proofs": missing, "failed_proofs": failed, "pending_tolerance_protocol": pending_protocol,
            "proof_checks": checks, "assertions_activated": False,
            "candidate_checks_required": list(contract.ROW_IDS)}


def pending_procedure(domain, source_sha256,*,material_mapping='clipped_cylinders_v2',strength_material_action_basis='finite_axial_window_v1'):
    """Explicit provisional limits; no proof, approval or activation is generated."""
    basis=dict(floor_material_mapping=material_mapping,strength_material_action_basis=strength_material_action_basis)
    material_basis_from_protocol(dict(material_action_basis=basis))
    return {"schema": SCHEMA, "method": {"id": SCHEMA, "source_sha256": copy.deepcopy(source_sha256)},
            "domain": copy.deepcopy(domain), "protocol": {
                "required_row_ids": list(contract.ROW_IDS), "instance_method": contract.METHOD,
                "material_action_basis":basis,
                "analysis_limits": {"mesh_per_bay": 2, "max_shells_per_floor": 4096, "max_coupled_dofs": 2400,
                    "max_case_count": 64, "reference_updates": 1, "reference_wall_seconds": 1800,
                    "primary_wall_seconds": 300, "physical_case_batch_size": 4,
                    "max_required_cuts":500000,"strength_wall_seconds":1800,"max_strength_evaluations":250,"max_chunk_cuts":128},
                "mesh_refinement": {"levels": [2, 4, 8], "moment_relative_tolerance": .05,
                    "shear_relative_tolerance": .05, "absolute_tolerances": None, "tolerance_basis": None}},
            "proofs": {name: [] for name in contract.ROW_IDS}}
