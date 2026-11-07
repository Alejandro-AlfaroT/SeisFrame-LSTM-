"""Required common-action evidence and a read-only, fail-closed replay boundary.

Saved check statuses and reviewed-profile assertions are never replay inputs.  The
analysis producer must reconstruct the checks from its primitive response, material,
development and method evidence.  This module binds that reconstruction to the exact
installed design, source files and saved payload; it performs no structural solve.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from Design.SMRF_Common import check_row_problems, not_evaluated
from Design.SMRF_Slab_Procedure import DISPOSITION_IDS

METHOD = "common_action_computed_contract_v1"
FAMILY = "common_actions"
ROW_IDS = (
    "common.analysis_and_case_coverage", "common.p5_signed_load_path",
    "common.shared_steel_concurrent_strength", "common.actual_bar_development",
    "common.torsion_concurrent", "common.diaphragm_collectors_chords",
) + tuple("common.disposition." + name for name in DISPOSITION_IDS)
CLAUSE = "ACI 318-19 Chapters 6, 8, 18, 21, 22, 25; ASCE 7-22 12.10; computed common-action evidence"
SOURCE_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SOURCE_FILES = (
    "Design/SMRF_Continuous_Torsion.py",
    "Design/SMRF_Common_Torsion_Stream.py",
    "Design/SMRF_Common_Action_Contract.py", "Design/Code_Check_Contract.py",
    "Design/SMRF_Common_Action_Analysis.py", "Design/SMRF_Physical_Floor_Strength.py",
    "Design/SMRF_Actual_Bar_Development.py", "Design/SMRF_Beam_Torsion.py",
    "Design/SMRF_Common_Torsion_Recovery.py", "Design/SMRF_Common_Torsion_Guards.py",
    "Design/SMRF_Common_Torsion_Stream.py", "Design/SMRF_Beam_Torsion_Detail.py",
    "Design/SMRF_Torsion_Station_Coverage.py",
    "Design/SMRF_Coupled_Floor_Design.py", "Design/SMRF_Floor_Substructure.py",
    "Design/SMRF_Beam_Shell_Connection.py",
    "Design/SMRF_Linear_Refinement.py",
    "Design/SMRF_Stable_Sparse_Assembly.py",
    "Design/SMRF_Axial_Action_Accounting.py", "Design/SMRF_Physical_Floor_Materials.py",
    "Design/SMRF_Common_Guard_Plan.py", "Design/SMRF_Primary_Moment_Guards.py",
)


def sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def payload_sha256(package):
    """The only excluded field is the payload's own digest (no circular hash)."""
    return sha256({key: value for key, value in package.items() if key != "payload_sha256"})


def source_problems(source):
    """Compare actual bytes, not two equal arbitrary source labels or file stems."""
    if not isinstance(source, dict) or not source:
        return ["source_identity must contain relative source paths and exact SHA256 hashes"]
    problems = ["missing source: " + name for name in REQUIRED_SOURCE_FILES if name not in source]
    for name, digest in source.items():
        if not isinstance(name, str) or "\\" in name:
            problems.append("source path must use relative forward slashes")
            continue
        path = (SOURCE_ROOT / name).resolve()
        if not path.is_relative_to(SOURCE_ROOT) or path.suffix != ".py":
            problems.append("source outside the Python method: " + name)
        elif not isinstance(digest, str) or len(digest) != 64 or not path.is_file():
            problems.append("missing source or malformed hash: " + name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append("changed source: " + name)
    return problems


def validate_package(state_identity, package):
    if not isinstance(package, dict):
        raise ValueError("actual common-action package is absent")
    if package.get('saved_physical_input_binding'):
        raise ValueError('Historical physical-input reuse is diagnostic evidence, not current-source candidate qualification')
    if not state_identity or package.get("state_identity") != state_identity:
        raise ValueError("common-action grouped-state identity differs")
    for key in ("method", "definition_sha256"):
        if not isinstance(package.get(key), str) or not package[key]:
            raise ValueError("common-action " + key + " is absent")
    if package.get("payload_sha256") != payload_sha256(package):
        raise ValueError("common-action payload digest differs")
    problems = source_problems(package.get("source_identity"))
    if problems:
        raise ValueError("; ".join(problems))
    if package.get("analysis_source_identity") != sha256(package["source_identity"]):
        raise ValueError("common-action mechanical source identity differs")


def required_inventory():
    return {(FAMILY, name, "") for name in ROW_IDS}


def unknown_checks(reason):
    return [not_evaluated(name, CLAUSE, reason) for name in ROW_IDS]


def assess_common_actions(state, package):
    """Recompute all required rows; missing proof never becomes a review exemption.

    ``replay_assessment`` is owned by the physical analysis producer.  It returns
    identities and standard numeric ``checks`` reconstructed from raw evidence,
    without analyzing again.  A missing row remains unknown beside real failures.
    A duplicate, invented row, malformed arithmetic or changed identity refuses
    the replay.  Neither package['checks'] nor package['status'] is consumed here.
    """
    try:
        validate_package(state.identity(), package)
        from Design.SMRF_Common_Action_Analysis import replay_assessment
        replay = replay_assessment(state, copy.deepcopy(package))
        if not isinstance(replay, dict):
            raise ValueError("common-action replay did not return a record")
        for key in ("state_identity", "source_identity", "definition_sha256", "payload_sha256"):
            if replay.get(key) != package.get(key):
                raise ValueError("common-action replay identity differs: " + key)
        emitted = replay.get("checks")
        if not isinstance(emitted, list):
            raise ValueError("common-action replay has no computed check list")
        rows = {}
        for row in emitted:
            if not isinstance(row, dict) or row.get("id") not in ROW_IDS or row.get("location", ""):
                raise ValueError("common-action replay contains an unexpected check identity")
            name = row["id"]
            if name in rows:
                raise ValueError("duplicate common-action replay check: " + name)
            if check_row_problems(row):
                raise ValueError("malformed computed common-action check: " + name)
            if (row.get("details") or {}).get("review_dependency"):
                raise ValueError("computed common-action check cannot be replaced by a review")
            rows[name] = copy.deepcopy(row)
        checks = [rows.get(name) or not_evaluated(name, CLAUSE, "Required computed proof absent from replay.")
                  for name in ROW_IDS]
        return {"method": METHOD, "checks": checks, "replay_completed": True,
                "state_identity": state.identity(), "payload_sha256": package["payload_sha256"],
                "accepted": all(row["status"] == "pass" for row in checks)}
    except Exception as exc:  # An absent producer or unreadable evidence is an unknown, never a pass.
        reason = f"Common-action replay unavailable: {type(exc).__name__}: {exc}"
        return {"method": METHOD, "checks": unknown_checks(reason), "replay_completed": False,
                "state_identity": state.identity(), "accepted": False, "reason": reason}


def record_input_identity(record):
    """Also bind the actual slab/mats, which the historical analysis digest omits."""
    from Design.SMRF_Design_Evidence import analysis_input_signature
    rebar = record.get("slab_reinforcement") or {}
    return sha256({"analysis_input_sha256": analysis_input_signature(record),
                   "slab": record.get("slab"), "slab_layout": rebar.get("layout"),
                   "slab_inputs": rebar.get("inputs"), "bar_layers": record.get("bar_layers"),
                   "analysis_policy": (record.get("request_identity") or {}).get("policy")})


def saved_binding(record, package):
    """Create an integrity binding, not an engineering approval or computed proof."""
    from Design.SMRF_Design_Evidence import analysis_input_signature
    return {"method": METHOD, "analysis_input_sha256": analysis_input_signature(record),
            "physical_input_sha256": record_input_identity(record), "package_sha256": sha256(package),
            "source_identity_sha256": sha256(package.get("source_identity")),
            "definition_sha256": package.get("definition_sha256"),
            "state_identity": (record.get("member_groups") or {}).get("sha256")}


def bound_saved_package(record):
    package = record.get("common_actions")
    if package is None:
        return None
    if not isinstance(package, dict) or record.get("common_actions_input_binding") != saved_binding(record, package):
        raise ValueError("Saved common actions are not bound to the exact physical inputs, payload and sources")
    validate_package((record.get("member_groups") or {}).get("sha256"), package)
    return package
