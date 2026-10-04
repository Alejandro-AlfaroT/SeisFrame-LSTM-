"""Fail-closed verification cache checks and process-owned launcher leases."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import threading

_INPUT_LOCK = threading.RLock()


@contextmanager
def exclusive_lease(path):
    """Nonblocking OS lock. The marker persists; the kernel releases ownership.

    Never unlink this file: another process may already have its inode open.
    This is for a local output root, not a distributed/network lock service.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another process owns {path}; no files were removed.") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_identity(case, cfg, profile_id=None):
    """Reproduce worker inputs without leaving global geometry/site/profile changes.

    Only identity construction touches globals here, serialized across launch
    threads. This does not run a design or an OpenSees analysis. ``profile_id``
    is the analysis profile the worker applies (Model/Analysis_Profile); the
    identity carries it, so a design made under another profile is refused.
    """
    import Structure_Parameters as sp
    import Geometry_Overrides as go
    from Design.Design_Driver import design_request_identity
    with _INPUT_LOCK:
        snapshot = dict(vars(sp))
        try:
            go.apply_geometry_overrides({
                "NUM_BAY_X": case["num_bay_x"], "NUM_BAY_Y": case["num_bay_y"],
                "NUM_FLOOR": case["num_floor"], "STORY_H": 12.0 * case["story_height_ft"],
                "BAY_X": 12.0 * case["bay_x_width_ft"], "BAY_Y": 12.0 * case["bay_y_width_ft"],
            }, variant_name=case["geometry_name"], emit=False)
            sp.apply_seismic_site(case["seismic_site"])
            if profile_id:
                from Model.Analysis_Profile import apply_profile
                apply_profile(profile_id)
            return design_request_identity(cfg)
        finally:
            for key in set(vars(sp)) - set(snapshot):
                delattr(sp, key)
            vars(sp).update(snapshot)


def validate_record(record, expected):
    """Bind the artifact to the actual requested geometry, policy and sources."""
    identity = record.get("request_identity")
    if not isinstance(identity, dict) or identity != expected:
        raise ValueError("Design request identity differs from current case/code/config; use a new output root.")
    payload = {k: v for k, v in identity.items() if k != "sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(canonical.encode()).hexdigest() != identity.get("sha256"):
        raise ValueError("Design request identity has an invalid digest.")
    if record.get("schema_version") != identity["schema"]:
        raise ValueError("Design schema differs from request identity.")
    geometry = record.get("geometry", {})
    for field, key in (("num_bay_x", "NUM_BAY_X"), ("num_bay_y", "NUM_BAY_Y"),
                       ("num_floor", "NUM_FLOOR"), ("bay_x_in", "BAY_X"),
                       ("bay_y_in", "BAY_Y"), ("story_h_in", "STORY_H")):
        if geometry.get(field) != identity["inputs"][key]:
            raise ValueError(f"Saved geometry {field} differs from request identity.")


def checked_result(case, out_dir, saved, expected, probe, probe_date):
    """Never promote a result label to evidence; inspect and requalify its design."""
    from Design.SMRF_Qualification import qualify_design
    out_dir = Path(out_dir)
    try:
        if (out_dir / ".design.json.lock").exists():
            raise ValueError("Design lock exists; ownership must be reviewed before reuse")
        if saved.get("case") != case:
            raise ValueError("Result belongs to a different plan case.")
        if saved.get("probe_assertions") != probe or saved.get("probe_date") != (probe_date if probe else None):
            raise ValueError("Result PROBE mode/date differs from this plan.")
        path = out_dir / "design.json"
        before = file_sha256(path)
        if saved.get("design_sha256") and saved["design_sha256"] != before:
            raise ValueError("Design artifact changed since the worker finished.")
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_record(record, expected)
        if record.get("seismic", {}).get("site_label") != case["seismic_site"]:
            raise ValueError("Design hazard differs from plan case.")
        q = qualify_design(record)
        if file_sha256(path) != before:
            raise ValueError("Design artifact changed while it was being checked.")
        by_status = {}
        for check in q["checks"]:
            by_status.setdefault(check["status"], set()).add(check["id"].split(":")[0])
        identity_payload = {k: expected[k] for k in ("schema", "policy", "source_sha256")}
        method = hashlib.sha256(json.dumps(identity_payload, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
        search = record.get("search") or {}
        torsion = (record.get("demand_basis") or {}).get("torsion") or {}
        return {**saved, "status": "designed", "design_sha256": before,
                "request_sha256": expected["sha256"], "methodology_sha256": method,
                "design_json_bytes": path.stat().st_size, "accepted": q["accepted"], "counts": q["counts"],
                "sections": record["sections"], "dcr": record["dcr"],
                "fail_ids": sorted(by_status.get("fail", [])),
                "not_evaluated_ids": sorted(by_status.get("not_evaluated", [])),
                "stop_reason": search.get("stop_reason"), "selected_iteration": search.get("selected_iteration"),
                "last_iteration": search.get("last_iteration"),
                "tir": torsion.get("tir", torsion.get("max_drift_ratio")),
                "torsional_irregularity": torsion.get("torsional_irregularity")}
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return {"case": case, "status": "error", "accepted": False,
                "error": f"Untrusted cached design: {exc}. Existing files preserved."}
