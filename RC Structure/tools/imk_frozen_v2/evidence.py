"""Shared helpers of the frozen IMK V2 verification (2026-10-29): locating the approved study, hashing, paths.

The approved twenty-run MCER/DBE study is read-only evidence. Every script here reads it and writes only into a NEW
output directory under RC Structure/outputs (git-ignored). The study is located by its root folder, passed with
--study; the saved-design harness folder (v6-mcer-hinges) is found beside it or under references/.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

RC = Path(__file__).resolve().parents[2]
if str(RC) not in sys.path:
    sys.path.insert(0, str(RC))

CASES = ("case_0018", "case_0035", "case_0064", "case_0079", "case_0088",
         "case_0013", "case_0045", "case_0038", "case_0080", "case_0065")
MODES = ("mce_average", "dbe_average")
RESEARCH_PROFILE = "v2_imk_half_member_reference_research_20261028"
DEFAULT_OUTPUT = RC / "outputs" / "imk_frozen_v2_verification_20261029"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def study_root(path):
    root = Path(path).resolve()
    for name in ("manifest.json", "profile.py", "final_summary.json", "verification.json"):
        if not (root / name).exists():
            raise SystemExit(f"{root} is not the approved study root (missing {name})")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("profile") != RESEARCH_PROFILE:
        raise SystemExit(f"{root}/manifest.json names profile {manifest.get('profile')!r}, not {RESEARCH_PROFILE}")
    return root


def harness_root(study):
    for candidate in (study.parent / "v6-mcer-hinges", study.parent / "references" / "v6-mcer-hinges"):
        if (candidate / "preflight.py").exists() and (candidate / "run_earthquake.py").exists():
            return candidate
    raise SystemExit(f"no v6-mcer-hinges harness beside {study}")


def new_output_dir(path):
    """A fresh directory: refuses an existing non-empty one so no earlier result is overwritten."""
    out = Path(path).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    return out


def under_study(path, study):
    try:
        Path(path).resolve().relative_to(study)
        return True
    except ValueError:
        return False


def git_state():
    import subprocess
    def run(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, cwd=str(RC.parent), timeout=30).stdout.strip()
        except Exception:                                   # noqa: BLE001 -- reported, never required
            return None
    return {"head": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "status_porcelain": run("status", "--porcelain=v1")}


def source_hashes():
    names = ("Structure_Parameters.py", "Model/IMK_Spring_Mapping.py", "Model/IMK_Calibration.py", "Model/IMK_Hinges.py",
             "Model/IMK_Materials.py", "Model/Analysis_Profile.py", "Analysis/Fixed_Design_Diagnostics.py")
    return {name: sha256_file(RC / name) for name in names}


def runtime():
    import platform
    import numpy
    import openseespy.opensees as ops
    try:
        from importlib.metadata import version
        openseespy_version = version("openseespy")
    except Exception:                                       # noqa: BLE001
        openseespy_version = None
    return {"python": sys.version, "platform": platform.platform(), "numpy": numpy.__version__,
            "openseespy": openseespy_version, "opensees": ops.version()}


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=1, default=str, allow_nan=True), encoding="utf-8")
