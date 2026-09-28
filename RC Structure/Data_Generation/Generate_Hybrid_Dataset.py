"""
Data_Generation/Generate_Hybrid_Dataset.py
==========================================

Automation entrypoint for RC Structure hybrid GNN + LSTM data generation.

Two modes are supported:
  1. Compile existing NTHA result folders into hybrid_sample.npz files.
  2. Optionally run Ground_Motion_Main.py first, then compile the new outputs.

Examples
--------
Compile existing successful NTHA outputs:
    python -B Data_Generation/Generate_Hybrid_Dataset.py --compile-existing

Run PEER result IDs 1 and 2, then compile:
    python -B Data_Generation/Generate_Hybrid_Dataset.py --run-ntha --result-id 1 --result-id 2

Run the first 5 pairs from the manifest, then compile:
    python -B Data_Generation/Generate_Hybrid_Dataset.py --run-ntha --limit 5

Stop a running batch safely from another terminal:
    python -B Data_Generation/Generate_Hybrid_Dataset.py --request-stop

The same generation command can then be run again. Successful NTHAs and
compiled samples are reused, while an interrupted active case is retried.
"""

import argparse
from datetime import datetime, timezone
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Geometry_Overrides import (
    add_geometry_arguments,
    geometry_cli_args_for_command,
    geometry_name_from_args,
    geometry_overrides_from_args,
)
from Ground_Motion_Main import analysis_run_name
try:
    from .Hybrid_Exporter import compile_ntha_root
except ImportError:  # Direct script execution places Data_Generation on sys.path.
    from Hybrid_Exporter import compile_ntha_root


STOP_FILE_NAME = "STOP_GENERATION.json"
STATE_FILE_NAME = "generation_state.json"
ATOMIC_REPLACE_ATTEMPTS = 10
ATOMIC_REPLACE_INITIAL_DELAY_SEC = 0.05


def _safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _replace_with_retry(source, destination):
    """Replace a file atomically, tolerating brief Windows sharing locks."""
    delay = ATOMIC_REPLACE_INITIAL_DELAY_SEC
    for attempt in range(ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == ATOMIC_REPLACE_ATTEMPTS:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 0.5)


def _write_json_atomic(path, payload):
    """Write a checkpoint without exposing a partially written JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    _replace_with_retry(temporary, path)


def _stop_file_path(args, dataset_dir):
    return Path(args.stop_file) if args.stop_file else Path(dataset_dir) / STOP_FILE_NAME


def _stop_requested(stop_path):
    return Path(stop_path).exists()


def _request_stop(stop_path):
    payload = {
        "requested_at": _utc_now(),
        "reason": "user_requested_stop",
        "instructions": "The running generator will checkpoint and stop safely.",
    }
    _write_json_atomic(stop_path, payload)
    return payload


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _status_success(path):
    status_path = Path(path) / "status.json"
    if not status_path.exists():
        return False
    with status_path.open(encoding="utf-8") as file:
        status = json.load(file)
    requested = int(status.get("npts_requested") or 0)
    completed = int(status.get("completed_steps") or 0)
    return not bool(status.get("failed")) and requested > 0 and completed == requested


def _pair_max_npts(row_x, row_y):
    return max(
        int(float(row_x.get("npts") or 0)),
        int(float(row_y.get("npts") or 0)),
    )


def _all_pair_keys(set_name, split, max_npts):
    # Import lazily so compile-only mode has no OpenSees/manifest dependency.
    sys.path.insert(0, str(_repo_root()))
    from Loads.Ground_Motion import ground_motion_pair_rows

    keys = []
    for key, row_x, row_y in ground_motion_pair_rows(set_name=set_name, split=split):
        match = re.match(r"peer_result_id:([0-9]+)$", key)
        if not match:
            continue

        result_id = int(match.group(1))
        npts = _pair_max_npts(row_x, row_y)
        if max_npts is not None and npts > max_npts:
            continue

        keys.append((key, result_id, npts))

    return sorted(keys, key=lambda item: item[1])


def _load_pair_keys(set_name, split, limit, start_index, max_npts):
    keys = _all_pair_keys(set_name, split, max_npts)
    keys = keys[start_index:]
    if limit is not None:
        keys = keys[:limit]
    return keys


def _terminate_process(process, log, timeout_sec=10.0):
    if process.poll() is not None:
        return
    log.write("\nSTOP: terminating the active analysis; it will be retried on resume.\n")
    log.flush()
    process.terminate()
    try:
        process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        log.write("STOP: graceful termination timed out; killing the active analysis.\n")
        log.flush()
        process.kill()
        process.wait()


def _run_command(command, cwd, log_path, stop_path=None, poll_interval_sec=1.0):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(str(item) for item in command) + "\n\n")
        log.flush()
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
        )
        interrupted = False
        stop_reason = None
        try:
            while process.poll() is None:
                if stop_path is not None and _stop_requested(stop_path):
                    interrupted = True
                    stop_reason = "stop_file"
                    _terminate_process(process, log)
                    break
                time.sleep(poll_interval_sec)
        except KeyboardInterrupt:
            interrupted = True
            stop_reason = "keyboard_interrupt"
            _terminate_process(process, log)
        returncode = process.wait()
    return {
        "returncode": returncode,
        "elapsed_sec": time.perf_counter() - start,
        "log_path": str(log_path),
        "interrupted": interrupted,
        "stop_reason": stop_reason,
    }


def parse_run_pair(text):
    """'<result_id>:<scale>' -> (int, float): one run exactly as the plan asked for it."""
    try:
        result_id, scale = str(text).split(":", 1)
        return int(result_id), float(scale)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--run-pair expects RESULT_ID:SCALE, got {text!r}") from exc


def expand_runs(selected, scale_levels, run_pairs=None):
    """The (key, result_id, npts, scale) runs to execute, and the pair ids that could not run.

    With explicit run pairs (the parent scheduler's plan) exactly those pairs
    run, in plan order; a pair whose record is not among ``selected`` is
    reported rather than silently replaced. Without pairs every selected
    record runs at every scale level. That cross product only matches a plan
    when each case has one record or one intensity: under a calibration every
    record carries its own target-matched factor, and the cross product ran
    pairings the plan never asked for (fixed 2026-09-26).
    """
    if run_pairs:
        by_id = {result_id: (key, result_id, npts) for key, result_id, npts in selected}
        runs, missing = [], []
        for result_id, scale in run_pairs:
            if result_id in by_id:
                key, _, npts = by_id[result_id]
                runs.append((key, result_id, npts, scale))
            else:
                missing.append(result_id)
        return runs, missing
    return [
        (key, result_id, npts, scale)
        for key, result_id, npts in selected
        for scale in scale_levels
    ], []


def run_ntha_batch(args, checkpoint=None):
    rc_dir = _repo_root()
    ntha_root = Path(args.ntha_root)
    logs_dir = Path(args.dataset_dir) / "run_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    max_npts = args.max_npts if args.max_npts and args.max_npts > 0 else None

    if args.result_id:
        allowed = {
            result_id: (key, result_id, npts)
            for key, result_id, npts in _all_pair_keys(args.set_name, args.split, max_npts)
        }
        selected = []
        filtered_ids = []
        for rid in args.result_id:
            if rid in allowed:
                selected.append(allowed[rid])
            else:
                filtered_ids.append(rid)
        if filtered_ids:
            print(
                "Skipping PEER result_id(s) above max_npts="
                f"{max_npts}: {filtered_ids}"
            )
    else:
        selected = _load_pair_keys(
            args.set_name,
            args.split,
            args.limit,
            args.start_index,
            max_npts,
        )

    # One record pair run at several intensities becomes several runs. The
    # scale factor is part of the run directory name, so the intensities of a
    # record coexist under one case without colliding. A parent scheduler
    # names each run explicitly with --run-pair; only a direct call without
    # pairs falls back to the records x scales cross product.
    scale_levels = list(args.scale_factor) if args.scale_factor else [None]
    expanded, missing_pairs = expand_runs(selected, scale_levels, getattr(args, "run_pair", None))
    if missing_pairs:
        print(
            "Skipping run pair(s) whose record is not selectable in this set "
            f"(max_npts={max_npts}): {sorted(set(missing_pairs))}"
        )

    summaries = []
    stop_path = Path(args._stop_path)
    args._selected_count = len(expanded)
    args._stop_reason = None
    if checkpoint:
        checkpoint(summaries, expanded, None, "running")

    for key, result_id, npts, scale_factor in expanded:
        if _stop_requested(stop_path):
            args._stop_reason = "stop_file"
            break
        run_name = analysis_run_name(
            _safe_name(key.replace("peer_result_id:", "peer_")),
            x_only=args.x_only,
            scale_factor=scale_factor,
            damping_ratio=args.damping_ratio,
            rayleigh_mode_i=args.rayleigh_mode_i,
            rayleigh_mode_j=args.rayleigh_mode_j,
            dt_factor=args.dt_factor,
        )
        out_dir = ntha_root / run_name
        sample_npz = Path(args.dataset_dir) / run_name / "hybrid_sample.npz"

        if args.skip_existing and _status_success(out_dir):
            sample_exists = sample_npz.exists()
            summaries.append(
                {
                    "key": key,
                    "result_id": result_id,
                    "scale_factor": scale_factor,
                    "run_name": run_name,
                    "npts": npts,
                    "skipped": True,
                    "reason": (
                        "successful NTHA and compiled sample already exist"
                        if sample_exists
                        else "successful NTHA exists; sample will be compiled at checkpoint"
                    ),
                }
            )
            print(f"Skipping {run_name}: existing successful NTHA output.")
            if checkpoint:
                checkpoint(summaries, selected, None, "running")
            continue

        command = [
            args.python_exe or sys.executable,
            "-B",
            str(rc_dir / "Ground_Motion_Main.py"),
            "--result-id",
            str(result_id),
            "--set-name",
            args.set_name,
            "--output-dir",
            str(ntha_root),
            "--damping-ratio",
            str(args.damping_ratio),
            "--rayleigh-mode-i",
            str(args.rayleigh_mode_i),
            "--rayleigh-mode-j",
            str(args.rayleigh_mode_j),
            "--dt-factor",
            str(args.dt_factor),
        ]
        if args.split:
            command.extend(["--split", args.split])
        if args.x_only:
            command.append("--x-only")
        if scale_factor is not None:
            command.extend(["--scale-factor", str(scale_factor)])
        if args.catalog_summary:
            command.append("--catalog-summary")
        command.extend(geometry_cli_args_for_command(args))

        print(f"Running NTHA {run_name} ({npts} points) -> {out_dir}")
        if checkpoint:
            checkpoint(summaries, selected, run_name, "running")
        run_result = _run_command(
            command,
            cwd=rc_dir,
            log_path=logs_dir / f"{run_name}.log",
            stop_path=stop_path,
        )
        run_result.update(
            {
                "key": key,
                "result_id": result_id,
                "scale_factor": scale_factor,
                "run_name": run_name,
                "npts": npts,
                "output_dir": str(out_dir),
                "skipped": False,
            }
        )
        summaries.append(run_result)
        if checkpoint:
            checkpoint(
                summaries,
                selected,
                None,
                "stopping" if run_result["interrupted"] else "running",
            )
        print(
            f"Finished {run_name}: returncode={run_result['returncode']}, "
            f"elapsed={run_result['elapsed_sec'] / 60.0:.1f} min"
        )
        if run_result["interrupted"]:
            args._stop_reason = run_result["stop_reason"] or "interrupted"
            break

    return summaries


def parse_args():
    rc_dir = _repo_root()
    default_ntha_root = rc_dir / "outputs" / "ntha"
    default_dataset_dir = rc_dir / "outputs" / "hybrid_dataset"

    parser = argparse.ArgumentParser(
        description="Run and/or compile RC Structure NTHA records for hybrid GNN/LSTM training."
    )
    parser.add_argument("--run-ntha", action="store_true", help="Run NTHA before compiling samples.")
    parser.add_argument("--compile-existing", action="store_true", help="Compile existing NTHA folders.")
    parser.add_argument("--result-id", type=int, action="append", help="Specific PEER result_id to run. May be repeated.")
    parser.add_argument("--set-name", default="peer_mle_all")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-npts",
        type=int,
        default=15000,
        help=(
            "Omit ground-motion pairs whose longer horizontal component has "
            "more than this many acceleration points. Use 0 to disable."
        ),
    )
    parser.add_argument("--x-only", action="store_true")
    parser.add_argument(
        "--scale-factor",
        type=float,
        action="append",
        default=None,
        help=(
            "Ground-motion scale factor. Repeat to run one record pair at "
            "several intensities; each becomes its own run directory."
        ),
    )
    parser.add_argument(
        "--run-pair",
        type=parse_run_pair,
        action="append",
        default=None,
        metavar="RESULT_ID:SCALE",
        help=(
            "One (record pair, scale factor) run exactly as the plan asked for "
            "it; repeat per run. When given, --scale-factor is ignored and no "
            "records x scales cross product is formed."
        ),
    )
    parser.add_argument("--damping-ratio", type=float, default=0.05)
    parser.add_argument("--rayleigh-mode-i", type=int, default=0)
    parser.add_argument("--rayleigh-mode-j", type=int, default=2)
    parser.add_argument("--dt-factor", type=float, default=1.0)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--ntha-root", default=str(default_ntha_root))
    parser.add_argument("--dataset-dir", default=str(default_dataset_dir))
    parser.add_argument(
        "--stop-file",
        default=None,
        help=(
            "Stop-request file watched while generation is running. Defaults to "
            f"<dataset-dir>/{STOP_FILE_NAME}."
        ),
    )
    parser.add_argument(
        "--request-stop",
        action="store_true",
        help=(
            "Create the stop-request file for a running generator and exit. "
            "Use the same geometry and dataset-dir arguments as the active run."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip successful NTHA runs. Any missing hybrid sample is compiled "
            "without rerunning the structural analysis."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-failed", action="store_true")
    parser.add_argument("--catalog-summary", action="store_true")
    add_geometry_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    geometry_name = geometry_name_from_args(args)
    default_ntha_root = _repo_root() / "outputs" / "ntha"
    default_dataset_dir = _repo_root() / "outputs" / "hybrid_dataset"
    if geometry_name:
        if Path(args.ntha_root) == default_ntha_root:
            args.ntha_root = str(default_ntha_root / geometry_name)
        if Path(args.dataset_dir) == default_dataset_dir:
            args.dataset_dir = str(default_dataset_dir / geometry_name)
    ntha_root = Path(args.ntha_root)
    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    stop_path = _stop_file_path(args, dataset_dir)

    if args.request_stop:
        _request_stop(stop_path)
        print(f"Stop requested: {stop_path}")
        print("The running generator will save its checkpoint and exit safely.")
        return

    args._stop_path = str(stop_path)
    started_at = _utc_now()

    automation_summary = {
        "status": "running",
        "started_at": started_at,
        "updated_at": started_at,
        "finished_at": None,
        "stop_reason": None,
        "stop_file": str(stop_path),
        "run_ntha": args.run_ntha,
        "compile_existing": args.compile_existing or not args.run_ntha,
        "ntha_root": str(ntha_root),
        "dataset_dir": str(dataset_dir),
        "max_npts": args.max_npts if args.max_npts and args.max_npts > 0 else None,
        "geometry_variant": geometry_name or "baseline",
        "geometry_overrides": geometry_overrides_from_args(args),
        "ntha_runs": [],
        "compiled_samples": [],
    }

    state_path = dataset_dir / STATE_FILE_NAME
    last_selected = []
    compiled_rows = []

    def checkpoint(summaries, selected, active_run, status):
        last_selected[:] = selected
        completed_ids = []
        failed_ids = []
        interrupted_ids = []
        for summary in summaries:
            result_id = summary.get("result_id")
            if summary.get("interrupted"):
                interrupted_ids.append(result_id)
            elif summary.get("skipped") or summary.get("returncode") == 0:
                completed_ids.append(result_id)
            else:
                failed_ids.append(result_id)

        selected_ids = [entry[1] for entry in selected]
        resumable_ids = set(completed_ids)
        remaining_ids = [result_id for result_id in selected_ids if result_id not in resumable_ids]
        state = {
            "status": status,
            "started_at": started_at,
            "updated_at": _utc_now(),
            "geometry_variant": geometry_name or "baseline",
            "ntha_root": str(ntha_root),
            "dataset_dir": str(dataset_dir),
            "stop_file": str(stop_path),
            "active_run": active_run,
            "selected_count": len(selected_ids),
            "attempted_count": len(summaries),
            "completed_count": len(completed_ids),
            "failed_count": len(failed_ids),
            "interrupted_count": len(interrupted_ids),
            "compiled_sample_count": sum(
                bool(row.get("compiled")) for row in compiled_rows
            ),
            "selected_result_ids": selected_ids,
            "completed_result_ids": completed_ids,
            "failed_result_ids": failed_ids,
            "interrupted_result_ids": interrupted_ids,
            "remaining_result_ids": remaining_ids,
            "runs": summaries,
        }
        _write_json_atomic(state_path, state)

    fatal_error = None
    stop_reason = None

    try:
        if args.run_ntha:
            automation_summary["ntha_runs"] = run_ntha_batch(args, checkpoint=checkpoint)
            stop_reason = args._stop_reason
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    except Exception as exc:
        fatal_error = exc
        automation_summary["error"] = f"{type(exc).__name__}: {exc}"

    if fatal_error is None:
        try:
            if last_selected:
                checkpoint(
                    automation_summary["ntha_runs"],
                    last_selected,
                    None,
                    "stopping" if stop_reason else "compiling",
                )
            rows = compile_ntha_root(
                ntha_root,
                dataset_dir=dataset_dir,
                require_success=not args.allow_failed,
                overwrite=args.overwrite,
            )
            compiled_rows[:] = rows
            automation_summary["compiled_samples"] = rows
            print(f"Hybrid manifest: {dataset_dir / 'hybrid_manifest.csv'}")
        except KeyboardInterrupt:
            stop_reason = stop_reason or "keyboard_interrupt_during_compile"
        except Exception as exc:
            fatal_error = exc
            automation_summary["error"] = f"{type(exc).__name__}: {exc}"

    final_status = "failed" if fatal_error else ("stopped" if stop_reason else "completed")
    finished_at = _utc_now()
    automation_summary.update(
        {
            "status": final_status,
            "updated_at": finished_at,
            "finished_at": finished_at,
            "stop_reason": stop_reason,
        }
    )
    _write_json_atomic(dataset_dir / "automation_summary.json", automation_summary)

    if last_selected:
        checkpoint(
            automation_summary["ntha_runs"],
            last_selected,
            None,
            final_status,
        )

    if stop_reason == "stop_file" and stop_path.exists():
        stop_path.unlink()
        print(f"Consumed stop request: {stop_path}")
    print(f"Automation summary: {dataset_dir / 'automation_summary.json'}")

    if stop_reason:
        print("Generation stopped safely. Re-run the same command to resume.")
    if fatal_error is not None:
        raise fatal_error


if __name__ == "__main__":
    main()
