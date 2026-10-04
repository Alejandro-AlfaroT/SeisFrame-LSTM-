"""
Data_Generation/Hybrid_Exporter.py
==================================

Compile completed RC Structure NTHA output folders into compact tensor samples
for hybrid GNN + LSTM training.

Expected input folder
---------------------
An NTHA result directory produced by Ground_Motion_Main.py or Main.py, e.g.
outputs/ntha/peer_1, containing:
  - nodes.csv
  - edges.csv
  - elements.csv
  - time_history.csv
  - story_drift_peaks.csv
  - node_envelope.csv
  - element_end_force_envelope.csv
  - global_parameters.json
  - record_summary_x.json
  - record_summary_y.json, optional for X-only runs
  - status.json
  - summary.json

Output
------
hybrid_sample.npz plus hybrid_metadata.json. The NPZ is intentionally neutral:
it can be loaded into PyTorch Geometric, vanilla PyTorch, or any custom hybrid
dataset class without depending on pandas.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


RC_STRUCTURE_DIR = Path(__file__).resolve().parents[1]
GROUND_MOTION_DIR = RC_STRUCTURE_DIR / "Ground_Motions"
GROUND_MOTION_MANIFEST = GROUND_MOTION_DIR / "metadata" / "record_manifest.csv"


ELEMENT_TYPE_ID = {
    "column": 0,
    "beam_x": 1,
    "beam_y": 2,
}

NODE_FEATURE_COLUMNS = [
    "x_in",
    "y_in",
    "z_in",
    "floor",
    "grid_i",
    "grid_j",
    "is_base",
    "is_floor_master",
]

EDGE_ATTR_COLUMNS = [
    "element_type_id",
    "length_in",
    "dir_x",
    "dir_y",
    "dir_z",
    "b_in",
    "h_in",
    "area_in2",
    "fc_ksi",
    "fy_ksi",
    "bar_area_in2",
    "top_bars",
    "bot_bars",
    "side_bars",
]

ELEMENT_FEATURE_COLUMNS = [
    "element_type_id",
    "length_in",
    "dir_x",
    "dir_y",
    "dir_z",
    "b_in",
    "h_in",
    "area_in2",
    "fc_ksi",
    "fy_ksi",
    "bar_area_in2",
    "top_bars",
    "bot_bars",
    "side_bars",
]

TIME_TARGET_COLUMNS = [
    "roof_disp_x_in",
    "roof_disp_y_in",
    "roof_disp_resultant_in",
]

STORY_DRIFT_COLUMNS = [
    "peak_drift_x_ratio",
    "peak_drift_y_ratio",
    "peak_drift_resultant_ratio",
]

NODE_ENVELOPE_COLUMNS = [
    "ux_max",
    "ux_min",
    "ux_abs_max",
    "uy_max",
    "uy_min",
    "uy_abs_max",
    "uz_max",
    "uz_min",
    "uz_abs_max",
    "rx_max",
    "rx_min",
    "rx_abs_max",
    "ry_max",
    "ry_min",
    "ry_abs_max",
    "rz_max",
    "rz_min",
    "rz_abs_max",
]

ELEMENT_FORCE_COLUMNS = [
    "axial_kip_max",
    "axial_kip_min",
    "shear_y_kip_max",
    "shear_y_kip_min",
    "shear_z_kip_max",
    "shear_z_kip_min",
    "torsion_kip_in_max",
    "torsion_kip_in_min",
    "moment_y_kip_in_max",
    "moment_y_kip_in_min",
    "moment_z_kip_in_max",
    "moment_z_kip_in_min",
]

GLOBAL_FEATURE_KEYS = [
    "num_bay_x",
    "num_bay_y",
    "num_floor",
    "bay_x_in",
    "bay_y_in",
    "story_h_in",
    "fc_col_ksi",
    "fc_beam_ksi",
    "fy_ksi",
    "b_col_in",
    "h_col_in",
    "b_beam_in",
    "h_beam_in",
    "col_bar_size",
    "col_top_bars",
    "col_bot_bars",
    "col_side_bars",
    "col_bar_area_in2",
    "col_stirrup_bar_size",
    "col_stirrup_legs",
    "col_stirrup_spacing_in",
    "beam_bar_size",
    "beam_top_bars",
    "beam_bot_bars",
    "beam_side_bars",
    "beam_bar_area_in2",
    "beam_stirrup_bar_size",
    "beam_stirrup_legs",
    "beam_stirrup_spacing_in",
    "floor_dead_load_ksf",
    "floor_live_load_ksf",
    "total_floor_gravity_load_kip",
    "element_formulation",
    "imk_apply_to_columns",
    "imk_apply_to_beams",
    "imk_hinge_stiffness_factor",
    "imk_beam_theta_y",
    "imk_column_theta_y",
    "imk_theta_p_pos",
    "imk_theta_pc_pos",
    "imk_theta_u_pos",
    "imk_fmaxfy_pos",
    "imk_fresfy_pos",
]

RECORD_FEATURE_KEYS = [
    "dt_sec",
    "npts",
    "duration_sec",
    "scale_factor",
    "pga_g",
    "pga_in_per_sec2",
]

HINGE_FEATURE_COLUMNS = [
    "ele_tag",
    "end_id",
    "axial_kip",
    "axial_ratio",
    "yield_moment_kip_in",
    "theta_y",
    "theta_p",
    "theta_pc",
    "theta_u",
    "rot_abs_max",
    "plastic_rotation",
    "damage_ratio",
    "yielded",
    "past_capping",
]

# Case-level damage summary. The inelasticity gate is recorded with the data
# rather than recomputed downstream so that what counted as inelastic is
# fixed at generation time and auditable afterwards.
DAMAGE_METRIC_COLUMNS = [
    "peak_interstory_drift_ratio",
    "roof_drift_ratio",
    "drift_concentration_factor",
    "residual_interstory_drift_ratio",
    "peak_hinge_damage_ratio",
    "fraction_hinges_yielded",
    "fraction_hinges_past_capping",
    "peak_base_shear_kip",
    "is_inelastic",
    "collapse_flag",
]

INELASTIC_DRIFT_THRESHOLD = 0.010
INELASTIC_ROOF_DRIFT_THRESHOLD = 0.00775
INELASTIC_DAMAGE_RATIO_THRESHOLD = 0.25
INELASTIC_YIELDED_FRACTION_THRESHOLD = 0.10
COLLAPSE_DRIFT_RATIO = 0.10

# Beyond this the solution has diverged, not deformed: a structure is gone well
# before 20% interstory drift. A diverged solve writes finite but meaningless
# numbers -- pilot30_s3 case_0003 recorded a drift ratio of 135 (13,508%) with
# a hinge damage ratio of 4.9e13 -- and those land in damage_metrics and
# target_peak, which embeddings.py and the training loaders read with raw
# np.load. Guarding only in schema_v3 was not enough: the surrogate's actual
# data path never goes through it. Metrics are therefore computed over the
# pre-divergence prefix, which also preserves the real collapse -- case_0022
# genuinely reached 9.9% drift before the solver lost it at 173%.
PHYSICAL_DRIFT_CEILING = 0.20

# plastic_rotation / theta_p. Past roughly this the section has disintegrated;
# larger values are divergence artifacts, not damage.
PHYSICAL_DAMAGE_RATIO_CEILING = 20.0

TARGET_PEAK_COLUMNS = [
    "max_abs_roof_disp_x_in",
    "max_abs_roof_disp_y_in",
    "max_story_drift_resultant_ratio",
    "max_story_drift_resultant_story",
    "analysis_failed",
    "completed_fraction",
    "num_recovery_steps",
    "recovery_rate",
]


def _read_csv(path):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    return path


def _to_float(value, default=math.nan):
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return default

    lowered = text.lower()
    if lowered in {"true", "yes", "y"}:
        return 1.0
    if lowered in {"false", "no", "n"}:
        return 0.0

    try:
        return float(text)
    except ValueError:
        return default


def _to_int(value, default=-1):
    number = _to_float(value, default=math.nan)
    if math.isnan(number):
        return default
    return int(number)


def _rows_to_array(rows, columns, dtype=np.float32):
    return np.asarray(
        [[_to_float(row.get(column), default=0.0) for column in columns] for row in rows],
        dtype=dtype,
    )


def _global_to_float(key, value):
    if key == "element_formulation":
        text = str(value or "").strip().lower()
        return {"fiber": 0.0, "imk": 1.0}.get(text, 0.0)
    return _to_float(value, default=0.0)


def _json_to_feature_array(data, keys):
    data = data or {}
    return np.asarray([_global_to_float(key, data.get(key)) for key in keys], dtype=np.float32)


def _resolve_acceleration_source(record_summary):
    source_value = (record_summary or {}).get("source_path")
    if source_value:
        source_path = Path(source_value)
        candidates = [source_path]
        if not source_path.is_absolute():
            candidates.insert(0, GROUND_MOTION_DIR / source_path)
        else:
            candidates.append(GROUND_MOTION_DIR / "processed" / source_path.name)

        for candidate in candidates:
            if candidate.exists():
                return candidate, (record_summary.get("units") or "in/sec^2")

    record_id = (record_summary or {}).get("record_id")
    if record_id and GROUND_MOTION_MANIFEST.exists():
        for row in _read_csv(GROUND_MOTION_MANIFEST):
            if row.get("record_id") != record_id:
                continue
            processed_file = (row.get("processed_file") or "").strip()
            if processed_file:
                processed_path = GROUND_MOTION_DIR / processed_file
                if processed_path.exists():
                    return processed_path, (row.get("processed_units") or "in/sec^2")
            break

    raise FileNotFoundError(
        "Ground motion source file not found for "
        f"{record_id or '<unknown record>'}: {source_value or '<no source_path>'}"
    )


def _acceleration_to_in_per_sec2(acceleration, units):
    normalized = str(units or "in/sec^2").strip().lower().replace(" ", "")
    if normalized in {"in/s2", "in/sec2", "in/sec^2"}:
        return acceleration
    if normalized in {"g", "grav", "gravity"}:
        return acceleration * 386.4
    if normalized in {"cm/s2", "cm/sec2", "cm/sec^2"}:
        return acceleration / 2.54
    if normalized in {"m/s2", "m/sec2", "m/sec^2"}:
        return acceleration * 39.37007874015748
    raise ValueError(f"Unsupported acceleration units in record summary: {units}")


def _load_acceleration_from_summary(record_summary, target_steps):
    if not record_summary:
        return np.zeros(target_steps, dtype=np.float32)

    path, source_units = _resolve_acceleration_source(record_summary)
    accel = np.loadtxt(path, dtype=np.float64)
    accel = _acceleration_to_in_per_sec2(
        np.atleast_1d(accel),
        source_units,
    )
    accel = (
        accel
        * _to_float(record_summary.get("scale_factor"), default=1.0)
    ).astype(np.float32)

    source_dt = _to_float(record_summary.get("dt_sec"), default=0.0)
    analysis_dt = _to_float(
        record_summary.get("analysis_dt_sec"),
        default=source_dt,
    )
    if (
        target_steps > 0
        and source_dt > 0.0
        and analysis_dt > 0.0
        and not math.isclose(source_dt, analysis_dt)
    ):
        source_time = np.arange(accel.size, dtype=np.float64) * source_dt
        target_time = np.arange(target_steps, dtype=np.float64) * analysis_dt
        return np.interp(
            target_time,
            source_time,
            accel,
            left=float(accel[0]),
            right=0.0,
        ).astype(np.float32)

    if accel.size >= target_steps:
        return accel[:target_steps]

    padded = np.zeros(target_steps, dtype=np.float32)
    padded[: accel.size] = accel
    return padded


def _portable_path(path, relative_to):
    return Path(
        os.path.relpath(
            Path(path).resolve(),
            start=Path(relative_to).resolve(),
        )
    ).as_posix()


def _status_features(status):
    status = status or {}
    requested = _to_float(status.get("npts_requested"), default=0.0)
    completed = _to_float(status.get("completed_steps"), default=0.0)
    convergence = status.get("convergence_summary") or {}
    return {
        "analysis_failed": 1.0 if status.get("failed") else 0.0,
        "completed_fraction": completed / requested if requested > 0 else 0.0,
        "num_recovery_steps": _to_float(convergence.get("num_recovery_steps"), default=0.0),
        "recovery_rate": _to_float(convergence.get("recovery_rate"), default=0.0),
    }


def _target_peak_array(summary, status):
    drift = (summary or {}).get("max_story_drift_resultant") or {}
    status_values = _status_features(status)
    values = {
        "max_abs_roof_disp_x_in": _to_float((summary or {}).get("max_abs_roof_disp_x_in"), default=0.0),
        "max_abs_roof_disp_y_in": _to_float((summary or {}).get("max_abs_roof_disp_y_in"), default=0.0),
        "max_story_drift_resultant_ratio": _to_float(drift.get("peak_drift_resultant_ratio"), default=0.0),
        "max_story_drift_resultant_story": _to_float(drift.get("story"), default=0.0),
        **status_values,
    }
    return np.asarray([values[column] for column in TARGET_PEAK_COLUMNS], dtype=np.float32)


def _damage_metrics(arrays, hinge_rows, global_parameters, status):
    """Case-level damage summary and the inelasticity gate.

    Peak drift alone is a poor gate: in the predecessor dataset 63 percent of
    cases already had a hinge at yield while almost none showed meaningful
    plastic action, because the hinge moment saturates at the capping
    strength. The gate therefore combines a global drift measure with a
    rotation-based one that does not saturate.
    """
    story_drift = arrays.get("story_drift")
    num_floor = int(_to_float(global_parameters.get("num_floor"), default=0.0)) or 1
    story_height = _to_float(global_parameters.get("story_h_in"), default=0.0)
    total_height = num_floor * story_height

    peak_drift = 0.0
    residual_drift = 0.0
    physical_steps = None
    truncated_steps = 0
    if story_drift is not None and story_drift.size:
        drift_x = story_drift[:, :num_floor]
        drift_y = story_drift[:, num_floor:2 * num_floor]
        resultant = np.sqrt(drift_x**2 + drift_y**2)

        # Keep only the prefix the solver was still describing a structure in.
        per_step = resultant.max(axis=1)
        physical = np.flatnonzero(per_step <= PHYSICAL_DRIFT_CEILING)
        if physical.size and physical[-1] + 1 < resultant.shape[0]:
            physical_steps = int(physical[-1] + 1)
            truncated_steps = int(resultant.shape[0] - physical_steps)
            resultant = resultant[:physical_steps]
        elif not physical.size:
            # Diverged before ever being physical; nothing usable here.
            physical_steps = 0
            truncated_steps = int(resultant.shape[0])
            resultant = resultant[:0]

        if resultant.size:
            peak_drift = float(resultant.max())
            # Average the final few percent of steps so a single noisy last step
            # does not decide the residual.
            tail = max(1, int(0.02 * resultant.shape[0]))
            residual_drift = float(resultant[-tail:].max(axis=1).mean())

    roof_drift = 0.0
    floor_disp = arrays.get("floor_disp")
    if floor_disp is not None and floor_disp.size and total_height > 0:
        roof = np.hypot(floor_disp[:, -2], floor_disp[:, -1])
        if physical_steps is not None:
            roof = roof[:physical_steps]
        roof_drift = float(roof.max() / total_height) if roof.size else 0.0

    peak_shear = 0.0
    base_shear = arrays.get("base_shear")
    if base_shear is not None and base_shear.size:
        peak_shear = float(np.abs(base_shear).max())

    # Hinge damage comes from the envelope, computed over the whole analysis,
    # so truncating the arrays above does not clean it. Drop the values that
    # are divergence artifacts rather than letting one 4.9e13 set the peak.
    damage = [
        float(row.get("damage_ratio") or 0.0) for row in hinge_rows
        if abs(float(row.get("damage_ratio") or 0.0)) <= PHYSICAL_DAMAGE_RATIO_CEILING
    ]
    yielded = [int(float(row.get("yielded") or 0)) for row in hinge_rows]
    capping = [int(float(row.get("past_capping") or 0)) for row in hinge_rows]
    peak_damage = max(damage) if damage else 0.0
    yielded_fraction = (sum(yielded) / len(yielded)) if yielded else 0.0
    capping_fraction = (sum(capping) / len(capping)) if capping else 0.0

    collapse = bool(
        peak_drift >= COLLAPSE_DRIFT_RATIO
        or (status.get("failed") and peak_drift >= INELASTIC_DRIFT_THRESHOLD)
    )
    inelastic = bool(
        peak_drift >= INELASTIC_DRIFT_THRESHOLD
        and roof_drift >= INELASTIC_ROOF_DRIFT_THRESHOLD
        and peak_damage >= INELASTIC_DAMAGE_RATIO_THRESHOLD
        and yielded_fraction >= INELASTIC_YIELDED_FRACTION_THRESHOLD
    )

    return {
        "peak_interstory_drift_ratio": peak_drift,
        "roof_drift_ratio": roof_drift,
        "drift_concentration_factor": (peak_drift / roof_drift) if roof_drift > 0 else 0.0,
        "residual_interstory_drift_ratio": residual_drift,
        "peak_hinge_damage_ratio": peak_damage,
        "fraction_hinges_yielded": yielded_fraction,
        "fraction_hinges_past_capping": capping_fraction,
        # Not in DAMAGE_METRIC_COLUMNS -- the array is built from that fixed
        # list, so these ride along for the metadata rather than widening it.
        "physical_steps": float(physical_steps if physical_steps is not None else -1),
        "truncated_steps": float(truncated_steps),
        "damage_values_discarded": float(len(hinge_rows) - len(damage)),
        "peak_base_shear_kip": peak_shear,
        "is_inelastic": 1.0 if inelastic else 0.0,
        "collapse_flag": 1.0 if collapse else 0.0,
    }


def _completed_ok(status):
    if not status:
        return False
    requested = _to_int(status.get("npts_requested"), default=0)
    completed = _to_int(status.get("completed_steps"), default=0)
    return not bool(status.get("failed")) and requested > 0 and completed == requested


def _sample_is_stale(ntha_dir, sample_path):
    """True when current analysis evidence no longer matches the sample.

    A retried run re-executes the NTHA but, with overwrite off, the existing
    sample was returned untouched -- so the sample described a previous
    attempt, not the analysis on disk. r150_s275 case_0003 carried a sample
    from four days before its NTHA, with a different step count. Comparing
    against status.json, which every completed analysis rewrites last, makes
    the sample follow the analysis without the caller having to know that a
    retry happened. New sidecars bind the exact status bytes by SHA256;
    legacy samples fall back to timestamps. A missing status is not reusable.
    """
    status_path = Path(ntha_dir) / "status.json"
    sample_path = Path(sample_path)
    if not sample_path.exists():
        return False
    if not status_path.exists():
        return True
    metadata = _read_json(sample_path.with_name("hybrid_metadata.json"), default={}) or {}
    digest = metadata.get("analysis_status_sha256")
    if digest:
        return digest != hashlib.sha256(status_path.read_bytes()).hexdigest()
    return status_path.stat().st_mtime_ns > sample_path.stat().st_mtime_ns


def required_files_present(ntha_dir):
    ntha_dir = Path(ntha_dir)
    required = [
        "nodes.csv",
        "edges.csv",
        "elements.csv",
        "time_history.csv",
        "story_drift_peaks.csv",
        "node_envelope.csv",
        "element_end_force_envelope.csv",
        "global_parameters.json",
        "record_summary_x.json",
        "status.json",
        "summary.json",
        "response_arrays.npz",
        "hinge_backbone.csv",
    ]
    missing = [name for name in required if not (ntha_dir / name).exists()]
    return missing


def compile_hybrid_sample(
    ntha_dir,
    output_dir=None,
    require_success=True,
    overwrite=False,
):
    """
    Compile one NTHA result directory into hybrid_sample.npz.

    Parameters
    ----------
    ntha_dir : str or Path
        Completed NTHA output directory.
    output_dir : str or Path, optional
        Destination directory. Defaults to ntha_dir.
    require_success : bool
        If True, skip failed or partial analyses.
    overwrite : bool
        If False and hybrid_sample.npz already exists, keep it.
    """
    ntha_dir = Path(ntha_dir)
    output_dir = Path(output_dir) if output_dir is not None else ntha_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_path = output_dir / "hybrid_sample.npz"
    metadata_path = output_dir / "hybrid_metadata.json"
    if sample_path.exists() and not overwrite and not _sample_is_stale(ntha_dir, sample_path):
        metadata = _read_json(metadata_path, default={}) or {}
        portable_source = _portable_path(ntha_dir, output_dir)
        portable_sample = _portable_path(sample_path, output_dir)
        needs_refresh = (
            not metadata_path.exists()
            or "reinforcement_geometry" not in metadata
            or metadata.get("run_name") != ntha_dir.name
            or metadata.get("source_ntha_dir") != portable_source
            or metadata.get("sample_npz") != portable_sample
        )
        metadata["run_name"] = ntha_dir.name
        metadata["source_ntha_dir"] = portable_source
        metadata["sample_npz"] = portable_sample
        # Reusing an existing tensor is not evidence that current source files
        # describe its reinforcement. Keep its recorded sidecar; missing old
        # provenance stays unknown instead of being inferred from live inputs.
        metadata.setdefault("reinforcement_geometry", None)
        if needs_refresh:
            _write_json(metadata_path, metadata)
        return metadata

    missing = required_files_present(ntha_dir)
    if missing:
        raise FileNotFoundError(f"{ntha_dir} is missing required NTHA files: {missing}")

    status_digest = hashlib.sha256((ntha_dir / "status.json").read_bytes()).hexdigest()
    nodes = _read_csv(ntha_dir / "nodes.csv")
    edges = _read_csv(ntha_dir / "edges.csv")
    elements = _read_csv(ntha_dir / "elements.csv")
    time_rows = _read_csv(ntha_dir / "time_history.csv")
    story_rows = _read_csv(ntha_dir / "story_drift_peaks.csv")
    node_env_rows = _read_csv(ntha_dir / "node_envelope.csv")
    force_env_rows = _read_csv(ntha_dir / "element_end_force_envelope.csv")

    status = _read_json(ntha_dir / "status.json", default={}) or {}
    summary = _read_json(ntha_dir / "summary.json", default={}) or {}
    global_parameters = _read_json(ntha_dir / "global_parameters.json", default={}) or {}
    record_x = _read_json(ntha_dir / "record_summary_x.json", default={}) or {}
    record_y = _read_json(ntha_dir / "record_summary_y.json", default=None)

    arrays = dict(np.load(ntha_dir / "response_arrays.npz"))
    hinge_rows = _read_csv(ntha_dir / "hinge_backbone.csv")
    damage = _damage_metrics(arrays, hinge_rows, global_parameters, status)

    # A run that stopped early because the structure collapsed is the most
    # informative sample in a fragility dataset, not a failure to discard.
    # It is kept and labelled. A run that stopped early while still nearly
    # elastic is a genuine numerical failure and is still rejected.
    collapsed = bool(damage["collapse_flag"])
    has_data = _to_int(status.get("completed_steps"), default=0) > 0
    if require_success and not _completed_ok(status) and not (collapsed and has_data):
        raise RuntimeError(f"NTHA run is not a complete successful sample: {ntha_dir}")

    n_steps = len(time_rows)
    if n_steps <= 0:
        raise RuntimeError(f"No time history rows found in {ntha_dir}")

    node_tags = np.asarray([_to_int(row.get("node_tag")) for row in nodes], dtype=np.int64)
    element_tags = np.asarray([_to_int(row.get("ele_tag")) for row in elements], dtype=np.int64)
    edge_element_tags = np.asarray([_to_int(row.get("ele_tag")) for row in edges], dtype=np.int64)

    x = _rows_to_array(nodes, NODE_FEATURE_COLUMNS)
    edge_index = np.asarray(
        [
            [_to_int(row.get("source")) - 1 for row in edges],
            [_to_int(row.get("target")) - 1 for row in edges],
        ],
        dtype=np.int64,
    )
    edge_attr = _rows_to_array(edges, EDGE_ATTR_COLUMNS)
    element_attr = _rows_to_array(elements, ELEMENT_FEATURE_COLUMNS)

    time_seconds = np.asarray([_to_float(row.get("time_sec"), default=0.0) for row in time_rows], dtype=np.float32)
    gm_x = _load_acceleration_from_summary(record_x, n_steps)
    gm_y = _load_acceleration_from_summary(record_y, n_steps)
    ground_motion = np.stack([gm_x, gm_y], axis=1).astype(np.float32)
    # An x-only run has no Y record, so its Y acceleration column and Y record
    # features are zero-filled. Zeros are a valid acceleration value, so the
    # arrays alone cannot distinguish "not excited in Y" from "absent". This
    # mask makes the distinction explicit for consumers, the same way
    # node_envelope_mask does for nodes missing an envelope row.
    ground_motion_present = np.asarray(
        [1.0 if record_x else 0.0, 1.0 if record_y else 0.0], dtype=np.float32
    )
    response_time_history = _rows_to_array(time_rows, TIME_TARGET_COLUMNS)

    story_drift_peaks = _rows_to_array(story_rows, STORY_DRIFT_COLUMNS)
    story_ids = np.asarray([_to_int(row.get("story")) for row in story_rows], dtype=np.int64)
    node_env_by_tag = {
        _to_int(row.get("node_tag")): row
        for row in node_env_rows
    }
    node_envelope = np.zeros((len(node_tags), len(NODE_ENVELOPE_COLUMNS)), dtype=np.float32)
    node_envelope_mask = np.zeros(len(node_tags), dtype=np.float32)
    for index, tag in enumerate(node_tags):
        row = node_env_by_tag.get(int(tag))
        if row is None:
            continue
        node_envelope[index, :] = _rows_to_array([row], NODE_ENVELOPE_COLUMNS)[0]
        node_envelope_mask[index] = 1.0
    node_envelope_tags = node_tags.copy()

    element_type_lookup = {row["ele_tag"]: ELEMENT_TYPE_ID.get(row.get("element_type"), -1) for row in elements}
    element_force_index = np.asarray(
        [
            [
                _to_int(row.get("ele_tag")),
                _to_int(row.get("node_tag")),
                0 if row.get("end") == "i" else 1,
                element_type_lookup.get(row.get("ele_tag"), ELEMENT_TYPE_ID.get(row.get("element_type"), -1)),
            ]
            for row in force_env_rows
        ],
        dtype=np.int64,
    )
    element_force_envelope = _rows_to_array(force_env_rows, ELEMENT_FORCE_COLUMNS)

    hinge_features = _rows_to_array(hinge_rows, HINGE_FEATURE_COLUMNS)
    damage_metrics = np.asarray(
        [damage[column] for column in DAMAGE_METRIC_COLUMNS], dtype=np.float32
    )
    if global_parameters.get("design_mode") == "grouped":
        # The global feature vector holds one column section, one beam section and one cage of each. A grouped
        # design has none of those; a missing value would be read as zero here, which is not a description of it.
        raise ValueError("This case was built from a grouped design: the global feature vector (GLOBAL_FEATURE_KEYS) has "
                         "one-section entries that do not exist for it. Its per-member sections, cages and groups are in "
                         "elements.csv; the hybrid sample needs a declared replacement for those global entries first.")
    global_features = _json_to_feature_array(global_parameters, GLOBAL_FEATURE_KEYS)
    record_features = np.stack(
        [
            _json_to_feature_array(record_x, RECORD_FEATURE_KEYS),
            _json_to_feature_array(record_y, RECORD_FEATURE_KEYS),
        ],
        axis=0,
    )
    target_peak = _target_peak_array(summary, status)

    if hashlib.sha256((ntha_dir / "status.json").read_bytes()).hexdigest() != status_digest:
        raise RuntimeError("Analysis status changed while compiling the sample; retry after the analysis finishes.")
    # Synchronized hinge moments (2026-10-01): the conjugate of hinge_rotation on the same rows, with
    # the explicit time and commit count of each row. Written only when the analysis produced them: a
    # run from before the moment history has no such array, and a None here would be stored as a
    # pickled object that a plain np.load refuses.
    hinge_moment_arrays = {name: arrays[name] for name in ("hinge_moment", "hinge_history_time", "hinge_history_commit_count")
                           if arrays.get(name) is not None}
    hinge_schema_path = ntha_dir / "hinge_moment_rotation_schema.json"
    hinge_schema = _read_json(hinge_schema_path) if hinge_schema_path.exists() else None
    np.savez_compressed(
        sample_path,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_tags=node_tags,
        element_tags=element_tags,
        edge_element_tags=edge_element_tags,
        element_attr=element_attr,
        time_seconds=time_seconds,
        ground_motion=ground_motion,
        ground_motion_present=ground_motion_present,
        response_time_history=response_time_history,
        story_ids=story_ids,
        story_drift_peaks=story_drift_peaks,
        node_envelope_tags=node_envelope_tags,
        node_envelope=node_envelope,
        node_envelope_mask=node_envelope_mask,
        element_force_index=element_force_index,
        element_force_envelope=element_force_envelope,
        global_features=global_features,
        record_features=record_features,
        target_peak=target_peak,
        hinge_features=hinge_features,
        damage_metrics=damage_metrics,
        floor_disp=arrays.get("floor_disp"),
        floor_vel=arrays.get("floor_vel"),
        floor_accel=arrays.get("floor_accel"),
        story_drift_history=arrays.get("story_drift"),
        base_shear=arrays.get("base_shear"),
        hinge_rotation=arrays.get("hinge_rotation"),
        element_force_history=arrays.get("element_force"),
        element_force_steps=arrays.get("element_force_steps"),
        element_force_tag_order=arrays.get("element_tag_order"),
        joint_force_history=arrays.get("joint_force"),
        joint_force_node_order=arrays.get("joint_node_order"),
        hinge_rotation_steps=arrays.get("hinge_rotation_steps"),
        hinge_tag_order=arrays.get("hinge_tag_order"),
        floor_master_nodes=arrays.get("floor_master_nodes"),
        **hinge_moment_arrays,
    )

    metadata = {
        "schema_version": "hybrid_gnn_lstm_v3",
        "analysis_status_sha256": status_digest,
        "run_name": ntha_dir.name,
        "source_ntha_dir": _portable_path(ntha_dir, output_dir),
        "sample_npz": _portable_path(sample_path, output_dir),
        "num_nodes": int(x.shape[0]),
        "num_edges": int(edge_index.shape[1]),
        "num_elements": int(element_attr.shape[0]),
        "num_time_steps": int(n_steps),
        "num_stories": int(len(story_rows)),
        "analysis_failed": bool(status.get("failed")),
        "completed_steps": _to_int(status.get("completed_steps"), default=0),
        "npts_requested": _to_int(status.get("npts_requested"), default=0),
        "record_id_x": record_x.get("record_id"),
        "record_id_y": (record_y or {}).get("record_id"),
        "node_feature_columns": NODE_FEATURE_COLUMNS,
        "edge_attr_columns": EDGE_ATTR_COLUMNS,
        "element_feature_columns": ELEMENT_FEATURE_COLUMNS,
        "ground_motion_columns": ["accel_x_in_per_sec2", "accel_y_in_per_sec2"],
        "x_only": record_y is None,
        "ground_motion_present_description": (
            "1.0 where that ground-motion component was applied; an x-only run "
            "is 0.0 in Y, whose acceleration column and record features are "
            "zero-filled rather than measured."
        ),
        "response_time_history_columns": TIME_TARGET_COLUMNS,
        "story_drift_peak_columns": STORY_DRIFT_COLUMNS,
        "node_envelope_columns": NODE_ENVELOPE_COLUMNS,
        "node_envelope_mask_description": "1.0 where node envelope was recorded; base nodes are 0.0 when absent from NTHA node_envelope.csv.",
        "element_force_index_columns": ["ele_tag", "node_tag", "end_id", "element_type_id"],
        "element_force_envelope_columns": ELEMENT_FORCE_COLUMNS,
        "global_feature_keys": GLOBAL_FEATURE_KEYS,
        # Preserve the 43-column tensor contract. Explicit slab/load provenance
        # is a sidecar field; it is not silently appended to trained inputs.
        "floor_loads": global_parameters.get("floor_loads"),
        "reinforcement_geometry": global_parameters.get("reinforcement_geometry"),
        "record_feature_keys": RECORD_FEATURE_KEYS,
        "target_peak_columns": TARGET_PEAK_COLUMNS,
        "hinge_feature_columns": HINGE_FEATURE_COLUMNS,
        "element_force_history_columns": [
            "axial_i", "shear_y_i", "shear_z_i", "torsion_i", "moment_y_i", "moment_z_i",
            "axial_j", "shear_y_j", "shear_z_j", "torsion_j", "moment_y_j", "moment_z_j",
        ],
        "element_force_history_frame": "member local",
        "joint_force_history_columns": ["fx", "fy", "fz", "mx", "my", "mz"],
        "joint_force_history_frame": "global",
        "joint_force_description": (
            "Internal force delivered to each joint, summed over the element "
            "ends framing into it, in the global frame."
        ),
        "damage_metric_columns": DAMAGE_METRIC_COLUMNS,
        # Divergence bookkeeping. truncated_steps > 0 means the solution left
        # the physical range and the metrics above describe only the prefix
        # before that; damage_values_discarded counts hinge damage ratios
        # rejected as divergence artifacts.
        "physical_steps": damage.get("physical_steps", -1.0),
        "truncated_steps": damage.get("truncated_steps", 0.0),
        "damage_values_discarded": damage.get("damage_values_discarded", 0.0),
        "physical_drift_ceiling": PHYSICAL_DRIFT_CEILING,
        "num_hinges": int(hinge_features.shape[0]) if hinge_features.size else 0,
        # hinge_rotation and hinge_moment share rows (hinge_rotation_steps), entity order
        # (hinge_tag_order) and columns: two per hinge, local y (direction 5) then local z (direction 6).
        "hinge_history_columns": ["local_y_direction_5", "local_z_direction_6"],
        "hinge_moment_present": "hinge_moment" in hinge_moment_arrays,
        "hinge_moment_rotation": (
            {key: hinge_schema.get(key) for key in ("schema_version", "available", "units", "quantities", "sign_convention",
                                                    "reference_state", "sampling", "missing", "identity", "status", "claims_not_made")}
            if hinge_schema else
            {"available": False, "reason": "the analysis output has no hinge_moment_rotation_schema.json (run predates the moment history)"}),
        "hinge_history_stride": int(
            arrays["hinge_rotation_steps"][1] - arrays["hinge_rotation_steps"][0]
        ) if arrays.get("hinge_rotation_steps") is not None
             and len(arrays["hinge_rotation_steps"]) > 1 else 1,
        "damage": damage,
        "is_inelastic": bool(damage["is_inelastic"]),
        "collapse": bool(damage["collapse_flag"]),
        "inelastic_gate": {
            "peak_interstory_drift_ratio": INELASTIC_DRIFT_THRESHOLD,
            "roof_drift_ratio": INELASTIC_ROOF_DRIFT_THRESHOLD,
            "peak_hinge_damage_ratio": INELASTIC_DAMAGE_RATIO_THRESHOLD,
            "fraction_hinges_yielded": INELASTIC_YIELDED_FRACTION_THRESHOLD,
        },
        "summary": {
            "max_abs_roof_disp_x_in": summary.get("max_abs_roof_disp_x_in"),
            "max_abs_roof_disp_y_in": summary.get("max_abs_roof_disp_y_in"),
            "max_story_drift_resultant": summary.get("max_story_drift_resultant"),
            "elapsed_sec": summary.get("elapsed_sec"),
        },
    }
    _write_json(metadata_path, metadata)
    return metadata


def discover_ntha_runs(ntha_root):
    ntha_root = Path(ntha_root)
    if not ntha_root.exists():
        return []
    return sorted(
        path for path in ntha_root.iterdir()
        if path.is_dir() and (path / "summary.json").exists()
    )


def compile_ntha_root(
    ntha_root,
    dataset_dir=None,
    require_success=True,
    overwrite=False,
    limit=None,
):
    ntha_root = Path(ntha_root)
    if dataset_dir is None:
        dataset_dir = ntha_root.parent / "hybrid_dataset"
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    runs = discover_ntha_runs(ntha_root)
    if limit is not None:
        runs = runs[:limit]

    for run_dir in runs:
        out_dir = dataset_dir / run_dir.name
        already_compiled = (out_dir / "hybrid_sample.npz").exists()
        row = {
            "run_name": run_dir.name,
            "source_ntha_dir": _portable_path(run_dir, dataset_dir),
            "sample_dir": _portable_path(out_dir, dataset_dir),
            "sample_npz": _portable_path(
                out_dir / "hybrid_sample.npz",
                dataset_dir,
            ),
            "compiled": False,
            "skipped": False,
            "error": "",
        }
        try:
            metadata = compile_hybrid_sample(
                run_dir,
                output_dir=out_dir,
                require_success=require_success,
                overwrite=overwrite,
            )
            row.update(
                {
                    "compiled": True,
                    "skipped": already_compiled and not overwrite,
                    "num_nodes": metadata.get("num_nodes"),
                    "num_edges": metadata.get("num_edges"),
                    "num_elements": metadata.get("num_elements"),
                    "num_time_steps": metadata.get("num_time_steps"),
                    "record_id_x": metadata.get("record_id_x"),
                    "record_id_y": metadata.get("record_id_y"),
                    "analysis_failed": metadata.get("analysis_failed"),
                }
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            if require_success:
                rows.append(row)
                continue
        rows.append(row)

    fieldnames = [
        "run_name",
        "source_ntha_dir",
        "sample_dir",
        "sample_npz",
        "compiled",
        "skipped",
        "num_nodes",
        "num_edges",
        "num_elements",
        "num_time_steps",
        "record_id_x",
        "record_id_y",
        "analysis_failed",
        "error",
    ]
    _write_csv(dataset_dir / "hybrid_manifest.csv", fieldnames, rows)
    _write_json(dataset_dir / "hybrid_manifest.json", rows)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile RC Structure NTHA outputs into hybrid GNN/LSTM tensor samples."
    )
    parser.add_argument("--ntha-dir", help="Compile one NTHA run directory.")
    parser.add_argument(
        "--ntha-root",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "ntha"),
        help="Root directory containing NTHA run folders.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Destination dataset directory. Defaults to outputs/hybrid_dataset.",
    )
    parser.add_argument("--allow-failed", action="store_true", help="Compile failed/partial runs too.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing hybrid_sample.npz files.")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    require_success = not args.allow_failed

    if args.ntha_dir:
        metadata = compile_hybrid_sample(
            args.ntha_dir,
            output_dir=args.dataset_dir,
            require_success=require_success,
            overwrite=args.overwrite,
        )
        print(json.dumps(metadata, indent=2))
        return

    rows = compile_ntha_root(
        args.ntha_root,
        dataset_dir=args.dataset_dir,
        require_success=require_success,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    compiled = sum(1 for row in rows if row.get("compiled"))
    failed = [row for row in rows if row.get("error")]
    print(f"Compiled {compiled}/{len(rows)} NTHA runs.")
    if failed:
        print(f"Skipped/failed {len(failed)} run(s). See hybrid_manifest.csv.")


if __name__ == "__main__":
    main()
