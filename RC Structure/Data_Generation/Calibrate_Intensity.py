"""
Data_Generation/Calibrate_Intensity.py
======================================

Turn a pilot batch into an intensity rule that produces a target drift
distribution.

Why a rule rather than a ladder
-------------------------------
A uniform set of scale factors applied to every record wastes most of the
compute. The drift a record produces depends on how hard it shakes a
structure at that structure's own period, and that varies by roughly an order
of magnitude across a dataset spanning four to nine stories and five hazard
levels. A uniform ladder therefore leaves short stiff frames elastic while
driving tall flexible ones to collapse, and the resulting drift distribution
is whatever falls out rather than what was wanted.

This module fits the relationship on a pilot batch and inverts it: given a
target drift, it returns the scale factor that should produce it.

The model
---------
    log(drift) = a + b log(SaGM(T1)) + c log(T1)

fitted by least squares on completed pilot runs. b near one is the
equal-displacement rule; letting it float lets the data disagree. The T1 term
absorbs the systematic difference between short and tall frames that remains
after normalizing by spectral demand.

Inverting for a target drift:

    log(Sa_required) = (log(target) - a - c log(T1)) / b
    scale_factor     = Sa_required / SaGM_unscaled(T1)

Scheduler constraint
--------------------
The scheduler must not import OpenSees, so it cannot compute T1 or a response
spectrum when it builds a plan. The calibration artifact therefore carries
everything needed to derive a scale factor from geometry alone: the fitted
coefficients, an empirical T1/Ta ratio so the period can be estimated from
the ASCE approximate formula, and the precomputed spectrum of every eligible
record pair.

Usage
-----
    python Calibrate_Intensity.py --dataset-root <pilot root> \\
        --output intensity_calibration.json
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import glob
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np


RC_DIR = Path(os.environ.get("RC_STRUCTURE_DIR", Path(__file__).resolve().parents[1])).resolve()
if str(RC_DIR) not in sys.path:
    sys.path.insert(0, str(RC_DIR))

from Analysis.Response_Spectrum import (  # noqa: E402
    SPECTRUM_PERIODS_SEC,
    pseudo_spectral_acceleration,
)

GROUND_MOTION_DIR = RC_DIR / "Ground_Motions"
RECORD_MANIFEST = GROUND_MOTION_DIR / "metadata" / "record_manifest.csv"

CALIBRATION_SCHEMA = "rc_intensity_calibration_v1"

# Target share of cases in each peak interstory drift band. Deliberately keeps
# a band of near-elastic cases: a surrogate trained only on damaged structures
# is unreliable at service levels, which is where most of a fragility curve's
# probability mass sits.
# Deliberately weighted toward heavy nonlinear response. This dataset trains a
# surrogate to predict the regime that is hard to analyse, so it over-samples
# damage relative to real seismic hazard, where a building portfolio is
# overwhelmingly elastic. That is a stated design choice, not an accident, and
# belongs in any methods description of the dataset.
#
# The shares are set from measurement, not intuition. Across 88 pilot runs the
# is_inelastic gate passes 0% below 1.25% drift, 18% in 1.5-2.0%, and 17 of 17
# above 2.0% -- the gate is dominated by hinge damage ratio, which tracks drift
# closely but crosses its 0.25 threshold near 2% drift, not the 1% the bands
# were previously built around. The old shares put 30% of the mass in the
# 1-2% band, which contributes almost nothing (9% pass), and expected only
# ~38% overall. Moving that mass into the severe band -- fully reachable
# within the 3.0 scale cap -- raises the expectation to ~66%, or ~60% after
# the near-collapse band's cap-limited losses.
TARGET_DRIFT_BANDS = (
    ("elastic", 0.0025, 0.005, 0.10),
    ("onset", 0.005, 0.010, 0.10),
    ("developed", 0.010, 0.020, 0.15),
    ("severe", 0.020, 0.040, 0.50),
    ("near_collapse", 0.040, 0.080, 0.15),
)

SCALE_FACTOR_MIN = 0.25
# Scaling a record far beyond its own intensity distorts spectral shape
# relative to the magnitude and distance it actually came from, so the
# response stops being representative of the hazard it claims to model.
# Capped at 3 deliberately. Measured consequence with record-target matching:
# every target band up to 4% drift stays fully reachable, and only the
# near-collapse band (4-8%) loses ground -- 40% of it, about 4% of all runs.
SCALE_FACTOR_MAX = 3.0

# Interstory drift above this is not a structure, it is a diverged solution.
# Collapse is conventionally taken at 5-10% drift; 20% is generous headroom
# for a genuine collapse still being integrated. Beyond it the model has no
# physical meaning and the number must not reach a least-squares fit.
COLLAPSE_DRIFT_CEILING = 0.20

# Fallback used when a pilot is too small to fit. b = 1 is equal displacement;
# c = 0 assumes no residual height dependence.
DEFAULT_COEFFICIENTS = {"a": math.log(0.02), "b": 1.0, "c": 0.0}
DEFAULT_PERIOD_RATIO = 1.2


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def record_spectra(record_ids=None):
    """5%-damped spectrum for every processed record, keyed by record id."""
    spectra = {}
    for row in _read_csv(RECORD_MANIFEST):
        record_id = row["record_id"]
        if record_ids is not None and record_id not in record_ids:
            continue
        processed = (row.get("processed_file") or "").strip()
        if not processed:
            continue
        path = GROUND_MOTION_DIR / processed.replace("/", os.sep)
        if not path.exists():
            continue
        acceleration = np.loadtxt(path)
        spectra[record_id] = pseudo_spectral_acceleration(
            acceleration, float(row["dt_sec"])
        )
    return spectra


def _sa_at(spectrum, period):
    return float(np.interp(period, SPECTRUM_PERIODS_SEC, np.asarray(spectrum)))


def geometric_mean_sa(spectra, record_x, record_y, period):
    """SaGM at one period for a record pair, in g."""
    sa_x = _sa_at(spectra[record_x], period) if record_x in spectra else 0.0
    sa_y = _sa_at(spectra[record_y], period) if record_y in spectra else sa_x
    return math.sqrt(max(sa_x, 1.0e-9) * max(sa_y, 1.0e-9))


def collect_pilot_observations(dataset_roots):
    """Read completed runs: period, record pair, applied scale, peak drift.

    Exclusions, as implemented: a failed run below 1% drift (it died before it
    said anything) and any run above COLLAPSE_DRIFT_CEILING (a diverged solve).
    A failed run at or above 1% is kept as an ordinary observation; nothing is
    censored.
    """
    observations = []
    design_schemas = {}
    for root in dataset_roots:
        pattern = str(Path(root) / "cases" / "*" / "ntha" / "*" / "summary.json")
        for path_str in sorted(glob.glob(pattern)):
            path = Path(path_str)
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue

            status = summary.get("status") or {}
            drift = (summary.get("max_story_drift_resultant") or {}).get(
                "peak_drift_resultant_ratio"
            )
            period = summary.get("elastic_reference_period_mode_1_sec")
            record_x = (summary.get("record_summary_x") or {}).get("record_id")
            record_y = (summary.get("record_summary_y") or {}).get("record_id") or record_x
            scale = float(
                (summary.get("record_summary_x") or {}).get("scale_factor") or 1.0
            )
            if not (drift and period and record_x) or drift <= 0.0:
                continue
            # Failed runs are kept only if they got far enough to be a real
            # collapse; a run that died early understates its own drift.
            if status.get("failed") and drift < 0.01:
                continue
            # ...and only if the number is still physical. A diverged solve
            # reports nonsense: pilot30_s3 case_0003 recorded a drift ratio of
            # 135 (13,508%) with hinge rotations of 1.9e12 rad after collapsing
            # legitimately at ~5.6%. That single point enters the fit 7.7 log
            # units off and would drag every coefficient with it, so the
            # ceiling applies to converged and failed runs alike.
            if drift > COLLAPSE_DRIFT_CEILING:
                continue

            global_parameters = json.loads(
                (path.parent / "global_parameters.json").read_text(encoding="utf-8")
            ) if (path.parent / "global_parameters.json").exists() else {}

            # The design route and hinge model this run was produced with, so the
            # artifact can say what it is valid for (see MODEL_IDENTITY_KEYS).
            case_dir = path.parents[2]
            if case_dir not in design_schemas:
                design_schemas[case_dir] = _design_schema_version(case_dir / "design.json")
            model = {key: global_parameters.get(key) for key in MODEL_IDENTITY_KEYS}
            model["design_schema_version"] = design_schemas[case_dir]

            observations.append(
                {
                    "case_id": case_dir.name,
                    "run_name": path.parent.name,
                    "period_sec": float(period),
                    "record_x": record_x,
                    "record_y": record_y,
                    "scale_factor": scale,
                    "peak_drift_ratio": float(drift),
                    "num_floor": int(global_parameters.get("num_floor") or 0),
                    "story_h_in": float(global_parameters.get("story_h_in") or 0.0),
                    "model": model,
                }
            )
    return observations


def _design_schema_version(design_path):
    try:
        return json.loads(Path(design_path).read_text(encoding="utf-8")).get("schema_version")
    except (OSError, ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Model identity: what a calibration is valid for
# ---------------------------------------------------------------------------
# A drift model fitted on one design route or hinge model does not transfer to
# another. The 2026-09-06/08 artifacts were fitted on rc_design_v1 designs with
# IMKBilin hinges and 10-14 ft stories; the current route differs on every
# count, and until 2026-09-26 nothing recorded that. The artifact now carries
# the identity of the pilot it came from, and the generator refuses one that
# does not match the running code unless told otherwise.
MODEL_IDENTITY_KEYS = (
    "design_schema_version",
    "element_formulation",
    "imk_material_type",
    "imk_deterioration_mode",
    "imk_use_calibrated_backbone",
    "imk_hinge_stiffness_mode",
    "imk_hinge_stiffness_factor",
)


def current_model_identity():
    """The design route and hinge model the running code would generate with.

    Read without importing the design driver, which imports OpenSees: the
    scheduler must stay OpenSees-free. Structure_Parameters is plain data.
    """
    import Structure_Parameters as sp

    source = (RC_DIR / "Design" / "Design_Driver.py").read_text(encoding="utf-8")
    match = re.search(r'^DESIGN_SCHEMA_VERSION\s*=\s*"([^"]+)"', source, re.M)
    return {
        "design_schema_version": match.group(1) if match else None,
        "element_formulation": sp.ELEMENT_FORMULATION,
        "imk_material_type": sp.IMK_MATERIAL_TYPE,
        "imk_deterioration_mode": getattr(sp, "IMK_DETERIORATION_MODE", "direct"),
        "imk_use_calibrated_backbone": bool(sp.IMK_USE_CALIBRATED_BACKBONE),
        "imk_hinge_stiffness_mode": getattr(sp, "IMK_HINGE_STIFFNESS_MODE", "member_stiffness_factor"),
        "imk_hinge_stiffness_factor": sp.IMK_HINGE_STIFFNESS_FACTOR,
    }


def pilot_model_identity(observations):
    """What the pilot runs actually used, from their global_parameters.json and design.json.

    One value per key when every run agrees; the list of values (and the key
    under "mixed_keys") otherwise; None when the pilot predates the key.
    """
    values, mixed = {}, []
    for key in MODEL_IDENTITY_KEYS:
        seen = []
        for item in observations:
            value = (item.get("model") or {}).get(key)
            if value is not None and value not in seen:
                seen.append(value)
        if not seen:
            values[key] = None
        elif len(seen) == 1:
            values[key] = seen[0]
        else:
            values[key] = seen
            mixed.append(key)
    return {"values": values, "mixed_keys": mixed}


def calibration_model_mismatch(payload, current=None):
    """Keys on which the artifact's model differs from the running code; {} when it matches.

    The pilot's own recorded value wins; a key the pilot predates falls back to
    the code state at fit time. An artifact with no model identity (fitted
    before 2026-09-26) mismatches on every key.
    """
    current = current or current_model_identity()
    recorded = payload.get("model_identity") or {}
    pilot = (recorded.get("pilot") or {}).get("values") or {}
    mixed = set((recorded.get("pilot") or {}).get("mixed_keys") or [])
    at_fit = recorded.get("current_at_fit") or {}
    differences = {}
    for key in MODEL_IDENTITY_KEYS:
        if key in mixed:
            differences[key] = {"calibration": pilot.get(key), "current": current.get(key),
                                "note": "the pilot mixed several values"}
            continue
        expected = pilot.get(key) if pilot.get(key) is not None else at_fit.get(key)
        if expected != current.get(key):
            differences[key] = {"calibration": expected, "current": current.get(key)}
    return differences


def fit_drift_model(observations, spectra):
    """Least-squares fit of log(drift) on log(Sa) and log(T1)."""
    rows, targets = [], []
    for item in observations:
        sa = item["scale_factor"] * geometric_mean_sa(
            spectra, item["record_x"], item["record_y"], item["period_sec"]
        )
        if sa <= 0.0:
            continue
        rows.append([1.0, math.log(sa), math.log(item["period_sec"])])
        targets.append(math.log(item["peak_drift_ratio"]))

    if len(rows) < 8:
        return dict(DEFAULT_COEFFICIENTS), {
            "sample_count": len(rows),
            "fitted": False,
            "reason": "fewer than 8 usable pilot runs; using equal-displacement default",
        }

    matrix = np.asarray(rows, dtype=np.float64)
    vector = np.asarray(targets, dtype=np.float64)
    solution, residuals, rank, _singular = np.linalg.lstsq(matrix, vector, rcond=None)
    predicted = matrix @ solution
    total = float(((vector - vector.mean()) ** 2).sum())
    residual = float(((vector - predicted) ** 2).sum())
    coefficients = {"a": float(solution[0]), "b": float(solution[1]), "c": float(solution[2])}

    # A non-positive exponent on spectral acceleration is physically
    # meaningless and cannot be inverted, so refuse it rather than emit
    # scale factors that move the wrong way.
    if coefficients["b"] <= 0.05:
        return dict(DEFAULT_COEFFICIENTS), {
            "sample_count": len(rows),
            "fitted": False,
            "reason": f"fitted b={coefficients['b']:.3f} is not usable; using default",
            "rejected_coefficients": coefficients,
        }

    return coefficients, {
        "sample_count": len(rows),
        "fitted": True,
        "rank": int(rank),
        "r_squared": (1.0 - residual / total) if total > 0 else None,
        "log_residual_std": float(np.sqrt(residual / max(1, len(rows) - 3))),
    }


def fit_period_ratio(observations):
    """Empirical T1 / Ta, so the scheduler can estimate a period from height."""
    ratios = []
    for item in observations:
        if item["num_floor"] <= 0 or item["story_h_in"] <= 0.0:
            continue
        height_ft = item["num_floor"] * item["story_h_in"] / 12.0
        ta = 0.016 * height_ft**0.9
        if ta > 0:
            ratios.append(item["period_sec"] / ta)
    if not ratios:
        return DEFAULT_PERIOD_RATIO, {"sample_count": 0, "fitted": False}
    return float(np.median(ratios)), {
        "sample_count": len(ratios),
        "fitted": True,
        "min": float(np.min(ratios)),
        "max": float(np.max(ratios)),
    }


def estimated_period(num_floor, story_h_in, period_ratio):
    """Estimate T1 from height using the fitted ratio on ASCE approximate Ta."""
    height_ft = num_floor * story_h_in / 12.0
    return period_ratio * 0.016 * height_ft**0.9


def scale_factor_for(target_drift, period_sec, unscaled_sa, coefficients):
    """Scale factor predicted to drive a structure to a target drift."""
    if unscaled_sa <= 0.0 or target_drift <= 0.0 or period_sec <= 0.0:
        return 1.0
    log_sa_required = (
        math.log(target_drift) - coefficients["a"] - coefficients["c"] * math.log(period_sec)
    ) / coefficients["b"]
    factor = math.exp(log_sa_required) / unscaled_sa
    return float(min(SCALE_FACTOR_MAX, max(SCALE_FACTOR_MIN, factor)))


def target_drift_sequence(count, seed=0):
    """Target drifts whose overall mix matches TARGET_DRIFT_BANDS.

    Bands are filled by quota rather than sampled independently so the
    realized distribution matches the intent even for small datasets, then
    shuffled so a case's runs are not ordered by severity.
    """
    rng = np.random.default_rng(seed)
    targets = []
    for _name, low, high, share in TARGET_DRIFT_BANDS:
        quota = int(round(share * count))
        if quota <= 0:
            continue
        targets.extend(rng.uniform(low, high, size=quota).tolist())
    while len(targets) < count:
        targets.append(float(rng.uniform(0.010, 0.020)))
    targets = targets[:count]
    rng.shuffle(targets)
    return targets


def build_calibration(dataset_roots, output_path, seed=0):
    observations = collect_pilot_observations(dataset_roots)
    needed = {item["record_x"] for item in observations} | {
        item["record_y"] for item in observations
    }
    spectra = record_spectra()
    coefficients, fit_report = fit_drift_model(observations, spectra)
    period_ratio, period_report = fit_period_ratio(observations)

    floors = [item["num_floor"] for item in observations if item.get("num_floor")]
    heights = [item["story_h_in"] for item in observations if item.get("story_h_in")]
    payload = {
        "schema_version": CALIBRATION_SCHEMA,
        "coefficients": coefficients,
        "fit": fit_report,
        "period_ratio": period_ratio,
        "period_ratio_fit": period_report,
        "model_identity": {
            "keys": list(MODEL_IDENTITY_KEYS),
            "pilot": pilot_model_identity(observations),
            "current_at_fit": current_model_identity(),
            "dataset_roots": [str(Path(root).resolve()) for root in dataset_roots],
            "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "num_floor_range": [min(floors), max(floors)] if floors else None,
            "story_h_in_range": [min(heights), max(heights)] if heights else None,
        },
        "target_drift_bands": [
            {"name": name, "low": low, "high": high, "share": share}
            for name, low, high, share in TARGET_DRIFT_BANDS
        ],
        "scale_factor_bounds": [SCALE_FACTOR_MIN, SCALE_FACTOR_MAX],
        "spectrum_periods_sec": SPECTRUM_PERIODS_SEC.tolist(),
        "record_spectra_g": {key: value.tolist() for key, value in spectra.items()},
        "pilot_observation_count": len(observations),
        "pilot_records_used": sorted(needed),
        "seed": seed,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return payload


def load_calibration(path, require_model=None):
    """Read a calibration artifact.

    With ``require_model`` (True for the running code, or a
    ``current_model_identity()``-shaped dict) an artifact fitted on a different
    design route or hinge model is refused.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != CALIBRATION_SCHEMA:
        raise ValueError(
            f"Unexpected calibration schema {payload.get('schema_version')!r}; "
            f"expected {CALIBRATION_SCHEMA!r}."
        )
    if require_model:
        current = current_model_identity() if require_model is True else require_model
        differences = calibration_model_mismatch(payload, current)
        if differences:
            detail = "; ".join(
                f"{key}: calibration {value['calibration']!r}, current {value['current']!r}"
                for key, value in differences.items()
            )
            raise ValueError(
                f"Calibration {path} was fitted on a different design route or hinge model "
                f"than the running code: {detail}. Refit it on a pilot generated by the current code."
            )
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        action="append",
        required=True,
        help="Pilot dataset root containing cases/; repeat for several roots.",
    )
    parser.add_argument("--output", required=True, help="Calibration artifact to write.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_calibration(args.dataset_root, args.output, seed=args.seed)
    fit = payload["fit"]
    coefficients = payload["coefficients"]
    print(f"Pilot runs used: {payload['pilot_observation_count']}")
    print(
        "Drift model: log(drift) = "
        f"{coefficients['a']:.3f} + {coefficients['b']:.3f} log(Sa) "
        f"+ {coefficients['c']:.3f} log(T1)"
    )
    if fit.get("fitted"):
        print(f"  R^2 {fit['r_squared']:.3f}, log residual sd {fit['log_residual_std']:.3f}")
    else:
        print(f"  NOT FITTED: {fit.get('reason')}")
    print(f"Period ratio T1/Ta: {payload['period_ratio']:.3f}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
