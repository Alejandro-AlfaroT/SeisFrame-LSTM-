"""Deterministic, parallel, stop/resume-safe RC dataset scheduler."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


# This file lives in "RC Structure/Data_Generation/", so the project root is
# its parent directory. Deriving it from __file__ keeps the scheduler portable
# across machines; RC_STRUCTURE_DIR still overrides for unusual layouts.
RC_DIR = Path(
    os.environ.get("RC_STRUCTURE_DIR", Path(__file__).resolve().parents[1])
).resolve()
sys.path.insert(0, str(RC_DIR))

# This module is used both as a script (python Data_Generation/Generate_...py)
# and imported as Data_Generation.Generate_Parameterized_Dataset by the tests.
# Only the script form puts this directory on the path, so sibling imports such
# as Calibrate_Intensity resolved in production and failed under import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Run_Naming import analysis_run_name  # noqa: E402


SEED = 20260731

# Two- and three-story frames are excluded. They need roughly four times the
# spectral demand of an eight-story frame to reach the same drift, which made
# a single intensity rule unreliable across the set, and they are the shortest
# periods where the equal-displacement assumption behind that rule is weakest.
# The upper bound moves to nine stories to keep the geometry pool large enough
# and because taller frames are the more inelastic ones.
# Bay width varies independently in X and Y. Forcing one width on both axes
# made every frame square in plan, so span-driven demand could never differ
# between the two directions and the biaxial ground motion had nothing
# asymmetric to excite.
# V2 population (user decisions 2026-10-01): column-grid centerline bays of 18 to 30 ft in each plan
# direction, independently, and one story height per building of 12 to 18 ft, both in 0.5-ft steps
# (25 bay-width levels, 13 story-height levels). The earlier 10-15 ft bays and 12-16 ft stories were
# narrower than practice (NIST GCR 16-917-40 reports 20 to 30 ft as the typical economical beam span;
# the 18-ft lower bound is the user's research choice, not the guide's). Bay counts (2..6) and floor
# counts (4..9) are unchanged and stay integers. Lengths are generated from integer half-foot ticks,
# so every value is an exact binary fraction and its inch value (6 x ticks) is exact. The basis name
# travels with every plan built from these ranges so a plan of an earlier population is never mistaken
# for this one.
POPULATION_BASIS = "v2_bays_18_30ft_stories_12_18ft_half_foot_steps_counts_unchanged_sdc_c_d_20261005"
GEOMETRY_INCREMENT_FT = 0.5
RANGES = {
    "num_bay_x": tuple(range(2, 7)),
    "num_bay_y": tuple(range(2, 7)),
    "num_floor": tuple(range(4, 10)),
    "story_height_ft": tuple(tick / 2 for tick in range(24, 37)),  # 12–18 ft inclusive, 0.5 ft steps: 13 levels.
    "bay_x_width_ft": tuple(tick / 2 for tick in range(36, 61)),   # 18–30 ft inclusive, 0.5 ft steps: 25 levels.
    "bay_y_width_ft": tuple(tick / 2 for tick in range(36, 61)),
}


def feet_label(value_ft):
    """A length in feet for a case name: 14 -> '14', 12.5 -> '12.5' (never '12.0', never rounded)."""
    return f"{float(value_ft):g}"


def feet_to_inches(value_ft):
    """Exact inches of a half-foot-tick length; anything off the 0.5-ft grid is refused, not rounded."""
    ticks = round(float(value_ft) / GEOMETRY_INCREMENT_FT)
    if abs(ticks * GEOMETRY_INCREMENT_FT - float(value_ft)) > 1e-12:
        raise ValueError(f"{value_ft!r} ft is not on the {GEOMETRY_INCREMENT_FT}-ft geometry grid.")
    return 6.0 * ticks


# ---- candidate planning: one named, seeded, replaceable method --------------------------------------
# The plan order of the candidate geometries comes from a named planner. Today there is one: the
# discrete Cartesian pool of RANGES shuffled with the seed, of which a plan takes a contiguous slice.
# A Latin hypercube planner is a candidate under review (record sampling_candidate_lhs_20261001),
# not implemented; it would be registered here beside this one, and every plan records which planner,
# seed, bounds and level mapping produced it, so plans of different methods are never confused.
SAMPLING_METHOD = "seeded_shuffled_cartesian_prefix_v1"


def _shuffled_cartesian_pool(seed):
    geometries = list(product(*RANGES.values()))
    random.Random(seed).shuffle(geometries)
    return geometries


CANDIDATE_PLANNERS = {SAMPLING_METHOD: _shuffled_cartesian_pool}


def candidate_geometries(seed, method=SAMPLING_METHOD):
    """Every candidate geometry in plan order: tuples in the key order of RANGES."""
    if method not in CANDIDATE_PLANNERS:
        raise ValueError(f"Unknown sampling method {method!r}; registered: {sorted(CANDIDATE_PLANNERS)}.")
    return CANDIDATE_PLANNERS[method](seed)


def sampling_provenance(seed, method=SAMPLING_METHOD, include_source=True):
    """What produced a plan's candidates: method, seed, bounds, the discrete level mapping and this source.

    ``include_source=False`` leaves the source hash out: that form is what a resumed plan is compared
    against, so an unrelated edit of this file does not strand a campaign, while a change of method,
    seed, bounds or levels still refuses the resume. The written plan always carries the hash."""
    if method not in CANDIDATE_PLANNERS:
        raise ValueError(f"Unknown sampling method {method!r}; registered: {sorted(CANDIDATE_PLANNERS)}.")
    pool = 1
    for values in RANGES.values():
        pool *= len(values)
    return {
        "method": method, "seed": seed, "population_basis": POPULATION_BASIS,
        "parameter_order": list(RANGES),
        "bounds": {key: [min(values), max(values)] for key, values in RANGES.items()},
        "levels": {key: list(values) for key, values in RANGES.items()},
        "level_counts": {key: len(values) for key, values in RANGES.items()},
        "discrete_mapping": ("each parameter takes one of its listed levels; lengths are integer half-foot ticks "
                             f"({GEOMETRY_INCREMENT_FT} ft), counts are integers; no continuous draw is rounded"),
        "pool_size": pool,
        "selection": "contiguous slice (geometry offset, count) of the seeded shuffle of the full Cartesian pool",
        "duplicates_possible": False,
        "hazard_assignment": "round-robin over the seeded shuffle of the site labels (seed + 2); not a sampled coordinate",
        **({"source_sha256": sha256_file(Path(__file__).resolve())} if include_source else {}),
        "candidate_methods_not_implemented": ["latin_hypercube (under review; not selected)"],
    }

# Design hazard, assigned per case. Geometry alone barely moves design demand,
# so without this axis the design loop returns near-identical members for
# different buildings. Labels must match Structure_Parameters.SEISMIC_SITE_OPTIONS.
# V2 research population restricted to SDC C-D by user decision (2026-10-05).
# The broader site library remains available for historical designs and code checks.
SEISMIC_SITES = ("sdc_c", "sdc_d_low", "sdc_d_high")

# The ladder of uncalibrated scale factors available to --runs-per-record.
# Under --intensity-calibration these are placeholders: the scale for each run
# comes from the calibration and only the count is read.
DEFAULT_INTENSITY_LEVELS = (1.0, 2.0, 3.0)

# One analysis per record pair by default.
#
# Tradeoff to be aware of: several runs per record would vary intensity
# *within* one structure, which is the cleanest way to separate structural
# response from intensity. At one run per record that variation has to come
# from across the dataset instead -- which the calibration provides, since it
# assigns each case its own target drift. So intensity stays identifiable
# between structures, but not within one.
DEFAULT_RUNS_PER_RECORD = 1
DEFAULT_INTENSITY_LADDER = DEFAULT_INTENSITY_LEVELS[:DEFAULT_RUNS_PER_RECORD]
# One record pair per structure.
#
# Splits are grouped by record pair (rc_hybrid_surrogate/data.py), so a
# structure carrying two record pairs can have one land in train and the
# other in test -- the same geometry, design and hazard on both sides of the
# split. With one pair per structure, grouping by record incidentally groups
# by structure too and that cannot happen.
#
# The intensity ladder already gives several runs per structure, so structure
# is not confounded with intensity. Record effects stay identifiable because
# round-robin assignment puts each record pair with many different
# structures across the dataset.
#
# Raising this above 1 requires the split to be blocked on structure as well
# as record, which it currently is not.
DEFAULT_RECORDS_PER_CASE = 1

# Bumped when plan semantics change. Version 3 adds the hazard axis, the
# intensity ladder, and multiple records per case, so a version 1 or 2 plan
# describes a different experiment and must not be silently reused.
PLAN_VERSION = 3
STOP_NAME = "STOP_GENERATION.json"
RECORD_MANIFEST = RC_DIR / "Ground_Motions" / "metadata" / "record_manifest.csv"
RECORD_SETS = RC_DIR / "Ground_Motions" / "metadata" / "record_sets.csv"
PEER_RESULT_RE = re.compile(r"PEER result_id=([0-9]+)")
ATOMIC_REPLACE_ATTEMPTS = 10
ATOMIC_REPLACE_INITIAL_DELAY_SEC = 0.05
NPTS_POLICY_OVERRIDE = "allow_npts_policy_excluded=true"
NPTS_POLICY_EXCLUSION = "omitted_from_generation=max_npts_gt_15000"


def now():
    return datetime.now(timezone.utc).isoformat()


def replace_with_retry(source, destination):
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


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")
    replace_with_retry(temp, path)


def write_csv(path, rows):
    path = Path(path)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    replace_with_retry(temp, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def eligible_record_ids(set_name, max_npts):
    """Read eligible PEER pairs without importing OpenSees into the scheduler."""
    with RECORD_MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        manifest = {row["record_id"]: row for row in csv.DictReader(file)}
    with RECORD_SETS.open(newline="", encoding="utf-8-sig") as file:
        selected = {
            row["record_id"]: row
            for row in csv.DictReader(file)
            if row.get("set_name") == set_name
        }
    pairs = {}
    for record_id, set_row in selected.items():
        row = manifest.get(record_id)
        if not row:
            continue
        usable = (row.get("usable") or "true").strip().lower() in {
            "1", "true", "yes", "y",
        }
        policy_override = (
            NPTS_POLICY_OVERRIDE in (set_row.get("notes") or "").lower()
            and NPTS_POLICY_EXCLUSION in (row.get("notes") or "").lower()
        )
        if not usable and not policy_override:
            continue
        match = PEER_RESULT_RE.search(row.get("notes") or "")
        if match:
            pairs.setdefault(int(match.group(1)), []).append(row)
    eligible = []
    for result_id, rows in pairs.items():
        if len(rows) < 2:
            continue
        pair_npts = max(int(float(row.get("npts") or 0)) for row in rows)
        if max_npts is None or max_npts <= 0 or pair_npts <= max_npts:
            eligible.append(result_id)
    return sorted(eligible)


def build_runs(case_records, intensity_levels, scale_factors=None, targets=None):
    """Expand a case's record and intensity assignments into concrete runs.

    Run names are produced by the same helper the analysis uses, so the
    scheduler can tell a finished run from a pending one without importing
    OpenSees or guessing at the naming convention. Under a calibration the
    target drift each factor was chosen for is kept on the run, so achieved
    versus target (and cap clipping) can be audited from the plan.
    """
    runs = []
    per_record = len(intensity_levels)
    for record_index, result_id in enumerate(case_records):
        for slot in range(per_record):
            position = record_index * per_record + slot
            scale = (
                scale_factors[position]
                if scale_factors is not None
                else intensity_levels[slot]
            )
            run = {
                "run_index": len(runs) + 1,
                "result_id": int(result_id),
                "scale_factor": float(scale),
                "run_name": analysis_run_name(
                    f"peer_{int(result_id)}", scale_factor=float(scale)
                ),
            }
            if targets is not None and position < len(targets):
                run["target_drift_ratio"] = float(targets[position])
            runs.append(run)
    return runs


def eligible_record_pairs(set_name, max_npts):
    """Map each eligible PEER result_id to its (X, Y) record ids.

    eligible_record_ids answers which pairs may be used; the calibration also
    needs to know which physical records those are, so it can read their
    spectra. Component order follows the manifest's horizontal_component note
    so X and Y stay consistent with what the analysis actually applies.
    """
    with RECORD_MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        manifest = list(csv.DictReader(file))
    allowed = set(eligible_record_ids(set_name, max_npts))

    grouped = {}
    for row in manifest:
        match = PEER_RESULT_RE.search(row.get("notes") or "")
        if not match:
            continue
        result_id = int(match.group(1))
        if result_id not in allowed:
            continue
        component = 2 if "horizontal_component=2" in (row.get("notes") or "") else 1
        grouped.setdefault(result_id, {})[component] = row["record_id"]

    pairs = {}
    for result_id, components in grouped.items():
        first = components.get(1)
        second = components.get(2, first)
        if first:
            pairs[result_id] = (first, second)
    return pairs


# How many records each case considers when matching a record to its target
# drift. Small enough that every record stays in circulation across the plan,
# large enough that a case aiming high can usually find a strong motion.
RECORD_MATCH_CANDIDATES = 8


def _unclamped_scale_factor(target, period, unscaled_sa, coefficients):
    """Scale factor before the [MIN, MAX] clamp, for ranking candidates.

    scale_factor_for clamps its result, which makes every over-ceiling record
    tie at the maximum. Ranking needs to tell "needs 8.1x" from "needs 40x".
    """
    if unscaled_sa <= 0.0 or target <= 0.0 or period <= 0.0:
        return None
    log_required = (
        math.log(target) - coefficients["a"] - coefficients["c"] * math.log(period)
    ) / coefficients["b"]
    return math.exp(log_required) / unscaled_sa


def _matched_case_records(
    calibration, case_geometry, record_pairs, records, targets,
    fallback_records, case_index, seed,
):
    """Choose records whose spectral demand suits this case's target drifts.

    Records are otherwise assigned by round robin before the target drift is
    known, so a case aiming at 4% can draw a weak motion needing 6x scaling
    while a strong record sits unused on an elastic target. Measured over a
    50-case plan, matching drops the median scale factor from 2.25 to 0.62
    and removes ceiling clipping entirely (6 of 50 runs to none).

    Selection is banded, not greedy: each case draws a deterministic candidate
    subset and takes the best member of that. Always taking the globally
    strongest record would minimise scaling and collapse record diversity,
    which the leakage-safe splits depend on -- the grouping key is the record
    pair, and too few distinct groups is what caused the validation plateau.

    Falls back to the round-robin choice for any slot it cannot price.
    """
    from Calibrate_Intensity import estimated_period, geometric_mean_sa

    coefficients = calibration["coefficients"]
    spectra = calibration["record_spectra_g"]
    period = estimated_period(
        case_geometry["num_floor"],
        case_geometry["story_height_in"],
        calibration["period_ratio"],
    )
    slots = max(1, len(fallback_records))
    per_slot = max(1, len(targets) // slots)
    chooser = random.Random(seed + 977 * case_index)

    chosen, used = [], set()
    for slot in range(slots):
        slot_targets = targets[slot * per_slot:(slot + 1) * per_slot] or targets[:1]
        available = [r for r in records if r not in used] or list(records)
        candidates = chooser.sample(
            available, min(RECORD_MATCH_CANDIDATES, len(available))
        )
        # The round-robin pick is always in the running, so matching can only
        # improve on it, never do worse.
        default = fallback_records[slot] if slot < len(fallback_records) else None
        if default is not None and default in available and default not in candidates:
            candidates.append(default)

        best, best_cost = None, None
        for result_id in candidates:
            pair = record_pairs.get(int(result_id))
            if not pair:
                continue
            unscaled = geometric_mean_sa(spectra, pair[0], pair[1], period)
            cost = 0.0
            for target in slot_targets:
                factor = _unclamped_scale_factor(target, period, unscaled, coefficients)
                if factor is None or factor <= 0.0:
                    cost = None
                    break
                # Symmetric in log space: a record needing 4x is penalised the
                # same as one needing 1/4x. Both are distortions of the motion.
                cost += abs(math.log(factor))
            if cost is None:
                continue
            if best_cost is None or cost < best_cost:
                best, best_cost = result_id, cost

        chosen.append(best if best is not None else default)
        if chosen[-1] is not None:
            used.add(chosen[-1])
    return [r for r in chosen if r is not None] or list(fallback_records)


def _calibrated_scale_factors(calibration, case_geometry, record_pairs, case_records, targets):
    """Scale factors predicted to land each run on its target drift."""
    from Calibrate_Intensity import (
        estimated_period,
        geometric_mean_sa,
        scale_factor_for,
    )

    coefficients = calibration["coefficients"]
    spectra = calibration["record_spectra_g"]
    period = estimated_period(
        case_geometry["num_floor"],
        case_geometry["story_height_in"],
        calibration["period_ratio"],
    )

    factors = []
    index = 0
    for result_id in case_records:
        pair = record_pairs.get(int(result_id))
        for _ in range(len(targets) // max(1, len(case_records))):
            target = targets[index % len(targets)]
            index += 1
            if not pair:
                factors.append(1.0)
                continue
            unscaled = geometric_mean_sa(spectra, pair[0], pair[1], period)
            factors.append(
                round(scale_factor_for(target, period, unscaled, coefficients), 3)
            )
    return factors, period


def build_plan(
    num_cases,
    record_ids,
    seed=SEED,
    geometry_offset=0,
    case_id_offset=0,
    seismic_sites=SEISMIC_SITES,
    intensity_levels=DEFAULT_INTENSITY_LADDER,
    records_per_case=DEFAULT_RECORDS_PER_CASE,
    calibration=None,
    record_pairs=None,
    sampling_method=SAMPLING_METHOD,
):
    if any(site not in SEISMIC_SITES for site in seismic_sites):
        raise ValueError(f"V2 generation requires SDC C-D presets: {list(SEISMIC_SITES)}.")
    # The candidate order comes from the named planner (already seeded); the plan takes a slice of it.
    geometries = candidate_geometries(seed, sampling_method)
    if geometry_offset < 0 or case_id_offset < 0:
        raise ValueError("geometry-offset and case-id-offset must be nonnegative.")
    if not 1 <= num_cases or geometry_offset + num_cases > len(geometries):
        raise ValueError(
            "Requested geometry slice must fit within the "
            f"{len(geometries)} available combinations; received offset="
            f"{geometry_offset}, num_cases={num_cases}."
        )
    if not record_ids:
        raise ValueError("No eligible seismic record pairs were found.")
    if not seismic_sites:
        raise ValueError("At least one seismic site is required.")
    if not intensity_levels:
        raise ValueError("At least one intensity level is required.")
    records = sorted(set(map(int, record_ids)))
    if records_per_case < 1:
        raise ValueError(f"records-per-case must be positive; received {records_per_case}.")
    if records_per_case > len(records):
        # A catalog smaller than the request is a degenerate catalog, not a
        # user error. Clamp rather than halt, but say so: silently assigning
        # fewer records than asked for would be invisible in the plan.
        print(
            f"WARNING: records-per-case {records_per_case} exceeds the "
            f"{len(records)} eligible record pair(s); clamping to {len(records)}."
        )
        records_per_case = len(records)
    random.Random(seed + 1).shuffle(records)
    sites = list(seismic_sites)
    random.Random(seed + 2).shuffle(sites)

    # With a calibration, intensity is chosen per run to hit a target drift
    # rather than taken from a fixed ladder. Targets are drawn once for the
    # whole plan so the realized drift distribution matches the intent across
    # the dataset, not merely within each case.
    runs_per_case = records_per_case * len(intensity_levels)
    plan_targets = None
    if calibration is not None:
        from Calibrate_Intensity import target_drift_sequence

        plan_targets = target_drift_sequence(num_cases * runs_per_case, seed=seed)
        record_pairs = record_pairs or {}

    cases = []
    selected_geometries = geometries[geometry_offset:geometry_offset + num_cases]
    for local_index, values in enumerate(selected_geometries, 1):
        bx, by, floors, story_ft, width_x_ft, width_y_ft = values
        case_index = case_id_offset + local_index
        case_id = f"case_{case_index:04d}"

        # Round-robin over both axes so hazard and records stay balanced
        # across any contiguous slice of the plan, including the per-device
        # ranges the scheduler hands out.
        site = sites[(case_index - 1) % len(sites)]
        start = ((case_index - 1) * records_per_case) % len(records)
        case_records = [
            records[(start + offset) % len(records)]
            for offset in range(records_per_case)
        ]

        scale_factors = None
        estimated_period_sec = None
        case_targets = None
        if plan_targets is not None:
            geometry = {"num_floor": floors, "story_height_in": feet_to_inches(story_ft)}
            start = (local_index - 1) * runs_per_case
            case_targets = plan_targets[start:start + runs_per_case]
            case_records = _matched_case_records(
                calibration, geometry, record_pairs, records, case_targets,
                case_records, case_index, seed,
            )
            scale_factors, estimated_period_sec = _calibrated_scale_factors(
                calibration, geometry, record_pairs, case_records, case_targets,
            )

        cases.append(
            {
                "case_index": case_index,
                "case_id": case_id,
                "geometry_name": (
                    f"{case_id}_bx{bx}_by{by}_s{floors}_"
                    f"sh{feet_label(story_ft)}ft_bwx{feet_label(width_x_ft)}ft_bwy{feet_label(width_y_ft)}ft_{site}"
                ),
                "seismic_site": site,
                "result_ids": case_records,
                "num_bay_x": bx,
                "num_bay_y": by,
                "num_floor": floors,
                "story_height_ft": story_ft,
                "story_height_in": feet_to_inches(story_ft),
                "bay_x_width_ft": width_x_ft,
                "bay_y_width_ft": width_y_ft,
                "bay_x_in": feet_to_inches(width_x_ft),
                "bay_y_in": feet_to_inches(width_y_ft),
                "runs": build_runs(case_records, intensity_levels, scale_factors, case_targets),
                "estimated_period_sec": estimated_period_sec,
            }
        )
    return cases


def plan_csv_rows(cases):
    """Flatten cases for the CSV mirror of the plan.

    The runs list cannot be a CSV cell, so it is summarized here. The JSON
    plan remains the authoritative record of every individual run.
    """
    rows = []
    for case in cases:
        row = {key: value for key, value in case.items() if key not in {"runs", "result_ids"}}
        runs = case_runs(case)
        row.update(
            {
                "result_ids": " ".join(str(value) for value in case.get("result_ids", [])),
                "intensity_levels": " ".join(
                    f"{value:g}" for value in sorted({run["scale_factor"] for run in runs})
                ),
                "run_names": " ".join(run["run_name"] for run in runs),
                "num_runs": len(runs),
            }
        )
        rows.append(row)
    return rows


def load_plan(
    root,
    num_cases,
    records,
    seed,
    max_npts,
    set_name="peer_mle_all",
    geometry_offset=0,
    case_id_offset=0,
    seismic_sites=SEISMIC_SITES,
    intensity_levels=DEFAULT_INTENSITY_LADDER,
    records_per_case=DEFAULT_RECORDS_PER_CASE,
    calibration=None,
    record_pairs=None,
    profile=None,
):
    path = root / "parameter_plan.json"
    # Every field is compared on every version. Earlier revisions left
    # set_name out of the version 1 comparison, so changing the record set
    # silently reused a plan built for a different catalog.
    expected = {
        "version": PLAN_VERSION,
        "seed": seed,
        "num_cases": num_cases,
        "max_npts": max_npts,
        "ranges": {key: list(value) for key, value in RANGES.items()},
        "population_basis": POPULATION_BASIS,
        "sampling": sampling_provenance(seed, include_source=False),
        "set_name": set_name,
        "geometry_offset": geometry_offset,
        "case_id_offset": case_id_offset,
        "seismic_sites": list(seismic_sites),
        "intensity_levels": [float(value) for value in intensity_levels],
        "records_per_case": records_per_case,
        "eligible_result_ids": list(records),
        # A plan built against a calibration describes a different experiment
        # from one built on the uniform ladder, so the fingerprint is part of
        # the conflict check.
        "intensity_source": "calibration" if calibration else "uniform_ladder",
        # Only meaningful under a calibration, and deliberately absent
        # otherwise: adding a key to every plan would make existing
        # uniform-ladder plans mismatch on reload and refuse to resume.
        **(
            {"record_selection": "target_matched"} if calibration else {}
        ),
        "calibration_fingerprint": (
            calibration["coefficients"] if calibration else None
        ),
        # The analysis profile every child run is built under (screening repair item 6, 2026-10-02): part of
        # the plan, so a root never mixes profiles. None for an unprofiled plan, which an older plan without
        # the key still matches.
        "analysis_profile": profile,
    }
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Existing plan conflicts with this command: {mismatches}. "
                "Choose another --output-root for a new plan."
            )
        return payload["cases"]
    cases = build_plan(
        num_cases,
        records,
        seed,
        geometry_offset=geometry_offset,
        case_id_offset=case_id_offset,
        seismic_sites=seismic_sites,
        intensity_levels=intensity_levels,
        records_per_case=records_per_case,
        calibration=calibration,
        record_pairs=record_pairs,
    )
    payload = dict(expected)
    payload.update(
        {
            "created_at": now(),
            "bay_width_policy": "independent widths in X and Y",
            "record_policy": "seeded shuffle then balanced round-robin",
            "sampling_source_sha256": sampling_provenance(seed)["source_sha256"],
            "eligible_result_ids": records,
            "cases": cases,
        }
    )
    write_json(path, payload)
    write_csv(root / "parameter_plan.csv", plan_csv_rows(cases))
    return cases


def paths_for(root, case):
    """Case-level paths. The design artifact is shared by every run below it."""
    base = root / "cases" / case["case_id"]
    return {
        "base": base,
        "ntha": base / "ntha",
        "dataset": base / "dataset",
        "design": base / "design.json",
        "child_stop": base / "dataset" / STOP_NAME,
        "log": base / "controller.log",
    }


def case_runs(case):
    """Runs belonging to a case: one per (record pair, intensity) combination."""
    return case.get("runs") or []


def run_paths_for(root, case, run):
    base = root / "cases" / case["case_id"]
    name = run["run_name"]
    return {
        "status": base / "ntha" / name / "status.json",
        "sample": base / "dataset" / name / "hybrid_sample.npz",
    }


def successful_status(path):
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        requested = int(value.get("npts_requested") or 0)
        completed = int(value.get("completed_steps") or 0)
        return requested > 0 and requested == completed and not value.get("failed")
    except (OSError, ValueError, TypeError):
        return False


def collapse_labelled(sample_path):
    """True when a compiled sample is labelled as a collapse.

    Reads the exporter's verdict rather than re-deriving one, so the
    scheduler and Hybrid_Exporter cannot disagree about what a collapse is.
    """
    metadata_path = Path(sample_path).with_name("hybrid_metadata.json")
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return bool(metadata.get("collapse"))


def complete_run(root, case, run):
    """Whether one run is finished and need not be attempted again.

    A run that stops early because the structure collapsed is finished, not
    failed. Hybrid_Exporter already keeps and labels such a run, and
    Calibrate_Intensity keeps it as an ordinary observation up to its 20%
    drift ceiling (it does not censor), but the
    scheduler only ever looked at the step count -- so a collapse stayed in
    the remaining list and was retried on every invocation. The analysis is
    deterministic, so each retry reproduced the same collapse and the batch
    could never report itself complete. At the dataset's 10 percent
    near-collapse target that is several hundred runs looping indefinitely.

    A compiled sample is required either way: a run that produced no sample
    has nothing to contribute regardless of why it stopped.
    """
    paths = run_paths_for(root, case, run)
    if not paths["sample"].exists():
        return False
    from Hybrid_Exporter import _sample_is_stale
    try:
        if _sample_is_stale(paths["status"].parent, paths["sample"]):
            return False
        if successful_status(paths["status"]):
            return True
        status = json.loads(paths["status"].read_text(encoding="utf-8"))
        metadata = json.loads(paths["sample"].with_name("hybrid_metadata.json").read_text(encoding="utf-8"))
        # Legacy sidecars need matching step counts as well as freshness.
        return (metadata.get("collapse") is True and int(status.get("completed_steps") or 0) > 0
                and all(metadata.get(k) == status.get(k) for k in ("completed_steps", "npts_requested")))
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def completed_run_count(root, case):
    return sum(1 for run in case_runs(case) if complete_run(root, case, run))


def complete(root, case):
    """A case is finished only when every one of its runs is finished."""
    runs = case_runs(case)
    return bool(runs) and all(complete_run(root, case, run) for run in runs)


def design_refusal(root, case):
    """Why this case's saved design was refused, or None.

    A case whose design.json exists but did not qualify is terminal: the
    child refuses to analyse it (Ground_Motion_Main.require_accepted_design)
    and will keep refusing it on every resume, at the full cost of a design.
    The refusal is reported with the failed and unevaluated check ids so the
    plan can be amended deliberately (``--retry-refused`` re-designs them).
    """
    path = paths_for(root, case)["design"]
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        qualification = record.get("qualification") or {}
        if qualification.get("accepted") is True:
            return None
        checks = qualification.get("checks") or []
        return {
            "schema_version": record.get("schema_version"),
            "counts": qualification.get("counts"),
            "failed": sorted({c["id"].split(":")[0] for c in checks if c.get("status") == "fail"}),
            "not_evaluated": sorted({c["id"].split(":")[0] for c in checks if c.get("status") == "not_evaluated"}),
        }
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return {"unreadable": repr(exc)}


def scan_completed(root, cases):
    """Perform the expensive filesystem scan once when a scheduler starts."""
    return {case["case_id"] for case in cases if complete(root, case)}


def scan_refused(root, cases):
    """Cases whose saved design was refused by qualification, with the reasons."""
    refused = {}
    for case in cases:
        reason = design_refusal(root, case)
        if reason is not None:
            refused[case["case_id"]] = reason
    return refused


def select_pending_cases(cases, completed_ids, case_start=1, case_end=None, run_limit=None, refused_ids=()):
    """Select unfinished cases from an inclusive, one-based device range.

    Cases in ``refused_ids`` (design saved but not qualified) are terminal
    and skipped; the caller passes an empty set to retry them.
    """
    case_end = len(cases) if case_end is None else case_end
    if case_start < 1 or case_end < case_start or case_end > len(cases):
        raise ValueError(
            f"case range must satisfy 1 <= start <= end <= {len(cases)}; "
            f"received {case_start}..{case_end}."
        )
    scope = cases[case_start - 1:case_end]
    refused_ids = set(refused_ids)
    selected = [case for case in scope
                if case["case_id"] not in completed_ids and case["case_id"] not in refused_ids]
    if run_limit is not None:
        selected = selected[:run_limit]
    return scope, selected


def command_for(args, case, paths):
    command = [
        args.python_exe,
        "-B",
        str(RC_DIR / "Data_Generation" / "Generate_Hybrid_Dataset.py"),
        "--run-ntha",
        "--set-name", args.set_name,
        "--max-npts", str(args.max_npts),
        "--python-exe", args.python_exe,
        "--ntha-root", str(paths["ntha"]),
        "--dataset-dir", str(paths["dataset"]),
        "--geometry-name", case["geometry_name"],
        "--num-bay-x", str(case["num_bay_x"]),
        "--num-bay-y", str(case["num_bay_y"]),
        "--num-floor", str(case["num_floor"]),
        "--bay-x", str(case["bay_x_in"]),
        "--bay-y", str(case["bay_y_in"]),
        "--story-h", str(case["story_height_in"]),
        "--seismic-site", str(case["seismic_site"]),
    ]
    # Each plan run is handed over as one explicit (record pair, scale), so the
    # child executes exactly the plan. A calibrated plan pairs every record
    # with its own target-matched factor; the former records x scales cross
    # product replaced that with pairings nobody asked for (fixed 2026-09-26).
    for result_id in case["result_ids"]:
        command.extend(["--result-id", str(result_id)])
    for run in case_runs(case):
        command.extend(["--run-pair", f"{int(run['result_id'])}:{float(run['scale_factor'])}"])
    if getattr(args, "profile", None):
        command.extend(["--profile", args.profile])
    return command


def request_child_stop(path):
    if not path.exists():
        write_json(path, {"requested_at": now(), "reason": "parent_stop"})


def run_case(args, case, stop_event, active, lock):
    root = Path(args.output_root)
    paths = paths_for(root, case)
    paths["ntha"].mkdir(parents=True, exist_ok=True)
    paths["dataset"].mkdir(parents=True, exist_ok=True)
    if complete(root, case):
        return {
            "case_id": case["case_id"], "status": "completed", "skipped": True,
            "completed_runs": len(case_runs(case)),
            "planned_runs": len(case_runs(case)),
        }
    if stop_event.is_set():
        return {"case_id": case["case_id"], "status": "pending", "skipped": True}
    command = command_for(args, case, paths)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    started = time.perf_counter()
    with paths["log"].open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=RC_DIR, env=env, stdout=log,
            stderr=subprocess.STDOUT, text=True, creationflags=flags,
        )
        with lock:
            active[case["case_id"]] = {"pid": process.pid}
        stop_sent = False
        while process.poll() is None:
            if stop_event.is_set() and not stop_sent:
                request_child_stop(paths["child_stop"])
                stop_sent = True
            time.sleep(0.5)
        returncode = process.wait()
        with lock:
            active.pop(case["case_id"], None)
    finished_runs = completed_run_count(root, case)
    refusal = None if finished_runs == len(case_runs(case)) else design_refusal(root, case)
    status = "completed" if finished_runs == len(case_runs(case)) else (
        "pending" if stop_event.is_set() else "design_refused" if refusal else "failed"
    )
    result = {
        "case_id": case["case_id"], "status": status,
        "skipped": False, "returncode": returncode,
        "completed_runs": finished_runs,
        "planned_runs": len(case_runs(case)),
        "elapsed_sec": time.perf_counter() - started, "updated_at": now(),
    }
    if refusal:
        result["design_refusal"] = refusal
    return result


def load_results(root):
    path = root / "case_results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("cases", {})


def progress(
    root, cases, results, active, status, started, completed_ids,
    write_manifest=False, invocation_case_ids=None, case_start=1, case_end=None,
):
    completed = [case["case_id"] for case in cases if case["case_id"] in completed_ids]
    failed = [
        case["case_id"]
        for case in cases
        if case["case_id"] not in completed_ids
        and results.get(case["case_id"], {}).get("status") == "failed"
    ]
    refused = [
        case["case_id"]
        for case in cases
        if case["case_id"] not in completed_ids
        and results.get(case["case_id"], {}).get("status") == "design_refused"
    ]
    remaining = [case["case_id"] for case in cases
                 if case["case_id"] not in completed_ids and case["case_id"] not in refused]
    state = {
        "status": status, "started_at": started, "updated_at": now(),
        "planned_count": len(cases), "completed_count": len(completed),
        "failed_count": len(failed), "design_refused_count": len(refused),
        "remaining_count": len(remaining),
        "active_cases": active, "completed_case_ids": completed,
        "failed_case_ids": failed, "design_refused_case_ids": refused,
        "design_refusals": {case_id: results[case_id].get("design_refusal") for case_id in refused},
        "remaining_case_ids": remaining,
    }
    if invocation_case_ids is not None:
        invocation_case_ids = list(invocation_case_ids)
        invocation_completed = [
            case_id for case_id in invocation_case_ids if case_id in completed_ids
        ]
        invocation_failed = [
            case_id
            for case_id in invocation_case_ids
            if case_id not in completed_ids
            and results.get(case_id, {}).get("status") == "failed"
        ]
        invocation_remaining = [
            case_id for case_id in invocation_case_ids if case_id not in completed_ids
        ]
        state.update(
            {
                "case_range_start": case_start,
                "case_range_end": len(cases) if case_end is None else case_end,
                "invocation_selected_count": len(invocation_case_ids),
                "invocation_completed_count": len(invocation_completed),
                "invocation_failed_count": len(invocation_failed),
                "invocation_remaining_count": len(invocation_remaining),
                "invocation_case_ids": invocation_case_ids,
                "invocation_completed_case_ids": invocation_completed,
                "invocation_failed_case_ids": invocation_failed,
                "invocation_remaining_case_ids": invocation_remaining,
            }
        )
    write_json(root / "generation_state.json", state)
    if write_manifest:
        rows = []
        for case in cases:
            paths = paths_for(root, case)
            result = results.get(case["case_id"], {})
            row = dict(case)
            row.update(
                {
                    "status": (
                        "completed"
                        if case["case_id"] in completed_ids
                        else result.get("status", "pending")
                    ),
                    "planned_runs": len(case_runs(case)),
                    "completed_runs": result.get(
                        "completed_runs",
                        len(case_runs(case)) if case["case_id"] in completed_ids else None,
                    ),
                    "case_dir": os.path.relpath(paths["base"], root),
                    "design_json": os.path.relpath(paths["design"], root),
                    "controller_log": os.path.relpath(paths["log"], root),
                    "returncode": result.get("returncode"),
                    "elapsed_sec": result.get("elapsed_sec"),
                }
            )
            rows.append(row)
        write_json(root / "case_results.json", {"updated_at": now(), "cases": results})
        write_json(root / "parameterized_manifest.json", rows)
        write_csv(root / "parameterized_manifest.csv", rows)
    return state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-cases", type=int, default=2500)
    parser.add_argument("--run-limit", type=int)
    parser.add_argument("--retry-refused", action="store_true",
                        help="Re-design cases whose saved design was refused by qualification "
                             "(normally terminal; use after changing the plan or the design inputs).")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--case-start",
        type=int,
        default=1,
        help="First one-based case number assigned to this device (inclusive).",
    )
    parser.add_argument(
        "--case-end",
        type=int,
        help="Last one-based case number assigned to this device (inclusive).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--set-name", default="peer_mle_all")
    parser.add_argument(
        "--seismic-site",
        action="append",
        dest="seismic_sites",
        help=(
            "Design hazard label to include in the plan; repeat to set the "
            "pool. Defaults to all of: " + ", ".join(SEISMIC_SITES) + "."
        ),
    )
    parser.add_argument(
        "--runs-per-record",
        type=int,
        default=None,
        help=(
            "How many analyses to run per record pair. Under "
            "--intensity-calibration the scale factor for each run comes from "
            "the calibration, so this count is the only thing that matters; "
            "without it, the first N of the default ladder ("
            + " ".join(f"{value:g}" for value in DEFAULT_INTENSITY_LEVELS)
            + ") are used. Total runs per case = records-per-case x this."
        ),
    )
    parser.add_argument(
        "--intensity-level",
        action="append",
        type=float,
        dest="intensity_levels",
        help=(
            "Explicit ground-motion scale factor; repeat for a ladder. Use "
            "this only to pin exact uncalibrated scale factors -- prefer "
            "--runs-per-record otherwise. Mutually exclusive with it."
        ),
    )
    parser.add_argument(
        "--intensity-calibration",
        default=None,
        help=(
            "Calibration artifact from Calibrate_Intensity.py. When given, "
            "each run scale factor is chosen to hit a target drift instead "
            "of using the uniform ladder."
        ),
    )
    parser.add_argument(
        "--allow-stale-calibration",
        action="store_true",
        help=(
            "Accept a calibration whose recorded design route or hinge model "
            "differs from the current code; it is refused otherwise."
        ),
    )
    parser.add_argument(
        "--records-per-case",
        type=int,
        default=DEFAULT_RECORDS_PER_CASE,
        help="Distinct record pairs assigned to each structure.",
    )
    parser.add_argument("--max-npts", type=int, default=15000)
    parser.add_argument(
        "--geometry-offset",
        type=int,
        default=0,
        help="Skip this many entries in the deterministic shuffled geometry pool.",
    )
    parser.add_argument(
        "--case-id-offset",
        type=int,
        default=0,
        help="Add this offset to generated case IDs; 2500 starts at case_2501.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--profile", default=None,
                        help="Analysis profile (Model/Analysis_Profile) recorded in the plan and applied by every child "
                             "run before its design is loaded or created, e.g. v2_nonlinear_flexure_screening_v1.")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--refresh-status",
        action="store_true",
        help="Rescan existing case artifacts and rebuild manifests without running analyses.",
    )
    parser.add_argument("--request-stop", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1 or (args.run_limit is not None and args.run_limit < 1):
        raise ValueError("workers and run-limit must be positive.")
    root = Path(args.output_root).resolve()
    args.output_root = str(root)
    root.mkdir(parents=True, exist_ok=True)
    stop_path = root / STOP_NAME
    if args.request_stop:
        write_json(stop_path, {"requested_at": now(), "reason": "user_requested_stop"})
        print(f"Stop requested: {stop_path}")
        return
    records = eligible_record_ids(args.set_name, args.max_npts)
    seismic_sites = tuple(args.seismic_sites or SEISMIC_SITES)
    if args.runs_per_record is not None and args.intensity_levels:
        raise ValueError(
            "Use either --runs-per-record or --intensity-level, not both: the "
            "first sets how many runs per record, the second pins their exact "
            "scale factors."
        )
    if args.runs_per_record is not None:
        if args.runs_per_record < 1:
            raise ValueError(
                f"runs-per-record must be positive; received {args.runs_per_record}."
            )
        if args.runs_per_record > len(DEFAULT_INTENSITY_LEVELS):
            # Only reachable without a calibration, where each run needs a
            # distinct scale factor and the default ladder has run out.
            raise ValueError(
                f"runs-per-record {args.runs_per_record} exceeds the "
                f"{len(DEFAULT_INTENSITY_LEVELS)} default intensity levels; "
                "pass --intensity-calibration, or set the ladder explicitly "
                "with repeated --intensity-level."
            )
        intensity_levels = tuple(DEFAULT_INTENSITY_LEVELS[: args.runs_per_record])
    else:
        intensity_levels = tuple(args.intensity_levels or DEFAULT_INTENSITY_LADDER)
    for site in seismic_sites:
        if site not in SEISMIC_SITES:
            raise ValueError(
                f"Unknown seismic site {site!r}; expected one of {list(SEISMIC_SITES)}."
            )
    calibration = None
    record_pairs = None
    if args.intensity_calibration:
        from Calibrate_Intensity import (
            calibration_model_mismatch,
            current_model_identity,
            load_calibration,
        )

        calibration = load_calibration(args.intensity_calibration)
        # A calibration fitted on another design route or hinge model would
        # place every run on the wrong intensity; refuse it unless overridden.
        stale = calibration_model_mismatch(calibration, current_model_identity())
        if stale and not args.allow_stale_calibration:
            detail = "; ".join(
                f"{key}: calibration {value['calibration']!r}, current {value['current']!r}"
                for key, value in stale.items()
            )
            raise SystemExit(
                f"Intensity calibration {args.intensity_calibration} was fitted on a different "
                f"design route or hinge model than this code generates with: {detail}. Refit it "
                "with Calibrate_Intensity.py on a pilot produced by the current code, or pass "
                "--allow-stale-calibration to proceed knowingly (into a new output root)."
            )
        record_pairs = eligible_record_pairs(args.set_name, args.max_npts)
        fit = calibration.get("fit", {})
        print(
            f"Intensity calibration: {args.intensity_calibration} "
            f"(fitted={fit.get('fitted')}, "
            f"pilot runs={calibration.get('pilot_observation_count')})"
        )
        if not fit.get("fitted"):
            print(f"  WARNING: {fit.get('reason')}")
        if stale:
            print(f"  WARNING: stale calibration accepted by --allow-stale-calibration: {sorted(stale)}")

    cases = load_plan(
        root,
        args.num_cases,
        records,
        args.seed,
        args.max_npts,
        set_name=args.set_name,
        geometry_offset=args.geometry_offset,
        case_id_offset=args.case_id_offset,
        seismic_sites=seismic_sites,
        intensity_levels=intensity_levels,
        records_per_case=args.records_per_case,
        calibration=calibration,
        record_pairs=record_pairs,
        profile=args.profile,
    )
    total_runs = sum(len(case_runs(case)) for case in cases)
    print(
        f"Plan: {len(cases)} cases x "
        f"{len(intensity_levels)} intensity level(s) = {total_runs} analyses; "
        f"hazards {list(seismic_sites)}"
    )
    plan_csv = root / "parameter_plan.csv"
    print(f"Plan: {root / 'parameter_plan.json'} ({len(cases)} cases, {len(records)} records)")
    print(f"Plan SHA256 (parameter_plan.csv): {sha256_file(plan_csv)}")
    if args.plan_only:
        return
    results = load_results(root)
    completed_ids = scan_completed(root, cases)
    # A refused design is terminal until the plan or the design inputs change;
    # record it as such so resumes do not pay for the same refusal again.
    refused = {} if args.retry_refused else scan_refused(root, cases)
    for case_id, reason in refused.items():
        if case_id not in completed_ids and results.get(case_id, {}).get("status") != "design_refused":
            results[case_id] = {"case_id": case_id, "status": "design_refused", "design_refusal": reason,
                                "updated_at": now()}
    if args.refresh_status:
        previous_state_path = root / "generation_state.json"
        previous_state = (
            json.loads(previous_state_path.read_text(encoding="utf-8"))
            if previous_state_path.exists()
            else {}
        )
        state = progress(
            root,
            cases,
            results,
            {},
            "refreshed",
            previous_state.get("started_at") or now(),
            completed_ids,
            write_manifest=True,
        )
        print(
            f"Refreshed manifests: {state['completed_count']}/{state['planned_count']} "
            f"complete; {state['failed_count']} failed; "
            f"{state['remaining_count']} remaining."
        )
        return
    scope, selected = select_pending_cases(
        cases,
        completed_ids,
        case_start=args.case_start,
        case_end=args.case_end,
        run_limit=args.run_limit,
        refused_ids=refused.keys(),
    )
    if refused:
        print(f"{len(refused)} case(s) with a refused design are skipped (terminal); "
              f"see generation_state.json design_refusals, or pass --retry-refused.")
    invocation_case_ids = [case["case_id"] for case in selected]
    resolved_case_end = args.case_end if args.case_end is not None else len(cases)
    started = now()
    stop_event, active, lock = threading.Event(), {}, threading.Lock()
    progress(
        root, cases, results, {}, "running", started, completed_ids,
        write_manifest=True,
        invocation_case_ids=invocation_case_ids,
        case_start=args.case_start,
        case_end=resolved_case_end,
    )
    print(
        f"Launching {len(selected)} pending cases from assigned range "
        f"{args.case_start}..{resolved_case_end} ({len(scope)} planned) "
        f"with {args.workers} workers."
    )
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = {executor.submit(run_case, args, case, stop_event, active, lock): case for case in selected}
    pending = set(futures)
    last_save = 0.0
    try:
        while pending:
            if stop_path.exists() and not stop_event.is_set():
                stop_event.set()
                print("Stop detected; active cases are checkpointing.")
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                case = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"case_id": case["case_id"], "status": "failed", "error": repr(exc)}
                results[case["case_id"]] = result
                if result.get("status") == "completed":
                    completed_ids.add(case["case_id"])
                print(f"{case['case_id']}: {result['status']}")
            if done or time.monotonic() - last_save > 5:
                with lock:
                    snapshot = dict(active)
                progress(
                    root, cases, results, snapshot,
                    "stopping" if stop_event.is_set() else "running",
                    started, completed_ids, write_manifest=bool(done),
                    invocation_case_ids=invocation_case_ids,
                    case_start=args.case_start,
                    case_end=resolved_case_end,
                )
                last_save = time.monotonic()
    except KeyboardInterrupt:
        stop_event.set()
        print("Ctrl+C received; active cases are checkpointing.")
        for case in selected:
            request_child_stop(paths_for(root, case)["child_stop"])
        for future, case in futures.items():
            try:
                result = future.result()
                results[case["case_id"]] = result
                if result.get("status") == "completed":
                    completed_ids.add(case["case_id"])
            except Exception as exc:
                results[case["case_id"]] = {"case_id": case["case_id"], "status": "failed", "error": repr(exc)}
    finally:
        executor.shutdown(wait=True)
    with lock:
        snapshot = dict(active)
    final = "stopped" if stop_event.is_set() else (
        "completed" if len(completed_ids) == len(cases) else "pilot_completed"
    )
    state = progress(
        root, cases, results, snapshot, final, started, completed_ids,
        write_manifest=True,
        invocation_case_ids=invocation_case_ids,
        case_start=args.case_start,
        case_end=resolved_case_end,
    )
    if stop_path.exists():
        stop_path.unlink()
    print(f"Completed {state['completed_count']}/{state['planned_count']}; remaining {state['remaining_count']}.")
    print(
        "Current invocation completed "
        f"{state['invocation_completed_count']}/{state['invocation_selected_count']}; "
        f"remaining {state['invocation_remaining_count']}."
    )


if __name__ == "__main__":
    main()
