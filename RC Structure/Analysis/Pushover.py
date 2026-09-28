import json
from pathlib import Path

import openseespy.opensees as ops

import Structure_Parameters as sp
from Model.Build_Model import apply_analysis_constraints, node_tag, roof_master_node
from Analysis.Diagnostics import (
    collect_failed_element_diagnostics,
    parse_failed_element_tags,
)
from Analysis.Mechanism_Checks import MechanismTracker
from RC_Design_Check import build_column_PM_diagram, get_element_tags, run_checks


def _base_shear_x():
    ops.reactions()
    shear = 0.0

    for j in range(sp.NUM_BAY_Y + 1):
        for i in range(sp.NUM_BAY_X + 1):
            base_node = node_tag(0, i, j)
            shear += ops.nodeReaction(base_node, 1)

    return -shear


def _restore_default_algorithm():
    ops.test("NormDispIncr", sp.PUSHOVER_TOL, sp.PUSHOVER_MAX_ITER)
    ops.algorithm("Newton")
    ops.integrator("DisplacementControl", roof_master_node(), 1, sp.PUSHOVER_DU)


def _run_recovery_step(roof_node, algorithm, tolerance, max_iter, du_factor=1.0):
    ops.test("NormDispIncr", tolerance, max_iter)
    ops.algorithm(*algorithm)
    ops.integrator("DisplacementControl", roof_node, 1, sp.PUSHOVER_DU * du_factor)
    try:
        ok = ops.analyze(1)
    finally:
        _restore_default_algorithm()
    return ok


def _try_recovery_step(roof_node):
    strategies = [
        ("NewtonRelaxed", ("Newton",), sp.PUSHOVER_FALLBACK_TOL, sp.PUSHOVER_FALLBACK_MAX_ITER, 1.0),
    ]

    for name, algorithm, tolerance, max_iter, du_factor in strategies:
        ok = _run_recovery_step(roof_node, algorithm, tolerance, max_iter, du_factor)
        if ok == 0:
            return ok, name

    return ok, strategies[-1][0]


def _roof_drift_ratio(roof_disp):
    return roof_disp / (sp.NUM_FLOOR * sp.STORY_H)


def _start_opensees_log(log_path):
    if log_path is None:
        return None

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        try:
            log_path.unlink()
        except PermissionError:
            stem = log_path.stem
            suffix = log_path.suffix
            for idx in range(1, 1000):
                candidate = log_path.with_name(f"{stem}_{idx}{suffix}")
                if not candidate.exists():
                    log_path = candidate
                    break
            else:
                raise

    ops.logFile(str(log_path))
    return log_path


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as file:
        json.dump(data, file, indent=2)

    return path


def _write_failure_snapshot(log_path, status):
    if log_path is None:
        return None

    snapshot_path = Path(log_path).with_name("pushover_failure_snapshot.json")
    return _write_json(snapshot_path, status)


def _write_failed_element_diagnostics(log_path, diagnostics):
    if log_path is None:
        return None

    diagnostics_path = Path(log_path).with_name("failed_element_diagnostics.json")
    return _write_json(diagnostics_path, diagnostics)


def _write_progress_snapshot(log_path, status):
    if log_path is None:
        return None

    progress_path = Path(log_path).with_name("pushover_progress.json")
    return _write_json(progress_path, status)


def _write_status_snapshot(log_path, status):
    if log_path is None:
        return None

    status_path = Path(log_path).with_name("pushover_status.json")
    return _write_json(status_path, status)


def _design_envelope_interval():
    return max(1, int(getattr(sp, "PUSHOVER_DESIGN_ENVELOPE_EVERY", 1)))


def _peak_shear_relative_step():
    return max(0.0, float(getattr(sp, "PUSHOVER_DESIGN_ENVELOPE_PEAK_SHEAR_REL_STEP", 0.05)))


def _drift_jump_relative_step():
    return max(0.0, float(getattr(sp, "PUSHOVER_DESIGN_ENVELOPE_DRIFT_JUMP_REL_STEP", 0.10)))


def _new_design_envelope():
    if not getattr(sp, "PUSHOVER_TRACK_DESIGN_ENVELOPE", False):
        return {
            "tracked": False,
            "sample_every": None,
            "num_samples": 0,
            "target_dcr_roof_drift_ratio": None,
            "target_dcr_snapshot": None,
            "elements": {},
            "summary": {
                "phase": "pushover_envelope",
                "all_design_checks_pass": None,
                "note": "Pushover design envelope tracking is disabled.",
            },
        }

    col_tags, beam_x_tags, beam_y_tags = get_element_tags()
    elements = {}

    for tag in col_tags:
        elements[str(tag)] = {
            "ele_tag": tag,
            "type": "column",
            "max_dcr_PM": 0.0,
            "max_dcr_V": 0.0,
            "max_governing_dcr": 0.0,
            "governing_limit_state": None,
            "governing_snapshot": None,
        }

    for tag in beam_x_tags + beam_y_tags:
        elements[str(tag)] = {
            "ele_tag": tag,
            "type": "beam",
            "max_dcr_pos": 0.0,
            "max_dcr_neg": 0.0,
            "max_dcr_V": 0.0,
            "max_governing_dcr": 0.0,
            "governing_limit_state": None,
            "governing_snapshot": None,
        }

    return {
        "tracked": True,
        "sample_every": _design_envelope_interval(),
        "num_samples": 0,
        "sampled_steps": [],
        "sample_reasons": [],
        "reason_counts": {},
        "target_dcr_roof_drift_ratio": float(
            getattr(sp, "PUSHOVER_DCR_CHECK_ROOF_DRIFT_RATIO", 0.02)
        ),
        "target_dcr_snapshot": None,
        "elements": elements,
        "summary": {},
        "_col_tags": col_tags,
        "_beam_tags": beam_x_tags + beam_y_tags,
        "_col_diagram": build_column_PM_diagram(),
        "_sampled_step_set": set(),
        "_last_forced_sample_step": None,
    }


def _snapshot_design_row(row, step, roof_disp, roof_drift, base_shear, load_factor):
    snapshot = dict(row)
    snapshot.update(
        {
            "step": step,
            "roof_disp_in": roof_disp,
            "roof_drift_ratio": roof_drift,
            "base_shear_kip": base_shear,
            "load_factor": load_factor,
        }
    )
    return snapshot


def _update_record(record, key, value, snapshot):
    if value > record.get(key, 0.0):
        record[key] = value
        record[f"{key}_snapshot"] = snapshot


def _governing_dcr_for_row(row):
    if row["type"] == "column":
        limit_states = {
            "PM": row["dcr_PM"],
            "shear": row["dcr_V"],
        }
    else:
        limit_states = {
            "flexure_pos": row["dcr_pos"],
            "flexure_neg": row["dcr_neg"],
            "shear": row["dcr_V"],
        }

    return max(limit_states.items(), key=lambda item: item[1])


def _summarize_design_snapshot(
    design_results,
    step,
    roof_disp,
    roof_drift,
    base_shear,
    load_factor,
    target_roof_drift,
):
    values = list(design_results.values())
    columns = [row for row in values if row["type"] == "column"]
    beams = [row for row in values if row["type"] == "beam"]
    failures = [int(tag) for tag, row in design_results.items() if not row["ok"]]
    failed_columns = [
        int(tag)
        for tag, row in design_results.items()
        if row["type"] == "column" and not row["ok"]
    ]
    failed_beams = [
        int(tag)
        for tag, row in design_results.items()
        if row["type"] == "beam" and not row["ok"]
    ]

    def max_value(rows, key):
        return max((row.get(key, 0.0) for row in rows), default=0.0)

    governing_rows = []
    for tag, row in design_results.items():
        limit_state, dcr = _governing_dcr_for_row(row)
        governing_rows.append(
            {
                "ele_tag": int(tag),
                "type": row["type"],
                "governing_limit_state": limit_state,
                "governing_dcr": dcr,
                "ok": row["ok"],
            }
        )
    governing_rows.sort(key=lambda row: row["governing_dcr"], reverse=True)

    summary = {
        "phase": "pushover_target_roof_drift",
        "target_roof_drift_ratio": target_roof_drift,
        "target_roof_disp_in": target_roof_drift * sp.NUM_FLOOR * sp.STORY_H,
        "actual_roof_drift_ratio": roof_drift,
        "actual_roof_disp_in": roof_disp,
        "step": step,
        "base_shear_kip": base_shear,
        "load_factor": load_factor,
        "all_design_checks_pass": len(failures) == 0,
        "num_failed_elements": len(failures),
        "failed_element_tags": failures,
        "num_failed_columns": len(failed_columns),
        "failed_column_tags": failed_columns,
        "num_failed_beams": len(failed_beams),
        "failed_beam_tags": failed_beams,
        "max_column_pm_dcr": max_value(columns, "dcr_PM"),
        "max_column_shear_dcr": max_value(columns, "dcr_V"),
        "max_beam_flexure_dcr": max(
            max_value(beams, "dcr_pos"),
            max_value(beams, "dcr_neg"),
        ),
        "max_beam_shear_dcr": max_value(beams, "dcr_V"),
        "top_governing_elements": governing_rows[:10],
        "notes": [
            "Snapshot is taken at the first pushover step reaching the configured roof-drift DCR check ratio.",
            "This snapshot is intended for lateral DCR reporting/design acceptance; the full 60 in pushover remains an ultimate-strength/collapse probe.",
        ],
    }

    return {
        "summary": summary,
        "results": {str(tag): dict(row) for tag, row in design_results.items()},
    }


def _record_design_envelope(
    envelope,
    step,
    roof_disp,
    roof_drift,
    base_shear,
    load_factor,
    reason="periodic",
    force=False,
):
    if not envelope.get("tracked"):
        return

    interval = envelope["sample_every"]
    if force and reason not in {
        "final_step",
        "load_reversal_stop",
        "target_roof_drift_dcr",
    }:
        min_gap = max(
            0,
            int(getattr(sp, "PUSHOVER_DESIGN_ENVELOPE_EVENT_MIN_STEP_GAP", 0)),
        )
        last_forced = envelope.get("_last_forced_sample_step")
        if (
            min_gap > 0
            and last_forced is not None
            and step - last_forced < min_gap
        ):
            envelope["sample_reasons"].append(
                {"step": step, "reason": f"{reason}_throttled"}
            )
            throttled_key = f"{reason}_throttled"
            envelope["reason_counts"][throttled_key] = (
                envelope["reason_counts"].get(throttled_key, 0) + 1
            )
            return

    if not force and step % interval != 0:
        return

    if step in envelope["_sampled_step_set"]:
        envelope["sample_reasons"].append({"step": step, "reason": reason})
        envelope["reason_counts"][reason] = envelope["reason_counts"].get(reason, 0) + 1
        return

    design_results = run_checks(
        envelope["_col_tags"],
        envelope["_beam_tags"],
        col_diagram=envelope["_col_diagram"],
    )
    envelope["num_samples"] += 1
    envelope["_sampled_step_set"].add(step)
    if force and reason not in {
        "final_step",
        "load_reversal_stop",
        "target_roof_drift_dcr",
    }:
        envelope["_last_forced_sample_step"] = step
    envelope["sampled_steps"].append(step)
    envelope["sample_reasons"].append({"step": step, "reason": reason})
    envelope["reason_counts"][reason] = envelope["reason_counts"].get(reason, 0) + 1

    if reason == "target_roof_drift_dcr":
        envelope["target_dcr_snapshot"] = _summarize_design_snapshot(
            design_results,
            step,
            roof_disp,
            roof_drift,
            base_shear,
            load_factor,
            envelope.get("target_dcr_roof_drift_ratio"),
        )

    for tag, row in design_results.items():
        record = envelope["elements"][str(tag)]
        snapshot = _snapshot_design_row(
            row,
            step,
            roof_disp,
            roof_drift,
            base_shear,
            load_factor,
        )

        if row["type"] == "column":
            limit_states = {
                "PM": row["dcr_PM"],
                "shear": row["dcr_V"],
            }
            _update_record(record, "max_dcr_PM", row["dcr_PM"], snapshot)
            _update_record(record, "max_dcr_V", row["dcr_V"], snapshot)
        else:
            limit_states = {
                "flexure_pos": row["dcr_pos"],
                "flexure_neg": row["dcr_neg"],
                "shear": row["dcr_V"],
            }
            _update_record(record, "max_dcr_pos", row["dcr_pos"], snapshot)
            _update_record(record, "max_dcr_neg", row["dcr_neg"], snapshot)
            _update_record(record, "max_dcr_V", row["dcr_V"], snapshot)

        governing_limit_state, governing_dcr = max(
            limit_states.items(),
            key=lambda item: item[1],
        )
        if governing_dcr > record["max_governing_dcr"]:
            record["max_governing_dcr"] = governing_dcr
            record["governing_limit_state"] = governing_limit_state
            record["governing_snapshot"] = snapshot


def _summarize_design_envelope(envelope):
    if not envelope.get("tracked"):
        return envelope

    elements = list(envelope["elements"].values())
    columns = [row for row in elements if row["type"] == "column"]
    beams = [row for row in elements if row["type"] == "beam"]
    failures = [
        row["ele_tag"]
        for row in elements
        if row.get("max_governing_dcr", 0.0) > 1.0
    ]
    failed_columns = [
        row["ele_tag"]
        for row in columns
        if row.get("max_governing_dcr", 0.0) > 1.0
    ]
    failed_beams = [
        row["ele_tag"]
        for row in beams
        if row.get("max_governing_dcr", 0.0) > 1.0
    ]

    def max_value(rows, key):
        return max((row.get(key, 0.0) for row in rows), default=0.0)

    envelope["summary"] = {
        "phase": "pushover_envelope",
        "tracked": True,
        "sample_every": envelope["sample_every"],
        "num_samples": envelope["num_samples"],
        "sampled_steps": envelope["sampled_steps"],
        "reason_counts": envelope["reason_counts"],
        "peak_base_shear_relative_milestone": _peak_shear_relative_step(),
        "interstory_drift_jump_relative_milestone": _drift_jump_relative_step(),
        "all_design_checks_pass": len(failures) == 0,
        "num_failed_elements": len(failures),
        "failed_element_tags": failures,
        "num_failed_columns": len(failed_columns),
        "failed_column_tags": failed_columns,
        "num_failed_beams": len(failed_beams),
        "failed_beam_tags": failed_beams,
        "max_column_pm_dcr": max_value(columns, "max_dcr_PM"),
        "max_column_shear_dcr": max_value(columns, "max_dcr_V"),
        "max_beam_flexure_dcr": max(
            max_value(beams, "max_dcr_pos"),
            max_value(beams, "max_dcr_neg"),
        ),
        "max_beam_shear_dcr": max_value(beams, "max_dcr_V"),
        "notes": [
            "Envelope values are sampled during the pushover history, not only at the final converged state.",
            "Each element stores the step and demand snapshot at its governing DCR.",
            "Samples include periodic checks plus forced checks at final step, hinge cap/ultimate events, significant running peak-base-shear milestones, and large interstory-drift jumps.",
        ],
    }

    envelope.pop("_col_tags", None)
    envelope.pop("_beam_tags", None)
    envelope.pop("_col_diagram", None)
    envelope.pop("_sampled_step_set", None)
    envelope.pop("_last_forced_sample_step", None)

    return envelope


def run_pushover(debug=True, log_path=None):
    roof_node = roof_master_node()
    log_path = _start_opensees_log(log_path)

    ops.wipeAnalysis()
    ops.system("BandGeneral")
    apply_analysis_constraints()
    ops.numberer("RCM")
    _restore_default_algorithm()
    ops.integrator("DisplacementControl", roof_node, 1, sp.PUSHOVER_DU)
    ops.analysis("Static")

    roof_disp = []
    roof_drift = []
    base_shear = []
    load_factors = []
    convergence = []
    mechanism_tracker = MechanismTracker()
    design_envelope = _new_design_envelope()
    status = {
        "completed_steps": 0,
        "requested_steps": sp.PUSHOVER_STEPS,
        "target_disp": sp.pushover_target_disp(),
        "failed": False,
        "failed_step": None,
        "failed_roof_disp": None,
        "failed_base_shear": None,
        "failed_element_tags": [],
        "failure_reason": None,
        "opensees_log": str(log_path) if log_path else None,
        "analysis_state": "running",
    }
    _write_progress_snapshot(log_path, status)
    _write_status_snapshot(log_path, status)
    peak_load_factor = 0.0
    peak_abs_base_shear = 0.0
    last_envelope_peak_shear_sample = 0.0
    previous_roof_drift = None
    previous_max_story_drift = None
    max_story_drift_jump = 0.0

    for step in range(sp.PUSHOVER_STEPS):
        algorithm_used = "Newton"
        ok = ops.analyze(1)

        if ok != 0:
            ok, algorithm_used = _try_recovery_step(roof_node)

        ux = ops.nodeDisp(roof_node, 1)
        drift = _roof_drift_ratio(ux)
        shear = _base_shear_x()

        if ok != 0:
            failed_element_tags = parse_failed_element_tags(log_path) if log_path else []
            status["failed"] = True
            status["failed_step"] = step
            status["failed_roof_disp"] = ux
            status["failed_base_shear"] = shear
            status["failed_element_tags"] = failed_element_tags
            status["failure_reason"] = "analysis_nonconvergence"
            status["analysis_state"] = "failed"
            status["failure_snapshot"] = str(_write_failure_snapshot(log_path, status))
            _write_status_snapshot(log_path, status)
            print(
                f"Pushover failed at step {step}: "
                f"Uroof={ux:.4f} in, drift={100.0 * drift:.2f}%, "
                f"Vbase={shear:.3f} kip"
            )
            break

        roof_disp.append(ux)
        roof_drift.append(drift)
        base_shear.append(shear)
        load_factors.append(ops.getLoadFactor(2))
        load_factor = load_factors[-1]
        peak_load_factor = max(peak_load_factor, load_factor)
        mechanism_step = mechanism_tracker.record_step(step, ux, shear)

        target_drift = design_envelope.get("target_dcr_roof_drift_ratio")
        if (
            design_envelope.get("tracked")
            and target_drift is not None
            and target_drift > 0.0
            and design_envelope.get("target_dcr_snapshot") is None
            and drift >= target_drift
            and (previous_roof_drift is None or previous_roof_drift < target_drift)
        ):
            _record_design_envelope(
                design_envelope,
                step,
                ux,
                drift,
                shear,
                load_factor,
                reason="target_roof_drift_dcr",
                force=True,
            )

        _record_design_envelope(
            design_envelope,
            step,
            ux,
            drift,
            shear,
            load_factor,
            reason="periodic",
        )
        forced_reasons = []
        abs_shear = abs(shear)
        if abs_shear > peak_abs_base_shear:
            peak_abs_base_shear = abs_shear
            rel_step = _peak_shear_relative_step()
            if (
                last_envelope_peak_shear_sample <= 0.0
                or abs_shear >= last_envelope_peak_shear_sample * (1.0 + rel_step)
            ):
                forced_reasons.append("peak_base_shear_milestone")
                last_envelope_peak_shear_sample = abs_shear

        if mechanism_step["new_hinge_cap_events"]:
            forced_reasons.append("hinge_cap_event")
        if mechanism_step["new_hinge_ultimate_events"]:
            forced_reasons.append("hinge_ultimate_event")
        if mechanism_step["new_column_yield_events"]:
            forced_reasons.append("column_yield_event")

        max_story_drift = mechanism_step["max_abs_story_drift_ratio"]
        if previous_max_story_drift is not None:
            drift_jump = abs(max_story_drift - previous_max_story_drift)
            rel_step = _drift_jump_relative_step()
            if (
                drift_jump > 1.0e-10
                and (
                    max_story_drift_jump <= 0.0
                    or drift_jump >= max_story_drift_jump * (1.0 + rel_step)
                )
            ):
                max_story_drift_jump = drift_jump
                forced_reasons.append("largest_interstory_drift_jump")
        previous_max_story_drift = max_story_drift

        for reason in forced_reasons:
            _record_design_envelope(
                design_envelope,
                step,
                ux,
                drift,
                shear,
                load_factor,
                reason=reason,
                force=True,
            )
        previous_roof_drift = drift
        convergence.append(
            {
                "step": step,
                "ok": ok,
                "algorithm": algorithm_used,
                "roof_disp": ux,
                "roof_drift": drift,
                "base_shear": shear,
                "load_factor": load_factor,
            }
        )
        status["completed_steps"] = step + 1
        status["current_roof_disp"] = ux
        status["current_base_shear"] = shear
        status["current_load_factor"] = load_factor

        if (
            sp.PUSHOVER_STOP_ON_LOAD_REVERSAL
            and peak_load_factor >= sp.PUSHOVER_LOAD_REVERSAL_MIN_PEAK_FACTOR
            and load_factor <= sp.PUSHOVER_LOAD_REVERSAL_LIMIT
        ):
            status["failed"] = True
            status["failed_step"] = step
            status["failed_roof_disp"] = ux
            status["failed_base_shear"] = shear
            status["failed_element_tags"] = []
            status["failure_reason"] = "lateral_load_factor_reversal"
            status["analysis_state"] = "failed"
            status["failure_snapshot"] = str(_write_failure_snapshot(log_path, status))
            _record_design_envelope(
                design_envelope,
                step,
                ux,
                drift,
                shear,
                load_factor,
                reason="load_reversal_stop",
                force=True,
            )
            _write_progress_snapshot(log_path, status)
            _write_status_snapshot(log_path, status)
            print(
                f"Pushover stopped at step {step}: "
                f"load factor reversed to {load_factor:.4f}, "
                f"Uroof={ux:.4f} in, drift={100.0 * drift:.2f}%, "
                f"Vbase={shear:.3f} kip"
            )
            break

        if debug and step % sp.PUSHOVER_DEBUG_EVERY == 0:
            _write_progress_snapshot(log_path, status)
            _write_status_snapshot(log_path, status)
            print(
                f"step={step:04d}, Uroof={ux:.3f} in, "
                f"drift={100.0 * drift:.2f}%, Vbase={shear:.2f} kip, "
                f"lambda={load_factor:.4f}, alg={algorithm_used}"
            )

    if not status["failed"]:
        status["analysis_state"] = "completed"
    if convergence:
        final = convergence[-1]
        _record_design_envelope(
            design_envelope,
            final["step"],
            final["roof_disp"],
            final["roof_drift"],
            final["base_shear"],
            final["load_factor"],
            reason="final_step",
            force=True,
        )
    _write_progress_snapshot(log_path, status)
    _write_status_snapshot(log_path, status)

    failed_element_diagnostics = collect_failed_element_diagnostics(
        status["failed_element_tags"]
    )

    if status["failed"]:
        diagnostics_path = _write_failed_element_diagnostics(
            log_path,
            failed_element_diagnostics,
        )
        if diagnostics_path is not None:
            status["failed_element_diagnostics"] = str(diagnostics_path)
        status["failure_snapshot"] = str(_write_failure_snapshot(log_path, status))

    return {
        "roof_node": roof_node,
        "roof_disp": roof_disp,
        "roof_drift": roof_drift,
        "base_shear": base_shear,
        "load_factors": load_factors,
        "convergence": convergence,
        "mechanism_diagnostics": mechanism_tracker.results(),
        "design_envelope": _summarize_design_envelope(design_envelope),
        "failed_element_diagnostics": failed_element_diagnostics,
        "status": status,
        "final_disp": (
            ops.nodeDisp(roof_node, 1),
            ops.nodeDisp(roof_node, 2),
            ops.nodeDisp(roof_node, 3),
        ),
    }
