import os
import inspect
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Ensure the terminal can render Unicode (φ, etc.) on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import openseespy.opensees as ops

import Structure_Parameters as sp
from Geometry_Overrides import apply_geometry_from_environment
from Analysis.Gravity import run_gravity_analysis
from Analysis.Modal import run_modal_analysis
from Analysis.Pushover import run_pushover
from Analysis.NTHA import run_ntha
from Analysis.Diagnostics import print_failed_element_diagnostics
from Data_Generation.Graph_Exporter import save_analysis_sample
from Loads.Gravity_Loads import apply_gravity_loads
from Loads.Seismic_ELF import apply_lateral_loads
from Loads.Ground_Motion import (
    ground_motion_catalog_summary,
    load_ground_motion_pair_by_result_id,
    load_ground_motion_pairs,
    load_ground_motion_record,
)
from Ground_Motion_Main import (
    analysis_run_name,
    save_ntha_outputs,
    validate_ntha_output_compatibility,
)
from Model.Build_Model import build_model
from Plotting.Plot_Results import (
    plot_pushover_curve,
    plot_roof_drift_curve,
    save_pushover_curve,
)
from Design.Config import DesignConfig
from RC_Design_Check import get_element_tags, print_summary, run_checks
from Redesign import apply_updates, run_iterative_redesign


FAILURE_ARTIFACTS = (
    "failed_element_diagnostics.json",
    "pushover_failure_snapshot.json",
)


def should_show_plots():
    value = os.environ.get("RC_SHOW_PLOTS", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def clear_failure_artifacts(outputs_dir):
    for filename in FAILURE_ARTIFACTS:
        path = outputs_dir / filename
        if path.exists():
            path.unlink()


def model_summary():
    return {
        "analysis_method": selected_analysis_method(),
        "geometry_variant": getattr(sp, "GEOMETRY_VARIANT_NAME", "baseline"),
        "asce_period_estimate": sp.asce_fundamental_period_check(),
        "num_bay_x": sp.NUM_BAY_X,
        "num_bay_y": sp.NUM_BAY_Y,
        "num_stories": sp.NUM_FLOOR,
        "num_floor": sp.NUM_FLOOR,
        "bay_x_in": sp.BAY_X,
        "bay_y_in": sp.BAY_Y,
        "story_height_in": sp.STORY_H,
        "total_height_in": sp.NUM_FLOOR * sp.STORY_H,
        "column_geometry": {
            "width_in": sp.B_COL,
            "depth_in": sp.H_COL,
            "area_in2": sp.rect_area(sp.B_COL, sp.H_COL),
            "iy_in4": sp.rect_iy(sp.B_COL, sp.H_COL),
            "iz_in4": sp.rect_iz(sp.B_COL, sp.H_COL),
        },
        "beam_geometry": {
            "width_in": sp.B_BEAM,
            "depth_in": sp.H_BEAM,
            "area_in2": sp.rect_area(sp.B_BEAM, sp.H_BEAM),
            "iy_in4": sp.rect_iy(sp.B_BEAM, sp.H_BEAM),
            "iz_in4": sp.rect_iz(sp.B_BEAM, sp.H_BEAM),
        },
    }


def print_model_summary(summary):
    print("\nModel summary:")
    print(f"  Analysis method: {summary.get('analysis_method', 'pushover')}")
    print(f"  Geometry variant: {summary.get('geometry_variant', 'baseline')}")
    print(
        f"  Plan: {summary['num_bay_x']} bay(s) in X @ {summary['bay_x_in']:g} in, "
        f"{summary['num_bay_y']} bay(s) in Y @ {summary['bay_y_in']:g} in"
    )
    print(
        f"  Vertical: {summary['num_stories']} storie(s) @ "
        f"{summary['story_height_in']:g} in = {summary['total_height_in']:g} in total"
    )
    col = summary["column_geometry"]
    beam = summary["beam_geometry"]
    print(
        f"  Columns: {col['width_in']:g} in x {col['depth_in']:g} in "
        f"(A={col['area_in2']:.2f} in^2)"
    )
    print(
        f"  Beams: {beam['width_in']:g} in x {beam['depth_in']:g} in "
        f"(A={beam['area_in2']:.2f} in^2)"
    )
    asce_check = summary.get("asce_period_estimate", {})
    if asce_check:
        print(
            f"  ASCE approximate Ta: {asce_check['asce_ta_sec']:.4f} sec "
            f"for h_n={asce_check['height_ft']:.2f} ft "
            f"({asce_check['equation']})"
        )


def print_gravity_results(results):
    print("Static gravity analysis succeeded")
    print("\nGravity-only roof control node displacement:")
    print(f"Ux = {results['ux']:.6e} in")
    print(f"Uy = {results['uy']:.6e} in")
    print(f"Uz = {results['uz']:.6e} in (diaphragm master UZ is constrained)")
    print("\nGravity-only physical roof floor vertical displacement:")
    print(f"Uz min = {results['roof_floor_uz_min']:.6e} in")
    print(f"Uz max = {results['roof_floor_uz_max']:.6e} in")
    print(f"Uz avg = {results['roof_floor_uz_avg']:.6e} in")


def print_modal_results(modes):
    print("\nModal Properties:")
    for item in modes:
        if not item["valid"]:
            print(
                f"Mode {item['mode']}: "
                f"skipped invalid eigenvalue lambda = {item['lambda']:.12e}"
            )
            continue

        print(
            f"Mode {item['mode']}: "
            f"T = {item['period']:.6f} sec, "
            f"omega = {item['omega']:.6f} rad/sec, "
            f"f = {item['frequency']:.6f} Hz"
        )



def first_valid_modal_period(modes):
    for item in modes:
        if item.get("valid") and item.get("period") is not None:
            return item["period"]
    return None


def asce_period_check_from_modal_results(modes, modal_source):
    return sp.asce_fundamental_period_check(
        model_period_sec=first_valid_modal_period(modes),
        modal_source=modal_source,
    )


def print_asce_period_check(check):
    print("\nASCE approximate fundamental period check:")
    print(
        f"  Ta = {check['asce_ta_sec']:.4f} sec "
        f"({check['equation']}, Ct={check['ct']}, x={check['exponent']}, "
        f"h_n={check['height_ft']:.2f} ft)"
    )
    if check.get("model_period_sec") is None:
        print("  Model period: not available yet")
        return

    print(
        f"  Model T1 = {check['model_period_sec']:.4f} sec "
        f"from {check.get('modal_source') or 'unknown source'}"
    )
    print(f"  T1 / Ta = {check['period_ratio_to_asce_ta']:.3f}")
    if check.get("warning"):
        print(f"  [WARNING] {check['warning']}")

def print_pushover_results(results, gravity_results):
    status = results["status"]
    print(f"\nPushover completed {status['completed_steps']} steps")

    if status["failed"]:
        print(f"Pushover failed at step {status['failed_step']}")
        if status.get("failed_element_tags"):
            print(f"Failed element tags: {status['failed_element_tags']}")
        print_failed_element_diagnostics(results.get("failed_element_diagnostics", []))

    ux, uy, uz = results["final_disp"]
    print("\nGravity + lateral roof control node displacement:")
    print(f"Ux = {ux:.6e} in")
    print(f"Uy = {uy:.6e} in")
    print(f"Uz = {uz:.6e} in (diaphragm master UZ is constrained)")

    print("\nIncrement due to lateral load:")
    print(f"dUx = {ux - gravity_results['ux']:.6e} in")
    print(f"dUy = {uy - gravity_results['uy']:.6e} in")
    print(f"dUz = {uz - gravity_results['uz']:.6e} in")

    mechanism = results.get("mechanism_diagnostics", {}).get("summary", {})
    if mechanism:
        print("\nMechanism diagnostics:")
        for target, snapshot in mechanism.get("story_drift_targets", {}).items():
            print(
                f"Roof drift {target}: max story drift = "
                f"{100.0 * snapshot['max_abs_story_drift_ratio']:.3f}% "
                f"between floors {snapshot['max_floor_pair']} "
                f"(story {snapshot['max_story']})"
            )
            ranking = snapshot.get("story_drift_ranking", [])
            if ranking:
                comparison = ", ".join(
                    f"{row['floor_pair']}: "
                    f"{100.0 * row['abs_story_drift_ratio']:.3f}%"
                    for row in ranking
                )
                print(f"  Interstory drift ranking: {comparison}")

        for target in mechanism.get("unreached_story_drift_targets", []):
            print(f"Roof drift {target}: not reached in this analysis")

        max_interstory = mechanism.get("max_interstory_drift_overall")
        if max_interstory:
            print(
                "Peak interstory drift over pushover: "
                f"{100.0 * max_interstory['abs_story_drift_ratio']:.3f}% "
                f"between floors {max_interstory['floor_pair']} "
                f"at roof drift {100.0 * max_interstory['roof_drift_ratio']:.3f}%"
            )

        first_cap = mechanism.get("first_hinge_cap")
        if first_cap:
            first_cap_group = mechanism.get("first_hinge_cap_group", [])
            print(
                "First hinge cap: "
                f"element {first_cap['physical_ele_tag']} "
                f"end {first_cap['end']} {first_cap['rot_dir']} "
                f"at roof drift {100.0 * first_cap['roof_drift_ratio']:.3f}% "
                f"({len(first_cap_group)} hinges at this step)"
            )
        else:
            print("First hinge cap: none reached")

        first_column = mechanism.get("first_column_nominal_yield")
        if first_column:
            print(
                "First column nominal yield: "
                f"element {first_column['ele_tag']} "
                f"end {first_column['end']} "
                f"at roof drift {100.0 * first_column['roof_drift_ratio']:.3f}%"
            )
        else:
            print("First column nominal yield: none detected")

        first_ultimate = mechanism.get("first_hinge_ultimate_exceedance")
        if first_ultimate:
            print(
                "First theta_u exceedance: "
                f"element {first_ultimate['physical_ele_tag']} "
                f"end {first_ultimate['end']} {first_ultimate['rot_dir']} "
                f"at roof drift {100.0 * first_ultimate['roof_drift_ratio']:.3f}%"
            )
        else:
            print("Theta_u exceedance: none detected")

        print(
            "Beam-sway before column mechanism indicator: "
            f"{mechanism.get('beam_sway_before_column_mechanism_indicator')}"
        )


def design_results_to_jsonable(design_results):
    return {str(k): v for k, v in design_results.items()}


def write_design_results(path, design_results):
    with path.open("w") as file:
        json.dump(design_results_to_jsonable(design_results), file, indent=2)


def summarize_design_results(
    design_results,
    phase,
    redesign_ok=None,
    n_iter=None,
    notes=None,
):
    values = list(design_results.values())
    columns = [row for row in values if row["type"] == "column"]
    beams = [row for row in values if row["type"] == "beam"]
    failures = [int(tag) for tag, row in design_results.items() if not row["ok"]]

    def max_value(rows, key, default=0.0):
        return max((row[key] for row in rows), default=default)

    beam_flex_max = max(
        (max(row["dcr_pos"], row["dcr_neg"]) for row in beams),
        default=0.0,
    )

    status = {
        "model": model_summary(),
        "phase": phase,
        "all_design_checks_pass": len(failures) == 0,
        "num_failed_elements": len(failures),
        "failed_element_tags": failures,
        "max_column_pm_dcr": max_value(columns, "dcr_PM"),
        "max_column_shear_dcr": max_value(columns, "dcr_V"),
        "max_beam_flexure_dcr": beam_flex_max,
        "max_beam_shear_dcr": max_value(beams, "dcr_V"),
        "final_column_reinforcement": {
            "bar_size": sp.COL_BAR_SIZE,
            "top_bars": sp.COL_TOP_BARS,
            "bottom_bars": sp.COL_BOT_BARS,
            "side_bars_per_face": sp.COL_SIDE_BARS,
            "bar_area_in2": sp.COL_BAR_AREA,
            "stirrup_bar_size": sp.COL_STIRRUP_BAR_SIZE,
            "stirrup_legs": sp.COL_STIRRUP_LEGS,
            "stirrup_spacing_in": sp.COL_STIRRUP_SPACING,
            "stirrup_area_in2": sp.COL_STIRRUP_LEGS * sp.rebar_area(sp.COL_STIRRUP_BAR_SIZE),
        },
        "final_beam_reinforcement": {
            "bar_size": sp.BEAM_BAR_SIZE,
            "top_bars": sp.BEAM_TOP_BARS,
            "bottom_bars": sp.BEAM_BOT_BARS,
            "bar_area_in2": sp.BEAM_BAR_AREA,
            "stirrup_bar_size": sp.BEAM_STIRRUP_BAR_SIZE,
            "stirrup_legs": sp.BEAM_STIRRUP_LEGS,
            "stirrup_spacing_in": sp.BEAM_STIRRUP_SPACING,
            "stirrup_area_in2": sp.BEAM_STIRRUP_LEGS * sp.rebar_area(sp.BEAM_STIRRUP_BAR_SIZE),
        },
    }

    if redesign_ok is not None:
        status["redesign_converged"] = redesign_ok
    if n_iter is not None:
        status["redesign_iterations"] = n_iter
    if notes is not None:
        status["notes"] = notes

    return status


def combine_design_statuses(
    gravity_status,
    pushover_final_status,
    pushover_envelope_status=None,
    pushover_redesign_history=None,
):
    lateral_status = pushover_envelope_status or pushover_final_status
    lateral_phase = lateral_status.get("phase")
    status = {
        "model": model_summary(),
        "all_design_checks_pass": (
            gravity_status["all_design_checks_pass"]
            and lateral_status["all_design_checks_pass"]
        ),
        "gravity": gravity_status,
        "pushover_final": pushover_final_status,
        "final_column_reinforcement": pushover_final_status["final_column_reinforcement"],
        "final_beam_reinforcement": pushover_final_status["final_beam_reinforcement"],
        "notes": [
            "Gravity checks use the post-redesign gravity-analysis state.",
            "Pushover-final checks use element forces at the final converged pushover state.",
            (
                "The configured target roof-drift DCR snapshot is used for lateral design acceptance; the full 60 in pushover envelope is retained for ultimate/collapse behavior."
                if lateral_phase == "pushover_target_roof_drift"
                else "When available, pushover-envelope checks are used for lateral design acceptance instead of the final-state snapshot."
            ),
            "Shear DCRs are included in pass/fail status, but the current redesign loop only changes longitudinal bars.",
        ],
    }

    if pushover_envelope_status is not None:
        status["pushover_lateral_design"] = pushover_envelope_status
        if lateral_phase == "pushover_target_roof_drift":
            status["pushover_2pct_drift"] = pushover_envelope_status
            full_envelope = pushover_envelope_status.get("full_history_envelope_summary")
            if full_envelope is not None:
                status["pushover_envelope"] = full_envelope
                status["ultimate_envelope_pass"] = full_envelope.get(
                    "all_design_checks_pass"
                )
                status["ultimate_envelope_num_failed_elements"] = full_envelope.get(
                    "num_failed_elements"
                )
                status["ultimate_envelope_failed_element_tags"] = full_envelope.get(
                    "failed_element_tags", []
                )
                status["ultimate_envelope_num_failed_columns"] = full_envelope.get(
                    "num_failed_columns"
                )
                status["ultimate_envelope_failed_column_tags"] = full_envelope.get(
                    "failed_column_tags", []
                )
                status["ultimate_envelope_num_failed_beams"] = full_envelope.get(
                    "num_failed_beams"
                )
                status["ultimate_envelope_failed_beam_tags"] = full_envelope.get(
                    "failed_beam_tags", []
                )
                status["ultimate_envelope_max_column_pm_dcr"] = full_envelope.get(
                    "max_column_pm_dcr"
                )
                status["ultimate_envelope_max_column_shear_dcr"] = full_envelope.get(
                    "max_column_shear_dcr"
                )
                status["ultimate_envelope_max_beam_flexure_dcr"] = full_envelope.get(
                    "max_beam_flexure_dcr"
                )
                status["ultimate_envelope_max_beam_shear_dcr"] = full_envelope.get(
                    "max_beam_shear_dcr"
                )
                status["notes"].append(
                    "Ultimate-envelope checks summarize the full 60 in pushover history and are reported separately from overall acceptance."
                )
        else:
            status["pushover_envelope"] = pushover_envelope_status

    if pushover_redesign_history is not None:
        status["pushover_redesign_history"] = pushover_redesign_history
        final_attempt = pushover_redesign_history[-1] if pushover_redesign_history else {}
        final_failed_drift = final_attempt.get("failed_roof_drift_ratio")
        min_required_drift = final_attempt.get(
            "minimum_acceptable_roof_drift_ratio",
            sp.PUSHOVER_MIN_ACCEPTABLE_DRIFT_RATIO,
        )
        pushover_drift_pass = (
            not final_attempt.get("failed", False)
            or (
                final_failed_drift is not None
                and final_failed_drift >= min_required_drift
            )
        )
        status["pushover_minimum_drift_pass"] = pushover_drift_pass
        status["minimum_acceptable_roof_drift_ratio"] = min_required_drift
        status["overall_acceptance_pass"] = (
            status["all_design_checks_pass"]
            and pushover_drift_pass
        )
        if not pushover_drift_pass:
            status["notes"].append(
                "Pushover failed before the configured minimum roof drift; the design checks pass, but the nonlinear drift acceptance target does not."
            )

    return status


def run_pushover_envelope_design_check(outputs_dir, pushover_results):
    envelope = pushover_results.get("design_envelope", {})
    if not envelope.get("tracked"):
        return None, None

    status = envelope.get("summary", {})
    elements = envelope.get("elements", {})
    target_snapshot = envelope.get("target_dcr_snapshot")

    print("\nPeak-over-history pushover design envelope:")
    print(
        f"  Samples checked: {status.get('num_samples', 0)} "
        f"(every {status.get('sample_every', '?')} step(s))"
    )
    print(
        f"  Column P-M max DCR : {status.get('max_column_pm_dcr', 0.0):.3f}"
    )
    print(
        f"  Column shear max DCR: {status.get('max_column_shear_dcr', 0.0):.3f}"
    )
    print(
        f"  Beam flexure max DCR: {status.get('max_beam_flexure_dcr', 0.0):.3f}"
    )
    print(
        f"  Beam shear max DCR  : {status.get('max_beam_shear_dcr', 0.0):.3f}"
    )
    print(
        f"  Envelope failures   : {status.get('num_failed_elements', 0)}"
    )
    print(
        f"    Columns failing   : {status.get('num_failed_columns', 0)}"
    )
    if status.get("failed_column_tags"):
        print(f"      Column tags     : {status['failed_column_tags']}")
    print(
        f"    Beams failing     : {status.get('num_failed_beams', 0)}"
    )
    if status.get("failed_beam_tags"):
        print(f"      Beam tags       : {status['failed_beam_tags']}")

    with (outputs_dir / "design_check_pushover_envelope.json").open("w") as file:
        json.dump(elements, file, indent=2)

    acceptance_status = status
    acceptance_results = elements

    if target_snapshot:
        target_status = target_snapshot.get("summary", {})
        target_results = target_snapshot.get("results", {})
        target_status["full_history_envelope_summary"] = status

        print(
            "\nPushover DCR snapshot at "
            f"{100.0 * target_status.get('target_roof_drift_ratio', 0.0):.1f}% "
            "roof drift:"
        )
        print(
            f"  Target roof displacement: "
            f"{target_status.get('target_roof_disp_in', 0.0):.3f} in"
        )
        print(
            f"  Actual snapshot: step {target_status.get('step')} | "
            f"Uroof={target_status.get('actual_roof_disp_in', 0.0):.3f} in | "
            f"drift={100.0 * target_status.get('actual_roof_drift_ratio', 0.0):.3f}% | "
            f"Vbase={target_status.get('base_shear_kip', 0.0):.3f} kip"
        )
        print(
            f"  Column P-M max DCR : "
            f"{target_status.get('max_column_pm_dcr', 0.0):.3f}"
        )
        print(
            f"  Column shear max DCR: "
            f"{target_status.get('max_column_shear_dcr', 0.0):.3f}"
        )
        print(
            f"  Beam flexure max DCR: "
            f"{target_status.get('max_beam_flexure_dcr', 0.0):.3f}"
        )
        print(
            f"  Beam shear max DCR  : "
            f"{target_status.get('max_beam_shear_dcr', 0.0):.3f}"
        )
        print(
            f"  Target-drift failures: "
            f"{target_status.get('num_failed_elements', 0)}"
        )
        print(
            f"    Columns failing: "
            f"{target_status.get('num_failed_columns', 0)}"
        )
        if target_status.get("failed_column_tags"):
            print(f"      Column tags: {target_status['failed_column_tags']}")
        print(
            f"    Beams failing  : "
            f"{target_status.get('num_failed_beams', 0)}"
        )
        if target_status.get("failed_beam_tags"):
            print(f"      Beam tags: {target_status['failed_beam_tags']}")

        with (outputs_dir / "design_check_pushover_2pct_drift.json").open("w") as file:
            json.dump(target_snapshot, file, indent=2)

        if getattr(sp, "PUSHOVER_USE_TARGET_DCR_FOR_REDESIGN", True):
            acceptance_status = target_status
            acceptance_results = target_results

    return acceptance_status, acceptance_results


def run_final_pushover_design_check(outputs_dir, pushover_failed=False):
    col_tags, beam_x_tags, beam_y_tags = get_element_tags()
    design_results = run_checks(col_tags, beam_x_tags + beam_y_tags)

    print("\nFinal-state design check at end of pushover:")
    print_summary(design_results, verbose=False)

    write_design_results(outputs_dir / "design_check_pushover_final.json", design_results)

    notes = [
        "Demand snapshot taken from the final converged pushover state.",
        "This is a final-state check only, not a peak-over-history envelope.",
    ]
    if pushover_failed:
        notes.append(
            "Pushover did not converge to the requested target; this check is a failed-state snapshot and should not be treated as a valid final lateral design envelope."
        )

    status = summarize_design_results(
        design_results,
        phase="pushover_final",
        notes=notes,
    )

    return status, design_results


def reinforcement_snapshot():
    return {
        "column": {
            "bar_size": sp.COL_BAR_SIZE,
            "top_bars": sp.COL_TOP_BARS,
            "bottom_bars": sp.COL_BOT_BARS,
            "side_bars_per_face": sp.COL_SIDE_BARS,
            "bar_area_in2": sp.COL_BAR_AREA,
            "stirrup_bar_size": sp.COL_STIRRUP_BAR_SIZE,
            "stirrup_legs": sp.COL_STIRRUP_LEGS,
            "stirrup_spacing_in": sp.COL_STIRRUP_SPACING,
            "stirrup_area_in2": sp.COL_STIRRUP_LEGS * sp.rebar_area(sp.COL_STIRRUP_BAR_SIZE),
        },
        "beam": {
            "bar_size": sp.BEAM_BAR_SIZE,
            "top_bars": sp.BEAM_TOP_BARS,
            "bottom_bars": sp.BEAM_BOT_BARS,
            "bar_area_in2": sp.BEAM_BAR_AREA,
            "stirrup_bar_size": sp.BEAM_STIRRUP_BAR_SIZE,
            "stirrup_legs": sp.BEAM_STIRRUP_LEGS,
            "stirrup_spacing_in": sp.BEAM_STIRRUP_SPACING,
            "stirrup_area_in2": sp.BEAM_STIRRUP_LEGS * sp.rebar_area(sp.BEAM_STIRRUP_BAR_SIZE),
        },
    }


def _next_value(current, values):
    for value in sorted(set(values)):
        if value > current:
            return value
    return None


def _next_count(current, bounds):
    _, upper = bounds
    if current < upper:
        return current + 1
    return None


def _failed_member_classes(failed_element_tags):
    if not failed_element_tags:
        return {"column", "beam"}

    col_tags, beam_x_tags, beam_y_tags = get_element_tags()
    columns = set(col_tags)
    beams = set(beam_x_tags + beam_y_tags)
    classes = set()

    for tag in failed_element_tags:
        if tag in columns:
            classes.add("column")
        elif tag in beams:
            classes.add("beam")

    return classes or {"column", "beam"}


def _column_reinforcement_ratio(update):
    total_bars = update["n_top"] + update["n_bot"] + 2 * update["n_side"]
    return total_bars * sp.rebar_area(update["bar_size"]) / (sp.B_COL * sp.H_COL)


def _column_total_steel_area(update):
    total_bars = update["n_top"] + update["n_bot"] + 2 * update["n_side"]
    return total_bars * sp.rebar_area(update["bar_size"])


def _first_valid_column_update(candidates, cfg):
    for update in candidates:
        if _column_reinforcement_ratio(update) <= cfg.rebar.rho_col_max:
            return update
    return None


def _column_strengthening_update(cfg, prefer_bar_count=False):
    candidates = []

    next_top = _next_count(sp.COL_TOP_BARS, cfg.rebar.col_n_top_range)
    if next_top is not None:
        candidates.append({
            "bar_size": sp.COL_BAR_SIZE,
            "n_top": next_top,
            "n_bot": next_top,
            "n_side": sp.COL_SIDE_BARS,
        })

    next_side = _next_value(sp.COL_SIDE_BARS, cfg.rebar.col_n_side_options)
    if next_side is not None:
        candidates.append({
            "bar_size": sp.COL_BAR_SIZE,
            "n_top": sp.COL_TOP_BARS,
            "n_bot": sp.COL_BOT_BARS,
            "n_side": next_side,
        })

    next_bar = _next_value(sp.COL_BAR_SIZE, cfg.rebar.bar_sizes_col)
    if next_bar is not None:
        candidates.append({
            "bar_size": next_bar,
            "n_top": sp.COL_TOP_BARS,
            "n_bot": sp.COL_BOT_BARS,
            "n_side": sp.COL_SIDE_BARS,
        })

    if not prefer_bar_count and candidates:
        candidates = candidates[-1:] + candidates[:-1]

    return _first_valid_column_update(candidates, cfg)


def _column_envelope_strengthening_update(cfg, envelope_summary):
    current = {
        "bar_size": sp.COL_BAR_SIZE,
        "n_top": sp.COL_TOP_BARS,
        "n_bot": sp.COL_BOT_BARS,
        "n_side": sp.COL_SIDE_BARS,
    }
    current_ast = _column_total_steel_area(current)
    governing_dcr = max(
        envelope_summary.get("max_column_pm_dcr") or 0.0,
        envelope_summary.get("max_column_shear_dcr") or 0.0,
    )
    target_dcr = max(0.1, getattr(cfg.dcr, "dcr_band_hi", 0.95))
    required_ast = current_ast * max(1.0, governing_dcr / target_dcr)

    candidates = []
    _, max_top = cfg.rebar.col_n_top_range

    for bar_size in sorted(set(cfg.rebar.bar_sizes_col)):
        if bar_size < sp.COL_BAR_SIZE:
            continue

        for n_top in range(sp.COL_TOP_BARS, max_top + 1):
            for n_side in sorted(set(cfg.rebar.col_n_side_options)):
                if n_side < sp.COL_SIDE_BARS:
                    continue

                update = {
                    "bar_size": bar_size,
                    "n_top": n_top,
                    "n_bot": n_top,
                    "n_side": n_side,
                }
                ast = _column_total_steel_area(update)

                if ast <= current_ast + 1.0e-9:
                    continue
                if _column_reinforcement_ratio(update) > cfg.rebar.rho_col_max:
                    continue

                candidates.append((ast, -n_top, n_side, bar_size, update))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[:4])

    for ast, _, _, _, update in candidates:
        if ast >= required_ast:
            return update

    return candidates[-1][4]


def _beam_strengthening_update(cfg):
    next_bar = _next_value(sp.BEAM_BAR_SIZE, cfg.rebar.bar_sizes_beam)
    if next_bar is not None:
        return {
            "bar_size": next_bar,
            "n_top": sp.BEAM_TOP_BARS,
            "n_bot": sp.BEAM_BOT_BARS,
        }

    next_top = _next_count(sp.BEAM_TOP_BARS, cfg.rebar.beam_n_range)
    next_bot = _next_count(sp.BEAM_BOT_BARS, cfg.rebar.beam_n_range)
    if next_top is not None or next_bot is not None:
        return {
            "bar_size": sp.BEAM_BAR_SIZE,
            "n_top": next_top if next_top is not None else sp.BEAM_TOP_BARS,
            "n_bot": next_bot if next_bot is not None else sp.BEAM_BOT_BARS,
        }

    return None


def apply_pushover_failure_redesign(pushover_results, cfg):
    failed_tags = pushover_results["status"].get("failed_element_tags", [])
    classes = _failed_member_classes(failed_tags)
    before = reinforcement_snapshot()
    col_update = (
        _column_strengthening_update(cfg, prefer_bar_count=True)
        if "column" in classes
        else None
    )
    beam_update = _beam_strengthening_update(cfg) if "beam" in classes else None

    if col_update is None and beam_update is None:
        return {
            "changed": False,
            "reason": "No stronger longitudinal reinforcement option is available in the configured search space.",
            "failed_member_classes": sorted(classes),
            "before": before,
            "after": before,
        }

    apply_updates(col_update, beam_update)
    after = reinforcement_snapshot()

    return {
        "changed": before != after,
        "reason": "Pushover failed before the minimum acceptable roof drift; strengthened affected member class(es).",
        "failed_member_classes": sorted(classes),
        "column_update": col_update,
        "beam_update": beam_update,
        "before": before,
        "after": after,
    }


def _pushover_envelope_summary(pushover_results):
    return (
        pushover_results
        .get("design_envelope", {})
        .get("summary", {})
    )


def _pushover_target_dcr_snapshot(pushover_results):
    return (
        pushover_results
        .get("design_envelope", {})
        .get("target_dcr_snapshot")
    )


def _pushover_lateral_design_summary(pushover_results):
    target_snapshot = _pushover_target_dcr_snapshot(pushover_results)
    if (
        getattr(sp, "PUSHOVER_USE_TARGET_DCR_FOR_REDESIGN", True)
        and target_snapshot
        and target_snapshot.get("summary")
    ):
        return target_snapshot["summary"]

    return _pushover_envelope_summary(pushover_results)


def needs_pushover_envelope_redesign(pushover_results):
    if pushover_results["status"].get("failed"):
        return False

    summary = _pushover_lateral_design_summary(pushover_results)
    if not summary.get("tracked", False):
        # Target-drift snapshots do not carry a "tracked" flag.
        if summary.get("phase") != "pushover_target_roof_drift":
            return False

    return summary.get("all_design_checks_pass") is False


def _envelope_governing_dcr(summary):
    if not summary:
        return 0.0

    return max(
        summary.get("max_column_pm_dcr") or 0.0,
        summary.get("max_column_shear_dcr") or 0.0,
        summary.get("max_beam_flexure_dcr") or 0.0,
        summary.get("max_beam_shear_dcr") or 0.0,
    )


def pushover_envelope_redesign_stalled(redesign_history):
    if not getattr(sp, "PUSHOVER_ENVELOPE_STOP_ON_STALLED_DCR", True):
        return None

    records = []
    for record in redesign_history:
        summary = (
            record.get("pushover_target_dcr", {})
            if getattr(sp, "PUSHOVER_USE_TARGET_DCR_FOR_REDESIGN", True)
            else record.get("pushover_envelope", {})
        )
        if not summary:
            summary = record.get("pushover_envelope", {})
        if summary.get("all_design_checks_pass") is False:
            records.append(
                {
                    "attempt_index": record.get("attempt_index"),
                    "governing_dcr": _envelope_governing_dcr(summary),
                    "num_failed_elements": summary.get("num_failed_elements"),
                    "failed_element_tags": summary.get("failed_element_tags", []),
                }
            )

    min_reruns = max(
        1,
        int(getattr(sp, "PUSHOVER_ENVELOPE_STALL_MIN_RERUNS", 2)),
    )
    if len(records) <= min_reruns:
        return None

    current = records[-1]
    prior_best = min(
        records[:-1],
        key=lambda row: row["governing_dcr"],
    )
    tol = max(
        0.0,
        float(getattr(sp, "PUSHOVER_ENVELOPE_STALL_DCR_TOL", 0.01)),
    )

    if current["governing_dcr"] >= prior_best["governing_dcr"] * (1.0 - tol):
        return {
            "stalled": True,
            "reason": (
                "Pushover envelope redesign stopped because the governing "
                "DCR did not improve enough after additional reinforcement."
            ),
            "current": current,
            "best_previous": prior_best,
            "relative_improvement_tolerance": tol,
            "completed_envelope_reruns": len(records) - 1,
        }

    return None


def apply_pushover_envelope_redesign(pushover_results, cfg):
    summary = _pushover_lateral_design_summary(pushover_results)
    failed_tags = summary.get("failed_element_tags", [])
    classes = _failed_member_classes(failed_tags)
    before = reinforcement_snapshot()
    col_update = (
        _column_envelope_strengthening_update(cfg, summary)
        if "column" in classes
        else None
    )
    beam_update = _beam_strengthening_update(cfg) if "beam" in classes else None

    if col_update is None and beam_update is None:
        return {
            "changed": False,
            "reason": "Pushover envelope failed, but no stronger longitudinal reinforcement option is available in the configured search space.",
            "failed_member_classes": sorted(classes),
            "envelope_summary": summary,
            "before": before,
            "after": before,
        }

    apply_updates(col_update, beam_update)
    after = reinforcement_snapshot()

    return {
        "changed": before != after,
        "reason": "Pushover envelope design checks failed; strengthened affected member class(es).",
        "failed_member_classes": sorted(classes),
        "column_update": col_update,
        "beam_update": beam_update,
        "envelope_summary": summary,
        "before": before,
        "after": after,
    }


def _pushover_failure_roof_drift(pushover_results):
    status = pushover_results["status"]
    if status.get("failed_roof_disp") is not None:
        return status["failed_roof_disp"] / (sp.NUM_FLOOR * sp.STORY_H)
    if pushover_results["roof_drift"]:
        return pushover_results["roof_drift"][-1]
    return 0.0


def needs_pushover_failure_redesign(pushover_results):
    status = pushover_results["status"]
    if not status.get("failed"):
        return False
    return _pushover_failure_roof_drift(pushover_results) < sp.PUSHOVER_MIN_ACCEPTABLE_DRIFT_RATIO


def pushover_attempt_record(attempt_index, pushover_results):
    record = {
        "attempt_index": attempt_index,
        "reinforcement": reinforcement_snapshot(),
        "completed_steps": pushover_results["status"]["completed_steps"],
        "failed": pushover_results["status"]["failed"],
        "failed_step": pushover_results["status"]["failed_step"],
        "failed_roof_drift_ratio": (
            _pushover_failure_roof_drift(pushover_results)
            if pushover_results["status"]["failed"]
            else None
        ),
        "minimum_acceptable_roof_drift_ratio": sp.PUSHOVER_MIN_ACCEPTABLE_DRIFT_RATIO,
        "failed_element_tags": pushover_results["status"].get("failed_element_tags", []),
    }

    envelope_summary = _pushover_envelope_summary(pushover_results)
    if envelope_summary:
        record["pushover_envelope"] = {
            "tracked": envelope_summary.get("tracked", False),
            "all_design_checks_pass": envelope_summary.get("all_design_checks_pass"),
            "num_failed_elements": envelope_summary.get("num_failed_elements"),
            "failed_element_tags": envelope_summary.get("failed_element_tags", []),
            "num_failed_columns": envelope_summary.get("num_failed_columns"),
            "failed_column_tags": envelope_summary.get("failed_column_tags", []),
            "num_failed_beams": envelope_summary.get("num_failed_beams"),
            "failed_beam_tags": envelope_summary.get("failed_beam_tags", []),
            "max_column_pm_dcr": envelope_summary.get("max_column_pm_dcr"),
            "max_column_shear_dcr": envelope_summary.get("max_column_shear_dcr"),
            "max_beam_flexure_dcr": envelope_summary.get("max_beam_flexure_dcr"),
            "max_beam_shear_dcr": envelope_summary.get("max_beam_shear_dcr"),
        }

    target_snapshot = _pushover_target_dcr_snapshot(pushover_results)
    if target_snapshot and target_snapshot.get("summary"):
        target_summary = target_snapshot["summary"]
        record["pushover_target_dcr"] = {
            "phase": target_summary.get("phase"),
            "target_roof_drift_ratio": target_summary.get("target_roof_drift_ratio"),
            "target_roof_disp_in": target_summary.get("target_roof_disp_in"),
            "actual_roof_drift_ratio": target_summary.get("actual_roof_drift_ratio"),
            "actual_roof_disp_in": target_summary.get("actual_roof_disp_in"),
            "step": target_summary.get("step"),
            "all_design_checks_pass": target_summary.get("all_design_checks_pass"),
            "num_failed_elements": target_summary.get("num_failed_elements"),
            "failed_element_tags": target_summary.get("failed_element_tags", []),
            "num_failed_columns": target_summary.get("num_failed_columns"),
            "failed_column_tags": target_summary.get("failed_column_tags", []),
            "num_failed_beams": target_summary.get("num_failed_beams"),
            "failed_beam_tags": target_summary.get("failed_beam_tags", []),
            "max_column_pm_dcr": target_summary.get("max_column_pm_dcr"),
            "max_column_shear_dcr": target_summary.get("max_column_shear_dcr"),
            "max_beam_flexure_dcr": target_summary.get("max_beam_flexure_dcr"),
            "max_beam_shear_dcr": target_summary.get("max_beam_shear_dcr"),
        }

    return record


def pushover_log_path(outputs_dir, attempt_index):
    return outputs_dir / f"opensees_pushover_attempt_{attempt_index}.log"


def run_modal_and_pushover(outputs_dir, attempt_index):
    build_model()
    print("\nModel rebuilt with final steel for modal/pushover.")

    apply_gravity_loads()
    gravity_results = run_gravity_analysis()

    modal_results = run_modal_analysis()
    print_modal_results(modal_results)
    pushover_period_check = asce_period_check_from_modal_results(
        modal_results,
        f"post_gravity_tangent_pushover_attempt_{attempt_index}",
    )
    print_asce_period_check(pushover_period_check)
    with (outputs_dir / f"asce_period_check_pushover_attempt_{attempt_index}.json").open("w", encoding="utf-8") as file:
        json.dump(pushover_period_check, file, indent=2)

    apply_lateral_loads()
    pushover_results = run_pushover(log_path=pushover_log_path(outputs_dir, attempt_index))
    print_pushover_results(pushover_results, gravity_results)

    return gravity_results, modal_results, pushover_results


def selected_analysis_method():
    return (
        getattr(sp, "ANALYSIS_METHOD", "pushover")
        .strip()
        .lower()
        .replace("-", "_")
    )


def is_ntha_method(method):
    return method in {"ntha", "nltha", "time_history", "ground_motion"}


def is_pushover_method(method):
    return method in {"pushover", "static_pushover"}


def _safe_output_name(value):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def _ntha_story_drift_rows(results):
    rows = []
    for story, (dx, dy) in enumerate(
        zip(results["peak_story_drift_x"], results["peak_story_drift_y"]),
        start=1,
    ):
        resultant = (dx * dx + dy * dy) ** 0.5
        rows.append(
            {
                "story": story,
                "floor_pair": f"{story - 1}-{story}",
                "peak_drift_x_ratio": dx,
                "peak_drift_y_ratio": dy,
                "peak_drift_x_percent": 100.0 * dx,
                "peak_drift_y_percent": 100.0 * dy,
                "peak_drift_resultant_ratio": resultant,
                "peak_drift_resultant_percent": 100.0 * resultant,
            }
        )
    return rows


def summarize_ntha_results(results, output_dir, elapsed_sec, file_counts):
    drift_rows = _ntha_story_drift_rows(results)
    max_drift_x = max(drift_rows, key=lambda row: row["peak_drift_x_ratio"])
    max_drift_y = max(drift_rows, key=lambda row: row["peak_drift_y_ratio"])
    max_drift_resultant = max(
        drift_rows,
        key=lambda row: row["peak_drift_resultant_ratio"],
    )

    roof_x = results["roof_disp_x"]
    roof_y = results["roof_disp_y"]

    summary = {
        "analysis_method": "ntha",
        "status": results["status"],
        "record_summary_x": results["record_summary_x"],
        "record_summary_y": results["record_summary_y"],
        "damping_ratio": results["damping_ratio"],
        "rayleigh_a0": results["rayleigh_a0"],
        "rayleigh_a1": results["rayleigh_a1"],
        "elapsed_sec": elapsed_sec,
        "max_abs_roof_disp_x_in": max((abs(v) for v in roof_x), default=0.0),
        "max_abs_roof_disp_y_in": max((abs(v) for v in roof_y), default=0.0),
        "max_story_drift_x": max_drift_x,
        "max_story_drift_y": max_drift_y,
        "max_story_drift_resultant": max_drift_resultant,
        "output_dir": str(output_dir),
    }
    summary.update(file_counts)
    return summary


def select_ntha_records_from_config():
    scale_factor = getattr(sp, "NTHA_SCALE_FACTOR", None)

    record_id_x = getattr(sp, "NTHA_RECORD_ID_X", None)
    if record_id_x:
        record_x = load_ground_motion_record(
            record_id_x,
            scale_factor=scale_factor,
        )
        record_y = None
        record_id_y = getattr(sp, "NTHA_RECORD_ID_Y", None)
        if record_id_y and not getattr(sp, "NTHA_X_ONLY", False):
            record_y = load_ground_motion_record(
                record_id_y,
                scale_factor=scale_factor,
            )
        return f"manual:{record_x.record_id}", record_x, record_y

    result_id = getattr(sp, "NTHA_RESULT_ID", None)
    if result_id is not None:
        key, record_x, record_y = load_ground_motion_pair_by_result_id(
            result_id,
            set_name=getattr(sp, "NTHA_SET_NAME", "peer_mle_all"),
            split=getattr(sp, "NTHA_SPLIT", None),
            scale_factor=scale_factor,
        )
        if getattr(sp, "NTHA_X_ONLY", False):
            record_y = None
        return key, record_x, record_y

    pairs = load_ground_motion_pairs(
        set_name=getattr(sp, "NTHA_SET_NAME", "peer_mle_all"),
        split=getattr(sp, "NTHA_SPLIT", None),
        limit=1,
        scale_factor=scale_factor,
    )
    if not pairs:
        raise RuntimeError("No ground-motion pair found for NTHA.")

    key, record_x, record_y = pairs[0]
    if getattr(sp, "NTHA_X_ONLY", False):
        record_y = None
    return key, record_x, record_y


def print_ntha_results(results):
    status = results["status"]
    drift_rows = _ntha_story_drift_rows(results)
    max_drift = max(drift_rows, key=lambda row: row["peak_drift_resultant_ratio"])

    print("\nNTHA results:")
    print(
        f"  Completed {status['completed_steps']} / "
        f"{status['npts_requested']} transient steps"
    )
    print(f"  Failed: {status['failed']}")
    print(f"  Recovery steps: {len(status.get('convergence_log', []))}")
    print(
        f"  Max roof |Ux| = "
        f"{max((abs(v) for v in results['roof_disp_x']), default=0.0):.6e} in"
    )
    print(
        f"  Max roof |Uy| = "
        f"{max((abs(v) for v in results['roof_disp_y']), default=0.0):.6e} in"
    )
    print(
        "  Peak resultant story drift = "
        f"{max_drift['peak_drift_resultant_percent']:.3f}% "
        f"at story {max_drift['story']} "
        f"(floors {max_drift['floor_pair']})"
    )


def run_modal_and_ntha(outputs_dir):
    build_model()
    print("\nModel rebuilt with final steel for modal/NTHA.")

    reference_modal_results = run_modal_analysis()
    print("\nElastic/reference modal properties for Rayleigh damping:")
    print_modal_results(reference_modal_results)
    reference_period_check = asce_period_check_from_modal_results(
        reference_modal_results,
        "elastic_reference_before_gravity",
    )
    print_asce_period_check(reference_period_check)

    apply_gravity_loads()
    gravity_results = run_gravity_analysis()

    modal_results = run_modal_analysis()
    print("\nPost-gravity tangent modal properties (diagnostic only):")
    print_modal_results(modal_results)
    tangent_period_check = asce_period_check_from_modal_results(
        modal_results,
        "post_gravity_tangent_diagnostic",
    )
    print_asce_period_check(tangent_period_check)

    if getattr(sp, "NTHA_PRINT_CATALOG_SUMMARY", True):
        print("\nGround-motion catalog summary:")
        print(json.dumps(ground_motion_catalog_summary(), indent=2))

    pair_key, record_x, record_y = select_ntha_records_from_config()
    run_name = analysis_run_name(
        _safe_output_name(pair_key.replace("peer_result_id:", "peer_")),
        x_only=record_y is None,
        scale_factor=record_x.scale_factor,
        damping_ratio=getattr(sp, "NTHA_DAMPING_RATIO", 0.05),
        rayleigh_mode_i=getattr(sp, "NTHA_RAYLEIGH_MODE_I", 0),
        rayleigh_mode_j=getattr(sp, "NTHA_RAYLEIGH_MODE_J", 2),
        dt_factor=getattr(sp, "NTHA_DT_FACTOR", 1.0),
    )
    ntha_output_dir = outputs_dir / "ntha" / run_name
    validate_ntha_output_compatibility(ntha_output_dir)

    print(
        "\nRunning NTHA with "
        f"X={record_x.record_id}"
        f"{', Y=' + record_y.record_id if record_y else ', X-only'}"
    )
    print(f"NTHA output directory: {ntha_output_dir}")

    ntha_kwargs = {
        "record_y": record_y,
        "damping_ratio": getattr(sp, "NTHA_DAMPING_RATIO", 0.05),
        "modal_results": reference_modal_results,
        "rayleigh_mode_i": getattr(sp, "NTHA_RAYLEIGH_MODE_I", 0),
        "rayleigh_mode_j": getattr(sp, "NTHA_RAYLEIGH_MODE_J", 2),
        "dt_factor": getattr(sp, "NTHA_DT_FACTOR", 1.0),
        "log_path": ntha_output_dir / "opensees_ntha.log",
    }

    run_ntha_signature = inspect.signature(run_ntha)
    if "progress_path" in run_ntha_signature.parameters:
        ntha_kwargs["progress_path"] = ntha_output_dir / "progress.json"
    if "progress_every" in run_ntha_signature.parameters:
        ntha_kwargs["progress_every"] = getattr(sp, "NTHA_PROGRESS_EVERY", 100)

    start = time.perf_counter()
    ntha_results = run_ntha(record_x, **ntha_kwargs)
    elapsed = time.perf_counter() - start

    file_counts = save_ntha_outputs(
        ntha_output_dir,
        ntha_results,
        gravity_results,
        reference_modal_results,
    )
    with (ntha_output_dir / "modal_results_elastic_reference.json").open("w", encoding="utf-8") as file:
        json.dump(reference_modal_results, file, indent=2)
    with (ntha_output_dir / "modal_results_post_gravity_tangent.json").open("w", encoding="utf-8") as file:
        json.dump(modal_results, file, indent=2)
    reference_valid_modes = [mode for mode in reference_modal_results if mode.get("valid")]
    tangent_valid_modes = [mode for mode in modal_results if mode.get("valid")]
    modal_diagnostics = {
        "rayleigh_modal_source": "elastic_reference_before_gravity",
        "elastic_reference_period_mode_1_sec": reference_valid_modes[0]["period"] if len(reference_valid_modes) > 0 else None,
        "elastic_reference_period_mode_2_sec": reference_valid_modes[1]["period"] if len(reference_valid_modes) > 1 else None,
        "post_gravity_tangent_period_mode_1_sec": tangent_valid_modes[0]["period"] if len(tangent_valid_modes) > 0 else None,
        "post_gravity_tangent_period_mode_2_sec": tangent_valid_modes[1]["period"] if len(tangent_valid_modes) > 1 else None,
        "gravity_roof_ux_in": gravity_results.get("ux"),
        "gravity_roof_uy_in": gravity_results.get("uy"),
        "asce_reference_period_check": reference_period_check,
        "asce_post_gravity_period_check": tangent_period_check,
    }
    with (ntha_output_dir / "modal_diagnostics.json").open("w", encoding="utf-8") as file:
        json.dump(modal_diagnostics, file, indent=2)

    summary = summarize_ntha_results(
        ntha_results,
        ntha_output_dir,
        elapsed,
        file_counts,
    )
    summary.update(modal_diagnostics)
    with (ntha_output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print_ntha_results(ntha_results)
    print(f"Saved NTHA summary: {ntha_output_dir / 'summary.json'}")

    return gravity_results, reference_modal_results, ntha_results, summary


def main():
    t_start = time.perf_counter()
    print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    apply_geometry_from_environment()

    project_dir = Path(__file__).resolve().parent
    outputs_dir = project_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    clear_failure_artifacts(outputs_dir)
    current_model_summary = model_summary()
    print_model_summary(current_model_summary)
    with (outputs_dir / "model_summary.json").open("w") as file:
        json.dump(current_model_summary, file, indent=2)
    with (outputs_dir / "asce_period_estimate.json").open("w", encoding="utf-8") as file:
        json.dump(current_model_summary["asce_period_estimate"], file, indent=2)

    # ── Iterative steel redesign (gravity-driven inner loop) ─────────────────
    cfg = DesignConfig.from_structure_parameters()
    redesign_ok, n_iter, gravity_results, design_results = run_iterative_redesign(
        cfg=cfg, max_iter=10, verbose=True
    )
    if not redesign_ok:
        print(f"\n[WARNING] Steel redesign did not converge in {n_iter} iterations. "
              "Proceeding with last iteration's design.")
    print_gravity_results(gravity_results)

    # Design check after convergence
    col_fail, beam_fail = print_summary(design_results, verbose=False)
    gravity_design_status = summarize_design_results(
        design_results,
        phase="gravity",
        redesign_ok=redesign_ok,
        n_iter=n_iter,
        notes=[
            "Demand snapshot taken from the gravity-analysis state after iterative redesign.",
            "Shear DCRs are included in pass/fail status, but the current redesign loop only changes longitudinal bars.",
        ],
    )
    write_design_results(outputs_dir / "design_check_gravity.json", design_results)
    write_design_results(outputs_dir / "design_check.json", design_results)
    with (outputs_dir / "design_status_gravity.json").open("w") as file:
        json.dump(gravity_design_status, file, indent=2)

    if not gravity_design_status["all_design_checks_pass"]:
        print(
            "[WARNING] Gravity design checks still fail: "
            f"{gravity_design_status['num_failed_elements']} element(s), "
            f"max column shear DCR={gravity_design_status['max_column_shear_dcr']:.3f}, "
            f"max column P-M DCR={gravity_design_status['max_column_pm_dcr']:.3f}, "
            f"max beam flexure DCR={gravity_design_status['max_beam_flexure_dcr']:.3f}."
        )

    # ── One-time modal + pushover after steel has converged ──────────────────

    analysis_method = selected_analysis_method()
    print(f"\nSelected analysis method: {analysis_method}")

    if is_ntha_method(analysis_method):
        gravity_results, modal_results, ntha_results, ntha_summary = run_modal_and_ntha(
            outputs_dir
        )
        analysis_status = {
            "analysis_method": "ntha",
            "gravity_design_status": gravity_design_status,
            "ntha_summary": ntha_summary,
        }
        with (outputs_dir / "analysis_status.json").open("w", encoding="utf-8") as file:
            json.dump(analysis_status, file, indent=2)

        ops.wipe()
        elapsed = time.perf_counter() - t_start
        print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        return

    if not is_pushover_method(analysis_method):
        raise ValueError(
            "Unsupported ANALYSIS_METHOD. Use 'ntha' or 'pushover', "
            f"got {analysis_method!r}."
        )

    # rebuild model with final steel configuration (run_iterative_redesign leaves
    # the OpenSees domain in a post-gravity state; rebuild cleanly for pushover)
    build_model()
    print("\nModel rebuilt with final steel for modal/pushover.")

    apply_gravity_loads()
    gravity_results = run_gravity_analysis()

    modal_results = run_modal_analysis()
    print_modal_results(modal_results)
    initial_pushover_period_check = asce_period_check_from_modal_results(
        modal_results,
        "post_gravity_tangent_pushover_initial",
    )
    print_asce_period_check(initial_pushover_period_check)
    with (outputs_dir / "asce_period_check_pushover_initial.json").open("w", encoding="utf-8") as file:
        json.dump(initial_pushover_period_check, file, indent=2)

    apply_lateral_loads()
    pushover_results = run_pushover(log_path=pushover_log_path(outputs_dir, 0))
    print_pushover_results(pushover_results, gravity_results)
    pushover_redesign_history = [pushover_attempt_record(0, pushover_results)]
    failure_redesign_attempts = 0
    envelope_redesign_attempts = 0

    while True:
        if needs_pushover_failure_redesign(pushover_results):
            if failure_redesign_attempts >= sp.PUSHOVER_REDESIGN_MAX_ATTEMPTS:
                print(
                    "[WARNING] Pushover still failed before "
                    f"{100.0 * sp.PUSHOVER_MIN_ACCEPTABLE_DRIFT_RATIO:.1f}% roof drift "
                    f"after {sp.PUSHOVER_REDESIGN_MAX_ATTEMPTS} failure redesign attempt(s)."
                )
                break

            redesign_update = apply_pushover_failure_redesign(pushover_results, cfg)
            pushover_redesign_history[-1]["failure_redesign_update"] = redesign_update
            retry_reason = "pushover-failure redesign"
            failure_redesign_attempts += 1

        elif needs_pushover_envelope_redesign(pushover_results):
            stalled = pushover_envelope_redesign_stalled(pushover_redesign_history)
            if stalled:
                pushover_redesign_history[-1]["envelope_redesign_stalled"] = stalled
                print(
                    "[WARNING] "
                    f"{stalled['reason']} "
                    f"Current DCR={stalled['current']['governing_dcr']:.3f}; "
                    f"best previous DCR={stalled['best_previous']['governing_dcr']:.3f}."
                )
                break

            if envelope_redesign_attempts >= sp.PUSHOVER_ENVELOPE_REDESIGN_MAX_ATTEMPTS:
                print(
                    "[WARNING] Pushover envelope checks still fail "
                    f"after {sp.PUSHOVER_ENVELOPE_REDESIGN_MAX_ATTEMPTS} envelope redesign attempt(s)."
                )
                break

            redesign_update = apply_pushover_envelope_redesign(pushover_results, cfg)
            pushover_redesign_history[-1]["envelope_redesign_update"] = redesign_update
            retry_reason = "pushover-envelope redesign"
            envelope_redesign_attempts += 1

        else:
            break

        if not redesign_update["changed"]:
            print(f"[WARNING] {redesign_update['reason']}")
            break

        attempt_index = pushover_redesign_history[-1]["attempt_index"] + 1
        print(
            f"\nRetrying modal/pushover after {retry_reason} "
            f"(attempt {attempt_index + 1})."
        )
        gravity_results, modal_results, pushover_results = run_modal_and_pushover(
            outputs_dir,
            attempt_index,
        )
        pushover_redesign_history.append(
            pushover_attempt_record(attempt_index, pushover_results)
        )
    (
        pushover_final_design_status,
        pushover_final_design_results,
    ) = run_final_pushover_design_check(
        outputs_dir,
        pushover_failed=pushover_results["status"]["failed"],
    )
    (
        pushover_envelope_design_status,
        pushover_envelope_design_results,
    ) = run_pushover_envelope_design_check(outputs_dir, pushover_results)
    design_status = combine_design_statuses(
        gravity_design_status,
        pushover_final_design_status,
        pushover_envelope_status=pushover_envelope_design_status,
        pushover_redesign_history=pushover_redesign_history,
    )

    with (outputs_dir / "design_status.json").open("w") as file:
        json.dump(design_status, file, indent=2)

    if not pushover_results["status"]["failed"]:
        clear_failure_artifacts(outputs_dir)

    with (outputs_dir / "pushover_status.json").open("w") as file:
        json.dump(pushover_results["status"], file, indent=2)

    with (outputs_dir / "mechanism_diagnostics.json").open("w") as file:
        json.dump(pushover_results["mechanism_diagnostics"], file, indent=2)

    csv_path = save_pushover_curve(
        pushover_results,
        outputs_dir / "pushover_curve.csv",
    )
    fig, _ = plot_pushover_curve(pushover_results, show=False)
    fig.savefig(outputs_dir / "pushover_curve.png", dpi=200)
    drift_fig, _ = plot_roof_drift_curve(pushover_results, show=False)
    drift_fig.savefig(outputs_dir / "pushover_drift_curve.png", dpi=200)

    design_check_export = {
        "gravity": design_results,
        "pushover_final": pushover_final_design_results,
    }
    if pushover_envelope_design_results is not None:
        lateral_check_name = (
            "pushover_2pct_drift"
            if (
                pushover_envelope_design_status
                and pushover_envelope_design_status.get("phase")
                == "pushover_target_roof_drift"
            )
            else "pushover_envelope"
        )
        design_check_export[lateral_check_name] = pushover_envelope_design_results

    sample_summary = save_analysis_sample(
        outputs_dir / "samples" / "baseline_frame",
        gravity_results,
        modal_results,
        pushover_results,
        design_status=design_status,
        design_check_results=design_check_export,
    )

    print(f"\nSaved pushover curve CSV: {csv_path}")
    print(f"Saved pushover curve plot: {outputs_dir / 'pushover_curve.png'}")
    print(f"Saved pushover drift plot: {outputs_dir / 'pushover_drift_curve.png'}")
    print(
        "Saved graph sample: "
        f"{outputs_dir / 'samples' / 'baseline_frame'} "
        f"({sample_summary['num_nodes']} nodes, "
        f"{sample_summary['num_elements']} elements, "
        f"{sample_summary['num_element_end_force_rows']} element-end force rows)"
    )

    if should_show_plots():
        plot_pushover_curve(pushover_results, show=True)
        plot_roof_drift_curve(pushover_results, show=True)

    ops.wipe()

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
