"""Re-run one generated case with recorders on its IMK hinges, and plot the
moment-rotation hysteresis.

Generated NTHA output only stores force/rotation ENVELOPES, never the full
moment-rotation history, so proving the IMK hinge model behaves correctly
(elastic slope matches the spring stiffness Kh, yield at My, capping at
Mc = FmaxFy*My reached at the native rotation My/Kh + dp, post-capping
degradation) requires re-running the case with element recorders attached.

This rebuilds the model exactly the way Ground_Motion_Main.py does
(build_gravity_modal_state() then run_ntha()) so the model is identical to
the one that produced the dataset, attaches recorders to the requested
hinges, runs the analysis, and plots the result.

Every loop is drawn against its OWN installed spring, read from the hinge
registry of the rebuilt model per member end, bending axis and native sign
(Model/IMK_Spring_Mapping.spring_points): native yield My/Kh, native capping
My/Kh + dp and, under the frozen IMK V2 calibration (2026-10-29), the
member-reference yield My/K0 for comparison. The plotted rotation is the native
zeroLength spring rotation in milliradians: not a story drift and not a member
chord rotation. The native du is a provisional numerical cutoff, not a collapse
rotation. The rerun installs the CURRENT member spring calibration; a case
generated under another one is reported, never silently mixed.

Important: hinges only exist where Structure_Parameters enables them.
With IMK_APPLY_TO_COLUMNS = False, column elements have no hinge at all and
their recorders will produce a file containing only a time column -- point
this at beam elements in that configuration.

Usage
-----
    python Record_Hinge_Hysteresis.py \\
        --case-dir "<root>/cases/case_3331" --set-name peer_expansion_30k \\
        --element 266 --element 262

Geometry and result_id are read from the case's own automation_summary.json,
so the rerun matches the original generation run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


RC_DIR = Path(
    os.environ.get("RC_STRUCTURE_DIR", Path(__file__).resolve().parents[1])
).resolve()
if str(RC_DIR) not in sys.path:
    sys.path.insert(0, str(RC_DIR))

import numpy as np  # noqa: E402
import openseespy.opensees as ops  # noqa: E402

from Analysis.NTHA import run_ntha  # noqa: E402
from Geometry_Overrides import apply_geometry_overrides  # noqa: E402
from Design.Design_Driver import DESIGN_ARTIFACT_NAME, load_or_create_design  # noqa: E402
from Ground_Motion_Main import build_gravity_modal_state  # noqa: E402
from Loads.Ground_Motion import load_ground_motion_pair_by_result_id  # noqa: E402
from Model import IMK_Spring_Mapping as spring_mapping  # noqa: E402
from Model.IMK_Hinges import hinge_element_tag, installed_spring_points  # noqa: E402

# zeroLength material index -> recorder label; the registry's spring axis of each label.
MATERIAL_DIRECTIONS = ((1, "roty"), (2, "rotz"))
SPRING_AXIS = {"roty": "y", "rotz": "z"}
END_NAMES = {1: "i", 2: "j"}


def read_case_settings(case_dir: Path):
    """Recover geometry overrides, result_id and scale factor from a case.

    The scale factor matters: rerunning a scale-3 case at 1.0 reproduces the
    geometry but not the demand, so the hinges stay near-elastic and the plot
    shows none of the behaviour it exists to verify.
    """
    summary_path = case_dir / "dataset" / "automation_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"No automation_summary.json under {case_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    overrides = summary.get("geometry_overrides") or {}
    runs = summary.get("ntha_runs") or []
    if not overrides or not runs:
        raise SystemExit(f"{summary_path} is missing geometry_overrides or ntha_runs.")
    scale_factor = float(runs[0].get("scale_factor") or 1.0)
    return overrides, int(runs[0]["result_id"]), runs[0]["run_name"], scale_factor


def attach_recorders(output_dir: Path, element_tags):
    for ele_tag in element_tags:
        for end_id in (1, 2):
            hinge_ele = hinge_element_tag(ele_tag, end_id)
            for material_index, label in MATERIAL_DIRECTIONS:
                path = output_dir / f"hinge_{ele_tag}_end{end_id}_{label}.out"
                ops.recorder(
                    "Element", "-file", str(path), "-time",
                    "-ele", hinge_ele, "material", material_index, "stressStrain",
                )
            print(f"  recorders: element {ele_tag} end {end_id} -> hinge {hinge_ele}")


def case_member_spring_calibration(case_dir: Path, run_name: str):
    """(calibration, basis): the member spring calibration the case's own NTHA run was generated under.

    Read from the run's global_parameters.json. A run that predates the key was generated before the frozen
    IMK V2 calibration (2026-10-29), under the pre-freeze rule the legacy replay reproduces. (None, basis) when
    the run wrote no global_parameters.json.
    """
    path = case_dir / "ntha" / run_name / "global_parameters.json"
    if not path.exists():
        return None, f"no {path.name} for run {run_name}"
    recorded = json.loads(path.read_text(encoding="utf-8")).get("imk_member_spring_calibration")
    if recorded is None:
        return spring_mapping.LEGACY_REPLAY, "key absent: the run predates the frozen IMK V2 calibration (2026-10-29)"
    return recorded, "recorded in the run's global_parameters.json"


def backbone_reference(ele_tag: int, end: str, axis: str):
    """The installed spring the recorded loop should agree with: spring_points of (element, end, axis).

    Read from the hinge registry of the model this rerun built, so it is the material that produced the loop,
    per native sign (beam ends are asymmetric) and at the column's own axial force. It replaces the earlier
    references, which took Ke from the nominal member theta_y of hinge_backbone.csv (not the spring stiffness)
    or the global IMK_THETA_* constants (not the calibrated backbone), and put capping at that theta_y + the
    member theta_p (not the native My/Kh + dp). None for a released (elastic) end or a member without hinges.
    """
    return installed_spring_points(ele_tag, end, axis)


def installed_spring_references(element_tags):
    """backbone_reference of every recorded spring, keyed by its recorder file stem."""
    return {f"hinge_{ele_tag}_end{end_id}_{label}": backbone_reference(ele_tag, END_NAMES[end_id], SPRING_AXIS[label])
            for ele_tag in element_tags for end_id in (1, 2) for _index, label in MATERIAL_DIRECTIONS}


def describe_references(references):
    """One line per recorded spring and native sign: the four rotations consumers must keep apart."""
    for stem, points in references.items():
        if points is None:
            print(f"  {stem}: no installed IMK spring (released end or member without hinges)")
            continue
        for sign in ("positive", "negative"):
            branch = points[sign]
            line = (f"  {stem} {sign} ({branch.get('physical_direction')}): My {branch['My_kip_in']:.1f} kip-in, "
                    f"Kh {points['Kh_kip_in_per_rad']:.3e} kip-in/rad, native My/Kh {branch['native_theta_y_rad']:.6f}, "
                    f"native capping My/Kh + dp {branch['native_theta_cap_rad']:.5f}, "
                    f"native du {branch['native_du_rad']:.4f} (numerical cutoff)")
            if branch.get("member_theta_y_reference_rad") is not None:
                line += (f"; member reference My/K0 {branch['member_theta_y_reference_rad']:.5f}, "
                         f"capping {branch['member_theta_cap_reference_rad']:.5f}")
            print(line)


def plot_recordings(output_dir: Path, references: dict, figure_path: Path, calibration=None,
                    case_calibration=None, case_basis=None):
    """Plot every recorder this rerun wrote that captured a real (non-degenerate) response against its own
    installed spring.

    ``references`` maps each recorder stem this rerun attached to backbone_reference (None: a released or elastic
    end, no reference lines); a hinge_*.out left in output_dir by an earlier rerun is not in it and is skipped.
    Rotation is the native zeroLength spring rotation in mrad; the ductility is per native yield My/Kh of each
    sign. When ``case_calibration`` differs from ``calibration`` the figure says so, so it cannot pass for the
    dataset's model.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    active = []
    for path in sorted(output_dir.glob("hinge_*.out")):
        if path.stem not in references:
            print(f"  {path.name}: not recorded by this rerun (earlier file) -- skipped")
            continue
        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] < 3:
            print(f"  {path.name}: no material response recorded (hinge absent) -- skipped")
            continue
        moment, rotation = data[:, 1], data[:, 2]
        if np.abs(rotation).max() < 1e-12:
            print(f"  {path.name}: rotation is identically zero (constrained DOF) -- skipped")
            continue
        active.append((path.stem, moment, rotation))

    if not active:
        print("No hinge produced a usable moment-rotation history; nothing to plot.")
        return None

    active.sort(key=lambda item: -np.abs(item[2]).max())
    active = active[:4]

    columns = min(2, len(active))
    rows = (len(active) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(6.5 * columns, 5.0 * rows), squeeze=False
    )

    for axis, (name, moment, rotation) in zip(axes.flat, active):
        axis.plot(rotation * 1000.0, moment, lw=0.6, color="tab:blue", alpha=0.85)
        points = references.get(name)
        if points is None:
            title = f"{name}\nno installed IMK spring (released or elastic end): no backbone reference"
        else:
            positive, negative = points["positive"], points["negative"]
            span = np.array([-negative["native_theta_y_rad"], positive["native_theta_y_rad"]]) * 1.4
            axis.plot(span * 1000.0, span * points["Kh_kip_in_per_rad"], "k--", lw=1.2, label="elastic Kh")
            # Each native sign on its own branch: beam ends are asymmetric.
            for sign, branch in ((1.0, positive), (-1.0, negative)):
                first = sign > 0
                axis.axhline(sign * branch["My_kip_in"], color="tab:red", ls=":", lw=1, label="My" if first else None)
                axis.axhline(sign * branch["Mc_kip_in"], color="tab:orange", ls=":", lw=1,
                             label="Mc = FmaxFy My" if first else None)
                axis.axvline(sign * branch["native_theta_cap_rad"] * 1000.0, color="tab:green", ls="-.", lw=1,
                             label="native capping My/Kh + dp" if first else None)
            demand = spring_mapping.rotation_demand(points, float(rotation.max()), float(rotation.min()))
            peak = {sign: demand["by_sign"][sign]["peak_rotation_rad"] for sign in ("positive", "negative")}
            cutoff = ("reached the native du (numerical cutoff, not collapse)" if demand["past_native_cutoff"]
                      else "below the native du cutoff")
            title = (f"{name}\npeak spring rotation +{peak['positive'] * 1000:.2f} / -{peak['negative'] * 1000:.2f} mrad"
                     f" = {peak['positive'] / positive['native_theta_y_rad']:.1f} / "
                     f"{peak['negative'] / negative['native_theta_y_rad']:.1f} x native My/Kh\n"
                     f"capping ratio {demand['capping_ratio']:.2f} ({demand['governing_sign']}; 1 = native capping); "
                     f"{cutoff}")
            axis.legend(fontsize=7, loc="lower right")
        axis.set_xlabel("Native spring rotation (mrad), not story drift")
        axis.set_ylabel("Spring moment (kip-in)")
        axis.set_title(title, fontsize=8)
        axis.grid(alpha=0.3)

    for axis in axes.flat[len(active):]:
        axis.axis("off")

    if calibration is not None:
        suptitle = f"Member spring calibration installed by this rerun: {calibration}"
        top = 0.97
        if case_calibration != calibration:
            generated = (f"case generated under {case_calibration}" if case_calibration is not None
                         else "case's member spring calibration unknown")
            suptitle += f"\n{generated} ({case_basis}): the loops are the rerun's model, not the dataset's"
            top = 0.95
        figure.suptitle(suptitle, fontsize=9)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, top))
    else:
        figure.tight_layout()
    figure.savefig(figure_path, dpi=150)
    print(f"\nWrote {figure_path}")
    return figure_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, help="Generated case directory.")
    parser.add_argument(
        "--set-name",
        required=True,
        help="Record set the case was generated with (e.g. peer_mle_all).",
    )
    parser.add_argument(
        "--element",
        action="append",
        type=int,
        required=True,
        help="Spine element tag whose hinges to record; repeat for several.",
    )
    parser.add_argument(
        "--member-type",
        default="beam",
        help="Unused since the frozen IMK V2 calibration (2026-10-29): every reference is read from the "
             "installed spring itself. Kept so existing commands still parse.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--damping-ratio", type=float, default=0.05)
    parser.add_argument("--rayleigh-mode-i", type=int, default=0)
    parser.add_argument("--rayleigh-mode-j", type=int, default=2)
    parser.add_argument("--dt-factor", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    case_dir = Path(args.case_dir).resolve()
    overrides, result_id, run_name, scale_factor = read_case_settings(case_dir)

    output_dir = Path(
        args.output_dir or RC_DIR / "outputs" / f"hinge_hysteresis_{case_dir.name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    apply_geometry_overrides(overrides, variant_name=f"{case_dir.name}_hysteresis")

    # Ground_Motion_Main.main() loads the case design before building, and
    # skipping it here silently built the model from Structure_Parameters
    # defaults instead of this case's sections and reinforcement. The hinge
    # that produced the recorded loop was then a different hinge entirely:
    # for case_0008 the rerun gave theta_y = 0.000207 rad against the 0.004
    # the case recorded (its nominal member theta_y column), and rotations 100x
    # too small. The plot looked like a near-elastic hinge and said nothing
    # about the real model.
    design_path = case_dir / DESIGN_ARTIFACT_NAME
    if not design_path.exists():
        raise SystemExit(
            f"No {DESIGN_ARTIFACT_NAME} in {case_dir}; the rerun cannot "
            "reproduce the generated model without it."
        )
    design_record, _created = load_or_create_design(design_path, verbose=False)
    sections = design_record["sections"]
    print(
        "Loaded case design: column %gx%g @%g ksi, beam %gx%g @%g ksi"
        % (
            sections["b_col_in"], sections["h_col_in"], sections["fc_col_ksi"],
            sections["b_beam_in"], sections["h_beam_in"], sections["fc_beam_ksi"],
        ),
        flush=True,
    )

    # The rerun installs the CURRENT member spring calibration. A case generated under another one is a
    # different model (frozen IMK V2 calibration, 2026-10-29): say so rather than present the loop as the case's.
    calibration = spring_mapping.active_calibration()
    case_calibration, basis = case_member_spring_calibration(case_dir, run_name)
    if case_calibration != calibration:
        print(
            f"WARNING: the case's run was generated under member spring calibration {case_calibration!r} "
            f"({basis}); this rerun installs {calibration!r}. The loops and their reference lines are the "
            "rerun's model, not the dataset's.",
            flush=True,
        )

    _, reference_modal_results, *_ = build_gravity_modal_state()
    print(f"Gravity+modal done at {time.perf_counter() - start:.0f}s", flush=True)

    attach_recorders(output_dir, args.element)
    # The installed springs, read now from the registry of the model just built.
    references = installed_spring_references(args.element)

    _, record_x, record_y = load_ground_motion_pair_by_result_id(
        result_id, set_name=args.set_name, scale_factor=scale_factor
    )
    print(
        f"Loaded records: {record_x.record_id} / {record_y.record_id} "
        f"at scale {scale_factor:g}",
        flush=True,
    )

    results = run_ntha(
        record_x,
        record_y=record_y,
        damping_ratio=args.damping_ratio,
        modal_results=reference_modal_results,
        rayleigh_mode_i=args.rayleigh_mode_i,
        rayleigh_mode_j=args.rayleigh_mode_j,
        dt_factor=args.dt_factor,
        log_path=output_dir / "opensees_ntha.log",
    )
    ops.wipe()

    status = results["status"]
    print(
        f"NTHA done: {status['completed_steps']}/{status['npts_requested']} steps, "
        f"failed={status['failed']}, elapsed={time.perf_counter() - start:.0f}s",
        flush=True,
    )

    print(f"\nInstalled spring references ({calibration}; native spring rotations, rad):")
    describe_references(references)
    plot_recordings(output_dir, references, output_dir / "hysteresis_plot.png", calibration=calibration,
                    case_calibration=case_calibration, case_basis=basis)


if __name__ == "__main__":
    main()
