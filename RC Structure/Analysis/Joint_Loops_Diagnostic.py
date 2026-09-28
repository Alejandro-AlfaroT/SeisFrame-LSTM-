"""Full-rate loops of the joint pinching springs in a fixed-design ground-motion diagnostic (2026-09-27).

The production ground-motion outputs keep the joint springs' envelopes (``joint_springs.csv``) and a
coarse force array; this module adds, for the diagnostic only and behind ``--joint-loops``, what the
hinge diagnostic keeps for the member hinges: one text recorder per plane over every joint spring
element at the analysis rate, an evaluation row per spring against its installed law, the energy-ranked
loop gallery and a compressed history file. Nothing here is written by the generation scheduler, so a
dataset run stays as light as before.

Planes: the joint element is ``zeroLength ... -mat m4 m5 -dir 4 5`` (Model/Joint_Springs), so recorder
material index 1 is direction 4 (the y-z plane, the springs the y beams load, axis "y") and index 2 is
direction 5 (the x-z plane, axis "x").
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openseespy.opensees as ops

from Analysis import Hinge_Hysteresis_Diagnostic as hd
from Model.Joint_Springs import joint_registry

MATERIAL_INDEX = {"y": 1, "x": 2}          # recorder material index per plane axis
YIELD_TOLERANCE = 1.02                     # |rotation| beyond theta_y by this factor counts as yielded


def attach_joint_recorders(output_dir):
    """One recorder per plane over every joint spring element, in ascending element order."""
    registry = joint_registry()
    if not registry:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = sorted(registry.values(), key=lambda e: int(e["element"]))
    tags = [int(e["element"]) for e in entries]
    domain = sorted(int(t) for t in ops.getEleTags() if int(t) in set(tags))
    if domain != tags:
        raise ValueError(f"joint recorder coverage: {len(set(tags) - set(domain))} registry springs not in the domain")
    files = {}
    for axis, index in MATERIAL_INDEX.items():
        path = output_dir / f"joint_spring_{axis}_material_{index}_stressStrain.out"
        ops.recorder("Element", "-file", str(path), "-time", "-precision", hd.RECORDER_PRECISION,
                     "-ele", *tags, "material", index, "stressStrain")
        files[axis] = str(path)
    joints = []
    for e in entries:
        planes = {}
        for axis, plane in e["planes"].items():
            planes[axis] = {k: plane.get(k) for k in ("category_id", "joint_class", "gamma", "aj_in2", "vn_kip", "mn_kip_in",
                                                        "ke_kip_in_per_rad", "theta_y_rad", "a_rad", "b_rad", "c_residual",
                                                        "kappa_f", "kappa_d", "axial_ratio", "material_tag", "direction")}
        joints.append({"element": int(e["element"]), "node": int(e["node"]), "beam_core": int(e["beam_core"]),
                       "floor": e["floor"], "grid_i": e["grid_i"], "grid_j": e["grid_j"], "kind": e["kind"], "level": e["level"],
                       "planes": planes})
    return {"files": files, "hinge_tag_order": tags, "joint_element_order": tags, "joints": joints,
            "material_index": dict(MATERIAL_INDEX), "precision_digits": hd.RECORDER_PRECISION,
            "columns": "time, then (moment kip-in, rotation rad) per joint spring in joint_element_order",
            "scope": "diagnostic only; the production outputs keep the joint envelope and a coarse force array"}


def read_joint_recorders(coverage, time_limit=None):
    """Same layout as the hinge reader: {axis: {"time", "moment", "rotation", "rows", ...}, "hinge_tag_order"}."""
    return hd.read_hinge_recorders(coverage, time_limit=time_limit)


def evaluate_joint_histories(recorded, coverage):
    """Per joint spring and plane: extrema, yield call against the installed theta_y, energy, damage against a."""
    index_of = {int(tag): k for k, tag in enumerate(recorded["hinge_tag_order"])}
    rows = []
    for joint in coverage["joints"]:
        k = index_of[joint["element"]]
        for axis, plane in joint["planes"].items():
            block = recorded.get(axis)
            base = {"element": joint["element"], "node": joint["node"], "floor": joint["floor"], "grid_i": joint["grid_i"],
                    "grid_j": joint["grid_j"], "kind": joint["kind"], "level": joint["level"], "spring_axis": axis,
                    "joint_class": plane["joint_class"], "gamma": plane["gamma"], "vn_kip": plane["vn_kip"],
                    "fy_positive_kip_in": plane["mn_kip_in"], "fy_negative_kip_in": plane["mn_kip_in"],
                    "ke_kip_in_per_rad": plane["ke_kip_in_per_rad"], "theta_y": plane["theta_y_rad"],
                    "a_rad": plane["a_rad"], "b_rad": plane["b_rad"], "c_residual": plane["c_residual"],
                    "kappa_f": plane["kappa_f"], "kappa_d": plane["kappa_d"], "axial_ratio": plane["axial_ratio"]}
            if block is None or not block["rows"]:
                rows.append({**base, "response_missing": True})
                continue
            rotation, moment = block["rotation"][:, k], block["moment"][:, k]
            max_rot, min_rot = float(rotation.max()), float(rotation.min())
            theta_y = float(plane["theta_y_rad"])
            peak = max(0.0, max(max_rot, -min_rot) - theta_y)
            dissipated, work = hd._loop_energy(moment, rotation, float(plane["ke_kip_in_per_rad"]))
            rows.append({**base, "response_missing": False,
                         "rotation_max": max_rot, "rotation_min": min_rot,
                         "moment_max_kip_in": float(moment.max()), "moment_min_kip_in": float(moment.min()),
                         "rotation_over_theta_y": max(max_rot, -min_rot) / theta_y if theta_y > 0 else None,
                         "moment_over_mn": max(abs(float(moment.max())), abs(float(moment.min()))) / float(plane["mn_kip_in"]),
                         "yielded": bool(max(max_rot, -min_rot) > YIELD_TOLERANCE * theta_y),
                         "plastic_rotation_peak": peak, "damage_ratio": (peak / plane["a_rad"]) if plane["a_rad"] else None,
                         "past_capping": bool(peak >= plane["a_rad"]) if plane["a_rad"] else False,
                         "dissipated_energy_kip_in": dissipated, "work_integral_kip_in": work, "samples": int(len(rotation))})
    return rows


def joint_summary(rows):
    present = [r for r in rows if not r.get("response_missing")]
    yielded = [r for r in present if r["yielded"]]
    return {"springs": len(rows), "springs_with_response": len(present), "yielded_springs": len(yielded),
            "max_rotation_over_theta_y": max((r["rotation_over_theta_y"] or 0.0 for r in present), default=0.0),
            "max_moment_over_mn": max((r["moment_over_mn"] for r in present), default=0.0),
            "max_dissipated_energy_kip_in": max((r["dissipated_energy_kip_in"] for r in present), default=0.0),
            "past_capping": sum(1 for r in present if r["past_capping"]),
            "basis": "yielded = |rotation| beyond 1.02 theta_y of the installed law; energy = work integral less the "
                     "elastic change on the installed Ke (Hinge_Hysteresis_Diagnostic._loop_energy)"}


def _title(r, detail=True):
    head = (f"joint {r['node']} floor {r['floor']} ({r['kind']}, {r['level']}), spring {r['spring_axis']} "
            f"[{r['joint_class']}, gamma {r['gamma']:g}]")
    if not detail:
        return head
    return head + "\n" + (f"rotation {100 * r['rotation_min']:+.3f}% to {100 * r['rotation_max']:+.3f}%  "
                          f"({r['rotation_over_theta_y']:.2f} theta_y); moment {r['moment_over_mn']:.2f} Mn; "
                          f"energy {r['dissipated_energy_kip_in']:.1f} kip-in; capping at {100 * (r['theta_y'] + r['a_rad']):.2f}%")


def plot_joint_loops(recorded, rows, output_dir, limit=24):
    """Per-spring PNGs and a gallery of the springs ranked by dissipated energy, then by rotation over theta_y."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_of = {int(tag): k for k, tag in enumerate(recorded["hinge_tag_order"])}
    present = [r for r in rows if not r.get("response_missing")]
    present.sort(key=lambda r: (-r["dissipated_energy_kip_in"], -(r["rotation_over_theta_y"] or 0.0)))
    selected = present[:limit]
    written = []
    for r in selected:
        k = index_of[r["element"]]
        block = recorded[r["spring_axis"]]
        figure, axis = plt.subplots(figsize=(7.5, 6.0))
        hd._loop_axis(axis, r, block["rotation"][:, k], block["moment"][:, k], _title(r), fontsize=10)
        figure.tight_layout()
        path = output_dir / f"joint_loop_{r['node']}_{r['spring_axis']}.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(str(path))
    if selected:
        yielded = sum(1 for r in present if r["yielded"])
        cols = min(3, len(selected))
        nrows = (len(selected) + cols - 1) // cols
        figure, axes = plt.subplots(nrows, cols, figsize=(5.6 * cols, 4.6 * nrows), squeeze=False)
        for ax, r in zip(axes.flat, selected):
            k = index_of[r["element"]]
            block = recorded[r["spring_axis"]]
            hd._loop_axis(ax, r, block["rotation"][:, k], block["moment"][:, k],
                          _title(r, detail=False) + "\n" +
                          f"Mn {r['fy_positive_kip_in']:.0f} kip-in, {r['rotation_over_theta_y']:.2f} theta_y, "
                          f"{r['moment_over_mn']:.2f} Mn, E {max(r['dissipated_energy_kip_in'], 0.0):.1f} kip-in",
                          fontsize=9, legend=False)
        for ax in axes.flat[len(selected):]:
            ax.axis("off")
        figure.suptitle(f"joint pinching spring moment (kip-in) vs rotation (%): {len(selected)} of {len(present)} springs ranked by "
                        f"dissipated energy then rotation / theta_y; {yielded} yielded; red dotted = installed Mn per sign, "
                        "grey dashed = installed Ke", fontsize=11)
        figure.tight_layout(rect=(0, 0, 1, 0.975))
        gallery = output_dir / "joint_loops_gallery.png"
        figure.savefig(gallery, dpi=120)
        plt.close(figure)
        written.append(str(gallery))
    return written


def write_joint_histories_npz(recorded, path):
    payload = {"joint_element_order": np.asarray(recorded["hinge_tag_order"], dtype=np.int64)}
    for axis in ("x", "y"):
        if axis in recorded:
            payload[f"time_{axis}"] = recorded[axis]["time"]
            payload[f"moment_{axis}"] = recorded[axis]["moment"].astype(np.float64)
            payload[f"rotation_{axis}"] = recorded[axis]["rotation"].astype(np.float64)
    np.savez_compressed(path, **payload)
    return str(path)


def run_joint_evaluation(coverage, out, time_limit, plot_limit=24):
    """Read, evaluate, plot and save; returns the manifest block."""
    out = Path(out)
    recorded = read_joint_recorders(coverage, time_limit=time_limit)
    rows = evaluate_joint_histories(recorded, coverage)
    summary = joint_summary(rows)
    plots = plot_joint_loops(recorded, rows, out / "joint_loops", limit=plot_limit) if summary["springs_with_response"] else []
    histories = write_joint_histories_npz(recorded, out / "joint_histories_full_rate.npz") if summary["springs_with_response"] else None
    (out / "joint_evaluation.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return {"summary": summary, "recorder_rows": {axis: recorded[axis]["rows"] for axis in ("x", "y") if axis in recorded},
            "histories_npz": histories, "evaluation_json": str(out / "joint_evaluation.json"), "plots": plots,
            "coverage": {k: v for k, v in coverage.items() if k != "joints"}, "joint_count": len(coverage["joints"])}
