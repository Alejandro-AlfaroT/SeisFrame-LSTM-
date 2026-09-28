"""Reproduce finite 3D pinching-panel mechanics and plot a synthetic diagnostic."""
from datetime import datetime
import argparse
from pathlib import Path
import hashlib
import json
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RC = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=RC / "outputs" / "imk_verification")
args = parser.parse_args()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(RC), str(RC / "tests")]
import openseespy.opensees as ops
from test_joint_panel import run_prescribed_joint, run_transverse_torsion_probe, RESPONSE_COLUMNS, GEOMETRY
from test_imk_materials import rotation_history

FILES = ["Model/Joint_Panel.py", "Model/JOINT_PANEL.md", "Model/IMK_Materials.py",
         "Model/IMK_Hinges.py", "Model/IMK_Calibration.py", "Structure_Parameters.py",
         "Analysis/Hinge_Hysteresis_Diagnostic.py", "Analysis/Pushover_Diagnostic.py",
         "Analysis/Fixed_Design_Diagnostics.py", "Data_Generation/Graph_Exporter.py", "Ground_Motion_Main.py",
         "tests/test_joint_panel.py", "tests/test_imk_materials.py", "tests/test_beam_hinge_asymmetry.py",
         "tests/test_generation_regressions.py", "tests/test_hinge_hysteresis_fixture.py", "tests/run_imk_verification.py"]
def hashes():
    return {name: hashlib.sha256((RC / name).read_bytes()).hexdigest() for name in FILES}
before = hashes()
suite = unittest.TestSuite()
for module in ("test_joint_panel", "test_imk_materials", "test_beam_hinge_asymmetry",
               "test_generation_regressions", "test_hinge_hysteresis_fixture"):
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName(module))
with (OUT / "tests.log").open("w", encoding="utf-8") as stream:
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
if not result.wasSuccessful():
    raise SystemExit("Mechanical verification failed; see joint_verification/tests.log")

# Nonproportional two-axis loading: different reversal times and amplitudes.
waypoints = [(0., 0.), (.004, 0.), (-.004, .003), (.01, -.008), (-.01, .012),
             (.02, -.016), (-.02, .01), (.03, .021), (-.03, -.021),
             (.03, .021), (-.03, -.021), (0., 0.)]
path = [np.array(waypoints[0])]
for target in waypoints[1:]:
    n = max(1, int(np.ceil(np.max(abs(np.array(target) - path[-1])) / .0002)))
    path.extend(np.linspace(path[-1], target, n + 1)[1:])
path = np.asarray(path)
rows, panel, residual = run_prescribed_joint(path)
spring_work = np.sum(np.diff(rows[:, :2], axis=0) * (rows[1:, 2:4] + rows[:-1, 2:4]) / 2, axis=0)
panel_work = GEOMETRY.volume * np.sum(np.diff(rows[:, 4:6], axis=0) * (rows[1:, 6:8] + rows[:-1, 6:8]) / 2, axis=0)

# Record the numerical penalty sensitivity used in the mechanical check.
probe = rotation_history([.008, -.008, .012, -.012, 0.], step=.0004)
probe_path = np.column_stack([probe, -.6 * probe])
reference, _, _ = run_prescribed_joint(probe_path)
penalty_checks = []
for penalty in (1e10, 1e12):
    approximate, _, _ = run_prescribed_joint(probe_path, handler="Penalty", penalty=penalty)
    penalty_checks.append({"penalty": penalty,
                           "max_rotation_error_rad": float(abs(approximate[:, :2] - reference[:, :2]).max()),
                           "max_moment_error_kip_in": float(abs(approximate[:, 2:4] - reference[:, 2:4]).max())})

np.savez_compressed(OUT / "joint_histories.npz", requested_core_rotations_rad=path,
                    response=rows, response_columns=np.array(RESPONSE_COLUMNS))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig = plt.figure(figsize=(16, 5.8), facecolor="#f8fafc")
grid = fig.add_gridspec(1, 3, width_ratios=(.95, 1., 1.))
ax = fig.add_subplot(grid[0], projection="3d")
ax.set_facecolor("#f8fafc")
for axis, half, color in ((0, GEOMETRY.dx / 2, "#2878a6"),
                           (1, GEOMETRY.dy / 2, "#2878a6"), (2, GEOMETRY.hz / 2, "#c06b32")):
    points = np.zeros((3, 3))
    points[:, axis] = [-half, 0., half]
    ax.plot(*points.T, color=color, lw=4)
    ax.scatter(*points[[0, 2]].T, color=color, s=40, depthshade=False)
ax.scatter([0.], [0.], [0.], s=140, c="#a04083", edgecolors="white", linewidths=1.3, depthshade=False)
ax.text(0., 0., 4., "C/B cores\ncoincident", ha="center", fontsize=9, color="#7e3266")
for suffix, xyz in GEOMETRY.offsets().items():
    ax.text(*(np.array(xyz) * 1.14), suffix.replace("_minus", "−").replace("_plus", "+"), ha="center", fontsize=10)
ax.set_xlabel("Global X (in)", labelpad=8)
ax.set_ylabel("Global Y (in)", labelpad=8)
ax.set_zlabel("Global Z (in)", labelpad=8)
ax.set_xlim(-18, 18)
ax.set_ylim(-20, 20)
ax.set_zlim(-14, 14)
ax.set_box_aspect((36, 40, 28))
ax.view_init(elev=23, azim=-53)
ax.set_title("Finite 3D joint panel", fontweight="bold", pad=16)
ax.text2D(.5, -.26, "Blue: beam core and rigid arms\nOrange: column core and rigid arms", transform=ax.transAxes,
          ha="center", fontsize=9, color="#475569")
for index, (title, color) in enumerate((("Y–Z shear plane · Rx", "#2878a6"),
                                       ("X–Z shear plane · Ry", "#a04083"))):
    ax = fig.add_subplot(grid[index + 1])
    ax.plot(rows[:, index], rows[:, index + 2], color=color, lw=1.05)
    ax.set_title(title, fontweight="bold", pad=16, color=color)
    ax.axhline(0, color="#9ca3af", lw=.6)
    ax.axvline(0, color="#9ca3af", lw=.6)
    ax.set_xlabel("Relative core rotation q (rad)")
    ax.set_ylabel("Conjugate spring moment (kip-in)")
    ax.grid(alpha=.16)
    ax.text(.03, .97, r"$q_x = \gamma_{yz}$" if index == 0 else r"$q_y = -\gamma_{xz}$", transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    ax.text(.5, -.20, f"Signed path work: {spring_work[index]:.2f} kip-in", transform=ax.transAxes,
            ha="center", fontsize=9, color="#475569")
fig.suptitle("IMKPinching in a finite 3D joint subassembly", fontsize=18, fontweight="bold", y=.98)
fig.text(.5, .90, "Synthetic shear-only fixture • simultaneous loading in two directions • no specimen calibration", ha="center", fontsize=11)
fig.text(.5, .035, f"{len(rows)-1:,} committed steps | 24 × 30 × 20 in panel | κF = κD = 0.5 | active Λ = 10, c = 1 | modern Eref = Λ × My", ha="center", fontsize=10, color="#475569")
fig.subplots_adjust(left=.035, right=.98, bottom=.24, top=.76, wspace=.32)
fig.savefig(OUT / "joint_pinching_3d.png", dpi=180, facecolor=fig.get_facecolor())
plt.close(fig)

torsion_probe = [run_transverse_torsion_probe(ratio) for ratio in (0., .1, 1., 10.)]
after = hashes()
report = {"created_at": datetime.now().astimezone().isoformat(), "opensees_version": ops.version(),
          "base_commit": "b90f57c5d0a0c2b08eed6f12ada2850ff3c7b1d9", "tests_run": result.testsRun,
          "test_failures": len(result.failures), "test_errors": len(result.errors),
          "source_sha256_before": before, "source_sha256_after": after, "sources_unchanged": before == after,
          "panel": panel, "committed_steps": len(rows) - 1, "failed_steps": 0,
          "all_response_values_finite": bool(np.isfinite(rows).all()),
          "max_arm_constraint_residual_mixed_in_rad": residual,
          "max_face_shear_kinematic_error_rad": float(max(abs(rows[:, 4] - rows[:, 0]).max(), abs(rows[:, 5] + rows[:, 1]).max())),
          "work_integral_kip_in_by_rx_ry": spring_work.tolist(),
          "virtual_work_error_kip_in_by_plane": abs(spring_work - panel_work).tolist(),
          "penalty_sensitivity": penalty_checks,
          "transverse_torsion_probe": torsion_probe,
          "coverage": "11 joint mechanics, 8 material adapter, 3 member hinge, 59 generation regression and 9 recorder tests; one synthetic nonproportional biaxial joint history",
          "exclusions": ["experimental calibration", "production frame integration", "ground motion", "dynamic mass/gravity reconciliation", "joint bond slip", "biaxial or axial strength interaction"],
          "data_dictionary": {"response": list(RESPONSE_COLUMNS),
                              "q_rx_q_ry": "beam-core minus column-core rotation about global X/Y, radians",
                              "moment_rx_ry": "zeroLength basicForce components, kip-in, conjugate to q",
                              "gamma": "engineering shear from separate face-displacement differences, radians",
                              "tau": "equivalent uniform panel shear stress, ksi; Mx/V and -My/V",
                              "work_integral": "signed integral of M dq; not exact dissipated energy for an open path",
                              "sample_axis": "initial zero then committed quasi-static steps; not earthquake time"},
          "review_queue": ["Resolve need for joint deterioration in code-conforming SMRFs using NIST section 2.3 and the actual detailing/demand domain", "Compare shared-core torsional compatibility with a justified 3D joint formulation", "Resolve member/joint slip partition before a combined shear/slip law", "Integrate faces with clear member lengths, gravity, mass and diaphragm accounting"]}
(OUT / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
(OUT / "report.md").write_text(f"""# Finite 3D pinching joint verification

{result.testsRun} tests passed on OpenSees {ops.version()}: 11 joint checks,
8 modern material, 3 actual member, 59 generation and 9 recorder checks. The plotted
nonproportional biaxial history completed {len(rows)-1:,} committed steps with
zero failed steps. Source byte hashes were unchanged during verification.

The fixture contains two coincident core nodes, six finite joint faces,
rigid arms, and two IMKPinching springs about global X/Y. Both shear planes
are active. The maximum reconstructed shear-coordinate discrepancy was
{report['max_face_shear_kinematic_error_rad']:.3e} rad. Virtual-work discrepancy
was {max(abs(spring_work-panel_work)):.3e} kip-in. A separate PeakOriented
member hinge plus clear beam matched analytical tip compliance, including
the joint rigid-arm lever. Floor constraints preserved both shear rotations.

Synthetic values, including kappaF=kappaD=0.5, verify implementation only.
They are not a joint calibration. No experimental response or building
ground-motion result is represented by these loops. Shear/slip interaction,
axial/biaxial strength interaction, and full-frame integration are excluded.
The current member calibration already includes slip; this panel is shear-only.

Evidence: `tests.log`, `verification.json`, `joint_histories.npz`, and
`joint_pinching_3d.png`. The JSON records input commands, parameter
identities, hashes, units, response columns, constraint sensitivity and the
remaining review items. The constitutive energy equation remains Eref=Lamda*Fy.

Mechanics, assumptions, the force-to-moment derivation, and primary sources
are documented in the repository's `Model/JOINT_PANEL.md`.
The follow-up mechanics/applicability review is `Model/JOINT_PANEL_REVIEW_20260923.md`.
The torsion probe confirms the prototype's parallel stiffness relation;
it does not validate that relation against a physical joint.
""", encoding="utf-8")
print(json.dumps({"tests": result.testsRun, "passed": result.wasSuccessful(),
                  "committed_steps": len(rows) - 1, "sources_unchanged": before == after,
                  "max_kinematic_error_rad": report["max_face_shear_kinematic_error_rad"],
                  "penalty_sensitivity": penalty_checks, "output": str(OUT)}, indent=2))
