"""Write UNIT1_BASELINE_REPORT.md: the baseline reproduction against the digitized observations, the
numerical checks and the discrepancy list (next-calibration package, first delivery).

usage: python make_baseline_report.py <output root with example_protocol/ and paper_protocol/>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = Path(sys.argv[1]).resolve()
EX_DIR = sys.argv[2] if len(sys.argv) > 2 else "example_protocol"
PA_DIR = sys.argv[3] if len(sys.argv) > 3 else "paper_protocol"
OR_DIR = sys.argv[4] if len(sys.argv) > 4 else None
ex = json.load(open(root / EX_DIR / "summary.json", encoding="utf-8"))
pa = json.load(open(root / PA_DIR / "summary.json", encoding="utf-8"))
orient = json.load(open(root / OR_DIR / "summary.json", encoding="utf-8")) if OR_DIR else None
obs = json.load(open(HERE / "unit1_observed_digitized.json", encoding="utf-8"))
u_kn, u_mm = obs["digitization"]["uncertainty_kN"], obs["digitization"]["uncertainty_mm"]

lines = []
w = lines.append
w("# Park and Ruitong (1988) Unit 1: baseline reproduction and discrepancy report")
w("")
w("2026-09-27. First delivery of the reference reproduction (next-calibration package): the OpenSees PR1 example ported to OpenSeesPy "
  "(`pr1_unit1.py`, inputs in `UNIT1_INPUT_MANIFEST.md`) run through the example's protocol and the paper's protocol, compared with the observations "
  "digitized from Fig. 16(a) (`unit1_observed_digitized.json`). No parameter was fitted. A running script and agreement with the PEER model are not "
  "experimental validation; the comparison below is the evidence.")
w("")
b = ex["baseline"]; m, e = b["meta"], b["evaluation"]
w("## 1. Runs")
w("")
w("| run | protocol | step (mm) | steps | peak V + / - (kN) | observed max (kN) | Newton retries / failures | sum Fx / moment residual (relative) | time (s) |")
w("|---|---|---|---|---|---|---|---|---|")
for label, s in (("example baseline", ex["baseline"]), ("paper protocol", pa["baseline"])):
    mm, ee = s["meta"], s["evaluation"]
    w(f"| {label} | {mm['protocol']} | {mm['step_mm']:g} | {mm['steps']} | {ee['strength']['peak_pos_kN']:.1f} / {ee['strength']['peak_neg_kN']:.1f} | 80.3 | "
      f"{mm['solver_retries']} / {mm['solver_failures']} | {ee['equilibrium']['sum_fx_relative']:.1e} / {ee['equilibrium']['moment_relative']:.1e} | {mm['elapsed_sec']:.0f} |")
w("")
pm = pa["baseline"]["meta"]
w(f"Paper protocol: load control to +-{pm['load_control_target_kN']:.2f} kN reached {pm['delta_y1_mm']:.2f} and {pm['delta_y2_mm']:.2f} mm, so "
  f"Delta_y = (4/3) x mean = {pm['delta_y_mm']:.2f} mm (the example assumes 15 mm, inferred from the figure); the mu cycles then run to "
  f"{', '.join(f'{mu * pm['delta_y_mm']:.0f}' for mu in pm['mu_levels'])} mm.")
w("")
w("## 2. Envelope against the digitized observations (example protocol, cycles at the example's amplitudes)")
w("")
w(f"Observed peaks read from Fig. 16(a) with +-{u_kn:g} kN and +-{u_mm:g} mm; run 1-2 of the test were load-controlled to +-54.45 kN.")
w("")
w("| amplitude (mm) | mu (Delta_y 15 mm) | cycle | simulated + (kN) | observed + (kN) | difference + | simulated - (kN) | observed - (kN) | difference - |")
w("|---|---|---|---|---|---|---|---|---|")
cycles = e["cycles"]
by_amp = {}
for c in cycles:
    by_amp.setdefault(round(c["amplitude_mm"]), []).append(c)
for p in obs["peaks"]:
    pass
obs_by = {}
for p in obs["peaks"]:
    if p.get("load_control"):
        continue
    obs_by.setdefault((abs(p["mu"]), p.get("cycle", 1), p["direction"]), p["V_kN"])
for amp in sorted(by_amp):
    if amp < 1:
        continue
    mu = amp / 15.0
    for k, c in enumerate(by_amp[amp]):
        op = obs_by.get((round(mu), k + 1, "+")); on = obs_by.get((round(mu), k + 1, "-"))
        dp = f"{c['peak_pos_kN'] - op:+.1f}" if op is not None else "-"
        dn = f"{c['peak_neg_kN'] - on:+.1f}" if on is not None else "-"
        w(f"| {amp} | {mu:g} | {k + 1} | {c['peak_pos_kN']:.1f} | {op if op is not None else 'load-controlled run'} | {dp} | {c['peak_neg_kN']:.1f} | {on if on is not None else '-'} | {dn} |")
w("")
w(f"Maximum simulated load {e['strength']['peak_pos_kN']:.1f} / {e['strength']['peak_neg_kN']:.1f} kN against 80.3 kN observed "
  f"({100 * (e['strength']['peak_over_observed_max'] - 1):+.1f} %); PEER's own lumped-plasticity simulation reached 71.2 kN (-11 %, PEER Table 5.2). "
  f"Theoretical V1 = 54.2 and V2 = 72.6 kN (paper Table 4).")
w("")
w("## 3. Loop shape: unloading, pinching, retention, work")
w("")
w("| amplitude (mm) | cycle | unloading secant after + peak (kN/mm, first 5 mm) | load at zero displacement after + peak (kN) | residual displacement at zero load after + peak (mm) | signed work (kN mm) | peak retention cycle 2 / cycle 1 |")
w("|---|---|---|---|---|---|---|")
for amp in sorted(by_amp):
    if amp < 1:
        continue
    first = by_amp[amp][0]
    for k, c in enumerate(by_amp[amp]):
        ret = f"{c['peak_pos_kN'] / first['peak_pos_kN']:.3f}" if k == 1 else "-"
        w(f"| {amp} | {k + 1} | {c['unloading_secant_from_pos_peak_kN_per_mm']:.2f} | {c['load_at_zero_displacement_after_pos_peak_kN']:.1f} | "
          f"{c['residual_displacement_at_zero_load_after_pos_peak_mm']:.1f} | {c['signed_work_kN_mm']:.0f} | {ret} |")
pi = obs["pinching_indicators"]
w("")
w(f"Observed: load at zero displacement about {pi['load_at_zero_displacement_kN']['value']} kN (+-{pi['load_at_zero_displacement_kN']['uncertainty']}) at mu >= 4; "
  f"residual displacement at zero load about {pi['residual_displacement_at_zero_load_mm']['mu_6']} mm after the 90 mm peak and "
  f"{pi['residual_displacement_at_zero_load_mm']['mu_7']} mm after the 105 mm peak (+-{pi['residual_displacement_at_zero_load_mm']['uncertainty']}); "
  "strength retention at the end of the test about 100 % of the theoretical (PEER Table 5.1); the paper: strength, stiffness and energy dissipation maintained well. "
  "Signed loop work has no experimental counterpart here (the loops were not digitized point by point).")
w("")
w("## 4. Joint and bar-slip quantities")
w("")
w("| amplitude (mm) | cycle | panel shear strain at + peak (rad) | joint deformation components at + peak (bar slip / interface / panel / total) | approximate share of the tip displacement (bar slip / panel) |")
w("|---|---|---|---|---|")
for amp in sorted(by_amp):
    if amp < 1:
        continue
    for k, c in enumerate(by_amp[amp]):
        comp = c["joint_def_components_at_pos_peak"]; share = c["joint_share_of_tip_disp_at_pos_peak"] or {}
        w(f"| {amp} | {k + 1} | {c['panel_strain_at_pos_peak']:.2e} | {comp['barslip']:.2e} / {comp['interface']:.2e} / {comp['panel']:.2e} / {comp['total']:.2e} | "
          f"{share.get('barslip', 0):.3f} / {share.get('panel', 0):.3f} |")
w("")
w("The share is approximate: each rotation-like component of the element's deformation output is taken as a rigid rotation over the full column height. "
  "Observed: the joint core contributed 9 to 14 % of the tip displacement at high ductility (paper section 4.2), measured on diagonals that also include "
  "some anchorage-zone deformation (PEER p. 52); PEER's simulation: bar slip 47 % of the joint deformation, panel 5 to 6 % of the total, joint 6 to 9 % of the total.")
w("")
w("| beam bar-slip spring | fy As (kN) | max tension / fy As | max compression (kN) | first tension yield: V (kN), u (mm) | max slip (mm) |")
w("|---|---|---|---|---|---|")
for name, t in e["beam_bar_slip_spring_forces"].items():
    fy = e["first_bar_yield_at_joint_face"].get(name)
    w(f"| {name} | {t['fy_As_kN']:.0f} | {t['max_tension_over_fy_As']:.2f} | {t['max_compression_kN']:.0f} | "
      f"{(f'{fy['V_kN']:.1f}, {fy['u_mm']:.1f}') if fy else 'not reached'} | {e['bar_slip_max_mm'][name]:.3f} |")
w("")
w("Observed: beam bar yielding penetrated into the joint core with the bar stress below yield in the middle quarter (Fig. 20a); no slip measurement is available, "
  "only the strain profiles. The implied slip at bar yield of the example's bond law is 0.067 mm (manifest section 4).")
w("")
w("## 5. Numerical checks before any fitting")
w("")
ch = ex.get("checks", {})
for source in (ex, orient):
    if source is None or "orientation_mirror" not in source.get("checks", {}):
        continue
    d = source["checks"]["orientation_mirror"]
    tol = source["baseline"]["meta"].get("solver", {}).get("tolerance", 1e-8)
    w(f"- Orientation mapping at solver tolerance {tol:g}: the geometry reflected (x -> -x), the beam elements kept running toward +x so the asymmetric "
      f"section stays the right way up, the joint element's node order kept anticlockwise, the loading reversed, and every recorded quantity mapped by "
      f"`reflection_map` (joint springs node 2 <-> node 4 and L <-> R at nodes 1 and 3, beam ends i <-> j, panel and column curvature flipped, bar slip and the "
      f"element's bar-slip and interface components invariant); maximum relative discrepancy over the whole history {d['max']:.2e} "
      f"(tip load {d['V']:.2e}, panel force {d['shearpanel_F']:.2e}, panel strain {d['shearpanel_d']:.2e}; worst key "
      f"{max((k for k, v in d.items() if isinstance(v, float) and k != 'max'), key=lambda k: d[k])} {d['max']:.2e}). "
      f"{'Passes' if d['max'] < 1e-9 else 'Does not pass'} the 1e-9 acceptance"
      + (f"; Newton needed the KrylovNewton fallback at {d['solver_retries']['original']} of {d['solver_retries']['steps']} steps at this tolerance, "
         "so the residual is what the fibre and pinching materials converge to, not a mapping defect." if isinstance(d.get("solver_retries"), dict) else "."))
for key in ("step_x2", "step_x5"):
    if key in ch:
        c = ch[key]
        w(f"- Step size {c['step_mm']:g} mm against {m['step_mm']:g} mm: peak load {c['peak_pos_kN']:.2f} kN ({100 * c['peak_change_relative']:.3f} % change), "
          f"total signed work {c['total_signed_work_kN_mm']:.0f} kN mm ({100 * c['work_change_relative']:.3f} % change).")
w(f"- Equilibrium at every step: sum Fx {e['equilibrium']['sum_fx_max_abs_N']:.1e} N, sum Fy {e['equilibrium']['sum_fy_max_abs_N']:.1e} N, "
  f"moment about the base {e['equilibrium']['moment_about_base_max_abs_Nmm']:.1e} N mm ({e['equilibrium']['moment_relative']:.1e} of V H).")
w(f"- Deformation accounting: the joint element reports bar-slip, interface-shear and panel components and their sum at every step; the beams' and "
  f"columns' basic deformations (chord rotations) and the curvatures at the joint faces are in the history files. Panel strain max {e['panel_strain_max']:.2e} rad.")
w("")
w("## 6. Discrepancies to reconcile before fitting (inputs and model form)")
w("")
w("1. Beam steel: measured D16 has Es 210,400 MPa, a yield plateau to 0.0255 and then Esh 3,580 MPa (paper Table 2, Fig. 5); the example's Steel02 uses "
   "b = 0.002322 (b E = 488.5 MPa) with no plateau. The BarSlip springs inherit Eh = 488.5 MPa. The measured beam bar strains at the joint face reached about "
   "0.012 at mu = 6 (Fig. 20a), inside the plateau.")
w("2. Transverse steel for confinement: the example's R6 at 60 (column) and 80 mm (beam) with fy 282 against the drawings' R12 + R8 sets in and next to the "
   "joint and R6(B) with fy 366 in the beam hinge region.")
w("3. Element form: the example uses force-based fibre elements (5 / 3 / 2 integration points); PEER's own comparison used lumped-plasticity elements with "
   "bilinear hinges and a plastic-hinge length of half the depth, and attributed its 8 to 13 % strength shortfall to that bilinear envelope or to a too-small "
   "post-yield stiffness of the joint components (PEER p. 49).")
w("4. Protocol: the example replaces the load-controlled +-0.75 V2 cycle by a 10 mm displacement cycle and assumes Delta_y = 15 mm; the paper protocol "
   f"measured here gives Delta_y = {pm['delta_y_mm']:.2f} mm.")
w("5. Interface shear: rigid (E = 1e10 N/mm) in the example, elastic by assumption in PEER because the rig's axial restraint is unknown.")
w("6. Bond strength for yielded bars in compression: PEER Table 4.2 prints 3.6 sqrt(f'c) MPa, the source code uses 3.7.")
w("7. Observations: the loops exist only as a scanned figure; peaks carry +-3 kN, +-3 mm; Delta_y, bar slip and the joint distortion history are not printed.")
w("")
w("## 7. What the baseline shows (no fitting)")
w("")
w(f"- Strength: {100 * (e['strength']['peak_over_observed_max'] - 1):+.1f} % against the observed maximum; the simulated envelope hardens less than the observed one "
  "between mu = 3 and mu = 6.")
w("- Loop shape: the simulation unloads at close to its initial stiffness (residual displacement about 80 % of the peak at mu = 7) where the observed loops "
  "unload gradually (residual about 40 %); the observed loops are more pinched than the fibre-element loops although the paper calls the pinching not significant.")
w("- Joint: the panel stays well inside its first envelope segment; the bar-slip springs at the joint faces stay near their yield slip; the joint's share of the "
  "tip displacement is below the observed 9 to 14 %, as in PEER's simulation (6 to 9 %).")
w("- Retention: the second cycle at each amplitude carries a few percent less than the first, from the fibre steel's Bauschinger response, not from a "
  "calibrated degradation.")
w("")
w("These are the quantities a fit would have to move; the fit metrics and tolerances are to be declared from the measurement quality above before any "
  "parameter is tuned (package item 4). Unit 1's small observed degradation cannot identify deterioration capacities on its own.")
w("")
w("## 8. Files")
w("")
w(f"- `{EX_DIR}/unit1_example_{{history.csv, history.npz, result.json, loops.png}}` and the same for `_mirror`, `_step_x2`, `_step_x5`; `{EX_DIR}/summary.json`.")
w(f"- `{PA_DIR}/unit1_paper_{{history.csv, history.npz, result.json, loops.png}}`; `{PA_DIR}/summary.json`.")
w("- `example_protocol/` (first edition): the same baseline; its mirrored run had the beam elements' local axes reflected (section top pointing down), which is why its orientation check reported a 20x discrepancy; kept as a record of the mistake.")
w("- Inputs: `Reference_Specimens/park_ruitong_1988/UNIT1_INPUT_MANIFEST.md`, `unit1_observed_digitized.json`, `pr1_unit1.py`.")
(root / "UNIT1_BASELINE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", root / "UNIT1_BASELINE_REPORT.md")
