"""Write UNIT1_COMPARISON_REPORT.md: the four-case untuned input comparison against the version-2 observations
under the fit criteria (2026-09-28), with the observation and drawing audits, the coupon check, the
numerical acceptance and the completeness status of every run. No parameter is fitted here.

usage: python make_comparison_report.py <output root> [<coupon dir>] [<observation audit dir>]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pr1_unit1 as pr1  # noqa: E402

root = Path(sys.argv[1]).resolve()
coupon_dir = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else root.parent / "diag_park_ruitong_unit1_cases_20260929" / "coupon"
audit_dir = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else root.parent / "diag_park_ruitong_unit1_cases_20260929" / "observation_audit"
obs = json.loads((HERE / "unit1_observed_digitized_v2.json").read_text(encoding="utf-8"))
coupon = json.loads((coupon_dir / "steel_coupon_check.json").read_text(encoding="utf-8")) if (coupon_dir / "steel_coupon_check.json").exists() else None
BAND_KN, BAND_MM, CROSS_MM, CROSS_KN = 3.0, 3.0, 6.0, 6.0
CASES = ["example_inputs", "measured_beam_steel_only", "verified_confinement_only", "both_corrections"]
SHORT = {"example_inputs": "example", "measured_beam_steel_only": "steel", "verified_confinement_only": "confinement", "both_corrections": "both"}


def load_run(folder, label_part=None):
    d = root / folder
    files = sorted(d.glob("*_result.json"))
    files = [f for f in files if "FAILED" not in f.name]
    if label_part:
        files = [f for f in files if label_part in f.name]
    files = [f for f in files if "mirror" not in f.name and "step_x" not in f.name] or files
    if not files:
        return None
    j = json.loads(files[0].read_text(encoding="utf-8"))
    j["file"] = files[0].name
    j["summary"] = json.loads((d / "summary.json").read_text(encoding="utf-8")) if (d / "summary.json").exists() else None
    return j


runs = {c: load_run(c) for c in CASES}
shifts = {(c, sgn): load_run(f"{c}_shift_{'plus' if sgn > 0 else 'minus'}3") for c in ("example_inputs", "both_corrections") for sgn in (3, -3)}
control = load_run("control_example_protocol")
paper = load_run("paper_protocol_self_scaled")
failed = sorted(p.name for p in root.rglob("*_FAILED_result.json"))

peaks_by_run = {p["run"]: p for p in obs["peaks"]}
branch_by_run = {b["run"]: b for b in obs["branches"]}


def cycle_runs(k):
    """Experiment protocol cycle k (1-based) -> (positive run, negative run) of the test."""
    return 2 * k + 1, 2 * k + 2


def observed_peak(run):
    p = peaks_by_run[run]
    lo, hi = (p["V_interval_kN"] if p.get("V_interval_kN") else (p["V_kN"], p["V_kN"]))
    return {"value": p["V_kN"], "lo": lo - BAND_KN, "hi": hi + BAND_KN, "interval": (lo, hi), "obscured": p.get("obscured", False),
            "separable": p.get("cycles_separable"), "status": p["status"], "delta": p["delta_mm"]}


def peak_errors(j):
    """Per test run: model peak at the same amplitude, direction and cycle against the observed reading."""
    rows = []
    for c in j["evaluation"]["cycles"]:
        rp, rn = cycle_runs(c["cycle"])
        for run, model in ((rp, c["peak_pos_kN"]), (rn, c["peak_neg_kN"])):
            o = observed_peak(run)
            err = model - o["value"]
            inside = o["lo"] <= model <= o["hi"]
            dist = 0.0 if inside else (o["lo"] - model if model < o["lo"] else model - o["hi"])
            rows.append({"run": run, "cycle": c["cycle"], "amplitude": c["amplitude_mm"], "model": model, "obs": o, "error": err,
                         "within": inside, "distance_outside": dist})
    return rows


def metrics(rows):
    e = np.array([r["error"] for r in rows])
    return {"n": len(rows), "max_abs": float(np.max(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e)),
            "within": int(sum(r["within"] for r in rows)), "score": float(np.mean((e / BAND_KN) ** 2)),
            "max_outside": float(max(r["distance_outside"] for r in rows))}


L = []
w = L.append
w("# Park and Ruitong (1988) Unit 1: untuned input comparison under the declared fit criteria")
w("")
w("2026-09-29. Second delivery of the reference reproduction, after the Unit 1 review of 2026-09-28. Everything below "
  "was produced without fitting any bond, pinching, damage or stiffness parameter: the four cases differ only in the "
  "measured beam steel and the drawn transverse-steel stations. The production flags (`GENERATION_RELEASE_READY`, "
  "`story_strength_model_verified`) are unchanged; no frame rerun, pilot or campaign was run.")
w("")
w("## 0. What changed since the first package")
w("")
w("- **Commit blocker 1 (ownership survives a new domain)**: `Model/Deformation_Ownership.py` now scopes the slip-interface "
  "registry to one OpenSees domain (`begin_domain` at every builder's wipe, registrations name their interface element, "
  "`validate_installed` refuses stale or unrealised claims); `Build_Model.build_model` opens the domain before any member "
  "exists and refuses any registration once its members exist; the subassembly registers and installs inside the same build. "
  "The domain probe now returns a theta_p ratio of exactly 1.0 with the stale record listed under the new domain. Tests: "
  "`tests/test_deformation_ownership.py::DomainLifecycle` (consecutive builds with a base end and an elevated end, "
  "registration honoured only while its element is installed).")
w("- **Commit blocker 2 (failed runs lose their history)**: `pr1_unit1.run` exports the completed states, the failing phase and "
  "target, the displacement and load at failure, every solver attempt of the failing step and the exception under a `*_FAILED` "
  "label before the error propagates; every accepted sub-step is recorded with its solver settings; mirrored histories are "
  "compared by physical phase, never truncated. Tests: `tests/test_pr1_failure_preservation.py` (injected failure after gravity, "
  "injected failure inside the cycles, sub-step recording, phase-matched comparison).")
w("- **Solver bookkeeping**: every recorded state carries its normalized equilibrium residual (characteristic scales: the observed "
  "maximum load 80.3 kN, its base moment, the gravity total) and the algorithm and tolerance that produced it; rows over 1e-6 are "
  "counted per run (section 4, section 9). Two stricter variants were tried and withdrawn with their failure records: sub-steps at "
  "the requested tolerance before any loose tolerance (a 0.001 mm sub-step hits a material update failure at 50 mm where the "
  "example's loose full step passes) and a zero-increment re-solve of an over-limit state (it fails at exactly those states and the "
  "failed attempts break the next step). The example's chain therefore stands; the 1e-6 limit is reported, not enforced.")
w("- **Observations**: `unit1_observed_digitized_v2.json` replaces the manual v1 reading with a pixel audit "
  "(`digitize_fig16a.py`, section 1). The v1 residual targets (35 and 42 mm) and the single unsigned pinching target are marked "
  "not accepted. The self-scaled paper protocol is archived as a diagnostic; the fixed experimental history is the comparison basis.")
w("- **Drawing audit**: the Fig. 7 station map (section 2) corrects the manifest: the R12 + R8 sets are joint hoops inside the "
  "beam depth, the column next to the joint carries 8 R6(A) at 60 crs (the example's input), the beam elevation is 1236.5 mm.")
w("")

# ---- 1. observation audit --------------------------------------------------------------------------
w("## 1. Observation audit (Fig. 16(a), pixel-traceable)")
w("")
t = obs["transformation"]; tk = obs["ticks"]
w(f"Source: {obs['image']['render']}; ink threshold {obs['image']['ink_threshold']}. Axes fitted as lines through their ink "
  f"(Delta axis slope {obs['axes']['skew']['delta_axis_slope_px_per_px']:+.4f}, V axis slope {obs['axes']['skew']['v_axis_slope_px_per_px']:+.4f} px/px: "
  f"the scan's skew is retained in the affine map). Ticks found: Delta {sorted(tk['delta_axis']['found'], key=float)} mm, "
  f"V {sorted(tk['v_axis']['found'], key=float)} kN (missing: Delta {tk['missing']['delta']}, V {tk['missing']['v']}, hidden by curves or labels); "
  f"affine fit on {t['fit_points']} points, residuals {t['residuals']['delta_rms_mm']:.2f} mm rms / {t['residuals']['delta_max_mm']:.2f} mm max and "
  f"{t['residuals']['v_rms_kN']:.2f} kN rms / {t['residuals']['v_max_kN']:.2f} kN max; {t['mm_per_px']:.4f} mm and {t['kN_per_px']:.4f} kN per pixel. "
  f"The V scale was searched within 10 % of the Delta scale (the figure's equal tick spacing) and verified on the overlay "
  f"`fig16a_audit_overlay.png`.")
w("")
w("### 1.1 Reversals (peaks)")
w("")
w("A reversal is the loop tip: the locally extreme-displacement ink pixel behind which two branches converge. Where the tip of a "
  "smaller loop enters the bundle of the larger loops' loading branches, the force is an interval from the emergence of the "
  "steep unloading branch to the far edge of the contiguous ink (OBSCURED). Where the two cycles at one amplitude share one "
  "tip at the scan's resolution, both runs carry the same reading (not separable). Reading band +-3 kN / +-3 mm on top.")
w("")
w("| run | dir | v1 manual reading (mm, kN) | pixel tip Delta (mm) | pixel tip V (kN) | V interval (kN) | obscured | cycles separable | v1 inside interval +- band |")
w("|---|---|---|---|---|---|---|---|---|")
for p in obs["peaks"]:
    iv = p.get("V_interval_kN")
    w(f"| {p['run']} | {p['direction']} | ({p['v1_reading']['delta_mm']}, {p['v1_reading']['V_kN']}) | {p['delta_mm']:.1f} | {p['V_kN']:.1f} | "
      f"{f'[{iv[0]:.1f}, {iv[1]:.1f}]' if iv else '-'} | {'yes' if p.get('obscured') else 'no'} | "
      f"{'-' if p.get('cycles_separable') is None else ('yes' if p['cycles_separable'] else 'no')} | "
      f"{'-' if p.get('v1_within_interval_or_band') is None else ('yes' if p['v1_within_interval_or_band'] else 'NO')} |")
w("")
aa = obs["amplitude_audit"]
w(f"Amplitude audit: pixel tips by nominal amplitude {aa['pixel_tip_delta_by_amplitude']} mm; {aa['status']}.")
w("")
lc = obs["load_controlled_first_cycle"]
w(f"Load-controlled runs 1 and 2 (target +-{lc['target_kN']} kN): traced reversals at "
  f"{lc['run_1']['delta_mm']:.1f} mm ({lc['run_1']['V_kN_at_tip']:.1f} kN at the tip, interval {lc['run_1'].get('V_interval_kN')}) and "
  f"{lc['run_2']['delta_mm']:.1f} mm ({lc['run_2']['V_kN_at_tip']:.1f} kN). {lc['note']} The v1 manual reading was +-12 mm at +-54.5 kN. "
  f"The displacement at exactly 54.45 kN on the first loading branch is not separately resolvable in the dense origin region; the "
  f"reading interval for the initial compliance is therefore 12 to 14 mm (+-3) positive and 12 to 16 mm (+-3) negative.")
w("")
w("### 1.2 Branches: zero-force crossings, zero-displacement crossings, unloading chords")
w("")
w("Each unloading branch was followed from its reversal by a heading-based walker through the crossings with other loops; a "
  "traced zero-force crossing is *resolved* when an independently detected axis crossing lies within 3 mm of it, *unconfirmed* "
  "otherwise, and *unavailable* where the walker lost the branch (the negative-side unloading branches merge with the positive "
  "reloading branches near zero force) or crossed to the wrong side. Chords are the 80 % to 20 % force levels on the traced "
  "branch with the bounds K_low = max(0, dF - 6)/(dU + 6), K_high = (dF + 6)/(dU - 6) (needs dU > 6 mm).")
w("")
w("| run | dir | trace | zero-force crossing (mm) | status | zero-displacement force (kN) | chord k (kN/mm) over dU (mm) | K_low / K_high |")
w("|---|---|---|---|---|---|---|---|")
for b in obs["branches"]:
    zf, zd, ch = b["zero_force_crossing"], b["zero_displacement_crossing"], b["unloading_chord_80_20"]
    w(f"| {b['run']} | {b['direction']} | {b['trace_status']} ({b['ambiguous_steps']} ambiguous steps) | "
      f"{f'{zf['delta_mm']:.1f}' if zf else 'unavailable'} | {zf['status'] if zf else (b['identity'] if b['identity'] != 'unresolved' else 'no trace')} | "
      f"{f'{zd['V_kN']:.1f} (unconfirmed)' if zd else '-'} | "
      f"{f'{ch['k_kN_per_mm']:.2f} over {ch['dU_mm']:.1f}' if ch and ch.get('k_kN_per_mm') else '-'} | "
      f"{f'{ch['k_low']:.2f} / ' + (f'{ch['k_high']:.2f}' if ch.get('k_high') else 'n/a (dU <= 6)') if ch else '-'} |")
w("")
ic = obs["independent_crossings"]
w("Independent axis crossings (no identity): zero force at " +
  ", ".join(f"{q['delta_mm']:.1f} mm ({q['strands']} strands)" for q in ic["zero_force_delta_mm"]) +
  "; zero displacement at " + ", ".join(f"{q['V_kN']:.1f} kN ({q['strands']} strands, range {q['V_range_kN'][0]:.1f} to {q['V_range_kN'][1]:.1f})" for q in ic["zero_displacement_V_kN"]) +
  ". The reloading branches of mu >= 3 cross the V axis in two bundles of 8 and 11 strands around -35 and +34 kN; their per-cycle identity is unresolved, "
  "so the force at zero displacement is compared against the bundle range, not per cycle.")
w("")
sv = obs["superseded_v1_targets"]
w(f"Superseded v1 targets: residual displacement {sv['residual_displacement_at_zero_load_mm']['mu_6']} / {sv['residual_displacement_at_zero_load_mm']['mu_7']} mm: "
  f"{sv['residual_displacement_at_zero_load_mm']['status']}; load at zero displacement {sv['load_at_zero_displacement_kN']['value']} kN: "
  f"{sv['load_at_zero_displacement_kN']['status']}. Run labels: {obs['run_label_audit']['run_25']}.")
w("")

# ---- 2. drawing audit ----------------------------------------------------------------------------
w("## 2. Drawing audit: transverse-steel stations (Fig. 7, Table 2(a))")
w("")
w("| station | location | ties | spacing (mm) | fy (MPa) | used for member confinement | in the model |")
w("|---|---|---|---|---|---|---|")
for st in pr1.COLUMN_STATIONS:
    w(f"| {st['station']} | {st['where']} | {st['ties']} | {st.get('spacing_mm', '-')} | {st.get('fy_MPa', '-')} | "
      f"{'yes' if st['used_for_member_confinement'] else 'no'} | {'column section (example input = this station)' if st['used_for_member_confinement'] else st.get('note', '')} |")
for st in pr1.BEAM_STATIONS:
    w(f"| {st['station']} | beam, {st['from_mm']:.0f} to {st['to_mm']:.0f} mm from the column face | {st['ties']} | {st['spacing_mm']:.0f} | {st['fy_MPa']:.0f} | "
      f"{'yes' if st['elements'] else 'no'} | {', '.join(st['elements']) if st['elements'] else st['note']} |")
w("")
w("The example's column confinement input (R6 at 60 crs, fy 282, 1853.5 mm of hoop per set = rectangle 1078 + diamond 775.5) is "
  "station C1, the column next to the joint: it stands. The joint hoops (station J) are never smeared into a member section. The "
  "`verified_confinement_only` case changes only the beam: station B1 (R6(B) at 80 crs, fy 366) for the inner elements and "
  "station B2 (R6(B) at 110 crs, fy 366) for the outer elements, against the example's uniform R6 at 80 crs with fy 282. "
  "The manifest's earlier claim of a 300 mm column region with R12/R8 ties is withdrawn (the callout is the joint core: "
  "75 + 4 x 75 + 20 + 20 + 42 = 457 mm).")
w("")

# ---- 3. coupon --------------------------------------------------------------------------------------
w("## 3. Beam steel coupon check (paper Table 2(b), Fig. 5)")
w("")
if coupon:
    w("| material | stress at 0.012 (MPa) | plateau within 1 % of fy | hardening onset strain | tangent 0.0255-0.0265 (MPa) | stress at 0.05 (MPa) | Fig. 5 reading at 0.05 | stress at 0.10 (MPa) |")
    w("|---|---|---|---|---|---|---|---|")
    for name, r in coupon["results"].items():
        w(f"| {name} | {r['stress_at_0.012_MPa']:.1f} | {'yes' if r['plateau_within_1pct_of_fy_at_0.012'] else 'no'} | "
          f"{r['hardening_onset_strain_(stress_>_1.01_fy)'] if r['hardening_onset_strain_(stress_>_1.01_fy)'] is not None else '-'} | "
          f"{r['tangent_0.0255_to_0.0265_MPa']:.0f} | {r['stress_at_0.05_MPa']:.1f} | {r['fig5_reading_at_0.05_MPa'][0]:.0f} +- {r['fig5_reading_at_0.05_MPa'][1]:.0f} "
          f"({'inside' if r['within_fig5_reading_at_0.05'] else 'outside'}) | {r['stress_at_0.10_MPa']:.1f} |")
    e = coupon["eps_su"]
    w("")
    w(f"eps_su is not measured ({e['status']}); with the declared bounds {e['bounds']} the stress at 0.05 strain spans "
      f"{e['stress_at_0.05_over_bounds_MPa'][0]:.1f} to {e['stress_at_0.05_over_bounds_MPa'][1]:.1f} MPa ({100 * e['spread_at_0.05_relative']:.2f} %) and "
      f"{100 * e['spread_at_0.10_relative']:.2f} % at 0.10: within the strain range the beam bars reach (section 5), the bound does not matter. "
      f"At 0.012 strain (the measured face-bar strain at mu = 6) the example's Steel02 gives {coupon['example_vs_measured_at_0.012']['Steel02_example']:.1f} MPa, "
      f"ReinforcingSteel {coupon['example_vs_measured_at_0.012']['ReinforcingSteel']:.1f} MPa and an immediate bilinear 3580 MPa would give "
      f"{coupon['example_vs_measured_at_0.012']['immediate_bilinear_3580_would_give']:.1f} MPa. {coupon['bar_slip_note']}. {coupon['cyclic_note']}. "
      f"Dodd_Restrepo is listed as a second candidate (its intermediate point is the Fig. 5 reading itself, so it is not an independent check).")
else:
    w("Coupon results not found.")
w("")

# ---- 4. runs ----------------------------------------------------------------------------------------
w("## 4. Runs and completeness")
w("")
w("| folder | case | protocol | amplitude shift (mm) | status | rows | sub-step rows | Newton retries | recoveries | rows over the 1e-6 equilibrium limit | finite | time (s) |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
all_runs = [(c, runs[c]) for c in CASES] + [(f"{c}_shift_{'plus' if s > 0 else 'minus'}3", j) for (c, s), j in shifts.items()] + \
           [("control_example_protocol", control), ("paper_protocol_self_scaled", paper)]
for folder, j in all_runs:
    if j is None:
        w(f"| {folder} | - | - | - | MISSING | | | | | | | |")
        continue
    m, e = j["meta"], j["evaluation"]
    w(f"| {folder} | {m['case']['beam_steel']} steel, {m['case']['confinement']} confinement | {m['protocol']} | {m.get('amplitude_shift_mm', 0):+g} | {m['status']} | "
      f"{m['steps']} | {e['substep_rows']} | {m['solver_retries']} | {m['solver_recoveries_by_kind'] or '-'} | {e.get('equilibrium_rows_over_limit', 'n/a')} | "
      f"{'yes' if e['all_finite'] else 'NO'} | {m['elapsed_sec']:.0f} |")
w("")
w(f"Failed-run records in this root: {failed if failed else 'none'}. Every listed run completed all its phases.")
w("")

# ---- 5. envelope ----------------------------------------------------------------------------------
w("## 5. Envelope: signed peaks per run at the experimental history")
w("")
w("Model peak on each half-cycle at the same amplitude, direction and cycle as the test run, against the pixel reading (with its "
  "interval where obscured) and the +-3 kN band. The error is model minus the tip reading; *within* means the model peak lies inside "
  "[interval low - 3, interval high + 3] kN.")
w("")
errs = {c: peak_errors(runs[c]) for c in CASES if runs[c]}
hdr = "| run | dir | amp (mm) | observed tip (kN) | interval + band (kN) | " + " | ".join(f"{SHORT[c]} (kN) / error" for c in errs) + " |"
w(hdr); w("|" + "---|" * (5 + len(errs)))
for i in range(len(next(iter(errs.values())))):
    r0 = errs[CASES[0]][i]
    o = r0["obs"]
    cells = " | ".join(f"{errs[c][i]['model']:.1f} / {errs[c][i]['error']:+.1f}{'' if errs[c][i]['within'] else ' *'}" for c in errs)
    w(f"| {r0['run']} | {'+' if r0['run'] % 2 else '-'} | {r0['amplitude']:.0f} | {o['value']:.1f}{' (obscured)' if o['obscured'] else ''} | [{o['lo']:.1f}, {o['hi']:.1f}] | {cells} |")
w("")
w("(* outside the interval + band)")
w("")
w("| case | n | max |error| (kN) | RMSE (kN) | signed bias (kN) | within interval + band | max distance outside (kN) | peak score mean((error/3)^2) |")
w("|---|---|---|---|---|---|---|---|---|")
for c, rows in errs.items():
    mm = metrics(rows)
    w(f"| {c} | {mm['n']} | {mm['max_abs']:.2f} | {mm['rmse']:.2f} | {mm['bias']:+.2f} | {mm['within']} / {mm['n']} | {mm['max_outside']:.2f} | {mm['score']:.2f} |")
w("")
for c in ("example_inputs", "both_corrections"):
    pk = {s: shifts[(c, s)]["evaluation"]["strength"]["peak_pos_kN"] for s in (3, -3) if shifts[(c, s)]}
    if pk and runs[c]:
        w(f"Coherent amplitude shift, {c}: peak {runs[c]['evaluation']['strength']['peak_pos_kN']:.2f} kN at the nominal amplitudes, "
          f"{pk.get(3, float('nan')):.2f} at +3 mm, {pk.get(-3, float('nan')):.2f} at -3 mm (the +-3 mm reading bound moves the peaks by "
          f"{max(abs(pk.get(3, 0) - runs[c]['evaluation']['strength']['peak_pos_kN']), abs(pk.get(-3, 0) - runs[c]['evaluation']['strength']['peak_pos_kN'])):.2f} kN at most).")
w("")

# ---- 6. initial compliance and retention ------------------------------------------------------------
w("## 6. Initial compliance and cycle retention")
w("")
w("| case | model Delta at +54.45 kN (mm) | at -54.45 kN (mm) | model-derived Delta_y (mm, reported only) | observed reading interval |")
w("|---|---|---|---|---|")
for c in CASES:
    j = runs[c]
    if j:
        m = j["meta"]
        w(f"| {c} | {m['delta_at_pos_target_mm']:.2f} | {m['delta_at_neg_target_mm']:.2f} | {m['delta_y_model_derived_mm']:.2f} | "
          f"+12 to +14 (+-3) / -12 to -16 (+-3) mm; v1 manual +-12 |")
w("")
w("Every case reaches the load target at 60 to 70 % of the observed reading interval: the model is stiffer than the specimen "
  "before yield. The measured steel changes this by 1 %; it is not a steel effect (candidates: the rigid interface springs, the "
  "uncracked fibre stiffness of the outer elements, the rig's unknown restraint).")
w("")
w("| amplitude (mm) | dir | observed P1, P2 (kN) | separable | observed retention interval | " + " | ".join(SHORT[c] for c in CASES) + " |")
w("|---|---|---|---|---|" + "---|" * len(CASES))
for k in range(1, 12, 2):
    amp = 15 * (k // 2 + 2)
    for sign, off in (("+", 1), ("-", 2)):
        r1, r2 = 2 * k + off, 2 * (k + 1) + off
        p1, p2 = peaks_by_run[r1], peaks_by_run[r2]
        sep = p2.get("cycles_separable")
        P1, P2 = abs(p1["V_kN"]), abs(p2["V_kN"])
        if sep:
            interval = f"[{(P2 - BAND_KN) / (P1 + BAND_KN):.3f}, {(P2 + BAND_KN) / (P1 - BAND_KN):.3f}]"
        else:
            interval = f"not separable: [{(P1 - BAND_KN) / (P1 + BAND_KN):.3f}, 1] by the band alone"
        cells = []
        for c in CASES:
            j = runs[c]
            if not j:
                cells.append("-"); continue
            cyc = j["evaluation"]["cycles"]
            key = "peak_pos_kN" if sign == "+" else "peak_neg_kN"
            v1, v2 = abs(cyc[k - 1][key]), abs(cyc[k][key])
            cells.append(f"{v2 / v1:.3f}")
        w(f"| {amp} | {sign} | {P1:.1f}, {P2:.1f} | {'yes' if sep else 'no'} | {interval} | " + " | ".join(cells) + " |")
w("")

# ---- 7. loop shape ----------------------------------------------------------------------------------
w("## 7. Loop shape: residual displacement, force at zero displacement, unloading chord")
w("")
w("Model quantities follow the declared definitions (first zero-force crossing after each reversal by linear interpolation; first "
  "zero-displacement crossing on the same branch with its sign; chord between the first 80 % and 20 % force levels toward zero). "
  "Observed values are the traced ones of section 1.2 with their status; an unavailable observation leaves that comparison open.")
w("")
w("| run | dir | observed residual (mm) [status] | " + " | ".join(f"{SHORT[c]} residual (mm)" for c in CASES) + " | observed force at zero disp. (kN) | " +
  " | ".join(f"{SHORT[c]} force at zero disp." for c in CASES) + " |")
w("|---|---|---|" + "---|" * len(CASES) + "---|" + "---|" * len(CASES))
for k in range(1, 13):
    rp, rn = cycle_runs(k)
    for run, side in ((rp, "after_pos_peak"), (rn, "after_neg_peak")):
        b = branch_by_run[run]
        zf, zd = b["zero_force_crossing"], b["zero_displacement_crossing"]
        cells_r, cells_f = [], []
        for c in CASES:
            j = runs[c]
            if not j:
                cells_r.append("-"); cells_f.append("-"); continue
            m = j["evaluation"]["cycles"][k - 1][side]
            cells_r.append(f"{m['residual_displacement_mm']:.1f}" if m["residual_displacement_mm"] is not None else "-")
            cells_f.append(f"{m['force_at_zero_displacement_kN']:.1f}" if m["force_at_zero_displacement_kN"] is not None else "-")
        w(f"| {run} | {'+' if run % 2 else '-'} | {f'{zf['delta_mm']:.1f} [{zf['status']}]' if zf else 'unavailable'} | " + " | ".join(cells_r) +
          f" | {f'{zd['V_kN']:.1f} [unconfirmed]' if zd else 'bundle -35 / +34'} | " + " | ".join(cells_f) + " |")
w("")
w("| run | dir | observed chord k (kN/mm) [K_low, K_high] | " + " | ".join(f"{SHORT[c]} chord k (kN/mm) over dU" for c in CASES) + " |")
w("|---|---|---|" + "---|" * len(CASES))
for k in range(1, 13):
    rp, rn = cycle_runs(k)
    for run, side in ((rp, "after_pos_peak"), (rn, "after_neg_peak")):
        ch = branch_by_run[run]["unloading_chord_80_20"]
        cells = []
        for c in CASES:
            j = runs[c]
            if not j:
                cells.append("-"); continue
            m = j["evaluation"]["cycles"][k - 1][side]["unloading_chord_80_20"]
            cells.append(f"{m['k_kN_per_mm']:.2f} over {m['du_mm']:.1f}" if m else "-")
        w(f"| {run} | {'+' if run % 2 else '-'} | " + (f"{ch['k_kN_per_mm']:.2f} [{ch['k_low']:.2f}, {ch['k_high']:.2f}]" if ch and ch.get("k_kN_per_mm") and ch.get("k_high") else
                                                     (f"{ch['k_kN_per_mm']:.2f} [K_high n/a]" if ch and ch.get("k_kN_per_mm") else "unavailable")) + " | " + " | ".join(cells) + " |")
w("")

# ---- 8. joint, bar strains, work --------------------------------------------------------------------
w("## 8. Joint quantities, bar strains and work (diagnostic)")
w("")
w("The joint share of the tip displacement is not compared: the experimental 9 to 14 % is a virtual diagonal-gauge quantity and "
  "the model's component-times-height number is not that measurement (conditional independent check, pending the gauge kinematics). "
  "Loop work is diagnostic only: there is no ordered experimental trace.")
w("")
w("| case | panel strain max (rad) | bar slip max at the beam faces (mm) | first top-layer tension yield at the face (V kN, u mm) | max top-layer strain at the face | max bottom-layer strain at the gravity point | total signed work (kN mm) |")
w("|---|---|---|---|---|---|---|")
for c in CASES:
    j = runs[c]
    if not j:
        continue
    e = j["evaluation"]
    fy = e["first_fibre_yield"].get("bLface_top") or e["first_fibre_yield"].get("bRface_top")
    w(f"| {c} | {e['panel_strain_max']:.2e} | {max(e['bar_slip_max_mm'][n] for n in ('node2BarSlipT', 'node4BarSlipT')):.3f} | "
      f"{f'{fy['V_kN']:.1f}, {fy['u_mm']:.1f}' if fy else 'not reached'} | {max(e['beam_bar_strain_max']['bLface_top'], e['beam_bar_strain_max']['bRface_top']):.4f} | "
      f"{max(e['beam_bar_strain_max']['bLgrav_bot'], e['beam_bar_strain_max']['bRgrav_bot']):.4f} | {e['total_signed_work_kN_mm']:.0f} |")
w("")

# ---- 9. numerical acceptance --------------------------------------------------------------------------
w("## 9. Numerical acceptance")
w("")
for c in ("example_inputs", "both_corrections"):
    j = runs[c]
    if not j or not j["summary"]:
        continue
    ch = j["summary"]["checks"]
    om = ch.get("orientation_mirror")
    if om:
        w(f"- Orientation ({c}): geometry reflected, every recorded quantity mapped by `reflection_map`, rows matched by phase "
          f"({'matched' if om['matching_phases'] else 'MISMATCH'}, {om['rows_compared']} rows); tip load differs by {om['global_budget']['V_kN']:.2e} kN and tip "
          f"displacement by {om['global_budget']['u_tip_mm']:.2e} mm against the 0.3 kN / 0.3 mm budget ({'passes' if om['global_budget']['passes'] else 'FAILS'}); "
          f"largest relative discrepancy over all keys {om['max']:.2e} ({max(om['relative'], key=om['relative'].get)}); solver retries original / mirror "
          f"{om['solver']['original']['retries']} / {om['solver']['mirror']['retries']}.")
    for key in ("step_x2", "step_x5"):
        if key in ch:
            s_ = ch[key]
            w(f"- Step size {s_['step_mm']:g} mm ({c}): peak change {s_['peak_change_kN']:.3f} kN ({'within' if s_['within_0_3_kN'] else 'OUTSIDE'} 0.3 kN), work change {100 * s_['work_change_relative']:.3f} %.")
for c in CASES:
    j = runs[c]
    if j:
        e = j["evaluation"]["equilibrium"]
        w(f"- Equilibrium ({c}): max normalized residuals sum Fx {e['sum_fx_relative']:.1e}, sum Fy {e['sum_fy_relative']:.1e}, moment {e['moment_relative']:.1e}; "
          f"{j['evaluation'].get('equilibrium_rows_over_limit', 'n/a')} rows over 1e-6; {'passes' if e['passes_1e-6'] else 'FAILS'} the 1e-6 limit.")
w("- Frozen control: the rewritten script reproduces the 2026-09-27 example-protocol baseline exactly (maximum difference 0.0 kN and 0.0 mm over 332,041 steps).")
w("")

# ---- 10. findings -------------------------------------------------------------------------------------
w("## 10. Findings (no tuning) and what stays open")
w("")
ex, both = runs["example_inputs"], runs["both_corrections"]
if ex and both:
    me, mb = metrics(errs["example_inputs"]), metrics(errs["both_corrections"])
    ms, mc = metrics(errs["measured_beam_steel_only"]), metrics(errs["verified_confinement_only"])
    w(f"- **Beam steel is the input that moves the envelope.** Peak {ex['evaluation']['strength']['peak_pos_kN']:.1f} kN with the example inputs "
      f"(bias {me['bias']:+.1f} kN, {me['within']}/{me['n']} peaks inside the reading interval + band), {runs['measured_beam_steel_only']['evaluation']['strength']['peak_pos_kN']:.1f} kN "
      f"with the measured plateau and hardening (bias {ms['bias']:+.1f} kN, {ms['within']}/{ms['n']} inside); the drawn beam confinement alone changes the peak by "
      f"{abs(runs['verified_confinement_only']['evaluation']['strength']['peak_pos_kN'] - ex['evaluation']['strength']['peak_pos_kN']):.2f} kN "
      f"({mc['within']}/{mc['n']} inside); both together {both['evaluation']['strength']['peak_pos_kN']:.1f} kN (bias {mb['bias']:+.1f}, {mb['within']}/{mb['n']} inside).")
w("- **Where each steel misses.** The example's Steel02 falls 5 to 9 kN below the readings from 75 mm upward (its 0.23 % hardening "
  "cannot follow the observed rise to 80 kN); ReinforcingSteel sits inside the reading interval at 30, 60, 75, 90 and 105 mm (first "
  "cycles) and misses in two places: at 45 mm both steels are 4 to 10 kN below the readings on both sides (the negative reading there "
  "is obscured), and at the second cycle at 105 mm the specimen dropped 11 kN (69.0 after 80.4) while no model degrades (82.9 kN, "
  "+13.9). The remaining error changes sign with amplitude, which a single stiffness or strength factor cannot absorb.")
w("- **Initial compliance is the largest untouched discrepancy**: about 8.5 mm at +-54.45 kN against 12 to 16 mm read from the figure, "
  "insensitive to the steel and the confinement. It belongs to the interface, the element stiffness before cracking or the rig, not to the section materials.")
w("- **Loop shape**: the model's residual displacements after the positive reversals exceed the traced crossings at every amplitude "
  "(18 vs 13.5 mm at 30 mm, 41 vs 29 at 60, 69 vs 51 at 90, 84 vs 60 to 65 at 105 with the example steel; the measured steel takes "
  "4 to 7 mm off the large-amplitude values): the specimen returns to zero force at 55 to 65 % of its amplitude, the model at 77 to 80 %. "
  "The observed 80-20 % unloading chord falls from 4.1 kN/mm at 30 mm to 1.8 at 105 mm while the model's stays at 3.5 to 3.8 "
  "(6.0 on the first unloading), outside the observed upper bound from 90 mm on. The force at zero displacement of the model's "
  "later cycles (35 to 39 kN) lies inside the observed bundle (34 to 35 kN); the first cycles at 30 and 45 mm cross at 47 to 54 kN, "
  "above it. Slip at the beam faces stays below 0.07 mm and the panel below 2.1e-3 rad in every case.")
w("- **Not identified here**: eps_su (bounded, immaterial below 0.10 strain); the cycle-2 reversals at 30 to 90 mm (not separable from cycle 1 "
  "on the scan); the negative-side residuals for runs 8 to 22 and every zero-displacement force per cycle (branches merge); the joint share "
  "(gauge kinematics); the interface material form; the footing anchorage basis. No deterioration energy is identified from this specimen.")
w("- **Stop condition**: no parameter was tuned; the next review selects the anchorage law and fits member and interface together, "
  "with the initial compliance and the loop-shape metrics above as the declared targets and the unresolved observations left unavailable.")
w("")
w("## 11. Files")
w("")
w(f"- Root `{root.name}`: one folder per run (`*_history.csv/.npz`, `*_result.json`, `*_loops.png`, `summary.json` where checks ran); `matrix_run.log`.")
w(f"- Observation audit `{audit_dir.name}`: `fig16a_unit1_400dpi_crop.png` (source crop), `fig16a_audit.json` (every pixel coordinate, transformation, "
  f"traces), `fig16a_audit_overlay.png`, `debug_tips_*.png` (zoom sheets); `Reference_Specimens/park_ruitong_1988/unit1_observed_digitized_v2.json`.")
w(f"- Coupon `{coupon_dir.name}`: `steel_coupon_check.json/.png`, `coupon_*.csv`.")
w("- Code: `pr1_unit1.py` (cases, experiment protocol, failure preservation, equilibrium acceptance), `digitize_fig16a.py`, `steel_coupon_check.py`, "
  "`make_comparison_report.py`; `Model/Deformation_Ownership.py`, `Model/Build_Model.py`, `Analysis/Joint_Slip_Subassembly.py` (domain scope); "
  "tests `test_deformation_ownership.py`, `test_pr1_failure_preservation.py`.")
(root / "UNIT1_COMPARISON_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("wrote", root / "UNIT1_COMPARISON_REPORT.md")
