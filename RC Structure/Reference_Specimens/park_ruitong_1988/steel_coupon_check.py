"""Coupon check of the beam-steel materials against the measured D16 properties (paper Table 2(b), Fig. 5),
independent of the specimen model (Unit 1 review, 2026-09-28: "implement and independently check the
measured yield plateau and subsequent hardening, not just a larger Steel02 ratio").

Materials checked: the example's Steel02 (bilinear, b E = 488.5 MPa, no plateau); ReinforcingSteel with the
measured fy, fsu, Es, Esh, eps_sh and eps_su at the centre and the bounds of its declared uncertainty (Fig. 5
ends at 0.055 strain, so the strain at fsu is not measured; 0.288 is the fracture strain); Dodd_Restrepo as a
second candidate with the same measured inputs and the Fig. 5 reading at 0.05 strain as its intermediate point.
Monotonic checks: plateau stress at 0.012 (the measured face-bar strain at mu = 6, Fig. 20a), hardening onset,
initial hardening tangent, stress at 0.05 against the Fig. 5 reading, sensitivity to eps_su. A cyclic history
is exported for review; the cyclic plateau shortening and reversal rules are the material's own assumptions and
no coupon test of them exists in the paper.

usage: python steel_coupon_check.py <output dir>
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import openseespy.opensees as ops

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pr1_unit1 as pr1  # noqa: E402

MONOTONIC = np.concatenate([np.linspace(0.0, 0.03, 601)[1:], np.linspace(0.03, 0.12, 181)[1:]])
CYCLIC_PEAKS = (0.004, -0.002, 0.012, -0.004, 0.02, -0.006, 0.03, -0.01, 0.05, 0.0)


def materials():
    ex = pr1.beam_steel_material("example")
    ms = pr1.beam_steel_material("measured")
    out = {"Steel02_example": (ex["type"], ex["args"]),
           "ReinforcingSteel_eps_su_0.20": (ms["type"], ms["args"])}
    for b in pr1.BS_EPS_SU_BOUNDS:
        out[f"ReinforcingSteel_eps_su_{b:.2f}"] = ("ReinforcingSteel", (pr1.B_FY, pr1.BS_FSU, pr1.B_ES, pr1.BS_ESH_MEASURED, pr1.BS_EPS_SH, b))
    out["Dodd_Restrepo_candidate"] = ("Dodd_Restrepo", (pr1.B_FY, pr1.BS_FSU, pr1.BS_EPS_SH, pr1.BS_EPS_SU, pr1.B_ES, 0.05, pr1.FIG5_D16_STRESS_AT_0_05[0]))
    return out


def trace(kind, args, strains):
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    ops.uniaxialMaterial(kind, 1, *args)
    ops.testUniaxialMaterial(1)
    out = []
    for e in strains:
        ops.setStrain(float(e))
        out.append((float(e), ops.getStress(), ops.getTangent()))
    ops.wipe()
    return np.array(out)


def cyclic_strains(peaks, n=200):
    s = [0.0]
    for p in peaks:
        s += list(np.linspace(s[-1], p, n + 1)[1:])
    return np.array(s)


def at(tr, eps):
    return float(np.interp(eps, tr[:, 0], tr[:, 1]))


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results, traces = {}, {}
    for name, (kind, args) in materials().items():
        mono = trace(kind, args, MONOTONIC)
        cyc = trace(kind, args, cyclic_strains(CYCLIC_PEAKS))
        traces[name] = {"monotonic": mono, "cyclic": cyc}
        s_012, s_0255, s_026, s_03, s_05, s_10 = (at(mono, e) for e in (0.012, 0.0255, 0.0265, 0.03, 0.05, 0.10))
        onset = next((float(e) for e, s, _ in mono if e > 0.005 and s > 1.01 * pr1.B_FY), None)
        results[name] = {"material": kind, "args": list(args),
                         "stress_at_0.012_MPa": s_012, "plateau_within_1pct_of_fy_at_0.012": bool(abs(s_012 - pr1.B_FY) <= 0.01 * pr1.B_FY),
                         "hardening_onset_strain_(stress_>_1.01_fy)": onset,
                         "tangent_0.0255_to_0.0265_MPa": (s_026 - s_0255) / 0.001,
                         "stress_at_0.03_MPa": s_03, "stress_at_0.05_MPa": s_05, "stress_at_0.10_MPa": s_10,
                         "fig5_reading_at_0.05_MPa": list(pr1.FIG5_D16_STRESS_AT_0_05),
                         "within_fig5_reading_at_0.05": bool(abs(s_05 - pr1.FIG5_D16_STRESS_AT_0_05[0]) <= pr1.FIG5_D16_STRESS_AT_0_05[1]),
                         "max_stress_MPa": float(np.max(mono[:, 1])), "strain_at_max_stress_within_0.12": float(mono[np.argmax(mono[:, 1]), 0]),
                         "cyclic_min_stress_MPa": float(np.min(cyc[:, 1])), "cyclic_max_stress_MPa": float(np.max(cyc[:, 1]))}
        with (out / f"coupon_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            w = csv.writer(handle)
            w.writerow(["history", "strain", "stress_MPa", "tangent_MPa"])
            for e, s, t in mono:
                w.writerow(["monotonic", e, s, t])
            for e, s, t in cyc:
                w.writerow(["cyclic", e, s, t])
    rs = {k: v for k, v in results.items() if k.startswith("ReinforcingSteel")}
    s05 = [v["stress_at_0.05_MPa"] for v in rs.values()]
    summary = {"measured_inputs": {"fy": pr1.B_FY, "Es": pr1.B_ES, "eps_sh": pr1.BS_EPS_SH, "Esh": pr1.BS_ESH_MEASURED, "fsu": pr1.BS_FSU,
                                   "eps_sf_fracture": 0.288, "source": "paper Table 2(b), Fig. 5"},
               "eps_su": {"value_used": pr1.BS_EPS_SU, "bounds": list(pr1.BS_EPS_SU_BOUNDS),
                          "status": "not measured: Fig. 5 ends at 0.055 strain; 0.288 is the fracture strain and is not the strain at fsu",
                          "stress_at_0.05_over_bounds_MPa": [min(s05), max(s05)], "spread_at_0.05_relative": (max(s05) - min(s05)) / max(s05),
                          "spread_at_0.10_relative": (max(v["stress_at_0.10_MPa"] for v in rs.values()) - min(v["stress_at_0.10_MPa"] for v in rs.values()))
                                                     / max(v["stress_at_0.10_MPa"] for v in rs.values())},
               "example_vs_measured_at_0.012": {"Steel02_example": results["Steel02_example"]["stress_at_0.012_MPa"],
                                                "ReinforcingSteel": results["ReinforcingSteel_eps_su_0.20"]["stress_at_0.012_MPa"],
                                                "immediate_bilinear_3580_would_give": pr1.B_FY + pr1.BS_ESH_MEASURED * (0.012 - pr1.B_FY / pr1.B_ES)},
               "bar_slip_note": f"the BarSlip springs keep the example's bilinear Eh = {pr1.BS_ESH:.1f} MPa in every case; BarSlip has no plateau-end input, "
                                "so changing its Eh to 3580 would not reproduce the coupon and is not done",
               "cyclic_note": "the cyclic history (peaks " + ", ".join(f"{p:g}" for p in CYCLIC_PEAKS) + ") is exported for review; the paper has no cyclic "
                              "coupon test, so plateau shortening on reversal is the material's assumption, not a checked quantity",
               "results": results}
    (out / "steel_coupon_check.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for name, tr in traces.items():
        axes[0].plot(tr["monotonic"][:, 0], tr["monotonic"][:, 1], lw=0.9, label=name)
        axes[1].plot(tr["cyclic"][:, 0], tr["cyclic"][:, 1], lw=0.7, label=name)
    axes[0].errorbar([0.05], [pr1.FIG5_D16_STRESS_AT_0_05[0]], yerr=[pr1.FIG5_D16_STRESS_AT_0_05[1]], fmt="s", color="red", capsize=3, label="Fig. 5 reading at 0.05")
    axes[0].axhline(pr1.B_FY, color="0.6", ls=":", lw=0.8); axes[0].axvline(pr1.BS_EPS_SH, color="0.6", ls=":", lw=0.8)
    axes[0].set_xlim(0, 0.12); axes[0].set_xlabel("strain"); axes[0].set_ylabel("stress (MPa)"); axes[0].set_title("monotonic coupon", fontsize=10)
    axes[0].legend(fontsize=7); axes[0].grid(True, lw=0.3, color="0.9")
    axes[1].set_xlabel("strain"); axes[1].set_ylabel("stress (MPa)"); axes[1].set_title("cyclic history (material assumption, no coupon test)", fontsize=10)
    axes[1].legend(fontsize=7); axes[1].grid(True, lw=0.3, color="0.9")
    fig.suptitle("D16 beam steel: measured properties (paper Table 2(b), Fig. 5) against the candidate materials", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out / "steel_coupon_check.png", dpi=130); plt.close(fig)
    for name, r in results.items():
        print(f"{name:32s} s(0.012) {r['stress_at_0.012_MPa']:6.1f}  onset {r['hardening_onset_strain_(stress_>_1.01_fy)']}  "
              f"tangent {r['tangent_0.0255_to_0.0265_MPa']:7.0f}  s(0.05) {r['stress_at_0.05_MPa']:6.1f}  s(0.10) {r['stress_at_0.10_MPa']:6.1f}  "
              f"max {r['max_stress_MPa']:6.1f} at {r['strain_at_max_stress_within_0.12']:.3f}")
    print(f"eps_su bounds {pr1.BS_EPS_SU_BOUNDS}: stress at 0.05 spans {min(s05):.1f}-{max(s05):.1f} MPa ({100 * summary['eps_su']['spread_at_0.05_relative']:.2f} %)")
    return summary


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE.parents[1] / "outputs" / "diag_park_ruitong_unit1_cases_20260929" / "coupon")
