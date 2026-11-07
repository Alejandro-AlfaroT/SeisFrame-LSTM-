"""Do the approved registries follow the written frozen equations? (consistency of the instruction and the evidence)

Independent of the production code: for every spring of the ten designs the pre-freeze baseline registry
(v6-mcer-hinges/cases/<case>/registry.json, member intervals installed natively) is converted with the equations of
the 2026-10-29 implementation instruction, written out here:

    K0 = Kh/20, Kb = 1.05 K0, Mc = 1.10 My
    dp  = (My/K0 + theta_p) - Mc/Kb - My/Kh,  dpc = theta_pc + Mc/Kb,  du unchanged
    Eref_S = Eref_C = lambda_H My_canonical (My_canonical/K0),  Eref_A = Eref_K = 1e12 My_canonical
    Lambda_m = Eref_m / My_native_positive

and the result is compared with the approved study registry, argument by argument.

    python -X utf8 -B -I tools/imk_frozen_v2/spec_equations_check.py --study <study root> --out <new dir>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evidence as ev                                                   # noqa: E402


def expected_command(baseline):
    command = list(baseline["command"])
    kh = command[2]
    k0 = kh / 20.0
    kb = 1.05 * k0
    for offset in (3, 9):                                               # dp, dpc, du, My, FmaxFy, FresFy per sign
        dp, dpc, du, my, fmax, fres = command[offset:offset + 6]
        mc = my * fmax
        command[offset] = my / k0 + dp - mc / kb - my / kh
        command[offset + 1] = dpc + mc / kb
    mapping = baseline["provenance"]["energy_mapping"]
    axis = baseline["provenance"]["deterioration"]["spring_axis"]
    lam = baseline["provenance"]["deterioration"]["deterioration_by_axis"][axis]["lambda_haselton_dimensionless"]
    my_can = mapping["reference_moment_kip_in"]
    my_pos = command[6]
    e_sc = lam * my_can * (my_can / k0)
    command[15] = command[16] = e_sc / my_pos                        # Lamda S, C
    command[17] = command[18] = 1.0e12 * my_can / my_pos             # Lamda A, K
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", required=True)
    parser.add_argument("--out", default=str(ev.DEFAULT_OUTPUT / "spec_equations"))
    args = parser.parse_args()
    study = ev.study_root(args.study)
    harness = ev.harness_root(study)
    out = ev.new_output_dir(args.out)
    names = ["Ke", "dp+", "dpc+", "du+", "My+", "FmaxFy+", "FresFy+", "dp-", "dpc-", "du-", "My-", "FmaxFy-",
             "FresFy-", "LamdaS", "LamdaC", "LamdaA", "LamdaK", "cS", "cC", "cA", "cK", "Dpos", "Dneg"]
    worst = {name: {"max_abs": 0.0, "max_rel": 0.0, "bitwise": 0} for name in names}
    springs, outside, examples = 0, 0, []
    for case in ev.CASES:
        baseline = json.loads((harness / "cases" / case / "registry.json").read_text())
        approved = json.loads((study / "mce_average" / "cases" / case / "registry.json").read_text())
        for tag, entry in baseline.items():
            for end, materials in entry["installed_materials"].items():
                for axis, material in materials.items():
                    springs += 1
                    want = expected_command(material)
                    got = approved[tag]["installed_materials"][end][axis]["command"]
                    for index, name in enumerate(names):
                        a, b = want[2 + index], got[2 + index]
                        err = abs(a - b)
                        rel = err / abs(b) if b else (0.0 if err == 0 else math.inf)
                        stats = worst[name]
                        stats["max_abs"], stats["max_rel"] = max(stats["max_abs"], err), max(stats["max_rel"], rel)
                        stats["bitwise"] += int(a == b)
                        if err > 1e-12 + 1e-10 * abs(b):
                            outside += 1
                            if len(examples) < 20:
                                examples.append({"where": f"{case}/{tag}/{end}/{axis}", "argument": name,
                                                 "equations": a, "approved": b})
    report = {"check": "approved registries against the instruction's equations applied to the pre-freeze baseline",
              "springs": springs, "outside_rtol_1e-10_atol_1e-12": outside, "examples": examples, "by_argument": worst,
              "consistent": outside == 0, "study": str(study), "harness": str(harness)}
    ev.write_json(out / "spec_equations_check.json", report)
    print(json.dumps({"springs": springs, "consistent": report["consistent"], "outside": outside,
                      "max_rel_by_argument": {k: v["max_rel"] for k, v in worst.items() if v["max_rel"]}}, indent=1))
    return 0 if report["consistent"] else 1


if __name__ == "__main__":
    sys.exit(main())
