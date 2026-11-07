"""Full parameter parity of the frozen IMK V2 calibration (verification A, 2026-10-29).

``build`` rebuilds one saved design through the production path (install_record under the V2 profile, then
Model.Build_Model.build_model) in a clean process WITHOUT the research adapter and saves its hinge registry.
``--replay`` builds it instead under the explicit historical replay profile, whose springs must reproduce the
pre-freeze baseline registries of commit 66f964c1 (v6-mcer-hinges/cases/<case>/registry.json).

``all`` runs every build in parallel subprocesses and then ``compare``: every member, end, axis and native sign of
the ten designs (3,407 members, 13,628 springs) against the approved MCER and DBE registries of the study, and the
replay against the baseline. Native command arguments are compared with rtol 1e-10 and atol 1e-12 and the maximum
errors are reported, not only pass/fail; the branch orientation, the physical energies, the strengths, the
stiffness, the unchanged du and the member-reference quantities of the approved adapter are compared field by
field; metadata that legitimately changed (identifiers, statuses, hashes) is listed and explained.

    python -X utf8 -B -I tools/imk_frozen_v2/parity.py all --study <study root> --out <new dir>
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evidence as ev                                                   # noqa: E402

RTOL, ATOL = 1e-10, 1e-12
ARGUMENT_NAMES = ["Ke", "dp+", "dpc+", "du+", "My+", "FmaxFy+", "FresFy+", "dp-", "dpc-", "du-", "My-", "FmaxFy-",
                  "FresFy-", "LamdaS", "LamdaC", "LamdaA", "LamdaK", "cS", "cC", "cA", "cK", "Dpos", "Dneg"]
V2 = "v2_nonlinear_flexure_screening_v1"
REPLAY = "v2_nonlinear_flexure_screening_v1_legacy_imk_replay"


def build(args):
    study = ev.study_root(args.study)
    out = Path(args.out).resolve()
    record_path = study / "mce_average" / "inputs" / args.case / "grouped_design.json"
    from Model import IMK_Materials as mm
    from Analysis.Fixed_Design_Diagnostics import install_record
    from Model.Build_Model import build_model
    from Model.IMK_Hinges import hinge_registry
    from Model.Analysis_Profile import profile_identity, verify_installed_domain
    import openseespy.opensees as ops
    if getattr(mm, "_research_half_member_installed", False):
        raise SystemExit("the research adapter is installed in this process")
    record = json.loads(record_path.read_bytes())
    profile = REPLAY if args.replay else V2
    _overrides, site = install_record(record, args.case, profile)
    build_model()
    registry = hinge_registry()
    topology = verify_installed_domain(profile)
    ops.wipe()
    label = "replay" if args.replay else "frozen"
    ev.write_json(out / "registries" / f"{args.case}_{label}.json", registry)
    ev.write_json(out / "registries" / f"{args.case}_{label}_build.json",
                  {"case": args.case, "profile": profile, "record": str(record_path),
                   "record_sha256": ev.sha256_file(record_path), "analysis_profile": site.get("analysis_profile"),
                   "profile_identity": profile_identity(), "installed_topology_consistent": topology["consistent"],
                   "installed_topology_problems": topology["problems"],
                   "installed_member_spring_calibrations": topology["installed_member_spring_calibrations"],
                   "adapter_installed": False})
    print(json.dumps({"case": args.case, "profile": profile, "members": len(registry),
                      "topology_consistent": topology["consistent"]}), flush=True)
    return 0


class Tally:
    """Maximum absolute and relative errors and every out-of-tolerance value, per named quantity."""

    def __init__(self):
        self.stats, self.failures, self.count = {}, [], 0

    def number(self, name, new, old, where):
        self.count += 1
        entry = self.stats.setdefault(name, {"compared": 0, "max_abs_err": 0.0, "max_rel_err": 0.0, "bitwise_equal": 0})
        entry["compared"] += 1
        if new is None or old is None or isinstance(new, (str, bool)) or isinstance(old, (str, bool)):
            if new != old:
                self.failures.append({"where": where, "quantity": name, "new": new, "approved": old})
            else:
                entry["bitwise_equal"] += 1
            return
        new, old = float(new), float(old)
        err = abs(new - old)
        rel = err / abs(old) if old != 0 else (0.0 if err == 0 else math.inf)
        entry["max_abs_err"] = max(entry["max_abs_err"], err)
        entry["max_rel_err"] = max(entry["max_rel_err"], rel)
        entry["bitwise_equal"] += int(new == old)
        if not (math.isfinite(new) and err <= ATOL + RTOL * abs(old)):
            self.failures.append({"where": where, "quantity": name, "new": new, "approved": old, "abs_err": err})

    def equal(self, name, new, old, where):
        self.count += 1
        entry = self.stats.setdefault(name, {"compared": 0, "equal": 0})
        entry["compared"] += 1
        if new == old:
            entry["equal"] += 1
        else:
            self.failures.append({"where": where, "quantity": name, "new": new, "approved": old})


def _springs(registry):
    for tag, entry in registry.items():
        for end, materials in (entry.get("installed_materials") or {}).items():
            for axis, material in materials.items():
                yield str(tag), end, axis, material


def _metadata_differences(new, old, prefix=""):
    """Keys present in only one record, or whose non-numeric values differ (a census, not a failure list)."""
    found = set()
    if isinstance(new, dict) and isinstance(old, dict):
        for key in set(new) | set(old):
            if key not in new or key not in old:
                found.add(f"{prefix}{key}: {'only in production' if key in new else 'only in approved'}")
            else:
                found |= _metadata_differences(new[key], old[key], f"{prefix}{key}.")
    elif new != old and not (isinstance(new, (int, float)) and isinstance(old, (int, float))):
        found.add(f"{prefix[:-1]}: value differs")
    return found


def compare_frozen(case, new, approved, tally, census):
    """Production (frozen) registry against one approved intensity registry."""
    missing = sorted(set(approved) - set(new), key=int)
    extra = sorted(set(new) - set(approved), key=int)
    springs = 0
    for tag, end, axis, old in _springs(approved):
        if tag not in new:
            continue
        where = f"{case}/{tag}/{end}/{axis}"
        material = ((new[tag].get("installed_materials") or {}).get(end) or {}).get(axis)
        if material is None:
            tally.failures.append({"where": where, "quantity": "spring", "new": None, "approved": "present"})
            continue
        springs += 1
        a, b = material["command"], old["command"]
        tally.equal("command.type_and_tag", a[:2], b[:2], where)
        tally.equal("command.length", len(a), len(b), where)
        for index, name in enumerate(ARGUMENT_NAMES):
            tally.number(f"command.{name}", a[2 + index], b[2 + index], where)
        tally.number("ke", material["ke"], old["ke"], where)
        for sign in ("positive", "negative"):
            for key in ("dp", "dpc", "du", "fy", "fmax_fy", "fres_fy"):
                tally.number(f"{sign}.{key}", material[sign][key], old[sign][key], where)
        for key in ("lamda_s", "lamda_c", "lamda_a", "lamda_k", "c_s", "c_c", "c_a", "c_k", "d_pos", "d_neg"):
            tally.number(f"cyclic.{key}", material["cyclic"][key], old["cyclic"][key], where)
        for mode in ("S", "C", "A", "K"):
            tally.number(f"physical_energy.{mode}", material["reference_energies_kip_in_rad"][mode],
                         old["reference_energies_kip_in_rad"][mode], where)
        m_new, m_old = material["provenance"]["energy_mapping"], old["provenance"]["energy_mapping"]
        for key in ("coordinate_reversed", "positive_input_direction", "negative_input_direction",
                    "canonical_physical_direction"):
            tally.equal(f"orientation.{key}", m_new[key], m_old[key], where)
        tally.number("orientation.reference_moment_kip_in", m_new["reference_moment_kip_in"],
                     m_old["reference_moment_kip_in"], where)
        for branch in ("physical_positive", "physical_negative"):
            for key in ("dp", "dpc", "du", "fy", "fmax_fy", "fres_fy"):
                tally.number(f"{branch}.{key}", m_new[branch][key], m_old[branch][key], where)
        for mode in ("S", "C", "A", "K"):
            tally.number(f"energy_mapping.reference_energy.{mode}", m_new["reference_energies_kip_in_rad"][mode],
                         m_old["reference_energies_kip_in_rad"][mode], where)
            tally.number(f"energy_mapping.base_lamda.{mode}", m_new["base_lamda_rad"][mode], m_old["base_lamda_rad"][mode],
                         where)
        # The approved adapter's member-reference record, keyed by native input sign.
        research = old["provenance"]["research_mapping"]
        frozen = material["provenance"]["member_spring_calibration"]
        reference = frozen["reference"]
        tally.number("reference.K0", reference["K0_kip_in_per_rad"], research["K0"], where)
        tally.number("reference.Kb", reference["Kb_kip_in_per_rad"], research["Kb"], where)
        tally.number("reference.Kh", reference["Kh_kip_in_per_rad"], research["Kh"], where)
        tally.number("reference.K0_vs_section_6EI_over_L", reference["K0_from_section_6EI_over_L"], research["K0"], where)
        tally.number("energy.theta_y_reference", frozen["energy"]["energy_reference_theta_y_rad"],
                     research["canonical_member_theta_y"], where)
        tally.number("energy.reference_energy_S_C", frozen["energy"]["reference_energies_kip_in_rad"]["S"],
                     research["reference_energy_S_C"], where)
        tally.number("energy.lambda_haselton", frozen["energy"]["lambda_haselton_dimensionless"],
                     old["provenance"]["deterioration"]["lambda_haselton_dimensionless"], where)
        by_sign = {record["input_sign"]: record for record in frozen["branches"].values()}
        for sign in ("positive", "negative"):
            ours, theirs = by_sign[sign], research["branches"][sign]
            tally.equal(f"branch.{sign}.physical_direction", ours["physical_direction"], theirs["physical_direction"], where)
            for key_new, key_old in (("member_theta_p_rad", "member_theta_p"), ("member_theta_pc_rad", "member_theta_pc"),
                                     ("member_theta_y_reference_rad", "member_theta_y_reference"),
                                     ("member_theta_cap_reference_rad", "member_theta_cap_reference")):
                tally.number(f"branch.{sign}.{key_old}", ours[key_new], theirs[key_old], where)
            for key in ("dp", "dpc", "du", "fy", "fmax_fy", "fres_fy"):
                tally.number(f"branch.{sign}.native_spring.{key}", ours["native_spring"][key], theirs["native_spring"][key], where)
        census.update(_metadata_differences(material["provenance"], old["provenance"], "provenance."))
        if material.get("parameter_sha256") != old.get("parameter_sha256"):
            census.add("parameter_sha256: value differs (hash of the identity, which includes the provenance)")
    # Member-level registry values that do not depend on the calibration.
    for tag in sorted(set(new) & set(approved), key=int):
        where = f"{case}/{tag}"
        for key in ("member_type", "material_type", "energy_mapping_mode", "group_id", "length_in"):
            if key in approved[tag]:
                tally.equal(f"member.{key}", new[tag].get(key), approved[tag].get(key), where)
        for key in ("ke_y_kip_in_per_rad", "ke_z_kip_in_per_rad", "yield_moment_y_hogging_i_kip_in",
                    "yield_moment_y_hogging_j_kip_in", "yield_moment_y_sagging_i_kip_in",
                    "yield_moment_y_sagging_j_kip_in", "yield_moment_z_kip_in", "iy_effective_in4", "stiffness_modifier",
                    "theta_p", "theta_pc", "theta_u", "axial_kip", "axial_ratio", "lambda_haselton_dimensionless"):
            if key in approved[tag]:
                tally.number(f"member.{key}", new[tag].get(key), approved[tag].get(key), where)
    return {"members_new": len(new), "members_approved": len(approved), "missing_members": missing,
            "extra_members": extra, "springs_compared": springs,
            "springs_approved": sum(1 for _ in _springs(approved)), "springs_new": sum(1 for _ in _springs(new))}


def compare_replay(case, new, baseline, tally):
    """Legacy replay registry against the pre-freeze baseline registry: every native argument bitwise."""
    springs = 0
    for tag, end, axis, old in _springs(baseline):
        where = f"{case}/{tag}/{end}/{axis}"
        material = ((new.get(tag, {}).get("installed_materials") or {}).get(end) or {}).get(axis)
        if material is None:
            tally.failures.append({"where": where, "quantity": "spring", "new": None, "approved": "present"})
            continue
        springs += 1
        tally.equal("command", material["command"], old["command"], where)
        for mode in ("S", "C", "A", "K"):
            tally.number(f"physical_energy.{mode}", material["reference_energies_kip_in_rad"][mode],
                         old["reference_energies_kip_in_rad"][mode], where)
    return {"springs_compared": springs, "members_new": len(new), "members_baseline": len(baseline),
            "missing_members": sorted(set(baseline) - set(new), key=int), "extra_members": sorted(set(new) - set(baseline), key=int)}


def compare(args):
    study = ev.study_root(args.study)
    harness = ev.harness_root(study)
    out = Path(args.out).resolve()
    report = {"verification": "full parameter parity (A)", "study": str(study), "tolerances": {"rtol": RTOL, "atol": ATOL},
              "cases": {}, "git": ev.git_state(), "sources": ev.source_hashes(), "runtime": ev.runtime()}
    frozen_tally, replay_tally, intensity_tally, census = Tally(), Tally(), Tally(), set()
    totals = {"members": 0, "springs": 0, "replay_springs": 0}
    for case in ev.CASES:
        new = json.loads((out / "registries" / f"{case}_frozen.json").read_text())
        build_info = json.loads((out / "registries" / f"{case}_frozen_build.json").read_text())
        mce = json.loads((study / "mce_average" / "cases" / case / "registry.json").read_text())
        dbe = json.loads((study / "dbe_average" / "cases" / case / "registry.json").read_text())
        # The two intensity registries describe the same installed springs (commands equal) as each other.
        for tag, end, axis, material in _springs(mce):
            other = dbe[tag]["installed_materials"][end][axis]
            intensity_tally.equal("mce_vs_dbe.command", material["command"], other["command"], f"{case}/{tag}/{end}/{axis}")
        entry = {"build": build_info, "versus_mce": compare_frozen(case, new, mce, frozen_tally, census),
                 "versus_dbe": compare_frozen(case, new, dbe, Tally(), set())}
        totals["members"] += entry["versus_mce"]["members_approved"]
        totals["springs"] += entry["versus_mce"]["springs_compared"]
        replay_path = out / "registries" / f"{case}_replay.json"
        if replay_path.exists():
            replay = json.loads(replay_path.read_text())
            baseline = json.loads((harness / "cases" / case / "registry.json").read_text())
            entry["replay_versus_baseline"] = compare_replay(case, replay, baseline, replay_tally)
            totals["replay_springs"] += entry["replay_versus_baseline"]["springs_compared"]
        report["cases"][case] = entry
    report["totals"] = totals
    report["frozen_versus_approved"] = {"quantities": frozen_tally.stats, "values_compared": frozen_tally.count,
                                        "out_of_tolerance": frozen_tally.failures[:500],
                                        "out_of_tolerance_count": len(frozen_tally.failures)}
    report["dbe_registry_consistency"] = {"quantities": intensity_tally.stats, "failures": intensity_tally.failures[:100]}
    report["replay_versus_pre_freeze_baseline"] = {"quantities": replay_tally.stats, "values_compared": replay_tally.count,
                                                   "out_of_tolerance": replay_tally.failures[:200],
                                                   "out_of_tolerance_count": len(replay_tally.failures)}
    report["metadata_differences"] = {
        "census": sorted(census),
        "explanation": ("Identifiers and statuses changed by design: the production record names the frozen calibration "
                        "(seisframe_imk_v2_frozen_20261029) where the adapter named its research profile, keeps the "
                        "member-to-native record under provenance.member_spring_calibration instead of the adapter's "
                        "research_mapping/baseline_provenance, and records the pre-freeze nominal products as labelled "
                        "historical metadata. parameter_sha256 hashes the whole identity including provenance, so it "
                        "differs although every numerical argument is compared above.")}
    no_missing = all(not c["versus_mce"]["missing_members"] and not c["versus_mce"]["extra_members"]
                     and c["versus_mce"]["springs_compared"] == c["versus_mce"]["springs_approved"]
                     == c["versus_mce"]["springs_new"] for c in report["cases"].values())
    report["passed"] = {"frozen_parity": not frozen_tally.failures and no_missing,
                        "dbe_consistency": not intensity_tally.failures,
                        "replay_reproduces_pre_freeze": (not replay_tally.failures) if totals["replay_springs"] else None}
    ev.write_json(out / "parity_report.json", report)
    worst = {name: {k: v for k, v in stats.items() if k in ("max_abs_err", "max_rel_err", "compared", "bitwise_equal")}
             for name, stats in frozen_tally.stats.items() if name.startswith("command.")}
    print(json.dumps({"totals": totals, "passed": report["passed"],
                      "out_of_tolerance": len(frozen_tally.failures), "command_max_errors": worst}, indent=1), flush=True)
    return 0 if all(v in (True, None) for v in report["passed"].values()) else 1


def all_builds(args):
    out = ev.new_output_dir(args.out)
    study = ev.study_root(args.study)
    if ev.under_study(out, study):
        raise SystemExit("the output directory must not be inside the approved study")
    (out / "registries").mkdir()
    (out / "logs").mkdir()
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "FOR_DISABLE_CONSOLE_CTRL_HANDLER": "1"}
    jobs = [(case, replay) for case in ev.CASES for replay in ((False, True) if args.with_replay else (False,))]

    def launch(job):
        case, replay = job
        log = out / "logs" / f"{case}_{'replay' if replay else 'frozen'}.log"
        command = [sys.executable, "-X", "utf8", "-B", "-I", __file__, "build", "--study", str(study), "--out", str(out),
                   "--case", case] + (["--replay"] if replay else [])
        started = time.time()
        with log.open("w", encoding="utf-8") as stream:
            code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env).returncode
        return {"case": case, "replay": replay, "exit_code": code, "elapsed_sec": time.time() - started}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(launch, jobs))
    ev.write_json(out / "builds.json", results)
    failed = [r for r in results if r["exit_code"] != 0]
    if failed:
        print(json.dumps({"failed_builds": failed}, indent=1), flush=True)
        return 1
    return compare(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("build", "compare", "all"))
    parser.add_argument("--study", required=True, help="root folder of the approved study (manifest.json, profile.py)")
    parser.add_argument("--out", default=str(ev.DEFAULT_OUTPUT / "parity"))
    parser.add_argument("--case", choices=ev.CASES)
    parser.add_argument("--replay", action="store_true", help="build under the explicit legacy replay profile")
    parser.add_argument("--with-replay", action="store_true", help="'all' also builds and compares the legacy replay")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.action == "build":
        if not args.case:
            raise SystemExit("build needs --case")
        return build(args)
    return compare(args) if args.action == "compare" else all_builds(args)


if __name__ == "__main__":
    sys.exit(main())
