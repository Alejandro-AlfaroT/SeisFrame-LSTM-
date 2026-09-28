"""
Design/Compare_SAP2000.py
=========================

Compare SAP2000 results against the design record's own numbers, per case.

Workflow
--------
1. Export_SAP2000.py writes <case>_frame.$2k and <case>_frame_targets.json.
2. Import the $2k into SAP2000, run all cases, run concrete design.
3. Export these tables to CSV (Display > Show Tables > File > Export):
     Modal Periods And Frequencies          -> modal.csv
     Joint Displacements                    -> joints.csv
     Base Reactions                         -> reactions.csv
     Concrete Design 1 - Column Summary     -> col_design.csv      (optional)
     Concrete Design 2 - Beam Summary       -> beam_design.csv     (optional)
   Export with the default SAP column headers, Kip-in units.
4. Run:
     python Design/Compare_SAP2000.py <case>_frame_targets.json --sap-dir <dir with the csvs>

What is compared, and which assertion each result speaks to
-----------------------------------------------------------
  period T1               model stiffness/mass       (sanity)
  base shear under EQX/EQY applied ELF               (sanity; should be exact)
  story drift, DRIFT_X/Y  elastic + Cd/Ie design     drift screen
  column / beam DCR       SAP concrete design        qualification.strength_model_verification
  base axials, DEAD       floor transfer             floor.independent_hand_verification (slab variant)

SAP's CSV headers vary slightly by version and by whether you export from a
selection. The reader is case-insensitive and tolerant of spaces, but if a
column is not found it says which one and stops rather than guessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_csv(path):
    """SAP CSVs have a title row, a header row, sometimes a units row."""
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = list(csv.reader(file))
    header_index = next((i for i, row in enumerate(reader)
                         if any(cell.strip().lower() in ("joint", "frame", "outputcase", "output case", "stepnum", "period") for cell in row)), 0)
    header = [cell.strip().lower().replace(" ", "") for cell in reader[header_index]]
    for row in reader[header_index + 1:]:
        if not any(cell.strip() for cell in row):
            continue
        if all(cell.strip().lower() in ("text", "kip", "in", "sec", "unitless", "rad", "in2", "in^2", "cyc/sec", "rad/sec", "rad2/sec2", "kip-in", "") for cell in row):
            continue
        rows.append(dict(zip(header, [cell.strip() for cell in row])))
    return rows


def col(row, *names):
    for name in names:
        key = name.lower().replace(" ", "")
        if key in row:
            return row[key]
    raise KeyError(f"None of {names} in columns {sorted(row)[:12]}...")


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def pct(a, b):
    return 100.0 * (a - b) / b if b else math.nan


def line(label, sap, ref, unit="", tol_pct=None):
    diff = pct(sap, ref) if ref else (0.0 if sap == 0 else math.nan)
    flag = ""
    status = "unavailable"
    if tol_pct is not None and all(math.isfinite(x) for x in (sap, ref, diff)):
        flag = "  ok" if abs(diff) <= tol_pct else "  ** %.0f%% tolerance" % tol_pct
        status = "pass" if abs(diff) <= tol_pct else "fail"
    else:
        flag = "  UNAVAILABLE (nonfinite value or missing comparison basis)"
    print("  %-40s SAP %12.4f   record %12.4f   %+7.2f%%%s %s" % (label, sap, ref, diff, flag, unit))
    return {"label": label, "status": status, "sap": sap, "reference": ref, "percent_difference": diff}


def compare(targets, sap_dir):
    sap_dir = Path(sap_dir)
    nf = len(targets["elf"]["story_forces_kip"])
    cd = float(targets["drift_screen"].get("cd", 5.5))
    ie = float(targets["drift_screen"].get("ie", math.nan))
    checks = []

    def report(*args, **kwargs):
        checks.append(line(*args, **kwargs))

    def unavailable(label, reason):
        print(f"  {label}: UNAVAILABLE -- {reason}")
        checks.append({"label": label, "status": "unavailable", "reason": reason})

    print("=" * 100)
    print("case %s  variant %s" % (targets["case"], targets["variant"]))
    print("=" * 100)

    # --- period ---------------------------------------------------------------
    modal = sap_dir / "modal.csv"
    if modal.exists():
        rows = read_csv(modal)
        periods = sorted((fnum(col(r, "Period")) for r in rows if col(r, "OutputCase", "Output Case").upper() == "MODAL"), reverse=True)
        if periods and targets.get("period_T1_sec"):
            print("\n[period]")
            report("T1 (s)", periods[0], float(targets["period_T1_sec"]), tol_pct=3)
            if len(periods) > 1:
                print("  SAP modes: " + ", ".join("%.3f" % p for p in periods[:4]))
        else:
            unavailable("T1", "MODAL rows or reference period missing")
    else:
        unavailable("T1", "modal.csv not found")

    # --- base shear -----------------------------------------------------------
    reactions = sap_dir / "reactions.csv"
    if reactions.exists():
        rows = read_csv(reactions)
        print("\n[base shear under applied ELF]  (should be exact: same forces)")
        for case, comp in (("EQX", "GlobalFX"), ("EQY", "GlobalFY")):
            for r in rows:
                if col(r, "OutputCase", "Output Case").upper() == case:
                    report("%s total %s (kip)" % (case, comp), abs(fnum(col(r, comp, comp.replace("Global", "")))),
                         float(targets["elf"]["base_shear_kip"]), tol_pct=0.5)
                    break
            else:
                unavailable(case, "base reaction row missing")
        print("\n[base axial under gravity -> floor transfer]")
        for case in ("DEAD_FLOOR", "SELF_WT", "LIVE"):
            for r in rows:
                if col(r, "OutputCase", "Output Case").upper() == case:
                    print("  %-40s SAP %12.2f kip" % (case + " total FZ", abs(fnum(col(r, "GlobalFZ", "FZ")))))
                    break
        totals = targets.get("floor_transfer_totals", {})
        for key in ("dead", "live"):
            if key in totals:
                print("  record transfer %-5s per floor: applied %.2f  beams %.2f  columns-direct %.2f  x %d floors" % (
                    key, totals[key].get("applied_kip", 0), totals[key].get("beam_kip", 0),
                    totals[key].get("column_direct_kip", 0), nf))
    else:
        unavailable("base shear", "reactions.csv not found")

    # --- story drift ----------------------------------------------------------
    joints = sap_dir / "joints.csv"
    if joints.exists():
        rows = read_csv(joints)
        print("\n[story drift]  elastic per story at governing node; design = Cd/Ie x elastic")
        by_case = {}
        for r in rows:
            case = col(r, "OutputCase", "Output Case").upper()
            by_case.setdefault(case, {})[int(fnum(col(r, "Joint")))] = (fnum(col(r, "U1")), fnum(col(r, "U2")))
        stories = targets["drift_screen"].get("stories") or []
        if not stories:
            unavailable("story drift", "reference stories missing")
        for st in stories:
            loc = st.get("location", "")
            try:
                story = int(loc.split(":")[1].split("/")[0])
                axis = loc.split("/")[-1].lower()
            except (IndexError, ValueError):
                unavailable("story drift", f"invalid reference location {loc!r}")
                continue
            case = "DRIFT_X" if axis == "x" else "DRIFT_Y"
            comp = 0 if axis == "x" else 1
            if case not in by_case:
                unavailable(loc, f"{case} joint results missing")
                continue
            gi, gj = (int(v) for v in st.get("governing_node", "0,0").split(","))
            # Joint tags are Model.Build_Model.node_tag(k, i, j); the geometry rides in the targets.
            geometry = targets["geometry"]
            per_floor = (int(geometry["num_bay_x"]) + 1) * (int(geometry["num_bay_y"]) + 1)
            n_top = story * per_floor + gj * (int(geometry["num_bay_x"]) + 1) + gi + 1
            n_bot = n_top - per_floor
            if n_top not in by_case[case] or n_bot not in by_case[case]:
                unavailable(loc, f"joint {n_top}/{n_bot} not in SAP export")
                continue
            elastic = abs(by_case[case][n_top][comp] - by_case[case][n_bot][comp])
            ref_elastic = float(st.get("elastic_drift_in", math.nan))
            report("story %d %s elastic drift (in)" % (story, axis.upper()), elastic, ref_elastic, tol_pct=5)
            ref_design = float(st.get("design_drift_in", math.nan))
            report("story %d %s design drift Cd/Ie (in)" % (story, axis.upper()),
                   elastic * cd / ie if math.isfinite(ie) and ie > 0 else math.nan, ref_design, tol_pct=5)
    else:
        unavailable("story drift", "joints.csv not found")

    # --- design DCR -----------------------------------------------------------
    dcr = targets.get("dcr") or {}
    for name, fname, ratio_cols in (("column", "col_design.csv", ("PMMRatio", "PMM Ratio", "Ratio")),
                                    ("beam", "beam_design.csv", ("Ratio", "FlexRatio", "PMMRatio"))):
        path = sap_dir / fname
        if not path.exists():
            unavailable(name + " DCR", fname + " not found")
            continue
        rows = read_csv(path)
        ratios = []
        for r in rows:
            for c in ratio_cols:
                key = c.lower().replace(" ", "")
                if key in r:
                    v = fnum(r[key])
                    if math.isfinite(v):
                        ratios.append(v)
                    break
        if ratios and len(ratios) != len(rows):
            unavailable(name + " DCR coverage", "some design rows have missing or nonfinite ratios")
        if ratios:
            print("\n[%s design DCR]  SAP max vs record's governing" % name)
            report("%s max utilization" % name, max(ratios), float(dcr.get(name, math.nan)), tol_pct=10)
            print("  SAP %s ratio: median %.3f  p90 %.3f  n=%d" % (name, sorted(ratios)[len(ratios) // 2],
                                                                 sorted(ratios)[int(0.9 * (len(ratios) - 1))], len(ratios)))
        else:
            unavailable(name + " DCR", "no finite dimensionless ratio field; steel areas and combination names are not ratios")

    print("\n[known differences to expect]")
    for note in targets.get("known_differences", []):
        print("  - " + note)
    status = "fail" if any(c["status"] == "fail" for c in checks) else (
        "incomplete" if not checks or any(c["status"] == "unavailable" for c in checks) else "pass")
    print(f"\nComparison status: {status}. Numerical comparison only; no engineering assertions are enabled.")
    return {"status": status, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", help="<case>_<variant>_targets.json from Export_SAP2000.py")
    parser.add_argument("--sap-dir", required=True, help="Directory holding the exported SAP CSV tables")
    args = parser.parse_args()
    targets = json.loads(Path(args.targets).read_text(encoding="utf-8"))
    result = compare(targets, args.sap_dir)
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
