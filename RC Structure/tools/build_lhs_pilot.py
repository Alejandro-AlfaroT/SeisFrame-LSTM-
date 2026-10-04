"""Build a Latin hypercube design pilot: screening manifests for Design/Verify_Designs.

Draws ``--count`` structures from the V2 population (Generate_Parameterized_Dataset.RANGES and SEISMIC_SITES)
by Latin hypercube sampling on the discrete levels, and writes

  screening_plan.json            every case, one manifest
  shard_NN_screening_plan.json   the same cases split into ``--shards`` manifests, one per machine; each shard
                                 has its own initial case, because a manifest run designs its initial case
                                 alone before the others
  sample.json                    the seed, the levels, and each case's level indices and unit-cube coordinates,
                                 so a later (adaptive) round can add points to the same space
  run_shard.ps1                  the two launches of one shard, with paths relative to the repository

Sampling (``latin_hypercube_discrete_v1``): each of the seven parameters gets one point in each of ``count``
equal strata of [0, 1), in an independent random order; a coordinate u maps to level floor(u * levels). The
marginal of every parameter is therefore as even as its level count allows. A draw that repeats a geometry
and site already taken is redrawn in its own stratum (recorded). Nothing is launched by this tool.

Usage:
  python tools/build_lhs_pilot.py --count 100 --seed 20261004 --shards 4 --name v2DesignPilot100 --out pilots/v2DesignPilot100
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Data_Generation"))

SAMPLING_METHOD = "latin_hypercube_discrete_v1"


def latin_hypercube(count, levels, seed):
    """``count`` points; levels = {parameter: (values...)}. Returns [{"u": {...}, "index": {...}, "values": {...}}]."""
    rng = random.Random(seed)
    columns = {}
    for name in levels:
        strata = list(range(count))
        rng.shuffle(strata)
        columns[name] = [(stratum + rng.random()) / count for stratum in strata]
    points, seen, redraws = [], set(), 0
    for row in range(count):
        for _attempt in range(1000):
            u = {name: columns[name][row] for name in levels}
            index = {name: min(int(u[name] * len(levels[name])), len(levels[name]) - 1) for name in levels}
            key = tuple(index[name] for name in levels)
            if key not in seen:
                break
            # a repeated structure: redraw the last coordinate inside its own stratum, then the next ones
            redraws += 1
            name = rng.choice(list(levels))
            stratum = int(columns[name][row] * count)
            columns[name][row] = (stratum + rng.random()) / count
        else:
            raise RuntimeError("could not draw distinct structures; the sample is too dense for the population")
        seen.add(key)
        points.append({"u": u, "index": index, "values": {name: levels[name][index[name]] for name in levels}})
    return points, redraws


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--name", required=True, help="Run name, used for the output root of every shard.")
    parser.add_argument("--out", type=Path, required=True, help="New directory for the manifests.")
    parser.add_argument("--max-section-iter", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4, help="Largest worker count a shard launch may use.")
    parser.add_argument("--stages", nargs="+", default=["design", "gravity_modal", "figures"])
    parser.add_argument("--profile", default="v2_nonlinear_flexure_screening_v1",
                        help="Analysis profile the designs are made under (Model/Analysis_Profile). A design made under one "
                             "profile is refused under another, so name the profile the response-history runs will use.")
    args = parser.parse_args()
    if args.count < 1 or args.shards < 1 or args.shards > args.count:
        raise SystemExit("count and shards must be positive, with at most one shard per case")

    import Structure_Parameters as sp
    from Generate_Parameterized_Dataset import (GEOMETRY_INCREMENT_FT, POPULATION_BASIS, RANGES, SEISMIC_SITES,
                                                feet_label)
    from Design.Screening_Manifest import MANIFEST_SCHEMA, load_manifest
    from Model.Analysis_Profile import PROFILES

    profile_id = args.profile
    if profile_id not in PROFILES:
        raise SystemExit(f"Unknown profile {profile_id!r}; known: {sorted(PROFILES)}")
    settings = PROFILES[profile_id]["settings"]
    levels = {**{key: tuple(values) for key, values in RANGES.items()}, "seismic_site": tuple(SEISMIC_SITES)}
    points, redraws = latin_hypercube(args.count, levels, args.seed)

    cases = []
    for number, point in enumerate(points, start=1):
        v = point["values"]
        name = (f"lhs_bx{v['num_bay_x']}_by{v['num_bay_y']}_s{v['num_floor']}_sh{feet_label(v['story_height_ft'])}"
                f"_bwx{feet_label(v['bay_x_width_ft'])}_bwy{feet_label(v['bay_y_width_ft'])}_{v['seismic_site']}")
        cases.append({"case_id": f"case_{number:04d}", "label": f"lhs_{number:03d}",
                      "num_bay_x": int(v["num_bay_x"]), "num_bay_y": int(v["num_bay_y"]), "num_floor": int(v["num_floor"]),
                      "bay_x_width_ft": v["bay_x_width_ft"], "bay_y_width_ft": v["bay_y_width_ft"],
                      "story_height_ft": v["story_height_ft"], "seismic_site": v["seismic_site"],
                      "geometry_name": name.replace(".", "p"),
                      "purpose": f"Latin hypercube point {number} of {args.count}, seed {args.seed}."})

    basis = sp.seismic_design_basis()
    sampling = {"method": SAMPLING_METHOD, "seed": args.seed, "count": args.count, "population_basis": POPULATION_BASIS,
                "parameter_order": list(levels), "levels": {key: list(values) for key, values in levels.items()},
                "level_counts": {key: len(values) for key, values in levels.items()},
                "discrete_mapping": "u in [0, 1) -> level floor(u * level_count); one point per stratum of width 1 / count",
                "redraws_for_repeated_structures": redraws,
                "extension_note": "a later round adds points to the same unit cube; keep this file with the dataset"}

    def manifest(case_list, label):
        return {
            "schema": MANIFEST_SCHEMA, "status": f"prepared_for_{args.name}_not_executed",
            "scope": ("Design pilot on a Latin hypercube sample of the V2 population: checks the automatic design across the "
                      "parameter space before generation. PROBE assertions; not a training dataset and not an acceptance."),
            "profile_id": profile_id,
            "design_basis": {"bay_and_story_height_increment_ft": GEOMETRY_INCREMENT_FT,
                             "risk_category": basis["risk_category"], "importance_factor": basis["importance_factor"],
                             **{key: [min(values), max(values)] for key, values in RANGES.items()},
                             "site_class": basis.get("site_class"), "population_basis": POPULATION_BASIS},
            "nonlinear_profile": {"element_formulation": settings["ELEMENT_FORMULATION"], "member_material": settings["IMK_MATERIAL_TYPE"],
                                  "apply_to_beams": settings["IMK_APPLY_TO_BEAMS"], "apply_to_columns": settings["IMK_APPLY_TO_COLUMNS"],
                                  "joint_model": settings["JOINT_MODEL"],
                                  "explicit_face_slip_interfaces": PROFILES[profile_id]["explicit_face_slip_interfaces"]},
            "execution": {"initial_case_id": case_list[0]["case_id"], "initial_workers": 1, "subsequent_max_workers": args.workers,
                          "max_section_iter": args.max_section_iter, "output_root": f"outputs/{args.name}_{label}",
                          "fresh_root_required": True,
                          "retry_policy": "No retries with a changed budget; a changed input, source or assumption needs a new output root."},
            "stages": list(args.stages),
            "acceptance_policy": {"production_acceptance": False,
                                  "note": "GENERATION_RELEASE_READY is not changed by this pilot; PROBE assertions certify nothing."},
            "sampling": {**sampling, "shard": label, "cases_in_this_manifest": len(case_list)},
            "cases": case_list,
        }

    args.out.mkdir(parents=True, exist_ok=False)
    written = {}

    def write(name, data):
        text = json.dumps(data, indent=1)
        (args.out / name).write_text(text, encoding="utf-8")
        written[name] = hashlib.sha256(text.encode("utf-8")).hexdigest()

    write("screening_plan.json", manifest(cases, "all"))
    # round-robin split, so every shard spans the whole sample rather than one corner of it
    for shard in range(args.shards):
        write(f"shard_{shard + 1:02d}_screening_plan.json", manifest(cases[shard::args.shards], f"shard_{shard + 1:02d}"))
    write("sample.json", {"sampling": sampling,
                          "points": [{"case_id": case["case_id"], "geometry_name": case["geometry_name"],
                                      "level_index": point["index"], "unit_cube": point["u"]}
                                     for case, point in zip(cases, points)]})
    stages = " ".join(args.stages)
    (args.out / "run_shard.ps1").write_text(f"""# {args.name}: one shard of the Latin hypercube design pilot on this machine.
# Usage, from anywhere:  .\\run_shard.ps1 -Shard 1 -ProbeDate 2026-10-05 [-Workers {args.workers}] [-Python python]
# The initial case runs alone, then the rest of the shard on the workers. Output: RC Structure\\outputs\\{args.name}_shard_NN
param([Parameter(Mandatory = $true)][int]$Shard, [Parameter(Mandatory = $true)][string]$ProbeDate,
      [int]$Workers = {args.workers}, [string]$Python = "python")
$rc = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$label = "shard_{{0:D2}}" -f $Shard
$manifest = Join-Path $PSScriptRoot "${{label}}_screening_plan.json"
$root = Join-Path $rc "outputs\\{args.name}_$label"
$initial = (Get-Content $manifest -Raw | ConvertFrom-Json).execution.initial_case_id
Set-Location $rc
& $Python -X utf8 -B "Design\\Verify_Designs.py" --manifest $manifest --case-ids $initial --probe-assertions --probe-date $ProbeDate --stages {stages} --output-root $root --workers 1
& $Python -X utf8 -B "Design\\Verify_Designs.py" --manifest $manifest --remaining --probe-assertions --probe-date $ProbeDate --stages {stages} --output-root $root --workers $Workers
""", encoding="utf-8")
    for name in written:
        load_manifest(args.out / name) if name.endswith("screening_plan.json") else None
    (args.out / "SHA256SUMS.json").write_text(json.dumps(written, indent=1), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "shards": args.shards, "redraws": redraws, "out": str(args.out),
                      "marginals": {key: {str(value): sum(1 for c in cases if c[key] == value) for value in values}
                                    for key, values in levels.items() if len(values) <= 6}}, indent=1))


if __name__ == "__main__":
    main()
