Column concrete revision (October 5, 2026): permitted grades are 6, 7, 8, 9 and 10 ksi, with a 6-ksi starting strength. Beam grades remain 5, 6 and 8 ksi. The design-source fingerprint changed; earlier designs and M1 evidence are historical and require verification under the revised source. Use the CD Fc6 launcher package with the refreshed baseline.

**v2DesignPilot100: Latin hypercube design pilot**

SDC C-D revision (October 5, 2026): the active 100-case manifests use 33 sdc_c, 34 sdc_d_low and 33 sdc_d_high cases. The seed, geometries, case IDs and four 25-case allocations are preserved; hazard assignments have changed. Use the new _CD output roots and revised launcher package on every device. Earlier results and qualification addenda do not establish acceptance of these revised cases. Historical E presets remain in the engineering site library for old records, but are excluded from generation and rejected by current screening manifests. The historical check5 manifest and run_check5.ps1 are not updated and are not commands for this revised population.

Purpose: check the automatic design across the V2 population before data generation. 100 structures, design
plus the gravity/modal build plus the figures. No ground motion. PROBE assertions: nothing here is accepted
and no release flag changes.

**Sample**

Latin hypercube on the discrete V2 levels (`latin_hypercube_discrete_v1`, seed 20261004), built by
`tools/build_lhs_pilot.py`. Seven parameters: bays in x and y (2 to 6), stories (4 to 9), story height
(12 to 18 ft), bay widths in x and y (18 to 30 ft), and the hazard preset (sdc_c, sdc_d_low, sdc_d_high). Every level of every parameter appears as evenly as 100 points allow (20 per bay count,
16 or 17 per story count, 33 C, 34 D-low and 33 D-high). `sample.json` keeps the seed, the levels and each case's
coordinates so a later adaptive round can add to the same space.

**Files**

| file | what |
|---|---|
| `screening_plan.json` | all 100 cases in one manifest |
| `shard_01` to `shard_04_screening_plan.json` | the same cases, 25 per machine, split round-robin so each shard spans the sample |
| `sample.json` | sampling record |
| `run_shard.ps1` | runs one shard on this machine |
| `SHA256SUMS.json` | hashes of the manifests |

**Run one shard on a machine**

From this folder, in PowerShell, with the OpPy environment active:

```
.\run_shard.ps1 -Shard 1 -ProbeDate 2026-10-05
```

Options: `-Workers 4` (the manifest allows at most 4) and `-Python <path to python.exe>`. The shard's first
case runs alone, then the rest on the workers. Results go to `RC Structure\outputs\v2DesignPilot100_CD_shard_NN`,
one folder per case plus `summary.md`, `summary.csv` and `summary.json` at the root. Use the same `-ProbeDate`
on every machine. The second launch (`--remaining`) only runs cases that have no result yet, so a stopped
shard can be continued by running the script again; this resume path has not been exercised on this pilot.

**Before running on another machine**

The machine needs this repository at the same source state as the one the manifests were built on (the
design identity includes the source hashes) and the OpPy environment (OpenSeesPy 3.8).

**Expected cost**

From the ten-case screens of 3 October: 4 to 30 minutes per design for small and medium floors, about
100 minutes for a 6 x 6 floor (the slab mesh refinement dominates). The sample has 25 cases with 20 or more
bays per floor, three of them 6 x 6. Budget roughly 60 to 100 machine-hours in total.

**Not covered before**

Earlier screens included SDC E. Those results are historical; the current C-D population requires the revised verification run.

**To rebuild with another size, seed or shard count** (the output directory must be new)

```
python tools/build_lhs_pilot.py --count 100 --seed 20261004 --shards 4 --name v2DesignPilot100_CD --out pilots/NEW_DIRECTORY
```
