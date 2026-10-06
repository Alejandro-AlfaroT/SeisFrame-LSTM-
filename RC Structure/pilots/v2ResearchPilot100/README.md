Column concrete revision (October 5, 2026): permitted grades are 6, 7, 8, 9 and 10 ksi, with a 6-ksi starting strength. Beam grades remain 5, 6 and 8 ksi. The design-source fingerprint changed; earlier designs and M1 evidence are historical and require verification under the revised source. Use the CD Fc6 launcher package with the refreshed baseline.

**v2ResearchPilot100: designs and response-history runs of the diagnostic research batch**

SDC C-D revision (October 5, 2026): the active 100-case manifests use 33 sdc_c, 34 sdc_d_low and 33 sdc_d_high cases. The seed, geometries, case IDs and four 25-case allocations are preserved; hazard assignments have changed. Use the new _CD output roots and revised launcher package on every device. Earlier results and qualification addenda do not establish acceptance of these revised cases. Historical E presets remain in the engineering site library for old records, but are excluded from generation and rejected by current screening manifests. The historical check5 manifest and run_check5.ps1 are not updated and are not commands for this revised population.

The same 100 Latin hypercube structures as `v2DesignPilot100` (seed 20261004), designed under the research
profile `v2_nonlinear_flexure_anchored_energy_research_v1`. A design made under one profile is refused under
another, so the designs that will be shaken must be made from these manifests, not from `v2DesignPilot100`.

This is a declared diagnostic batch. Production release remains closed. Original design records remain
unchanged; verified per-design M1 addenda can supply supplemental research qualification, recorded beside
the original state in every run. See the batch safeguards below for the required evidence options.

**Step 1: designs (per machine)**

```
.\run_shard.ps1 -Shard 1 -ProbeDate 2026-10-06
```

Same use as the design pilot: the shard's first case alone, then the rest on the workers; results in
`RC Structure\outputs\v2ResearchPilot100_CD_shard_NN`. Use one `-ProbeDate` on every machine. Copy every shard
folder back whole to the machine that builds the plan.

**Step 2: the run plan (once, on one machine)**

```
python Data_Generation/Run_Research_Batch.py --manifest pilots/v2ResearchPilot100/screening_plan.json --design-roots outputs/v2ResearchPilot100_CD_shard_01 outputs/v2ResearchPilot100_CD_shard_02 outputs/v2ResearchPilot100_CD_shard_03 outputs/v2ResearchPilot100_CD_shard_04 --output-root outputs/v2ResearchBatch100_CD --plan-only
```

This writes `outputs/v2ResearchBatch100_CD/ntha_plan.json`: for every structure its record pair (both horizontal
components, with the SHA-256 of each record file), its intensity group (33 at 0.5, 34 at 1.0 and 33 at 1.5
times the design spectrum at the reference period) and the SHA-256 of the design it will run. Copy
`ntha_plan.json` into the same output folder on every machine, together with the design folders.

**Step 3: runs (per machine)**

```
python Data_Generation/Run_Research_Batch.py --output-root outputs/v2ResearchBatch100_CD --design-roots outputs/v2ResearchPilot100_CD_shard_01 outputs/v2ResearchPilot100_CD_shard_02 outputs/v2ResearchPilot100_CD_shard_03 outputs/v2ResearchPilot100_CD_shard_04 --shard 1 --of 4 --workers 2
```

Each structure runs in its own process and claims its folder first. Running the command again continues the
shard: finished cases are skipped. A case that was interrupted keeps its `.claim` file and is NOT rerun
automatically; review and preserve its evidence, then retry in a new attempt output root. A failed analysis
keeps everything it wrote and is reported as `analysis_failed`.

**Step 4: merge and check (one machine)**

Copy the case folders of every machine into one `outputs/v2ResearchBatch100_CD` (they do not overlap), then

```
python Data_Generation/Run_Research_Batch.py --output-root outputs/v2ResearchBatch100_CD --summarize-only
```

writes `batch_status.csv` and `batch_status.json`: every case with its state, steps, peak drift, how many beam
and column springs yielded, open and failed design checks, run time and size. `training_eligible` requires
a completed, untruncated run with complete finite aligned histories and a complete effective research
checklist; it is not production acceptance or physical-model validation.

**Before running on another machine**

`python tools/source_fingerprint.py --check <baseline.json>` must report a match (source files, OpenSees and
library versions), and the machine needs the `Ground_Motions` folder: the run checks each record file against
the hash in the plan and stops on a difference.

**Record set**

Default `peer_mle_all`: 54 usable pairs of at most 15,000 points, so 46 of 100 structures reuse a pair. Runs
that share `record_group` share an earthquake and station and belong on one side of a train/test split.
`--record-set peer_strong_63` gives 63 pairs but includes 30,000-point records (about twice the run time).

**Mechanics check**

`--smoke-duration-sec 4` runs four seconds of the record and marks the run as a smoke test; it is never
training-eligible.

**Batch safeguards added October 4**

The runner now creates version 2 plans. They pin the design/result files, the complete generation/review
source identity and runtime versions. Source/environment mismatches stop execution before model build;
old version 1 plans require a new output root and are never silently upgraded. Final collection rejects
foreign-plan, mismatched-identity and unplanned case folders. Keep each device's allocation disjoint:
a claim file protects one shared folder, not separate copies on different computers.

To carry the scoped M1 qualification, supply BOTH options at plan creation and execution:

```
--m1-addendum-roots outputs/YOUR_M1_ADDENDA
--m1-review-roots outputs/YOUR_M1_REVIEWS
```

Each root contains case_NNNN folders. The review folder must include review.json, status.json and the full
pm65/pm129 JSON evidence. Copy the adopted pilots/v2ResearchPilot100/torsional_strength_assertion.json
unchanged with the source. Addenda are independently reconstructed from saved evidence before use; this
rechecks qualification and numerical records, without rerunning structural analysis or optimization.
Original-machine paths in historical addenda are treated only as labels; the supplied roots locate copies.
Do not edit the copied evidence or addenda to change those labels. Result files remain unchanged.

A plan pins each addendum's exact hash and effective qualification. Missing/altered addenda or conflicting
copies are errors. If no addendum was supplied, original qualification remains in force. Open or failed
checks make training_eligible false even when a declared diagnostic analysis completes. Production
acceptance remains false; scoped research checklist completion does not validate the physical model.

All scheduled full-rate hinge rows and positive, increasing commit counts are required for training
eligibility. This also excludes empty/short histories and nonfinite time arrays. Full independent
recorder/sign/units and physical-response review is still required on the final-baseline end-to-end screen.

The existing run_shard.ps1 still performs design/gravity/figures; run the M1 review/assertion separately
for every applicable design. run_check5.ps1 still names historical output roots: use fresh roots and the
explicit runner command for post-repair checks. Preserve failed and interrupted attempts; retry into a
new attempt root instead of deleting evidence. Include this README in device handoffs (it is currently
untracked). Refresh baseline_fingerprint.json after any generation/review tool change and recheck it on
all devices. Generation-only changes do not invalidate unchanged design-source identities.
