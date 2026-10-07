# School weighted seismic calibration

Prepared for fresh C–D, column-fc 6–10 ksi design verification results. This is an intensity diagnostic using the existing research hinge model. It does not fit physical IMK deterioration parameters or release training data.

## What is frozen

One simultaneous X/Y ground-motion pair per eligible design; both components receive the same raw acceleration multiplier. The 54 usable `peer_mle_all` pairs are shuffled reproducibly (seed 20261006), used once before repeating, and assigned across four disjoint device shards. All 100 eligible designs produce 25 searches per device. The plan is built once after the new designs arrive and copied unchanged to every device.

| Metric | Reference | Weight |
| --- | ---: | ---: |
| Fraction of hinge ends yielded | 10% | 45% |
| Peak hinge rotation-demand damage proxy | 0.25 | 35% |
| Peak resultant interstory drift | 1.25% | 15% |
| Peak resultant roof drift | 1.5% | 5% |

Score = sum(weight × metric/reference), target 1.0. Ratios are uncapped and compensate for each other; drift references may be exceeded. This does not guarantee yielding at every story or cyclic damage coverage. The damage metric is plastic rotation demand divided by theta_p, not fatigue damage or dissipated-energy deterioration. Existing solver, complete-history, recorder-agreement and collapse checks still apply. The old `is_inelastic` label is retained separately.

Default search: raw factors 1.0–3.5, starting at 1.0, at most seven full-record trials per design/pair, relative bracket tolerance 0.1, one-hour timeout per trial, one worker per device. The reported scale is the smallest passing scale actually tested, not a proven global optimum. These are raw multipliers, not spectral alpha. A 100-design batch can require up to 700 NTHAs; completion time depends on the first full-record searches.

## 1. Install the prepared files on all four devices

Use the existing project checkout on each device. Copy the package's `repository_overlay` contents into that checkout, preserving directories and replacing the corresponding prepared source files. Preserve unrelated local work first; do not merge alternative model edits into this frozen batch. The overlay includes the current source files, launchers, record metadata, and 108 processed acceleration files. It includes no design-result folders. Keep the checkout path short (for example `C:\SeisFrame`); do not run analyses inside a deeply nested extracted ZIP folder. Windows and recorder path-length limits can otherwise stop output creation.

Activate the same OpPy environment used for design verification. The setup check compares Python, OpenSees, NumPy, SciPy and matplotlib versions, OS/architecture, the source snapshot, launchers and record hashes. See `school_setup_identity.json` for exact versions. An environment mismatch must be resolved before launch; do not bypass the check or install arbitrary latest versions.

In a **PowerShell PyCharm terminal at the repository root**, on every device:

```powershell
$cal = '.\RC Structure\pilots\v2WeightedCalibration100'
& "$cal\Test-Setup.ps1" -Python python
```

If Python is not the active OpPy interpreter, substitute its full path, for example `-Python 'C:\Users\YOUR_USER\anaconda3\envs\OpPy\python.exe'`. The repository path and user name can differ between machines. If the local terminal blocks `.ps1` scripts, use the supplied `DIRECT_PYTHON_COMMANDS.txt` without changing school execution policy.

## 2. Receive and inventory tomorrow's designs on one coordinator

Complete design verification first. Copy **all four complete output roots**, including `design.json`, `result.json` and supporting evidence, into the coordinator's `RC Structure\outputs`:

```text
v2ResearchPilot100_CD_shard_01
v2ResearchPilot100_CD_shard_02
v2ResearchPilot100_CD_shard_03
v2ResearchPilot100_CD_shard_04
```

Run this without launching an analysis:

```powershell
& "$cal\Prepare-Plan.ps1" -Python python -InventoryOnly
```

Review `RC Structure\pilots\v2WeightedCalibration100\prepared\wcal100_CD_Fc6_r1\design_inventory.json`. Every requested case is eligible or explicitly excluded. Missing/duplicate cases, stale source identities, failed design checks, wrong profiles and incomplete/failed gravity-modal verification are refused.

The launcher declares an **open-M1 diagnostic**: the sole open `demands.torsional_irregularity` check may remain. It does not assert M1 or change original qualification. Fully qualified designs also qualify. To require fully qualified designs only, add `-RequireQualified` to inventory and preparation. All runs retain `training_release=false`.

Once reviewed, build the master plan **once**:

```powershell
& "$cal\Prepare-Plan.ps1" -Python python
```

If any cases are excluded, this stops with the inventory saved. To deliberately run only its listed eligible subset, repeat with `-AllowExcludedCases`. Never relabel failed or stale designs as accepted. For nondefault folders, supply `-DesignRoots @('outputs/root1','outputs/root2','outputs/root3','outputs/root4')`; paths must remain inside `RC Structure`. Changes to budget or run name are made at preparation, before freezing. A new run name requires passing its corresponding `-Plan` to device and collection scripts.

Copy the entire generated `prepared\wcal100_CD_Fc6_r1` folder and **all four design roots** to the same relative locations in every device checkout. The plan includes source, runtime, roster, design, result and motion identities. It must not be rebuilt independently on each device. Source fixes after design verification require new matching designs and a new calibration plan.

## 3. Preflight and run each device

These read-only checks must pass on their respective devices:

```powershell
# Device 1
& "$cal\Device1.ps1" -Python python -PreflightOnly
# Device 2
& "$cal\Device2.ps1" -Python python -PreflightOnly
# Device 3
& "$cal\Device3.ps1" -Python python -PreflightOnly
# Device 4
& "$cal\Device4.ps1" -Python python -PreflightOnly
```

Run **only the command for that device**:

```powershell
# Device 1
& "$cal\Device1.ps1" -Python python
# Device 2
& "$cal\Device2.ps1" -Python python
# Device 3
& "$cal\Device3.ps1" -Python python
# Device 4
& "$cal\Device4.ps1" -Python python
```

Each command first runs that device's first assigned full-record search. If it completes successfully and reaches the weighted target, the launcher continues through its remaining jobs. An unusable first search or unmet target stops the launcher for review. This is a full search, not a shortened excitation. Use `-FirstJobOnly` to stop after it regardless of outcome.

Keep machines powered, awake and connected to their local output disks. Do not let two machines own the same device shard. Avoid running design verification concurrently with calibration on the same checkout. Trial artifacts can be large: retain full histories and ensure enough disk space based on the first searches; the record library size is not an estimate of result storage.

Completed searches survive restart. After reviewing a stop, use the matching device command with `-Resume` to verify existing evidence and continue remaining jobs. `-SummarizeOnly` reads saved progress without running NTHA. An interrupted/timeout trial is unusable and retained; it is never silently rerun or overwritten. Interrupted searches remain unfinished for collection, even when a stop manifest was written. Retrying one, or recovering a search without its final manifest, requires a new plan/output root after review. A stale lock after a hard crash must be removed only after confirming the recorded host/PID is stopped.

Exit codes: 0 = executed searches reached the target; 1 = error or unusable evidence; 2 = usable completed searches without every target reached; 130 = interrupted. A later unusable trial remains a failure even if an earlier trial passed. Preserve logs and review the reported outcomes before further use.

## 4. Collect complete device folders

After processes stop, copy each complete folder (including every trial, NPZ/CSV, raw recorder, manifest and log) from:

```text
RC Structure\outputs\wcal100_CD_Fc6_r1\device_01
RC Structure\outputs\wcal100_CD_Fc6_r1\device_02
RC Structure\outputs\wcal100_CD_Fc6_r1\device_03
RC Structure\outputs\wcal100_CD_Fc6_r1\device_04
```

Place them under the same output root on the coordinator, retaining its frozen plan and prepared folder. Then:

```powershell
& "$cal\Collect-Results.ps1" -Python python
```

The collector verifies copied artifacts and recomputes usable trial metrics and scores. It writes `calibration_summary.json` and `calibration_summary.csv` in `RC Structure\outputs\wcal100_CD_Fc6_r1_collection`. To inspect a deliberately incomplete return, use `-AllowPartial`; missing searches remain explicit. A complete collection means all evidence was collected, not that all targets passed. Separate fields report target success and unusable trials. Collection never grants training eligibility.

## Validation and remaining live check

The prepared workflow has focused score, design eligibility, frozen-plan, four-shard assignment, resume/lock, timeout/interruption and portable collection tests. A genuine saved diagnostic was also accepted by the current trial assessor without rerunning its analysis. No new 100-case batch has been launched. The fresh design inventory and each school's full-record first search remain the final live checks. This preparation leaves the existing design-model fingerprint unchanged.
