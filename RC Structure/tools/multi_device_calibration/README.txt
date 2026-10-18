V3 RESULTS / V2 WEIGHTED SEISMIC INTENSITY CALIBRATION

Use START_HERE.txt for the complete school sequence and short Python commands.
No PowerShell execution-policy change is needed for those commands.

Purpose and preserved settings
This selects ground-motion intensity using the current V2 research hinge model.
It does not fit IMK deterioration parameters or release a training dataset.
V3 refers to the design-verification batch, not a new constitutive model.

Metric                              Reference       Weight
Fraction of hinge ends yielded      10%             45%
Peak hinge rotation-demand proxy    0.25            35%
Peak resultant interstory drift     1.25%            15%
Peak resultant roof drift           1.5%              5%

Score = sum(weight * measured value / reference); target 1.0. Ratios are uncapped,
so one metric can compensate for another. Drift references may be exceeded;
they are not acceptance caps. The separate existing collapse and solver checks
still apply. The damage proxy is plastic rotation demand / theta_p, not fatigue
damage or measured energy deterioration. The old is_inelastic label stays separate.

One simultaneous X/Y pair per eligible design, selected reproducibly from the
54 usable peer_mle_all pairs, with a maximum of 15,000 points per component.
Seed 20261006 shuffles the pool; pairs repeat only after the pool is used.
The same raw acceleration multiplier applies to both components. This multiplier
is not spectral alpha or a uniform target spectral acceleration across buildings.
Bounds: 1.0 to 3.5, initial 1.0, at most 7 full-record trials per design/pair,
relative bracket tolerance 0.1, 3,600 seconds per trial, one worker per device.
The result is the smallest passing scale actually tested, not a global optimum.
A design already passing at 1.0 is not reduced below 1.0. An unmet target at 3.5
is retained as unmet. One hundred eligible cases can require up to 700 NTHAs.
Use first-search timings and storage to assess throughput; no school-day runtime
or whole-batch disk/memory requirement is guaranteed by these preparation checks.

What is installed
The V3 pilot is separate from v2WeightedCalibration100. The earlier setup identity
and prepared plans are preserved. Model, Design, Analysis, Loads and Data_Generation
implementations are unchanged by this preparation. The expected engineering
source is db7ecfe16304967c38c6687652604bfb9d5b81c4ea559f9568e25ff34d47aba3,
from the repairs committed at e5c4538c. The updated setup checks the complete
calibration source, runtime, records, roster, baseline and pilot files.
Setup hashes for launchers and the verification roster/baseline normalize UTF-8
BOM and CRLF/LF for Git checkout portability. Motion files and frozen plan inputs
remain bound to their exact bytes.

Using Git: commit/push this new pilot and its tests, then pull on each device.
Copy the bundled Ground_Motions folder as well; processed accelerations are not
in Git. The expected Python/library versions are in school_setup_identity.json.
Do not copy the old calibration overlay over the current V3 model repairs.

Using the ZIP: extract its complete repository_overlay to a short local path
and use that as a separate working copy with the matching OpPy environment.
Keep the setup and saved results together. The ZIP contains current source and
motions, but no Python environment and no forthcoming school verification results.

Preparation and eligibility
Run inventory only after all four verification processes finish and results are
copied. All 100 original IDs are requested. No case is silently resampled, repaired
or declared passing. A deliberate eligible subset requires --allow-excluded.
The diagnostic admits fully qualified designs or designs whose only open item
is demands.torsional_irregularity (M1), with zero failures, consistent qualification
records, passing gravity/modal stage and matching source/profile identities.
M1 remains open, original acceptance is preserved, and training_release stays false.
The local case_0036 budget24 investigation is not a replacement for its school
16-evaluation result. Tomorrow's saved evidence determines actual eligibility.

Build one plan and copy it with ALL four complete design roots to all devices.
The eligible roster, source, runtime, design/result files, records and assignments
are frozen. The four calibration shards are balanced over the eligible sorted
roster; they need not match each device's earlier design-verification membership.
If all 100 qualify, there are 25 searches per device. Never infer full completion
from a plan or from the word designed in a result file.

Recovery and collection
Fresh output roots only: default outputs/wcalV3_r1. A new attempt can use
school.py prepare --run-name NAME; then pass the resulting --plan path to every
run and collect command. Never change a plan or rescore saved evidence in place.
Inventory and preparation refuse to rewrite the inputs of an existing plan.
Concurrent owners of a shard are blocked locally; do not assign the same device
number to two machines. Never run design and calibration against a changing
shared checkout, and do not write active outputs to a network/OneDrive folder.

A first search that fails or misses target stops continuation, including resume.
A later unusable trial remains a failed search even if an earlier trial passed.
Stop/timeout logs and partial histories are retained. Interrupted searches are
not retried by resume and remain unfinished for collection. Retrying requires a
new plan/output root after review. For a stale lock, verify the recorded host/PID
has stopped before removing it; do not delete active locks or failed evidence.
--summarize-only reads saved progress. --allow-partial on collection reports an
incomplete return without treating missing searches as successes.

Preserve every trial folder, NPZ, CSV, recorder, manifest and log. Collection checks
plan/job identities and artifact hashes and recomputes usable response scores.
A complete collection can contain unmet targets or unusable histories; the summary
reports these separately. No collection operation grants training eligibility.

Live checks still needed
Tomorrow's full verification inventory, one frozen master plan, preflight on the
actual four school devices and each device's first full-record search remain to
be completed. Preparation tests and local saved-design checks are not those runs.
Physical IMK calibration on the selected exterior connection remains separate.

Most-damaged beam hinge export
The school run and collection commands automatically export graphs after their
analysis/collection step, including a stopped search's earlier usable trials.
Preflight and summary commands do not export or launch analyses. If analysis and
plotting both fail, the original analysis failure remains the reported failure.
An export failure after successful analysis is also reported; saved results are
preserved and graphs can be regenerated using school.py plots.

Selection uses the maximum saved calibration damage ratio among BEAM hinge ends,
not maximum moment or maximum rotation across the structure. The selected hinge
has one shared damage proxy based on the maximum absolute rotation over both
bending axes and its saved theta_y_spring and theta_p. Both local y and z
moment/rotation pairs are shown without changing signs or subtracting gravity.
Graph moments are kip-ft; exported CSV moments retain the source's kip-in.
Exact damage ties choose the smallest hinge tag and list all ties in JSON.
All-zero beam damage is explicitly labeled as a representative tied beam hinge.
Missing beam hinges or ambiguous member types are reported; there is no column
fallback. The structure-wide peak-damage metric and calibration score remain
unchanged and are retained separately in each plot's JSON provenance.
Use school.py plots --member-scope all only for the previous all-member selection.
Default beam plots use a new output folder, preserving earlier all-member plots.

Graphs use full-precision scheduled snapshots, checked against the scored arrays
and committed-state counts. They are not the full adaptive-substep recorders.
Incomplete, unusable, mismatched or changed trial evidence is not plotted as a
valid response. The PNG, CSV and metadata live OUTSIDE hashed device/analysis
folders so original evidence and collection checksums remain unchanged.
Per-device graph reports and collection reports list exports, skips and errors.
The historical example bundled with the ZIP illustrates the format only;
it is not a result from the forthcoming V3 calibration batch.

PowerShell alternatives
The Device1.ps1 ... Device4.ps1, Prepare-Plan.ps1 and Collect-Results.ps1 wrappers
offer the same default setup. Run from the repository root and provide -Python
with the active OpPy interpreter. Prepare-Plan supports -InventoryOnly and
-AllowExcludedCases; Run-Device supports -PreflightOnly, -FirstJobOnly and -Resume.
Use the direct school.py commands if the terminal blocks .ps1 scripts.
