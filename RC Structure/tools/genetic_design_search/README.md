# Experimental genetic design search

`run.py` offers two engines, both calling the existing complete grouped candidate evaluator. The existing grouped and production runners continue to use their existing search.

- `--engine heuristic_v2` (the default, unchanged): population search, uniform crossover and failure-guided mutation (`Design/Genetic_Search.py`), described under "Search behavior" below.
- `--engine generation_v3`: the bounded serial generation controller (`Design/Genetic_Generation.py`), described under "Generation controller (v3)".

## Generation controller (v3)

Modules:

| Module | Role |
|---|---|
| `Genetic_Diagnostics.py` | The normalized diagnostic adapter, run while the evaluation is in memory. |
| `Genetic_Operators.py` | The six operators. |
| `Genetic_Population.py` | The versioned selection policy and the archive views. |
| `Genetic_Generation.py` | The controller. |

What it does:

- **Fixed per run:** one case, one grouping, one longitudinal steel grade and one evaluation basis. The genes and the profile repair are those of `DesignSpace`. The inner designer selects every cage.
- **Every distinct child is fully evaluated** and assessed by `Code_Check_Contract.assess_evaluation` before it is ranked or stored.
- **Mutation.** Parents are drawn uniformly from the working population. With probability `--guided-probability` (0.75), the controller routes the parent's own diagnostics:
  - **H1, strong-column / weak-beam:** connected column ties, with size and grade as separate alternatives.
  - **H2, member strength:** beam depth, width and grade; column size and grade. For column capacity shear the size step is withheld while the selected cage has the least longitudinal area the section admits and the probable-moment mechanism sets Ve: a larger section then raises that shear faster than the 8 sqrt(f'c) b d limit. The flag is read only on a settled longitudinal selection whose record shows the mechanism shear governing under the column-own Ve method (the only method the grouped route builds). On the production route a cage that passes its P-M check is expected to have a mechanism-governed Ve, but that is not established, so the record is read each time and the flag left unknown otherwise. It is a sufficient condition only; a cage the selection made heavier for another reason, or one outside the admissible domain, withholds nothing. A capacity-shear move waits until the hoop selection has settled with a recorded outcome; a hoop ladder exhausted by the same over-demand is such an outcome. The gravity column capacity-shear row is routed only when it compares the governing direction's Ve with that direction's hoop phi Vn. The producer currently pairs it with the lighter direction's phi Vn; such a row is invalid evidence for routing, listed under `unroutable_failures` (see the implementation report). With Vs uncapped, as the producer records it per direction, an authenticated row cannot fail today, so genuine section deficits route through `column.capacity_shear_section`.
  - **H3, reinforcement space:** beam width for horizontal bar room; the joint core for threading. A slab-mat overlap abstains until its blocked dimension is diagnosed. A beam strength shortfall of a lane-limited group (the cages that reach the demand do not thread the column) is bar room, not strength: beam depth, beam width and, on a consistent joint topology, the cores of the joints the beam frames into; never a concrete grade. Any topology problem withholds those cores as `joint_topology_inconsistent`, as it invalidates every joint-based diagnostic (also when none of the cores it names is a group of the parent); on a consistent topology, a beam it names at no joint has its column lever withheld as `no_joint_incidence`.
  - **H4, joint:** core size and grade for joint shear; core size for through-bar depth and hook embedment.
  - **H5, drift / stability:** the SFRS columns of the story and the beams (depth) of both floors that bound it.

  The controller chooses an operator uniformly among those with a proposal, then one of its proposals uniformly. Otherwise, or when every diagnostic abstains, it uses R1 (a uniformly chosen gene, then a uniformly chosen local neighbour).

  `counts.abstentions` and `counts.withheld_levers` are counted once per parent routed on a guided draw (a parent drawn only on the R1 branch contributes nothing). Within a parent, a withheld lever counts once per diagnostic and group whose step exists (a larger square size), as abstentions count once per diagnostic; a lever without a target is listed by its reason.
- **Not part of v3:** crossover and adaptive operator weights.
- **Unknown or invalid evidence never strengthens.** Unknown rows, solver failures and invalid evidence abstain with a named reason. A not-evaluated strong-column row becomes an evidence task, never a hierarchy move.
- **Diagnostic binding.** Each diagnostic is bound to the parent's realized identity and the run's evaluation basis. Its dependencies come from the record itself: rows, completion flags, capacity, strength and action evidence.
- **Before any evaluation:** repair, then post-repair validation (no-op, rejected, ready), then deduplication.
- **Two identities:**
  - `input_identity`: the basis plus the complete submitted design, seed cages included. It is the cache key.
  - `realized_identity`: the final design with its cages and slab. It is the registry key.
- **Recorded per child:** the requested change, the changes repair induced, and the reinforcement changes against the submitted design and against the parent. A child with a parent also records `vs_parent`: each requesting diagnostic's check (or strength component) in the parent and in the child, kept apart from the child's new governing maximum; collateral regressions over every check; evidence gained; both sets of material quantities.
- **Selection** (`seisframe_ga_selection_v2`):
  - **Tiers:** code-accepted, then complete with known failures, then unresolved with known failures.
  - **Evidence and recovery queues:** a record without an observed failure but with missing evidence goes to the evidence queue. A stopped, raising or numerically incomplete record goes to the analysis-recovery queue. Every child of a generation is in its selection, so errored children appear in that generation's queues.
  - **Coverage** is compared only under the same expected inventory.
  - **Progress** (max known exceedance, mean per-family max exceedance) is compared only under equal inventory, known-check mask and check bases. It forms nondominated fronts.
  - **Tie-breaking** uses fixed-domain section/grade diversity.
  - **Local regression witnesses** are kept.
  - **Quarantine** is computed over the whole history: a design any observation contradicts is never a parent.
  - A survivor is a search parent, not an accepted design.
- **Views recomputed from the full history:**
  - the accepted registry: every distinct code-accepted realized design, whatever its material quantities; every accepted-classified observation it excludes is listed as refused or quarantined;
  - a bounded diversity view;
  - the evidence queue, grouped by missing source, with full per-source counts (each realized design counted once, quarantined ones left to the quarantine view); an unavailable material quantity is listed whenever it occurs. An unevaluated row without a reason code gets one only from the producer's own record: a gravity column capacity-shear row of a column without selected hoops, whose governing Ve is known, is `hoops_not_selected` (the upstream hoop selection), not evidence to obtain;
  - the analysis-recovery queue;
  - the unroutable failures: failing rows the router's first gates keep from every operator, by kind and check. `invalid_evidence` is a row the adapter could not authenticate, such as the gravity column direction mixing, or a strength row whose DCR does not follow from its components or whose group has no strength evidence (a whole-row strength item is never authenticated, however it is marked). `no_operator` is a failure without a mutation reason, such as `column.hoops_selected`, or the analysis-shear component of a strength row (named `#shear`). `dependencies_incomplete` is a mapped failure whose operator dependencies the record does not show. `rows` counts failing rows (a strength row and its components are one). Each still blocks acceptance; none is a strengthening trigger. A failure that passes these gates but on which every operator still abstains (no legal move at the end of the catalog, for example) is not listed here; it is in `counts.abstentions`;
  - quarantined contradictory observations;
  - descriptor collisions.

  Qualification stays `not_performed`, and nonlinear performance stays `not_requested`.
- **Stops:**
  - **Budgets and limits:** evaluation, proposal, time (checked between steps), generation, stagnation or error bounds. A stop between steps first selects whatever was evaluated since the last selection.
  - **Stagnation** counts generations that had parents in which no progress was made. Progress is a code-accepted identity added to the registry, or a new realized design (evaluated in that generation, and realized by no earlier observation of the run) that enters the population and either:
    - dominates a member of the previous population (a better tier, more coverage under the same inventory, or better progress measures under the same comparison signature; `improving_entrants`). Each member is compared on its own record, so a member deduplicated under an earlier id that a cache hit readmitted still counts; a member quarantined in the same selection is not compared; or
    - enters while that population, less its members quarantined in the same selection, had a free slot (`free_slot_entry`).

    Any other entry is not progress: by the diversity tie-break (noncomparable with the member it replaced; the tie-break's seeded anchor is redrawn every generation, so such entries happen by chance), or by rank (a nondominated design that dominates no member). Progress is measured against the previous population, not against every earlier parent. A design reached again through another input is listed as `rediscovered_designs`, not counted, and a population change among earlier designs (a readmitted cache hit) is not progress, even into a free slot. A selection that ends underfilled keeps every eligible candidate, and the population shrinks only by quarantine, so free-slot entries reset the counter at most population-size times plus once per quarantined member. A generation drawn while no parent existed (exploration) does not increase the counter; its entries fill free slots and reset it, within the exploration budget. On a flat landscape (no design dominates another) the bound ends the run after that many generations once the population is full (generations that fill it reset the counter). Designs whose known checks differ (masks or bases) are noncomparable, so a run whose exceedances still fall across such designs can end on the bound too: it is a bound, not a finding.
  - **No new candidate:** a generation with parents that produced no new evaluation (only no-ops, duplicates, cache hits or rejections) stops the search, even when a readmitted cache hit changes the population.
  - **Targets:** distinct accepted designs, or distinct accepted section profiles.
  - **`waiting_on_dependencies`:** no eligible parent and the exploration budget is spent. The exploration budget counts exploration evaluations over the whole run; no-op, duplicate and cache-hit draws are not counted. Only an empty population draws exploration designs; an underfilled one breeds from the parents it has. The stop detail lists the evidence queue, the analysis-recovery queue and the quarantined observations.
  - **Evaluation errors:** an exception raised by the evaluator or while its result is assessed, diagnosed or saved is a recorded error observation. It keeps every artifact already written for it (the evaluation, for analysis recovery; a design file is named as soon as it exists). It holds only recordable values of the evaluator's result, so a status that cannot be serialized (NaN, an object) is kept as text. A write that fails removes its temporary file. Its input is a cache key like any other, so it is not evaluated again within the run; it waits in the analysis-recovery queue. The run continues until `--evaluation-error-limit` errors.

  No stop is a statement of infeasibility. Each stop carries a note specific to its reason.
- **Checkpoints and resume.** Every step is checkpointed with the random-generator state. `--resume` continues a run only with identical inputs and configuration fingerprint (case included). Every repository module loaded by the runner is hashed into the request, so an edited source refuses the resume; keep the tree unchanged for the length of a run.
  - An evaluation that finished before an interruption is adopted, with its own time added to the run's elapsed time. An unfinished one is evaluated again, and the checkpoint after it carries its time.
  - Only records an interruption can leave are accepted after the checkpoint. These are the pending evaluation, one cache-hit origin of the next step, log lines of the next step (of the checkpoint's own step when it was taken inside an evaluation), and one cut final line. Anything else refuses the resume and changes no run file (taking the lease may create `.generation.lease`, which holds no run data). So does a history naming one observation id twice, or a checkpoint whose pending evaluation names an observation already in the history.
  - The interrupted process's `error.json` is set aside first, as `error.before_resume.N.json`, so a rename that fails changes no other file. Set-aside lines then go to durable sidecar files, and each rewritten file is replaced atomically. A sidecar number already on disk (from a resume cut before its own checkpoint) is skipped, never overwritten. A run that a resume started (its request saved, no checkpoint yet) records that, and the error record it set aside, in its first checkpoint. A resume that fails after that rename leaves the record unnamed; the next resume names it in `unrecorded_error_records`.
  - One process at a time owns the run directory, through an OS lease on `.generation.lease` that the kernel releases if the process dies. A second `--resume` while the first run is alive is refused. Every run file is written under the lease: the request a new run claims the directory with, and `error.json`, which the controller writes before it releases the lease.
  - A controller runs once. When its run returns or raises it is spent: it writes nothing (also when releasing its lease failed), and `start` or `resume` on it is refused before anything is read or written; a new controller continues the run. A second `start` or `resume` on a controller that already owns the directory, or that loaded a history, is refused the same way and keeps its lease. A `start` without a request refuses a directory that holds another run's saved request, and a `resume` without the request's sha256 refuses a directory without a checkpoint (there is no run to resume). `summary()` still gives the summary without writing, as a copy throughout (views included). A `start` or `resume` that raised keeps no partial run in memory, so `summary()` on it is refused.
  - On Windows without long-path support, an output directory whose artifacts would exceed 259 characters is refused when the controller is built, before any directory exists.
  - Solve counts are work accounting: the evaluator's floor caches live in the process, so a resumed run may count more solves than an uninterrupted one. A raising evaluation records its floor solves, from the shared cache, and an unknown frame count.
- **Initialization** (`--initialization`):
  - `seed_only_v1`
  - `seed_r1_neighbors_v1` (the default)
  - `uniform_then_repair_v1`
  - `constructive_profiles_v1`: the research grammar sampler, audited for `system_three_story_v1` only and refused elsewhere. Every draw must come back from repair unchanged.

Run from `RC Structure` (plan first: `--plan-only` solves nothing and writes nothing):

```powershell
python -X utf8 -B tools/genetic_design_search/run.py --engine generation_v3 --case-file <case.json> --profile <profile> --seed-column 30 30 6 --seed-beam 16 30 5 --population-size 4 --max-generations 3 --max-evaluations 16 --max-wall-seconds 1800 --random-seed 2026103101 --output-dir <new-directory>
```

The generation_v3 option names (`--proposal-budget`, `--evaluation-error-limit` and the rest) were chosen so that no heuristic_v2 option, full or abbreviated as the earlier runner accepted it, changes meaning. Short abbreviations such as `--o` or `--g` are now ambiguous; argparse refuses them, and none is reassigned. Use full option names.

Exit codes: `0` the search stopped (the reason is in `result.json`), `1` the controller raised (`error.json`) or releasing the lease failed after the run finished (`result.json` holds the run; no `error.json`), `3` the search stopped at its evaluation-error bound (the first error is printed).

Output directories:
- A new or empty `--output-dir` starts a run. So does one holding only the lease marker or a cut `request.json.tmp`.
- The runner checks the directory first, so a directory that check refuses gets nothing, not even the lease marker. The controller then checks it again under the lease and saves `request.json` there, durably and atomically, before the first checkpoint. Two launches cannot both claim one directory, and the one that loses never replaces the winner's request. A cut request write leaves no request, and the same command starts again.
- An existing run directory (it holds `request.json`) is refused. If its request is this one, the refusal points to `--resume`; if it is another request's (including a launch that lost the claim to it), it points to a new `--output-dir`; if it is not a whole JSON object, it says to inspect it. A directory owned by a running process is refused by the lease; that run continues with its own command plus `--resume` once it has stopped.
- A directory holding a matching `request.json` and no checkpoint or log (interrupted before its first checkpoint) starts on `--resume`; any other file in it is left in place. A `request.json` that is not a whole JSON object (cut, undecodable, or another JSON value) refuses the resume.
- A non-empty directory without `request.json` is refused in both modes: choose a new `--output-dir` or empty it.
- A configuration or output path the controller refuses leaves no directory.

Outputs:

| File | Contents |
|---|---|
| `request.json` | Inputs, fingerprints, loaded-source hashes, genes, basis and configuration. |
| `checkpoint.json` | Resumable state with the random-generator state. |
| `observations.jsonl` | Append-only evaluation records and cache-hit origin events. |
| `proposals.jsonl` | Every proposal attempt with its outcome. |
| `generations.jsonl` | Selection decisions, fronts and regression witnesses (from the initial selection on). |
| `evaluations/oNNNN.json` | The public evaluation of each evaluated design (an error observation names the one written before its processing failed). |
| `designs/oNNNN.json` | Each code-accepted realized design: member groups, slab thickness and layout with its inputs and floor context, every code-check row, and whether the realized identity recomputes from them. Every file is named by its observation; the registry lists only accepted records. |
| `topology/<sha>.json` | The joint topology the diagnostics were routed on. |
| `result.json` | Stop, counts, metrics and all views. |
| `.generation.lease` | The ownership lock marker. It persists; the lock itself is released by the kernel. |
| `observations.partial.N.txt`, `observations.after_checkpoint.N.txt`, `<log>.after_checkpoint.N.txt` | The lines resume N set aside. |
| `error.json`, `error.before_resume.N.json` | A controller exception (written by the controller under its lease), and the one a later resume set aside. |

Without an asserted slab-action review, every strong-column row is unknown, so no candidate can be code-accepted. The run then exercises known-failure guidance and the evidence queue, not acceptance.

## Search behavior (heuristic_v2)

- Genes represent section sizes per design tie (or per untied group), and concrete grades per declared family. One layout and longitudinal steel grade remain fixed for each run. Reinforcement is selected by the existing cage selector, starting from the same seed cages for each candidate.
- With probability **0.75**, mutation tries a move suggested by the evaluated parent's failed checks. It can grow affected groups or grades, propose wider/deeper beams for lane congestion, or reduce material from feasible parents. Guidance applied after crossover is a hypothesis, checked by a new evaluation.
- Otherwise, or if no usable guided move exists, one random gene changes. Most random changes use a neighboring rung; 20% can jump elsewhere on that gene's declared ladder. Crossover probability is **0.8**. These are starting settings, not tuned results.
- Repair preserves ties, grade families, the column ceiling, section order and system coupling. It can enlarge neighboring bands. All resulting alleles must remain on the declared ladders. An off-ladder seed is refused rather than silently changing the domain.
- Completed feasible candidates outrank failures. Feasible survivors retain concrete/total-steel tradeoffs through nondominated sorting and crowding. The representative follows the existing material rule: neither quantity increases, and at least one decreases; fewer form sizes break a quantity tie. Concrete grade has no price premium in this objective.
- Every evaluation passes the code-check contract (`Design/Code_Check_Contract.py`, 2026-10-30) before it is ranked. The contract checks the record's schema and explicit completion flags, the seed identity captured before evaluation, the final state and its identity, and the code-check rows bound to that final design. It derives the required checks from the final design's own topology and system assignment, including every required strong-column joint, direction and sway sign, and recomputes the result. A candidate is **feasible** only when that complete scoped code check passed, the evaluation completed and the material quantities are available.
- Completed failed checks (**check_failed**) rank by the dimensionless exceedances of their known failures. An unfinished analysis, unsettled reinforcement/slab iteration (except reinforcement passes whose proven cycle was resolved on verification: contract v2 completes the two selection flags for that final state only), unknown or missing evidence (for example strong-column/weak-beam rows left unevaluated because the slab actions await their engineering assertion), or unknown total steel is **unresolved**. Those candidates keep their known failures as fields and as secondary ranking keys, but cannot enter the feasible archive. A record that is malformed, unbound to its final design, or contradicts its own rows is **invalid** and ranks last. Review-dependent items (fire resistance, slab-column local steel, the story-strength model M1, coupler product qualification and so on) are listed, never asserted. Exceptions abort with an error record. A search budget stop proves neither infeasibility nor optimality.
- Evaluation is serial. Repaired genomes are deduplicated within the run, and the evaluator's content-keyed floor caches are reused. Default limits are 8 population members, 24 distinct evaluations and 10 offspring generations after initialization. A bounded number of proposal attempts prevents endless duplicate loops. Candidate limits are not wall-clock limits; one full evaluation can still be expensive.

## Run

From `RC Structure`, in the existing OpenSees Python environment:

```powershell
python tools/genetic_design_search/run.py --manifest <screening_plan.json> --case-id case_0030 --seed-column 36 36 6 --seed-beam 16 30 5 --population-size 8 --max-evaluations 24 --random-seed 42 --plan-only
```

Choose sections on that case's declared ladders. `--plan-only` prints the configured inputs, source hashes and all genes without solving or writing files. Replace `--plan-only` with `--output-dir <new-directory>` to run. Existing output directories are refused. `--grouping` can select another existing grouping policy; `auto` resolves the starting layout and does not escalate layouts within this search.

Alternatively use `--case-file <case.json>` with a JSON object containing `case_id`, `geometry_name`, `num_bay_x`, `num_bay_y`, `num_floor`, `story_height_ft`, `bay_x_width_ft`, `bay_y_width_ft` and `seismic_site`. `--profile` optionally selects an existing analysis profile. Without it, the configured model profile is retained and recorded.

Outputs:

- `request.json`: case, source/configuration identity, seed, domains and budgets.
- `candidates.jsonl`: each evaluated candidate's dimensions/cages, checks, quantities, mutation reason, parents and repairs; rejected invalid proposals and evaluation errors are also recorded.
- `result.json`: stop reason, work counts, outcome classes, feasible tradeoffs and a representative, or an error record.

Every result is marked experimental and `production_accepted: false`. This entry point does not install synthetic PROBE assertions, create a qualified design record, or run nonlinear ground motions. Without an asserted slab-action review, no slab layout is selected, so the strong-column checks stay unknown and every candidate is unresolved. That is the honest outcome, not a defect of the search. A candidate called feasible here has passed the scoped code check of the contract. It is not a qualified design: the saved record's qualification, independent review and `GENERATION_RELEASE_READY` remain separate.

## Validation and comparison

```powershell
python -B -m unittest discover -s tests -p test_genetic_search.py -v
```

The behavioral tests use controlled evaluator responses to check budgets, reproducibility, crossover, congestion alternatives, repair, ranking, duplicate avoidance, logging and restoration of shared state. They are not a structural benchmark. Before replacing the current search, compare the same cases, seeds, evaluation budgets and evaluator revision across several random seeds. Report feasibility, unresolved outcomes, concrete/steel quantities, solve counts and elapsed time. The heuristic operator changes search proposals; it does not fix the audited engineering-check defects.
