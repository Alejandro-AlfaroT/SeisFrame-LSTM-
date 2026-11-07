# Frozen SeisFrame V2 member-hinge calibration

**Identity:** `seisframe_imk_v2_frozen_20261029`. The source of truth is `Model/IMK_Spring_Mapping.py`. The
configuration key is `Structure_Parameters.IMK_MEMBER_SPRING_CALIBRATION`.

**Approval:** 29 October 2026, by the principal researcher, reporting the RA's agreement. The approval froze the
calibration of the completed twenty-run MCER/DBE study and made it the normal V2 member-hinge calibration. It is
numerically equivalent to research profile `v2_imk_half_member_reference_research_20261028`. The scientific
baseline is commit `66f964c10d6a9ce915aba5b1ec5db40ca3d72822` (Verification1).

**Not experimental validation.** The approval freezes a modelling choice for V2. It does not prove universal
applicability or certify a dataset release, and it authorizes no production generation. The retained native du is
a provisional numerical cutoff, not an experimentally established collapse rotation. A zero-moment event in an
analysis is a numerical material event, not an observed failure. Numerical success, plausible loops and the
earlier roughly 0.087 rad zero-moment event are not experimental validation.

## Scope

The calibration converts each member-end flexural spring the model assigns IMK to, once: both ends of every beam
and column, both bending axes (local 5 = y, 6 = z), and both native signs. It does not touch:

- released gravity-beam ends (Elastic numerical pins);
- joint springs (`Model/Joint_Springs`, IMKPinching);
- the low-level native command adapter `IMK_Materials.define_rotational_imk`, which still installs
  already-native inputs unchanged and never reinterprets an arbitrary command as a member backbone.

The approved study used the V2 profile: rigid centerline joints, ACI 318-19 Table 6.6.3.1.1(a) modifiers
0.35 Ig for beams and 0.70 Ig for columns, and no face slip interfaces. A build outside that scope is not refused.
It is recorded per spring, under `member_spring_calibration.scope`.

## Half-member reference (per spring)

The existing effective EI0 and the model centerline length L are used; the nominal shear span is Ls = L/2.

| Quantity | Definition |
|---|---|
| K0 | 3 EI0 / Ls = 6 EI0 / L = Kh / n |
| n | 20 |
| Kh | n K0, the installed spring stiffness (unchanged) |
| EIe | 1.05 EI0, the elastic element between the springs (applied once) |
| Kb | 3 EIe / Ls = 1.05 K0 |

With this retained split the assembled initial rotational stiffness is 420/421 K0, which is 0.2375 % soft in
double curvature. The discrepancy is stated, not corrected: n, Kh and the interior factor 1.05 are frozen.
K0 is taken as Kh/20, as the approved adapter did, and is checked against the section's own 6 EI0/L to 1e-12.

## Each physical branch

My is the branch's existing adopted yield moment, in positive magnitudes.

    Mc            = 1.10 My
    theta_y_ref   = My / K0
    theta_cap_ref = theta_y_ref + theta_p,member
    dp_native     = theta_cap_ref - Mc/Kb - My/Kh
    dpc_native    = theta_pc,member + Mc/Kb
    du_native     = theta_p,member + theta_pc,member + legacy padding   (unchanged)
    theta_reference_chord = theta_native_spring + M/Kb

Notes on these equations:

- The member intervals come from the unchanged Haselton et al. (2008) regressions, with their existing inputs,
  floors and clamps: Eq. 3.10 (theta_p), Eq. 3.16 (theta_pc, with the published 0.10 ceiling on the member
  interval) and Eq. 3.20 (lambda).
- dpc_native may exceed 0.10 and is not clamped back.
- The legacy padding is 0.005 for beams and 0.004 for columns. It is named `legacy_cutoff_padding_rad` and is not
  the member yield rotation of any active rule.
- On the reference, the native capping point is exactly theta_cap_ref. The post-capping slope is -Mc/theta_pc and
  the residual begins at theta_cap_ref + theta_pc (1 - Mr/Mc), with Mr/My = 0.20.

## Deterioration energy

    theta_y_ref,E = My_canonical / K0
    Eref_S = Eref_C = lambda_H * My_canonical * theta_y_ref,E
    Eref_A = Eref_K = 1e12 * My_canonical
    Lambda_native,m = Eref_m / My_native_positive

My_canonical is the canonical PHYSICAL direction of the `canonical_physical_orientation_baseline_v1`
orientation: hogging for a beam's vertical bending, even where sagging is stronger, and the section's positive
branch for the symmetric springs. Reversing the native coordinate therefore keeps every physical energy. A and K
stay suppressed with the existing large finite capacities, and c = 1 and D = 1. Neither the nominal 0.005/0.004 rad
yield rotation nor My/Kh is an active energy rotation. The pre-freeze nominal products are kept only as labelled
source metadata (`deterioration_historical_nominal`).

## Worked example (golden checkpoint)

The example is case_0064, beam 173, end j, local axis y, material 3017325. The native positive branch is sagging.

| Quantity | Value |
|---|---|
| K0 | 2359414.450118823 kip-in/rad |
| Kh | 47188289.00237646 kip-in/rad |
| Kb | 2477385.172624764 kip-in/rad |
| Positive branch (sagging) | My 5805.419123130262, dp 0.06592058298671885, dpc 0.10257770212965206, du 0.17116077795789097 |
| Negative branch (hogging) | My 8249.551921462938, dp 0.06581945877939033, dpc 0.10366293752537273, du 0.17116077795789097 |
| theta_y_ref,E | 0.0034964403651285094 rad |
| Eref_S = Eref_C | 3047.590937189035 kip-in-rad |
| Lambda_S = Lambda_C | 0.524956230127582 |
| Lambda_A = Lambda_K | 1421008844752.4194 |

`tests/test_imk_frozen_v2_calibration.py` pins this command argument for argument.

## Retained, not changed

- Effective stiffness modifiers and the member strength and slab-development calculations.
- Mc/My = 1.10, Mr/My = 0.20, and the residual and tail.
- The regression predictors, the fixed axial calibration basis and the legacy guards. The guards are now reported
  per member under `regression_guards`: the nu clamp, the rho_sh clip (120 columns in the ten study designs), the
  theta_p and theta_pc floors, and the 0.10 ceiling. No new rejection or quarantine was added.
- Bond-slip ownership and the a_sl policy, section geometry, reinforcement rows, element topology, releases and
  damping.

## Configuration, profiles and replay

`IMK_MEMBER_SPRING_CALIBRATION` defaults to the frozen calibration, so every ordinary build uses it.

`Model/IMK_Spring_Mapping.configuration_problems` refuses any configuration that would mix the frozen calibration
with another policy: another material, stiffness rule or factor, deterioration or energy-mapping mode, Mc/My,
Mr/My, c, D, or supplied member energies. The legacy-unmapped, anchored and explicit-energy mappings and IMKBilin
therefore need the replay.

`legacy_direct_interval_nominal_energy_66f964c1` reproduces the pre-freeze rule exactly: member intervals installed
as native values, and Lamda = lambda times the nominal yield rotation. It is used only for explicit historical
replay. The profiles are:

- `v2_nonlinear_flexure_screening_v1` (the V2 profile) requires the frozen calibration. A process configured for
  the replay is refused rather than switched.
- `v2_nonlinear_flexure_screening_v1_legacy_imk_replay` is the same model under the pre-freeze rule. It
  reinstalls designs made for V2.
- `v2_nonlinear_flexure_anchored_energy_research_v1` is a pre-freeze research convention and now sets the replay
  explicitly.

The calibration and its stiffness rule are part of the profile identity, and so of every new design request and
of every intensity and output identity. Saved designs never restore IMK settings: `install_record` reinstalls
only geometry, site and the design. It reports, per key, where the record's profile identity differs from the
installed one, for example a pre-freeze record with no `IMK_MEMBER_SPRING_CALIBRATION`. The process-local research
adapter is refused if it is present, because it would map the frozen native values a second time.

## Exports

Each installed material's provenance carries `member_spring_calibration`:

- `reference`: K0, Kh, Kb, n, the interior factor, L, Ls and the section check;
- `branches`: per physical branch, the input sign, My, Mc, Mr, the member theta_p, theta_pc, theta_y_ref and
  cap, and the native theta_y, dp, dpc, cap and du;
- `energy`: lambda_H, My_canonical, theta_y_ref,E, every Eref and the equations;
- `du_policy`, `retained_assumptions`, `approval` and `scope`.

The material record holds the exact native command and its parameter hash. Each hinge-registry entry carries
`member_spring_calibration` and `spring_points` per end and axis
(`IMK_Spring_Mapping.spring_points`: native My/Kh, My/Kh + dp and du per sign, beside the member-reference values).

The flat `theta_p`/`theta_pc` keys are member intervals and `theta_u` is the retained native du. `theta_y_target`
is None under the frozen calibration; the nominal value is `historical_nominal_theta_y_rad`.

Downstream classification reads the installed per-end, per-axis and per-sign points through `rotation_demand` and
`hinge_end_demand`. That includes mechanism checks, pushover and hysteresis diagnostics, the hinge tables and the
dataset features. They never read the global `IMK_THETA_*` constants, and `imk_hinge_thresholds` refuses them
under the frozen calibration.

`damage_ratio` now means the native capping ratio of the governing axis and sign: 1 at the native capping point
My/Kh + dp. The du event is `native_cutoff_du_exceeded` (a provisional numerical cutoff, not a collapse). The signed
work integral is `net_work_minus_initial_ke_storage_kip_in` (not dissipation). The changed export meanings carry
new versions, so pre-freeze and frozen outputs are refused rather than mixed:

| Export | Version |
|---|---|
| `hinge_backbone.csv` | `hinge_backbone_v2_native_spring_demand` |
| Hybrid dataset sample | `hybrid_gnn_lstm_v4` (38 global features, v4 hinge features) |
| Hinge moment-rotation export | `hinge_moment_rotation_v2` |
| Hysteresis diagnostic | `hinge_hysteresis_diagnostic_v4_native_spring_points` |
| Pushover diagnostic | `pushover_diagnostic_v2_native_spring_points` |
| Mechanism diagnostics | `mechanism_diagnostics_v2_installed_native_spring_points` |
| Response score policy and score | `seisframe_seismic_response_score_policy_v2`, `seisframe_seismic_response_score_v2` |

The NTHA output identity, the intensity-calibration model identity and the profile identity all include
`imk_member_spring_calibration`. Resuming a dataset folder does not count a sample of another hybrid schema as
complete.

The SeisFrame-LSTM readers in `RC Hybrid Surrogate Model` (schema_v3, 43 global features) refuse v4 samples. They
must be updated before the surrogate can train on frozen-calibration data; this freeze does not change them.

## Verification (29 October 2026)

The runtime is OpenSees 3.8.0 (openseespy 3.8.0.0), the same as the study, with Python 3.12.13 and numpy 2.5.1.
Commands are in `tools/imk_frozen_v2/README.md`.

**A. Full parameter parity.** The ten saved designs were rebuilt without the adapter: 3,407 members and 13,628
springs.
- Every native command argument of every spring is bitwise equal to the approved MCER registries, with maximum
  absolute and relative error 0.0. The comparison also covers branch orientation, physical energies, strengths,
  stiffness, the unchanged du and the adapter's member-reference records.
- No springs are missing or extra.
- The MCER and DBE registries hold identical commands.
- The legacy replay reproduces the pre-freeze baseline registries command for command.

**B. Mechanics and invariance.** `tests/test_imk_frozen_v2_calibration.py` checks:
- the golden command;
- the cap target, the post-capping slope -Mc/theta_pc on the reference and the residual onset, all against the
  native OpenSees backbone;
- the 420/421 split, the unchanged du, and the native dpc above 0.10;
- mirrored energies of asymmetric beams, hogging as the canonical direction even when it is the weaker branch, and
  A/K free of the energy rotation;
- that the padding does not anchor the energy and the nominal theta_y is inactive;
- refusals of double conversion, invalid inputs, a cutoff at or before the mapped cap, and mixed policies;
- the profiles, and every spring of an assembled frame against hand formulas.

**C. Native response.** Production-created commands replayed the archived fresh probes for six springs:
- case_0064 beam 173 end j, column 74 end j, beam 156 end i and beam 161 end j;
- case_0035 column 85, a symmetric spring of another design;
- case_0013 beam 319, an X-direction beam.

All 30 runs (monotonic, FEMA 461 chord, NCREE 2015 chord and legacy local spring) are bitwise identical to the
archived command run under the same runtime. Against the archive itself, these match exactly:
- the monotonic curves;
- the moments and chord coordinates;
- the work, excursion peaks, endpoint moments and step counts;
- every zero-moment event (coordinates, level, cycle and leg).

The only differences are in the archive files, and the archived command alone reproduces them, so the calibration
does not cause them:
- spring rotations in rows at exactly zero moment, where chord-controlled equilibrium fixes the rotation only to
  the 1e-6 My tolerance (at most 2.3e-9 rad);
- one extra stored final row in two springs.

**D. Frame integration.** case_0064 was rerun at MCER and at DBE with the saved scales (1.9668667648819056 and
1.3112445099212704) and the study's harness, without the adapter, into a new directory. Both runs reproduce the
approved batch bitwise:
- completion over 4,400 steps;
- every drift, roof displacement and base-shear history, with peak drifts 1.122 %/1.426 % at MCER and
  0.763 %/0.991 % at DBE;
- all 800 hinge histories;
- every IMK state vector after gravity and after the record;
- the selected hinges and rankings.

## Known limitations (unchanged by this freeze)

- Haselton's lambda was regressed with yield energies taken at the secant stiffness EIy (PEER 2007/03,
  nomenclature, p. 15). The frozen rule takes theta_y,E from the model's effective K0 instead. This was reviewed
  on 2026-10-29 and is recorded as an open scientific question, not changed here.
- The beam regressions extrapolate column regressions.
- The du is a provisional cutoff.
- The joint and slip partition and the 420/421 split are retained modelling assumptions.
