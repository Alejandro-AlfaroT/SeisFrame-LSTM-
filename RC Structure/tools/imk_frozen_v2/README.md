# Frozen IMK V2 calibration: verification tools

These scripts verify that the production member-spring calibration `seisframe_imk_v2_frozen_20261029`
(`Model/IMK_Spring_Mapping.py`, specification in `Model/IMK_FROZEN_V2_CALIBRATION.md`) reproduces the
approved twenty-run MCER/DBE research study numerically, without the research adapter.

The approved study is read-only evidence. Its root holds `manifest.json`, `profile.py` and `probes.py`. The
saved-design harness folder `v6-mcer-hinges` must sit beside the study root or under `../references/`. The
portable package is at `D:\SeisFrame_RA_Review_2026-10-29\current-study`; there is also a desktop copy. Every
script writes only into a new directory under `RC Structure/outputs/` (git-ignored) and refuses to write inside
the study or over a non-empty directory.

Run every command from `RC Structure` with the OpPy interpreter. Set `STUDY` to the study root first.

```
$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:FOR_DISABLE_CONSOLE_CTRL_HANDLER="1"
$STUDY = "D:\SeisFrame_RA_Review_2026-10-29\current-study"
```

**A. Full parameter parity.** Rebuilds the ten saved designs in clean processes and compares all 3,407 members
and 13,628 springs with the approved MCER and DBE registries. The comparison covers every native argument
(rtol 1e-10, atol 1e-12, with the maximum errors reported), branch orientation, physical energies, strengths,
stiffness, the unchanged du, and the adapter's member-reference records. `--with-replay` also rebuilds the designs
under the legacy replay profile and checks them against the pre-freeze baseline registries
(`v6-mcer-hinges/cases/<case>/registry.json`).

```
python -X utf8 -B -I tools/imk_frozen_v2/parity.py all --with-replay --study $STUDY --out outputs/imk_frozen_v2_verification_20261029/parity
```

**C. Native response regression.** Replays the archived fresh probes with the production-created command, using
the study's own protocol code (`probes.py simulate`, `fema_probes monotonic`), with the archived amplitudes,
cycles, chord or spring control, accepted steps and zero-history initial state. It also runs the archived command
in the same process as a control.

```
python -X utf8 -B -I tools/imk_frozen_v2/probe_replay.py all --study $STUDY --out outputs/imk_frozen_v2_verification_20261029/probes
```

**D. Frame integration regression.** Reruns case_0064 at each intensity through the study's saved-design
earthquake harness without the adapter: the saved average-intensity scale, the RSN864 pair, dt factor 0.5,
UmfPack and 5 % Rayleigh. It then compares the run with the approved batch.

```
python -X utf8 -B -I tools/imk_frozen_v2/frame_regression.py run --study $STUDY --mode mce_average --case case_0064 --out outputs/imk_frozen_v2_verification_20261029/frame
python -X utf8 -B -I tools/imk_frozen_v2/frame_regression.py compare --study $STUDY --mode mce_average --case case_0064 --out outputs/imk_frozen_v2_verification_20261029/frame
```

Repeat with `--mode dbe_average`. The comparison reads only the summary, npz, registry and state-vector files of
the approved run, which both copies of the study hold; the raw `all_y.bin`/`all_z.bin` recorders, which the
portable package omits, are not needed. The harness's `preflight.py` points at this repository's desktop path.

**B. Mechanics and invariance.** These checks are unit tests and need no evidence:

```
python -X utf8 -B -m unittest discover -s tests -p test_imk_frozen_v2_calibration.py -v
```

The tools verify numerical equivalence to the approved study. They do not establish experimental validation, and
they do not authorize production generation.
