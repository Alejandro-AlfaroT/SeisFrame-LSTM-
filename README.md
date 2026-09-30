# SeisFrame

Simulation-to-dataset pipeline and hybrid GNN-LSTM surrogate for the nonlinear
seismic response of reinforced concrete special moment resisting frames
(RC SMRFs).

Nonlinear time history analysis (NTHA) is the most reliable way to evaluate the
seismic response of an RC SMRF, but its cost rules it out for large-scale uses
such as fragility analysis and regional risk assessment. SeisFrame designs
code-compliant RC SMRFs, models them in OpenSees with concentrated-plasticity
Ibarra-Medina-Krawinkler (IMK) hinges and beam-column joint elements, runs
gravity, modal, pushover and bidirectional NTHA under PEER NGA-West2 ground
motions, and exports every run as a graph dataset. A hybrid surrogate then
combines a graph neural network over the structural graph with an LSTM over
time to predict the response history at a small fraction of the simulation
cost. See `abstract_draft.md` for the current research abstract.

## Repository layout

| Path | Purpose |
|---|---|
| `RC Structure/` | The OpenSees pipeline. `Main.py` builds, designs and analyses one frame. `Design/` holds the ACI 318-19 SMRF design loop, slab and joint checks, and SAP2000 export/comparison for independent verification. `Model/` holds sections, IMK hinge calibration, joint panels and springs. `Analysis/` holds gravity, modal, pushover, response-spectrum and NTHA drivers plus diagnostics. `Loads/` holds gravity, ELF and ground-motion loading. `Data_Generation/` holds the deterministic multi-device dataset scheduler, intensity calibration and graph export. `Ground_Motions/` holds the record manifest and set definitions (raw and processed records stay out of git). `Reference_Specimens/` holds the Park and Ruitong (1988) Unit 1 validation package. `tests/` holds the pytest suite. |
| `RC Hybrid Surrogate Model/` | The GNN-LSTM surrogate: dataset schema, feature engineering, model, losses, metrics, training and evaluation scripts, and experiment configs. Its own `README.md` documents the baseline task, leakage control and training commands. |
| `structural-analysis-kit/` | A portable analysis workspace for inspecting and comparing generated datasets across devices, with per-device profiles kept out of git. See its `README.md`. |
| `OpenSees/` | Early exploratory OpenSeesPy scripts (2D and 3D frames, fiber sections, El Centro response spectra). Not used by the pipeline. |
| `projector_share/` | A TensorFlow Embedding Projector bundle of surrogate embeddings for the 3,800-case dataset. |
| `abstract_draft.md` | Current research abstract. |
| `run_rc_structure.ps1` | Windows launcher that runs `RC Structure/Main.py` in the `OpPy` conda environment. |

## Environment

The generation pipeline runs in a conda environment named `OpPy`, specified in
`RC Structure/environment-generation.yml`:

- Python 3.12
- NumPy, SciPy, Matplotlib
- `openseespy==3.8.0.0`

The surrogate additionally needs PyTorch and PyTorch Geometric
(`RC Hybrid Surrogate Model/requirements.txt`).

## Running

All commands below are run from the repository root. On Windows, substitute the
`OpPy` interpreter for `python` as `run_rc_structure.ps1` does.

Design and analyse one frame:

```powershell
.\run_rc_structure.ps1
```

Generate the parameterized dataset for one device's case range (see
`RC Structure/Data_Generation/MULTI_DEVICE_SETUP.md` for range assignment and
the plan hash check):

```text
python "RC Structure/Data_Generation/Generate_Parameterized_Dataset.py" --help
```

Run a design-only verification sweep over the plan's geometries
(`RC Structure/Design/DESIGN_VERIFICATION_RUN.md`):

```text
python "RC Structure/Design/Verify_Designs.py" --count 150 --workers 4 --output-root outputs/design_verification
```

Smoke-test and train the surrogate (`RC Hybrid Surrogate Model/README.md`):

```text
python "RC Hybrid Surrogate Model/train.py" --smoke-test
python "RC Hybrid Surrogate Model/train.py"
```

Run the test suites:

```text
python -m pytest "RC Structure/tests"
python -m pytest "RC Hybrid Surrogate Model/tests"
```

## Lineage

This repository started as a fork of StructGNN, the static-analysis graph
neural network of Chou, Chang, Jean, Chang, Huang and Chen (2024),
"StructGNN: an efficient graph neural network framework for static structural
analysis", *Computers and Structures*. That code, its linear static dataset,
and a later SAP2000-based extension were removed from the working tree once
SeisFrame no longer depended on them. They remain in the git history before
the cleanup commit.
