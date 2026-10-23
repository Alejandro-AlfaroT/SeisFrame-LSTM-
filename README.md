# SeisFrame

Simulation-to-dataset pipeline and hybrid GNN-LSTM surrogate (SeisFrame-LSTM) for the nonlinear seismic response of
reinforced concrete buildings whose lateral system is a perimeter special moment frame with an interior gravity system.

Nonlinear time-history analysis (NTHA) is the most reliable way to evaluate the seismic response of a reinforced
concrete frame, but its cost rules it out for large-scale uses such as fragility analysis and regional risk
assessment. SeisFrame designs code-compliant buildings automatically (ACI 318-19 and ASCE 7-22: perimeter special
moment frames as the seismic force-resisting system, every interior beam and column as a member not designated as
part of that system), models them in OpenSees with concentrated-plasticity Ibarra-Medina-Krawinkler (IMK) hinges at
every member end, calibrates the ground-motion intensities, runs gravity, modal and bidirectional NTHA under PEER
NGA-West2 ground motions, and exports every run as a graph dataset. The surrogate combines a graph neural network
over the structural graph with an LSTM over time to predict the response history at a small fraction of the
simulation cost. See `abstract_draft.md` for the research abstract.

## Repository layout

| Path | Purpose |
|---|---|
| `RC Structure/` | The OpenSees pipeline. `Design/` holds the design automation: the grouped-design engine (`Grouped_Runner.py` is the entry point; `Grouped_Design`, `Grouped_Search`, `Grouped_Record`, `Group_Selection`, `Group_Capacity`), the two-system basis (`Perimeter_Systems.py`, `Gravity_Frame_Members.py`), the ACI 318-19 checks, capacity design, joints, slab, diaphragm and qualification modules, and the uniform route the grouped engine builds on (`Design_Driver`, `Uniform_Search`, `Section_Design`, `Verify_Designs`). `Model/` holds member groups, sections, IMK hinge calibration and the model builder. `Analysis/` holds gravity, modal, pushover, response-spectrum and NTHA drivers. `Loads/` holds gravity, ELF and ground-motion loading. `Data_Generation/` holds the parameterized dataset generator, the weighted seismic-intensity calibration and the graph and hybrid exporters. `Ground_Motions/` holds the record manifest and set definitions (raw and processed records stay out of git). `tools/` holds the roster builder, the batch driver and the multi-device calibration launchers. `tests/` holds the unit tests. `Main.py` builds, designs and analyses one frame by hand. |
| `RC Hybrid Surrogate Model/` | The GNN-LSTM surrogate: dataset schema, feature engineering, model, losses, metrics, training and evaluation scripts, and experiment configs. Its own `README.md` documents the baseline task, leakage control and training commands. |
| `structural-analysis-kit/` | A portable analysis workspace for inspecting and comparing generated datasets across devices, with per-device profiles kept out of git. See its `README.md`. |
| `projector_share/` | A TensorFlow Embedding Projector bundle of surrogate embeddings for the 3,800-case dataset. |
| `code_map/` | One node per script and one edge per import; `python code_map/build_code_map.py` regenerates it. |
| `cortex/` | CORTEX, a local knowledge base for the project's notes, facts, decisions and sessions (FastAPI + SQLite FTS5, React + Three.js brain view), seeded from the SeisFrame Obsidian vault. See its `README.md`. |
| `abstract_draft.md` | Research abstract. |
| `run_rc_structure.ps1` | Windows launcher that runs `RC Structure/Main.py` in the `OpPy` conda environment. |

## Design basis (2026-10-16)

- Perimeter special moment frames carry the seismic forces; interior beams and columns are the gravity system
  (ACI 318-19 18.14), designed for gravity and checked for compatibility with the design displacement
  (ASCE 7-22 12.12.5). One design per system: one perimeter column, one interior column, one perimeter beam, one
  interior beam, the same at every story; one concrete grade per member type.
- The iterative design moves concrete strength first, then reinforcement, then section size; columns are capped at
  32 in. The grouping policy, the lever order and every rule version are part of each design record's request
  identity, and qualification recomputes the evidence from the saved record.
- The nonlinear model keeps every member with its IMK hinges; the design credits the perimeter only.

## Running

Design a roster of structures (two-system basis, five cases at a time):

```
python "RC Structure/tools/build_lhs_pilot.py" --count 100 --seed <seed> --shards 1 --name <name> --out "RC Structure/pilots/<name>"
python "RC Structure/tools/run_grouped_batch.py" --manifest "RC Structure/pilots/<name>/screening_plan.json" --output-root <new dir> --workers 5 --probe-date <YYYY-MM-DD>
```

One case by hand: `python "RC Structure/Design/Grouped_Runner.py" --manifest <plan> --case-id case_0001 --seed-column 18 18 6 --seed-beam 14 20 5 --max-feasibility-trials 16 --max-reduction-trials 6 --output-dir <new dir>`.

Generation, calibration and training: `RC Structure/Data_Generation/Generate_Parameterized_Dataset.py`,
`Run_Weighted_Seismic_Calibration.py` (with the launchers in `tools/multi_device_calibration/`), and
`RC Hybrid Surrogate Model/train.py` (see that README).

## Environment

The generation pipeline runs in a conda environment named `OpPy`, specified in `RC Structure/environment-generation.yml`:
Python 3.12, NumPy, SciPy, Matplotlib, `openseespy==3.8.0.0`. The surrogate additionally needs PyTorch and PyTorch
Geometric (`RC Hybrid Surrogate Model/requirements.txt`). Tests: `python -m unittest discover -s "RC Structure/tests"`
from `RC Structure`.
