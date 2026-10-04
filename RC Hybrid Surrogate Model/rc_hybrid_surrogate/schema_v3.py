"""Reader for the hybrid_gnn_lstm_v3 sample contract.

The v2 pipeline predicted one roof-displacement trace from a pooled graph
embedding, so it only ever read six arrays. v3 exports the full structural
response -- per-floor kinematics, per-element forces, per-hinge rotations,
per-joint forces -- which is what the per-node and per-element decoders need.

Two things about v3 drive the design here.

Sampling rate is not uniform. Floor kinematics, story drift and base shear are
recorded every step; element forces, joint forces and hinge rotations are
recorded every NTHA_ELEMENT_HISTORY_STRIDE (8) steps. The strided arrays ship
with an explicit index array (element_force_steps, hinge_rotation_steps)
giving the exact step each row was taken at, so a decoder running at full rate
can be supervised on the strided quantities by gathering those indices. No
resampling or assumed alignment is required, and none should be introduced.

Node and element counts vary per sample. Every per-node and per-element array
is therefore ragged across a batch and has to be concatenated with an index
vector, the way torch_geometric batches node features, rather than stacked.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "hybrid_gnn_lstm_v3"

# Recorded every analysis step.
FULL_RATE_ARRAYS = (
    "floor_disp",
    "floor_vel",
    "floor_accel",
    "story_drift_history",
    "base_shear",
)

# Recorded every stride-th step. Each maps to the array naming the steps it
# was sampled at; that index array is what aligns it to a full-rate decoder.
STRIDED_ARRAYS = {
    "element_force_history": "element_force_steps",
    "hinge_rotation": "hinge_rotation_steps",
    # The conjugate spring moments (kip-in), read from the same committed state as hinge_rotation and
    # stored on the same rows; absent from samples compiled before 2026-10-01 (then simply skipped).
    "hinge_moment": "hinge_rotation_steps",
    # Joint forces are written on the element stride and share its index array.
    "joint_force_history": "element_force_steps",
}

# Per-entity histories are exported FLATTENED to 2-D: [num_sampled_steps,
# num_entities * num_components], entity-major, so column block
# j*C:(j+1)*C belongs to entity_order[j]. Verified against the force
# envelopes on real exporter output: the entity-major reshape never exceeds
# the full-rate envelope, while a component-major read violates that bound
# on 599 of 1064 element ends. Each entry maps the array to the npz array
# naming its entity order, and to the metadata key naming its columns (None
# where the exporter writes no column list and the width must be inferred).
ENTITY_LAYOUTS = {
    "element_force_history": ("element_force_tag_order", "element_force_history_columns"),
    "hinge_rotation": ("hinge_tag_order", None),
    # Two components per hinge, local y (direction 5) then local z (direction 6), as hinge_rotation.
    "hinge_moment": ("hinge_tag_order", "hinge_history_columns"),
    # Joints are a SUBSET of nodes -- interior joints only, not base or
    # floor-master nodes -- so this order is what a per-node decoder must
    # target. It is shorter than num_nodes and must never be assumed equal.
    "joint_force_history": ("joint_force_node_order", "joint_force_history_columns"),
}

# Present in v2 as well; the conditioning inputs and the graph itself.
GRAPH_ARRAYS = ("x", "edge_index", "edge_attr", "element_attr")
CONDITION_ARRAYS = ("global_features", "record_features", "ground_motion")


# Interstory drift above this is a diverged solve, not a structure. Collapse
# is conventionally 5-10% drift; 20% leaves headroom for a genuine collapse
# still being integrated. Mirrors COLLAPSE_DRIFT_CEILING in the generation
# side's Calibrate_Intensity, and the two must not drift apart.
PHYSICAL_DRIFT_CEILING = 0.20


class SchemaError(RuntimeError):
    """Raised when a sample does not satisfy the v3 contract."""


@dataclass(frozen=True)
class SampleV3:
    """One decoded v3 sample. Arrays are numpy; nothing is normalized yet."""

    sample_id: str
    path: Path
    metadata: dict
    arrays: dict

    # -- graph ------------------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return int(self.arrays["x"].shape[0])

    @property
    def num_elements(self) -> int:
        return int(self.arrays["element_attr"].shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.arrays["ground_motion"].shape[0])

    def get(self, name: str):
        return self.arrays.get(name)

    def require(self, name: str):
        value = self.arrays.get(name)
        if value is None:
            raise SchemaError(f"{self.path} is missing required array {name!r}")
        return value

    # -- strided alignment ------------------------------------------------
    def strided_steps(self, name: str) -> np.ndarray:
        """Step indices at which a strided array was sampled.

        Returned as int64 suitable for gathering along a full-rate time axis.
        """
        if name not in STRIDED_ARRAYS:
            raise KeyError(f"{name!r} is not a strided array; expected one of {sorted(STRIDED_ARRAYS)}")
        steps = self.arrays.get(STRIDED_ARRAYS[name])
        if steps is None:
            raise SchemaError(
                f"{self.path} has {name!r} but not its step index array "
                f"{STRIDED_ARRAYS[name]!r}; the two cannot be aligned without it."
            )
        return np.asarray(steps, dtype=np.int64).reshape(-1)

    # -- per-entity history layout ----------------------------------------
    def entity_order(self, name: str) -> np.ndarray:
        """Tags of the entities the columns of a flattened history refer to."""
        if name not in ENTITY_LAYOUTS:
            raise KeyError(
                f"{name!r} is not a per-entity history; expected one of "
                f"{sorted(ENTITY_LAYOUTS)}"
            )
        order_name = ENTITY_LAYOUTS[name][0]
        order = self.arrays.get(order_name)
        if order is None:
            raise SchemaError(
                f"{self.path} has {name!r} but not its entity order array "
                f"{order_name!r}; its columns cannot be attributed to entities."
            )
        return np.asarray(order).reshape(-1)

    def num_components(self, name: str) -> int:
        """Components per entity in a flattened history."""
        array = self.require(name)
        width = int(array.shape[1])
        count = int(self.entity_order(name).shape[0])
        if count == 0 or width % count:
            raise SchemaError(
                f"{self.path}: {name} is {width} columns wide, which is not a "
                f"whole multiple of its {count} entities."
            )
        return width // count

    def as_entity_history(self, name: str) -> np.ndarray:
        """A flattened history as [sampled_steps, entities, components].

        The exporter writes these 2-D; the decoders want the entity axis
        separated. Entity j of the result is entity_order(name)[j].
        """
        array = self.require(name)
        return array.reshape(array.shape[0], -1, self.num_components(name))

    @property
    def analysis_failed(self) -> bool:
        """True when the NTHA did not run to completion for this sample."""
        return bool(self.metadata.get("analysis_failed", False))

    def ground_motion_present(self) -> np.ndarray:
        """[x_present, y_present]; a zero column is absent, not measured zero."""
        present = self.arrays.get("ground_motion_present")
        if present is None:
            # Written from the first generation run that included the mask.
            # Older v3 samples predate it; both components were applied unless
            # the metadata says the run was x-only.
            x_only = bool(self.metadata.get("x_only", False))
            present = np.asarray([1.0, 0.0 if x_only else 1.0], dtype=np.float32)
        return np.asarray(present, dtype=np.float32).reshape(-1)


def _read_metadata(sample_path: Path) -> dict:
    metadata_path = sample_path.with_name("hybrid_metadata.json")
    if not metadata_path.exists():
        raise SchemaError(f"No hybrid_metadata.json beside {sample_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_sample(sample_path: str | Path, strict: bool = True) -> SampleV3:
    """Load one hybrid_sample.npz as a v3 sample.

    strict=True rejects anything that is not schema v3. A v2 sample is
    rejected with a message naming the version rather than failing later on a
    missing array, because a v2 file is structurally valid and would otherwise
    silently train a v3 model on roof displacement alone.
    """
    sample_path = Path(sample_path)
    metadata = _read_metadata(sample_path)
    version = metadata.get("schema_version")
    if strict and version != SCHEMA_VERSION:
        raise SchemaError(
            f"{sample_path} is schema {version!r}, not {SCHEMA_VERSION!r}. "
            "Regenerate the dataset, or read it with the v2 loader."
        )

    with np.load(sample_path, allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}

    sample_id = str(metadata.get("run_name") or sample_path.parent.name)
    sample = SampleV3(
        sample_id=sample_id, path=sample_path, metadata=metadata, arrays=arrays
    )
    if strict:
        validate(sample)
    return sample


def validate(sample: SampleV3) -> None:
    """Check the invariants a decoder depends on, with actionable messages."""
    for name in GRAPH_ARRAYS + CONDITION_ARRAYS:
        sample.require(name)

    num_nodes = sample.num_nodes
    num_elements = sample.num_elements
    num_steps = sample.num_steps

    edge_index = sample.require("edge_index")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise SchemaError(f"{sample.path}: edge_index must be [2, num_edges]")
    if edge_index.size and int(edge_index.max()) >= num_nodes:
        raise SchemaError(
            f"{sample.path}: edge_index references node "
            f"{int(edge_index.max())} but only {num_nodes} nodes exist"
        )

    for name in FULL_RATE_ARRAYS:
        array = sample.get(name)
        if array is None:
            continue
        if array.shape[0] != num_steps:
            raise SchemaError(
                f"{sample.path}: {name} has {array.shape[0]} steps but the "
                f"ground motion has {num_steps}; full-rate arrays must match."
            )

    for name in STRIDED_ARRAYS:
        array = sample.get(name)
        if array is None:
            continue
        steps = sample.strided_steps(name)
        if array.shape[0] != steps.shape[0]:
            raise SchemaError(
                f"{sample.path}: {name} has {array.shape[0]} rows but its step "
                f"index has {steps.shape[0]}; they must correspond one to one."
            )
        if steps.size and int(steps.max()) >= num_steps:
            raise SchemaError(
                f"{sample.path}: {name} is sampled at step {int(steps.max())} "
                f"but only {num_steps} steps were run."
            )

    # Per-entity histories are 2-D and entity-major. Check the width divides
    # by the entity order, which is what makes the reshape well defined; an
    # earlier version of this checked shape[1] against num_elements on a 3-D
    # array, which never fired because the exporter writes 2-D.
    for name, (order_name, column_key) in ENTITY_LAYOUTS.items():
        array = sample.get(name)
        if array is None:
            continue
        if array.ndim != 2:
            raise SchemaError(
                f"{sample.path}: {name} has {array.ndim} dimensions; per-entity "
                "histories are exported as [steps, entities * components]."
            )
        components = sample.num_components(name)  # raises if width is ragged
        columns = sample.metadata.get(column_key) if column_key else None
        if columns is not None and len(columns) != components:
            raise SchemaError(
                f"{sample.path}: {name} implies {components} components per "
                f"entity but {column_key} names {len(columns)}."
            )

    # A diverged solve exports structurally valid arrays holding nonsense.
    # pilot30_s3 case_0003 collapsed legitimately at ~5.6% drift, then the
    # solver blew up and wrote a drift ratio of 135 (13,508%) with hinge
    # rotations of 1.9e12 rad. The values are finite, so a NaN check misses
    # them, and one such sample would destroy target normalisation for the
    # whole training set.
    drift_peaks = sample.get("story_drift_peaks")
    if drift_peaks is not None and drift_peaks.size:
        worst = float(np.abs(drift_peaks).max())
        if not np.isfinite(worst) or worst > PHYSICAL_DRIFT_CEILING:
            raise SchemaError(
                f"{sample.path}: peak story drift {worst:.4g} exceeds the "
                f"physical ceiling {PHYSICAL_DRIFT_CEILING}; the analysis "
                f"diverged (metadata analysis_failed="
                f"{sample.metadata.get('analysis_failed')}). Exclude this run "
                "rather than training on it."
            )

    for name in FULL_RATE_ARRAYS + tuple(STRIDED_ARRAYS):
        array = sample.get(name)
        if array is not None and array.size and not np.isfinite(array).all():
            raise SchemaError(f"{sample.path}: {name} holds non-finite values")

    element_forces = sample.get("element_force_history")
    if element_forces is not None:
        covered = int(sample.entity_order("element_force_history").shape[0])
        if covered != num_elements:
            raise SchemaError(
                f"{sample.path}: element_force_history covers {covered} "
                f"elements but element_attr has {num_elements}."
            )
