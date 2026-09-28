"""
Analysis/NTHA.py
================
Nonlinear Time-History Analysis (NLTHA) for the RC 3D MRF.

Design decisions
----------------
* Follows the same step-by-step convergence-recovery pattern as Pushover.py.
* Element force envelopes are tracked in-memory at every converged step
  using the same ``localForce`` query as Graph_Exporter, so the results are
  directly compatible with ``collect_element_end_force_rows()``.
* Rayleigh damping is applied with mass-proportional (a0) and
  initial-stiffness-proportional (a1) coefficients derived from two modal
  periods (default: modes 1 and 3, which for a symmetric 3D frame capture
  both translational directions).
* Gravity loads must already be applied before calling ``run_ntha()``.
  The function adds the ground motion pattern on top of the existing
  gravity state.

Return value
------------
dict with keys:
    status              convergence summary (see _make_status)
    time_history        list[float]  – time at each completed step (s)
    roof_disp_x         list[float]  – roof X-displacement history (in)
    roof_disp_y         list[float]  – roof Y-displacement history (in)
    peak_story_drift_x  list[float]  – per-story peak |drift ratio| in X
    peak_story_drift_y  list[float]  – per-story peak |drift ratio| in Y
    element_envelope    dict  {ele_tag: {'max': [12 floats], 'min': [12 floats]}}
    node_envelope       dict  {node_tag: {'max': [6 floats], 'min': [6 floats]}}
    record_summary_x    dict  – summarize_record() output for X ground motion
    record_summary_y    dict  – summarize_record() output for Y GM, or None
    damping_ratio       float
    rayleigh_a0         float  – mass coefficient
    rayleigh_a1         float  – initial-stiffness coefficient
"""

import math
import sys
from pathlib import Path

import openseespy.opensees as ops

import Structure_Parameters as sp
from Model.Build_Model import apply_analysis_constraints, floor_master_node, node_tag
from Loads.Ground_Motion import (
    GroundMotionRecord,
    define_path_time_series,
    apply_uniform_excitation,
    summarize_record,
)
from Model.IMK_Hinges import hinge_registry


# ================================================================
#  CONSTANTS
# ================================================================

# Pattern / time-series tags — chosen to not collide with gravity (pattern 1)
_TS_TAG_X   = 100
_TS_TAG_Y   = 101
_PAT_TAG_X  = 100
_PAT_TAG_Y  = 101

# Default NLTHA tolerances and iteration limits
_TOL_PRIMARY  = 1.0e-6
_ITER_PRIMARY = 25
_TOL_FALLBACK = 1.0e-4
_ITER_FALLBACK = 50
_DT_SUBDIVIDE  = 10      # subdivide dt by this factor in last-resort recovery


# ================================================================
#  ELEMENT TAG ENUMERATION
#  Mirrors the sequential tagging in Graph_Exporter.collect_element_rows()
# ================================================================

def _physical_element_tags():
    """
    Return all physical element tags in the same order as Graph_Exporter.
    IMK hinge zero-length elements (tag >= IMK_HINGE_ELEMENT_TAG_BASE) are
    excluded; their contribution to beam-end moments is captured through
    the elastic beam element forces plus the hinge element envelope.
    """
    n_col    = sp.NUM_FLOOR * (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1)
    n_beam_x = sp.NUM_FLOOR * sp.NUM_BAY_X       * (sp.NUM_BAY_Y + 1)
    n_beam_y = sp.NUM_FLOOR * (sp.NUM_BAY_X + 1) * sp.NUM_BAY_Y
    return list(range(1, n_col + n_beam_x + n_beam_y + 1))


def _joint_spring_element_tags():
    """The scissors joint springs (Model/Joint_Springs), which sit between the members and the hinges."""
    base = getattr(sp, "JOINT_ELEMENT_TAG_BASE", None)
    if base is None:
        return []
    try:
        all_tags = ops.getEleTags()
    except Exception:
        return []
    return sorted(t for t in all_tags if base <= t < sp.IMK_HINGE_ELEMENT_TAG_BASE)


def _imk_hinge_element_tags():
    """
    Return all IMK hinge zeroLength element tags.
    These are tracked separately because their Mz response gives the
    connection moment, which is more meaningful than the elastic spine Mz.
    Only populated when IMK_APPLY_TO_BEAMS or IMK_APPLY_TO_COLUMNS is True.
    """
    # Hinge elements are built inside Model/IMK_Hinges.py with a known
    # tag base. We query the domain for all elements and filter by base.
    base = sp.IMK_HINGE_ELEMENT_TAG_BASE
    try:
        all_tags = ops.getEleTags()
    except Exception:
        return []
    return [t for t in all_tags if t >= base]


# ================================================================
#  RAYLEIGH DAMPING
# ================================================================

def _compute_rayleigh(modal_results, damping_ratio, mode_i=0, mode_j=2):
    """
    Compute mass-proportional (a0) and initial-stiffness-proportional (a1)
    coefficients for Rayleigh damping targeting ``damping_ratio`` at two modes.

    Parameters
    ----------
    modal_results : list of dicts from run_modal_analysis()
    damping_ratio : float  (fraction, e.g. 0.05 for 5%)
    mode_i, mode_j : 0-based indices into the valid modal results list
    """
    valid = [m for m in modal_results if m["valid"]]
    if len(valid) < max(mode_i, mode_j) + 1:
        raise ValueError(
            f"Not enough valid modes for Rayleigh damping. "
            f"Need modes {mode_i} and {mode_j} (0-based); "
            f"got {len(valid)} valid modes."
        )
    omega_i = valid[mode_i]["omega"]
    omega_j = valid[mode_j]["omega"]
    a0 = 2.0 * damping_ratio * omega_i * omega_j / (omega_i + omega_j)
    a1 = 2.0 * damping_ratio / (omega_i + omega_j)
    return a0, a1


# ================================================================
#  FORCE / DISPLACEMENT QUERIES
# ================================================================

def _query_element_forces(ele_tags):
    """
    Query local element forces for every tag.
    Returns {ele_tag: [f0..f11]} (12 local DOFs: 6 at i-end, 6 at j-end).
    """
    forces = {}
    for tag in ele_tags:
        f = ops.eleResponse(tag, "localForce")
        if not f:
            f = ops.eleForce(tag)
        forces[tag] = list(f)
    return forces


def _update_envelope(envelope, current_forces):
    """In-place update of max/min envelope dict."""
    for tag, forces in current_forces.items():
        if tag not in envelope:
            envelope[tag] = {
                "max": list(forces),
                "min": list(forces),
            }
        else:
            env = envelope[tag]
            for k, v in enumerate(forces):
                if v > env["max"][k]:
                    env["max"][k] = v
                if v < env["min"][k]:
                    env["min"][k] = v


def _query_node_displacements(tags):
    """Returns {node_tag: [ux, uy, uz, rx, ry, rz]}."""
    return {
        tag: [ops.nodeDisp(tag, dof) for dof in range(1, 7)]
        for tag in tags
    }


def _update_node_envelope(envelope, current_disps):
    """In-place update of node displacement envelope."""
    for tag, disps in current_disps.items():
        if tag not in envelope:
            envelope[tag] = {"max": list(disps), "min": list(disps)}
        else:
            env = envelope[tag]
            for k, v in enumerate(disps):
                if v > env["max"][k]:
                    env["max"][k] = v
                if v < env["min"][k]:
                    env["min"][k] = v


def _roof_disp():
    """Current X and Y displacement of the roof diaphragm master node."""
    roof = floor_master_node(sp.NUM_FLOOR)
    return ops.nodeDisp(roof, 1), ops.nodeDisp(roof, 2)


def _current_story_drifts():
    """
    Current absolute interstory drift ratios per story in X and Y.
    Returns (list_x, list_y), each of length NUM_FLOOR.
    """
    dx, dy = [], []
    prev_ux = prev_uy = 0.0
    for k in range(1, sp.NUM_FLOOR + 1):
        m = floor_master_node(k)
        ux, uy = ops.nodeDisp(m, 1), ops.nodeDisp(m, 2)
        dx.append(abs(ux - prev_ux) / sp.STORY_H)
        dy.append(abs(uy - prev_uy) / sp.STORY_H)
        prev_ux, prev_uy = ux, uy
    return dx, dy


def _all_non_base_node_tags():
    """All structural (non-base) node tags in floor order."""
    tags = []
    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                tags.append(node_tag(k, i, j))
    return tags


def _floor_master_nodes():
    """Diaphragm master node for every elevated floor, in floor order."""
    return [floor_master_node(k) for k in range(1, sp.NUM_FLOOR + 1)]


def _query_floor_kinematics(masters):
    """Displacement, velocity and acceleration at each floor, X and Y.

    Returns three flat lists of length 2 * NUM_FLOOR ordered
    [floor1_x, floor1_y, floor2_x, ...]. Velocity and acceleration are
    recorded because the equation of motion and any energy balance need them;
    displacement alone cannot express either. Accelerations are relative to
    the ground, matching the UniformExcitation formulation.
    """
    displacement, velocity, acceleration = [], [], []
    for master in masters:
        for dof in (1, 2):
            displacement.append(ops.nodeDisp(master, dof))
            velocity.append(ops.nodeVel(master, dof))
            acceleration.append(ops.nodeAccel(master, dof))
    return displacement, velocity, acceleration


def _base_story_column_tags():
    """Element tags of the first-story columns.

    create_column_elements numbers columns with the story index outermost
    starting at tag 1, so the first (nx+1)(ny+1) tags are the base story.
    """
    per_story = (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1)
    return list(range(1, per_story + 1))


def _base_shear_xy(base_column_tags):
    """Total base shear in global X and Y, kips.

    Summed from the first-story column forces rather than from support
    reactions. With IMK columns the member attaches to a hinge node that is
    tied to the fixed base node through equalDOF, and nodeReaction does not
    report force transferred across a multi-point constraint -- it returns
    zero translational reaction, which silently reads as zero base shear.
    Column forces are also already global, and cost less than a reactions()
    assembly.
    """
    shear_x = 0.0
    shear_y = 0.0
    for tag in base_column_tags:
        forces = ops.eleForce(tag)
        if not forces:
            continue
        shear_x += forces[0]
        shear_y += forces[1]
    return -shear_x, -shear_y


def _element_node_map(ele_tags):
    """Structural end nodes for each physical element.

    For an IMK member the element spans two zero-length hinge nodes, so
    ops.eleNodes reports those rather than the joints the member frames into.
    Assembling joint forces from them silently matches nothing and yields an
    all-zero result, so the physical end nodes recorded at build time are
    preferred wherever they exist.
    """
    registry = hinge_registry()
    node_map = {}
    for tag in ele_tags:
        entry = registry.get(int(tag))
        if entry and "node_i" in entry:
            node_map[tag] = [entry["node_i"], entry["node_j"]]
        else:
            node_map[tag] = list(ops.eleNodes(tag))
    return node_map


def _assemble_joint_forces(ele_tags, node_map, node_positions):
    """Internal force delivered to each joint, summed over framing members.

    Element end forces are summed in the GLOBAL frame. Local forces cannot be
    added across members that point in different directions, so the local
    values recorded for element-level targets are not usable here.

    Returns a flat list of 6 * len(node_positions) values, ordered by the
    node_positions mapping.
    """
    totals = [0.0] * (6 * len(node_positions))
    for tag in ele_tags:
        forces = ops.eleForce(tag)
        if not forces or len(forces) < 12:
            continue
        for end, node in enumerate(node_map.get(tag, ())):
            slot = node_positions.get(node)
            if slot is None:
                continue
            offset = 6 * slot
            base = 6 * end
            for dof in range(6):
                totals[offset + dof] += forces[base + dof]
    return totals


def _query_hinge_rotations(hinge_ele_tags):
    """Rotation of every IMK spring, keyed by hinge element tag.

    The hinge force envelope saturates at the capping strength, so it cannot
    distinguish a spring that just yielded from one about to fail. Rotation is
    the only quantity that separates them, and it is what every damage measure
    downstream is built from.
    """
    rotations = {}
    for tag in hinge_ele_tags:
        response = ops.eleResponse(tag, "deformation")
        if response:
            rotations[tag] = list(response)
    return rotations


def _update_rotation_envelope(envelope, current):
    """Track peak positive, negative and absolute rotation per hinge."""
    for tag, values in current.items():
        entry = envelope.get(tag)
        if entry is None:
            envelope[tag] = {
                "max": list(values),
                "min": list(values),
                "abs_max": [abs(value) for value in values],
            }
            continue
        for index, value in enumerate(values):
            if value > entry["max"][index]:
                entry["max"][index] = value
            if value < entry["min"][index]:
                entry["min"][index] = value
            if abs(value) > entry["abs_max"][index]:
                entry["abs_max"][index] = abs(value)


# ================================================================
#  CONVERGENCE / RECOVERY
# ================================================================

def _set_primary_algorithm():
    ops.test("NormDispIncr", _TOL_PRIMARY, _ITER_PRIMARY, 0)
    ops.algorithm("Newton")


def _try_step(dt):
    """
    Try one transient step; fall back through recovery strategies if needed.

    Returns (ok, strategy_name)
    """
    # Primary: Newton
    _set_primary_algorithm()
    ok = ops.analyze(1, dt)
    if ok == 0:
        return 0, "Newton"

    # Fallback 1: KrylovNewton, relaxed tolerance
    ops.test("NormDispIncr", _TOL_FALLBACK, _ITER_FALLBACK, 0)
    ops.algorithm("KrylovNewton")
    ok = ops.analyze(1, dt)
    _set_primary_algorithm()
    if ok == 0:
        return 0, "KrylovNewton"

    # Fallback 2: subdivide the step
    ops.test("NormDispIncr", _TOL_FALLBACK, _ITER_FALLBACK, 0)
    ops.algorithm("KrylovNewton")
    dt_sub = dt / _DT_SUBDIVIDE
    ok = ops.analyze(_DT_SUBDIVIDE, dt_sub)
    _set_primary_algorithm()
    if ok == 0:
        return 0, f"KrylovNewton+subdivide({_DT_SUBDIVIDE}x)"

    return ok, "failed"


# ================================================================
#  LOGGING
# ================================================================

def _open_log(log_path):
    if log_path is None:
        return None
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path.open("w", encoding="utf-8")


def _log(fh, msg):
    if fh is not None:
        fh.write(msg + "\n")
        fh.flush()


# ================================================================
#  STATUS HELPERS
# ================================================================


def _summarize_convergence_log(convergence_log, completed_steps):
    """Return compact convergence-quality metrics for status.json."""
    entries = convergence_log or []
    recovery_entries = [entry for entry in entries if entry.get("ok")]
    failed_entries = [entry for entry in entries if not entry.get("ok")]

    strategy_counts = {}
    for entry in recovery_entries:
        strategy = entry.get("strategy", "unknown")
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

    max_consecutive = 0
    current_consecutive = 0
    previous_step = None
    for entry in recovery_entries:
        step = entry.get("step")
        if isinstance(step, int) and previous_step is not None and step == previous_step + 1:
            current_consecutive += 1
        else:
            current_consecutive = 1
        previous_step = step
        if current_consecutive > max_consecutive:
            max_consecutive = current_consecutive

    recovery_count = len(recovery_entries)
    recovery_rate = recovery_count / completed_steps if completed_steps else 0.0

    return {
        "num_recovery_steps": recovery_count,
        "recovery_rate": recovery_rate,
        "recovery_strategy_counts": strategy_counts,
        "max_consecutive_recovery_steps": max_consecutive,
        "num_failed_recovery_steps": len(failed_entries),
    }


def _make_status(
    npts_requested,
    completed_steps,
    failed,
    failed_step=None,
    failed_time=None,
    convergence_log=None,
):
    convergence_log = convergence_log or []
    convergence_summary = _summarize_convergence_log(convergence_log, completed_steps)
    return {
        "npts_requested":       npts_requested,
        "completed_steps":      completed_steps,
        "failed":               failed,
        "failed_step":          failed_step,
        "failed_time_sec":      failed_time,
        "convergence_summary":  convergence_summary,
        "convergence_log":      convergence_log,
    }


def _analysis_step_schedule(record_x, record_y=None, dt_factor=1.0):
    if dt_factor <= 0.0:
        raise ValueError("dt_factor must be positive.")

    dt = record_x.dt_sec * dt_factor
    excitation_duration = record_x.dt_sec * record_x.npts
    if record_y is not None:
        excitation_duration = max(
            excitation_duration,
            record_y.dt_sec * record_y.npts,
        )
    n_steps = max(1, int(math.ceil(excitation_duration / dt - 1.0e-12)))
    return dt, n_steps


# ================================================================
#  MAIN NTHA FUNCTION
# ================================================================

def run_ntha(
    record_x: GroundMotionRecord,
    record_y: GroundMotionRecord | None = None,
    damping_ratio: float = 0.05,
    modal_results: list | None = None,
    rayleigh_mode_i: int = 0,
    rayleigh_mode_j: int = 2,
    dt_factor: float = 1.0,
    log_path=None,
):
    """
    Run NLTHA on the current OpenSees model (gravity already applied).

    Parameters
    ----------
    record_x : GroundMotionRecord
        X-direction (DOF 1) ground motion.
    record_y : GroundMotionRecord, optional
        Y-direction (DOF 2) ground motion. If None, X-only excitation.
    damping_ratio : float
        Target modal damping ratio (default 0.05 = 5%).
    modal_results : list of dicts
        Output of ``run_modal_analysis()``. Required for Rayleigh damping.
        If None, a fallback based on T1 = 1.0 s is used (not recommended).
    rayleigh_mode_i, rayleigh_mode_j : int
        0-based indices into the valid modal results for Rayleigh targeting.
        Default (0, 2) → modes 1 & 3 (captures both translational periods
        for a symmetric 3D frame).
    dt_factor : float
        Multiply the record dt by this factor (use < 1.0 to sub-step).
    log_path : str or Path, optional
        Path for a plain-text convergence log.

    Returns
    -------
    dict  (see module docstring)
    """
    log_fh = _open_log(log_path)

    try:
        return _run_ntha_impl(
            record_x, record_y,
            damping_ratio, modal_results,
            rayleigh_mode_i, rayleigh_mode_j,
            dt_factor, log_fh,
        )
    finally:
        if log_fh is not None:
            log_fh.close()


def _run_ntha_impl(
    record_x, record_y,
    damping_ratio, modal_results,
    rayleigh_mode_i, rayleigh_mode_j,
    dt_factor, log_fh,
):
    # ------------------------------------------------------------------
    # 1. Rayleigh damping
    # ------------------------------------------------------------------
    if modal_results is not None:
        a0, a1 = _compute_rayleigh(
            modal_results, damping_ratio,
            rayleigh_mode_i, rayleigh_mode_j,
        )
    else:
        # Fallback: assume T1 = 1.0 s, T2 = 0.3 s
        _log(log_fh, "[WARNING] modal_results not provided; using T1=1.0s, T2=0.3s for Rayleigh.")
        omega1 = 2.0 * math.pi / 1.0
        omega2 = 2.0 * math.pi / 0.3
        a0 = 2.0 * damping_ratio * omega1 * omega2 / (omega1 + omega2)
        a1 = 2.0 * damping_ratio / (omega1 + omega2)

    ops.rayleigh(a0, 0.0, a1, 0.0)   # mass + initial-stiffness proportional
    _log(log_fh, f"Rayleigh: a0={a0:.6e}, a1={a1:.6e}, zeta={damping_ratio:.3f}")

    # ------------------------------------------------------------------
    # 2. Ground motion time series and patterns
    # ------------------------------------------------------------------
    define_path_time_series(_TS_TAG_X, record_x)
    apply_uniform_excitation(_PAT_TAG_X, _TS_TAG_X, direction=1)

    if record_y is not None:
        define_path_time_series(_TS_TAG_Y, record_y)
        apply_uniform_excitation(_PAT_TAG_Y, _TS_TAG_Y, direction=2)

    # ------------------------------------------------------------------
    # 3. Analysis settings
    # ------------------------------------------------------------------
    apply_analysis_constraints()
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")
    _set_primary_algorithm()

    # ------------------------------------------------------------------
    # 4. Collect element / node tag lists
    # ------------------------------------------------------------------
    phys_ele_tags  = _physical_element_tags()
    hinge_ele_tags = _imk_hinge_element_tags()
    joint_ele_tags = _joint_spring_element_tags()
    non_base_nodes = _all_non_base_node_tags()

    # ------------------------------------------------------------------
    # 5. Analysis loop
    # ------------------------------------------------------------------
    dt, n_steps = _analysis_step_schedule(record_x, record_y, dt_factor)

    time_history   = []
    roof_disp_x    = []
    roof_disp_y    = []
    peak_drift_x   = [0.0] * sp.NUM_FLOOR
    peak_drift_y   = [0.0] * sp.NUM_FLOOR

    element_envelope = {}
    hinge_envelope   = {}
    node_envelope    = {}

    # Spatial and kinematic response. Previously the only stored time series
    # was roof displacement, so the per-step cost of querying the whole model
    # was paid and then discarded, and nothing downstream could see how
    # response was distributed over the height.
    floor_masters  = _floor_master_nodes()
    base_columns   = _base_story_column_tags()
    floor_disp_history  = []
    floor_vel_history   = []
    floor_accel_history = []
    story_drift_history = []
    base_shear_x = []
    base_shear_y = []

    # Element end forces over time. The envelope query below already
    # computes these every step; only the strided copies are kept, since at
    # full resolution they would dominate dataset storage.
    element_stride = max(1, int(getattr(sp, "NTHA_ELEMENT_HISTORY_STRIDE", 8)))
    element_tag_order = list(phys_ele_tags)
    element_force_history = []
    element_history_steps = []

    joint_node_order = list(non_base_nodes)
    joint_node_positions = {tag: index for index, tag in enumerate(joint_node_order)}
    element_node_map = _element_node_map(element_tag_order)
    joint_force_history = []

    hinge_rotation_envelope = {}
    hinge_rotation_history  = []
    hinge_history_steps     = []
    hinge_stride = max(1, int(getattr(sp, "NTHA_HINGE_HISTORY_STRIDE", 8)))
    hinge_tag_order = list(hinge_ele_tags)
    joint_rotation_envelope = {}          # joint springs: (Rx, Ry) per joint element, peaks every step

    convergence_log = []
    failed          = False
    failed_step     = None
    failed_time     = None

    _log(
        log_fh,
        f"Starting NLTHA: record={record_x.record_id}, "
        f"record_npts={record_x.npts}, analysis_steps={n_steps}, dt={dt:.6f}s",
    )

    for step in range(n_steps):
        ok, strategy = _try_step(dt)

        if ok != 0:
            failed      = True
            failed_step = step
            failed_time = step * dt
            msg = f"Step {step}: FAILED after all recovery strategies at t={failed_time:.4f}s"
            _log(log_fh, msg)
            convergence_log.append({"step": step, "strategy": "failed", "ok": False})
            break

        if strategy != "Newton":
            msg = f"Step {step}: recovered with {strategy}"
            _log(log_fh, msg)
            convergence_log.append({"step": step, "strategy": strategy, "ok": True})

        # -- Snapshot --
        t    = ops.getTime()
        ux_r, uy_r = _roof_disp()
        dx, dy = _current_story_drifts()

        time_history.append(t)
        roof_disp_x.append(ux_r)
        roof_disp_y.append(uy_r)

        displacement, velocity, acceleration = _query_floor_kinematics(floor_masters)
        floor_disp_history.append(displacement)
        floor_vel_history.append(velocity)
        floor_accel_history.append(acceleration)
        story_drift_history.append(list(dx) + list(dy))

        shear_x, shear_y = _base_shear_xy(base_columns)
        base_shear_x.append(shear_x)
        base_shear_y.append(shear_y)

        for k in range(sp.NUM_FLOOR):
            if dx[k] > peak_drift_x[k]:
                peak_drift_x[k] = dx[k]
            if dy[k] > peak_drift_y[k]:
                peak_drift_y[k] = dy[k]

        # Element force envelopes
        element_forces = _query_element_forces(phys_ele_tags)
        _update_envelope(element_envelope, element_forces)

        if step % element_stride == 0:
            element_history_steps.append(step)
            element_force_history.append(
                [
                    value
                    for tag in element_tag_order
                    for value in (element_forces.get(tag) or [0.0] * 12)[:12]
                ]
            )
            joint_force_history.append(
                _assemble_joint_forces(
                    element_tag_order, element_node_map, joint_node_positions
                )
            )
        if hinge_ele_tags:
            _update_envelope(hinge_envelope, _query_element_forces(hinge_ele_tags))

            # Peaks every step so none is missed; history decimated.
            rotations = _query_hinge_rotations(hinge_tag_order)
            _update_rotation_envelope(hinge_rotation_envelope, rotations)
            if step % hinge_stride == 0:
                hinge_history_steps.append(step)
                hinge_rotation_history.append(
                    [value for tag in hinge_tag_order for value in rotations.get(tag, (0.0, 0.0))]
                )
        if joint_ele_tags:
            _update_rotation_envelope(joint_rotation_envelope, _query_hinge_rotations(joint_ele_tags))

        # Node displacement envelopes
        _update_node_envelope(node_envelope, _query_node_displacements(non_base_nodes))

    # ------------------------------------------------------------------
    # 6. Assemble and return results
    # ------------------------------------------------------------------
    ops.wipeAnalysis()

    status = _make_status(
        npts_requested  = n_steps,
        completed_steps = len(time_history),
        failed          = failed,
        failed_step     = failed_step,
        failed_time     = failed_time,
        convergence_log = convergence_log,
    )

    _log(
        log_fh,
        f"Completed {status['completed_steps']}/{n_steps} steps. "
        f"{'FAILED' if failed else 'OK'}"
    )

    record_summary_x = summarize_record(record_x)
    record_summary_x.update(
        {
            "analysis_dt_sec": dt,
            "analysis_steps": n_steps,
        }
    )
    record_summary_y = summarize_record(record_y) if record_y else None
    if record_summary_y is not None:
        record_summary_y.update(
            {
                "analysis_dt_sec": dt,
                "analysis_steps": n_steps,
            }
        )

    return {
        "status":             status,
        "time_history":       time_history,
        "roof_disp_x":        roof_disp_x,
        "roof_disp_y":        roof_disp_y,
        "peak_story_drift_x": peak_drift_x,
        "peak_story_drift_y": peak_drift_y,
        "element_envelope":   element_envelope,
        "hinge_envelope":     hinge_envelope,   # IMK spring forces
        "hinge_rotation_envelope": hinge_rotation_envelope,
        "joint_rotation_envelope": joint_rotation_envelope,
        "hinge_rotation_history":  hinge_rotation_history,
        "hinge_rotation_steps":    hinge_history_steps,
        "hinge_tag_order":         hinge_tag_order,
        "element_force_history":   element_force_history,
        "element_force_steps":     element_history_steps,
        "element_tag_order":       element_tag_order,
        "element_history_stride":  element_stride,
        "joint_force_history":     joint_force_history,
        "joint_node_order":        joint_node_order,
        "hinge_history_stride":    hinge_stride,
        "node_envelope":      node_envelope,
        "floor_disp_history":  floor_disp_history,
        "floor_vel_history":   floor_vel_history,
        "floor_accel_history": floor_accel_history,
        "story_drift_history": story_drift_history,
        "base_shear_x":        base_shear_x,
        "base_shear_y":        base_shear_y,
        "floor_master_nodes":  floor_masters,
        "record_summary_x":   record_summary_x,
        "record_summary_y":   record_summary_y,
        "damping_ratio":      damping_ratio,
        "rayleigh_a0":        a0,
        "rayleigh_a1":        a1,
    }


# ================================================================
#  GRAPH EXPORTER EXTENSION
#  Converts NTHA element_envelope into the same row format as
#  Graph_Exporter.collect_element_end_force_rows(), ready to merge
#  with the static element table for GNN training.
# ================================================================

_FORCE_KEYS = [
    "axial_kip",
    "shear_y_kip",
    "shear_z_kip",
    "torsion_kip_in",
    "moment_y_kip_in",
    "moment_z_kip_in",
]

_END_OFFSETS = {"i": 0, "j": 6}


def envelope_to_end_force_rows(element_envelope, element_rows):
    """
    Convert the NTHA element_envelope dict to the same list-of-dicts
    format as Graph_Exporter.collect_element_end_force_rows().

    Adds ``_max`` and ``_min`` suffix columns for each force component
    so both envelope bounds are retained for the surrogate training set.

    Parameters
    ----------
    element_envelope : dict  {ele_tag: {'max': [12], 'min': [12]}}
        From run_ntha() results['element_envelope'].
    element_rows : list of dicts
        From Graph_Exporter.collect_element_rows() — supplies ele_tag,
        node_i, node_j, element_type, etc.

    Returns
    -------
    list of dicts, two rows per element (end 'i' and 'j').
    """
    rows = []
    for elem in element_rows:
        tag = elem["ele_tag"]
        if tag not in element_envelope:
            continue
        env = element_envelope[tag]

        for end, offset in _END_OFFSETS.items():
            row = {
                "ele_tag":      tag,
                "node_tag":     elem[f"node_{end}"],
                "end":          end,
                "element_type": elem["element_type"],
            }
            for i, key in enumerate(_FORCE_KEYS):
                row[f"{key}_max"] = env["max"][offset + i]
                row[f"{key}_min"] = env["min"][offset + i]

            rows.append(row)

    return rows
