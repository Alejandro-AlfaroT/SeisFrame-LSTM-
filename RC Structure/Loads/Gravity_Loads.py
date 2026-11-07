"""
Gravity_Loads.py
Apply floor gravity loads as tributary-area-based uniform beam loads.

Each beam receives a downward uniform load (kip/in) equal to the floor area
load intensity (kip/in²) multiplied by its tributary width:
  - Edge beams   → half the adjacent bay width
  - Interior beams → full bay width (half from each side)

"""

import openseespy.opensees as ops

import Structure_Parameters as sp
from Model import Member_Groups as mg
from Model.Build_Model import node_tag


def _column_count():
    return sp.NUM_FLOOR * (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1)


def _apply_nodal_gravity_loads(load_factor=1.0):
    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                ops.load(
                    node_tag(k, i, j),
                    0.0,
                    0.0,
                    -load_factor * sp.node_gravity_load_kip(i, j, floor=k),
                    0.0,
                    0.0,
                    0.0,
                )


def _apply_beam_uniform_gravity_loads(load_factor=1.0):
    """
    Apply tributary-area-scaled uniform loads to all floor beams.

    The floor load is split equally between X-beams and Y-beams so the total
    applied load matches the nodal approach (180 kips/floor for the default
    parameters).  Each beam receives half the full-tributary intensity:

        wz = 0.5 × floor_load_ksi × tributary_width_perp

    Without the 0.5 factor, both beam directions would each apply the full
    floor load independently, doubling the total to 360 kips/floor.

    X-beams span in the X-direction; tributary width is in the Y-direction
    and depends on the beam's j-position (edge vs. interior).

    Y-beams span in the Y-direction; tributary width is in the X-direction
    and depends on the beam's i-position (edge vs. interior).
    """
    ele_tag = _column_count() + 1

    # ── X beams (half-tributary in the Y-direction) ───────────────────────────
    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            wz = load_factor * sp.beam_gravity_wz_kip_per_in("x", j, floor=k) / 2.0
            for i in range(sp.NUM_BAY_X):
                ops.eleLoad("-ele", ele_tag, "-type", "-beamUniform", 0.0, wz, 0.0)
                ele_tag += 1

    # ── Y beams (half-tributary in the X-direction) ───────────────────────────
    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y):
            for i in range(sp.NUM_BAY_X + 1):
                wz = load_factor * sp.beam_gravity_wz_kip_per_in("y", i, floor=k) / 2.0
                ops.eleLoad("-ele", ele_tag, "-type", "-beamUniform", 0.0, wz, 0.0)
                ele_tag += 1
    if sp.SLAB_THICKNESS_IN is not None:
        from Model import Floor_Footprint as fp
        for k in range(1,sp.NUM_FLOOR+1):
            physical = fp.physical_floor(k)
            cx, cy = physical["centroid_in"]
            weight = load_factor * sp.total_floor_gravity_load(k)
            mx = -.5*weight*(cy-sp.NUM_BAY_Y*sp.BAY_Y/2)
            my = .5*weight*(cx-sp.NUM_BAY_X*sp.BAY_X/2)
            if mx or my:
                ops.load(node_tag(k,0,0),0.,0.,0.,mx,my,0.)


def _beam_element_tag(k, axis, line_index, span_index):
    """Element tag of floor k's beam, matching the builders' creation order."""
    n_col = _column_count()
    if axis == "x":
        return n_col + ((k - 1) * (sp.NUM_BAY_Y + 1) + line_index) * sp.NUM_BAY_X + span_index + 1
    n_beam_x = sp.NUM_FLOOR * sp.NUM_BAY_X * (sp.NUM_BAY_Y + 1)
    return n_col + n_beam_x + ((k - 1) * sp.NUM_BAY_Y + span_index) * (sp.NUM_BAY_X + 1) + line_index + 1


def _apply_perimeter_strip(k, transfer, dead_factor, live_factor, live_pattern):
    """Put floor k's perimeter strip on the frame (Loads.Perimeter_Strip, 2026-11-05).

    The transfer is the centreline plate's. The slab layer over the outer half of each perimeter beam, with
    its superimposed dead and live load, is a uniform line load on that beam; the couple of its eccentricity
    about the beam axis goes half to each end joint, and each corner square to its corner joint.
    """
    from Loads import Perimeter_Strip as strip
    loads = strip.floor_loads(k, transfer, dead_factor, live_factor, live_pattern)
    for (axis, line_index, span_index), w in loads["beams"].items():
        ops.eleLoad("-ele", _beam_element_tag(k, axis, line_index, span_index), "-type", "-beamUniform", 0.0, -w, 0.0)
    for (i, j), (load, mx, my) in loads["joints"].items():
        if load or mx or my:
            ops.load(node_tag(k, i, j), 0.0, 0.0, -load, mx, my, 0.0)


def _apply_slab_transfer_loads(dead_factor=1.0, live_factor=1.0, live_pattern="all", perimeter_strip=True):
    """
    Apply the slab-to-frame transfer on every floor.

    Each beam receives the discrete downward node loads the flexible-beam
    floor model delivered to it (local -z point loads at their span
    fractions), and each column node the footprint share that bypasses the
    beams. Dead and live unit cases are combined linearly with the given
    factors; ``live_pattern`` selects the all-panel live case ("all") or a
    saved ACI 6.4.2 arrangement (``live_pattern_<id>``). Member self-weight
    is applied separately, as in every mode.

    Uniform designs carry one transfer, applied to every floor. A grouped
    design carries one transfer per mechanically distinct floor (its own beam
    lines and supports), and every floor receives its own; a uniform transfer
    is refused there, and there is no tributary fallback.

    A transfer is solved for the centreline plate and is validated against that plate's load. The rest of
    the physical floor the weights use, the perimeter strip, is added floor by floor
    (``_apply_perimeter_strip``, 2026-11-05); ``perimeter_strip=False`` leaves it out for a comparison
    against a model of the centreline plate alone.
    """
    from Design.SMRF_Floor_Transfer import validate_floor_transfer
    geometry = {"num_bay_x": sp.NUM_BAY_X, "num_bay_y": sp.NUM_BAY_Y, "bay_x_in": sp.BAY_X, "bay_y_in": sp.BAY_Y}
    from Loads import Perimeter_Strip as strip
    if mg.is_grouped():
        from Design import SMRF_Floor_Sections as floor_sections
        from Design.SMRF_Floor_Transfer import is_by_floor, validate_floor_transfers_by_floor
        if not is_by_floor(sp.FLOOR_TRANSFER):
            raise mg.GroupedStateError(
                "The installed floor transfer is one common floor's; a grouped design needs the transfer of each of its "
                "own floors (Design.SMRF_Floor_Transfer.build_floor_transfers_by_floor). It is not applied.")
        # The plate each transfer was solved for; plate + strip = the footprint the weights use, or this raises.
        dead_kip, live_kip = strip.plate_loads_kip(sp.floor_dead_load_ksf(), sp.FLOOR_LIVE_LOAD_KSF)
        transfers = validate_floor_transfers_by_floor(
            sp.FLOOR_TRANSFER, {**geometry, "num_floor": sp.NUM_FLOOR}, floor_sections.by_floor(), sp.SLAB_THICKNESS_IN,
            dead_kip, live_kip)
        accumulated = {}
        for k in range(1, sp.NUM_FLOOR + 1):
            transfer = transfers[k]
            if id(transfer) not in accumulated:
                accumulated[id(transfer)] = _accumulate_transfer(transfer, dead_factor, live_factor, live_pattern)
            _apply_floor_transfer(k, *accumulated[id(transfer)])
            if perimeter_strip:
                _apply_perimeter_strip(k, transfer, dead_factor, live_factor, live_pattern)
        return
    dead_kip, live_kip = strip.plate_loads_kip(sp.floor_dead_load_ksf(), sp.FLOOR_LIVE_LOAD_KSF)
    transfer = validate_floor_transfer(
        sp.FLOOR_TRANSFER, geometry,
        {"b_beam_in": sp.B_BEAM, "h_beam_in": sp.H_BEAM, "fc_beam_ksi": sp.FC_BEAM_KSI,
         "b_col_in": sp.B_COL, "h_col_in": sp.H_COL},
        sp.SLAB_THICKNESS_IN, dead_kip, live_kip,
    )
    loads = _accumulate_transfer(transfer, dead_factor, live_factor, live_pattern)
    for k in range(1, sp.NUM_FLOOR + 1):
        _apply_floor_transfer(k, *loads)
        if perimeter_strip:
            _apply_perimeter_strip(k, transfer, dead_factor, live_factor, live_pattern)


def _accumulate_transfer(transfer, dead_factor, live_factor, live_pattern):
    """One floor's loads from one transfer: (beam point loads, bending couples, column loads, joint moments)."""
    from Design.SMRF_Floor_Transfer import global_couple
    beams, columns = {}, {}
    bending_couples = {}   # (axis, line, span) -> {x_fraction: local-y couple (kip-in)}
    node_moments = {}      # (i, j) -> [Mx, My] global, per floor
    live_case = "live" if live_pattern in (None, "all") else f"live_pattern_{live_pattern}"
    if live_factor and live_case not in transfer["unit_cases"]:
        raise ValueError(f"Floor transfer has no live case for pattern {live_pattern!r}.")

    def add_moment(i, j, mx, my):
        current = node_moments.setdefault((i, j), [0.0, 0.0])
        current[0] += mx
        current[1] += my

    for name, factor in (("dead", dead_factor), (live_case, live_factor)):
        case = transfer["unit_cases"].get(name)
        if case is None or factor == 0.0:
            continue
        for beam in case["beams"]:
            axis, line, span = beam["axis"], beam["line_index"], beam["span_index"]
            loads = beams.setdefault((axis, line, span), {})
            for x_fraction, load in beam["node_loads"]:
                loads[x_fraction] = loads.get(x_fraction, 0.0) + factor * load
            # A single frame element has no interior nodes. The torsion
            # couple (about the beam axis) goes to the two end joints in
            # proportion to position: the element's twist is linear, so that
            # split is its consistent nodal load. The vertical-plane bending
            # couple is NOT split -- on a fixed-ended element that gives no
            # response at all -- it is applied as a force pair inside the
            # element (see _bending_couple_pair), whose fixed-end actions
            # and internal moment jump the element carries exactly.
            end_i = (span, line) if axis == "x" else (line, span)
            end_j = (span + 1, line) if axis == "x" else (line, span + 1)
            couples = bending_couples.setdefault((axis, line, span), {})
            for x_fraction, local_x, local_y in beam.get("node_couples", []):
                gx, gy = global_couple(axis, factor * local_x, 0.0)
                add_moment(*end_i, (1.0 - x_fraction) * gx, (1.0 - x_fraction) * gy)
                add_moment(*end_j, x_fraction * gx, x_fraction * gy)
                couples[x_fraction] = couples.get(x_fraction, 0.0) + factor * local_y
        for column in case["columns"]:
            key = (column["grid_i"], column["grid_j"])
            columns[key] = columns.get(key, 0.0) + factor * column["direct_load_kip"]
            add_moment(*key, factor * column.get("couple_global_mx_kip_in", 0.0),
                       factor * column.get("couple_global_my_kip_in", 0.0))
    return beams, bending_couples, columns, node_moments


def _apply_floor_transfer(k, beams, bending_couples, columns, node_moments):
    """Put one floor's accumulated transfer loads on floor k of the frame."""
    for (axis, line_index, span_index), loads in beams.items():
        tag = _beam_element_tag(k, axis, line_index, span_index)
        length = sp.BAY_X if axis == "x" else sp.BAY_Y
        for x_fraction in sorted(loads):
            ops.eleLoad("-ele", tag, "-type", "-beamPoint", 0.0, -loads[x_fraction], x_fraction)
        for x_fraction, couple in sorted(bending_couples.get((axis, line_index, span_index), {}).items()):
            for fraction, force in _bending_couple_pair(x_fraction, couple, length):
                ops.eleLoad("-ele", tag, "-type", "-beamPoint", 0.0, force, fraction)
    for (i, j), load in columns.items():
        mx, my = node_moments.get((i, j), (0.0, 0.0))
        if load or mx or my:
            ops.load(node_tag(k, i, j), 0.0, 0.0, -load, mx, my, 0.0)
    for (i, j), (mx, my) in node_moments.items():
        if (i, j) not in columns and (mx or my):
            ops.load(node_tag(k, i, j), 0.0, 0.0, 0.0, mx, my, 0.0)


# Half-separation of the force pair that carries an interior bending couple.
# The pair is statically identical to the couple; its response differs from
# the exact couple only inside the 2-in window, by O((arm/L)^2) at the nodes.
TRANSFER_COUPLE_ARM_IN = 1.0


def _bending_couple_pair(x_fraction, couple_kip_in, length_in, arm_in=TRANSFER_COUPLE_ARM_IN):
    """Force pair equivalent to a local-y couple at x on a beam element.

    Local z is up (vecxz = (0, 0, 1)); a positive local-y couple is
    +P at (x - arm) and -P at (x + arm) with P = M / (2 arm): the two
    ``-beamPoint`` rows. The arm shrinks near an end so both forces stay
    strictly inside the element. Verified against an explicit interior node
    carrying the couple (tests/test_smrf_transfer_mechanics.py).
    """
    if couple_kip_in == 0.0:
        return []
    a = x_fraction * length_in
    arm = min(arm_in, 0.5 * a, 0.5 * (length_in - a))
    if arm <= 0.0:
        raise ValueError("Interior bending couples must lie strictly inside the beam element.")
    force = couple_kip_in / (2.0 * arm)
    return [((a - arm) / length_in, +force), ((a + arm) / length_in, -force)]


def _apply_element_self_weight(load_factor=1.0):
    """
    Apply distributed self-weight to all physical column and beam elements.

    Column self-weight acts along the column axis (global -Z = local -x for a
    vertical element), applied as the axial Wx component of beamUniform.

    Beam self-weight acts downward (global -Z = local -z for a horizontal
    element), applied as the Wz component — the same convention used by
    _apply_beam_uniform_gravity_loads().

    IMK hinge zero-length elements (tags >= IMK_HINGE_ELEMENT_TAG_BASE) are
    skipped; they have zero length so their self-weight is zero and they do
    not support distributed element loads.
    """
    n_col    = sp.NUM_FLOOR * (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1)
    n_beam_x = sp.NUM_FLOOR * sp.NUM_BAY_X       * (sp.NUM_BAY_Y + 1)
    n_beam_y = sp.NUM_FLOOR * (sp.NUM_BAY_X + 1) * sp.NUM_BAY_Y

    if mg.is_grouped() or sp.SLAB_THICKNESS_IN is not None:
        # Every member weighs its own section between its own joint faces (Model.Member_Properties).
        from Model import Member_Properties as mp
        for tag in range(1, n_col + n_beam_x + n_beam_y + 1):
            member = mg.resolve(tag)
            w = load_factor * mp.member_self_weight_kip_per_in(member)
            if member.is_column:
                ops.eleLoad("-ele", tag, "-type", "-beamUniform", 0.0, 0.0, -w)
                couple = [load_factor*v for v in mp.column_weight_eccentricity_kipin(member)]
                if any(couple):
                    ops.load(member.node_j, 0., 0., 0., *couple)
            else:
                ops.eleLoad("-ele", tag, "-type", "-beamUniform", 0.0, -w, 0.0)
                couple = [.5*load_factor*v for v in mp.beam_weight_eccentricity_kipin(member)]
                if any(couple):
                    ops.load(member.node_i, 0., 0., 0., *couple)
                    ops.load(member.node_j, 0., 0., 0., *couple)
        return

    w_col  = load_factor * sp.col_self_weight_kip_per_in()   # kip/in, magnitude
    w_beam_x = load_factor * sp.beam_self_weight_kip_per_in("x")
    w_beam_y = load_factor * sp.beam_self_weight_kip_per_in("y")

    # Columns — self-weight along local -x (axial, downward for vertical column)
    for tag in range(1, n_col + 1):
        ops.eleLoad("-ele", tag, "-type", "-beamUniform", 0.0, 0.0, -w_col)

    # Beams — self-weight along local -z (transverse, downward for horizontal beam)
    for tag in range(n_col + 1, n_col + n_beam_x + 1):
        ops.eleLoad("-ele", tag, "-type", "-beamUniform", 0.0, -w_beam_x, 0.0)
    for tag in range(n_col + n_beam_x + 1, n_col + n_beam_x + n_beam_y + 1):
        ops.eleLoad("-ele", tag, "-type", "-beamUniform", 0.0, -w_beam_y, 0.0)


def _apply_roof_extension_weight(load_factor=1.0):
    """The column extension above each roof joint: its weight as a vertical load at the roof node."""
    from Model import Roof_Extension as roof
    for j in range(sp.NUM_BAY_Y + 1):
        for i in range(sp.NUM_BAY_X + 1):
            weight = load_factor * roof.weight_kip(i, j)
            if weight:
                ops.load(node_tag(sp.NUM_FLOOR, i, j), 0.0, 0.0, -weight, 0.0, 0.0, 0.0)


def apply_gravity_loads(floor_factor=1.0, self_weight_factor=None,
                        dead_factor=None, live_factor=None, live_pattern="all", roof_extension=True,
                        perimeter_strip=True):
    """Apply gravity loads, optionally factored for a design load combination.

    floor_factor
        Multiplier on the combined dead+live floor load. Pass
        sp.seismic_combination_floor_factor() for the ASCE 7 2.3.6 seismic
        combination; leave at 1.0 for the unfactored service-load state used
        by the response-history analyses.
    self_weight_factor
        Multiplier on element self weight, which is pure dead load. Defaults
        to floor_factor when not given.
    dead_factor, live_factor
        Separate floor dead/live factors, honoured by the slab-transfer mode
        (whose unit cases are kept apart). Both default to floor_factor. The
        tributary modes bundle D+L and keep using floor_factor.
    live_pattern
        "all" or a saved ACI 6.4.2 arrangement id (slab-transfer mode only).
    roof_extension
        The weight of the column extensions above the roof joints (Model/Roof_Extension), with the
        self-weight factor. False only for a comparison against a model that has no such extension.
    perimeter_strip
        Slab-transfer mode only: the strip of the physical floor outside the centreline plate the transfer
        was solved for (Loads/Perimeter_Strip), with the floor dead and live factors. False only for a
        comparison against a model of the centreline plate alone. The tributary modes load the physical
        floor themselves.
    """
    if live_pattern not in (None, "all") and sp.effective_gravity_load_model() != "slab_transfer":
        raise ValueError("Live-load patterns require the slab-transfer load model.")
    if self_weight_factor is None:
        self_weight_factor = floor_factor
    if dead_factor is None:
        dead_factor = floor_factor
    if live_factor is None:
        live_factor = floor_factor

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)

    mode = sp.effective_gravity_load_model()
    if mode == "slab_transfer":
        _apply_slab_transfer_loads(dead_factor, live_factor, live_pattern, perimeter_strip)
    elif mode == "nodal":
        _apply_nodal_gravity_loads(floor_factor)
    elif mode == "beam_uniform":
        _apply_beam_uniform_gravity_loads(floor_factor)
    else:
        raise ValueError(f"Unknown GRAVITY_LOAD_MODEL: {mode}")

    _apply_element_self_weight(self_weight_factor)
    if roof_extension:
        _apply_roof_extension_weight(self_weight_factor)
