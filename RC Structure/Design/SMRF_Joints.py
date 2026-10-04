"""Pure, evidence-based checks for RC special moment-frame connections.

Units are kip, inch, ksi. This module does not import OpenSees or mutable
Structure_Parameters. Capacities must be evaluated at joint faces upstream.
It checks the supplied evidence, not the completeness of the structural model.

``evaluate_joints`` input groups (all are lists):
* ``joints``: {id, directions: {x: {positive: state, negative: state}, y: ...}}.
  An SCWB state has ``nominal_strengths: True``, ``column_capacities`` and
  ``beam_capacities``. Each column supplies ``mn_kip_in``,
  ``factored_axial_kip``, and ``axial_envelope_checked: True``: its Mn is the
  lowest appropriate strength over the factored axial envelope for this sway.
  Each beam supplies rectangular ``mn_kip_in``, explicit ``slab_mn_kip_in``
  and ``slab_basis``: no_slab, not_in_tension, developed_effective_width,
  terminated_undeveloped (exterior hogging end whose slab bars are not
  developed: zero slab term) or flange_concrete_undeveloped_bars (exterior
  sagging end: the flange concrete in compression, no slab bars). An
  additive slab term is a *change in section strength*, not As*fy times an
  arbitrary lever arm. Both capacities come from section compatibility.
* ``beam_capacity_shear``: see ``beam_capacity_shear_checks``.
* ``through_bar_anchorage``: see ``through_bar_anchorage_checks``.
* ``joint_shear``: see ``joint_shear_check``. This function does not select
  a Table 18.8.4.3 coefficient or infer confinement from an interior label.

Omitted groups and unknown evidence produce not_evaluated, not a pass.
The exception of ACI 318-19 18.7.3.1 is applied per connection and per sway
(user decision 2026-10-02): where the column is discontinuous above the
connection and its factored axial compression is below Ag f'c / 10 in every
supplied combination, 18.7.3.2 is not required there; the check records the
axial load against the limit and keeps the strength sums it did not require.
Without that evidence on the sway state the rule applies as before; one
column at a roof is never doubled.

Basis: ACI 318-19 18.6.5.1, 18.7.3.2, 18.8.2.3, 18.8.4 and 21.2.4.4.
Publisher's 318-19 changes (joint depth and joint-shear phi=0.85):
https://www.concrete.org/publications/getarticle.aspx?m=icap&pubID=51732620
ACI 318-19 publisher presentation, including SCWB:
https://www.concreteros.org/uploads/2/4/0/2/24020264/aci_318-19_presentation.pdf
Equilibrium and methodology: NIST GCR 16-917-40, Sections 5.3--5.5:
https://nvlpubs.nist.gov/nistpubs/gcr/2016/NIST.GCR.16-917-40.pdf
NIST uses an older code edition; its joint coefficient table is NOT adopted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from Design.SMRF_Common import make_check, not_evaluated


SCWB_CLAUSE = "ACI 318-19 18.7.3.2"
SCWB_EXCEPTION_CLAUSE = "ACI 318-19 18.7.3.1 (exception to 18.7.3.2)"
# Columns satisfy 18.7.3.2 "except at connections where the column is discontinuous above the connection
# and the column factored axial compressive force Pu under load combinations including E is less than
# Ag f'c / 10" (18.7.3.1). False restores the rule at every connection.
SCWB_DISCONTINUOUS_COLUMN_EXCEPTION = True
BEAM_SHEAR_CLAUSE = "ACI 318-19 18.6.5.1"
JOINT_SHEAR_CLAUSE = "ACI 318-19 18.8.4; 21.2.4.4"
SLAB_BASES = {"no_slab", "not_in_tension", "developed_effective_width", "terminated_undeveloped",
              "flange_concrete_undeveloped_bars"}
SLAB_BASES_WITH_STRENGTH = {"developed_effective_width", "flange_concrete_undeveloped_bars"}
MPR_BASIS = "fy_at_least_1.25_phi_1.0"


def _number(value, name, *, minimum=None, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _records(value, name):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty list.")
    return value


def scwb_check(state, *, location="", check_id="scwb"):
    """Check one direction and sway sign from nominal joint-face capacities.

    ``axial_envelope_checked`` must mean the supplied column Mn is the
    minimum over the applicable factored axial envelope, NOT a single
    service-gravity estimate. This module never clamps/extrapolates a P-M
    curve and never silently substitutes the zero-axial moment strength.
    """
    try:
        if not isinstance(state, Mapping):
            raise ValueError("Nominal joint-face capacity evidence is missing.")
        if state.get("nominal_strengths") is not True:
            raise ValueError("Explicit nominal, unfactored moment strengths are required.")
        columns = _records(state.get("column_capacities"), "column_capacities")
        beams = _records(state.get("beam_capacities"), "beam_capacities")
        column_sum, beam_sum = 0.0, 0.0
        axial_loads = []
        for col in columns:
            if not isinstance(col, Mapping) or col.get("axial_envelope_checked") is not True:
                raise ValueError("Every column needs its governing factored axial envelope.")
            axial_loads.append(_number(col.get("factored_axial_kip"), "factored_axial_kip"))
            column_sum += _number(col.get("mn_kip_in"), "column mn_kip_in", minimum=0)
        for beam in beams:
            if not isinstance(beam, Mapping) or beam.get("slab_basis") not in SLAB_BASES:
                raise ValueError("Every beam needs an explicit developed slab contribution or absence basis.")
            slab = _number(beam.get("slab_mn_kip_in"), "slab_mn_kip_in", minimum=0)
            if beam["slab_basis"] not in SLAB_BASES_WITH_STRENGTH and slab != 0:
                raise ValueError("A nonzero slab contribution requires developed_effective_width or "
                                 "flange_concrete_undeveloped_bars.")
            beam_sum += _number(beam.get("mn_kip_in"), "beam mn_kip_in", minimum=0) + slab
        if beam_sum <= 0:
            raise ValueError("Positive total beam flexural strength is required.")
        exception = _discontinuous_column_exception(state, columns)
    except ValueError as exc:
        return not_evaluated(check_id, SCWB_CLAUSE, str(exc), location)
    if exception is not None:
        return make_check(
            check_id, SCWB_EXCEPTION_CLAUSE, exception["pu_max_kip"], exception["limit_kip"],
            units="kip", location=location,
            details={"sum_mnc_kip_in": column_sum, "sum_mnb_kip_in": beam_sum,
                     "ratio_provided": column_sum / beam_sum, "ratio_required": None,
                     "column_factored_axial_kip": axial_loads,
                     "strength_basis": "nominal joint-face strengths; no phi (recorded, not required here)",
                     "roof_exemption_applied": True, "exception": exception},
        )
    return make_check(
        check_id, SCWB_CLAUSE, 1.2 * beam_sum, column_sum,
        units="kip-in", location=location,
        details={"sum_mnc_kip_in": column_sum, "sum_mnb_kip_in": beam_sum,
                 "ratio_provided": column_sum / beam_sum, "ratio_required": 1.2,
                 "column_factored_axial_kip": axial_loads,
                 "strength_basis": "nominal joint-face strengths; no phi",
                 "roof_exemption_applied": False},
    )


def _discontinuous_column_exception(state, columns):
    """The 18.7.3.1 exception at this connection, or None when 18.7.3.2 applies.

    Needs the sway state's ``connection`` evidence (column discontinuous above, gross area, f'c) and the single
    column's largest factored axial compression over every supplied combination (more combinations than the
    clause names, so never less than its Pu). Strictly below Ag f'c / 10.
    """
    connection = state.get("connection")
    if not SCWB_DISCONTINUOUS_COLUMN_EXCEPTION or not isinstance(connection, Mapping):
        return None
    if connection.get("column_discontinuous_above") is not True or len(columns) != 1:
        return None
    if columns[0].get("axial_max_kip") is None:
        return None
    area = _number(connection.get("column_gross_area_in2"), "column_gross_area_in2", positive=True)
    fc = _number(connection.get("column_fc_ksi"), "column_fc_ksi", positive=True)
    pu_max = _number(columns[0].get("axial_max_kip"), "axial_max_kip")
    limit = area * fc / 10.0
    if not pu_max < limit:
        return None
    return {"clause": "ACI 318-19 18.7.3.1", "pu_max_kip": pu_max, "limit_kip": limit,
            "column_gross_area_in2": area, "column_fc_ksi": fc,
            "basis": ("the column is discontinuous above the connection and its largest factored axial compression "
                      "over every supplied combination is below Ag f'c / 10")}


def beam_capacity_shear_envelope(data):
    """Return shear at the clear-span faces for both beam sway mechanisms.

    Required fields: clear_span_in, mpr_left_positive_kip_in,
    mpr_left_negative_kip_in, mpr_right_positive_kip_in,
    mpr_right_negative_kip_in, gravity_reactions_kip=[left,right].
    Mpr values are nonnegative magnitudes. Gravity reactions are signed
    upward-positive face reactions for the factored transverse loads alone
    with zero end moments, not reactions already including flexural actions.
    For positive sway, left moment is negative and right moment positive.
    Each gravity combination must be evaluated separately by the caller.
    """
    if not isinstance(data, Mapping):
        raise ValueError("Beam equilibrium inputs are required.")
    length = _number(data.get("clear_span_in"), "clear_span_in", positive=True)
    moments = {key: _number(data.get(key), key, minimum=0) for key in (
        "mpr_left_positive_kip_in", "mpr_left_negative_kip_in",
        "mpr_right_positive_kip_in", "mpr_right_negative_kip_in",
    )}
    reactions = _records(data.get("gravity_reactions_kip"), "gravity_reactions_kip")
    if len(reactions) != 2:
        raise ValueError("Exactly two signed gravity face reactions are required.")
    left, right = (_number(value, "gravity reaction") for value in reactions)
    positive = (moments["mpr_left_negative_kip_in"] + moments["mpr_right_positive_kip_in"]) / length
    negative = (moments["mpr_left_positive_kip_in"] + moments["mpr_right_negative_kip_in"]) / length
    face_shears = {"positive": [left + positive, right - positive],
                   "negative": [left - negative, right + negative]}
    return {"face_shears_kip": face_shears,
            "left_required_kip": max(abs(v[0]) for v in face_shears.values()),
            "right_required_kip": max(abs(v[1]) for v in face_shears.values()),
            "mechanism_shear_positive_kip": positive,
            "mechanism_shear_negative_kip": negative,
            "clear_span_in": length}


def beam_capacity_shear_checks(data, *, location=""):
    """Compare Mpr equilibrium demands to explicitly supplied design shears.

    Besides equilibrium inputs, require mpr_basis=MPR_BASIS and
    gravity_basis='factored_zero_end_moment_reactions'. Passing comparisons
    require phi_vn_left_kip and phi_vn_right_kip, plus
    shear_capacity_requirements_checked=True, affirming applicable Vc=0,
    transverse reinforcement and maximum shear-strength limits were assessed.
    No design strength is inferred merely from stirrup area and spacing.
    """
    try:
        if not isinstance(data, Mapping) or data.get("mpr_basis") != MPR_BASIS:
            raise ValueError("Probable strength basis must confirm steel stress >=1.25fy and phi=1.")
        if data.get("gravity_basis") != "factored_zero_end_moment_reactions":
            raise ValueError("Factored gravity-only face reactions must be identified explicitly.")
        envelope = beam_capacity_shear_envelope(data)
    except ValueError as exc:
        return [not_evaluated("beam_capacity_shear", BEAM_SHEAR_CLAUSE, str(exc), location)]
    checks = []
    for end in ("left", "right"):
        demand = envelope[f"{end}_required_kip"]
        try:
            if data.get("shear_capacity_requirements_checked") is not True:
                raise ValueError("Shear capacity needs transverse-detailing, Vc applicability and strength-limit checks.")
            capacity = _number(data.get(f"phi_vn_{end}_kip"), f"phi_vn_{end}_kip", minimum=0)
            check = make_check(f"beam_capacity_shear_{end}", BEAM_SHEAR_CLAUSE,
                               demand, capacity, units="kip", location=location,
                               details=envelope)
        except ValueError as exc:
            check = not_evaluated(f"beam_capacity_shear_{end}", BEAM_SHEAR_CLAUSE,
                                  str(exc), location)
            check.update(demand=demand, units="kip")
            check["details"]["equilibrium"] = envelope
        checks.append(check)
    return checks


def rectangular_joint_area(*, column_depth_in, column_width_in,
                           beam_width_in, beam_center_offset_in):
    """Effective Aj for one rectangular, axis-aligned beam/joint footprint.

    Implements bj=min(column_width, beam_width+column_depth,
    beam_width+2*nearest_edge_extension) when the beam is narrower. Offset
    must be supplied even when zero; a narrower beam partly outside the
    column footprint is unsupported. Wider beams must cover the column.
    For unlike beams on opposite faces, compute appropriate footprints
    upstream; this helper does not invent an intersection/average width.
    """
    depth = _number(column_depth_in, "column_depth_in", positive=True)
    width = _number(column_width_in, "column_width_in", positive=True)
    beam = _number(beam_width_in, "beam_width_in", positive=True)
    offset = abs(_number(beam_center_offset_in, "beam_center_offset_in"))
    if beam <= width:
        extension = (width - beam) / 2.0 - offset
        if extension < 0:
            raise ValueError("Beam footprint extends beyond the rectangular column; joint area needs separate evaluation.")
        effective = min(width, beam + depth, beam + 2.0 * extension)
    else:
        if offset > (beam - width) / 2.0:
            raise ValueError("Wide beam does not cover the column footprint; joint area needs separate evaluation.")
        effective = width
    return effective * depth


def through_bar_anchorage_checks(data, *, location=""):
    """Check the 318-19 through-bar joint dimension, not full development.

    Inputs: bars_pass_through=True, joint_depth_in, concrete_type
    ('normalweight' or 'lightweight'), beam_depths_in (all framing beams
    generating shear in the considered direction), bars=[{grade_ksi,
    diameter_in}]. Grade60 and Grade80 are implemented. Exterior terminating
    bars require a separate anchorage evaluation and never pass this check.
    """
    clause = "ACI 318-19 18.8.2.3"
    try:
        if not isinstance(data, Mapping) or data.get("bars_pass_through") is not True:
            raise ValueError("Through-bar applicability is missing; terminating bars need a separate development check.")
        depth = _number(data.get("joint_depth_in"), "joint_depth_in", positive=True)
        concrete = data.get("concrete_type")
        if concrete not in {"normalweight", "lightweight"}:
            raise ValueError("Concrete type must be explicit: normalweight or lightweight.")
        beam_depths = [_number(d, "beam depth", positive=True)
                       for d in _records(data.get("beam_depths_in"), "beam_depths_in")]
        limits, grades = [], []
        for bar in _records(data.get("bars"), "bars"):
            if not isinstance(bar, Mapping):
                raise ValueError("Each through bar needs grade_ksi and diameter_in.")
            grade = _number(bar.get("grade_ksi"), "grade_ksi", positive=True)
            db = _number(bar.get("diameter_in"), "diameter_in", positive=True)
            if grade not in (60, 80):
                raise ValueError("Only Grade60 and Grade80 through bars are implemented.")
            grades.append(grade)
            multiplier = (20.0 / (0.75 if concrete == "lightweight" else 1.0)) if grade == 60 else 26.0
            limits.append(multiplier * db)
        required = max(max(limits), max(beam_depths) / 2.0)
    except ValueError as exc:
        return [not_evaluated("joint_through_bar_depth", clause, str(exc), location)]
    checks = [make_check("joint_through_bar_depth", clause, required, depth,
                         units="in", location=location,
                         details={"bar_depth_limits_in": limits,
                                  "half_max_beam_depth_in": max(beam_depths) / 2.0,
                                  "scope": "joint depth only; not full anchorage/development"})]
    if 80 in grades:
        checks.append(make_check("joint_grade80_normalweight", "ACI 318-19 18.8.2.3.1",
                                 1 if concrete == "lightweight" else 0, 0,
                                 location=location, details={"concrete_type": concrete}))
    return checks


def joint_shear_check(data, *, location=""):
    """Check supplied joint design strength against signed probable actions.

    ``beam_face_forces_kip`` are signed horizontal forces on the chosen
    joint half free body; ``column_shear_kip`` is subtracted in that SAME
    sign convention. Require probable_face_forces_complete=True and
    column_shear_consistent_with_mpr=True. Strength must be provided as
    nominal_vn_kip, capacity_basis='ACI318-19_Table18.8.4.3', and
    capacity_topology_and_confinement_checked=True. Caller is responsible
    for the actual rectangular area, 2019 continuity/confinement category
    and required joint transverse steel. This module only applies phi=.85.
    """
    try:
        if not isinstance(data, Mapping):
            raise ValueError("Signed probable joint face actions and capacity evidence are missing.")
        if data.get("probable_face_forces_complete") is not True:
            raise ValueError("Complete signed probable joint face actions are required.")
        if data.get("column_shear_consistent_with_mpr") is not True:
            raise ValueError("Column shear must be consistent with the same probable-strength mechanism.")
        face_forces = [_number(v, "beam face force") for v in
                       _records(data.get("beam_face_forces_kip"), "beam_face_forces_kip")]
        column_shear = _number(data.get("column_shear_kip"), "column_shear_kip")
        demand = abs(sum(face_forces) - column_shear)
    except ValueError as exc:
        return not_evaluated("joint_shear", JOINT_SHEAR_CLAUSE, str(exc), location)
    try:
        if (data.get("capacity_basis") != "ACI318-19_Table18.8.4.3"
                or data.get("capacity_topology_and_confinement_checked") is not True):
            missing = data.get("unevaluated_evidence")
            raise ValueError("Verified 2019 joint-area, continuity, confinement and transverse-steel capacity is required."
                             + (f" Unevaluated: {', '.join(str(m) for m in missing)}." if isinstance(missing, list) and missing else ""))
        nominal = _number(data.get("nominal_vn_kip"), "nominal_vn_kip", minimum=0)
    except ValueError as exc:
        check = not_evaluated("joint_shear", JOINT_SHEAR_CLAUSE, str(exc), location)
        check.update(demand=demand, units="kip")
        return check
    return make_check("joint_shear", JOINT_SHEAR_CLAUSE, demand, 0.85 * nominal,
                      units="kip", location=location,
                      details={"phi": 0.85, "nominal_vn_kip": nominal,
                               "signed_face_forces_kip": face_forces,
                               "signed_column_shear_kip": column_shear,
                               "capacity_basis": data["capacity_basis"]})


def evaluate_joints(inputs):
    """Evaluate explicit evidence; every omitted group remains unevaluated."""
    inputs = inputs if isinstance(inputs, Mapping) else {}
    checks = []
    joints = inputs.get("joints")
    if not isinstance(joints, (list, tuple)) or not joints:
        checks.append(not_evaluated("scwb", SCWB_CLAUSE, "Joint inventory and nominal capacity evidence are missing."))
    else:
        for index, joint in enumerate(joints):
            joint = joint if isinstance(joint, Mapping) else {}
            location = str(joint.get("id", f"joint_{index}"))
            directions = joint.get("directions", {})
            directions = directions if isinstance(directions, Mapping) else {}
            for axis in ("x", "y"):
                states = directions.get(axis, {})
                states = states if isinstance(states, Mapping) else {}
                for sign in ("positive", "negative"):
                    checks.append(scwb_check(states.get(sign), location=f"{location}/{axis}/{sign}"))
    groups = (("beam_capacity_shear", BEAM_SHEAR_CLAUSE, beam_capacity_shear_checks),
              ("through_bar_anchorage", "ACI 318-19 18.8.2.2--18.8.2.3; 18.8.5", through_bar_anchorage_checks),
              ("joint_shear", JOINT_SHEAR_CLAUSE, joint_shear_check))
    for name, clause, evaluate in groups:
        entries = inputs.get(name)
        if not isinstance(entries, (list, tuple)) or not entries:
            checks.append(not_evaluated(name, clause, "Explicit member/joint evidence is missing."))
            continue
        for index, entry in enumerate(entries):
            location = str(entry.get("id", f"{name}_{index}")) if isinstance(entry, Mapping) else f"{name}_{index}"
            result = evaluate(entry, location=location)
            checks.extend(result if isinstance(result, list) else [result])
    return checks
