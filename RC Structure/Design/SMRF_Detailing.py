"""Pure checks for a declared Grade-60, normalweight rectangular RC scope.

ACI 318-19: 18.6.2--18.6.4, 18.7.2, 18.7.4--18.7.5, 25.2.
Units: kip, inch, ksi. Each member dictionary carries dimensions, bar areas,
diameters, counts, clear cover to OUTSIDE of hoops, and hoop spacing. Columns
use n_side_per_face (excluding corners). Geometry spans are center-to-center;
column h is along X, b along Y. Beam bar_stacking supplies declared layers;
records without it retain the legacy single-row geometry.

This does not create a complete bar cage or certify hooks, anchorage, splices,
confinement area, development, or capacity shear. Those remain explicit open
checks. NIST GCR16-917-40 provides methodology (older code edition); the
numeric rules here use the declared ACI318-19 Grade60 scope.
https://www.ocf.berkeley.edu/~chiep/wp-content/uploads/2024/01/CE-123-ACI-318-19.pdf
"""
from __future__ import annotations

import math

from Design.SMRF_Common import make_check, not_evaluated


def _num(data, key, minimum=0.0, strictly_positive=False):
    value = data.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} requires a number, not a boolean.")
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Missing or invalid {key}.") from exc
    if not math.isfinite(value) or value < minimum or (strictly_positive and value <= 0):
        raise ValueError(f"{key} must be finite and {'positive' if strictly_positive else f'>= {minimum}' }.")
    return value


def _count(data, key, minimum=0):
    value = _num(data, key, minimum)
    if not value.is_integer():
        raise ValueError(f"{key} must be an integer.")
    return int(value)


def _dimensions(member):
    return (_num(member, "b_in", strictly_positive=True),
            _num(member, "h_in", strictly_positive=True))


def _bar_geometry(member):
    cover = _num(member, "clear_cover_in")
    hoop = _num(member, "stirrup_db_in", strictly_positive=True)
    db = _num(member, "bar_db_in", strictly_positive=True)
    return cover + hoop + db / 2, db


def select_transverse_geometry(inputs, minimum_spacing_in=3, spacing_step_in=1):
    """Select bounded Grade-60 end-zone geometry, NOT a complete hoop cage.

    The input member/material dictionaries are the same as evaluate_detailing.
    Each member must declare its current/requested hoop_spacing_in; selection
    never increases it. Allowed spacings are minimum_spacing_in + n*step for
    integers n >= 0, rounded DOWN beneath both the request and the code bound.
    No allowed spacing raises ValueError rather than returning a fallback.

    ACI 318-19 18.6.4.1/.4: beam spacing <= min(d/4,6db,6), end zone 2h,
    first hoop <=2 in from the support face. ACI 18.7.5.1/.3: column end zone
    max(b,h,clear_height/6,18), spacing <= min(min(b,h)/4,6db,4). The 4-inch
    column bound is conservative for the permitted so range [4,6]; it does
    not infer hx, supported bars, crossties, confinement area or capacity shear.
    Additional within-span yielding zones and anchorage are not selected here.
    """
    if not isinstance(inputs, dict) or not isinstance(inputs.get("material"), dict):
        raise ValueError("Explicit member and material dictionaries are required.")
    material = inputs["material"]
    if _num(material, "fy_ksi", strictly_positive=True) != 60 or material.get("normalweight") is not True:
        raise ValueError("Transverse geometry supports explicit Grade60 normalweight scope only.")
    policy = {"minimum_spacing_in": minimum_spacing_in, "spacing_step_in": spacing_step_in}
    minimum = _num(policy, "minimum_spacing_in", strictly_positive=True)
    step = _num(policy, "spacing_step_in", strictly_positive=True)
    result = {"method_version": "aci318_19_grade60_transverse_scalar_geometry_v1",
              "stage": "scalar_geometry_only", "full_cage_verified": False,
              "minimum_spacing_in": minimum, "spacing_step_in": step}
    for name in ("beam", "column"):
        member = inputs.get(name)
        if not isinstance(member, dict):
            raise ValueError(f"Explicit {name} inputs are required.")
        b, h = _dimensions(member)
        offset, db = _bar_geometry(member)
        if 2 * offset >= min(b, h):
            raise ValueError(f"{name} bar centroids must lie inside its rectangular section.")
        requested = _num(member, "hoop_spacing_in", strictly_positive=True)
        if name == "beam":
            d = h - offset
            bounds = {"effective_depth_quarter_in": d / 4,
                      "six_longitudinal_diameters_in": 6 * db, "absolute_bound_in": 6.0}
            zone = 2 * h
        else:
            clear_height = _num(member, "clear_height_in", strictly_positive=True)
            bounds = {"minimum_dimension_quarter_in": min(b, h) / 4,
                      "six_longitudinal_diameters_in": 6 * db,
                      "conservative_so_bound_in": 4.0}
            zone = max(b, h, clear_height / 6, 18)
        code_bound = min(bounds.values())
        ceiling = min(requested, code_bound)
        if ceiling < minimum:
            raise ValueError(f"No allowed {name} hoop spacing: minimum {minimum:g} in exceeds "
                             f"the requested/code ceiling {ceiling:g} in.")
        quotient = (ceiling - minimum) / step
        if not math.isfinite(quotient) or quotient > 1e12:
            raise ValueError("Transverse spacing grid is too fine for stable bounded selection.")
        index = math.floor(quotient)
        spacing = minimum + index * step
        # Floating-point lattice arithmetic must never round above the ceiling.
        if spacing > ceiling:
            index -= 1
            spacing = minimum + index * step
        if index < 0 or spacing < minimum or spacing > ceiling:
            raise ValueError(f"No representable allowed {name} hoop spacing.")
        selected = {"hoop_spacing_in": spacing, "requested_hoop_spacing_in": requested,
                    "code_spacing_bound_in": code_bound, "spacing_bounds": bounds,
                    "end_zone_length_in": zone, "zone_origin": "joint_face",
                    "full_cage_verified": False}
        if name == "beam":
            first = _num(member, "first_hoop_distance_in") if "first_hoop_distance_in" in member else 2.0
            selected["first_hoop_distance_in"] = min(first, 2.0)
            selected["effective_depth_in"] = d
        else:
            selected["so_basis"] = "Conservative 4-inch bound only; hx and actual supported-bar layout are not assumed."
        result[name] = selected
    result["limitations"] = [
        "Scalar end-zone spacing/length selection only; no hoop/crosstie geometry or supported-bar inventory.",
        "No confinement area, capacity-design shear, splice/anchorage, joint shear or additional yielding-zone verification.",
    ]
    return result


def evaluate_detailing(inputs):
    inputs = inputs if isinstance(inputs, dict) else {}
    beam, col = inputs.get("beam", {}), inputs.get("column", {})
    geo, material = inputs.get("geometry", {}), inputs.get("material", {})
    checks = []

    def check(name, clause, demand, capacity, comparison="<=", location="", units="in"):
        checks.append(make_check(name, clause, demand, capacity, comparison,
                                 units=units, location=location))

    def unknown(name, clause, error):
        checks.append(not_evaluated(name, clause, str(error)))

    supported = material.get("fy_ksi") == 60 and material.get("normalweight") is True
    if supported:
        check("detailing.material_scope", "Declared Grade60/normalweight implementation scope", 1, 1, "==", units="")
    else:
        unknown("detailing.material_scope", "Declared Grade60/normalweight implementation scope",
                "An explicit Grade60, normalweight basis is required for reinforcement rules in this module.")
    try:
        b, h = _dimensions(beam)
        check("beam.minimum_width", "ACI 318-19 18.6.2.1(b)", b, min(0.3 * h, 10), ">=")
        cb, ch = _dimensions(col)
        check("column.minimum_dimension", "ACI 318-19 18.7.2.1(a)", min(cb, ch), 12, ">=")
        check("column.aspect_ratio", "ACI 318-19 18.7.2.1(b)", min(cb, ch) / max(cb, ch), 0.4, ">=", units="ratio")
        offset, _ = _bar_geometry(beam)
        d = h - offset
        if d <= 0:
            raise ValueError("Beam effective depth must be positive.")
        for axis, column_depth, column_width in (("x", ch, cb), ("y", cb, ch)):
            span = _num(geo, f"span_{axis}_in", strictly_positive=True)
            check("beam.clear_span", "ACI 318-19 18.6.2.1(a)", span - column_depth, 4 * d, ">=", axis)
            # Centered beam: projection measured separately on each side.
            projection = max(0, (b - column_width) / 2)
            check("beam.column_projection", "ACI 318-19 18.6.2.1(c)", projection,
                  min(column_width, 0.75 * column_depth), location=axis)
    except (ValueError, TypeError, AttributeError) as exc:
        unknown("detailing.geometry_complete", "ACI 318-19 18.6.2; 18.7.2", exc)

    for name, member in (("beam", beam), ("column", col)):
        try:
            b, h = _dimensions(member)
            offset, db = _bar_geometry(member)
            area = _num(member, "bar_area_in2", strictly_positive=True)
            top, bottom = _count(member, "n_top", 2), _count(member, "n_bottom", 2)
            aggregate = _num(member, "aggregate_size_in", strictly_positive=True)
            clearance = max(1.0, db, 4 * aggregate / 3) if name == "beam" else max(1.5, 1.5 * db, 4 * aggregate / 3)
            if name == "beam" and "bar_stacking" in member:
                from Design.SMRF_Beam_Spacing import evaluate_layered_beam_spacing
                checks.extend(evaluate_layered_beam_spacing(member, col))
            else:
                for face, count in (("top", top), ("bottom", bottom)):
                    spacing = (b - 2 * offset) / (count - 1) - db
                    check(f"{name}.bar_clear_spacing", "ACI 318-19 25.2.1; 25.2.3", spacing,
                          clearance, ">=", face)
            if name == "column":
                side = _count(member, "n_side_per_face")
                check("column.side_bar_clear_spacing", "ACI 318-19 25.2.3",
                      (h - 2 * offset) / (side + 1) - db, clearance, ">=")
        except (ValueError, TypeError, AttributeError, KeyError, OverflowError) as exc:
            unknown(f"{name}.bar_fit_complete", "ACI 318-19 25.2", exc)
        # Steel ratios do not require an aggregate size; keep independent evidence.
        try:
            b, h = _dimensions(member)
            offset, db = _bar_geometry(member)
            area = _num(member, "bar_area_in2", strictly_positive=True)
            top, bottom = _count(member, "n_top", 2), _count(member, "n_bottom", 2)
            if name == "column":
                side = _count(member, "n_side_per_face")
                rho = (top + bottom + 2 * side) * area / (b * h)
                check("column.minimum_rho", "ACI 318-19 18.7.4.1", rho, 0.01, ">=", units="ratio")
                check("column.maximum_rho", "ACI 318-19 18.7.4.1", rho, 0.06, units="ratio")
            elif supported:
                fc = _num(member, "fc_ksi", strictly_positive=True)
                d = h - offset
                if d <= 0:
                    raise ValueError("Beam effective depth must be positive.")
                amin = max(3 * math.sqrt(1000 * fc) / 60000, 200 / 60000) * b * d
                for face, count in (("top", top), ("bottom", bottom)):
                    check("beam.minimum_steel", "ACI 318-19 18.6.3.1; 9.6.1.2", count * area, amin, ">=", face, "in2")
                    check("beam.maximum_rho", "ACI 318-19 18.6.3.1", count * area / (b * d), 0.025, location=face, units="ratio")
        except (ValueError, TypeError, AttributeError) as exc:
            unknown(f"{name}.longitudinal_steel_complete", "ACI 318-19 18.6.3; 18.7.4", exc)

    try:
        strengths = {key: _num(beam, key, strictly_positive=True) for key in
                     ("mn_positive_left_kipin", "mn_negative_left_kipin",
                      "mn_positive_right_kipin", "mn_negative_right_kipin",
                      "mn_positive_min_along_kipin", "mn_negative_min_along_kipin")}
        for end in ("left", "right"):
            check("beam.reversal_balance", "ACI 318-19 18.6.3.2",
                  strengths[f"mn_positive_{end}_kipin"], 0.5 * strengths[f"mn_negative_{end}_kipin"], ">=", end, "kip-in")
        maximum = max(strengths[k] for k in strengths if "min_along" not in k)
        for sign in ("positive", "negative"):
            check("beam.span_strength", "ACI 318-19 18.6.3.2", strengths[f"mn_{sign}_min_along_kipin"],
                  0.25 * maximum, ">=", sign, "kip-in")
        for face in ("top", "bottom"):
            count = _count(beam, f"continuous_{face}_bars")
            check("beam.continuous_bars", "ACI 318-19 18.6.3.1", count, 2, ">=", face, "bars")
    except (ValueError, TypeError, AttributeError) as exc:
        unknown("beam.continuity_and_strength_balance", "ACI 318-19 18.6.3", exc)

    for name, member in (("beam", beam), ("column", col)):
        # Spacing, zone length, first-hoop location and bar-support geometry
        # are independent evidence. Missing one must not hide failure of another.
        try:
            if not supported:
                raise ValueError("Grade60 transverse spacing rules require the declared material scope.")
            b, h = _dimensions(member)
            offset, db = _bar_geometry(member)
            spacing = _num(member, "hoop_spacing_in", strictly_positive=True)
            if name == "beam":
                if h - offset <= 0:
                    raise ValueError("Beam effective depth must be positive.")
                limit = min((h - offset) / 4, 6 * db, 6)
                check("beam.end_hoop_spacing", "ACI 318-19 18.6.4.4", spacing, limit)
            else:
                # Bounds independent of hx still reveal known violations.
                basic_limit = min(min(b, h) / 4, 6 * db)
                check("column.end_hoop_spacing_dimension_bar", "ACI 318-19 18.7.5.3(a),(b)",
                      spacing, basic_limit)
                try:
                    hx = _num(member, "hx_in", strictly_positive=True)
                except (ValueError, TypeError, AttributeError):
                    conservative, absolute = min(basic_limit, 4), min(basic_limit, 6)
                    if spacing <= conservative:
                        check("column.end_hoop_spacing", "ACI 318-19 18.7.5.3; conservative so=4 bound only",
                              spacing, conservative)
                    elif spacing > absolute:
                        check("column.end_hoop_spacing", "ACI 318-19 18.7.5.3; maximum possible so=6",
                              spacing, absolute)
                    else:
                        unknown("column.end_hoop_spacing", "ACI 318-19 18.7.5.3",
                                "Spacing exceeds the conservative so=4 bound but may be permitted by an actual verified hx; hx is missing/invalid.")
                else:
                    so = min(6, max(4, 4 + (14 - hx) / 3))
                    check("column.end_hoop_spacing", "ACI 318-19 18.7.5.3", spacing, min(basic_limit, so))
        except (ValueError, TypeError, AttributeError) as exc:
            unknown(f"{name}.end_hoop_spacing", "ACI 318-19 18.6.4.4; 18.7.5.3", exc)
        try:
            b, h = _dimensions(member)
            zone = _num(member, "end_zone_length_in", strictly_positive=True)
            if name == "beam":
                check("beam.end_zone_length", "ACI 318-19 18.6.4.1", zone, 2 * h, ">=")
            else:
                clear_h = _num(member, "clear_height_in", strictly_positive=True)
                check("column.end_zone_length", "ACI 318-19 18.7.5.1", zone, max(b, h, clear_h / 6, 18), ">=")
        except (ValueError, TypeError, AttributeError) as exc:
            unknown(f"{name}.end_zone_length", "ACI 318-19 18.6.4.1; 18.7.5.1", exc)
        if name == "beam":
            try:
                first = _num(member, "first_hoop_distance_in")
                check("beam.first_hoop", "ACI 318-19 18.6.4.4", first, 2)
            except (ValueError, TypeError, AttributeError) as exc:
                unknown("beam.first_hoop", "ACI 318-19 18.6.4.4", exc)
        else:
            try:
                hx = _num(member, "hx_in", strictly_positive=True)
                check("column.supported_bar_distance_basic", "ACI 318-19 18.7.5.2(e); stricter axial/fc applicability remains open",
                      hx, 14)
            except (ValueError, TypeError, AttributeError) as exc:
                unknown("column.supported_bar_distance_basic", "ACI 318-19 18.7.5.2(e)", exc)

    for name, clause, reason in (
        ("column.confinement_area_and_support", "ACI 318-19 18.7.5.2--18.7.5.4", "Actual hoop/crosstie layout, core dimensions, axial envelope and lateral bar support must be designed and verified."),
        ("beam.hoop_layout_and_axial_applicability", "ACI 318-19 18.6.4", "Hoop geometry, hooks, bar support and beam axial-load applicability are not established by scalar spacing."),
        ("detailing.anchorage_splices_and_cover", "ACI 318-19 18.6--18.8; 20.5; 25.4--25.5", "Exposure-specific cover, development, terminating bars, splice and cutoff locations require explicit design evidence."),
    ):
        unknown(name, clause, reason)
    return checks
