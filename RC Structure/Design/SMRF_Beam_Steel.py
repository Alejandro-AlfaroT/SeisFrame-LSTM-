"""Shared web-steel limits for candidate selection and saved-design detailing.

Areas include all bars on the tension face; d is measured from the opposite
compression face to their area centroid. Equal bar sizes within a face are
required. Slab steel is handled separately by the composite strength model.
Units: inches, square inches, ksi. The longitudinal grade is always explicit.
"""
import math

from Design.SMRF_Beam_Slab_Strength import validate_bar_rows


def beam_face_steel_limits(*, b_in, h_in, fc_ksi, fy_ksi, bar_area_in2, count, rows):
    """ACI 318-19 9.6.1.2 / 18.6.3.1 using validated physical bar rows.

    ``rows`` contains per_layer and offsets_in from the tension face, plus an
    optional centroid_in which must agree with those rows. A legacy single-row
    caller must supply its nominal offset as one explicit row.
    """
    import Structure_Parameters as sp

    for name, value in (("b_in", b_in), ("h_in", h_in), ("fc_ksi", fc_ksi),
                        ("fy_ksi", fy_ksi), ("bar_area_in2", bar_area_in2)):
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if isinstance(count, bool) or not math.isfinite(float(count)) or int(count) != count or count < 2:
        raise ValueError("A beam face needs an integer total of at least two bars")
    if fy_ksi not in (60, 80):
        raise ValueError("Beam steel limits require longitudinal Grade 60 or Grade 80")
    layers = validate_bar_rows(rows, count, "beam face bars", h_in)
    centroid = sum(n * offset for n, offset in layers) / count
    d = h_in - centroid
    steel = count * bar_area_in2
    fy_psi = 1000.0 * fy_ksi
    minimum = max(3.0 * math.sqrt(1000.0 * fc_ksi) / fy_psi, 200.0 / fy_psi) * b_in * d
    return {"bar_count": count, "bar_area_in2": bar_area_in2,
            "per_layer": [n for n, _ in layers], "offsets_in": [offset for _, offset in layers],
            "centroid_in": centroid, "d_in": d, "as_in2": steel, "as_min_in2": minimum,
            "rho": steel / (b_in * d), "rho_max": sp.beam_rho_max_code(fy_ksi), "fy_ksi": fy_ksi}
