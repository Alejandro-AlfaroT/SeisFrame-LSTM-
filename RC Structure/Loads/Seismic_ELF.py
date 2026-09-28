"""
Loads/Seismic_ELF.py
====================

ASCE 7-22 Chapter 12 equivalent lateral force (ELF) procedure.

This is the seismic demand that drives the iterative design loop. Before this
module existed the redesign was gravity-only, which is why every generated
structure converged on the same section: gravity demand barely varies with
building height, so a 2-story and an 8-story frame were designed to nearly
the same forces.

Seismic weight
--------------
W is taken from the same accounted weight that Model/Build_Model.py turns into nodal
mass (sp.total_floor_seismic_weight()). Legacy mode retains floor D+100%L and
omits member mass. Slab-aware mode includes computed slab+superimposed dead,
the configured seismic live-load fraction, and consistently lumped member
weight. Occupancy/partition/roof assumptions still need project-specific review.

Unit system: kip, inch, second.
"""

import openseespy.opensees as ops

import Structure_Parameters as sp
from Model.Build_Model import floor_master_node


ELF_SERIES_TAG_X = 21
ELF_SERIES_TAG_Y = 22
ELF_PATTERN_TAG_X = 21
ELF_PATTERN_TAG_Y = 22


def seismic_weight_per_floor():
    """Effective seismic weight at each elevated floor, kips."""
    return [sp.total_floor_seismic_weight() for _ in range(sp.NUM_FLOOR)]


def floor_heights_ft():
    """Height of each elevated floor above the base, feet."""
    return [(k * sp.STORY_H) / 12.0 for k in range(1, sp.NUM_FLOOR + 1)]


def design_period_sec(model_period_sec=None):
    """ASCE 7-22 12.8.2: T is capped at Cu*Ta when a model period is used."""
    ta = sp.asce_approx_fundamental_period_sec()
    if model_period_sec is None or model_period_sec <= 0.0:
        return ta, ta, False
    capped = min(float(model_period_sec), sp.ASCE_CU * ta)
    return capped, ta, capped < float(model_period_sec)


def seismic_response_coefficient(period_sec):
    """ASCE 7-22 12.8.1.1 Cs, with the 12.8-3/12.8-5/12.8-6 bounds applied."""
    r_over_i = sp.ASCE_R / sp.ASCE_IE
    cs = sp.ASCE_SDS / r_over_i

    if period_sec <= sp.ASCE_TL:
        cs_max = sp.ASCE_SD1 / (period_sec * r_over_i)
    else:
        cs_max = sp.ASCE_SD1 * sp.ASCE_TL / (period_sec**2 * r_over_i)

    cs_min = max(0.044 * sp.ASCE_SDS * sp.ASCE_IE, 0.01)
    if sp.ASCE_S1 >= 0.6:
        cs_min = max(cs_min, 0.5 * sp.ASCE_S1 / r_over_i)

    governing = "Cs_basic"
    if cs > cs_max:
        cs, governing = cs_max, "Cs_max (12.8-3)"
    if cs < cs_min:
        cs, governing = cs_min, "Cs_min (12.8-5/6)"

    return {
        "cs": cs,
        "cs_basic": sp.ASCE_SDS / r_over_i,
        "cs_max": cs_max,
        "cs_min": cs_min,
        "governing_bound": governing,
    }


def vertical_distribution_exponent(period_sec):
    """ASCE 7-22 12.8.3: k = 1 at T <= 0.5 s, 2 at T >= 2.5 s, linear between."""
    if period_sec <= 0.5:
        return 1.0
    if period_sec >= 2.5:
        return 2.0
    return 1.0 + (period_sec - 0.5) / 2.0


def elf_story_forces(model_period_sec=None):
    """Full ELF result: base shear, per-floor forces, and the governing terms."""
    period, ta, was_capped = design_period_sec(model_period_sec)
    coefficients = seismic_response_coefficient(period)
    weights = seismic_weight_per_floor()
    heights = floor_heights_ft()
    total_weight = sum(weights)
    base_shear = coefficients["cs"] * total_weight

    k = vertical_distribution_exponent(period)
    products = [w * h**k for w, h in zip(weights, heights)]
    total_product = sum(products)
    if total_product <= 0.0:
        raise ValueError("Degenerate ELF vertical distribution; check geometry.")
    forces = [base_shear * p / total_product for p in products]

    return {
        "design_period_sec": period,
        "asce_ta_sec": ta,
        "model_period_sec": model_period_sec,
        "period_capped_at_cu_ta": was_capped,
        "seismic_weight_kip": total_weight,
        "base_shear_kip": base_shear,
        "vertical_distribution_k": k,
        "story_forces_kip": forces,
        "story_weights_kip": weights,
        "story_heights_ft": heights,
        **coefficients,
    }


def accidental_torsion_moment(direction, story_force_kip, ratio=0.05, amplification=1.0):
    """ASCE 7-22 12.8.4.2/12.8.4.3: Mta = Ax * (ratio * B_perp) * Fx at one level.

    B_perp is the plan dimension perpendicular to the applied force.
    """
    perpendicular = sp.NUM_BAY_Y * sp.BAY_Y if direction == "x" else sp.NUM_BAY_X * sp.BAY_X
    return amplification * ratio * perpendicular * story_force_kip


def apply_elf_loads(direction, model_period_sec=None, load_factor=1.0, elf=None,
                    accidental_torsion_ratio=0.0, torsion_amplification=1.0, torsion_sign=1.0):
    """Apply the ELF story forces at the rigid-diaphragm master nodes.

    direction : "x" or "y"
    load_factor : seismic load-effect factor for the design combination
                  (1.0 for the ASCE 7 2.3.6 combinations, which already
                  treat E at strength level).
    accidental_torsion_ratio : 0.05 applies the 12.8.4.2 accidental torsion
                  as a moment about the diaphragm master's vertical axis at
                  every level, amplified by Ax (12.8.4.3) and signed by
                  torsion_sign. The building is doubly symmetric with one
                  section family per member type, so the member envelope
                  over all frames is the same for either torsion sign; one
                  sign per combination is applied and recorded.
    """
    direction = direction.lower()
    if direction not in {"x", "y"}:
        raise ValueError(f"direction must be 'x' or 'y'; received {direction!r}.")

    elf = elf or elf_story_forces(model_period_sec)
    series_tag = ELF_SERIES_TAG_X if direction == "x" else ELF_SERIES_TAG_Y
    pattern_tag = ELF_PATTERN_TAG_X if direction == "x" else ELF_PATTERN_TAG_Y

    ops.timeSeries("Linear", series_tag)
    ops.pattern("Plain", pattern_tag, series_tag)
    torsion = []
    for index, force in enumerate(elf["story_forces_kip"], start=1):
        master = floor_master_node(index)
        scaled = load_factor * force
        moment = 0.0
        if accidental_torsion_ratio:
            moment = torsion_sign * accidental_torsion_moment(direction, scaled, accidental_torsion_ratio,
                                                              torsion_amplification)
        torsion.append(moment)
        if direction == "x":
            ops.load(master, scaled, 0.0, 0.0, 0.0, 0.0, moment)
        else:
            ops.load(master, 0.0, scaled, 0.0, 0.0, 0.0, moment)

    return {**elf, "accidental_torsion_kip_in": torsion, "accidental_torsion_ratio": accidental_torsion_ratio,
            "torsion_amplification_ax": torsion_amplification, "torsion_sign": torsion_sign}


def apply_lateral_loads():
    """Uniform test pattern: sp.FX_FLOOR in X at every floor master (Main.py's legacy lateral case).

    Not the ELF demand; kept beside it so every lateral load pattern lives in
    one module (moved here from Loads/Lateral_Loads.py on 2026-09-25).
    """
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)

    for k in range(1, sp.NUM_FLOOR + 1):
        master = floor_master_node(k)
        ops.load(master, sp.FX_FLOOR, 0.0, 0.0, 0.0, 0.0, 0.0)
