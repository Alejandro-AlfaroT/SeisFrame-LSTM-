"""
Design/Section_Design.py
========================

Discrete section-dimension and concrete-strength ladders for the iterative
design loop, plus the heuristics that pick a rung from a demand-capacity
ratio.

Redesign.py resizes longitudinal bars and stirrup spacing. That is only the
last ~20% of the design space: once a section is at its maximum practical
steel ratio, the only way to reduce DCR further is a bigger section or
stronger concrete. Without this module the loop saturates and every geometry
converges on whatever section Structure_Parameters happened to start with.

Ladder ordering
---------------
Rungs are ordered by increasing capacity. Concrete strength is stepped before
dimensions at a given size because it is the cheaper way to buy capacity, and
because deeper members attract more seismic demand through added mass and
stiffness. Only when the strength options at a size are exhausted does the
section grow.

Unit system: kip, inch, ksi.
"""

from __future__ import annotations

import math


# ACI 318-19 18.7.2.1: special moment frame columns, least dimension >= 12 in
# and least/perpendicular dimension ratio >= 0.4. Square columns satisfy the
# ratio automatically and keep biaxial capacity symmetric. The ladder was
# capped at 36 in until 2026-09-27; the user raised the cap to 42 in (with a
# 40-in rung) after the joint-assembly rules showed that at 36 in a cage
# strong enough for the strong-column check leaves the beam bars only three
# lanes per layer on the faces the y beams meet. Steps (user decision
# 2026-10-02, V2): 1 in from 14 to 24, then 2 in from 24 to 42.
COLUMN_SIZES_IN = tuple(float(size) for size in (*range(14, 25), *range(26, 43, 2)))

# ACI 318-19 18.6.2.1(b) uses min(0.3h, 10 in). The ladder's
# max(0.3h, 10 in) is a stricter PRACTICAL sizing preference, not that rule.
BEAM_DEPTHS_IN = (16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 36.0)

# Concrete grades by element (user decision 2026-10-03): the range moves from 4-8 ksi to 5-10 ksi, with
# separate ladders. Columns reach 10 ksi, where joint shear strength (sqrt f'c) and the 25.4.1.4 development
# credit both still count in full; beams and slab stop at 8 ksi (the slab is cast with the beams). Column
# concrete stronger than 1.4 times the floor's is carried through the joint by placing column-strength
# concrete in the floor at the column (ACI 318-19 15.5(a)), which is also what the joint strength assumes.
COLUMN_CONCRETE_STRENGTHS_KSI = (5.0, 6.0, 8.0, 10.0)
BEAM_CONCRETE_STRENGTHS_KSI = (5.0, 6.0, 8.0)
CONCRETE_STRENGTHS_KSI = tuple(sorted(set(COLUMN_CONCRETE_STRENGTHS_KSI) | set(BEAM_CONCRETE_STRENGTHS_KSI)))
# Large geometry is the last resort (user decision 2026-10-03; the tallest frames are nine stories): a column
# past this size is selected only after the top grade failed at or below it, and is reported as an exception.
COLUMN_SIZE_LAST_RESORT_IN = 32.0

COLUMN_MIN_DIMENSION_IN = 12.0
BEAM_MIN_WIDTH_IN = 10.0
BEAM_MIN_WIDTH_RATIO = 0.30
# Each depth is offered at its paired width and wider variants in 4 in
# steps: the capacity-design shear section (22.5.1.2 with the 18.6.5.1 Ve),
# short clear spans (18.6.2.1(a) ln >= 4d caps the depth) and, since
# 2026-09-27, the threading of the beam bars between the column bars in one
# layer (SMRF_Capacity_Design.design_bar_threading) answer to width, not
# depth. The fifth variant supplies 32/34-in webs at the deeper rungs;
# the former four-variant cap stopped at 30 in while columns could reach
# 42 in, excluding cages that need a wider beam for clearance/confinement.
# 18.6.2.1(c) (projection beyond the column) is checked per rung against
# the columns in SMRF_Detailing.
BEAM_WIDTH_STEP_IN = 4.0
BEAM_WIDTH_VARIANTS = 5
# A stated proportioning preference (not a code rule; 18.6.2.1(b) allows
# up to 3h): no offered variant is wider than it is deep, so the ladder does
# not fill with band beams that outrank deeper rungs on the capacity proxy.
BEAM_MAX_WIDTH_DEPTH_RATIO = 1.0

# Proportioning limits that keep generated frames buildable.
#
# Both span/depth limits below are preliminary-sizing preferences. Actual
# clear-span >= 4d is a separate check using the adjoining columns.
BEAM_MIN_SPAN_DEPTH_RATIO = 5.0
BEAM_MAX_SPAN_DEPTH_RATIO = 20.0   # h >= span / 20  (deflection control)
BEAM_MAX_DEPTH_FRACTION_OF_STORY = 0.40


def beam_width_for_depth(depth_in):
    """Beam width paired with a depth: about h/2, snapped to 2 in, ACI-legal."""
    raw = max(depth_in / 2.0, BEAM_MIN_WIDTH_RATIO * depth_in, BEAM_MIN_WIDTH_IN)
    snapped = 2.0 * math.ceil(raw / 2.0)
    return max(snapped, BEAM_MIN_WIDTH_IN)


def beam_widths_for_depth(depth_in):
    """The widths offered at a depth: the paired width and the 4-in wider variants, no wider than deep."""
    base = beam_width_for_depth(depth_in)
    widths = [base + k * BEAM_WIDTH_STEP_IN for k in range(BEAM_WIDTH_VARIANTS)]
    return [w for w in widths if w <= max(base, BEAM_MAX_WIDTH_DEPTH_RATIO * depth_in)]


def column_ladder():
    """Ordered (b, h, fc) rungs for columns, increasing capacity."""
    return [
        (size, size, fc)
        for size in COLUMN_SIZES_IN
        for fc in COLUMN_CONCRETE_STRENGTHS_KSI
    ]


def beam_ladder(span_in=None, story_height_in=None):
    """(b, h, fc) rungs for beams, filtered by span proportioning.

    Ordered by the flexural capacity proxy b h^2 sqrt(f'c), so a jump lands
    on the lightest rung that should carry the demand and every step up the
    ladder adds capacity; depth, width and f'c are mixed in that order.
    """
    rungs = []
    for depth in BEAM_DEPTHS_IN:
        if span_in is not None:
            if depth > span_in / BEAM_MIN_SPAN_DEPTH_RATIO:
                continue
            if depth < span_in / BEAM_MAX_SPAN_DEPTH_RATIO:
                continue
        if story_height_in is not None:
            if depth > BEAM_MAX_DEPTH_FRACTION_OF_STORY * story_height_in:
                continue
        for width in beam_widths_for_depth(depth):
            for fc in BEAM_CONCRETE_STRENGTHS_KSI:
                rungs.append((width, depth, fc))

    if not rungs:
        raise ValueError("No beam rung meets the declared span/story sizing limits; "
                         "revise the geometry or explicitly extend the ladder.")
    return sorted(rungs, key=lambda rung: (flexural_capacity_proxy(rung), rung[1], rung[0], rung[2]))


def nearest_rung_index(ladder, b_in, h_in, fc_ksi):
    """Index of the rung closest to a given section, for resuming a search."""
    def distance(rung):
        rb, rh, rfc = rung
        return (rb - b_in) ** 2 + (rh - h_in) ** 2 + 4.0 * (rfc - fc_ksi) ** 2

    return min(range(len(ladder)), key=lambda i: distance(ladder[i]))


def flexural_capacity_proxy(rung):
    """Relative flexural capacity of a rung: b*h^2*sqrt(fc).

    Used only to order and to size a jump. The real capacity comes from the
    ACI checks in RC_Design_Check.py; this just avoids walking the ladder one
    rung at a time when the demand is far from the target.
    """
    b, h, fc = rung
    return b * h * h * math.sqrt(fc)


def suggest_rung_index(ladder, current_index, governing_dcr, target_dcr):
    """Jump to the rung whose capacity proxy should land DCR near the target.

    Returns the current index when the demand-capacity ratio is already at the
    target, so a converged design does not oscillate.
    """
    if not ladder:
        raise ValueError("Section ladder is empty.")
    current_index = max(0, min(current_index, len(ladder) - 1))
    if governing_dcr <= 0.0 or target_dcr <= 0.0:
        return current_index

    required = flexural_capacity_proxy(ladder[current_index]) * (governing_dcr / target_dcr)
    feasible = [i for i, rung in enumerate(ladder) if flexural_capacity_proxy(rung) >= required]
    if not feasible:
        return len(ladder) - 1
    return min(feasible)


# Beam growth rule (user decision 2026-10-02). A search heuristic, not a code limit: the strength jump
# lands on the lightest rung that carries the governing factored moment with this tension steel ratio
# (ACI 318-19 18.6.3.1 allows 0.025; the steel pass still selects the bars). The effective depth is
# estimated as h less this offset (cover, hoop, stacked orthogonal cage, half a bar).
BEAM_GROWTH_STEEL_RATIO = 0.015
BEAM_GROWTH_DEPTH_OFFSET_IN = 3.5


def beam_rung_carries_moment(rung, mu_kip_in, fy_ksi, steel_ratio=BEAM_GROWTH_STEEL_RATIO, phi=0.9):
    """Whether a singly reinforced b x d section at ``steel_ratio`` reaches phi Mn >= Mu (rectangular stress block)."""
    b, h, fc = rung
    d = h - BEAM_GROWTH_DEPTH_OFFSET_IN
    if d <= 0:
        return False
    area = steel_ratio * b * d
    a = area * fy_ksi / (0.85 * fc * b)
    return phi * area * fy_ksi * (d - a / 2.0) >= mu_kip_in


def suggest_beam_rung_for_moment(ladder, current_index, mu_kip_in, fy_ksi, steel_ratio=BEAM_GROWTH_STEEL_RATIO):
    """First rung at or above ``current_index`` that carries ``mu_kip_in`` at ``steel_ratio``; the top rung when none does."""
    if not ladder:
        raise ValueError("Section ladder is empty.")
    current_index = max(0, min(current_index, len(ladder) - 1))
    for index in range(current_index, len(ladder)):
        if beam_rung_carries_moment(ladder[index], mu_kip_in, fy_ksi, steel_ratio):
            return index
    return len(ladder) - 1


def rung_summary(rung, member_type):
    b, h, fc = rung
    return {
        "member_type": member_type,
        "b_in": b,
        "h_in": h,
        "fc_ksi": fc,
        "capacity_proxy": flexural_capacity_proxy(rung),
    }


def validate_rung(rung, member_type, span_in=None, story_height_in=None,
                  column_depth_in=None, effective_depth_in=None):
    """Raise if a rung violates the ACI dimensional limits it claims to honor."""
    b, h, fc = rung
    if not all(math.isfinite(v) and v > 0 for v in (b, h, fc)):
        raise ValueError("Section dimensions and concrete strength must be finite and positive.")
    if member_type not in ("column", "beam"):
        raise ValueError("member_type must be column or beam.")
    if member_type == "column":
        if min(b, h) < COLUMN_MIN_DIMENSION_IN:
            raise ValueError(f"Column least dimension {min(b, h)} in < 12 in (ACI 18.7.2.1).")
        if min(b, h) / max(b, h) < 0.4:
            raise ValueError("Column dimension ratio < 0.4 (ACI 18.7.2.1).")
    else:
        if b < min(BEAM_MIN_WIDTH_RATIO * h, BEAM_MIN_WIDTH_IN):
            raise ValueError(f"Beam width {b} in violates ACI 18.6.2.1 for depth {h} in.")
        if story_height_in is not None and h > BEAM_MAX_DEPTH_FRACTION_OF_STORY * story_height_in:
            raise ValueError(f"Beam depth {h} in exceeds {BEAM_MAX_DEPTH_FRACTION_OF_STORY} of story height.")
        if span_in is not None and (not math.isfinite(span_in) or span_in <= 0):
            raise ValueError("Beam span must be finite and positive.")
        if column_depth_in is not None:
            if span_in is None or effective_depth_in is None:
                raise ValueError("Clear-span checking requires span and effective depth.")
            if not math.isfinite(column_depth_in) or column_depth_in <= 0:
                raise ValueError("Supporting column depth must be finite and positive.")
            if not math.isfinite(effective_depth_in) or not 0 < effective_depth_in < h:
                raise ValueError("Effective depth must lie within the beam section.")
            clear_span = span_in - column_depth_in  # equal columns at both ends
            if clear_span < 4.0 * effective_depth_in:
                raise ValueError("Beam clear span < 4d (ACI 318-19 18.6.2.1(a)).")
    if fc <= 0.0:
        raise ValueError("Concrete strength must be positive.")
    return True
