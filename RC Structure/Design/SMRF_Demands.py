"""Pure, explicitly scoped SMRF load-combination and drift evidence helpers.

Units are kip and inch; no OpenSees imports or model state are used here. These
helpers do not establish ELF eligibility or complete building-code compliance.
ASCE/SEI 7-22 Chapters 2 and 12 are the declared basis. Primary references:
https://www.asce.org/publications-and-news/codes-and-standards/asce-sei-7-22
https://nvlpubs.nist.gov/nistpubs/gcr/2016/NIST.GCR.16-917-40.pdf

The NIST guide explains the approach but predates ASCE 7-22. In particular,
7-22 permits a 0.10 lower bound on theta_max and limits beta >= 1.25/Omega0;
these newer details must not be taken from an older worked example.

Edition discipline (2026-09-18): every rule here names the edition it comes
from. ASCE 7-22 12.6 lists the ELF procedure of 12.8 as permitted for any
structure; the 7-16 Table 12.6-1 height/period/irregularity restriction was
deleted (NCSEA, Design Guide for Structural Irregularities, "Permitted
Analytical Procedures", p. 4; S. K. Ghosh, Significant Changes in ASCE 7-22,
SEAU 2024, slide 116). Torsional irregularity is 7-22 Table 12.3-1 Type 1
from the torsional irregularity ratio of 12.3.2.1.1 (TIR > 1.2, a ratio of
story drifts) or the 75% one-sided strength criterion; the former Type 1b is
Type 1 with TIR > 1.4 (Ghosh slides 101-105). The 7-16 12.3.3.1
prohibition of extreme torsional irregularity in SDC E/F was REMOVED in
7-22 (FEMA P-2192 Vol. 1 sec. 1.4.5; California 2025 Title 24 change record
for 1617.12.6 / 1617A.1.10: "Repealed all language related to the extreme
torsional irregularities. Revised to align with ASCE 7-22"), so the
population's TIR <= 1.4 ceiling is project policy in every SDC. The 12.8.4.3
amplification Ax is a ratio of level displacements, computed per level with
Ax = 1, and is kept separate from the TIR.
"""
from __future__ import annotations

import math
from datetime import date
from collections.abc import Mapping

from Design.SMRF_Common import assertion_provenance_valid, make_check, not_evaluated, summarize_checks
# The risk tables have one home (Structure_Parameters); only the tables are read here, never the module state.
from Structure_Parameters import (SEISMIC_IMPORTANCE_FACTOR_BY_RISK_CATEGORY as IMPORTANCE_FACTOR_BY_RISK_CATEGORY,
                                  ALLOWABLE_STORY_DRIFT_RATIO_BY_RISK_CATEGORY, ASCE_SITE_CLASSES)


CODE_EDITION = "ASCE 7-22"
SUPPORTED_EDITIONS = ("ASCE 7-22", "ASCE 7-16")
# Strength combinations carry rho = 1.3 without claiming either condition of
# 12.3.4.2 (conditions 1 and 2, with (a) to (c) inside condition 1 in the 2022
# text); drift forces use rho = 1.0 (12.3.4.1).
REDUNDANCY_FACTOR_STRENGTH = 1.3
TIR_TYPE_1 = 1.2                  # Table 12.3-1 Type 1 threshold (story-drift ratio, 12.3.2.1.1)
TIR_FORMER_TYPE_1B = 1.4          # 7-16 extreme (1b) threshold; in 7-22 only the 12.3.4.2.1 trigger, no prohibition
TIR_PROJECT_POLICY_LIMIT = 1.4    # research population ceiling in every SDC; not a code rule under ASCE 7-22
ONE_SIDE_STRENGTH_TYPE_1 = 0.75   # Table 12.3-1 Type 1: > 75% of story strength at or on one side of the center of mass
AX_DIVISOR = 1.2                  # 12.8.4.3: Ax = (delta_max / (1.2 delta_avg))^2
AX_MAX = 3.0
TORSION_CASES = ("x+", "x-", "y+", "y-")   # the 12.3.2.1.1 accidental torsion cases assessed
_TOL = 1e-9
DRIFT_DISPLACEMENT_TOLERANCE = 1e-6        # story drift vs adjacent-level displacement difference at one edge (relative)


def _number(value, name, minimum=None, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return number


def live_load_patterns(num_bay_x, num_bay_y):
    """ACI 318-19 6.4.2 arrangements for the frame beams, as floor panel sets.

    Beams are loaded by the slab, so a span pattern is a panel pattern:
    alternate spans (panel columns/rows of one parity, for maximum positive
    moment) and two adjacent spans about every interior support line (for
    maximum negative moment there). Patterns are zero-based [i, j] panels.
    """
    patterns = []
    for axis, count, other in (("x", num_bay_x, num_bay_y), ("y", num_bay_y, num_bay_x)):
        for parity, name in ((0, "even"), (1, "odd")):
            panels = [[i, j] for i in range(num_bay_x) for j in range(num_bay_y)
                      if ((i if axis == "x" else j) % 2) == parity]
            if panels:
                patterns.append({"id": f"alternate_{axis}_{name}", "rule": "ACI 318-19 6.4.2(c)", "panels": panels})
        for support in range(1, count):
            panels = [[i, j] for i in range(num_bay_x) for j in range(num_bay_y)
                      if (i if axis == "x" else j) in (support - 1, support)]
            patterns.append({"id": f"adjacent_{axis}_support_{support}", "rule": "ACI 318-19 6.4.2(b)",
                             "panels": panels})
    return patterns


def strength_load_combinations(sds, redundancy_factor=1.3,
                               orthogonal_fraction=0.3, live_patterns=()):
    """Return the signed D/L/QEx/QEy strength-load descriptors.

    ``live_patterns`` (from ``live_load_patterns``) adds one 1.2D + 1.6L
    gravity combination per arrangement, carrying ``live_pattern`` = its id;
    the unpatterned combinations carry ``live_pattern`` = "all".

    ``dead`` applies to ALL dead load, including member self-weight. ``live``
    applies only to L. ``ex``/``ey`` multiply signed strength-level *unfactored*
    ELF actions QEx/QEy: they already include rho and must not be multiplied by
    it again. Both 100X/30Y and 30X/100Y are included with independent signs.

    This is a D/L/seismic subset, not all ASCE Chapter 2 combinations. Roof
    live, snow, rain, wind and other loads require a separate applicability
    determination. L is not reduced to 0.5. Unpatterned full L is not always
    conservative, so live-load patterning remains a separate prerequisite.
    Rho=1.3 is a conservative assumption, not an assessment of redundancy.
    """
    sds = _number(sds, "sds", minimum=0)
    rho = _number(redundancy_factor, "redundancy_factor", minimum=1)
    fraction = _number(orthogonal_fraction, "orthogonal_fraction", minimum=0.3)
    if fraction > 1:
        raise ValueError("orthogonal_fraction must not exceed 1.")
    combinations = [
        {"id": "gravity_1.4D", "family": "gravity", "dead": 1.4,
         "live": 0.0, "ex": 0.0, "ey": 0.0, "live_pattern": "all"},
        {"id": "gravity_1.2D_1.6L", "family": "gravity", "dead": 1.2,
         "live": 1.6, "ex": 0.0, "ey": 0.0, "live_pattern": "all"},
    ]
    for pattern in live_patterns:
        combinations.append({"id": f"gravity_1.2D_1.6L_{pattern['id']}", "family": "gravity_pattern",
                             "dead": 1.2, "live": 1.6, "ex": 0.0, "ey": 0.0,
                             "live_pattern": pattern["id"]})
    for family, dead, live in (
        ("seismic_high_gravity", 1.2 + 0.2 * sds, 1.0),
        ("seismic_low_gravity", 0.9 - 0.2 * sds, 0.0),
    ):
        for primary, fx, fy in (("X", 1.0, fraction), ("Y", fraction, 1.0)):
            for sx in (-1, 1):
                for sy in (-1, 1):
                    combinations.append({
                        "id": f"{family}_{primary}_x{sx:+d}_y{sy:+d}",
                        "family": family, "dead": dead, "live": live,
                        "ex": rho * sx * fx, "ey": rho * sy * fy, "live_pattern": "all",
                    })
    return combinations


def floor_load_factor(dead_factor, live_factor, dead_load, live_load):
    """Weighted multiplier for a uniform combined-D+L floor-load interface.

    This is only valid when D and L share the same floor distribution. Apply
    ``dead_factor`` separately to member self-weight. It does not implement
    live-load patterning or represent different roof/floor load distributions.
    """
    d = _number(dead_load, "dead_load", minimum=0)
    l = _number(live_load, "live_load", minimum=0)
    df = _number(dead_factor, "dead_factor")
    lf = _number(live_factor, "live_factor")
    if d + l <= 0:
        raise ValueError("The combined floor dead and live load must be positive.")
    return (df * d + lf * l) / (d + l)


def story_node_deltas(upper_displacements, lower_displacements):
    """Subtract aligned floor-node XY displacements using stable grid IDs.

    Values are mappings ``{'x': ux, 'y': uy}``; base displacements must be
    explicitly provided (zero for the declared fixed base). A missing node
    raises rather than silently removing an edge from the drift calculation.
    """
    if not upper_displacements or set(upper_displacements) != set(lower_displacements):
        raise ValueError("Upper and lower floors require identical nonempty grid-node IDs.")
    return {
        node: {direction: _number(upper[direction], "upper displacement") -
               _number(lower_displacements[node][direction], "lower displacement")
               for direction in ("x", "y")}
        for node, upper in upper_displacements.items()
    }


def seismic_design_category(sds, sd1, s1, risk_category="II"):
    """ASCE 7-22 11.6: SDC from Tables 11.6-1/11.6-2, with the S1 >= 0.75 rule."""
    risk = str(risk_category).upper()
    if risk not in ("I", "II", "III", "IV"):
        raise ValueError("risk_category must be I, II, III or IV.")
    if s1 >= 0.75:
        return "F" if risk == "IV" else "E"
    def from_sds(value):
        if value < 0.167:
            return "A"
        if value < 0.33:
            return "B"
        if value < 0.50:
            return "C"
        return "D"
    def from_sd1(value):
        if value < 0.067:
            return "A"
        if value < 0.133:
            return "B"
        if value < 0.20:
            return "C"
        return "D"
    short, one = from_sds(sds), from_sd1(sd1)
    if risk == "IV":
        short = {"A": "A", "B": "C", "C": "D", "D": "D"}[short]
        one = {"A": "A", "B": "C", "C": "D", "D": "D"}[one]
    return max(short, one)


def elf_eligibility(sdc, height_ft, regular, period_sec, ts_sec, edition=CODE_EDITION):
    """Is the ELF procedure (12.8) a permitted analysis procedure, under a named edition?

    ASCE 7-22 12.6 lists (a) the ELF procedure of 12.8, (b) modal response
    spectrum analysis, (c) linear response history analysis or (d) an
    AHJ-approved procedure, for any structure; the ASCE 7-16 Table 12.6-1
    restriction (SDC D-F: irregular, or over 160 ft with T >= 3.5 Ts, not
    permitted) was deleted. NCSEA, Design Guide for Structural
    Irregularities, "Permitted Analytical Procedures", p. 4: "This table has
    been deleted, and ELF procedure is allowed for all cases with no
    restrictions in ASCE 7-22." S. K. Ghosh, Significant Changes in ASCE
    7-22 (SEAU, 2024-02-20), slide 116 quotes the 7-22 12.6 text.

    The 7-16 branch reproduces the previously implemented subset of that
    table so its verdict can be recorded beside the 7-22 one; it is not the
    declared basis of this project. Any other edition is an error, never a
    silent default.
    """
    if edition == "ASCE 7-22":
        return True, ("ASCE 7-22 12.6(a): the equivalent lateral force procedure of 12.8 is permitted for any "
                      "structure; the ASCE 7-16 Table 12.6-1 height/period/irregularity restriction was deleted")
    if edition == "ASCE 7-16":
        if sdc in ("A", "B", "C"):
            return True, "ASCE 7-16 Table 12.6-1: SDC B/C, all structures permitted (SDC A per 11.7)"
        if regular and height_ft <= 160.0:
            return True, "ASCE 7-16 Table 12.6-1: no structural irregularities and height <= 160 ft"
        if regular and period_sec < 3.5 * ts_sec:
            return True, "ASCE 7-16 Table 12.6-1: no irregularities, T < 3.5 Ts"
        return False, "ASCE 7-16 Table 12.6-1: ELF not permitted for this height/period/irregularity combination"
    raise ValueError(f"Unknown code edition {edition!r}; supported: {SUPPORTED_EDITIONS}.")


def story_drift_ratio(delta_a_in, delta_b_in):
    """Table 12.3-1 / 12.3.2.1.1 ratio at one story: the larger edge story drift over the average of the two."""
    a = abs(_number(delta_a_in, "edge story drift a"))
    b = abs(_number(delta_b_in, "edge story drift b"))
    average = 0.5 * (a + b)
    return max(a, b) / average if average > 0 else 1.0


def amplification_from_level_displacements(delta_a_in, delta_b_in):
    """ASCE 7-22 12.8.4.3 at one level: Ax = (delta_max / (1.2 delta_avg))^2, 1 <= Ax <= 3.

    ``delta_a_in``/``delta_b_in`` are the displacements at the two extreme
    points of the structure at that level (signed; magnitudes are used),
    computed with Ax = 1. This is a ratio of *level displacements*, not of
    story drifts: the TIR of Table 12.3-1 uses story drifts and the two can
    differ (edge displacements (0.10, 0.10) at level 1 and (0.30, 0.20) at
    level 2 give a story-drift ratio of 1.333 but a level ratio of 1.2).
    Returns {"ratio", "ax"}.
    """
    a = abs(_number(delta_a_in, "edge level displacement a"))
    b = abs(_number(delta_b_in, "edge level displacement b"))
    average = 0.5 * (a + b)
    ratio = max(a, b) / average if average > 0 else 1.0
    return {"ratio": ratio, "ax": min(AX_MAX, max(1.0, (ratio / AX_DIVISOR) ** 2))}


def one_sided_strength_fraction(positions, strengths, center):
    """Largest fraction of a story's lateral strength provided at or on one side of the center of mass.

    ASCE 7-22 Table 12.3-1 Type 1, second criterion: "more than 75% of any
    story's lateral strength below the diaphragm is provided at or on one
    side of the center of mass" (Ghosh, SEAU 2024, slide 103). "At or on"
    counts a line through the center on both sides, so three identical
    lines at -1, 0, +1 give 2/3, not 1/2. ``positions`` are the frame-line
    coordinates perpendicular to the direction considered, ``strengths``
    their story lateral strengths (any consistent unit), ``center`` the
    center-of-mass coordinate. Returns (fraction, detail).
    """
    positions = [_number(p, "position") for p in positions]
    strengths = [_number(s, "strength", minimum=0.0) for s in strengths]
    center = _number(center, "center")
    if len(positions) != len(strengths) or not positions:
        raise ValueError("positions and strengths must be nonempty and of equal length.")
    total = sum(strengths)
    if total <= 0:
        raise ValueError("The story lateral strength must be positive.")
    tolerance = 1e-9 * max(1.0, max(abs(p) for p in positions))
    low = sum(s for p, s in zip(positions, strengths) if p <= center + tolerance)
    high = sum(s for p, s in zip(positions, strengths) if p >= center - tolerance)
    fraction = max(low, high) / total
    return fraction, {"center": center, "total_strength": total, "at_or_below_center": low, "at_or_above_center": high,
                      "lines_at_center": sum(1 for p in positions if abs(p - center) <= tolerance),
                      "criterion": "more than 75% at or on one side of the center of mass (Table 12.3-1 Type 1)"}


STRENGTH_MODEL_REVIEW_ITEM = "M1"
STRENGTH_MODEL_STATUSES = ("provisional", "verified")


def classify_torsional_irregularity(tir, one_side_strength_fraction=None, strength_model_verified=False):
    """ASCE 7-22 Table 12.3-1 Type 1 torsional irregularity from its two criteria.

    Type 1 exists where the torsional irregularity ratio (TIR, 12.3.2.1.1:
    the largest edge story drift over the average of the two edge story
    drifts, from ELF forces with accidental torsion and Ax = 1.0, over every
    story, direction and accidental torsion case) exceeds 1.2, or where more
    than 75% of any story's lateral strength below the diaphragm is provided
    at or on one side of the center of mass (Ghosh, SEAU 2024, slides
    101-103). 7-22 has no separate extreme type: Type 1 with TIR > 1.4 is
    what 7-16 called Type 1b (slide 105); 7-22 removed the 7-16 12.3.3.1
    prohibition of that case in SDC E/F (FEMA P-2192 Vol. 1 sec. 1.4.5;
    California 2025 Title 24, 1617.12.6) and addresses it through
    12.3.4.2.1 for rho in SDC D-F. The 12.8.4.3 amplification is NOT derived
    here: it is a ratio of level displacements
    (amplification_from_level_displacements), not of the TIR.

    The two criteria stay distinct. The strength fraction comes from a
    story-strength model whose applicability is a review decision
    (STRENGTH_MODEL_REVIEW_ITEM); ``strength_model_verified`` says whether
    that decision has been asserted. An unknown fraction leaves the
    criterion unevaluated; a fraction from a provisional model can
    establish Type 1 (conservative) but never its absence; only a verified
    model resolves the criterion both ways. So the label is ``type_1`` when
    either criterion triggers, ``none`` when the TIR is at or below 1.2 AND
    a verified model places at most 75% on one side, and ``unresolved``
    otherwise. A positive TIR establishes Type 1 on its own.
    """
    tir = _number(tir, "tir", minimum=1.0)
    fraction = (None if one_side_strength_fraction is None
                else _number(one_side_strength_fraction, "one_side_strength_fraction", minimum=0.0))
    if fraction is not None and fraction > 1.0:
        raise ValueError("one_side_strength_fraction must not exceed 1.")
    by_tir = tir > TIR_TYPE_1
    by_strength = None if fraction is None else fraction > ONE_SIDE_STRENGTH_TYPE_1
    verified = bool(strength_model_verified) and fraction is not None
    status = "unevaluated" if fraction is None else ("verified" if verified else "provisional")
    type_1 = by_tir or bool(by_strength)
    absence_established = (not by_tir) and by_strength is False and verified
    label = "type_1" if type_1 else ("none" if absence_established else "unresolved")
    return {"tir": tir, "by_tir": by_tir, "by_strength_distribution": by_strength,
            "one_side_strength_fraction": fraction, "type_1": type_1,
            "label": label,
            "strength_criterion_evaluated": by_strength is not None,
            "strength_criterion_status": status,
            "strength_criterion_resolved": verified,
            "strength_model_review_item": STRENGTH_MODEL_REVIEW_ITEM,
            "absence_established": absence_established,
            "tir_exceeds_1_4": tir > TIR_FORMER_TYPE_1B,
            "edition": CODE_EDITION,
            "basis": ("ASCE 7-22 Table 12.3-1 Type 1: TIR > 1.2 (12.3.2.1.1, story drifts) or more than 75% of a story's "
                      "lateral strength at or on one side of the center of mass; the former Type 1b is Type 1 with "
                      "TIR > 1.4 (12.3.4.2.1 for rho; no 7-22 prohibition); Ax per 12.8.4.3 from level displacements; "
                      "the strength criterion is resolved only on a verified story-strength model (review item "
                      f"{STRENGTH_MODEL_REVIEW_ITEM}); a provisional model can establish Type 1, not its absence")}


def torsional_irregularity_limit(sdc):
    """The TIR ceiling applied to a design: a project population policy in every SDC.

    ASCE 7-16 12.3.3.1 prohibited extreme torsional irregularity (Type 1b)
    in SDC E and F. ASCE 7-22 removed that prohibition: FEMA P-2192 Vol. 1
    sec. 1.4.5 (2020 NEHRP Design Examples) states it, and California's 2025
    Title 24 adoption record for ASCE 7 12.3.3.1 (1617.12.6 / 1617A.1.10)
    reads "Repealed all language related to the extreme torsional
    irregularities. Revised to align with ASCE 7-22." The S. K. Ghosh slide
    109 redline that appeared to retain a Type 1b clause is a flattened
    deletion. The 1.4 ceiling therefore has no code basis in any SDC under
    7-22 and is retained only as a research population policy pending a
    decision; Type 1 with TIR > 1.4 still carries 12.3.4.2.1 for rho in
    SDC D-F. Returns (limit, kind, basis); ``sdc`` is recorded, not decisive.
    """
    return (TIR_PROJECT_POLICY_LIMIT, "project_policy",
            f"project population policy TIR <= 1.4 (SDC {sdc or 'unknown'}): ASCE 7-22 removed the 7-16 12.3.3.1 "
            "prohibition of extreme torsional irregularity (FEMA P-2192 Vol. 1 sec. 1.4.5; California 2025 Title 24 "
            "1617.12.6); no edition-current provision prohibits TIR > 1.4, so this ceiling is a research restriction "
            "pending a decision, not a code provision")


def validate_torsion_assessment(torsion, num_floor, policy_ratio=None):
    """Recompute the torsion assessment from its row primitives and reject anything inconsistent.

    Returns a dict with ``valid``; when valid it carries the recomputed
    ``tir``, ``tir_by_case``, per-row and per-level 12.8.4.3 ``ax`` values,
    the envelope ``ax_required_envelope`` and ``cases``; otherwise ``reason``
    (and ``legacy`` True for a saved single-sign assessment, which is valid
    evidence of what it was, not of the 7-22 case set). Rules: every row
    names a story in 1..num_floor, a direction, an accidental torsion case
    consistent with its eccentricity sign, finite nonnegative edge story
    drifts whose ratio reproduces the stored one, and finite signed edge
    level displacements whose ratio and Ax reproduce the stored ones; the
    rows cover every (story, direction, sign) exactly once; the stored
    ``cases``, ``tir_by_case``, ``tir``/``max_drift_ratio`` and
    ``amplification_by_level`` equal the recomputed values; the eccentricity
    ratio is at least 5% and equals the declared policy value; the
    assessment ran with Ax = 1. Stored scalars are never trusted on their
    own.
    """
    if not isinstance(torsion, dict) or not torsion:
        return {"valid": False, "reason": "no torsion assessment was saved with this design", "legacy": False}
    rows = torsion.get("stories")
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        return {"valid": False, "reason": "the assessment carries no per-story rows; scalar TIR/Ax alone are not evidence",
                "legacy": False}
    if "cases" not in torsion or any("case" not in r for r in rows):
        return {"valid": False, "legacy": True,
                "reason": ("legacy single-eccentricity-sign assessment (rows carry no accidental-torsion case); "
                           "ASCE 7-22 12.3.2.1.1 needs each case and 12.8.4.3 needs level displacements")}
    try:
        floors = int(num_floor)
        if floors <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return {"valid": False, "reason": "the story count is unknown", "legacy": False}
    seen, per_case, ax_by_level, ax_rows = set(), {}, {}, []
    for r in rows:
        try:
            story = r.get("story")
            if type(story) is not int or not 1 <= story <= floors:
                raise ValueError(f"story {story!r} is not in 1..{floors}")
            direction = r.get("direction")
            sign = r.get("eccentricity_sign")
            if direction not in ("x", "y") or sign not in (1, 1.0, -1, -1.0) or isinstance(sign, bool):
                raise ValueError(f"direction {direction!r} / sign {sign!r} are not a valid accidental torsion case")
            case = f"{direction}{'+' if sign > 0 else '-'}"
            if r.get("case") != case:
                raise ValueError(f"row case {r.get('case')!r} does not match direction/sign {case}")
            key = (story, direction, 1 if sign > 0 else -1)
            if key in seen:
                raise ValueError(f"duplicate row for story {story} case {case}")
            seen.add(key)
            drift = story_drift_ratio(_number(r.get("delta_end_a_in"), "delta_end_a_in", minimum=0.0),
                                      _number(r.get("delta_end_b_in"), "delta_end_b_in", minimum=0.0))
            stored = _number(r.get("delta_max_over_avg"), "delta_max_over_avg")
            if not math.isclose(drift, stored, rel_tol=_TOL, abs_tol=_TOL):
                raise ValueError(f"story {story} {case}: stored drift ratio {stored} != recomputed {drift}")
            if "delta_level_a_in" not in r or "delta_level_b_in" not in r:
                raise ValueError(f"story {story} {case}: edge level displacements are not recorded (12.8.4.3)")
            level = amplification_from_level_displacements(_number(r.get("delta_level_a_in"), "delta_level_a_in"),
                                                           _number(r.get("delta_level_b_in"), "delta_level_b_in"))
            for field, value in (("level_max_over_avg", level["ratio"]), ("ax_level", level["ax"])):
                if not math.isclose(_number(r.get(field), field), value, rel_tol=_TOL, abs_tol=_TOL):
                    raise ValueError(f"story {story} {case}: stored {field} {r.get(field)} != recomputed {value}")
        except ValueError as exc:
            return {"valid": False, "reason": str(exc), "legacy": False}
        per_case[case] = max(per_case.get(case, 0.0), drift)
        ax_by_level[story] = max(ax_by_level.get(story, 0.0), level["ax"])
        ax_rows.append({"story": story, "case": case, "ratio": level["ratio"], "ax": level["ax"]})
    expected = {(k, d, s) for k in range(1, floors + 1) for d in ("x", "y") for s in (1, -1)}
    if seen != expected:
        missing = sorted(expected - seen)[:6]
        return {"valid": False, "reason": f"rows do not cover every story, direction and eccentricity sign exactly once; "
                                          f"missing e.g. {missing}", "legacy": False}
    # The two primitives come from one solved state: at each edge and case the
    # story drift is the change of that edge's level displacement between
    # adjacent levels, the base being fixed (u_0 = 0). Checked to
    # DRIFT_DISPLACEMENT_TOLERANCE (relative, with a 1e-9 in floor), the rigid
    # diaphragm giving every node of an edge line the same in-plane
    # displacement to solver precision.
    if torsion.get("base", "fixed") != "fixed":
        return {"valid": False, "reason": f"assessment base condition {torsion.get('base')!r} is not the fixed base the "
                                          "drift/displacement consistency check assumes", "legacy": False}
    by_case_story = {(r["case"], r["story"]): r for r in rows}
    for case in TORSION_CASES:
        previous = {"a": 0.0, "b": 0.0}
        for story in range(1, floors + 1):
            r = by_case_story[(case, story)]
            for edge in ("a", "b"):
                level = float(r[f"delta_level_{edge}_in"])
                implied = abs(level - previous[edge])
                stored = float(r[f"delta_end_{edge}_in"])
                if not math.isclose(implied, stored, rel_tol=DRIFT_DISPLACEMENT_TOLERANCE, abs_tol=1e-9):
                    return {"valid": False, "legacy": False,
                            "reason": (f"story {story} {case} edge {edge}: recorded story drift {stored} != |u_{story} - u_{story - 1}| "
                                       f"= {implied} from the recorded edge level displacements (fixed base)")}
                previous[edge] = level
    cases = torsion.get("cases")
    if not isinstance(cases, list) or sorted(cases) != sorted(TORSION_CASES) or sorted(per_case) != sorted(TORSION_CASES):
        return {"valid": False, "reason": f"stored cases {cases!r} are not the four 12.3.2.1.1 accidental torsion cases",
                "legacy": False}
    tir = max(per_case.values())
    stored_by_case = torsion.get("tir_by_case")
    if not isinstance(stored_by_case, dict) or set(stored_by_case) != set(per_case) or any(
            not _finite_close(stored_by_case.get(c), per_case[c]) for c in per_case):
        return {"valid": False, "reason": "stored tir_by_case does not reproduce the per-case maxima of the rows",
                "legacy": False}
    for field in ("tir", "max_drift_ratio"):
        if not _finite_close(torsion.get(field), tir):
            return {"valid": False, "reason": f"stored {field} {torsion.get(field)!r} != recomputed TIR {tir}", "legacy": False}
    ratio = torsion.get("ratio")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio < 0.05:
        return {"valid": False, "reason": f"eccentricity ratio {ratio!r} is not a finite value of at least 5% (12.8.4.2)",
                "legacy": False}
    if policy_ratio is not None and not _finite_close(ratio, policy_ratio):
        return {"valid": False, "reason": f"eccentricity ratio {ratio} differs from the declared policy value {policy_ratio}",
                "legacy": False}
    if not _finite_close(torsion.get("assessment_amplification"), 1.0):
        return {"valid": False, "reason": "the assessment did not record Ax = 1 (12.3.2.1.1 / 12.8.4.3 compute the "
                                          "ratios assuming Ax = 1)", "legacy": False}
    stored_levels = torsion.get("amplification_by_level")
    if stored_levels is not None:
        if (not isinstance(stored_levels, dict) or {int(k) for k in stored_levels} != set(ax_by_level)
                or any(not _finite_close(stored_levels[k], ax_by_level[int(k)]) for k in stored_levels)):
            return {"valid": False, "reason": "stored amplification_by_level does not reproduce the per-level 12.8.4.3 values",
                    "legacy": False}
    return {"valid": True, "reason": None, "legacy": False, "tir": tir, "tir_by_case": per_case,
            "ax_by_level": ax_by_level, "ax_rows": ax_rows, "ax_required_envelope": max(ax_by_level.values()),
            "cases": list(TORSION_CASES), "rows": len(rows), "eccentricity_ratio": float(ratio)}


def _finite_close(value, target):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    return math.isclose(float(value), float(target), rel_tol=_TOL, abs_tol=_TOL)


STRENGTH_FAMILIES = ("x_edge", "x_interior", "y_edge", "y_interior")


def line_story_strengths(direction, geometry, families, story_h_in):
    """Frame-line positions and beam-mechanism story strengths for one direction (the declared model).

    Direction x is resisted by the frames on the y-grid lines (num_bay_y + 1
    lines at bay_y spacing, each spanning num_bay_x bays); direction y by
    the x-grid lines. A line's story strength is bays x (Mn- + Mn+) / h of
    its beam family: the edge family on the two perimeter lines, the
    interior family elsewhere. Returns (positions, strengths, line_families).
    """
    if direction == "x":
        lines, spacing, bays = int(geometry["num_bay_y"]) + 1, float(geometry["bay_y_in"]), int(geometry["num_bay_x"])
    elif direction == "y":
        lines, spacing, bays = int(geometry["num_bay_x"]) + 1, float(geometry["bay_x_in"]), int(geometry["num_bay_y"])
    else:
        raise ValueError("direction must be x or y.")
    h = _number(story_h_in, "story_h_in", positive=True)
    positions, strengths, names = [], [], []
    for j in range(lines):
        family = f"{direction}_{'edge' if j in (0, lines - 1) else 'interior'}"
        values = families[family]
        positions.append(j * spacing)
        strengths.append(bays * (_number(values["mn_negative_kip_in"], "mn_negative_kip_in", minimum=0.0)
                                 + _number(values["mn_positive_kip_in"], "mn_positive_kip_in", minimum=0.0)) / h)
        names.append(family)
    return positions, strengths, names


def strength_distribution_fraction(regularity, geometry=None, families=None):
    """The validated one-sided strength fraction from the saved regularity evidence: (fraction or None, reason).

    Evidence contract (line evidence, the only accepted form): the block
    names its ``model`` and carries ``uniform_over_height`` (claim and
    basis), ``strength_inputs`` (story height, bays and the four beam
    families' Mn- / Mn+ it was priced on) and ``by_direction`` with exactly
    ``x`` and ``y``, each listing every frame line of the geometry at its
    grid position with its story strength, the center of mass and the
    direction's fraction. Everything is recomputed: the line roster and
    positions against ``geometry``, each line strength from the recorded
    family strengths, each direction's fraction and the overall maximum,
    and, when the beam families recomputed from the record's final cage are
    supplied (``families``), the recorded family strengths against them.
    A scalar with a model name, a single direction, a short or displaced
    roster, or a family strength that does not reproduce is unknown, never
    a fraction. No analytical-bound format is accepted: absent line
    evidence stays unevaluated. The block must also carry an
    ``applicability`` record whose ``status`` is ``provisional`` or
    ``verified`` (STRENGTH_MODEL_STATUSES): complete arithmetic is not
    scientific applicability, and the evaluator decides from that status
    (checked against the record's assertion policy) whether the criterion
    can establish the absence of the irregularity.
    """
    block = (regularity or {}).get("lateral_strength_distribution") if isinstance(regularity, dict) else None
    if not isinstance(block, dict):
        return None, "no lateral_strength_distribution evidence is saved"
    fraction = block.get("one_side_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        return None, f"one_side_fraction {fraction!r} is not a finite value in [0, 1]"
    if not isinstance(block.get("model"), str) or not block["model"].strip():
        return None, "the strength-distribution model is not named"
    applicability = block.get("applicability")
    if not isinstance(applicability, dict) or applicability.get("status") not in STRENGTH_MODEL_STATUSES:
        return None, (f"the strength model carries no applicability status ({STRENGTH_MODEL_STATUSES}); complete line "
                      f"arithmetic does not establish scientific applicability (review item {STRENGTH_MODEL_REVIEW_ITEM})")
    if isinstance(block.get("stories"), list):
        return _story_strength_fraction(block, geometry, fraction)
    uniform = block.get("uniform_over_height")
    if not isinstance(uniform, dict) or uniform.get("claim") is not True or not isinstance(uniform.get("basis"), str) or not uniform["basis"].strip():
        return None, "story coverage is not established: no uniform_over_height claim with a basis"
    inputs = block.get("strength_inputs")
    if not isinstance(inputs, dict) or not isinstance(inputs.get("families"), dict):
        return None, "the strength inputs (families, bays, story height) the lines were priced on are not recorded"
    recorded = inputs["families"]
    if set(recorded) != set(STRENGTH_FAMILIES):
        return None, f"strength inputs must carry the four beam families, found {sorted(recorded)}"
    by_direction = block.get("by_direction")
    if not isinstance(by_direction, dict) or set(by_direction) != {"x", "y"}:
        return None, f"line evidence must cover exactly directions x and y, found {sorted(by_direction) if isinstance(by_direction, dict) else None}"
    if geometry is None:
        return None, "the geometry needed to check the line roster is unknown"
    try:
        story_h = _number(inputs.get("story_h_in"), "story_h_in", positive=True)
        if not _finite_close(story_h, geometry.get("story_h_in")):
            return None, f"strength inputs story height {story_h} != geometry {geometry.get('story_h_in')}"
        bays = inputs.get("bays") or {}
        if bays.get("x") != geometry.get("num_bay_x") or bays.get("y") != geometry.get("num_bay_y"):
            return None, f"strength inputs bays {bays} != geometry ({geometry.get('num_bay_x')}, {geometry.get('num_bay_y')})"
        if families is not None:
            for name in STRENGTH_FAMILIES:
                for key in ("mn_negative_kip_in", "mn_positive_kip_in"):
                    if not _finite_close(recorded[name].get(key), families[name][key]):
                        return None, (f"recorded {name} {key} {recorded[name].get(key)!r} does not reproduce from the record's "
                                      f"final cage ({families[name][key]}): stale strength inputs")
        recomputed = []
        for direction in ("x", "y"):
            item = by_direction[direction]
            if not isinstance(item, dict):
                return None, f"direction {direction}: line evidence is not a record"
            positions, strengths, names = line_story_strengths(direction, geometry, recorded, story_h)
            saved_positions = item.get("line_positions_in") or []
            saved_strengths = item.get("line_story_strength_kip") or []
            if len(saved_positions) != len(positions) or len(saved_strengths) != len(strengths):
                return None, (f"direction {direction}: {len(saved_positions)} lines recorded, the geometry has {len(positions)} "
                              f"frame lines")
            if any(not _finite_close(a, b) for a, b in zip(saved_positions, positions)):
                return None, f"direction {direction}: recorded line positions {saved_positions} != grid positions {positions}"
            if any(not _finite_close(a, b) for a, b in zip(saved_strengths, strengths)):
                return None, f"direction {direction}: recorded line strengths do not reproduce from the recorded family strengths"
            if not _finite_close(item.get("center"), 0.5 * positions[-1]):
                return None, f"direction {direction}: center {item.get('center')!r} is not the plan center {0.5 * positions[-1]}"
            value, detail = one_sided_strength_fraction(positions, strengths, 0.5 * positions[-1])
            if not _finite_close(item.get("one_side_fraction"), value):
                return None, f"direction {direction}: stored fraction {item.get('one_side_fraction')!r} != recomputed {value}"
            if item.get("lines_at_center") != detail["lines_at_center"]:
                return None, f"direction {direction}: lines_at_center {item.get('lines_at_center')!r} != {detail['lines_at_center']}"
            recomputed.append(value)
    except (ValueError, TypeError, KeyError) as exc:
        return None, f"line evidence invalid ({type(exc).__name__}: {exc})"
    if not _finite_close(fraction, max(recomputed)):
        return None, f"overall fraction {fraction} != the largest direction value {max(recomputed)}"
    return float(fraction), None


def _story_strength_fraction(block, geometry, fraction):
    """The story-by-story form of the line evidence (a grouped design): (fraction or None, reason).

    A grouped design prices every story on the beams installed there
    (Design.Grouped_Design.story_strength_distribution), so the block claims
    no uniformity over height and carries ``stories``: one entry per story,
    each with ``by_direction`` x and y listing every frame line at its grid
    position with its story strength. Recomputed here: the story roster, the
    line roster and positions against ``geometry``, each direction's story
    strength and one-sided fraction, and the overall maximum. The line
    strengths themselves are recomputed from the installed members and
    compared by the grouped qualification (Design.Grouped_Record), which is
    the only caller that can resolve them.
    """
    uniform = block.get("uniform_over_height")
    if (not isinstance(uniform, dict) or uniform.get("claim") is not False
            or not isinstance(uniform.get("basis"), str) or not uniform["basis"].strip()):
        return None, "story-by-story line evidence must state that it claims no uniformity over height, with a basis"
    if geometry is None:
        return None, "the geometry needed to check the line roster is unknown"
    try:
        stories, num_floor = block["stories"], int(geometry["num_floor"])
        if [entry.get("story") if isinstance(entry, dict) else None for entry in stories] != list(range(1, num_floor + 1)):
            return None, f"the story roster does not list stories 1..{num_floor} in order"
        recomputed = []
        for entry in stories:
            by_direction = entry.get("by_direction")
            if not isinstance(by_direction, dict) or set(by_direction) != {"x", "y"}:
                return None, f"story {entry['story']}: line evidence must cover exactly directions x and y"
            for direction in ("x", "y"):
                item = by_direction[direction]
                if not isinstance(item, dict):
                    return None, f"story {entry['story']} direction {direction}: line evidence is not a record"
                if direction == "x":
                    lines, spacing = int(geometry["num_bay_y"]) + 1, float(geometry["bay_y_in"])
                else:
                    lines, spacing = int(geometry["num_bay_x"]) + 1, float(geometry["bay_x_in"])
                positions = [j * spacing for j in range(lines)]
                saved_positions = item.get("line_positions_in") or []
                strengths = item.get("line_story_strength_kip") or []
                where = f"story {entry['story']} direction {direction}"
                if len(saved_positions) != lines or len(strengths) != lines:
                    return None, f"{where}: {len(saved_positions)} lines recorded, the geometry has {lines} frame lines"
                if any(not _finite_close(a, b) for a, b in zip(saved_positions, positions)):
                    return None, f"{where}: recorded line positions {saved_positions} != grid positions {positions}"
                strengths = [_number(value, "line_story_strength_kip", minimum=0.0) for value in strengths]
                if not _finite_close(item.get("story_strength_kip"), sum(strengths)):
                    return None, f"{where}: story strength {item.get('story_strength_kip')!r} is not the sum of its lines"
                if not _finite_close(item.get("center"), 0.5 * positions[-1]):
                    return None, f"{where}: center {item.get('center')!r} is not the plan center {0.5 * positions[-1]}"
                value, detail = one_sided_strength_fraction(positions, strengths, 0.5 * positions[-1])
                if not _finite_close(item.get("one_side_fraction"), value):
                    return None, f"{where}: stored fraction {item.get('one_side_fraction')!r} != recomputed {value}"
                if item.get("lines_at_center") != detail["lines_at_center"]:
                    return None, f"{where}: lines_at_center {item.get('lines_at_center')!r} != {detail['lines_at_center']}"
                recomputed.append(value)
    except (ValueError, TypeError, KeyError) as exc:
        return None, f"line evidence invalid ({type(exc).__name__}: {exc})"
    if not _finite_close(fraction, max(recomputed)):
        return None, f"overall fraction {fraction} != the largest story and direction value {max(recomputed)}"
    return float(fraction), None


def redundancy_requirement(sdc, conditions_12_3_4_2_verified=False):
    """rho required by ASCE 7-22 12.3.4 for the strength combinations.

    12.3.4.1 permits rho = 1.0 for SDC B and C (and for drift, P-Delta and
    the other listed uses in every SDC); 12.3.4.2 requires rho = 1.3 for
    SDC D, E and F unless (a) or (b) is demonstrated. Those demonstrations
    are not made for the archetype, so 1.3 is the requirement there;
    carrying 1.3 in every strength combination satisfies both cases
    conservatively. 12.3.4.2.1 sets rho = 1.3 for the whole structure in both
    directions where a Type 1 irregularity has TIR > 1.4 in both directions
    (SDC D-F); 1.3 is carried in every case there, so it is met without a branch.
    """
    if sdc in ("D", "E", "F"):
        required = 1.0 if conditions_12_3_4_2_verified else REDUNDANCY_FACTOR_STRENGTH
        basis = "ASCE 7-22 12.3.4.2: rho = 1.3 for SDC D-F unless condition 1 or condition 2 is demonstrated"
    else:
        required = 1.0
        basis = "ASCE 7-22 12.3.4.1: rho = 1.0 for SDC B and C"
    return {"required": required, "basis": basis, "conditions_12_3_4_2_verified": bool(conditions_12_3_4_2_verified)}


def _redundancy_factor_used(record, basis):
    """rho carried by the saved strength combinations (declared, else read from the action factors)."""
    declared = basis.get("redundancy_factor")
    if isinstance(declared, (int, float)) and not isinstance(declared, bool) and math.isfinite(declared):
        return float(declared)
    combinations = (record.get("design_actions") or {}).get("combinations") or []
    factors = [abs(c.get("ex", 0.0)) for c in combinations if str(c.get("family", "")).startswith("seismic")]
    factors += [abs(c.get("ey", 0.0)) for c in combinations if str(c.get("family", "")).startswith("seismic")]
    finite = [f for f in factors if isinstance(f, (int, float)) and math.isfinite(f)]
    return max(finite) if finite else None


SITE_CLASSES = ASCE_SITE_CLASSES              # ASCE 7-22 11.4.2, Table 20.2-1 (one source: Structure_Parameters)
SITE_RESPONSE_ANALYSIS_CLASSES = ("F",)       # 11.4.7: site response analysis for Site Class F only (20.3.1 exemption aside)
RISK_CATEGORIES = ("I", "II", "III", "IV")


def demand_policy_problems(policy):
    """Why a DemandPolicy is not an acceptable declaration; empty when it is.

    A declaration is a named, dated basis with values in their domains. A
    blank or whitespace author, date or basis, a date that is not an ISO
    calendar date, a site class or risk category that is not exactly one of
    ASCE 7-22's (no case folding, no surrounding whitespace: the value is
    used verbatim downstream, so a noncanonical spelling is rejected rather
    than normalised in one place and not another), a blank occupancy, a
    nonfinite or negative load value, an accidental torsion ratio below the
    5% of 12.8.4.2, or a non-Boolean flag is a problem, listed explicitly so
    a typo never reads as approved evidence.
    """
    problems = []
    if not isinstance(policy, dict):
        return ["DemandPolicy is missing"]
    for key in ("declared_by", "declaration_date", "declaration_basis"):
        value = policy.get(key)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{key} is blank")
    stamp = policy.get("declaration_date")
    if isinstance(stamp, str) and stamp.strip():
        try:
            if date.fromisoformat(stamp).isoformat() != stamp:
                problems.append("declaration_date is not an ISO calendar date")
        except ValueError:
            problems.append("declaration_date is not an ISO calendar date")
    site_class = policy.get("site_class")
    if not isinstance(site_class, str) or site_class not in SITE_CLASSES:
        problems.append(f"site_class {site_class!r} is not exactly one of {SITE_CLASSES}")
    risk = policy.get("risk_category")
    if not isinstance(risk, str) or risk not in RISK_CATEGORIES:
        problems.append(f"risk_category {risk!r} is not exactly one of {RISK_CATEGORIES}")
    occupancy = policy.get("occupancy")
    if not isinstance(occupancy, str) or not occupancy.strip():
        problems.append("occupancy is blank")
    for key in ("partition_allowance_ksf", "storage_live_fraction_in_weight", "roof_live_load_ksf", "snow_load_ksf"):
        value = policy.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            problems.append(f"{key} {value!r} is not a finite nonnegative number")
    fraction = policy.get("storage_live_fraction_in_weight")
    if isinstance(fraction, (int, float)) and not isinstance(fraction, bool) and math.isfinite(fraction) and fraction > 1:
        problems.append("storage_live_fraction_in_weight exceeds 1")
    ratio = policy.get("accidental_torsion_ratio")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio < 0.05:
        problems.append(f"accidental_torsion_ratio {ratio!r} is below the 5% of ASCE 7-22 12.8.4.2")
    for key in ("wind_governs", "rain_ponding_excluded", "live_load_patterning"):
        if not isinstance(policy.get(key), bool):
            problems.append(f"{key} is not Boolean")
    return problems


def risk_basis_check(policy, seismic, drift, identity_inputs=None):
    """demands.risk_basis: the saved category, importance factor and drift criterion agree with each
    other and with ASCE 7-22 Tables 1.5-2 and 12.12-1. Reads the record only; no module state."""
    clause = "ASCE 7-22 Table 1.5-2; Table 12.12-1; 12.12.1.1"
    category, ie = seismic.get("risk_category"), seismic.get("importance_factor")
    if category is None or ie is None:
        return not_evaluated("demands.risk_basis", clause,
                             "The record's seismic block carries no risk category and importance factor (designed before "
                             "the explicit risk basis). Redesign it under the current schema; do not relabel it.")
    problems = []
    if policy.get("risk_category") != category:
        problems.append(f"DemandPolicy declares {policy.get('risk_category')!r}, the design was made for {category!r}")
    base = drift.get("base_drift_limit_ratio")
    try:
        table = story_drift_basis(category, ie, base)
    except (ValueError, TypeError) as exc:
        table = None
        problems.append(str(exc))
    if base is None:
        problems.append("the drift assumptions carry no base drift limit")
    if drift.get("risk_category") not in (None, category):
        problems.append(f"the drift screen was evaluated for {drift.get('risk_category')!r}")
    if drift.get("importance_factor") is not None and abs(float(drift["importance_factor"]) - float(ie)) > 1e-12:
        problems.append(f"the drift screen used Ie = {drift['importance_factor']}")
    if drift.get("low_rise_allowance_asserted"):
        problems.append("the low-rise drift allowance is asserted; its conditions are not verified in this scope")
    recorded = (identity_inputs or {}).get("ASCE_IE")
    if recorded is not None and abs(float(recorded) - float(ie)) > 1e-12:
        problems.append(f"the request identity carries ASCE_IE = {recorded}")
    if (identity_inputs or {}).get("ASCE_RISK_CATEGORY") not in (None, category):
        problems.append(f"the request identity carries risk category {identity_inputs.get('ASCE_RISK_CATEGORY')!r}")
    return make_check("demands.risk_basis", clause, int(not problems), 1, "==",
                      details={"risk_category": category, "importance_factor": ie,
                               "table_importance_factor": IMPORTANCE_FACTOR_BY_RISK_CATEGORY.get(category),
                               "base_drift_limit_ratio": base, "table_drift_limit_ratio": (table or {}).get("table_limit"),
                               "effective_drift_limit_ratio": drift.get("effective_drift_limit_ratio"),
                               "low_rise_allowance_asserted": bool(drift.get("low_rise_allowance_asserted")),
                               "problems": problems,
                               "basis": "category, Ie and the Table 12.12-1 row are one set; the limit is divided by rho for "
                                        "moment frames in SDC D-F; the low-rise allowance is not asserted"})


def evaluate_demand_basis(record, strength_families=None):
    """Evaluate the demand-scope items from the saved policy, loads and results.

    Everything here reads what the design record carries: the declared
    ``DemandPolicy`` (request identity), the load inventory, the torsion
    assessment and the drift/ELF assumptions. Items stay not_evaluated
    while the policy carries no declaration basis. ``strength_families``
    are the four beam families recomputed from the record's final cage
    (qualification passes the ones it rebuilt); without them they are
    rebuilt here, and a record they cannot be rebuilt from leaves the
    strength-distribution criterion unevaluated.
    """
    policy = ((record.get("request_identity") or {}).get("policy") or {}).get("demands") \
        or (record.get("demand_basis") or {}).get("policy") or {}
    basis = record.get("demand_basis") or {}
    seismic = record.get("seismic") or {}
    loads = record.get("floor_loads") or {}
    geometry = record.get("geometry") or {}
    checks = []
    problems = demand_policy_problems(policy)
    declared = not problems
    def declaration_missing(key, clause, what):
        return not_evaluated(f"demands.{key}", clause,
                             f"{what}: declare DemandPolicy (declared_by, declaration_date, declaration_basis) in "
                             f"Design/Config.py. Rejected: {'; '.join(problems) if problems else 'not declared'}.")
    # Site hazard and SDC.
    try:
        sdc = seismic_design_category(seismic["sds"], seismic["sd1"], seismic["s1"], policy.get("risk_category", "II"))
    except (KeyError, ValueError, TypeError):
        sdc = None
    label = seismic.get("site_label") or ""
    label_sdc = label.split("_")[1].upper() if label.startswith("sdc_") and len(label.split("_")) > 1 else None
    if sdc is None or not declared:
        checks.append(declaration_missing("site_hazard", "ASCE 7-22 Chapters 11 and 21",
                                          "Site class and risk category are declarations"))
    else:
        # ``declared`` guarantees the canonical spelling; no normalisation here. ASCE 7-22 requires a site
        # response analysis for Site Class F only (11.4.7); the 7-16 rule for Site Class D, E or F with
        # S1 >= 0.2 (its 11.4.8) is not in this edition, whose geodatabase spectra carry the site effect.
        site_specific = policy["site_class"] in SITE_RESPONSE_ANALYSIS_CLASSES
        checks.append(make_check("demands.site_hazard", "ASCE 7-22 11.4.2, 11.4.7, 11.6, Tables 11.6-1/11.6-2, Table 20.2-1",
                                 int(label_sdc == sdc and not site_specific), 1, "==",
                                 details={"derived_sdc": sdc, "site_label": label, "label_sdc": label_sdc,
                                          "site_class": policy.get("site_class"), "risk_category": policy.get("risk_category"),
                                          "site_specific_ground_motion_required": site_specific,
                                          "basis": "SDS/SD1/S1 are declared design values for the declared site class; ASCE 7-22 "
                                                   "11.4.7 requires a site response analysis for Site Class F only (20.3.1 "
                                                   "exemption aside); the edition has no 11.4.8 and no S1 rule for Site Class D or E"}))
    # Risk basis (2026-10-01): the category, Ie and the drift criterion saved with the design must be
    # one consistent set. A record without them predates the explicit basis: it cannot be shown to
    # be a design of its declared category, and it is redesigned, never relabelled.
    checks.append(risk_basis_check(policy, seismic, basis.get("drift") or {},
                                   ((record.get("request_identity") or {}).get("inputs") or {})))
    # Analysis procedure permission (edition-explicit). The torsional
    # classification and its consequences are separate items below: 7-22
    # 12.6 permits ELF for any structure, so an irregularity no longer
    # decides the procedure, only what the procedure must include.
    regularity = basis.get("regularity") or {}
    torsion = basis.get("torsion") or {}
    edition = basis.get("code_edition") or CODE_EDITION
    height_ft = geometry.get("num_floor", 0) * geometry.get("story_h_in", 0.0) / 12.0
    # The torsion items are derived from the validated per-row primitives
    # only: stored scalars and a stored classification are compared with the
    # recomputation and any disagreement, missing row, missing case or
    # missing strength evidence leaves the item unevaluated.
    policy_ratio = policy.get("accidental_torsion_ratio") if declared else None
    assessment = validate_torsion_assessment(torsion, geometry.get("num_floor"), policy_ratio)
    families = strength_families
    # A grouped design carries story-by-story line evidence, which needs no beam families (its line
    # strengths are recomputed from the installed members by Design.Grouped_Record).
    strength_block = regularity.get("lateral_strength_distribution")
    story_form = isinstance(strength_block, dict) and isinstance(strength_block.get("stories"), list)
    if story_form:
        families = {}
    if families is None:
        try:
            from Design.SMRF_Beam_Slab_Strength import beam_slab_strengths
            families = beam_slab_strengths(record)[1]
        except Exception as exc:                        # noqa: BLE001 -- unknown is the honest answer
            families = None
            family_problem = f"beam families cannot be rebuilt from the record ({type(exc).__name__}: {exc})"
    if families is None:
        strength_fraction, strength_problem = None, family_problem
    elif story_form:
        strength_fraction, strength_problem = strength_distribution_fraction(regularity, geometry)
    else:
        strength_fraction, strength_problem = strength_distribution_fraction(regularity, geometry, families)
    # Scientific applicability of the story-strength model (review item M1):
    # the evidence block states a status, and a "verified" status must be
    # backed by the record's own story_strength_model_verified assertion with
    # author, date and basis. A claim without its assertion, or an assertion
    # the evidence was not priced under, is stale evidence: unknown.
    verification = ((record.get("request_identity") or {}).get("policy") or {}).get("verification") \
        or basis.get("verification") or {}
    asserted_verified = assertion_provenance_valid(verification) and verification.get("story_strength_model_verified") is True
    block_status = (((regularity.get("lateral_strength_distribution") or {}).get("applicability") or {}).get("status")
                    if isinstance(regularity.get("lateral_strength_distribution"), dict) else None)
    if strength_fraction is not None:
        if block_status == "verified" and not asserted_verified:
            strength_fraction, strength_problem = None, ("the evidence claims a verified strength model but the record carries "
                                                         "no story_strength_model_verified assertion with author, date and basis")
        elif block_status == "provisional" and asserted_verified:
            strength_fraction, strength_problem = None, ("the record asserts story_strength_model_verified but the evidence was "
                                                         "priced as provisional: stale evidence")
    model_verified = strength_fraction is not None and block_status == "verified" and asserted_verified
    classification, torsion_problem = None, assessment.get("reason")
    if assessment["valid"]:
        classification = classify_torsional_irregularity(assessment["tir"], strength_fraction,
                                                         strength_model_verified=model_verified)
        stored = torsion.get("classification")
        if isinstance(stored, dict) and (
                stored.get("type_1") != classification["type_1"] or stored.get("label") != classification["label"]
                or stored.get("by_strength_distribution") != classification["by_strength_distribution"]
                or not _finite_close(stored.get("tir"), classification["tir"])):
            torsion_problem = ("the stored classification does not reproduce from the validated rows and the saved "
                               "strength-distribution evidence (stale or inconsistent evidence): stored label "
                               f"{stored.get('label')!r}, recomputed {classification['label']!r}")
            classification = None
    torsion_label = classification["label"] if classification else "not evaluated"
    if sdc is None or not regularity or not declared:
        checks.append(declaration_missing("elf_eligibility", "ASCE 7-22 12.6", "ELF procedure permission"))
    elif edition not in SUPPORTED_EDITIONS:
        checks.append(not_evaluated("demands.elf_eligibility", "ASCE 7-22 12.6",
                                    f"Unknown code edition {edition!r} in the saved demand basis."))
    else:
        # Regularity is known only with a classification; unknown counts as
        # irregular for the superseded 7-16 verdict (7-22 permits ELF either way).
        regular = regularity.get("regular") is True and classification is not None and not classification["type_1"]
        permitted, reason = elf_eligibility(sdc, height_ft, regular, basis.get("design_period_sec", 0.0),
                                            basis.get("ts_sec", 0.0), edition=edition)
        superseded = elf_eligibility(sdc, height_ft, regular, basis.get("design_period_sec", 0.0),
                                     basis.get("ts_sec", 0.0), edition="ASCE 7-16")
        checks.append(make_check("demands.elf_eligibility", f"{edition} 12.6 (analysis procedure selection)",
                                 int(permitted), 1, "==",
                                 details={"code_edition": edition, "reason": reason, "height_ft": height_ft,
                                          "regular": regular, "regularity": regularity,
                                          "torsional_irregularity": torsion_label,
                                          "period_cap": basis.get("period_basis"),
                                          "asce_7_16_table_12_6_1": {"permitted": superseded[0], "reason": superseded[1],
                                                                     "status": "superseded rule, recorded for traceability only"},
                                          "scope": "procedure permission only; system limitations (12.2), torsion, "
                                                   "drift and redundancy are their own items"}))
    # Torsional irregularity classification, its amplification consequence, and the TIR ceiling.
    ti_clause = "ASCE 7-22 Table 12.3-1 Type 1; 12.3.2.1.1; 12.8.4.3"
    at_clause = "ASCE 7-22 12.8.4.2; project policy TIR <= 1.4 (no ASCE 7-22 prohibition in any SDC)"
    if torsion_problem is not None:
        checks.append(not_evaluated("demands.torsional_irregularity", ti_clause, torsion_problem))
        checks.append(not_evaluated("demands.accidental_torsion", at_clause, torsion_problem))
    else:
        if classification["label"] == "unresolved":
            # TIR at or below 1.2 and the strength branch cannot certify absence:
            # either no fraction, or a fraction from a provisional model. Type 1
            # is neither established nor excluded; the item stays open, the
            # TIR ceiling below still evaluates.
            if not classification["strength_criterion_evaluated"]:
                reason = ("the strength-distribution criterion of Table 12.3-1 Type 1 is not evaluated "
                          f"({strength_problem}); a TIR at or below 1.2 does not establish the absence of Type 1")
            else:
                reason = (f"TIR {classification['tir']:.4f} <= 1.2 does not establish Type 1 and the strength-distribution "
                          f"criterion is unresolved: the one-sided fraction {classification['one_side_strength_fraction']:.4f} "
                          f"comes from a provisional story-strength model (review item {STRENGTH_MODEL_REVIEW_ITEM}, not "
                          "asserted as story_strength_model_verified); a provisional model cannot certify the absence "
                          "of the strength irregularity")
            check = not_evaluated("demands.torsional_irregularity", ti_clause, reason)
            check["details"].update({"classification": classification, "sdc": sdc,
                                     "strength_criterion_status": classification["strength_criterion_status"],
                                     "ax_by_level": assessment["ax_by_level"],
                                     "ax_required_envelope_if_type_1": assessment["ax_required_envelope"],
                                     "ax_applied": torsion.get("amplification"),
                                     "strength_distribution": regularity.get("lateral_strength_distribution")})
            checks.append(check)
        else:
            required = assessment["ax_required_envelope"] if classification["type_1"] else 1.0
            applied = torsion.get("amplification")
            stored_required = torsion.get("amplification_required")
            if stored_required is not None and not _finite_close(stored_required, required):
                checks.append(not_evaluated("demands.torsional_irregularity", ti_clause,
                                            f"stored amplification_required {stored_required!r} does not reproduce the "
                                            f"12.8.4.3 envelope {required} recomputed from the level displacements"))
            elif isinstance(applied, bool) or not isinstance(applied, (int, float)) or not math.isfinite(applied):
                checks.append(not_evaluated("demands.torsional_irregularity", ti_clause,
                                            "the Ax applied to the strength combinations and drift runs is not recorded"))
            else:
                # An Ax applied from the same level displacements reproduces the
                # requirement to rounding; equal within 1e-9 is equal, not a shortfall.
                applied = required if _finite_close(applied, required) else float(applied)
                checks.append(make_check("demands.torsional_irregularity", ti_clause, applied, required, ">=", "Ax",
                                         details={**classification, "sdc": sdc,
                                                  "strength_branch_note": (
                                                      "Type 1 established by the TIR; the strength-distribution criterion is "
                                                      f"{classification['strength_criterion_status']} and did not decide the outcome"
                                                      if classification["by_tir"] and not classification["strength_criterion_resolved"]
                                                      else "strength-distribution criterion "
                                                           f"{classification['strength_criterion_status']}"),
                                                  "ax_by_level": assessment["ax_by_level"],
                                                  "ax_required_envelope": assessment["ax_required_envelope"],
                                                  "ax_applied_at_every_level": float(applied),
                                                  "assessment_cases": assessment["cases"],
                                                  "rows_validated": assessment["rows"],
                                                  "assessment_basis": torsion.get("basis"),
                                                  "strength_distribution": regularity.get("lateral_strength_distribution"),
                                                  "basis": ("TIR per 12.3.2.1.1 from the edge story drifts of the ELF runs with 5% "
                                                            "accidental torsion and Ax = 1, every story, direction and eccentricity "
                                                            "sign; Ax per 12.8.4.3 from the edge level displacements of the same runs, "
                                                            "per level, case and direction; the single Ax applied to every level of "
                                                            "every seismic strength combination and drift run must cover the largest "
                                                            "per-level value (conservative envelope), and 1.0 where Type 1 is absent")}))
        limit, kind, limit_basis = torsional_irregularity_limit(sdc)
        checks.append(make_check("demands.accidental_torsion", at_clause, assessment["tir"], limit, "<=",
                                 "TIR = delta_max/delta_avg (story drifts)",
                                 details={"tir_by_case": assessment["tir_by_case"], "cases": assessment["cases"],
                                          "rows_validated": assessment["rows"],
                                          "eccentricity_ratio": assessment["eccentricity_ratio"],
                                          "sign_symmetry_residual_max": torsion.get("sign_symmetry_residual_max"),
                                          "sdc": sdc, "limit_kind": kind, "limit_basis": limit_basis,
                                          "assessment_basis": torsion.get("basis"),
                                          "basis": "5% eccentricity at every level in every seismic combination, amplified by "
                                                   "Ax; TIR recomputed from the validated rows of the drift runs with torsion "
                                                   "applied and Ax = 1"}))
    # Redundancy factor.
    rho_used = _redundancy_factor_used(record, basis)
    if sdc is None or rho_used is None:
        checks.append(not_evaluated("demands.redundancy", "ASCE 7-22 12.3.4",
                                    "The SDC or the rho carried by the strength combinations is unknown."))
    else:
        requirement = redundancy_requirement(sdc)
        checks.append(make_check("demands.redundancy", "ASCE 7-22 12.3.4.1 / 12.3.4.2", rho_used, requirement["required"],
                                 ">=", "rho",
                                 details={**requirement, "sdc": sdc, "rho_strength_combinations": rho_used,
                                          "rho_drift_forces": 1.0,
                                          "tir_exceeds_1_4": classification["tir_exceeds_1_4"] if classification else None,
                                          "note_12_3_4_2_1": ("12.3.4.2.1 requires rho = 1.3 in both directions where TIR > 1.4 "
                                                              "in both directions (SDC D-F); rho = 1.3 is carried in every "
                                                              "strength combination there, and such designs fail the project "
                                                              "TIR ceiling above"),
                                          "basis": "rho = 1.3 in every seismic strength combination without claiming 12.3.4.2 condition 1 or 2; "
                                                   "drift forces at rho = 1.0 (12.3.4.1) with the D-F drift limit divided by rho"}))
    # Load scope.
    if not declared:
        checks.append(declaration_missing("load_scope", "ASCE 7-22 Chapter 2", "Applicable loads are a declaration"))
    else:
        roof_ok = loads.get("floor_live_load_ksf", 0.0) >= policy.get("roof_live_load_ksf", 0.0) + policy.get("snow_load_ksf", 0.0)
        checks.append(make_check("demands.load_scope", "ASCE 7-22 Chapter 2; 4.8; 7; 8; 26",
                                 int(roof_ok and not policy.get("wind_governs", False) and policy.get("rain_ponding_excluded", False)), 1, "==",
                                 details={"roof_live_load_ksf": policy.get("roof_live_load_ksf"),
                                          "floor_live_applied_to_roof_ksf": loads.get("floor_live_load_ksf"),
                                          "snow_load_ksf": policy.get("snow_load_ksf"), "wind_governs": policy.get("wind_governs"),
                                          "rain_ponding_excluded": policy.get("rain_ponding_excluded"),
                                          "declared_by": policy.get("declared_by"), "declaration_basis": policy.get("declaration_basis")}))
    # Live-load patterning.
    patterns = basis.get("live_load_patterns")
    if patterns is None:
        checks.append(not_evaluated("demands.live_load_patterning", "ACI 318-19 6.4.2",
                                    "No live-load arrangements were analyzed."))
    else:
        checks.append(make_check("demands.live_load_patterning", "ACI 318-19 6.4.2(a)-(c)",
                                 int(len(patterns) > 0 and basis.get("patterns_in_strength_envelope") is True), 1, "==",
                                 details={"arrangements": [p["id"] for p in patterns],
                                          "basis": "alternate spans and two adjacent spans about every interior support, "
                                                   "as slab panel patterns through the floor transfer, in the 1.2D+1.6L family"}))
    # Effective seismic weight.
    if not declared:
        checks.append(declaration_missing("effective_seismic_weight", "ASCE 7-22 12.7.2", "The weight inventory needs the occupancy declaration"))
    else:
        partitions = policy.get("partition_allowance_ksf", 0.0)
        sdl = loads.get("floor_superimposed_dead_load_ksf") or 0.0
        storage = policy.get("storage_live_fraction_in_weight", 0.0)
        live_fraction = loads.get("seismic_live_load_fraction", 0.0)
        office = policy.get("occupancy", "") == "office"
        ok = (partitions >= 0.010 and partitions <= sdl and (storage == 0.0 if office else storage >= 0.25)
              and live_fraction >= storage and policy.get("snow_load_ksf", 0.0) == 0.0)
        checks.append(make_check("demands.effective_seismic_weight", "ASCE 7-22 12.7.2", int(ok), 1, "==",
                                 details={"inventory": "slab self-weight + SDL (finishes/MEP/partitions) + beam drops + columns",
                                          "partition_allowance_ksf": partitions, "superimposed_dead_load_ksf": sdl,
                                          "storage_live_fraction": storage, "seismic_live_load_fraction": live_fraction,
                                          "occupancy": policy.get("occupancy"),
                                          "total_floor_seismic_weight_kip": loads.get("total_floor_seismic_weight_kip")}))
    # Drift analysis basis.
    drift = basis.get("drift") or {}
    if not drift:
        checks.append(not_evaluated("demands.drift_analysis_basis", "ASCE 7-22 12.8.6; ACI 318-19 6.6.3",
                                    "No drift basis was saved with this design."))
    else:
        ok = (abs(drift.get("beam_stiffness_modifier", 0) - 0.35) < 1e-9 and abs(drift.get("column_stiffness_modifier", 0) - 0.70) < 1e-9
              and drift.get("cd") == 5.5 and drift.get("rho_for_drift_load") == 1.0 and drift.get("second_order_included") is True)
        checks.append(make_check("demands.drift_analysis_basis", "ASCE 7-22 12.8.6, 12.8.7; ACI 318-19 Table 6.6.3.1.1(a)",
                                 int(ok), 1, "==", details={**drift, "basis": "elastic cracked stiffness 0.35 Ig beams / 0.70 Ig columns, "
                                                            "rho = 1 drift forces at the Cu Ta-capped period (conservative for drift), Cd = 5.5, "
                                                            "P-Delta stability coefficient, full D+L gravity state"}))
    return checks


def demand_scope_checks():
    """Kept for callers that have no design record: every item not_evaluated."""
    return evaluate_demand_basis({})


def story_drift_basis(risk_category, importance_factor, drift_limit_ratio=None):
    """Category, Ie and the Table 12.12-1 base limit as one consistent set; a contradiction raises.

    The importance factor must be the one Table 1.5-2 gives for the category, and an explicit
    limit may be stricter than the category's row but never more permissive (no low-rise
    allowance in this scope). Pure: the tables are read, no module state is consulted.
    """
    if not isinstance(risk_category, str) or risk_category not in RISK_CATEGORIES:
        raise ValueError(f"risk_category {risk_category!r} is not exactly one of {RISK_CATEGORIES}.")
    ie = _number(importance_factor, "importance_factor", positive=True)
    table_ie = IMPORTANCE_FACTOR_BY_RISK_CATEGORY[risk_category]
    if abs(ie - table_ie) > 1e-12:
        raise ValueError(f"importance_factor {ie:g} contradicts Risk Category {risk_category} "
                         f"(Ie = {table_ie:g}, ASCE 7-22 Table 1.5-2); category and factor come from one basis.")
    base = ALLOWABLE_STORY_DRIFT_RATIO_BY_RISK_CATEGORY[risk_category]
    limit = base if drift_limit_ratio is None else _number(drift_limit_ratio, "drift_limit_ratio", positive=True)
    if limit > base + 1e-15:
        raise ValueError(f"drift_limit_ratio {limit:g} exceeds the Risk Category {risk_category} allowable {base:g} "
                         "(ASCE 7-22 Table 12.12-1, all other structures; the low-rise allowance is not asserted).")
    return {"risk_category": risk_category, "importance_factor": table_ie, "table_limit": base, "limit": limit}


def evaluate_drift_and_stability(stories, cd=5.5, importance_factor=1.0,
                                 drift_limit_ratio=None, redundancy_factor=1.3,
                                 seismic_design_category="D", beta=1.0,
                                 overstrength_factor=3.0,
                                 second_order_included=False,
                                 analysis_succeeded=True, risk_category="II"):
    """Evaluate node-envelope drift and story stability from elastic results.

    Each story has ``id``, ``height_in``, ``node_deltas_in`` mapping grid IDs to
    ``{'x': dx, 'y': dy}``, ``expected_node_ids`` listing *all* structural floor
    grid IDs, ``story_shear_kip`` mapping directions to shears, and
    ``gravity_above_kip`` (P at and above that story, not just that floor).
    ``directions`` may be supplied to evaluate a unidirectional analysis; it
    defaults to ('x', 'y'). Input deltas come from QEx/QEy drift analyses with
    rho=1, not factored strength combinations or nonlinear time histories.

    Cd/Ie amplifies elastic drift. The base limit is the Table 12.12-1 "all
    other structures" row of ``risk_category`` (0.020h for I and II, 0.015h for
    III, 0.010h for IV; no low-rise allowance); solely moment-frame SDC D/E/F
    limits are divided by rho (12.12.1.1). ``importance_factor`` must be the
    Table 1.5-2 value of that category and an explicit ``drift_limit_ratio``
    may only be stricter than the row: a contradictory set raises
    (story_drift_basis) instead of evaluating against a stale default. The
    defaults are the self-consistent Risk II set; the design driver passes the
    configured basis explicitly.
    theta=P*delta_elastic/(V*h); 7-22 permits theta_max at least 0.10. The
    optional theta/(1+theta) correction applies only to analysis that already
    includes P-Delta. A first-order theta>0.10 requires a subsequent design
    amplification/reanalysis, recorded as not_evaluated here, not silently
    applied to an unrelated member-demand envelope.
    """
    cd = _number(cd, "cd", positive=True)
    drift_basis = story_drift_basis(risk_category, importance_factor, drift_limit_ratio)
    ie, limit, base_limit = drift_basis["importance_factor"], drift_basis["limit"], drift_basis["limit"]
    rho = _number(redundancy_factor, "redundancy_factor", minimum=1)
    beta = _number(beta, "beta", positive=True)
    omega = _number(overstrength_factor, "overstrength_factor", positive=True)
    if beta < 1.25 / omega:
        raise ValueError("ASCE 7-22 beta cannot be less than 1.25/Omega0.")
    sdc = str(seismic_design_category).upper()
    if sdc not in ("A", "B", "C", "D", "E", "F"):
        raise ValueError("seismic_design_category must be A through F.")
    limit /= rho if sdc in ("D", "E", "F") else 1.0
    theta_max = max(0.10, min(0.5 / (beta * cd), 0.25))
    checks, rows = [], []
    stories = list(stories)
    if not analysis_succeeded:
        checks.append(not_evaluated(
            "demands.drift_analysis", "ASCE 7-22 12.8.6-12.8.7",
            "Analysis did not complete; drift and stability are unknown. A solver failure alone does not prove structural instability."))
        return {"checks": checks, "stories": rows, **summarize_checks(checks)}
    if not stories:
        checks.append(not_evaluated("demands.story_results", "ASCE 7-22 12.8.6",
                                    "No story results were supplied."))
    for index, story in enumerate(stories, 1):
        location = f"story:{story.get('id', index)}"
        directions = tuple(story.get("directions", ("x", "y")))
        if not directions or any(d not in ("x", "y") for d in directions) or len(set(directions)) != len(directions):
            raise ValueError("Each story directions must be a nonempty unique subset of x/y.")
        deltas = story.get("node_deltas_in", {})
        expected = story.get("expected_node_ids")
        if not isinstance(deltas, Mapping):
            deltas = {}
        if expected is None:
            checks.append(not_evaluated("demands.node_coverage", "ASCE 7-22 12.8.6.5",
                                        "Expected grid-node IDs are missing; edge coverage cannot be confirmed.", location))
        else:
            expected = list(expected)
            matched = bool(expected) and len(expected) == len(set(expected)) and set(deltas) == set(expected)
            checks.append(make_check("demands.node_coverage", "ASCE 7-22 12.8.6.5",
                                     int(matched), 1, comparison="==", location=location,
                                     details={"expected_nodes": len(expected), "supplied_nodes": len(deltas)}))
        for direction in directions:
            loc = f"{location}/{direction}"
            try:
                height = _number(story.get("height_in"), "height_in", positive=True)
                if not deltas:
                    raise ValueError("No structural grid-node drifts were supplied.")
                values = {str(node): abs(_number(value[direction], "node delta"))
                          for node, value in deltas.items()}
                governing_node = max(values, key=values.get)
                elastic_delta = values[governing_node]
            except (ValueError, KeyError, TypeError) as exc:
                for key in ("story_drift", "stability"):
                    checks.append(not_evaluated(f"demands.{key}", "ASCE 7-22 12.8.6-12.8.7", str(exc), loc))
                continue
            amplified = elastic_delta * cd / ie
            ratio = amplified / height
            row = {"location": loc, "governing_node": governing_node,
                   "elastic_drift_in": elastic_delta, "design_drift_in": amplified,
                   "drift_ratio": ratio, "allowable_drift_ratio": limit,
                   "theta": None, "theta_limit": theta_max,
                   "required_second_order_amplification": None}
            checks.append(make_check("demands.story_drift", "ASCE 7-22 12.8.6; 12.12.1.1; Table 12.12-1",
                                     ratio, limit, location=loc, units="in/in",
                                     details={"governing_node": governing_node, "cd": cd, "importance_factor": ie}))
            try:
                p = _number(story.get("gravity_above_kip"), "gravity_above_kip", minimum=0)
                shear = abs(_number(story.get("story_shear_kip", {}).get(direction), "story_shear_kip"))
                if shear <= 0:
                    raise ValueError("Nonzero corresponding story shear is required to evaluate stability.")
                theta_raw = p * elastic_delta / (shear * height)
                theta = theta_raw / (1 + theta_raw) if second_order_included else theta_raw
                row.update(theta=theta, theta_raw=theta_raw)
                checks.append(make_check("demands.stability", "ASCE 7-22 12.8.7 Eqs. 12.8-18, 12.8-19",
                                         theta, theta_max, units="dimensionless", location=loc,
                                         details={"second_order_included": second_order_included, "beta": beta}))
                needs_amplification = theta > 0.10 and not second_order_included
                row["required_second_order_amplification"] = 1 / (1 - theta) if needs_amplification and theta < 1 else 1.0 if not needs_amplification else None
                if needs_amplification:
                    checks.append(not_evaluated("demands.second_order_effects", "ASCE 7-22 12.8.7",
                                                "First-order theta exceeds 0.10; displacement AND member-demand amplification/reanalysis has not been performed.", loc))
            except (ValueError, TypeError, AttributeError) as exc:
                checks.append(not_evaluated("demands.stability", "ASCE 7-22 12.8.7", str(exc), loc))
            rows.append(row)
    return {"checks": checks, "stories": rows,
            "assumptions": {"risk_category": drift_basis["risk_category"], "system": "solely_moment_frames",
                            "sdc": sdc, "rho": rho, "cd": cd, "importance_factor": ie,
                            "base_drift_limit_ratio": base_limit,
                            "table_drift_limit_ratio": drift_basis["table_limit"],
                            "limit_divided_by_rho": sdc in ("D", "E", "F"),
                            "effective_drift_limit_ratio": limit,
                            "low_rise_allowance_asserted": False,
                            "drift_limit_source": "ASCE 7-22 Table 12.12-1 (all other structures); 12.12.1.1",
                            "drift_load_rho": 1.0, "theta_max_lower_bound": 0.10,
                            "not_complete_code_acceptance": True},
            **summarize_checks(checks)}
