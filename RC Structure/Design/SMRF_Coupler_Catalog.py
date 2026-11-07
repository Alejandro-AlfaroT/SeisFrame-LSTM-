"""Pinned nVent LENTON A2 envelopes; IAPMO ER-0129, revised 2025-02-21.

Only published size/grade combinations are represented. A missing pair is
unknown evidence, never an invented product or a failed structural geometry.
Hexagonal stock is bounded by its circumscribed circle, not across-flats size.
"""
from __future__ import annotations

import math

CATALOG_ID = "nvent_lenton_A2_ER0129_20250221"
SOURCE_URL = "https://www.nvent.com/sites/default/files/dam/iapmo-ues-er-0129-en.pdf"
SOURCE_SHA256 = "4d817003d1d08f2282d8187c45c427350a68ab7e707efe8343a247e36e29f4fe"
# Table 2: part, across-flats or round diameter, body length, bar engagement, hex.
_SAME = {
    4: ("EL12A2", 11/16, 1+5/8, 9/16, True),
    5: ("EL16A2", 7/8, 2+3/16, 7/8, True),
    6: ("EL20A2", 1+1/16, 2+13/16, 1+1/8, True),
    7: ("EL22A2", 1+3/16, 3+5/32, 1+1/4, True),
    8: ("EL25A2", 1+3/8, 3+11/32, 1+3/8, False),
    9: ("EL28A2", 1+1/2, 3+19/32, 1+1/2, False),
    10: ("EL32A2", 1+3/4, 3+25/32, 1+9/16, False),
    11: ("EL36A2", 1+7/8, 3+31/32, 1+11/16, False),
    14: ("EL43TA2", 2+1/4, 5+1/4, 2+1/8, False),
    18: ("EL57TA2", 3., 6+1/2, 2+3/4, False),
}
# Table 4: part, diameter, length, larger/smaller engagement, hex.
_TRANSITION = {
    (4, 5): ("EL1612A2", 7/8, 2+5/16, 7/8, 9/16, True),
    (5, 6): ("EL2016A2", 1+1/16, 3., 1+1/8, 7/8, True),
    (6, 7): ("EL2220A2", 1+3/16, 3+13/32, 1+1/4, 1+1/8, True),
    (7, 8): ("EL2522A2", 1+3/8, 3+11/16, 1+3/8, 1+1/4, False),
    (8, 9): ("EL2825A2", 1+1/2, 3+29/32, 1+1/2, 1+3/8, False),
    (9, 10): ("EL3228A2", 1+3/4, 4+1/8, 1+9/16, 1+1/2, False),
    (10, 11): ("EL3632A2", 1+7/8, 4+5/16, 1+11/16, 1+9/16, False),
    (11, 14): ("EL43T36A2", 2+1/4, 4+21/32, 2+1/8, 1+11/16, False),
    (11, 18): ("EL57T36A2", 3., 5+11/16, 2+3/4, 1+11/16, False),
    (14, 18): ("EL57T43TA2", 3., 6+5/16, 2+3/4, 2+1/8, False),
}


def lookup_coupler(lower_size, upper_size, fy_ksi, specification, protected_region=False):
    """Return explicit qualification/envelope evidence for a Type 2 splice."""
    a, b = sorted((int(lower_size), int(upper_size)))
    out = {"catalog_id": CATALOG_ID, "source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256,
           "report_revision": "2025-02-21", "report_valid_through": "2027-02-28",
           "status": "not_evaluated", "type": 2, "lower_bar_size": int(lower_size),
           "upper_bar_size": int(upper_size), "protected_region": bool(protected_region)}
    if float(fy_ksi) not in (60., 80.) or specification != f"ASTM A706 Grade {int(fy_ksi)}":
        return {**out, "reason": "No pinned A2 qualification for this exact bar specification and grade."}
    if protected_region and float(fy_ksi) != 60.:
        return {**out, "status": "fail", "reason": "ER-0129 section 2.7 restricts protected/yield-region Type 2 splices to Grade 60."}
    if a == b and a in _SAME:
        part, diameter, length, engagement, hexagonal = _SAME[a]
        large, small, table = engagement, engagement, 2
    elif (a, b) in _TRANSITION:
        part, diameter, length, large, small, hexagonal = _TRANSITION[a, b]
        table = 4
    else:
        return {**out, "reason": f"No published A2 product for No. {a} to No. {b} in pinned Tables 2/4."}
    return {**out, "status": "pass", "reason": "Type 2 under ER-0129 Table 1; dimensional envelope from the stated table.",
            "part_number": part, "dimension_table": table, "stock_dimension_in": diameter,
            "stock_shape": "hexagonal_across_flats" if hexagonal else "round",
            "envelope_diameter_in": diameter / math.cos(math.pi/6) if hexagonal else diameter,
            "length_in": length, "large_bar_engagement_in": large, "small_bar_engagement_in": small,
            "installation": "At least one bar free to rotate; approved LENTON thread fabrication and manufacturer installation per ER-0129 sections 3.3--3.5.",
            "cover_basis": "Concrete cover measured from outside splice device (ER-0129 3.2)."}
