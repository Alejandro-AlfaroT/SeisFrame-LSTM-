"""Hoop and crosstie arrangement for the column and beam cages (ACI 318-19).

The capacity design selects a hoop bar, a leg count and a spacing. This
module says what those legs are: a perimeter hoop (two legs each way) plus
crossties, each engaging a longitudinal bar on one face and the bar
opposite it. From the bar positions it decides which bars need support
and produces the arrangement, so the leg count the shear and confinement
quantities use is one the cage can actually contain, and the supported-bar
spacing those quantities depend on is the arrangement's, not an assumption.

Rules applied (all pure geometry; hook geometry and placement are not
drawn here):

* 25.7.2.3, invoked by 18.7.5.2(d) for columns and 18.6.4.4 for beams:
  every corner and alternate longitudinal bar has lateral support from a
  hoop corner or a crosstie, and no unsupported bar is farther than 6 in
  clear along the hoop from a supported bar.
* 18.7.5.2(e): hx, the spacing of laterally supported column bars around
  the perimeter, at most 14 in; 18.7.5.2(f): every bar supported and
  hx <= 8 in where Pu > 0.3 Ag f'c or f'c > 10 ksi.
* Crossties are single-leg ties, each engaging one interior bar, with a
  135-degree seismic hook at one end and a 90-degree hook at the other,
  alternated end for end (18.7.5.2(c)). The default is the code-minimal
  pattern above (user decision 2026-10-17: code-compliant alternating
  hooks); COLUMN_CROSSTIE_EVERY_BAR = True ties every interior column bar
  instead (the ACI example figure), and the legs across a face pair are
  then the bars on that face.
* 18.7.5.2(b): a crosstie engages a peripheral longitudinal bar, so a face
  with n bars can carry at most n - 2 crossties in the direction across
  it, and the legs across that direction number at most n.
* 18.6.4.4: beam bars nearest the tension and compression faces are
  supported per 25.7.2.3 with supported spacing at most 14 in.

A leg count is *constructible* for a direction when it lies between the
minimum the support rules require and the maximum the bars can engage.
The two directions of a column are independent: the legs that cross the b
faces (the perimeter hoop's two legs parallel to h plus the crossties on
the b-face bars) and the legs that cross the h faces are counted, tied and
used for Av and Ash separately, as 18.7.5.4 defines Ash per direction.
"""
from __future__ import annotations

import math

HX_MAX_IN = 14.0
HX_MAX_HIGH_AXIAL_IN = 8.0
# True ties every interior column bar (the ACI example figure); False, the default (user decision 2026-10-17), keeps the
# 25.7.2.3 minimal pattern with the hooks alternated end for end. Beams always use the minimal pattern.
COLUMN_CROSSTIE_EVERY_BAR = False
UNSUPPORTED_CLEAR_MAX_IN = 6.0
COLUMN_DIRECTIONS = ("across_b_face", "across_h_face")
_OMITTED = object()      # grid_in left out: the module's grid (Structure_Parameters.MODULE_GRID); None itself is "no grid"


def legs_by_direction(legs):
    """Column leg count as {direction: legs}; an int means the same count both ways."""
    if isinstance(legs, dict):
        missing = [d for d in COLUMN_DIRECTIONS if d not in legs]
        if missing:
            raise ValueError(f"Column legs need both directions; missing {missing}.")
        return {d: int(legs[d]) for d in COLUMN_DIRECTIONS}
    return {d: int(legs) for d in COLUMN_DIRECTIONS}


def face_bar_positions(width_in, clear_cover_in, hoop_db_in, bar_db_in, count, grid_in=_OMITTED):
    """Centre positions of ``count`` bars along one face, on the bar position grid.

    The corner bars sit in the hoop corners at the first grid step at or beyond
    clear cover + the layout hoop + half the bar (Structure_Parameters.first_bar_offset_in);
    the interior bars take grid pitches, equal where the equal pitch is a grid
    multiple and otherwise symmetric pitches one grid step apart
    (Structure_Parameters.face_bar_positions_in; design-basis rule 2026-10-14).
    ``grid_in``: the record's grid (None: even spacing of a record from before the
    rule); omitted, the module's grid.
    """
    import Structure_Parameters as sp
    grid = sp.MODULE_GRID if grid_in is _OMITTED else grid_in
    if count < 2:
        raise ValueError("A face needs at least two (corner) bars.")
    first = sp.first_bar_offset_in(clear_cover_in, hoop_db_in, bar_db_in, grid_in=grid)
    last = width_in - first
    if last <= first:
        raise ValueError("Face too narrow for the cover and bars.")
    return sp.face_bar_positions_in(width_in, first, int(count), grid_in=grid)


def support_pattern(positions, bar_db_in, hx_max_in=HX_MAX_IN, every_bar=False,
                    clear_max_in=UNSUPPORTED_CLEAR_MAX_IN):
    """Minimum set of supported bars on a face, corners included.

    Walks the face; an interior bar stays unsupported only when the bar
    before it is supported, the bar after it will be, its clear distance to
    both is within ``clear_max_in`` and the supported spacing across it is
    within ``hx_max_in``. Returns {"supported": [bool per bar], "hx_in",
    "max_unsupported_clear_in", "crossties"} for the minimal pattern.
    """
    n = len(positions)
    supported = [True] + [False] * (n - 2) + [True] if n > 2 else [True] * n
    if not every_bar:
        i = 1
        while i < n - 1:
            prev_supported = supported[i - 1]
            clear_prev = positions[i] - positions[i - 1] - bar_db_in
            clear_next = positions[i + 1] - positions[i] - bar_db_in
            span = positions[i + 1] - positions[i - 1]
            can_skip = (prev_supported and clear_prev <= clear_max_in and clear_next <= clear_max_in
                        and span <= hx_max_in)
            supported[i] = not can_skip
            i += 1
    else:
        supported = [True] * n
    return _describe(positions, bar_db_in, supported)


def arrangement_with_crossties(positions, bar_db_in, crossties, hx_max_in=HX_MAX_IN, every_bar=False):
    """Support pattern realizing exactly ``crossties`` interior supports.

    Starts from the minimal pattern the rules require and adds crossties to
    the unsupported bars with the largest clear distance to a supported
    neighbour, so every realized pattern still satisfies the rules the
    minimal one did. Fewer crossties than the minimum is not constructible.
    """
    n = len(positions)
    minimal = support_pattern(positions, bar_db_in, hx_max_in, every_bar=every_bar)
    supported = list(minimal["supported"])
    needed = crossties - (sum(supported) - 2)
    if needed < 0 or crossties > n - 2:
        raise ValueError("Crosstie count outside what the face supports.")
    while needed > 0:
        described = _describe(positions, bar_db_in, supported)
        candidates = [i for i in range(1, n - 1) if not supported[i]]
        if not candidates:
            raise ValueError("No unsupported bar left to tie.")
        # Largest clear distance to the nearest supported bar first, then leftmost.
        def gap(i):
            left = next((positions[i] - positions[j] - bar_db_in for j in range(i - 1, -1, -1) if supported[j]), math.inf)
            right = next((positions[j] - positions[i] - bar_db_in for j in range(i + 1, n) if supported[j]), math.inf)
            return min(left, right)
        supported[max(candidates, key=lambda i: (gap(i), -i))] = True
        needed -= 1
    return _describe(positions, bar_db_in, supported)


def _describe(positions, bar_db_in, supported):
    n = len(positions)
    supported_positions = [x for x, s in zip(positions, supported) if s]
    hx = max((b - a for a, b in zip(supported_positions, supported_positions[1:])), default=0.0)
    worst_clear = 0.0
    alternate_ok = True
    for i in range(n):
        if supported[i]:
            continue
        left = next((positions[i] - positions[j] - bar_db_in for j in range(i - 1, -1, -1) if supported[j]), math.inf)
        right = next((positions[j] - positions[i] - bar_db_in for j in range(i + 1, n) if supported[j]), math.inf)
        worst_clear = max(worst_clear, min(left, right))
        if i + 1 < n and not supported[i + 1]:
            alternate_ok = False
    return {"positions_in": positions, "supported": supported, "hx_in": hx,
            "max_unsupported_clear_in": worst_clear, "crossties": sum(supported) - 2,
            "alternate_bars_supported": alternate_ok}


def column_cage(b_in, h_in, clear_cover_in, hoop_db_in, bar_db_in, top_bars, side_bars, high_axial=False,
                legs=None, grid_in=_OMITTED):
    """Arrangement and constructible leg range for a rectangular column cage.

    Faces: two faces of width ``b_in`` with ``top_bars`` bars each (the
    legs across them run in the h direction), two faces of depth ``h_in``
    with ``side_bars`` interior bars each plus the corners. With ``legs``
    given -- an int for the same count both ways, or
    {"across_b_face": n, "across_h_face": m} -- the arrangement realizes
    that many legs in each direction (perimeter hoop + legs - 2 crossties
    on the face pair the legs cross) when constructible in both.
    """
    hx_max = HX_MAX_HIGH_AXIAL_IN if high_axial else HX_MAX_IN
    every_bar = bool(high_axial or COLUMN_CROSSTIE_EVERY_BAR)
    faces = {
        "b_face": face_bar_positions(b_in, clear_cover_in, hoop_db_in, bar_db_in, top_bars, grid_in=grid_in),
        "h_face": face_bar_positions(h_in, clear_cover_in, hoop_db_in, bar_db_in, side_bars + 2, grid_in=grid_in),
    }
    minimal = {name: support_pattern(pos, bar_db_in, hx_max, every_bar=every_bar) for name, pos in faces.items()}
    # Legs across the b faces run along h and count 2 + crossties on the b face; and vice versa.
    legs_min = {"across_b_face": 2 + minimal["b_face"]["crossties"], "across_h_face": 2 + minimal["h_face"]["crossties"]}
    legs_max = {"across_b_face": top_bars, "across_h_face": side_bars + 2}
    result = {"faces": faces, "minimal_support": minimal, "legs_min": legs_min, "legs_max": legs_max,
              "hx_max_in": hx_max, "high_axial": high_axial, "every_bar_supported": every_bar,
              "support_basis": ("a crosstie on every interior bar, 135/90-degree hooks alternated end for end (18.7.5.2(c); "
                                "user decision 2026-10-17)" if every_bar else
                                "25.7.2.3 minimal pattern: corners and alternate bars, 6 in clear, hx per 18.7.5.2(e)"),
              # Per direction, every count from the minimum the support rules
              # need to the bars the face can engage is constructible; the
              # common list is what a single count both ways could use.
              "constructible_legs_by_direction": {d: list(range(legs_min[d], legs_max[d] + 1)) for d in COLUMN_DIRECTIONS},
              "constructible_legs": [n for n in range(2, max(legs_max.values()) + 1)
                                     if all(legs_min[d] <= n <= legs_max[d] for d in legs_min)]}
    if legs is not None:
        legs = legs_by_direction(legs)
        feasible = all(legs_min[d] <= legs[d] <= legs_max[d] for d in COLUMN_DIRECTIONS)
        result["legs"] = legs
        result["constructible"] = feasible
        if feasible:
            # Legs across the b faces are the crossties on the b-face bars (plus the hoop's two).
            crossties = {"b_face": legs["across_b_face"] - 2, "h_face": legs["across_h_face"] - 2}
            realized = {name: arrangement_with_crossties(pos, bar_db_in, crossties[name], hx_max, every_bar=every_bar)
                        for name, pos in faces.items()}
            result["arrangement"] = realized
            result["hx_in"] = max(r["hx_in"] for r in realized.values())
            result["checks"] = _cage_checks(realized, hx_max, "column")
        else:
            result["arrangement"] = None
            result["hx_in"] = None
            result["checks"] = [{"rule": "18.7.5.2(b)/(d)", "passes": False,
                                 "detail": f"{legs} legs are not constructible: bars allow {legs_max}, support rules need {legs_min}"}]
    return result


def beam_cage(b_in, clear_cover_in, hoop_db_in, bar_db_in, top_bars, bot_bars, legs=None, grid_in=_OMITTED):
    """Arrangement and constructible leg range for a beam hoop (18.6.4.4 / 25.7.2.3)."""
    faces = {"top": face_bar_positions(b_in, clear_cover_in, hoop_db_in, bar_db_in, top_bars, grid_in=grid_in),
             "bottom": face_bar_positions(b_in, clear_cover_in, hoop_db_in, bar_db_in, bot_bars, grid_in=grid_in)}
    minimal = {name: support_pattern(pos, bar_db_in, HX_MAX_IN) for name, pos in faces.items()}
    # A crosstie engages a top bar and a bottom bar; legs across the beam width.
    legs_min = 2 + max(m["crossties"] for m in minimal.values())
    legs_max = min(top_bars, bot_bars)
    result = {"faces": faces, "minimal_support": minimal, "legs_min": legs_min, "legs_max": legs_max,
              "hx_max_in": HX_MAX_IN,
              "constructible_legs": [n for n in range(2, legs_max + 1) if n >= legs_min]}
    if legs is not None:
        feasible = legs_min <= legs <= legs_max
        result["legs"] = legs
        result["constructible"] = feasible
        if feasible:
            realized = {name: arrangement_with_crossties(pos, bar_db_in, legs - 2) for name, pos in faces.items()}
            result["arrangement"] = realized
            result["hx_in"] = max(r["hx_in"] for r in realized.values())
            result["checks"] = _cage_checks(realized, HX_MAX_IN, "beam")
        else:
            result["arrangement"] = None
            result["hx_in"] = None
            result["checks"] = [{"rule": "18.6.4.4 / 25.7.2.3", "passes": False,
                                 "detail": f"{legs} legs are not constructible: bars allow {legs_max}, support rules need {legs_min}"}]
    return result


def _cage_checks(realized, hx_max, member):
    hx_rule = "18.7.5.2(e)/(f)" if member == "column" else "18.6.4.4"
    checks = []
    for name, face in realized.items():
        checks.append({"rule": "25.7.2.3(a) alternate bars supported", "face": name,
                       "passes": face["alternate_bars_supported"]})
        checks.append({"rule": "25.7.2.3(b) unsupported bar within 6 in clear", "face": name,
                       "value_in": face["max_unsupported_clear_in"], "limit_in": UNSUPPORTED_CLEAR_MAX_IN,
                       "passes": face["max_unsupported_clear_in"] <= UNSUPPORTED_CLEAR_MAX_IN + 1e-9})
        checks.append({"rule": f"{hx_rule} supported-bar spacing", "face": name,
                       "value_in": face["hx_in"], "limit_in": hx_max, "passes": face["hx_in"] <= hx_max + 1e-9})
    checks.append({"rule": "18.7.5.2(b)/(c) crossties engage bars, alternate end for end" if member == "column"
                   else "25.7.2.3 crossties engage bars", "passes": True,
                   "detail": "crosstie count never exceeds the interior bars it engages; 135-degree hooks and end alternation are "
                             "fabrication requirements stated with the arrangement, not drawn"})
    return checks


def cage_passes(cage):
    return bool(cage.get("constructible")) and all(c["passes"] for c in cage.get("checks", []))
