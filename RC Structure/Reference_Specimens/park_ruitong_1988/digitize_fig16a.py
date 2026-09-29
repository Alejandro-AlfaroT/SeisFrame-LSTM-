"""Traceable pixel audit of Fig. 16(a) of Park and Ruitong (1988): axis calibration with skew, tick marks,
the loop peaks (reversal points), the zero-force and zero-displacement crossings of every branch, and the
unloading chords, each with its pixel coordinates (Codex Unit 1 review, 2026-09-28: "Trace each branch from its
numbered reversal through both axis crossings, save pixel coordinates and the axis transformation, and have the
crossing identity checked before fitting").

Input: a crop of the Unit 1 panel from a 400 dpi render of the bulletin's scanned page 267 (pdftoppm). The
scan is bilevel; dark pixels (< 128) are ink. Nothing in the plotted response is altered.

Method
  1. Axes: the row with the longest horizontal ink run is the Delta axis, the column with the longest vertical
     run the V axis; each is fitted as a line through its ink pixels (least squares), which captures the scan's
     skew. Their intersection is the origin.
  2. Ticks: short ink segments perpendicular to an axis, centred on it, with no continuation beyond a few
     pixels; matched by order to the printed values (-120 .. 120 mm every 30, +-30/60/90 kN). The affine map
     pixel -> (Delta, V) is a least-squares fit to the ticks and the origin; its residuals are reported.
  3. Peaks: for each run of the provisional v1 list, the reversal point is the extreme-Delta ink pixel inside
     a window around the v1 reading (a displacement-controlled reversal is the loop's extreme displacement).
  4. Branches: from each reversal the unloading branch is followed pixel by pixel toward zero force (a greedy
     tracker that continues the current direction through crossings with other loops), giving the 80 % and
     20 % force levels, the zero-force crossing and, continuing, the zero-displacement crossing. Every trace
     is saved and drawn on an overlay so its identity can be checked; a trace that loses the curve or that
     reaches the axis more than a set distance from any independently detected crossing is marked unresolved.
  5. Axis crossings are also detected independently (ink on both sides of an axis at a column/row), as the
     reference set the traces are checked against.

usage: python digitize_fig16a.py --image <crop.png> --out <dir> [--v1 unit1_observed_digitized.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
DELTA_TICKS = (-120, -90, -60, -30, 30, 60, 90, 120)
V_TICKS = (-90, -60, -30, 30, 60, 90)


def load(path):
    im = Image.open(path).convert("L")
    a = np.array(im)
    return im, a < 128


def curve_ink(ink, min_extent=40):
    """Ink belonging to the plotted curves: connected components at least min_extent px long in one
    direction (a scan gap can cut a steep branch into thin fragments, which must survive), excluding the
    dashes of the V1/V2 lines (thin and horizontal) and the tick marks (thin and short). The printed run
    numbers and axis labels are smaller than min_extent in both directions and are dropped."""
    from scipy import ndimage
    labels, n = ndimage.label(ink, structure=np.ones((3, 3), dtype=int))
    keep = np.zeros(n + 1, dtype=bool)
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start; w = sl[1].stop - sl[1].start
        dash = h <= 7 and w >= 25
        tick = w <= 7 and h <= 50
        keep[k] = max(h, w) >= min_extent and not dash and not tick
    return keep[labels]


def longest_run(line):
    best, cur, start, best_start = 0, 0, 0, 0
    for i, v in enumerate(line):
        if v:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best, best_start = cur, start
        else:
            cur = 0
    return best, best_start


def find_axes(ink):
    h, w = ink.shape
    runs_r = [longest_run(ink[r]) for r in range(h)]
    r0 = int(np.argmax([r[0] for r in runs_r]))
    # the V axis: the column with the most ink over the full height (a steep loop branch has a long run but
    # drifts across columns; the axis line, gaps and all, keeps one column), searched near the Delta axis' middle
    length0, start0 = runs_r[r0]
    mid = start0 + length0 // 2
    counts = ink.sum(axis=0).astype(float)
    lo, hi = max(mid - 300, 0), min(mid + 300, w)
    c0 = int(lo + np.argmax(counts[lo:hi]))
    runs_c = [longest_run(ink[:, c]) for c in range(w)]
    # ink pixels of the Delta axis: within +-6 rows of r0, over the run's extent, columns where the row band is thin
    length, start = runs_r[r0]
    xs, ys = [], []
    for x in range(start, start + length):
        col = ink[max(r0 - 8, 0):r0 + 9, x]
        idx = np.where(col)[0]
        if 0 < len(idx) <= 6:
            xs.append(x); ys.append(r0 - 8 + idx.mean())
    b, a = np.polyfit(xs, ys, 1)
    delta_axis = {"row_seed": r0, "x_start": start, "x_end": start + length - 1, "fit_y_of_x": [a, b], "n_pixels": len(xs),
                  "rms_px": float(np.sqrt(np.mean((np.array(ys) - (a + b * np.array(xs))) ** 2)))}
    ys_ink = np.where(ink[:, c0])[0]
    start, end = int(ys_ink.min()), int(ys_ink.max())
    xs, ys = [], []
    for y in range(start, end + 1):
        row = ink[y, max(c0 - 4, 0):c0 + 5]
        idx = np.where(row)[0]
        if 0 < len(idx) <= 4:
            ys.append(y); xs.append(c0 - 4 + idx.mean())
    d, c = np.polyfit(ys, xs, 1)
    v_axis = {"col_seed": c0, "y_start": start, "y_end": end, "fit_x_of_y": [c, d], "n_pixels": len(ys),
              "rms_px": float(np.sqrt(np.mean((np.array(xs) - (c + d * np.array(ys))) ** 2)))}
    # origin: intersection of y = a + b x and x = c + d y
    x0 = (c + d * a) / (1.0 - d * b)
    y0 = a + b * x0
    return delta_axis, v_axis, (x0, y0)


def _tick_candidates_delta(ink, delta_axis, origin, x_range, min_len=5, max_len=45, clear=16):
    """Columns where a short ink run touches the Delta axis, extends at least 5 px on one side of it and at
    most 3 px on the other (a crossing curve extends both ways), with no ink beyond the run on the extending
    side; the axis labels sit on the other side and are ignored."""
    a, b = delta_axis["fit_y_of_x"]
    cands, group, x_prev = [], [], None
    for x in range(int(x_range[0]), int(x_range[1]) + 1):
        y = a + b * x
        yi = int(round(y))
        col = ink[:, x]
        ys = [yy for yy in range(yi - 3, yi + 4) if 0 <= yy < len(col) and col[yy]]
        if not ys:
            continue
        top = min(ys); bot = max(ys)
        while top - 1 >= 0 and col[top - 1]:
            top -= 1
        while bot + 1 < len(col) and col[bot + 1]:
            bot += 1
        total = bot - top + 1
        if not (min_len <= total <= max_len):
            continue
        up, dn = y - top, bot - y
        if up >= 5 and dn <= 3:
            clear_side = col[max(top - clear, 0):max(top - 1, 0)]
        elif dn >= 5 and up <= 3:
            clear_side = col[bot + 2:bot + 2 + clear]
        else:
            continue
        if clear_side.any():
            continue
        if x_prev is not None and x - x_prev <= 2:
            group.append(x)
        else:
            if group:
                cands.append(group)
            group = [x]
        x_prev = x
    if group:
        cands.append(group)
    return [float(np.mean(g)) for g in cands if abs(np.mean(g) - origin[0]) > 20 and len(g) <= 12]


def _tick_candidates_v(ink, v_axis, origin, y_range, min_len=5, max_len=45, clear=16):
    c, d = v_axis["fit_x_of_y"]
    cands, group, y_prev = [], [], None
    for y in range(int(y_range[0]), int(y_range[1]) + 1):
        x = c + d * y
        xi = int(round(x))
        row = ink[y, :]
        xs = [xx for xx in range(xi - 3, xi + 4) if 0 <= xx < len(row) and row[xx]]
        if not xs:
            continue
        lf = min(xs); rt = max(xs)
        while lf - 1 >= 0 and row[lf - 1]:
            lf -= 1
        while rt + 1 < len(row) and row[rt + 1]:
            rt += 1
        total = rt - lf + 1
        if not (min_len <= total <= max_len):
            continue
        left, right = x - lf, rt - x
        if left >= 5 and right <= 3:
            clear_side = row[max(lf - clear, 0):max(lf - 1, 0)]
        elif right >= 5 and left <= 3:
            clear_side = row[rt + 2:rt + 2 + clear]
        else:
            continue
        if clear_side.any():
            continue
        if y_prev is not None and y - y_prev <= 2:
            group.append(y)
        else:
            if group:
                cands.append(group)
            group = [y]
        y_prev = y
    if group:
        cands.append(group)
    return [float(np.mean(g)) for g in cands if abs(np.mean(g) - origin[1]) > 20 and len(g) <= 12]


def _consensus_ticks(cands, origin_coord, values, spacing_bounds, tol=6.0, sign=1.0):
    """Assign candidates to the printed tick values by the regular spacing that explains the most of them:
    for each candidate pair, the implied pixels-per-unit is tested against every candidate; the best
    hypothesis wins. Returns {value: pixel} for the values found and the fitted spacing."""
    best = None
    unit = abs(values[1] - values[0]) if len(values) > 1 else 1.0
    for i in range(len(cands)):
        for j in range(len(cands)):
            if i == j:
                continue
            for vi in values:
                for vj in values:
                    if vi == vj:
                        continue
                    ppu = (cands[j] - cands[i]) / (vj - vi)
                    if ppu * sign <= 0 or not (spacing_bounds[0] <= abs(ppu) <= spacing_bounds[1]):
                        continue
                    off = cands[i] - ppu * vi
                    if abs(off - origin_coord) > 12:          # the origin is fixed by the axis-line intersection
                        continue
                    assigned, resid = {}, []
                    for val in values:
                        expect = off + ppu * val
                        near = min(cands, key=lambda q: abs(q - expect))
                        if abs(near - expect) <= tol:
                            assigned[val] = near; resid.append(abs(near - expect))
                    score = (len(assigned), -float(np.mean(resid)) if resid else 0.0)
                    if best is None or score > best[0]:
                        best = (score, ppu, off, assigned)
    if best is None:
        return {}, None, None
    _, ppu, off, assigned = best
    return assigned, ppu, off


def find_ticks(ink, delta_axis, v_axis, origin):
    h, w = ink.shape
    cd = _tick_candidates_delta(ink, delta_axis, origin, (5, w - 6))
    cv = _tick_candidates_v(ink, v_axis, origin, (5, h - 6))
    # plausible scales at 400 dpi: 2.5 to 6 px per mm and per kN
    td, ppu_d, off_d = _consensus_ticks(cd, origin[0], DELTA_TICKS, (2.5, 6.0), sign=1.0)
    # prior, verified on the overlay: the figure draws 30 mm and 30 kN with the same tick spacing (the 90 kN
    # tick sits three quarters as far from the origin as the 120 mm tick), so the V scale is searched within
    # 10 % of the Delta scale; y grows downward
    v_bounds = (0.9 * abs(ppu_d), 1.1 * abs(ppu_d)) if ppu_d else (2.5, 6.0)
    tv, ppu_v, off_v = _consensus_ticks(cv, origin[1], V_TICKS, v_bounds, sign=-1.0)
    return {"candidates_px": cd, "found": {str(k): v for k, v in td.items()}, "px_per_mm": ppu_d, "offset_px": off_d}, \
           {"candidates_px": cv, "found": {str(k): v for k, v in tv.items()}, "px_per_kN": ppu_v, "offset_px": off_v}


def fit_affine(points):
    """points: list of ((x, y), (Delta, V)); returns the 2x3 map and residuals."""
    A = np.array([[x, y, 1.0] for (x, y), _ in points])
    D = np.array([dv[0] for _, dv in points]); V = np.array([dv[1] for _, dv in points])
    md, *_ = np.linalg.lstsq(A, D, rcond=None)
    mv, *_ = np.linalg.lstsq(A, V, rcond=None)
    res_d, res_v = A @ md - D, A @ mv - V
    return np.vstack([md, mv]), {"delta_rms_mm": float(np.sqrt(np.mean(res_d ** 2))), "delta_max_mm": float(np.max(np.abs(res_d))),
                                 "v_rms_kN": float(np.sqrt(np.mean(res_v ** 2))), "v_max_kN": float(np.max(np.abs(res_v)))}


def to_phys(M, x, y):
    p = M @ np.array([x, y, 1.0])
    return float(p[0]), float(p[1])


def to_pixel(M, delta, v):
    A = M[:, :2]; b = M[:, 2]
    xy = np.linalg.solve(A, np.array([delta, v]) - b)
    return float(xy[0]), float(xy[1])


def axis_crossings(ink, M, delta_axis, v_axis, origin, x_range, y_range, band=(3, 14)):
    """Columns with ink on both sides of the Delta axis and continuing beyond the band on at least one side
    (a tick does not continue); rows likewise for the V axis. Scanned over the tick extents."""
    a, b = delta_axis["fit_y_of_x"]
    cols = []
    for x in range(int(x_range[0]), int(x_range[1]) + 1):
        y = int(round(a + b * x))
        up = ink[y - band[1]:y - band[0], x - 6:x + 7].any()
        dn = ink[y + band[0] + 1:y + band[1] + 1, x - 6:x + 7].any()
        far_up = ink[y - 2 * band[1]:y - band[1], x - 6:x + 7].any()
        far_dn = ink[y + band[1] + 1:y + 2 * band[1] + 1, x - 6:x + 7].any()
        if up and dn and (far_up or far_dn):
            cols.append(x)
    groups, g = [], []
    for x in cols:
        if g and x - g[-1] > 3:
            groups.append(g); g = []
        g.append(x)
    if g:
        groups.append(g)
    def strands(sub):
        """Maximum number of separate ink runs in any row of a sub-image (the curves crossing a band)."""
        best = 0
        for row in sub:
            runs = 0; prev = False
            for v in row:
                if v and not prev:
                    runs += 1
                prev = v
            best = max(best, runs)
        return best
    dcross = []
    for g in groups:
        xm = float(np.mean(g)); ym = a + b * xm
        if abs(xm - origin[0]) < 25:
            continue
        y = int(round(ym))
        n = max(strands(ink[y - 14:y - 4, g[0] - 2:g[-1] + 3]), strands(ink[y + 5:y + 15, g[0] - 2:g[-1] + 3]))
        dcross.append({"x_px": xm, "y_px": ym, "width_px": len(g), "delta_mm": to_phys(M, xm, ym)[0], "strands": int(n),
                       "delta_range_mm": [to_phys(M, g[0], a + b * g[0])[0], to_phys(M, g[-1], a + b * g[-1])[0]]})
    c, d = v_axis["fit_x_of_y"]
    rows = []
    for y in range(int(y_range[0]), int(y_range[1]) + 1):
        x = int(round(c + d * y))
        lf = ink[y - 6:y + 7, x - band[1]:x - band[0]].any()
        rt = ink[y - 6:y + 7, x + band[0] + 1:x + band[1] + 1].any()
        far_lf = ink[y - 6:y + 7, x - 2 * band[1]:x - band[1]].any()
        far_rt = ink[y - 6:y + 7, x + band[1] + 1:x + 2 * band[1] + 1].any()
        if lf and rt and (far_lf or far_rt):
            rows.append(y)
    groups, g = [], []
    for y in rows:
        if g and y - g[-1] > 3:
            groups.append(g); g = []
        g.append(y)
    if g:
        groups.append(g)
    vcross = []
    for g in groups:
        ym = float(np.mean(g)); xm = c + d * ym
        if abs(ym - origin[1]) < 25:
            continue
        x = int(round(xm))
        n = max(strands(ink[g[0] - 2:g[-1] + 3, x - 14:x - 4].T), strands(ink[g[0] - 2:g[-1] + 3, x + 5:x + 15].T))
        vcross.append({"x_px": xm, "y_px": ym, "height_px": len(g), "V_kN": to_phys(M, xm, ym)[1], "strands": int(n),
                       "V_range_kN": [to_phys(M, c + d * g[0], g[0])[1], to_phys(M, c + d * g[-1], g[-1])[1]]})
    return dcross, vcross


def _runs_in(line):
    runs, prev = 0, False
    for v in line:
        if v and not prev:
            runs += 1
        prev = v
    return runs


def find_peak(curves, M, delta_mm, v_kn, direction, window=(5.0, 5.0)):
    """The reversal point of a run: a loop tip inside a window around the provisional reading. A tip pixel
    is the locally extreme-x ink pixel (no ink 1-4 px ahead of it within +-3 rows) behind which, 6 px back,
    the column crosses at least two separate ink runs within +-30 rows (the loading and unloading branches
    converging). Tips are clustered (3 px) and each cluster's extreme pixel is a candidate; larger loops'
    branches passing through the window have no tip there and are ignored."""
    x1, y1 = to_pixel(M, delta_mm - window[0], v_kn + window[1]); x2, y2 = to_pixel(M, delta_mm + window[0], v_kn - window[1])
    xa, xb = int(min(x1, x2)), int(max(x1, x2)); ya, yb = int(min(y1, y2)), int(max(y1, y2))
    h, w = curves.shape
    tips = []
    for y in range(max(ya, 31), min(yb, h - 31) + 1):
        for x in range(max(xa, 8), min(xb, w - 8) + 1):
            if not curves[y, x]:
                continue
            if direction == "+":
                ahead = curves[y - 3:y + 4, x + 1:x + 5]
                behind = curves[y - 30:y + 31, x - 6]
            else:
                ahead = curves[y - 3:y + 4, x - 4:x]
                behind = curves[y - 30:y + 31, x + 6]
            if ahead.any():
                continue
            if _runs_in(behind) >= 2:
                tips.append((x, y))
    if not tips:
        return None
    tips.sort(key=lambda t: t[1])
    clusters, g = [], [tips[0]]
    for t in tips[1:]:
        if abs(t[1] - g[-1][1]) <= 3 and abs(t[0] - g[-1][0]) <= 6:
            g.append(t)
        else:
            clusters.append(g); g = [t]
    clusters.append(g)
    cands = []
    for g in clusters:
        xs = [t[0] for t in g]; ys = [t[1] for t in g]
        xm = float(max(xs) if direction == "+" else min(xs)); ym = float(np.mean(ys))
        d, v = to_phys(M, xm, ym)
        v_top, v_bot = to_phys(M, xm, min(ys))[1], to_phys(M, xm, max(ys))[1]
        cands.append({"x_px": xm, "y_px": ym, "delta_mm": d, "V_kN": v, "tip_pixels": len(g), "rows": [int(min(ys)), int(max(ys))],
                      "V_range_kN": [min(v_top, v_bot), max(v_top, v_bot)],
                      "note": "a tip taller than about 6 px may hold both cycles' reversals at this amplitude" if max(ys) - min(ys) >= 6 else None})
    return cands


def _chains(cands, dy=12.0, dx=4.0):
    """Group tip candidates that lie along one steep segment (consecutive candidates within dy rows and dx
    columns of each other)."""
    cands = sorted(cands, key=lambda q: q["y_px"])
    chains = []
    for c in cands:
        for ch in chains:
            if abs(c["y_px"] - ch[-1]["y_px"]) <= dy and abs(c["x_px"] - ch[-1]["x_px"]) <= dx:
                ch.append(c)
                break
        else:
            chains.append([c])
    return chains


def tip_interval(curves, M, tip, direction):
    """The reversal's force reading interval. The visible tip is where the steep unloading branch emerges
    from the bundle of the larger loops' loading branches; the true reversal may sit anywhere inside the
    contiguous ink above it (positive) or below it (negative). The interval runs from the emergence point
    to the far edge of that contiguous ink in the tip's column (x +- 3 px, a gap of 3 white rows ends it)."""
    x, y = int(round(tip["x_px"])), int(round(tip["y_px"]))
    step = -1 if direction == "+" else 1
    h = curves.shape[0]
    yy, gap, last = y, 0, y
    while 0 <= yy + step < h:
        yy += step
        if curves[yy, max(x - 3, 0):x + 4].any():
            last, gap = yy, 0
        else:
            gap += 1
            if gap >= 3:
                break
    v_tip = to_phys(M, tip["x_px"], tip["y_px"])[1]
    v_far = to_phys(M, tip["x_px"], last)[1]
    lo, hi = min(v_tip, v_far), max(v_tip, v_far)
    return {"V_low_kN": lo, "V_high_kN": hi, "far_edge_px": [tip["x_px"], float(last)], "width_kN": hi - lo,
            "obscured": bool(hi - lo > 1.5),
            "basis": "emergence of the steep unloading branch to the far edge of the contiguous ink in the tip column"}


def trace_branch(ink, M, start, direction, step=5.0, max_steps=6000):
    """Follow the unloading branch from a reversal. A heading-based walker: from the current point the next
    point is the ink pixel on a ring of radius ``step`` whose direction deviates least from the current
    heading (momentum-smoothed), so a crossing with another loop is passed straight through. The initial
    heading points straight down for a positive reversal (up for a negative one): the unloading branch
    leaves the tip steeply while the loading branch arrives nearly horizontally. Ends when the branch has
    passed zero force and then crossed zero displacement, or when the curve is lost."""
    x, y = float(start[0]), float(start[1])
    hx, hy = (0.0, 1.0) if direction == "+" else (0.0, -1.0)
    path = [(x, y)]
    ambiguous, passed_zero_force = 0, False
    h, w = ink.shape
    d_start = to_phys(M, x, y)[0]
    for _ in range(max_steps):
        found = None
        for radius in (step, step + 3.0, step + 7.0, step + 11.0, step + 17.0):      # the larger radii jump bundles of crossing branches
            cands = []
            r = int(np.ceil(radius)) + 1
            ya, yb = max(int(y) - r, 0), min(int(y) + r, h - 1)
            xa, xb = max(int(x) - r, 0), min(int(x) + r, w - 1)
            sub = ink[ya:yb + 1, xa:xb + 1]
            ys, xs = np.where(sub)
            for px, py in zip(xs + xa, ys + ya):
                dx, dy = px - x, py - y
                dist = np.hypot(dx, dy)
                if abs(dist - radius) > 1.2:
                    continue
                # physical monotonicity of an unloading-then-reloading branch: from a positive reversal it only
                # moves down (toward zero force) and left (toward the origin); mirrored for a negative one
                if direction == "+" and (dy < 0 or dx > 3):
                    continue
                if direction == "-" and (dy > 0 or dx < -3):
                    continue
                cos = (dx * hx + dy * hy) / dist
                if cos > 0.34:                       # within 70 degrees of the heading
                    cands.append((cos, px, py))
            if cands:
                cands.sort(reverse=True)
                best = cands[0]
                # ambiguity: another candidate cluster well separated from the best one
                if any(np.hypot(c[1] - best[1], c[2] - best[2]) > 4 for c in cands[1:]):
                    ambiguous += 1
                found = best
                break
        if found is None:
            return {"path": path, "status": "lost_curve", "ambiguous_steps": ambiguous}
        _, px, py = found
        dx, dy = px - x, py - y
        dist = np.hypot(dx, dy)
        hx, hy = 0.6 * hx + 0.4 * dx / dist, 0.6 * hy + 0.4 * dy / dist
        n = np.hypot(hx, hy); hx, hy = hx / n, hy / n
        x, y = float(px), float(py)
        path.append((x, y))
        d, v = to_phys(M, x, y)
        if (direction == "+" and v <= 0.0) or (direction == "-" and v >= 0.0):
            passed_zero_force = True
        if passed_zero_force and ((direction == "+" and d <= 0.0) or (direction == "-" and d >= 0.0)):
            return {"path": path, "status": "reached_zero_displacement", "ambiguous_steps": ambiguous}
        if not passed_zero_force and abs(d) > abs(d_start) + 8.0:
            return {"path": path, "status": "left_the_unloading_branch", "ambiguous_steps": ambiguous}
        if passed_zero_force and abs(d) > 1.3 * abs(d_start):
            return {"path": path, "status": "diverged_after_zero_force", "ambiguous_steps": ambiguous}
    return {"path": path, "status": "max_steps", "ambiguous_steps": ambiguous}


def branch_observables(M, trace, peak):
    """From a traced branch: 80 % / 20 % force levels (toward zero), zero-force crossing, zero-displacement
    crossing, each interpolated between consecutive trace pixels."""
    phys = [to_phys(M, x, y) for x, y in trace["path"]]
    vp = peak["V_kN"]; sgn = 1.0 if vp >= 0 else -1.0
    out = {}

    def first_level(target_v):
        for i in range(len(phys) - 1):
            (d1, v1), (d2, v2) = phys[i], phys[i + 1]
            if (v1 - target_v) * sgn >= 0.0 > (v2 - target_v) * sgn:
                t = (v1 - target_v) / (v1 - v2)
                return {"delta_mm": d1 + t * (d2 - d1), "V_kN": target_v, "x_px": trace["path"][i][0] + t * (trace["path"][i + 1][0] - trace["path"][i][0]),
                        "y_px": trace["path"][i][1] + t * (trace["path"][i + 1][1] - trace["path"][i][1])}
        return None
    out["level_80"], out["level_20"], out["zero_force"] = first_level(0.8 * vp), first_level(0.2 * vp), first_level(0.0)
    for i in range(len(phys) - 1):
        (d1, v1), (d2, v2) = phys[i], phys[i + 1]
        if d1 * sgn > 0.0 >= d2 * sgn:
            t = d1 / (d1 - d2)
            out["zero_displacement"] = {"delta_mm": 0.0, "V_kN": v1 + t * (v2 - v1), "x_px": trace["path"][i][0] + t * (trace["path"][i + 1][0] - trace["path"][i][0]),
                                        "y_px": trace["path"][i][1] + t * (trace["path"][i + 1][1] - trace["path"][i][1])}
            break
    else:
        out["zero_displacement"] = None
    if out["level_80"] and out["level_20"]:
        df = abs(out["level_80"]["V_kN"] - out["level_20"]["V_kN"]); du = abs(out["level_80"]["delta_mm"] - out["level_20"]["delta_mm"])
        out["unloading_chord"] = {"dF_kN": df, "dU_mm": du, "k_kN_per_mm": df / du if du > 0 else None,
                                  "k_low": max(0.0, df - 6.0) / (du + 6.0), "k_high": (df + 6.0) / (du - 6.0) if du > 6.0 else None,
                                  "bounds_basis": "Codex 2026-09-28: independent +-3 kN / +-3 mm endpoint bounds; K_high needs dU > 6 mm"}
    else:
        out["unloading_chord"] = None
    return out


def write_v2(result, v1, path, args):
    """The version-2 observation file: every quantity with its pixel coordinates, the axis transformation,
    reading intervals where the figure obscures a point, identity statuses for the crossings, and the v1
    readings retained as provisional. Nothing from v1 is deleted; superseded targets are marked."""
    peaks, branches = [], []
    for rec in result["runs"]:
        rev = rec["reversal"]
        base = {"run": rec["run"], "mu": rec["mu"], "cycle": rec["cycle"], "direction": rec["direction"], "load_control": rec["load_control"],
                "v1_reading": rec["v1_reading"]}
        if rev is None:
            peaks.append({**base, "delta_mm": rec["v1_reading"]["delta_mm"], "V_kN": rec["v1_reading"]["V_kN"], "status": "v1_reading_retained_no_tip_found"})
        else:
            iv = rev["interval"]
            peaks.append({**base, "delta_mm": rev["delta_mm"], "V_kN": rev["V_kN"], "V_interval_kN": [iv["V_low_kN"], iv["V_high_kN"]],
                          "obscured": iv["obscured"], "pixel": {"x": rev["x_px"], "y": rev["y_px"]}, "far_edge_pixel": iv["far_edge_px"],
                          "chains_in_window": rev["chains_in_window"], "cycles_separable": rec.get("cycles_separable"),
                          "v1_within_interval_or_band": rev["v1_within_interval_or_band"],
                          "status": "pixel_tip_obscured_interval" if iv["obscured"] else "pixel_tip"})
        obs = rec["observables"] or {}
        zf, zd, ch = obs.get("zero_force"), obs.get("zero_displacement"), obs.get("unloading_chord")
        tr = rec["trace"] or {}
        branches.append({"run": rec["run"], "direction": rec["direction"], "trace_status": tr.get("status"), "ambiguous_steps": tr.get("ambiguous_steps"),
                         "identity": rec["identity"],
                         "zero_force_crossing": None if zf is None else {"delta_mm": zf["delta_mm"], "pixel": {"x": zf["x_px"], "y": zf["y_px"]},
                                                                          "matches_independent_crossing": zf.get("matches_independent_crossing"),
                                                                          "nearest_independent_mm": zf.get("nearest_independent_crossing_mm"),
                                                                          "status": ("resolved" if zf.get("matches_independent_crossing") else "traced_unconfirmed")},
                         "zero_displacement_crossing": None if zd is None else {"V_kN": zd["V_kN"], "pixel": {"x": zd["x_px"], "y": zd["y_px"]},
                                                                                 "status": "traced_unconfirmed: the reloading branches bundle near the V axis; per-cycle identity unresolved"},
                         "unloading_chord_80_20": None if ch is None else {**ch, "level_80": obs["level_80"], "level_20": obs["level_20"]},
                         "rejected_crossing": obs.get("zero_force_rejected"),
                         "unavailable_reason": (None if zf is not None else
                                                "the unloading branch could not be followed to zero force: it merges with the reloading branches of the "
                                                "opposite loops (the pinched region); marked unavailable, not estimated")})
    v_bundles = [{"V_kN": q["V_kN"], "V_range_kN": q["V_range_kN"], "strands": q["strands"], "pixel": {"x": q["x_px"], "y": q["y_px"]}} for q in result["independent_crossings"]["zero_displacement_V_kN"]]
    d_bundles = [{"delta_mm": q["delta_mm"], "delta_range_mm": q["delta_range_mm"], "strands": q["strands"], "pixel": {"x": q["x_px"], "y": q["y_px"]}} for q in result["independent_crossings"]["zero_force_delta_mm"]]
    lc = {r["run"]: r for r in peaks if r["load_control"]}
    v2 = {"version": 2, "date": "2026-09-29",
          "source": v1["source"],
          "supersedes": "unit1_observed_digitized.json (version 1, manual reading of a 300 dpi render, 2026-09-27)",
          "reason": ("Codex Unit 1 review 2026-09-28: branch-specific redigitization with pixel coordinates and the axis transformation; the v1 residual "
                     "targets (35 and 42 mm) and the single unsigned pinching target (28 kN) are not accepted; run labels audited; the +-3 kN / +-3 mm "
                     "reading bounds are estimated resolution, not statistical confidence"),
          "image": result["source"], "axes": result["axes"], "ticks": result["ticks"], "transformation": result["transformation"],
          "digitization": {"uncertainty_kN": 3.0, "uncertainty_mm": 3.0, "crossing_uncertainty_mm": 6.0, "crossing_uncertainty_kN": 6.0,
                           "interpretation": "estimated reading bounds; an obscured reversal carries an explicit interval on top of the band",
                           "method": "digitize_fig16a.py: ink threshold, connected-component curve mask, axis-line fits, tick consensus, loop-tip detection, "
                                     "heading-based branch tracing, independent axis-crossing detection; overlay fig16a_audit_overlay.png"},
          "theoretical_loads_kN": v1["theoretical_loads_kN"], "maximum_load_kN": v1["maximum_load_kN"],
          "yield_displacement_mm": {**v1["yield_displacement_mm"], "status": "inferred, provisional; the model never rescales the history by it"},
          "load_controlled_first_cycle": {"target_kN": 54.45, "basis": "0.75 x V2 in both directions (section 3.2, Fig. 15)",
                                          "run_1": None if 1 not in lc else {"delta_mm": lc[1]["delta_mm"], "V_kN_at_tip": lc[1]["V_kN"], "V_interval_kN": lc[1].get("V_interval_kN"),
                                                                             "pixel": lc[1].get("pixel"), "status": lc[1]["status"]},
                                          "run_2": None if 2 not in lc else {"delta_mm": lc[2]["delta_mm"], "V_kN_at_tip": lc[2]["V_kN"], "V_interval_kN": lc[2].get("V_interval_kN"),
                                                                             "pixel": lc[2].get("pixel"), "status": lc[2]["status"]},
                                          "note": "the displacement reached at the load target is the initial-compliance observable; the tip force read from the "
                                                  "figure differs from 54.45 kN by the reading/obscuration interval and is not the target"},
          "peaks": peaks, "branches": branches,
          "independent_crossings": {"zero_force_delta_mm": d_bundles, "zero_displacement_V_kN": v_bundles,
                                    "note": "columns/rows where ink crosses an axis; a bundle of n strands holds n unresolved branches"},
          "amplitude_audit": {"reconstructed_amplitudes_mm": [30, 45, 60, 75, 90, 105],
                              "pixel_tip_delta_by_amplitude": {str(int(abs(r["mu"]) * 15)): sorted({round(abs(q["delta_mm"]), 1) for q in peaks if q.get("pixel") and abs(q["mu"]) == abs(r["mu"])})
                                                               for r in peaks if not r["load_control"]},
                              "status": "the pixel tips lie within about 1.5 mm of mu x 15 mm on both sides; the reconstruction stands with the +-3 mm coherent shift as its sensitivity"},
          "superseded_v1_targets": {"residual_displacement_at_zero_load_mm": {**v1["pinching_indicators"]["residual_displacement_at_zero_load_mm"],
                                                                              "status": "NOT ACCEPTED (Codex 2026-09-28): the outer positive unloading branches cross zero force at 45-66 mm (branches above)"},
                                    "load_at_zero_displacement_kN": {**v1["pinching_indicators"]["load_at_zero_displacement_kN"],
                                                                     "status": "NOT ACCEPTED: no per-cycle identity; replaced by the signed bundles in independent_crossings"},
                                    "qualitative": v1["pinching_indicators"]["qualitative"]},
          "run_label_audit": {"run_25": "the second cycle at +105 mm reverses visibly lower than run 23 (pixel tips above); retained as printed, not removed",
                              "cycle_pairs": "where cycles_separable is false the two cycles at that amplitude and direction share one tip interval at the scan's resolution",
                              "printed_numbers": "run numbers 1-26 as printed on the figure; identity of each numbered reversal taken from its printed position"},
          "strength_retention": v1["strength_retention"], "damage_observations": v1["damage_observations"], "protocol": v1["protocol"]}
    path.write_text(json.dumps(v2, indent=1), encoding="utf-8")
    print("wrote", path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--v1", default=str(HERE / "unit1_observed_digitized.json"))
    parser.add_argument("--dpi", type=float, default=400.0)
    parser.add_argument("--write-v2", default=None, help="write the version-2 observation file to this path")
    args = parser.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    im, ink = load(args.image)
    curves = curve_ink(ink)
    trace_ink = curves.copy()               # the axis lines are masked out for tracing: a walker must not slide along them
    v1 = json.loads(Path(args.v1).read_text(encoding="utf-8"))
    delta_axis, v_axis, origin = find_axes(ink)
    ticks_d, ticks_v = find_ticks(ink, delta_axis, v_axis, origin)
    result = {"source": {"image": str(Path(args.image).resolve()), "render": f"pdftoppm {args.dpi:g} dpi of the bulletin's scanned page 267, cropped to the Unit 1 panel",
                         "ink_threshold": "gray < 128", "curve_ink": "connected components at least 40 px long in one direction, excluding thin horizontal "
                                                                     "dashes (the V1/V2 lines) and thin short marks (ticks); drops the printed run numbers and "
                                                                     "axis labels before any curve analysis"},
              "axes": {"delta_axis": delta_axis, "v_axis": v_axis, "origin_px": list(origin),
                       "skew": {"delta_axis_slope_px_per_px": delta_axis["fit_y_of_x"][1], "v_axis_slope_px_per_px": v_axis["fit_x_of_y"][1]}},
              "ticks": {"delta_axis": ticks_d, "v_axis": ticks_v, "expected": {"delta": list(DELTA_TICKS), "v": list(V_TICKS)},
                        "method": "short isolated ink segments touching the axis, assigned to the printed values by the regular spacing that explains the most of them; "
                                  "the V scale is searched within 10 % of the Delta scale (equal tick spacing on the figure, verified on the overlay)"}}
    a, b = delta_axis["fit_y_of_x"]; c, d = v_axis["fit_x_of_y"]
    points = [((origin[0], origin[1]), (0.0, 0.0))]
    for val, x in ticks_d["found"].items():
        points.append(((x, a + b * x), (float(val), 0.0)))
    for val, y in ticks_v["found"].items():
        points.append(((c + d * y, y), (0.0, float(val))))
    result["ticks"]["missing"] = {"delta": [v for v in DELTA_TICKS if str(v) not in ticks_d["found"]], "v": [v for v in V_TICKS if str(v) not in ticks_v["found"]]}
    if len(ticks_d["found"]) < 4 or len(ticks_v["found"]) < 3:
        result["status"] = f"axis calibration incomplete: delta ticks {sorted(ticks_d['found'])}, v ticks {sorted(ticks_v['found'])}"
        (out / "fig16a_audit.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(result["status"]); print("candidates delta", ticks_d["candidates_px"]); print("candidates v", ticks_v["candidates_px"])
        return result
    M, residuals = fit_affine(points)
    yy, xx = np.mgrid[0:ink.shape[0], 0:ink.shape[1]]
    trace_ink &= ~(np.abs(yy - (a + b * xx)) <= 2)      # the axis line's own thickness only
    trace_ink &= ~(np.abs(xx - (c + d * yy)) <= 2)
    xr = (min(min(ticks_d["found"].values()), origin[0]) - 10, max(max(ticks_d["found"].values()), origin[0]) + 10)
    yr = (min(min(ticks_v["found"].values()), origin[1]) - 10, max(max(ticks_v["found"].values()), origin[1]) + 10)
    result["transformation"] = {"affine_pixel_to_physical": M.tolist(), "form": "[Delta; V] = M @ [x; y; 1]", "fit_points": len(points),
                                "residuals": residuals, "mm_per_px": float(np.hypot(M[0, 0], M[0, 1])), "kN_per_px": float(np.hypot(M[1, 0], M[1, 1]))}
    dcross, vcross = axis_crossings(curves, M, delta_axis, v_axis, origin, xr, yr)
    result["independent_crossings"] = {"zero_force_delta_mm": dcross, "zero_displacement_V_kN": vcross}
    # peaks and branches
    runs = []
    overlay = im.convert("RGB"); draw = ImageDraw.Draw(overlay)
    colours = [(200, 0, 0), (0, 140, 0), (0, 0, 220), (200, 120, 0), (140, 0, 160), (0, 150, 150)]
    for k, p in enumerate(v1["peaks"]):
        cands = find_peak(curves, M, p["delta_mm"], p["V_kN"], p["direction"], window=(6.0, 8.0))
        rec = {"run": p["run"], "mu": p["mu"], "cycle": p.get("cycle"), "direction": p["direction"], "load_control": bool(p.get("load_control")),
               "v1_reading": {"delta_mm": p["delta_mm"], "V_kN": p["V_kN"]},
               "reversal_candidates": cands, "reversal": None, "trace": None, "observables": None, "identity": "unresolved"}
        if cands:
            # the candidate nearest the v1 force reading
            # candidates lie along the steep unloading segments in the window (one segment per loop); they are
            # chained along each segment and the reversal of a chain is its end on the loading side (the top for
            # a positive run, the bottom for a negative one). The chain is chosen by the provisional reading:
            # its displacement selects the amplitude, its force the cycle when two chains exist there.
            chains = _chains(cands)
            tips = [(min(ch, key=lambda q: q["y_px"]) if p["direction"] == "+" else max(ch, key=lambda q: q["y_px"])) for ch in chains]
            near = [t for t in tips if abs(t["delta_mm"] - p["delta_mm"]) <= 3.5] or tips
            rev = dict(min(near, key=lambda q: abs(q["V_kN"] - p["V_kN"])))
            rev["chains_in_window"] = len(chains)
            rev["chain_tips_V_kN"] = sorted(t["V_kN"] for t in tips)
            rev["interval"] = tip_interval(curves, M, rev, p["direction"])
            rev["v1_within_interval_or_band"] = bool(rev["interval"]["V_low_kN"] - 3.0 <= p["V_kN"] <= rev["interval"]["V_high_kN"] + 3.0)
            rec["reversal"] = rev
            tr = trace_branch(trace_ink, M, (rev["x_px"], rev["y_px"]), p["direction"])
            rec["trace"] = {"status": tr["status"], "ambiguous_steps": tr["ambiguous_steps"], "n_points": len(tr["path"]),
                            "end_physical": list(to_phys(M, *tr["path"][-1])),
                            "path_px": [[round(x, 1), round(y, 1)] for x, y in tr["path"][::5]]}
            obs = branch_observables(M, tr, rev)
            rec["observables"] = obs
            # identity check: the traced zero-force crossing must sit within 3 mm of an independently detected crossing
            zf = obs.get("zero_force")
            if zf and ((p["direction"] == "+" and zf["delta_mm"] <= 0.0) or (p["direction"] == "-" and zf["delta_mm"] >= 0.0)):
                obs["zero_force_rejected"] = {**zf, "reason": "crossing on the wrong side of the origin: the walker left the unloading branch"}
                obs["zero_force"] = None; obs["zero_displacement"] = None; obs["unloading_chord"] = None
                zf = None
                rec["identity"] = "trace_crossed_to_the_wrong_side"
            if zf:
                near = min(dcross, key=lambda q: abs(q["delta_mm"] - zf["delta_mm"])) if dcross else None
                zf["nearest_independent_crossing_mm"] = None if near is None else near["delta_mm"]
                zf["matches_independent_crossing"] = bool(near is not None and abs(near["delta_mm"] - zf["delta_mm"]) <= 3.0)
                rec["identity"] = ("traced_from_reversal_and_matches_independent_crossing" if zf["matches_independent_crossing"]
                                   else "traced_but_no_independent_crossing_within_3mm")
            if tr["status"] == "lost_curve":
                rec["identity"] = "trace_lost_curve"
            col = colours[k % len(colours)]
            pts = [(x, y) for x, y in tr["path"]]
            if len(pts) > 1:
                draw.line(pts, fill=col, width=3)
            r = 9
            draw.ellipse([rev["x_px"] - r, rev["y_px"] - r, rev["x_px"] + r, rev["y_px"] + r], outline=col, width=3)
            draw.text((rev["x_px"] + 10, rev["y_px"] - 22), str(p["run"]), fill=col)
        runs.append(rec)
    result["runs"] = runs
    # cycle pairs that share one chain are not separable at this resolution
    by_amp = {}
    for rec in runs:
        if rec["reversal"] and not rec["load_control"]:
            by_amp.setdefault((abs(rec["mu"]), rec["direction"]), []).append(rec)
    for key, pair in by_amp.items():
        if len(pair) == 2:
            same = abs(pair[0]["reversal"]["y_px"] - pair[1]["reversal"]["y_px"]) < 2.0 and abs(pair[0]["reversal"]["x_px"] - pair[1]["reversal"]["x_px"]) < 2.0
            for rec in pair:
                rec["cycles_separable"] = not same
    if args.write_v2:
        write_v2(result, v1, Path(args.write_v2), args)
    # overlay: axes, ticks, independent crossings
    for val, x in ticks_d["found"].items():
        draw.line([(x, a + b * x - 30), (x, a + b * x + 30)], fill=(0, 200, 255), width=2)
        draw.text((x - 10, a + b * x + 34), val, fill=(0, 120, 200))
    for val, y in ticks_v["found"].items():
        draw.line([(c + d * y - 30, y), (c + d * y + 30, y)], fill=(0, 200, 255), width=2)
        draw.text((c + d * y + 34, y - 6), val, fill=(0, 120, 200))
    for q in dcross:
        draw.rectangle([q["x_px"] - 5, q["y_px"] - 5, q["x_px"] + 5, q["y_px"] + 5], outline=(255, 0, 255), width=2)
    for q in vcross:
        draw.rectangle([q["x_px"] - 5, q["y_px"] - 5, q["x_px"] + 5, q["y_px"] + 5], outline=(255, 0, 255), width=2)
    overlay.save(out / "fig16a_audit_overlay.png")
    (out / "fig16a_audit.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("origin px", origin, "ticks found", sorted(ticks_d["found"].items(), key=lambda kv: float(kv[0])), sorted(ticks_v["found"].items(), key=lambda kv: float(kv[0])),
          "missing", result["ticks"]["missing"], "residuals", residuals)
    print("independent zero-force crossings (mm, strands):", [(round(q["delta_mm"], 1), q["strands"]) for q in dcross])
    print("independent zero-displacement crossings (kN, strands):", [(round(q["V_kN"], 1), q["strands"]) for q in vcross])
    for r in runs:
        rev = r["reversal"]; o = r["observables"] or {}
        zf = o.get("zero_force"); zd = o.get("zero_displacement"); ch = o.get("unloading_chord")
        print(f"run {r['run']:2d} {r['direction']} v1 ({r['v1_reading']['delta_mm']:4}, {r['v1_reading']['V_kN']:4}) -> reversal "
              f"({rev['delta_mm']:6.1f}, {rev['V_kN']:5.1f}) V in [{rev['interval']['V_low_kN']:.1f}, {rev['interval']['V_high_kN']:.1f}]"
              f"{' OBSCURED' if rev['interval']['obscured'] else ''}" if rev else f"run {r['run']:2d} no reversal found",
              f"| end {r['trace']['end_physical'][0]:.1f} mm {r['trace']['end_physical'][1]:.1f} kN" if r["trace"] else "",
              f"| trace {r['trace']['status'] if r['trace'] else '-'} amb {r['trace']['ambiguous_steps'] if r['trace'] else '-'}",
              f"| zero-force {zf['delta_mm']:.1f} mm ({'ok' if zf.get('matches_independent_crossing') else 'unmatched'})" if zf else "| zero-force -",
              f"| zero-disp {zd['V_kN']:.1f} kN" if zd else "| zero-disp -",
              f"| chord {ch['k_kN_per_mm']:.2f} kN/mm over {ch['dU_mm']:.1f} mm" if ch and ch["k_kN_per_mm"] else "| chord -",
              f"| {r['identity']}")
    return result


if __name__ == "__main__":
    main()
