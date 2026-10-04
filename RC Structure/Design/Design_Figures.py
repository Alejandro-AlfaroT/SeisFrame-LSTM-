"""Figures of one uniform design record: the structure in 3D, the member cross sections with their reinforcement,
and the joint detailing with the joint shear table (verification stage ``figures``, 2026-10-02).

Everything is read from the saved record (sections, reinforcement, the capacity-design cages, joints, anchorage and
splices, the slab layout, the detailing zones). Nothing is recomputed and no model is built. A value the record does
not carry is left out of the drawing and named in ``notes``; the stage never fails a case over a missing label. A
grouped record (``design_mode`` "grouped") is not drawn here: its members differ by group and the figures of the
grouped design are a later extension.

    draw_design_figures(record, out_dir, label="case_0001") -> {"files": [...], "notes": [...]}
"""
from __future__ import annotations

import math
from pathlib import Path

FILES = ("structure_3d.png", "reinforcement_detailing.png", "joint_detailing.png")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _bar(ax, x, y, dia, Circle):
    ax.add_patch(Circle((x, y), dia / 2.0, facecolor="black", edgecolor="black"))


def _hoop(ax, x0, y0, w, h, FancyBboxPatch, lw=2.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                fill=False, lw=lw, edgecolor="black", ls=ls))


def _caption(ax, text, size=8.4):
    ax.text(0.5, -0.02, text, transform=ax.transAxes, ha="center", va="top", fontsize=size, linespacing=1.4)


def _dim(ax, p0, p1, text, offset, rotation=0, ha="center", va="top"):
    ax.annotate("", p0, p1, arrowprops=dict(arrowstyle="<->", lw=0.9))
    mid = ((p0[0] + p1[0]) / 2.0 + offset[0], (p0[1] + p1[1]) / 2.0 + offset[1])
    ax.text(mid[0], mid[1], text, ha=ha, va=va, rotation=rotation, fontsize=8.5)


def _column_bars(record):
    """Bar centroids of the column section as (x along b, y along h), from the capacity-design cage when saved."""
    s, r = record["sections"], record["reinforcement"]
    bc, hc, c = s["b_col_in"], s["h_col_in"], r["col_longitudinal_centroid_offset_in"]
    cage = ((record.get("capacity_design") or {}).get("columns") or {}).get("cage") or {}
    faces = cage.get("faces") or {}
    xs = list(faces.get("b_face") or [])
    ys = list(faces.get("h_face") or [])
    if not xs or not ys:
        n_face, n_side = r["col_top_bars"], r["col_side_bars"]
        xs = [c + k * (bc - 2 * c) / (n_face - 1) for k in range(n_face)] if n_face > 1 else [bc / 2.0]
        ys = [c] + [c + k * (hc - 2 * c) / (n_side + 1) for k in range(1, n_side + 1)] + [hc - c]
    bars = [(x, ys[0]) for x in xs] + [(x, ys[-1]) for x in xs] + [(xs[0], y) for y in ys[1:-1]] + [(xs[-1], y) for y in ys[1:-1]]
    support = cage.get("minimal_support") or {}
    ties_x = [p for p, ok in zip(xs[1:-1], (support.get("b_face") or {}).get("supported", [True] * len(xs))[1:-1]) if ok]
    ties_y = [p for p, ok in zip(ys[1:-1], (support.get("h_face") or {}).get("supported", [True] * len(ys))[1:-1]) if ok]
    return bars, xs, ys, ties_x, ties_y, bool(faces)


def _beam_layers(record, axis="x"):
    """[(face, offset_from_face_in, count)] of the beam bars of one framing direction."""
    r = record["reinforcement"]
    stacking = (r.get("beam_bar_stacking") or {}).get("layers") or {}
    rows = []
    for face, count in (("top", r["beam_top_bars"]), ("bottom", r["beam_bot_bars"])):
        entry = (stacking.get(axis) or {}).get(face) or {}
        offsets, per_layer = entry.get("offsets_in"), entry.get("per_layer")
        if offsets and per_layer:
            rows.extend((face, float(o), int(n)) for o, n in zip(offsets, per_layer))
        else:
            rows.append((face, float(r["beam_longitudinal_centroid_offset_in"]), int(count)))
    return rows


def _figure_structure(record, out, label, notes):
    plt = _plt()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    g, s, r = record["geometry"], record["sections"], record["reinforcement"]
    bx, by, nf = g["num_bay_x"], g["num_bay_y"], g["num_floor"]
    Lx, Ly, H = g["bay_x_in"] / 12.0, g["bay_y_in"] / 12.0, g["story_h_in"] / 12.0
    slab = (record.get("slab") or {}).get("thickness_in")
    fig = plt.figure(figsize=(13, 8.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    for k in range(nf):
        for j in range(by + 1):
            for i in range(bx + 1):
                ax.plot([i * Lx] * 2, [j * Ly] * 2, [k * H, (k + 1) * H], color="0.2", lw=1.8)
    for k in range(1, nf + 1):
        z = k * H
        for j in range(by + 1):
            ax.plot([0, bx * Lx], [j * Ly] * 2, [z, z], color="0.45", lw=1.1)
        for i in range(bx + 1):
            ax.plot([i * Lx] * 2, [0, by * Ly], [z, z], color="0.45", lw=1.1)
        ax.add_collection3d(Poly3DCollection([[(0, 0, z), (bx * Lx, 0, z), (bx * Lx, by * Ly, z), (0, by * Ly, z)]],
                                             facecolors="0.85", edgecolors="none", alpha=0.25))
    ax.set_box_aspect((bx * Lx, by * Ly, nf * H))
    ax.set_xlabel("x (ft)"); ax.set_ylabel("y (ft)"); ax.set_zlabel("z (ft)")
    ax.view_init(elev=22, azim=-55)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.0)
    ax.grid(False)
    period = (record.get("demand") or {}).get("model_period_sec")
    seismic = record.get("seismic") or {}
    ax.set_title(f"{label}: {bx} x {by} bays ({Lx:g} ft x {Ly:g} ft), {nf} stories at {H:g} ft; "
                 f"{(bx + 1) * (by + 1)} columns per story, every grid line a special moment frame\n"
                 f"columns {s['b_col_in']:g} x {s['h_col_in']:g} in (f'c {s['fc_col_ksi']:g} ksi), beams {s['b_beam_in']:g} x {s['h_beam_in']:g} in "
                 f"(f'c {s['fc_beam_ksi']:g} ksi)" + (f", slab {slab:g} in" if slab else "")
                 + f"\n{2 * r['col_top_bars'] + 2 * r['col_side_bars']} #{r['col_bar_size']} column bars, "
                 f"{r['beam_top_bars']} + {r['beam_bot_bars']} #{r['beam_bar_size']} beam bars; "
                 + (f"T1 = {period:.3f} s; " if isinstance(period, (int, float)) else "")
                 + f"site {seismic.get('site_label')}, SDS {seismic.get('sds')}, SD1 {seismic.get('sd1')}", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out / "structure_3d.png", dpi=150)
    plt.close(fig)


def _figure_reinforcement(record, out, label, notes):
    plt = _plt()
    from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
    s, r = record["sections"], record["reinforcement"]
    det = record.get("detailing") or {}
    cd = record.get("capacity_design") or {}
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.6), gridspec_kw={"width_ratios": [1.0, 1.1, 1.5]})

    # ---- column section
    ax = axes[0]
    bc, hc = s["b_col_in"], s["h_col_in"]
    cover, db = r["col_clear_cover_in"], r["col_bar_diameter_in"]
    bars, xs, ys, ties_x, ties_y, from_cage = _column_bars(record)
    ax.add_patch(Rectangle((0, 0), bc, hc, facecolor="0.92", edgecolor="black", lw=1.5))
    _hoop(ax, cover, cover, bc - 2 * cover, hc - 2 * cover, FancyBboxPatch)
    for x, y in bars:
        _bar(ax, x, y, db, Circle)
    for x in ties_x:
        ax.plot([x, x], [ys[0], ys[-1]], color="black", lw=1.3, ls="--")
    for y in ties_y:
        ax.plot([xs[0], xs[-1]], [y, y], color="black", lw=1.3, ls="--")
    _dim(ax, (0, -2.5), (bc, -2.5), f"{bc:g} in", (0, -1.7))
    _dim(ax, (-2.5, 0), (-2.5, hc), f"{hc:g} in", (-1.9, 0), rotation=90, ha="right", va="center")
    ax.set_xlim(-9, bc + 5); ax.set_ylim(-9, hc + 3); ax.set_aspect("equal"); ax.axis("off")
    n_bars = 2 * r["col_top_bars"] + 2 * r["col_side_bars"]
    rho = n_bars * r["col_bar_area_in2"] / (bc * hc)
    legs = r.get("col_stirrup_legs_by_direction") or {}
    bond = cd.get("column_bar_bond") or {}
    bond_text = (f"; 18.7.4.3: 1.25 ld = {bond['factored_ld_in']:.1f} in <= {bond['half_clear_height_in']:.1f} in"
                 if bond.get("factored_ld_in") else "")
    ax.set_title("Column section", fontsize=11, fontweight="bold")
    _caption(ax, f"{n_bars} #{r['col_bar_size']} bars (rho = {rho * 100:.2f} %), f'c {s['fc_col_ksi']:g} ksi, "
                 f"{(record.get('materials') or {}).get('reinforcement_specification', 'Grade 60')}\n"
                 f"#{r['col_stirrup_bar_size']} hoops at {r['col_stirrup_spacing_in']:g} in, legs {legs.get('across_b_face', r['col_stirrup_legs'])} across the b faces "
                 f"and {legs.get('across_h_face', r['col_stirrup_legs'])} across the h faces\n"
                 f"clear cover {cover:g} in outside the hoops; bar centroid {r['col_longitudinal_centroid_offset_in']:g} in from the face"
                 + ("; bar positions from the saved cage" if from_cage else "; bars evenly spaced (no cage saved)")
                 + "\ndashed = crossties on the supported interior bars" + bond_text)

    # ---- beam section with slab (the x direction's layers; the y direction mirrors them at the joint)
    ax = axes[1]
    bw, hb = s["b_beam_in"], s["h_beam_in"]
    ts = (record.get("slab") or {}).get("thickness_in") or 0.0
    bcov, sd, bdb = r["beam_clear_cover_in"], r["beam_stirrup_diameter_in"], r["beam_bar_diameter_in"]
    flange = max(16.0, 0.8 * bw)
    if ts:
        ax.add_patch(Rectangle((-flange, hb - ts), bw + 2 * flange, ts, facecolor="0.95", edgecolor="black", lw=1.2))
    ax.add_patch(Rectangle((0, 0), bw, hb, facecolor="0.92", edgecolor="black", lw=1.5))
    _hoop(ax, bcov, bcov, bw - 2 * bcov, hb - 2 * bcov, FancyBboxPatch)
    beam_cage = (cd.get("beams") or {}).get("cage") or {}
    top_support = (beam_cage.get("minimal_support") or {}).get("top") or {}
    for pos, ok in zip(top_support.get("positions_in") or [], top_support.get("supported") or []):
        if ok and bcov + sd + bdb < pos < bw - bcov - sd - bdb:
            ax.plot([pos, pos], [bcov, hb - bcov], color="black", lw=1.3, ls="--")
    inner = bw - 2 * (bcov + sd)
    layers = _beam_layers(record, "x")
    for face, offset, n in layers:
        y = hb - offset if face == "top" else offset
        positions = [bcov + sd + bdb / 2 + k * (inner - bdb) / (n - 1) for k in range(n)] if n > 1 else [bw / 2.0]
        for x in positions:
            _bar(ax, x, y, bdb, Circle)
    layout = (record.get("slab_reinforcement") or {}).get("layout") or {}
    mats = layout.get("layers") or {}
    sl = mats.get("x_top") or {}
    if ts and sl.get("spacing_in"):
        scov = sl.get("clear_cover_outer_mat_in", 0.75)
        x = -flange + 2.0
        while x < bw + flange:
            _bar(ax, x, hb - scov - 0.25, 0.5, Circle)
            _bar(ax, x, hb - ts + scov + 0.25, 0.5, Circle)
            x += sl["spacing_in"]
    _dim(ax, (0, -2.5), (bw, -2.5), f"{bw:g} in", (0, -1.7))
    _dim(ax, (bw + flange + 2, 0), (bw + flange + 2, hb), f"{hb:g} in", (1.6, 0), rotation=90, va="center", ha="left")
    if ts:
        _dim(ax, (-flange - 2, hb - ts), (-flange - 2, hb), f"{ts:g} in", (-1.6, 0), rotation=90, va="center", ha="right")
    ax.set_xlim(-flange - 8, bw + flange + 8); ax.set_ylim(-9, hb + 3); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Beam section with slab (x beams)", fontsize=11, fontweight="bold")
    layer_text = ", ".join(f"{face} {n} at {offset:g} in" for face, offset, n in layers)
    d_eff = (det.get("beam") or {}).get("effective_depth_in")
    _caption(ax, f"{r['beam_top_bars']} #{r['beam_bar_size']} top and {r['beam_bot_bars']} bottom, f'c {s['fc_beam_ksi']:g} ksi; layers: {layer_text}\n"
                 f"#{r['beam_stirrup_bar_size']} stirrups, {r['beam_stirrup_legs']} legs, at {r['beam_stirrup_spacing_in']:g} in; clear cover {bcov:g} in"
                 + (f"; d = {d_eff:g} in" if isinstance(d_eff, (int, float)) else "")
                 + (f"\nslab {ts:g} in, #{sl.get('bar_size')} at {sl.get('spacing_in'):g} in each way, top and bottom mats, cover {sl.get('clear_cover_outer_mat_in', 0.75):g} in"
                    if ts and sl.get("spacing_in") else "\nslab mats not drawn (no layout saved)")
                 + "\n(slab width schematic, not the effective flange)")

    # ---- elevation at a joint: hoop zones
    ax = axes[2]
    beam_det, col_det = det.get("beam") or {}, det.get("column") or {}
    zone_b = beam_det.get("end_zone_length_in") or 2.0 * hb
    zone_c = col_det.get("end_zone_length_in") or max(hc, 18.0)
    Lb, Lc = 1.45 * zone_b, 1.3 * zone_c
    xj, zj = 0.0, 0.0
    ax.add_patch(Rectangle((xj - hc / 2, zj - hb / 2 - Lc), hc, Lc, facecolor="0.92", edgecolor="black"))
    ax.add_patch(Rectangle((xj - hc / 2, zj + hb / 2), hc, Lc, facecolor="0.92", edgecolor="black"))
    ax.add_patch(Rectangle((xj - hc / 2 - Lb, zj - hb / 2), Lb, hb, facecolor="0.92", edgecolor="black"))
    ax.add_patch(Rectangle((xj + hc / 2, zj - hb / 2), Lb, hb, facecolor="0.92", edgecolor="black"))
    ax.add_patch(Rectangle((xj - hc / 2, zj - hb / 2), hc, hb, facecolor="0.85", edgecolor="black"))
    sb = beam_det.get("hoop_spacing_in") or r["beam_stirrup_spacing_in"]
    fh = beam_det.get("first_hoop_distance_in", 2.0)
    for side in (-1, 1):
        x = xj + side * (hc / 2 + fh)
        while abs(x - xj) < hc / 2 + Lb - 2:
            lw = 1.6 if abs(x - xj) <= hc / 2 + zone_b else 0.8
            ax.plot([x, x], [zj - hb / 2 + bcov, zj + hb / 2 - bcov], color="black", lw=lw)
            x += side * sb
    for side, zl, va in ((-1, zj + hb / 2 + 3, "bottom"), (1, zj - hb / 2 - 3, "top")):
        ax.annotate("", (xj + side * hc / 2, zl), (xj + side * (hc / 2 + zone_b), zl), arrowprops=dict(arrowstyle="<->", lw=0.9))
        ax.text(xj + side * (hc / 2 + zone_b / 2), zl + (1.2 if va == "bottom" else -1.2),
                f"2h = {zone_b:g} in: #{r['beam_stirrup_bar_size']} at {sb:g} in, first at {fh:g} in", ha="center", va=va, fontsize=7.5)
    sc = col_det.get("hoop_spacing_in") or r["col_stirrup_spacing_in"]
    for side in (-1, 1):
        z = zj + side * (hb / 2 + 2.0)
        while abs(z - zj) < hb / 2 + Lc - 2:
            lw = 1.6 if abs(z - zj) <= hb / 2 + zone_c else 0.8
            ax.plot([xj - hc / 2 + cover, xj + hc / 2 - cover], [z, z], color="black", lw=lw)
            z += side * sc
        ax.annotate("", (xj - hc / 2 - 3, zj + side * hb / 2), (xj - hc / 2 - 3, zj + side * (hb / 2 + zone_c)),
                    arrowprops=dict(arrowstyle="<->", lw=0.9))
        ax.text(xj - hc / 2 - 5, zj + side * (hb / 2 + zone_c / 2), f"l0 = {zone_c:g} in\n#{r['col_stirrup_bar_size']} at {sc:g} in",
                ha="right", va="center", fontsize=7.5)
    jh = (((cd.get("joints") or {}).get("joint_transverse") or {}).get("hoops")) or {}
    if jh.get("spacing_in"):
        z = zj - hb / 2 + 1.5
        while z < zj + hb / 2 - 1.0:
            ax.plot([xj - hc / 2 + cover, xj + hc / 2 - cover], [z, z], color="black", lw=1.6, ls=":")
            z += jh["spacing_in"]
        jlegs = jh.get("legs") or {}
        ax.text(xj + hc / 2 + 4, zj + hb / 2 + 8, f"joint: #{jh.get('bar_size')} at {jh['spacing_in']:g} in, "
                f"{jlegs.get('across_b_face', '?')} / {jlegs.get('across_h_face', '?')} legs", ha="left", va="bottom", fontsize=7.5)
        ax.annotate("", (xj + hc / 2 + 6, zj + hb / 2 + 7.5), (xj + hc / 4, zj + hb / 2 - 2), arrowprops=dict(arrowstyle="->", lw=0.8))
    else:
        notes.append("joint transverse hoops not in the record; joint hoops not drawn")
    ax.set_xlim(xj - hc / 2 - Lb - 24, xj + hc / 2 + Lb + 4); ax.set_ylim(zj - hb / 2 - Lc - 4, zj + hb / 2 + Lc + 4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Transverse reinforcement at a joint (elevation)", fontsize=11, fontweight="bold")
    _caption(ax, "Heavy lines: hoops in the recorded end zones (beam 2h from the joint face, column l0); dotted: joint hoops.\n"
                 "The analysis and the saved cage use the end-zone spacing over the whole member.")
    mats_ = record.get("materials") or {}
    fig.suptitle(f"{label} reinforcement detailing (design record {record.get('schema_version')}, "
                 f"{mats_.get('reinforcement_specification', '')}, aggregate {mats_.get('aggregate_size_in', '?')} in)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out / "reinforcement_detailing.png", dpi=150)
    plt.close(fig)


def _figure_joint(record, out, label, notes):
    plt = _plt()
    from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
    s, r = record["sections"], record["reinforcement"]
    cd = record.get("capacity_design") or {}
    joints = (cd.get("joints") or {})
    cats = joints.get("joints") or {}
    jh = (joints.get("joint_transverse") or {}).get("hoops") or {}
    anch = ((cd.get("anchorage") or {}).get("directions") or {}).get("x") or {}
    splices = cd.get("splices") or {}
    bc, hc, bw, hb = s["b_col_in"], s["h_col_in"], s["b_beam_in"], s["h_beam_in"]
    cover, hoop_d, db = r["col_clear_cover_in"], r["col_stirrup_diameter_in"], r["col_bar_diameter_in"]
    bdb = r["beam_bar_diameter_in"]
    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], width_ratios=[1.0, 1.25])
    axp, axe, axt = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    # ---- (a) plan section through an interior joint at beam mid-depth
    ax = axp
    ext = 22.0
    for (x0, y0, w, h) in ((-ext, (bc - bw) / 2, ext, bw), (hc, (bc - bw) / 2, ext, bw),
                           ((hc - bw) / 2, -ext, bw, ext), ((hc - bw) / 2, bc, bw, ext)):
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor="0.88", edgecolor="black", lw=1.0, hatch="//"))
    ax.add_patch(Rectangle((0, 0), hc, bc, facecolor="0.94", edgecolor="black", lw=1.6))
    _hoop(ax, cover, cover, hc - 2 * cover, bc - 2 * cover, FancyBboxPatch)
    bars, xs, ys, ties_x, ties_y, _from_cage = _column_bars(record)
    # the plan is drawn with h along the page x axis: swap the section's axes
    for x, y in bars:
        _bar(ax, y, x, db, Circle)
    for y in ties_x:
        ax.plot([ys[0], ys[-1]], [y, y], color="black", lw=1.3, ls="--")
    for x in ties_y:
        ax.plot([x, x], [xs[0], xs[-1]], color="black", lw=1.3, ls="--")
    threading = ((cd.get("bar_threading") or {}).get("by_direction") or {}).get("x") or {}
    positions = threading.get("threaded_positions_in") or []
    if not positions:
        inner = bw - 2 * (r["beam_clear_cover_in"] + r["beam_stirrup_diameter_in"])
        n = r["beam_top_bars"]
        positions = [(bc - bw) / 2 + r["beam_clear_cover_in"] + r["beam_stirrup_diameter_in"] + bdb / 2 + k * (inner - bdb) / max(n - 1, 1)
                     for k in range(n)]
    for y in positions:
        ax.plot([-ext, hc + ext], [y, y], color="black", lw=0.9, alpha=0.75)
    x_ext = (bc - bw) / 2
    bj = min(bc, bw + hc, bw + 2 * x_ext)
    ax.add_patch(Rectangle((0, (bc - bj) / 2), hc, bj, facecolor="none", edgecolor="black", lw=1.2, ls=":"))
    _dim(ax, (0, -ext - 3), (hc, -ext - 3), f"{hc:g} in", (0, -1.5))
    _dim(ax, (hc + ext + 3, (bc - bw) / 2), (hc + ext + 3, (bc + bw) / 2), f"{bw:g} in beam", (1.5, 0), rotation=90, va="center", ha="left")
    ax.set_xlim(-ext - 4, hc + ext + 12); ax.set_ylim(-ext - 10, bc + ext + 2); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Plan section through an interior joint at beam mid-depth", fontsize=11, fontweight="bold")
    jlegs, jcross = jh.get("legs") or {}, jh.get("crossties_per_direction") or {}
    confined = bw / bc >= 0.75
    _caption(ax, f"column {hc:g} x {bc:g} in with its {len(bars)} #{r['col_bar_size']} bars; beams {bw:g} in wide on all four faces, "
                 f"#{r['beam_bar_size']} beam bars pass through ({len(positions)} per layer)\n"
                 + (f"joint hoops #{jh.get('bar_size')} at {jh.get('spacing_in'):g} in, {jlegs.get('across_b_face', '?')} / {jlegs.get('across_h_face', '?')} legs "
                    f"({jcross.get('across_b_face', '?')} / {jcross.get('across_h_face', '?')} crossties); Ash provided "
                    f"{jh.get('ash_provided_per_in', float('nan')):.3f} vs required {jh.get('ash_required_per_in', float('nan')):.3f} in2/in\n"
                    if jh.get("spacing_in") else "joint hoops not in the record\n")
                 + f"dotted: effective joint area Aj = bj h = {bj:g} x {hc:g} = {bj * hc:g} in2, bj = min(b_col, b_w + h, b_w + 2x) "
                 f"= min({bc:g}, {bw + hc:g}, {bw + 2 * x_ext:g})\n"
                 f"beam width / column face = {bw / bc:.2f} {'>=' if confined else '<'} 0.75: transverse beams "
                 f"{'can' if confined else 'do not'} confine the joint (15.2.8)")

    # ---- (b) elevation: exterior joint with hooked beam bars, interior joint with through bars
    ax = axe
    Lb, Lc = 44.0, 40.0
    cb = r["beam_longitudinal_centroid_offset_in"]
    tail = 12 * bdb

    def joint_block(x0, left_beam, right_beam):
        ax.add_patch(Rectangle((x0, -hb / 2 - Lc), hc, Lc, facecolor="0.94", edgecolor="black"))
        ax.add_patch(Rectangle((x0, hb / 2), hc, Lc, facecolor="0.94", edgecolor="black"))
        if left_beam:
            ax.add_patch(Rectangle((x0 - Lb, -hb / 2), Lb, hb, facecolor="0.94", edgecolor="black"))
        if right_beam:
            ax.add_patch(Rectangle((x0 + hc, -hb / 2), Lb, hb, facecolor="0.94", edgecolor="black"))
        ax.add_patch(Rectangle((x0, -hb / 2), hc, hb, facecolor="0.86", edgecolor="black"))
        c = r["col_longitudinal_centroid_offset_in"]
        for xb in (x0 + c, x0 + hc - c):
            ax.plot([xb, xb], [-hb / 2 - Lc, hb / 2 + Lc], color="black", lw=1.6)
        if jh.get("spacing_in"):
            z = -hb / 2 + 1.5
            while z < hb / 2 - 1.0:
                ax.plot([x0 + cover, x0 + hc - cover], [z, z], color="black", lw=1.4, ls=":")
                z += jh["spacing_in"]

    X1 = 0.0
    joint_block(X1, False, True)
    for zb, sign in ((hb / 2 - cb, -1), (-hb / 2 + cb, 1)):
        x_end = X1 + cover + hoop_d + bdb / 2
        ax.plot([X1 + hc + Lb, x_end + 2.0], [zb, zb], color="black", lw=1.6)
        ax.plot([x_end + 2.0, x_end], [zb, zb + sign * 2.0], color="black", lw=1.6)
        ax.plot([x_end, x_end], [zb + sign * 2.0, zb + sign * (2.0 + tail)], color="black", lw=1.6)
    if anch.get("ldh_required_in") is not None:
        ax.annotate("", (X1 + hc - cover - hoop_d, hb / 2 + 4), (X1 + cover + hoop_d, hb / 2 + 4), arrowprops=dict(arrowstyle="<->", lw=0.9))
        ax.text(X1 + hc / 2, hb / 2 + 5.5, f"embedment {anch.get('embedment_available_in', float('nan')):.1f} in\n(ldh required {anch['ldh_required_in']:.1f} in)",
                ha="center", va="bottom", fontsize=7.5)
    else:
        notes.append("anchorage evidence not in the record; ldh not labelled")
    ax.text(X1 + hc / 2 + 14, -hb / 2 - Lc - 3, f"Exterior joint: #{r['beam_bar_size']} beam bars with a 90-degree hook\n"
            f"into the confined core (18.8.5.1), tail 12db = {tail:g} in", ha="center", va="top", fontsize=8)
    X2 = hc + Lb + 62.0
    joint_block(X2, True, True)
    for zb in (hb / 2 - cb, -hb / 2 + cb):
        ax.plot([X2 - Lb, X2 + hc + Lb], [zb, zb], color="black", lw=1.6)
    ax.annotate("", (X2, hb / 2 + 4), (X2 + hc, hb / 2 + 4), arrowprops=dict(arrowstyle="<->", lw=0.9))
    through = anch.get("through_bar_depth_required_in")
    ax.text(X2 + hc / 2, hb / 2 + 5.5, f"h = {hc:g} in" + (f"\n(20db = {through:g} in needed)" if through else ""), ha="center", va="bottom", fontsize=7.5)
    ax.text(X2 + hc / 2, -hb / 2 - Lc - 3, "Interior joint: beam bars continuous through the joint\n"
            "(18.8.2.3); one bar family per direction", ha="center", va="top", fontsize=8)
    ax.set_xlim(X1 - 4, X2 + hc + Lb + 4); ax.set_ylim(-hb / 2 - Lc - 14, hb / 2 + Lc + 2)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Beam bar anchorage at the joints (elevation)", fontsize=11, fontweight="bold")
    col_s, beam_s = splices.get("column") or {}, splices.get("beam") or {}
    col_text = (f"Class B laps ({col_s['class_b_lap_in']:.0f} in) in the center half of the clear height (18.7.4.4)"
                if col_s.get("lap_splice_feasible") else
                f"mechanical splices ({col_s.get('splice_type', 'type not saved')}); a Class B lap would need {col_s.get('class_b_lap_in', float('nan')):.0f} in")
    beam_text = (f"Class B laps ({beam_s.get('class_b_lap_top_in', float('nan')):.0f} in) between the hinge zones"
                 if beam_s.get("lap_splice_feasible") else
                 f"laps do not fit between the hinge zones, so mechanical splices (18.2.7)")
    _caption(ax, f"Column bars: one cage over the height, {col_text}.\nBeam bars: {beam_text}.")

    # ---- (c) joint shear table
    ax = axt
    ax.axis("off")
    rows = []
    for key in sorted(cats):
        cat = cats[key]
        parts = key.split("/")
        level, kind, axis = (parts[1:] + ["", "", ""])[:3]
        phi_vn = cat.get("phi_vn_kip") or float("nan")
        rows.append([f"{level} {kind.replace('_', ' ')}", axis, f"{cat.get('beams_in_direction', '?')} / {cat.get('beams_perpendicular', '?')}",
                     f"{cat.get('gamma', float('nan')):g}", f"{cat.get('nominal_vn_kip', float('nan')):.0f}", f"{phi_vn:.0f}",
                     f"{cat.get('vj_kip', float('nan')):.0f}", f"{(cat.get('vj_kip') or float('nan')) / phi_vn:.2f}",
                     (cat.get("gamma_basis") or "")[:60]])
    if rows:
        cols = ["joint", "plane", "beams in dir / transv.", "gamma", "Vn (kip)", "phi Vn (kip)", "Vj (kip)", "Vj / phi Vn", "Table 18.8.4.3 row"]
        table = ax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center", colLoc="center",
                         colWidths=[0.11, 0.05, 0.11, 0.06, 0.08, 0.09, 0.08, 0.08, 0.34])
        table.auto_set_font_size(False)
        table.set_fontsize(8.0)
        table.scale(1.0, 1.3)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_facecolor("white")
            if row == 0:
                cell.set_text_props(fontweight="bold")
    else:
        notes.append("no joint shear categories in the record; table left empty")
    ax.set_title("Joint shear by category (ACI 318-19 18.8.4.3, phi = 0.85)", fontsize=11, fontweight="bold")
    ax.text(0.5, 0.0, "Vj = probable beam face forces minus the column shear (capacity design). The row is the record's own "
            "classification under its declared continuity; the\nphysical cage that realises it is independent verification, not this drawing. "
            "V2 profile: rigid centerline joints in the nonlinear model, joint design checks retained.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.2, style="italic", linespacing=1.4)
    fig.suptitle(f"{label} joint detailing and joint shear (design record)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "joint_detailing.png", dpi=150)
    plt.close(fig)


def draw_design_figures(record, out_dir, label=None):
    """Write the three figures of a uniform design record into ``out_dir``; returns {"files", "notes"}."""
    if record.get("design_mode") == "grouped" or "sections" not in record:
        raise ValueError("Design figures are drawn for uniform design records only; a grouped record has no one section.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    label = label or (record.get("geometry") or {}).get("name") or "design"
    notes = []
    _figure_structure(record, out, label, notes)
    _figure_reinforcement(record, out, label, notes)
    _figure_joint(record, out, label, notes)
    return {"files": [str(out / name) for name in FILES], "notes": notes}
