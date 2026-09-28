"""Cage geometry and joint assembly review of one design record (2026-09-27 detailing review, items 1-2).

    python -m tools.review_cage_geometry --design <case>/design.json --output-root <folder>

Writes cage_geometry.json, cage_geometry.md and three drawings: the column and beam cages with their
hoops, crossties and hook orientations for two successive sets; the joint plan with the beam bars threaded
between the column bars; and the joint elevation with the stacked orthogonal layers and the slab mats.
Read-only on the record. Everything comes from Design/SMRF_Cage_Geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

RC = Path(__file__).resolve().parents[1]
if str(RC) not in sys.path:
    sys.path.insert(0, str(RC))

from Design.SMRF_Cage_Geometry import bar_diameter, evaluate_cage_geometry  # noqa: E402


def markdown(res, record, design_path):
    s, r = record["sections"], record["reinforcement"]
    lines = [f"**Cage geometry review of {design_path}**", "",
             f"Column {s['b_col_in']:g} x {s['h_col_in']:g} in, {2 * r['col_top_bars'] + 2 * r['col_side_bars']} #{r['col_bar_size']}, "
             f"#{r['col_stirrup_bar_size']} hoops at {r['col_stirrup_spacing_in']:g} in; beam {s['b_beam_in']:g} x {s['h_beam_in']:g} in, "
             f"{r['beam_top_bars']} #{r['beam_bar_size']} top and {r['beam_bot_bars']} bottom, #{r['beam_stirrup_bar_size']} stirrups at "
             f"{r['beam_stirrup_spacing_in']:g} in.", ""]
    lines += ["**Item 1: hoops, crossties, hooks**", ""]
    for name, part in (("Column", res["column"]), ("Beam", res["beam"])):
        arr = part["arrangement"]
        set0 = part["tie_sets"][0]
        lines.append(f"{name}: constructible {arr.get('constructible')}, hx {arr.get('hx_in')} in; perimeter hoop out-to-out "
                     f"{set0['hoop']['out_to_out_in'][0]:g} x {set0['hoop']['out_to_out_in'][1]:g} in, inside bend {set0['hoop']['inside_bend_diameter_in']:g} in, "
                     f"seismic hooks with {set0['hoop']['hook_extension_in']:g} in extension; {len(set0['crossties'])} crossties per set "
                     f"(cut length {set0['crossties'][0]['cut_length_in']:.1f} in, 90-degree end alternating {[t['ninety_degree_end'] for t in set0['crossties']]} "
                     f"then {[t['ninety_degree_end'] for t in part['tie_sets'][1]['crossties']]}); clear spacing minimum "
                     f"{min(min(v) for v in part['clearances_in'].values() if v):.2f} in against {part['clear_limit_in']:.2f} in."
                     if set0["crossties"] else f"{name}: constructible {arr.get('constructible')}, no crossties.")
        for c in part["checks"]:
            lines.append(f"- {'pass' if c['passes'] else 'FAIL'}: {c['rule']}" + (f" ({c.get('face', c.get('direction', ''))})" if c.get('face') or c.get('direction') else "")
                         + (f": {c['detail']}" if isinstance(c.get("detail"), str) else ""))
        lines.append("")
    lines += ["**Item 2: the assembled joint**", "", f"Stacking convention selected: {res['stacking_selected']} "
              f"(the one with fewer failed checks; both are in the JSON).", ""]
    j = res["joint"]
    for axis, d in j["directions"].items():
        lines.append(f"- {axis}-beams: column bar lanes at {[round(v, 1) for v in d['column_bar_lanes_blocked_in']]} in across the "
                     f"{d['beam_band_in'][0]:g}-{d['beam_band_in'][1]:g} in beam band; {d['bars_per_layer_that_fit']} of {r['beam_top_bars']} bars fit per layer "
                     f"(threaded at {[round(v, 1) for v in d['threaded_positions_in']]} in); layers needed {d['layers_needed']}"
                     + (f"; two layers move the group centroid to {d['two_layer_centroid_from_face_in']:.2f} in (lever-arm ratio {d['two_layer_lever_arm_ratio']:.3f})"
                        if d.get("two_layer_centroid_from_face_in") else ""))
    st = j["stacking"]
    lines.append(f"- Stacking: {st['upper_direction']} bars on top at {st['upper_layer_centroid_from_face_in']:g} in, {st['lower_direction']} bars at "
                 f"{st['lower_layer_centroid_from_face_in']:g} in; effective depth {st['effective_depth_nominal_in']:g} -> {st['effective_depth_lower_in']:g} in "
                 f"for the lower direction, lever-arm ratio {st['lower_direction_lever_arm_ratio']:.3f}. {st['strength_basis_note']}.")
    lines.append(f"- Slab mats (centroid from the top face): " + ", ".join(f"{m['mat']} {m['centroid_from_top_in']:g} in" for m in j["slab_mats"])
                 + f"; overlaps with crossing beam bars: {len(j['slab_clashes'])}; tight crossings under 1 in: {len(j['slab_tight_crossings'])}.")
    h = j["exterior_hooks"]
    lines.append(f"- Exterior joints: #{r['beam_bar_size']} 90-degree hooks (bend {h['hook']['inside_bend_diameter_in']:g} in, tail {h['hook']['extension_in']:g} in) "
                 f"reach {h['tail_reaches_from_top_face_in']:.1f} in from the top face inside the {h['joint_depth_in']:g} in joint; straight embedment "
                 f"{h['straight_embedment_to_hook_in']:.1f} in against ldh {h['ldh_required_in']}.")
    lines.append("")
    for c in j["checks"]:
        detail = c["detail"] if isinstance(c.get("detail"), str) else json.dumps(c.get("detail"))
        lines.append(f"- {'pass' if c['passes'] else 'FAIL'}: {c['rule']}" + (f" ({c['direction']})" if c.get("direction") else "") + f": {detail}")
    lines += ["", f"Scope: {res['scope']}", ""]
    return "\n".join(lines)


def draw(res, record, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Circle, FancyBboxPatch, Rectangle

    s, r = record["sections"], record["reinforcement"]
    bc, hc, bw, hb = s["b_col_in"], s["h_col_in"], s["b_beam_in"], s["h_beam_in"]
    col_db, col_tie = bar_diameter(r["col_bar_size"]), bar_diameter(r["col_stirrup_bar_size"])
    beam_db, beam_tie = bar_diameter(r["beam_bar_size"]), bar_diameter(r["beam_stirrup_bar_size"])

    def hook_mark(ax, x, y, kind, direction, size):
        """A small mark for a hook end: 135-degree seismic hooks as a filled dot with a tail, 90-degree as an open square."""
        if kind == "seismic_135":
            ax.add_patch(Circle((x, y), size * 0.35, facecolor="black"))
            ax.plot([x, x + direction[0] * size], [y, y + direction[1] * size], color="black", lw=1.2)
        else:
            ax.add_patch(Rectangle((x - size * 0.3, y - size * 0.3), size * 0.6, size * 0.6, fill=False, lw=1.2))

    # ---- figure 1: column and beam cages, two successive tie sets each
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    for row, (name, part, w, h, cover, tie_db, db) in enumerate((("Column", res["column"], hc, bc, r["col_clear_cover_in"], col_tie, col_db),
                                                                  ("Beam", res["beam"], bw, hb, r["beam_clear_cover_in"], beam_tie, beam_db))):
        arr = part["arrangement"]
        c = cover + tie_db + db / 2
        for col, tset in enumerate(part["tie_sets"]):
            ax = axes[row, col]
            ax.add_patch(Rectangle((0, 0), w, h, facecolor="0.94", edgecolor="black", lw=1.4))
            ax.add_patch(FancyBboxPatch((cover, cover), w - 2 * cover, h - 2 * cover, boxstyle="round,pad=0,rounding_size=1.0",
                                        fill=False, lw=2.0, edgecolor="black"))
            # hoop seismic hooks at the top-right corner
            hook_mark(ax, w - c, h - c, "seismic_135", (-0.7, -0.7), 1.8)
            hook_mark(ax, w - c, h - c, "seismic_135", (-0.7, 0.7), 1.8)
            if name == "Column":
                bars = res["joint"]["column_bars_plan_in"]
                for x, y in bars:
                    ax.add_patch(Circle((x, y), db / 2, facecolor="black"))
            else:
                for pos in arr["faces"]["top"]:
                    ax.add_patch(Circle((pos, h - c), db / 2, facecolor="black"))
                for pos in arr["faces"]["bottom"]:
                    ax.add_patch(Circle((pos, c), db / 2, facecolor="black"))
            for tie in tset["crossties"]:
                if tie["axis"] == "x":
                    x0, x1, y0 = c, w - c, tie["position_in"]
                    ax.plot([x0, x1], [y0, y0], color="black", lw=1.3, ls="--")
                    for end, xx, dirn in (("near", x0, (1, 0)), ("far", x1, (-1, 0))):
                        hook_mark(ax, xx, y0, tie["hooks"][end], (dirn[0] * 0.0, 0.8), 1.6)
                else:
                    y0, y1, x0 = c, h - c, tie["position_in"]
                    ax.plot([x0, x0], [y0, y1], color="black", lw=1.3, ls="--")
                    for end, yy in (("near", y0), ("far", y1)):
                        hook_mark(ax, x0, yy, tie["hooks"][end], (0.8, 0.0), 1.6)
            ax.set_xlim(-4, w + 4); ax.set_ylim(-4, h + 4); ax.set_aspect("equal"); ax.axis("off")
            hg = tset["hook_geometry"]
            ax.set_title(f"{name} cage, tie set {col + 1}: #{tset['hoop']['bar_size']} hoop {tset['hoop']['out_to_out_in'][0]:g} x "
                         f"{tset['hoop']['out_to_out_in'][1]:g} in, bend {hg['seismic_135']['inside_bend_diameter_in']:g} in\n"
                         f"{len(tset['crossties'])} crossties, 90-degree end at {[t['ninety_degree_end'] for t in tset['crossties']]}",
                         fontsize=9)
    fig.text(0.5, 0.955, "Hoops and crossties with hook orientation; successive sets alternate the 90-degree end "
             "(ACI 318-19 18.7.5.2(c), 18.6.4.3, Table 25.3.2).\n"
             f"Marks: dot with tail = 135-degree seismic hook (extension {res['column']['tie_sets'][0]['hook_geometry']['seismic_135']['extension_in']:g} in column, "
             f"{res['beam']['tie_sets'][0]['hook_geometry']['seismic_135']['extension_in']:g} in beam); square = 90-degree hook "
             f"({res['column']['tie_sets'][0]['hook_geometry']['crosstie_90']['extension_in']:g} in column, "
             f"{res['beam']['tie_sets'][0]['hook_geometry']['crosstie_90']['extension_in']:g} in beam).",
             ha="center", va="top", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out / "cage_ties_and_hooks.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: joint plan, beam bars threaded between column bars
    j = res["joint"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, axis in zip(axes, ("x", "y")):
        d = j["directions"][axis]
        ax.add_patch(Rectangle((0, 0), hc, bc, facecolor="0.94", edgecolor="black", lw=1.4))
        lo, hi = d["beam_band_in"]
        if axis == "x":
            ax.add_patch(Rectangle((-14, lo), hc + 28, bw, facecolor="none", edgecolor="black", lw=1.0, hatch="//"))
        else:
            ax.add_patch(Rectangle((lo, -14), bw, bc + 28, facecolor="none", edgecolor="black", lw=1.0, hatch="//"))
        for x, y in j["column_bars_plan_in"]:
            ax.add_patch(Circle((x, y), col_db / 2, facecolor="black"))
        for p in d["nominal_positions_in"]:
            (ax.plot([-14, hc + 14], [p, p], color="0.6", lw=0.8, ls=":") if axis == "x" else ax.plot([p, p], [-14, bc + 14], color="0.6", lw=0.8, ls=":"))
        for p in d["threaded_positions_in"]:
            (ax.plot([-14, hc + 14], [p, p], color="black", lw=2.0) if axis == "x" else ax.plot([p, p], [-14, bc + 14], color="black", lw=2.0))
        ax.set_xlim(-16, hc + 16); ax.set_ylim(-16, bc + 16); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{axis}-beam bars through the joint: dotted = record's {r['beam_top_bars']} positions in one layer, "
                     f"solid = the {d['bars_per_layer_that_fit']} that pass\nbetween the column bars with {d['clearance_used_in']:g} in clear; "
                     f"layers needed {d['layers_needed']}", fontsize=9)
    fig.suptitle("Item 2: beam bars must pass between the column bars at the joint (plan at the top-bar level)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "joint_plan_threading.png", dpi=150)
    plt.close(fig)

    # ---- figure 3: joint elevation with the stacked layers and the slab mats
    fig, ax = plt.subplots(figsize=(11, 6.5))
    st = j["stacking"]
    ts = (record.get("slab") or {}).get("thickness_in", 0.0)
    span = 40.0
    ax.add_patch(Rectangle((-span, -hb), hc + 2 * span, hb, facecolor="0.94", edgecolor="black"))
    ax.add_patch(Rectangle((-span, -ts), hc + 2 * span, ts, facecolor="0.9", edgecolor="black", lw=0.8))
    ax.add_patch(Rectangle((0, -hb - 20), hc, hb + 20 + 20, facecolor="none", edgecolor="black", lw=1.2))
    for layer in j["beam_top_layers"]:
        y = -layer["centroid_from_top_in"]
        style = dict(color="black", lw=2.0) if layer["direction"] == "x" else dict(color="black", lw=2.0, ls="--")
        ax.plot([-span, hc + span], [y, y], **style)
        ax.text(hc + span + 1, y, layer["layer"] + f" ({layer['centroid_from_top_in']:g} in)", va="center", fontsize=8)
    for n, m in enumerate(sorted(j["slab_mats"], key=lambda m: m["centroid_from_top_in"])):
        y = -m["centroid_from_top_in"]
        ax.plot([-span, hc + span], [y, y], color="0.45", lw=0.9, ls="-" if m["runs_along"] == "x" else "--")
        # labels fanned out to the left so mats 0.5 in apart stay legible
        label_y = 2.0 - 2.4 * n
        ax.plot([-span - 1, -span - 6], [y, label_y], color="0.6", lw=0.5)
        ax.text(-span - 7, label_y, f"{m['mat']} ({m['centroid_from_top_in']:g} in)", ha="right", va="center", fontsize=8)
    for cl in j["slab_clashes"]:
        ax.text(hc / 2, -hb - 3, "OVERLAP: " + " vs ".join(cl["between"]), ha="center", va="top", fontsize=8.5, fontweight="bold")
    ax.set_xlim(-span - 26, hc + span + 30); ax.set_ylim(-hb - 8, 6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Item 2: stacking at the joint, convention {st['convention']}: {st['upper_direction']} bars on top, {st['lower_direction']} bars "
                 f"{st['lower_layer_centroid_from_face_in']:g} in down\n(effective depth {st['effective_depth_nominal_in']:g} -> {st['effective_depth_lower_in']:g} in, "
                 f"lever-arm ratio {st['lower_direction_lever_arm_ratio']:.3f}); slab mats grey (solid along x, dashed along y)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "joint_elevation_stacking.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stacking", choices=("x_over_y", "y_over_x"), default=None)
    parser.add_argument("--no-drawings", action="store_true")
    args = parser.parse_args(argv)
    if args.design.resolve().parent == args.output_root.resolve():
        raise SystemExit("Use a separate output directory; the design package is read-only here.")
    raw = args.design.read_bytes()
    record = json.loads(raw)
    res = evaluate_cage_geometry(record, stacking=args.stacking)
    res["source_design_path"] = str(args.design.resolve())
    res["source_design_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "cage_geometry.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    (args.output_root / "cage_geometry.md").write_text(markdown(res, record, args.design), encoding="utf-8")
    if not args.no_drawings:
        draw(res, record, args.output_root)
    print(json.dumps({"column_cage_passes": res["summary"]["column_cage_passes"], "beam_cage_passes": res["summary"]["beam_cage_passes"],
                      "joint_assembly_passes": res["summary"]["joint_assembly_passes"], "stacking_selected": res["stacking_selected"],
                      "failed": len(res["summary"]["failed_checks"]), "output": str(args.output_root)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
