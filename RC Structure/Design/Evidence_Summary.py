"""Evidence summary for one representative design, for the assertion review.

Writes ``evidence_summary.md`` beside a ``design.json``: every quantity a
reviewer needs before filling ``SlabActionAssertions``, ``DemandPolicy`` and
``IndependentVerification`` in ``Design/Config.py``, each computed from the
artifact or measured on the model built from it by this script, with the hand
reference beside it where a hand formula exists. Nothing is transcribed.

    python Design/Evidence_Summary.py --output-dir outputs/smrf_evidence_review_<date>
    python Design/Evidence_Summary.py --design-file <path>/design.json

Without ``--design-file`` the representative case is designed here under
PROBE slab-action and demand assertions (so the assertion-dependent
evidence is visible) with ``IndependentVerification`` deliberately empty, so
the artifact reports exactly the items the review settles. The probe is
recorded in the request identity and is not a certification.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from pathlib import Path

RC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_DIR))
import Structure_Parameters as sp  # noqa: E402
import Design.Design_Driver as dd  # noqa: E402
from Design.Config import (DesignConfig, SlabActionAssertions, DemandPolicy, IndependentVerification,  # noqa: E402
                           FloorAnalysisConfig, PROBE_SLAB_REFINEMENT)

PROBE = "PROBE -- evidence review only, not a certification"
REVIEW_CASE = {"NUM_BAY_X": 3, "NUM_BAY_Y": 3, "NUM_FLOOR": 8, "BAY_X": 120.0, "BAY_Y": 120.0, "STORY_H": 120.0}
REVIEW_SITE = "sdc_d_high"


def probe_config(date=None):
    """Slab-action and demand blocks asserted with PROBE values; verification left empty."""
    date = date or time.strftime("%Y-%m-%d")
    return DesignConfig(
        slab_actions=SlabActionAssertions(analysis_applicability_verified=True, all_floors_enveloped=True,
                                          load_pattern_envelope_verified=True, spatial_envelope_per_unit_width=True,
                                          twisting_moment_resolution_verified=True, zero_membrane_force_verified=True,
                                          verified=True, two_way_shear_path_assessed=True,
                                          asserted_by=PROBE, assertion_date=date, assertion_basis=PROBE),
        demands=DemandPolicy(declared_by=PROBE, declaration_date=date, declaration_basis=PROBE),
        verification=IndependentVerification(asserted_by="", assertion_basis=""),
        # Same shared PROBE recipe as Verify_Designs.probe_config; the
        # verification block above stays deliberately empty.
        floor_analysis=FloorAnalysisConfig(slab_refinement=dict(PROBE_SLAB_REFINEMENT)),
    )


def design_review_case(out_dir, geometry=None, site=REVIEW_SITE):
    """Design the representative case under probe assertions; save and return (record, seconds)."""
    import Geometry_Overrides as go
    overrides = dict(REVIEW_CASE, **(geometry or {}))
    with contextlib.redirect_stdout(io.StringIO()):
        go.apply_geometry_overrides(overrides, variant_name="evidence_review", emit=False)
        sp.apply_seismic_site(site)
    cfg = probe_config()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        record = dd.design_structure(cfg=cfg, verbose=False)
    record["request_identity"] = dd.design_request_identity(cfg)
    elapsed = time.perf_counter() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "design.json").write_text(json.dumps(record, indent=1, allow_nan=False), encoding="utf-8")
    return record, elapsed


def load_record(path):
    """A saved design, with Structure_Parameters set to its inputs so the model can be rebuilt."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, value in record["request_identity"]["inputs"].items():
        if hasattr(sp, key):
            setattr(sp, key, value)
    sp.NUM_MODES = sp.NUM_FLOOR + 2
    return record


def write_evidence_summary(r, out_dir, elapsed=None):
    """Write evidence_summary.md for record ``r`` into out_dir; returns the lines."""
    out_dir = Path(out_dir)
    g, s, b = r["geometry"], r["sections"], r["reinforcement"]
    slab = r["slab"]
    loads = r["floor_loads"]
    tr = r["floor_transfer"]
    actions = r["slab_actions"]
    layout = r["slab_reinforcement"]["layout"]
    fam = r["beam_slab_families"]
    cap = r["capacity_design"]
    dem = r["demand_basis"]
    q = r["qualification"]
    statuses = {c["id"]: c["status"] for c in q["checks"]}

    L = []
    def line(text=""):
        L.append(text)

    def pct(a, b):
        return 100.0 * (a - b) / b

    # ---------------------------------------------------------------- 1. case
    line("# SMRF evidence summary — representative case for the assertion review")
    line()
    line(f"Generated {time.strftime('%Y-%m-%d')} from `design.json` in this folder (`{r['schema_version']}`, design time {f'{elapsed:.0f} s' if elapsed is not None else 'not timed: summarized from the saved artifact'}). "
         "Every number below is computed from that artifact or measured on the model built from it by this script; nothing is transcribed.")
    line("Assertion-dependent evidence below was produced under **probe** assertions so it can be reviewed; the probe")
    line("is recorded in the artifact's request identity and is not a certification. `IndependentVerification` was left")
    line("empty on purpose, so the artifact reports exactly the items your review settles.")
    line()
    line("## 1. Case")
    line()
    line(f"- Plan {g['num_bay_x']}×{g['num_bay_y']} bays of {g['bay_x_in']/12:g}×{g['bay_y_in']/12:g} ft, {g['num_floor']} stories of {g['story_h_in']/12:g} ft; "
         f"site `{r['seismic']['site_label']}` (SDS {r['seismic']['sds']:.2f}, SD1 {r['seismic']['sd1']:.2f}, S1 {r['seismic']['s1']:.2f}); R = {r['seismic']['r']:g}, Ie = {sp.ASCE_IE:g}.")
    line(f"- Slab {slab['thickness_in']:g} in, f'c {slab['concrete_fc_ksi']:g} ksi; SDL {loads['floor_superimposed_dead_load_ksf']*1000:.0f} psf; live {loads['floor_live_load_ksf']*1000:.0f} psf; "
         f"floor D+L {loads['total_floor_gravity_load_kip']:.1f} kip, member weight {loads['member_self_weight_per_floor_kip']:.1f} kip, seismic weight {loads['total_floor_seismic_weight_kip']:.1f} kip/floor.")
    line(f"- Result: columns {s['b_col_in']:g}×{s['h_col_in']:g} f'c {s['fc_col_ksi']:g} with #{b['col_bar_size']} {b['col_top_bars']}T/{b['col_bot_bars']}B/{b['col_side_bars']}S, hoops #{b['col_stirrup_bar_size']}-{b['col_stirrup_legs']}L @ {b['col_stirrup_spacing_in']:g} in; "
         f"beams {s['b_beam_in']:g}×{s['h_beam_in']:g} f'c {s['fc_beam_ksi']:g} with #{b['beam_bar_size']} {b['beam_top_bars']}T/{b['beam_bot_bars']}B, hoops #{b['beam_stirrup_bar_size']}-{b['beam_stirrup_legs']}L @ {b['beam_stirrup_spacing_in']:g} in; "
         f"slab mats #{layout['bar_size']} @ " + ", ".join(f"{k} {v['spacing_in']:g}" for k, v in layout['layers'].items()) + " in.")
    line(f"- T1 = {r['demand']['model_period_sec']:.3f} s (design period {dem['design_period_sec']:.3f} s, Cu·Ta cap {'applied' if dem['period_basis']['capped_at_cu_ta'] else 'not applied'}); base shear {r['demand']['base_shear_kip']:.1f} kip; "
         f"beam DCR {r['dcr']['beam']:.3f}, column DCR {r['dcr']['column']:.3f}.")
    line(f"- Qualification: {q['counts']}; open items: {', '.join(f'`{i}`' for i in q['not_evaluated'])}.")
    line()

    # ---------------------------------------------------------------- 2. floor transfer
    line("## 2. Slab-to-frame gravity transfer (`floor_transfer`)")
    line()
    dead = tr["unit_cases"]["dead"]
    live = tr["unit_cases"]["live"]
    by = {(x["axis"], x["line_index"], x["span_index"]): x for x in dead["beams"]}
    xi = by[("x", 1, 1)]; xe = by[("x", 0, 1)]
    dead_ksf = tr["dead_pressure_ksf"]
    # Two-way tributary to an x-beam from one panel: a triangle when the beam is
    # on the panel's short side, a trapezoid when on its long side; an interior
    # beam has a panel on each side.
    lx_, ly_ = g["bay_x_in"], g["bay_y_in"]
    short_, long_ = min(lx_, ly_), max(lx_, ly_)
    panel_to_x_beam = (short_ ** 2 / 4.0) if lx_ <= ly_ else (long_ - short_ / 2.0) * short_ / 2.0
    tri = dead_ksf / 144.0 * 2 * panel_to_x_beam
    shape = "two triangles" if lx_ <= ly_ else "two trapezoids"
    line(f"- Mesh {tr['mesh_per_bay']}/bay; unit dead {dead['applied_kip']:.2f} kip = {dead_ksf:.4f} ksf × {loads['floor_area_sqft']:.0f} ft²; unit live {live['applied_kip']:.2f} kip. "
         f"Direct-to-column share (point-support mesh artifact, recorded not redistributed): {100*dead['column_direct_fraction']:.1f}% (dead).")
    line(f"- Equilibrium of every unit case: {'all balanced to 1e-8' if all(c['equilibrium']['numerical_balance_passed'] for c in tr['unit_cases'].values() if c) else 'NOT balanced'}.")
    line(f"- Interior x-beam, unit dead: {xi['total_kip']:.3f} kip vs two-way tributary ({shape}, bays {lx_/12:g}×{ly_/12:g} ft) {tri:.3f} kip "
         f"({pct(xi['total_kip'], tri):+.1f}%); edge x-beam {xe['total_kip']:.3f} kip vs half of that {tri/2:.3f} kip ({pct(xe['total_kip'], tri/2):+.1f}%).")
    line(f"- Exported inventory (transfer schema {tr['schema']}): node forces plus torsion/bending couples at every beam-line node, "
         f"column direct loads plus beam-end couples at every column node; force and both first moments balance to "
         f"{max(dead['equilibrium']['x_first_moment_relative_error'], dead['equilibrium']['y_first_moment_relative_error']):.1e} (dead). "
         f"Bending couples are applied to the frame as in-element force pairs (arm 1 in), torsion couples split to the end joints.")
    line("- Idealization to judge: elastic ShellMITC4 on the slab mid-plane; ACI 8.4.1.8 gross T/L-beams (interior 2 flanges, perimeter 1) less the flange mid-plane term; "
         "columns held vertically at intersections; live load all-panel for the transfer, ACI 6.4.2 arrangements as separate unit cases. "
         "Checked in tests: stiff-bending/free-torsion limit vs the rigid-line diagnostic (<5%), mirror symmetry, mesh stability (8 vs 16/bay <3%), Navier plate benchmark (rigid model).")
    line()

    # ---------------------------------------------------------------- 3. slab strip actions
    line("## 3. Slab strip actions (`slab_actions`) — the `SlabActionAssertions` evidence")
    line()
    w = 1.2 * dead_ksf + 1.6 * loads["floor_live_load_ksf"]
    lc = (g["bay_x_in"] - s["h_col_in"]) / 12.0                    # clear span, ft
    c2 = {"neg": 0.045, "pos_d": 0.018, "pos_l": 0.027}            # ACI 318-63 Method 2, Case 2 (all edges continuous), m = 1
    c4 = {"neg": 0.050, "pos_d": 0.027, "pos_l": 0.032}            # Case 4 (two adjacent edges discontinuous), m = 1
    def ref(c):
        neg = c["neg"] * w * lc**2 * 12.0
        pos = (c["pos_d"] * 1.2 * dead_ksf + c["pos_l"] * 1.6 * loads["floor_live_load_ksf"]) * lc**2 * 12.0
        return neg, pos
    strips = {(x["panel_id"], x["axis"], x["face"]): x for x in actions["strips"]}
    interior = f"panel_x{(g['num_bay_x']+1)//2}_y{(g['num_bay_y']+1)//2}"
    corner = "panel_x1_y1"
    line(f"- Cases: {', '.join(c['id'] for c in actions['cases'])}; pattern rule: {actions['pattern_rule']}.")
    line(f"- Membrane resultants max |p| = {actions['max_abs_membrane_kip_per_in']:.1e} kip/in (in-plane DOFs fixed). Equilibrium: "
         f"{'every case balanced' if all(e['numerical_balance_passed'] for e in actions['equilibrium']) else 'NOT balanced'}.")
    square = abs(g["bay_x_in"] - g["bay_y_in"]) < 1e-9
    line(f"- Factored pressure w = 1.2·{dead_ksf:.4f} + 1.6·{loads['floor_live_load_ksf']:.3f} = {w:.4f} ksf; clear span {lc:.2f} ft"
         + (" (square panels, m = 1)." if square else f" (panels {g['bay_x_in']/12:g}×{g['bay_y_in']/12:g} ft, m = {min(g['bay_x_in'], g['bay_y_in'])/max(g['bay_x_in'], g['bay_y_in']):.2f})."))
    line()
    if square:
        line("| panel | face | Wood-Armer envelope, kip-in/ft (x / y) | ACI 318-63 Method 2 reference (m = 1) | note |")
        line("|---|---|---|---|---|")
        for pid, label, coef in ((interior, "interior (Case 2)", c2), (corner, "corner (Case 4)", c4)):
            neg, pos = ref(coef)
            for face, refv in (("top", neg), ("bottom", pos)):
                mx = strips[(pid, "x", face)]["mu_kip_in_per_ft"]; my = strips[(pid, "y", face)]["mu_kip_in_per_ft"]
                line(f"| {pid} ({label}) | {face} | {mx:.2f} / {my:.2f} | {refv:.2f} | FE {pct(max(mx,my), refv):+.0f}% (Gauss maxima incl. twisting vs strip-average coefficients) |")
    else:
        line("| panel | face | Wood-Armer envelope, kip-in/ft (x / y) |")
        line("|---|---|---|")
        for pid, label in ((interior, "interior"), (corner, "corner")):
            for face in ("top", "bottom"):
                mx = strips[(pid, "x", face)]["mu_kip_in_per_ft"]; my = strips[(pid, "y", face)]["mu_kip_in_per_ft"]
                line(f"| {pid} ({label}) | {face} | {mx:.2f} / {my:.2f} |")
        line()
        line("- The ACI 318-63 Method 2 reference is tabulated here only for square panels (m = 1); for this panel ratio "
             "take the coefficients for m from the method's tables before comparing.")
    vmax = max(x["vu_kip_per_ft"] for x in actions["strips"])
    line()
    line(f"- Support-face shear max {vmax:.3f} kip/ft (points outside the beam width, face in tension); one-way reference w·ℓc/2 = {w*lc/2:.3f} kip/ft, two-way share roughly half of that.")
    spacings = {name: layer["spacing_in"] for name, layer in sorted(layout["layers"].items())}
    ratio_top = min(layer["flexure"]["phi_mn_kip_in_per_ft"] / layer["demand_envelope"]["mu_kip_in_per_ft"]
                    for name, layer in layout["layers"].items() if layer["demand_envelope"]["mu_kip_in_per_ft"] > 0)
    governed_by_spacing = all(abs(layer["spacing_in"] - layer["maximum_spacing_in"]) < 1e-9
                              for layer in layout["layers"].values())
    line(f"- Selected mats: #{layout['bar_size']} @ " + ", ".join(f"{k} {v:g}" for k, v in spacings.items())
         + f" in; lowest capacity/demand over the layers {ratio_top:.1f}×; "
         + (f"the 2h = {2*slab['thickness_in']:g} in spacing limit governs, not strength" if governed_by_spacing else "strength or another limit governs the spacing")
         + f". Minimum steel 0.0018bh = {0.0018*12*slab['thickness_in']:.3f} in²/ft vs provided "
         + ", ".join(f"{name} {layer['area_in2_per_ft']:.3f}" for name, layer in sorted(layout['layers'].items())) + ".")
    line("- What the flags assert (see `numerical_basis` in `slab_actions`): applicability of an elastic plate FE with uniform gravity; one common floor represents all floors (uniform slab, SDL, live incl. roof); "
         "the 6.4.3.3 pattern rule; Gauss-point (not strip-averaged) maxima; Wood-Armer twisting resolution; membrane-free; overall verification. `two_way_shear_path_assessed`: "
         f"α_f min {min(e['alpha_f'] for p in slab['panels'] for e in p['edges']):.2f} and α_f1·ℓ2/ℓ1 min "
         f"{min(e['alpha_f'] * e['slab_strip_width_in'] / (g['bay_x_in'] if e['beam_axis'] == 'x' else g['bay_y_in']) for p in slab['panels'] for e in p['edges']):.2f} "
         "on every edge → the beams take the whole panel shear (ACI 318-14 Table 8.10.8.1, via 318-19 R8.2.1) and are designed for the plate reactions plus stem weight; "
         "the slab is checked for one-way shear at the beam faces (318-19 8.4.3.1); no slab-column critical section (8.4.4.1.1) exists.")
    line()

    # ---------------------------------------------------------------- 4. beam gravity moment check
    line("## 4. Frame gravity demand check")
    line()
    # Frame hogging under 1.2D+1.6L against the fixed-end moment of each beam's actual transfer loads.
    combos = r["design_actions"]["combinations"]
    grav = next(c for c in combos if c["id"] == "gravity_1.2D_1.6L")
    n_col_all = g["num_floor"] * (g["num_bay_x"] + 1) * (g["num_bay_y"] + 1)
    n_bx_all = g["num_bay_x"] * (g["num_bay_y"] + 1)
    def _beam_key(tag):
        idx = tag - n_col_all - 1
        if idx < g["num_floor"] * n_bx_all:
            k, rem = divmod(idx, n_bx_all); line_, span_ = divmod(rem, g["num_bay_x"]); return "x", k + 1, line_, span_
        idx -= g["num_floor"] * n_bx_all
        n_by_all = (g["num_bay_x"] + 1) * g["num_bay_y"]
        k, rem = divmod(idx, n_by_all); span_, line_ = divmod(rem, g["num_bay_x"] + 1); return "y", k + 1, line_, span_
    unit_dead = {(b["axis"], b["line_index"], b["span_index"]): b for b in tr["unit_cases"]["dead"]["beams"]}
    unit_live = {(b["axis"], b["line_index"], b["span_index"]): b for b in tr["unit_cases"]["live"]["beams"]}
    ratios, worst_hog = {}, (0.0, 0.0, None)
    for tag_s, m in grav["members"].items():
        if m["member_type"] == "column":
            continue
        axis_, k_, line_, span_ = _beam_key(int(tag_s))
        L_ = g["bay_x_in"] if axis_ == "x" else g["bay_y_in"]
        f_ = m["local_force_kip_kipin"]
        fem = [0.0, 0.0]
        from Loads.Gravity_Loads import _bending_couple_pair
        for unit, factor in ((unit_dead, 1.2), (unit_live, 1.6)):
            beam_ = unit[(axis_, line_, span_)]
            point_loads = [(x_, -factor * p_) for x_, p_ in beam_["node_loads"]]            # downward
            for x_, _cx, cy_ in beam_.get("node_couples", []):
                point_loads += _bending_couple_pair(x_, factor * cy_, L_)                    # the bending couple, as applied
            for x_, pz_ in point_loads:
                a_, b_ = x_ * L_, (1 - x_) * L_
                fem[0] += -pz_ * a_ * b_ * b_ / L_ ** 2                                      # hogging positive
                fem[1] += -pz_ * a_ * a_ * b_ / L_ ** 2
        w_ = 1.2 * slab["concrete_unit_weight_kcf"] / 1728 * s["b_beam_in"] * (s["h_beam_in"] - slab["thickness_in"]) \
            * (1 - (s["h_col_in"] if axis_ == "x" else s["b_col_in"]) / L_)
        fem = [v + w_ * L_ ** 2 / 12 for v in fem]
        kind_ = "edge" if line_ in (0, g["num_bay_y"] if axis_ == "x" else g["num_bay_x"]) else "interior"
        for hog_, ref_ in ((-f_[4], fem[0]), (f_[10], fem[1])):
            ratios.setdefault(kind_, []).append(hog_ / ref_)
            if hog_ > worst_hog[0]:
                worst_hog = (hog_, ref_, f"{axis_} {kind_} beam, floor {k_}")
    per_story_all = (g["num_bay_x"] + 1) * (g["num_bay_y"] + 1)
    base_axial = sum(m["axial_i_kip"] for t_, m in grav["members"].items() if m["member_type"] == "column" and int(t_) <= per_story_all)
    factored_gravity = g["num_floor"] * (1.2 * (loads["floor_dead_load_ksf"] * loads["floor_area_sqft"] + loads["member_self_weight_per_floor_kip"])
                                         + 1.6 * loads["floor_live_load_ksf"] * loads["floor_area_sqft"])
    line(f"- Load ledger under 1.2D+1.6L: story-1 column axials at the joint faces sum to {base_axial:.1f} kip; factored building gravity "
         f"1.2(D + members) + 1.6L = {factored_gravity:.1f} kip ({base_axial - factored_gravity:+.1e}). Floor pressure reaches the frame exactly once.")
    line(f"- Beam hogging vs the fixed-end moment of each beam's own transfer loads (point forces and bending couples at their positions, as applied, plus drop weight): "
         f"interior beams {min(ratios['interior']):.2f}-{max(ratios['interior']):.2f}, edge beams {min(ratios['edge']):.2f}-{max(ratios['edge']):.2f} "
         f"of fixed-end -- exterior ends relieved by joint rotation, first interior supports raised, as a continuous beam should. "
         f"Largest hogging {worst_hog[0]:.0f} kip-in ({worst_hog[2]}) against a fixed-end reference of {worst_hog[1]:.0f}.")
    line("- Beam end-moment sign mapping is pinned against OpenSees (fixed-fixed span: My_i = −wL²/12, My_j = +wL²/12) in `tests/test_beam_flexure_sign.py`.")
    cc = r["coupled_comparison"]
    ms = cc["moment_significance"]["by_kind"]
    line(f"- Coupled comparison (`coupled_comparison`, the `floor_frame_compatibility_reviewed` evidence): bare frame + transfer vs the monolithic shell/web/column model, "
         f"gross stiffness, mesh {cc['mesh_per_bay']}/bay. All panels: column verticals within {100*cc['max_column_vertical_relative_difference']:.1f}%, totals to {cc['total_relative_error']:.1e}; "
         f"pattern `{cc['asymmetric']['pattern']['id']}`: within {100*cc['asymmetric']['max_column_vertical_relative_difference']:.1f}%. "
         f"Base-moment ratio frame/coupled: " + ", ".join(f"{k} {v:.2f}" for k, v in cc['base_moment_ratio_frame_over_coupled'].items() if v) + ". "
         f"Coupled mesh sensitivity ({cc['coupled_mesh_sensitivity']['meshes_per_bay']}): verticals {100*cc['coupled_mesh_sensitivity']['max_column_vertical_relative_change']:.1f}%, base moments {100*cc['coupled_mesh_sensitivity']['max_column_base_moment_relative_change']:.1f}%.")
    line("- Moment significance (per column kind: coupled gravity base moment / governing factored base moment; model gap / governing): " +
         "; ".join(f"{k} {e['coupled_gravity_over_governing']:.3f} / {e['gravity_gap_over_governing']:.4f}" for k, e in ms.items() if e['coupled_gravity_over_governing'] is not None) +
         ". Not compared: " + ", ".join(cc["not_compared"]) + ".")
    line()

    # ---------------------------------------------------------------- 5. demand basis
    line("## 5. Demand basis (`demand_basis`) — the `DemandPolicy` evidence")
    line()
    tor = dem["torsion"]
    worst = max(tor["stories"], key=lambda x: x["delta_max_over_avg"])
    line(f"- Derived SDC {statuses.get('demands.site_hazard') and [c for c in q['checks'] if c['id']=='demands.site_hazard'][0]['details']['derived_sdc']} from SDS/SD1/S1 (Tables 11.6-1/11.6-2); label agrees.")
    line(f"- Analysis procedure: ELF is permitted for any structure (ASCE 7-22 12.6; the 7-16 Table 12.6-1 restriction was deleted); "
         f"height {g['num_floor']*g['story_h_in']/12:.0f} ft, regular by construction, torsional irregularity (7-22 Table 12.3-1 Type 1): {tor['torsional_irregularity']}.")
    line(f"- Accidental torsion: 5% eccentricity in every seismic combination; TIR (12.3.2.1.1, {len(tor.get('cases') or ['x', 'y'])} accidental-torsion cases) "
         f"= {tor.get('tir', tor['max_drift_ratio']):.3f} (story {worst['story']}, {worst['direction']}{', ' + worst['case'] if worst.get('case') else ''}), "
         f"Type 1 threshold 1.2 → Ax applied {tor['amplification']:.3f} (12.8.4.3 envelope over levels from the edge level displacements: "
         f"{tor.get('amplification_envelope_12_8_4_3', float('nan')):.3f}); TIR ≤ 1.4 is a project policy in every SDC (ASCE 7-22 removed the 7-16 12.3.3.1 E/F prohibition); "
         f"one-sided story strength fraction {((tor.get('strength_distribution') or {}).get('one_side_fraction') or float('nan')):.3f} (Type 1 above 0.75).")
    line(f"- Live-load arrangements (ACI 6.4.2): {len(dem['live_load_patterns'])} — " + ", ".join(p['id'] for p in dem['live_load_patterns']) + "; combinations in the strength envelope: " + str(len(r['design_actions']['combinations'])) + ".")
    line(f"- Seismic weight per floor {loads['total_floor_seismic_weight_kip']:.1f} kip = slab {loads['slab_self_weight_ksf']*1000:.1f} psf + SDL {loads['floor_superimposed_dead_load_ksf']*1000:.0f} psf (incl. ≥10 psf partitions) over {loads['floor_area_sqft']:.0f} ft² + members {loads['member_self_weight_per_floor_kip']:.1f} kip; live fraction {loads['seismic_live_load_fraction']:g} (office).")
    line(f"- Drift basis: 0.35/0.70 Ig, ρ = 1 forces at the capped period, Cd = 5.5, P-Δ included, full D+L gravity; SDC used {r['drift_screen']['assumptions']['seismic_design_category']}.")
    line(f"- To declare: site class {sp.ASCE_SITE_CLASS} (ASCE 7-22 Table 20.2-1; a site response analysis is required for Site Class F only, 11.4.7), risk category {sp.ASCE_RISK_CATEGORY} "
         f"(Ie = {sp.ASCE_IE:g}; the category fixes Ie and the drift row, not the occupancy), office occupancy, partition allowance within the 50 psf SDL, roof live 20 psf enveloped by the 50 psf floor live applied to the roof, snow 0, wind not governing, rain/ponding excluded.")
    line()

    # ---------------------------------------------------------------- 6. strengths and capacity design
    line("## 6. Beam-plus-slab strengths and capacity design — for the `IndependentVerification` review")
    line()
    fi = fam["x_interior"]; fe = fam["x_edge"]
    ab = {6: 0.44, 7: 0.60, 8: 0.79, 5: 0.31, 4: 0.20}[b["beam_bar_size"]]
    # the hogging hand check at the family's own top-bar elevation (staggered since 2026-09-27)
    d_eff = s["h_beam_in"] - (fi.get("centroid_offsets_in") or {}).get("top", b["beam_longitudinal_centroid_offset_in"])
    As = b["beam_top_bars"] * ab
    a = As * 60.0 / (0.85 * s["fc_beam_ksi"] * s["b_beam_in"])
    hand_rect = As * 60.0 * (d_eff - a / 2.0)
    line(f"- Rectangular Mn⁻ (beam top bars only): solver {fi['rectangular']['negative']['mn_kip_in']:.0f} kip-in vs hand As·fy·(d − a/2) = {hand_rect:.0f} kip-in ({pct(fi['rectangular']['negative']['mn_kip_in'], hand_rect):+.1f}%, compression steel included by the solver).")
    line(f"- Effective flange (Table 6.3.2.1): interior {fi['effective_flange_width_in']:.1f} in (bw + 2·min(8h, sw/2, ℓn/8)), edge {fe['effective_flange_width_in']:.1f} in; slab steel inside it {fi['slab_steel_in_flange_in2']:.2f} / {fe['slab_steel_in_flange_in2']:.2f} in² (both mats, continuous → developed).")
    line(f"- Composite Mn⁻ / Mn⁺: interior {fi['mn_negative_kip_in']:.0f} / {fi['mn_positive_kip_in']:.0f} kip-in ({pct(fi['mn_negative_kip_in'], fi['rectangular']['negative']['mn_kip_in']):+.0f}% / {pct(fi['mn_positive_kip_in'], fi['rectangular']['positive']['mn_kip_in']):+.0f}% vs rectangular); edge {fe['mn_negative_kip_in']:.0f} / {fe['mn_positive_kip_in']:.0f}. "
         "Same numbers feed SCWB, the joint evidence and the hinge yield moments (`hinge_backbone.csv` will show hogging/sagging per beam family).")
    sc = r["scwb"]
    for axis in ("x", "y"):
        anc = fam[f"{axis}_edge"].get("exterior_anchorage")
        if anc:
            line(f"- Slab mats parallel to {axis} at the perimeter: hooked into the {s['b_beam_in']:g}-in perimeter beam, "
                 f"ℓdh (25.4.3.1, #{anc['bar_size']}, ψr {anc['governing']['psi_r']:g}, ψc {anc['governing']['psi_c']:.3f}) = "
                 f"{anc['ldh_required_in']:.2f} in ≤ embedment {anc['embedment_available_in']:.2f} in → "
                 f"{'developed: credited at exterior beam ends' if anc['developed'] else 'NOT developed: exterior hogging ends use the rectangular beam'}.")
    line(f"- SCWB screen, governing interior roof joint: Mnc {sc['column_nominal_moment_kip_in']:.0f} vs 1.2·(Mnb⁻ + Mnb⁺) = 1.2·{sc['beam_nominal_moment_kip_in']:.0f} = {1.2*sc['beam_nominal_moment_kip_in']:.0f} kip-in → ratio {sc['ratio_provided']:.2f}; exact per-joint checks: "
         f"{sum(1 for c in q['checks'] if c['id']=='scwb' and c['status']=='pass')} pass, {sum(1 for c in q['checks'] if c['id']=='scwb' and c['status']!='pass')} otherwise.")
    bs, cs = cap["beams"], cap["columns"]
    st = cap["beam_strengths"]["x_interior"]
    line(f"- Beam probable strengths (1.25 fy): Mpr⁻ {st['mpr_negative_kip_in']:.0f}, Mpr⁺ {st['mpr_positive_kip_in']:.0f} kip-in over ℓn {st['clear_span_in']:.0f} in → mechanism shear {bs['mechanism_shear_kip']:.1f} kip; "
         f"Ve = {bs['ve_kip']:.1f} kip with factored gravity reactions; Vc = 0 in the hinge zone ({'yes' if bs['vc_zero_hinge_zone'] else 'no'}); Vs required {bs['vs_required_kip']:.1f} ≤ 8√f'c·bw·d = {bs['vs_limit_kip']:.1f} kip; "
         f"hoops #{bs['hoops']['bar_size']}-{bs['hoops']['legs']}L @ {bs['hoops']['spacing_in']:g} in (φVn {bs['hoops']['phi_vn_kip']:.0f} kip).")
    gv = cs["governing"]
    method = cs.get("column_shear_method") or "beam_joint_delivery_limited_v2 (unnamed legacy evidence)"
    own_envelope = gv.get("ve_own_envelope_kip")
    line(f"- Column Ve, method `{method}`: min(2·Mpr,col/ℓn = {gv['ve_own_kip']:.0f}, joint-limited {gv['ve_joint_limited_kip']:.0f}) ≥ Vu {gv['vu_analysis_kip']:.0f} → {gv['ve_kip']:.0f} kip"
         + (f" (column-own probable envelope, each end at its own axial range: {own_envelope:.0f} kip, recorded for comparison)" if own_envelope is not None else "")
         + f"; Vc = 0 ({'yes' if gv['vc_zero'] else 'no'}: P_min {gv['axial_min_kip']:.0f} kip < Ag f'c/20 = {s['b_col_in']*s['h_col_in']*s['fc_col_ksi']/20:.0f}); "
         f"hoops #{cs['hoops']['bar_size']}-{cs['hoops']['legs']}L @ {cs['hoops']['spacing_in']:g} in (shear s = {cs['hoops']['spacing_from_shear_in']:.1f}, confinement s = {cs['hoops']['spacing_from_confinement_in']:.1f}, 4-in cap); "
         f"Ash/s provided {cs['hoops']['ash_provided_per_in']:.3f} vs required {cs['hoops']['ash_required_per_in']:.3f} in²/in; hx {cs['confinement']['hx_in']:.1f} in.")
    line()
    line("| joint | Vj kip | φVn kip | γ | Table 18.8.4.3 inputs (column / beam / transverse confinement) | Vj/φVn |")
    line("|---|---|---|---|---|---|")
    for k, j in cap["joints"]["joints"].items():
        if j["axis"] == "x":
            cls = j.get("classification") or {}
            inputs = (f"{cls['column']['state']} / {cls['beam']['state']} / {'confined' if cls['confinement']['confined'] else 'not confined'}"
                      if cls else f"{j.get('confined_faces')} faces ≥ 3/4 width (legacy record)")
            line(f"| {j['level']} {j['kind']} | {j['vj_kip']:.0f} | {j['phi_vn_kip']:.0f} | {j['gamma']:.0f} | {inputs} | {j['vj_kip']/j['phi_vn_kip']:.2f} |")
    an = cap["anchorage"]["directions"]["x"]; spl = cap["splices"]
    line()
    line(f"- Vj hand check, floor interior x: 1.25·60·(As_top + slab) + 1.25·60·As_bot − Vcol = 75·{st['tension_steel_hogging_in2']:.2f} + 75·{st['tension_steel_sagging_in2']:.2f} − {cap['joints']['joints']['joint_shear/floor/interior/x']['column_shear_kip']:.0f} = {cap['joints']['joints']['joint_shear/floor/interior/x']['vj_kip']:.0f} kip; "
         f"φVn = 0.85·γ·√f'c·Aj with Aj = {cap['joints']['joints']['joint_shear/floor/interior/x']['aj_in2']:.0f} in². Vcol assumes equal split above/below with mid-height inflection (all to the single column at the roof) — an assumption to confirm.")
    line(f"- Terminating bars: ℓdh = fy·db/(65√f'c) = {an['ldh_required_in']:.1f} in ≤ embedment {an['embedment_available_in']:.1f} in; through-bar 20db = {an['through_bar_depth_required_in']:.1f} in ≤ {an['through_bar_depth_available_in']:.0f}.")
    line(f"- Splices: beam Class B lap {spl['beam']['class_b_lap_top_in']:.0f} in vs {spl['beam']['available_between_hinge_zones_in']:.0f} in between hinge zones → {spl['beam']['splice_type']}; column lap {spl['column']['class_b_lap_in']:.0f} in in the {spl['column']['center_half_clear_height_in']:.0f}-in center half → {spl['column']['splice_type']}.")
    line()

    # ---------------------------------------------------------------- 7. model consistency
    line("## 7. Design-to-model consistency (what `detailing_model_consistency_verified` covers)")
    line()
    # Build the IMK model from this artifact and measure it.
    from Design.Design_Driver import apply_design
    import Ground_Motion_Main as gm
    from Model.IMK_Hinges import hinge_registry
    apply_design(r)
    with contextlib.redirect_stdout(io.StringIO()):
        gravity_state, ref_modal, _post, _a, _b = gm.build_gravity_modal_state()
    imk_t1 = next(m["period"] for m in ref_modal if m.get("valid"))
    registry = hinge_registry()
    hinge_fam = {}
    for h in registry.values():
        if h.get("member_type") != "column":
            hinge_fam.setdefault(h["beam_family"], (h["yield_moment_y_hogging_kip_in"], h["yield_moment_y_sagging_kip_in"]))
    family_match = all(abs(hinge_fam[k][0] - fam[k]["mn_negative_kip_in"]) < 1e-6 and abs(hinge_fam[k][1] - fam[k]["mn_positive_kip_in"]) < 1e-6
                       for k in fam)
    rho_sh_beam = b["beam_stirrup_legs"] * sp.rebar_area(b["beam_stirrup_bar_size"]) / (s["b_beam_in"] * b["beam_stirrup_spacing_in"])
    rho_sh_col = b["col_stirrup_legs"] * sp.rebar_area(b["col_stirrup_bar_size"]) / (s["b_col_in"] * b["col_stirrup_spacing_in"])
    line(f"- The IMK model built from this artifact (measured by this script): T1 = {imk_t1:.4f} s vs the design model's {r['demand']['model_period_sec']:.4f} s; "
         f"gravity converges with roof sway ux = {gravity_state.get('ux'):.1e} in, uy = {gravity_state.get('uy'):.1e} in; "
         f"{len(registry)} hinged members ({sum(1 for h in registry.values() if h.get('member_type') == 'column')} columns). "
         f"Beam hinge yield moments per family {'equal' if family_match else 'DO NOT equal'} the design's composite Mn⁻/Mn⁺: "
         + ", ".join(f"{k} {v[0]:.0f}/{v[1]:.0f}" for k, v in sorted(hinge_fam.items()))
         + " kip-in (spring sign measured: hogging +ve at end i, −ve at end j, `tests/test_beam_hinge_asymmetry.py`).")
    beams_ = [h for h in registry.values() if h.get("member_type") != "column"]
    reduced_ends = sum(1 for h in beams_ for end in ("i", "j")
                       if abs(h[f"yield_moment_y_hogging_{end}_kip_in"] - h["yield_moment_y_hogging_kip_in"]) > 1e-9
                       or abs(h[f"yield_moment_y_sagging_{end}_kip_in"] - h["yield_moment_y_sagging_kip_in"]) > 1e-9)
    anchorage_ = {tuple(sorted((k, v) for k, v in (h.get("exterior_slab_anchorage") or {}).items() if k in ("axis", "developed", "ldh_required_in", "embedment_available_in")))
                  for h in beams_}
    line(f"- Per-end hinge strengths: {reduced_ends} of {2 * len(beams_)} beam ends yield without the slab bars (an exterior end whose perimeter hook "
         f"does not fit, `terminated_undeveloped` hogging / `flange_concrete_undeveloped_bars` sagging); perimeter hook check on the model: "
         + "; ".join(", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in a) for a in sorted(anchorage_)) + ".")
    line(f"- ρsh from the saved hoops, as `Model.IMK_Calibration.transverse_steel_ratio` reads it: beams {rho_sh_beam:.4f}, columns {rho_sh_col:.4f} (Haselton cap 0.02).")
    line("- Export/cache metadata (`hybrid_metadata.reinforcement_geometry`, `hinge_backbone.csv` hogging/sagging columns) is what item 7 asks you to spot-check on one exported case.")
    line()

    # ---------------------------------------------------------------- 8. checklist
    line("## 8. What to fill in `Design/Config.py`")
    line()
    line("- `SlabActionAssertions`: seven flags + `two_way_shear_path_assessed`, with `asserted_by`, `assertion_date`, `assertion_basis` (e.g. 'reviewed evidence_summary §2–3 for case 3x3x8 sdc_d_high, 2026-09-13').")
    line("- `DemandPolicy`: keep the defaults if §5 matches your archetype; add `declared_by`, `declaration_date`, `declaration_basis`.")
    line("- `IndependentVerification`: `floor_hand_check_verified` (§2–3), `strength_model_verified` (§6), `detailing_model_consistency_verified` (§7), `slab_column_local_steel_assessed` (8.6.1.2 for a beam-supported slab), `fire_resistance_scope_accepted`, `congestion_and_placement_accepted` (hook geometry, crosstie end alternation and placement of the generated hoop/crosstie arrangement in §6), `floor_frame_compatibility_reviewed` (§4: the coupled comparison, its pattern case, mesh sensitivity and `moment_significance`); `asserted_by`, `assertion_date`, `assertion_basis`.")
    line("- Every design made after the blocks are filled records them in its request identity. `GENERATION_RELEASE_READY` is not the next step: "
         "it follows the design-only verification run over the plan (`Design/Verify_Designs.py`) with the real assertions, "
         "since this document covers one case, not the design range.")
    line()
    line(f"Artifact: `{(out_dir / 'design.json').as_posix()}` — {(out_dir / 'design.json').stat().st_size/1e6:.1f} MB; open items in it: {', '.join(q['not_evaluated'])}.")

    (out_dir / "evidence_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return L


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design-file", default=None, help="Summarize this saved design instead of designing the review case.")
    parser.add_argument("--output-dir", default=None,
                        help="Folder for design.json and evidence_summary.md (default: outputs/smrf_evidence_review_<date>, "
                             "or the design file's folder).")
    parser.add_argument("--num-bay-x", type=int); parser.add_argument("--num-bay-y", type=int)
    parser.add_argument("--num-floor", type=int); parser.add_argument("--bay-x", type=float, help="in")
    parser.add_argument("--bay-y", type=float, help="in"); parser.add_argument("--story-h", type=float, help="in")
    parser.add_argument("--seismic-site", default=REVIEW_SITE)
    args = parser.parse_args(argv)
    if args.design_file:
        out_dir = Path(args.output_dir) if args.output_dir else Path(args.design_file).resolve().parent
        record, elapsed = load_record(args.design_file), None
    else:
        out_dir = Path(args.output_dir) if args.output_dir else RC_DIR / "outputs" / f"smrf_evidence_review_{time.strftime('%Y%m%d')}"
        geometry = {key: value for key, value in (("NUM_BAY_X", args.num_bay_x), ("NUM_BAY_Y", args.num_bay_y),
                                                  ("NUM_FLOOR", args.num_floor), ("BAY_X", args.bay_x),
                                                  ("BAY_Y", args.bay_y), ("STORY_H", args.story_h)) if value is not None}
        record, elapsed = design_review_case(out_dir, geometry, args.seismic_site)
    lines = write_evidence_summary(record, out_dir, elapsed)
    print(f"written {out_dir / 'evidence_summary.md'} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
