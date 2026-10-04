"""Full-rate hinge moment-rotation measurement for fixed-design diagnostics (2026-09-21).

Production NTHA keeps hinge rotation histories at a stride and force
ENVELOPES only, so a loop cannot be drawn from a generated run. This module
attaches OpenSees element recorders to every IMK hinge spring before an
analysis, reads them back afterwards, and evaluates each spring against the
backbone actually installed on that hinge end and sign
(Analysis.Pushover_Diagnostic.hinge_inventory reads Model.IMK_Hinges
.hinge_registry). Nothing here redesigns, rebuilds a design record, or
writes outside the caller's output directory.

Measurement path
----------------
* One recorder per spring axis covers every hinge zero-length element:
  ``recorder Element -file <p> -time -precision 12 -ele <tags...> material <k> stressStrain``.
  Each row is [analysis time, (moment, rotation) per element in tag order]:
  the IMKBilin material's "stress" is the spring moment (kip-in) and its
  "strain" the spring rotation (rad), the conjugate pair of the spring in
  its own frame (probed 2026-09-21: identical to ``basicForce`` and
  ``deformation`` of the element at every step). Spring index 1 is the
  y' material (about global X for a column, vertical bending for a beam);
  index 2 the z' material (about global Y for a column, in-plane bending
  for a beam). Signs are the spring's: for a beam, hogging is POSITIVE
  rotation at end i and NEGATIVE at end j, and the installed positive /
  negative strengths follow (Model.IMK_Hinges._create_end_hinge). The
  element-end negation used by the pushover diagnostic's joint equilibrium
  is NOT applied here: a loop is plotted in the spring's own pair.
* Rows are written at every committed analysis step, including the
  sub-steps of a recovery strategy (NTHA's subdivide fallback commits ten
  sub-steps for one record step), so histories are aligned by the time
  column, never by row index.
* Yield is identified from the FULL-RATE history of each spring in two
  ways that are both reported. The path-aware call decomposes every step
  into an elastic part dM / Ke and a plastic part dtheta - dM / Ke and
  counts a step as plastic when its tangent is below half the installed
  Ke; accumulated positive and negative plastic rotation then say whether
  the spring yielded in each direction, INCLUDING yielding on the shifted
  (kinematic) branch after a reversal, where IMKBilin yields on the other
  side while the rotation is still on the first side and the moment is
  below that sign's Fy (review finding, 2026-09-21). The virgin-backbone
  call (peak rotation beyond Fy / Ke of that sign, peak moment reaching
  that sign's Fy, both from the extrema) is kept beside it, as is the
  production one-sided peak-based plastic rotation of
  Ground_Motion_Main._hinge_backbone_rows; disagreements are flagged.
  This path-aware decomposition is a legacy Bilin diagnostic, not a validated
  plastic-rotation measure for peak-oriented or pinched reloading. For those
  laws we report virgin-envelope exceedance and leave accumulated plastic
  rotation unavailable. Reversal yielding below that envelope can be missed.
  Dissipated energy is the work integral less the change in stored
  elastic energy on Ke (an exact discrete identity: zero for an elastic
  spring on any open path). Capping / ultimate flags use the installed
  theta_p / theta_u.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import openseespy.opensees as ops

import Structure_Parameters as sp
from Analysis.Pushover_Diagnostic import element_inventory, hinge_inventory, model_audit
from Model.IMK_Hinges import hinge_registry

DIAGNOSTIC_VERSION = "hinge_hysteresis_diagnostic_v3_material_aware"
SPRING_MATERIAL_INDEX = {"y": 1, "z": 2}
RECORDER_PRECISION = 12
YIELD_MOMENT_TOLERANCE = 1e-3      # virgin backbone: peak |M| >= (1 - tol) Fy of that sign
YIELD_ROTATION_TOLERANCE = 1e-6    # virgin backbone: peak |theta| > (1 + tol) Fy / Ke of that sign
PLASTIC_TANGENT_FRACTION = 0.5     # a step whose tangent dM/dtheta is below this fraction of Ke is plastic
PLASTIC_ROTATION_TOLERANCE = 1e-6  # accumulated plastic rotation (rad) above which a direction counts as yielded


# ---------------------------------------------------------------------------
# Installed-design verification
# ---------------------------------------------------------------------------

def _close(a, b, rel=1e-9):
    try:
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-9)
    except (TypeError, ValueError):
        return a == b


def verify_installed_design(record):
    """Compare the live Structure_Parameters state and the hinge registry with the saved record.

    Returns {"consistent": bool, "checks": [...], "differences": [...],
    "registry": {...}}. Sections, longitudinal bars, hoops, covers, slab
    thickness and layout presence, and the four beam families' hinge
    strengths are compared value by value; column hinge strengths are
    recorded with their axial estimates (the record holds no per-member
    column hinge strength to compare against, which is stated, not hidden).
    """
    if record.get("design_mode") == "grouped":
        return verify_installed_grouped_design(record)
    sections, rebar, slab = record["sections"], record["reinforcement"], record.get("slab") or {}
    checks, differences = [], []

    def check(name, saved, installed, rel=1e-9):
        ok = _close(saved, installed, rel)
        checks.append({"item": name, "saved": saved, "installed": installed, "match": ok})
        if not ok:
            differences.append(f"{name}: saved {saved!r} installed {installed!r}")

    for key, attr in (("b_col_in", "B_COL"), ("h_col_in", "H_COL"), ("fc_col_ksi", "FC_COL_KSI"),
                      ("b_beam_in", "B_BEAM"), ("h_beam_in", "H_BEAM"), ("fc_beam_ksi", "FC_BEAM_KSI")):
        check(f"sections.{key}", sections[key], getattr(sp, attr))
    for key, attr in (("col_bar_size", "COL_BAR_SIZE"), ("col_top_bars", "COL_TOP_BARS"), ("col_bot_bars", "COL_BOT_BARS"),
                      ("col_side_bars", "COL_SIDE_BARS"), ("beam_bar_size", "BEAM_BAR_SIZE"), ("beam_top_bars", "BEAM_TOP_BARS"),
                      ("beam_bot_bars", "BEAM_BOT_BARS"), ("beam_side_bars", "BEAM_SIDE_BARS"),
                      ("col_stirrup_bar_size", "COL_STIRRUP_BAR_SIZE"), ("col_stirrup_legs", "COL_STIRRUP_LEGS"),
                      ("col_stirrup_spacing_in", "COL_STIRRUP_SPACING"), ("beam_stirrup_bar_size", "BEAM_STIRRUP_BAR_SIZE"),
                      ("beam_stirrup_legs", "BEAM_STIRRUP_LEGS"), ("beam_stirrup_spacing_in", "BEAM_STIRRUP_SPACING"),
                      ("beam_clear_cover_in", "BEAM_CLEAR_COVER_IN"), ("col_clear_cover_in", "COL_CLEAR_COVER_IN")):
        check(f"reinforcement.{key}", rebar[key], getattr(sp, attr))
    check("reinforcement.col_stirrup_legs_by_direction", rebar.get("col_stirrup_legs_by_direction"),
          getattr(sp, "COL_STIRRUP_LEGS_BY_DIRECTION", None))
    check("slab.thickness_in", slab.get("thickness_in"), sp.SLAB_THICKNESS_IN)
    saved_layout = bool(((record.get("slab_reinforcement") or {}).get("layout")))
    installed_layout = bool(((sp.SLAB_REINFORCEMENT or {}).get("layout")))
    check("slab_reinforcement.layout_present", saved_layout, installed_layout)
    check("gravity_load_model", record.get("gravity_load_model"), sp.effective_gravity_load_model())
    materials = record.get("materials") or {}
    check("materials.fy_ksi", materials.get("fy_ksi"), sp.FY_KSI)
    check("materials.es_ksi", materials.get("es_ksi"), sp.ES_KSI)
    # Hinge registry against the saved beam families.
    registry = hinge_registry()
    inventory = element_inventory()
    expected_hinged = (len(inventory["columns"]) if sp.IMK_APPLY_TO_COLUMNS else 0) + (len(inventory["beams"]) if sp.IMK_APPLY_TO_BEAMS else 0)
    check("registry.hinged_member_count", expected_hinged, len(registry))
    families_saved = record.get("beam_slab_families") or {}
    evidence_gaps = []
    by_family = {}
    # Per member, end and sign: the record's beam_slab_strengths entries ('<tag>/<end>/<positive|negative>',
    # mn_composite_kip_in) are the strengths the design's joint checks used; the installed hinge end must
    # yield at exactly those (hogging = negative sign, sagging = positive sign) when a layout exists.
    entries_saved = record.get("beam_slab_strengths") or {}
    per_end_checked, per_end_mismatch = 0, []
    for entry in registry.values():
        if entry["member_type"] == "column":
            continue
        fam = entry.get("beam_family")
        by_family.setdefault(fam, {"hogging": set(), "sagging": set(), "hogging_ends": set(), "sagging_ends": set(), "count": 0})
        by_family[fam]["hogging"].add(round(entry["yield_moment_y_hogging_kip_in"], 6))
        by_family[fam]["sagging"].add(round(entry["yield_moment_y_sagging_kip_in"], 6))
        for end in ("i", "j"):
            by_family[fam]["hogging_ends"].add(round(entry[f"yield_moment_y_hogging_{end}_kip_in"], 6))
            by_family[fam]["sagging_ends"].add(round(entry[f"yield_moment_y_sagging_{end}_kip_in"], 6))
            for sign, key in (("negative", f"yield_moment_y_hogging_{end}_kip_in"), ("positive", f"yield_moment_y_sagging_{end}_kip_in")):
                saved_entry = entries_saved.get(f"{entry['ele_tag']}/{end}/{sign}")
                saved_value = (saved_entry or {}).get("mn_composite_kip_in") if isinstance(saved_entry, dict) else None
                if saved_value is None:
                    continue
                per_end_checked += 1
                if not _close(saved_value, entry[key], 1e-6):
                    per_end_mismatch.append(f"beam {entry['ele_tag']} end {end} {sign}: record {saved_value} installed {entry[key]}")
        by_family[fam]["count"] += 1
    if per_end_checked:
        check("registry.beam_end_strengths_vs_record_entries", f"{per_end_checked} entries", f"{per_end_checked - len(per_end_mismatch)} match", rel=0)
        if per_end_mismatch:
            differences.extend(per_end_mismatch[:20])
            checks[-1]["match"] = False
        else:
            checks[-1]["match"] = True
            differences[:] = [d for d in differences if not d.startswith("registry.beam_end_strengths")]
    for fam, data in sorted(by_family.items()):
        saved = families_saved.get(fam) if isinstance(families_saved, dict) else None
        saved_neg = (saved or {}).get("mn_negative_kip_in")
        saved_pos = (saved or {}).get("mn_positive_kip_in")
        # One convention on both sides (2026-09-21): the hinge yields at the record's family strengths whether or
        # not a layout exists -- with a layout the composite section with its developed mats, without one the
        # bare rectangle in hogging and the flange concrete (no mats) in sagging (SMRF_Beam_Slab_Strength
        # .composite_beam_strengths with layout None, used by Model.IMK_Hinges.beam_yield_moments alike).
        rule = ("composite family strength with developed slab mats (layout established)" if saved_layout else
                "rectangular hogging and flange-concrete sagging with no mats (no layout established)")
        check(f"registry.{fam}.hogging_kip_in", saved_neg, max(data["hogging"]) if data["hogging"] else None, rel=1e-6)
        check(f"registry.{fam}.sagging_kip_in", saved_pos, max(data["sagging"]) if data["sagging"] else None, rel=1e-6)
        checks.append({"item": f"registry.{fam}.members", "saved": None, "installed": data["count"], "match": True, "rule": rule,
                       "note": ("per-end hogging strengths " + str(sorted(data["hogging_ends"])) + ", sagging " + str(sorted(data["sagging_ends"])) +
                                " (an exterior end without developed slab bars yields lower)")
                       if len(data["hogging_ends"]) > 1 or len(data["sagging_ends"]) > 1 else None})
    columns = [e for e in registry.values() if e["member_type"] == "column"]
    registry_summary = {
        "hinged_members": len(registry), "beam_families": {fam: {"count": d["count"], "hogging_kip_in": sorted(d["hogging"]),
                                                                 "sagging_kip_in": sorted(d["sagging"])} for fam, d in by_family.items()},
        "beam_end_strengths_verified_against_record_entries": per_end_checked,
        "column_hinges": {"count": len(columns),
                          "yield_moment_range_kip_in": [min(e["yield_moment_y_kip_in"] for e in columns), max(e["yield_moment_y_kip_in"] for e in columns)] if columns else None,
                          "axial_estimate_range_kip": [min(e["axial_kip"] for e in columns), max(e["axial_kip"] for e in columns)] if columns else None,
                          "basis": "nominal P-M at the tributary gravity axial estimate fixed at build time; the same value for both axes; "
                                   "the saved record holds no per-member column hinge strength, so this is recorded, not verified"},
        "rotation_capacities_and_ke": {"theta_p_range": [min(e["theta_p"] for e in registry.values()), max(e["theta_p"] for e in registry.values())] if registry else None,
                                       "theta_u_range": [min(e["theta_u"] for e in registry.values()), max(e["theta_u"] for e in registry.values())] if registry else None,
                                       "basis": "Haselton 2008 per member from the installed cage and hoops, and Ke = IMK_HINGE_STIFFNESS_FACTOR x 6EI/L; the "
                                                "record stores no independent copy (production hinge_backbone.csv is written from this same registry), so "
                                                "these are recorded, not verified"},
        "stiffness_mode": getattr(sp, "IMK_HINGE_STIFFNESS_MODE", None), "stiffness_factor": sp.IMK_HINGE_STIFFNESS_FACTOR,
        "beam_spring_ke_family_note": ("the installed spring Ke is computed from the member's own properties (its line's T-section I, "
                                       "a column's axial state) since 2026-09-24 (Model.IMK_Hinges._create_end_hinge passes props) and is "
                                       "recorded per member as ke_y/ke_z_kip_in_per_rad; before that the INTERIOR line's I served every "
                                       "family and edge-line springs were stiffer than n x 6EI/L of their own member"),
    }
    return {"consistent": not differences, "checks": checks, "differences": differences, "registry": registry_summary,
            "design_evidence_gaps": evidence_gaps,
            "verified": ["sections", "longitudinal bars", "hoops (bar, legs, spacing, legs by direction)", "covers", "slab thickness",
                         "slab layout presence", "gravity load model", "steel materials", "hinged member count",
                         "beam family hogging/sagging strengths", "beam end strengths per member, end and sign against the record's "
                         "beam_slab_strengths entries" if per_end_checked else "beam end strengths (no record entries to compare)"],
            "recorded_not_verified": ["column hinge strengths and axial estimates", "theta_p / theta_pc / theta_u", "spring Ke"]}


def verify_installed_grouped_design(record):
    """Compare the installed grouped design and the hinge registry with a saved grouped record, member by member.

    Every hinged member's registry entry carries the design its hinges were built from; it is compared with
    the record's own row for that physical member (group, section, bars, hoops), each beam end's hinge
    yield strengths with the record's ``beam_end_strengths``, and each beam's bar rows with the record's
    group rows. Column hinge strengths, rotation capacities and spring stiffness are recorded with their
    ranges per group; the record holds no independent copy of them, which is stated, not hidden.
    """
    from Model import Member_Groups as mg
    slab = record.get("slab") or {}
    block = record.get("member_groups") or {}
    checks, differences = [], []

    def check(name, saved, installed, rel=1e-9):
        ok = _close(saved, installed, rel)
        checks.append({"item": name, "saved": saved, "installed": installed, "match": ok})
        if not ok:
            differences.append(f"{name}: saved {saved!r} installed {installed!r}")

    state = mg.active()
    check("member_groups.installed", True, state is not None)
    check("member_groups.sha256", block.get("sha256"), None if state is None else state.identity())
    check("slab.thickness_in", slab.get("thickness_in"), sp.SLAB_THICKNESS_IN)
    check("slab_reinforcement.layout_present", bool((record.get("slab_reinforcement") or {}).get("layout")),
          bool((sp.SLAB_REINFORCEMENT or {}).get("layout")))
    check("gravity_load_model", record.get("gravity_load_model"), sp.effective_gravity_load_model())
    materials = record.get("materials") or {}
    check("materials.fy_ksi", materials.get("fy_ksi"), sp.FY_KSI)
    check("materials.es_ksi", materials.get("es_ksi"), sp.ES_KSI)
    for key, attr in (("beam_clear_cover_in", "BEAM_CLEAR_COVER_IN"), ("col_clear_cover_in", "COL_CLEAR_COVER_IN")):
        check(f"materials.{key}", materials.get(key), getattr(sp, attr))
    registry = hinge_registry()
    inventory = element_inventory()
    expected_hinged = (len(inventory["columns"]) if sp.IMK_APPLY_TO_COLUMNS else 0) + (len(inventory["beams"]) if sp.IMK_APPLY_TO_BEAMS else 0)
    check("registry.hinged_member_count", expected_hinged, len(registry))
    rows = {int(row["member_tag"]): row for row in block.get("members") or []}
    design_keys = ("b_in", "h_in", "fc_ksi", "bar_size", "top_bars", "bot_bars", "side_bars", "stirrup_bar_size", "stirrup_legs",
                   "stirrup_spacing_in", "stirrup_legs_by_direction")
    saved_strengths = record.get("beam_end_strengths") or {}
    saved_rows = (record.get("bar_layers") or {}).get("groups") or {}
    member_mismatch, end_mismatch, row_mismatch, ends_checked, by_group = [], [], [], 0, {}
    for entry in registry.values():
        tag = int(entry["ele_tag"])
        row = rows.get(tag)
        installed = entry.get("installed_design") or {}
        if row is None or entry.get("group_id") != row.get("group_id"):
            member_mismatch.append(f"member {tag}: group {entry.get('group_id')!r} installed, {None if row is None else row.get('group_id')!r} saved")
        else:
            bad = [key for key in design_keys if not _close(row.get(key), installed.get(key))]
            if bad:
                member_mismatch.append(f"member {tag} ({row['group_id']}): {', '.join(bad)} differ from the saved member row")
        group = by_group.setdefault(entry.get("group_id"), {"members": 0, "hogging": set(), "sagging": set(), "yield": set(), "axial": []})
        group["members"] += 1
        if entry["member_type"] == "column":
            group["yield"].add(round(entry["yield_moment_y_kip_in"], 6))
            group["axial"].append(entry["axial_kip"])
            continue
        saved = saved_strengths.get(str(tag)) or {}
        for end in ("i", "j"):
            for sign in ("hogging", "sagging"):
                installed_value = entry[f"yield_moment_y_{sign}_{end}_kip_in"]
                group[sign].add(round(installed_value, 6))
                ends_checked += 1
                if not _close(saved.get(f"{sign}_{end}_kip_in"), installed_value, 1e-6):
                    end_mismatch.append(f"beam {tag} end {end} {sign}: record {saved.get(f'{sign}_{end}_kip_in')!r} installed {installed_value}")
        installed_rows, group_rows = entry.get("beam_bar_rows") or {}, saved_rows.get(entry.get("group_id")) or {}
        for face in ("top", "bottom"):
            a, b = installed_rows.get(face) or {}, group_rows.get(face) or {}
            if (a.get("per_layer") != b.get("per_layer") or len(a.get("offsets_in") or []) != len(b.get("offsets_in") or [])
                    or any(not _close(x, y) for x, y in zip(a.get("offsets_in") or [], b.get("offsets_in") or []))):
                row_mismatch.append(f"beam {tag} {face} bar rows differ from the saved rows of {entry.get('group_id')}")
    for name, total, bad in (("registry.member_designs_vs_record_rows", len(registry), member_mismatch),
                             ("registry.beam_end_strengths_vs_record", ends_checked, end_mismatch),
                             ("registry.beam_bar_rows_vs_record", sum(1 for e in registry.values() if e["member_type"] != "column"),
                              row_mismatch)):
        checks.append({"item": name, "saved": f"{total} entries", "installed": f"{total - len(bad)} match", "match": not bad})
        differences.extend(bad[:20])
    groups = {gid: {"members": data["members"],
                    **({"yield_moment_range_kip_in": [min(data["yield"]), max(data["yield"])],
                        "axial_estimate_range_kip": [min(data["axial"]), max(data["axial"])]} if data["yield"] else
                       {"hogging_kip_in": sorted(data["hogging"]), "sagging_kip_in": sorted(data["sagging"])})}
              for gid, data in sorted(by_group.items(), key=lambda item: str(item[0]))}
    registry_summary = {
        "hinged_members": len(registry), "groups": groups,
        "beam_end_strengths_verified_against_record_entries": ends_checked,
        "column_hinges": {"basis": "nominal P-M of each column's own section and cage at its tributary gravity axial estimate; the "
                                   "saved record holds no per-member column hinge strength, so this is recorded, not verified"},
        "rotation_capacities_and_ke": {"theta_p_range": [min(e["theta_p"] for e in registry.values()), max(e["theta_p"] for e in registry.values())] if registry else None,
                                       "theta_u_range": [min(e["theta_u"] for e in registry.values()), max(e["theta_u"] for e in registry.values())] if registry else None,
                                       "basis": "Haselton 2008 per member from its own group's cage and hoops, and Ke from the member's own "
                                                "properties; the record stores no independent copy, so these are recorded, not verified"},
        "stiffness_mode": getattr(sp, "IMK_HINGE_STIFFNESS_MODE", None), "stiffness_factor": sp.IMK_HINGE_STIFFNESS_FACTOR,
    }
    return {"consistent": not differences, "checks": checks, "differences": differences, "registry": registry_summary,
            "design_evidence_gaps": [],
            "verified": ["member groups digest", "every hinged member's group, section, longitudinal bars and hoops against its saved row",
                         "covers", "slab thickness", "slab layout presence", "gravity load model", "steel materials",
                         "hinged member count", "beam end strengths per member, end and sign", "beam bar rows per member"],
            "recorded_not_verified": ["column hinge strengths and axial estimates", "theta_p / theta_pc / theta_u", "spring Ke"]}


# ---------------------------------------------------------------------------
# Recorders
# ---------------------------------------------------------------------------

def attach_hinge_recorders(output_dir, hinges):
    """One text recorder per spring axis over every hinge element, in ``hinges`` order.

    Returns the coverage record: file per axis, hinge tag order, column
    layout, precision. Every hinge in the inventory is covered; the
    recorder cannot silently skip an element (OpenSees errors on an unknown
    tag), and the reader checks the column count against the tag list.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tags = [int(h["hinge_tag"]) for h in hinges]
    # Coverage against the DOMAIN, not only the inventory: every zero-length hinge element OpenSees holds.
    domain_hinges = sorted(int(t) for t in ops.getEleTags() if int(t) >= sp.IMK_HINGE_ELEMENT_TAG_BASE)
    uncovered = sorted(set(domain_hinges) - set(tags))
    extra = sorted(set(tags) - set(domain_hinges))
    if uncovered or extra:
        raise ValueError(f"hinge recorder coverage: {len(uncovered)} domain hinges not in the inventory {uncovered[:10]}, "
                         f"{len(extra)} inventory hinges not in the domain {extra[:10]}")
    files = {}
    for axis, index in SPRING_MATERIAL_INDEX.items():
        path = output_dir / f"hinge_spring_{axis}_material_{index}_stressStrain.out"
        ops.recorder("Element", "-file", str(path), "-time", "-precision", RECORDER_PRECISION,
                     "-ele", *tags, "material", index, "stressStrain")
        files[axis] = str(path)
    return {"files": files, "hinge_tag_order": tags, "hinge_count": len(tags), "domain_hinge_count": len(domain_hinges),
            "coverage_complete": True,
            "columns": "time, then (moment kip-in, rotation rad) per hinge in hinge_tag_order",
            "precision_digits": RECORDER_PRECISION, "material_index": dict(SPRING_MATERIAL_INDEX),
            "rows": "one per committed analysis step, including recovery sub-steps; align by time",
            "spring_frame": "spring's own conjugate pair (IMK stress = moment, strain = rotation); no element-end negation"}


def close_recorders():
    """Flush and remove every recorder (the files are complete only after this)."""
    ops.remove("recorders")


def read_hinge_recorders(coverage, time_limit=None):
    """Read the recorder files back: {axis: {"time": (n,), "moment": (n, H), "rotation": (n, H)}} plus the tag order.

    ``time_limit`` keeps only rows at or before that analysis time (the
    retained response window: sub-steps committed by a recovery attempt
    that then failed lie beyond the last snapshot and are dropped, and
    counted in ``rows_dropped_beyond_window``). An empty file (a solve that
    failed before its first commit) gives zero rows, not an error: the
    caller reports the missing response.
    """
    tags = list(coverage["hinge_tag_order"])
    out = {"hinge_tag_order": tags}
    for axis, path in coverage["files"].items():
        path = Path(path)
        data = np.loadtxt(path, ndmin=2) if path.exists() and path.stat().st_size > 0 else np.zeros((0, 1 + 2 * len(tags)))
        expected = 1 + 2 * len(tags)
        if data.shape[1] != expected:
            raise ValueError(f"{path}: {data.shape[1]} columns, expected {expected} for {len(tags)} hinges")
        dropped = 0
        if time_limit is not None and data.shape[0]:
            keep = data[:, 0] <= time_limit + 1e-9
            dropped = int((~keep).sum())
            data = data[keep]
        times = data[:, 0]
        monotone = bool(np.all(np.diff(times) > 0)) if len(times) > 1 else True
        out[axis] = {"time": times, "moment": data[:, 1::2], "rotation": data[:, 2::2], "rows": int(data.shape[0]),
                     "rows_dropped_beyond_window": dropped, "time_strictly_increasing": monotone}
    return out


# ---------------------------------------------------------------------------
# Evaluation against the installed backbones
# ---------------------------------------------------------------------------

def _loop_energy(moment, rotation, ke):
    """Dissipated energy of a spring history (kip-in): the work integral less the change in stored elastic energy.

    W = trapezoid integral of M dtheta from the first recorded row to the
    last; the recoverable part is (M_last^2 - M_first^2) / (2 Ke) on the
    installed elastic stiffness, so an elastic spring on any open path
    returns exactly zero and a yielded spring returns the area its loops
    enclose. After cyclic stiffness deterioration the recoverable term is
    an estimate on Ke; the raw work integral is returned beside it.
    """
    if len(rotation) < 2:
        return 0.0, 0.0
    work = float(np.sum(0.5 * (moment[1:] + moment[:-1]) * np.diff(rotation)))
    stored = float((moment[-1] ** 2 - moment[0] ** 2) / (2.0 * ke)) if ke > 0 else 0.0
    return work - stored, work


def plastic_increments(moment, rotation, ke, tangent_fraction=PLASTIC_TANGENT_FRACTION):
    """Signed plastic rotation increment of every step: dtheta - dM / Ke on steps whose tangent is below tangent_fraction x Ke.

    Elastic loading and unloading branches of IMKBilin run at Ke (or a
    deteriorated stiffness still far above the hardening slope), the
    hardening branch at about Fmax/Fy - 1 times Fy / theta_p (three
    orders below Ke) and the post-capping branch below zero, so the
    tangent separates the branches. On a step that crosses a yield point
    the decomposition still gives the plastic part exactly (the elastic
    part of any step is dM / Ke). Returns (increments, plastic_mask).
    """
    d_theta = np.diff(rotation)
    d_m = np.diff(moment)
    with np.errstate(divide="ignore", invalid="ignore"):
        tangent = np.where(np.abs(d_theta) > 0, d_m / np.where(np.abs(d_theta) > 0, d_theta, 1.0), np.inf)
    plastic = (np.abs(d_theta) > 0) & (tangent < tangent_fraction * ke)
    increments = np.where(plastic, d_theta - d_m / ke, 0.0)
    return increments, plastic


def evaluate_hinge_histories(recorded, hinges, moment_tolerance=YIELD_MOMENT_TOLERANCE, rotation_tolerance=YIELD_ROTATION_TOLERANCE,
                             plastic_tolerance=PLASTIC_ROTATION_TOLERANCE):
    """Per hinge and spring axis: full-rate extrema, path-aware and virgin-backbone yield calls per sign, plastic rotation, energy.

    For Bilin, ``yielded*`` and ``plastic_rotation_*_accumulated`` use the
    legacy path-aware tangent proxy. For PeakOriented/Pinching, ``yielded*``
    reports virgin-envelope exceedance and accumulated plastic quantities
    are None. These are diagnostics, not internal material state variables.
    ``virgin_yield_*`` are the
    origin-referenced criteria on the extrema; ``plastic_rotation_peak*``
    is the production one-sided measure (peak rotation less that sign's
    elastic limit, Ground_Motion_Main._hinge_backbone_rows);
    ``spring_rotation_over_elastic_limit_*`` is the spring's own ratio
    (the spring is IMK_HINGE_STIFFNESS_FACTOR times stiffer than the
    member, so this is not a member ductility). A spring with no rows
    (missing response) yields a row with ``response_missing`` True.
    """
    index_of = {int(tag): k for k, tag in enumerate(recorded["hinge_tag_order"])}
    rows = []
    for h in hinges:
        material_type = h.get("material_type", "IMKBilin")  # compatibility with historical inventories
        if material_type not in ("IMKBilin", "IMKPeakOriented", "IMKPinching"):
            raise ValueError(f"Unsupported hinge diagnostic material {material_type!r}")
        use_bilin_path = material_type == "IMKBilin"
        k = index_of[int(h["hinge_tag"])]
        for axis in ("y", "z"):
            block = recorded.get(axis)
            if block is None:
                continue
            spring = h["springs"][axis]
            fy_pos, fy_neg, ke = spring["fy_positive_kip_in"], spring["fy_negative_kip_in"], spring["ke"]
            ty_pos, ty_neg = spring["theta_y_positive"], spring["theta_y_negative"]
            base = {"hinge_tag": int(h["hinge_tag"]), "member_tag": h["member_tag"], "member_type": h["member_type"], "end": h["end"],
                    "material_type": material_type,
                    "installed_material": h.get("installed_materials", {}).get(axis),
                    "yield_detection_basis": "legacy_bilin_tangent_proxy" if use_bilin_path else "virgin_envelope_exceedance",
                    "accumulated_plastic_rotation_available": use_bilin_path,
                    "energy_measure": "work_minus_initial_Ke_storage_estimate",
                    "spring_axis": axis, "beam_family": h.get("beam_family"), "joint_node": h["joint_node"],
                    "fy_positive_kip_in": fy_pos, "fy_negative_kip_in": fy_neg, "ke_kip_in_per_rad": ke,
                    "theta_y_positive": ty_pos, "theta_y_negative": ty_neg,
                    "theta_p": h["theta_p"], "theta_pc": h["theta_pc"], "theta_u": h["theta_u"]}
            if block["rows"] == 0:
                rows.append({**base, "response_missing": True, "yielded": None, "yielded_positive": None, "yielded_negative": None,
                             "samples": 0})
                continue
            moment, rotation = block["moment"][:, k], block["rotation"][:, k]
            max_rot, min_rot = float(rotation.max()), float(rotation.min())
            max_m, min_m = float(moment.max()), float(moment.min())
            i_max_rot, i_min_rot = int(rotation.argmax()), int(rotation.argmin())
            # Virgin-backbone (origin-referenced) criteria on the extrema.
            v_pos_rot = ty_pos is not None and max_rot > (1.0 + rotation_tolerance) * ty_pos
            v_neg_rot = ty_neg is not None and -min_rot > (1.0 + rotation_tolerance) * ty_neg
            v_pos_mom = fy_pos > 0 and max_m >= (1.0 - moment_tolerance) * fy_pos
            v_neg_mom = fy_neg > 0 and -min_m >= (1.0 - moment_tolerance) * fy_neg
            virgin_yielded = bool((v_pos_rot and v_pos_mom) or (v_neg_rot and v_neg_mom))
            # A reduced reloading tangent in PeakOriented/Pinching is not
            # evidence of a separable plastic-rotation increment. Preserve
            # missing values rather than mislabeling this contribution.
            acc_pos = acc_neg = first_plastic = plastic_steps = None
            if use_bilin_path:
                increments, plastic_mask = plastic_increments(moment, rotation, ke)
                acc_pos = float(increments[increments > 0].sum())
                acc_neg = float(-increments[increments < 0].sum())
                plastic_steps = int(plastic_mask.sum())
                first_plastic = int(np.argmax(plastic_mask)) + 1 if plastic_mask.any() else None
                yielded_pos, yielded_neg = acc_pos > plastic_tolerance, acc_neg > plastic_tolerance
            else:
                yielded_pos, yielded_neg = bool(v_pos_rot and v_pos_mom), bool(v_neg_rot and v_neg_mom)
            yielded = yielded_pos or yielded_neg
            # Production one-sided peak measure.
            peak_pos = max(0.0, max_rot - (ty_pos or 0.0))
            peak_neg = max(0.0, -min_rot - (ty_neg or 0.0))
            peak = max(peak_pos, peak_neg)
            dissipated, work = _loop_energy(moment, rotation, ke)
            rows.append({
                **base, "response_missing": False,
                "rotation_max": max_rot, "rotation_min": min_rot, "moment_max_kip_in": max_m, "moment_min_kip_in": min_m,
                "time_of_rotation_max": float(block["time"][i_max_rot]), "time_of_rotation_min": float(block["time"][i_min_rot]),
                "moment_at_rotation_max_kip_in": float(moment[i_max_rot]), "moment_at_rotation_min_kip_in": float(moment[i_min_rot]),
                "yielded": bool(yielded), "yielded_positive": bool(yielded_pos), "yielded_negative": bool(yielded_neg),
                "plastic_rotation_positive_accumulated": acc_pos, "plastic_rotation_negative_accumulated": acc_neg,
                "plastic_rotation_accumulated": acc_pos + acc_neg if use_bilin_path else None, "plastic_steps": plastic_steps,
                "time_of_first_plastic_step": float(block["time"][first_plastic]) if first_plastic is not None else None,
                "virgin_yield_positive_by_rotation": bool(v_pos_rot), "virgin_yield_positive_by_moment": bool(v_pos_mom),
                "virgin_yield_negative_by_rotation": bool(v_neg_rot), "virgin_yield_negative_by_moment": bool(v_neg_mom),
                "virgin_yielded": virgin_yielded,
                "criteria_disagree": bool(virgin_yielded != yielded) if use_bilin_path else None,
                "plastic_rotation_peak_positive": peak_pos, "plastic_rotation_peak_negative": peak_neg, "plastic_rotation_peak": peak,
                "plastic_rotation": peak,
                "spring_rotation_over_elastic_limit_positive": (max_rot / ty_pos) if ty_pos else None,
                "spring_rotation_over_elastic_limit_negative": (-min_rot / ty_neg) if ty_neg else None,
                "damage_ratio": (peak / h["theta_p"]) if h["theta_p"] > 0 else None,
                "past_capping": bool(h["theta_p"] > 0 and peak >= h["theta_p"]),
                "past_ultimate": bool(max(max_rot, -min_rot) >= h["theta_u"]),
                "dissipated_energy_kip_in": dissipated, "work_integral_kip_in": work,
                "samples": int(len(rotation)),
            })
    return rows


def yield_summary(rows):
    present = [r for r in rows if not r.get("response_missing")]
    missing = [r for r in rows if r.get("response_missing")]
    yielded = [r for r in present if r["yielded"]]
    by_type = {}
    for r in present:
        key = (r["member_type"], r["spring_axis"])
        entry = by_type.setdefault(f"{key[0]}/{key[1]}", {"springs": 0, "yielded": 0, "yielded_both_signs": 0, "criteria_disagree": 0, "past_capping": 0})
        entry["springs"] += 1
        entry["yielded"] += int(r["yielded"])
        entry["yielded_both_signs"] += int(r["yielded_positive"] and r["yielded_negative"])
        entry["criteria_disagree"] += int(bool(r["criteria_disagree"]))
        entry["past_capping"] += int(r["past_capping"])
    return {"springs": len(rows), "springs_with_response": len(present), "springs_missing_response": len(missing),
            "yielded_springs": len(yielded),
            "yielded_hinges": len({(r["hinge_tag"]) for r in yielded}),
            "yielded_members": len({r["member_tag"] for r in yielded}),
            "yielded_both_signs": sum(1 for r in yielded if r["yielded_positive"] and r["yielded_negative"]),
            "by_type": by_type,
            "max_plastic_rotation_peak": max((r["plastic_rotation_peak"] for r in present), default=0.0),
            "max_plastic_rotation_accumulated": max((r["plastic_rotation_accumulated"] for r in present
                                                     if r["plastic_rotation_accumulated"] is not None), default=None),
            "springs_with_accumulated_plastic_rotation": sum(r["plastic_rotation_accumulated"] is not None for r in present),
            "yield_detection_bases": sorted({r.get("yield_detection_basis", "legacy_bilin_tangent_proxy") for r in present}),
            "max_damage_ratio": max((r["damage_ratio"] or 0.0 for r in present), default=0.0),
            "max_spring_rotation_over_elastic_limit": max((max(r["spring_rotation_over_elastic_limit_positive"] or 0.0,
                                                               r["spring_rotation_over_elastic_limit_negative"] or 0.0) for r in present), default=0.0),
            "total_dissipated_energy_kip_in": float(sum(r["dissipated_energy_kip_in"] for r in present)),
            "criteria_disagreements": sum(1 for r in present if r["criteria_disagree"]),
            "yield_criteria": {"path_aware": f"accumulated plastic rotation in a direction > {PLASTIC_ROTATION_TOLERANCE:g} rad, plastic steps "
                                             f"being those with tangent < {PLASTIC_TANGENT_FRACTION:g} Ke (dtheta - dM/Ke)",
                               "virgin_rotation": f"peak |theta| > (1 + {YIELD_ROTATION_TOLERANCE:g}) Fy/Ke of that sign",
                               "virgin_moment": f"peak |M| >= (1 - {YIELD_MOMENT_TOLERANCE:g}) Fy of that sign",
                               "yielded": "Bilin: legacy tangent proxy. PeakOriented/Pinching: virgin-envelope exceedance; "
                                          "may miss reversal yielding below the original envelope; not an internal material-state query",
                               "energy": "work minus initial-Ke elastic-storage estimate; raw work integral also reported",
                               "plastic_rotation": "peak-based one-sided measure (production definition); accumulated values beside it",
                               "spring_rotation_over_elastic_limit": "spring-level ratio; the spring is IMK_HINGE_STIFFNESS_FACTOR x the member "
                                                                     "stiffness, so this is not a member ductility"}}


def stride_justification(recorded, hinges, rows, strides=(2, 4, 8)):
    """Compare decimated histories (every k-th row) with the full-rate ones: yield calls, extrema and energy.

    Justifies or refutes a production stride: for each stride the share of
    full-rate yielded springs the decimated history still calls yielded,
    the largest relative shortfall in peak rotation, peak moment and
    dissipated energy over the yielded springs, and the largest absolute
    shortfall in plastic rotation.
    """
    full = {(r["hinge_tag"], r["spring_axis"]): r for r in rows if not r.get("response_missing")}
    yielded_keys = [k for k, r in full.items() if r["yielded"]]
    out = {}
    for stride in strides:
        decimated = {"hinge_tag_order": recorded["hinge_tag_order"]}
        for axis in ("y", "z"):
            if axis in recorded:
                b = recorded[axis]
                decimated[axis] = {"time": b["time"][::stride], "moment": b["moment"][::stride], "rotation": b["rotation"][::stride],
                                   "rows": int(b["moment"][::stride].shape[0])}
        dec_rows = {(r["hinge_tag"], r["spring_axis"]): r for r in evaluate_hinge_histories(decimated, hinges) if not r.get("response_missing")}
        recovered = sum(1 for k in yielded_keys if dec_rows[k]["yielded"])
        false_new = sum(1 for k, r in dec_rows.items() if r["yielded"] and not full[k]["yielded"])
        rot_short, mom_short, energy_short, plastic_short = 0.0, 0.0, 0.0, 0.0
        for k in yielded_keys:
            f, d = full[k], dec_rows[k]
            f_rot = max(abs(f["rotation_max"]), abs(f["rotation_min"])); d_rot = max(abs(d["rotation_max"]), abs(d["rotation_min"]))
            f_mom = max(abs(f["moment_max_kip_in"]), abs(f["moment_min_kip_in"])); d_mom = max(abs(d["moment_max_kip_in"]), abs(d["moment_min_kip_in"]))
            rot_short = max(rot_short, (f_rot - d_rot) / f_rot if f_rot > 0 else 0.0)
            mom_short = max(mom_short, (f_mom - d_mom) / f_mom if f_mom > 0 else 0.0)
            if f["dissipated_energy_kip_in"] > 0:
                energy_short = max(energy_short, (f["dissipated_energy_kip_in"] - d["dissipated_energy_kip_in"]) / f["dissipated_energy_kip_in"])
            plastic_short = max(plastic_short, f["plastic_rotation_peak"] - d["plastic_rotation_peak"])
        out[str(stride)] = {"rows_full_rate": recorded["y"]["rows"] if "y" in recorded else None, "rows_decimated": decimated["y"]["rows"] if "y" in decimated else None,
                            "yielded_full_rate": len(yielded_keys), "yielded_recovered": recovered, "yielded_missed": len(yielded_keys) - recovered,
                            "yielded_only_in_decimated": false_new,
                            "max_relative_shortfall_peak_rotation": rot_short, "max_relative_shortfall_peak_moment": mom_short,
                            "max_relative_shortfall_dissipated_energy": energy_short, "max_absolute_shortfall_plastic_rotation_rad": plastic_short}
    return {"strides": out, "production_strides": {"hinge_rotation": getattr(sp, "NTHA_HINGE_HISTORY_STRIDE", None),
                                                   "element_force": getattr(sp, "NTHA_ELEMENT_HISTORY_STRIDE", None)},
            "basis": "decimation of the full-rate recorder rows (every k-th committed row), re-evaluated with the same criteria; peaks and "
                     "yield calls of the yielded springs compared with the full-rate ones"}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

LOOP_COLOR = "#1f4e9c"
ZERO_LINE = {"color": "black", "lw": 1.1, "alpha": 0.9, "zorder": 1}


def _loop_axis(ax, r, rotation, moment, title, fontsize=10, legend=True):
    """One readable moment-rotation loop: symmetric frame on the loop, bold zero lines, reference lines and labels."""
    x_lim = 1.2 * max(abs(r["rotation_max"]), abs(r["rotation_min"]), 1e-6) * 100.0
    y_lim = 1.15 * max(abs(r["moment_max_kip_in"]), abs(r["moment_min_kip_in"]),
                       sp.IMK_FMAXFY_POS * r["fy_positive_kip_in"], sp.IMK_FMAXFY_NEG * r["fy_negative_kip_in"])
    ax.axhline(0.0, **ZERO_LINE)
    ax.axvline(0.0, **ZERO_LINE)
    ax.plot(rotation * 100.0, moment, lw=0.8, color=LOOP_COLOR, alpha=0.95, zorder=3, label="spring response")
    ke = r["ke_kip_in_per_rad"]
    ax.plot([-x_lim, x_lim], [-x_lim / 100.0 * ke, x_lim / 100.0 * ke], ls="--", lw=1.0, color="0.45", zorder=2, label="installed Ke")
    ax.axhline(r["fy_positive_kip_in"], color="#c0392b", ls=":", lw=1.4, zorder=2, label=f"installed Fy  +{r['fy_positive_kip_in']:.0f} / -{r['fy_negative_kip_in']:.0f}")
    ax.axhline(-r["fy_negative_kip_in"], color="#c0392b", ls=":", lw=1.4, zorder=2)
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_xlabel("spring rotation (%)", fontsize=fontsize)
    ax.set_ylabel("spring moment (kip-in)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    ax.grid(True, alpha=0.25, zorder=0)
    ax.set_title(title, fontsize=fontsize)
    if legend:
        ax.legend(fontsize=fontsize - 2, loc="lower right", framealpha=0.9)


def _loop_title(r, detail=True):
    head = (f"{r['member_type']} {r['member_tag']} end {r['end']}, spring {r['spring_axis']}"
            f"{' (' + r['beam_family'] + ')' if r['beam_family'] else ''}")
    if not detail:
        return head
    line2 = (f"rotation {100 * r['rotation_min']:+.2f}% to {100 * r['rotation_max']:+.2f}%   "
             f"peak beyond elastic limit {r['plastic_rotation_peak']:.4f} rad; {_accumulated_label(r)}")
    line3 = (f"energy estimate {r['dissipated_energy_kip_in']:.0f} kip-in   "
             f"capping at {100 * ((r['theta_y_positive'] or 0.0) + r['theta_p']):.1f}% (damage {r['damage_ratio'] if r['damage_ratio'] is not None else 0:.2f})")
    return "\n".join([head, line2, line3])


def _accumulated_label(row):
    value = row["plastic_rotation_accumulated"]
    return "accumulated plastic rotation unavailable" if value is None else f"accumulated {value:.4f} rad"


def plot_hinge_loops(recorded, hinges, rows, output_dir, limit=24, only_yielded=True):
    """One PNG per selected spring (moment vs rotation with the installed backbone lines) and a gallery."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_of = {int(tag): k for k, tag in enumerate(recorded["hinge_tag_order"])}
    selected = [r for r in rows if not r.get("response_missing") and (r["yielded"] or not only_yielded)]
    selected.sort(key=lambda r: -r["dissipated_energy_kip_in"])
    selected = selected[:limit]
    written = []
    for r in selected:
        k = index_of[r["hinge_tag"]]
        block = recorded[r["spring_axis"]]
        figure, axis = plt.subplots(figsize=(7.5, 6.0))
        _loop_axis(axis, r, block["rotation"][:, k], block["moment"][:, k], _loop_title(r), fontsize=10)
        figure.tight_layout()
        path = output_dir / f"loop_{r['member_type']}_{r['member_tag']}_{r['end']}_{r['spring_axis']}.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(str(path))
    if selected:
        cols = min(3, len(selected))
        nrows = (len(selected) + cols - 1) // cols
        figure, axes = plt.subplots(nrows, cols, figsize=(5.6 * cols, 4.6 * nrows), squeeze=False)
        for ax, r in zip(axes.flat, selected):
            k = index_of[r["hinge_tag"]]
            block = recorded[r["spring_axis"]]
            _loop_axis(ax, r, block["rotation"][:, k], block["moment"][:, k],
                       _loop_title(r, detail=False) + "\n" +
                       f"Fy +{r['fy_positive_kip_in']:.0f} / -{r['fy_negative_kip_in']:.0f} kip-in, E {max(r['dissipated_energy_kip_in'], 0.0):.0f} kip-in",
                       fontsize=9, legend=False)
        for ax in axes.flat[len(selected):]:
            ax.axis("off")
        figure.suptitle("spring moment (kip-in) vs rotation (%): yielded springs ranked by dissipated energy; "
                        "black = zero lines, red dotted = installed Fy per sign, grey dashed = installed Ke", fontsize=11)
        figure.tight_layout(rect=(0, 0, 1, 0.975))
        gallery = output_dir / "loops_gallery.png"
        figure.savefig(gallery, dpi=120)
        plt.close(figure)
        written.append(str(gallery))
    return written


def read_histories_npz(path):
    """The inverse of write_histories_npz: {axis: {"time", "moment", "rotation", "rows"}, "hinge_tag_order"}."""
    loaded = np.load(path)
    out = {"hinge_tag_order": [int(t) for t in loaded["hinge_tag_order"]]}
    for axis in ("y", "z"):
        if f"moment_{axis}" in loaded:
            out[axis] = {"time": loaded[f"time_{axis}"], "moment": loaded[f"moment_{axis}"], "rotation": loaded[f"rotation_{axis}"],
                         "rows": int(loaded[f"moment_{axis}"].shape[0])}
    return out


def plot_selected_loops(recorded, rows, selection, output_dir, name):
    """A titled row of chosen springs (list of (member_tag, end, spring_axis)) at their own scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index_of = {int(tag): k for k, tag in enumerate(recorded["hinge_tag_order"])}
    chosen = [next(r for r in rows if (r["member_tag"], r["end"], r["spring_axis"]) == key) for key in selection]
    figure, axes = plt.subplots(1, len(chosen), figsize=(5.6 * len(chosen), 5.0), squeeze=False)
    for ax, r in zip(axes.flat, chosen):
        k = index_of[r["hinge_tag"]]
        block = recorded[r["spring_axis"]]
        _loop_axis(ax, r, block["rotation"][:, k], block["moment"][:, k],
                   _loop_title(r, detail=False) + "\n" +
                   f"Fy +{r['fy_positive_kip_in']:.0f} / -{r['fy_negative_kip_in']:.0f} kip-in; peak beyond elastic limit "
                   f"{r['plastic_rotation_peak']:.4f} rad; {_accumulated_label(r)}", fontsize=9, legend=(ax is axes.flat[0]))
    figure.tight_layout()
    path = Path(output_dir) / name
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)


def column_story_shears_from_hinges(recorded, inventory, story_h_in):
    """Full-rate chord story shear per story and direction from the column hinge moments (kip).

    A column's chord shear follows from its two end moments: with the
    element's local end moments My (about global Y) and Mz (about global X)
    the story shear it carries is (My_i + My_j) / L in X and (Mz_i + Mz_j) /
    L in Y (Analysis.Pushover_Diagnostic conventions, validated on a
    cantilever); the spring's recorded moment is -Mz for the y spring and
    +My for the z spring, so V_x = (sz_i + sz_j) / L and V_y = -(sy_i +
    sy_j) / L with L the centerline story height. This is the chord shear
    of the constitutive moments only: it carries neither the P-Delta part
    of the global shear nor the velocity-proportional (stiffness-
    proportional damping) part the recorded element forces include, which
    together reached about 6% of the peak base shear on the 0074 pilot; the
    caller quantifies the difference against the exact recorded base
    shear before using these for story loops. Returns {"x": (n, stories),
    "y": (n, stories)}.
    """
    index_of = {int(tag): k for k, tag in enumerate(recorded["hinge_tag_order"])}
    n = recorded["y"]["rows"]
    stories = max(c["story"] for c in inventory["columns"])
    shear = {"x": np.zeros((n, stories)), "y": np.zeros((n, stories))}
    for c in inventory["columns"]:
        tag = c["tag"]
        ki = index_of.get(sp.IMK_HINGE_ELEMENT_TAG_BASE + 10 * tag + 1)
        kj = index_of.get(sp.IMK_HINGE_ELEMENT_TAG_BASE + 10 * tag + 2)
        if ki is None or kj is None:
            continue
        s = c["story"] - 1
        shear["x"][:, s] += (recorded["z"]["moment"][:, ki] + recorded["z"]["moment"][:, kj]) / story_h_in
        shear["y"][:, s] += -(recorded["y"]["moment"][:, ki] + recorded["y"]["moment"][:, kj]) / story_h_in
    return shear


def plot_force_displacement(time, roof_disp, base_shear, floor_disp, story_shear, story_h_in, output_dir, base_shear_check=None):
    """Global and story force-displacement loops.

    ``roof_disp`` and ``base_shear``: {"x": (n,), "y": (n,)} exact histories
    from the NTHA outputs. ``floor_disp``: (n, 2 * stories) signed floor
    master displacements [f1x, f1y, f2x, ...]. ``story_shear``: from
    column_story_shears_from_hinges. Writes base shear vs roof
    displacement (two panels) and story shear vs story drift ratio (stories
    x directions), and returns the paths plus the story-1 check against the
    exact base shear when ``base_shear_check`` is requested.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written, check = [], {}
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, d in zip(axes, ("x", "y")):
        ax.plot(roof_disp[d], base_shear[d], lw=0.7, color=LOOP_COLOR, zorder=3)
        ax.axhline(0.0, **ZERO_LINE); ax.axvline(0.0, **ZERO_LINE)
        ax.set_xlabel(f"roof displacement {d.upper()} (in)"); ax.set_ylabel(f"base shear {d.upper()} (kip)")
        ax.set_title(f"base shear vs roof displacement, {d.upper()}: peak V {np.abs(base_shear[d]).max():.0f} kip, "
                     f"peak u {np.abs(roof_disp[d]).max():.2f} in", fontsize=9)
        ax.grid(alpha=0.3)
    figure.tight_layout()
    path = output_dir / "force_displacement_global.png"
    figure.savefig(path, dpi=130); plt.close(figure); written.append(str(path))
    stories = story_shear["x"].shape[1]
    drift = {}
    for d, col in (("x", 0), ("y", 1)):
        u = floor_disp[:, col::2]                       # (n, stories) signed level displacements in this direction
        lower = np.concatenate([np.zeros((u.shape[0], 1)), u[:, :-1]], axis=1)
        drift[d] = (u - lower) / story_h_in
    if base_shear_check is not None:
        for d in ("x", "y"):
            exact, derived = np.asarray(base_shear[d]), story_shear[d][:, 0]
            check[d] = {"max_abs_difference_kip": float(np.abs(exact - derived).max()), "peak_exact_kip": float(np.abs(exact).max()),
                        "max_relative_to_peak": float(np.abs(exact - derived).max() / max(1e-9, np.abs(exact).max())),
                        "basis": "story-1 hinge-derived chord shear vs the exact recorded base shear (global column forces, P-Delta included)"}
    figure, axes = plt.subplots(stories, 2, figsize=(10, 2.9 * stories), squeeze=False)
    for s in range(stories):
        for j, d in enumerate(("x", "y")):
            ax = axes[stories - 1 - s, j]
            ax.plot(drift[d][:, s] * 100.0, story_shear[d][:, s], lw=0.55, color="tab:blue")
            if s == 0 and base_shear_check is not None:
                ax.plot(drift[d][:, 0] * 100.0, base_shear[d], lw=0.5, color="tab:red", alpha=0.6, label="exact base shear (incl. P-Delta)")
                ax.legend(fontsize=7, loc="lower right")
            ax.axhline(0.0, **ZERO_LINE); ax.axvline(0.0, **ZERO_LINE)
            ax.set_title(f"story {s + 1} {d.upper()}: peak V {np.abs(story_shear[d][:, s]).max():.0f} kip, peak drift {100 * np.abs(drift[d][:, s]).max():.3f}%", fontsize=8)
            ax.set_xlabel("story drift ratio (%)", fontsize=8); ax.set_ylabel("story shear (kip)", fontsize=8)
            ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
    figure.suptitle("story shear (from column hinge moments, chord shear) vs signed story drift, full rate", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    path = output_dir / "force_displacement_stories.png"
    figure.savefig(path, dpi=120); plt.close(figure); written.append(str(path))
    return written, check, drift


def write_histories_npz(recorded, path):
    payload = {"hinge_tag_order": np.asarray(recorded["hinge_tag_order"], dtype=np.int64)}
    for axis in ("y", "z"):
        if axis in recorded:
            payload[f"time_{axis}"] = recorded[axis]["time"]
            payload[f"moment_{axis}"] = recorded[axis]["moment"].astype(np.float64)
            payload[f"rotation_{axis}"] = recorded[axis]["rotation"].astype(np.float64)
    np.savez_compressed(path, **payload)
    return str(path)


def run_audit():
    """The model audit of the built frame (Analysis.Pushover_Diagnostic.model_audit) plus the hinge-specific notes, for the manifest."""
    inventory = element_inventory()
    hinges = hinge_inventory(inventory)
    audit = model_audit(inventory, hinges)
    audit["not_represented"] = list(audit["not_represented"]) + [
        "the same nominal column hinge strength serves bending about both axes (my = mz)",
    ]
    return audit, inventory, hinges


def measured_gravity_state(inventory):
    """Vertical equilibrium of the gravity state: story-1 columns' global end-i forces against the load ledger."""
    from Analysis.Pushover_Diagnostic import DiagnosticSettings, expected_gravity_total_kip
    total = 0.0
    for c in inventory["columns"]:
        if c["story"] == 1:
            total += list(ops.eleResponse(c["tag"], "force"))[2]
    expected, basis = expected_gravity_total_kip(DiagnosticSettings(gravity_dead_factor=1.0, gravity_live_factor=1.0,
                                                                     gravity_self_weight_factor=1.0, live_pattern="all"))
    return {"vertical_reaction_kip": total, "expected_vertical_load_kip": expected, "expected_basis": basis,
            "residual_kip": (total - expected) if expected is not None else None,
            "reaction_basis": "sum of the story-1 columns' global end-i vertical forces (nodeReaction is blind to the penalty-tied base)"}
