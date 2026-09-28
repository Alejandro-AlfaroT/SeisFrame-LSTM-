"""Developed beam-plus-slab flexural strengths for capacity design and hinges.

ACI 318-19 18.7.3.2 counts slab reinforcement within the 6.3.2 effective
flange width toward the beam flexural strength used for strong-column /
weak-beam. The same strength has to be what the nonlinear beam hinge yields
at, otherwise the modeled beams are weaker relative to their columns than the
design assumed. This module is the single source for both.

Strain compatibility (22.2): ecu = 0.003 at the compression face, Whitney
block with beta1 from 22.2.2.4.3, elastic-perfectly-plastic steel, every bar
layer in the composite section (beam top and bottom bars, slab top and bottom
mats within the flange) as a discrete layer, zero axial force. Displaced
concrete at steel layers is ignored (slightly conservative for compression
steel). Negative (hogging) bending puts the web in compression; positive
(sagging) bending puts the flange in compression, with the block allowed to
pass into the web.

Effective flange width follows Table 6.3.2.1: interior beams add
min(8h, sw/2, ln/8) each side; perimeter beams add min(6h, sw/2, ln/12) on
their one slab side. h is the slab thickness, sw the clear distance to the
adjacent web and ln the clear span. Slab mats are continuous and uniform, so
the bars inside the flange are developed at every beam end.
"""
from __future__ import annotations

import math

_BAR_AREA = {3: 0.11, 4: 0.20, 5: 0.31, 6: 0.44, 7: 0.60, 8: 0.79, 9: 1.00, 10: 1.27, 11: 1.56}
ES_KSI = 29000.0


def beta1(fc_ksi):
    return max(0.65, min(0.85, 0.85 - 0.05 * (fc_ksi - 4.0)))


def effective_flange_width(bw, slab_h, clear_span, clear_to_adjacent_web, slab_sides):
    """ACI 318-19 Table 6.3.2.1 total flange width for 1 or 2 slab sides.

    The same flange sets the frame's beam flexural stiffness
    (Structure_Parameters.beam_flexural_section), so the section analysed
    is the section checked; one implementation serves both.
    """
    import Structure_Parameters as sp
    return sp.effective_flange_width_in(bw, slab_h, clear_span, clear_to_adjacent_web, slab_sides)


def section_moment(layers, fc_ksi, fy_ksi, depth, web_width, flange_width=None, flange_depth=0.0):
    """Nominal moment (kip-in) of a section with steel ``layers`` and no axial load.

    ``layers``: iterable of (depth_from_compression_face_in, area_in2). The
    compression block occupies ``flange_width`` down to ``flange_depth`` and
    ``web_width`` below it; pass flange_width=None for a rectangular section.
    """
    layers = [(float(y), float(a)) for y, a in layers if a > 0]
    if not layers or depth <= 0:
        raise ValueError("A section needs steel layers and positive depth.")
    b1 = beta1(fc_ksi)
    bf = web_width if flange_width is None else flange_width

    def block(c):
        a = min(b1 * c, depth)
        if flange_width is None or a <= flange_depth:
            force = 0.85 * fc_ksi * bf * a
            return force, a / 2.0, a
        flange = 0.85 * fc_ksi * bf * flange_depth
        web = 0.85 * fc_ksi * web_width * (a - flange_depth)
        centroid = (flange * flange_depth / 2.0 + web * (flange_depth + (a - flange_depth) / 2.0)) / (flange + web)
        return flange + web, centroid, a

    def steel(c):
        forces = []
        for y, area in layers:
            strain = 0.003 * (c - y) / c
            stress = max(-fy_ksi, min(fy_ksi, ES_KSI * strain))   # compression positive
            forces.append((y, area * stress))
        return forces

    def unbalance(c):
        force, _, _ = block(c)
        return force + sum(f for _, f in steel(c))

    low, high = 1e-6, depth
    if unbalance(low) > 0 or unbalance(high) < 0:
        raise ValueError("No neutral axis satisfies equilibrium; check the steel layers.")
    for _ in range(100):
        mid = 0.5 * (low + high)
        if unbalance(mid) > 0:
            high = mid
        else:
            low = mid
    c = 0.5 * (low + high)
    force, centroid, a = block(c)
    forces = steel(c)
    moment = -(force * centroid + sum(y * f for y, f in forces))
    extreme = max(y for y, _ in layers)
    tension_strain = 0.003 * (extreme - c) / c
    return {"mn_kip_in": moment, "neutral_axis_in": c, "stress_block_in": a,
            "tension_strain": tension_strain, "tension_controlled": tension_strain >= fy_ksi / ES_KSI + 0.003,
            "beta1": b1, "steel_forces_kip": [(y, f) for y, f in forces]}


def validate_bar_rows(face_rows, count, name="beam bars", depth_in=None):
    """The actual bar rows of one face as [(bars, elevation_from_that_face_in), ...], checked.

    ``face_rows`` is one face of Structure_Parameters.beam_bar_layers (per_layer, offsets_in and,
    when present, layers and centroid_in). Rejected (2026-09-27, Codex item 3): per_layer and
    offsets_in of different lengths (a zip would silently drop bars), a declared layer count that
    does not match, counts that do not add up to ``count``, non-positive or non-increasing
    elevations, an elevation outside the section, or a declared centroid the rows do not give.
    """
    try:
        per = list(face_rows["per_layer"])
        offs = [float(o) for o in face_rows["offsets_in"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name}: rows need per_layer and offsets_in ({exc})") from exc
    if not per or len(per) != len(offs):
        raise ValueError(f"{name}: per_layer has {len(per)} entries and offsets_in {len(offs)}; they must match")
    if any(type(n) is not int or n < 1 for n in per):
        raise ValueError(f"{name}: every layer must hold a positive integer number of bars")
    if "layers" in face_rows and int(face_rows["layers"]) != len(per):
        raise ValueError(f"{name}: declared {face_rows['layers']} layers but {len(per)} rows")
    if sum(per) != int(count):
        raise ValueError(f"{name}: rows add up to {sum(per)} bars, not the {count} of the face")
    if any(not math.isfinite(o) or o <= 0.0 for o in offs) or any(b <= a for a, b in zip(offs, offs[1:])):
        raise ValueError(f"{name}: elevations must be positive and strictly increasing from the face")
    if depth_in is not None and offs[-1] >= float(depth_in) / 2.0:
        raise ValueError(f"{name}: an elevation of {offs[-1]:g} in reaches past mid-depth of a {depth_in:g} in section")
    if "centroid_in" in face_rows:
        centroid = sum(n * o for n, o in zip(per, offs)) / sum(per)
        if not math.isclose(centroid, float(face_rows["centroid_in"]), abs_tol=1e-6):
            raise ValueError(f"{name}: declared centroid {face_rows['centroid_in']} does not follow from the rows ({centroid:.4f})")
    return list(zip(per, offs))


def _slab_layers_from_top(slab_h, layout, axis, flange_width):
    """Slab mats parallel to the beam inside the flange: (depth from slab top, area)."""
    result = []
    for face in ("top", "bottom"):
        layer = layout["layers"][f"{axis}_{face}"]
        area = layer["bar_area_in2"] * flange_width / layer["spacing_in"]
        y_from_top = (slab_h - layer["effective_depth_in"]) if face == "top" else layer["effective_depth_in"]
        result.append((y_from_top, area, face))
    return result


def composite_beam_strengths(beam, slab, layout, geometry, axis, position):
    """Rectangular and beam-plus-slab Mn for one beam family.

    beam: b_in, h_in, fc_ksi, fy_ksi, bar_size, top_bars, bot_bars, centroid_offset_in and,
    when the bars sit in more than one row, ``layers`` = {"top": rows, "bottom": rows} (one
    direction of Structure_Parameters.beam_bar_layers): every row is then its own steel layer
    (Codex item 1, 2026-09-27), so a second row of the compression face that lies below the
    neutral axis is counted in tension where it belongs. Without ``layers`` each face is one
    row at its centroid offset (a number, or {"top", "bottom"} for the stacked orthogonal cages).
    slab: thickness_in. layout: the slab reinforcement layout record.
    geometry: bay_x_in, bay_y_in, h_col_in, b_col_in. axis 'x'/'y';
    position 'edge' or 'interior'.
    """
    bw, h, fc, fy = beam["b_in"], beam["h_in"], beam["fc_ksi"], beam["fy_ksi"]
    ab = _BAR_AREA[beam["bar_size"]]
    rows = beam.get("layers")
    if rows:
        top_rows = validate_bar_rows(rows["top"], beam["top_bars"], "top bars", h)
        bot_rows = validate_bar_rows(rows["bottom"], beam["bot_bars"], "bottom bars", h)
    else:
        # One offset for both faces, or {"top": .., "bottom": ..} when the orthogonal cages stack at
        # the joints (Structure_Parameters.beam_bar_stacking_offsets_in).
        offset = beam["centroid_offset_in"]
        if isinstance(offset, dict):
            off_top, off_bot = float(offset["top"]), float(offset["bottom"])
        else:
            off_top = off_bot = float(offset)
        top_rows, bot_rows = [(beam["top_bars"], off_top)], [(beam["bot_bars"], off_bot)]
    off_top = sum(n * o for n, o in top_rows) / max(1, sum(n for n, _ in top_rows))
    off_bot = sum(n * o for n, o in bot_rows) / max(1, sum(n for n, _ in bot_rows))

    def web_layers(sign):
        """(depth from the compression face, area) of every beam bar row for one sign of bending."""
        near, far = (bot_rows, top_rows) if sign == "negative" else (top_rows, bot_rows)
        return [(o, n * ab) for n, o in near] + [(h - o, n * ab) for n, o in far]

    t = slab["thickness_in"]
    if axis == "x":
        clear_span, clear_web = geometry["bay_x_in"] - geometry["h_col_in"], geometry["bay_y_in"] - bw
    else:
        clear_span, clear_web = geometry["bay_y_in"] - geometry["b_col_in"], geometry["bay_x_in"] - bw
    sides = 2 if position == "interior" else 1
    bf, overhang = effective_flange_width(bw, t, clear_span, clear_web, sides)
    top_area, bot_area = beam["top_bars"] * ab, beam["bot_bars"] * ab
    slab_layers = _slab_layers_from_top(t, layout, axis, bf) if layout is not None else []
    slab_area = sum(area for _, area, _ in slab_layers)
    # Rectangular beam only. Layers are (depth from the compression face, area), one per bar row.
    rect_neg = section_moment(web_layers("negative"), fc, fy, h, bw)
    rect_pos = section_moment(web_layers("positive"), fc, fy, h, bw)
    # Composite. Negative: compression face is the beam bottom.
    neg_layers = web_layers("negative") + [(h - y, a) for y, a, _ in slab_layers]
    comp_neg = section_moment(neg_layers, fc, fy, h, bw)
    pos_layers = web_layers("positive") + [(y, a) for y, a, _ in slab_layers]
    comp_pos = section_moment(pos_layers, fc, fy, h, bw, flange_width=bf, flange_depth=t)
    # Where the slab bars are not developed (an exterior end whose hook does
    # not fit the perimeter beam) neither mat is counted in either sign: the
    # bottom mat would otherwise add tension steel to sagging. The flange
    # concrete stays in compression under sagging; it needs no development.
    undeveloped_pos = section_moment(web_layers("positive"), fc, fy, h, bw, flange_width=bf, flange_depth=t)
    return {
        "axis": axis, "position": position, "slab_sides": sides,
        "centroid_offsets_in": {"top": off_top, "bottom": off_bot},
        "bar_rows": {"top": [{"bars": n, "elevation_from_face_in": o} for n, o in top_rows],
                     "bottom": [{"bars": n, "elevation_from_face_in": o} for n, o in bot_rows]},
        "steel_area_in2": {"top": top_area, "bottom": bot_area},
        "effective_flange_width_in": bf, "flange_overhang_in": overhang,
        "clear_span_in": clear_span, "clear_to_adjacent_web_in": clear_web,
        "slab_steel_in_flange_in2": slab_area,
        "slab_layers_from_top": [{"face": face, "depth_in": y, "area_in2": a} for y, a, face in slab_layers],
        "rectangular": {"positive": rect_pos, "negative": rect_neg},
        "composite": {"positive": comp_pos, "negative": comp_neg},
        "undeveloped": {"positive": undeveloped_pos, "negative": rect_neg},
        "mn_positive_kip_in": comp_pos["mn_kip_in"], "mn_negative_kip_in": comp_neg["mn_kip_in"],
        "mn_undeveloped_positive_kip_in": undeveloped_pos["mn_kip_in"],
        "mn_undeveloped_negative_kip_in": rect_neg["mn_kip_in"],
        "slab_increment_positive_kip_in": comp_pos["mn_kip_in"] - rect_pos["mn_kip_in"],
        "slab_increment_negative_kip_in": comp_neg["mn_kip_in"] - rect_neg["mn_kip_in"],
        "flange_concrete_increment_positive_kip_in": undeveloped_pos["mn_kip_in"] - rect_pos["mn_kip_in"],
        "basis": ("ACI 318-19 6.3.2.1 effective flange; 18.7.3.2 slab mats within it as discrete layers; "
                  "22.2 strain compatibility, ecu = 0.003, Whitney block, EPP steel, zero axial force; "
                  "continuous uniform slab mats are developed at interior beam ends by continuity (25.4.2.4 "
                  "within half the adjacent clear span); exterior ends are credited per the perimeter hook check"),
    }


_BAR_DIAMETER = {3: 0.375, 4: 0.5, 5: 0.625, 6: 0.75, 7: 0.875, 8: 1.0, 9: 1.128, 10: 1.27, 11: 1.41}


def hook_development_length_in(bar_size, fc_ksi, fy_ksi, spacing_in):
    """ACI 318-19 25.4.3.1 standard-hook development for an uncoated, normalweight bar.

    ldh = fy psi_e psi_r psi_o psi_c / (55 lambda sqrt(f'c)) * db^1.5, at least
    8 db and 6 in. psi_e = 1 (uncoated); psi_r = 1.0 where the hooked bars are
    spaced at least 6 db, else 1.6 (Table 25.4.3.2, without confining ties);
    psi_o = 1.0 (the hook terminates inside a beam that continues along the
    perimeter, so side cover normal to the hook plane is at least 6 db);
    psi_c = f'c/15000 + 0.6 below 6 ksi, 1.0 above; lambda = 1.
    """
    db = _BAR_DIAMETER[bar_size]
    fc_psi = fc_ksi * 1000.0
    psi_r = 1.0 if spacing_in >= 6.0 * db else 1.6
    psi_c = fc_psi / 15000.0 + 0.6 if fc_psi < 6000.0 else 1.0
    ldh = fy_ksi * 1000.0 * 1.0 * psi_r * 1.0 * psi_c / (55.0 * math.sqrt(fc_psi)) * db ** 1.5
    return max(ldh, 8.0 * db, 6.0), {"psi_e": 1.0, "psi_r": psi_r, "psi_o": 1.0, "psi_c": psi_c, "db_in": db}


def perimeter_slab_bar_anchorage(layout, axis, perimeter_beam_width_in, beam_clear_cover_in, hoop_db_in,
                                 fc_beam_ksi, fy_ksi):
    """Can the slab mats parallel to ``axis`` be developed where the slab ends at the perimeter?

    At a beam's exterior end the slab bars in its flange run to the building
    edge, where the perimeter beam (perpendicular to them) is the only
    concrete beyond the critical section. They are developed there by a
    standard hook into that beam: ldh (25.4.3.1) against the embedment from
    the beam's near face to the outside of the hook, the beam width less the
    far-side clear cover and hoop. Both mats of the axis are checked (both
    are in tension under hogging); the worse governs. Returns the record the
    strength entries carry.
    """
    if layout is None:
        return None
    layers = {face: layout["layers"][f"{axis}_{face}"] for face in ("top", "bottom")}
    worst, detail, bar = 0.0, None, None
    for face, layer in layers.items():
        bar = layout.get("bar_size") or layer.get("bar_size") or next(
            size for size, area in _BAR_AREA.items() if abs(area - layer["bar_area_in2"]) < 1e-9)
        ldh, factors = hook_development_length_in(bar, fc_beam_ksi, fy_ksi, layer["spacing_in"])
        if ldh > worst:
            worst, detail = ldh, {"face": face, "spacing_in": layer["spacing_in"], **factors}
    available = perimeter_beam_width_in - beam_clear_cover_in - hoop_db_in
    return {"axis": axis, "bar_size": bar, "ldh_required_in": worst, "embedment_available_in": available,
            "developed": worst <= available, "governing": detail,
            "hook": "standard 90-degree hook into the perimeter beam, ACI 318-19 25.4.3.1 (not the 18.8.5.1 joint formula)",
            "basis": ("slab mats parallel to the beam terminate at the building edge; the perimeter beam is the "
                      "only embedment beyond the exterior critical section")}


def beam_family(kind, line_index, num_bay_x, num_bay_y):
    axis = kind[-1]
    last = num_bay_y if axis == "x" else num_bay_x
    return axis, ("edge" if line_index in (0, last) else "interior")


def beam_members(num_bay_x, num_bay_y, num_floor):
    """(tag, kind, line_index) for every beam in the builders' creation order."""
    return [(tag, kind, line) for tag, kind, line, _span in beam_members_with_spans(num_bay_x, num_bay_y, num_floor)]


def beam_members_with_spans(num_bay_x, num_bay_y, num_floor):
    """(tag, kind, line_index, span_index) for every beam in the builders' creation order."""
    tag = num_floor * (num_bay_x + 1) * (num_bay_y + 1)
    members = []
    for _k in range(num_floor):
        for j in range(num_bay_y + 1):
            for i in range(num_bay_x):
                tag += 1
                members.append((tag, "beam_x", j, i))
    for _k in range(num_floor):
        for j in range(num_bay_y):
            for i in range(num_bay_x + 1):
                tag += 1
                members.append((tag, "beam_y", i, j))
    return members


def exterior_ends(kind, span_index, num_bay_x, num_bay_y):
    """Which ends of a beam terminate at the building perimeter in the beam's own direction."""
    last = (num_bay_x if kind == "beam_x" else num_bay_y) - 1
    return {"i": span_index == 0, "j": span_index == last}


def beam_slab_strengths(record):
    """Joint-adapter entries and per-family strengths from a design record.

    Requires record['slab_reinforcement']['layout']; returns
    (entries keyed '<tag>/<end>/<positive|negative>', families keyed
    '<axis>_<position>'). Without a layout the entries declare no_slab.
    """
    sections, rebar, geometry = record["sections"], record["reinforcement"], record["geometry"]
    layout = (record.get("slab_reinforcement") or {}).get("layout")
    beam = {"b_in": sections["b_beam_in"], "h_in": sections["h_beam_in"], "fc_ksi": sections["fc_beam_ksi"],
            "fy_ksi": record["materials"]["fy_ksi"], "bar_size": rebar["beam_bar_size"],
            "top_bars": rebar["beam_top_bars"], "bot_bars": rebar["beam_bot_bars"],
            "centroid_offset_in": rebar["beam_longitudinal_centroid_offset_in"]}
    slab = {"thickness_in": record["slab"]["thickness_in"]}
    geom = {"bay_x_in": geometry["bay_x_in"], "bay_y_in": geometry["bay_y_in"],
            "h_col_in": sections["h_col_in"], "b_col_in": sections["b_col_in"]}
    # Stacked orthogonal layers (reinforcement.beam_bar_stacking) give each direction its own
    # top and bottom offsets and, since 2026-09-27, its actual rows; older records carry the
    # single nominal offset.
    stacking_record = rebar.get("beam_bar_stacking") or {}
    stacking = stacking_record.get("offsets_in") or {}
    rows_by_axis = stacking_record.get("layers") or {}
    families = {}
    for axis in ("x", "y"):
        beam_axis = {**beam, "centroid_offset_in": stacking.get(axis, beam["centroid_offset_in"]),
                     "layers": rows_by_axis.get(axis)}
        for position in ("edge", "interior"):
            families[f"{axis}_{position}"] = composite_beam_strengths(beam_axis, slab, layout, geom, axis, position)
    # Where the slab ends at the perimeter its bars are credited only if the
    # hook into the perimeter beam develops them (ACI 318-19 25.4.3.1).
    anchorage = {axis: perimeter_slab_bar_anchorage(layout, axis, sections["b_beam_in"], rebar["beam_clear_cover_in"],
                                                    rebar["beam_stirrup_diameter_in"], sections["fc_beam_ksi"],
                                                    record["materials"]["fy_ksi"])
                 for axis in ("x", "y")}
    for axis in ("x", "y"):
        for position in ("edge", "interior"):
            families[f"{axis}_{position}"]["exterior_anchorage"] = anchorage[axis]
    entries = {}
    for tag, kind, line, span in beam_members_with_spans(geometry["num_bay_x"], geometry["num_bay_y"], geometry["num_floor"]):
        axis, position = beam_family(kind, line, geometry["num_bay_x"], geometry["num_bay_y"])
        family = families[f"{axis}_{position}"]
        exterior = exterior_ends(kind, span, geometry["num_bay_x"], geometry["num_bay_y"])
        for end in ("i", "j"):
            for sign in ("positive", "negative"):
                if layout is None:
                    # A slab exists but its reinforcement is not established:
                    # that is unknown slab strength, never "no slab". Leaving
                    # the compatibility flag unset keeps the joint SCWB checks
                    # not_evaluated instead of passing on zero slab strength.
                    entries[f"{tag}/{end}/{sign}"] = {"slab_basis": "unknown", "slab_mn_kip_in": None,
                                                      "section_compatibility_verified": False,
                                                      "reason": "slab reinforcement not established (slab action "
                                                                "evidence not asserted as verified)"}
                    continue
                terminated = exterior[end] and not anchorage[axis]["developed"]
                if terminated:
                    # An exterior end whose slab bars cannot be hooked into the
                    # perimeter beam: the mats are not developed at this critical
                    # section and are counted in neither sign. Hogging is the
                    # rectangular beam; sagging keeps the flange concrete in
                    # compression (no bar development involved) but not the
                    # bottom mat, which would otherwise add tension steel.
                    strength = family["undeveloped"][sign]["mn_kip_in"]
                    rectangular = family["rectangular"][sign]["mn_kip_in"]
                    entries[f"{tag}/{end}/{sign}"] = {
                        "slab_basis": "terminated_undeveloped" if sign == "negative" else "flange_concrete_undeveloped_bars",
                        "slab_mn_kip_in": 0.0 if sign == "negative" else max(0.0, strength - rectangular),
                        "section_compatibility_verified": True,
                        "family": f"{axis}_{position}",
                        "mn_composite_kip_in": strength,
                        "mn_rectangular_kip_in": rectangular,
                        "effective_flange_width_in": family["effective_flange_width_in"],
                        "slab_steel_in_flange_in2": 0.0,
                        "exterior_end": True, "exterior_anchorage": anchorage[axis],
                        "basis": ("rectangular beam at an exterior end: slab bars terminate undeveloped at the perimeter"
                                  if sign == "negative" else
                                  "beam with the flange concrete in compression at an exterior end: slab bars terminate "
                                  "undeveloped at the perimeter and are not counted in either mat"),
                    }
                    continue
                entries[f"{tag}/{end}/{sign}"] = {
                    "slab_basis": "developed_effective_width",
                    "slab_mn_kip_in": max(0.0, family[f"slab_increment_{sign}_kip_in"]),
                    "section_compatibility_verified": True,
                    "family": f"{axis}_{position}",
                    "mn_composite_kip_in": family[f"mn_{sign}_kip_in"],
                    "mn_rectangular_kip_in": family["rectangular"][sign]["mn_kip_in"],
                    "effective_flange_width_in": family["effective_flange_width_in"],
                    "slab_steel_in_flange_in2": family["slab_steel_in_flange_in2"],
                    "exterior_end": exterior[end],
                    "exterior_anchorage": anchorage[axis] if exterior[end] else None,
                    "basis": family["basis"] + (" ; exterior end: slab bars hooked into the perimeter beam, "
                                                "ldh per 25.4.3.1 fits" if exterior[end] else ""),
                }
    return entries, families


def governing_beam_nominal_moment(families):
    """Largest composite Mn over families and signs, for the column SCWB screen."""
    return max(f[f"mn_{sign}_kip_in"] for f in families.values() for sign in ("positive", "negative"))
