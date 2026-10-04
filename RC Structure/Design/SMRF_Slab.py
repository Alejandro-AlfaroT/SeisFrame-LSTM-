"""One thickness per research building: a bounded two-way slab thickness screen.

Basis: ACI 318-19 Table 8.3.1.2, 8.3.1.2.1, and 8.4.1.8. This is NOT a
slab strength, reinforcement, shear, or full serviceability design. Concrete is
normalweight; slab and downstand beams are monolithic, with the same concrete;
all rectangular panels have beams on all four sides. No openings, cantilevers,
drops, prestressing, or variation between floors is represented.

Source (original ACI text, pp. 101-104):
https://www.ocf.berkeley.edu/~chiep/wp-content/uploads/2024/01/CE-123-ACI-318-19.pdf
"""
from __future__ import annotations

import math

from . import SMRF_Floor_Sections as floor_sections
from .SMRF_Common import make_check, not_evaluated


METHOD_VERSION = "aci318_19_uniform_beam_supported_slab_screen_v1"


class SlabSizingError(ValueError):
    """Supported request exhausted its thickness ladder; a deeper beam may help."""


_DEFAULT_POLICY = {
    "minimum_thickness_in": 5.0,
    "maximum_thickness_in": 14.0,
    "thickness_increment_in": 0.5,
    "superimposed_dead_load_ksf": 0.05,
    "live_load_mass_fraction": 0.0,
    "fy_ksi": 60.0,
    "concrete_unit_weight_kcf": 0.15,
    "slab_system": "monolithic_beam_supported_two_way",
    "concrete_type": "normalweight",
    "prestressed": False,
}


def _number(value, name, *, allow_zero=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not a boolean.")
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}.")
    return value


def _inputs(geometry, sections, policy):
    if not all(isinstance(item, dict) for item in (geometry, sections, policy)):
        raise ValueError("Slab geometry, sections, and policy must be dictionaries.")
    g, s = {}, {}
    try:
        for key in ("num_bay_x", "num_bay_y", "num_floor"):
            value = _number(geometry[key], key)
            if not value.is_integer():
                raise ValueError(f"{key} must be a positive integer.")
            g[key] = int(value)
        for key in ("bay_x_in", "bay_y_in", "story_h_in"):
            g[key] = _number(geometry[key], key)
        if floor_sections.is_by_floor(sections):
            # A grouped design: every mechanically distinct floor with its own beam lines.
            s = floor_sections.validate_by_floor(sections, g["num_bay_x"], g["num_bay_y"], g["num_floor"])
        else:
            for key in ("b_beam_in", "h_beam_in", "fc_beam_ksi"):
                s[key] = _number(sections[key], key)
    except KeyError as exc:
        raise ValueError(f"Missing slab input: {exc.args[0]}.") from exc
    p = dict(_DEFAULT_POLICY)
    unknown = set(policy) - set(p)
    if unknown:
        raise ValueError(f"Unsupported slab policy fields: {sorted(unknown)}.")
    p.update(policy)
    if (p["slab_system"] != "monolithic_beam_supported_two_way"
            or p["concrete_type"] != "normalweight" or p["prestressed"] is not False):
        raise ValueError("Only monolithic, nonprestressed, normalweight beam-supported two-way slabs are implemented.")
    for key in ("minimum_thickness_in", "maximum_thickness_in", "thickness_increment_in",
                "fy_ksi", "concrete_unit_weight_kcf"):
        p[key] = _number(p[key], key)
    for key in ("superimposed_dead_load_ksf", "live_load_mass_fraction"):
        p[key] = _number(p[key], key, allow_zero=True)
    if p["fy_ksi"] != 60.0:
        raise ValueError("The slab screen currently supports Grade 60 reinforcement only.")
    if p["live_load_mass_fraction"] > 1:
        raise ValueError("live_load_mass_fraction must be between zero and one.")
    if p["maximum_thickness_in"] < p["minimum_thickness_in"]:
        raise ValueError("maximum_thickness_in must be at least minimum_thickness_in.")
    if floor_sections.is_by_floor(s):
        grades = set()
        for _sha, floor, _floors in floor_sections.distinct_floors(s):
            grades.add(floor["fc_beam_ksi"])
            beams = floor["beam_lines"]["x"] + floor["beam_lines"]["y"]
            if max(beam["b_in"] for beam in beams) >= min(g["bay_x_in"], g["bay_y_in"]):
                raise ValueError("Slab clear spans must be positive after subtracting beam width.")
            if max(beam["h_in"] for beam in beams) >= g["story_h_in"]:
                raise ValueError("The downstand beam must fit within the story height.")
            for i in range(g["num_bay_x"]):
                for j in range(g["num_bay_y"]):
                    clear_x, clear_y = _panel_clear_spans(g, floor, i, j)
                    if max(clear_x, clear_y) / min(clear_x, clear_y) > 2.0:
                        raise ValueError("Clear-span aspect ratio exceeds 2; one-way slab design is not implemented (ACI R8.3.1.2).")
        if len(grades) != 1:
            raise ValueError("The slab is cast with the beams: every floor must use the one beam concrete grade.")
        return g, s, p
    if s["b_beam_in"] >= min(g["bay_x_in"], g["bay_y_in"]):
        raise ValueError("Slab clear spans must be positive after subtracting beam width.")
    if s["h_beam_in"] >= g["story_h_in"]:
        raise ValueError("The downstand beam must fit within the story height.")
    beta = max(g["bay_x_in"], g["bay_y_in"]) - s["b_beam_in"]
    beta /= min(g["bay_x_in"], g["bay_y_in"]) - s["b_beam_in"]
    if beta > 2.0:
        raise ValueError("Clear-span aspect ratio exceeds 2; one-way slab design is not implemented (ACI R8.3.1.2).")
    return g, s, p


def _beam_inertia(width, depth, slab_h, flange_count, transverse_bay):
    """Gross centroidal T/L-section I about the horizontal bending axis.

    The web below the slab and the complete top flange do not overlap. A slab
    flange on an interior beam occurs on each side; perimeter beams have one.
    Capping each projection at half the clear transverse bay prevents overlapping
    effective flanges in unusually narrow bays (a conservative extra bound).
    """
    projection = min(depth - slab_h, 4.0 * slab_h, 0.5 * (transverse_bay - width))
    flange_width = width + flange_count * projection
    web_h = depth - slab_h
    flange_area, web_area = flange_width * slab_h, width * web_h
    flange_y, web_y = slab_h / 2.0, slab_h + web_h / 2.0
    centroid = (flange_area * flange_y + web_area * web_y) / (flange_area + web_area)
    inertia = (flange_width * slab_h ** 3 / 12.0 + flange_area * (flange_y - centroid) ** 2
               + width * web_h ** 3 / 12.0 + web_area * (web_y - centroid) ** 2)
    return inertia, projection, flange_width


def _required_thickness(long_clear, beta, alpha_fm, weak_discontinuous_edge, fy_ksi=60.0):
    """Return equation and floor separately so the edge factor applies correctly."""
    if alpha_fm <= 0.2:
        return None
    numerator = long_clear * (0.8 + fy_ksi * 1000.0 / 200000.0)
    if alpha_fm <= 2.0:
        expression = numerator / (36.0 + 5.0 * beta * (alpha_fm - 0.2))
        floor, branch = 5.0, "0.2 < alpha_fm <= 2.0"
    else:
        expression = numerator / (36.0 + 9.0 * beta)
        floor, branch = 3.5, "alpha_fm > 2.0"
    edge_factor = 1.1 if weak_discontinuous_edge else 1.0
    return {"equation_thickness_in": expression, "absolute_minimum_in": floor,
            "discontinuous_edge_factor": edge_factor,
            "required_thickness_in": max(floor, expression * edge_factor),
            "branch": branch}


def _panel_clear_spans(geometry, floor, i, j):
    """Clear spans of panel (i, j) between the faces of its own four beams."""
    widths = floor_sections.panel_face_widths(floor, i, j)
    return (geometry["bay_x_in"] - 0.5 * sum(widths["x"]), geometry["bay_y_in"] - 0.5 * sum(widths["y"]))


def _panels_by_line(geometry, floor, policy, slab_h):
    """The panels of one floor described line by line: every edge on its own beam, every panel on its own clear spans."""
    g = geometry
    results = []
    for i in range(g["num_bay_x"]):
        for j in range(g["num_bay_y"]):
            clear_x, clear_y = _panel_clear_spans(g, floor, i, j)
            long_clear, short_clear = max(clear_x, clear_y), min(clear_x, clear_y)
            edges = []
            definitions = (("x_min", "y", i, i == 0, g["bay_x_in"]),
                           ("x_max", "y", i + 1, i == g["num_bay_x"] - 1, g["bay_x_in"]),
                           ("y_min", "x", j, j == 0, g["bay_y_in"]),
                           ("y_max", "x", j + 1, j == g["num_bay_y"] - 1, g["bay_y_in"]))
            for side, beam_axis, line, discontinuous, transverse_bay in definitions:
                beam = floor_sections.beam_line(floor, beam_axis, line)
                n_flange = 1 if discontinuous else 2
                ib, projection, flange_width = _beam_inertia(beam["b_in"], beam["h_in"], slab_h, n_flange, transverse_bay)
                slab_i = transverse_bay * slab_h ** 3 / 12.0
                edges.append({"side": side, "beam_axis": beam_axis, "beam_line_index": line,
                              "beam_width_in": beam["b_in"], "beam_depth_in": beam["h_in"],
                              "discontinuous": discontinuous, "flange_count": n_flange,
                              "flange_projection_in": projection, "flange_width_in": flange_width,
                              "beam_inertia_in4": ib, "slab_strip_width_in": transverse_bay,
                              "slab_strip_inertia_in4": slab_i, "alpha_f": ib / slab_i})
            alpha_fm = sum(edge["alpha_f"] for edge in edges) / 4.0
            weak_edges = [edge["side"] for edge in edges if edge["discontinuous"] and edge["alpha_f"] < 0.8]
            minimum = _required_thickness(long_clear, long_clear / short_clear, alpha_fm,
                                          bool(weak_edges), policy["fy_ksi"])
            count = sum(edge["discontinuous"] for edge in edges)
            panel = {"panel_id": f"panel_x{i + 1}_y{j + 1}", "bay_x_index": i + 1,
                     "bay_y_index": j + 1,
                     "panel_type": "interior" if count == 0 else "edge" if count == 1 else "corner_or_multi_edge",
                     "discontinuous_edge_count": count, "clear_span_x_in": clear_x,
                     "clear_span_y_in": clear_y, "long_clear_span_in": long_clear,
                     "beta": long_clear / short_clear, "alpha_fm": alpha_fm,
                     "weak_discontinuous_edges": weak_edges, "edges": edges,
                     "thickness_in": slab_h}
            if minimum is None:
                panel.update({"status": "not_evaluated", "required_thickness_in": None,
                              "reason": "alpha_fm <= 0.2 invokes ACI 8.3.1.1; weak-beam/flat-slab scope is not implemented."})
            else:
                panel.update(minimum)
                panel["status"] = "pass" if slab_h >= minimum["required_thickness_in"] else "fail"
            results.append(panel)
    return results


def _choose_slab_by_floor(g, s, p):
    """One thickness for the whole building, passing every panel of every mechanically distinct floor."""
    lower, upper, increment = (p[key] for key in ("minimum_thickness_in", "maximum_thickness_in", "thickness_increment_in"))
    steps = int(math.floor((upper - lower) / increment + 1e-10)) + 1
    if steps > 10000:
        raise ValueError("Slab thickness ladder exceeds 10,000 trials; increase the thickness increment.")
    distinct = floor_sections.distinct_floors(s)
    shallowest = min(beam["h_in"] for _sha, floor, _floors in distinct
                     for beam in floor["beam_lines"]["x"] + floor["beam_lines"]["y"])
    trials = []
    for step in range(steps):
        slab_h = lower + step * increment
        if slab_h >= min(shallowest, g["story_h_in"]):
            trials.append({"thickness_in": slab_h, "status": "not_evaluated",
                           "reason": "Slab must be thinner than every downstand beam and the story height."})
            continue
        by_floor = [{"floor_sections_sha256": sha, "floors": floors, "panels": _panels_by_line(g, floor, p, slab_h)}
                    for sha, floor, floors in distinct]
        rows = [(entry, panel) for entry in by_floor for panel in entry["panels"]]
        supported = [(entry, panel) for entry, panel in rows if panel["required_thickness_in"] is not None]
        governing = max(supported, key=lambda item: item[1]["required_thickness_in"]) if supported else None
        passed = all(panel["status"] == "pass" for _entry, panel in rows)
        trials.append({"thickness_in": slab_h,
                       "status": "pass" if passed else "not_evaluated" if len(supported) < len(rows) else "fail",
                       "required_thickness_in": governing[1]["required_thickness_in"] if governing else None,
                       "governing_panel_id": governing[1]["panel_id"] if governing else None,
                       "governing_floors": governing[0]["floors"] if governing else None,
                       "unsupported_panel_count": len(rows) - len(supported),
                       "alpha_fm_min": min(panel["alpha_fm"] for _entry, panel in rows),
                       "alpha_fm_max": max(panel["alpha_fm"] for _entry, panel in rows)})
        if passed:
            weight = p["concrete_unit_weight_kcf"] * slab_h / 12.0
            per_floor = len(by_floor[0]["panels"])
            return {"method_version": METHOD_VERSION, "stage": "thickness_screen_only",
                    "thickness_screen_passed": True, "thickness_in": slab_h,
                    "uniform_thickness_all_floors_including_roof": True,
                    "concrete_fc_ksi": distinct[0][1]["fc_beam_ksi"], "concrete_type": p["concrete_type"],
                    "concrete_unit_weight_kcf": p["concrete_unit_weight_kcf"],
                    "superimposed_dead_load_ksf": p["superimposed_dead_load_ksf"],
                    "self_weight_ksf": weight, "total_dead_load_ksf": weight + p["superimposed_dead_load_ksf"],
                    "live_load_mass_fraction": p["live_load_mass_fraction"],
                    "inputs": {"geometry": g, "sections": s, "policy": p},
                    "panel_count_per_floor": per_floor, "panel_count_building": per_floor * g["num_floor"],
                    "governing_panel_id": governing[1]["panel_id"], "governing_floors": governing[0]["floors"],
                    "governing_floor_sections_sha256": governing[0]["floor_sections_sha256"],
                    "required_thickness_in": governing[1]["required_thickness_in"],
                    # The governing floor's panels, in the place the one-floor record keeps its panels; every floor's
                    # panels are under floor_panels.
                    "panels": governing[0]["panels"], "floor_panels": by_floor, "distinct_floor_count": len(by_floor),
                    "trial_history": trials,
                    "stiffness_basis": "Gross monolithic T/L beam per ACI 8.4.1.8 on every beam line's own web; same Ec for slab/beam; "
                                       "full transverse bay slab strip even at perimeter (conservative); one thickness passing every "
                                       "panel of every mechanically distinct floor.",
                    "scope_limitations": ["Thickness screen only; no slab flexure, shear, reinforcement, or explicit long-term deflection analysis.",
                                          "No openings, cantilevers, drops, or prestressing; one slab thickness on every floor.",
                                          "Slab stiffness here sizes thickness only; it does not itself change frame-analysis beam stiffness."]}
    unsupported = sum(trial["status"] == "not_evaluated" for trial in trials)
    raise SlabSizingError(f"No slab thickness passes the supported screen on every floor in {lower:g}-{upper:g} in at {increment:g} in "
                          f"increments ({unsupported}/{len(trials)} unsupported trials). Revisit beam/slab geometry or design "
                          "policy; no fallback thickness was assigned.")


def _panels(geometry, sections, policy, slab_h):
    g, s = geometry, sections
    width, depth = s["b_beam_in"], s["h_beam_in"]
    clear_x, clear_y = g["bay_x_in"] - width, g["bay_y_in"] - width
    long_clear, short_clear = max(clear_x, clear_y), min(clear_x, clear_y)
    results = []
    for i in range(g["num_bay_x"]):
        for j in range(g["num_bay_y"]):
            edges = []
            definitions = (("x_min", "y", i == 0, g["bay_x_in"]),
                           ("x_max", "y", i == g["num_bay_x"] - 1, g["bay_x_in"]),
                           ("y_min", "x", j == 0, g["bay_y_in"]),
                           ("y_max", "x", j == g["num_bay_y"] - 1, g["bay_y_in"]))
            for side, beam_axis, discontinuous, transverse_bay in definitions:
                n_flange = 1 if discontinuous else 2
                ib, projection, flange_width = _beam_inertia(width, depth, slab_h, n_flange, transverse_bay)
                # Interior strip width follows adjacent panel centerlines. At an
                # edge retain the full bay instead of the narrower edge strip;
                # this lowers alpha and is conservative for this thickness screen.
                slab_i = transverse_bay * slab_h ** 3 / 12.0
                edges.append({"side": side, "beam_axis": beam_axis,
                              "discontinuous": discontinuous, "flange_count": n_flange,
                              "flange_projection_in": projection, "flange_width_in": flange_width,
                              "beam_inertia_in4": ib, "slab_strip_width_in": transverse_bay,
                              "slab_strip_inertia_in4": slab_i, "alpha_f": ib / slab_i})
            alpha_fm = sum(edge["alpha_f"] for edge in edges) / 4.0
            weak_edges = [edge["side"] for edge in edges if edge["discontinuous"] and edge["alpha_f"] < 0.8]
            minimum = _required_thickness(long_clear, long_clear / short_clear, alpha_fm,
                                          bool(weak_edges), policy["fy_ksi"])
            count = sum(edge["discontinuous"] for edge in edges)
            panel = {"panel_id": f"panel_x{i + 1}_y{j + 1}", "bay_x_index": i + 1,
                     "bay_y_index": j + 1, "num_floors_represented": g["num_floor"],
                     "panel_type": "interior" if count == 0 else "edge" if count == 1 else "corner_or_multi_edge",
                     "discontinuous_edge_count": count, "clear_span_x_in": clear_x,
                     "clear_span_y_in": clear_y, "long_clear_span_in": long_clear,
                     "beta": long_clear / short_clear, "alpha_fm": alpha_fm,
                     "weak_discontinuous_edges": weak_edges, "edges": edges,
                     "thickness_in": slab_h}
            if minimum is None:
                panel.update({"status": "not_evaluated", "required_thickness_in": None,
                              "reason": "alpha_fm <= 0.2 invokes ACI 8.3.1.1; weak-beam/flat-slab scope is not implemented."})
            else:
                panel.update(minimum)
                panel["status"] = "pass" if slab_h >= minimum["required_thickness_in"] else "fail"
            results.append(panel)
    return results


def choose_slab(geometry: dict, sections: dict, policy: dict) -> dict:
    """Choose the first allowed thickness passing every panel at that thickness.

    Recompute alpha for every trial because slab I varies with h cubed. Never
    substitute the maximum thickness when the screen is unsupported or fails.
    The returned floor dead load excludes live load and member self-weight.
    """
    g, s, p = _inputs(geometry, sections, policy)
    if floor_sections.is_by_floor(s):
        return _choose_slab_by_floor(g, s, p)
    lower, upper, increment = (p[key] for key in ("minimum_thickness_in", "maximum_thickness_in", "thickness_increment_in"))
    steps = int(math.floor((upper - lower) / increment + 1e-10)) + 1
    if steps > 10000:
        raise ValueError("Slab thickness ladder exceeds 10,000 trials; increase the thickness increment.")
    trials = []
    for step in range(steps):
        slab_h = lower + step * increment
        if slab_h >= min(s["h_beam_in"], g["story_h_in"]):
            trials.append({"thickness_in": slab_h, "status": "not_evaluated",
                           "reason": "Slab must be thinner than the downstand beam and story height."})
            continue
        panels = _panels(g, s, p, slab_h)
        supported = [panel for panel in panels if panel["required_thickness_in"] is not None]
        governing = max(supported, key=lambda panel: panel["required_thickness_in"]) if supported else None
        passed = all(panel["status"] == "pass" for panel in panels)
        trials.append({"thickness_in": slab_h, "status": "pass" if passed else "not_evaluated" if len(supported) < len(panels) else "fail",
                       "required_thickness_in": governing["required_thickness_in"] if governing else None,
                       "governing_panel_id": governing["panel_id"] if governing else None,
                       "unsupported_panel_count": len(panels) - len(supported),
                       "alpha_fm_min": min(panel["alpha_fm"] for panel in panels),
                       "alpha_fm_max": max(panel["alpha_fm"] for panel in panels)})
        if passed:
            weight = p["concrete_unit_weight_kcf"] * slab_h / 12.0
            return {"method_version": METHOD_VERSION, "stage": "thickness_screen_only",
                    "thickness_screen_passed": True, "thickness_in": slab_h,
                    "uniform_thickness_all_floors_including_roof": True,
                    "concrete_fc_ksi": s["fc_beam_ksi"], "concrete_type": p["concrete_type"],
                    "concrete_unit_weight_kcf": p["concrete_unit_weight_kcf"],
                    "superimposed_dead_load_ksf": p["superimposed_dead_load_ksf"],
                    "self_weight_ksf": weight, "total_dead_load_ksf": weight + p["superimposed_dead_load_ksf"],
                    "live_load_mass_fraction": p["live_load_mass_fraction"],
                    "inputs": {"geometry": g, "sections": s, "policy": p},
                    "panel_count_per_floor": len(panels), "panel_count_building": len(panels) * g["num_floor"],
                    "governing_panel_id": governing["panel_id"],
                    "required_thickness_in": governing["required_thickness_in"],
                    "panels": panels, "trial_history": trials,
                    "stiffness_basis": "Gross monolithic T/L beam per ACI 8.4.1.8; same Ec for slab/beam; full transverse bay slab strip even at perimeter (conservative).",
                    "scope_limitations": ["Thickness screen only; no slab flexure, shear, reinforcement, or explicit long-term deflection analysis.",
                                          "No openings, cantilevers, drops, changes between floors, or prestressing.",
                                          "Slab stiffness here sizes thickness only; it does not itself change frame-analysis beam stiffness."]}
    unsupported = sum(trial["status"] == "not_evaluated" for trial in trials)
    raise SlabSizingError(f"No slab thickness passes the supported screen in {lower:g}-{upper:g} in at {increment:g} in increments ({unsupported}/{len(trials)} unsupported trials). Revisit beam/slab geometry or design policy; no fallback thickness was assigned.")


def evaluate_slab(record) -> list:
    """Recompute the screen instead of trusting a saved success flag or evidence."""
    clause = "ACI 318-19 8.3.1.2, 8.3.1.2.1; slab research-scope audit"
    if not isinstance(record, dict) or not record:
        return [not_evaluated("slab_thickness_screen", clause, "Automatic slab thickness evidence is missing.")]
    inputs = record.get("inputs", {})
    try:
        expected = choose_slab(inputs.get("geometry"), inputs.get("sections"), inputs.get("policy"))
    except (ValueError, TypeError, AttributeError, OverflowError) as exc:
        return [not_evaluated("slab_thickness_screen", clause, f"Cannot independently recompute the slab screen: {exc}")]
    matches = record == expected
    return [make_check("slab_thickness_evidence", clause, int(matches), 1, comparison="==",
                       details={"reason": "The saved slab record must exactly match deterministic recomputation.",
                                "method_version": METHOD_VERSION}),
            make_check("slab_thickness_screen", "ACI 318-19 Table 8.3.1.2 and 8.3.1.2.1",
                       expected["required_thickness_in"], record.get("thickness_in"), units="in",
                       location=expected["governing_panel_id"],
                       details={"scope": "Minimum thickness screen only; strip actions, reinforcement, shear and "
                                         "serviceability are the slab_strip_* checks."})]
