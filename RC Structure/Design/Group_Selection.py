"""Reinforcement selection for a grouped design: one cage per group, on the group's own demands (2026-10-02).

The uniform design picks one column cage from the building's worst column and one beam cage from the
largest beam moments. Here every group selects its own cage from the same candidate domain
(Design.Config.RebarConfig: bar sizes, counts, steel ratios) and the same constructibility rules
(Redesign._col_candidates and _beam_candidates), with each rule evaluated on the members that actually
adjoin:

  beams     candidates are priced on the group's own sagging and hogging envelopes, at the bar depth the
            orthogonal layers give them where they cross the other beams (Model.Member_Bar_Layers), and
            must pass the lanes of every column cage they thread and 20 db through every joint core
            they pass (18.8.2.3);
  columns   the groups of one plan location form a chain over the height. Each band's candidates are
            priced on its members' concurrent P, My, Mz in every combination; adjacent bands must form a
            supported transition (Design.SMRF_Transitions); every joint of the chain must satisfy
            18.7.3.2 with the two columns actually framing in (one at the roof, no exemption), each at
            its own factored axial range. A column whose axial range leaves a cage's P-M surface has no
            strength there: that cage is not a candidate for it (no clamp, no exception). The least
            cost chain is found exactly by a pass over the bands.

Objective: Design.Config.DCRTargets.objective, as in the uniform pick. ``min_deviation``: the cage whose
governing DCR is closest to the target (within the convergence tolerance), then the least steel;
``min_volume``: the least steel with DCR within the ceiling. The target is a research preference: a
column is never refused for being below it. The strong-column estimate used during selection is the
nominal P-M sweep at the ends of each column's axial range; the exact joint evaluation of the capacity
design is the authority, and ``scwb_margin`` lets the caller ask for more when that evaluation fails.
"""
from __future__ import annotations

import math
from dataclasses import replace

import Structure_Parameters as sp
from Design import Group_Checks as checks
from Design.SMRF_Beam_Slab_Strength import BeamBarRowsError
from Design.SMRF_Transitions import column_transition
from Model import Member_Bar_Layers as bar_layers
from Model import Member_Groups as mg
from Model import Member_Properties as mp
from Model.IMK_Calibration import column_moment_at_axial

RULES_ID = "group_reinforcement_selection_v1"
# Under the system policy (Model.Member_Groups.SYSTEM_POLICY_ID, 2026-10-16) the groups of a design tie select one
# cage together, and the strong-column rule applies only in the directions in which a column is part of a
# perimeter frame (ACI 318-19 18.7.3.2 is a special-moment-frame rule; an interior column and the gravity
# direction of an edge column are members not designated as part of the SFRS, 18.14).
SYSTEM_RULES_ID = "system_reinforcement_selection_v1"
BEAM_UNIT_SWEEPS = 3                      # beam units re-selected within a pass until none changes (crossing beams share joints)
COLUMN_SUPPORTED_BAR_HX_MAX_IN = 14.0


def _trial_state(state, gid, design):
    """A trial with one group's design changed; under the system policy the whole design tie moves with it."""
    tie = state.tie_of(gid)
    return state.with_tied_designs({tie: design}) if tie else state.with_designs({gid: design})


def _scwb_axes(state, member):
    """The directions in which 18.7.3.2 applies at this column's joints: both under the band policy (every frame is
    part of the SFRS); under the system policy the perimeter-frame directions of the column
    (Model.Member_Groups.column_sfrs_directions), none for an interior column."""
    if state.policy_id != mg.SYSTEM_POLICY_ID:
        return ("x", "y")
    directions = mg.column_sfrs_directions(member.grid_i, member.grid_j, sp.NUM_BAY_X, sp.NUM_BAY_Y)
    return tuple(axis for axis in ("x", "y") if axis in directions)


def _beam_units(state):
    """[(label, group ids)] selected together: each beam group alone under the band policy, each beam tie under
    the system policy."""
    ties = state.ties()
    if ties:
        return [(tie, tuple(gids)) for tie, gids in sorted(ties.items()) if mg.SYSTEM_DESIGN_TIES[tie]["member_type"] == "beam"]
    return [(gid, (gid,)) for gid in sorted(g for g, group in state.groups.items() if group["member_type"] != "column")]


def _column_units(state):
    """[(label, plan locations)] whose column groups form one chain: each location under the band policy, the
    locations of each column tie under the system policy."""
    if state.ties():
        return [(tie, tuple(spec["locations"])) for tie, spec in mg.SYSTEM_DESIGN_TIES.items()
                if spec["member_type"] == "column" and tie in state.ties()]
    return [(location, (location,)) for location in mg.COLUMN_LOCATIONS]


def _objective(dcr, steel, cfg):
    if cfg.dcr.objective == "min_volume":
        return (0.0, steel)
    tol = max(cfg.iteration.convergence_tol, 1e-12)
    return (round(abs(dcr - cfg.dcr.dcr_target) / tol) * tol, steel)


# ---- candidate cages --------------------------------------------------------------------------------------
def column_cage_candidates(design, cfg, rho_max=None):
    """Column cages of the declared domain that one section can hold, as MemberDesigns (hoops unchanged).

    The static rules of Redesign._col_candidates on the slab-aware path: ACI ratio limits, clear spacing
    on both faces, supported bars within 14 in (18.7.5.2(f)), no more face bars than the hoop ladder can
    tie, and a layout the largest hoop can confine at the minimum spacing. ``rho_max`` defaults to the
    practical ceiling the uniform strong-column escalation uses (RebarConfig.rho_col_practical_max).
    """
    from Design.SMRF_Capacity_Design import column_layout_confinable
    from Redesign import COLUMN_MAX_TIED_FACE_BARS
    b, h, ag = design.b_in, design.h_in, design.b_in * design.h_in
    low = max(0.01, cfg.rebar.rho_col_min)
    high = min(0.06, cfg.rebar.rho_col_max, cfg.rebar.rho_col_practical_max if rho_max is None else rho_max)
    result = []
    for bar_size in cfg.rebar.bar_sizes_col:
        ab, db = sp.rebar_area(bar_size), sp.rebar_diameter(bar_size)
        clear_min = sp.longitudinal_clear_spacing_in("column", bar_size)
        cover = sp.longitudinal_cover_in("column", bar_size, design.stirrup_bar_size)
        for n_top in cfg.rebar.col_n_top_iter():
            for n_side in cfg.rebar.col_n_side_options:
                if n_top < 2 or n_side < 0:
                    continue
                rho = (2 * n_top + 2 * n_side) * ab / ag
                if not low <= rho <= high + 1e-12:
                    continue
                pitch_b, pitch_h = (b - 2 * cover) / (n_top - 1), (h - 2 * cover) / (n_side + 1)
                if pitch_b - db < clear_min or pitch_h - db < clear_min:
                    continue
                if pitch_b > COLUMN_SUPPORTED_BAR_HX_MAX_IN or pitch_h > COLUMN_SUPPORTED_BAR_HX_MAX_IN:
                    continue
                if max(n_top, n_side + 2) > COLUMN_MAX_TIED_FACE_BARS:
                    continue
                if not column_layout_confinable(b, h, design.fc_ksi, sp.FY_KSI, sp.COL_CLEAR_COVER_IN, n_top, n_side):
                    continue
                result.append(replace(design, bar_size=bar_size, top_bars=n_top, bot_bars=n_top, side_bars=n_side))
    return result


def beam_cage_candidates(design, cfg, through_depth_in=None):
    """Beam cages of the declared domain for one section (bar size and counts; hoops unchanged)."""
    result = []
    for bar_size in cfg.rebar.bar_sizes_beam:
        if through_depth_in is not None and 20.0 * sp.rebar_diameter(bar_size) > through_depth_in:
            continue                                                    # 18.8.2.3: through-bar joint depth
        for n_top in cfg.rebar.beam_n_iter():
            for n_bot in ((n_top,) if cfg.rebar.grouped_beam_symmetric else cfg.rebar.beam_n_iter()):
                if n_top < 2 or n_bot < 2:
                    continue
                result.append(replace(design, bar_size=bar_size, top_bars=n_top, bot_bars=n_bot))
    return result


# ---- beams ------------------------------------------------------------------------------------------------
def _beam_through_depth(state, gid):
    """Smallest joint-core depth a through bar of this group passes (None when every joint is an end)."""
    depths = []
    for tag in state.groups[gid]["member_tags"]:
        member = state.member(tag)
        k, i, j = member.story_or_floor, member.grid_i, member.grid_j
        ends = ((i, j), (i + 1, j)) if member.member_type == "beam_x" else ((i, j), (i, j + 1))
        for gi, gj in ends:
            joint = mg.joint_members(k, gi, gj)
            axis = mp.beam_axis(member)
            if joint[f"beam_{axis}_minus"] is not None and joint[f"beam_{axis}_plus"] is not None:
                core = mp.joint_core_dimensions(k, gi, gj)
                depths.append(core[0] if axis == "x" else core[1])
    return min(depths) if depths else None


def evaluate_beam_candidate(state, gid, candidate, demand, cfg):
    """Price one beam cage in its place: rows from the layering, code limits, governing flexural DCR.

    ``demand`` = (largest sagging, largest hogging moment over the group's members and combinations).
    Returns {"admissible", "threads", "layers", "dcr", "steel_in2", "reason"}.
    """
    from Design.SMRF_Beam_Slab_Strength import validate_bar_rows
    design = candidate
    trial = _trial_state(state, gid, design)
    with mg.installed(trial):
        try:
            arrangement = bar_layers.arrangement()
        except mg.GroupedStateError as exc:
            return {"admissible": False, "reason": f"bar layers do not settle: {exc}"}
    rows = arrangement[gid]
    n = max(design.top_bars, design.bot_bars)
    fit = arrangement["_basis"].get("bars_per_layer_governing", {}).get(gid)
    stacked = arrangement["_basis"].get("stacked", False)
    threads = (not stacked) or (fit is not None and fit > 0 and math.ceil(n / fit) <= sp.BEAM_BAR_MAX_LAYERS)
    db = sp.rebar_diameter(design.bar_size)
    if not threads:
        # No lane arrangement: the bars are kept in one row (the threading check fails on it); they must
        # at least fit across the width.
        cover = sp.longitudinal_cover_in("beam", design.bar_size, design.stirrup_bar_size)
        if (design.b_in - 2 * cover) / (n - 1) - db < sp.longitudinal_clear_spacing_in("beam", design.bar_size):
            return {"admissible": False, "reason": "bars do not fit across the width"}
    try:
        validate_bar_rows(rows["top"], design.top_bars, "top bars", design.h_in)
        validate_bar_rows(rows["bottom"], design.bot_bars, "bottom bars", design.h_in)
    except ValueError as exc:
        return {"admissible": False, "reason": str(exc)}
    strengths = checks.beam_flexural_strengths(design, rows)
    d = min(strengths["positive"]["d_in"], strengths["negative"]["d_in"])
    ab = sp.rebar_area(design.bar_size)
    as_top, as_bot = design.top_bars * ab, design.bot_bars * ab
    as_min = max(strengths["positive"]["as_min_in2"], strengths["negative"]["as_min_in2"])
    if d <= 0 or max(as_top, as_bot) > 0.025 * design.b_in * d:
        return {"admissible": False, "reason": "steel ratio above 0.025 (18.6.3.1)"}
    if min(as_top, as_bot) < as_min:
        return {"admissible": False, "reason": "below minimum steel (9.6.1.2)"}

    def nominal(area):
        a = area * sp.FY_KSI / (0.85 * design.fc_ksi * design.b_in)
        return area * sp.FY_KSI * (d - a / 2.0)

    mn_pos, mn_neg = nominal(as_bot), nominal(as_top)
    if min(mn_pos, mn_neg) <= 0 or mn_pos < 0.5 * mn_neg or min(mn_pos, mn_neg) < 0.25 * max(mn_pos, mn_neg):
        return {"admissible": False, "reason": "end reversal or along-span strength balance (18.6.3.2)"}
    mu_pos, mu_neg = demand
    dcr = max(mu_pos / strengths["positive"]["phi_mn_kip_in"], mu_neg / strengths["negative"]["phi_mn_kip_in"])
    return {"admissible": True, "threads": threads, "layers": max(rows["top"]["layers"], rows["bottom"]["layers"]),
            "dcr": dcr, "steel_in2": as_top + as_bot, "reason": None}


def select_beam_group(state, gid, actions, cfg):
    """The cage one beam group selects, given every other group as it stands. Returns (design, report)."""
    return select_beam_groups(state, (gid,), actions, cfg)


def _merge_beam_results(results):
    """One cage priced in several groups' places: it threads only if it threads everywhere, its layers and DCR are
    the worst, its steel the sum."""
    return {"admissible": True, "threads": all(r["threads"] for r in results), "layers": max(r["layers"] for r in results),
            "dcr": max(r["dcr"] for r in results), "steel_in2": sum(r["steel_in2"] for r in results), "reason": None,
            "by_group": [{k: r[k] for k in ("threads", "layers", "dcr")} for r in results]}


def select_beam_groups(state, gids, actions, cfg):
    """The cage the beam groups ``gids`` select together: one group under the band policy, the groups of a design
    tie under the system policy. Priced on the union of their demands, admissible only in every group's own place
    (its own layering, joint cores and column lanes). Returns (design typed as the first group, report)."""
    gids = tuple(gids)
    current = state.designs[gids[0]]
    rows = [row for gid in gids for row in checks.beam_demand_rows(state.groups[gid]["member_tags"], actions)]
    demand = (max(r[2] for r in rows), max(r[3] for r in rows))
    priced = []
    with mg.installed(state):
        depths = [_beam_through_depth(state, gid) for gid in gids]
    through = min((d for d in depths if d is not None), default=None)
    for candidate in beam_cage_candidates(current, cfg, through):
        results = [evaluate_beam_candidate(state, gid, replace(candidate, member_type=state.groups[gid]["member_type"]), demand, cfg)
                   for gid in gids]
        if all(r["admissible"] for r in results):
            priced.append((candidate, _merge_beam_results(results)))
    report = {"group_id": gids[0] if len(gids) == 1 else None, "groups": list(gids),
              "demand_mu_positive_kip_in": demand[0], "demand_mu_negative_kip_in": demand[1],
              "candidates_admissible": len(priced), "through_bar_depth_in": through}
    if not priced:
        return current, {**report, "status": "no_admissible_cage", "exhausted": True,
                         "detail": "no cage of the declared domain satisfies the code limits in this section"}
    pool = [item for item in priced if item[1]["threads"]] or priced
    report["lanes_available"] = any(item[1]["threads"] for item in priced)
    feasible = [item for item in pool if item[1]["dcr"] <= cfg.dcr.dcr_hard_max]
    if not feasible:
        candidate, result = min(pool, key=lambda item: item[1]["dcr"])
        # Lane-limited: a cage that reaches the demand exists but does not thread the column bars (room, not strength).
        lane_limited = any(item[1]["dcr"] <= cfg.dcr.dcr_hard_max for item in priced if not item[1]["threads"])
        return candidate, {**report, "status": "strength_short", "exhausted": True, "dcr": result["dcr"],
                           "lane_limited": lane_limited, "strongest_dcr_any_cage": min(item[1]["dcr"] for item in priced),
                           "detail": ("the cages that reach the demand do not thread the column cage in enough lanes; the strongest "
                                      "threading cage is installed so the shortfall is measured" if lane_limited else
                                      "no admissible cage reaches the demand; the strongest is installed so the shortfall is measured")}
    if cfg.rebar.beam_prefer_fewest_layers:
        fewest = min(item[1]["layers"] for item in feasible)
        feasible = [item for item in feasible if item[1]["layers"] == fewest]
    candidate, result = min(feasible, key=lambda item: _objective(item[1]["dcr"], item[1]["steel_in2"], cfg))
    return candidate, {**report, "status": "selected", "exhausted": False, "dcr": result["dcr"], "layers": result["layers"],
                       "threads": result["threads"]}


# ---- columns ----------------------------------------------------------------------------------------------
def _column_chain(state, location):
    """Per band from the base, the column group ids of one plan location (a tuple of one), or of several locations
    selected together (the locations of a design tie under the system policy)."""
    locations = (location,) if isinstance(location, str) else tuple(location)
    groups = [g for g in state.groups.values() if g["member_type"] == "column" and g["location_class"] in locations]
    bands = {}
    for group in groups:
        bands.setdefault(min(group["stories_or_floors"]), []).append(group["group_id"])
    return [tuple(sorted(bands[start])) for start in sorted(bands)]


def _axial_ranges(tags, actions):
    """{tag: {"i": (min, max), "j": (min, max)}} joint-face axial range of each column over the combinations."""
    result = {}
    for tag in tags:
        ends = {}
        for end in ("i", "j"):
            values = [float(a["members"][str(tag)][f"axial_{end}_kip"]) for a in actions]
            ends[end] = (min(values), max(values))
        result[tag] = ends
    return result


def _column_mn_estimate(design, axis, axial_range):
    """Least nominal moment over the ends of an axial range (0 when the range leaves the P-M surface)."""
    diagram = mp.column_pm_diagram(design, "h" if axis == "x" else "b")
    return min(column_moment_at_axial(p, diagram) for p in axial_range)


def _joint_requirements(state, established):
    """{(level, i, j): {axis: required sum of column Mn}}: 1.2 times the larger beam sum of the two sways."""
    if not established:
        return {}
    requirements = {}
    for level in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                joint = mg.joint_members(level, i, j)
                entry = {}
                for axis in ("x", "y"):
                    sums = {"positive": 0.0, "negative": 0.0}
                    for side, end in (("minus", "j"), ("plus", "i")):
                        beam = joint[f"beam_{axis}_{side}"]
                        if beam is None:
                            continue
                        basis = mp.beam_strengths(beam)["basis"]
                        for sign in ("positive", "negative"):
                            flexure = sign if end == "j" else ("negative" if sign == "positive" else "positive")
                            sums[sign] += basis[f"{'sagging' if flexure == 'positive' else 'hogging'}_{end}_kip_in"]
                    entry[axis] = sp.SCWB_RATIO_MIN * max(sums.values())
                requirements[(level, i, j)] = entry
    return requirements


def _core_beams(state, gid):
    """[(beam design, axis)] of the beams whose bars thread this column group's cage (it is their joint core)."""
    pairs = {}
    for tag in state.groups[gid]["member_tags"]:
        member = state.member(tag)
        joint = mg.joint_members(member.story_or_floor, member.grid_i, member.grid_j)
        for axis in ("x", "y"):
            for side in ("minus", "plus"):
                beam = joint[f"beam_{axis}_{side}"]
                if beam is not None:
                    pairs[(beam.group_id, axis)] = (beam.design, axis)
    return list(pairs.values())


def _column_clear_height_face(member):
    from Design.Group_Capacity import column_clear_heights
    return column_clear_heights(member)["face"]


def _cage_develops_its_bars(candidate, clear_height_in):
    """ACI 318-19 18.7.4.3 with Eq. (25.4.2.4a) at the cage's own cb and Ktr = 0 (Design/ACI_Checks)."""
    from Design.ACI_Checks import column_bar_bond_18_7_4_3, column_bar_cb_in
    cb = column_bar_cb_in(candidate.b_in, candidate.h_in, sp.COL_CLEAR_COVER_IN, candidate.stirrup_bar_size,
                          candidate.bar_size, candidate.top_bars, candidate.side_bars)["cb_in"]
    return column_bar_bond_18_7_4_3(candidate.bar_size, candidate.fc_ksi, sp.FY_KSI, cb, clear_height_in)["passes"]


def select_column_chain(state, location, actions, cfg, scwb_margin=1.0, requirements=None):
    """Cages of the column groups of one plan location, chosen together. Returns ({gid: design}, report).

    ``requirements`` are the strong-column sums of _joint_requirements built once per selection pass on the
    beams just selected (their rows validated against the cages of that moment); built here when not given.

    A pass over the bands from the base: each band's candidates carry the least total cost of a feasible
    chain below them. Node feasibility: strength on the band's own concurrent demands, and 18.7.3.2 at
    the joints whose two columns are both in the band and at the roof. Link feasibility: a supported
    transition, and 18.7.3.2 at the band-boundary joints with the lower band's cage below and the upper
    band's above.
    """
    chain = _column_chain(state, location)
    label = location if isinstance(location, str) else "+".join(location)
    if not chain:
        return {}, {"location": label, "status": "absent"}
    # The strong-column estimate screens the cages whenever the beams have end strengths to sum (a slab is
    # installed). Without an established slab layout the beam sums carry no slab mats, so the estimate is the
    # grouped form of the uniform proxy screen and the exact joint evaluation stays not_evaluated.
    established = sp.SLAB_THICKNESS_IN is not None
    layout_established = established and (sp.SLAB_REINFORCEMENT or {}).get("layout") is not None
    alpha = getattr(cfg.dcr, "biaxial_contour_exponent", 1.5)
    with mg.installed(state):
        if requirements is None:
            requirements = _joint_requirements(state, established)
        core_beams = {gids: [pair for gid in gids for pair in _core_beams(state, gid)] for gids in chain}
    report = {"location": label, "chain": [list(gids) for gids in chain], "scwb_margin": scwb_margin,
              "scwb_in_selection": established,
              "scwb_screen": ("beam plus developed slab mats" if layout_established else
                              "proxy: beams without slab mats (no slab layout established)" if established else "none"),
              "bands": {}}
    nodes = []                                                      # per band: [(candidate, cost, info)]
    for gids in chain:
        key_label, current = "+".join(gids), state.designs[gids[0]]
        tags = [tag for gid in gids for tag in state.groups[gid]["member_tags"]]
        rows = checks.column_demand_rows(tags, actions)
        ranges = _axial_ranges(tags, actions)
        members = {tag: state.member(tag) for tag in tags}
        stories = set().union(*(set(state.groups[gid]["stories_or_floors"]) for gid in gids))
        priced, rejected = [], {"strength": 0, "axial_domain": 0, "scwb": 0, "bar_bond": 0}
        clear_height = min(_column_clear_height_face(member) for member in members.values())
        for candidate in column_cage_candidates(current, cfg):
            if not _cage_develops_its_bars(candidate, clear_height):
                # ACI 318-19 18.7.4.3 (2026-10-02): 1.25 ld at Ktr = 0 must fit in half the clear height.
                rejected["bar_bond"] += 1
                continue
            envelope = checks.column_pm_envelope(checks.column_pm_diagrams(candidate), rows, alpha)
            dcr = envelope["dcr"]
            if envelope["outside_domain"]:
                # A demand outside the cage's axial domain, in compression or in tension, rejects the cage by name.
                rejected["axial_domain"] += 1
                continue
            if dcr > cfg.dcr.dcr_hard_max:
                rejected["strength"] += 1
                continue
            needed = [math.ceil(max(b.top_bars, b.bot_bars) / fit) if fit and fit > 0 else 99
                      for b, axis in core_beams[gids] for fit in (bar_layers.bars_per_layer(candidate, b, axis),)]
            layers = max(needed, default=1)
            # Strong column at the joints that involve only this band (both columns in it, or the roof).
            mn = {(tag, end, axis): _column_mn_estimate(candidate, axis, ranges[tag][end])
                  for tag in tags for end in ("i", "j") for axis in ("x", "y")}
            ok = True
            for tag, member in members.items():
                level = member.story_or_floor                          # the joint at this column's top
                key = (level, member.grid_i, member.grid_j)
                if key not in requirements:
                    continue
                above_story = level + 1
                for axis in _scwb_axes(state, member):
                    required = scwb_margin * requirements[key][axis]
                    if above_story > sp.NUM_FLOOR:
                        provided = mn[(tag, "j", axis)]
                    elif above_story in stories:
                        above = mg_column_tag(above_story, member.grid_i, member.grid_j)
                        provided = mn[(tag, "j", axis)] + mn[(above, "i", axis)]
                    else:
                        continue                                    # a band boundary: decided on the link
                    if provided < required:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                rejected["scwb"] += 1
                continue
            steel = candidate.longitudinal_area_in2 * len(tags) * sp.STORY_H
            priced.append({"design": candidate, "dcr": dcr, "layers": layers, "mn": mn, "steel": steel,
                           "threads": layers <= sp.BEAM_BAR_MAX_LAYERS})
        threading = [item for item in priced if item["threads"]]
        preferred = threading or priced
        if cfg.rebar.col_prefer_fewest_beam_layers and preferred:
            fewest = min(item["layers"] for item in preferred)
            preferred = [item for item in preferred if item["layers"] == fewest]
        report["bands"][key_label] = {"candidates_admissible": len(priced), "lanes_available": bool(threading),
                                      "rejected": rejected, "members": len(tags), "groups": list(gids)}
        # The beam-layer preferences narrow a band's cages; they are preferences, so the chain is solved on the
        # preferred cages first and on the wider pools only when no chain exists among them.
        nodes.append({"gids": gids, "key": key_label, "pool": preferred, "pools": [preferred, threading or priced, priced],
                      "members": members, "ranges": ranges, "stories": stories})

    def boundary_ok(lower_node, lower, upper_node, upper):
        """18.7.3.2 at the joints between two bands, lower cage below and upper cage above."""
        top_story = max(lower_node["stories"])
        for tag, member in lower_node["members"].items():
            if member.story_or_floor != top_story:
                continue
            key = (top_story, member.grid_i, member.grid_j)
            if key not in requirements:
                continue
            above = mg_column_tag(top_story + 1, member.grid_i, member.grid_j)
            for axis in _scwb_axes(state, member):
                if lower["mn"][(tag, "j", axis)] + upper["mn"][(above, "i", axis)] < scwb_margin * requirements[key][axis]:
                    return False
        return True

    # Per band boundary: the clear height of the upper story and the joint depth an offset bend has to fit in.
    boundary = {}
    with mg.installed(state):
        from Design.Group_Capacity import column_clear_heights
        for k in range(1, len(nodes)):
            member = min(nodes[k]["members"].values(), key=lambda m: m.member_tag)      # a column of the band's first story
            joint = mg.joint_members(member.story_or_floor - 1, member.grid_i, member.grid_j)
            depths = [joint[key].design.h_in for key in ("beam_x_minus", "beam_x_plus", "beam_y_minus", "beam_y_plus")
                      if joint[key] is not None]
            boundary[k] = (column_clear_heights(member)["physical"], min(depths))
    transitions = {}

    def transition_ok(k, lower, upper):
        key = (k, lower["design"], upper["design"])
        if key not in transitions:
            clear, depth = boundary[k]
            transition = column_transition(lower["design"], upper["design"], fy_ksi=sp.FY_KSI,
                                           clear_cover_in=sp.COL_CLEAR_COVER_IN, upper_clear_height_in=clear,
                                           joint_depth_in=depth)
            transitions[key] = (transition["supported"],
                                transition.get("extra_bar_length_in", 0.0) * sp.rebar_area(lower["design"].bar_size))
        return transitions[key]

    def forward():
        """best[k][n] = (cost of the cheapest feasible chain ending in candidate n of band k, predecessor)."""
        best = []
        for k, node in enumerate(nodes):
            column = []
            for item in node["pool"]:
                own = _objective(item["dcr"], item["steel"], cfg)
                if k == 0:
                    column.append((own, None))
                    continue
                options = []
                for index, previous in enumerate(nodes[k - 1]["pool"]):
                    if best[k - 1][index] is None:
                        continue
                    supported, extra = transition_ok(k, previous, item)
                    if not supported or not boundary_ok(nodes[k - 1], previous, node, item):
                        continue
                    total = tuple(a + b for a, b in zip(best[k - 1][index][0], own))
                    lines = len(node["members"]) // max(1, len(node["stories"]))        # column lines of this location
                    options.append(((total[0], total[1] + extra * lines), index))
                column.append(min(options, key=lambda o: o[0]) if options else None)
            best.append(column)
        return best, ([(entry[0], index) for index, entry in enumerate(best[-1]) if entry is not None] if best and best[-1] else [])

    tiers = ("preferred_beam_layers", "cages_the_beam_bars_thread", "all_admissible_cages")
    for tier, name in enumerate(tiers):
        if tier and all(len(node["pools"][tier]) == len(node["pools"][tier - 1]) for node in nodes):
            continue                                                    # the wider pools add nothing here
        for node in nodes:
            node["pool"] = node["pools"][tier]
        best, last = forward()
        report["cage_pool"] = name
        if last:
            break
    if not last:
        # No feasible chain: say where it breaks and install, per band, the cage that band would select on its own
        # demands (its own objective, the chain rules set aside), so the checks measure the shortfall on a cage of
        # the size the band needs. The heaviest cage would raise the probable moments and with them the capacity
        # shears of this line, and the failure would be read as a shear problem. Nothing is clamped or passed.
        blocked_index = next((k for k, column in enumerate(best) if not any(e is not None for e in column)), 0)
        blocked = nodes[blocked_index]["key"]
        selected = {}
        for node in nodes:
            current = state.designs[node["gids"][0]]
            if node["pool"]:
                design = min(node["pool"], key=lambda item: _objective(item["dcr"], item["steel"], cfg))["design"]
            else:
                everything = column_cage_candidates(current, cfg)
                design = max(everything, key=lambda c: c.longitudinal_area_in2) if everything else current
            selected.update({gid: design for gid in node["gids"]})
        report.update(status="no_feasible_chain", exhausted=True, blocked_at=blocked,
                      blocked_by=("no admissible cage of this band meets its own strength and strong-column demands"
                                  if not nodes[blocked_index]["pool"] else
                                  "no cage of this band continues a feasible chain from the band below (transition or the "
                                  "strong-column rule at the boundary joints)"),
                      detail=("no combination of admissible cages along this column line satisfies strength, the "
                              "strong-column rule and a supported transition at every band; each band carries the cage it "
                              "would select on its own demands so the checks measure the shortfall"))
        return selected, report
    _cost, index = min(last, key=lambda item: item[0])
    selected = {}
    for k in range(len(nodes) - 1, -1, -1):
        item = nodes[k]["pool"][index]
        selected.update({gid: item["design"] for gid in nodes[k]["gids"]})
        report["bands"][nodes[k]["key"]].update(dcr=item["dcr"], layers=item["layers"], threads=item["threads"],
                                                rho=item["design"].longitudinal_area_in2 / item["design"].gross_area_in2)
        index = best[k][index][1]
    report.update(status="selected", exhausted=False)
    return selected, report


def mg_column_tag(story, i, j):
    return (story - 1) * (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1) + j * (sp.NUM_BAY_X + 1) + i + 1


# ---- every group ------------------------------------------------------------------------------------------
def select_reinforcement(state, actions, cfg, scwb_margin=1.0, max_passes=4):
    """Longitudinal cages of every group of ``state`` from the solved actions. Returns (state, report).

    Beams first (their strengths set what the columns must exceed), then each column line; repeated
    until no cage changes, because the layers of a beam depend on the cages it threads and crosses. The
    frame actions do not depend on the bars, so no re-analysis is needed between passes.
    """
    tied = bool(state.ties())
    report = {"rules": SYSTEM_RULES_ID if tied else RULES_ID, "objective": cfg.dcr.objective, "scwb_margin": scwb_margin,
              "passes": [], "settled": False, "policy": state.policy_id,
              "units": {"beams": [label for label, _ in _beam_units(state)], "columns": [label for label, _ in _column_units(state)]}}
    for index in range(max_passes):
        changes, beams, columns = {}, {}, {}
        # The beam units until none changes within the pass: a beam's layers at a joint depend on the beams it
        # crosses, so a unit selected before another changed is priced again on the final crossing beams.
        for _sweep in range(BEAM_UNIT_SWEEPS):
            moved = False
            for label, gids in _beam_units(state):
                design, beams[label] = select_beam_groups(state, gids, actions, cfg)
                wanted = {gid: replace(design, member_type=state.groups[gid]["member_type"]) for gid in gids}
                updates = {gid: d for gid, d in wanted.items() if d != state.designs[gid]}
                if updates:
                    changes.update(updates)
                    moved = True
                    state = state.with_tied_designs({label: design}) if tied else state.with_designs(updates)
            if not moved:
                break
        # The strong-column sums once per pass, on the beams just selected: a column unit that takes a heavier cage
        # changes the lanes of the beams that thread it, and their rows are re-laid only by the next pass's beam
        # selection, so the requirements are not rebuilt between column units.
        try:
            with mg.installed(state):
                requirements = _joint_requirements(state, sp.SLAB_THICKNESS_IN is not None)
        except BeamBarRowsError as exc:
            # The beams as selected do not stack at a joint (a layer past mid-depth): no column is priced on them this
            # pass; the next pass re-selects the beams, and a design that never settles is reported, not solved around.
            report["passes"].append({"pass": index + 1, "changed_groups": sorted(changes), "beams": beams, "columns": {},
                                     "beam_rows_problem": str(exc)})
            continue
        for label, locations in _column_units(state):
            selected, columns[label] = select_column_chain(state, locations if tied else locations[0], actions, cfg, scwb_margin,
                                                           requirements=requirements)
            updates = {gid: design for gid, design in selected.items() if design != state.designs[gid]}
            if updates:
                changes.update(updates)
                state = state.with_designs(updates)
        report["passes"].append({"pass": index + 1, "changed_groups": sorted(changes), "beams": beams, "columns": columns})
        if not changes:
            report["settled"] = True
            break
    final = report["passes"][-1]
    report["exhausted_groups"] = sorted([gid for item in final["beams"].values() if item.get("exhausted") for gid in item["groups"]]
                                        + [gid for item in final["columns"].values() if item.get("exhausted")
                                           for band in item.get("chain", []) for gid in band])
    return state, report
