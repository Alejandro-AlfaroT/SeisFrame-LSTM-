"""Story lateral strength by the conventional story mechanism, for the ASCE 7-22 Table 12.3-2 weak-story types (2026-11-07).

Story lateral strength is the total lateral strength of the seismic-force-resisting elements sharing the story shear in the
direction considered (ASCE 7-22 Table 12.3-2, Types 4a and 4b). For a moment frame each column of the story contributes the
shear (M_top + M_bottom) / clear height, where an end moment is the column's own nominal strength or its share of the
nominal strength of the beams framing into that joint, whichever is less. The beams' sum at a joint (one hogging, one sagging,
the developed slab steel included as the strong-column rule includes it) is shared between the column below and the column
above in proportion to their strengths; at the roof the column below takes it all; at the base the column's own strength
governs. Column strengths are nominal at the axial force of the gravity state of the seismic combinations. Both sway signs
are computed and the lesser is the story strength.

This replaces, on the frame-plus-transfer route, the certified equilibrium bounds of SMRF_Story_Adapter, which stay for the
computed route. Nothing is solved; the result is a deterministic function of the saved actions and capacity design.
"""
from __future__ import annotations

import Structure_Parameters as sp

METHOD = "conventional_story_mechanism_strength_v1"


def _gravity_members(actions, gravity_actions):
    seismic = {a["id"] for a in actions if a.get("ex") or a.get("ey")}
    for case in gravity_actions or []:
        if set(case.get("represented_design_combination_ids") or []) & seismic:
            return case["members"], case.get("id")
    raise ValueError("no gravity state of the seismic combinations is available")


def _beam_sum(record):
    total = 0.0
    for beam in record.get("beam_capacities") or []:
        if "mn_kip_in" not in beam:
            raise ValueError(f"beam {beam.get('tag')} has no nominal strength at this joint")
        total += float(beam["mn_kip_in"]) + float(beam.get("slab_mn_kip_in") or 0.0)
    return total


def calculate(state, actions, capacity, gravity_actions):
    """{"method", "stories": [{"story", "by_direction": {axis: {"story_strength_kip", ...}}}], "applicability", ...}."""
    from Design.Group_Capacity import column_clear_heights, column_nominal_moment
    from Model import Member_Groups as mg
    members, gravity_id = _gravity_members(actions, gravity_actions)
    joints = {(j["floor"], j["grid_i"], j["grid_j"]): j for j in (capacity.get("scwb") or {}).get("joints") or []}
    layout = mg.layout_of(state.policy_id)
    nx, ny, nf = sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR
    columns = {(m.story_or_floor, m.grid_i, m.grid_j): m for m in state.members("column")}

    def nominal(member, axis, end):
        p = float(members[str(member.member_tag)][f"axial_{end}_kip"])
        return column_nominal_moment(member.design, axis, p)[0]

    def end_moment(member, axis, sign, level, position):
        """The moment one column end can take at the joint of ``level``: its own strength or its share of the beams'."""
        own = nominal(member, axis, "j" if position == "below" else "i")
        joint = joints.get((level, member.grid_i, member.grid_j))
        if joint is None:
            raise ValueError(f"no joint record at level {level}, grid ({member.grid_i}, {member.grid_j})")
        beams = _beam_sum(joint["directions"][axis][sign])
        below = columns.get((level, member.grid_i, member.grid_j))
        above = columns.get((level + 1, member.grid_i, member.grid_j))
        strengths = []
        if below is not None:
            strengths.append(nominal(below, axis, "j"))
        if above is not None:
            strengths.append(nominal(above, axis, "i"))
        share = beams * own / sum(strengths) if len(strengths) > 1 else beams
        return min(own, share), {"own_kip_in": own, "beam_sum_kip_in": beams, "share_kip_in": share}

    stories = []
    for k in range(1, nf + 1):
        by_direction = {}
        for axis in ("x", "y"):
            signs = {}
            for sign in ("positive", "negative"):
                total, contributions = 0.0, []
                for (story, i, j), member in sorted(columns.items()):
                    if story != k or axis not in mg.column_sfrs_directions(i, j, nx, ny, layout):
                        continue
                    top, top_detail = end_moment(member, axis, sign, k, "below")
                    if k == 1:
                        bottom, bottom_detail = nominal(member, axis, "i"), {"basis": "fixed base: the column's own strength"}
                    else:
                        bottom, bottom_detail = end_moment(member, axis, sign, k - 1, "above")
                    height = column_clear_heights(member)["physical"]
                    shear = (top + bottom) / height
                    total += shear
                    contributions.append({"tag": member.member_tag, "shear_kip": shear, "top_kip_in": top,
                                          "bottom_kip_in": bottom, "clear_height_in": height, "top": top_detail,
                                          "bottom": bottom_detail})
                if not contributions:
                    raise ValueError(f"story {k} has no seismic-force-resisting column in direction {axis}")
                signs[sign] = {"strength_kip": total, "columns": contributions}
            strength = min(signs["positive"]["strength_kip"], signs["negative"]["strength_kip"])
            by_direction[axis] = {"story_strength_kip": strength, "story_strength_lower_kip": strength,
                                  "story_strength_upper_kip": strength, "by_sign": signs}
        stories.append({"story": k, "by_direction": by_direction})
    import math
    high = [a for a in actions if a.get("family") == "seismic_high_gravity"]
    low = [a for a in actions if a.get("family") == "seismic_low_gravity"]
    full_ev = bool(high and low) and all(math.isclose(a["dead"], 1.2 + 0.2 * sp.ASCE_SDS, rel_tol=1e-10) for a in high) \
        and all(math.isclose(a["dead"], 0.9 - 0.2 * sp.ASCE_SDS, rel_tol=1e-10) for a in low)
    return {"method": METHOD, "gravity_state": gravity_id, "stories": stories,
            "unreduced_sds_for_vertical_ground_motion": full_ev, "unreduced_sds": sp.ASCE_SDS,
            "applicability": {"status": "verified", "verified": True,
                              "basis": ("story mechanism of the frame columns with joint shares of the beam strengths; nominal "
                                        "strengths, gravity axial forces of the seismic combinations")}}


def model_valid(result):
    """The result is this method's and complete: a story strength for every story and direction."""
    try:
        return (result.get("method") == METHOD and len(result["stories"]) == sp.NUM_FLOOR
                and all(row["by_direction"][axis]["story_strength_kip"] > 0.0 for row in result["stories"] for axis in ("x", "y")))
    except (KeyError, TypeError, AttributeError):
        return False
