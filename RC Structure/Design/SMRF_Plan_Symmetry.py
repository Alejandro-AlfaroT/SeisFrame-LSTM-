"""A capacity-independent bound on one-sided strength for the symmetric archetype.

ASCE 7-22 Table 12.3-1 counts strength at OR on one side of the mass center.
Pairwise reflected frames of identical physical members have equal nominal
strength, even if columns, joints or shear govern. With no SFRS line at the
center, either side therefore has exactly half the total. This does not give
the total strength, establish a vertical mechanism, or verify M1.

Domain: the declared rectangular regular grid, uniform floor loading/slab,
centered mass, identical fixed bases, reflected loading envelopes, and a
perimeter/core system policy. Every actual member, material/cage and physical
column-bar reflection is checked. Unknown/asymmetric domains return unresolved.
"""
from __future__ import annotations
import math
import Structure_Parameters as sp
from Model import Member_Groups as mg

METHOD = "reflected_frame_pair_strength_fraction_v1"


def strength_distribution_bound(state, geometry, *, clear_cover_in, grid_in):
    problems=[]
    if state is None or not mg.is_system_policy(state.policy_id):
        return {"method": METHOD, "verified": False, "reason": "not a supported perimeter/core grouped archetype"}
    counts=state.counts
    if any(geometry.get(k)!=counts[k] for k in ("num_bay_x","num_bay_y","num_floor")):
        problems.append("geometry and installed member counts differ")
    if any(not isinstance(geometry.get(k),(int,float)) or not math.isfinite(geometry[k]) or geometry[k]<=0
           for k in ("bay_x_in","bay_y_in","story_h_in")):
        problems.append("uniform positive grid dimensions are not established")
    nx,ny=counts["num_bay_x"],counts["num_bay_y"]
    members=state.members()
    index={(m.member_type,m.story_or_floor,m.grid_i,m.grid_j):m for m in members}
    layout=mg.layout_of(state.policy_id)
    pairs={"x":0,"y":0};central={"x":[],"y":[]}
    for axis in ("x","y"):
        for member in members:
            i,j=member.grid_i,member.grid_j
            if axis=="x":j=ny-j-(member.member_type=="beam_y")
            else:i=nx-i-(member.member_type=="beam_x")
            other=index.get((member.member_type,member.story_or_floor,i,j))
            if other is None or other.design.section_key()!=member.design.section_key():
                problems.append(f"{axis} reflection changes member {member.member_tag} or its section/material/cage")
            if member.member_type==f"beam_{axis}" and mg.member_system(
                    member.member_type,member.grid_i,member.grid_j,nx,ny,layout)==mg.SYSTEM_SFRS:
                line=member.grid_j if axis=="x" else member.grid_i
                last=ny if axis=="x" else nx
                if 2*line==last:central[axis].append(member.member_tag)
                elif 2*line<last:pairs[axis]+=1
        if central[axis]:problems.append(f"{axis} has an SFRS line at the mass center")
        if not pairs[axis]:problems.append(f"{axis} has no reflected SFRS frame pairs")
    checked=set()
    for member in members:
        d=member.design
        if not d.is_column or d.section_key() in checked:continue
        checked.add(d.section_key())
        try:
            cover=sp.first_bar_offset_in(clear_cover_in,sp.rebar_diameter(d.stirrup_bar_size),
                                          sp.rebar_diameter(d.bar_size),grid_in)
            bars=[]
            for y,count in ((d.h_in/2-cover,d.top_bars),(-d.h_in/2+cover,d.bot_bars)):
                bars.extend((x-d.b_in/2,y) for x in sp.face_bar_positions_in(d.b_in,cover,count,grid_in))
            for x in (-d.b_in/2+cover,d.b_in/2-cover):
                bars.extend((x,y-d.h_in/2) for y in sp.face_bar_positions_in(d.h_in,cover,d.side_bars+2,grid_in)[1:-1])
            canonical=sorted((round(x,8),round(y,8)) for x,y in bars)
            if any(canonical!=sorted((round(sx*x,8),round(sy*y,8)) for x,y in bars) for sx,sy in ((-1,1),(1,-1))):
                problems.append(f"column cage {member.group_id} is not doubly symmetric")
        except ValueError as exc:
            problems.append(f"column cage {member.group_id}: {exc}")
    return {"method":METHOD,"verified":not problems,"one_side_fraction":.5 if not problems else None,
            "state_identity":state.identity(),"geometry":dict(geometry),"column_clear_cover_in":clear_cover_in,
            "bar_position_grid_in":grid_in,"member_reflection_checks":2*len(members),
            "distinct_column_cages_checked":len(checked),"sfrs_span_pairs_by_direction":pairs,
            "centerline_sfrs_members":central,"problems":sorted(set(problems)),
            "scope":"one-sided fraction only; total/vertical story strength M1 remains separate",
            "basis":"identical reflected frame pairs, no centerline SFRS, centered uniform-load archetype; each side is S/(2S) for positive total nominal strength"}


def bound_from_record(record):
    """Recompute from the saved physical state; never trust a stored verified flag."""
    try:
        state=mg.GroupedDesign.from_record(record["member_groups"])
        grid=record["request_identity"]["policy"]["rebar"]["bar_position_grid_in"]
        return strength_distribution_bound(state,record["geometry"],
            clear_cover_in=record["materials"]["col_clear_cover_in"],grid_in=grid)
    except (KeyError,TypeError,ValueError,mg.GroupedStateError) as exc:
        return {"method":METHOD,"verified":False,"reason":f"saved symmetry inputs do not reconstruct: {exc}"}
