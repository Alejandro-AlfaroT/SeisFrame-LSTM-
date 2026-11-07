"""Actual beam-bar centerlines for column-transition physical verification.

This adapter uses the installed shared row arrangement and its swept-column
lane algorithm. Same-group bars through an interior joint have ONE owner.
Terminating groups have separate standard90deg hooks; each hook family's
actual embedment is checked after staggered vertical tails are placed.
"""
from __future__ import annotations

import math
import json
from collections import OrderedDict

import Structure_Parameters as sp
from Model import Member_Bar_Layers as bar_layers
from Design.SMRF_Cage_Geometry import beam_bars_through_inventory,clear_spacing_limit,hook_geometry
from Design.SMRF_Transitions import bar_coordinates
from Design.SMRF_Transition_Physical import _tube,_clearance

_CONTEXT_CACHE=OrderedDict()


def beam_context(entry,transition=None,column_hoops=None):
    from Model import Member_Groups as mg
    key=(mg.active().identity(),entry["id"],json.dumps((transition or {}).get("paths"),sort_keys=True),
         tuple(sorted((column_hoops or {}).items())),sp.FY_KSI,sp.FYT_KSI,sp.REINFORCEMENT_SPECIFICATION,
         sp.COL_CLEAR_COVER_IN,sp.BEAM_CLEAR_COVER_IN,sp.AGGREGATE_MAX_SIZE_IN,sp.BAR_POSITION_GRID_IN,
         sp.BAR_LAYOUT_HOOP_DB_IN,sp.BAY_X,sp.BAY_Y,sp.STORY_H,sp.ROOF_COLUMN_EXTENSION,sp.SLAB_THICKNESS_IN,
         sp.BEAM_BAR_LAYER_ORDER,sp.BEAM_BAR_STACKING,
         json.dumps((sp.SLAB_REINFORCEMENT or {}).get("layout"),sort_keys=True))
    if key in _CONTEXT_CACHE:
        _CONTEXT_CACHE.move_to_end(key)
        return dict(_CONTEXT_CACHE[key])
    result=_beam_context(entry,transition,column_hoops)
    _CONTEXT_CACHE[key]=result
    if len(_CONTEXT_CACHE)>128:_CONTEXT_CACHE.popitem(last=False)
    return dict(result)


def _beam_context(entry,transition=None,column_hoops=None):
    from Design.SMRF_Splice_Physical import group_beam_lanes
    common_lanes=group_beam_lanes()
    core=entry["core"].design
    above=entry["above"].design if entry["above"] is not None else core
    inventory=bar_layers.transition_inventory(core,above,transition) if transition is not None else None
    if inventory is None:
        inventory=tuple((p["y_in"]+core.h_in/2,p["x_in"]+core.b_in/2,sp.rebar_diameter(core.bar_size))
                        for p in bar_coordinates(core,sp.COL_CLEAR_COVER_IN))
    context={"method":"actual_grouped_beam_segments_v2_bounded_hook_families","beam_segments":[],"beam_geometry_complete":False,
             "beam_geometry_checks":[],"beam_hook_families":[],"beam_longitudinal_adjustment_in3":0.,
             "upper_hoop_spacing_in":(column_hoops or {}).get((entry["above"] or entry["core"]).group_id,above.stirrup_spacing_in),
             "transverse_fy_ksi":sp.FYT_KSI,"aggregate_max_size_in":sp.AGGREGATE_MAX_SIZE_IN,
             "longitudinal_specification":sp.REINFORCEMENT_SPECIFICATION,
             "nominal_coordinate_basis":"x alongcolumnb; y alongcolumnh; z0jointtop"}
    if isinstance(inventory,str):
        context["reason"]="No declared column path inventory exists for beam threading."
        return context
    checks=context["beam_geometry_checks"]
    segments=context["beam_segments"]
    from Design.SMRF_Joint_Mat_Physical import joint_mat_segments
    mats=joint_mat_segments(entry)
    # Continuous bars are fixed obstacles before any terminating hook is
    # placed, regardless of the order in which its orthogonal group is read.
    fixed=[]
    straight_by_axis={"x":[],"y":[]}
    for direction in ("x","y"):
        members=entry["beams"][direction]
        for member in members:
            depth=core.h_in if direction=="x" else core.b_in
            for p in common_lanes[member.group_id]["coordinates"]:
                z=-member.design.h_in/2+p["y_in"]
                start=[p["x_in"],-depth/2,z] if direction=="x" else [-depth/2,p["x_in"],z]
                end=[p["x_in"],depth/2,z] if direction=="x" else [depth/2,p["x_in"],z]
                straight_by_axis[direction].append(_tube(start,end,sp.rebar_diameter(member.design.bar_size),"fixed_beam_row"))
        if len(members)==2 and not (entry.get("mixed") or {}).get(direction,False):
            fixed+=straight_by_axis[direction]
    joint_depth=min(b.design.h_in for members in entry["beams"].values() for b in members)
    for item in inventory:
        *plan,col_db=item
        x0,y0,x1,y1=plan if len(plan)==4 else (*plan,*plan)
        fixed.append(_tube([y0-core.b_in/2,x0-core.h_in/2,-joint_depth],
                           [y1-core.b_in/2,x1-core.h_in/2,0.],col_db,"fixed_column_path"))
    for axis in ("x","y"):
        members=entry["beams"][axis]
        if not members:
            continue
        # Global frameX spans columnh, so its axial direction is physicaly.
        axial,lateral,depth,width=(1,0,core.h_in,core.b_in) if axis=="x" else (0,1,core.b_in,core.h_in)
        continuous=len(members)==2 and not (entry.get("mixed") or {}).get(axis,False)
        selected=[members[0]] if continuous else members
        for member in selected:
            design=member.design
            db=sp.rebar_diameter(design.bar_size)
            rows=bar_layers.group_rows(member.group_id)
            hook=hook_geometry(design.bar_size,"development_90")
            radius=(hook["inside_bend_diameter_in"]+db)/2
            side=-1 if entry["joint"].get(f"beam_{axis}_minus") is not None and entry["joint"][f"beam_{axis}_minus"].member_tag==member.member_tag else 1
            travel=-side
            family=0
            for face in ("top","bottom"):
                for row,(count,offset) in enumerate(zip(rows[face]["per_layer"],rows[face]["offsets_in"])):
                    lane_record=common_lanes[member.group_id]
                    placed=[p["x_in"]+width/2 for p in lane_record["coordinates"]
                            if p["source_face"]==face and p["row"]==row]
                    checks.append({"rule":"actual_beam_row_lanes","passes":len(placed)==count,
                                   "group":member.group_id,"face":face,"row":row,"requested":count,"placed":len(placed)})
                    checks.append({"rule":"one_straight_beam_lane_inventory","passes":lane_record["status"]=="pass",
                                   "group":member.group_id,"status":lane_record["status"],"method":lane_record["method"]})
                    z=-offset if face=="top" else -design.h_in+offset
                    turn=-1 if face=="top" else 1
                    # Stagger whole hook families along embedment, retaining
                    # horizontal beam rows and the full standard12db tail.
                    stagger=family*(db+max(1.,db))
                    colcover=sp.hoop_clear_cover_from_centroid_in(
                        sp.first_bar_offset_in(sp.COL_CLEAR_COVER_IN,sp.rebar_diameter(core.stirrup_bar_size),sp.rebar_diameter(core.bar_size)),
                        sp.rebar_diameter(core.stirrup_bar_size),sp.rebar_diameter(core.bar_size))
                    far=travel*(depth/2-colcover-sp.rebar_diameter(core.stirrup_bar_size)-db/2-stagger)
                    tangent=far-travel*radius
                    ldh=max(sp.FY_KSI*1000*db/(65*math.sqrt(core.fc_ksi*1000)),8*db,6.)
                    if not continuous:
                        base_far=far
                        selected=None
                        trials=0
                        # A bounded fabrication choice, not a weaker fit
                        # test: move a whole hook family inward on the grid,
                        # recompute ldh and every collision at each trial.
                        grid=sp.BAR_POSITION_GRID_IN or .5
                        for trial in range(17):
                            trials+=1
                            extra=trial*grid
                            candidate_far=base_far-travel*extra
                            if depth/2+travel*candidate_far+db/2<ldh-1e-9:break
                            candidate_tangent=candidate_far-travel*radius
                            if travel*(candidate_tangent-side*depth/2)<-1e-9:break
                            proposed=[]
                            for lane in placed:
                                def at(a,zz):
                                    p=[0.,0.,zz];p[axial]=a;p[lateral]=lane-width/2;return p
                                proposed.append(_tube(at(side*depth/2,z),at(candidate_tangent,z),db,"candidate_hook"))
                                arc=[at(candidate_tangent+travel*radius*math.sin(math.pi/2*j/24),
                                        z+turn*radius*(1-math.cos(math.pi/2*j/24))) for j in range(25)]
                                allowance=radius*(1-math.cos(math.pi/96))
                                proposed.extend(_tube(a,b,db,"candidate_hook",allowance=allowance) for a,b in zip(arc,arc[1:]))
                                proposed.append(_tube(arc[-1],at(candidate_far,z+turn*(radius+hook["extension_in"])),db,"candidate_hook"))
                            obstacles=fixed+segments+mats["segments"]+straight_by_axis["y" if axis=="x" else "x"]
                            if not any(_clearance(a,b)<-1e-6 for a in proposed for b in obstacles):
                                selected=(candidate_far,candidate_tangent,stagger+extra)
                                break
                        checks.append({"rule":"bounded_actual_hook_family_placement","passes":selected is not None,
                                       "group":member.group_id,"face":face,"row":row,"maximum_trials":17,"trials":trials})
                        if selected is not None:
                            far,tangent,stagger=selected
                    # ldh terminates at the hook OUTSIDE edge, not its tangent.
                    available=depth/2+travel*far+db/2
                    if not continuous:
                        hook_length=abs(tangent-side*depth/2)+math.pi*radius/2+hook["extension_in"]
                        context["beam_longitudinal_adjustment_in3"]+=count*(hook_length-depth/2)*sp.rebar_area(design.bar_size)
                        checks.append({"rule":"staggered_hook_development","passes":available>=ldh-1e-9,
                                       "group":member.group_id,"face":face,"row":row,
                                       "available_in":available,"required_in":ldh,"stagger_in":stagger})
                        tail_end=z+turn*(radius+hook["extension_in"])
                        checks.append({"rule":"standard_hook_tail_fits_member_depth","passes":
                                       min(z,tail_end)-db/2>=-design.h_in+sp.BEAM_CLEAR_COVER_IN-1e-9 and
                                       max(z,tail_end)+db/2<=-sp.BEAM_CLEAR_COVER_IN+1e-9,
                                       "group":member.group_id,"face":face,"row":row,"tail_end_z_in":tail_end})
                        context["beam_hook_families"].append({"group":member.group_id,"face":face,"row":row,
                            "member_tag":member.member_tag,"tangent_axial_local_in":tangent,
                            "bend_radius_in":radius,"standard_extension_in":hook["extension_in"],
                            "stagger_in":stagger,"available_ldh_in":available,"required_ldh_in":ldh})
                    for k,lane in enumerate(placed):
                        name=f"{axis}/{member.group_id}/{face}/{row}/{k}"
                        def point(a,zz):
                            p=[0.,0.,zz]
                            p[axial]=a
                            p[lateral]=lane-width/2
                            return p
                        if continuous:
                            segments.append(_tube(point(-depth/2,z),point(depth/2,z),db,name))
                            continue
                        segments.append(_tube(point(side*depth/2,z),point(tangent,z),db,name))
                        arc=[point(tangent+travel*radius*math.sin(math.pi/2*j/24),
                                   z+turn*radius*(1-math.cos(math.pi/2*j/24))) for j in range(25)]
                        allowance=radius*(1-math.cos(math.pi/96))
                        segments.extend(_tube(a,b,db,name,allowance=allowance) for a,b in zip(arc,arc[1:]))
                        segments.append(_tube(arc[-1],point(far,z+turn*(radius+hook["extension_in"])),db,name))
                    family+=1
    collisions=[]
    for i,a in enumerate(segments):
        for b in segments[i+1:]:
            if a["id"]==b["id"]:
                continue
            gap=_clearance(a,b)
            if gap<-1e-6:
                pair=tuple(sorted((a["id"],b["id"])))
                if not any(c["bars"]==list(pair) for c in collisions):
                    collisions.append({"bars":list(pair),"clearance_in":gap})
    checks.append({"rule":"actual_beam_bar_segments_nonintersecting","passes":not collisions,"collisions":collisions})
    context["beam_geometry_complete"]=True
    context["slab_mat_context"]={k:v for k,v in mats.items() if k!="segments"}
    context["placement_obstacle_segments"]=mats["segments"]
    context["beam_geometry_complete"]=mats["status"]!="not_evaluated"
    checks.append({"rule":"one_global_slab_mat_phase","passes":mats["status"]=="pass","status":mats["status"]})
    clashes=[]
    for a in segments:
        for b in mats["segments"]:
            if _clearance(a,b)<-1e-6:
                clashes.append([a["id"],b["id"]])
    checks.append({"rule":"actual_beam_and_slab_segments_nonintersecting","passes":not clashes,"collisions":clashes[:30]})
    context["status"]="pass" if all(c["passes"] for c in checks) else "fail"
    if mats["status"]=="not_evaluated":
        context["status"]="not_evaluated"
    context["beam_bar_length_inside_joint_in"]=sum(math.dist(s["start_in"],s["end_in"]) for s in segments)
    return context


def installed_beam_straight_intervals(state,member_tag):
    """Actual horizontal steel intervals for material/section inventories.

    At continuous joints the member owns steel up to its grid endpoint;
    at terminating joints its straight bar ends at the selected hook's
    tangent. Hook curves and tails are never represented as axial steel.
    """
    import hashlib
    from Model import Member_Groups as mg
    from Design.Group_Capacity import joint_inventory,column_clear_heights
    from Design.SMRF_Transitions import column_transition
    from Design.SMRF_Splice_Physical import group_beam_lanes
    member_tag=int(member_tag)
    member=state.member(member_tag)
    out={"status":"not_evaluated","method":"actual_hook_tangent_to_tangent_owned_straight_runs_v1",
         "state_identity":state.identity(),"member_tag":int(member_tag),"bars":[],"end_evidence":[]}
    if member.is_column or mg.active() is None or mg.active().identity()!=state.identity():
        out["reason"]="Installed grouped beam state is required"
        return out
    axis=member.member_type[-1]
    step=sp.BAY_X if axis=="x" else sp.BAY_Y
    start=(member.grid_i if axis=="x" else member.grid_j)*step
    ends=[(member.grid_i,member.grid_j),
          (member.grid_i+(axis=="x"),member.grid_j+(axis=="y"))]
    entries={(e["grid_i"],e["grid_j"]):e for e in joint_inventory(state)["joints"] if e["level"]==member.story_or_floor}
    contexts=[]
    for endpoint in ends:
        entry=entries[endpoint]
        preliminary=None
        if entry["above"] is not None:
            preliminary=column_transition(entry["core"].design,entry["above"].design,fy_ksi=sp.FY_KSI,
                clear_cover_in=sp.COL_CLEAR_COVER_IN,upper_clear_height_in=column_clear_heights(entry["above"])["physical"],
                joint_depth_in=min(b.design.h_in for members in entry["beams"].values() for b in members))
        context=beam_context(entry,preliminary)
        contexts.append(context)
        out["end_evidence"].append({"joint_id":entry["id"],"beam_context_status":context["status"],
                                    "geometry_failures":[c for c in context['beam_geometry_checks'] if c.get('passes') is not True],
                                    "hook_families":[f for f in context["beam_hook_families"] if f["member_tag"]==member_tag]})
    lanes=group_beam_lanes(state)[member.group_id]
    statuses=[lanes["status"]]+[c["status"] for c in contexts]
    out["status"]="fail" if "fail" in statuses else "pass" if all(s=="pass" for s in statuses) else "not_evaluated"
    out['geometry_complete']=bool(lanes['coordinates']) and lanes['status']=='pass' and all(c.get('beam_geometry_complete') is True for c in contexts)
    out['physical_detail_status']=out['status']
    for lane in lanes["coordinates"]:
        interval=[start,start+step]
        for end,context in enumerate(contexts):
            family=next((f for f in context["beam_hook_families"] if f["member_tag"]==member_tag and
                         f["face"]==lane["source_face"] and f["row"]==lane["row"]),None)
            if family is not None:
                interval[end]+=family["tangent_axial_local_in"]
        if interval[1]<=interval[0]:
            out["status"]='fail'
            out['physical_detail_status']='fail'
        out["bars"].append({"id":f"member_{member_tag}/beam/{lane['id']}","source_lane_id":lane["id"],
                            "straight_run_interval_in":interval,
                            "owned_straight_interval_in":[max(start,interval[0]),min(start+step,interval[1])],
                            "member_grid_interval_in":[start,start+step]})
    out["evidence_id"]=hashlib.sha256(json.dumps(out,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return out
