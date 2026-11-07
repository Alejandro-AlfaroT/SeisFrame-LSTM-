"""Physical ordinary splice inventory; same beam lanes at every joint.

Local fabrication details preserve the installed nominal Av/s. Published
devices may be selected instead of geometrically undeclared contact laps;
no bar or section proposal is removed from its production catalog.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import replace

import Structure_Parameters as sp
from Design import SMRF_Transition_Physical as ph
from Design.SMRF_Coupler_Catalog import lookup_coupler
from Design.SMRF_Cage_Layout import _describe,_cage_checks

_LANES={}


def group_beam_lanes(state=None):
    from Model import Member_Groups as mg,Member_Bar_Layers as layers
    from Design.Group_Capacity import joint_inventory
    from Design.SMRF_Cage_Geometry import thread_bars,clear_spacing_limit
    from Design.SMRF_Transitions import bar_coordinates
    state=state or mg.active()
    key=(state.identity(),sp.COL_CLEAR_COVER_IN,sp.BEAM_CLEAR_COVER_IN,sp.AGGREGATE_MAX_SIZE_IN,
         sp.BAR_POSITION_GRID_IN,sp.BAR_LAYOUT_HOOP_DB_IN,sp.SLAB_THICKNESS_IN,sp.BAY_X,sp.BAY_Y,sp.STORY_H,sp.FY_KSI,
         sp.BEAM_BAR_LAYER_ORDER,json.dumps((sp.SLAB_REINFORCEMENT or {}).get("layout"),sort_keys=True))
    if key in _LANES:
        return _LANES[key]
    obstacles={gid:[] for gid,d in state.designs.items() if not d.is_column}
    missing={gid:[] for gid in obstacles}
    for kind in joint_inventory(state)["types"].values():
        entry=kind["representative"]
        core=entry["core"].design
        moved=layers.joint_inventory(entry["level"],entry["grid_i"],entry["grid_j"])
        bars=(moved or {}).get("bars") if moved else tuple((p["y_in"]+core.h_in/2,p["x_in"]+core.b_in/2,
             sp.rebar_diameter(core.bar_size)) for p in bar_coordinates(core,sp.COL_CLEAR_COVER_IN))
        for axis,members in entry["beams"].items():
            for member in members:
                gid=member.group_id
                if isinstance(bars,str):
                    missing[gid].append(entry["id"])
                    continue
                shift=((core.b_in if axis=="x" else core.h_in)-member.design.b_in)/2
                for bar in bars:
                    *plan,db=bar
                    x0,y0,x1,y1=plan if len(plan)==4 else (*plan,*plan)
                    a,b=(y0,y1) if axis=="x" else (x0,x1)
                    obstacles[gid].append((min(a,b)-shift,max(a,b)-shift,db))
    result={}
    for gid,items in obstacles.items():
        design=state.designs[gid]
        rows=layers.group_rows(gid)
        maximum=max(n for face in ("top","bottom") for n in rows[face]["per_layer"])
        db=sp.rebar_diameter(design.bar_size)
        first=sp.first_bar_offset_in(sp.BEAM_CLEAR_COVER_IN,sp.rebar_diameter(design.stirrup_bar_size),db)
        unique=sorted(set(items))
        pool=thread_bars(maximum,db,first,design.b_in-first,[(a,b) for a,b,_ in unique],
                         [d for _,_,d in unique],clear_spacing_limit("beam",db,sp.AGGREGATE_MAX_SIZE_IN))
        points=[]
        for face in ("top","bottom"):
            for row,(count,offset) in enumerate(zip(rows[face]["per_layer"],rows[face]["offsets_in"])):
                # The smaller row is a subset including the same extreme
                # lanes. All candidates share one straight beam-bar axis.
                indices=([round(k*(len(pool)-1)/(count-1)) for k in range(count)] if count>1 else [len(pool)//2]) if pool else []
                for index,i in enumerate(indices):
                    points.append({"id":f"{face}_{row}_{index}","face":face if row==0 else "interior",
                        "row":row,"source_face":face,"index":index,"x_in":pool[i]-design.b_in/2,
                        "y_in":design.h_in/2-offset if face=="top" else -design.h_in/2+offset})
        result[gid]={"method":"one_common_straight_lane_inventory_v1","status":"not_evaluated" if missing[gid] else
                     "pass" if len(pool)==maximum else "fail","missing_transition_contexts":missing[gid],
                     "coordinates":points,"pool_from_left_in":pool,"requested_maximum":maximum,
                     "distinct_joint_obstacle_spans":len(unique),"rows":rows}
    if len(_LANES)>64:
        _LANES.clear()
    _LANES[key]=result
    return result


def beam_cage_geometry(design,coordinates,clear_cover_in,legs=None,perimeter_cover_in=None):
    """Actual rectangular hoop around shared extreme beam lanes and row depths.

    Crossties join common top/bottom lanes, so their shear projection is
    vertical. At most 4096 support subsets are considered; no diagonal leg
    is credited as vertical reinforcement.
    """
    top=sorted((p for p in coordinates if p["face"]=="top"),key=lambda p:p["x_in"])
    bottom=sorted((p for p in coordinates if p["face"]=="bottom"),key=lambda p:p["x_in"])
    count=design.stirrup_legs if legs is None else int(legs)
    bad={"passes":False,"checks":[{"rule":"beam_hoop_actual_supported_bar_pattern","passes":False}],"segments":[],"total_length_in":0.}
    if min(len(top),len(bottom))<2 or count>min(len(top),len(bottom)):
        return bad
    if any(abs(top[k]["x_in"]-bottom[k]["x_in"])>ph.TOL for k in (0,-1)):
        return bad
    db=sp.rebar_diameter(design.bar_size)
    shared=sorted({p["x_in"] for p in top}&{p["x_in"] for p in bottom})
    supports=None
    for chosen in itertools.islice(itertools.combinations(shared[1:-1],count-2),4096):
        proposed=[shared[0],*chosen,shared[-1]]
        faces={name:_describe([p["x_in"] for p in bars],db,[p["x_in"] in proposed for p in bars])
               for name,bars in (("top",top),("bottom",bottom))}
        if all(c["passes"] for c in _cage_checks(faces,14.,"beam")):
            supports=proposed
            break
    if supports is None:
        return bad
    center=[(top[0]["x_in"]+top[-1]["x_in"])/2,(top[0]["y_in"]+bottom[0]["y_in"])/2]
    translated=[{**p,"x_in":p["x_in"]-center[0],"y_in":p["y_in"]-center[1]} for p in coordinates]
    synthetic=replace(design,member_type="column",top_bars=len(top),bot_bars=len(top),side_bars=0,
                      stirrup_legs=max(count,2),stirrup_legs_by_direction=(count,2))
    cage=ph.rectangular_cage_geometry(synthetic,translated,0.,support_positions_in=[x-center[0] for x in supports],
                                    perimeter_cover_in=perimeter_cover_in)
    if not cage.get("segments"):
        return cage
    for segment in cage["segments"]:
        for end in ("start_in","end_in"):
            for k in (0,1):
                segment[end][k]+=center[k]
    cover=min(min(design.b_in/2-abs(p[0]),design.h_in/2-abs(p[1]))-
              (s["diameter_in"]/2-s.get("chord_enclosure_allowance_in",0.))
              for s in cage["segments"] for p in (s["start_in"],s["end_in"]))
    cage["checks"].append({"rule":"beam_actual_hoop_cover","passes":cover>=clear_cover_in-ph.TOL,
                            "provided_in":cover,"required_in":clear_cover_in})
    cage["passes"]=cage["passes"] and cover>=clear_cover_in-ph.TOL
    cage["actual_cover_in"]=cover
    cage["alternation_center_in"]=center
    cage["support_lanes_in"]=supports
    cage["vertical_leg_equivalent"]=(count-2)+2*cage["hoop_leg_horizontal_projection_factor"]
    return cage


def verify_beam_mechanical_splice(design,coordinates,*,clear_span_in,spacing_in,fy_ksi,specification,
                                  clear_cover_in,aggregate_in=.75):
    product=lookup_coupler(design.bar_size,design.bar_size,fy_ksi,specification,False)
    out={"method":"beam_A2_actual_rows_and_transverse_v1","status":product["status"],"product":product,
         "checks":[],"bodies":[],"coordinates":coordinates,"additional_transverse_steel_volume_in3":0.}
    if product["status"]!="pass":
        return out
    checks=out["checks"]
    def check(rule,passes,detail):
        checks.append({"rule":rule,"passes":bool(passes),"status":"pass" if passes else "fail","detail":detail})
    diameter,length=product["envelope_diameter_in"],product["length_in"]
    points=[[p["x_in"],p["y_in"]] for p in coordinates]
    db=sp.rebar_diameter(design.bar_size)
    cover=min((min(design.b_in/2-abs(x),design.h_in/2-abs(y))-diameter/2 for x,y in points),default=-math.inf)
    check("beam_splice_device_cover",cover>=clear_cover_in-ph.TOL,{"provided_in":cover,"required_in":clear_cover_in})
    minimum=max(1.,db,4*aggregate_in/3)
    clear=min((math.dist(a,b)-diameter for i,a in enumerate(points) for b in points[i+1:]),default=-math.inf)
    check("beam_splice_device_spacing",clear>=minimum-ph.TOL,{"provided_in":clear,"required_in":minimum})
    base=beam_cage_geometry(design,coordinates,clear_cover_in)
    check("beam_original_transverse_geometry",base["passes"],base["checks"])
    station=clear_span_in/2
    # All local devices and changed hoops are outside the 2h hinge zones,
    # including Grade60; this is a conservative construction choice.
    protected=2*design.h_in
    nominal_av=design.stirrup_legs*sp.rebar_area(design.stirrup_bar_size)
    options=[]
    for bar in sorted(set([int(design.stirrup_bar_size),4,5])):
        if bar<design.stirrup_bar_size:
            continue
        for legs in range(design.stirrup_legs,min(sum(p["face"]=="top" for p in coordinates),sum(p["face"]=="bottom" for p in coordinates))+1):
            candidate=replace(design,stirrup_bar_size=bar,stirrup_legs=legs)
            cage=beam_cage_geometry(candidate,coordinates,clear_cover_in)
            if not cage["passes"]:
                continue
            av=cage["vertical_leg_equivalent"]*sp.rebar_area(bar)
            # 18.6.4.6 outside hinge zones: spacing <= d/2. The local
            # Av/s is at least the original nominal design value.
            effective=min(design.h_in/2-p["y_in"] for p in coordinates if p["source_face"]=="bottom")
            maximum=min(effective/2,spacing_in*av/nominal_av)
            gap=math.floor(maximum+1e-9)
            if gap<spacing_in-ph.TOL or gap<length+sp.rebar_diameter(bar)-ph.TOL:
                continue
            if ph.transverse_collisions(ph._positioned_cage_segments(cage,0,0),ph._positioned_cage_segments(cage,gap,1)):
                continue
            options.append((cage["total_length_in"]*sp.rebar_area(bar)/gap,bar,legs,gap,cage,av))
    chosen=None
    for _,bar,legs,gap,cage,av in sorted(options,key=lambda v:v[:4]):
        transverse=[s for i in range(-3,4) for s in ph._positioned_cage_segments(cage,station+i*gap,i)]
        bodies=[]
        for i,xy in enumerate(points):
            body=None
            for delta in sorted([(-.5+k/32)*gap for k in range(33)],key=lambda x:(abs(x),x)):
                interval=[station+delta-length/2,station+delta+length/2]
                if interval[0]<protected or interval[1]>clear_span_in-protected:
                    continue
                if any(ph.segment_cylinder_clearance(t,xy,interval,diameter/2)<-ph.TOL for t in transverse):
                    continue
                body={"id":f"beam_coupler_{i}","geometry":"finite_longitudinal_cylinder",
                      "center_xy_in":xy,"z_interval_in":interval,"diameter_in":diameter,"stagger_in":delta,
                      "part_number":product["part_number"]}
                break
            if body is None:
                break
            bodies.append(body)
        if len(bodies)!=len(points):
            continue
        first=math.floor((min(b["z_interval_in"][0] for b in bodies)-station)/gap)
        last=math.ceil((max(b["z_interval_in"][1] for b in bodies)-station)/gap)
        positions=[station+k*gap for k in range(first,last+1)]
        zone=[positions[0]-sp.rebar_diameter(bar)/2,positions[-1]+cage["set_height_in"]+sp.rebar_diameter(bar)/2]
        if zone[0]<protected or zone[1]>clear_span_in-protected:
            continue
        if base["passes"] and (ph.transverse_collisions(ph._positioned_cage_segments(base,positions[0]-spacing_in,0),ph._positioned_cage_segments(cage,positions[0],first)) or
            ph.transverse_collisions(ph._positioned_cage_segments(cage,positions[-1],last),ph._positioned_cage_segments(base,positions[-1]+spacing_in,0))):
            continue
        chosen=(bar,legs,gap,cage,av,bodies,positions,zone)
        break
    check("beam_splice_device_and_hoop_layout",chosen is not None,{"candidates":len(options),"protected_end_in":protected,
           "stagger_trials_per_device":33,"original_nominal_Av_per_in":nominal_av/spacing_in})
    if chosen:
        bar,legs,gap,cage,av,bodies,positions,zone=chosen
        out.update(bodies=bodies,hoop_stations_in=positions,transverse_detail={"bar_size":bar,"legs":legs,
            "spacing_in":gap,"zone_in":zone,"set_transverse_length_in":cage["total_length_in"],
            "provided_Av_per_in":av/gap,"original_Av_per_in":nominal_av/spacing_in})
        removed=max(0,math.floor((positions[-1]-station)/spacing_in)-math.ceil((positions[0]-station)/spacing_in)+1)
        base_volume=base.get("total_length_in",0)*sp.rebar_area(design.stirrup_bar_size)
        out["additional_transverse_steel_volume_in3"]=max(0.,len(positions)*cage["total_length_in"]*sp.rebar_area(bar)+(2-removed)*base_volume)
    out["device_count"]=len(points)
    out["solid_envelope_steel_volume_in3"]=len(points)*math.pi*diameter**2/4*length
    out["baseline_bar_volume_replaced_in3"]=len(points)*length*sp.rebar_area(design.bar_size)
    out["net_envelope_steel_volume_in3"]=max(0.,out["solid_envelope_steel_volume_in3"]-out["baseline_bar_volume_replaced_in3"])
    out["quantity_basis"]="solid circumscribed device-envelope upper bound; no product weight asserted"
    out["status"]="pass" if all(c["passes"] for c in checks) else "fail"
    return out
