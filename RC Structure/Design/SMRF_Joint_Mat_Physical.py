"""One globally phased slab-mat inventory, clipped to each actual joint."""
from __future__ import annotations

import hashlib
import json
import math

import Structure_Parameters as sp
from Design.SMRF_Transition_Physical import _tube,_clearance

_CACHE={}


def global_mat_placement(state=None):
    from Model import Member_Groups as mg,Member_Bar_Layers as layers
    from Design.Group_Capacity import joint_inventory
    from Design.SMRF_Transitions import bar_coordinates
    from Model.Floor_Footprint import physical_floor
    from Design.SMRF_Cage_Geometry import hook_geometry
    from Design.SMRF_Roof_Anchorage import head_product
    state=state or mg.active()
    layout=(sp.SLAB_REINFORCEMENT or {}).get("layout")
    if sp.SLAB_THICKNESS_IN is None:
        return {"status":"pass","layers":{},"basis":"No slab is present."}
    mats,complete=layers.slab_mats()
    if not complete or not layout:
        return {"status":"not_evaluated","layers":{},"reason":"No complete installed slab mat layout."}
    signature=hashlib.sha256(json.dumps(layout,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    # The phase preference reads the beam lanes, so the lanes are part of the key (2026-11-05).
    from Design.SMRF_Splice_Physical import group_beam_lanes
    beam_lanes=group_beam_lanes(state)
    lane_signature=hashlib.sha256(json.dumps({g:[r.get("status"),[[p.get("x_in"),p.get("y_in")] for p in r.get("coordinates") or []]]
                                              for g,r in beam_lanes.items()},sort_keys=True,default=str).encode()).hexdigest()
    key=(state.identity(),signature,sp.BAR_POSITION_GRID_IN,sp.SLAB_THICKNESS_IN,sp.BAY_X,sp.BAY_Y,sp.STORY_H,sp.FY_KSI,
         sp.ROOF_COLUMN_EXTENSION,sp.BEAM_CLEAR_COVER_IN,sp.COL_CLEAR_COVER_IN,sp.REINFORCEMENT_SPECIFICATION,lane_signature)
    if key in _CACHE:
        return _CACHE[key]
    entries=joint_inventory(state)["joints"]
    result={"method":"global_slab_mat_phase_v3_outer_faces_parallel_lanes","state_identity":state.identity(),"layout_sha256":signature,
            "origin_in":[0.,0.],"floors":{},"layers":{},
            "phase_basis":("One on-grid phase per installed layer for the entire floor; straight mats, no local joint relocation. "
                           "A phase clears every column bar; among those, the first that also clears, where a beam of the "
                           "layer's direction terminates at a joint, every bar lane of that beam whose elevation the layer's "
                           "hooked end sweeps (the beam bar's straight run passes the mat hook axis; parallel bars in separate "
                           "vertical planes cannot touch) is preferred. Without one, the first column-clear phase stands and "
                           "the hook-family trials and collision checks decide.")}
    for floor in range(1,sp.NUM_FLOOR+1):
        footprint=physical_floor(floor,state)
        boundary={}
        for entry in entries:
            if entry["level"]!=floor:
                continue
            for side,bound,coordinate,axis in (("x_minus",0,"grid_i","y"),("x_plus",sp.NUM_BAY_X,"grid_i","y"),
                                               ("y_minus",0,"grid_j","x"),("y_plus",sp.NUM_BAY_Y,"grid_j","x")):
                if entry[coordinate]!=bound:
                    continue
                for member in entry["beams"][axis]:
                    boundary.setdefault(side,[]).append(member.design)
        floor_record={"footprint":footprint,"layers":{}}
        largest_db=max(m["db_in"] for m in mats)
        largest_radius=max((hook_geometry(layout["layers"][m["mat"]]["bar_size"],"development_90")["inside_bend_diameter_in"]+m["db_in"])/2 for m in mats)
        largest_stagger=largest_db+max(1.,largest_db)
        for mat in mats:
            name,axis=mat["mat"],mat["runs_along"]
            cross="y" if axis=="x" else "x"
            db=mat["db_in"]
            hoop={side:max(sp.rebar_diameter(d.stirrup_bar_size) for d in values) for side,values in boundary.items()}
            # The existing corner convention puts the outermost bar just
            # inside its parallel edge beam's hoop, not outside that hoop.
            corner_buffer=largest_stagger+largest_radius+(db+largest_db)/2
            transverse=[footprint[f"{cross}_min_in"]+sp.BEAM_CLEAR_COVER_IN+hoop[f"{cross}_minus"]+db/2+corner_buffer,
                        footprint[f"{cross}_max_in"]-sp.BEAM_CLEAR_COVER_IN-hoop[f"{cross}_plus"]-db/2-corner_buffer]
            stagger=db+max(1.,db) if name.endswith("bottom") else 0.
            ends=[footprint[f"{axis}_min_in"]+sp.BEAM_CLEAR_COVER_IN+hoop[f"{axis}_minus"]+db/2+stagger,
                  footprint[f"{axis}_max_in"]-sp.BEAM_CLEAR_COVER_IN-hoop[f"{axis}_plus"]-db/2-stagger]
            hook=hook_geometry(layout["layers"][name]["bar_size"],"development_90")
            radius=(hook["inside_bend_diameter_in"]+db)/2
            straight=[ends[0]+radius,ends[1]-radius]
            tail=mat["centroid_from_top_in"]+radius+hook["extension_in"]
            tail_limit=min(d.h_in-sp.BEAM_CLEAR_COVER_IN-db/2 for side in (f"{axis}_minus",f"{axis}_plus") for d in boundary[side])
            floor_record["layers"][name]={**mat,"transverse_interval_in":transverse,"hook_vertical_axes_in":ends,
                "straight_run_interval_in":straight,"hook_radius_in":radius,"hook_extension_in":hook["extension_in"],
                "hook_axial_stagger_in":stagger,"development_embedment_adjustment_in":-stagger,
                "corner_crossing_buffer_in":corner_buffer,
                "hook_tail_depth_in":tail,"hook_tail_depth_limit_in":tail_limit,"hook_tail_fits":tail<=tail_limit+1e-6,
                "bar_cut_length_in":straight[1]-straight[0]+math.pi*radius+2*hook["extension_in"]}
        result["floors"][str(floor)]=floor_record
    inventories=[]
    for entry in entries:
        core=entry["core"].design
        moved=layers.joint_inventory(entry["level"],entry["grid_i"],entry["grid_j"])
        bars=(moved or {}).get("bars") if moved else tuple((p["y_in"]+core.h_in/2,p["x_in"]+core.b_in/2,
                 sp.rebar_diameter(core.bar_size)) for p in bar_coordinates(core,sp.COL_CLEAR_COVER_IN))
        if isinstance(bars,str):
            result.update(status="not_evaluated",reason="A joint column transition has no declared physical paths.")
            return result
        inventories.append((entry,core,bars))
    for mat in mats:
        name=mat["mat"]
        layer=layout["layers"][name]
        spacing=float(layer["spacing_in"])
        db=mat["db_in"]
        axis=mat["runs_along"]
        grid=sp.BAR_POSITION_GRID_IN or .5
        trials=max(1,math.ceil(spacing/grid))
        chosen=None;fallback=None
        for trial in range(trials):
            phase=trial*grid
            positions={}
            for floor,record in result["floors"].items():
                lo,hi=record["layers"][name]["transverse_interval_in"]
                positions[floor]=[phase+k*spacing for k in range(math.ceil((lo-phase)/spacing),math.floor((hi-phase)/spacing)+1)]
            collision=False;lane_conflict=False
            for entry,core,bars in inventories:
                origin=entry["grid_j"]*sp.BAY_Y if axis=="x" else entry["grid_i"]*sp.BAY_X
                half=(core.b_in if axis=="x" else core.h_in)/2
                local=[v-origin for v in positions[str(entry["level"])] if abs(v-origin)<=half+db]
                roof_head=(head_product(core.bar_size,sp.FY_KSI,sp.REINFORCEMENT_SPECIFICATION)
                           if entry["above"] is None and not sp.ROOF_COLUMN_EXTENSION else {})
                for item in bars:
                    *plan,col_db=item
                    if roof_head.get("status")=="pass":
                        col_db=max(col_db,roof_head["diameter_in"])
                    x0,y0,x1,y1=plan if len(plan)==4 else (*plan,*plan)
                    a,b=(y0-half,y1-half) if axis=="x" else (x0-half,x1-half)
                    if any(max(min(a,b)-v,v-max(a,b),0.)<(db+col_db)/2-1e-6 for v in local):
                        collision=True
                        break
                if collision:
                    break
                # Parallel beams that terminate here: their hooked bars and the hooked mat ends share this joint. A
                # mat bar in a lane's plane meets that beam bar only where its hooked end (bend and tail) sweeps the
                # lane's elevation.
                members=entry["beams"][axis]
                if not lane_conflict and not (len(members)==2 and not (entry.get("mixed") or {}).get(axis,False)):
                    record=result["floors"][str(entry["level"])]["layers"][name]
                    z_top=-mat["centroid_from_top_in"]
                    z_tail=z_top-record["hook_radius_in"]-record["hook_extension_in"]
                    for member in members:
                        if beam_lanes[member.group_id].get("status")!="pass" or not beam_lanes[member.group_id].get("coordinates"):
                            # Lanes not established: no phase can be called lane-clear for this beam.
                            lane_conflict=True
                            break
                        beam_db=sp.rebar_diameter(member.design.bar_size)
                        reach=(db+beam_db)/2
                        near=[v-origin for v in positions[str(entry["level"])] if abs(v-origin)<=member.design.b_in/2+db]
                        swept=[p["x_in"] for p in beam_lanes[member.group_id].get("coordinates") or []
                               if z_tail-reach+1e-6 < -member.design.h_in/2+p["y_in"] < z_top+reach-1e-6]
                        if any(abs(v-x)<reach-1e-6 for v in near for x in swept):
                            lane_conflict=True
                            break
            if not collision:
                if fallback is None:
                    fallback=(phase,positions)
                if not lane_conflict:
                    chosen=(phase,positions)
                    break
        phase_rule="columns_and_swept_parallel_lanes" if chosen else "columns_only" if fallback else None
        chosen=chosen or fallback
        result["layers"][name]={**mat,"bar_size":layer["bar_size"],"spacing_in":spacing,"phase_trials":trials,
             "status":"pass" if chosen else "fail","phase_in":None if chosen is None else chosen[0],"phase_rule":phase_rule}
        for floor,record in result["floors"].items():
            record["layers"][name].update(global_transverse_coordinates_in=[] if chosen is None else chosen[1][floor],
                phase_in=None if chosen is None else chosen[0],bar_size=layer["bar_size"],spacing_in=spacing)
    result["status"]="pass" if (all(v["status"]=="pass" for v in result["layers"].values()) and
        all(m["hook_tail_fits"] for f in result["floors"].values() for m in f["layers"].values())) else "fail"
    if len(_CACHE)>64:
        _CACHE.clear()
    _CACHE[key]=result
    return result


def joint_mat_segments(entry,placement=None):
    placement=placement or global_mat_placement()
    core=entry["core"].design
    origin=[entry["grid_i"]*sp.BAY_X,entry["grid_j"]*sp.BAY_Y]
    segments=[]
    floor=(placement.get("floors") or {}).get(str(entry["level"]),{})
    for name,mat in floor.get("layers",{}).items():
        axis=mat["runs_along"]
        axial=0 if axis=="x" else 1
        half_axial=(core.h_in if axis=="x" else core.b_in)/2
        half_cross=(core.b_in if axis=="x" else core.h_in)/2
        half_cross=max([half_cross]+[b.design.b_in/2 for b in entry["beams"][axis]])
        half_axial=max([half_axial]+[b.design.b_in/2 for b in entry["beams"]["y" if axis=="x" else "x"]])
        lo=max(mat["straight_run_interval_in"][0]-origin[axial],-half_axial)
        hi=min(mat["straight_run_interval_in"][1]-origin[axial],half_axial)
        if lo>hi:
            continue
        for i,v in enumerate(mat.get("global_transverse_coordinates_in") or ()):
            cross=v-origin[1-axial]
            if abs(cross)>half_cross+mat["db_in"]:
                continue
            z=-mat["centroid_from_top_in"]
            def point(a,zz):
                return [cross,a,zz] if axis=="x" else [a,cross,zz]
            a,b=([cross,lo,z],[cross,hi,z]) if axis=="x" else ([lo,cross,z],[hi,cross,z])
            segments.append(_tube(a,b,mat["db_in"],f"slab/{name}/{i}"))
            for end,travel in ((0,-1),(1,1)):
                far=mat["hook_vertical_axes_in"][end]-origin[axial]
                if abs(far)>half_axial+mat["db_in"]:
                    continue
                radius=mat["hook_radius_in"]
                tangent=far-travel*radius
                arc=[point(tangent+travel*radius*math.sin(math.pi*j/48),z-radius*(1-math.cos(math.pi*j/48))) for j in range(25)]
                allowance=radius*(1-math.cos(math.pi/96))
                segments.extend(_tube(a,b,mat["db_in"],f"slab/{name}/{i}",allowance=allowance) for a,b in zip(arc,arc[1:]))
                segments.append(_tube(arc[-1],point(far,z-radius-mat["hook_extension_in"]),mat["db_in"],f"slab/{name}/{i}"))
    collisions=[]
    for i,a in enumerate(segments):
        for b in segments[i+1:]:
            if a["id"]!=b["id"] and _clearance(a,b)<-1e-6:
                pair=sorted((a["id"],b["id"]))
                if pair not in collisions:
                    collisions.append(pair)
    status="fail" if collisions else placement["status"]
    return {"status":status,"segments":segments,"perimeter_hook_geometry_complete":True,"mat_collisions":collisions,
            "placement":{k:v for k,v in placement.items() if k not in ("layers","floors")},
            "physical_floor":floor.get("footprint"),
            "layer_phases":{n:{k:v for k,v in layer.items() if k!="global_transverse_coordinates_in"} for n,layer in floor.get("layers",{}).items()}}


def mat_quantities(placement):
    """Exact integer bars and full planar hook cut lengths, per actual floor."""
    floors={}
    total=0.
    for floor,record in placement.get("floors",{}).items():
        layers={}
        for name,mat in record["layers"].items():
            count=len(mat["global_transverse_coordinates_in"])
            volume=count*mat["bar_cut_length_in"]*sp.rebar_area(mat["bar_size"])
            layers[name]={"bars":count,"bar_cut_length_in":mat["bar_cut_length_in"],"steel_volume_in3":volume,"phase_in":mat["phase_in"]}
            total+=volume
        floors[floor]={"layers":layers,"concrete_footprint":record["footprint"]}
    return {"status":placement["status"],"steel_volume_in3":total if placement["status"]=="pass" else None,
            "floors":floors,"layout_sha256":placement.get("layout_sha256"),"method":placement.get("method")}


def member_slab_inventory(member,flange,placement=None):
    """Exact installed parallel bars within one member's actual effective flange.

    Each reported bar is a full straight run between the perimeter hook
    tangencies; its end development is evaluated separately at both critical
    datums by Member_Properties.exterior_end_anchorage. Unsupported interior
    cuts are unknown, never assumed to develop. This inventory is also the
    coordinate source for the common slab/web strength field.
    """
    placement=placement or global_mat_placement()
    from Model import Member_Properties as mp
    axis=mp.beam_axis(member)
    center=member.grid_j*sp.BAY_Y if axis=="x" else member.grid_i*sp.BAY_X
    half=member.design.b_in/2
    overhang={r["side"]:r["overhang_in"] for r in flange["overhangs"]}
    band=[center-half-overhang.get("minus",0.),center+half+overhang.get("plus",0.)]
    record={"status":placement["status"],"member_tag":member.member_tag,"axis":axis,"floor":member.story_or_floor,
            "effective_flange_interval_in":band,"layout_sha256":placement.get("layout_sha256"),"bars":[],"layers":{},
            "method":"integer_installed_global_mat_bars_in_actual_flange_v1",
            "development_basis":"full continuous straight run; exterior standard hooks checked at both support and joint faces"}
    mats=placement.get("floors",{}).get(str(member.story_or_floor),{}).get("layers",{})
    for face in ("top","bottom"):
        name=f"{axis}_{face}"
        mat=mats.get(name)
        if mat is None:
            record.update(status="not_evaluated",reason="The actual floor has no complete physical layer inventory.")
            continue
        if mat.get("phase_in") is None:
            record.update(status="fail" if placement["status"]=="fail" else "not_evaluated",
                          reason="No installed global phase exists for this layer.")
            continue
        chosen=[(i,s) for i,s in enumerate(mat["global_transverse_coordinates_in"]) if band[0]-1e-8<=s<=band[1]+1e-8]
        bars=[{"id":f"floor_{member.story_or_floor}/slab/{name}/{i}","owner":"slab","face":face,"axis":axis,
               "bar_size":mat["bar_size"],"area_in2":sp.rebar_area(mat["bar_size"]),"db_in":mat["db_in"],
               "global_transverse_coordinate_in":s,"centroid_from_top_in":mat["centroid_from_top_in"],
               "fy_ksi":sp.SLAB_FY_KSI,"es_ksi":sp.ES_KSI,"straight_run_interval_in":mat["straight_run_interval_in"],
               "hook_vertical_axes_in":mat["hook_vertical_axes_in"],"hook_axial_stagger_in":mat["hook_axial_stagger_in"],
               "interior_cut_or_splice":False} for i,s in chosen]
        record["bars"].extend(bars)
        record["layers"][face]={"count":len(bars),"area_in2":sum(b["area_in2"] for b in bars),
                                  "depth_in":mat["centroid_from_top_in"],"phase_in":mat["phase_in"],
                                  "global_transverse_coordinates_in":[s for _,s in chosen]}
    return record
