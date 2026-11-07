"""Explicit nominal-envelope detailing checks for declared column transitions.

Coordinates are inches, x along b, y along h, z=0 at joint top. No analysis
engine is used. Evidence absence and a demonstrated geometric failure have
different statuses. These checks cannot qualify an unpublished splice.
"""
from __future__ import annotations

import math

from Design.SMRF_Coupler_Catalog import lookup_coupler

METHOD = "column_transition_physical_v2_rigid_fabrication_configurations"
TOL = 1.e-6
_DB = {3:.375, 4:.5, 5:.625, 6:.75, 7:.875, 8:1., 9:1.128, 10:1.27, 11:1.41, 14:1.693, 18:2.257}


def segment_distance(a, b, c, d):
    """Shortest distance between two finite 3D segments, including points."""
    u, v, w = [[q-p for p,q in zip(x,y)] for x,y in ((a,b),(c,d),(c,a))]
    dot = lambda x,y: sum(p*q for p,q in zip(x,y))
    aa, bb, cc, dd, ee = dot(u,u), dot(u,v), dot(v,v), dot(u,w), dot(v,w)
    if aa < 1.e-24 and cc < 1.e-24:
        return math.dist(a,c)
    if aa < 1.e-24:
        s, t = 0., max(0., min(1., ee/cc))
    elif cc < 1.e-24:
        s, t = max(0., min(1., -dd/aa)), 0.
    else:
        den = aa*cc-bb*bb
        s = max(0., min(1., (bb*ee-cc*dd)/den)) if den > 1.e-24 else 0.
        t = (bb*s+ee)/cc
        if t < 0.:
            t, s = 0., max(0., min(1., -dd/aa))
        elif t > 1.:
            t, s = 1., max(0., min(1., (bb-dd)/aa))
    return math.sqrt(sum((w[k]+s*u[k]-t*v[k])**2 for k in range(3)))


def _tube(a, b, diameter, name, owner=(), allowance=0.):
    return {"start_in":list(a), "end_in":list(b), "diameter_in":diameter+2*allowance,
            "id":name, "owners":list(owner),"chord_enclosure_allowance_in":allowance,
            "cut_length_upper_in":math.dist(a,b)}


def _clearance(a, b):
    radius = (a["diameter_in"]+b["diameter_in"])/2
    # Exact bounding-box separation avoids most 3D distance work.
    if any(max(min(a["start_in"][k],a["end_in"][k])-max(b["start_in"][k],b["end_in"][k]),
               min(b["start_in"][k],b["end_in"][k])-max(a["start_in"][k],a["end_in"][k])) > radius
           for k in range(3)):
        return math.inf
    return segment_distance(a["start_in"],a["end_in"],b["start_in"],b["end_in"])-radius


def offset_bend_geometry(offset_in, bar_size, height_in=None):
    """Tangent circular bends joined by a straight segment, slope <= 1:6.

    The Table 25.3.1 standard-hook bend diameter is deliberately adopted as
    a conservative fabrication radius for these shallow offset bends; that
    table is not misrepresented as an explicit offset-bend requirement.
    """
    delta, db = float(offset_in), _DB[int(bar_size)]
    if delta < 0 or not math.isfinite(delta):
        raise ValueError("Offset must be nonnegative and finite.")
    inside = db * (6 if bar_size <= 8 else 8 if bar_size <= 11 else 10)
    radius, max_angle = (inside+db)/2, math.atan(1/6)
    if delta <= TOL:
        return {"height_in":float(height_in or 0), "angle_rad":0., "radius_in":radius,
                "inside_bend_diameter_in":inside, "straight_in":float(height_in or 0), "offset_in":delta}
    angle = min(max_angle, math.acos(max(-1.,1-delta/(2*radius))))
    straight = max(0., (delta-2*radius*(1-math.cos(angle)))/math.sin(angle))
    minimum = 2*radius*math.sin(angle)+straight*math.cos(angle)
    height = minimum if height_in is None else float(height_in)
    if height < minimum-TOL:
        return {"height_in":height, "minimum_height_in":minimum, "fits":False,
                "radius_in":radius,"inside_bend_diameter_in":inside,"offset_in":delta}
    lo, hi = 0., angle
    for _ in range(60):
        mid = (lo+hi)/2
        displacement = height*math.tan(mid)-2*radius*(1/math.cos(mid)-1)
        if displacement < delta:
            lo = mid
        else:
            hi = mid
    angle = (lo+hi)/2
    straight = max(0.,(height-2*radius*math.sin(angle))/math.cos(angle))
    return {"height_in":height,"minimum_height_in":minimum,"fits":True,"angle_rad":angle,
            "radius_in":radius,"inside_bend_diameter_in":inside,"straight_in":straight,"offset_in":delta,
            "slope":math.tan(angle),"radius_basis":"declared conservative Table 25.3.1 fabrication diameter"}


def offset_path_segments(path, bar_size, height_in, joint_depth_in):
    """Conservative tubes enclosing a tangent-arc centerline (16 chords/arc)."""
    origin, goal = path["lower_xy_in"], path["target_xy_in"]
    delta, db = math.dist(origin,goal), _DB[int(bar_size)]
    name = path.get("id",f"{path['face']}_{path['lower_index']}")
    geometry = offset_bend_geometry(delta,bar_size,height_in)
    if delta <= TOL:
        return [_tube([*origin,-joint_depth_in],[*goal,0],db,name)], geometry
    if not geometry["fits"]:
        return [], geometry
    r,theta,length = geometry["radius_in"],geometry["angle_rad"],geometry["straight_in"]
    vector = [(goal[k]-origin[k])/delta for k in (0,1)]
    points = [(0.,-joint_depth_in),(0.,-height_in)]
    points += [(r*(1-math.cos(theta*j/16)),-height_in+r*math.sin(theta*j/16)) for j in range(1,17)]
    x1,z1 = points[-1]
    x2,z2 = x1+length*math.sin(theta),z1+length*math.cos(theta)
    points.append((x2,z2))
    points += [(x2+r*(math.cos(theta*(1-j/16))-math.cos(theta)),
                z2+r*(math.sin(theta)-math.sin(theta*(1-j/16)))) for j in range(1,17)]
    xyz = [[origin[0]+vector[0]*x,origin[1]+vector[1]*x,z] for x,z in points]
    sagitta = r*(1-math.cos(theta/32))
    segments = [_tube(a,b,db,name,allowance=sagitta) for a,b in zip(xyz,xyz[1:]) if math.dist(a,b)>TOL]
    geometry["chord_enclosure_allowance_in"] = sagitta
    return segments, geometry


def lap_group_spacing(paths, lower_db, upper_db, aggregate_in=.75):
    """25.2.3/25.5.1.2 spacing between contact-lap groups and adjacent bars.

    The two partners in one lap are not erroneously tested as two unrelated
    column bars. Their own nonintersection and 25.5.1.3 limit are separate.
    """
    bars = []
    for i,p in enumerate(paths):
        if p["path"] in ("lap_spliced","terminated"):
            bars.append((i,"lower",p["target_xy_in"],lower_db))
        if p["upper_xy_in"] is not None:
            bars.append((i,"upper",p["upper_xy_in"],upper_db))
    failures, minimum = [], None
    for j,a in enumerate(bars):
        for b in bars[j+1:]:
            clear = math.dist(a[2],b[2])-(a[3]+b[3])/2
            if a[0] == b[0] and abs(clear)<=TOL:
                continue  # The explicit contact-lap exception, 25.5.1.2.
            limit = max(1.5,1.5*max(a[3],b[3]),4*aggregate_in/3)
            entry = {"bars":[f"{a[1]}_{a[0]}",f"{b[1]}_{b[0]}"],"clear_in":clear,"required_in":limit}
            if minimum is None or clear-limit < minimum["clear_in"]-minimum["required_in"]:
                minimum = entry
            if clear < limit-TOL:
                failures.append(entry)
    return {"passes":not failures,"minimum_margin_pair":minimum,"failures":failures,
            "basis":"25.2.3 column spacing; 25.5.1.2 contact-pair exception. Conservative column spacing retained between lap groups (25.5.1.2 permits 25.2.1)."}


def _arc(center,radius,start,stop,z,diameter,name,owner):
    count = max(8,math.ceil(abs(stop-start)/(math.pi/32)))
    points = [[center[0]+radius*math.cos(start+(stop-start)*k/count),
               center[1]+radius*math.sin(start+(stop-start)*k/count),z] for k in range(count+1)]
    allowance = radius*(1-math.cos(abs(stop-start)/count/2))
    return [{**_tube(a,b,diameter,name,owner,allowance),"cut_length_upper_in":radius*abs(stop-start)/count}
            for a,b in zip(points,points[1:])],points


def rectangular_cage_geometry(design, coordinates, clear_cover_in, z=0., high_axial=False, offset_support=False,
                              bar_envelope_db_in=None,perimeter_cover_in=None,support_positions_in=None):
    """Draw the selected rounded perimeter hoop and alternating 135/90 crossties.

    Orthogonal crossties occupy separate elevations, so coincident crossing
    bars are never accepted as a realizable tie set. Hoop-corner and
    crosstie ownership is explicit. Hook extensions use Table 25.3.2 and
    25.3.4; the perimeter closure hooks are conservatively enveloped inside
    the corner with their two end tails at separate elevations.
    """
    from Design.SMRF_Cage_Layout import column_cage,arrangement_with_crossties,_cage_checks
    coordinates=[{**p,"id":p.get("id",f"{p['face']}_{p['index']}")} for p in coordinates]
    db,dt = _DB[int(design.bar_size)],_DB[int(design.stirrup_bar_size)]
    db=max(db,float(bar_envelope_db_in or db))
    legs = design.stirrup_legs_by_direction or (design.stirrup_legs,)*2
    if offset_support:
        legs=(design.top_bars,design.side_bars+2)
    layout = column_cage(design.b_in,design.h_in,clear_cover_in,dt,db,design.top_bars,design.side_bars,
                         high_axial=high_axial,legs={"across_b_face":legs[0],"across_h_face":legs[1]})
    pitch=dt+.125
    # Actual intermediate coordinates can differ from the section's grid
    # coordinates. Crossties follow those bars, rather than empty grid points.
    top=sorted((p for p in coordinates if p["face"]=="top"),key=lambda p:p["x_in"])
    left=sorted((p for p in coordinates if p["face"]=="left"),key=lambda p:p["y_in"])
    face_positions={"b_face":[p["x_in"]+design.b_in/2 for p in top],
                    "h_face":[-top[-1]["y_in"]+design.h_in/2]+[p["y_in"]+design.h_in/2 for p in left]+[top[-1]["y_in"]+design.h_in/2]}
    try:
        layout["arrangement"]={face:arrangement_with_crossties(pos,db,legs[i]-2,layout["hx_max_in"],high_axial)
                               for i,(face,pos) in enumerate(face_positions.items())}
        layout["checks"]=_cage_checks(layout["arrangement"],layout["hx_max_in"],"column")
        layout["hx_in"]=max(a["hx_in"] for a in layout["arrangement"].values())
        layout["constructible"]=all(c["passes"] for c in layout["checks"])
        layout["faces"]=face_positions
        layout["legs"]={"across_b_face":legs[0],"across_h_face":legs[1]}
        if support_positions_in is not None:
            from Design.SMRF_Cage_Layout import _describe
            realized={}
            for face in ("top","bottom"):
                values=sorted(p["x_in"] for p in coordinates if p["face"]==face)
                realized[face]=_describe(values,db,[any(abs(v-q)<TOL for q in support_positions_in) for v in values])
            layout["arrangement"]["b_face"]={**realized["top"],"positions_in":[v+design.b_in/2 for v in realized["top"]["positions_in"]]}
            layout["checks"]=_cage_checks(realized,14.,"beam")
            layout["constructible"]=all(c["passes"] for c in layout["checks"])
    except ValueError:
        layout["constructible"]=False
    out = {"passes":bool(layout["constructible"]),"layout":layout,"segments":[],"ownership":{},"engagement_envelopes":[],
           "checks":[],"set_height_in":3*pitch,"bar_envelope_basis":"ACI nominal equivalent ASTM diameters",
           "hoop_form":"one-turn continuously wound rectangular hoop; seismic hooks both ends (ACI 2.3 hoop definition)",
           "hoop_pitch_in":pitch,"longitudinal_coordinates":coordinates}
    if not layout["constructible"]:
        out["checks"].append({"rule":"selected_crossties_constructible","passes":False})
        return out
    xmax,ymax=top[-1]["x_in"],top[-1]["y_in"]
    minimum_ri=max((2 if dt<=.625 else 3)*dt,db/2)
    # A small explicit radius allowance bounds the additional 3D curvature
    # of a smooth zero-end-slope winding pitch (verified below).
    ri=minimum_ri+.001
    if perimeter_cover_in is not None:
        # A larger mandrel can put the hoop at its specified cover while
        # keeping the same corner bars. Useful around known coupler bodies;
        # no longitudinal bar is relocated and no minimum radius reduced.
        first=min(design.b_in/2-xmax,design.h_in/2-ymax)
        candidate=(first-float(perimeter_cover_in)-dt-db/(2*math.sqrt(2)))/(1-1/math.sqrt(2))
        ri=max(ri,candidate)
    # Make a minimum-radius hoop tangent to the actual diagonal corner bar.
    diagonal = (ri-db/2)/math.sqrt(2)
    r = ri+dt/2
    # Corner centres lie INWARD from the corner bar for a containing hoop.
    cx,cy = xmax-diagonal,ymax-diagonal
    cover_b,cover_h = design.b_in/2-(cx+r+dt/2),design.h_in/2-(cy+r+dt/2)
    valid = min(cover_b,cover_h)>=clear_cover_in-TOL and min(cx,cy)>0
    out.update(inside_bend_radius_in=ri,actual_cover_in=[cover_b,cover_h])
    out["core_outside_dimensions_in"]=[2*(cx+ri+dt),2*(cy+ri+dt)]
    out["core_area_in2"]=(2*(cx+ri+dt))*(2*(cy+ri+dt))-(4-math.pi)*(ri+dt)**2
    out["checks"].append({"rule":"rounded_hoop_corner_cover_and_containment","passes":valid,
                          "inside_radius_in":ri,"corner_clearance_in":ri-(math.sqrt(2)*diagonal+db/2)})
    segments = out["segments"]
    corner_ids = {}
    for sx,sy,start in ((1,1,0),(-1,1,math.pi/2),(-1,-1,math.pi),(1,-1,3*math.pi/2)):
        owner = next((p["id"] for p in coordinates if abs(p["x_in"]-sx*xmax)<TOL and abs(p["y_in"]-sy*ymax)<TOL),None)
        corner_ids[sx,sy] = owner
        arcs,_ = _arc((sx*cx,sy*cy),r,start,start+math.pi/2,z,dt,"perimeter_corner",[owner])
        segments.extend(arcs)
        out["ownership"][owner] = ["perimeter_hoop_corner"]
        out["engagement_envelopes"].append({"owner":owner,"center_in":[sx*cx,sy*cy],
                                             "inside_radius_in":ri,"angle_start_rad":start,
                                             "angle_stop_rad":start+math.pi/2,"axes":["x","y"],
                                             "transverse_bar_id":"perimeter_winding","z_bounds_in":[z,z+pitch]})
    segments += [_tube([-cx,cy+r,z],[cx,cy+r,z],dt,"perimeter_top"),
                 _tube([-cx,-cy-r,z],[cx,-cy-r,z],dt,"perimeter_bottom"),
                 _tube([cx+r,-cy,z],[cx+r,cy,z],dt,"perimeter_right"),
                 _tube([-cx-r,-cy,z],[-cx-r,cy,z],dt,"perimeter_left")]
    extension = max(6*dt,3.)
    # Replace the zero-pitch closed sketch by ONE CONNECTED bar with a
    # one-diameter-plus-1/8-in pitch. Smoothstep z(s) has zero derivative at
    # BOTH ends, joining each planar hook with a continuous 3D tangent.
    ring=[]
    for sx,sy,a,b in ((1,1,math.pi/4,math.pi/2),(-1,1,math.pi/2,math.pi),
                      (-1,-1,math.pi,3*math.pi/2),(1,-1,3*math.pi/2,2*math.pi),
                      (1,1,0,math.pi/4)):
        arc,_=_arc((sx*cx,sy*cy),r,a,b,z,dt,"perimeter_wound_body",[corner_ids[sx,sy]])
        if ring:
            ring.append(_tube(ring[-1]["end_in"],arc[0]["start_in"],dt,"perimeter_wound_body"))
        ring+=arc
    total=sum(math.dist(s["start_in"],s["end_in"]) for s in ring)
    traversed=0.
    for tube in ring:
        length=math.dist(tube["start_in"],tube["end_in"])
        u=traversed/total
        tube["start_in"][2]=z+pitch*(3*u*u-2*u*u*u)
        traversed+=length
        u=traversed/total
        tube["end_in"][2]=z+pitch*(3*u*u-2*u*u*u)
        vertical_chord_error=(6*pitch/total**2)*length**2/8
        tube["vertical_chord_enclosure_in"]=vertical_chord_error
        tube["diameter_in"]+=2*vertical_chord_error
        tube["chord_enclosure_allowance_in"]+=vertical_chord_error
        tube["cut_length_upper_in"]*=math.sqrt(1+(1.5*pitch/total)**2)
    segments[:]=ring
    for stop,level in ((-math.pi/2,z),(math.pi,z+pitch)):
        hook,pts=_arc((cx,cy),r,math.pi/4,stop,level,dt,"perimeter_135_hook",[corner_ids[1,1]])
        segments+=hook
        end=pts[-1]
        sign=1 if stop>math.pi/4 else -1
        segments.append(_tube(end,[end[0]-sign*math.sin(stop)*extension,
                                   end[1]+sign*math.cos(stop)*extension,level],dt,"perimeter_135_tail",[corner_ids[1,1]]))
    out["hoop_leg_horizontal_projection_factor"] = 1/math.sqrt(1+(1.5*pitch/total)**2)
    radius_bound=1/math.sqrt(1/r**2+(6*pitch/total**2)**2)
    out["checks"].append({"rule":"continuous_winding_3d_bend_radius","passes":radius_bound>=minimum_ri+dt/2-TOL,
                          "minimum_centerline_radius_bound_in":radius_bound,"required_in":minimum_ri+dt/2,
                          "pitch_end_slopes":[0.,0.]})
    for tube in segments:
        tube["physical_bar_id"]="perimeter_winding"
    perimeter_ri=ri
    ri=minimum_ri
    r=ri+dt/2  # planar crossties need no winding-curvature allowance
    for direction,face,key,axis,plane in (("y","b_face","x_in",0,z+2*pitch),
                                         ("x","h_face","y_in",1,z+3*pitch)):
        pattern=layout["arrangement"][face]
        dimension=design.b_in if axis==0 else design.h_in
        for pos,supported in zip(pattern["positions_in"][1:-1],pattern["supported"][1:-1]):
            if not supported:
                continue
            coordinate=pos-dimension/2
            faces=("top","bottom") if axis==0 else ("left","right")
            owners=[p for p in coordinates if abs(p[key]-coordinate)<TOL and p["face"] in faces]
            if len(owners)!=2:
                valid=False
                continue
            owners.sort(key=lambda p:p["y_in"] if axis==0 else p["x_in"])
            ids=[p["id"] for p in owners]
            for pid in ids:
                out["ownership"].setdefault(pid,[]).append(f"crosstie_{direction}_135_90_alternating")
            lower,upper=owners[0],owners[1]
            # Local u across the tie and v along its tension leg. The
            # straight leg is tangent to both owned longitudinal cylinders.
            u0=coordinate
            v0=(lower["y_in"] if axis==0 else lower["x_in"])+(ri-db/2)
            v1=(upper["y_in"] if axis==0 else upper["x_in"])-(ri-db/2)
            local=[_tube([u0+r,v0,plane],[u0+r,v1,plane],dt,f"crosstie_{direction}",ids)]
            for v,start,stop in ((v1,0,3*math.pi/4),(v0,0,-math.pi/2)):
                hook_owner=ids[1] if v==v1 else ids[0]
                arcs,pts=_arc((u0,v),r,start,stop,plane,dt,f"crosstie_{direction}_hook",[hook_owner])
                local+=arcs
                sign=1 if stop>start else -1
                end=pts[-1]
                local.append(_tube(end,[end[0]-sign*math.sin(stop)*extension,
                                         end[1]+sign*math.cos(stop)*extension,plane],dt,f"crosstie_{direction}_tail",[hook_owner]))
                center=[u0,v] if axis==0 else [v,u0]
                out["engagement_envelopes"].append({"owner":ids[1] if v==v1 else ids[0],
                    "center_in":center,"inside_radius_in":ri,"angle_start_rad":start,
                    "angle_stop_rad":stop,"swap_axes":axis==1,"axes":[direction],
                    "transverse_bar_id":f"crosstie_{direction}/"+"/".join(ids),"z_bounds_in":[plane,plane]})
            if axis==1:
                for tube in local:
                    for endpoint in ("start_in","end_in"):
                        tube[endpoint][0],tube[endpoint][1]=tube[endpoint][1],tube[endpoint][0]
            for tube in local:
                tube["physical_bar_id"]=f"crosstie_{direction}/"+"/".join(ids)
            segments+=local
    if offset_support:
        # Along-face offset components require a real tie in that direction.
        # Opposite bars on each face are linked by additional 135/90 ties,
        # sharing the plane of the parallel ordinary crossties. They are
        # additional steel, never assumed to be the ordinary orthogonal leg.
        for face,axis in (("top",0),("bottom",0),("left",1),("right",1)):
            bars=sorted((p for p in coordinates if p["face"]==face),key=lambda p:p["x_in"] if axis==0 else p["y_in"])
            if face in ("top","bottom"):
                bars=bars[1:-1]  # existing perimeter corners own both axes
            for a,b in zip(bars[:len(bars)//2],reversed(bars[len(bars)-len(bars)//2:])):
                local,envelopes=_supplemental_face_tie(a,b,axis,ri,dt,db,z+(3 if axis==0 else 2)*pitch)
                segments+=local
                out["engagement_envelopes"]+=envelopes
                for p in (a,b):
                    out["ownership"].setdefault(p["id"],[]).append("supplemental_along_face_offset_restraint")
    ri=perimeter_ri
    violations=[]
    for tube in segments:
        rad=tube["diameter_in"]/2-tube.get("chord_enclosure_allowance_in",0.)
        if any(abs(p[0])+rad>design.b_in/2-clear_cover_in+TOL or abs(p[1])+rad>design.h_in/2-clear_cover_in+TOL
               for p in (tube["start_in"],tube["end_in"])):
            # Chord enclosure can extend outside by its deliberately added
            # sagitta although the exact arc remains inside its cover.
            if tube["id"] != "perimeter_wound_body":
                violations.append(tube["id"])
    out["checks"].append({"rule":"hook_extensions_inside_cover","passes":not violations,"violations":sorted(set(violations))})
    collisions=[]
    for tie in segments:
        for bar in coordinates:
            if bar["id"] in tie["owners"]:
                continue
            longitudinal=_tube([bar["x_in"],bar["y_in"],z-10],[bar["x_in"],bar["y_in"],z+10],db,bar["id"])
            if _clearance(tie,longitudinal)<-TOL:
                collisions.append([tie["id"],bar["id"]])
    out["checks"].append({"rule":"hooks_clear_other_longitudinal_bars","passes":not collisions,"collisions":collisions})
    contained=[]
    for p in coordinates:
        qx,qy=abs(p["x_in"])-cx,abs(p["y_in"])-cy
        signed=math.hypot(max(qx,0),max(qy,0))+min(max(qx,qy),0)-ri
        if signed+db/2>TOL:
            contained.append(p["id"])
    out["checks"].append({"rule":"all_longitudinal_cylinders_inside_rounded_hoop","passes":not contained,"outside":contained})
    physical_bars=transverse_bar_connectivity(segments)
    out["checks"].append({"rule":"transverse_bars_are_connected","passes":physical_bars["passes"],"bars":physical_bars["bars"]})
    clashes=transverse_collisions(segments,segments,same_set=True)
    out["checks"].append({"rule":"transverse_bars_do_not_intersect","passes":not clashes,"collisions":clashes[:20]})
    out["passes"] = valid and all(c["passes"] for c in out["checks"]+layout["checks"])
    out["total_length_in"] = sum(s["cut_length_upper_in"] for s in segments)
    out["cut_length_basis"]="exact planar arcs plus a conservative smooth-winding 3D arc-length bound"
    return out


def _supplemental_face_tie(a,b,axis,ri,dt,db,z):
    normal=1-axis
    normal_key="y_in" if normal==1 else "x_in"
    axial_key="x_in" if axis==0 else "y_in"
    n=(a[normal_key]+b[normal_key])/2
    sign=1 if n>=0 else -1
    low,high=sorted((a,b),key=lambda p:p[axial_key])
    r=ri+dt/2
    v0,v1=low[axial_key]+ri-db/2,high[axial_key]-ri+db/2
    owners=[low["id"],high["id"]]
    def point(u,v):
        p=[0.,0.,z]
        p[normal],p[axis]=n+sign*u,v
        return p
    name="supplemental_face_"+("x" if axis==0 else "y")
    tubes=[_tube(point(r,v0),point(r,v1),dt,name,owners)]
    envelopes=[]
    for v,stop,owner in ((v1,3*math.pi/4,high["id"]),(v0,-math.pi/2,low["id"])):
        parts,pts=_arc((0,v),r,0,stop,z,dt,name,[owner])
        for part in parts:
            part["start_in"]=point(*part["start_in"][:2])
            part["end_in"]=point(*part["end_in"][:2])
        tubes+=parts
        turn=1 if stop>0 else -1
        extension=max(6*dt,3.)
        end=pts[-1]
        tubes.append(_tube(point(*end[:2]),point(end[0]-turn*math.sin(stop)*extension,
                                              end[1]+turn*math.cos(stop)*extension),dt,name,[owner]))
        center=point(0,v)[:2]
        envelopes.append({"owner":owner,"center_in":center,"inside_radius_in":ri,"angle_start_rad":0.,
                          "angle_stop_rad":stop,"axial_index":axis,"normal_index":normal,"normal_sign":sign,
                          "axes":["x" if axis==0 else "y"],"transverse_bar_id":name+"/"+"/".join(owners),
                          "z_bounds_in":[z,z]})
    for tube in tubes:
        tube["physical_bar_id"]=name+"/"+"/".join(owners)
    return tubes,envelopes


def transverse_bar_connectivity(segments):
    """Trace each actual fabricated bar and locate its segments along it."""
    groups={}
    for i,s in enumerate(segments):
        groups.setdefault(s["physical_bar_id"],[]).append(i)
    records=[]
    for name,indices in groups.items():
        graph={}
        key=lambda p:tuple(round(x,8) for x in p)
        for i in indices:
            for end in ("start_in","end_in"):
                graph.setdefault(key(segments[i][end]),[]).append(i)
        ends=[point for point,edges in graph.items() if len(edges)==1]
        good=len(ends)==2 and all(len(edges)<=2 for edges in graph.values())
        point=ends[0] if ends else None
        visited=set()
        distance=0.
        while point is not None:
            choices=[i for i in graph[point] if i not in visited]
            if not choices:
                break
            i=choices[0]
            visited.add(i)
            s=segments[i]
            length=math.dist(s["start_in"],s["end_in"])
            s["fabricated_bar_interval_in"]=[distance,distance+length]
            distance+=length
            point=key(s["end_in"]) if point==key(s["start_in"]) else key(s["start_in"])
        good=good and len(visited)==len(indices)
        records.append({"physical_bar_id":name,"connected":good,"endpoints":len(ends),"length_in":distance})
    return {"passes":all(r["connected"] for r in records),"bars":records}


def transverse_collisions(first,second,same_set=False):
    failures=[]
    seen=set()
    for i,a in enumerate(first):
        for j,b in enumerate(second):
            if same_set and j<=i:
                continue
            if same_set and a["physical_bar_id"]==b["physical_bar_id"]:
                aa=a.get("fabricated_bar_interval_in")
                bb=b.get("fabricated_bar_interval_in")
                if aa and bb and max(aa[0]-bb[1],bb[0]-aa[1])<=2*max(a["diameter_in"],b["diameter_in"]):
                    continue  # only local continuous-bar capsule adjacency
            gap=_clearance(a,b)
            if gap<-TOL:
                pair=(a["physical_bar_id"],b["physical_bar_id"])
                if pair not in seen:
                    seen.add(pair)
                    failures.append({"physical_bars":list(pair),"clearance_in":gap})
    return failures


def endpoint_engagement(cage,xy,diameter,owner):
    """Actual circular hook/corner ownership at a proposed bend endpoint."""
    axes=set()
    for hook in cage["engagement_envelopes"]:
        if hook["owner"]!=owner:
            continue
        delta=[xy[k]-hook["center_in"][k] for k in (0,1)]
        if "axial_index" in hook:
            delta=[delta[hook["normal_index"]]*hook["normal_sign"],delta[hook["axial_index"]]]
        elif hook.get("swap_axes"):
            delta.reverse()
        if math.hypot(*delta)+diameter/2>hook["inside_radius_in"]+TOL:
            continue
        theta=math.atan2(delta[1],delta[0])
        a,b=sorted((hook["angle_start_rad"],hook["angle_stop_rad"]))
        if math.hypot(*delta)<=TOL or any(a-TOL<=theta+2*math.pi*k<=b+TOL for k in (-1,0,1)):
            axes.update(hook["axes"])
    return sorted(axes)


def path_xy_at_z(path,geometry,z):
    """Exact point of the two circular offset arcs and their common tangent."""
    start,end=path["lower_xy_in"],path["target_xy_in"]
    delta=path["offset_in"]
    if delta<=TOL:
        return list(start)
    h,r,t=geometry["height_in"],geometry["radius_in"],geometry["angle_rad"]
    if z<=-h:
        return list(start)
    if z>=0:
        return list(end)
    arc_height=r*math.sin(t)
    if z<-h+arc_height:
        offset=r-math.sqrt(max(0.,r*r-(z+h)**2))
    elif z>-arc_height:
        offset=delta-r+math.sqrt(max(0.,r*r-z*z))
    else:
        offset=r*(1-math.cos(t))+(z+h-arc_height)*math.tan(t)
    return [start[k]+(end[k]-start[k])*offset/delta for k in (0,1)]


def development_transverse_index(cage,bar_area_in2,spacing_in,bars_in_plane,governing_db_in):
    """Ktr from actual projected crossing steel; no nominal winding-leg credit."""
    legs=cage.get('layout',{}).get('legs',{})
    projection=cage.get('hoop_leg_horizontal_projection_factor',0.)
    crossing=min((max(0.,count-2)+2*projection)*bar_area_in2 for count in legs.values()) if legs else 0.
    ktr=40*crossing/(spacing_in*bars_in_plane)
    return {'Ktr_in':ktr,'required_in':.5*governing_db_in,'actual_crossing_area_in2':crossing,
        'hoop_leg_projection_factor':projection,'bars_in_splitting_plane':bars_in_plane,
        'passes':ktr>=.5*governing_db_in-TOL}


def verify_transition(result,lower,upper,*,fy_ksi,clear_cover_in,upper_clear_height_in,joint_depth_in,
                      lower_hoop_spacing_in=None,physical_context=None):
    """Compute checks against the paths actually proposed; never repair silently."""
    from Design.SMRF_Transitions import bar_coordinates,_BAR
    ctx = physical_context or {}
    checks=[]
    def check(rule,passes,detail,clause):
        checks.append({"rule":rule,"status":"not_evaluated" if passes is None else "pass" if passes else "fail",
                       "passes":passes,"detail":detail,"clause":clause})
    out={"method":METHOD,"status":"not_evaluated","checks":checks,"coordinate_basis":"hoop_corner_containment_v1",
         "nominal_envelope_scope":"Code nominal bar diameters; product outside dimensions; fabrication tolerances require shop specification.",
         "paths_3d":[],"tie_sets":[],"couplers":[],"additional_transverse_steel_volume_in3":0.}
    out["longitudinal_quantity_adjustment_in3"]=0.
    out["minimum_joint_section_dimensions_in"]=[min(lower.b_in,upper.b_in),min(lower.h_in,upper.h_in)]
    if not result.get("geometry_supported",result["supported"]):
        check("preliminary_transition_rules",False,"At least one declared path/scope/anchorage rule fails.","declared transition scope")
    paths=result["paths"]
    if not paths:
        out["status"]="fail"
        out["failure_rules"]=[c["rule"] for c in checks if c["status"]=="fail"]
        out["unknown_rules"]=[]
        return out
    db,udb,dt=_DB[int(lower.bar_size)],_DB[int(upper.bar_size)],_DB[int(lower.stirrup_bar_size)]
    maximum=max(p["offset_in"] for p in paths)
    height=offset_bend_geometry(maximum,lower.bar_size)["height_in"] if maximum>TOL else 0.
    check("finite_radius_bends_fit",height<=joint_depth_in+TOL,
          {"required_height_in":height,"available_in":joint_depth_in},"10.7.4.1; declared Table 25.3.1 fabrication radii")
    tubes=[]
    for i,p in enumerate(paths):
        name=f"{p['face']}_{p['lower_index']}"
        part,geometry=offset_path_segments({**p,"id":name},lower.bar_size,height,joint_depth_in)
        tubes+=part
        out["paths_3d"].append({"id":name,"geometry":geometry,"segments":part})
        if p["offset_in"]>TOL and geometry.get("fits"):
            out["longitudinal_quantity_adjustment_in3"]+=(geometry["straight_in"]+2*geometry["radius_in"]*geometry["angle_rad"]-height)*_BAR[int(lower.bar_size)][1]
    overlap=lap_group_spacing(paths,db,udb,float(ctx.get("aggregate_max_size_in",.75)))
    check("lap_group_clear_spacing",overlap["passes"],overlap,"25.2.3/25.5.1.2")
    beam=ctx.get("beam_segments",[])+ctx.get("placement_obstacle_segments",[])
    complete=ctx.get("beam_geometry_complete") is True
    check("beam_bar_geometry_available",True if complete else None,
          f"{len(beam)} supplied actual beam-bar segments; completeness={complete}","physical beam/column placement")
    for evidence in ctx.get("beam_geometry_checks",[]):
        check(evidence["rule"],None if evidence.get("status")=="not_evaluated" else evidence["passes"],evidence,"25.3.1;18.8.5.1;actual installed beam placement")
    out["beam_geometry_method"]=ctx.get("method")
    out["beam_hook_families"]=ctx.get("beam_hook_families",[])
    collisions=[]
    for col in tubes:
        for other in beam:
            gap=_clearance(col,other)
            if gap < -TOL:
                collisions.append({"column_bar":col["id"],"beam_bar":other.get("id"),"clearance_in":gap})
    check("column_paths_clear_beam_bars",not collisions if complete else None,collisions[:30],"physical nominal-envelope geometry")
    from dataclasses import replace
    axial=ctx.get("axial_max_kip")
    high=bool(axial is not None and axial>.3*min(lower.b_in*lower.h_in,upper.b_in*upper.h_in)*lower.fc_ksi) or lower.fc_ksi>10
    check("actual_confinement_axial_input",True if axial is not None else None,axial,"18.7.5.2(f);18.7.5.4")
    geometries={p["id"]:p["geometry"] for p in out["paths_3d"]}
    path_map={f"{p['face']}_{p['lower_index']}":p for p in paths}
    has_bends=maximum>TOL
    def coordinates_at(z):
        placed=[]
        for p in paths:
            pid=f"{p['face']}_{p['lower_index']}"
            xy=path_xy_at_z(p,geometries[pid],z)
            placed.append({"id":pid,"face":p["face"],"index":p["lower_index"],"x_in":xy[0],"y_in":xy[1]})
        return placed
    coords=coordinates_at(-joint_depth_in)
    cage=rectangular_cage_geometry(lower,coords,clear_cover_in,high_axial=high)
    out["lower_cage"]={k:v for k,v in cage.items() if k!="segments"}
    check("hoop_corner_and_hook_geometry",cage["passes"],cage["checks"],"18.7.5.2;25.3.2/25.3.4/25.3.5")
    spacing=min(float(lower_hoop_spacing_in or lower.stirrup_spacing_in),float(ctx.get("upper_hoop_spacing_in") or upper.stirrup_spacing_in))
    set_height=cage["set_height_in"]
    check("tie_set_vertical_fit",set_height+dt<=spacing+TOL,
          {"stack_height_in":set_height,"bar_diameter_in":dt,"spacing_in":spacing},"physical orthogonal tie placement")
    cage_cache={}
    def cage_at(station):
        if not has_bends:
            return cage
        key=round(station,8)
        if key not in cage_cache:
            # The lower bars keep their actual identities through the joint.
            # Above it, the smaller section bounds this lower-bar stub cage.
            section=lower if station<0 else replace(lower,b_in=upper.b_in,h_in=upper.h_in)
            cage_cache[key]=rectangular_cage_geometry(section,coordinates_at(station-dt/2),clear_cover_in,
                high_axial=high,offset_support=True,bar_envelope_db_in=db*math.sqrt(1+1/36))
        return cage_cache[key]
    def shifted_segments(detail,station,index):
        answer=[]
        for tie in detail["segments"]:
            moved={**tie,"start_in":[*tie["start_in"][:2],tie["start_in"][2]+station],
                   "end_in":[*tie["end_in"][:2],tie["end_in"][2]+station]}
            if index%2 and (tie["id"].startswith("crosstie_") or tie["id"].startswith("supplemental_face_")):
                axis=1 if "_y" in tie["id"] else 0
                for end in ("start_in","end_in"):
                    moved[end][axis]*=-1
                moved["owners"]=[reflected_owner(detail,owner,axis) for owner in tie["owners"]]
            answer.append(moved)
        return answer
    def reflected_owner(detail,owner,axis):
        coordinates=detail["longitudinal_coordinates"]
        original=next(p for p in coordinates if p["id"]==owner)
        target=[original["x_in"],original["y_in"]]
        target[axis]*=-1
        return min(coordinates,key=lambda p:math.hypot(p["x_in"]-target[0],p["y_in"]-target[1]))["id"]
    def actual_hooks(detail,index):
        hooks=[]
        for hook in detail["engagement_envelopes"]:
            transformed=dict(hook)
            if index%2 and hook["transverse_bar_id"]!="perimeter_winding":
                axis=0 if hook["axes"]==["x"] else 1
                transformed["owner"]=reflected_owner(detail,hook["owner"],axis)
                transformed["center_in"]=list(hook["center_in"])
                transformed["center_in"][axis]*=-1
                transformed["angle_start_rad"]=-hook["angle_start_rad"]
                transformed["angle_stop_rad"]=-hook["angle_stop_rad"]
            hooks.append(transformed)
        return hooks
    def restraints(details):
        evidence=[]
        area=_BAR[int(lower.stirrup_bar_size)][1]
        for p in paths:
            if not p["bent"]:
                continue
            pid=f"{p['face']}_{p['lower_index']}"
            angle=geometries[pid]["angle_rad"]
            for end,zz,xy in (("lower",-height,p["lower_xy_in"]),("upper",0.,p["target_xy_in"])):
                for component,axis in enumerate(("x","y")):
                    required=1.5*_BAR[lower.bar_size][1]*fy_ksi*math.tan(angle)*abs(p["offset_xy_in"][component])/p["offset_in"]
                    if required<=TOL:
                        continue
                    available=0.
                    owners=[]
                    for number,(station,detail) in enumerate(details):
                        for hook in actual_hooks(detail,number):
                            if hook["owner"]!=pid or axis not in hook["axes"]:
                                continue
                            if max(abs(station+h-zz) for h in hook["z_bounds_in"])>6+TOL:
                                continue
                            if axis not in endpoint_engagement({"engagement_envelopes":[hook]},xy,db,pid):
                                continue
                            factor=detail["hoop_leg_horizontal_projection_factor"] if hook["transverse_bar_id"]=="perimeter_winding" else 1.
                            available+=area*float(ctx.get("transverse_fy_ksi",60.))*factor
                            owners.append({"set":number,"bar":hook["transverse_bar_id"]})
                    evidence.append({"bar":pid,"bend":end,"axis":axis,"required_kip":required,
                                     "provided_kip":available,"transverse_owners":owners,"passes":available>=required-TOL})
        return evidence
    selected=None
    failure_details=[]
    phases=sorted(set([spacing*k/16 for k in range(16)]+[.0625,.125]))
    orientations=[(False,False,False),(False,False,True),
                  (True,False,False),(False,True,False),(True,True,False),
                  (True,False,True),(False,True,True),(True,True,True)]
    orientations=[(*o,swap) for swap in (False,True) for o in orientations]
    trials=[(orientation,phase) for orientation in orientations for phase in phases]
    orientation_cache={}
    attempted=0
    for orientation,phase in trials:
        attempted+=1
        stations=[]
        zz=-joint_depth_in-(6 if has_bends else 0)+phase
        while zz<=(6 if has_bends else 0)+TOL:
            stations.append(zz)
            zz+=spacing
        details=[]
        for station in stations:
            base=cage_at(station)
            key=(id(base),orientation)
            if key not in orientation_cache:
                orientation_cache[key]=oriented_cage(base,*orientation)
            detail=orientation_cache[key]
            if detail is None:
                break
            details.append((station,detail))
        if len(details)!=len(stations):
            failure_details=[{"rule":"orientation_does_not_map_actual_longitudinal_bars"}]
            continue
        invalid=[{"station":station,"checks":[c for c in detail["checks"] if not c["passes"]]}
                 for station,detail in details if not detail["passes"]]
        if invalid:
            failure_details=invalid[:3]
            continue
        force_checks=restraints(details)
        if any(not c["passes"] for c in force_checks):
            failure_details=[{"restraints":[c for c in force_checks if not c["passes"]][:6]}]
            continue
        previous=None
        collision=False
        for index,(station,detail) in enumerate(details):
            shifted=shifted_segments(detail,station,index)
            if previous is not None and transverse_collisions(previous,shifted):
                collision=True
                failure_details=[{"rule":"adjacent_transverse_sets_intersect"}]
                break
            previous=shifted
            for tie in shifted:
                hit=next((b for b in beam if _clearance(tie,b)<-TOL),None)
                if hit is not None:
                    collision=True
                    failure_details=[{"rule":"tie_beam_collision","station":station,"tie":tie["id"],
                                      "beam":hit.get("id"),"beam_z_in":[hit['start_in'][2],hit['end_in'][2]],
                                      "clearance_in":_clearance(tie,hit)}]
                    break
                # An owned bar is exempt only after an analytic enclosure
                # of its actual inclined cylinder inside THAT hook/corner.
                # Sharing an owner label is not evidence of clearance.
                for col in tubes:
                    if _clearance(tie,col)>=-TOL:
                        continue
                    certified=False
                    if col["id"] in tie["owners"]:
                        hooks=[h for h in actual_hooks(detail,index) if h["owner"]==col["id"] and
                               h["transverse_bar_id"]==tie["physical_bar_id"]]
                        rad=tie["diameter_in"]/2
                        elevations=[min(tie["start_in"][2],tie["end_in"][2])-rad,
                                    max(tie["start_in"][2],tie["end_in"][2])+rad]
                        envelope=db*math.sqrt(1+1/36) if path_map[col["id"]]["bent"] else db
                        # Successive crosstie reversal changes its physical
                        # hook ownership. Enclosure for these two bars is
                        # symmetric under the same end-for-end reflection.
                        for hook in hooks:
                            certified=all(endpoint_engagement({"engagement_envelopes":[hook]},
                                path_xy_at_z(path_map[col["id"]],geometries[col["id"]],elevation),envelope,col["id"])
                                for elevation in elevations)
                            if certified:
                                break
                    if not certified:
                        collision=True
                        failure_details=[{"rule":"tie_column_collision_without_valid_hook_enclosure","station":station,"tie":tie["id"],"column":col["id"]}]
                        break
                if collision:
                    break
            if collision:
                break
        if not collision:
            selected=details
            out["bend_restraints"]=force_checks
            break
    check("joint_transverse_detail_and_station_clearance",selected is not None if complete else None,
          {"phase_trials":len(phases),"orientation_trials":len(orientations),"trials_attempted":attempted,
           "spacing_in":spacing,"last_refusal":failure_details},"10.7.6.4;18.8.3;physical tie/beam placement")
    out["tie_sets"]=[{"z_in":station,"spacing_in":spacing,"crossties_reversed":bool(i%2),
                      "bar_coordinates":coordinates_at(station-dt/2),
                      "core_outside_dimensions_in":detail["core_outside_dimensions_in"],"core_area_in2":detail["core_area_in2"],
                      "transverse_length_in":detail["total_length_in"],"hoop_form":detail["hoop_form"],
                      "fabrication_orientation":detail.get("fabrication_orientation",{"reflect_x":False,"reflect_y":False,"reverse_z":False}),
                      "hoop_leg_horizontal_projection_factor":detail["hoop_leg_horizontal_projection_factor"]}
                     for i,(station,detail) in enumerate(selected or [])]
    if selected:
        min_detail=min((d for _,d in selected),key=lambda d:d["core_area_in2"])
        ach=min_detail["core_area_in2"]
        bc=min_detail["core_outside_dimensions_in"]
        ag=lower.b_in*lower.h_in
        fyt=float(ctx.get("transverse_fy_ksi",60.))
        ratio=max(.3*(ag/ach-1)*lower.fc_ksi/fyt,.09*lower.fc_ksi/fyt)
        if high and axial is not None:
            n=len(paths)
            ratio=max(ratio,.2*max(1.,lower.fc_ksi/25+.6)*n/max(1,n-2)*max(0.,axial)/(fyt*ach))
        legs=min_detail["layout"]["legs"]
        projection=min(d["hoop_leg_horizontal_projection_factor"] for _,d in selected)
        area=_BAR[int(lower.stirrup_bar_size)][1]
        available=[(legs["across_b_face"]-2+2*projection)*area,(legs["across_h_face"]-2+2*projection)*area]
        required=[ratio*spacing*bc[0],ratio*spacing*bc[1]]
        physical_core={"outside_dimensions_in":bc,"area_in2":ach,"gross_area_in2":ag,"axial_max_kip":axial,
                       "ash_provided_in2":available,"ash_required_in2":required,"hoop_leg_projection_factor":projection,
                       "hx_in":max(d["layout"]["hx_in"] for _,d in selected),"spacing_in":spacing}
        out["minimum_joint_confinement_core"]=physical_core
        check("actual_joint_core_confinement",all(a>=r-TOL for a,r in zip(available,required)) if axial is not None else None,
              physical_core,"18.8.3.1;18.7.5.4")
        # Added ties are real fabricated bars. No negative quantity credit.
        lower_baseline=cage["total_length_in"]*area
        upper_baseline=rectangular_cage_geometry(upper,bar_coordinates(upper,clear_cover_in),clear_cover_in)
        upper_volume=upper_baseline.get("total_length_in",0.)*_BAR[int(upper.stirrup_bar_size)][1]
        out["additional_transverse_steel_volume_in3"]=sum(max(0.,d["total_length_in"]*area-(lower_baseline if z<=0 else upper_volume)) for z,d in selected)
    has_lap=any(p["path"]=="lap_spliced" for p in paths)
    if has_lap or any(p["path"]=="terminated" for p in paths):
        # The lower cage is explicitly retained throughout development/lap;
        # the existing upper cage alone does not own every lower tail.
        lower_fits=lower.b_in<=upper.b_in+TOL and lower.h_in<=upper.h_in+TOL
        check("lap_zone_lower_cage_continuation",lower_fits,
              "Lower hoops/crossties continue through the entire lap and every terminated tail; a reduced section cannot contain that cage.","18.7.4.4;18.7.5.2")
        lap=result["splice"]["class_b_lap_in"] if has_lap else 0.
        start=(upper_clear_height_in-lap)/2 if has_lap else 0.
        stop=max(start+lap,upper_clear_height_in/2)
        # A hoop cannot keep claiming a hook around a tail after that bar
        # ends. Extend all terminated lower tails to the end of the actual
        # overlap/detail zone and price that explicit extra longitudinal steel.
        out["terminated_tail_length_above_joint_in"]=stop
        out["longitudinal_quantity_adjustment_in3"]+=sum(
            stop*_BAR[int(lower.bar_size)][1] if p["path"]=="terminated" else
            stop*_BAR[int(lower.bar_size)][1]-start*_BAR[int(upper.bar_size)][1] if p["path"]=="lap_spliced" else 0.
            for p in paths)
        overlap_inventory=[]
        upper_coordinates=[]
        upper_by_key={(p["face"],p["index"]):p for p in bar_coordinates(upper,clear_cover_in)}
        for p in paths:
            pid=f"{p['face']}_{p['lower_index']}"
            if p["path"] in ("lap_spliced","terminated"):
                overlap_inventory.append({"id":pid,"xy_in":p["target_xy_in"],"diameter_in":db})
            if p["upper_xy_in"] is not None:
                uid=f"upper_{pid}" if p["path"]=="lap_spliced" else pid
                overlap_inventory.append({"id":uid,"xy_in":p["upper_xy_in"],"diameter_in":udb})
                upper_coordinates.append({**upper_by_key[p["face"],p["upper_index"]],"id":uid})
        lower_lap=rectangular_cage_geometry(replace(lower,b_in=upper.b_in,h_in=upper.h_in),coordinates_at(0),clear_cover_in,high_axial=high)
        upper_lap=rectangular_cage_geometry(upper,upper_coordinates,clear_cover_in,high_axial=high)
        zone_checks=[]
        chosen_phase=None
        zone_mode="separate_lower_and_upper_cages"
        # First test the actual combined lap inventory against the lower
        # cage's owned supports. A newly lapped bar may be an allowed
        # alternate unsupported bar between two directly supported lower
        # bars; duplicating an entire upper cage would invent congestion.
        from Design.SMRF_Cage_Layout import _describe,_cage_checks
        combined_faces={}
        for face,coordinate in (("top",0),("left",1)):
            if face=="top":
                members=[v for v in overlap_inventory if v["id"].startswith(("top_","upper_top_"))]
            else:
                members=[v for v in overlap_inventory if v["id"].startswith(("left_","upper_left_"))]
                members += [v for v in overlap_inventory if v["id"] in ("top_0","bottom_0")]
            members.sort(key=lambda v:v["xy_in"][coordinate])
            supported=[v["id"] in lower_lap["ownership"] for v in members]
            combined_faces[face]=_describe([v["xy_in"][coordinate] for v in members],max(db,udb),supported)
        combined_support=_cage_checks(combined_faces,8. if high else 14.,"column")
        if high:
            combined_support.append({"rule":"all_lap_bars_supported_at_high_axial","passes":all(
                v["id"] in lower_lap["ownership"] for v in overlap_inventory)})
        union_clear=True
        for tie in lower_lap["segments"]:
            for bar in overlap_inventory:
                if bar["id"] in tie["owners"]:
                    continue
                cylinder=_tube([*bar["xy_in"],-10],[*bar["xy_in"],20],bar["diameter_in"],bar["id"])
                if _clearance(tie,cylinder)<-TOL:
                    union_clear=False
        if lower_fits and lower_lap["passes"] and union_clear and all(c["passes"] for c in combined_support):
            chosen_phase=0.
            zone_mode="one_lower_cage_supports_combined_lap_inventory"
        if lower_lap["passes"] and upper_lap["passes"] and lower_fits:
            for k in range(1,32):
                if chosen_phase is not None:
                    break
                phase=spacing*k/32
                okay=True
                for index in (0,1):
                    lo=shifted_segments(lower_lap,index*spacing,index)
                    up=shifted_segments(upper_lap,index*spacing+phase,index)
                    nxt=shifted_segments(lower_lap,(index+1)*spacing,index+1)
                    if transverse_collisions(lo,up) or transverse_collisions(up,nxt):
                        okay=False
                        break
                    for tie in lo+up:
                        for bar in overlap_inventory:
                            if bar["id"] in tie["owners"]:
                                continue  # individually verified vertical owner/hook geometry
                            cylinder=_tube([*bar["xy_in"],-10],[*bar["xy_in"],20],bar["diameter_in"],bar["id"])
                            if _clearance(tie,cylinder)<-TOL:
                                okay=False
                                break
                        if not okay:
                            break
                    if not okay:
                        break
                if okay:
                    chosen_phase=phase
                    break
        zone_checks=[c for d in (lower_lap,upper_lap) for c in d["checks"] if not c["passes"]]
        check("lap_zone_actual_hoops_and_crossties",chosen_phase is not None,
              {"detail_mode":zone_mode,"lower_phase_in":0.,"upper_phase_in":chosen_phase,
               "spacing_in":spacing,"phase_trials":31,"geometry_failures":zone_checks,
               "combined_support_checks":combined_support,"overlap_bars":overlap_inventory},"18.7.4.4;18.7.5.2;25.3.5")
        out["lap_zone"]={"interval_in":[start,stop],"bar_inventory":overlap_inventory,
                         "lower_hoop_phase_in":0.,"upper_hoop_phase_in":chosen_phase,"spacing_in":spacing,
                         "detail_mode":zone_mode,"combined_support":combined_faces,
                         "detail":"Actual combined lap bars are checked against owned physical hooks and alternate-bar support rules; separate cages are used only if jointly collision-free."}
        joint_top=max((z for z,_ in selected or []),default=0.)
        last_allowed=stop-lower_lap["set_height_in"]-_DB[int(lower.stirrup_bar_size)]/2
        stations=[]
        z=joint_top+spacing
        while z<=last_allowed+TOL:
            stations.append(z)
            z+=spacing
        last_lower=stations[-1] if stations else joint_top
        first_upper=last_lower+spacing
        handoff_clear=not transverse_collisions(_positioned_cage_segments(lower_lap,last_lower,0),
                                               _positioned_cage_segments(upper_lap,first_upper,1))
        check("development_zone_cage_handoff",handoff_clear and upper_lap["passes"],
              {"last_lower_set_z_in":last_lower,"first_upper_set_z_in":first_upper,
               "terminated_bar_tips_z_in":stop,"maximum_set_gap_in":spacing},"18.7.5.2;physical bar termination/hoop ownership")
        out["lap_zone"]["cage_handoff"]={"last_lower_set_z_in":last_lower,"first_upper_set_z_in":first_upper}
        if selected and stations:
            last_joint_z,last_joint_detail=selected[-1]
            handoff=not transverse_collisions(shifted_segments(last_joint_detail,last_joint_z,len(selected)-1),
                                              shifted_segments(lower_lap,stations[0],len(selected)))
            check('joint_to_development_cage_handoff',handoff and stations[0]-last_joint_z<=spacing+TOL,
                  {'last_joint_set_z_in':last_joint_z,'first_development_set_z_in':stations[0],
                   'last_joint_fabrication_orientation':last_joint_detail.get('fabrication_orientation')},
                  '18.7.5.2;physical bar/hoop ownership')
        # Only stations above those already owned by the joint detail are
        # priced here. The lower cage replaces the upper ordinary grid;
        # a second upper cage is counted only in the explicit two-cage mode.
        lower_volume=lower_lap.get("total_length_in",0.)*_BAR[int(lower.stirrup_bar_size)][1]
        upper_volume=upper_lap.get("total_length_in",0.)*_BAR[int(upper.stirrup_bar_size)][1]
        zone_volume=len(stations)*(lower_volume+(upper_volume if zone_mode=="separate_lower_and_upper_cages" else 0.))
        replaced=len(stations)*upper_volume*spacing/float(ctx.get("upper_hoop_spacing_in") or upper.stirrup_spacing_in)
        out["additional_transverse_steel_volume_in3"]+=max(0.,zone_volume-replaced)
        out["lap_zone"]["quantity"]={"lower_stations_in":sorted(stations),"actual_steel_volume_in3":zone_volume,
            "replaced_upper_baseline_volume_in3":replaced,"joint_owned_top_station_in":joint_top}
        if fy_ksi>=80:
            # Each direction cuts at most half the longitudinal bars. Use
            # the smaller effective crossing area to bound Ktr downward.
            actual_face_count=max(len(v["positions_in"]) for v in combined_faces.values())
            ktr=development_transverse_index(lower_lap,_BAR[int(lower.stirrup_bar_size)][1],
                spacing,actual_face_count,max(db,udb))
            check("grade80_lap_transverse_index",ktr['passes'],ktr,
                  "10.7.1.3;25.4.2.2;25.5.1.5")
    mechanical=[p for p in paths if p["path"]=="mechanically_spliced"]
    if mechanical:
        placement=verify_mechanical_splice(upper,fy_ksi=fy_ksi,clear_cover_in=clear_cover_in,
            clear_height_in=upper_clear_height_in,spacing_in=float(ctx.get("upper_hoop_spacing_in") or upper.stirrup_spacing_in),
            specification=ctx.get("longitudinal_specification"),aggregate_in=float(ctx.get("aggregate_max_size_in",.75)),
            lower_bar_size=lower.bar_size,coordinates=[p["upper_xy_in"] for p in mechanical])
        product=placement["product"]
        out["coupler_product"]=product
        out["mechanical_splice_detail"]=placement
        check("mechanical_splice_product",None if product["status"]=="not_evaluated" else product["status"]=="pass",
              product,"18.2.7;ER-0129")
        for entry in placement["checks"]:
            check(entry["rule"],entry["passes"],entry["detail"],"18.2.7;18.7.5.5;actual product/transverse geometry")
        out["couplers"]=placement["bodies"]
        out["additional_transverse_steel_volume_in3"]+=placement.get("additional_transverse_steel_volume_in3",0.)
        out["mechanical_device_solid_envelope_steel_volume_in3"]=placement.get("solid_envelope_steel_volume_in3",0.)
        out["mechanical_device_net_steel_volume_in3"]=placement.get("net_envelope_steel_volume_in3",0.)
        out["longitudinal_quantity_adjustment_in3"]+=sum(sum(b["z_interval_in"])/2 for b in placement.get("bodies",[]))*(_BAR[int(lower.bar_size)][1]-_BAR[int(upper.bar_size)][1])
    states=[c["status"] for c in checks]
    out["status"]="fail" if "fail" in states else "not_evaluated" if "not_evaluated" in states else "pass"
    out["failure_rules"]=[c["rule"] for c in checks if c["status"]=="fail"]
    out["unknown_rules"]=[c["rule"] for c in checks if c["status"]=="not_evaluated"]
    return out


def segment_cylinder_clearance(segment,center_xy,z_interval,radius):
    """Conservative distance of a tube to a finite vertical solid cylinder.

    Distance to a closed convex cylinder is convex along a segment. Golden
    minimization retains the minimum bracket; its residual Lipschitz width
    is subtracted, so numerical uncertainty never creates clearance.
    """
    a,b=segment["start_in"],segment["end_in"]
    vertical_allowance=segment.get("vertical_chord_enclosure_in",0.)
    tube=segment["diameter_in"]/2-vertical_allowance
    # Winding-pitch approximation error is vertical, not radial. Expanding
    # the finite cylinder axially preserves an enclosing distance bound
    # without falsely growing a straight hoop leg toward a tangent device.
    z_interval=[z_interval[0]-vertical_allowance,z_interval[1]+vertical_allowance]
    if (max(a[2],b[2])+tube<z_interval[0] or min(a[2],b[2])-tube>z_interval[1] or
        any(max(a[k],b[k])+tube<center_xy[k]-radius or min(a[k],b[k])-tube>center_xy[k]+radius for k in (0,1))):
        return math.inf
    def distance(t):
        point=[a[k]+t*(b[k]-a[k]) for k in range(3)]
        radial=max(0.,math.hypot(point[0]-center_xy[0],point[1]-center_xy[1])-radius)
        axial=max(z_interval[0]-point[2],point[2]-z_interval[1],0.)
        return math.hypot(radial,axial)
    lo,hi=0.,1.
    ratio=(math.sqrt(5)-1)/2
    u,v=hi-ratio*(hi-lo),lo+ratio*(hi-lo)
    fu,fv=distance(u),distance(v)
    for _ in range(64):
        if fu<fv:
            hi,v,fv=v,u,fu
            u=hi-ratio*(hi-lo)
            fu=distance(u)
        else:
            lo,u,fu=u,v,fv
            v=lo+ratio*(hi-lo)
            fv=distance(v)
    bound=min(fu,fv,distance(0),distance(1))-math.dist(a,b)*(hi-lo)
    return bound-tube


def oriented_cage(cage,reflect_x=False,reflect_y=False,reverse_z=False,swap_cross_tie_planes=False):
    """Rigid reflections of one fabricated set; preserve steel and curvature.

    Plan reflections are allowed only when each physical longitudinal bar
    maps exactly to another actual bar. They relocate the winding closure,
    not the column cage. One orientation applies to every station, keeping
    every corresponding leg at the selected spacing.
    """
    if not any((reflect_x,reflect_y,reverse_z,swap_cross_tie_planes)):
        return cage
    from copy import deepcopy
    result=deepcopy(cage)
    if swap_cross_tie_planes:
        pitch=cage['hoop_pitch_in']
        def shift(name):
            if name.startswith(('crosstie_y','supplemental_face_y')):return pitch
            if name.startswith(('crosstie_x','supplemental_face_x')):return -pitch
            return 0.
        for tube in result['segments']:
            dz=shift(tube['id'])
            for end in ('start_in','end_in'):tube[end][2]+=dz
        for hook in result['engagement_envelopes']:
            dz=shift(hook['transverse_bar_id'])
            hook['z_bounds_in']=[z+dz for z in hook['z_bounds_in']]
        # Relative interbar distances can change, so this is explicitly
        # rechecked; plane permutation is not assumed collision-free.
        if transverse_collisions(result['segments'],result['segments'],same_set=True):
            return None
    coordinates=cage["longitudinal_coordinates"]
    center=cage.get("alternation_center_in",[0.,0.])
    def xy(point):
        return [2*center[i]-point[i] if (reflect_x,reflect_y)[i] else point[i] for i in (0,1)]
    owners={}
    for p in coordinates:
        target=xy([p["x_in"],p["y_in"]])
        matches=[q["id"] for q in coordinates if math.hypot(q["x_in"]-target[0],q["y_in"]-target[1])<=TOL]
        if len(matches)!=1:
            return None
        owners[p["id"]]=matches[0]
    height=cage["set_height_in"]
    def zz(value):
        return height-value if reverse_z else value
    for tube in result["segments"]:
        for end in ("start_in","end_in"):
            tube[end]=xy(tube[end])+[zz(tube[end][2])]
        tube["owners"]=[owners[o] for o in tube["owners"]]
    result["ownership"]={owners[k]:v for k,v in result["ownership"].items()}
    for hook in result["engagement_envelopes"]:
        hook["owner"]=owners[hook["owner"]]
        hook["center_in"]=xy(hook["center_in"])
        hook["z_bounds_in"]=sorted(zz(v) for v in hook["z_bounds_in"])
        if "normal_index" in hook:
            if (reflect_x,reflect_y)[hook["normal_index"]]:
                hook["normal_sign"]*=-1
            if (reflect_x,reflect_y)[hook["axial_index"]]:
                for key in ("angle_start_rad","angle_stop_rad"):
                    hook[key]=-hook[key]
        else:
            for axis,reflect in enumerate((reflect_x,reflect_y)):
                if not reflect:
                    continue
                local_axis=1-axis if hook.get("swap_axes") else axis
                for key in ("angle_start_rad","angle_stop_rad"):
                    hook[key]=math.pi-hook[key] if local_axis==0 else -hook[key]
    result["fabrication_orientation"]={"reflect_x":reflect_x,"reflect_y":reflect_y,"reverse_z":reverse_z,
                                       "swap_cross_tie_planes":swap_cross_tie_planes}
    return result


def _positioned_cage_segments(cage,station,index=0):
    answer=[]
    for tie in cage["segments"]:
        out={**tie,"start_in":[*tie["start_in"][:2],tie["start_in"][2]+station],
             "end_in":[*tie["end_in"][:2],tie["end_in"][2]+station]}
        if index%2 and (tie["id"].startswith("crosstie_") or tie["id"].startswith("supplemental_face_")):
            axis=1 if "_y" in tie["id"] else 0
            for end in ("start_in","end_in"):
                out[end][axis]=2*cage.get("alternation_center_in",[0.,0.])[axis]-out[end][axis]
        answer.append(out)
    return answer


def verify_mechanical_splice(design,*,fy_ksi,clear_cover_in,clear_height_in,spacing_in,specification,
                             station_in=None,aggregate_in=.75,lower_bar_size=None,coordinates=None):
    """Documented Type2 devices with actual cylinders, hooks and steel quantities.

    Device centers may stagger within one transverse spacing. A local hoop
    from the unchanged production ladder may be selected outside lo only
    if it preserves EVERY original Av/s and Ash/(s bc) bound. This creates
    real device space without weakening shear, confinement or spacing.
    """
    from dataclasses import replace
    from Design.SMRF_Transitions import bar_coordinates,_BAR
    from Design.SMRF_Capacity_Design import COLUMN_HOOP_BARS,SPACING_GRID_IN
    station=clear_height_in/2 if station_in is None else float(station_in)
    lower_size=design.bar_size if lower_bar_size is None else int(lower_bar_size)
    product=lookup_coupler(lower_size,design.bar_size,fy_ksi,specification,False)
    out={"method":"documented_A2_splice_layout_v2_actual_device_stagger","status":product["status"],"product":product,
         "checks":[],"bodies":[],"additional_transverse_steel_volume_in3":0.}
    if product["status"]!="pass":
        return out
    bars=[{**p,"id":f"{p['face']}_{p['index']}"} for p in bar_coordinates(design,clear_cover_in)]
    points=coordinates or [[p["x_in"],p["y_in"]] for p in bars]
    diameter,length=product["envelope_diameter_in"],product["length_in"]
    checks=out["checks"]
    def check(rule,condition,detail):
        checks.append({"rule":rule,"status":"pass" if condition else "fail","passes":bool(condition),"detail":detail})
    endzone=max(design.b_in,design.h_in,clear_height_in/6,18.)
    protected=max(endzone,2*max(design.b_in,design.h_in)) if fy_ksi>=80 else endzone
    cover=min(min(design.b_in/2-abs(x),design.h_in/2-abs(y))-diameter/2 for x,y in points)
    check("splice_device_cover",cover>=clear_cover_in-TOL,{"provided_in":cover,"required_in":clear_cover_in})
    clear=min((math.dist(a,b)-diameter for i,a in enumerate(points) for b in points[i+1:]),default=1e30)
    required=max(1.5,1.5*max(_DB[lower_size],_DB[int(design.bar_size)]),4*aggregate_in/3)
    check("splice_device_clear_spacing",clear>=required-TOL,{"provided_in":clear,"required_in":required,
          "basis":"conservative column-bar spacing applied to outer device envelopes"})
    base=rectangular_cage_geometry(design,bars,clear_cover_in)
    check("original_transverse_geometry",base["passes"],base["checks"])
    if not base["passes"]:
        out["status"]="fail"
        return out
    base_legs=design.stirrup_legs_by_direction or (design.stirrup_legs,)*2
    base_av=[n*_BAR[int(design.stirrup_bar_size)][1] for n in base_legs]
    base_bc=base["core_outside_dimensions_in"]
    code_spacing=min(6.,(5 if fy_ksi>=80 else 6)*min(_DB[lower_size],_DB[int(design.bar_size)]))
    options=[]
    full=(design.top_bars,design.side_bars+2)
    for bar in sorted(set([int(design.stirrup_bar_size)]+[b for b in COLUMN_HOOP_BARS if b>=design.stirrup_bar_size])):
        for legs in sorted(set([tuple(base_legs),full])):
            candidate=replace(design,bar_size=max(lower_size,design.bar_size),stirrup_bar_size=bar,
                              stirrup_legs=max(legs),stirrup_legs_by_direction=legs)
            cage=rectangular_cage_geometry(candidate,bars,clear_cover_in,perimeter_cover_in=clear_cover_in)
            if not cage["passes"]:
                continue
            area=_BAR[bar][1]
            factor=cage["hoop_leg_horizontal_projection_factor"]
            av=[(n-2+2*factor)*area for n in legs]
            bc=cage["core_outside_dimensions_in"]
            maximum=min([code_spacing]+[spacing_in*av[i]/base_av[i]*min(1.,base_bc[i]/bc[i]) for i in (0,1)])
            gap=math.floor((maximum+1e-9)/SPACING_GRID_IN)*SPACING_GRID_IN
            if gap<spacing_in-TOL or gap<length+_DB[bar]-TOL:
                continue
            if transverse_collisions(_positioned_cage_segments(cage,0,0),_positioned_cage_segments(cage,gap,1)):
                continue
            options.append((cage["total_length_in"]*area/gap,bar,legs,gap,cage,av,bc))
    chosen=None
    last_reason="No existing-ladder hoop spacing preserves both reinforcement-per-length bounds and device length."
    for _,bar,legs,gap,cage,av,bc in sorted(options,key=lambda x:x[:4]):
        # Three periods on each side cover all staggered device choices.
        transverse=[]
        for index in range(-3,4):
            transverse+=_positioned_cage_segments(cage,station+index*gap,index)
        bodies=[]
        offsets=sorted([(-.5+k/32)*gap for k in range(33)],key=lambda x:(abs(x),x))
        for i,xy in enumerate(points):
            placed=None
            for delta in offsets:
                z0,z1=station+delta-length/2,station+delta+length/2
                if z0<protected-TOL or z1>clear_height_in-protected+TOL:
                    continue
                if any(segment_cylinder_clearance(tie,xy,[z0,z1],diameter/2)<-TOL for tie in transverse):
                    continue
                # Unspliced/terminated column bars also remain present.
                if any(math.dist(xy,[p['x_in'],p['y_in']])>TOL and
                       math.dist(xy,[p['x_in'],p['y_in']])<diameter/2+max(_DB[lower_size],_DB[int(design.bar_size)])/2+required-TOL for p in bars):
                    continue
                placed={"id":f"coupler_{i}","geometry":"finite_vertical_cylinder","center_xy_in":list(xy),
                        "z_interval_in":[z0,z1],"diameter_in":diameter,"part_number":product["part_number"],"stagger_in":delta}
                break
            if placed is None:
                last_reason=f"No collision-free permitted device elevation at bar {i} with No.{bar} hoops at {gap:g}in."
                break
            bodies.append(placed)
        if len(bodies)!=len(points):
            continue
        first=math.floor((min(b['z_interval_in'][0] for b in bodies)-station)/gap)
        last=math.ceil((max(b['z_interval_in'][1] for b in bodies)-station)/gap)
        hoop_stations=[station+k*gap for k in range(first,last+1)]
        zone=[hoop_stations[0]-_DB[bar]/2,hoop_stations[-1]+cage["set_height_in"]+_DB[bar]/2]
        if zone[0]<endzone-TOL or zone[1]>clear_height_in-endzone+TOL:
            last_reason="The local upgraded transverse zone reaches the mandatory end confinement region."
            continue
        # Explicit bridge sets keep gaps to the ordinary cage <= its original
        # spacing; the physical steel at both junctions must remain clear.
        before=_positioned_cage_segments(base,hoop_stations[0]-spacing_in,0)
        after=_positioned_cage_segments(base,hoop_stations[-1]+spacing_in,0)
        if (transverse_collisions(before,_positioned_cage_segments(cage,hoop_stations[0],first)) or
            transverse_collisions(_positioned_cage_segments(cage,hoop_stations[-1],last),after)):
            last_reason="Local upgraded cage intersects a declared ordinary-cage bridge set."
            continue
        chosen=(bar,legs,gap,cage,av,bc,bodies,hoop_stations,zone)
        break
    check("splice_devices_and_transverse_detail",chosen is not None,{"candidate_details":len(options),"device_elevation_trials_per_bar":33,
          "last_refusal":None if chosen else last_reason,"outside_lo_spacing_limit_in":code_spacing,
          "protected_device_end_zone_in":protected,"clause":"18.2.7;18.7.5.5;ER-0129"})
    if chosen:
        bar,legs,gap,cage,av,bc,bodies,hoop_stations,zone=chosen
        ratios={"provided_Av_per_in":[a/gap for a in av],"original_Av_per_in":[a/spacing_in for a in base_av],
                "provided_Ash_over_sbc":[av[i]/(gap*bc[i]) for i in (0,1)],
                "original_Ash_over_sbc":[base_av[i]/(spacing_in*base_bc[i]) for i in (0,1)]}
        out["transverse_detail"]={"bar_size":bar,"legs_by_direction":list(legs),"spacing_in":gap,"zone_in":zone,
                                  "hoop_form":cage["hoop_form"],"set_height_in":cage["set_height_in"],"ratios":ratios,
                                  "perimeter_inside_bend_radius_in":cage["inside_bend_radius_in"],
                                  "actual_hoop_cover_in":cage["actual_cover_in"],"set_transverse_length_in":cage["total_length_in"],
                                  "core_area_in2":cage["core_area_in2"],"core_outside_dimensions_in":bc,
                                  "original_spacing_in":spacing_in,"bridge_stations_in":[hoop_stations[0]-spacing_in,hoop_stations[-1]+spacing_in]}
        out["bodies"],out["hoop_stations_in"]=bodies,hoop_stations
        area=_BAR[bar][1]
        base_volume=base["total_length_in"]*_BAR[int(design.stirrup_bar_size)][1]
        removed=max(0,math.floor((hoop_stations[-1]-station)/spacing_in)-math.ceil((hoop_stations[0]-station)/spacing_in)+1)
        # The two explicitly added bridge sets are conservatively included;
        # no assumed reuse of an unrecorded ordinary-grid station is credited.
        out["additional_transverse_steel_volume_in3"]=max(0.,len(hoop_stations)*cage["total_length_in"]*area+(2-removed)*base_volume)
    out["device_count"]=len(points)
    out["solid_envelope_steel_volume_in3"]=len(points)*math.pi*diameter**2/4*length
    out["baseline_bar_volume_replaced_in3"]=len(points)*length*(_BAR[lower_size][1]+_BAR[int(design.bar_size)][1])/2
    out["net_envelope_steel_volume_in3"]=max(0.,out["solid_envelope_steel_volume_in3"]-out["baseline_bar_volume_replaced_in3"])
    out["quantity_basis"]="solid circumscribed device-envelope upper bound, not manufacturer weight; explicit added bridge ties included conservatively"
    out["status"]="pass" if all(c["passes"] for c in checks) else "fail"
    return out
