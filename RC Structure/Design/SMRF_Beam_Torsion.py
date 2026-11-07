"""ACI 318-19 nonprestressed solid-beam torsion at a concurrent section state.

Units are kip, inch and ksi. No compatibility-torsion redistribution, concrete
torsional strength, compression cancellation, or interior-crosstie torsion
credit is taken. Missing physical/concurrent evidence is unknown.
"""
from __future__ import annotations

import math

import Structure_Parameters as sp
from Design.SMRF_Splice_Physical import beam_cage_geometry
from Design import SMRF_Transition_Physical as ph

METHOD="aci31819_22_7_actual_hoop_concurrent_reserve_v1"
SOURCE_SHA256="9cf6a9503428dc053bcf08bbbe6c9cdd0f0186ce75b60e50448778b47a18f4a0"
PHI=.75


def _finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def solid_section(rectangles,fc_ksi,*,lambda_factor=1.,section_id=None):
    """Exact union area/perimeter of connected axis-aligned solid rectangles.

    Rectangle tuples are (x0,x1,z0,z1); overlaps are counted once. A T/L
    section may use the declared physical flange and web. The caller supplies
    the concrete strength applicable to that same section (a minimum of its
    different constituent strengths is conservative). Hollow/disconnected
    sections are explicitly outside this helper's formula family.
    """
    out={"section_id":section_id,"status":"not_evaluated","fc_ksi":fc_ksi,"lambda":lambda_factor,
         "rectangles":rectangles,"section_type":"nonprestressed_solid"}
    if not rectangles or not _finite(fc_ksi) or fc_ksi<=0 or not _finite(lambda_factor) or not 0<lambda_factor<=1:
        return {**out,"reason":"Positive concrete strength, applicable lambda and declared physical rectangles are required."}
    if any(len(r)!=4 or not all(_finite(v) for v in r) or r[1]<=r[0] or r[3]<=r[2] for r in rectangles):
        return {**out,"status":"fail","reason":"Invalid rectangle geometry."}
    xs=sorted({v for r in rectangles for v in r[:2]});zs=sorted({v for r in rectangles for v in r[2:]})
    cells={(i,j) for i in range(len(xs)-1) for j in range(len(zs)-1)
           if any(r[0]<(xs[i]+xs[i+1])/2<r[1] and r[2]<(zs[j]+zs[j+1])/2<r[3] for r in rectangles)}
    seen=set();todo=[next(iter(cells))]
    while todo:
        p=todo.pop()
        if p in seen:continue
        seen.add(p)
        todo.extend(q for q in ((p[0]-1,p[1]),(p[0]+1,p[1]),(p[0],p[1]-1),(p[0],p[1]+1)) if q in cells and q not in seen)
    if seen!=cells:return {**out,"status":"fail","reason":"Disconnected concrete section."}
    # Flood the complement with a one-cell exterior frame to detect holes.
    empty={(i,j) for i in range(-1,len(xs)) for j in range(-1,len(zs)) if (i,j) not in cells}
    outside=set();todo=[(-1,-1)]
    while todo:
        p=todo.pop()
        if p in outside:continue
        outside.add(p)
        todo.extend(q for q in ((p[0]-1,p[1]),(p[0]+1,p[1]),(p[0],p[1]-1),(p[0],p[1]+1)) if q in empty and q not in outside)
    if outside!=empty:return {**out,"reason":"A hollow section requires the separate hollow-section code equations."}
    area=perimeter=0.
    for i,j in cells:
        dx,dz=xs[i+1]-xs[i],zs[j+1]-zs[j]
        area+=dx*dz
        perimeter+=sum(dz for p in ((i-1,j),(i+1,j)) if p not in cells)
        perimeter+=sum(dx for p in ((i,j-1),(i,j+1)) if p not in cells)
    return {**out,"status":"pass","Acp_in2":area,"Ag_in2":area,"pcp_in":perimeter,
            "geometry_basis":"exact connected solid rectangle union; shared interfaces removed"}


def threshold(section,Tu_kip_in,N_tension_kip,*,phi=PHI):
    out={"status":"not_evaluated","section_id":section.get("section_id"),"below_threshold":False,
         "Tth_kip_in":None,"phi_Tth_kip_in":None,"Tcr_kip_in":None,"phi":phi,
         "basis":"22.7.4.1(a)(c),22.7.5.1(c); full same-section axial force, compression positive in code"}
    if section.get("status")!="pass" or not all(_finite(v) for v in (Tu_kip_in,N_tension_kip,phi)) or not 0<phi<=PHI:
        return {**out,"reason":"Declared solid section and finite same-section Tu/N/phi are required."}
    root=min(100.,math.sqrt(section["fc_ksi"]*1000))
    lam=section["lambda"]
    compression=-N_tension_kip
    radicand=1+compression*1000/(4*section["Ag_in2"]*lam*root)
    out.update(axial_compression_kip=compression,axial_radicand=radicand,sqrt_fc_used_psi=root)
    if radicand<0:
        return {**out,"reason":"Axial tension leaves the uncracked threshold formula domain; no threshold exemption is established."}
    value=lam*root*section["Acp_in2"]**2/section["pcp_in"]*math.sqrt(radicand)/1000
    return {**out,"status":"pass","Tth_kip_in":value,"phi_Tth_kip_in":phi*value,"Tcr_kip_in":4*value,
            "below_threshold":abs(Tu_kip_in)<phi*value,"Tu_abs_kip_in":abs(Tu_kip_in)}


def beam_torsion_section(bw_in,h_in,slab_t_in,available_overhangs_in,fc_ksi,*,lambda_factor=1.,section_id=None):
    """9.2.4.4 torsional section for the declared flush-top beam/slab detail.

    Available overhangs are actual slab distances from each web face to its
    physical edge, not the 6.3.2 flexural effective widths. The same-section
    axial force must subsequently be recovered for this selected geometry.
    """
    values=[bw_in,h_in,slab_t_in,*[available_overhangs_in.get(k) for k in ("minus","plus")]]
    if not all(_finite(v) for v in values) or min(bw_in,h_in)<=0 or not 0<=slab_t_in<h_in or min(values[3:])<0:
        return {"status":"fail","reason":"Invalid flush-top beam/slab torsion-section geometry.","section_id":section_id}
    projection=min(h_in-slab_t_in,4*slab_t_in)
    overhang={k:min(projection,available_overhangs_in[k]) for k in ("minus","plus")}
    web=solid_section([[-bw_in/2,bw_in/2,-h_in,0.]],fc_ksi,lambda_factor=lambda_factor,section_id=section_id)
    if slab_t_in==0:return {**web,"overhangs_in":overhang,"flanges_omitted":True,"clause":"9.2.4.4"}
    composite=solid_section([[-bw_in/2,bw_in/2,-h_in,0.],
        [-bw_in/2-overhang["minus"],bw_in/2+overhang["plus"],-slab_t_in,0.]],fc_ksi,
        lambda_factor=lambda_factor,section_id=section_id)
    omitted=composite["Acp_in2"]**2/composite["pcp_in"]<web["Acp_in2"]**2/web["pcp_in"]
    return {**(web if omitted else composite),"overhangs_in":overhang,"flanges_omitted":omitted,
            "clause":"9.2.4.4","axial_force_requirement":"recover full Nu over these exact selected rectangles"}


def _perimeter_bars(coordinates,cage):
    """Map actual outer bars to nearest exact rounded-hoop arc length."""
    center=cage.get("alternation_center_in",[0.,0.])
    dt=cage["torsion_geometry"]["hoop_db_in"]
    r=cage["inside_bend_radius_in"]+dt/2
    B,H=cage["core_outside_dimensions_in"]
    cx,cy=(B-dt)/2-r,(H-dt)/2-r
    q=math.pi*r/2
    length=4*q+4*(cx+cy)
    starts=[0.,q,q+2*cx,2*q+2*cx,2*q+2*cx+2*cy,3*q+2*cx+2*cy,3*q+4*cx+2*cy,4*q+4*cx+2*cy]
    segments=[("arc",cx,cy,0.,math.pi/2),("line",cx,cy+r,-cx,cy+r),
              ("arc",-cx,cy,math.pi/2,math.pi),("line",-cx-r,cy,-cx-r,-cy),
              ("arc",-cx,-cy,math.pi,1.5*math.pi),("line",-cx,-cy-r,cx,-cy-r),
              ("arc",cx,-cy,1.5*math.pi,2*math.pi),("line",cx+r,-cy,cx+r,cy)]
    xmin,xmax=min(p["x_in"] for p in coordinates),max(p["x_in"] for p in coordinates)
    ymin,ymax=min(p["y_in"] for p in coordinates),max(p["y_in"] for p in coordinates)
    result=[]
    for p in coordinates:
        if min(abs(p["x_in"]-xmin),abs(p["x_in"]-xmax),abs(p["y_in"]-ymin),abs(p["y_in"]-ymax))>1e-6:continue
        x,y=p["x_in"]-center[0],p["y_in"]-center[1]
        candidates=[]
        for start,seg in zip(starts,segments):
            kind,a,b,c,d=seg
            if kind=="arc":
                angle=math.atan2(y-b,x-a)%(2*math.pi)
                if c==1.5*math.pi and angle<math.pi:angle+=2*math.pi
                angle=max(c,min(d,angle))
                px,py=a+r*math.cos(angle),b+r*math.sin(angle)
                station=start+r*(angle-c)
            else:
                distance=math.hypot(c-a,d-b)
                u=max(0.,min(1.,((x-a)*(c-a)+(y-b)*(d-b))/distance**2))
                px,py=a+u*(c-a),b+u*(d-b);station=start+u*distance
            candidates.append((math.hypot(x-px,y-py),station%length))
        result.append({**p,"hoop_station_in":min(candidates)[1]})
    result.sort(key=lambda p:p["hoop_station_in"])
    for i,p in enumerate(result):
        before=(p["hoop_station_in"]-result[i-1]["hoop_station_in"])%length
        after=(result[(i+1)%len(result)]["hoop_station_in"]-p["hoop_station_in"])%length
        p.update(tributary_perimeter_in=(before+after)/2,tributary_fraction=(before+after)/(2*length),next_bar_gap_in=after)
    return result


def assess_torsion(section,Tu_kip_in,N_tension_kip,*,design,coordinates,spacing_in,clear_cover_in,
                   shear_reservations=None,longitudinal_state=None,development=None,identity=None):
    """Check torsion without spending steel already allocated to N/M/V.

    ``shear_reservations`` keys vertical,inplane each require vu_kip,
    capacity_ve_kip (zero only when no such demand applies), vc_kip, d_in,
    width_in. All forces are factored except nominal Vc. The larger absolute
    actual/capacity shear is reserved. ``longitudinal_state`` is the shared
    seven-resultant nominal equilibrium state at factored demand/strength_phi;
    bars use id, nominal_stress_ksi, area_in2,fy_ksi. The torsion reserve is not
    multiplied by strength_phi a second time.
    """
    out={"method":METHOD,"source_sha256":SOURCE_SHA256,"status":"not_evaluated","identity":identity,
         "redistribution_used":False,"checks":[],"threshold":threshold(section,Tu_kip_in,N_tension_kip)}
    checks=out["checks"]
    def check(rule,value,details):
        checks.append({"rule":rule,"passes":value,"status":"not_evaluated" if value is None else "pass" if value else "fail","details":details})
    th=out["threshold"]
    if section.get("status")=="fail":
        check("torsion_physical_section_geometry",False,section)
        out["status"]="fail"
        return out
    if th["status"]!="pass":
        check("torsion_threshold_inputs_and_domain",None,th)
        return out
    if th["below_threshold"]:
        return {**out,"status":"pass","branch":"below_threshold","torsion_effects_neglected":True,
                "basis":"9.5.4.1;22.7.1.1: strict Tu < phi Tth; other N/M/V checks remain required"}
    out.update(branch="reinforcement_required",torsion_effects_neglected=False)
    if not _finite(spacing_in) or spacing_in<=0:
        check("torsion_actual_spacing",None,"Finite positive installed hoop spacing is required.")
        return out
    cage=beam_cage_geometry(design,coordinates,clear_cover_in)
    check("torsion_physical_closed_hoop",bool(cage["passes"]),cage["checks"])
    if not cage["passes"] or not cage.get("core_area_in2"):
        out["status"]="fail";return out
    clash=ph.transverse_collisions(ph._positioned_cage_segments(cage,0.,0),ph._positioned_cage_segments(cage,spacing_in,1))
    check("torsion_adjacent_actual_hoops_nonintersecting",not clash,clash[:20])
    dt=sp.rebar_diameter(design.stirrup_bar_size)
    area=sp.rebar_area(design.stirrup_bar_size)
    projection=cage["hoop_leg_horizontal_projection_factor"]
    B,H=cage["core_outside_dimensions_in"]
    r=cage["inside_bend_radius_in"]+dt/2
    aoh=(B-dt)*(H-dt)-(4-math.pi)*r*r
    ph_plan=2*(B+H-2*dt)-(8-2*math.pi)*r
    ph_upper=ph_plan/projection
    ao=.85*aoh
    fyt=min(60.,sp.FYT_KSI);fy=min(60.,sp.FY_KSI)
    geometry={"Aoh_in2":aoh,"Ao_in2":ao,"ph_plan_in":ph_plan,"ph_upper_in":ph_upper,"hoop_db_in":dt,
              "theta_deg":45.,"fy_used_ksi":fy,"fyt_used_ksi":fyt,"hoop_projection_factor":projection,
              "outer_single_leg_area_in2":area*projection}
    cage["torsion_geometry"]=geometry
    out["geometry"]=geometry
    check("torsion_spacing_9_7_6_3_3",spacing_in<=min(ph_plan/8,12.)+1e-8,
          {"provided_in":spacing_in,"maximum_in":min(ph_plan/8,12.)})
    perimeter=_perimeter_bars(coordinates,cage)
    check("torsion_longitudinal_distribution_9_7_5_1",len(perimeter)>=4 and all(p["next_bar_gap_in"]<=12.+1e-8 for p in perimeter),
          {"bars":[{"id":p["id"],"next_bar_gap_in":p["next_bar_gap_in"],"tributary_fraction":p["tributary_fraction"]} for p in perimeter]})
    check("torsion_longitudinal_diameter_9_7_5_2",sp.rebar_diameter(design.bar_size)>=max(.375,.042*spacing_in),
          {"provided_in":sp.rebar_diameter(design.bar_size),"minimum_in":max(.375,.042*spacing_in)})
    valid_shear=isinstance(shear_reservations,dict) and all(a in shear_reservations and
        all(_finite(shear_reservations[a].get(k)) for k in ("vu_kip","capacity_ve_kip","vc_kip","d_in","width_in")) and
        min(shear_reservations[a]["d_in"],shear_reservations[a]["width_in"])>0 and shear_reservations[a]["vc_kip"]>=0
        for a in ("vertical","inplane"))
    remaining=None
    if valid_shear:
        reserved={};shear_stress=0.;stress_limits=[]
        for axis,data in shear_reservations.items():
            if axis not in ("vertical","inplane"):continue
            demand=max(abs(data["vu_kip"]),abs(data["capacity_ve_kip"]))
            av_required=max(0.,demand/PHI-data["vc_kip"])*spacing_in/(fyt*data["d_in"])
            inner=(design.stirrup_legs-2)*area if axis=="vertical" else 0.
            outer_reserved=max(0.,av_required-inner)/2
            reserved[axis]={"vu_used_kip":demand,"required_all_leg_area_in2":av_required,"inner_leg_area_in2":inner,
                            "reserved_outer_single_leg_in2":outer_reserved}
            shear_stress+=demand*1000/(data["width_in"]*data["d_in"])
            stress_limits.append(PHI*(data["vc_kip"]*1000/(data["width_in"]*data["d_in"])+8*math.sqrt(section["fc_ksi"]*1000)))
        remaining=min(area*projection-v["reserved_outer_single_leg_in2"] for v in reserved.values())
        out["transverse_reservations"]=reserved
        torsion_stress=abs(Tu_kip_in)*1000*ph_upper/(1.7*aoh*aoh)
        check("torsion_shear_section_limit_22_7_7_1",math.hypot(shear_stress,torsion_stress)<=min(stress_limits)+1e-8,
              {"combined_stress_psi":math.hypot(shear_stress,torsion_stress),"limit_psi":min(stress_limits),
               "basis":"two simultaneous orthogonal shear stresses conservatively summed before the solid-section root-sum-square with torsion"})
        phi_tn_transverse=PHI*2*ao*max(0.,remaining)*fyt/spacing_in
        check("torsion_transverse_reserve_9_5_4_3_22_7_6_1a",phi_tn_transverse>=abs(Tu_kip_in)-1e-8,
              {"phi_Tn_kip_in":phi_tn_transverse,"Tu_abs_kip_in":abs(Tu_kip_in),"remaining_single_leg_in2":remaining})
        minimum=max(.75*math.sqrt(section["fc_ksi"]*1000)*design.b_in/(fyt*1000),50*design.b_in/(fyt*1000))
        check("torsion_minimum_outer_transverse_9_6_4_2",2*area*projection/spacing_in>=minimum-1e-8,
              {"provided_outer_legs_in2_per_in":2*area*projection/spacing_in,"minimum_in2_per_in":minimum})
    else:
        check("torsion_concurrent_shear_reservation",None,"Both actual and full capacity-design shear demands with applicable Vc/d/width are required.")
    longitudinal=longitudinal_state or {}
    complete=(longitudinal.get("status")=="pass" and longitudinal.get("all_concurrent_demands_included") is True and
              _finite(longitudinal.get("strength_phi")) and 0<longitudinal["strength_phi"]<=1)
    if identity:
        complete=complete and all(longitudinal.get(k)==v for k,v in identity.items() if k in
                                 ("case_id","cut_id","state_identity","recovery_id"))
    states={p.get("id"):p for p in longitudinal.get("bars",[])}
    # A zero-stress bar outside its developed domain is installed steel, not
    # unused developed torsion capacity. The shared-state provider must say
    # explicitly which physical bars participated in its strength model.
    complete=complete and all(states.get(p["id"],{}).get("strength_credited") is True for p in perimeter)
    complete=complete and all(p["id"] in states and all(_finite(states[p["id"]].get(k)) for k in
                        ("nominal_stress_ksi","area_in2","fy_ksi")) for p in perimeter)
    if complete:
        complete=(len(states)==len(longitudinal["bars"]) and all(states[p["id"]]["fy_ksi"]>0 and
                  abs(states[p["id"]]["area_in2"]-sp.rebar_area(design.bar_size))<=1e-8 for p in perimeter))
    if complete and remaining is not None:
        amin=max(0.,min(5*math.sqrt(section["fc_ksi"]*1000)*section["Acp_in2"]/(fy*1000)-
                        max(0.,remaining)/spacing_in*ph_plan*fyt/fy,
                        5*math.sqrt(section["fc_ksi"]*1000)*section["Acp_in2"]/(fy*1000)-25*design.b_in/(fyt*1000)*ph_plan*fyt/fy))
        nominal_force=max(abs(Tu_kip_in)/PHI*ph_upper/(2*ao),amin*fy)
        allocations=[]
        for p in perimeter:
            state=states[p["id"]]
            cap=min(60.,state["fy_ksi"])
            available=state["area_in2"]*max(0.,cap-abs(state["nominal_stress_ksi"]))
            required=nominal_force*p["tributary_fraction"]
            allocations.append({"id":p["id"],"available_nominal_tension_kip":available,"required_nominal_tension_kip":required,
                                "stress_reserved_ksi":abs(state["nominal_stress_ksi"]),"passes":available>=required-1e-8})
        check("torsion_longitudinal_reserve_9_5_4_3_9_6_4_3_22_7_6_1b",all(p["passes"] for p in allocations),
              {"minimum_torsion_area_in2":amin,"required_additional_nominal_tension_kip":nominal_force,
               "per_bar":allocations,"compression_zone_reduction_used":False,"strength_phi_already_in_nominal_stresses":longitudinal["strength_phi"]})
    else:
        check("torsion_same_state_longitudinal_reserve",None,"A passing same-cut/case nominal-equivalent N/M7 bar stress inventory and shear reservation are required.")
    detail=development or {}
    declared=detail.get("status")=="pass" and bool(detail.get("source_check_ids"))
    supplied=set(detail.get("longitudinal_bar_ids",[]))
    if not declared:
        check("torsion_zone_extension_and_support_development",None,"Actual 9.7.5.3/.4 and 9.7.6.3.2 zone/end development evidence is required.")
    else:
        check("torsion_zone_extension_and_support_development",all(detail.get(k) is True for k in
              ("longitudinal_extends_required_zone","transverse_extends_required_zone","both_supports_developed")) and
              {p["id"] for p in perimeter}<=supplied,detail)
    statuses=[c["status"] for c in checks]
    out["status"]="fail" if "fail" in statuses else "not_evaluated" if "not_evaluated" in statuses else "pass"
    return out
