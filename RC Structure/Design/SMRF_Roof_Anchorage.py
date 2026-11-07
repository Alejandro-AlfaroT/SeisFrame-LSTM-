"""Roof column termination with pinned Class HA heads and physical cages."""
from __future__ import annotations

import math

from Design import SMRF_Transition_Physical as ph
from Design.SMRF_Transitions import bar_coordinates,_BAR

HEAD_SOURCE_URL="https://forms.iapmo.org/ues_reports/reports/er_0188.pdf"
HEAD_SOURCE_SHA256="8dd5ff5c0ac447bf988bac7b8d6a61388fd6298e94d69c5be17221b73ff16200"
_D6={4:("EL12D6",1.375,.5625),5:("EL16D6",1.5,.875),6:("EL20D6",1.875,1.125),
     7:("EL22D6",2.,1.25),8:("EL25D6",2.25,1.375),9:("EL28D6",2.75,1.5),
     10:("EL32D6",3.,1.5625),11:("EL36D6",3.25,1.6875)}


def straight_roof_development_bound(design,*,extension_in,joint_depth_in,fy_ksi,clear_cover_in,normalweight=True):
    """Chapter25 straight development on actual coordinates, Ktr=0.

    Vertical uncoated bars use psi_t=psi_e=1. No excess-steel reduction or
    seismic stress multiplier is invented. This independent general check
    does not replace the special reversed-loading anchorage provisions.
    """
    points=bar_coordinates(design,clear_cover_in)
    db=_BAR[int(design.bar_size)][0]
    side=min(min(design.b_in/2-abs(p['x_in']),design.h_in/2-abs(p['y_in'])) for p in points)
    separation=min(math.hypot(a['x_in']-b['x_in'],a['y_in']-b['y_in'])
                   for i,a in enumerate(points) for b in points[i+1:])
    cb=min(side,separation/2)
    grade={60.:1.,80.:1.15}
    if float(fy_ksi) not in grade or cb<=0:
        return {'status':'not_evaluated','reason':'Unsupported steel grade or invalid physical cb geometry'}
    root=min(math.sqrt(design.fc_ksi*1000),100.)
    lam=1. if normalweight else .75
    ratio=min(cb/db,2.5)
    ld=max(12.,3/40*fy_ksi*1000*grade[float(fy_ksi)]*(.8 if design.bar_size<=6 else 1.)/(lam*root)/ratio*db)
    available=joint_depth_in+extension_in-clear_cover_in-db/2
    return {'status':'pass' if available>=ld-ph.TOL else 'fail','ld_in':ld,'available_in':available,
        'cb_in':cb,'half_minimum_center_spacing_in':separation/2,'minimum_centroid_cover_in':side,
        'ktr_in':0.,'ktr_basis':'25.4.2.4 permits zero as a design simplification; no transverse bond credit',
        'confinement_ratio':ratio,'psi_t':1.,'psi_e':1.,'psi_g':grade[float(fy_ksi)],'stress_ksi':fy_ksi,
        'critical_section_z_in':-joint_depth_in,'bar_tip_z_in':extension_in-clear_cover_in-db/2,
        'clause':'25.4.2.1;25.4.2.4;Table25.4.2.5; necessary general straight-bar check, not seismic substitution'}


def verify_declared_roof(design,*,extension_in,joint_depth_in,fy_ksi,clear_cover_in,spacing_in,specification,
                         physical_context,axial_max_kip,normalweight=True):
    """Keep the existing roof extension; never terminate its bars at z=0."""
    if extension_in<=0:
        return verify_roof_anchorage(design,joint_depth_in=joint_depth_in,fy_ksi=fy_ksi,clear_cover_in=clear_cover_in,
            spacing_in=spacing_in,specification=specification,physical_context=physical_context,axial_max_kip=axial_max_kip,
            normalweight=normalweight)
    from Design.SMRF_Transitions import column_transition
    result=column_transition(design,design,fy_ksi=fy_ksi,clear_cover_in=clear_cover_in,
        upper_clear_height_in=extension_in,joint_depth_in=joint_depth_in,lower_hoop_spacing_in=spacing_in,
        physical_context={**physical_context,"axial_max_kip":axial_max_kip,"upper_hoop_spacing_in":spacing_in})
    out={**result["physical_detail"],"method":"declared_roof_extension_straight_anchorage_v1",
         "termination_type":"straight_bar_at_declared_extension_top","extension_in":extension_in,
         "extension_concrete_in3":design.b_in*design.h_in*extension_in,"net_head_steel_volume_in3":0.}
    out["checks"]=list(out["checks"])
    def check(rule,passes,detail):
        out["checks"].append({"rule":rule,"status":"not_evaluated" if passes is None else "pass" if passes else "fail",
                              "passes":passes,"detail":detail})
    db,area=_BAR[int(design.bar_size)]
    dt,tie_area=_BAR[int(design.stirrup_bar_size)]
    bars=bar_coordinates(design,clear_cover_in)
    cage=ph.rectangular_cage_geometry(design,bars,clear_cover_in)
    tip=extension_in-clear_cover_in-db/2
    stations=[]
    joint_sets=out.get("tie_sets") or []
    if joint_sets and cage['passes']:
        orientation=joint_sets[-1].get('fabrication_orientation') or {}
        cage=ph.oriented_cage(cage,**orientation)
    if joint_sets and cage["passes"]:
        z=max(v["z_in"] for v in joint_sets)+spacing_in
        while z+cage["set_height_in"]+dt/2<=tip+ph.TOL:
            stations.append(z)
            z+=spacing_in
    last=stations[-1] if stations else max((v["z_in"] for v in joint_sets),default=None)
    clashes=[]
    if joint_sets and cage["passes"]:
        sequence=[max(v["z_in"] for v in joint_sets)]+stations
        for i,(a,b) in enumerate(zip(sequence,sequence[1:])):
            clashes+=ph.transverse_collisions(ph._positioned_cage_segments(cage,a,len(joint_sets)-1+i),
                                             ph._positioned_cage_segments(cage,b,len(joint_sets)+i))
    check("roof_extension_adjacent_transverse_sets",not clashes,clashes[:20])
    check("roof_extension_is_actual_full_column_depth",extension_in>=max(design.b_in,design.h_in)-ph.TOL,
          {"declared_extension_in":extension_in,"required_for_15_2_6_in":max(design.b_in,design.h_in)})
    check("roof_extension_has_owned_transverse_sets",last is not None and cage["passes"] and
          tip-(last+cage.get("set_height_in",0.))<=spacing_in+ph.TOL,
          {"bar_tip_z_in":tip,"extension_set_stations_in":stations,"last_station_in":last,"maximum_end_interval_in":spacing_in})
    length=joint_depth_in+tip
    general=straight_roof_development_bound(design,extension_in=extension_in,joint_depth_in=joint_depth_in,
        fy_ksi=fy_ksi,clear_cover_in=clear_cover_in,normalweight=normalweight)
    check('roof_general_straight_development',None if general['status']=='not_evaluated' else general['status']=='pass',general)
    confined=max(0.,min(length,joint_depth_in+(last if last is not None else -joint_depth_in)))
    effective=confined+(length-confined)/1.6
    if int(design.bar_size)<=11:
        root=min(math.sqrt(design.fc_ksi*1000),100.)
        lam=1. if normalweight else .75
        ldh=max(fy_ksi*1000*db/(65*lam*root),(8 if normalweight else 10)*db,6. if normalweight else 7.5)
        ld=3.25*ldh  # conservative 18.8.5.3(b); no favourable cast-depth assertion
        ldc=max(.02*fy_ksi*1000/root*db,.0003*fy_ksi*1000*db,8.)
        check("roof_straight_bar_tension_and_compression_development",effective>=ld-ph.TOL and length>=ldc-ph.TOL,
              {"ld_required_in":ld,"ldc_required_in":ldc,"available_in":length,"confined_length_in":confined,
               "effective_length_in":effective,"outside_core_multiplier":1.6,"clauses":"18.8.5.3(b);18.8.5.4;25.4.9"})
    else:
        check("roof_straight_bar_tension_and_compression_development",None,
              "18.8.5.3 provides the seismic straight-bar formula only through No11; no primary-supported larger-bar roof termination detail is supplied.")
    out["extension_transverse_sets"]=[{"z_in":z,"transverse_length_in":cage.get("total_length_in",0.)} for z in stations]
    out["extension_longitudinal_steel_in3"]=len(bars)*area*max(0.,tip)
    out["extension_transverse_steel_in3"]=len(stations)*cage.get("total_length_in",0.)*tie_area
    states=[v["status"] for v in out["checks"]]
    out["status"]="fail" if "fail" in states else "not_evaluated" if "not_evaluated" in states else "pass"
    out["failure_rules"]=[v["rule"] for v in out["checks"] if v["status"]=="fail"]
    out["unknown_rules"]=[v["rule"] for v in out["checks"] if v["status"]=="not_evaluated"]
    return out


def head_product(bar_size,fy_ksi,specification):
    out={"catalog_id":"nvent_D6_ER0188_20250925","source_url":HEAD_SOURCE_URL,"source_sha256":HEAD_SOURCE_SHA256,
         "report_revision":"2025-09-25","report_valid_through":"2027-09-30","status":"not_evaluated"}
    if fy_ksi!=60. or specification!="ASTM A706 Grade 60" or bar_size not in _D6:
        return {**out,"reason":"ER-0188 Table1 footnote requires Chapter17 or an approved alternative for Grade80/No>11; no such evidence is supplied. Exact A706Grade60 No4–11 is the pinned simple route."}
    part,diameter,length=_D6[bar_size]
    return {**out,"status":"pass","part_number":part,"diameter_in":diameter,"length_in":length,
            "net_bearing_area_ratio_min":4.,"astm_class":"A970-18 Class HA",
            "thread_tip_projection_bound_in":2*3.5/25.4,
            "projection_basis":"Table2 permits +/-2 threads; section4.1 maximum stated pitch3.5mm conservatively bounds all sizes.",
            "installation":"Taper threads prepared by an ERICO-approved fabricator; installation and inspection per ER-0188. No construction approval asserted."}


def headed_development(design,coordinates,fy_ksi):
    db=_BAR[int(design.bar_size)][0]
    clear=min(min(design.b_in/2-abs(p["x_in"]),design.h_in/2-abs(p["y_in"]))-db/2 for p in coordinates)
    center=min(math.hypot(a["x_in"]-b["x_in"],a["y_in"]-b["y_in"]) for i,a in enumerate(coordinates) for b in coordinates[i+1:])
    psi_p=1. if center>=6*db-ph.TOL else 1.6  # no unverified parallel-tie credit
    psi_o=1. if clear>=2.5-ph.TOL else 1.25
    psi_c=min(1.,design.fc_ksi*1000/15000+.6)
    root=min(math.sqrt(design.fc_ksi*1000),100.)
    tension=max(1.25*fy_ksi*1000*psi_p*psi_o*psi_c/(75*root)*db**1.5,8*db,6.)
    compression=max(.02*fy_ksi*1000/root*db,.0003*fy_ksi*1000*db,8.)
    return {"tension_ldt_in":tension,"compression_ldc_in":compression,"psi_e":1.,"psi_p":psi_p,"psi_o":psi_o,
            "psi_c":psi_c,"bar_clear_cover_in":clear,"bar_center_spacing_in":center,
            "bar_cover_passes":clear>=2*db-ph.TOL,"bar_spacing_passes":center>=3*db-ph.TOL,
            "basis":"18.8.5.2 at1.25fy;25.4.4.1--.3;compression25.4.9 with no head/confinement reduction credit"}


def verify_roof_anchorage(design,*,joint_depth_in,fy_ksi,clear_cover_in,spacing_in,specification,
                          physical_context,axial_max_kip,normalweight=True):
    product=head_product(design.bar_size,fy_ksi,specification)
    out={"method":"roof_column_D6_physical_v1","status":product["status"],"product":product,"checks":[],
         "heads":[],"tie_sets":[],"additional_transverse_steel_volume_in3":0.,"net_head_steel_volume_in3":0.}
    if product["status"]!="pass":
        return out
    def check(rule,passes,detail):
        out["checks"].append({"rule":rule,"status":"not_evaluated" if passes is None else "pass" if passes else "fail",
                              "passes":passes,"detail":detail})
    bars=[{**p,"id":f"{p['face']}_{p['index']}"} for p in bar_coordinates(design,clear_cover_in)]
    db,area=_BAR[int(design.bar_size)]
    dt,tie_area=_BAR[int(design.stirrup_bar_size)]
    diameter,length=product["diameter_in"],product["length_in"]
    development=headed_development(design,bars,fy_ksi)
    check("roof_head_code_conditions",normalweight and development["bar_cover_passes"] and development["bar_spacing_passes"],development)
    head_cover=min(min(design.b_in/2-abs(p["x_in"]),design.h_in/2-abs(p["y_in"]))-diameter/2 for p in bars)
    check("roof_head_outside_device_cover",head_cover>=clear_cover_in-ph.TOL,{"provided_in":head_cover,"required_in":clear_cover_in})
    head_clear=development["bar_center_spacing_in"]-diameter
    check("roof_head_device_clearance",head_clear>=max(1.5,1.5*db)-ph.TOL,{"provided_in":head_clear,"required_in":max(1.5,1.5*db)})
    complete=physical_context.get("beam_geometry_complete") is True
    check("roof_actual_beam_and_slab_geometry",True if complete else None,physical_context.get("status"))
    obstacles=physical_context.get("beam_segments",[])+physical_context.get("placement_obstacle_segments",[])
    high=axial_max_kip is not None and axial_max_kip>.3*design.b_in*design.h_in*design.fc_ksi or design.fc_ksi>10.
    cage=ph.rectangular_cage_geometry(design,bars,clear_cover_in,high_axial=high)
    check("roof_actual_transverse_cage",cage["passes"],cage["checks"])
    if not cage["passes"]:
        out["status"]="fail"
        return out
    # The final cage is wound downward: its crossties remain around the
    # straight bar shanks below the bearing faces, not through the heads.
    for segment in cage["segments"]:
        for end in ("start_in","end_in"):
            segment[end][2]*=-1
    top=-clear_cover_in-product["thread_tip_projection_bound_in"]
    nominal_bearing=top-length
    required=max(development["tension_ldt_in"],development["compression_ldc_in"])
    check("roof_head_development_fits",joint_depth_in+nominal_bearing>=required-ph.TOL,
          {**development,"available_to_bearing_face_in":joint_depth_in+nominal_bearing})
    chosen=None
    for trial in range(17):
        shift=trial*spacing_in/16
        bearing=nominal_bearing-shift
        if joint_depth_in+bearing<required-ph.TOL:
            continue
        heads=[{"id":f"head/{p['id']}","center_xy_in":[p["x_in"],p["y_in"]],"z_interval_in":[bearing,top-shift],
                "diameter_in":diameter,"part_number":product["part_number"]} for p in bars]
        if any(ph.segment_cylinder_clearance(s,h["center_xy_in"],h["z_interval_in"],diameter/2)<-ph.TOL for h in heads for s in obstacles):
            continue
        for phase in range(17):
            top_station=bearing-dt/2-phase*spacing_in/16
            # Bound the last concrete-face interval conservatively by s.
            if -top_station>spacing_in+ph.TOL:
                continue
            stations=[]
            z=top_station
            while z-cage["set_height_in"]-dt/2>=-joint_depth_in-ph.TOL:
                stations.append(z)
                z-=spacing_in
            if not stations or stations[-1]-cage["set_height_in"]+joint_depth_in>spacing_in+ph.TOL:
                continue
            sets=[ph._positioned_cage_segments(cage,z,i) for i,z in enumerate(stations)]
            if any(ph.transverse_collisions(a,b) for a,b in zip(sets,sets[1:])):
                continue
            ties=[s for group in sets for s in group]
            if any(ph._clearance(s,o)<-ph.TOL for s in ties for o in obstacles):
                continue
            if any(ph.segment_cylinder_clearance(s,h["center_xy_in"],h["z_interval_in"],diameter/2)<-ph.TOL for h in heads for s in ties):
                continue
            chosen=heads,stations,bearing
            break
        if chosen:
            break
    check("roof_heads_and_actual_downward_wound_hoops",chosen is not None if complete else None,
          {"head_elevations":17,"hoop_phases_per_elevation":17,"end_interval_limit_in":spacing_in,
           "basis":"All hook owners remain straight through the complete inverted set; heads are above every transverse bar."})
    ach=cage["core_area_in2"]
    bc=cage["core_outside_dimensions_in"]
    ag=design.b_in*design.h_in
    fyt=float(physical_context.get("transverse_fy_ksi",60.))
    ratio=max(.3*(ag/ach-1)*design.fc_ksi/fyt,.09*design.fc_ksi/fyt)
    if high and axial_max_kip is not None:
        n=len(bars)
        ratio=max(ratio,.2*max(1.,design.fc_ksi/25+.6)*n/max(1,n-2)*max(0.,axial_max_kip)/(fyt*ach))
    legs=design.stirrup_legs_by_direction or (design.stirrup_legs,)*2
    av=[(n-2+2*cage["hoop_leg_horizontal_projection_factor"])*tie_area for n in legs]
    demand=[ratio*spacing_in*v for v in bc]
    check("roof_actual_core_confinement",all(a>=b-ph.TOL for a,b in zip(av,demand)) if axial_max_kip is not None else None,
          {"ash_provided_in2":av,"ash_required_in2":demand,"core_area_in2":ach,"axial_max_kip":axial_max_kip})
    if chosen:
        heads,stations,bearing=chosen
        out.update(heads=heads,tie_sets=[{"z_in":z,"winding":"downward","transverse_length_in":cage["total_length_in"]} for z in stations],
                   minimum_joint_confinement_core={"area_in2":ach,"outside_dimensions_in":bc},
                   development={**development,"provided_in":joint_depth_in+bearing})
        out["additional_transverse_steel_volume_in3"]=max(0.,(len(stations)-math.ceil(joint_depth_in/spacing_in))*cage["total_length_in"]*tie_area)
        # Baseline column bars run to the roof surface. Replace the head
        # interval once and remove the declared uncovered top bar length.
        out["net_head_steel_volume_in3"]=len(bars)*(math.pi*diameter**2/4*length-area*length-area*(clear_cover_in+shift))
    statuses=[c["status"] for c in out["checks"]]
    out["status"]="fail" if "fail" in statuses else "not_evaluated" if "not_evaluated" in statuses else "pass"
    out["failure_rules"]=[c["rule"] for c in out["checks"] if c["status"]=="fail"]
    return out
