"""Actual cut/bar development, independent of any zero-N slab acceptance flags.

Full fy is conservatively required in both tension and compression on both
sides of every selected station (25.4.1.1). Hooks supply tension only. No excess
steel reduction, invented transverse bond credit, or partial development credit
is used. A failure here is the full-fy assumption failing; it is not a proof that
every possible compatible stress field fails.
"""
from __future__ import annotations
import hashlib
import json
import math

METHOD='actual_inventory_full_fy_cut_development_aci31819_v2_explicit_local_ids'
ACI_SHA256='9cf6a9503428dc053bcf08bbbe6c9cdd0f0186ce75b60e50448778b47a18f4a0'
TOL=1e-8

def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def straight_lengths(*,db_in,bar_size,fy_ksi,fc_ksi,cb_in,concrete_below_in,
                     minimum_center_spacing_in,ktr_actual_lower_bound_in=0.,normalweight=True):
    """25.4.2.4 Ktr=0 length plus separate mandatory 25.4.2.2 confinement."""
    if fy_ksi not in (40.,60.,80.,100.) or min(db_in,fc_ksi,cb_in)<=0:
        return {'status':'unknown','reason':'Unsupported grade or unavailable actual cover/spacing'}
    root=min(math.sqrt(1000*fc_ksi),100.)
    factors={'psi_t':1.3 if concrete_below_in>12 else 1.,'psi_e':1.,
        'psi_s':.8 if bar_size<=6 else 1.,'psi_g':{40.:1.,60.:1.,80.:1.15,100.:1.3}[fy_ksi],
        'lambda':1. if normalweight else .75,'ktr_length_credit_in':0.}
    ratio=min(cb_in/db_in,2.5)
    ld=max(12.,.075*fy_ksi*1000*factors['psi_t']*factors['psi_s']*factors['psi_g']/(factors['lambda']*root*ratio)*db_in)
    # No favourable confinement reduction of compression development.
    ldc=max(.02*fy_ksi*1000/root*db_in,.0003*fy_ksi*1000*db_in,8.)
    required=.5*db_in if fy_ksi>=80 and minimum_center_spacing_in<6-TOL else 0.
    confinement=ktr_actual_lower_bound_in>=required-TOL
    return {'status':'pass' if confinement else 'fail','ld_tension_in':ld,'ld_compression_in':ldc,
        'cb_in':cb_in,'confinement_ratio':ratio,'factors':factors,
        'minimum_center_spacing_in':minimum_center_spacing_in,
        'ktr_required_25_4_2_2_in':required,'ktr_actual_lower_bound_in':ktr_actual_lower_bound_in,
        'mandatory_confinement_passes':confinement,'clauses':'25.4.1.1-.4;25.4.2.1-.5;25.4.9'}

def hook_tension(*,db_in,bar_size,fy_ksi,fc_ksi,minimum_hook_spacing_in,
                 side_cover_in,top_bottom_cover_in,radius_in,extension_in,tail_fits,
                 available_in,normalweight=True):
    """Actual standard 90 hook, no beneficial enclosing-tie credit."""
    if bar_size not in (4,5,6):return {'status':'unknown','reason':'Mat hook provider supports actual No.4/5/6 only'}
    if min(db_in,fc_ksi,radius_in,extension_in)<=0:return {'status':'unknown','reason':'Incomplete hook geometry'}
    psi_r=1. if minimum_hook_spacing_in>=6*db_in-TOL else 1.6
    psi_o=1. if side_cover_in>=6*db_in-TOL else 1.25
    psi_c=fc_ksi/15+.6 if fc_ksi<6 else 1.
    root=min(math.sqrt(1000*fc_ksi),100.);lam=1. if normalweight else .75
    ld=max(fy_ksi*1000*psi_r*psi_o*psi_c/(55*lam*root)*db_in**1.5,8*db_in,6.)
    geometric=(2*radius_in-db_in>=6*db_in-TOL and extension_in>=12*db_in-TOL and tail_fits)
    mandatory_ties=side_cover_in<2.5-TOL and top_bottom_cover_in<2.5-TOL
    # Both low covers need explicit enclosing ties, not an assumed slab exemption.
    status='fail' if not geometric or available_in<ld-TOL else 'unknown' if mandatory_ties else 'pass'
    return {'status':status,'ldh_required_in':ld,'available_in':available_in,
        'standard_hook_geometry_passes':geometric,'mandatory_enclosing_ties_unproved':mandatory_ties,
        'side_cover_in':side_cover_in,'top_bottom_cover_in':top_bottom_cover_in,
        'psi_r':psi_r,'psi_o':psi_o,'psi_c':psi_c,'lambda':lam,
        'inside_bend_diameter_in':2*radius_in-db_in,'extension_in':extension_in,
        'clauses':'25.3.1;25.4.3.1-.4; hooks do not develop compression (25.4.1.2)'}

def _status(states):
    return 'fail' if 'fail' in states else 'unknown' if any(s!='pass' for s in states) else 'pass'

def side_development(station,run,lengths,hooks):
    """Independent arithmetic on actual end coordinates; used by both owners."""
    if lengths.get('status')=='unknown':return {'status':'unknown','reason':lengths.get('reason')}
    rows=[]
    for side,available in enumerate((station-run[0],run[1]-station)):
        straight=available>=lengths['ld_tension_in']-TOL and lengths['mandatory_confinement_passes']
        hook=hooks[side] if hooks else {'status':'unknown','reason':'No supported terminal hook at this owned endpoint'}
        tension='pass' if straight or hook.get('status')=='pass' else ('unknown' if hook.get('status')=='unknown' else 'fail')
        compression=available>=lengths['ld_compression_in']-TOL
        # Missing straight mandatory transverse reinforcement is a known failure
        # for that path; a separately supported hook may still develop tension.
        rows.append({'side':'minus' if side==0 else 'plus','straight_available_in':available,
            'straight_tension_passes':straight,'hook_tension':hook,
            'tension_status':tension,'compression_status':'pass' if compression else 'fail',
            'status':_status([tension,'pass' if compression else 'fail'])})
    return {'status':_status([r['status'] for r in rows]),'station_along_bar_in':station,
            'straight_run_interval_in':list(run),'straight_required':lengths,'sides':rows}

def full_fy_developed_domain(check):
    """Where the SAME full-fy formulas pass; no stress/excess-steel reduction.

    A capacity consumer may neglect undeveloped end-zone reinforcement only
    while retaining its actual concrete void and physical quantities. This
    domain alone does not define a bond-slip/force-decay field or alter Mn/Mpr.
    """
    if not all(k in check for k in ('straight_run_interval_in','straight_required','sides','station_along_bar_in')):
        return {'status':'unknown','reason':'Actual length/path evidence is incomplete'}
    if check.get('physical_detail_status') not in (None,'pass'):
        return {'status':check['physical_detail_status'],'reason':'Underlying installed physical beam path does not pass'}
    lengths=check['straight_required'];run=check['straight_run_interval_in'];station=check['station_along_bar_in']
    if lengths.get('status')=='unknown':return {'status':'unknown','reason':'Actual straight development formula unavailable'}
    requirements=[]
    for end,side in enumerate(check['sides']):
        options=[]
        if lengths.get('mandatory_confinement_passes'):options.append(('straight',lengths['ld_tension_in']))
        hook=side.get('hook_tension',{})
        geometric=(hook.get('standard_hook_geometry_passes') is True and not hook.get('mandatory_enclosing_ties_unproved'))
        geometric=geometric or hook.get('selected_hook_context_passes') is True
        if geometric:
            extra=hook['available_in']-side['straight_available_in']
            options.append(('actual_standard_hook',max(0.,hook['ldh_required_in']-extra)))
        if not options:return {'status':'unknown','reason':'No supported full-fy tension development path for an end'}
        route,retreat=min(options,key=lambda item:item[1])
        required=max(retreat,lengths['ld_compression_in'])
        requirements.append({'side':side['side'],'tension_route':route,'tension_retreat_in':retreat,
            'compression_retreat_in':lengths['ld_compression_in'],'governing_retreat_in':required})
    interval=[run[0]+requirements[0]['governing_retreat_in'],run[1]-requirements[1]['governing_retreat_in']]
    return {'status':'computed' if interval[1]>=interval[0]-TOL else 'fail',
        'interval_in':interval,'physical_straight_run_interval_in':run,'endpoint_requirements':requirements,
        'basis':'full_fy_both_signs_both_sides_no_25_4_10_reduction',
        'void_and_quantity_ownership':'retain complete installed bar; this interval supplies strength credit only',
        'actual_demand_failure_established':False}

def _nearest(bar,bars):
    candidates=[math.hypot(float(bar['s_in'])-float(other['s_in']),float(bar['z_in'])-float(other['z_in']))
        for other in bars if other['id']!=bar['id'] and other['axis']==bar['axis'] and
        min(bar['straight_run_interval_in'][1],other['straight_run_interval_in'][1])>
        max(bar['straight_run_interval_in'][0],other['straight_run_interval_in'][0])+TOL]
    return min(candidates,default=1e12)

def _mat_station(bar,station,mat,footprint,slab_h,fc,maximum_cast_depth,all_bars,normalweight,*,nearest_center_spacing_in=None):
    db=bar['db_in'];depth=slab_h/2-bar['z_in'];axis=bar['axis'];cross='y' if axis=='x' else 'x'
    side=min(bar['s_in']-footprint[f'{cross}_min_in'],footprint[f'{cross}_max_in']-bar['s_in'])-db/2
    nearest=_nearest(bar,all_bars) if nearest_center_spacing_in is None else nearest_center_spacing_in
    cb=min(depth,slab_h-depth,side+db/2,nearest/2)
    lengths=straight_lengths(db_in=db,bar_size=mat['bar_size'],fy_ksi=bar['fy_ksi'],fc_ksi=fc,
        cb_in=cb,concrete_below_in=max(0.,maximum_cast_depth-depth),minimum_center_spacing_in=nearest,
        normalweight=normalweight)
    hooks=[]
    for end in (0,1):
        far=mat['hook_vertical_axes_in'][end]
        available=station-far+db/2 if end==0 else far-station+db/2
        hooks.append(hook_tension(db_in=db,bar_size=mat['bar_size'],fy_ksi=bar['fy_ksi'],fc_ksi=fc,
            minimum_hook_spacing_in=min(mat['spacing_in'],nearest),side_cover_in=side,
            top_bottom_cover_in=min(depth-db/2,mat['hook_tail_depth_limit_in']-mat['hook_tail_depth_in']),
            radius_in=mat['hook_radius_in'],extension_in=mat['hook_extension_in'],tail_fits=mat['hook_tail_fits'],
            available_in=available,normalweight=normalweight))
    result=side_development(station,mat['straight_run_interval_in'],lengths,hooks)
    return result['status'],result

def _beam_station(bar,station,inventory,members,all_bars,slab_h,slab_fc,normalweight,cage_cache):
    """Follow only identical, explicitly continuous cages across ownership planes."""
    import Structure_Parameters as sp
    from Design.SMRF_Splice_Physical import beam_cage_geometry,group_beam_lanes
    from Design import SMRF_Transition_Physical as ph
    from Model import Member_Groups as mg
    member=members[str(bar['member_id'])];lane_id=bar['source_lane_id']
    face,row_index=lane_id.split('_')[:2];row_index=int(row_index)
    chain=[bar];seen={bar['id']}
    def endpoint(value,end):
        geometry=inventory['beam_geometry'][str(value['member_id'])]
        proof=geometry['straight_run_evidence'];e=proof['end_evidence'][end]
        hook=next((f for f in e['hook_families'] if f['face']==face and f['row']==row_index),None)
        lane=next(r for r in proof['bars'] if r['source_lane_id']==lane_id)
        return proof,e,hook,lane
    for end in (0,1):
        while True:
            last=chain[0] if end==0 else chain[-1];proof,e,hook,lane=endpoint(last,end)
            if proof.get('geometry_complete') is not True:return {'status':'unknown','reason':'Beam endpoint path is incomplete'}
            if hook is not None:break
            coordinate=lane['member_grid_interval_in'][end]
            choices=[]
            for other in all_bars:
                if other.get('operator_owner')!='web_line' or other['id'] in seen or other['axis']!=bar['axis']:continue
                target=members[str(other['member_id'])]
                if target.group_id!=member.group_id or target.design!=member.design or other.get('source_lane_id')!=lane_id:continue
                if any(abs(other[k]-bar[k])>TOL for k in ('s_in','z_in','db_in','fy_ksi')):continue
                op,oe,oh,ol=endpoint(other,1-end)
                if abs(ol['member_grid_interval_in'][1-end]-coordinate)<=TOL and oh is None and op.get('geometry_complete') is True:choices.append(other)
            if len(choices)!=1:return {'status':'unknown','reason':'No unique same-cage continuous neighbor at a nonhooked ownership plane'}
            nxt=choices[0];seen.add(nxt['id'])
            if end==0:chain.insert(0,nxt)
            else:chain.append(nxt)
    run=[endpoint(chain[0],0)[3]['straight_run_interval_in'][0],endpoint(chain[-1],1)[3]['straight_run_interval_in'][1]]
    designs=[members[str(x['member_id'])].design for x in chain]
    depth=slab_h/2-bar['z_in'];db=bar['db_in'];nearest=_nearest(bar,all_bars)
    side=member.design.b_in/2-abs(bar['s_in']-inventory['beam_geometry'][str(bar['member_id'])]['line_in'])
    cb=min(side,depth,min(d.h_in for d in designs)-depth,nearest/2)
    key=member.group_id
    if key not in cage_cache:
        lanes=group_beam_lanes(mg.active())[key]
        cage=beam_cage_geometry(member.design,lanes['coordinates'],sp.BEAM_CLEAR_COVER_IN)
        adjacent=ph.transverse_collisions(ph._positioned_cage_segments(cage,0.,0),
            ph._positioned_cage_segments(cage,member.design.stirrup_spacing_in,1)) if cage.get('passes') else []
        # A mechanical splice may replace the nominal hoops with a wider
        # spacing while preserving TOTAL Av/s. That does not preserve the
        # two perimeter-leg splitting restraint. Use the permitted Ktr=0
        # lower bound over the full actual bar run; never inherit nominal
        # hoop confinement through a changed splice zone.
        ktr=0.
        cage_cache[key]=(ktr,cage.get('passes') and not adjacent)
    ktr,cage_pass=cage_cache[key]
    fc=min([slab_fc]+[d.fc_ksi for d in designs])
    lengths=straight_lengths(db_in=db,bar_size=member.design.bar_size,fy_ksi=bar['fy_ksi'],fc_ksi=fc,cb_in=cb,
        concrete_below_in=max(d.h_in for d in designs)-depth,minimum_center_spacing_in=nearest,
        ktr_actual_lower_bound_in=ktr,normalweight=normalweight)
    hooks=[];physical=[endpoint(value,0)[0].get('physical_detail_status') for value in chain]
    for end,value in enumerate((chain[0],chain[-1])):
        proof,e,hook,lane=endpoint(value,end);physical.append(proof.get('physical_detail_status'))
        radius=hook['bend_radius_in'];outer=run[end]+(-1 if end==0 else 1)*(radius+db/2)
        available=station-outer if end==0 else outer-station
        required=hook['required_ldh_in']
        status='pass' if available>=required-TOL else 'fail'
        if e.get('beam_context_status')!='pass':status='fail' if e.get('beam_context_status')=='fail' else 'unknown'
        hooks.append({'status':status,'available_in':available,'ldh_required_in':required,
            'selected_hook_context_passes':e.get('beam_context_status')=='pass',
            'source_evidence_id':proof['evidence_id'],'joint_id':e['joint_id'],'selected_family':hook,
            'clause':'18.8.5.1 actual selected standard hook; no artificial shortening of its critical datum'})
    result=side_development(station,run,lengths,hooks)
    result.update(continuous_physical_bar_owners=[x['id'] for x in chain],actual_hoop_ktr_lower_bound_in=ktr,
        transverse_geometry_passes=bool(cage_pass),physical_detail_status=_status(physical))
    result['status']=_status([result['status']]+physical)
    return result

def actual_bar_development(state,inventory,cut_specs,*,slab_fc_ksi,normalweight=True):
    """Per-cut records scoped to the exact actual material identity and geometry.

    cut_specs contains floor, axis, station_in, interval_in,
    axial_accounting_interval_in and optional physical_bar_ids/member_id.
    Alternatively bar_selection_basis='explicit_local_physical_ids' requires
    the exact nonempty bar IDs selected by the local material consumer. That
    branch computes end development for those physical bars without imposing
    an unrelated axial averaging window. It supplies no material/action or
    continuous-coverage qualification to that consumer.
    Transverse mat bars are checked along their OWN run at both clipped strip
    boundaries. No state can inherit another cut's developed-bar assertion.
    """
    import Structure_Parameters as sp
    from Model import Member_Groups as mg
    from Design.SMRF_Joint_Mat_Physical import global_mat_placement
    if not isinstance(cut_specs,(list,tuple)) or not cut_specs:raise ValueError('Explicit nonempty cut inventory required')
    if state.identity()!=inventory.get('state_identity') or mg.active().identity()!=state.identity():raise ValueError('Actual installed state identity differs')
    if inventory.get('identity_sha256')!=_hash({k:v for k,v in inventory.items() if k!='identity_sha256'}):raise ValueError('Material inventory identity does not match its content')
    placement=global_mat_placement(state)
    if placement.get('layout_sha256')!=inventory.get('layout_sha256') or float(sp.SLAB_THICKNESS_IN)!=inventory['slab_thickness_in']:raise ValueError('Actual installed mat layout/thickness differs')
    if not isinstance(slab_fc_ksi,(int,float)) or not math.isfinite(slab_fc_ksi) or slab_fc_ksi<=0:raise ValueError('Actual slab concrete strength is required')
    if normalweight is not True:raise ValueError('Current installed floor inventory is qualified for declared normalweight concrete only')
    members={str(m.member_tag):m for m in state.members() if not m.is_column};cage_cache={}
    output={'method':METHOD,'primary_aci_sha256':ACI_SHA256,'state_identity':state.identity(),
        'material_inventory_id':inventory['identity_sha256'],'cuts':[],
        'basis':'full_fy_sufficiency','full_fy_only':True,'actual_demand_failure_established':False,
        'stress_basis':'Full actual fy in tension and compression, both sides; no excess-steel reduction or phi multiplier'}
    for spec in cut_specs:
        floor=str(int(spec['floor']));axis=spec['axis'];station=float(spec['station_in'])
        lo,hi=map(float,spec['interval_in'])
        selection=spec.get('bar_selection_basis','finite_axial_window')
        if selection not in ('finite_axial_window','explicit_local_physical_ids'):raise ValueError('Explicit supported physical bar selection basis required')
        local=selection=='explicit_local_physical_ids'
        requested_ids=spec.get('physical_bar_ids',[])
        if local and (not isinstance(requested_ids,(list,tuple)) or not requested_ids or
                      any(not isinstance(v,str) or not v for v in requested_ids) or len(set(requested_ids))!=len(requested_ids)):
            raise ValueError('Local development requires unique nonempty physical bar IDs')
        if axis not in ('x','y') or not all(math.isfinite(x) for x in (station,lo,hi)) or hi<=lo:raise ValueError('Invalid physical cut geometry')
        if local:
            if spec.get('axial_accounting_interval_in') is not None:raise ValueError('Local physical IDs do not define an axial averaging window')
            window=None
        else:
            a,b=map(float,spec['axial_accounting_interval_in'])
            if not all(math.isfinite(x) for x in (a,b)) or b<=a or not a<=station<=b:raise ValueError('Invalid physical cut geometry')
            window=[a,b]
        bars=inventory['floors'][floor]['bars'];actual=placement.get('floors',{}).get(floor,{})
        selected=[]
        requested=set(requested_ids)
        for bar in bars:
            if requested and bar['id'] not in requested:continue
            db=bar['db_in'];s=bar['s_in'];run=bar['straight_run_interval_in']
            if bar['axis']==axis:
                if not run[0]-TOL<=station<=run[1]+TOL or not lo+db/2-TOL<=s<=hi-db/2+TOL:continue
                stations=[station]
            else:
                if not local and not a+db/2-TOL<=s<=b-db/2+TOL:continue
                start,end=max(lo,run[0]),min(hi,run[1])
                if end<=start:continue
                stations=[start,end]
            evidence={'method':METHOD,'state_identity':state.identity(),'material_inventory_id':inventory['identity_sha256'],
                'basis':'full_fy_sufficiency','full_fy_only':True,'actual_demand_failure_established':False,
                'floor':int(floor),'axis':axis,'interval_in':[lo,hi],'axial_accounting_interval_in':window,
                'bar_selection_basis':selection,
                'station_interval_in':[station,station],'physical_bar_ids':[bar['id']],
                'bar_axis':bar['axis'],'critical_stations_along_bar_in':stations,'checks':[]}
            if bar['operator_owner']=='slab':
                mat=actual.get('layers',{}).get(bar.get('mat_name',''))
                if mat is None or placement.get('status')!='pass':
                    evidence.update(status='fail' if placement.get('status')=='fail' else 'unknown',reason='Actual installed mat geometry is not complete and passing')
                elif mat['bar_size'] not in (4,5,6):evidence.update(status='unknown',reason='Mat development provider supports retained No.4/5/6 only')
                else:
                    floor_members=[m for m in members.values() if m.story_or_floor==int(floor)]
                    fc=min([slab_fc_ksi]+[m.design.fc_ksi for m in floor_members])
                    depth=max([inventory['slab_thickness_in']]+[m.design.h_in for m in floor_members])
                    statuses=[]
                    for where in stations:
                        status,record=_mat_station(bar,where,mat,actual['footprint'],inventory['slab_thickness_in'],fc,depth,bars,normalweight)
                        statuses.append(status);evidence['checks'].append(record)
                    evidence['status']=_status(statuses)
                    evidence['conservative_material_basis']={'fc_ksi':fc,'maximum_actual_cast_depth_in':depth,'ktr_length_credit_in':0.}
            else:
                if bar['axis']!=axis:
                    evidence.update(status='unknown',reason='Crossing beam/joint is not a slab transverse row')
                else:
                    for where in stations:evidence['checks'].append(_beam_station(bar,where,inventory,members,bars,
                        inventory['slab_thickness_in'],slab_fc_ksi,normalweight,cage_cache))
                    evidence['status']=_status([r['status'] for r in evidence['checks']])
            domains=[full_fy_developed_domain(check) for check in evidence['checks']]
            if domains and all(d['status']=='computed' for d in domains):
                domain={**domains[0],'interval_in':[max(d['interval_in'][0] for d in domains),min(d['interval_in'][1] for d in domains)]}
                if domain['interval_in'][1]<domain['interval_in'][0]-TOL:domain['status']='fail'
                evidence['full_fy_development_domain']=domain
            else:evidence['full_fy_development_domain']={'status':'unknown','station_domains':domains}
            evidence['evidence_id']=_hash(evidence);selected.append(evidence)
        missing=sorted(requested-{x['physical_bar_ids'][0] for x in selected})
        record={'specification':dict(spec),'evidence':selected,'missing_physical_bar_ids':missing,
            'status':_status([e['status'] for e in selected]+(['unknown'] if missing or not selected else []))}
        output['cuts'].append(record)
    output['status']=_status([c['status'] for c in output['cuts']]);output['evidence_id']=_hash(output)
    return output
