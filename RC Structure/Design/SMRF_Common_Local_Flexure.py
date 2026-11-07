"""Local slab-only support transfer on the actual concurrent seven-action cuts.

ACI318-19 8.4.2.2.1-.6: the required signed flexural transfer is concentrated
inside column width+3h, clipped by free edges. Preserve each physical section
state. An additional design guard supplies only a signed transfer shortfall,
equally across the existing support faces. This is a necessary local design
check, not a new equilibrated FE solution or moment redistribution claim.
Beam-owned steel displaces concrete but supplies no local slab resistance.
"""
from __future__ import annotations
import copy
import math
import numpy as np
from Design.SMRF_Coupled_Floor_Design import _hash
from Design.SMRF_Physical_Floor_Cuts import common_strip_cut,bar_clear_interval
from Design.SMRF_Primary_Moment_Guards import physical_cut_states

METHOD='actual_concurrent_slab_only_support_transfer_v3_explicit_local_binding'


def inward_bar_clear_band(interval,bars):
    """Largest retained interval after moving cut boundaries INTO the code band."""
    a,b=map(float,interval)
    circles=[(r['s_in']-r['db_in']/2,r['s_in']+r['db_in']/2) for r in bars]
    for _ in range(len(circles)+1):
        prior=(a,b)
        for lo,hi in circles:
            if lo+1e-9<a<hi-1e-9:a=hi+1e-8
            if lo+1e-9<b<hi-1e-9:b=lo-1e-8
        if prior==(a,b):break
    if b<=a:raise ValueError('No bar-clear positive local band remains inside the code effective width')
    return [a,b]


def transfer_shortfall(moments,outward_normals,required):
    """One signed local couple, not a sum of independently credited capacities.

    The section moments use their positive global normal. For Y sections the
    caller converts physical Mx to -Myy before calling. Equal face increments
    keep the two-face mean moment unchanged. The physical state is retained
    even where one guard moment has smaller magnitude than its physical one.
    """
    m=np.asarray(moments,float);n=np.asarray(outward_normals,float);required=float(required)
    if m.ndim!=1 or len(m) not in (1,2) or m.shape!=n.shape or not np.isfinite(m).all() or not math.isfinite(required):
        raise ValueError('Finite actual one/two-face moments and signed transfer required')
    if any(v not in (-1.,1.) for v in n) or len(set(n))!=len(n):raise ValueError('Unique actual outward face normals required')
    actual=float(n@m);sign=1. if required>=0 else -1.
    missing=sign*max(0.,abs(required)-sign*actual);increments=n*missing/len(m)
    guarded=m+increments;got=float(n@guarded)
    if sign*got<abs(required)-1e-8:raise ValueError('Local required signed flexural couple did not close')
    return dict(physical_face_moments_kipin=m.tolist(),outward_normals=n.tolist(),
        required_transfer_kipin=required,physical_transfer_kipin=actual,
        added_transfer_kipin=missing,face_increments_kipin=increments.tolist(),
        guarded_face_moments_kipin=guarded.tolist(),guarded_transfer_kipin=got,
        residual_kipin=got-actual-missing,physical_states_retained=True,
        capacity_summed_between_faces=False,physical_equilibrium_solution_claimed=False)


def one_way_from_development(inventory,raw_cut,development,*,fc_ksi):
    """Face shear; no d-away reduction, and the actual concurrent slab axial N.

    Both faces separately supply a sufficient bound for reversal. If that
    does not pass, all installed longitudinal steel at the largest actual d
    supplies a relaxed upper bound. Insufficient conservative face credit
    alone cannot prove a physical shear failure.
    """
    from Design.SMRF_Common_Local_Checks import _bars,credited_steel
    from Design.SMRF_Slab_Local_Shear import check_one_way_strength
    spec=raw_cut['specification'];axis=spec['axis'];j=0 if axis=='x' else 1
    material=inventory['floors'][str(spec['floor'])];h=inventory['slab_thickness_in'];rows=[]
    for face in ('top','bottom'):
        bars=_bars(material,axis,face,spec['interval_in'],spec['station_in'])
        credit=credited_steel(bars,development['evidence'],spec['station_in'])
        if not bars:return dict(status='unknown',reason='No actual local tension mat face',specification=spec)
        depth=min(h/2+abs(b['z_in']) for b in bars)
        data=dict(fc_ksi=fc_ksi,slab_thickness_in=h,cut_width_in=spec['interval_in'][1]-spec['interval_in'][0],
            effective_depth_in=depth,n_tension_kip=raw_cut['physical_slab_integral8'][j],
            q_vertical_kip=raw_cut['physical_slab_integral8'][6+j])
        proved=check_one_way_strength(**data,developed_tension_area_in2=credit['provided_developed_in2'])
        upper=check_one_way_strength(**data,developed_tension_area_in2=credit['installed_area_in2'])
        status='pass' if proved['status']=='pass' else 'fail' if upper['status']=='fail' else 'unknown'
        rows.append(dict(face=face,status=status,credit=credit,developed_check=proved,installed_steel_upper_check=upper))
    allbars=[b for b in material['bars'] if b['operator_owner']=='slab' and b['axis']==axis and
        spec['interval_in'][0]+b['db_in']/2-1e-9<=b['s_in']<=spec['interval_in'][1]-b['db_in']/2+1e-9 and
        b['straight_run_interval_in'][0]-1e-9<=spec['station_in']<=b['straight_run_interval_in'][1]+1e-9]
    upper=check_one_way_strength(**dict(data,effective_depth_in=max(h/2+abs(b['z_in']) for b in allbars)),
        developed_tension_area_in2=sum(b['area_in2'] for b in allbars))
    status='pass' if all(r['developed_check']['status']=='pass' for r in rows) else 'fail' if upper['status']=='fail' else 'unknown'
    return dict(status=status,specification=spec,faces=rows,relaxed_all_installed_steel_upper_check=upper,
        raw_source_sha256=raw_cut['source_sha256'],d_away_support_reduction_used=False)


def _projected_slab(projector,projection,interval,sref):
    """Exact slab part of the zero-total-N full-T demand field; no web credit."""
    a,b=max(interval[0],projector.flange[0]),min(interval[1],projector.flange[1]);v=np.zeros(7)
    if b<=a:return v
    e,kz,ks=projection['generalized_normal_strain'];j=projector.j
    n0=projector.ef*projector.t*(e-ks*projector.sref);n1=projector.ef*projector.t*ks
    v[j]=n0*(b-a)+n1*(b*b-a*a)/2
    v[j+3]=projector.ef*projector.t**3/12*kz*(b-a)
    v[6]=n0*(b*b-a*a)/2+n1*(b**3-a**3)/3-sref*v[j]
    return v


def _guard(base,resultants,proof,name):
    r=copy.deepcopy(base);r['resultants']=list(map(float,resultants))
    # A physical window is not the axial law of a newly prescribed face guard.
    r.pop('axial_action_window',None)
    r['identity']['recovery_id']+='|'+name;r['source_identities'].append(dict(r['identity']))
    r['strength_guard']=proof;r['accounting'].update(strength_only_design_guard=True,
        physical_equilibrium_solution_claimed=False,equilibrium_scope='physical_base_and_prescribed_local_section_demand',
        evidence_id=_hash(proof))
    return r


def plan_local_transfer(common,state,*,selected_floor_case_pair,maximum_cuts,axial_accounting_length_in=24.,prepared_sources=None,local_pitch_actions=False):
    """Reconstruct every support/axis/trace local state in one explicit pair.

    Original physical, primary strong/weak/both, all seven P5 membrane, and
    simultaneous primary/P5 strong/both branches use the same slab-only bars.
    Every branch retains its complete physical transverse membrane/bending/
    twist actions. The signed code flexural shortfall is checked in addition.
    """
    from Design.SMRF_Common_Action_Analysis import physical_response_bundles
    from Design.SMRF_Common_Guard_Plan import _PhysicalSource,primary_guard_increment
    from Design.SMRF_Physical_Action_Readout import installed_full_t_projector
    from Design.SMRF_Primary_Moment_Guards import _normal,primary_membrane_source_from_readout
    from Design.SMRF_Primary_Diaphragm_Batch import PrimaryDiaphragmReadout
    from Design.SMRF_Local_Transfer_Binding import local_transfer_recipe,bind_local_transfer_action,projector_parameters
    if not isinstance(local_pitch_actions,bool):raise ValueError('Explicit boolean local transfer action selection required')
    sid,level,cid=selected_floor_case_pair;level=int(level)
    if state.identity()!=common['state_identity']:raise ValueError('Exact installed local-transfer state required')
    from Design.SMRF_Common_Guard_Plan import PreparedCommonGuardSources
    if prepared_sources is not None and not isinstance(prepared_sources,PreparedCommonGuardSources):raise ValueError('Invocation-local common source context required')
    context=None if prepared_sources is None else prepared_sources.validate(common)
    bundles=physical_response_bundles(common) if context is None else context.bundles
    sources={r['source_id']:_PhysicalSource(r,common) for r in bundles} if context is None else context.sources
    source=sources[sid];actual,q=source.fields[(level,cid)];desc=next(f for f in common['definition']['floors'] if f['level']==level)
    inventory=common['inventory'];material=inventory['floors'][str(level)];fp=material['footprint'];h=inventory['slab_thickness_in']
    local=[r for r in common['local_checks']['records'] if r['source_id']==sid and r['floor']==level and r['case_id']==cid]
    expected={(s['id'],t) for s in desc['supports'] for t in ('inside','outside')}
    if {(r['support_id'],r['trace']) for r in local}!=expected or len(local)!=len(expected):raise ValueError('Complete actual local support/critical-trace inventory required')
    seismic=cid in {c['id'] for c in common['combinations'] if c['ex'] or c['ey']}
    readout=(PrimaryDiaphragmReadout(common['p5']) if context is None else context.readout) if seismic else None
    if seismic and (common['p5']['common_input_signature']!=common['input_signature'] or
        common['p5']['source_identity']!=common['analysis_source_identity'] or common['p5']['definition_sha256']!=common['definition_sha256'] or
        common['p5']['state_identity']!=common['state_identity'] or common['p5']['combination_inventory_sha256']!=_hash(common['combinations'])):
        raise ValueError('Local P5 state/source/input/combination basis differs')
    specs=[];cuts={};proofs=[];oneway=[]
    def append(spec,row,name,recipe=None,face_index=None,include_shortfall=False):
        if local_pitch_actions:
            original=row
            row=bind_local_transfer_action(recipe,face_index,include_shortfall,floor=level,inventory=inventory,
                analysis_input_identity=common['input_signature'],source_response_sha256=source.bundle['response_payload_sha256'],
                definition_sha256=common['definition_sha256'])
            if {k:v for k,v in row.items() if k!='local_point_binding'}!=original:
                raise ValueError('Local transfer binding changed original signed action or accounting')
        key=spec['cut_id']+'|'+name
        if key in cuts:raise ValueError('Duplicate local transfer cut identity')
        specs.append(dict(spec,cut_id=key,local_branch=name));cuts[key]=row
        if len(specs)>maximum_cuts:raise ValueError('Complete local transfer inventory exceeds declared cut budget')
    for record in local:
        support=next(s for s in desc['supports'] if s['id']==record['support_id'])
        for axis,cross,dim in (('x','y','b_in'),('y','x','h_in')):
            j=0 if axis=='x' else 1;center=support[axis];along=support['h_in' if axis=='x' else 'b_in']/2
            codeband=[max(fp[cross+'_min_in'],support[cross]-support[dim]/2-1.5*h),min(fp[cross+'_max_in'],support[cross]+support[dim]/2+1.5*h)]
            bars=[b for b in material['bars'] if b['axis']==axis and abs(b['z_in'])<h/2+b['db_in']/2]
            band=inward_bar_clear_band(codeband,bars)
            faces=[]
            for normal in (-1.,1.):
                station=center+normal*along
                if not fp[axis+'_min_in']+1e-8<station<fp[axis+'_max_in']-1e-8:continue
                trace=('minus' if normal>0 else 'plus') if record['trace']=='inside' else ('plus' if normal>0 else 'minus')
                name=f'local/{sid}/{level}/{cid}/{support["id"]}/{axis}/{record["trace"]}/{normal:+.0f}'
                transverse=[dict(db_in=b['db_in'],global_transverse_coordinates_in=[b['s_in']]) for b in material['bars'] if b['axis']!=axis and abs(b['z_in'])<h/2+b['db_in']/2]
                alo,ahi=fp[axis+'_min_in'],fp[axis+'_max_in']
                from Design.SMRF_Axial_Action_Accounting import select_axial_window
                selection=select_axial_window(station,axial_accounting_length_in,[alo,ahi],transverse)
                accounting=(selection['interval_in'] if selection['status']=='computed' else
                    bar_clear_interval([max(alo,station-axial_accounting_length_in/2),min(ahi,station+axial_accounting_length_in/2)],[alo,ahi],transverse))
                spec=dict(cut_id=name,floor=level,axis=axis,station_in=station,interval_in=band,trace=trace,
                    axial_accounting_interval_in=accounting,axial_window_selection=selection,material_scope='slab_only',member_id=None,
                    source_id=sid,support_id=support['id'],code_band_in=codeband,physical_region_kind='local_slab_column_transfer')
                raw=source.cut(spec,cid,member=None)
                normalized,_=physical_cut_states([raw],outer_interval_in=band);base=normalized[0]
                branches={'physical':base};branchproofs={}
                local_inputs=dict(name=name,raw=raw,primary=None,p5_readouts={})
                if seismic:
                    gid=common['gravity_case_mapping'][cid];g=sources['actual_frozen_gravity'].cut(spec,gid,member=None)
                    members=[b for b in desc['beams'] if b['axis']==axis and abs(b['line']-support[cross])<1e-8 and b['start']<station<b['end']]
                    if len(members)!=1:raise ValueError('Unique actual local support-face primary beam required')
                    member=members[0];projector=installed_full_t_projector(state.member(int(member['id'])))
                    # The frame web terminates at its clear support face.
                    # Its full-T target must use the span-side trace even
                    # while retaining both local slab traces at that face.
                    exact=dict(spec,interval_in=list(projector.flange),member_id=member['id'],trace='plus' if normal>0 else 'minus')
                    pe=source.cut(exact,cid);ge=sources['actual_frozen_gravity'].cut(exact,gid)
                    inc=primary_guard_increment(common,cid,member['id'],station)
                    if local_pitch_actions:local_inputs['primary']=dict(increment=inc,physical_effective=pe,gravity_effective=ge,
                        gravity_local=g,projector_parameters=projector_parameters(projector))
                    delta=_normal(ge,axis,projector.sref)+np.array([0.,inc['m_z_kip_in'],inc['m_s_kip_in']])-_normal(pe,axis,projector.sref)
                    for mode,dz,ds in (('strong',delta[1],0.),('weak',0.,delta[2]),('both',delta[1],delta[2])):
                        projection=projector.project(n_tension_kip=0.,m_z_kip_in=dz,m_s_kip_in=ds,increment_id=name+'/'+mode)
                        proof=dict(primary_increment=inc,projection=projection,exact_physical_sha256=_hash(pe),exact_gravity_sha256=_hash(ge),slab_only=True)
                        branches['primary/'+mode]=_guard(base,np.asarray(base['resultants'])+_projected_slab(projector,projection,band,base['reference_s_in']),proof,'primary/'+mode)
                    for branch in readout.floor(level)['witnesses'][cid]['cases']:
                        bid=branch['id'];p=readout.cut(level,cid,bid,axis=axis,station_in=station,interval_in=band,trace=trace,transverse_reference_in=base['reference_s_in'])
                        if local_pitch_actions:local_inputs['p5_readouts'][bid]=p
                        verified=primary_membrane_source_from_readout(p,identity=raw['identity'])
                        target=np.asarray(base['resultants']).copy();target[:3]=np.asarray(g['slab']['integral'][:3])+np.asarray(verified['slab_membrane_integral_kip'])
                        target[6]=g['slab']['transverse_first_moment'][j]+verified['slab_membrane_global_first_moment_kipin'][j]-base['reference_s_in']*target[j]
                        proof=dict(primary_slab_source=verified,gravity_slab_sha256=_hash(g),slab_only=True,web_design_force_omitted_from_slab_resistance=True)
                        branches['P5/'+bid]=_guard(base,target,proof,'P5/'+bid)
                        for mode,ds in (('strong',0.),('both',inc['m_s_kip_in'])):
                            projection=projector.project(n_tension_kip=0.,m_z_kip_in=delta[1],m_s_kip_in=ds,increment_id=name+'/P5/'+bid+'/'+mode)
                            fullproof=dict(proof,primary_increment=inc,projection=projection,exact_physical_sha256=_hash(pe),exact_gravity_sha256=_hash(ge))
                            branches['P5/'+bid+'/'+mode]=_guard(base,target+_projected_slab(projector,projection,band,base['reference_s_in']),fullproof,'P5/'+bid+'/'+mode)
                faces.append(dict(spec=spec,branches=branches,normal=normal,raw=raw,local_recipe_inputs=local_inputs))
                oneway.append(dict(specification=spec,physical_slab_integral8=raw['slab']['integral'],source_sha256=_hash(raw)))
            if not faces:raise ValueError('Actual local support has no slab transfer face')
            required=record['flexural_transfer_required_kipin'][axis]*(1 if axis=='x' else -1)
            if any(set(f['branches'])!=set(faces[0]['branches']) for f in faces):raise ValueError('Local paired faces have different concurrent branch inventories')
            for branch in faces[0]['branches']:
                proof=transfer_shortfall([f['branches'][branch]['resultants'][j+3] for f in faces],[f['normal'] for f in faces],required)
                proof.update(method=METHOD,support_record_id=record['id'],source_record_sha256=_hash(record),axis=axis,branch=branch,
                    code_band_in=codeband,actual_bar_clear_band_in=band,code_basis='ACI318-19 8.4.2.2.1-.6',slab_only_resistance=True)
                proofs.append(proof)
                recipe=local_transfer_recipe(faces,branch,record,axis,codeband,band) if local_pitch_actions else None
                for index,(f,increment) in enumerate(zip(faces,proof['face_increments_kipin'])):
                    base=f['branches'][branch];append(f['spec'],base,branch,recipe,index)
                    target=np.asarray(base['resultants']).copy();target[j+3]+=increment
                    append(f['spec'],_guard(base,target,proof,branch+'/gamma_f'),branch+'/gamma_f',recipe,index,True)
    result=dict(method=METHOD,pair=list(selected_floor_case_pair),state_identity=common['state_identity'],
        analysis_input_identity=common['input_signature'],material_inventory_id=inventory['identity_sha256'],
        cut_specs=specs,cuts=cuts,required_cut_ids=[s['cut_id'] for s in specs],required_case_ids=[cid],
        required_cut_count=len(specs),transfer_proofs=proofs,one_way_raw_cuts=oneway,
        nominal_probable_capacity_inventory_modified=False,complete_design_qualified=False)
    if local_pitch_actions:
        from Design.SMRF_Common_Guard_Plan import LOCAL_PLAN_CONTRACT
        result.update(action_contract=LOCAL_PLAN_CONTRACT,continuous_capacity_verified=False)
    result['payload_sha256']=_hash(result);return result
