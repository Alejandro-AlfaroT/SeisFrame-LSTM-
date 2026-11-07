"""Atomic disjoint gravity + primary flexure + P5 physical cut composition.

Every field is a signed full tensor on an actual interval; both beam axes
can contribute to one slab cut. No C0 field is accepted. Primary normal
projection has zero total N; P5 exclusively owns external diaphragm N.
Algebraic seven-action transfer is distinguished from elastic/ultimate strain
agreement and from local transverse-shear field qualification.
"""
from __future__ import annotations
import hashlib,json,math
import numpy as np

METHOD='physical_common_seven_action_composition_v1'
KEYS=('state_identity','combination_id','location_id','recovery_id')


def _hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _identity(x):
    if not isinstance(x,dict) or any(not isinstance(x.get(k),str) or not x[k] for k in KEYS):
        raise ValueError('Complete atomic state/combination/location/recovery identity required')
    return {k:x[k] for k in KEYS}


def _close(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and bool(np.all(abs(a-b)<=1e-8+1e-7*np.maximum(abs(a),abs(b))))


def polynomial_slab_field(*,source_id,source_kind,identity,axis,station_in,interval_in,origin_in,
        coefficients,sample_coordinates_in,sample_tensors,branch_id=None,
        local_shear_field_verified=False,proof=None):
    """Freeze an actual upstream polynomial line field and check its samples.

    coefficients[k,:] multiply(s-origin)^k for all eight common shell fields:
    Nxx,Nyy,Nxy,Mxx,Myy,Mxy,Qx,Qy. Degree0..2 is integrated exactly below.
    Caller must supply actual sampled source fields, not only an acceptance
    flag. This verifies transfer/interpolation, not its upstream analysis.
    """
    ident=_identity(identity);c=np.asarray(coefficients,float);x=np.asarray(sample_coordinates_in,float);v=np.asarray(sample_tensors,float)
    if axis not in ('x','y') or not math.isfinite(float(station_in)):raise ValueError('Actual global cut axis/station required')
    lo,hi=map(float,interval_in);origin=float(origin_in)
    if source_kind not in ('primary_elf_slab','p5_slab','p5_physical_slab') or not source_id:raise ValueError('Explicit primary or P5 slab source required')
    if c.ndim!=2 or c.shape[1]!=8 or not 1<=len(c)<=3 or not np.isfinite(c).all():raise ValueError('Degree0..2 full eight-tensor polynomial required')
    if not all(math.isfinite(z) for z in (lo,hi,origin)) or hi<=lo:raise ValueError('Actual positive finite physical field interval required')
    if x.ndim!=1 or len(set(x.tolist()))<len(c) or v.shape!=(len(x),8) or np.any(x<lo-1e-9) or np.any(x>hi+1e-9):
        raise ValueError('Enough distinct actual source samples inside the field interval required')
    reconstructed=np.vander(x-origin,N=len(c),increasing=True)@c
    if not _close(reconstructed,v):raise ValueError('Polynomial transfer does not reconstruct actual source tensor samples')
    if source_kind=='p5_slab' and (not branch_id or np.max(abs(c[:,3:]))>1e-12):
        raise ValueError('P5 membrane branch must be named and cannot invent bending/transverse shear')
    shear_verified=source_kind=='p5_slab'
    if source_kind=='p5_physical_slab':
        error=float((proof or {}).get('physical_force_partition_residual',math.nan))
        if not branch_id or not math.isfinite(error) or abs(error)>1e-9 or not (proof or {}).get('physical_response_sha256'):
            raise ValueError('Actual coupled physical P5 response identity and numerical force partition witness required')
        shear_verified=True
    if source_kind=='primary_elf_slab' and local_shear_field_verified:
        witness=(proof or {}).get('shear_equilibrium')
        if not isinstance(witness,dict) or not witness.get('source_evidence_id'):
            raise ValueError('Actual local shear derivative/interface-equilibrium witness required')
        names=('dMxx_dx','dMxy_dy','dMyy_dy','dMxy_dx','interface_couple_x','interface_couple_y')
        data={name:np.asarray(witness[name],float) for name in names}
        if any(value.shape!=x.shape or not np.isfinite(value).all() for value in data.values()):raise ValueError('Actual derivative/interface witness at every tensor sample required')
        predicted=np.c_[-data['dMxx_dx']-data['dMxy_dy']+data['interface_couple_x'],
                         -data['dMyy_dy']-data['dMxy_dx']+data['interface_couple_y']]
        if not _close(predicted,v[:,6:]):raise ValueError('Local transverse shear does not satisfy supplied actual derivative/interface equilibrium')
        shear_verified=True
    return dict(source_id=str(source_id),source_kind=source_kind,identity=ident,axis=axis,station_in=float(station_in),interval_in=[lo,hi],origin_in=origin,
        coefficients=c.tolist(),sample_coordinates_in=x.tolist(),sample_tensors=v.tolist(),
        sample_residual=(reconstructed-v).tolist(),branch_id=branch_id,
        local_shear_field_verified=shear_verified,proof=proof)


def integrate_slab_field(term,interval_in):
    lo=max(float(interval_in[0]),term['interval_in'][0]);hi=min(float(interval_in[1]),term['interval_in'][1])
    if hi<=lo:return np.zeros(8),np.zeros(8)
    c=np.asarray(term['coefficients']);a,b=lo-term['origin_in'],hi-term['origin_in']
    integral=sum((c[k]*(b**(k+1)-a**(k+1))/(k+1) for k in range(len(c))),np.zeros(8))
    first=term['origin_in']*integral+sum((c[k]*(b**(k+2)-a**(k+2))/(k+2) for k in range(len(c))),np.zeros(8))
    return integral,first


def full_t_normal_terms(projector,projection,*,identity,source_id,member_id,station_in):
    """Actual aligned primary full-T slab field plus one disjoint web source.

    N=0 is mandatory because P5 owns the external diaphragm axial action.
    Local Q is explicitly unresolved; use the projector's derivative witness
    for integrated shear and a separately justified spatial shear recovery.
    """
    ident=_identity(identity);p=projector.properties()
    if projection.get('properties')!=p:raise ValueError('Full-T field geometry/reference differs from projector')
    q=np.asarray(projection['generalized_normal_strain']);target=np.asarray(projection['reconstructed_resultants'])
    if not _close(projector.K@q,target) or abs(target[0])>1e-8 or projection.get('requires_transverse_action_accounting'):
        raise ValueError('Complete zero-total-N full-T normal projection required')
    if projection.get('transverse_action_specified') is not True:
        raise ValueError('Actual weak first moment, including explicit zero, must be retained')
    j=projector.j;c=np.zeros((2,8));c[0,j]=projector.ef*projector.t*q[0];c[1,j]=projector.ef*projector.t*q[2]
    c[0,j+3]=projector.ef*projector.t**3/12*q[1]
    points=list(projector.flange);samples=[list(projector.slab_field(projection,s))+[0.,0.] for s in points]
    field=polynomial_slab_field(source_id=source_id+'/slab',source_kind='primary_elf_slab',identity=ident,axis=projector.axis,station_in=station_in,
        interval_in=projector.flange,origin_in=projector.sref,coefficients=c,
        sample_coordinates_in=points,sample_tensors=samples,proof=dict(full_T_projection=projection,axial_scope='zero_total_N_primary'))
    w=next(r for r in projection['regions'] if r['id']=='web')
    web=dict(source_id=source_id+'/web',source_kind='primary_elf_web',identity=ident,axis=projector.axis,
        station_in=float(station_in),
        member_id=str(member_id),interval_in=list(projector.web),reference_s_in=projector.sref,reference_z_in=0.,
        n_tension_kip=w['n_tension_kip'],m_z_kip_in=w['m_z_kip_in'],m_s_kip_in=w['m_s_kip_in'],
        full_T_projection=projection,axial_scope='zero_total_N_primary')
    return dict(slab=field,web=web,normal_reconstruction_residual=(projector.K@q-target).tolist(),
                local_transverse_shear_field_verified=False)


def physical_response_terms(response,cut,*,branch_id,source_prefix,source_kind='p5_physical'):
    """Recover actual coupled P5 slab8 and web3 sources on this physical cut.

    The response is the actual zero-element-load additional-floor-load solve.
    Native GP fields are interpolated only inside their own physical cell.
    The resulting line tensor is bilinear restricted to a line, hence affine;
    endpoints and midpoint independently verify the exact polynomial transfer.
    The actual web state preserves N, both normal first moments, Q/V/T and
    its force-end equilibrium. No planar-N datum approximation is inserted.
    """
    from Design.SMRF_Physical_Floor_Cuts import slab_tensor
    if source_kind!='p5_physical' or not branch_id or not source_prefix:
        raise ValueError('Explicit physical P5 branch/source identities required')
    ident=_identity(cut['identity']);slab=cut['slab'];axis=slab['axis'];j=0 if axis=='x' else 1
    station=float(slab['station_in']);sref=float(cut['transverse_reference_in']);error=float(response.get('force_partition_relative_error',math.nan))
    if not math.isfinite(error) or abs(error)>1e-9:raise ValueError('Actual coupled P5 floor force partition did not close')
    response_hash=_hash(response);elements={r['id']:r for r in response['elements']};fields=[];webs=[];total=np.zeros(7)
    proof=dict(physical_response_sha256=response_hash,physical_force_partition_residual=error,
               native_field_method=response.get('method'),upstream_model_mesh_qualification_not_inferred=True,
               primary_force_composition_authorized=False,
               branch_scope='independent physical additional-load response about its own gravity baseline')
    for piece in slab['pieces']:
        row=elements[piece['element_id']];a,b=piece['interval_in'];coords=[a,(a+b)/2,b]
        values=np.array([slab_tensor(row,station,s) if axis=='x' else slab_tensor(row,s,station) for s in coords])
        coefficients=np.stack([values[0],(values[2]-values[0])/(b-a)])
        term=polynomial_slab_field(source_id=f'{source_prefix}/slab/{row["id"]}',source_kind='p5_physical_slab',
            identity=ident,axis=axis,station_in=station,interval_in=[a,b],origin_in=a,coefficients=coefficients,
            sample_coordinates_in=coords,sample_tensors=values,branch_id=branch_id,proof=proof)
        integral,first=integrate_slab_field(term,[a,b])
        if not _close(integral,piece['integral']) or not _close(first,piece['transverse_first_moment']):
            raise ValueError('Physical coupled P5 cell tensors differ from the actual cut integrals')
        total+=np.r_[integral[:6],first[j]-sref*integral[j]];fields.append(term)
    web=cut.get('web')
    if web is not None:
        row=elements[web['element_id']];res=np.asarray(web['end_equilibrium_residual']);scale=np.asarray(web['end_equilibrium_scale'])
        if res.shape!=scale.shape or np.any(scale<=0) or not np.isfinite(res).all() or np.any(abs(res)>1e-8*scale):
            raise ValueError('Actual coupled P5 web end equilibrium did not close')
        props=row['section'];line=float(web['transverse_reference_in']);n=float(web['N_tension_kip'])
        m=float(web['M_centroid_kipin'])+float(web['centroid_z_in'])*n;ms=float(web['Ms_transverse_first_moment_kipin'])
        term=dict(source_id=f'{source_prefix}/web/{web["element_id"]}',source_kind='p5_physical_web',identity=ident,
            axis=axis,station_in=station,branch_id=branch_id,member_id=str(web['member_id']),
            interval_in=[line-props['b_in']/2,line+props['b_in']/2],reference_s_in=line,reference_z_in=0.,
            n_tension_kip=n,m_z_kip_in=m,m_s_kip_in=ms,source_web_state=web,source_evidence_id=response_hash,
            source_element_global_forces=row['global_resisting_force'],physical_force_partition_residual=error)
        term['primary_force_composition_authorized']=False
        webs.append(term);total[j]+=n;total[j+3]+=m;total[6]+=ms+(line-sref)*n
    expected=np.r_[cut['resultants'],cut['Ms_transverse_first_moment_kipin']]
    if not _close(total,expected):raise ValueError('Physical P5 terms do not reconstruct all seven actual cut actions')
    return dict(method=METHOD,slab_fields=fields,web_additions=webs,
        required_source_ids=[v['source_id'] for v in fields+webs],branch_id=branch_id,
        reference_resultants=expected.tolist(),reconstruction_residual=(total-expected).tolist(),
        physical_response_sha256=response_hash,local_transverse_shear_field_verified=True,
        primary_force_composition_authorized=False,
        web_other_actions=None if web is None else {k:web[k] for k in ('Q_vertical_kip','V_inplane_kip','T_about_positive_axis_kipin')},
        model_mesh_qualification_inferred=False)


def physical_partition(cuts,*,outer_interval_in):
    """Verify one actual floor trace has complete disjoint physical strips."""
    low,high=map(float,outer_interval_in);cursor=low;rows=[]
    for cut in sorted(cuts,key=lambda c:c['slab']['interval_in'][0]):
        slab=cut['slab'];a,b=slab['interval_in'];ident=_identity(cut['identity'])
        if abs(a-cursor)>1e-8 or b<=a:raise ValueError('Physical common/remote strip coverage has a gap or overlap')
        if not _close(np.sum([p['integral'] for p in slab['pieces']],axis=0),slab['integral']):raise ValueError('Gravity slab piece integrals do not reconstruct cut')
        if not _close(np.sum([p['transverse_first_moment'] for p in slab['pieces']],axis=0),slab['transverse_first_moment']):
            raise ValueError('Gravity slab first moments do not reconstruct cut')
        rows.append(dict(location_id=ident['location_id'],cut_sha256=_hash(cut),interval_in=[a,b],
                         state_identity=ident['state_identity'],combination_id=ident['combination_id'],
                         axis=slab['axis'],station_in=slab['station_in'],trace=slab['trace']))
        cursor=b
    if not rows or abs(cursor-high)>1e-8:raise ValueError('Physical strips do not cover entire actual floor trace')
    for key in ('state_identity','combination_id','axis','station_in','trace'):
        if len({r[key] for r in rows})!=1:raise ValueError('Partition mixes nonconcurrent floor traces')
    result=dict(method=METHOD,outer_interval_in=[low,high],rows=rows,coverage_residual_in=cursor-high)
    result['evidence_id']=_hash(result);return result


def compose_common_cut(gravity_cut,*,slab_fields,web_additions,required_source_ids,partition):
    """Add all concurrent signed fields exactly once on the actual cut width."""
    ident=_identity(gravity_cut['identity']);slab=gravity_cut['slab'];axis=slab['axis'];j=0 if axis=='x' else 1
    lo,hi=slab['interval_in'];sref=float(gravity_cut['transverse_reference_in']);width=hi-lo
    if partition.get('evidence_id')!=_hash({k:v for k,v in partition.items() if k!='evidence_id'}):raise ValueError('Physical partition evidence changed after verification')
    if not any(r['cut_sha256']==_hash(gravity_cut) for r in partition['rows']):raise ValueError('Cut is absent from actual verified physical partition')
    terms=list(slab_fields)+list(web_additions);ids=[v['source_id'] for v in terms]
    if len(set(ids))!=len(ids) or set(ids)!=set(required_source_ids):raise ValueError('Complete explicit source inventory required exactly once')
    branches={f.get('branch_id') for f in terms if f['source_kind'].startswith('p5')}
    if len(branches)>1 or None in branches:raise ValueError('P5 original/amplified branches cannot be mixed in one cut')
    for term in terms:
        tid=_identity(term['identity'])
        if any(tid[k]!=ident[k] for k in KEYS[:-1]):raise ValueError('Source belongs to a different state/combination/location')
        if term.get('axis')!=axis or not math.isclose(float(term.get('station_in',math.nan)),float(slab['station_in']),rel_tol=0,abs_tol=1e-9):
            raise ValueError('Source tensor is on a different actual cut axis/station')
        if term['interval_in'][0]<partition['outer_interval_in'][0]-1e-9 or term['interval_in'][1]>partition['outer_interval_in'][1]+1e-9:
            raise ValueError('Complete physical trace does not cover the added field interval')
        if term['source_kind'].startswith('primary') and term.get('axial_scope',term.get('proof',{}).get('axial_scope'))!='zero_total_N_primary':
            raise ValueError('Primary/P5 external axial ownership is ambiguous')
    gi=np.asarray(slab['integral'],float);gf=np.asarray(slab['transverse_first_moment'],float)
    reconstructed=np.r_[gi[:6],gf[j]-sref*gi[j]]
    gw=gravity_cut.get('web')
    if gw is not None:
        reconstructed[j]+=gw['N_tension_kip'];reconstructed[j+3]+=gw['M_centroid_kipin']+gw['centroid_z_in']*gw['N_tension_kip']
        reconstructed[6]+=gw['Ms_transverse_first_moment_kipin']+(gw['transverse_reference_in']-sref)*gw['N_tension_kip']
    total=np.r_[np.asarray(gravity_cut['resultants'],float),gravity_cut['Ms_transverse_first_moment_kipin']]
    if total.shape!=(7,):raise ValueError('Gravity physical cut must retain all six actions and normal transverse first moment')
    if not _close(total,reconstructed):raise ValueError('Gravity common cut does not reconstruct its disjoint slab/web fields')
    physical_p5=not any(v['source_kind'] in ('p5_slab','p5_web') for v in terms)
    mixed_independent=(any(v['source_kind'].startswith('primary') for v in terms) and
                       any(v['source_kind'].startswith('p5_physical') for v in terms))
    contributions=[];known_q=np.asarray(slab['integral'][6:],float);local_q=physical_p5
    for term in slab_fields:
        checked=polynomial_slab_field(**{k:term[k] for k in ('source_id','source_kind','identity','axis','station_in','interval_in','origin_in',
            'coefficients','sample_coordinates_in','sample_tensors','branch_id','local_shear_field_verified','proof')})
        if term['source_kind']=='primary_elf_slab':
            proof=term.get('proof') or {};projections=proof.get('full_T_projections',[proof.get('full_T_projection')])
            if not projections or any(p is None for p in projections):raise ValueError('Actual primary full-T matrix/source projection witness required')
            for p in projections:
                k=np.asarray(p['properties']['full_three_resultant_matrix']);q=np.asarray(p['generalized_normal_strain']);r=np.asarray(p['reconstructed_resultants'])
                if not _close(k@q,r) or abs(r[0])>1e-8 or not p.get('transverse_action_specified'):
                    raise ValueError('Primary field witness does not close all actual zero-N normal resultants')
        integral,first=integrate_slab_field(term,[lo,hi]);value=np.r_[integral[:6],first[j]-sref*integral[j]]
        total+=value;known_q+=integral[6:];local_q &=term['local_shear_field_verified']
        contributions.append(dict(source_id=term['source_id'],resultants=value.tolist(),integral8=integral.tolist(),first_moment8=first.tolist()))
    for term in web_additions:
        if term['source_kind']=='primary_elf_web':
            p=term['full_T_projection'];w=next(r for r in p['regions'] if r['id']=='web')
            if not _close([term['n_tension_kip'],term['m_z_kip_in'],term['m_s_kip_in']],
                          [w['n_tension_kip'],w['m_z_kip_in'],w['m_s_kip_in']]):
                raise ValueError('Primary web source differs from its actual full-T decomposition')
        elif term['source_kind']=='p5_web':
            ends=np.asarray(term.get('signed_axial_end_forces_kip'),float)
            if ends.shape!=(2,) or not _close(ends,[-term['n_tension_kip'],term['n_tension_kip']]) or not term.get('source_evidence_id'):
                raise ValueError('P5 actual beam axial end-force/identity witness required')
            if float(term['reference_z_in'])!=0. or abs(float(term['m_z_kip_in']))>1e-10 or abs(float(term['m_s_kip_in']))>1e-10:
                raise ValueError('P5 axial load-plane wrench must be N at slab midplane and its own member line')
        elif term['source_kind']=='p5_physical_web':
            w=term['source_web_state'];n=float(w['N_tension_kip'])
            target=[n,w['M_centroid_kipin']+w['centroid_z_in']*n,w['Ms_transverse_first_moment_kipin']]
            if not _close([term['n_tension_kip'],term['m_z_kip_in'],term['m_s_kip_in']],target) or not term.get('source_evidence_id'):
                raise ValueError('Physical P5 web actions differ from actual coupled response')
            res=np.asarray(w['end_equilibrium_residual']);scale=np.asarray(w['end_equilibrium_scale'])
            if res.shape!=scale.shape or np.any(scale<=0) or np.any(abs(res)>1e-8*scale):raise ValueError('Physical P5 web end forces do not equilibrate')
        else:raise ValueError('Explicit primary or P5 web source required')
        a,b=term['interval_in'];overlap=min(b,hi)-max(a,lo)
        if overlap<=1e-9:continue
        if a<lo-1e-9 or b>hi+1e-9:raise ValueError('Physical common cut cannot split an added web/cage')
        if term['axis']!=axis:raise ValueError('Crossing web requires an actual joint/three-dimensional section')
        if gravity_cut.get('web') is None or str(gravity_cut['web']['member_id'])!=str(term['member_id']):
            raise ValueError('Added web source has no matching disjoint physical gravity web')
        n=float(term['n_tension_kip']);m=float(term['m_z_kip_in'])+float(term['reference_z_in'])*n
        ms=float(term['m_s_kip_in'])+(float(term['reference_s_in'])-sref)*n
        value=np.zeros(7);value[j]=n;value[j+3]=m;value[6]=ms;total+=value
        contributions.append(dict(source_id=term['source_id'],resultants=value.tolist()))
    gravity=np.r_[gravity_cut['resultants'],gravity_cut['Ms_transverse_first_moment_kipin']]
    residual=total-gravity-sum((np.asarray(r['resultants']) for r in contributions),np.zeros(7))
    tolerance=1e-8+1e-7*np.maximum(abs(total),abs(gravity))
    inherited=gravity_cut['accounting'];er=np.asarray(inherited['equilibrium_residual'],float);et=np.asarray(inherited['equilibrium_tolerance'],float)
    transfer_verified=_close(residual,np.zeros(7)) and er.shape==et.shape and np.isfinite(er).all() and np.all(abs(er)<=et)
    evidence=dict(gravity_cut_sha256=_hash(gravity_cut),sources=terms,partition_id=partition['evidence_id'])
    accounting=dict(method='disjoint_actual_fields',evidence_id=_hash(evidence),equilibrium_residual=np.r_[er,residual].tolist(),
        equilibrium_tolerance=np.r_[et,tolerance].tolist(),weak_first_moment_residual_kip_in=float(residual[6]),
        weak_first_moment_tolerance_kip_in=float(tolerance[6]),spatial_seven_field_projection_verified=bool(transfer_verified),
        core_remote_coverage_verified=True,all_concurrent_demands_included=set(ids)==set(required_source_ids),
        coverage_scope='this complete physical floor trace; all design traces and mesh witnesses remain separate',
        projection_scope='actual physical-width six tensors plus normal transverse first moment; ultimate strength has its own compatible strain solution',
        elastic_ultimate_strain_equality_required=False,local_transverse_shear_field_verified=bool(local_q),
        physical_diaphragm_embedding_verified=physical_p5,
        independent_physical_branch_added_to_primary=bool(mixed_independent),
        primary_force_composition_authorized=not mixed_independent,
        primary_axial_scope='zero_total_N_primary_P5_exclusively_owns_external_N')
    return dict(method=METHOD,identity=ident,source_identities=[ident]+[v['identity'] for v in terms],axis=axis,
        reference_z_in=0.,reference_s_in=sref,cut_width_in=width,units='integrated_strip',resultants=total.tolist(),
        contributions=contributions,accounting=accounting,known_transverse_shear_integrals=known_q.tolist(),
        local_transverse_shear_status='available' if local_q else 'unknown',
        complete_design_qualified=False,nominal_probable_capacity_inventory_modified=False)
