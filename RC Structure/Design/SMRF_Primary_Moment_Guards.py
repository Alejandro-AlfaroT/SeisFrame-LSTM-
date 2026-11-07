"""Additional primary-SFRS strength demands on complete physical floor states.

These are design guards, not equilibrated extra load cases. The physical
G+ELF state remains intact. A zero-N full-T field raises/reverses one named
moment to the signed primary target G_actual + E_primary. All other concurrent
physical actions remain. Nominal/probable reinforcement inventories are never
changed. Both the original physical state and every guard must be checked.
"""
from __future__ import annotations
import copy
import numpy as np
from Design.SMRF_Common_Floor_Actions import _hash, _close, physical_partition, compose_common_cut

METHOD='physical_state_with_separate_primary_strength_moment_guards_v4_P5_axial_fields'


def primary_membrane_source_from_readout(readout,*,identity):
    """Verify B's actual Q4 cut fields and map to the strength-guard contract."""
    if readout.get('identity_sha256')!=_hash({k:v for k,v in readout.items() if k!='identity_sha256'}):
        raise ValueError('Actual prescribed-primary cut payload changed')
    context=readout.get('adapter_context') or {}
    if context.get('state_identity')!=identity.get('state_identity') or readout.get('combination_id')!=identity.get('combination_id'):
        raise ValueError('Prescribed-primary actual state/combination differs from physical strength cut')
    total=np.zeros(3);first=np.zeros(3);cursor=float(readout['interval_in'][0])
    for piece in sorted(readout['slab_fields'],key=lambda p:p['interval_in'][0]):
        a,b=map(float,piece['interval_in']);origin=float(piece['origin_in']);c=np.asarray(piece['coefficients'],float)
        if abs(a-cursor)>1e-8 or b<=a or c.shape!=(2,8) or not np.isfinite(c).all() or np.max(abs(c[:,3:]))>1e-12:
            raise ValueError('Actual planar slab field inventory has missing/overlapping/nonmembrane material')
        x=np.asarray(piece['sample_coordinates_in']);v=np.asarray(piece['sample_tensors'])
        if not _close(np.column_stack([np.ones(len(x)),x-origin])@c,v):raise ValueError('Primary slab affine transfer sample mismatch')
        left,right=a-origin,b-origin;integral=c[0,:3]*(right-left)+c[1,:3]*(right**2-left**2)/2
        moment=origin*integral+c[0,:3]*(right**2-left**2)/2+c[1,:3]*(right**3-left**3)/3
        if not _close(integral,piece['integrated_N_kip']) or not _close(moment,piece['transverse_first_moment_kipin']):
            raise ValueError('Primary slab affine integral or first moment mismatch')
        total+=integral;first+=moment;cursor=b
    if abs(cursor-readout['interval_in'][1])>1e-8 or not _close(total,readout['slab_integrated_N_kip']) or not _close(first,readout['slab_transverse_first_moment_kipin']):
        raise ValueError('Complete primary slab strip/resultants not reconstructed')
    webs=readout['webs']
    if len(webs)>1:raise ValueError('One strength common strip can own only one complete physical web')
    web=None;j=0 if readout['axis']=='x' else 1;alln=total.copy();allfirst=first.copy()
    if webs:
        w=webs[0]
        if float(w['reference_z_in'])!=0 or abs(float(w['m_z_kip_in']))>1e-12 or abs(float(w['m_s_kip_in']))>1e-12:
            raise ValueError('Primary planar web design force must retain actual slab-midplane reference')
        web=dict(member_id=w['member_id'],N_tension_kip=w['n_tension_kip'],transverse_reference_in=w['reference_s_in'])
        alln[j]+=web['N_tension_kip'];allfirst[j]+=web['transverse_reference_in']*web['N_tension_kip']
    ms=allfirst[j]-readout['transverse_reference_in']*alln[j]
    if not _close(alln,readout['total_integrated_N_kip']) or not _close([ms],[readout['Ms_transverse_first_moment_kipin']]):
        raise ValueError('Primary slab/web total membrane and first moment do not reconstruct')
    out=dict(identity=dict(identity),axis=readout['axis'],station_in=readout['station_in'],interval_in=readout['interval_in'],
        reference_z_in=0.,source_evidence_id=readout['identity_sha256'],branch_id=readout['branch_id'],gravity_case_id=readout['gravity_case_id'],
        slab_membrane_integral_kip=total.tolist(),slab_membrane_global_first_moment_kipin=first.tolist(),web=web,
        floor_freebody_residual=readout['floor_freebody_residual'],floor_freebody_tolerance=readout['floor_freebody_tolerance'],
        actual_readout=readout)
    return out


def physical_cut_states(cuts, *, outer_interval_in):
    partition=physical_partition(cuts,outer_interval_in=outer_interval_in)
    states=[]
    for cut in cuts:
        row=compose_common_cut(cut,slab_fields=[],web_additions=[],required_source_ids=[],partition=partition)
        for key in ('material_inventory_id','analysis_input_identity','source_response_sha256','definition_sha256','axial_action_window'):
            if key in cut:row[key]=cut[key]
        states.append(row)
    return states,partition


def _normal(cut,axis,sref):
    if cut['slab']['axis']!=axis or float(cut['reference_z_in'])!=0:
        raise ValueError('Actual same-axis slab-midplane cut required')
    j=0 if axis=='x' else 1;n=float(cut['resultants'][j])
    return np.array([n,cut['resultants'][j+3],cut['Ms_transverse_first_moment_kipin']+
                     (cut['transverse_reference_in']-sref)*n])


def primary_moment_guards(physical_cuts, gravity_effective_cut, *, projector,
        primary_increment, guard_identity, components=('strong','weak'),
        physical_effective_cut=None,outer_interval_in=None):
    """Return physical states plus separate signed primary moment guards.

    physical_cuts partition exactly this actual code-effective flange, with
    its complete web in one strip. gravity_effective_cut is the same physical
    flange/station and same factored G baseline, supplied separately from the
    released-primary increment. The increment carries explicit baseline case
    identity, weak-axis reference and zero external N. No E-only target is
    accepted. A caller wishing to cover more of the floor uses adjacent trace
    partitions too, without double-crediting their material.
    """
    if not guard_identity or not physical_cuts:raise ValueError('Named guard and actual physical cuts required')
    outer=projector.flange if outer_interval_in is None else outer_interval_in
    states,partition=physical_cut_states(physical_cuts,outer_interval_in=outer)
    if outer[0]>projector.flange[0]+1e-9 or outer[1]<projector.flange[1]-1e-9:
        raise ValueError('Physical bar-clear partition must contain the whole actual effective flange')
    if not _close(outer,projector.flange) and physical_effective_cut is None:
        raise ValueError('Enclosing physical widths require a separate exact effective-flange physical cut')
    first=physical_cuts[0];identity=first['identity'];slab=first['slab'];inc=primary_increment
    if not _close(gravity_effective_cut['slab']['interval_in'],projector.flange):
        raise ValueError('Gravity target must use the identical actual effective flange')
    for row in [gravity_effective_cut]+physical_cuts:
        if row['identity']['state_identity']!=identity['state_identity'] or row['slab']['axis']!=projector.axis or not _close(
                [row['slab']['station_in']],[slab['station_in']]) or row['slab']['trace']!=slab['trace']:
            raise ValueError('Physical/gravity cuts must retain actual state, axis, station and trace')
    if inc.get('source_operation')!='released_loaded_minus_identical_released_gravity':
        raise ValueError('Retained primary increment must subtract its identical released gravity baseline')
    if inc.get('state_identity')!=identity['state_identity'] or inc.get('gravity_load_id')!=gravity_effective_cut['identity']['combination_id']:
        raise ValueError('Primary increment must bind this actual state and exact gravity load identity')
    if inc.get('physical_combination_id')!=identity['combination_id'] or inc.get('axis')!=projector.axis or not _close(
            [float(inc.get('station_in',float('nan')))],[slab['station_in']]):
        raise ValueError('Primary increment must bind this actual physical combination/axis/station')
    if abs(float(inc['n_tension_kip']))>1e-8 or float(inc['reference_z_in'])!=0 or 'm_s_kip_in' not in inc:
        raise ValueError('Complete zero-N primary moment increment at slab midplane required')
    if physical_effective_cut is not None:
        pc=physical_effective_cut
        if (not _close(pc['slab']['interval_in'],projector.flange) or pc['identity']['state_identity']!=identity['state_identity'] or
                pc['identity']['combination_id']!=identity['combination_id'] or pc['slab']['trace']!=slab['trace'] or
                not _close([pc['slab']['station_in']],[slab['station_in']])):
            raise ValueError('Exact physical effective-flange cut differs in actual scope/concurrency')
        physical=_normal(pc,projector.axis,projector.sref)
    else:physical=sum((_normal(c,projector.axis,projector.sref) for c in physical_cuts),np.zeros(3))
    gravity=_normal(gravity_effective_cut,projector.axis,projector.sref)
    increment=np.array([0.,inc['m_z_kip_in'],inc['m_s_kip_in']])
    target=gravity+increment;j=projector.j
    gwindow=physical_cut_states([gravity_effective_cut],outer_interval_in=projector.flange)[0][0]
    pwindow=(physical_cut_states([physical_effective_cut],outer_interval_in=projector.flange)[0][0]
             if physical_effective_cut is not None else states[0] if len(states)==1 else None)
    if not components or len(set(components))!=len(components) or any(c not in ('strong','weak','both') for c in components):
        raise ValueError('Explicit distinct strong, weak or both design guard components required')
    output=[dict(branch_id='physical',kind='complete_physical_response',states=states)]
    for component in components:
        delta=np.zeros(3)
        if component in ('strong','both'):delta[1]=target[1]-physical[1]
        if component in ('weak','both'):delta[2]=target[2]-physical[2]
        name=f'{guard_identity}/{component}'
        projection=projector.project(n_tension_kip=0,m_z_kip_in=delta[1],m_s_kip_in=delta[2],increment_id=name)
        split=projector.partition(projection,[dict(id=c['identity']['location_id'],interval_in=c['slab']['interval_in']) for c in physical_cuts])
        byid={r['id']:r for r in split['strips']};guarded=[];sumdelta=np.zeros(3)
        proof=dict(method=METHOD,guard_identity=name,primary_increment=inc,gravity_effective_cut_sha256=_hash(gravity_effective_cut),
                   physical_effective_cut_sha256=None if physical_effective_cut is None else _hash(physical_effective_cut),
                   physical_partition=partition,projection=projection,partition=split,
                   physical_normal=physical.tolist(),gravity_normal=gravity.tolist(),target_normal=target.tolist(),
                   demand_increment=delta.tolist(),strength_only=True,physical_equilibrium_solution_claimed=False)
        for source in states:
            row=copy.deepcopy(source);part=byid[row['identity']['location_id']]
            row.pop('axial_action_window',None)
            r=np.asarray(part['integrated_six_resultants']);n=r[j]
            ms=part['transverse_first_moment_kip_in']+(projector.sref-row['reference_s_in'])*n
            add=np.r_[r,ms];row['resultants']=(np.asarray(row['resultants'])+add).tolist()
            sumdelta +=[n,r[j+3],ms+(row['reference_s_in']-projector.sref)*n]
            row['identity']['recovery_id']+=f'|strength_guard:{name}'
            row['source_identities'].append(dict(row['identity']))
            row['strength_guard']=dict(proof,physical_cut_resultants=source['resultants'],added_resultants=add.tolist())
            row['accounting']['evidence_id']=_hash(row['strength_guard'])
            row['accounting']['strength_only_design_guard']=True
            row['accounting']['physical_equilibrium_solution_claimed']=False
            # Existing numerical physical equilibrium remains provenance of
            # the base state; projection closure proves only added cut demand.
            row['accounting']['equilibrium_scope']='physical_base_and_separate_section_demand_projection'
            if pwindow is not None:
                from Design.SMRF_Axial_Action_Accounting import primary_moment_window_evidence
                raw=next(c for c in physical_cuts if c['identity']['location_id']==source['identity']['location_id'])
                row['axial_action_window']=primary_moment_window_evidence(row,source,gwindow,pwindow,
                    projector=projector,primary_increment=inc,component=component,
                    strip_interval=raw['slab']['interval_in'],owns_web=raw['web'] is not None)
            guarded.append(row)
        if not _close(sumdelta,delta):raise ValueError('Disjoint guard strips do not reconstruct N/M/Mweak')
        output.append(dict(branch_id=name,kind='additional_strength_only_primary_guard',states=guarded,
                           reconstruction_residual=(sumdelta-delta).tolist(),proof=proof))
    payload=dict(method=METHOD,state_identity=identity['state_identity'],combination_id=identity['combination_id'],
                 branches=output,nominal_probable_capacity_inventory_modified=False,
                 force_bound_claimed=False,physical_equilibrium_solution_claimed_for_guards=False)
    payload['payload_sha256']=_hash(payload)
    return payload


def primary_membrane_guards(physical_cuts,gravity_cuts,primary_membrane_sources,*,
        outer_interval_in,guard_identity,projector=None,primary_increment=None,
        physical_effective_cut=None,gravity_effective_cut=None):
    """Preserve prescribed primary collector forces in extra strength states.

    Every planar source contains the actual integrated slab membrane tensor,
    its GLOBAL transverse first moments, optional web N at its actual line,
    exact scope/identity and the numerical six-component primary floor
    freebody residual/tolerance. No centroid bending is invented: this is a
    specified normal demand at slab-midplane z=0 for a fresh capacity solution.

    The membrane guard retains physical G+E Mxx/Myy/Mxy and Q, and replaces
    membrane/weak first moment by G_actual + P5_primary. Optional simultaneous
    primary moment guards use the SAME projected demand and physical steel.
    A strong guard changes no total weak moment; the simultaneous strong+weak
    target is G_actual + P5_primary + E_primary. This is not an elastic field
    superposition or a proof one stiffness model bounds another.
    """
    if not guard_identity:raise ValueError('Named primary membrane guard required')
    states,partition=physical_cut_states(physical_cuts,outer_interval_in=outer_interval_in)
    if len(gravity_cuts)!=len(states) or len(primary_membrane_sources)!=len(states):
        raise ValueError('Every physical strip needs exact gravity and primary membrane sources')
    source_by={s['identity']['location_id']:s for s in primary_membrane_sources}
    gravity_by={c['identity']['location_id']:c for c in gravity_cuts}
    if len(source_by)!=len(states) or len(gravity_by)!=len(states):raise ValueError('Unique matching strip source inventory required')
    guarded=[];proofs=[];gstrong=0.;primary_ms=0.;physical_strong=0.
    j=0 if states[0]['axis']=='x' else 1
    for physical,raw in zip(states,physical_cuts):
        location=physical['identity']['location_id'];s=source_by.get(location);g=gravity_by.get(location)
        if s is None or g is None:raise ValueError('Required actual physical strip source is missing')
        for key in ('state_identity','combination_id','location_id'):
            if s['identity'].get(key)!=physical['identity'][key]:raise ValueError('Primary membrane source is nonconcurrent')
        if g['identity']['state_identity']!=physical['identity']['state_identity']:
            raise ValueError('Gravity membrane baseline differs from installed state')
        if s.get('gravity_case_id')!=g['identity']['combination_id']:
            raise ValueError('Primary membrane guard requires its exact concurrent gravity baseline')
        for row in (s,g['slab']):
            if row['axis']!=raw['slab']['axis'] or not _close([row['station_in']],[raw['slab']['station_in']]) or not _close(
                    row['interval_in'],raw['slab']['interval_in']):raise ValueError('Primary/gravity membrane physical scope differs')
        if not s.get('source_evidence_id') or not s.get('branch_id') or float(s.get('reference_z_in',float('nan')))!=0:
            raise ValueError('Signed primary branch and actual slab-midplane evidence required')
        er=np.asarray(s.get('floor_freebody_residual'),float);tol=np.asarray(s.get('floor_freebody_tolerance'),float)
        if er.shape!=(6,) or tol.shape!=(6,) or not np.isfinite(er).all() or not np.isfinite(tol).all() or np.any(tol<0) or np.any(abs(er)>tol):
            raise ValueError('Prescribed primary diaphragm freebody does not numerically close')
        ni=np.asarray(s['slab_membrane_integral_kip'],float);first=np.asarray(s['slab_membrane_global_first_moment_kipin'],float)
        if ni.shape!=(3,) or first.shape!=(3,) or not np.isfinite(ni).all() or not np.isfinite(first).all():
            raise ValueError('Actual full slab membrane tensor and first moments required')
        membrane=ni.copy();ms=float(first[j]-physical['reference_s_in']*ni[j]);web=s.get('web')
        if web is not None:
            if raw.get('web') is None or str(web['member_id'])!=str(raw['web']['member_id']):
                raise ValueError('Primary web axial source lacks matching actual whole web owner')
            n=float(web['N_tension_kip']);line=float(web['transverse_reference_in'])
            if not np.isfinite([n,line]).all():raise ValueError('Finite primary web axial state required')
            membrane[j]+=n;ms+=(line-physical['reference_s_in'])*n
        row=copy.deepcopy(physical);target=np.asarray(row['resultants'])
        row.pop('axial_action_window',None)
        gn=np.asarray(g['resultants'][:3]);gms=(float(g['Ms_transverse_first_moment_kipin'])+
            (g['transverse_reference_in']-physical['reference_s_in'])*gn[j])
        target[:3]=gn+membrane;target[6]=gms+ms;row['resultants']=target.tolist()
        row['identity']['recovery_id']+=f'|primary_membrane_guard:{guard_identity}'
        row['source_identities'].append(dict(row['identity']))
        proof=dict(method=METHOD,guard_identity=guard_identity,physical_cut_sha256=_hash(raw),gravity_cut_sha256=_hash(g),
                   primary_source=s,strength_only=True,physical_equilibrium_solution_claimed=False)
        row['strength_guard']=proof;row['accounting'].update(strength_only_design_guard=True,
            physical_equilibrium_solution_claimed=False,primary_required_collector_membrane_retained=True,
            equilibrium_scope='physical_base_and_prescribed_primary_section_demand',evidence_id=_hash(proof))
        if s.get('axial_action') is not None:
            from Design.SMRF_Axial_Action_Accounting import linear_combination_window
            gb=physical_cut_states([g],outer_interval_in=g['slab']['interval_in'])[0][0]
            primary=s['axial_action'];keep=np.zeros((7,7));keep[3:6,3:6]=np.eye(3)
            gm=np.zeros((7,7));gm[:3,:3]=np.eye(3);gm[6,6]=1.
            gm[6,j]=gb['reference_s_in']-row['reference_s_in']
            pm=gm.copy();pm[6,j]=primary['reference_s_in']-row['reference_s_in']
            row['axial_action_window']=linear_combination_window(row,[physical,gb,primary],[keep,gm,pm],
                derivation_method='physical_bending_plus_G_and_named_P5_membrane_axial_field_v1')
        guarded.append(row);proofs.append(proof)
        gstrong+=g['resultants'][j+3];physical_strong+=physical['resultants'][j+3]
        if projector is not None:primary_ms+=ms+(physical['reference_s_in']-projector.sref)*membrane[j]
    if len({s['branch_id'] for s in primary_membrane_sources})!=1:raise ValueError('Different primary diaphragm branches cannot be mixed')
    branches=[dict(branch_id='physical',kind='complete_physical_response',states=states),
              dict(branch_id=guard_identity,kind='additional_strength_only_primary_membrane_guard',states=guarded)]
    if (projector is None)!=(primary_increment is None):raise ValueError('Simultaneous primary moment guard needs projector and actual increment together')
    if projector is not None:
        inc=primary_increment
        gwindow=(physical_cut_states([gravity_effective_cut],outer_interval_in=projector.flange)[0][0]
                 if gravity_effective_cut is not None else physical_cut_states([gravity_cuts[0]],outer_interval_in=projector.flange)[0][0]
                 if len(gravity_cuts)==1 else None)
        pwindow=(physical_cut_states([physical_effective_cut],outer_interval_in=projector.flange)[0][0]
                 if physical_effective_cut is not None else states[0] if len(states)==1 else None)
        if outer_interval_in[0]>projector.flange[0]+1e-9 or outer_interval_in[1]<projector.flange[1]-1e-9:
            raise ValueError('Simultaneous actual material partition must contain the effective flange')
        if not _close(outer_interval_in,projector.flange) and (physical_effective_cut is None or gravity_effective_cut is None):
            raise ValueError('Enclosing bar-clear material partition requires both exact effective-flange target cuts')
        for exact,combo in ((physical_effective_cut,states[0]['identity']['combination_id']),(gravity_effective_cut,inc.get('gravity_load_id'))):
            if exact is not None:
                if (not _close(exact['slab']['interval_in'],projector.flange) or exact['identity']['combination_id']!=combo or
                    exact['identity']['state_identity']!=states[0]['identity']['state_identity'] or exact['slab']['axis']!=projector.axis or
                    not _close([exact['slab']['station_in']],[physical_cuts[0]['slab']['station_in']]) or
                    exact['slab']['trace']!=physical_cuts[0]['slab']['trace']):
                    raise ValueError('Exact effective-flange source scope differs from physical strength partition')
        if physical_effective_cut is not None:physical_strong=_normal(physical_effective_cut,projector.axis,projector.sref)[1]
        if gravity_effective_cut is not None:gstrong=_normal(gravity_effective_cut,projector.axis,projector.sref)[1]
        if inc.get('source_operation')!='released_loaded_minus_identical_released_gravity' or abs(float(inc['n_tension_kip']))>1e-8:
            raise ValueError('Identical-baseline zero-N released primary increment required')
        if inc.get('state_identity')!=states[0]['identity']['state_identity'] or inc.get('physical_combination_id')!=states[0]['identity']['combination_id']:
            raise ValueError('Simultaneous primary increment differs from actual state/combination')
        if inc.get('axis')!=projector.axis or float(inc.get('reference_z_in',float('nan')))!=0 or not _close(
                [float(inc.get('station_in',float('nan')))],[physical_cuts[0]['slab']['station_in']]):
            raise ValueError('Simultaneous primary increment differs from actual cut axis/station/datum')
        if any(g['identity']['combination_id']!=inc.get('gravity_load_id') for g in gravity_cuts):
            raise ValueError('Simultaneous moment target requires exactly the actual gravity baseline')
        for mode in ('strong','both'):
            delta=np.array([0.,gstrong+float(inc['m_z_kip_in'])-physical_strong,
                            float(inc['m_s_kip_in']) if mode=='both' else 0.])
            projection=projector.project(n_tension_kip=0,m_z_kip_in=delta[1],m_s_kip_in=delta[2],increment_id=guard_identity+'/'+mode)
            split=projector.partition(projection,[dict(id=r['identity']['location_id'],interval_in=raw['slab']['interval_in']) for r,raw in zip(guarded,physical_cuts)])
            parts={v['id']:v for v in split['strips']};combined=[]
            for base in guarded:
                row=copy.deepcopy(base);p=parts[row['identity']['location_id']];six=np.asarray(p['integrated_six_resultants'])
                row.pop('axial_action_window',None)
                ms=p['transverse_first_moment_kip_in']+(projector.sref-row['reference_s_in'])*six[j]
                row['resultants']=(np.asarray(row['resultants'])+np.r_[six,ms]).tolist()
                row['identity']['recovery_id']+=f'|primary_moment:{mode}'
                row['strength_guard'].update(primary_increment=inc,projection=projection,projection_partition=split,
                    physical_effective_cut_sha256=None if physical_effective_cut is None else _hash(physical_effective_cut),
                    gravity_effective_cut_sha256=None if gravity_effective_cut is None else _hash(gravity_effective_cut),
                    simultaneous_primary_membrane_and_moment=True)
                row['accounting']['evidence_id']=_hash(row['strength_guard'])
                if gwindow is not None and pwindow is not None:
                    from Design.SMRF_Axial_Action_Accounting import primary_moment_window_evidence
                    raw=next(c for c in physical_cuts if c['identity']['location_id']==base['identity']['location_id'])
                    row['axial_action_window']=primary_moment_window_evidence(row,base,gwindow,pwindow,
                        projector=projector,primary_increment=inc,component=mode,
                        strip_interval=raw['slab']['interval_in'],owns_web=raw['web'] is not None,
                        primary_weak_increment_only=True)
                combined.append(row)
            branches.append(dict(branch_id=guard_identity+'/'+mode,kind='additional_strength_only_simultaneous_primary_guard',states=combined))
    out=dict(method=METHOD,branches=branches,physical_partition=partition,strength_only=True,
             nominal_probable_capacity_inventory_modified=False,physical_equilibrium_solution_claimed_for_guards=False)
    out['payload_sha256']=_hash(out);return out
