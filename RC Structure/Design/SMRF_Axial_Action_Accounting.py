"""Match seven section actions to the physical transverse-bar counting length.

The spatial material kernel homogenizes actual transverse bars over an axial
length. Integrating the same seven work-conjugate actions over that length
establishes the associated average. A point cut is not automatically that
average. This module retains both values and never lowers a point demand.
An average by itself establishes neither local capacity nor mesh convergence.
"""
from __future__ import annotations
import math
import numpy as np
from Design.SMRF_Common_Floor_Actions import _hash

METHOD='actual_seven_action_axial_window_v3_selection_scope'


def select_axial_window(station_in,length_in,available_interval,bar_families):
    """Translate a full accounting length into its physical domain.

    This is a geometry selection, not a strength or homogenization approval.
    An end window is one-sided. Neither the requested length nor physical
    bar circles may be truncated to manufacture a supported window.
    """
    from Design.SMRF_Physical_Floor_Cuts import bar_clear_interval
    station,length=float(station_in),float(length_in)
    lo,hi=map(float,available_interval)
    if not all(math.isfinite(v) for v in (station,length,lo,hi)) or length<=0 or hi<=lo or not lo<=station<=hi:
        raise ValueError('Finite positive axial domain containing the station required')
    out=dict(method='full_length_translated_bar_clear_window_v1',station_in=station,
        requested_length_in=length,available_interval_in=[lo,hi],length_shortened=False,
        material_homogenization_qualified=False)
    if hi-lo<length:
        return dict(out,status='not_evaluated',reason='physical_domain_shorter_than_requested_accounting_length')
    left=min(max(station-length/2,lo),hi-length)
    try:
        interval=bar_clear_interval([left,left+length],[lo,hi],bar_families)
    except ValueError as exc:
        if str(exc)!='A physical mat bar crosses the slab/accounting outer boundary':raise
        return dict(out,status='not_evaluated',reason='bar_clear_window_cannot_fit_inside_actual_domain')
    return dict(out,status='computed',interval_in=interval,
        translation_in=left-(station-length/2),actual_length_in=interval[1]-interval[0])


def connected_web_domain(response,axis,member_id,station_in,trace):
    """Actual contiguous web component containing this selected point trace."""
    if axis not in ('x','y') or trace not in ('minus','plus'):
        raise ValueError('Actual axis and one-sided physical trace required')
    j=0 if axis=='x' else 1;segments=[];station=float(station_in)
    for row in response['elements']:
        if row['kind']!='web' or str(row['member_id'])!=str(member_id):continue
        points=np.asarray(row['positions_in'],float)
        if row['axis']!=axis or points.shape!=(2,3) or not np.isfinite(points).all():
            raise ValueError('Same-axis finite actual web endpoints required')
        a,b=points[:,j]
        if b<=a:raise ValueError('Positive oriented physical web segment required')
        segments.append((float(a),float(b)))
    groups=[]
    for a,b in sorted(segments):
        if groups and a<groups[-1][1]:raise ValueError('Overlapping actual web segments')
        if groups and a==groups[-1][1]:groups[-1][1]=b
        else:groups.append([a,b])
    chosen=[v for v in groups if v[0]<station<v[1] or station==v[0 if trace=='plus' else 1]]
    if len(chosen)!=1:raise ValueError('Unique connected physical web domain required')
    return chosen[0]


def integrate_window(pieces,interval):
    """Exact polynomial integration; every inch and all seven actions required."""
    a,b=map(float,interval)
    if not math.isfinite(a+b) or b<=a or not pieces:raise ValueError('Nonempty finite axial window required')
    cursor=a;integral=np.zeros(7);first=np.zeros(7);audits=[]
    for row in pieces:
        lo,hi=map(float,row['interval_in']);c=np.asarray(row['coefficients'],float)
        if lo!=cursor or hi<=lo or hi>b or c.shape!=(3,7) or not np.isfinite(c).all():
            raise ValueError('Contiguous complete quadratic seven-action window required')
        samples=np.asarray(row['sample_values'],float);quarter=np.asarray(row['quarter_witness'],float)
        matrix=np.array([[1.,0,0],[1.,.5,.25],[1.,1.,1.]])
        if samples.shape!=(3,7) or quarter.shape!=(7,) or not np.isfinite(samples).all() or not np.isfinite(quarter).all():
            raise ValueError('Finite independent physical polynomial witnesses required')
        error=np.vstack((matrix@c-samples,np.array([1.,.25,.0625])@c-quarter))
        target=np.vstack((samples,quarter))
        if np.any(abs(error)>1e-7+1e-7*abs(target)):raise ValueError('Actual axial polynomial witness differs')
        length=hi-lo;mean=np.array([1.,.5,1/3])@c
        moment=np.array([.5,1/3,.25])@c
        integral+=length*mean;first+=length*((lo-a)*mean+length*moment)
        audits.append(dict(interval_in=[lo,hi],integral= (length*mean).tolist(),
                           maximum_polynomial_residual=float(np.max(abs(error)))))
        cursor=hi
    if cursor!=b:raise ValueError('Axial action window is incomplete')
    return dict(average_resultants=(integral/(b-a)).tolist(),integral_resultants=integral.tolist(),
                first_axial_moment_about_window_start=first.tolist(),piece_integrals=audits)


def _binding(action):
    return dict(state_identity=action['identity']['state_identity'],combination_id=action['identity']['combination_id'],
        recovery_id=action['identity']['recovery_id'],
        material_inventory_id=action.get('material_inventory_id'),analysis_input_identity=action.get('analysis_input_identity'),
        axis=action['axis'],cut_width_in=action['cut_width_in'],reference_s_in=action['reference_s_in'],
        reference_z_in=action['reference_z_in'],units=action['units'])


def make_window_evidence(action,interval,pieces,*,source_response_sha256):
    if not source_response_sha256:raise ValueError('Actual source response digest required')
    result=integrate_window(pieces,interval)
    out=dict(method=METHOD,status='computed',binding=_binding(action),interval_in=list(map(float,interval)),
        pieces=pieces,source_response_sha256=source_response_sha256,**result,
        point_demand_replaced=False,local_capacity_bound_claimed=False,mesh_qualification_inferred=False)
    # Keep the full raw polynomial witnesses, not just integration summaries.
    out['pieces']=pieces;out['payload_sha256']=_hash(out);return out


def verify_point_accounting(action,spec):
    """A point target passes only if its complete physical window matches it."""
    evidence=action.get('axial_action_window')
    pending='transverse_bar_length_and_concurrent_action_basis_not_established'
    selection=spec.get('axial_window_selection')
    if selection is not None:
        if selection.get('status')!='computed':
            return dict(status='unknown',pending=[pending],corresponding_action_average_verified=False,
                physical_window_selection_reason=selection.get('reason'))
        a,b=map(float,spec['axial_accounting_interval_in']);lo,hi=map(float,selection['available_interval_in'])
        length=float(selection['requested_length_in']);station=float(spec['station_in'])
        if (selection.get('method')!='full_length_translated_bar_clear_window_v1' or
                not all(math.isfinite(v) for v in (a,b,lo,hi,length,station)) or
                not lo<=a<=station<=b<=hi or length<=0 or b-a<length or
                selection.get('interval_in')!=[a,b] or selection.get('actual_length_in')!=b-a or
                selection.get('station_in')!=station or selection.get('length_shortened') is not False):
            raise ValueError('Computed axial window differs from its full physical domain selection')
    if not evidence or evidence.get('status')!='computed':
        return dict(status='unknown',pending=[pending],corresponding_action_average_verified=False)
    if evidence.get('method')!=METHOD or evidence.get('payload_sha256')!=_hash({k:v for k,v in evidence.items() if k!='payload_sha256'}):
        raise ValueError('Axial action-window evidence changed or method differs')
    if (evidence['binding']!=_binding(action) or evidence['interval_in']!=list(spec['axial_accounting_interval_in']) or
            evidence.get('source_response_sha256')!=action.get('source_response_sha256')):
        raise ValueError('Axial action-window actual material/case/source/geometry differs')
    result=integrate_window(evidence['pieces'],evidence['interval_in'])
    for key in ('average_resultants','integral_resultants','first_axial_moment_about_window_start','piece_integrals'):
        if result[key]!=evidence[key]:raise ValueError('Saved axial action integration differs from raw polynomial witnesses')
    value=np.asarray(action['resultants'],float);average=np.asarray(result['average_resultants'])
    if value.shape!=(7,) or not np.isfinite(value).all():raise ValueError('Seven finite concurrent actions required')
    residual=value-average;tol=1e-6+1e-6*np.maximum(abs(value),abs(average))
    passed=bool(np.all(abs(residual)<=tol))
    return dict(status='pass' if passed else 'unknown',pending=[] if passed else [pending],
        corresponding_action_average_verified=passed,point_resultants=value.tolist(),average_resultants=average.tolist(),
        residual=residual.tolist(),tolerance=tol.tolist(),window_evidence_sha256=evidence['payload_sha256'],
        point_demand_replaced=False,local_capacity_bound_claimed=False)


def physical_window_evidence(response,spec,action,*,uniform_web_loads):
    """Integrate the original piecewise bilinear shell/quadratic web fields.

    The producer still keeps the original point target. Crossing an unresolved
    joint or a missing actual web gives an explicit domain gap, not an average
    assembled from the pieces that happened to exist.
    """
    from Design.SMRF_Physical_Floor_Cuts import common_strip_cut
    from Design.SMRF_Primary_Moment_Guards import physical_cut_states
    selection=spec.get('axial_window_selection')
    if selection is not None and selection['status']!='computed':
        return dict(method=METHOD,status='not_evaluated',reason=selection['reason'],
            interval_in=spec['axial_accounting_interval_in'],point_demand_replaced=False,
            mesh_qualification_inferred=False)
    a,b=map(float,spec['axial_accounting_interval_in']);j=0 if spec['axis']=='x' else 1
    breaks={a,b};lo,hi=map(float,spec['interval_in']);selected=[]
    for row in response['elements']:
        points=np.asarray(row['positions_in'],float)
        if points[:,j].max()<=a or points[:,j].min()>=b:continue
        if row['kind']=='slab':
            if points[:,1-j].max()<=lo or points[:,1-j].min()>=hi:continue
        elif str(row.get('member_id'))!=str(spec.get('member_id')):continue
        selected.append(row);breaks.update(float(p[j]) for p in row['positions_in'] if a<float(p[j])<b)
    local=dict(elements=selected,force_partition_relative_error=response['force_partition_relative_error'])
    breaks=sorted(breaks);pieces=[];matrix=np.array([[1.,0,0],[1.,.5,.25],[1.,1.,1.]])
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        values=[]
        for fraction in (0.,.5,1.,.25):
            station=lo+fraction*(hi-lo)
            try:
                raw=common_strip_cut(local,axis=spec['axis'],station_in=station,interval_in=spec['interval_in'],
                    trace='minus' if fraction==1 else 'plus',web_id=spec.get('member_id'),uniform_web_loads=uniform_web_loads,
                    state_identity=action['identity']['state_identity'],combination_id=action['identity']['combination_id'],
                    location_id=action['identity']['location_id'],recovery_id=action['identity']['recovery_id'])
            except ValueError as exc:
                if str(exc) not in ('Common cut needs exactly one physical web trace outside its joint',
                    'Slab cut contains a gap or overlapping physical elements','Slab cut does not cover its entire declared physical interval'):
                    raise
                return dict(method=METHOD,status='not_evaluated',reason=str(exc),interval_in=[a,b],
                    unresolved_interval_in=[lo,hi],point_demand_replaced=False,mesh_qualification_inferred=False)
            normalized,_=physical_cut_states([raw],outer_interval_in=spec['interval_in']);values.append(normalized[0]['resultants'])
        coefficients=np.linalg.solve(matrix,np.asarray(values[:3],float))
        pieces.append(dict(interval_in=[lo,hi],coefficients=coefficients.tolist(),sample_values=values[:3],quarter_witness=values[3]))
    return make_window_evidence(action,[a,b],pieces,source_response_sha256=action['source_response_sha256'])


def primary_moment_window_evidence(guard,physical,gravity_effective,physical_effective,*,
        projector,primary_increment,component,strip_interval,owns_web,primary_weak_increment_only=False):
    """Exact separate seven-action field for a primary moment demand guard.

    The original physical polynomials, exact effective-flange gravity and
    physical fields, and the affine unloaded primary increment determine the
    guard at every station. The full-T projection is linear and its geometry
    is fixed over this member domain. Integrating the pointwise guard does not
    assert that its average bounds point strength or qualifies bar smearing.
    """
    missing=dict(method=METHOD,status='not_evaluated',reason='primary_guard_axial_sources_incomplete',
                 point_demand_replaced=False,mesh_qualification_inferred=False)
    if component not in ('strong','weak','both'):raise ValueError('Named primary moment component required')
    sources=(physical,gravity_effective,physical_effective);windows=[]
    for action in sources:
        w=action.get('axial_action_window')
        if not w or w.get('status')!='computed':return missing
        if (w.get('method')!=METHOD or w.get('payload_sha256')!=_hash({k:v for k,v in w.items() if k!='payload_sha256'})
                or w['binding']!=_binding(action) or w['source_response_sha256']!=action.get('source_response_sha256')):
            raise ValueError('Primary guard axial source binding or payload differs')
        integration=integrate_window(w['pieces'],w['interval_in'])
        if any(integration[k]!=w[k] for k in integration):raise ValueError('Primary guard source polynomial integral differs')
        windows.append(w)
    a,b=windows[0]['interval_in']
    if any(w['interval_in']!=[a,b] for w in windows):
        return dict(missing,reason='primary_guard_sources_have_different_axial_windows')
    inc=primary_increment;cut=inc.get('actual_primary_cut',{})
    domain=cut.get('physical_member_interval_in');derivative=cut.get('normal_derivative_per_in')
    if domain is None or derivative is None:return dict(missing,reason='primary_increment_axial_law_or_domain_missing')
    from Design.SMRF_Physical_Floor_Cuts import identity
    if cut.get('identity_sha256')!=identity({k:v for k,v in cut.items() if k!='identity_sha256'}):
        raise ValueError('Primary increment actual affine law payload differs')
    dr=np.asarray(derivative,float);span=np.asarray(domain,float)
    if dr.shape!=(3,) or span.shape!=(2,) or not np.isfinite(dr).all() or not np.isfinite(span).all() or span[1]<=span[0]:
        raise ValueError('Finite affine primary law and actual member domain required')
    if not span[0]<=a<b<=span[1]:return dict(missing,reason='primary_guard_window_outside_actual_primary_member')
    if (cut['axis']!=projector.axis or cut['station_in']!=inc['station_in'] or dr[0]!=0. or
            not np.allclose(dr[1:],[-cut['Q_vertical_kip'],cut['V_inplane_kip']],rtol=0.,atol=1e-12) or
            cut['state_identity']!=physical['identity']['state_identity'] or
            cut['physical_gravity_case_id']!=gravity_effective['identity']['combination_id'] or
            not np.allclose([cut['M_centroid_kipin'],cut['Ms_transverse_first_moment_kipin']],
                            [inc['m_z_kip_in'],inc['m_s_kip_in']],rtol=0.,atol=1e-10)):
        raise ValueError('Primary affine source differs from the actual signed guard target')
    j=projector.j
    def value(window,x,trace):
        selected=[p for p in window['pieces'] if p['interval_in'][0]<x<p['interval_in'][1] or
            x==p['interval_in'][1 if trace=='minus' else 0]]
        if len(selected)!=1:raise ValueError('Unique one-sided primary window polynomial required')
        p=selected[0];lo,hi=p['interval_in'];t=(x-lo)/(hi-lo)
        return np.array([1.,t,t*t])@np.asarray(p['coefficients'])
    def normal(v,action):
        return np.array([v[j],v[j+3],v[6]+(action['reference_s_in']-projector.sref)*v[j]])
    def recover(x,trace):
        original=value(windows[0],x,trace)
        g=normal(value(windows[1],x,trace),gravity_effective)
        p=normal(value(windows[2],x,trace),physical_effective)
        delta=g+np.array([0.,inc['m_z_kip_in'],inc['m_s_kip_in']])+dr*(x-inc['station_in'])-p
        dz=delta[1] if component in ('strong','both') else 0.
        ds=delta[2] if component in ('weak','both') else 0.
        if primary_weak_increment_only and component in ('weak','both'):
            ds=inc['m_s_kip_in']+dr[2]*(x-inc['station_in'])
        projection=projector.project(n_tension_kip=0.,m_z_kip_in=dz,m_s_kip_in=ds,increment_id='axial_guard_projection')
        # partition() also supplies the actual web only to its whole owner.
        parts=[];lo,hi=map(float,strip_interval);left,right=projector.flange
        outer=[min(left,lo),max(right,hi)]
        if lo>outer[0]:parts.append(dict(id='left',interval_in=[outer[0],lo]))
        parts.append(dict(id='target',interval_in=[lo,hi]))
        if hi<outer[1]:parts.append(dict(id='right',interval_in=[hi,outer[1]]))
        split=projector.partition(projection,parts)
        part=next(p for p in split['strips'] if p['id']=='target')
        # A mismatched material owner is an error even if its current N is0.
        intersects_web=min(hi,projector.web[1])>max(lo,projector.web[0])
        if intersects_web!=bool(owns_web):raise ValueError('Primary axial projection web ownership differs')
        r=np.asarray(part['integrated_six_resultants']);n=r[j]
        ms=part['transverse_first_moment_kip_in']+(projector.sref-physical['reference_s_in'])*n
        return original+np.r_[r,ms]
    station=inc['station_in']
    # The actual point is always reproduced by the same polynomial side that
    # generated its base action; at a jump test both, require an exact match.
    traces=['plus','minus'] if a<station<b else ['plus' if station==a else 'minus']
    matches=[t for t in traces if np.allclose(value(windows[0],station,t),physical['resultants'],rtol=1e-7,atol=1e-7)]
    if not matches or not any(np.allclose(recover(station,t),guard['resultants'],rtol=1e-7,atol=1e-7) for t in matches):
        raise ValueError('Primary guard axial law does not reproduce its original point demand')
    breaks=sorted({v for w in windows for p in w['pieces'] for v in p['interval_in']})
    pieces=[];matrix=np.array([[1.,0.,0.],[1.,.5,.25],[1.,1.,1.]])
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        samples=[recover(lo+f*(hi-lo),'minus' if f==1. else 'plus') for f in (0.,.5,1.,.25)]
        coeff=np.linalg.solve(matrix,np.asarray(samples[:3]))
        pieces.append(dict(interval_in=[lo,hi],coefficients=coeff.tolist(),sample_values=[v.tolist() for v in samples[:3]],quarter_witness=samples[3].tolist()))
    result=make_window_evidence(guard,[a,b],pieces,source_response_sha256=guard['source_response_sha256'])
    result['derivation']=dict(method='pointwise_primary_moment_projection_axial_integrals_v1',component=component,
        primary_weak_increment_only=bool(primary_weak_increment_only),
        source_window_sha256=[w['payload_sha256'] for w in windows],primary_affine_cut_sha256=cut['identity_sha256'],
        original_point_reproduced=True,physical_equilibrium_solution_claimed=False)
    result['payload_sha256']=_hash({k:v for k,v in result.items() if k!='payload_sha256'})
    return result


def _verified_window(action):
    w=action.get('axial_action_window')
    if not w or w.get('status')!='computed':return None
    if (w.get('method')!=METHOD or w.get('payload_sha256')!=_hash({k:v for k,v in w.items() if k!='payload_sha256'})
            or w['binding']!=_binding(action) or w['source_response_sha256']!=action.get('source_response_sha256')):
        raise ValueError('Axial source binding or payload differs')
    integration=integrate_window(w['pieces'],w['interval_in'])
    if any(integration[k]!=w[k] for k in integration):raise ValueError('Axial source integral differs')
    return w


def _window_value(window,x,trace):
    chosen=[p for p in window['pieces'] if p['interval_in'][0]<x<p['interval_in'][1] or
        x==p['interval_in'][1 if trace=='minus' else 0]]
    if len(chosen)!=1:raise ValueError('Unique one-sided axial polynomial required')
    p=chosen[0];lo,hi=p['interval_in'];t=(x-lo)/(hi-lo)
    return np.array([1.,t,t*t])@np.asarray(p['coefficients'])


def linear_combination_window(guard,sources,maps,*,derivation_method):
    """Retain all seven components of an explicit constant linear action map."""
    if not derivation_method or not sources or len(sources)!=len(maps):raise ValueError('Named complete axial action map required')
    matrices=[np.asarray(m,float) for m in maps]
    if any(m.shape!=(7,7) or not np.isfinite(m).all() for m in matrices):raise ValueError('Finite seven-by-seven action maps required')
    for s in sources:
        if (s['identity']['state_identity']!=guard['identity']['state_identity'] or
                any(s.get(k)!=guard.get(k) for k in ('axis','units','reference_z_in','cut_width_in','material_inventory_id','analysis_input_identity'))):
            raise ValueError('Axial combination physical material/state/axes differ')
    windows=[_verified_window(s) for s in sources]
    missing=dict(method=METHOD,status='not_evaluated',reason='guard_axial_sources_incomplete',
        point_demand_replaced=False,mesh_qualification_inferred=False)
    if any(w is None for w in windows):return missing
    interval=windows[0]['interval_in']
    if any(w['interval_in']!=interval for w in windows):return dict(missing,reason='guard_sources_have_different_axial_windows')
    point=sum((m@np.asarray(s['resultants']) for m,s in zip(matrices,sources)),np.zeros(7))
    if not np.allclose(point,guard['resultants'],rtol=1e-7,atol=1e-7):raise ValueError('Axial combination differs from original guard point')
    breaks=sorted({x for w in windows for p in w['pieces'] for x in p['interval_in']});pieces=[]
    fit=np.array([[1.,0.,0.],[1.,.5,.25],[1.,1.,1.]])
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        samples=[]
        for f in (0.,.5,1.,.25):
            x=lo+f*(hi-lo);trace='minus' if f==1. else 'plus'
            samples.append(sum((m@_window_value(w,x,trace) for m,w in zip(matrices,windows)),np.zeros(7)))
        coeff=np.linalg.solve(fit,np.asarray(samples[:3]))
        pieces.append(dict(interval_in=[lo,hi],coefficients=coeff.tolist(),sample_values=[r.tolist() for r in samples[:3]],quarter_witness=samples[3].tolist()))
    result=make_window_evidence(guard,interval,pieces,source_response_sha256=guard['source_response_sha256'])
    result['derivation']=dict(method=derivation_method,source_window_sha256=[w['payload_sha256'] for w in windows],
        action_maps=[m.tolist() for m in matrices],original_point_reproduced=True,physical_equilibrium_solution_claimed=False)
    result['payload_sha256']=_hash({k:v for k,v in result.items() if k!='payload_sha256'})
    return result


def primary_membrane_axial_action(reader,spec,raw,verified_source):
    """Recover the named P5 branch over its actual complete Q4/rod window.

    This uses the original planar fields and retains every element break and
    one-sided trace. It supplies only the prescribed membrane action, including
    its actual transverse first moment. It adds no fictitious plate bending.
    """
    import copy
    from Design.SMRF_Primary_Moment_Guards import primary_membrane_source_from_readout,physical_cut_states
    physical=physical_cut_states([raw],outer_interval_in=raw['slab']['interval_in'])[0][0]
    cid=raw['identity']['combination_id'];branch=verified_source['branch_id'];level=spec['floor']
    floor=reader.floor(level);j=0 if spec['axis']=='x' else 1;sref=physical['reference_s_in']
    def vector(source):
        v=np.zeros(7);v[:3]=source['slab_membrane_integral_kip']
        v[6]=source['slab_membrane_global_first_moment_kipin'][j]-sref*v[j]
        web=source.get('web')
        if web is not None:
            v[j]+=web['N_tension_kip'];v[6]+=(web['transverse_reference_in']-sref)*web['N_tension_kip']
        return v
    action={k:copy.deepcopy(physical[k]) for k in ('identity','material_inventory_id','analysis_input_identity',
        'axis','cut_width_in','reference_s_in','reference_z_in','units')}
    action['identity']['recovery_id']+='|prescribed_membrane:'+branch
    action.update(resultants=vector(verified_source).tolist(),source_response_sha256=floor['payload_sha256'])
    missing=dict(method=METHOD,status='not_evaluated',reason='primary_membrane_axial_domain_missing',
        point_demand_replaced=False,mesh_qualification_inferred=False)
    action['axial_action_window']=missing
    if 'axial_accounting_interval_in' not in spec:return action
    selection=spec.get('axial_window_selection')
    if selection is not None and selection.get('status')!='computed':
        missing['reason']=selection.get('reason','primary_membrane_axial_domain_missing');return action
    a,b=map(float,spec['axial_accounting_interval_in'])
    key=_hash(dict(binding=_binding(action),interval=[a,b],selection=selection,member=spec.get('member_id'),
                   source=floor['payload_sha256']))
    cache=reader._axial_action_windows
    def match_point(window):
        at=float(spec['station_in']);trace=spec['trace']
        if at==a:trace='plus'
        elif at==b:trace='minus'
        if not np.allclose(_window_value(window,at,trace),action['resultants'],rtol=1e-7,atol=1e-7):
            raise ValueError('P5 axial field differs from its actual original point')
    if key in cache:
        match_point(cache[key]);action['axial_action_window']=copy.deepcopy(cache[key]);return action
    coordinates=np.asarray(floor['mesh']['nodes'],float)[:,j]
    if not float(min(coordinates))<=a<b<=float(max(coordinates)):return action
    breaks=sorted({a,b,*[float(x) for x in coordinates if a<x<b]});pieces=[]
    fit=np.array([[1.,0.,0.],[1.,.5,.25],[1.,1.,1.]])
    point_ids=[]
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        samples=[]
        for f in (0.,.5,1.,.25):
            x=lo+f*(hi-lo);trace='minus' if f==1. else 'plus'
            cut=reader.cut(level,cid,branch,axis=spec['axis'],station_in=x,interval_in=raw['slab']['interval_in'],
                trace=trace,transverse_reference_in=sref)
            source=primary_membrane_source_from_readout(cut,identity=raw['identity'])
            web=source.get('web');expected=raw.get('web')
            if (web is None)!=(expected is None) or (web is not None and str(web['member_id'])!=str(expected['member_id'])):
                action['axial_action_window']=dict(missing,reason='primary_membrane_web_ownership_changes_inside_window')
                return action
            samples.append(vector(source));point_ids.append(cut['identity_sha256'])
        coeff=np.linalg.solve(fit,np.asarray(samples[:3]))
        pieces.append(dict(interval_in=[lo,hi],coefficients=coeff.tolist(),sample_values=[r.tolist() for r in samples[:3]],quarter_witness=samples[3].tolist()))
    window=make_window_evidence(action,[a,b],pieces,source_response_sha256=action['source_response_sha256'])
    match_point(window)
    window['derivation']=dict(method='actual_prescribed_primary_membrane_axial_field_v1',
        branch_id=branch,source_floor_sha256=floor['payload_sha256'],point_readout_sha256=point_ids,
        original_point_reproduced=True,plate_bending_added=False)
    window['payload_sha256']=_hash({k:v for k,v in window.items() if k!='payload_sha256'})
    cache[key]=copy.deepcopy(window);action['axial_action_window']=window
    return action
