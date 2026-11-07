"""Exact concurrent9.2.4.4 beam torsion cuts from actual floor responses.

Raw ShellMITC4 positive-face vertical traction is +Q; the physical web's
diagram shear has positive-face traction -Qweb. Native K face-force and
twisting-couple witnesses fix this distinction. Every original gravity,
physical ELF and signed compatibility state is retained. This producer
performs algebra only; it does not qualify a mesh or create unused steel.
"""
from __future__ import annotations
import math
import numpy as np
import Structure_Parameters as sp
from Model import Member_Properties as mp
from Design.SMRF_Physical_Floor_Cuts import common_strip_cut,identity
from Design.SMRF_Beam_Torsion_Adapter import actual_member_torsion_section

METHOD='actual_same_section_torsion_recovery_v2_guard_and_detail_stations'
FIELDS=('N_tension_kip','T_about_positive_axis_kipin','Q_vertical_kip','V_inplane_kip',
        'M_centroid_kipin','Ms_transverse_first_moment_kipin')


def rectangle_union_centroid(rectangles):
    """Exact area/centroid of the same solid union used by ACI22.7."""
    xs=sorted({v for r in rectangles for v in r[:2]});zs=sorted({v for r in rectangles for v in r[2:]})
    area=sx=sz=0.
    for a,b in zip(xs[:-1],xs[1:]):
        for c,d in zip(zs[:-1],zs[1:]):
            x,z=(a+b)/2,(c+d)/2
            if any(r[0]<x<r[1] and r[2]<z<r[3] for r in rectangles):
                da=(b-a)*(d-c);area+=da;sx+=da*x;sz+=da*z
    if area<=0:raise ValueError('Positive actual torsion section union required')
    return dict(area_in2=area,transverse_in=sx/area,z_from_top_in=sz/area)


def cut_wrench(cut,*,reference_s_in,reference_z_in):
    """Direct positive-face r x traction, about an explicit physical line.

    Mxy=int(z*tau_xy); its X-face couple is -Mxy and Y-face +Mxy.
    Q_vertical is the beam diagram convention (negative positive-face Fz).
    V_inplane is the positive-face global transverse traction, as web_section.
    """
    slab,web=cut['slab'],cut['web'];j=0 if slab['axis']=='x' else 1;sign=1. if j==0 else -1.
    sr,zr=float(reference_s_in),float(reference_z_in)
    q=float(slab['integral'][6+j]);nxy=float(slab['integral'][2]);mxy=float(slab['integral'][5])
    first=float(slab['transverse_first_moment'][6+j])-sr*q
    slab_t=sign*(first-mxy+zr*nxy)
    web_t=web['T_about_positive_axis_kipin']-sign*((web['transverse_reference_in']-sr)*web['Q_vertical_kip']+
        (web['centroid_z_in']-zr)*web['V_inplane_kip'])
    n=float(cut['resultants'][j]);m=float(cut['resultants'][j+3])-zr*n
    ms=float(cut['Ms_transverse_first_moment_kipin'])+(cut['transverse_reference_in']-sr)*n
    return dict(N_tension_kip=n,T_about_positive_axis_kipin=float(slab_t+web_t),
        Q_vertical_kip=float(web['Q_vertical_kip']-q),V_inplane_kip=float(web['V_inplane_kip']+nxy),
        M_centroid_kipin=m,Ms_transverse_first_moment_kipin=ms,
        reference_s_in=sr,reference_z_in=zr,
        torque_components_kipin=dict(slab=slab_t,transported_web=web_t),
        positive_face_Fz_kip=float(q-web['Q_vertical_kip']))


def recover_cut(actual,*,section,station_in,trace,uniform_web_loads,state_identity,
                case_id,cut_id,recovery_id,slab_thickness_in,beam_depth_in):
    """One physical cut with full same-selected-section N and signed T/Q/V."""
    if section.get('status')!='pass':raise ValueError('Actual computable9.2.4.4 section required')
    line=section['global_transverse_origin_in'];rects=section['rectangles']
    interval=[line+min(r[0] for r in rects),line+max(r[1] for r in rects)]
    raw=common_strip_cut(actual,axis=section['axis'],station_in=station_in,interval_in=interval,trace=trace,
        web_id=str(section['member_tag']),uniform_web_loads=uniform_web_loads,state_identity=state_identity,
        combination_id=case_id,location_id=cut_id,recovery_id=recovery_id)
    zr=(slab_thickness_in-beam_depth_in)/2
    selected=cut_wrench(raw,reference_s_in=line,reference_z_in=zr)
    centroid=rectangle_union_centroid(rects)
    if not math.isclose(centroid['area_in2'],section['Ag_in2'],rel_tol=0,abs_tol=1e-8):
        raise ValueError('Torsional axial section union differs from selected22.7 area')
    central=cut_wrench(raw,reference_s_in=line+centroid['transverse_in'],
        reference_z_in=slab_thickness_in/2+centroid['z_from_top_in'])
    return dict(status='pass',member_tag=section['member_tag'],case_id=case_id,cut_id=cut_id,
        state_identity=state_identity,recovery_id=recovery_id,section_id=section['section_id'],
        same_section_axial_recovered=True,station_in=float(station_in),trace=trace,interval_in=interval,
        reference_basis='actual beam analysis line and gross-depth midplane',**selected,
        section_centroid=centroid,centroid_wrench=central,raw_cut=raw,
        force_partition_relative_error=actual.get('force_partition_relative_error'))


def analysis_case_inventory(common,bundles):
    """Explicit source/raw-case associations, never a mixed N/T envelope."""
    original={c['id']:c for c in common['combinations']};gmap=common['gravity_case_mapping']
    if len(original)!=len(common['combinations']) or set(gmap)!=set(original):
        raise ValueError('Exact unique original combination and gravity mapping required')
    cases=[];excluded=[];seen=set();direct=set()
    for bundle in bundles:
        sid=bundle['source_id'];response=bundle['response']
        for raw in response['cases']:
            rid=raw['id'];key=sid+'/'+rid
            if key in seen:raise ValueError('Duplicate source/raw physical case')
            seen.add(key)
            if rid in original:
                originals=[rid];direct.add(rid);kind='actual_physical_ELF'
                if raw.get('gravity_case_id')!=gmap[rid]:raise ValueError('Physical ELF case has a different gravity baseline')
            elif rid in set(gmap.values()):
                originals=sorted(k for k,v in gmap.items() if v==rid);kind='actual_gravity'
                direct.update(k for k in originals if not(original[k]['ex'] or original[k]['ey']))
            elif raw.get('branch')=='physical_deformation_compatibility':
                gid=raw.get('gravity_case_id');originals=sorted(k for k,v in gmap.items() if v==gid and (original[k]['ex'] or original[k]['ey']))
                if not originals or not raw.get('imposed'):raise ValueError('Compatibility needs actual excitation and seismic gravity family')
                kind='actual_signed_Cd_compatibility'
            elif rid=='service_DL':
                excluded.append(dict(source_id=sid,raw_case_id=rid,basis='service reference only; not an original strength combination'))
                continue
            else:raise ValueError('Unmapped physical torsion source case: '+key)
            for name in originals:
                if any(not math.isclose(float(raw[k]),float(original[name][k]),rel_tol=0,abs_tol=1e-12) for k in ('dead','live')):
                    raise ValueError('Physical torsion case differs from its actual gravity coefficients')
                if raw.get('live_pattern','all')!=original[name].get('live_pattern','all'):
                    raise ValueError('Physical torsion case differs from its actual live pattern')
            cases.append(dict(analysis_case_id=key,source_id=sid,raw_case_id=rid,original_case_ids=originals,
                branch=kind,source_excitation=raw))
    if direct!=set(original):raise ValueError('Original gravity/ELF physical case coverage is incomplete')
    return dict(cases=cases,excluded_reference_cases=excluded,expected_case_ids=list(original))


def polynomial_stations(values,section):
    """Scalar stationary points and exact |T|/sqrt(1-N/k) extrema.

    values are the exact start/mid/end field. The latter ratio can peak
    inside an affine N/T interval even when neither component does. Domain
    boundaries are retained, never treated as a threshold exemption.
    """
    values=np.asarray(values,float)
    if values.shape!=(3,len(FIELDS)) or not np.isfinite(values).all():raise ValueError('Three complete finite concurrent samples required')
    c=np.linalg.solve(np.array([[1,0,0],[1,.5,.25],[1,1,1]]),values)
    if np.max(abs(c[2,:4]))>1e-8+1e-9*np.max(abs(values)):
        raise ValueError('Torsion/axial/shear clear-interval fields must be affine')
    points={0.,.5,1.};reasons={0.:'face_or_mesh_trace',.5:'interval_midpoint',1.:'face_or_mesh_trace'}
    for b,d in zip(c[1],c[2]):
        if abs(d)>1e-12:
            x=-b/(2*d)
            if 1e-10<x<1-1e-10:points.add(float(x));reasons[float(x)]='scalar_stationary_point'
    scale=4*section['Ag_in2']*section['lambda']*min(100.,math.sqrt(section['fc_ksi']*1000))/1000
    r0,r1=1-c[0,0]/scale,-c[1,0]/scale;t0,t1=c[0,1],c[1,1]
    if abs(r1)>1e-14:
        x=-r0/r1
        if 1e-10<x<1-1e-10:points.add(float(x));reasons[float(x)]='threshold_axial_formula_domain_boundary'
    if abs(t1*r1)>1e-14:
        x=(r1*t0-2*t1*r0)/(t1*r1)
        if 1e-10<x<1-1e-10 and r0+r1*x>0:
            points.add(float(x));reasons[float(x)]='concurrent_threshold_ratio_stationary_point'
    return c,[(x,reasons[x]) for x in sorted(points)]


def recover_common_torsion(state,common,*,maximum_records,include_strength_inputs=False,
                          axial_accounting_length_in=24.,strength_inputs_above_threshold_only=False,
                          prepared_sources=None,selected_member_tags=None):
    """Complete bounded physical algebra inventory; above-threshold reserve pending.

    The cut union spans every actual beam clear face, FE/web boundary and
    concurrent scalar/threshold stationary point from every physical branch.
    Every retained source/raw case is evaluated at the complete union. This
    does not prove continuous nonlinear reinforcement-reserve interaction or
    qualify the FE traces. Those remain mandatory independent method checks.
    """
    from Design.SMRF_Common_Action_Analysis import physical_response_bundles
    from Design.SMRF_Primary_Moment_Guards import physical_cut_states
    from Design.SMRF_Physical_Floor_Cuts import bar_clear_interval
    from Design.SMRF_Common_Guard_Plan import PreparedCommonGuardSources
    from Design.SMRF_Common_Torsion_Guards import plan_record
    from Design.SMRF_Beam_Torsion_Detail import member_detail
    from Design.SMRF_Torsion_Station_Coverage import interval_points,detail_boundaries
    if type(maximum_records) is not int or maximum_records<1:raise ValueError('Explicit positive torsion record budget required')
    if not math.isfinite(axial_accounting_length_in) or axial_accounting_length_in<=0:raise ValueError('Positive physical accounting length required')
    context=PreparedCommonGuardSources(common) if prepared_sources is None else prepared_sources.validate(common)
    definition,inventory=common['definition'],common['inventory']
    bundles=physical_response_bundles(common) if context is None else context.bundles
    if any(v!=state.identity() for v in (common.get('state_identity'),definition.get('state_identity'),inventory.get('state_identity'))):
        raise ValueError('Physical geometry/material/grouped identities differ')
    if common['definition_sha256']!=definition['identity_sha256'] or common['analysis_source_identity']!=identity(common['source_identity']):
        raise ValueError('Physical torsion source or geometry digest differs')
    bysource={b['source_id']:b for b in bundles}
    if len(bysource)!=len(bundles):raise ValueError('Unique physical response sources required')
    desc={int(f['level']):f for f in definition['floors']};actual={};grids={}
    for b in bundles:
        r=b['response'];sid=b['source_id']
        if (r.get('definition_sha256')!=definition['identity_sha256'] or b.get('material_inventory_id')!=inventory['identity_sha256']
                or b.get('analysis_input_identity')!=common['input_signature'] or b.get('response_payload_sha256')!=(identity(r) if context is None else context.physical.digest(b))):
            raise ValueError('Actual torsion response has different geometry/material/input/payload')
        if {int(f['level']) for f in r['floors']}!=set(desc):raise ValueError('Every actual response must cover every floor')
        required={c['id'] for c in r['cases']}
        for floor in r['floors']:
            f=int(floor['level']);grids[sid,f]=floor['grid']
            if {a['case_id'] for a in floor['responses']}!=required or len(floor['responses'])!=len(required):
                raise ValueError('Every physical floor requires every raw case exactly once')
            for a in floor['responses']:actual[sid,a['case_id'],f]=a
    mapping=analysis_case_inventory(common,bundles);cases=mapping['cases']
    slab_fc=min(f['fc_ksi'] for f in definition['floors']);sections={};cuts={};records=[];audits=[];specs=[];actions={};guard_audits=[];details={}
    members=[m for m in state.members() if not m.is_column]
    full_member_tags=[m.member_tag for m in members]
    if selected_member_tags is not None:
        selected=set(selected_member_tags)
        if not selected or len(selected)!=len(selected_member_tags) or not selected<=set(full_member_tags):raise ValueError('Explicit distinct actual torsion member subset required')
        members=[m for m in members if m.member_tag in selected]
    source_id=identity(dict(method=METHOD,input=common['input_signature'],state=state.identity(),
        responses={b['source_id']:b['response_payload_sha256'] for b in bundles},mapping=mapping))
    for member in members:
        tag=member.member_tag;floor=member.story_or_floor;section=actual_member_torsion_section(member,slab_fc);sections[str(tag)]=section
        if section.get('status')!='pass':raise ValueError('Actual torsional section is unavailable for member '+str(tag))
        axis=section['axis'];j=0 if axis=='x' else 1;length=mp.beam_span_in(member)
        start=(member.grid_i if axis=='x' else member.grid_j)*(sp.BAY_X if axis=='x' else sp.BAY_Y)
        fi,fj=mp.beam_face_distances_in(member);lo,hi=start+fi,start+length-fj
        breaks={lo,hi}
        detail=member_detail(state,inventory,member,slab_fc_ksi=slab_fc)
        details[str(tag)]=dict(evidence_id=detail['evidence_id'],status=detail['splice_detail']['status'],
            boundaries_from_face_in=detail_boundaries(detail,hi-lo,member.design.stirrup_spacing_in))
        breaks.update(lo+x for x in details[str(tag)]['boundaries_from_face_in'])
        breaks.update(float(v) for v in context.readout.floor(floor)['mesh'][axis+'_in'] if lo<float(v)<hi)
        for b in bundles:
            breaks.update(float(v) for v in grids[b['source_id'],floor][axis+'_in'] if lo<float(v)<hi)
            for row in actual[b['source_id'],b['response']['cases'][0]['id'],floor]['elements']:
                if row['kind']=='web' and str(row['member_id'])==str(tag):
                    breaks.update(float(p[j]) for p in row['positions_in'] if lo<float(p[j])<hi)
        breaks=sorted(breaks);locations={};cache={}
        def recover(case,station,trace):
            key=(case['analysis_case_id'],float(station),trace)
            if key not in cache:
                sid,rid=case['source_id'],case['raw_case_id'];a=actual[sid,rid,floor]
                q={e['id']:-float(case['source_excitation']['dead'])*definition['gamma_kip_in3']*e['section']['b_in']*
                    (e['section']['h_in']-definition['slab_thickness_in']) for e in a['elements'] if e['kind']=='web'}
                cid=f'member_{tag}/{station:.15g}/{trace}'
                r=recover_cut(a,section=section,station_in=station,trace=trace,uniform_web_loads=q,
                    state_identity=state.identity(),case_id=case['analysis_case_id'],cut_id=cid,
                    recovery_id=sid+'/'+bysource[sid]['response_payload_sha256'],
                    slab_thickness_in=definition['slab_thickness_in'],beam_depth_in=member.design.h_in)
                r.update(source_id=sid,raw_case_id=rid,original_case_ids=case['original_case_ids'],
                    source_response_sha256=bysource[sid]['response_payload_sha256'],material_inventory_id=inventory['identity_sha256'],
                    analysis_input_identity=common['input_signature'],definition_sha256=definition['identity_sha256'])
                cache[key]=r
            return cache[key]
        for a,b in zip(breaks[:-1],breaks[1:]):
            for case in cases:
                samples=[recover(case,x,trace) for x,trace in [(a,'plus'),((a+b)/2,'plus'),(b,'minus')]]
                values=[[r[k] for k in FIELDS] for r in samples];coeff,points=polynomial_stations(values,section)
                witness=recover(case,a+(b-a)/3,'plus');pred=np.array([1,1/3,1/9])@coeff
                err=pred-np.array([witness[k] for k in FIELDS])
                if np.any(abs(err)>1e-8+1e-9*np.maximum(abs(pred),np.abs([witness[k] for k in FIELDS]))):
                    raise ValueError('Concurrent torsion field failed independent polynomial witness')
                audits.append(dict(member_tag=tag,case_id=case['analysis_case_id'],interval_in=[a,b],
                    coefficients=coeff.tolist(),independent_third_residual=err.tolist()))
                for x,why in points:
                    station=a+x*(b-a);trace='minus' if x==1 else 'plus'
                    locations.setdefault((station,trace),set()).add(why)
                # Physical G and Cd cases have only their already-sampled
                # physical guards. Original signed ELF also needs all P5 and
                # primary projected N/T extrema before threshold exemption.
                original=next((c for c in common['combinations'] if c['id']==case['raw_case_id']),None)
                if original is not None and (original['ex'] or original['ey']):
                    def provider(x,trace):
                        return plan_record(state,common,recover(case,a+x*(b-a),trace),section,
                            maximum_cuts=maximum_records,prepared_sources=context,include_material=False)
                    guarded,proof=interval_points(provider,section)
                    guard_audits.append(dict(member_tag=tag,case_id=case['analysis_case_id'],interval_in=[a,b],guards=proof))
                    for x,reasons in guarded:
                        locations.setdefault((a+x*(b-a),'minus' if x==1 else 'plus'),set()).update(reasons)
        if len(records)+len(cases)*len(locations)>maximum_records:raise ValueError('Complete torsion case/cut inventory exceeds explicit bound; no truncation')
        cuts[str(tag)]=[f'member_{tag}/{at:.15g}/{trace}' for at,trace in sorted(locations)]
        for case in cases:
            for (station,trace),why in sorted(locations.items()):
                r=dict(recover(case,station,trace));raw=r.pop('raw_cut');r['sampling_kinds']=sorted(why)
                r['longitudinal_state']=None;r['development']=None;r['shear_reservations']=None
                records.append(r)
                from Design.SMRF_Beam_Torsion import threshold
                r['torsion_threshold']=threshold(section,r['T_about_positive_axis_kipin'],r['N_tension_kip'])
                need_strength=not(strength_inputs_above_threshold_only and r['torsion_threshold'].get('below_threshold'))
                if include_strength_inputs and need_strength:
                    # Optional exact7 material input. It remains unqualified;
                    # no all-concurrent/method assertion is manufactured here.
                    key=case['analysis_case_id']+'/'+r['cut_id']
                    fp=desc[floor]['footprint'];cross='y' if axis=='x' else 'x'
                    families=[dict(db_in=v['db_in'],global_transverse_coordinates_in=[v['s_in']])
                        for v in inventory['floors'][str(floor)]['bars'] if v['axis']==axis]
                    interval=bar_clear_interval(r['interval_in'],[fp[cross+'_min_in'],fp[cross+'_max_in']],families)
                    a=actual[case['source_id'],case['raw_case_id'],floor]
                    q={e['id']:-float(case['source_excitation']['dead'])*definition['gamma_kip_in3']*e['section']['b_in']*
                        (e['section']['h_in']-definition['slab_thickness_in']) for e in a['elements'] if e['kind']=='web'}
                    raw=common_strip_cut(a,axis=axis,station_in=station,interval_in=interval,trace=trace,
                        web_id=str(tag),uniform_web_loads=q,state_identity=state.identity(),combination_id=case['raw_case_id'],
                        location_id=key,recovery_id=r['recovery_id'])
                    for k in ('material_inventory_id','analysis_input_identity','source_response_sha256','definition_sha256'):raw[k]=r[k]
                    normalized,_=physical_cut_states([raw],outer_interval_in=interval)
                    alo,ahi=fp[axis+'_min_in'],fp[axis+'_max_in']
                    transverse=[dict(db_in=v['db_in'],global_transverse_coordinates_in=[v['s_in']])
                        for v in inventory['floors'][str(floor)]['bars'] if v['axis']!=axis and v['operator_owner']=='slab']
                    account=bar_clear_interval([max(alo,station-axial_accounting_length_in/2),min(ahi,station+axial_accounting_length_in/2)],
                        [alo,ahi],transverse)
                    specs.append(dict(cut_id=key,torsion_cut_id=r['cut_id'],floor=floor,axis=axis,station_in=station,
                        interval_in=interval,axial_accounting_interval_in=account,member_id=str(tag),trace=trace,source_id=case['source_id'],
                        raw_case_id=case['raw_case_id'],analysis_case_id=case['analysis_case_id'],
                        exact_torsion_section_interval_in=r['interval_in'],physical_region_kind='torsion_enclosing_bar_clear_common_section'))
                    actions[key]=normalized[0]
        cache.clear()
    result=dict(method=METHOD,state_identity=state.identity(),source_id=source_id,analysis_input_identity=common['input_signature'],
        source_identity=common['source_identity'],definition_sha256=definition['identity_sha256'],material_inventory_id=inventory['identity_sha256'],
        expected_case_ids=mapping['expected_case_ids'],analysis_case_ids=[c['analysis_case_id'] for c in cases],
        analysis_case_original_ids={c['analysis_case_id']:c['original_case_ids'] for c in cases},analysis_cases=cases,
        excluded_reference_cases=mapping['excluded_reference_cases'],expected_cuts_by_member=cuts,
        cut_inventory_complete=len(members)==len(full_member_tags),full_member_tags=full_member_tags,
        selected_member_tags=[m.member_tag for m in members],sections=sections,slab_fc_ksi=slab_fc,lambda_factor=1.,records=records,
        polynomial_audits=audits,guard_polynomial_audits=guard_audits,detail_station_inventory=details,
        primary_guard_threshold_stations_complete=True,maximum_records=maximum_records,required_record_count=len(records),
        mesh_qualification_inferred=False,continuous_nonlinear_reserve_bound_claimed=False,
        primary_strength_guard_reserve_complete=False,complete_design_qualified=False,
        strength_inputs=dict(cut_specs=specs,cuts=actions,required_cut_ids=[s['cut_id'] for s in specs],
            required_case_ids=sorted({c['identity']['combination_id'] for c in actions.values()}),
            above_threshold_only=bool(strength_inputs_above_threshold_only)) if include_strength_inputs else None)
    result['payload_sha256']=identity(result);return result
