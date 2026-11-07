"""Replayable atomic physical-floor material, development and seven-action checks.

The input explicitly lists every required case/cut. This module evaluates
that inventory; it does not invent a whole-floor mesh/trace qualification or
declare the pilot ready. Inverse failure and inadequate full-fy evidence stay
unknown; only a proved section upper-bound exceedance is a capacity failure.
"""
from __future__ import annotations
import copy
import math
import numpy as np
from Design.SMRF_Common_Floor_Actions import _hash
from Design.SMRF_Physical_Floor_Materials import spatial_section_from_inventory
from Design.SMRF_Shared_Floor_Strength import SpatialSharedFloorSection
from Design.SMRF_Axial_Action_Accounting import (verify_point_accounting,physical_window_evidence,
    select_axial_window,connected_web_domain)

METHOD='actual_physical_floor_atomic_strength_family_v4_explicit_local_event_planner'


class PreparedPhysicalResponses:
    """Invocation-local integrity checks for sealed raw response dictionaries.

    Repeated case chunks can reuse the initial full payload digest. Saved
    acceptance creates a fresh instance and recomputes all digests; this is
    never a persisted qualification flag or an inter-candidate cache.
    """
    def __init__(self,bundles):
        self.bundles=bundles;self.records={}
        for b in bundles:
            digest=_hash(b['response']);sid=b['source_id']
            if sid in self.records or digest!=b['response_payload_sha256']:raise ValueError('Duplicate or changed sealed physical response')
            self.records[sid]=(b['response'],digest)

    def digest(self,bundle):
        prior,digest=self.records[bundle['source_id']]
        if prior is not bundle['response'] or bundle['response_payload_sha256']!=digest:raise ValueError('Prepared response object/binding changed')
        return digest


def plan_physical_cut_family(definition,inventory,response_bundles,*,planner_identity,
        axial_accounting_length_in=24.,maximum_cuts=100000,include_scalar_extrema=True,
        selected_floor_case_pairs=None,prepared_responses=None,local_pitch_actions=False,local_material_events_by_floor=None):
    """Deterministic actual FE/member common/remote cut inventory, no inverse solves.

    bundles=[{source_id,response:<complete CoupledFloorSystem response>}].
    Both traces at every clear mesh/support face, every interval midpoint,
    and all seven scalar polynomial extrema are retained. Physical webs own
    whole bar-clear bands. Remote slab bands split at actual FE boundaries
    moved only to clear a longitudinal bar circle. Orthogonal beam bands are
    assigned to their own-axis checks; finite column-footprint bands are
    explicitly enumerated for the separate local/joint checks.

    This is complete sampling of these declared section polynomials, not a
    theorem that checking their separate scalar extrema bounds all nonlinear
    interaction between samples. Mesh/section-width comparison remains a
    finite method witness. Budget overflow raises; it never truncates checks.
    """
    from Design.SMRF_Physical_Floor_Cuts import common_strip_cut,bar_clear_interval,disjoint_transverse_partition
    from Design.SMRF_Primary_Moment_Guards import physical_cut_states
    if not planner_identity or not isinstance(maximum_cuts,int) or maximum_cuts<1:
        raise ValueError('Named deterministic planner and explicit positive cut budget required')
    if definition['state_identity']!=inventory['state_identity']:raise ValueError('Geometry/material states differ')
    length=float(axial_accounting_length_in)
    if not math.isfinite(length) or length<=0:raise ValueError('Positive actual axial accounting length required')
    descriptions={int(f['level']):f for f in definition['floors']};specs=[];cuts={};deferred=[];coverage=[];bundle_ids=[]
    universe=[(str(b['source_id']),int(f['level']),str(c['id'])) for b in response_bundles
              for f in definition['floors'] for c in b['response']['cases']]
    if len(set(universe))!=len(universe):raise ValueError('Unique full source/floor/case universe required')
    selected=universe if selected_floor_case_pairs is None else [tuple(p) for p in selected_floor_case_pairs]
    if not selected or len(set(selected))!=len(selected) or not set(selected)<=set(universe):
        raise ValueError('Selected floor/case pairs must be a unique nonempty subset of the full physical universe')
    chosen=set(selected)
    if not isinstance(local_pitch_actions,bool):raise ValueError('Explicit boolean local action selection required')
    local_events={}
    if local_pitch_actions:
        from Design.SMRF_Local_Material_Events import verify_local_material_events
        from Design.SMRF_Local_Pitch_Strength import bind_local_pitch_action
        needed={str(p[1]) for p in selected}
        if not isinstance(local_material_events_by_floor,dict) or set(local_material_events_by_floor)!=needed:
            raise ValueError('Complete actual material events for every selected floor required')
        for level in needed:
            row=verify_local_material_events(local_material_events_by_floor[level],inventory)
            if str(row['floor'])!=level:raise ValueError('Local material event floor differs')
            local_events[level]=row
    elif local_material_events_by_floor is not None:
        raise ValueError('Local material events require the explicit local action contract')
    def clear_bound(value,lo,hi,circles):
        options=[value]
        for a,b in circles:
            if a+1e-9<value<b-1e-9:options=[a-1e-8,b+1e-8];break
        feasible=[v for v in options if lo+1e-8<v<hi-1e-8 and not any(a+1e-9<v<b-1e-9 for a,b in circles)]
        return min(feasible,key=lambda v:abs(v-value)) if feasible else None
    for bundle in response_bundles:
        sid=str(bundle['source_id']);response=bundle['response'];bundle_ids.append(sid)
        if not sid or bundle_ids.count(sid)>1:raise ValueError('Unique coherent response source identities required')
        if response.get('definition_sha256')!=definition['identity_sha256']:raise ValueError('Response geometry identity differs')
        if prepared_responses is not None and not isinstance(prepared_responses,PreparedPhysicalResponses):raise ValueError('Actual invocation-local prepared response inventory required')
        sourcehash=_hash(response) if prepared_responses is None else prepared_responses.digest(bundle);binding=bundle
        if (binding.get('material_inventory_id')!=inventory['identity_sha256'] or
                binding.get('response_payload_sha256')!=sourcehash or not binding.get('analysis_input_identity')):
            raise ValueError('Response must bind exact actual material inventory, producer input identity and payload bytes')
        case_by={c['id']:c for c in response['cases']}
        if len(case_by)!=len(response['cases']):raise ValueError('Duplicate actual cases in physical response')
        if {int(f['level']) for f in response['floors']}!=set(descriptions):raise ValueError('Coherent response must cover every physical floor')
        for floorrow in response['floors']:
            level=int(floorrow['level']);desc=descriptions[level];fp=desc['footprint'];bars=inventory['floors'][str(level)]['bars']
            if {r['case_id'] for r in floorrow['responses']}!=set(case_by):raise ValueError('Every floor must retain every coherent case')
            for actual in floorrow['responses']:
                if (sid,level,str(actual['case_id'])) not in chosen:continue
                case=case_by[actual['case_id']];webs=[e for e in actual['elements'] if e['kind']=='web'];web_domains={}
                q={e['id']:-float(case['dead'])*definition['gamma_kip_in3']*e['section']['b_in']*(e['section']['h_in']-definition['slab_thickness_in']) for e in webs}
                for axis in ('x','y'):
                    j=0 if axis=='x' else 1;cross='y' if axis=='x' else 'x'
                    alo,ahi=fp[axis+'_min_in'],fp[axis+'_max_in'];tlo,thi=fp[cross+'_min_in'],fp[cross+'_max_in']
                    breaks=set(float(v) for v in floorrow['grid'][axis+'_in'])
                    if local_pitch_actions:breaks.update(local_events[str(level)]['events'][axis]['coordinates_in'])
                    for beam in desc['beams']:
                        if beam['axis']!=axis:breaks.update([max(alo,beam['line']-beam['b_in']/2),min(ahi,beam['line']+beam['b_in']/2)])
                    for support in desc['supports']:
                        c=support[axis];d=support['h_in' if axis=='x' else 'b_in']
                        breaks.update([max(alo,c-d/2),min(ahi,c+d/2)])
                    breaks=sorted(v for v in breaks if alo<=v<=ahi)
                    families=[dict(db_in=b['db_in'],global_transverse_coordinates_in=[b['s_in']]) for b in bars if b['axis']==axis]
                    transverse=[dict(db_in=b['db_in'],global_transverse_coordinates_in=[b['s_in']]) for b in bars if b['axis']!=axis and b['operator_owner']=='slab']
                    circles=[(b['s_in']-b['db_in']/2,b['s_in']+b['db_in']/2) for b in bars if b['axis']==axis]
                    for aa,bb in zip(breaks[:-1],breaks[1:]):
                        if local_pitch_actions and bb-aa<=1e-8:raise ValueError('Local material event interval is below current geometric resolution; cannot omit it')
                        if bb-aa<=1e-8:continue
                        mid=(aa+bb)/2;active={}
                        for e in webs:
                            p=np.asarray(e['positions_in'])
                            if e['axis']==axis and p[0,j]-1e-9<mid<p[1,j]+1e-9:
                                active[str(e['member_id'])]=e
                        # Every station in this clear interval belongs to the
                        # same actual slab element row. Passing only that row
                        # avoids rescanning the entire floor for every scalar
                        # witness; original field dictionaries and full source
                        # response hash remain unchanged.
                        slabrow=[e for e in actual['elements'] if e['kind']=='slab' and
                            min(p[j] for p in e['positions_in'])-1e-9<mid<max(p[j] for p in e['positions_in'])+1e-9]
                        strip_response=dict(elements=slabrow+list(active.values()),
                            force_partition_relative_error=actual['force_partition_relative_error'])
                        bands=[]
                        for member,e in active.items():
                            line=e['positions_in'][0][1-j];bw=e['section']['b_in']
                            a,b=bar_clear_interval([line-bw/2,line+bw/2],[tlo,thi],families);bands.append((member,a,b))
                        base=disjoint_transverse_partition([tlo,thi],bands)['pieces'];parts=[]
                        for part in base:
                            if part['kind']=='common_beam_strip':parts.append(part);continue
                            lo,hi=part['interval_in'];sub={lo,hi};regions=[]
                            for support in desc['supports']:
                                d=support['h_in' if axis=='x' else 'b_in'];w=support['b_in' if axis=='x' else 'h_in']
                                if support[axis]-d/2-1e-9<mid<support[axis]+d/2+1e-9:
                                    a,b=max(lo,support[cross]-w/2),min(hi,support[cross]+w/2)
                                    if b>a:sub.update([a,b]);regions.append((a,b,'local_joint',str(support['id'])))
                            for e in webs:
                                if e['axis']==axis:continue
                                line=e['positions_in'][0][j];bw=e['section']['b_in']
                                if line-bw/2-1e-9<mid<line+bw/2+1e-9:
                                    a,b=max(lo,e['positions_in'][0][1-j]),min(hi,e['positions_in'][1][1-j])
                                    if b>a:sub.update([a,b]);regions.append((a,b,'orthogonal_common_beam',str(e['member_id'])))
                            for v in floorrow['grid'][cross+'_in']:
                                if lo+1e-8<v<hi-1e-8:
                                    adjusted=clear_bound(v,lo,hi,circles)
                                    if adjusted is not None:sub.add(adjusted)
                            sub=sorted(sub)
                            for a,b in zip(sub[:-1],sub[1:]):
                                if b-a<=1e-8:continue
                                owners=[r for r in regions if r[0]-1e-9<(a+b)/2<r[1]+1e-9]
                                kind=owners[0][2] if owners else 'remote_slab'
                                parts.append(dict(kind=kind,interval_in=[a,b],assigned_ids=sorted({r[3] for r in owners})))
                        if abs(sum(p['interval_in'][1]-p['interval_in'][0] for p in parts)-(thi-tlo))>1e-7:
                            raise ValueError('Actual full floor trace partition does not cover its physical width')
                        coverage.append(dict(source_id=sid,case_id=case['id'],floor=level,axis=axis,axial_interval_in=[aa,bb],parts=parts))
                        for pi,part in enumerate(parts):
                            if part['kind'] in ('local_joint','orthogonal_common_beam'):
                                deferred.append(dict(source_id=sid,case_id=case['id'],floor=level,axis=axis,axial_interval_in=[aa,bb],**part));continue
                            domain=[alo,ahi];member=part.get('member_id')
                            if member is not None:
                                domain=web_domains.get((axis,member))
                                if domain is None or not domain[0]<mid<domain[1]:
                                    domain=connected_web_domain({'elements':webs},axis,member,mid,'plus')
                                    web_domains[(axis,member)]=domain
                            def recover(station,trace,location):
                                cut=common_strip_cut(strip_response,axis=axis,station_in=station,interval_in=part['interval_in'],trace=trace,
                                    web_id=part.get('member_id'),uniform_web_loads=q,state_identity=inventory['state_identity'],
                                    combination_id=case['id'],location_id=location,recovery_id=sid+'/'+sourcehash)
                                cut.update(material_inventory_id=inventory['identity_sha256'],analysis_input_identity=binding['analysis_input_identity'],
                                           source_response_sha256=sourcehash,definition_sha256=definition['identity_sha256'])
                                return cut
                            sample=[recover(aa,'plus','polynomial_start'),recover(mid,'minus','polynomial_mid'),recover(bb,'minus','polynomial_end')]
                            locations=[(aa,'plus','face_or_mesh_trace'),(mid,'minus','mesh_interval_midpoint'),(bb,'minus','face_or_mesh_trace')]
                            polynomial_audit=None
                            if include_scalar_extrema:
                                vals=np.array([c['resultants']+[c['Ms_transverse_first_moment_kipin']] for c in sample])
                                coefficients=np.linalg.solve(np.array([[1,0,0],[1,.5,.25],[1,1,1]]),vals)
                                witness=recover(aa+.25*(bb-aa),'minus','polynomial_independent_witness')
                                target=np.array(witness['resultants']+[witness['Ms_transverse_first_moment_kipin']]);pred=np.array([1,.25,.25**2])@coefficients
                                residual=pred-target;passed=bool(np.all(abs(residual)<=1e-7+1e-7*np.maximum(abs(pred),abs(target))))
                                if not passed:raise ValueError('Actual clear-segment section actions are not the verified degree-two field')
                                for b,c in zip(coefficients[1],coefficients[2]):
                                    if abs(c)>1e-12:
                                        root=-b/(2*c)
                                        if 1e-8<root<1-1e-8:locations.append((aa+root*(bb-aa),'minus','scalar_resultant_extremum'))
                                polynomial_audit=dict(normalized_coefficients=coefficients.tolist(),independent_quarter_residual=residual.tolist())
                            seen=[]
                            for station,trace,kind in locations:
                                if any(abs(station-s)<1e-7 and trace==t for s,t in seen):continue
                                seen.append((station,trace));key=f'{sid}/{case["id"]}/f{level}/{axis}/{aa:.9g}:{bb:.9g}/p{pi}/{station:.12g}/{trace}'
                                if local_pitch_actions:
                                    key=f'{sid}/{case["id"]}/f{level}/{axis}/{float(aa).hex()}:{float(bb).hex()}/p{pi}/{float(station).hex()}/{trace}'
                                if key in cuts:raise ValueError('Duplicate actual physical cut identity cannot replace a required point')
                                raw=recover(station,trace,key);normalized,_=physical_cut_states([raw],outer_interval_in=part['interval_in'])
                                selection=select_axial_window(station,length,domain,transverse)
                                accounting=(selection['interval_in'] if selection['status']=='computed' else
                                    bar_clear_interval([max(alo,station-length/2),min(ahi,station+length/2)],[alo,ahi],transverse))
                                specs.append(dict(cut_id=key,floor=level,axis=axis,station_in=station,interval_in=part['interval_in'],
                                    axial_accounting_interval_in=accounting,axial_window_selection=selection,member_id=part.get('member_id'),trace=trace,
                                    source_id=sid,sampling_kind=kind,physical_region_kind=part['kind'],polynomial=polynomial_audit))
                                raw['axial_action_window']=physical_window_evidence(actual,specs[-1],normalized[0],uniform_web_loads=q)
                                if local_pitch_actions:
                                    specs[-1]['material_scope']='common_slab_web' if member is not None else 'slab_only'
                                    cuts[key]=bind_local_pitch_action(raw,floor=level,inventory=inventory,
                                        analysis_input_identity=binding['analysis_input_identity'],source_response_sha256=sourcehash,
                                        definition_sha256=definition['identity_sha256'])
                                else:
                                    normalized,_=physical_cut_states([raw],outer_interval_in=part['interval_in'])
                                    cuts[key]=normalized[0]
                                if len(specs)>maximum_cuts:raise ValueError('Complete actual required-cut inventory exceeds declared budget; no truncation allowed')
    cases=sorted({c['identity']['combination_id'] for c in cuts.values()})
    out=dict(method=METHOD+'/planner',planner_identity=planner_identity,state_identity=inventory['state_identity'],
        material_inventory_id=inventory['identity_sha256'],definition_sha256=definition['identity_sha256'],
        required_cut_ids=[s['cut_id'] for s in specs],required_case_ids=cases,cut_specs=specs,cuts=cuts,
        required_cut_count=len(specs),coverage=coverage,assigned_joint_or_other_axis_regions=deferred,
        all_floor_trace_partitions_verified=True,scalar_polynomial_extrema_included=bool(include_scalar_extrema),
        nonlinear_interaction_between_samples_bound_claimed=False,mesh_qualification_inferred=False,
        complete_design_qualified=False)
    if local_pitch_actions:
        from Design.SMRF_Common_Guard_Plan import LOCAL_PLAN_CONTRACT
        out.update(action_contract=LOCAL_PLAN_CONTRACT,local_material_events_by_floor=local_events,
            continuous_capacity_verified=False,material_event_boundaries_included=True)
    out['scope']=dict(full_floor_case_universe=[list(p) for p in universe],selected_floor_case_pairs=[list(p) for p in selected],
        full_universe_sha256=_hash(universe),complete_required_universe=set(selected)==set(universe),
        response_payloads={b['source_id']:b['response_payload_sha256'] for b in response_bundles})
    out['payload_sha256']=_hash(out);return out


def assess_physical_cut_family(state, inventory, cut_specs, cuts, *, slab_fc_ksi,
        ec_ksi, phi_requirements, required_cut_ids, required_case_ids,
        assessment_identity, max_evaluations=250, development_result=None,
        apply_development_domains=True,section_initialization_cache=None):
    """Evaluate exact named normalized seven-action cuts, with replay payload.

    cut_specs: cut_id,floor,axis,station_in,interval_in,
    axial_accounting_interval_in,optional member_id. cuts maps cut_id to the
    normalized action shape accepted by SpatialSharedFloorSection. A guard
    is an additional named state, not a replacement physical branch.
    development_result may be saved output from the authoritative provider;
    every individual evidence record is re-scoped by the inventory adapter.
    """
    if not assessment_identity:raise ValueError('Unique assessment identity required')
    specs=list(cut_specs);ids=[r['cut_id'] for r in specs];required=list(required_cut_ids);cases=list(required_case_ids)
    if len(set(ids))!=len(ids) or len(set(required))!=len(required) or not required or not cases or len(set(cases))!=len(cases):
        raise ValueError('Explicit nonempty unique required case and physical cut inventories required')
    if set(ids)!=set(required) or set(cuts)!=set(required):raise ValueError('Every required cut must appear exactly once in specs/actions')
    observed={r['identity']['combination_id'] for r in cuts.values()}
    if observed!=set(cases):raise ValueError('Action combinations differ from explicit required case inventory')
    if state is not None and state.identity()!=inventory['state_identity']:raise ValueError('Installed state differs from physical inventory')
    if development_result is None:
        if state is None:raise ValueError('Live state or complete saved development evidence required')
        from Design.SMRF_Actual_Bar_Development import actual_bar_development
        development_result=actual_bar_development(state,inventory,specs,slab_fc_ksi=slab_fc_ksi)
    development=development_result.get('cuts',[])
    if len(development)!=len(specs):raise ValueError('Authoritative development evidence must cover each requested cut')
    material=dict(slab_fc_ksi=float(slab_fc_ksi),ec_ksi=float(ec_ksi),phi_requirements=dict(phi_requirements),
                  apply_development_domains=bool(apply_development_domains))
    payload=dict(method=METHOD,assessment_identity=str(assessment_identity),state_identity=inventory['state_identity'],
        material_inventory_id=inventory['identity_sha256'],inventory=inventory,cut_specs=specs,cuts=cuts,
        development_result=development_result,material=material,required_cut_ids=required,required_case_ids=cases,
        max_evaluations=int(max_evaluations))
    records=[];cache={} if section_initialization_cache is None else section_initialization_cache
    for spec,dev in zip(specs,development):
        name=spec['cut_id'];action=cuts[name];pending=[]
        if action['identity']['state_identity']!=inventory['state_identity']:raise ValueError('Physical action state differs from actual steel')
        if action.get('material_inventory_id')!=inventory['identity_sha256'] or not action.get('analysis_input_identity'):
            raise ValueError('Physical actions must bind exact actual mat/cage inventory and producing analysis input')
        if action['axis']!=spec['axis'] or action['cut_width_in']!=spec['interval_in'][1]-spec['interval_in'][0]:
            raise ValueError('Actual cut axis/width differs from requested material inventory')
        args={k:spec[k] for k in ('floor','axis','station_in','interval_in','axial_accounting_interval_in')}
        built=spatial_section_from_inventory(inventory,**args,member_id=spec.get('member_id'),
            material_scope=spec.get('material_scope','common_slab_web'),
            transverse_reference=action['reference_s_in'],fc_ksi=slab_fc_ksi,ec_ksi=ec_ksi,
            phi_requirements=phi_requirements,development_evidence=dev.get('evidence',[]),
            apply_development_domains=apply_development_domains)
        if built['section'] is None:pending.append('physical_section_inventory_not_constructed')
        if built['development']['status']!='pass':pending.append('full_fy_development_credit_not_established')
        if built['physical_detail_status']!='pass':pending.append('physical_detail_inventory_not_passed')
        if built['unqualified_geometry_diagnostic']:pending.append('explicit_diagnostic_state_not_accepted_candidate')
        axial_accounting=verify_point_accounting(action,spec);pending.extend(axial_accounting['pending'])
        if built['axial_material_uniformity']['status']!='pass':pending.append('physical_material_varies_inside_axial_accounting_window')
        section=None
        if built['section'] is not None:
            geometry=copy.deepcopy(built['constructor'])
            for bar in geometry['longitudinal_bars']:bar.pop('id',None)
            for row in geometry['transverse_rows']:
                row['physical_bar_count']=len(row.pop('physical_bar_ids'));row.pop('id',None)
            if geometry['web'] is not None:geometry['web'].pop('id',None)
            reference=geometry['reference_s_in'];geometry['reference_s_in']=0.
            geometry['slab_interval_in']=[v-reference for v in geometry['slab_interval_in']]
            for bar in geometry['longitudinal_bars']:bar['s_in']-=reference
            if geometry['web'] is not None:geometry['web']['interval_in']=[v-reference for v in geometry['web']['interval_in']]
            for row in geometry['transverse_rows']:
                row['run_interval_in']=[v-reference for v in row['run_interval_in']]
                if row.get('strength_run_interval_in') is not None:row['strength_run_interval_in']=[v-reference for v in row['strength_run_interval_in']]
            key=_hash(geometry);target=np.asarray(action['resultants'],float)/action['cut_width_in']
            nearby=cache.get(key,[]);initial=None;initial_source=None
            if nearby:
                nearest=min(nearby,key=lambda r:float(np.linalg.norm((target-np.asarray(r['target']))/(1+abs(target)))))
                initial=nearest['strain'];initial_source=nearest['action_sha256']
            section=SpatialSharedFloorSection(built['section'],state_identity=inventory['state_identity'],
                material_inventory_id=inventory['identity_sha256']).evaluate(action,max_evaluations=max_evaluations,initial_strain=initial)
            section['numerical_initialization']=dict(geometry_sha256=key,prior_action_sha256=initial_source,
                acceptance_reused=False,all_current_equilibrium_domain_quadrature_checks_repeated=True)
            numerical=section.get('numerical_state')
            if numerical and numerical['restricted_model_state_verified']:
                cache.setdefault(key,[]).append(dict(target=target.tolist(),strain=numerical['generalized_strain'],action_sha256=_hash(action)))
                cache[key]=cache[key][-16:]
            pending.extend(section.get('pending',[]))
        if pending or section is None or section['status']=='unknown':status='unknown'
        elif section['status']=='capacity_exceeded':status='fail'
        else:status='pass'
        records.append(dict(cut_id=name,case_id=action['identity']['combination_id'],status=status,
            pending=sorted(set(pending)),constructor=built['constructor'],material_cut_identity=built['identity'],
            material_cut_sha256=built['identity_sha256'],physical_bar_ids=built['physical_bar_ids'],
            inventory_issues=built['issues'],development=built['development'],section_assessment=section,
            axial_action_accounting=axial_accounting,
            axial_material_uniformity=built['axial_material_uniformity'],
            action_payload=action,action_sha256=_hash(action),
            proved_capacity_exceeded=bool(section and section['section_status']=='capacity_exceeded'),
            actual_development_demand_failure_established=False))
    status='fail' if any(r['status']=='fail' for r in records) else 'unknown' if any(r['status']=='unknown' for r in records) else 'pass'
    return dict(method=METHOD,assessment_identity=str(assessment_identity),status=status,
        state_identity=inventory['state_identity'],material_inventory_id=inventory['identity_sha256'],
        required_cut_ids=required,actual_cut_ids=ids,required_case_ids=cases,actual_case_ids=sorted(observed),
        payload_sha256=_hash(payload),replay_payload=payload,records=records,
        complete_design_qualified=False,assessment_scope='only explicit required case/cut inventory',
        upstream_mesh_trace_and_method_qualification_inferred=False,
        nominal_probable_capacity_inventory_modified=False)


def replay_physical_cut_assessment(record):
    payload=copy.deepcopy(record['replay_payload'])
    if record.get('payload_sha256')!=_hash(payload):raise ValueError('Saved atomic assessment payload changed')
    material=payload['material']
    return assess_physical_cut_family(None,payload['inventory'],payload['cut_specs'],payload['cuts'],
        **material,required_cut_ids=payload['required_cut_ids'],required_case_ids=payload['required_case_ids'],
        assessment_identity=payload['assessment_identity'],max_evaluations=payload['max_evaluations'],
        development_result=payload['development_result'])


def verify_physical_cut_assessment(record):
    """Verify saved feasible strain or infeasible dual witnesses, no optimizer.

    Reconstruct actual material/development from the bound raw payload; check
    every current constitutive point, final quadrature pair and seven force
    residuals. A failure certificate is independently recomputed from the
    exact continuous support function. Unknown numerical outcomes stay unknown.
    """
    payload=record['replay_payload']
    if record.get('payload_sha256')!=_hash(payload):raise ValueError('Saved actual assessment raw payload changed')
    inventory=payload['inventory'];specs=payload['cut_specs'];actions=payload['cuts'];material=payload['material']
    ids=[s['cut_id'] for s in specs]
    if len(ids)!=len(set(ids)) or set(ids)!=set(payload['required_cut_ids']) or set(actions)!=set(ids):
        raise ValueError('Saved complete required cut inventory differs')
    if {c['identity']['combination_id'] for c in actions.values()}!=set(payload['required_case_ids']):
        raise ValueError('Saved complete required case inventory differs')
    saved={r['cut_id']:r for r in record['records']}
    if set(saved)!=set(ids) or len(saved)!=len(record['records']):raise ValueError('Saved assessment results omit/duplicate required cuts')
    development=payload['development_result']['cuts']
    if len(development)!=len(specs):raise ValueError('Saved development evidence inventory differs')
    output=[]
    for spec,dev in zip(specs,development):
        ident=spec['cut_id'];action=actions[ident];prior=saved[ident];pending=[]
        if action.get('material_inventory_id')!=inventory['identity_sha256'] or not action.get('analysis_input_identity'):
            raise ValueError('Saved action does not bind actual mat/cage/analysis identity')
        if action['identity']['state_identity']!=inventory['state_identity'] or prior['action_sha256']!=_hash(action):
            raise ValueError('Saved action identity changed')
        args={k:spec[k] for k in ('floor','axis','station_in','interval_in','axial_accounting_interval_in')}
        built=spatial_section_from_inventory(inventory,**args,member_id=spec.get('member_id'),transverse_reference=action['reference_s_in'],
            material_scope=spec.get('material_scope','common_slab_web'),
            fc_ksi=material['slab_fc_ksi'],ec_ksi=material['ec_ksi'],phi_requirements=material['phi_requirements'],
            development_evidence=dev.get('evidence',[]),apply_development_domains=material['apply_development_domains'])
        if built['constructor']!=prior['constructor'] or built['identity_sha256']!=prior['material_cut_sha256']:
            raise ValueError('Saved section constructor differs from current exact physical inventory/development')
        for source in action['source_identities']:
            if any(source.get(k)!=action['identity'][k] for k in ('state_identity','combination_id','location_id')):
                raise ValueError('Saved nonconcurrent action source')
        a=action['accounting'];er=np.asarray(a['equilibrium_residual']);tol=np.asarray(a['equilibrium_tolerance'])
        if er.shape!=tol.shape or not er.size or not np.isfinite(er).all() or not np.isfinite(tol).all() or np.any(tol<0) or np.any(abs(er)>tol):
            pending.append('upstream_force_moment_equilibrium_not_verified')
        if a.get('method')!='disjoint_actual_fields' or not a.get('evidence_id'):pending.append('actual_disjoint_spatial_action_evidence_not_established')
        if a.get('spatial_seven_field_projection_verified') is not True:pending.append('seven_resultant_physical_field_transfer_not_verified')
        if a.get('core_remote_coverage_verified') is not True:pending.append('physical_core_remote_coverage_not_verified')
        if a.get('physical_diaphragm_embedding_verified') is False:pending.append('planar_diaphragm_force_datum_does_not_establish_physical_slab_web_bending_response')
        if a.get('primary_force_composition_authorized') is False:pending.append('independent_physical_additional_load_branch_is_not_a_primary_response_correction')
        weak=float(a.get('weak_first_moment_residual_kip_in',math.nan));wtol=float(a.get('weak_first_moment_tolerance_kip_in',math.nan))
        if not math.isfinite(weak) or not math.isfinite(wtol) or wtol<0 or abs(weak)>wtol:pending.append('weak_first_moment_action_reconstruction_not_verified')
        if built['development']['status']!='pass':pending.append('full_fy_development_credit_not_established')
        if built['physical_detail_status']!='pass':pending.append('physical_detail_inventory_not_passed')
        if built['unqualified_geometry_diagnostic']:pending.append('explicit_diagnostic_state_not_accepted_candidate')
        axial_accounting=verify_point_accounting(action,spec);pending.extend(axial_accounting['pending'])
        if built['axial_material_uniformity']['status']!='pass':pending.append('physical_material_varies_inside_axial_accounting_window')
        if prior.get('axial_material_uniformity')!=built['axial_material_uniformity']:
            raise ValueError('Saved axial material/development window differs from its actual inventory')
        if prior.get('axial_action_accounting')!=axial_accounting:
            raise ValueError('Saved axial action/material accounting differs from its raw witness')
        section=built['section'];s=prior.get('section_assessment');witness=None;section_status='unknown'
        if section is None:pending.append('physical_section_inventory_not_constructed')
        elif s is not None:
            if action['axis']!=section.axis or float(action['reference_z_in'])!=0 or abs(action['reference_s_in']-section.sref)>1e-10 or abs(action['cut_width_in']-section.width)>1e-9:
                raise ValueError('Saved action/material physical cut reference differs')
            target=np.asarray(action['resultants'],float)
            if action['units']=='integrated_strip':target=target/section.width
            elif action['units']!='per_unit_width':raise ValueError('Saved explicit units missing')
            if target.shape!=(7,) or not np.isfinite(target).all() or not np.allclose(target,s['factored_resultants_per_in'],rtol=0,atol=1e-12):
                raise ValueError('Saved section target differs from raw concurrent action')
            certificate=s.get('combined_capacity_upper_bound_certificate') or {}
            if certificate.get('capacity_exceeded'):
                direction=np.asarray(certificate['direction']);upper=section.support_upper_bound(direction);demand=float(direction@target)
                allowance=1e-8+1e-9*max(abs(upper),abs(demand));passed=demand>upper+allowance
                witness=dict(kind='exact_dual_support',verified=bool(passed),demand=demand,upper_bound=upper,allowance=allowance)
                if passed:section_status='capacity_exceeded'
            elif s.get('capacity_bound_violations'):
                b=section.component_capacity_upper_bounds();checks=[]
                for v in s['capacity_bound_violations']:
                    j=v['component'];value=target[j];bound=b['positive'][j] if value>=0 else b['negative_magnitude'][j]
                    checks.append(abs(value)>bound+1e-8+1e-9*max(abs(value),bound))
                passed=bool(checks and any(checks));witness=dict(kind='component_support',verified=passed)
                if passed:section_status='capacity_exceeded'
            elif s.get('numerical_state'):
                old=s['numerical_state'];q=np.asarray(old['generalized_strain']);orders=old['quadrature']['orders']
                coarse=section.state(q,quadrature_order=int(orders[0]));fine=section.state(q,quadrature_order=int(orders[1]))
                cr=np.asarray(coarse['nominal_resultants']);fr=np.asarray(fine['nominal_resultants']);qe=abs(cr-fr)
                quadrature=bool(np.all(qe<=1e-7+1e-5*np.maximum(abs(cr),abs(fr))))
                error=np.asarray(fine['design_resultants'])-target;balanced=bool(np.all(abs(error)<=1e-6+1e-6*abs(target)))
                passed=balanced and quadrature and fine['material_domain_valid']
                witness=dict(kind='compatible_strain',verified=bool(passed),generalized_strain=q.tolist(),
                    equilibrium_residual=error.tolist(),equilibrium_verified=balanced,quadrature_verified=quadrature,
                    quadrature_absolute_errors=qe.tolist(),orders=orders,material_domain_errors=fine['material_domain_errors'])
                if passed:section_status='pass'
        if section_status=='unknown':pending.append('saved_capacity_or_feasibility_witness_not_verified')
        status='unknown' if pending else 'fail' if section_status=='capacity_exceeded' else 'pass'
        claim_verified=(prior['status']=='unknown' or status==prior['status'])
        output.append(dict(cut_id=ident,status=status,section_status=section_status,pending=sorted(set(pending)),
            saved_claim_verified=claim_verified,witness=witness,material_cut_sha256=built['identity_sha256']))
    status='fail' if any(r['status']=='fail' for r in output) else 'unknown' if any(r['status']=='unknown' for r in output) else 'pass'
    return dict(method=METHOD+'/saved_witness_verification',status=status,records=output,
        saved_claims_verified=all(r['saved_claim_verified'] for r in output),payload_sha256=record['payload_sha256'],
        required_cut_ids=payload['required_cut_ids'],required_case_ids=payload['required_case_ids'],
        inverse_solves=0,dual_searches=0,complete_design_qualified=False)
