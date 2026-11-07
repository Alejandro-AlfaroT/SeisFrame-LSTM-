"""Explicit local-pitch section assessment and optimizer-free saved verification.

This is a named alternate material/action contract. Its local transverse
layers come from the installed pitches and require no axial-window equality.
It does not qualify that idealization, complete spatial interaction coverage,
support behavior or a pilot. Those remain separate required method evidence.
"""
from __future__ import annotations
import copy
import math
import numpy as np
from Design.SMRF_Common_Floor_Actions import _hash
from Design.SMRF_Pitch_Spatial_Section import pitch_spatial_section,developed_pitch_spatial_section,METHOD as SECTION_METHOD
from Design.SMRF_Shared_Floor_Strength import SpatialSharedFloorSection

METHOD='actual_local_pitch_atomic_strength_and_witness_family_v2_primary_guard_replay'


def bind_local_pitch_action(raw_cut,*,floor,inventory,analysis_input_identity,source_response_sha256,definition_sha256):
    """Normalize an actual physical cut and retain its full geometric source.

    Source fields must be reconstructed from the sealed physical response by
    the caller's physics replay. A supplied digest alone never qualifies them.
    This binder cannot create a primary guard from an unrelated force vector.
    """
    from Design.SMRF_Primary_Moment_Guards import physical_cut_states
    if raw_cut['identity']['state_identity']!=inventory['state_identity'] or str(floor) not in inventory['floors']:
        raise ValueError('Actual floor and state must belong to the installed inventory')
    if not all(isinstance(v,str) and v for v in (analysis_input_identity,source_response_sha256,definition_sha256)):
        raise ValueError('Explicit analysis, source response and definition identities required')
    result=physical_cut_states([raw_cut],outer_interval_in=raw_cut['slab']['interval_in'])[0][0]
    point=dict(floor=int(floor),axis=raw_cut['slab']['axis'],station_in=raw_cut['slab']['station_in'],
        interval_in=raw_cut['slab']['interval_in'],trace=raw_cut['slab']['trace'],
        member_id=None if raw_cut['web'] is None else str(raw_cut['web']['member_id']))
    result.update(material_inventory_id=inventory['identity_sha256'],analysis_input_identity=analysis_input_identity,
        source_response_sha256=source_response_sha256,definition_sha256=definition_sha256,
        local_point_binding=dict(method=METHOD,point=point,raw_cut=copy.deepcopy(raw_cut),raw_cut_sha256=_hash(raw_cut),
            normalized_base_sha256=_hash(result)))
    return result


def _check_point(action,spec,inventory):
    binding=action.get('local_point_binding')
    if isinstance(binding,dict) and binding.get('kind')=='local_transfer':
        from Design.SMRF_Local_Transfer_Binding import verify_local_transfer_binding
        verify_local_transfer_binding(action,spec,inventory)
        return
    if isinstance(binding,dict) and binding.get('kind') in ('primary_guard','primary_partition'):
        from Design.SMRF_Local_Primary_Guards import verify_local_primary_guard_binding
        verify_local_primary_guard_binding(action,spec,inventory)
        return
    if not isinstance(binding,dict) or binding.get('method')!=METHOD:raise ValueError('Bound actual local point action required')
    raw=binding['raw_cut']
    if binding['raw_cut_sha256']!=_hash(raw):raise ValueError('Saved local raw cut changed')
    # Re-execute physical normalization and all polynomial transfer checks.
    rebuilt=bind_local_pitch_action(raw,floor=spec['floor'],inventory=inventory,analysis_input_identity=action['analysis_input_identity'],
        source_response_sha256=action['source_response_sha256'],definition_sha256=action['definition_sha256'])
    if rebuilt!=action:raise ValueError('Local action differs from its actual raw cut reconstruction')
    expected={k:spec[k] for k in ('floor','axis','station_in','interval_in','trace')};expected['member_id']=None if spec.get('member_id') is None else str(spec['member_id'])
    if binding['point']!=expected:raise ValueError('Local material and action station/trace/width/member differ')
    if action['material_inventory_id']!=inventory['identity_sha256']:raise ValueError('Local actual material inventory differs')


def _prepare(inventory,spec,action,material,evidence):
    _check_point(action,spec,inventory)
    args={k:spec[k] for k in ('floor','axis','station_in','interval_in','trace')}
    built=pitch_spatial_section(inventory,**args,member_id=spec.get('member_id'),material_scope=spec.get('material_scope','common_slab_web'),
        transverse_reference=action['reference_s_in'],fc_ksi=material['slab_fc_ksi'],ec_ksi=material['ec_ksi'],
        phi_requirements=material['phi_requirements'],development_evidence=evidence)
    pending=[]
    if built['section'] is None:pending.append('actual_local_section_inventory_not_constructed')
    if built['development_status']!='pass':pending.append('actual_local_full_fy_domains_not_established')
    if inventory.get('physical_detail_status')!='pass':pending.append('physical_detail_inventory_not_passed')
    if inventory.get('unqualified_geometry_diagnostic',False):pending.append('explicit_diagnostic_state_not_accepted_candidate')
    a=action['accounting'];er=np.asarray(a.get('equilibrium_residual',[]),float);tol=np.asarray(a.get('equilibrium_tolerance',[]),float)
    if not (er.size and er.shape==tol.shape and np.isfinite(er).all() and np.isfinite(tol).all() and np.all(tol>=0) and np.all(abs(er)<=tol)):
        pending.append('upstream_force_moment_equilibrium_not_verified')
    for key in ('spatial_seven_field_projection_verified','core_remote_coverage_verified','all_concurrent_demands_included'):
        if a.get(key) is not True:pending.append(key+'_not_established')
    weak=float(a.get('weak_first_moment_residual_kip_in',math.nan));wtol=float(a.get('weak_first_moment_tolerance_kip_in',math.nan))
    if not math.isfinite(weak) or not math.isfinite(wtol) or wtol<0 or abs(weak)>wtol:pending.append('weak_first_moment_action_reconstruction_not_verified')
    if a.get('physical_diaphragm_embedding_verified') is False:pending.append('physical_diaphragm_embedding_not_established')
    if a.get('primary_force_composition_authorized') is False:pending.append('primary_force_composition_not_authorized')
    return built,pending


def actual_local_cut_development(state,inventory,spec,action,material):
    """Recompute actual full-fy evidence for the same point material owner IDs."""
    if state is None or state.identity()!=inventory['state_identity']:raise ValueError('Actual local development state required')
    _check_point(action,spec,inventory)
    first=developed_pitch_spatial_section(state,inventory,**{k:spec[k] for k in ('floor','axis','station_in','interval_in','trace')},
        fc_ksi=material['slab_fc_ksi'],ec_ksi=material['ec_ksi'],phi_requirements=material['phi_requirements'],member_id=spec.get('member_id'),
        material_scope=spec.get('material_scope','common_slab_web'),transverse_reference=action['reference_s_in'])
    return first.get('actual_development_evidence',{}).get('cuts',[{}])[0].get('evidence',[])


def assess_local_pitch_cut_family(state,inventory,cut_specs,cuts,*,slab_fc_ksi,ec_ksi,phi_requirements,
        required_cut_ids,required_case_ids,assessment_identity,max_evaluations=250,development_by_cut=None,initial_strains=None):
    """Evaluate every explicitly required actual cut, never a sampled-domain proof."""
    specs=copy.deepcopy(list(cut_specs));ids=[s['cut_id'] for s in specs];required=list(required_cut_ids);cases=list(required_case_ids)
    if not assessment_identity or not ids or len(ids)!=len(set(ids)) or len(required)!=len(set(required)) or set(ids)!=set(required) or set(cuts)!=set(ids):
        raise ValueError('Complete unique actual required cut inventory required')
    if not cases or len(cases)!=len(set(cases)) or {c['identity']['combination_id'] for c in cuts.values()}!=set(cases):
        raise ValueError('Complete unique actual required combination inventory required')
    if state is not None and state.identity()!=inventory['state_identity']:raise ValueError('Actual state differs from installed local inventory')
    material=dict(slab_fc_ksi=float(slab_fc_ksi),ec_ksi=float(ec_ksi),phi_requirements=dict(phi_requirements),section_method=SECTION_METHOD)
    development=copy.deepcopy(development_by_cut) if development_by_cut is not None else {}
    if development_by_cut is not None and set(development)!=set(ids):raise ValueError('Development records must cover each exact required cut')
    records=[]
    for spec in specs:
        ident=spec['cut_id'];action=cuts[ident];_check_point(action,spec,inventory)
        if development_by_cut is None:
            development[ident]=actual_local_cut_development(state,inventory,spec,action,material)
        built,pending=_prepare(inventory,spec,action,material,development[ident]);assessment=None
        if built['section'] is not None:
            assessment=SpatialSharedFloorSection(built['section'],state_identity=inventory['state_identity'],material_inventory_id=inventory['identity_sha256']).evaluate(
                action,max_evaluations=max_evaluations,initial_strain=None if initial_strains is None else initial_strains.get(ident))
            pending.extend(assessment['pending'])
        status='unknown' if pending or assessment is None or assessment['status']=='unknown' else 'fail' if assessment['section_status']=='capacity_exceeded' else 'pass'
        records.append(dict(cut_id=ident,status=status,pending=sorted(set(pending)),constructor=built['constructor'],material_cut_sha256=built['identity_sha256'],
            physical_bar_ids=built['physical_bar_ids'],excluded_beam_strength_ids=built['excluded_beam_strength_ids'],development=built['development'],
            development_status=built['development_status'],inventory_issues=built['issues'],section_assessment=assessment,action_sha256=_hash(action)))
    payload=dict(method=METHOD,assessment_identity=assessment_identity,inventory=copy.deepcopy(inventory),material=material,cut_specs=specs,
        cuts=copy.deepcopy(cuts),development_by_cut=development,required_cut_ids=required,required_case_ids=cases,max_evaluations=max_evaluations)
    return dict(method=METHOD,status='fail' if any(r['status']=='fail' for r in records) else 'unknown' if any(r['status']=='unknown' for r in records) else 'pass',
        records=records,payload_sha256=_hash(payload),replay_payload=payload,continuous_coverage_verified=False,engineering_qualified=False,
        assessment_scope='exact required point/case inventory only',axial_window_equivalence_claimed=False,nominal_probable_capacity_inventory_modified=False)


def verify_local_pitch_cut_assessment(record):
    """Reconstruct actual local material and numeric witnesses, without optimizer."""
    payload=record['replay_payload']
    if record.get('method')!=METHOD or payload.get('method')!=METHOD or record['payload_sha256']!=_hash(payload):raise ValueError('Saved local assessment method or payload differs')
    if payload['material']['section_method']!=SECTION_METHOD:raise ValueError('Saved local section method differs')
    specs=payload['cut_specs'];ids=[r['cut_id'] for r in specs];saved={r['cut_id']:r for r in record['records']}
    if not ids or len(ids)!=len(set(ids)) or len(saved)!=len(record['records']) or len(payload['required_cut_ids'])!=len(set(payload['required_cut_ids'])) or set(ids)!=set(payload['required_cut_ids']) or set(saved)!=set(ids) or set(payload['cuts'])!=set(ids) or set(payload['development_by_cut'])!=set(ids):
        raise ValueError('Saved exact required point/development inventory differs')
    if not payload['required_case_ids'] or len(payload['required_case_ids'])!=len(set(payload['required_case_ids'])) or {c['identity']['combination_id'] for c in payload['cuts'].values()}!=set(payload['required_case_ids']):raise ValueError('Saved required cases differ')
    output=[]
    for spec in specs:
        ident=spec['cut_id'];prior=saved[ident];action=payload['cuts'][ident]
        built,pending=_prepare(payload['inventory'],spec,action,payload['material'],payload['development_by_cut'][ident])
        if built['constructor']!=prior['constructor'] or built['identity_sha256']!=prior['material_cut_sha256'] or _hash(action)!=prior['action_sha256']:
            raise ValueError('Saved local material/action constructor differs')
        s=built['section'];assessment=prior['section_assessment'];section_status='unknown';witness=None
        if s is not None and assessment is not None:
            target=np.array(action['resultants'])/s.width
            if assessment['factored_resultants_per_in']!=target.tolist():raise ValueError('Saved local section target differs')
            # _check_point rebuilt all physical accounting from the raw cut;
            # finite section reconstruction remains distinct from method gates.
            cert=assessment.get('combined_capacity_upper_bound_certificate') or {}
            if cert.get('capacity_exceeded'):
                d=np.asarray(cert['direction']);upper=s.support_upper_bound(d);demand=float(d@target);tol=1e-8+1e-9*max(abs(upper),abs(demand))
                verified=bool(demand>upper+tol);witness=dict(kind='dual_upper_bound',verified=verified,upper=upper,demand=demand,allowance=tol)
                if verified:section_status='capacity_exceeded'
            elif assessment.get('capacity_bound_violations'):
                bounds=s.component_capacity_upper_bounds();checks=[]
                for row in assessment['capacity_bound_violations']:
                    j=row['component'];bound=bounds['positive'][j] if target[j]>=0 else bounds['negative_magnitude'][j]
                    checks.append(bool(abs(target[j])>bound+1e-8+1e-9*max(abs(target[j]),bound)))
                verified=any(checks);witness=dict(kind='component_upper_bound',verified=verified,checks=checks)
                if verified:section_status='capacity_exceeded'
            elif assessment.get('numerical_state'):
                numerical=assessment['numerical_state'];q=numerical['generalized_strain'];orders=numerical['quadrature']['orders']
                if len(orders)!=2 or any(type(n) is not int or n<4 or n>64 for n in orders) or orders[1]!=min(64,2*orders[0]):raise ValueError('Actual independent quadrature pair required')
                coarse=s.state(q,quadrature_order=orders[0]);fine=s.state(q,quadrature_order=orders[1]);a=np.array(coarse['nominal_resultants']);b=np.array(fine['nominal_resultants'])
                residual=np.array(fine['design_resultants'])-target;quad=bool(np.all(abs(a-b)<=1e-7+1e-5*np.maximum(abs(a),abs(b))))
                balanced=bool(np.all(abs(residual)<=1e-6+1e-6*abs(target)));verified=quad and balanced and fine['material_domain_valid']
                witness=dict(kind='compatible_strain',verified=bool(verified),equilibrium_residual=residual.tolist(),quadrature_verified=quad,
                    material_domain_valid=fine['material_domain_valid'],material_domain_errors=fine['material_domain_errors'])
                if verified:section_status='pass'
        if section_status=='unknown':pending.append('local_numeric_witness_not_verified')
        status='unknown' if pending else 'fail' if section_status=='capacity_exceeded' else 'pass'
        output.append(dict(cut_id=ident,status=status,section_status=section_status,pending=sorted(set(pending)),witness=witness,
            saved_claim_verified=prior['status']=='unknown' or status==prior['status']))
    status='fail' if any(r['status']=='fail' for r in output) else 'unknown' if any(r['status']=='unknown' for r in output) else 'pass'
    return dict(method=METHOD+'/saved_verification',status=status,
        records=output,saved_claims_verified=all(r['saved_claim_verified'] for r in output) and (record['status']=='unknown' or record['status']==status),inverse_solves=0,dual_searches=0,
        continuous_coverage_verified=False,engineering_qualified=False)
