"""Bounded actual-layout selection through complete physical assessment callbacks.

Each candidate requires its own installed geometry, analysis and common
strength payload. No zero-N slab assertion, stale action cache, or successful
layout geometry alone can activate a candidate. Catalog order is explicit.
"""
from __future__ import annotations
import copy
import math
import itertools
from Design.SMRF_Common_Floor_Actions import _hash

METHOD='bounded_actual_slab_layout_common_strength_selection_v1'


def catalog_proposals(slab_inputs,policy=None,*,maximum_candidates=30000):
    """Enumerate the unchanged original three-bar/four-layer spacing catalog.

    Uses the existing policy (default #4/#5/#6,3..12in) and four independent
    spacings, never a coarser replacement catalog. Only computed minimum
    steel, clear spacing, maximum spacing and finite four-layer depth reject
    a proposal here. Actual global phase/cage/hook geometry and simultaneous
    actions are pending. Sorting uses nominal slab density merely to order
    trials; final selection uses exact installed steel volume.
    """
    from Design.SMRF_Slab_Reinforcement import _inputs,_policy,_layout,computed_layout
    inputs=_inputs(slab_inputs);resolved=_policy(policy);bars=resolved['bar_sizes'];spacings=resolved['spacing_options_in']
    total=len(bars)*len(spacings)**4
    if not isinstance(maximum_candidates,int) or maximum_candidates<total:
        raise ValueError('Full original catalog exceeds declared bound; specify a larger bound or explicit original policy')
    candidates=[];rejected=[];names=('x_top','x_bottom','y_top','y_bottom')
    for bar in bars:
        templates={(axis,spacing):_layout(inputs,resolved,bar,axis,spacing) for axis in ('x','y') for spacing in spacings}
        for values in itertools.product(spacings,repeat=4):
            layers={name:copy.deepcopy(templates[(name[0],spacing)]) for name,spacing in zip(names,values)}
            failures=[]
            for name,row in layers.items():
                if row is None:failures.append(name+'/nonpositive_depth');continue
                if row['area_in2_per_ft']<row['minimum_area_in2_per_ft']-1e-10:failures.append(name+'/minimum_steel')
                if row['horizontal_clear_spacing_in']<row['minimum_clear_spacing_in']-1e-10:failures.append(name+'/clear_spacing')
                if row['spacing_in']>row['maximum_spacing_in']+1e-10:failures.append(name+'/maximum_spacing')
            sample=next((v for v in layers.values() if v is not None),None)
            if sample is not None and inputs['thickness_in']<=2*resolved['clear_cover_in']+4*sample['bar_diameter_in']:
                failures.append('four_mat_layers_do_not_fit_slab_depth')
            key=f'No{bar}/'+','.join(f'{name}:{spacing:g}' for name,spacing in zip(names,values))
            if failures:
                rejected.append(dict(candidate_id=key,status='fail',basis='computed_basic_geometry_or_minimum_steel',failures=failures));continue
            # Canonical proposal metadata is retained for exact comparison;
            # its zero-N strip capacities never qualify concurrent strength.
            layout=computed_layout(inputs,bar,dict(zip(names,values)),resolved)
            if layout['mat_clear_gap_in']<layout['minimum_mat_clear_gap_in']-1e-10:
                rejected.append(dict(candidate_id=key,status='fail',basis='computed_basic_geometry_or_minimum_steel',failures=['minimum_between_mat_clear_gap']));continue
            candidates.append(dict(candidate_id=key,layout=layout,layout_sha256=_hash(layout),
                nominal_steel_area_sum_in2_per_ft=sum(v['area_in2_per_ft'] for v in layers.values()),
                geometry_status='actual_placement_not_evaluated',common_strength_status='not_evaluated'))
    candidates.sort(key=lambda r:(r['nominal_steel_area_sum_in2_per_ft'],r['layout']['bar_size'],r['candidate_id']))
    output=dict(method=METHOD+'/original_catalog',inputs=inputs,policy=resolved,enumerated_count=total,
        candidates=candidates,rejected=rejected,physical_phase_hook_cage_verified=False,
        capacity_or_zero_N_assertion_used=False,installed_state_mutated=False)
    output['payload_sha256']=_hash(output);return output


def select_physical_layout(candidates, evaluate, *, selection_identity,
        maximum_candidates, required_case_ids, required_cut_ids):
    """Evaluate bounded named catalog layouts without mutating caller state.

    candidates are {candidate_id,layout}. evaluate(candidate,identity) returns
    layout_sha256, state_identity, installed_layout, assessment, quantities,
    and qualification={status,evidence_id}. assessment is the complete
    SMRF_Physical_Floor_Strength result for the explicitly required inventory.
    quantities.steel_volume_in3 is the actual installed physical inventory,
    including hooks. Method/mesh/refinement closure remains an explicit
    computed qualification supplied by the production orchestrator.
    """
    rows=list(candidates)
    if not selection_identity or not isinstance(maximum_candidates,int) or maximum_candidates<1:
        raise ValueError('Named selection and positive bounded candidate count required')
    if len(rows)>maximum_candidates:raise ValueError('Candidate inventory exceeds declared selection budget')
    names=[r['candidate_id'] for r in rows]
    if not rows or len(names)!=len(set(names)):raise ValueError('Nonempty unique actual catalog candidates required')
    cases=set(required_case_ids);cuts=set(required_cut_ids)
    if not cases or not cuts:raise ValueError('Explicit complete required cases/cuts required')
    outputs=[];admissible=[]
    for index,candidate in enumerate(rows):
        source=copy.deepcopy(candidate);identity=f'{selection_identity}/{index:03d}/{source["candidate_id"]}'
        result=evaluate(copy.deepcopy(source),identity);assessment=result.get('assessment',{});pending=[]
        geometry=result.get('geometry_assessment') or {}
        if (geometry.get('status')=='fail' and geometry.get('state_identity')==result.get('state_identity') and
                geometry.get('layout_sha256')==_hash(source['layout']) and geometry.get('evidence_id') and geometry.get('failures') and
                result.get('installed_layout')==source['layout']):
            outputs.append(dict(candidate_id=source['candidate_id'],evaluation_identity=identity,candidate=source,
                status='fail',failure_class='computed_actual_geometry',pending=[],result=result,steel_volume_in3=None))
            continue
        if _hash(result.get('installed_layout'))!=_hash(source['layout']):pending.append('evaluated_layout_differs_from_catalog_candidate')
        if result.get('layout_sha256')!=_hash(source['layout']):pending.append('actual_layout_payload_hash_not_bound')
        if not result.get('state_identity') or result['state_identity']!=assessment.get('state_identity'):
            pending.append('actual_installed_state_not_bound_to_assessment')
        if set(assessment.get('required_case_ids',[]))!=cases or set(assessment.get('actual_case_ids',[]))!=cases:
            pending.append('complete_required_case_inventory_not_assessed')
        if set(assessment.get('required_cut_ids',[]))!=cuts or set(assessment.get('actual_cut_ids',[]))!=cuts:
            pending.append('complete_required_cut_inventory_not_assessed')
        if not assessment.get('payload_sha256') or assessment['payload_sha256']!=_hash(assessment.get('replay_payload')):
            pending.append('exact_common_strength_payload_not_bound')
        qualification=result.get('qualification',{})
        if qualification.get('status')!='pass' or not qualification.get('evidence_id'):
            pending.append('physical_method_mesh_trace_qualification_not_established')
        volume=float(result.get('quantities',{}).get('steel_volume_in3',math.nan))
        if not math.isfinite(volume) or volume<=0:pending.append('actual_installed_steel_quantity_missing')
        status='unknown' if pending or assessment.get('status') not in ('pass','fail') else assessment['status']
        row=dict(candidate_id=source['candidate_id'],evaluation_identity=identity,candidate=source,
                 status=status,pending=pending,result=result,steel_volume_in3=volume if math.isfinite(volume) else None)
        outputs.append(row)
        if status=='pass':admissible.append((volume,index,row))
    selected=min(admissible,key=lambda r:(r[0],r[1]))[2] if admissible else None
    output=dict(method=METHOD,selection_identity=str(selection_identity),maximum_candidates=maximum_candidates,
        evaluated_candidates=len(outputs),required_case_ids=list(required_case_ids),required_cut_ids=list(required_cut_ids),
        status='selected' if selected else 'unknown' if any(r['status']=='unknown' for r in outputs) else 'no_feasible_candidate',
        selected_candidate_id=None if selected is None else selected['candidate_id'],
        selected_state_identity=None if selected is None else selected['result']['state_identity'],
        records=outputs,installed_state_mutated=False,old_zero_axial_assertion_used=False)
    output['payload_sha256']=_hash(output)
    return output
