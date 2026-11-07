"""Bounded complete physical/primary cut processing with explicit unprocessed scope.

An early proven failure may stop this necessary-condition check. A wall/cut
limit is unknown; neither limit nor an incomplete stream can establish a pass.
Raw compatible-strain/dual witnesses are stored per bounded chunk and replayed
without optimizers. Original physical source payloads are never sliced/relabelled.
"""
from __future__ import annotations
import copy
import gzip
import hashlib
import json
from pathlib import Path
import time
from Design.SMRF_Coupled_Floor_Design import _hash
from Design.SMRF_Physical_Floor_Strength import plan_physical_cut_family,assess_physical_cut_family,verify_physical_cut_assessment
from Design.SMRF_Common_Guard_Plan import build_common_guard_plan

METHOD='complete_common_floor_strength_stream_v2_explicit_material_action_basis'
LOCAL_PITCH_BASIS='actual_local_pitch_point_strength_v1'
LEGACY_BASIS='finite_axial_window_v1'


def _local_basis(basis,common):
    from Design.SMRF_Bar_Pitch_Layers import METHOD as PITCH_METHOD
    if basis not in (LEGACY_BASIS,LOCAL_PITCH_BASIS):raise ValueError('Explicit supported strength material/action basis required')
    if 'strength_material_action_basis' in common and common['strength_material_action_basis']!=basis:
        raise ValueError('Strength stream material/action basis differs from the bound common input')
    if basis==LOCAL_PITCH_BASIS and common.get('floor_material_mapping')!=PITCH_METHOD:
        raise ValueError('Local strength requires the identical explicit fixed-pitch analysis mapping')
    return basis==LOCAL_PITCH_BASIS


def _save(path,value):
    if path.exists():raise ValueError('A strength witness artifact cannot overwrite earlier evidence')
    with gzip.open(path,'wt',encoding='utf-8') as stream:json.dump(value,stream,allow_nan=False,separators=(',',':'))
    return dict(artifact_file=str(path.resolve()),artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),payload_sha256=value['payload_sha256'])


def read_chunk(reference):
    path=Path(reference['artifact_file'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=reference['artifact_sha256']:raise ValueError('Strength artifact bytes changed')
    with gzip.open(path,'rt',encoding='utf-8') as stream:value=json.load(stream)
    if value['payload_sha256']!=reference['payload_sha256']:raise ValueError('Strength artifact payload differs')
    return value


def _slice_plan(plan,specs):
    names=[s['cut_id'] for s in specs]
    value={**plan,'cut_specs':specs,'cuts':{k:plan['cuts'][k] for k in names},'required_cut_ids':names,'required_cut_count':len(names)}
    value['required_case_ids']=sorted({v['identity']['combination_id'] for v in value['cuts'].values()})
    value.pop('payload_sha256',None);value['payload_sha256']=_hash(value);return value


def assess_common_strength(state,common,*,limits,artifact_directory,progress=None,stop_on_proven_failure=True,prepared_sources=None,
        material_action_basis=LEGACY_BASIS):
    from Design.SMRF_Common_Action_Analysis import physical_response_bundles
    import Structure_Parameters as sp
    local=_local_basis(material_action_basis,common)
    from Design.SMRF_Local_Material_Events import installed_local_material_events
    from Design.SMRF_Local_Pitch_Strength import assess_local_pitch_cut_family,verify_local_pitch_cut_assessment
    started=time.monotonic();destination=Path(artifact_directory).resolve();destination.mkdir(parents=True,exist_ok=True)
    if any(destination.iterdir()):raise ValueError('Fresh strength artifact directory required')
    context=None if prepared_sources is None else prepared_sources.validate(common)
    bundles=physical_response_bundles(common) if context is None else context.bundles
    definition=common['definition'];inventory=common['inventory']
    universe=[[b['source_id'],int(f['level']),c['id']] for b in bundles for f in definition['floors'] for c in b['response']['cases']]
    if len({_hash(p) for p in universe})!=len(universe):raise ValueError('Duplicate full physical floor/case universe')
    wall=float(limits['strength_wall_seconds']);maximum=int(limits['max_required_cuts']);chunk_size=int(limits['max_chunk_cuts'])
    if wall<=0 or maximum<1 or chunk_size<1:raise ValueError('Explicit positive finite strength stream limits required')
    result=dict(method=METHOD,material_action_basis=material_action_basis,material_event_records={},
        state_identity=common['state_identity'],material_inventory_id=inventory['identity_sha256'],
        analysis_input_identity=common['input_signature'],source_identity=common['analysis_source_identity'],
        full_floor_case_universe=universe,full_universe_sha256=_hash(universe),chunks=[],scope_records=[],
        limits={k:limits[k] for k in ('strength_wall_seconds','max_required_cuts','max_chunk_cuts','max_strength_evaluations')},
        completed_pairs=[],processed_cut_count=0,stop_reason=None,remaining_pairs=[],status='unknown',
        strength_phis=dict(NM=.65,diaphragm_shear=.6),nominal_probable_capacity_inventory_modified=False)
    cache={};failed=False;unknown=False;fc=min(f['fc_ksi'] for f in definition['floors']);ec=sp.concrete_ec_ksi(fc)
    def timed_out():return time.monotonic()-started>=wall
    for pair_index,pair in enumerate(universe):
        if timed_out():result['stop_reason']='declared_strength_wall_bound';break
        options={}
        if local:
            floor=str(pair[1])
            if floor not in result['material_event_records']:
                result['material_event_records'][floor]=installed_local_material_events(state,inventory,pair[1],slab_fc_ksi=fc)
            options=dict(local_pitch_actions=True,local_material_events_by_floor={floor:result['material_event_records'][floor]})
            if timed_out():result['stop_reason']='declared_strength_wall_bound';break
        try:
            plan=plan_physical_cut_family(definition,inventory,bundles,planner_identity=common['input_signature'],
                maximum_cuts=maximum,selected_floor_case_pairs=[pair],prepared_responses=None if context is None else context.physical,**options)
        except ValueError as exc:
            if 'exceeds declared budget' not in str(exc):raise
            result['stop_reason']='required_physical_cut_bound';break
        scope=dict(pair=pair,physical_plan_payload_sha256=plan['payload_sha256'],physical_cut_ids=plan['required_cut_ids'],
            physical_cut_count=plan['required_cut_count'],processed_physical_cut_ids=[],complete=False,
            assigned_joint_or_other_axis_regions=plan['assigned_joint_or_other_axis_regions'])
        result['scope_records'].append(scope)
        # At most16 physical seeds create a bounded guard plan; each guard
        # plan is split again into the declared exact material solve chunks.
        for first in range(0,len(plan['cut_specs']),16):
            if timed_out():result['stop_reason']='declared_strength_wall_bound';break
            seed_specs=plan['cut_specs'][first:first+16];partial=_slice_plan(plan,seed_specs)
            try:
                guarded=build_common_guard_plan(common,partial,common['p5'],state=state,
                    maximum_cuts=maximum-result['processed_cut_count'],allow_partial_plan=True,prepared_sources=context,local_pitch_actions=local)
            except ValueError as exc:
                if 'exceed' not in str(exc) or 'budget' not in str(exc):raise
                result['stop_reason']='required_primary_guard_cut_bound';break
            for offset in range(0,len(guarded['cut_specs']),chunk_size):
                if timed_out():result['stop_reason']='declared_strength_wall_bound';break
                chosen=guarded['cut_specs'][offset:offset+chunk_size];names=[s['cut_id'] for s in chosen]
                cuts={k:guarded['cuts'][k] for k in names};cases=sorted({c['identity']['combination_id'] for c in cuts.values()})
                assess=assess_local_pitch_cut_family if local else assess_physical_cut_family
                record=assess(state,inventory,chosen,cuts,slab_fc_ksi=fc,ec_ksi=ec,
                    phi_requirements=result['strength_phis'],required_cut_ids=names,required_case_ids=cases,
                    assessment_identity=common['input_signature']+f'/chunk{len(result["chunks"])}',
                    max_evaluations=int(limits['max_strength_evaluations']),**({} if local else dict(section_initialization_cache=cache)))
                audit=(verify_local_pitch_cut_assessment if local else verify_physical_cut_assessment)(record)
                if not audit['saved_claims_verified']:raise ValueError('Current strength witnesses did not independently replay')
                reference=_save(destination/f'chunk_{len(result["chunks"]):06d}.json.gz',record)
                result['chunks'].append(dict(**reference,pair=pair,guard_plan_payload_sha256=guarded['payload_sha256'],
                    physical_seed_ids=partial['required_cut_ids'],required_cut_ids=names,required_case_ids=cases,
                    status=record['status'],verified_status=audit['status']))
                result['processed_cut_count']+=len(names);failed |= audit['status']=='fail';unknown |= audit['status']=='unknown'
                if progress:progress(dict(stage='common_strength_chunk',chunk_count=len(result['chunks']),
                    processed_cut_count=result['processed_cut_count'],pair_index=pair_index,total_pairs=len(universe),elapsed_seconds=time.monotonic()-started))
                if failed and stop_on_proven_failure:result['stop_reason']='verified_capacity_failure';break
            if result['stop_reason']:break
            scope['processed_physical_cut_ids'].extend(partial['required_cut_ids'])
        if result['stop_reason']:break
        scope['complete']=scope['processed_physical_cut_ids']==scope['physical_cut_ids'];result['completed_pairs'].append(pair)
    result['remaining_pairs']=[p for p in universe if p not in result['completed_pairs']]
    complete=not result['remaining_pairs'] and result['stop_reason'] is None
    result.update(complete_required_inventory=complete,status='fail' if failed else 'pass' if complete and not unknown else 'unknown',
        elapsed_seconds=time.monotonic()-started,method_qualification_inferred=False,complete_design_qualified=False,
        continuous_capacity_verified=False)
    result['payload_sha256']=_hash(result);return result


def verify_strength_stream(record,common,*,state,prepared_sources=None):
    """Recompute full scope and cut actions, then replay saved material witnesses."""
    from Design.SMRF_Common_Action_Analysis import physical_response_bundles
    from Design.SMRF_Local_Material_Events import installed_local_material_events
    from Design.SMRF_Local_Pitch_Strength import actual_local_cut_development,verify_local_pitch_cut_assessment
    if record.get('method')!=METHOD or record['payload_sha256']!=_hash({k:v for k,v in record.items() if k!='payload_sha256'}):raise ValueError('Strength stream method or payload changed')
    local=_local_basis(record['material_action_basis'],common)
    if (record['state_identity']!=state.identity() or record['analysis_input_identity']!=common['input_signature']
        or record['material_inventory_id']!=common['inventory']['identity_sha256'] or record['source_identity']!=common['analysis_source_identity']):raise ValueError('Strength stream design/material/input/source changed')
    context=None if prepared_sources is None else prepared_sources.validate(common)
    bundles=physical_response_bundles(common) if context is None else context.bundles
    universe=[[b['source_id'],int(f['level']),c['id']] for b in bundles for f in common['definition']['floors'] for c in b['response']['cases']]
    if universe!=record['full_floor_case_universe'] or _hash(universe)!=record['full_universe_sha256']:raise ValueError('Full physical case universe changed')
    checked=[];failure=False;unknown=False;seen=set();completed=[]
    events={};fc=min(f['fc_ksi'] for f in common['definition']['floors'])
    if not local and record['material_event_records']:raise ValueError('Legacy strength stream cannot borrow local material events')
    for floor,prior in record['material_event_records'].items():
        current=installed_local_material_events(state,common['inventory'],int(floor),slab_fc_ksi=fc)
        if current!=prior:raise ValueError('Saved material events differ from actual installed end development')
        events[floor]=current
    for scope in record['scope_records']:
        floor=str(scope['pair'][1])
        if local and floor not in events:raise ValueError('Saved local stream omits required floor material events')
        options=dict(local_pitch_actions=True,local_material_events_by_floor={floor:events[floor]}) if local else {}
        plan=plan_physical_cut_family(common['definition'],common['inventory'],bundles,planner_identity=common['input_signature'],
            maximum_cuts=record['limits']['max_required_cuts'],selected_floor_case_pairs=[scope['pair']],prepared_responses=None if context is None else context.physical,**options)
        if plan['payload_sha256']!=scope['physical_plan_payload_sha256'] or plan['required_cut_ids']!=scope['physical_cut_ids']:
            raise ValueError('Saved physical required-cut scope differs from reconstructed current fields')
        references=[r for r in record['chunks'] if r['pair']==scope['pair']];groups={}
        for reference in references:groups.setdefault(tuple(reference['physical_seed_ids']),[]).append(reference)
        done=[]
        for seed_ids,refs in groups.items():
            seeds=[s for s in plan['cut_specs'] if s['cut_id'] in seed_ids]
            if [s['cut_id'] for s in seeds]!=list(seed_ids):raise ValueError('Unknown/reordered physical guard seeds')
            guarded=build_common_guard_plan(common,_slice_plan(plan,seeds),common['p5'],state=state,
                maximum_cuts=record['limits']['max_required_cuts'],allow_partial_plan=True,prepared_sources=context,local_pitch_actions=local)
            received=[]
            for reference in refs:
                if reference['guard_plan_payload_sha256']!=guarded['payload_sha256']:raise ValueError('Saved primary guard plan differs')
                chunk=read_chunk(reference);payload=chunk['replay_payload'];names=payload['required_cut_ids']
                from Design.SMRF_Actual_Bar_Development import actual_bar_development
                if local:
                    development={s['cut_id']:actual_local_cut_development(state,common['inventory'],s,payload['cuts'][s['cut_id']],payload['material']) for s in payload['cut_specs']}
                    if development!=payload['development_by_cut']:raise ValueError('Saved local full-fy development differs from actual detailing')
                else:
                    development=actual_bar_development(state,common['inventory'],payload['cut_specs'],slab_fc_ksi=payload['material']['slab_fc_ksi'])
                    if development!=payload['development_result']:raise ValueError('Saved full-fy development differs from current actual detailing')
                if names!=reference['required_cut_ids'] or any(k in seen for k in names):raise ValueError('Duplicate/changed saved cut chunks')
                for spec in payload['cut_specs']:
                    expected=next((s for s in guarded['cut_specs'] if s['cut_id']==spec['cut_id']),None)
                    if spec!=expected or payload['cuts'][spec['cut_id']]!=guarded['cuts'][spec['cut_id']]:raise ValueError('Strength witness is not the actual reconstructed common cut')
                audit=(verify_local_pitch_cut_assessment if local else verify_physical_cut_assessment)(chunk);checked.append(audit);seen.update(names);received.extend(names)
                failure |= audit['status']=='fail';unknown |= audit['status']=='unknown'
            if received==guarded['required_cut_ids']:done.extend(seed_ids)
            elif not set(received)<=set(guarded['required_cut_ids']):raise ValueError('Unrecognized cut guard evidence')
        if done==plan['required_cut_ids']:completed.append(scope['pair'])
        if scope['complete']!=(done==plan['required_cut_ids']):raise ValueError('Saved completed scope claim differs')
    complete=completed==universe and len(seen)==record['processed_cut_count']
    if complete!=record['complete_required_inventory']:raise ValueError('Saved complete cut universe claim differs')
    return dict(method=METHOD,status='fail' if failure else 'pass' if complete and not unknown else 'unknown',
        complete_required_inventory=complete,processed_cut_count=len(seen),chunk_count=len(checked),inverse_solves=0,dual_searches=0,
        remaining_pairs=[p for p in universe if p not in completed],saved_claims_verified=True,continuous_capacity_verified=False,
        material_action_basis=record['material_action_basis'])
