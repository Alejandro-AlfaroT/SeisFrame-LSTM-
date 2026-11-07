"""Bounded local flexure/one-way-shear calculation and optimizer-free readback."""
from __future__ import annotations
from pathlib import Path
import time
from Design.SMRF_Coupled_Floor_Design import _hash
from Design.SMRF_Common_Local_Flexure import plan_local_transfer,one_way_from_development
from Design.SMRF_Common_Strength_Stream import _save,read_chunk
from Design.SMRF_Physical_Floor_Strength import assess_physical_cut_family,verify_physical_cut_assessment
from Design.SMRF_Actual_Bar_Development import actual_bar_development
from Design.SMRF_Common_Strength_Stream import LEGACY_BASIS,_local_basis

METHOD='all_actual_local_transfer_strength_and_shear_stream_v2_explicit_local_basis'


def local_universe(common,prepared_sources=None):
    from Design.SMRF_Common_Action_Analysis import physical_response_bundles
    bundles=physical_response_bundles(common) if prepared_sources is None else prepared_sources.validate(common).bundles
    return [[b['source_id'],int(f['level']),c['id']] for b in bundles
            for f in common['definition']['floors'] for c in b['response']['cases']]


def assess_local_strength(state,common,*,limits,artifact_directory,progress=None,stop_on_proven_failure=True,prepared_sources=None,
        material_action_basis=LEGACY_BASIS):
    import Structure_Parameters as sp
    from Design.SMRF_Local_Pitch_Strength import assess_local_pitch_cut_family,verify_local_pitch_cut_assessment
    local=_local_basis(material_action_basis,common)
    dest=Path(artifact_directory).resolve();dest.mkdir(parents=True,exist_ok=True)
    if any(dest.iterdir()):raise ValueError('Fresh local strength witness directory required')
    started=time.monotonic();universe=local_universe(common,prepared_sources);cache={};fc=min(f['fc_ksi'] for f in common['definition']['floors'])
    result=dict(method=METHOD,material_action_basis=material_action_basis,source_identity=common.get('analysis_source_identity'),
        state_identity=common['state_identity'],analysis_input_identity=common['input_signature'],
        material_inventory_id=common['inventory']['identity_sha256'],full_universe=universe,scopes=[],chunks=[],
        completed_pairs=[],remaining_pairs=[],processed_cut_count=0,one_way_checks=[],limits=dict(limits),stop_reason=None)
    failed=False;unknown=False
    def bounded():return time.monotonic()-started>=limits['strength_wall_seconds']
    for pair in universe:
        if bounded():result['stop_reason']='declared_local_strength_wall_bound';break
        try:plan=plan_local_transfer(common,state,selected_floor_case_pair=pair,maximum_cuts=int(limits['max_required_cuts'])-result['processed_cut_count'],prepared_sources=prepared_sources,local_pitch_actions=local)
        except ValueError as exc:
            if 'exceeds declared cut budget' not in str(exc):raise
            result['stop_reason']='declared_local_cut_bound';break
        scope=dict(pair=pair,plan_sha256=plan['payload_sha256'],required_cut_ids=plan['required_cut_ids'],processed_cut_ids=[],complete=False)
        result['scopes'].append(scope)
        for at in range(0,len(plan['cut_specs']),int(limits['max_chunk_cuts'])):
            if bounded():result['stop_reason']='declared_local_strength_wall_bound';break
            specs=plan['cut_specs'][at:at+int(limits['max_chunk_cuts'])];names=[s['cut_id'] for s in specs]
            options={} if local else dict(development_result=actual_bar_development(state,common['inventory'],specs,slab_fc_ksi=fc),section_initialization_cache=cache)
            assess=assess_local_pitch_cut_family if local else assess_physical_cut_family
            assessment=assess(state,common['inventory'],specs,{n:plan['cuts'][n] for n in names},slab_fc_ksi=fc,
                ec_ksi=sp.concrete_ec_ksi(fc),phi_requirements=dict(NM=.65,diaphragm_shear=.6),required_cut_ids=names,
                required_case_ids=plan['required_case_ids'],assessment_identity=common['input_signature']+'/local/'+str(len(result['chunks'])),
                max_evaluations=int(limits['max_strength_evaluations']),**options)
            verified=(verify_local_pitch_cut_assessment if local else verify_physical_cut_assessment)(assessment)
            if not verified['saved_claims_verified']:raise ValueError('New local material witness failed independent readback')
            ref=_save(dest/f'local_chunk_{len(result["chunks"]):06d}.json.gz',assessment)
            result['chunks'].append(dict(ref,pair=pair,plan_sha256=plan['payload_sha256'],required_cut_ids=names))
            scope['processed_cut_ids'].extend(names);result['processed_cut_count']+=len(names)
            failed |= verified['status']=='fail';unknown |= verified['status']=='unknown'
            if progress:progress(dict(stage='local_transfer_strength_chunk',chunks=len(result['chunks']),processed_cuts=result['processed_cut_count'],elapsed_seconds=time.monotonic()-started))
            if failed and stop_on_proven_failure:result['stop_reason']='verified_local_capacity_failure';break
        if result['stop_reason']:break
        shear_specs=[r['specification'] for r in plan['one_way_raw_cuts']]
        shear_dev=actual_bar_development(state,common['inventory'],shear_specs,slab_fc_ksi=fc)
        checks=[one_way_from_development(common['inventory'],raw,dev,fc_ksi=fc) for raw,dev in zip(plan['one_way_raw_cuts'],shear_dev['cuts'])]
        result['one_way_checks'].append(dict(pair=pair,plan_sha256=plan['payload_sha256'],checks=checks,development=shear_dev))
        failed |= any(r['status']=='fail' for r in checks);unknown |= any(r['status']=='unknown' for r in checks)
        scope['complete']=scope['processed_cut_ids']==scope['required_cut_ids'];result['completed_pairs'].append(pair)
        if failed and stop_on_proven_failure:result['stop_reason']='verified_local_one_way_shear_failure';break
    result['remaining_pairs']=[p for p in universe if p not in result['completed_pairs']]
    result.update(complete_required_inventory=not result['remaining_pairs'],
        status='fail' if failed else 'pass' if not result['remaining_pairs'] and not unknown else 'unknown',
        elapsed_seconds=time.monotonic()-started,nominal_probable_capacity_inventory_modified=False,method_qualification_inferred=False,
        continuous_capacity_verified=False)
    result['payload_sha256']=_hash(result);return result


def verify_local_strength(record,common,*,state,prepared_sources=None):
    from Design.SMRF_Local_Pitch_Strength import actual_local_cut_development,verify_local_pitch_cut_assessment
    if record.get('method')!=METHOD or record['payload_sha256']!=_hash({k:v for k,v in record.items() if k!='payload_sha256'}):raise ValueError('Saved local stream method or content changed')
    local=_local_basis(record['material_action_basis'],common)
    if (record['state_identity']!=state.identity() or record['analysis_input_identity']!=common['input_signature'] or
        record['material_inventory_id']!=common['inventory']['identity_sha256'] or record['source_identity']!=common.get('analysis_source_identity')):raise ValueError('Saved local stream actual state/input/material/source differs')
    universe=local_universe(common,prepared_sources)
    if universe!=record['full_universe']:raise ValueError('Saved local complete case universe changed')
    done=[];seen=set();failed=False;unknown=False;fc=min(f['fc_ksi'] for f in common['definition']['floors']);shear_count=0
    for scope in record['scopes']:
        plan=plan_local_transfer(common,state,selected_floor_case_pair=scope['pair'],maximum_cuts=int(record['limits']['max_required_cuts']),prepared_sources=prepared_sources,local_pitch_actions=local)
        if plan['payload_sha256']!=scope['plan_sha256'] or plan['required_cut_ids']!=scope['required_cut_ids']:raise ValueError('Reconstructed local physical/primary/gamma_f plan differs')
        received=[]
        for ref in [r for r in record['chunks'] if r['pair']==scope['pair']]:
            chunk=read_chunk(ref);p=chunk['replay_payload'];names=p['required_cut_ids']
            if ref['plan_sha256']!=plan['payload_sha256'] or names!=ref['required_cut_ids'] or set(names)&seen:raise ValueError('Local chunk scope changed/duplicated')
            expected=[s for s in plan['cut_specs'] if s['cut_id'] in names]
            if expected!=p['cut_specs'] or p['cuts']!={n:plan['cuts'][n] for n in names}:raise ValueError('Local saved action is not the reconstructed actual support state')
            if local:
                dev={s['cut_id']:actual_local_cut_development(state,common['inventory'],s,p['cuts'][s['cut_id']],p['material']) for s in expected}
                if dev!=p['development_by_cut']:raise ValueError('Local actual bar development changed')
            else:
                dev=actual_bar_development(state,common['inventory'],expected,slab_fc_ksi=fc)
                if dev!=p['development_result']:raise ValueError('Local actual bar development changed')
            verified=(verify_local_pitch_cut_assessment if local else verify_physical_cut_assessment)(chunk)
            if not verified['saved_claims_verified']:raise ValueError('Local saved capacity claim could not be verified')
            failed |= verified['status']=='fail';unknown |= verified['status']=='unknown';seen.update(names);received.extend(names)
        shear=[r for r in record['one_way_checks'] if r['pair']==scope['pair']]
        complete=received==plan['required_cut_ids'] and len(shear)==1
        if len(shear)>1:raise ValueError('Duplicate local one-way shear record')
        if shear:
            specs=[r['specification'] for r in plan['one_way_raw_cuts']]
            dev=actual_bar_development(state,common['inventory'],specs,slab_fc_ksi=fc)
            checks=[one_way_from_development(common['inventory'],r,d,fc_ksi=fc) for r,d in zip(plan['one_way_raw_cuts'],dev['cuts'])]
            if shear[0]!=dict(pair=scope['pair'],plan_sha256=plan['payload_sha256'],checks=checks,development=dev):raise ValueError('Local one-way shear differs from actual actions/development')
            failed |= any(r['status']=='fail' for r in checks);unknown |= any(r['status']=='unknown' for r in checks);shear_count+=len(checks)
        if complete:done.append(scope['pair'])
        if complete!=scope['complete']:raise ValueError('Saved local scope completion claim differs')
    complete=done==universe and len(seen)==record['processed_cut_count']
    if complete!=record['complete_required_inventory'] or done!=record['completed_pairs']:raise ValueError('Saved local total inventory claim differs')
    return dict(method=METHOD,status='fail' if failed else 'pass' if complete and not unknown else 'unknown',
        complete_required_inventory=complete,processed_cut_count=len(seen),one_way_cut_count=shear_count,
        inverse_solves=0,dual_searches=0,saved_claims_verified=True,method_qualification_inferred=False,
        continuous_capacity_verified=False,material_action_basis=record['material_action_basis'])
