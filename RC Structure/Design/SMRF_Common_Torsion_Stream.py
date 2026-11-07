"""Bounded actual torsion/capacity/detail assessment and optimizer-free readback."""
from __future__ import annotations
from pathlib import Path
import time
import Structure_Parameters as sp
from Design.SMRF_Coupled_Floor_Design import _hash
from Design.SMRF_Common_Torsion_Recovery import recover_common_torsion
from Design.SMRF_Common_Torsion_Guards import plan_record,longitudinal_reserve
from Design.SMRF_Beam_Torsion_Detail import member_detail,transverse_variants,variant_design
from Design.SMRF_Beam_Torsion import assess_torsion
from Design.SMRF_Common_Guard_Plan import PreparedCommonGuardSources
from Design.SMRF_Common_Strength_Stream import _save,read_chunk
from Design.SMRF_Physical_Floor_Strength import assess_physical_cut_family
from Design.SMRF_Continuous_Torsion import continuous_member_coverage

METHOD='computed_guarded_beam_torsion_stream_v3_continuous_threshold'
CHUNK_METHOD='computed_guarded_beam_torsion_member_chunk_v2_continuous_threshold'
RECORD_ORDER='original_ELF_then_signed_compatibility_then_gravity_v1'


def _ordered_records(raw):
    priority={'actual_physical_ELF':0,'actual_signed_Cd_compatibility':1,'actual_gravity':2}
    branches={c['analysis_case_id']:c['branch'] for c in raw['analysis_cases']}
    return sorted(raw['records'],key=lambda r:priority[branches[r['case_id']]])


def capacity_inventory(state,common):
    from Design.Group_Capacity import beam_group_capacity
    transfers=common['capacity_gravity_transfer']
    return {gid:beam_group_capacity(gid,state,transfers,sp.ASCE_SDS) for gid,d in state.designs.items() if not d.is_column}


def capacity_binding(capacity):
    return {k:capacity[k] for k in ('group_id','members','vc_kip','effective_depth_in')}


def assess_record(state,record,section,plan,detail,capacity,material=None):
    """Recompute actual threshold, full-Mpr shear and every installed hoop variant."""
    member=state.member(record['member_tag']);design=member.design
    identity={k:record[k] for k in ('case_id','cut_id','state_identity','recovery_id')}
    if plan['record_identity']!=identity or plan['section_id']!=section['section_id']:raise ValueError('Torsion guard record identity differs')
    if plan['payload_sha256']!=_hash({k:v for k,v in plan.items() if k!='payload_sha256'}):raise ValueError('Torsion guard payload changed')
    if plan['all_demands_below_threshold']:
        # Do not consume the saved boolean; every threshold is recomputed.
        rows=[assess_torsion(section,d['T_about_positive_axis_kipin'],d['N_tension_kip'],design=design,
            coordinates=[],spacing_in=design.stirrup_spacing_in,clear_cover_in=sp.BEAM_CLEAR_COVER_IN,identity=identity) for d in plan['demands']]
        if not rows or any(r.get('branch')!='below_threshold' for r in rows):raise ValueError('Saved all-guard torsion threshold claim differs')
        return dict(status='pass',branch='all_exact_section_guards_below_threshold',assessments=rows,material_status=None)
    if detail is None:raise ValueError('Above-threshold actual member detail inventory required')
    ids=detail['development']['longitudinal_bar_ids']
    reserve=None if material is None else longitudinal_reserve(material,plan['material_plan'],record,ids)
    local=transverse_variants(design,detail['splice_detail'],station_from_left_face_in=record['station_in']-detail['development']['support_faces_in'][0],
        clear_span_in=detail['development']['clear_span_in'])
    fullve=max((m['governing_ve_kip'] for m in capacity['members'] if str(m['member_tag'])==str(record['member_tag'])),default=None)
    if fullve is None:raise ValueError('Actual beam full-Mpr capacity shear is missing')
    # The prescribed frame's in-plane member action is retained. Its plastic
    # hinge mechanism is the beam strong-axis Mpr mechanism; no weak-axis
    # plastic hinge is designated in this SFRS. Vc=0 is used in-plane.
    left=[-p['x_in'] for p in detail['coordinates'] if p['x_in']<0.]
    right=[p['x_in'] for p in detail['coordinates'] if p['x_in']>0.]
    if not left or not right:raise ValueError('Actual reinforcement on both in-plane faces required')
    inplane_d=design.b_in/2+min(sum(left)/len(left),sum(right)/len(right))
    rows=[]
    for demand in plan['demands']:
        for variant in local['variants']:
            shear=dict(vertical=dict(vu_kip=demand['Q_vertical_kip'],capacity_ve_kip=fullve,vc_kip=capacity['vc_kip'],
                d_in=capacity['effective_depth_in'],width_in=design.b_in),
                inplane=dict(vu_kip=demand['V_inplane_kip'],capacity_ve_kip=0.,vc_kip=0.,d_in=inplane_d,width_in=design.h_in))
            value=assess_torsion(section,demand['T_about_positive_axis_kipin'],demand['N_tension_kip'],design=variant_design(design,variant),
                coordinates=detail['coordinates'],spacing_in=variant['spacing_in'],clear_cover_in=sp.BEAM_CLEAR_COVER_IN,
                shear_reservations=shear,longitudinal_state=reserve,development=detail['development'],identity=identity)
            rows.append(dict(guard_cut_id=demand['guard_cut_id'],demand_kind=demand['kind'],hoop_variant=variant,**value))
    # A measured clash/cover/spacing failure is a real invalid installed
    # detail. Missing product or geometry evidence remains not_evaluated.
    status='fail' if local['status']=='fail' or any(r['status']=='fail' for r in rows) else 'pass' if rows and local['status']=='pass' and all(r['status']=='pass' for r in rows) else 'not_evaluated'
    return dict(status=status,branch='explicit_reinforcement_reserve',assessments=rows,
        longitudinal_reserve=reserve,transverse_inventory=local,full_Mpr_capacity_ve_kip=fullve,
        physical_detail_status=detail['splice_detail']['status'],
        physical_detail_failure_checks=[c for c in detail['splice_detail'].get('checks',[]) if c.get('status')=='fail'],
        material_status=None if material is None else material['status'])


def _groups(state,raw,records,complete,continuous):
    out={}
    for gid,d in state.designs.items():
        if d.is_column:continue
        tags=set(state.groups[gid]['member_tags']);rows=[r for r in records if r['member_tag'] in tags]
        expected=sum(r['member_tag'] in tags for r in raw['records'])
        whole=complete and len(rows)==expected and expected>0
        covered=all(continuous['members'].get(str(t),{}).get('status')=='pass' for t in tags)
        status='fail' if any(r['assessment']['status']=='fail' for r in rows) else 'pass' if whole and covered and all(r['assessment']['status']=='pass' for r in rows) else 'not_evaluated'
        out[gid]=dict(status=status,processed_records=len(rows),required_records=expected,complete_required_inventory=whole,
            continuous_interval_coverage_complete=covered)
    return out


def _assess_torsion_chunk(state,common,*,limits,artifact_directory,selected_member_tags,prepared_sources=None,progress=None,stop_on_proven_failure=True):
    started=time.monotonic();context=PreparedCommonGuardSources(common) if prepared_sources is None else prepared_sources.validate(common)
    dest=Path(artifact_directory).resolve();dest.mkdir(parents=True,exist_ok=True)
    if any(dest.iterdir()):raise ValueError('Fresh torsion evidence directory required')
    raw=recover_common_torsion(state,common,maximum_records=int(limits['max_required_cuts']),prepared_sources=context,selected_member_tags=selected_member_tags)
    continuous=continuous_member_coverage(raw)
    out=dict(method=CHUNK_METHOD,state_identity=state.identity(),analysis_input_identity=common['input_signature'],source_id=raw['source_id'],
        material_inventory_id=common['inventory']['identity_sha256'],raw_reference=_save(dest/'raw_torsion.json.gz',raw),
        limits=dict(limits),record_order=RECORD_ORDER,records=[],material_cut_count=0,stop_reason=None,
        continuous_coverage=continuous)
    capacities=None;details={};cache={};fc=raw['slab_fc_ksi']
    for record in _ordered_records(raw):
        if time.monotonic()-started>=limits['strength_wall_seconds']:out['stop_reason']='declared_torsion_wall_bound';break
        plan=plan_record(state,common,record,raw['sections'][str(record['member_tag'])],maximum_cuts=int(limits['max_required_cuts']),prepared_sources=context)
        material=None;detail=None;capacity=None
        if not plan['all_demands_below_threshold']:
            if capacities is None:capacities=capacity_inventory(state,common)
            tag=record['member_tag'];member=state.member(tag)
            if tag not in details:details[tag]=member_detail(state,common['inventory'],member,slab_fc_ksi=fc)
            detail=details[tag];capacity=capacity_binding(capacities[member.group_id]);p=plan['material_plan']
            if out['material_cut_count']+len(p['cut_specs'])>limits['max_required_cuts']:
                out['stop_reason']='declared_torsion_material_cut_bound';break
            material=assess_physical_cut_family(state,common['inventory'],p['cut_specs'],p['cuts'],slab_fc_ksi=fc,
                ec_ksi=sp.concrete_ec_ksi(fc),phi_requirements=dict(NM=.65,diaphragm_shear=.6),required_cut_ids=p['required_cut_ids'],
                required_case_ids=p['required_case_ids'],assessment_identity=common['input_signature']+'/torsion/'+str(len(out['records'])),
                max_evaluations=int(limits['max_strength_evaluations']),section_initialization_cache=cache)
            out['material_cut_count']+=len(p['cut_specs'])
        assessed=assess_record(state,record,raw['sections'][str(record['member_tag'])],plan,detail,capacity,material)
        witness=dict(record=record,plan=plan,detail=detail,capacity=capacity,material=material,assessment=assessed)
        witness['payload_sha256']=_hash(witness)
        ref=_save(dest/f'torsion_{len(out["records"]):06d}.json.gz',witness)
        out['records'].append(dict(member_tag=record['member_tag'],case_id=record['case_id'],cut_id=record['cut_id'],reference=ref,assessment=dict(status=assessed['status'])))
        if progress:progress(dict(stage='torsion_record',processed_records=len(out['records']),required_records=len(raw['records']),elapsed_seconds=time.monotonic()-started))
        if assessed['status']=='fail' and stop_on_proven_failure:out['stop_reason']='verified_torsion_failure';break
    complete=raw['cut_inventory_complete'] and len(out['records'])==len(raw['records'])
    out.update(complete_required_inventory=complete,required_record_count=len(raw['records']),groups=_groups(state,raw,out['records'],complete,continuous),
        selected_member_tags=raw['selected_member_tags'],complete_selected_inventory=len(out['records'])==len(raw['records']),
        elapsed_seconds=time.monotonic()-started,method_qualification_inferred=False)
    statuses=[g['status'] for g in out['groups'].values()]
    out['status']='fail' if 'fail' in statuses else 'pass' if statuses and all(s=='pass' for s in statuses) else 'not_evaluated'
    out['payload_sha256']=_hash(out);return out


def _verify_torsion_chunk(saved,common,*,state,prepared_sources=None,beam_capacity=None):
    if saved.get('method')!=CHUNK_METHOD or saved.get('record_order')!=RECORD_ORDER:raise ValueError('Supported actual torsion member chunk/order required')
    if saved['payload_sha256']!=_hash({k:v for k,v in saved.items() if k!='payload_sha256'}):raise ValueError('Saved torsion stream changed')
    if saved['state_identity']!=state.identity() or saved['analysis_input_identity']!=common['input_signature'] or saved['material_inventory_id']!=common['inventory']['identity_sha256']:
        raise ValueError('Saved torsion input/material/state changed')
    context=PreparedCommonGuardSources(common) if prepared_sources is None else prepared_sources.validate(common)
    raw=recover_common_torsion(state,common,maximum_records=int(saved['limits']['max_required_cuts']),prepared_sources=context,
        selected_member_tags=saved['selected_member_tags'])
    continuous=continuous_member_coverage(raw)
    if continuous!=saved.get('continuous_coverage'):raise ValueError('Saved continuous torsion interval evidence differs')
    if read_chunk(saved['raw_reference'])!=raw:raise ValueError('Saved torsion physical inventory differs from current raw recovery')
    bykey={(r['member_tag'],r['case_id'],r['cut_id']):r for r in raw['records']};seen=set();results=[];details={};capacities=beam_capacity;material_count=0
    ordered=[(r['member_tag'],r['case_id'],r['cut_id']) for r in _ordered_records(raw)]
    for row in saved['records']:
        key=(row['member_tag'],row['case_id'],row['cut_id'])
        if key in seen or key not in bykey or key!=ordered[len(seen)]:raise ValueError('Duplicate, missing or reordered actual torsion record')
        seen.add(key);record=bykey[key];witness=read_chunk(row['reference'])
        if witness['payload_sha256']!=_hash({k:v for k,v in witness.items() if k!='payload_sha256'}):raise ValueError('Torsion witness payload changed')
        section=raw['sections'][str(record['member_tag'])]
        plan=plan_record(state,common,record,section,maximum_cuts=int(saved['limits']['max_required_cuts']),prepared_sources=context)
        if witness['record']!=record or witness['plan']!=plan:raise ValueError('Torsion guard is not the reconstructed exact physical/primary/P5 state')
        detail=None;capacity=None;material=witness['material']
        if not plan['all_demands_below_threshold']:
            tag=record['member_tag'];member=state.member(tag)
            if tag not in details:details[tag]=member_detail(state,common['inventory'],member,slab_fc_ksi=raw['slab_fc_ksi'])
            detail=details[tag]
            if capacities is None:capacities=capacity_inventory(state,common)
            capacity=capacity_binding(capacities[member.group_id])
            from Design.SMRF_Actual_Bar_Development import actual_bar_development
            if material is None:raise ValueError('Required saved compatible material witness is absent')
            material_count+=len(material['required_cut_ids'])
            dev=actual_bar_development(state,common['inventory'],plan['material_plan']['cut_specs'],slab_fc_ksi=raw['slab_fc_ksi'])
            if dev!=material['replay_payload']['development_result']:raise ValueError('Saved torsion guard development differs from actual bars')
        if witness['detail']!=detail or witness['capacity']!=capacity:raise ValueError('Actual torsion support/splice/full-Mpr shear detail changed')
        result=assess_record(state,record,section,plan,detail,capacity,material)
        if result!=witness['assessment'] or row['assessment']!={'status':result['status']}:raise ValueError('Saved torsion capacity result differs from independent replay')
        results.append(row)
    complete=raw['cut_inventory_complete'] and len(seen)==len(bykey)
    selected_complete=len(seen)==len(bykey)
    if material_count!=saved['material_cut_count']:raise ValueError('Saved torsion material work count differs')
    if (saved['complete_selected_inventory']!=selected_complete or saved['required_record_count']!=len(bykey)
        or saved['selected_member_tags']!=raw['selected_member_tags']):raise ValueError('Saved torsion selected inventory differs')
    groups=_groups(state,raw,results,complete,continuous)
    if complete!=saved['complete_required_inventory'] or groups!=saved['groups']:raise ValueError('Saved torsion complete/group claim differs')
    statuses=[g['status'] for g in groups.values()]
    status='fail' if 'fail' in statuses else 'pass' if statuses and all(s=='pass' for s in statuses) else 'not_evaluated'
    covered=all(r['status']=='pass' for r in continuous['members'].values())
    selected_status='fail' if any(r['assessment']['status']=='fail' for r in results) else 'pass' if selected_complete and covered and results and all(r['assessment']['status']=='pass' for r in results) else 'not_evaluated'
    return dict(method=CHUNK_METHOD,status=status,groups=groups,complete_required_inventory=complete,processed_records=len(seen),
        required_records=len(bykey),selected_member_tags=raw['selected_member_tags'],selected_status=selected_status,
        complete_selected_inventory=selected_complete,saved_claims_verified=True,inverse_solves=0,new_frame_solves=0)


def _member_roster(state):
    tags=sorted(t for gid,g in state.groups.items() if not state.designs[gid].is_column for t in g['member_tags'])
    if not tags or len(tags)!=len(set(tags)):raise ValueError('Distinct complete installed beam inventory required')
    return tags


def _chunk_summary(chunk):
    rows=chunk['records'];whole=chunk['complete_selected_inventory']
    covered=all(r['status']=='pass' for r in chunk['continuous_coverage']['members'].values())
    status='fail' if any(r['assessment']['status']=='fail' for r in rows) else 'pass' if whole and covered and rows and all(r['assessment']['status']=='pass' for r in rows) else 'not_evaluated'
    return dict(selected_member_tags=chunk['selected_member_tags'],selected_status=status,complete_selected_inventory=whole,
        processed_records=len(rows),required_records=chunk['required_record_count'])


def _aggregate_members(state,roster,summaries):
    members={}
    for row in summaries:
        if len(row['selected_member_tags'])!=1:raise ValueError('Exactly one beam per bounded torsion chunk required')
        tag=row['selected_member_tags'][0]
        if tag in members or tag not in roster:raise ValueError('Duplicate or foreign beam torsion chunk')
        members[tag]=row
    if list(members)!=roster[:len(members)]:raise ValueError('Actual torsion chunks must follow the complete fixed member roster')
    groups={}
    for gid,d in state.designs.items():
        if d.is_column:continue
        tags=state.groups[gid]['member_tags'];seen=[members[t] for t in tags if t in members]
        pending=[t for t in tags if t not in members];whole=not pending and all(r['complete_selected_inventory'] for r in seen)
        status='fail' if any(r['selected_status']=='fail' for r in seen) else 'pass' if whole and seen and all(r['selected_status']=='pass' for r in seen) else 'not_evaluated'
        groups[gid]=dict(status=status,complete_required_inventory=whole,required_member_count=len(tags),
            pending_member_tags=pending,processed_records=sum(r['processed_records'] for r in seen),
            required_records_in_enumerated_members=sum(r['required_records'] for r in seen))
    statuses=[g['status'] for g in groups.values()]
    return dict(groups=groups,complete_required_inventory=bool(groups) and all(g['complete_required_inventory'] for g in groups.values()),
        status='fail' if 'fail' in statuses else 'pass' if statuses and all(s=='pass' for s in statuses) else 'not_evaluated')


def assess_torsion_stream(state,common,*,limits,artifact_directory,prepared_sources=None,progress=None,stop_on_proven_failure=True):
    """Bounded per-member inventory; pending members remain explicitly unknown.

    A verified failing beam can reject a candidate without recovering unrelated
    remaining beams. Passing acceptance still requires every original member,
    raw physical case, guard extremum and actual detailing-zone boundary.
    Finite passing reinforcement-reserve samples cannot establish continuous
    adequacy. Currently a whole-interval threshold certificate can close that
    requirement; above-threshold continuous reserve remains explicitly unknown.
    """
    if limits!=common['limits']:raise ValueError('Torsion stream must retain actual common input bounds')
    started=time.monotonic();context=PreparedCommonGuardSources(common) if prepared_sources is None else prepared_sources.validate(common)
    dest=Path(artifact_directory).resolve();dest.mkdir(parents=True,exist_ok=True)
    if any(dest.iterdir()):raise ValueError('Fresh torsion evidence directory required')
    roster=_member_roster(state);out=dict(method=METHOD,state_identity=state.identity(),analysis_input_identity=common['input_signature'],
        material_inventory_id=common['inventory']['identity_sha256'],member_inventory=roster,limits=dict(limits),chunks=[],
        material_cut_count=0,enumerated_record_count=0,stop_reason=None)
    out['source_id']=_hash(dict(method=METHOD,input=common['input_signature'],members=roster))
    summaries=[]
    for tag in roster:
        seconds=limits['strength_wall_seconds']-(time.monotonic()-started)
        remaining=int(limits['max_required_cuts'])-max(out['enumerated_record_count'],out['material_cut_count'])
        if seconds<=0:out['stop_reason']='declared_torsion_wall_bound';break
        if remaining<=0:out['stop_reason']='declared_torsion_cut_bound';break
        chunk_limits=dict(limits,strength_wall_seconds=seconds,max_required_cuts=remaining)
        chunk=_assess_torsion_chunk(state,common,limits=chunk_limits,artifact_directory=dest/f'member_{tag}',
            selected_member_tags=[tag],prepared_sources=context,progress=progress,stop_on_proven_failure=stop_on_proven_failure)
        summary=_chunk_summary(chunk);summaries.append(summary)
        out['chunks'].append(dict(member_tag=tag,summary=summary,reference=_save(dest/f'member_{tag}.json.gz',chunk)))
        out['enumerated_record_count']+=chunk['required_record_count'];out['material_cut_count']+=chunk['material_cut_count']
        if progress:progress(dict(stage='torsion_member_completed',member_tag=tag,enumerated_members=len(summaries),
            required_members=len(roster),status=summary['selected_status'],elapsed_seconds=time.monotonic()-started))
        if chunk['stop_reason']:
            out['stop_reason']=chunk['stop_reason'];break
    out.update(_aggregate_members(state,roster,summaries),elapsed_seconds=time.monotonic()-started,method_qualification_inferred=False)
    out['payload_sha256']=_hash(out);return out


def verify_torsion_stream(saved,common,*,state,prepared_sources=None,beam_capacity=None):
    """Re-enumerate every saved member chunk; reconstruct complete/pending roster."""
    if saved.get('method')!=METHOD or saved.get('payload_sha256')!=_hash({k:v for k,v in saved.items() if k!='payload_sha256'}):raise ValueError('Saved torsion stream changed or unsupported')
    roster=_member_roster(state)
    if (saved['state_identity']!=state.identity() or saved['analysis_input_identity']!=common['input_signature']
        or saved['material_inventory_id']!=common['inventory']['identity_sha256'] or saved['member_inventory']!=roster
        or saved['limits']!=common['limits']
        or saved['source_id']!=_hash(dict(method=METHOD,input=common['input_signature'],members=roster))):raise ValueError('Saved torsion actual input/material/member inventory changed')
    context=PreparedCommonGuardSources(common) if prepared_sources is None else prepared_sources.validate(common)
    summaries=[];required=material=0
    for row in saved['chunks']:
        chunk=read_chunk(row['reference'])
        expected_remaining=int(saved['limits']['max_required_cuts'])-max(required,material)
        if (chunk['limits']['max_required_cuts']!=expected_remaining or chunk['limits']['strength_wall_seconds']>saved['limits']['strength_wall_seconds']
            or any(chunk['limits'][k]!=v for k,v in saved['limits'].items() if k not in ('max_required_cuts','strength_wall_seconds'))):
            raise ValueError('Saved torsion chunk bounds differ from original finite allocation')
        audit=_verify_torsion_chunk(chunk,common,state=state,prepared_sources=context,beam_capacity=beam_capacity)
        summary={k:audit[k] for k in ('selected_member_tags','selected_status','complete_selected_inventory','processed_records','required_records')}
        if summary!=row['summary'] or summary['selected_member_tags']!=[row['member_tag']]:raise ValueError('Saved member torsion summary differs from actual replay')
        summaries.append(summary);required+=audit['required_records'];material+=chunk['material_cut_count']
    result=_aggregate_members(state,roster,summaries)
    if any(result[k]!=saved[k] for k in ('groups','complete_required_inventory','status')):raise ValueError('Saved torsion complete/group claim differs')
    if required!=saved['enumerated_record_count'] or material!=saved['material_cut_count']:raise ValueError('Saved torsion work counts differ')
    return dict(method=METHOD,**result,enumerated_members=len(summaries),required_members=len(roster),
        processed_records=sum(r['processed_records'] for r in summaries),required_records_in_enumerated_members=required,
        saved_claims_verified=True,inverse_solves=0,new_frame_solves=0)
