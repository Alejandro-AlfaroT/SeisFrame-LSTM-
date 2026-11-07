"""Bind ACI torsion evidence to every actual beam/case/cut, without a solve."""
from __future__ import annotations
import hashlib
import json
import math
import Structure_Parameters as sp
from Design.SMRF_Common import make_check,not_evaluated
from Design.SMRF_Beam_Torsion import beam_torsion_section,assess_torsion

ROW_ID='beam.torsion_concurrent'


def actual_member_torsion_section(member,slab_fc_ksi,lambda_factor=1.):
    """9.2.4.4 geometry whose complete Nu must accompany the recovered Tu.

    Coordinates are transverse to the member and from the slab top, with
    zero at the beam centerline and negative depth. This is deliberately
    independent of the6.3.2 effective flange used for flexural strength.
    """
    from Model.Floor_Footprint import physical_floor
    from Model import Member_Groups as mg
    if member.is_column or not isinstance(slab_fc_ksi,(int,float)) or not math.isfinite(slab_fc_ksi) or slab_fc_ksi<=0:
        return {'status':'not_evaluated','reason':'Actual beam and positive slab concrete strength required'}
    fp=physical_floor(member.story_or_floor,mg.active())
    axis=member.member_type[-1]
    center=member.grid_j*sp.BAY_Y if axis=='x' else member.grid_i*sp.BAY_X
    transverse='y' if axis=='x' else 'x'
    b,h=member.design.b_in,member.design.h_in
    available={'minus':center-b/2-fp[transverse+'_min_in'],'plus':fp[transverse+'_max_in']-center-b/2}
    result=beam_torsion_section(b,h,sp.SLAB_THICKNESS_IN,available,min(member.design.fc_ksi,slab_fc_ksi),lambda_factor=lambda_factor)
    result.update(member_tag=member.member_tag,state_identity=mg.active().identity(),
                  global_transverse_origin_in=center,axis=axis,floor=member.story_or_floor)
    result['section_id']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result


def evidence(state,expected_ids,beam_capacity,common_torsion=None,*,common_actions=None):
    """Return one mandatory row per beam group; unknown is never a pass.

    ``expected_cuts_by_member`` is the recovery method's complete physical
    cut inventory, independently enumerated before checking its records.
    Every expected combination is required at every listed cut. Section
    identity binds Nu to this helper's actual9.2.4.4 selected geometry.
    """
    from Design.SMRF_Splice_Physical import group_beam_lanes
    package=common_torsion if isinstance(common_torsion,dict) else {}
    from Design.SMRF_Common_Torsion_Stream import METHOD as STREAM_METHOD
    if package.get('method')==STREAM_METHOD:
        from Design.SMRF_Common_Torsion_Stream import verify_torsion_stream
        checks=[];audit=None;problem=None
        try:
            if common_actions is None or [c['id'] for c in common_actions['combinations']]!=list(expected_ids):
                raise ValueError('Computed torsion needs its exact current common actions and original combination roster')
            if common_actions.get('common_torsion')!=package:raise ValueError('Torsion stream is not the current common-action child')
            audit=verify_torsion_stream(package,common_actions,state=state,beam_capacity=beam_capacity)
        except (ValueError,KeyError,TypeError,RuntimeError,OSError) as exc:problem=str(exc)
        for gid,d in state.designs.items():
            if d.is_column:continue
            group=None if audit is None else audit['groups'].get(gid)
            if group is None or group['status']=='not_evaluated':row=not_evaluated(ROW_ID,'ACI 318-19 9.2.4.4;22.7;9.7.5/9.7.6.3',problem or 'Complete actual torsion reserve/detail coverage remains unresolved',gid)
            else:row=make_check(ROW_ID,'ACI 318-19 9.2.4.4;22.7;9.7.5/9.7.6.3',int(group['status']=='pass'),1,'==',location=gid)
            row['details'].update(torsion=group,source_id=package.get('source_id'));checks.append(row)
        return dict(method='all_actual_beam_case_cut_torsion_v2_common_readback',source_id=package.get('source_id'),state_identity=state.identity(),
            inventory_problems=[] if problem is None else [problem],groups={} if audit is None else audit['groups'],checks=checks,
            status='fail' if any(r['status']=='fail' for r in checks) else 'pass' if checks and all(r['status']=='pass' for r in checks) else 'not_evaluated')
    beam_tags={int(t) for gid,g in state.groups.items() if not state.designs[gid].is_column for t in g['member_tags']}
    cuts=package.get('expected_cuts_by_member') or {}
    problems=[]
    if package.get('state_identity')!=state.identity():problems.append('missing_or_different_state_identity')
    if not package.get('source_id'):problems.append('missing_recovery_source_identity')
    if package.get('cut_inventory_complete') is not True:problems.append('physical_cut_inventory_not_complete')
    if (len(set(expected_ids))!=len(expected_ids) or not isinstance(package.get('expected_case_ids'),list) or
        sorted(package.get('expected_case_ids',[]),key=str)!=sorted(expected_ids,key=str)):problems.append('strength_case_inventory_mismatch')
    analysis_ids=expected_ids
    case_map={}
    if 'analysis_case_ids' in package:
        analysis_ids=package.get('analysis_case_ids');case_map=package.get('analysis_case_original_ids') or {}
        if (not isinstance(analysis_ids,list) or not analysis_ids or any(not isinstance(c,str) or not c for c in analysis_ids)
                or len(analysis_ids)!=len(set(analysis_ids)) or set(case_map)!=set(analysis_ids)
                or any(not isinstance(v,list) or not v or len(v)!=len(set(v)) or not set(v)<=set(expected_ids) for v in case_map.values())
                or set().union(*(set(v) for v in case_map.values()))!=set(expected_ids)):
            problems.append('actual_source_case_mapping_mismatch');analysis_ids=[]
        if package.get('method')!='actual_same_section_torsion_recovery_v1':
            problems.append('unsupported_actual_source_case_method')
        digest=hashlib.sha256(json.dumps({k:v for k,v in package.items() if k!='payload_sha256'},sort_keys=True,
            separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if package.get('payload_sha256')!=digest:problems.append('actual_torsion_payload_changed')
    try:
        cuts={int(t):values for t,values in cuts.items()}
        if set(cuts)!=beam_tags or any(not isinstance(v,list) or not v or any(not isinstance(c,str) or not c for c in v)
                                      or len(v)!=len(set(v)) for v in cuts.values()):
            problems.append('physical_member_cut_inventory_mismatch')
            cuts={t:v for t,v in cuts.items() if isinstance(v,list) and all(isinstance(c,str) and c for c in v)}
    except (TypeError,ValueError):
        cuts={};problems.append('malformed_cut_inventory')
    records={};duplicates=set()
    for record in package.get('records',[]):
        if not isinstance(record,dict):problems.append('malformed_action_record');continue
        try:
            key=(int(record['member_tag']),record['case_id'],record['cut_id'])
            hash(key)
        except (KeyError,TypeError,ValueError):problems.append('malformed_action_identity');continue
        if key in records:duplicates.add(key)
        records[key]=record
    required={(t,c,cut) for t in beam_tags for c in analysis_ids for cut in cuts.get(t,[]) if isinstance(cut,str)}
    if set(records)-required:problems.append('unexpected_member_case_cut')
    result={'method':'all_actual_beam_case_cut_torsion_v1','source_id':package.get('source_id'),
            'state_identity':state.identity(),'inventory_problems':sorted(set(problems)),
            'groups':{},'checks':[]}
    lanes=None
    for gid,group in sorted(state.groups.items()):
        design=state.designs[gid]
        if design.is_column:continue
        evaluated=[];missing=[]
        for tag in group['member_tags']:
            section=actual_member_torsion_section(state.member(tag),package.get('slab_fc_ksi'),package.get('lambda_factor',1.))
            for case_id in analysis_ids:
                for cut_id in cuts.get(int(tag),[]):
                    key=(int(tag),case_id,cut_id);record=records.get(key)
                    identity={'case_id':case_id,'cut_id':cut_id,'state_identity':state.identity(),
                              'recovery_id':(record or {}).get('recovery_id')}
                    local=[]
                    if record is None or key in duplicates:local.append('missing_or_duplicate_action')
                    else:
                        if record.get('status')!='pass':local.append('recovery_not_pass')
                        if record.get('state_identity')!=state.identity() or not record.get('recovery_id'):local.append('recovery_identity_mismatch')
                        if record.get('section_id')!=section.get('section_id') or record.get('same_section_axial_recovered') is not True:
                            local.append('same_torsional_section_axial_not_established')
                        if case_map and record.get('original_case_ids')!=case_map.get(case_id):
                            local.append('actual_source_case_record_mapping_mismatch')
                        if any(not isinstance(record.get(k),(int,float)) or not math.isfinite(record[k]) for k in
                               ('N_tension_kip','T_about_positive_axis_kipin','Q_vertical_kip','V_inplane_kip')):
                            local.append('missing_actual_concurrent_resultants')
                    if local or problems:
                        missing.append({'member_tag':tag,'identity':identity,'problems':local or list(problems)})
                        continue
                    if lanes is None:lanes=group_beam_lanes(state)
                    coordinates=[{**p,'id':f"member_{tag}/beam/{p['id']}"} for p in lanes[gid]['coordinates']]
                    shear=record.get('shear_reservations')
                    if isinstance(shear,dict):
                        shear={axis:dict(values) for axis,values in shear.items() if isinstance(values,dict)}
                        if any(a not in shear or any(not isinstance(shear[a].get(k),(int,float)) or not math.isfinite(shear[a][k])
                            for k in ('capacity_ve_kip','vc_kip','d_in','width_in')) for a in ('vertical','inplane')):
                            shear={}
                        for axis,force in (('vertical','Q_vertical_kip'),('inplane','V_inplane_kip')):
                            if axis in shear:shear[axis]['vu_kip']=record[force]
                        capacity=beam_capacity.get(gid,{})
                        full_ve=max((m['governing_ve_kip'] for m in capacity.get('members',[]) if int(m['member_tag'])==int(tag)),default=None)
                        if 'vertical' in shear and full_ve is not None:
                            shear['vertical']['capacity_ve_kip']=max(abs(shear['vertical'].get('capacity_ve_kip',0.)),full_ve)
                            shear['vertical']['vc_kip']=min(shear['vertical'].get('vc_kip',0.),capacity.get('vc_kip',0.))
                            shear['vertical']['d_in']=capacity['effective_depth_in']
                            shear['vertical']['width_in']=design.b_in
                        else:shear=None
                    assessed=assess_torsion(section,record['T_about_positive_axis_kipin'],record['N_tension_kip'],
                        design=design,coordinates=coordinates,spacing_in=design.stirrup_spacing_in,
                        clear_cover_in=sp.BEAM_CLEAR_COVER_IN,shear_reservations=shear,
                        longitudinal_state=record.get('longitudinal_state'),development=record.get('development'),identity=identity)
                    evaluated.append({'member_tag':tag,'case_id':case_id,'cut_id':cut_id,'section':section,**assessed})
        statuses=[v['status'] for v in evaluated]
        status='fail' if 'fail' in statuses else 'not_evaluated' if problems or missing or not evaluated or 'not_evaluated' in statuses else 'pass'
        data={'status':status,'cases':evaluated,'missing':missing,'inventory_problems':sorted(set(problems))}
        result['groups'][gid]=data
        clause='ACI 318-19 9.2.4.4;9.5.4;9.6.4;9.7.5/9.7.6.3;22.7'
        if status=='not_evaluated':row=not_evaluated(ROW_ID,clause,'Actual concurrent torsion/cut inventory or reinforcement reserve evidence is incomplete',gid)
        else:row=make_check(ROW_ID,clause,int(status=='pass'),1,'==',location=gid)
        row['details'].update(torsion=data,source_id=package.get('source_id'))
        result['checks'].append(row)
    result['status']='fail' if any(c['status']=='fail' for c in result['checks']) else 'pass' if result['checks'] and all(c['status']=='pass' for c in result['checks']) else 'not_evaluated'
    return result
