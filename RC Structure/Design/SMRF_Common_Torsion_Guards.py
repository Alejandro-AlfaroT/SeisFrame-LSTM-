"""Same-section torsion demands and compatible longitudinal reserve witnesses.

Threshold geometry is the exact 9.2.4.4 solid section. Material cuts may
enlarge that width to avoid cutting a bar; their axial force is NEVER used
for the threshold. Prescribed-primary torsion is an additional design state,
not a superposed three-dimensional displacement solution.
"""
from __future__ import annotations
import copy
import math
from Design.SMRF_Coupled_Floor_Design import _hash
from Design.SMRF_Physical_Floor_Cuts import bar_clear_interval
from Design.SMRF_Primary_Moment_Guards import physical_cut_states
from Design.SMRF_Common_Guard_Plan import PreparedCommonGuardSources,build_common_guard_plan,primary_guard_increment
from Design.SMRF_Common_Torsion_Recovery import cut_wrench
from Design.SMRF_Beam_Torsion import threshold

METHOD='exact_torsion_section_all_primary_guards_v1'


def _seed(common,context,spec,raw_case):
    raw=context.sources[spec['source_id']].cut(spec,raw_case)
    states,_=physical_cut_states([raw],outer_interval_in=spec['interval_in'])
    universe=[[b['source_id'],int(f['level']),c['id']] for b in context.bundles
              for f in common['definition']['floors'] for c in b['response']['cases']]
    out=dict(state_identity=common['state_identity'],material_inventory_id=common['inventory']['identity_sha256'],
        definition_sha256=common['definition_sha256'],cut_specs=[spec],cuts={spec['cut_id']:states[0]},
        required_cut_ids=[spec['cut_id']],required_cut_count=1,required_case_ids=[raw_case],
        scope=dict(full_floor_case_universe=universe,full_universe_sha256=_hash(universe),
                   selected_floor_case_pairs=[[spec['source_id'],spec['floor'],raw_case]],complete_required_universe=False))
    out['payload_sha256']=_hash(out)
    return out,raw


def plan_record(state,common,record,section,*,maximum_cuts,prepared_sources=None,accounting_length_in=24.,include_material=True):
    """Every same-cut physical/primary/P5 torsion demand and material state.

P5 changes the slab-plane membrane shear only. Its torque transport is
axis_sign * z_reference * delta_Nxy. Primary beam torque is recovered from
its native force vector at the slab plane, with its own in-plane shear
transported once. For P5 states the external in-plane increment belongs to
P5; it is not added again from the primary beam.
"""
    context=PreparedCommonGuardSources(common) if prepared_sources is None else prepared_sources.validate(common)
    if (record['state_identity']!=state.identity() or record['section_id']!=section['section_id'] or
        record['analysis_input_identity']!=common['input_signature'] or
        record['material_inventory_id']!=common['inventory']['identity_sha256']):
        raise ValueError('Exact current torsion state/section/input/material required')
    member=state.member(record['member_tag']);floor=member.story_or_floor;axis=section['axis'];j=0 if axis=='x' else 1
    desc=next(f for f in common['definition']['floors'] if f['level']==floor);fp=desc['footprint'];cross='y' if axis=='x' else 'x'
    bars=common['inventory']['floors'][str(floor)]['bars']
    families=[dict(db_in=b['db_in'],global_transverse_coordinates_in=[b['s_in']]) for b in bars if b['axis']==axis]
    transverse=[dict(db_in=b['db_in'],global_transverse_coordinates_in=[b['s_in']]) for b in bars if b['axis']!=axis and b['operator_owner']=='slab']
    station=record['station_in'];alo,ahi=fp[axis+'_min_in'],fp[axis+'_max_in']
    account=bar_clear_interval([max(alo,station-accounting_length_in/2),min(ahi,station+accounting_length_in/2)],[alo,ahi],transverse)
    exact=list(record['interval_in']);material=bar_clear_interval(exact,[fp[cross+'_min_in'],fp[cross+'_max_in']],families)
    key=record['case_id']+'/'+record['cut_id'];raw_case=record['raw_case_id']
    common_spec=dict(floor=floor,axis=axis,station_in=station,axial_accounting_interval_in=account,
        member_id=str(record['member_tag']),trace=record['trace'],source_id=record['source_id'],raw_case_id=raw_case,
        analysis_case_id=record['case_id'],torsion_cut_id=record['cut_id'])
    plans={};raws={}
    for name,width in [('exact_threshold',exact)]+([('material',material)] if include_material else []):
        spec=dict(common_spec,cut_id=key+'|'+name,interval_in=width,physical_region_kind='torsion_'+name)
        seed,raw=_seed(common,context,spec,raw_case)
        plans[name]=build_common_guard_plan(common,seed,common['p5'],state=state,maximum_cuts=maximum_cuts,
            allow_partial_plan=True,prepared_sources=context);raws[name]=raw
    exact_plan=plans['exact_threshold'];raw=raws['exact_threshold'];zr=record['reference_z_in'];sr=record['reference_s_in']
    physical=cut_wrench(raw,reference_s_in=sr,reference_z_in=zr)
    for name in ('N_tension_kip','T_about_positive_axis_kipin','Q_vertical_kip','V_inplane_kip'):
        if not math.isclose(physical[name],record[name],rel_tol=1e-9,abs_tol=1e-8):raise ValueError('Torsion record differs from native exact-section recovery')
    seismic=next((c for c in common['combinations'] if c['id']==raw_case and (c['ex'] or c['ey'])),None)
    primary=None;gravity=None;sign=1. if axis=='x' else -1.
    if seismic:
        primary=primary_guard_increment(common,raw_case,record['member_tag'],station)['actual_primary_cut']
        gid=common['gravity_case_mapping'][raw_case]
        grow=context.sources['actual_frozen_gravity'].cut(exact_plan['cut_specs'][0],gid)
        gravity=cut_wrench(grow,reference_s_in=sr,reference_z_in=zr)
    demands=[]
    for spec in exact_plan['cut_specs']:
        # Remote effective-flange strips own no web and cannot supply Nu for
        # this 9.2.4.4 section. Its complete core is retained in every branch.
        if spec.get('member_id')!=str(record['member_tag']) or spec['interval_in']!=exact:continue
        action=exact_plan['cuts'][spec['cut_id']];n=action['resultants'][j]
        dv=action['resultants'][2]-raw['resultants'][2]
        variants=[('physical_T_with_same_guard_membrane',physical['T_about_positive_axis_kipin']+sign*zr*dv,
                   physical['Q_vertical_kip'],physical['V_inplane_kip']+dv)]
        if primary is not None:
            branch=spec.get('guard_branch_id','')
            if branch.startswith('P5'):
                # Replace the entire physical external membrane increment;
                # retain the exact gravity web shear and P5 slab-plane shear.
                increment_v=action['resultants'][2]-grow['resultants'][2]
                v=gravity['V_inplane_kip']+increment_v
            else:
                increment_v=primary['V_inplane_kip'];v=gravity['V_inplane_kip']+increment_v
            t=gravity['T_about_positive_axis_kipin']+primary['T_about_positive_axis_kipin']+sign*zr*increment_v
            variants.append(('gravity_plus_primary_T',t,gravity['Q_vertical_kip']+primary['Q_vertical_kip'],v))
        for kind,t,q,v in variants:
            demands.append(dict(guard_cut_id=spec['cut_id'],kind=kind,N_tension_kip=float(n),T_about_positive_axis_kipin=float(t),
                Q_vertical_kip=float(q),V_inplane_kip=float(v),threshold=threshold(section,t,n)))
    if not demands:raise ValueError('No complete same-section torsion guard core')
    out=dict(method=METHOD,record_identity={k:record[k] for k in ('case_id','cut_id','state_identity','recovery_id')},
        exact_torsion_interval_in=exact,material_interval_in=material,section_id=section['section_id'],demands=demands,
        exact_threshold_plan=exact_plan,material_plan=plans.get('material'),all_demands_below_threshold=all(d['threshold']['below_threshold'] for d in demands),
        physical_displacement_superposition_claimed=False,complete_design_qualified=False)
    out['payload_sha256']=_hash(out);return out


def longitudinal_reserve(assessment,plan,record,beam_bar_ids):
    """Verified componentwise envelope of actual compatible core bar stresses.

The envelope is a conservative reservation, not a new equilibrium state.
Every required material state must pass; missing/undeveloped bars cannot
be converted into unused torsion capacity.
"""
    from Design.SMRF_Physical_Floor_Strength import verify_physical_cut_assessment
    names=plan['required_cut_ids']
    if assessment['required_cut_ids']!=names or assessment['replay_payload']['cut_specs']!=plan['cut_specs'] or assessment['replay_payload']['cuts']!=plan['cuts']:
        raise ValueError('Torsion longitudinal assessment does not cover its exact guard plan')
    audit=verify_physical_cut_assessment(assessment)
    out=dict(status='not_evaluated',**{k:record[k] for k in ('case_id','cut_id','state_identity','recovery_id')},
        all_concurrent_demands_included=False,bars=[],required_guard_cut_ids=names,
        verified_material_status=audit['status'],reservation_basis='per_bar_maximum_absolute_nominal_stress_over_all_required_compatible_guards')
    if audit['status']!='pass':return out
    required=set(beam_bar_ids);envelope={};sources=[];phis=set()
    for row in assessment['records']:
        spec=next(s for s in plan['cut_specs'] if s['cut_id']==row['cut_id'])
        if spec.get('member_id')!=str(record['member_tag']):continue
        longitudinal=(row.get('section_assessment') or {}).get('longitudinal_state') or {}
        if longitudinal.get('status')!='pass' or longitudinal.get('all_concurrent_demands_included') is not True:return out
        bars={b['id']:b for b in longitudinal.get('bars',[])}
        if not required<=set(bars) or any(bars[k].get('strength_credited') is not True for k in required):return out
        phis.add(longitudinal['strength_phi']);sources.append(row['cut_id'])
        for key in required:
            b=bars[key]
            if key in envelope and any(envelope[key][k]!=b[k] for k in ('area_in2','fy_ksi')):raise ValueError('Same physical torsion bar changes material between guards')
            if key not in envelope or abs(b['nominal_stress_ksi'])>abs(envelope[key]['nominal_stress_ksi']):envelope[key]=copy.deepcopy(b)
    if set(envelope)!=required or not required or len(phis)!=1:return out
    out.update(status='pass',all_concurrent_demands_included=True,bars=[envelope[k] for k in sorted(required)],
        strength_phi=next(iter(phis)),source_guard_cut_ids=sources)
    out['evidence_id']=_hash(out);return out
