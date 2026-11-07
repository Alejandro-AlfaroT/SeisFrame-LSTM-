"""Source-bound common physical-gravity, primary-ELF and floor-action producer.

One invocation belongs to ONE installed cage/mat/section state. Any change
requires another invocation before final acceptance. Numerical responses do
not activate a profile or establish the seven method dispositions. Complete
common strength, prescribed-primary diaphragm guards and their independent
method evidence are separately replayed by the acceptance contract.
"""
from __future__ import annotations
import hashlib
import gzip
import json
import math
from pathlib import Path
import time
import numpy as np
import Structure_Parameters as sp
from Model import Member_Groups as mg,Member_Properties as mp
from Design.SMRF_Coupled_Floor_Design import CoupledFloorSystem,installed_definition,_hash
from Design.SMRF_Physical_Floor_Materials import installed_inventory
from Design.SMRF_Frozen_Gravity_Design import frozen_gravity_family
from Design.SMRF_Primary_Frame_Transfer import PrimaryFrame,installed_primary_definition,transferred_floor_influences
from Design.SMRF_Physical_Diaphragm import installed_grid_masses
from Design.SMRF_Primary_Demands import floor_loads,torsion_assessment,drift_screen
from Design.SMRF_Physical_Seismic_Design import PhysicalSeismicSystem
from Design.SMRF_Physical_Member_Actions import build_member_actions,PreparedMemberResponses
from Design.SMRF_Story_Adapter import collect_physical_gravity_cases
from Design.SMRF_Demands import live_load_patterns,strength_load_combinations,seismic_design_category
from Loads.Seismic_ELF import elf_story_forces
from Design.SMRF_Common_Strength_Stream import LEGACY_BASIS,_local_basis

METHOD='coordinated_actual_common_floor_member_actions_v6_continuous_strength_required'
from Design.SMRF_Common_Action_Procedure import LIMIT_KEYS
_LIMITS=tuple(sorted(LIMIT_KEYS))


def current_source_identity():
    """Actual production Python bytes; tests and output directories excluded."""
    root=Path(__file__).resolve().parents[1];out={}
    for p in sorted(root.rglob('*.py')):
        rel=p.relative_to(root)
        if any(v in ('tests','outputs','pilots','__pycache__') for v in rel.parts):continue
        out[rel.as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _gravity_cases(combinations):
    service=dict(id='service_DL',dead=1.,live=1.,live_pattern='all',ex=0.,ey=0.)
    cases=[service];bykey={(1.,1.,'all'):service['id']};mapping={}
    for case in combinations:
        key=(float(case['dead']),float(case['live']),case.get('live_pattern','all'))
        if key not in bykey:
            name='G_'+_hash(key)[:20];bykey[key]=name
            cases.append(dict(id=name,dead=key[0],live=key[1],live_pattern=key[2],ex=0.,ey=0.))
        mapping[case['id']]=bykey[key]
    return cases,mapping


def common_input_identity(state_identity,definition,inventory,source,limits,combinations,model_period_sec,demand_parameters,elf,mass,*,material_mapping='clipped_cylinders_v2',strength_material_action_basis=LEGACY_BASIS):
    _local_basis(strength_material_action_basis,dict(floor_material_mapping=material_mapping))
    return _hash(dict(method=METHOD,state=state_identity,definition=definition['identity_sha256'],
        inventory=inventory['identity_sha256'],source=source,limits=limits,combinations=combinations,
        model_period_sec=model_period_sec,demand_parameters=demand_parameters,elf=elf,grid_masses=mass,material_mapping=material_mapping,
        strength_material_action_basis=strength_material_action_basis))


def verify_common_input_identity(common):
    """Bind the explicit analysis and strength choices before consuming fields."""
    if common.get('method')!=METHOD:raise ValueError('Current common action producer method required')
    if 'method_procedure' in common:
        from Design.SMRF_Common_Action_Procedure import material_basis_from_protocol
        declared=material_basis_from_protocol(common['method_procedure']['procedure']['protocol'])
        actual=dict(material_mapping=common['floor_material_mapping'],strength_material_action_basis=common['strength_material_action_basis'])
        if declared!=actual:raise ValueError('Common material choices differ from the method-proof protocol')
    if _hash(common['source_identity'])!=common['analysis_source_identity']:raise ValueError('Common source identity differs')
    expected=common_input_identity(common['state_identity'],common['definition'],common['inventory'],
        common['analysis_source_identity'],common['limits'],common['combinations'],common['model_period_sec'],
        common['demand_parameters'],common['elf'],common['grid_masses'],material_mapping=common['floor_material_mapping'],
        strength_material_action_basis=common['strength_material_action_basis'])
    if expected!=common['input_signature']:raise ValueError('Common input identity does not bind actual analysis and strength material choices')
    return common['strength_material_action_basis']


def compatibility_cases(primary,drift_responses,gravity_case_ids,*,cd,importance_factor):
    """Exact original signed Cd/Ie targets from the saved/solved primary drift."""
    cases=[]
    for gid in gravity_case_ids:
        for key,row in drift_responses.items():
            response=primary.influence_response(row['response']);amplification=cd/importance_factor
            for sign in (1.,-1.):
                cases.append(dict(id=f'compat:{gid}:{key}:{sign:+.0f}',gravity_case_id=gid,sign=sign,axis=key[0],
                    reference_xy_in=primary.master_xy,levels=[dict(floor=k,design_in=dict(
                        ux=amplification*u[0],uy=amplification*u[1],rz_rad=amplification*u[2]))
                        for k,u in enumerate(response['master_displacements'],1)]))
    return cases


def physical_response_bundles(common):
    """Exact material/input-bound primitives for complete cut recovery."""
    output=[]
    def add(name,response):
        output.append(dict(source_id=name,response=response,
            material_inventory_id=common['inventory']['identity_sha256'],
            analysis_input_identity=common['input_signature'],response_payload_sha256=_hash(response)))
    add('actual_frozen_gravity',common['gravity']['design_response'])
    for index,batch in enumerate(common['physical_branches']):
        for gid,row in batch['groups'].items():add(f'physical_ELF/{index}/{gid}',row['total_response'])
    add('actual_Cd_compatibility',common['compatibility']['response'])
    return output


def analyze_installed(state,cfg,*,source_identity,model_period_sec,limits,progress=None,artifact_directory=None,final_assessment=True,
                      material_mapping='clipped_cylinders_v2',strength_material_action_basis=LEGACY_BASIS):
    """Compute explicit bounded actions for the exact installed physical state.

    Limits are supplied and recorded, not copied from a launch budget. The
    original strength combination IDs and live patterns are preserved. This
    producer currently returns all primitive response evidence needed by the
    shared-strength/method evaluator; its result is never code acceptance.
    """
    started=time.monotonic()
    _local_basis(strength_material_action_basis,dict(floor_material_mapping=material_mapping))
    if type(model_period_sec) not in (int,float) or not math.isfinite(model_period_sec) or model_period_sec<=0:
        raise ValueError('Explicit finite positive scalar model period required before physical analysis')
    if mg.active() is None or mg.active().identity()!=state.identity():raise ValueError('Exact candidate must be installed')
    if not isinstance(source_identity,dict) or not source_identity:raise ValueError('Actual path-to-SHA production source map required')
    if set(limits)!=set(_LIMITS) or any(not isinstance(limits[k],(int,float)) or limits[k]<0 for k in _LIMITS):
        raise ValueError('Complete explicit bounded common-analysis limits required')
    if limits['max_case_count']<1 or limits['physical_case_batch_size']<1:raise ValueError('Positive case bounds required')
    actual_source=current_source_identity()
    if source_identity!=actual_source:raise ValueError('Source map differs from current production Python bytes')
    source=_hash(source_identity);definition=installed_definition(state)
    def report(stage,**data):
        if progress:progress(dict(stage=stage,elapsed_seconds=time.monotonic()-started,**data))
    report('actual_common_inventory')
    inventory=installed_inventory(state)
    patterns=live_load_patterns(sp.NUM_BAY_X,sp.NUM_BAY_Y) if cfg.demands.live_load_patterning else []
    combinations=strength_load_combinations(sp.ASCE_SDS,redundancy_factor=1.3,live_patterns=patterns)
    if len(combinations)>limits['max_case_count']:raise ValueError('Original combination inventory exceeds the declared bound')
    gcases,gmap=_gravity_cases(combinations);bygravity={c['id']:c for c in gcases}
    parameters=dict(accidental_ratio=cfg.demands.accidental_torsion_ratio,risk_category=cfg.demands.risk_category,
        cd=sp.ASCE_CD,importance_factor=sp.ASCE_IE,redundancy_factor=1.3,
        seismic_live_load_fraction=sp.SEISMIC_LIVE_LOAD_FRACTION,live_load_patterning=cfg.demands.live_load_patterning)
    mass=installed_grid_masses(definition);elf=elf_story_forces(model_period_sec)
    identity=common_input_identity(state.identity(),definition,inventory,source,limits,combinations,model_period_sec,parameters,elf,mass,material_mapping=material_mapping,
        strength_material_action_basis=strength_material_action_basis)
    common=dict(method=METHOD,state_identity=state.identity(),definition_sha256=definition['identity_sha256'],
        source_identity=source_identity,analysis_source_identity=source,input_signature=identity,
        definition=definition,inventory=inventory,limits=dict(limits),combinations=combinations,floor_material_mapping=material_mapping,
        strength_material_action_basis=strength_material_action_basis,
        gravity_case_mapping=gmap,qualification_asserted=False,official_pilot_started=False,
        method_dispositions={},common_strength=None,demand_parameters=parameters,grid_masses=mass,elf=elf,model_period_sec=model_period_sec)
    def checkpoint(name,value):
        if artifact_directory is None:return
        destination=Path(artifact_directory);destination.mkdir(parents=True,exist_ok=True)
        path=destination/(name+'.json.gz')
        if path.exists():raise ValueError('Physical checkpoint path must be fresh; no prior evidence overwritten')
        data=dict(input_signature=identity,source_identity=source,state_identity=state.identity(),
            definition_sha256=definition['identity_sha256'],material_inventory_id=inventory['identity_sha256'],value=value)
        data['payload_sha256']=_hash(data)
        with gzip.open(path,'wt',encoding='utf-8') as stream:json.dump(data,stream,allow_nan=False)
    checkpoint('installed_common_inputs',common)
    mesh=dict(subdivisions_per_bay=int(limits['mesh_per_bay']),max_shells_per_floor=int(limits['max_shells_per_floor']),
              max_coupled_dofs=int(limits['max_coupled_dofs']))
    predictor=CoupledFloorSystem(definition,[gcases[0]],source_identity=source,
        stiffness_basis='Gross own-slab predictor; beam.35; column.7; equilibrium GJ retained',
        beam_bending_factor=.35,slab_bending_factor=1.,web_torsion_factor=1.,column_bending_factor=.7,**mesh)
    frozen,gravity_system=frozen_gravity_family(predictor,inventory,gcases,protocol_id=identity,
        reference_updates=int(limits['reference_updates']),maximum_wall_seconds=limits['reference_wall_seconds'],
        maximum_coupled_dofs=int(limits['max_coupled_dofs']),patterns=patterns,progress=progress,material_mapping=material_mapping)
    common['gravity']=frozen
    checkpoint('actual_frozen_gravity',frozen)
    if gravity_system is None:
        common.update(status='unknown',reason=frozen.get('reason'),elapsed_seconds=time.monotonic()-started)
        common['payload_sha256']=_hash(common);return common
    gravity=frozen['design_response']
    primaries={};primary_records={}
    def primary_for(case_id):
        if case_id not in primaries:
            d=installed_primary_definition(definition,bygravity[case_id]);p=PrimaryFrame(d,source_identity=source)
            transfer=p.transfer_gravity(gravity,case_id=case_id);primaries[case_id]=p
            primary_records[case_id]=dict(definition=d,gravity_transfer=transfer,increments={})
        return primaries[case_id]
    service=primary_for('service_DL')
    # This provisional distribution is never an assertion of story capacity.
    # The actual final connected-frame strength is added by the caller.
    report('actual_P_primary_torsion')
    torsion,torsion_responses=torsion_assessment(service,definition,elf,mass,
        accidental_ratio=cfg.demands.accidental_torsion_ratio,strength_distribution={'one_side_fraction':None})
    risk=cfg.demands.risk_category;sdc=seismic_design_category(sp.ASCE_SDS,sp.ASCE_SD1,sp.ASCE_S1,risk)
    basis=sp.seismic_design_basis(risk_category=risk,redundancy_factor=1.3,
        seismic_design_category=sdc if sdc in ('D','E','F') else 'D')
    drift,drift_responses=drift_screen(service,definition,elf,mass,accidental_ratio=cfg.demands.accidental_torsion_ratio,
        amplification=torsion['amplification'],basis=basis,sdc=sdc,cd=sp.ASCE_CD,
        gravity_above_by_story={k:mp.gravity_above_kip(k) for k in range(1,sp.NUM_FLOOR+1)})
    common.update(torsion=torsion,torsion_responses=torsion_responses,drift_screen=drift,drift_responses=drift_responses)
    adapted=[];physical_cases=[];witnesses={};member_responses=PreparedMemberResponses(gravity)
    for combination in combinations:
        gid=gmap[combination['id']];active=bool(combination['ex'] or combination['ey']);p=None;increment=None
        if active:
            p=primary_for(gid);load=floor_loads(definition,elf,mass,ex=combination['ex'],ey=combination['ey'],
                accidental_ratio=cfg.demands.accidental_torsion_ratio,amplification=torsion['amplification'])
            increment=p.solve_increment(load['loads']);primary_records[gid]['increments'][combination['id']]=increment
            physical_cases.append(dict(id=combination['id'],gravity_case_id=gid,**load))
            witnesses.setdefault(gid,[]).append(dict(id=combination['id'],loads=load['loads'],
                strength_factor=max(abs(combination['ex']),abs(combination['ey']))))
        report('actual_common_member_fields',case_id=combination['id'])
        adapted.append(build_member_actions(state,definition,gravity,combination,gravity_case_id=gid,
            primary_definition=p.definition if p else None,primary_increment=increment,
            primary_transfer=primary_records[gid]['gravity_transfer'] if p else None,prepared_responses=member_responses))
    common.update(actions=[r['action'] for r in adapted],primary=primary_records)
    common['story_strength_gravity_actions']=collect_physical_gravity_cases(adapted,[c['id'] for c in combinations])
    checkpoint('actual_primary_member_actions',dict(primary=primary_records,actions=common['actions'],
        story_strength_gravity_actions=common['story_strength_gravity_actions'],torsion=torsion,drift_screen=drift))
    report('complete_physical_ELF_branches')
    physical=PhysicalSeismicSystem(definition,gcases,source_identity=source,
        seismic_live_load_fraction=sp.SEISMIC_LIVE_LOAD_FRACTION,max_cases=int(limits['max_case_count']),patterns=patterns,**mesh)
    batches=[];batch=int(limits['physical_case_batch_size'])
    for first in range(0,len(physical_cases),batch):
        batches.append(physical.solve(physical_cases[first:first+batch],input_identity=identity))
    common['physical_branches']=batches
    checkpoint('complete_physical_ELF_branches',batches)
    compatibility=compatibility_cases(service,drift_responses,witnesses,cd=sp.ASCE_CD,importance_factor=sp.ASCE_IE)
    common['compatibility']=physical.compatibility(compatibility,input_identity=identity)
    from Design.SMRF_Common_Compatibility import adapt_compatibility
    common['compatibility_actions']=adapt_compatibility(state,definition,common['compatibility'],
        cd=sp.ASCE_CD,importance_factor=sp.ASCE_IE)
    checkpoint('complete_physical_compatibility',common['compatibility'])
    gactions=common['story_strength_gravity_actions']
    common['capacity_gravity_transfer']=dict(physical_gravity_member_actions=gactions,
        expected_physical_gravity_case_ids=[c['id'] for c in gactions],state_identity=state.identity(),
        source_identity=source,physical_definition_sha256=definition['identity_sha256'])
    report('prescribed_primary_diaphragm_influence_bases')
    common['p5_influences']={}
    for gid,rows in witnesses.items():
        p=primary_for(gid)
        common['p5_influences'][gid]=transferred_floor_influences(p,definition,mass,witness_cases=rows,
            protocol_id=identity,max_direct_solves=12*sp.NUM_FLOOR+2*len(rows),
            max_wall_seconds=limits['primary_wall_seconds'],force_amplitude_kip=1.,torque_amplitude_kipin=1000.)
    checkpoint('prescribed_primary_influence_bases',common['p5_influences'])
    report('prescribed_primary_diaphragm_signed_fields')
    from Design.SMRF_Primary_Diaphragm_Batch import analyze_primary_diaphragms
    common['p5']=analyze_primary_diaphragms(common,sds=sp.ASCE_SDS,importance_factor=sp.ASCE_IE,
        omega0=3.,seismic_live_load_fraction=sp.SEISMIC_LIVE_LOAD_FRACTION,
        subdivisions_per_bay=int(limits['mesh_per_bay']),maximum_nodes_per_floor=int(limits['max_shells_per_floor'])*4,
        demand_factors_by_floor={k:dict(transfer_factor=1.,irregularity_factor=1.25) for k in range(1,sp.NUM_FLOOR+1)},
        max_witness_floor_pairs=sp.NUM_FLOOR*len(physical_cases),
        max_wall_seconds=limits['primary_wall_seconds']*sp.NUM_FLOOR,progress=progress,
        artifact_directory=None if artifact_directory is None else Path(artifact_directory)/'primary_diaphragms')
    common['p5_factor_basis']='Retained exact transfer components; conservative1.25 diaphragm irregularity factor, Omega0 collector cases; no strength-classification reduction'
    common['method_procedure']=cfg.floor_analysis.common_action_procedure
    common['p5_parameters']=dict(sds=sp.ASCE_SDS,importance_factor=sp.ASCE_IE,omega0=3.,
        seismic_live_load_fraction=sp.SEISMIC_LIVE_LOAD_FRACTION,
        demand_factors_by_floor={k:dict(transfer_factor=1.,irregularity_factor=1.25) for k in range(1,sp.NUM_FLOOR+1)})
    common.update(status='responses_computed',elapsed_seconds=time.monotonic()-started,
        primary_policy_retained=True,source_unchanged=current_source_identity()==source_identity,
        strength_assessment_performed=False,mesh_qualification_inferred=False,complete_design_qualified=False)
    common['payload_sha256']=_hash(common)
    return assess_installed(state,common,progress=progress,artifact_directory=artifact_directory) if final_assessment else common


def assess_installed(state,common,*,progress=None,artifact_directory=None):
    """Assess one unchanged installed action state after cage coordination.

    Reuses only exactly current force fields. No geometry, mat, source or
    input identity can change between action generation and this stage.
    All final strength streams remain independently replayable.
    """
    if common['payload_sha256']!=_hash({k:v for k,v in common.items() if k!='payload_sha256'}):raise ValueError('Actual common action payload changed before final assessment')
    strength_basis=verify_common_input_identity(common)
    if (state.identity()!=common['state_identity'] or installed_definition(state)!=common['definition'] or
        installed_inventory(state)!=common['inventory'] or current_source_identity()!=common['source_identity']):
        raise ValueError('Final strength assessment requires the unchanged installed state/material/source')
    common=dict(common);common.pop('payload_sha256');started=time.monotonic()
    definition=common['definition'];inventory=common['inventory'];identity=common['input_signature'];limits=common['limits']
    def report(stage,**data):
        if progress:progress(dict(stage=stage,elapsed_seconds=time.monotonic()-started,**data))
    if artifact_directory is None:
        import uuid
        artifact_directory=Path(__file__).resolve().parents[1]/'outputs'/'common_actions'/(identity+'_'+uuid.uuid4().hex)
    artifact_directory=Path(artifact_directory)
    from Design.SMRF_Common_Guard_Plan import PreparedCommonGuardSources
    prepared=PreparedCommonGuardSources(common)
    report('actual_local_support_shear_and_steel')
    from Design.SMRF_Common_Local_Checks import assess_local_checks
    common['local_checks']=assess_local_checks(state,definition,inventory,prepared.bundles,
        assessment_identity=identity,slab_fc_ksi=min(f['fc_ksi'] for f in definition['floors']),slab_fy_ksi=sp.SLAB_FY_KSI)
    report('complete_adjacent_corner_panel_envelopes')
    from Design.SMRF_Common_Corner_Demands import corner_demands,primary_corner_envelopes
    common['slab_corner_demands']=corner_demands(common,guard_envelopes=primary_corner_envelopes(state,common,prepared_sources=prepared))
    report('complete_common_material_strength_stream')
    from Design.SMRF_Common_Strength_Stream import assess_common_strength
    strength_directory=artifact_directory/'strength'
    common['common_strength']=assess_common_strength(state,common,limits=limits,artifact_directory=strength_directory,progress=progress,prepared_sources=prepared,
        material_action_basis=strength_basis)
    report('actual_concurrent_local_flexural_transfer_and_one_way_shear')
    from Design.SMRF_Common_Local_Stream import assess_local_strength
    common['local_strength']=assess_local_strength(state,common,limits=limits,artifact_directory=artifact_directory/'local_strength',progress=progress,prepared_sources=prepared,
        material_action_basis=strength_basis)
    report('actual_torsion_primary_P5_and_full_Mpr_reserve')
    from Design.SMRF_Common_Torsion_Stream import assess_torsion_stream
    common['common_torsion']=assess_torsion_stream(state,common,limits=limits,artifact_directory=artifact_directory/'torsion',progress=progress,prepared_sources=prepared)
    common.update(strength_assessment_performed=True,strength_assessment_elapsed_seconds=time.monotonic()-started,
        source_unchanged=current_source_identity()==common['source_identity'])
    common['payload_sha256']=_hash(common)
    return common


def replay_assessment(state,package):
    """Reconstruct thirteen common-action rows; absent calculation stays unknown.

    No saved check status activates a row. Each implemented reader verifies
    actual raw operators/fields or constitutive witnesses under current bytes.
    Method proofs and per-candidate checks remain distinct requirements.
    """
    from Design.SMRF_Common_Action_Contract import ROW_IDS,CLAUSE,validate_package
    from Design.SMRF_Common import make_check,not_evaluated
    from Design.SMRF_Common_Physics_Replay import replay_physics
    from Design.SMRF_Primary_Diaphragm_Batch import replay_primary_diaphragms
    from Design.SMRF_Common_Strength_Stream import verify_strength_stream,read_chunk
    from Design.SMRF_Common_Local_Checks import replay_local_checks
    validate_package(state.identity(),package)
    rows={name:not_evaluated(name,CLAUSE,'Required actual computed evidence has not been supplied.') for name in ROW_IDS};audits={}
    prepared=None
    def sources():
        nonlocal prepared
        from Design.SMRF_Common_Guard_Plan import PreparedCommonGuardSources
        if prepared is None:prepared=PreparedCommonGuardSources(package)
        return prepared
    def unknown(name,exc):rows[name]=not_evaluated(name,CLAUSE,str(exc))
    def count_check(name,failed,details):rows[name]=make_check(name,CLAUSE,failed,0,'<=','failed required checks',details=details)
    try:
        audits['physics']=replay_physics(state,package)
        count_check('common.analysis_and_case_coverage',0,audits['physics'])
    except (ValueError,KeyError,TypeError,RuntimeError) as exc:unknown('common.analysis_and_case_coverage',exc)
    try:
        parameters=dict(package['p5_parameters']);parameters['demand_factors_by_floor']={int(k):v for k,v in parameters['demand_factors_by_floor'].items()}
        audits['p5']=replay_primary_diaphragms(package['p5'],package,**parameters)
        count_check('common.p5_signed_load_path',0,dict(force_cases=audits['p5']['force_cases_checked'],
            maximum_normalized_nodal_residual=audits['p5']['maximum_normalized_nodal_residual'],factorizations=audits['p5']['factorizations']))
    except (ValueError,KeyError,TypeError,RuntimeError) as exc:unknown('common.p5_signed_load_path',exc)
    try:
        audits['strength']=verify_strength_stream(package['common_strength'],package,state=state,prepared_sources=sources())
        strength=audits['strength'];name='common.shared_steel_concurrent_strength'
        if strength['status']=='unknown' or (strength['status']=='pass' and strength.get('continuous_capacity_verified') is not True):
            unknown(name,'Actual common material cuts or continuous capacity remain unverified: '+str(strength))
        else:count_check(name,int(strength['status']=='fail'),strength)
        pending=0;development_count=0
        for chunk in package['common_strength']['chunks']:
            saved=read_chunk(chunk)
            for r in saved['records']:pending+=int(r['development']['status']!='pass');development_count+=1
        if not strength['complete_required_inventory'] or pending:
            unknown('common.actual_bar_development',f'Complete developed full-fy strength domains not established: {pending} pending, {development_count} evaluated cuts')
        else:count_check('common.actual_bar_development',0,dict(actual_development_cuts=development_count))
        if strength['complete_required_inventory'] and rows['common.p5_signed_load_path']['status']=='pass':
            if strength['status']=='unknown' or (strength['status']=='pass' and strength.get('continuous_capacity_verified') is not True):
                unknown('common.diaphragm_collectors_chords','Concurrent actual P5/primary guard continuous capacities remain unresolved')
            else:count_check('common.diaphragm_collectors_chords',int(strength['status']=='fail'),strength)
    except (ValueError,KeyError,TypeError,RuntimeError,OSError) as exc:
        for name in ('common.shared_steel_concurrent_strength','common.actual_bar_development','common.diaphragm_collectors_chords'):unknown(name,exc)
    # Local primitive recovery and constitutive evidence remain necessary
    # even when the independent method/application witnesses have passed.
    try:
        local=package['local_checks'];payload=local['replay_payload']
        if _hash(payload['response_bundles'])!=_hash(physical_response_bundles(package)) or payload['inventory']!=package['inventory']:
            raise ValueError('Local actions do not bind the exact current physical source/material inventory')
        rebuilt=replay_local_checks(local)
        if _hash(rebuilt)!=_hash(local):raise ValueError('Saved local support checks differ from actual raw calculation')
        audits['local']=dict(records=len(rebuilt['records']),shear_failures=sum(r['shear']['status']=='fail' for r in rebuilt['records']),
            minimum_failures=sum(r['local_minimum']['status']=='fail' for r in rebuilt['records']),
            minimum_unknowns=sum(r['local_minimum']['status']=='unknown' for r in rebuilt['records']))
        from Design.SMRF_Common_Local_Stream import verify_local_strength
        audits['local_strength']=verify_local_strength(package['local_strength'],package,state=state,prepared_sources=sources())
        audits['local']['local_flexural_transfer_complete']=audits['local_strength']['complete_required_inventory']
        from Design.SMRF_Common_Corner_Demands import corner_demands,primary_corner_envelopes
        corners=corner_demands(package,guard_envelopes=primary_corner_envelopes(state,package,prepared_sources=sources()))
        if corners!=package['slab_corner_demands']:raise ValueError('Saved corner panel envelope differs from reconstructed physical/primary fields')
        audits['corner']=dict(verified=True,records=len(corners['records']))
    except (ValueError,KeyError,TypeError,RuntimeError) as exc:audits['local']=dict(pending=str(exc))
    procedure=package.get('method_procedure') or {}
    try:
        from Design.SMRF_Common_Action_Procedure import assessment
        method=assessment(procedure['procedure'],procedure['evidence_base_dir']);audits['method']=method
        for name in ROW_IDS:
            if not name.startswith('common.disposition.'):continue
            proofs=method['proof_checks'].get(name,[])
            if not proofs or method['method_problems'] or method['pending_tolerance_protocol']:
                unknown(name,'Current-source numeric method/domain/tolerance proof remains incomplete')
            else:count_check(name,sum(p['status']!='pass' for p in proofs),dict(proofs=proofs))
        # Same-case local checks and complete global physics stay necessary
        # even after independent applicability examples have passed.
        local=audits.get('local',{})
        name='common.disposition.local_slab_column_steel'
        strength=audits.get('local_strength',{})
        if (not local.get('local_flexural_transfer_complete') or strength.get('status')=='unknown' or
            (strength.get('status')=='pass' and strength.get('continuous_capacity_verified') is not True and
             not local.get('minimum_failures')) or
            local.get('minimum_unknowns') or not audits.get('corner',{}).get('verified')):
            unknown(name,'Actual local continuous capacity, gamma_f transfer, development, one-way shear or corner envelope remains incomplete')
        elif rows[name]['status']=='pass':
            count_check(name,local['minimum_failures']+int(strength['status']=='fail'),dict(local=local,strength=strength,method_proofs=method['proof_checks'].get(name)))
        shear_name='common.disposition.two_way_shear_path'
        if rows[shear_name]['status']=='pass':
            if 'shear_failures' not in local:unknown(shear_name,'Actual complete local punching shear is not verified')
            else:count_check(shear_name,local['shear_failures'],dict(local=local,method_proofs=method['proof_checks'].get(shear_name)))
    except (ValueError,KeyError,TypeError,RuntimeError,OSError) as exc:
        for name in ROW_IDS:
            if name.startswith('common.disposition.'):unknown(name,exc)
    try:
        from Design.SMRF_Common_Torsion_Stream import verify_torsion_stream
        audits['torsion_strength']=verify_torsion_stream(package['common_torsion'],package,state=state,prepared_sources=sources())
        torsion=audits['torsion_strength']
        if torsion['status']=='not_evaluated':unknown('common.torsion_concurrent','Actual torsion reserve/detail inventory remains incomplete: '+str(torsion))
        else:count_check('common.torsion_concurrent',int(torsion['status']=='fail'),torsion)
    except (ValueError,KeyError,TypeError,RuntimeError,OSError) as exc:unknown('common.torsion_concurrent',exc)
    return {**{k:package[k] for k in ('state_identity','source_identity','definition_sha256','payload_sha256')},
        'checks':[rows[name] for name in ROW_IDS],'audits':audits,'native_analyze_calls':0,'frame_solves':0}
