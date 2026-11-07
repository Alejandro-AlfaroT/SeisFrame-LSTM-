"""Reconstruct installed operators and check saved raw fields without solving frames."""
from __future__ import annotations
from Design.SMRF_Coupled_Floor_Design import CoupledFloorSystem,installed_definition,_hash
from Design.SMRF_Frozen_Gravity_Design import frozen_gravity_family
from Design.SMRF_Physical_Seismic_Design import PhysicalSeismicSystem
from Design.SMRF_Physical_Floor_Materials import installed_inventory
from Design.SMRF_Primary_Frame_Transfer import PrimaryFrame,installed_primary_definition,verify_transferred_floor_influences
from Design.SMRF_Demands import live_load_patterns,strength_load_combinations


def verify_derived_member_actions(state,common):
    """Rebuild selection/capacity/story actions from the verified native fields.

    Equal saved labels or field hashes are not a substitute for extraction.
    This repeats every polynomial cut and full primary wrench transport, with
    invocation-local hashing only; no mechanical analysis is performed.
    """
    from Design.SMRF_Physical_Member_Actions import build_member_actions,PreparedMemberResponses
    from Design.SMRF_Story_Adapter import collect_physical_gravity_cases
    from Design.SMRF_Common_Compatibility import adapt_compatibility
    gravity=common['gravity']['design_response'];prepared=PreparedMemberResponses(gravity);adapted=[]
    for combination in common['combinations']:
        gid=common['gravity_case_mapping'][combination['id']];active=bool(combination['ex'] or combination['ey'])
        primary=common['primary'][gid] if active else None
        adapted.append(build_member_actions(state,common['definition'],gravity,combination,gravity_case_id=gid,
            primary_definition=None if primary is None else primary['definition'],
            primary_increment=None if primary is None else primary['increments'][combination['id']],
            primary_transfer=None if primary is None else primary['gravity_transfer'],prepared_responses=prepared))
    actions=[r['action'] for r in adapted]
    if _hash(actions)!=_hash(common['actions']):raise ValueError('Saved selection/strength member actions differ from current native field extraction')
    gcases=collect_physical_gravity_cases(adapted,[c['id'] for c in common['combinations']])
    if _hash(gcases)!=_hash(common['story_strength_gravity_actions']):raise ValueError('Saved story-strength gravity fields differ from current native cut extraction')
    transfer=dict(physical_gravity_member_actions=gcases,expected_physical_gravity_case_ids=[c['id'] for c in gcases],
        state_identity=state.identity(),source_identity=common['analysis_source_identity'],physical_definition_sha256=common['definition_sha256'])
    if _hash(transfer)!=_hash(common['capacity_gravity_transfer']):raise ValueError('Saved capacity gravity transfer differs from reconstructed actual field')
    compatibility=adapt_compatibility(state,common['definition'],common['compatibility'],
        cd=common['demand_parameters']['cd'],importance_factor=common['demand_parameters']['importance_factor'])
    if _hash(compatibility)!=_hash(common['compatibility_actions']):raise ValueError('Saved compatibility member forces differ from actual physical responses')
    return dict(signed_member_cases=len(actions),gravity_member_cases=len(gcases),all_derived_member_fields_verified=True)


def replay_physics(state,common):
    import Structure_Parameters as sp
    from Model import Member_Properties as mp
    from Design.SMRF_Common_Action_Analysis import _gravity_cases,current_source_identity,common_input_identity,compatibility_cases,verify_common_input_identity
    from Design.SMRF_Physical_Diaphragm import installed_grid_masses
    from Design.SMRF_Primary_Demands import floor_loads,torsion_assessment,drift_screen
    from Design.SMRF_Demands import seismic_design_category
    from Loads.Seismic_ELF import elf_story_forces
    if common['source_identity']!=current_source_identity():raise ValueError('Common physics source changed')
    strength_basis=verify_common_input_identity(common)
    definition=installed_definition(state);inventory=installed_inventory(state)
    if definition!=common['definition'] or inventory!=common['inventory']:raise ValueError('Actual installed physics geometry/material differs')
    parameters=common['demand_parameters'];risk=parameters['risk_category'];ratio=parameters['accidental_ratio']
    if (parameters['cd']!=sp.ASCE_CD or parameters['importance_factor']!=sp.ASCE_IE or parameters['redundancy_factor']!=1.3 or
        parameters['seismic_live_load_fraction']!=sp.SEISMIC_LIVE_LOAD_FRACTION):raise ValueError('Actual seismic parameters differ from installed demand basis')
    combinations=common['combinations'];patterned=parameters['live_load_patterning']
    patterns=live_load_patterns(sp.NUM_BAY_X,sp.NUM_BAY_Y) if patterned else []
    if combinations!=strength_load_combinations(sp.ASCE_SDS,redundancy_factor=1.3,live_patterns=patterns):
        raise ValueError('Actual common original strength combination inventory differs')
    gcases,gmap=_gravity_cases(combinations)
    if gmap!=common['gravity_case_mapping']:raise ValueError('Actual common gravity case mapping differs')
    limits=common['limits'];source=common['analysis_source_identity'];identity=common['input_signature']
    mass=installed_grid_masses(definition);elf=elf_story_forces(common['model_period_sec'])
    if _hash(mass)!=_hash(common['grid_masses']) or _hash(elf)!=_hash(common['elf']):raise ValueError('Saved actual mass/ELF basis differs')
    material_mapping=common['floor_material_mapping']
    if common_input_identity(state.identity(),definition,inventory,source,limits,combinations,common['model_period_sec'],parameters,elf,mass,material_mapping=material_mapping,
            strength_material_action_basis=strength_basis)!=identity:
        raise ValueError('Saved mechanical input identity does not bind the actual full demand basis')
    mesh=dict(subdivisions_per_bay=int(limits['mesh_per_bay']),max_shells_per_floor=int(limits['max_shells_per_floor']),max_coupled_dofs=int(limits['max_coupled_dofs']))
    predictor=CoupledFloorSystem(definition,[gcases[0]],source_identity=source,
        stiffness_basis='Gross own-slab predictor; beam.35; column.7; equilibrium GJ retained',
        beam_bending_factor=.35,slab_bending_factor=1.,web_torsion_factor=1.,column_bending_factor=.7,**mesh)
    rebuilt,system=frozen_gravity_family(predictor,inventory,gcases,protocol_id=identity,
        reference_updates=int(limits['reference_updates']),maximum_wall_seconds=limits['reference_wall_seconds'],
        maximum_coupled_dofs=int(limits['max_coupled_dofs']),patterns=patterns,saved_result=common['gravity'],material_mapping=material_mapping)
    if system is None or rebuilt['status']!='responses_computed':raise ValueError('Saved gravity material reference unresolved')
    del predictor,system
    physical=PhysicalSeismicSystem(definition,gcases,source_identity=source,
        seismic_live_load_fraction=sp.SEISMIC_LIVE_LOAD_FRACTION,max_cases=int(limits['max_case_count']),patterns=patterns,**mesh)
    observed=[]
    bycombo={c['id']:c for c in combinations}
    for batch in common['physical_branches']:
        if batch['input_identity']!=identity:raise ValueError('Saved physical branch input identity differs')
        for saved in batch['case_descriptions']:
            c=bycombo[saved['id']]
            required=dict(id=c['id'],gravity_case_id=gmap[c['id']],**floor_loads(definition,elf,mass,ex=c['ex'],ey=c['ey'],
                accidental_ratio=ratio,amplification=common['torsion']['amplification']))
            if _hash(saved)!=_hash(required):raise ValueError('Saved physical load differs from actual signed ELF/accidental torque')
        physical.solve(batch['case_descriptions'],input_identity=identity,saved_result=batch)
        observed.extend(c['id'] for c in batch['case_descriptions'])
    expected=[c['id'] for c in combinations if c['ex'] or c['ey']]
    if len(observed)!=len(set(observed)) or set(observed)!=set(expected):raise ValueError('Physical original signed ELF case inventory incomplete')
    gravity=rebuilt['design_response'];primary_count=0;service=None;influences=[]
    bygravity={r['id']:r for r in gcases}
    required_groups={'service_DL'}|{gmap[c] for c in expected}
    if set(common['primary'])!=required_groups:raise ValueError('Actual primary gravity group inventory differs')
    for gid,saved in common['primary'].items():
        pd=installed_primary_definition(definition,bygravity[gid])
        if pd!=saved['definition']:raise ValueError('Retained primary model differs from actual installed policy')
        primary=PrimaryFrame(pd,source_identity=source)
        primary.transfer_gravity(gravity,case_id=gid,saved_transfer=saved['gravity_transfer'])
        if set(saved['increments'])!={cid for cid in expected if gmap[cid]==gid}:raise ValueError('Primary signed increment roster differs')
        for cid,inc in saved['increments'].items():
            c=bycombo[cid];required=floor_loads(definition,elf,mass,ex=c['ex'],ey=c['ey'],accidental_ratio=ratio,amplification=common['torsion']['amplification'])
            if _hash(inc['floor_loads'])!=_hash(required['loads']):raise ValueError('Saved primary load differs from actual signed ELF')
            primary.solve_increment(required['loads'],saved_increment=inc);primary_count+=1
        if gid!='service_DL':
            influences.append(verify_transferred_floor_influences(primary,definition,mass,common['p5_influences'][gid]))
        if gid=='service_DL':service=primary
    for name in ('torsion_responses','drift_responses'):
        if len(common[name])!=4:raise ValueError('Complete original signed service torsion/drift response inventory required')
    torsion,tresponses=torsion_assessment(service,definition,elf,mass,accidental_ratio=ratio,strength_distribution={'one_side_fraction':None},saved_responses=common['torsion_responses'])
    if _hash(torsion)!=_hash(common['torsion']):raise ValueError('Saved TIR/Ax differs from reconstructed actual displacement increments')
    sdc=seismic_design_category(sp.ASCE_SDS,sp.ASCE_SD1,sp.ASCE_S1,risk)
    basis=sp.seismic_design_basis(risk_category=risk,redundancy_factor=1.3,seismic_design_category=sdc if sdc in ('D','E','F') else 'D')
    drift,dresponses=drift_screen(service,definition,elf,mass,accidental_ratio=ratio,amplification=torsion['amplification'],basis=basis,sdc=sdc,
        cd=sp.ASCE_CD,gravity_above_by_story={k:mp.gravity_above_kip(k) for k in range(1,sp.NUM_FLOOR+1)},saved_responses=common['drift_responses'])
    if _hash(drift)!=_hash(common['drift_screen']):raise ValueError('Saved drift/stability screen differs from actual primary responses')
    primary_count+=8
    groups=list(dict.fromkeys(gmap[cid] for cid in expected))
    imposed=compatibility_cases(service,dresponses,groups,cd=sp.ASCE_CD,importance_factor=sp.ASCE_IE)
    saved_imposed=[c['imposed'] for c in common['compatibility']['response']['cases']]
    if _hash(imposed)!=_hash(saved_imposed):raise ValueError('Saved compatibility targets differ from actual signed Cd/Ie primary drift')
    physical.compatibility(imposed,input_identity=identity,saved_result=common['compatibility']);del physical
    derived=verify_derived_member_actions(state,common)
    return dict(method='reconstructed_common_native_operator_and_raw_field_replay_v1',
        state_identity=state.identity(),source_identity=source,input_signature=identity,physical_seismic_cases=len(observed),
        gravity_cases=len(gcases),compatibility_cases=len(imposed),primary_increments=primary_count,
        primary_influence_witnesses=influences,derived_member_actions=derived,
        frame_solves=0,native_analyze_calls=0,all_raw_operator_fields_verified=True)
