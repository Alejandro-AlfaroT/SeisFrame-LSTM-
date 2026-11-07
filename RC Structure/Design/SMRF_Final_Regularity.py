"""Final installed actions to independently certified regularity evidence.

The nominal story-resistance proof and the reflected-plan distribution proof
have different jobs. A total resistance certificate does not invent a split
between parallel frame lines. Existing actual-P torsion/drift responses are
replayed and retained; no new displacement solve or Ax reduction occurs here.
"""
from __future__ import annotations
import copy
import math
from Design.SMRF_Coupled_Floor_Design import _hash,validate_definition
from Design.SMRF_Demands import classify_torsional_irregularity,validate_torsion_assessment,seismic_design_category
from Design.SMRF_Story_Adapter import computed_verification_valid,_validate_inventory

METHOD='final_actual_member_regularities_with_replayed_nominal_certificates_v1'


def refresh_actual_pattern_basis(basis,combinations,*,sds,num_bay_x,num_bay_y,patterning_enabled):
    """Use the complete solved case roster, independently of the old load cache."""
    from Design.SMRF_Demands import live_load_patterns,strength_load_combinations
    patterns=live_load_patterns(num_bay_x,num_bay_y) if patterning_enabled else []
    expected=strength_load_combinations(sds,redundancy_factor=1.3,live_patterns=patterns)
    if combinations!=expected:raise ValueError('Final actual signed/patterned combinations differ from the prescribed complete inventory')
    return {**basis,'live_load_patterns':patterns,'patterns_in_strength_envelope':bool(patterning_enabled),
        'live_pattern_source':dict(method=METHOD,complete_combination_inventory_sha256=_hash(combinations),
            actual_pattern_combination_ids=[c['id'] for c in combinations if c['live_pattern']!='all'],
            basis='Every prescribed panel arrangement is represented by its own actual physical gravity response')}


def refresh_torsion_classification(torsion,strength,*,num_floors,accidental_ratio):
    """Reclassify only after replaying the original four-case torsion record."""
    audit=validate_torsion_assessment(torsion,num_floors,policy_ratio=accidental_ratio)
    if not audit['valid']:raise ValueError('Final torsion source failed replay: '+audit['reason'])
    classification=classify_torsional_irregularity(audit['tir'],strength.get('one_side_fraction'),
        strength_model_verified=computed_verification_valid(strength),
        independent_distribution_bound=strength.get('independent_distribution_bound'))
    required=audit['ax_required_envelope'] if classification['type_1'] else 1.
    applied=torsion.get('amplification')
    if not isinstance(applied,(int,float)) or not math.isfinite(applied) or applied+1e-10<required:
        raise ValueError('Final classification requires greater Ax than the actual demand analyses applied')
    output=copy.deepcopy(torsion)
    output.update(classification=classification,torsional_irregularity=classification['label'],
        amplification_required=required,strength_distribution=strength,
        final_regularity_binding=dict(method=METHOD,original_torsion_sha256=_hash(torsion),
            final_strength_sha256=_hash(strength),original_applied_amplification=applied,
            new_displacement_solves=0,computed_story_certificate_valid=computed_verification_valid(strength)))
    return output


def _validate_common(state,common):
    from Model import Member_Groups as mg
    from Design.SMRF_Common_Action_Analysis import METHOD as COMMON_METHOD,current_source_identity
    if (common.get('method')!=COMMON_METHOD or common.get('status')!='responses_computed'
        or common.get('source_unchanged') is not True
        or common.get('payload_sha256')!=_hash({k:v for k,v in common.items() if k!='payload_sha256'})):
        raise ValueError('Complete immutable source-bound common response required for final regularity')
    definition=common['definition'];validate_definition(definition)
    if (mg.active() is None or mg.active().identity()!=state.identity() or common.get('state_identity')!=state.identity()
        or definition['state_identity']!=state.identity() or common.get('definition_sha256')!=definition['identity_sha256']):
        raise ValueError('Final regularity actual installed state/definition differs')
    if common['source_identity']!=current_source_identity() or common['analysis_source_identity']!=_hash(common['source_identity']):
        raise ValueError('Final regularity actual action source is stale')
    expected=[c['id'] for c in common['combinations']]
    if len(set(expected))!=len(expected) or set(common['gravity_case_mapping'])!=set(expected):
        raise ValueError('Complete unique final combination/gravity mapping required')
    _validate_inventory(state,common['actions'],expected,common['story_strength_gravity_actions'])
    by_id={c['id']:c for c in common['combinations']}
    for action in common['actions']+common['story_strength_gravity_actions']:
        if (action.get('state_identity')!=state.identity() or action.get('source_identity')!=common['analysis_source_identity']
            or action.get('physical_definition_sha256')!=definition['identity_sha256']):
            raise ValueError('Final actual member action physical/source binding differs')
    for action in common['actions']:
        if any(action.get(k,'all' if k=='live_pattern' else None)!=by_id[action['id']].get(k,'all' if k=='live_pattern' else None)
               for k in ('dead','live','ex','ey','live_pattern')) or action['physical_gravity_case_id']!=common['gravity_case_mapping'][action['id']]:
            raise ValueError('Final signed action combination/gravity identity differs')
    service=common['primary']['service_DL'];torsion=common['torsion']
    if (torsion.get('source_identity')!=common['analysis_source_identity']
        or torsion.get('primary_definition_sha256')!=service['definition']['identity_sha256']
        or service['definition']['physical_definition_sha256']!=definition['identity_sha256']
        or torsion.get('gravity_transfer')!=service['gravity_transfer']['column_axial_ledger']):
        raise ValueError('Final actual-P torsion source/physical gravity binding differs')
    return expected


def refresh_final_regularities(cfg,common,capacity,ledger,*,state=None):
    """Calculate final story certificates, then refresh torsion/vertical evidence.

    Call on the final unchanged installed cage, after capacity construction.
    The common response is not mutated. Missing bounds remain unresolved;
    stale state/source/combination evidence raises a software/input error.
    """
    import Structure_Parameters as sp
    from Model import Member_Groups as mg
    from Design import Grouped_Design as grouped,Design_Driver as driver
    state=state or mg.active();expected=_validate_common(state,common)
    weights={r['floor']:r['seismic_weight_kip'] for r in ledger['floors']}
    if (len(weights)!=len(ledger['floors']) or set(weights)!=set(range(1,sp.NUM_FLOOR+1))
        or any(not math.isfinite(weights[k]) or weights[k]<=0
            or not math.isclose(weights[k],common['elf']['story_weights_kip'][k-1],rel_tol=1e-9,abs_tol=1e-8) for k in weights)):
        raise ValueError('Final regularity weight ledger differs from actual ELF mass inventory')
    capacity_hash=_hash(capacity)
    strength=grouped.story_strength_distribution(cfg,state=state,actions=common['actions'],expected_ids=expected,
        capacity=capacity,gravity_actions=common['story_strength_gravity_actions'])
    torsion=refresh_torsion_classification(common['torsion'],strength,num_floors=sp.NUM_FLOOR,
        accidental_ratio=cfg.demands.accidental_torsion_ratio)
    vertical=grouped.vertical_regularity(common['drift_screen'],common['elf'],strength,ledger)
    regularity=grouped.grouped_regularity(strength,vertical,state)
    demand_basis=driver._demand_basis(cfg,torsion,common['elf'],common['drift_screen'],strength,regularity)
    demand_basis=refresh_actual_pattern_basis(demand_basis,common['combinations'],sds=sp.ASCE_SDS,
        num_bay_x=sp.NUM_BAY_X,num_bay_y=sp.NUM_BAY_Y,patterning_enabled=cfg.demands.live_load_patterning)
    sdc=seismic_design_category(sp.ASCE_SDS,sp.ASCE_SD1,sp.ASCE_S1,cfg.demands.risk_category)
    checks=grouped.regularity_checks(vertical,sdc,strength_model_verified=computed_verification_valid(strength))
    _validate_common(state,common)
    if _hash(capacity)!=capacity_hash:raise ValueError('Final capacity evidence changed during story verification')
    result=dict(method=METHOD,common_input_signature=common['input_signature'],common_payload_sha256=common['payload_sha256'],
        state_identity=state.identity(),analysis_source_identity=common['analysis_source_identity'],
        strength=strength,torsion=torsion,vertical=vertical,regularity=regularity,demand_basis=demand_basis,checks=checks,
        computed_story_certificate_valid=computed_verification_valid(strength),new_primary_displacement_solves=0,
        qualification_asserted=False)
    result['payload_sha256']=_hash(result);return result
