"""Actual-mat frozen static gravity references and coherent design responses.

ACI318-19 6.3.1.1/R6.3.1.1: explicit static stiffness idealizations with
independent sensitivity evidence. A gross-slab predictor initializes a
service D+L field. Local actual-mat full relaxed D is frozen for the next
linear analysis; this is not asserted to be nonlinear material equilibrium.
The reference update count is finite and declared, never a convergence claim.
Every reference and final design response remains individually identifiable.
"""
from __future__ import annotations
import time
import numpy as np
from Design.SMRF_Physical_Floor_Materials import FrozenGravityReferenceBuilder
from Design.SMRF_Coupled_Floor_Design import _hash

METHOD='actual_mat_frozen_static_gravity_reference_family_v3_paired_physical_state'


def physical_reference_state_identity(response):
    """Bind material references to every physical high/low DOF, not just joints."""
    frame=response['frame'];joint=[];floors=[]
    def pair(record,high_key,low_key,size):
        if low_key not in record:raise ValueError('Actual paired reference state requires both displacement parts')
        h,l=np.asarray(record[high_key],float),np.asarray(record[low_key],float)
        if h.shape!=(size,) or l.shape!=h.shape or not np.isfinite(h).all() or not np.isfinite(l).all():
            raise ValueError('Complete finite physical reference state required')
        return dict(high=h.tolist(),low=l.tolist())
    for case in frame['cases']:
        joint.append(pair(case,'physical_displacements','physical_displacements_lo',len(frame['physical_dof_ids'])))
    for floor in response['floors']:
        states=[]
        for case in floor['responses']:
            states.append(dict(case_id=case['case_id'],dof_ids=case['dof_ids'],
                **pair(case,'retained_displacements','retained_displacements_lo',len(case['dof_ids']))))
        if len(states)!=len(joint):raise ValueError('Complete paired floor reference case inventory required')
        floors.append(dict(id=floor['id'],states=states))
    if not joint or not floors:raise ValueError('Actual nonempty paired reference inventory required')
    return _hash(dict(frame_dof_ids=frame['physical_dof_ids'],joint_states=joint,floors=floors))


def frozen_gravity_family(predictor,inventory,design_cases,*,protocol_id,
                          reference_updates=1,maximum_wall_seconds,
                          maximum_coupled_dofs,web_torsion_factor=1.,patterns=(),progress=None,saved_result=None,
                          material_mapping='clipped_cylinders_v2'):
    """Build/recover one or more finite actual references, then design cases.

    predictor must have one actual D+L reference case. Its stiffness factors
    remain recorded. This function reuses its immutable geometry and starts a
    new system for each reference so earlier operators cannot become stale.
    Any material-domain/ownership/quadrature unknown returns its full evidence;
    it is never converted to an engineering capacity failure.
    """
    if not protocol_id or len(predictor.cases)!=1 or not design_cases:
        raise ValueError('Protocol, one actual reference and nonempty design cases required')
    if inventory['state_identity']!=predictor.definition['state_identity']:
        raise ValueError('Reference reinforcement differs from the actual frozen geometry')
    if type(reference_updates) is not int or not 0<=reference_updates<=3 or maximum_wall_seconds<=0:
        raise ValueError('Explicit finite reference-update and wall bounds required')
    case=predictor.cases[0]
    if float(case['dead'])!=1. or float(case['live'])!=1. or case.get('live_pattern','all')!='all':
        raise ValueError('Initial reference must be the declared actual all-panel D+L service load')
    # Validate the explicit material choice before any predictor solve. The
    # builder remains responsible for exact local material/ownership checks.
    FrozenGravityReferenceBuilder(inventory,reference_identity=protocol_id,material_mapping=material_mapping)
    start=time.monotonic();references=[];current=predictor;cracked_points=set()
    def report(stage,**data):
        if progress is not None:progress(dict(stage=stage,elapsed_seconds=time.monotonic()-start,**data))
    report('predictor_solve')
    if saved_result is not None and (saved_result.get('method')!=METHOD or saved_result.get('material_mapping')!=material_mapping
        or saved_result['protocol_id']!=protocol_id or saved_result['reference_updates']!=reference_updates
        or saved_result['definition_sha256']!=predictor.definition['identity_sha256'] or saved_result['source_identity']!=predictor.source_identity
        or saved_result['inventory_identity']!=inventory['identity_sha256'] or len(saved_result['references'])!=reference_updates+1):
        raise ValueError('Saved actual gravity reference protocol/inventory differs')
    response=current.solve(pdelta=True,saved_response=None if saved_result is None else saved_result['predictor_response'])
    result=dict(method=METHOD,protocol_id=protocol_id,definition_sha256=predictor.definition['identity_sha256'],
        source_identity=predictor.source_identity,inventory_identity=inventory['identity_sha256'],material_mapping=material_mapping,
        reference_updates=reference_updates,predictor_response=response,references=references,
        final_combination_nonlinear_equilibrium_claimed=False,force_bound_claimed=False,
        qualification_asserted=False,official_pilot_started=False)
    if response['status']!='equilibrated':
        result.update(status='unresolved',reason='reference predictor did not equilibrate');return result,None
    for iteration in range(reference_updates+1):
        if time.monotonic()-start>maximum_wall_seconds:raise RuntimeError('Frozen reference family wall bound exhausted')
        physical_identity=physical_reference_state_identity(response)
        reference_id=_hash(dict(method=METHOD,protocol=protocol_id,iteration=iteration,
            inventory=inventory['identity_sha256'],material_mapping=material_mapping,frame=response['frame']['identity_sha256'],
            paired_physical_reference_state_sha256=physical_identity))
        if saved_result is not None and saved_result['references'][iteration]['reference_identity']!=reference_id:
            raise ValueError('Saved actual gravity reference identity differs from its rebuilt inputs')
        builder=FrozenGravityReferenceBuilder(inventory,reference_identity=reference_id,
                                             prior_cracked_point_ids=cracked_points,material_mapping=material_mapping)
        fields={r['id']:r['responses'][0] for r in response['floors']};built={}
        for row in current.floors:
            report('build_actual_frozen_reference',iteration=iteration,floor=row['description']['level'])
            built[row['description']['id']]=builder.build(row,fields[row['description']['id']],
                reference_loading={'case':case,'frame_identity':response['frame']['identity_sha256'],
                                   'reference_iteration':iteration})
        record=dict(iteration=iteration,reference_identity=reference_id,
            paired_physical_reference_state_sha256=physical_identity,
            material_mapping=material_mapping,
            floor_references={key:value['audit'] for key,value in built.items()})
        if saved_result is not None:
            saved=saved_result['references'][iteration]
            if any(_hash(saved[key])!=_hash(record[key]) for key in record):
                raise ValueError('Saved actual gravity material audit differs from reconstructed physical reference')
        for value in built.values():cracked_points.update(value['audit'].get('cracked_point_ids',()))
        references.append(record)
        if any(r['status']!='reference_constructed' for r in built.values()):
            result.update(status='unknown',reason='actual reference material/domain/ownership/quadrature unresolved',
                          elapsed_seconds=time.monotonic()-start);return result,None
        following=current.with_cases([case],patterns=patterns,case_identity=reference_id)
        report('assemble_frozen_reference',iteration=iteration)
        operators={row['description']['id']:builder.frozen_operator(row,built[row['description']['id']],
            beam_bending_factor=.35,web_torsion_factor=web_torsion_factor,
            source_identity=predictor.source_identity,maximum_coupled_dofs=maximum_coupled_dofs)
            for row in following.floors}
        following.set_floor_operators(operators,iteration_identity=reference_id)
        response=following.solve(pdelta=True,saved_response=None if saved_result is None else saved_result['references'][iteration]['response']);record['response']=response
        if response['status']!='equilibrated':
            result.update(status='unresolved',reason='frozen reference response did not equilibrate',
                          elapsed_seconds=time.monotonic()-start);return result,None
        current=following
    if time.monotonic()-start>maximum_wall_seconds:raise RuntimeError('Frozen reference family wall bound exhausted')
    design=current.with_cases(design_cases,patterns=patterns,case_identity=_hash(dict(
        protocol=protocol_id,reference=references[-1]['reference_identity'],material_mapping=material_mapping,cases=design_cases)))
    report('design_case_solve',cases=len(design_cases));design_result=design.solve(pdelta=True,
        saved_response=None if saved_result is None else saved_result['design_response'])
    result.update(status='responses_computed' if design_result['status']=='equilibrated' else 'unresolved',
        design_response=design_result,elapsed_seconds=time.monotonic()-start,
        targeted_response_sensitivity_verified=False)
    return result,design
