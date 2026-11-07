"""Serialized existing primary demand guards for the explicit local-pitch basis.

Guards remain additional design states. Replaying a recipe reconstructs its
original Full-T or P5 calculation; it never presents that demand change as a
new physical equilibrium solution or changes nominal/probable capacity steel.
"""
from __future__ import annotations
import copy
from Design.SMRF_Common_Floor_Actions import _hash
from Design.SMRF_Full_T_Projection import FullTIncrementProjector
from Design.SMRF_Primary_Moment_Guards import (METHOD as GUARD_METHOD,primary_moment_guards,
    primary_membrane_guards,primary_membrane_source_from_readout)

METHOD='serialized_primary_guard_local_point_binding_v1'


def primary_guard_recipe(kind,*,projector=None,**arguments):
    """Freeze original guard inputs; preserve all physically signed branches."""
    if kind not in ('moment','membrane'):raise ValueError('Named primary moment or membrane guard recipe required')
    args=copy.deepcopy(arguments)
    if projector is not None:
        if not isinstance(projector,FullTIncrementProjector):raise ValueError('Actual Full-T projector required')
        args['projector_parameters']=dict(slab_thickness_in=projector.t,beam_depth_in=projector.h,
            flange_interval_in=list(projector.flange),web_interval_in=list(projector.web),slab_E_ksi=projector.ef,
            web_E_ksi=projector.ew,axis=projector.axis,transverse_reference_in=projector.sref)
    result=dict(method=METHOD,guard_method=GUARD_METHOD,kind=kind,arguments=args)
    result['payload_sha256']=_hash(result)
    return result


def reconstruct_primary_guard_recipe(recipe):
    if recipe.get('method')!=METHOD or recipe.get('guard_method')!=GUARD_METHOD or recipe.get('payload_sha256')!=_hash({k:v for k,v in recipe.items() if k!='payload_sha256'}):
        raise ValueError('Saved primary guard recipe method or content differs')
    args=copy.deepcopy(recipe['arguments']);kind=recipe['kind']
    if kind=='moment':
        required={'physical_cuts','gravity_effective_cut','primary_increment','guard_identity','projector_parameters'}
        allowed=required|{'components','physical_effective_cut','outer_interval_in'}
    elif kind=='membrane':
        required={'physical_cuts','gravity_cuts','primary_membrane_sources','outer_interval_in','guard_identity'}
        allowed=required|{'projector_parameters','primary_increment','physical_effective_cut','gravity_effective_cut'}
    else:raise ValueError('Unknown saved primary guard recipe kind')
    if not required<=set(args) or not set(args)<=allowed:raise ValueError('Exact original primary guard argument inventory required')
    if 'projector_parameters' in args:args['projector']=FullTIncrementProjector(**args.pop('projector_parameters'))
    if kind=='moment':return primary_moment_guards(**args)
    sources=[]
    for source in args['primary_membrane_sources']:
        if not source.get('actual_readout'):raise ValueError('Actual P5 readout needed for local guard replay')
        rebuilt=primary_membrane_source_from_readout(source['actual_readout'],identity=source['identity'])
        if {k:v for k,v in source.items() if k!='axial_action'}!=rebuilt:
            raise ValueError('Primary membrane summary differs from independently rebuilt actual readout')
        if 'axial_action' in source:rebuilt['axial_action']=source['axial_action']
        sources.append(rebuilt)
    args['primary_membrane_sources']=sources
    return primary_membrane_guards(**args)


def bind_local_primary_guard_action(recipe,branch_id,location_id,*,floor,inventory,analysis_input_identity,source_response_sha256,definition_sha256):
    """Reconstruct one named guard and bind the unchanged actual physical cut."""
    from Design.SMRF_Local_Pitch_Strength import bind_local_pitch_action
    family=reconstruct_primary_guard_recipe(recipe)
    branches=[b for b in family['branches'] if b['branch_id']==branch_id]
    if len(branches)!=1:raise ValueError('Exact unique original guard branch required')
    matches=[s for s in branches[0]['states'] if s['identity']['location_id']==location_id]
    raw=[r for r in recipe['arguments']['physical_cuts'] if r['identity']['location_id']==location_id]
    if len(matches)!=1 or len(raw)!=1:raise ValueError('Exact unique physical cut within guard partition required')
    context=dict(floor=floor,inventory=inventory,analysis_input_identity=analysis_input_identity,
        source_response_sha256=source_response_sha256,definition_sha256=definition_sha256)
    base=bind_local_pitch_action(raw[0],**context)
    row=copy.deepcopy(matches[0])
    if branch_id=='physical':
        if 'strength_guard' in row or row['resultants']!=base['resultants']:
            raise ValueError('Physical partition branch changed the original actual cut demand')
    elif 'strength_guard' not in row or row['accounting'].get('physical_equilibrium_solution_claimed') is not False:
        raise ValueError('A local primary guard must remain an explicit additional design state')
    for key in ('material_inventory_id','analysis_input_identity','source_response_sha256','definition_sha256'):
        if key in row and row[key]!=base[key]:raise ValueError('Primary guard source identity differs from physical cut')
        row[key]=base[key]
    row['local_point_binding']=dict(method=METHOD,kind='primary_partition' if branch_id=='physical' else 'primary_guard',point=base['local_point_binding']['point'],
        raw_cut_sha256=base['local_point_binding']['raw_cut_sha256'],recipe=copy.deepcopy(recipe),branch_id=branch_id,
        location_id=location_id,normalized_guard_sha256=_hash(row))
    return row


def verify_local_primary_guard_binding(action,spec,inventory):
    binding=action.get('local_point_binding') or {}
    if binding.get('method')!=METHOD or binding.get('kind') not in ('primary_guard','primary_partition'):raise ValueError('Exact local primary guard binding required')
    rebuilt=bind_local_primary_guard_action(binding['recipe'],binding['branch_id'],binding['location_id'],floor=spec['floor'],inventory=inventory,
        analysis_input_identity=action['analysis_input_identity'],source_response_sha256=action['source_response_sha256'],definition_sha256=action['definition_sha256'])
    if rebuilt!=action:raise ValueError('Local primary guard action differs from its reconstructed recipe')
    expected={k:spec[k] for k in ('floor','axis','station_in','interval_in','trace')}
    expected['member_id']=None if spec.get('member_id') is None else str(spec['member_id'])
    if binding['point']!=expected:raise ValueError('Primary guard actual local geometry differs from material cut')
    if action['material_inventory_id']!=inventory['identity_sha256']:raise ValueError('Primary guard actual material inventory differs')
