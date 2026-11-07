"""Reconstruct prescribed slab-only support-face and paired-transfer actions."""
from __future__ import annotations
import copy
import numpy as np
from Design.SMRF_Common_Floor_Actions import _hash
from Design.SMRF_Full_T_Projection import FullTIncrementProjector
from Design.SMRF_Primary_Moment_Guards import physical_cut_states,primary_membrane_source_from_readout,primary_moment_guards,_normal

METHOD='actual_local_support_transfer_point_recipe_v1'


def projector_parameters(projector):
    return dict(slab_thickness_in=projector.t,beam_depth_in=projector.h,flange_interval_in=list(projector.flange),
        web_interval_in=list(projector.web),slab_E_ksi=projector.ef,web_E_ksi=projector.ew,axis=projector.axis,transverse_reference_in=projector.sref)


def reconstruct_local_face_branch(face,branch):
    """Same original prescribed equations; actual source fields rechecked."""
    from Design.SMRF_Common_Local_Flexure import _guard,_projected_slab
    raw=face['raw'];band=raw['slab']['interval_in'];axis=raw['slab']['axis'];j=0 if axis=='x' else 1
    if raw['web'] is not None:raise ValueError('Local transfer resistance must retain a slab-only physical cut')
    base=physical_cut_states([raw],outer_interval_in=band)[0][0]
    if branch=='physical':return base
    primary=face['primary'];inc=primary['increment'];pe=primary['physical_effective'];ge=primary['gravity_effective'];g=primary['gravity_local']
    projector=FullTIncrementProjector(**primary['projector_parameters'])
    # Reuse the full existing signed target/concurrency validator, including
    # zero external primary N and identical gravity/effective-flange geometry.
    primary_moment_guards([pe],ge,projector=projector,primary_increment=inc,guard_identity='local_recipe_source_validation',
        components=('both',),physical_effective_cut=pe,outer_interval_in=projector.flange)
    if (inc['state_identity']!=raw['identity']['state_identity'] or inc['physical_combination_id']!=raw['identity']['combination_id']
        or inc['axis']!=axis or inc['station_in']!=raw['slab']['station_in'] or g['identity']['combination_id']!=inc['gravity_load_id']
        or g['identity']['state_identity']!=raw['identity']['state_identity'] or g['web'] is not None
        or any(g['slab'][key]!=raw['slab'][key] for key in ('axis','station_in','interval_in','trace'))):
        raise ValueError('Local physical/gravity/primary state or section differs')
    delta=_normal(ge,axis,projector.sref)+np.array([0.,inc['m_z_kip_in'],inc['m_s_kip_in']])-_normal(pe,axis,projector.sref)
    name=face['name'];parts=branch.split('/')
    if len(parts)==2 and parts[0]=='primary' and parts[1] in ('strong','weak','both'):
        mode=parts[1];dz=delta[1] if mode in ('strong','both') else 0.;ds=delta[2] if mode in ('weak','both') else 0.
        projection=projector.project(n_tension_kip=0.,m_z_kip_in=dz,m_s_kip_in=ds,increment_id=name+'/'+mode)
        proof=dict(primary_increment=inc,projection=projection,exact_physical_sha256=_hash(pe),exact_gravity_sha256=_hash(ge),slab_only=True)
        return _guard(base,np.asarray(base['resultants'])+_projected_slab(projector,projection,band,base['reference_s_in']),proof,branch)
    if len(parts) not in (2,3) or parts[0]!='P5' or (len(parts)==3 and parts[2] not in ('strong','both')):
        raise ValueError('Original explicit local physical/primary/P5 branch required')
    bid=parts[1];readout=face['p5_readouts'][bid]
    if any(readout[key]!=raw['slab'][key] for key in ('axis','station_in','interval_in','trace')):
        raise ValueError('Local P5 actual cut geometry differs')
    if readout['transverse_reference_in']!=base['reference_s_in'] or readout['gravity_case_id']!=inc['gravity_load_id']:
        raise ValueError('Local P5 reference or gravity differs')
    verified=primary_membrane_source_from_readout(readout,identity=raw['identity'])
    target=np.asarray(base['resultants']).copy();target[:3]=np.asarray(g['slab']['integral'][:3])+np.asarray(verified['slab_membrane_integral_kip'])
    target[6]=g['slab']['transverse_first_moment'][j]+verified['slab_membrane_global_first_moment_kipin'][j]-base['reference_s_in']*target[j]
    proof=dict(primary_slab_source=verified,gravity_slab_sha256=_hash(g),slab_only=True,web_design_force_omitted_from_slab_resistance=True)
    if len(parts)==2:return _guard(base,target,proof,branch)
    mode=parts[2];ds=0. if mode=='strong' else inc['m_s_kip_in']
    projection=projector.project(n_tension_kip=0.,m_z_kip_in=delta[1],m_s_kip_in=ds,increment_id=name+'/P5/'+bid+'/'+mode)
    proof=dict(proof,primary_increment=inc,projection=projection,exact_physical_sha256=_hash(pe),exact_gravity_sha256=_hash(ge))
    return _guard(base,target+_projected_slab(projector,projection,band,base['reference_s_in']),proof,branch)


def local_transfer_recipe(faces,branch,record,axis,code_band,actual_band):
    from Design.SMRF_Common_Local_Flexure import METHOD as PRODUCER_METHOD
    selected=[]
    for face in faces:
        value=copy.deepcopy(face['local_recipe_inputs']);value['normal']=face['normal']
        if branch=='physical':value['primary']=None;value['p5_readouts']={}
        elif branch.startswith('P5/'):
            bid=branch.split('/')[1];value['p5_readouts']={bid:value['p5_readouts'][bid]}
        else:value['p5_readouts']={}
        selected.append(value)
    result=dict(method=METHOD,producer_method=PRODUCER_METHOD,faces=selected,branch=branch,support_record=copy.deepcopy(record),
        axis=axis,code_band_in=list(code_band),actual_band_in=list(actual_band))
    result['payload_sha256']=_hash(result);return result


def bind_local_transfer_action(recipe,face_index,include_shortfall,*,floor,inventory,analysis_input_identity,source_response_sha256,definition_sha256):
    from Design.SMRF_Common_Local_Flexure import METHOD as PRODUCER_METHOD,transfer_shortfall,_guard
    from Design.SMRF_Local_Pitch_Strength import bind_local_pitch_action
    if recipe.get('method')!=METHOD or recipe.get('producer_method')!=PRODUCER_METHOD or recipe.get('payload_sha256')!=_hash({k:v for k,v in recipe.items() if k!='payload_sha256'}):
        raise ValueError('Exact current local transfer recipe required')
    if type(face_index) is not int or not 0<=face_index<len(recipe['faces']) or not isinstance(include_shortfall,bool):
        raise ValueError('Exact unique face and explicit shortfall selection required')
    rows=[reconstruct_local_face_branch(f,recipe['branch']) for f in recipe['faces']]
    record=recipe['support_record'];axis=recipe['axis'];j=0 if axis=='x' else 1
    for face,row in zip(recipe['faces'],rows):
        raw=face['raw']
        if (record['floor']!=floor or record['case_id']!=row['identity']['combination_id']
            or raw['slab']['axis']!=axis or raw['slab']['interval_in']!=recipe['actual_band_in']):
            raise ValueError('Paired local transfer source/geometry differs')
    required=record['flexural_transfer_required_kipin'][axis]*(1 if axis=='x' else -1)
    proof=transfer_shortfall([r['resultants'][j+3] for r in rows],[f['normal'] for f in recipe['faces']],required)
    proof.update(method=PRODUCER_METHOD,support_record_id=record['id'],source_record_sha256=_hash(record),axis=axis,branch=recipe['branch'],
        code_band_in=recipe['code_band_in'],actual_bar_clear_band_in=recipe['actual_band_in'],code_basis='ACI318-19 8.4.2.2.1-.6',slab_only_resistance=True)
    row=rows[face_index];raw=recipe['faces'][face_index]['raw']
    base=bind_local_pitch_action(raw,floor=floor,inventory=inventory,analysis_input_identity=analysis_input_identity,
        source_response_sha256=source_response_sha256,definition_sha256=definition_sha256)
    if include_shortfall:
        target=np.asarray(row['resultants']).copy();target[j+3]+=proof['face_increments_kipin'][face_index]
        row=_guard(row,target,proof,recipe['branch']+'/gamma_f')
    for key in ('material_inventory_id','analysis_input_identity','source_response_sha256','definition_sha256'):
        if key in row and row[key]!=base[key]:raise ValueError('Local transfer source identity differs')
        row[key]=base[key]
    row['local_point_binding']=dict(method=METHOD,kind='local_transfer',point=base['local_point_binding']['point'],
        recipe=copy.deepcopy(recipe),face_index=face_index,include_shortfall=include_shortfall,normalized_action_sha256=_hash(row))
    return row


def verify_local_transfer_binding(action,spec,inventory):
    binding=action['local_point_binding']
    if binding.get('method')!=METHOD or binding.get('kind')!='local_transfer':raise ValueError('Exact local transfer binding required')
    rebuilt=bind_local_transfer_action(binding['recipe'],binding['face_index'],binding['include_shortfall'],floor=spec['floor'],inventory=inventory,
        analysis_input_identity=action['analysis_input_identity'],source_response_sha256=action['source_response_sha256'],definition_sha256=action['definition_sha256'])
    if rebuilt!=action:raise ValueError('Local transfer action differs from its paired-face source recipe')
    expected={k:spec[k] for k in ('floor','axis','station_in','interval_in','trace')};expected['member_id']=None
    if spec.get('member_id') is not None or spec.get('material_scope')!='slab_only' or binding['point']!=expected:
        raise ValueError('Local transfer resistance requires identical slab-only point geometry')
