"""Signed actual-section extraction for the shared floor action contract.

All locations are global inch coordinates and all moments reference the slab
midplane. Primary ELF increments have already had their identical auxiliary
gravity baseline removed. Their zero-N composite bending field is separate
from the P5 external membrane/collector load path. No C0 field is added here.
"""
from __future__ import annotations
import numpy as np
from Design.SMRF_Physical_Floor_Cuts import web_section,identity
from Design.SMRF_Full_T_Projection import FullTIncrementProjector

METHOD='physical_section_primary_increment_readout_v2_actual_axial_domain'


def primary_beam_cut(definition,increment,member_id,station_in):
    if increment.get('status')!='equilibrated':raise ValueError('Complete primary ELF increment required')
    if increment.get('primary_definition_sha256')!=definition['identity_sha256']:
        raise ValueError('Primary increment belongs to a different physical model')
    matches=[b for b in definition['beams'] if str(b['id'])==str(member_id)]
    if len(matches)!=1:raise ValueError('One actual primary beam required')
    member=matches[0];nodes={n['id']:np.array(n['xyz'],float) for n in definition['nodes']}
    p=np.array([nodes[member['i']],nodes[member['j']]])
    delta=p[1]-p[0]
    if abs(delta[2])>1e-9:raise ValueError('Physical floor beam must be horizontal')
    axis='x' if delta[0]>0 and abs(delta[1])<=1e-9 else 'y' if delta[1]>0 and abs(delta[0])<=1e-9 else None
    if axis is None:raise ValueError('Physical positive X/Y beam direction required')
    p[:,2]-=p[0,2]
    row=dict(kind='web',id=str(member_id)+'/primary_increment',member_id=str(member_id),axis=axis,
        positions_in=p.tolist(),global_resisting_force=increment['beams'][str(member_id)]['global_force_increment'])
    cut=web_section(row,station_in,uniform_vertical_load_kip_in=0.)
    cut.update(method=METHOD,primary_definition_sha256=definition['identity_sha256'],
        state_identity=definition['state_identity'],physical_gravity_case_id=increment['physical_gravity_case_id'],
        primary_increment_identity=increment['frame']['identity_sha256'],
        reference_z_in=0.,normal_derivative_per_in=[0.,-cut['Q_vertical_kip'],cut['V_inplane_kip']],
        physical_member_interval_in=p[:,0 if axis=='x' else 1].tolist(),
        source_scope='complete_primary_increment_before_external_N_partition')
    cut['identity_sha256']=identity(cut)
    return cut


def installed_full_t_projector(member):
    import Structure_Parameters as sp
    from Model import Member_Groups as mg,Member_Properties as mp
    if member.is_column:raise ValueError('Physical beam required')
    state=mg.active()
    if state is None or state.member(member.member_tag).design!=member.design:
        raise ValueError('Exact physical member must be installed')
    axis='x' if member.member_type=='beam_x' else 'y';d=member.design
    line=member.grid_j*sp.BAY_Y if axis=='x' else member.grid_i*sp.BAY_X
    flange=mp.effective_flange(member);overhangs={p['side']:p['overhang_in'] for p in flange['overhangs']}
    # Member_Properties names actual minus/plus sides explicitly.
    if set(overhangs)-{'minus','plus'}:raise ValueError('Unknown effective flange side convention')
    interval=[line-d.b_in/2-overhangs.get('minus',0.),line+d.b_in/2+overhangs.get('plus',0.)]
    e=sp.concrete_ec_ksi(d.fc_ksi)
    return FullTIncrementProjector(slab_thickness_in=sp.SLAB_THICKNESS_IN,beam_depth_in=d.h_in,
        flange_interval_in=interval,web_interval_in=[line-d.b_in/2,line+d.b_in/2],
        slab_E_ksi=e,web_E_ksi=e,axis=axis,transverse_reference_in=line)


def primary_flexural_projection(projector,cut,*,external_axial_owner):
    """Project full strong/weak bending once; external N belongs to named P5.

    The discarded primary normal force is reported, never silently added to
    the physical diaphragm force. The rigid-diaphragm model normally gives
    zero member axial extension; its measured residual is retained as a check.
    """
    if not isinstance(external_axial_owner,str) or not external_axial_owner:
        raise ValueError('Exclusive actual external diaphragm axial owner required')
    if projector.axis!=cut['axis']:raise ValueError('Actual cut and full-T axes differ')
    projection=projector.project(n_tension_kip=0.,m_z_kip_in=cut['M_centroid_kipin'],
        m_s_kip_in=cut['Ms_transverse_first_moment_kipin'],increment_id=cut['identity_sha256'])
    projection['source_partition']=dict(primary_normal_force_removed_kip=cut['N_tension_kip'],
        external_axial_owner=external_axial_owner,primary_component='zero_total_N_full_T_flexural_couple',
        double_external_axial_credit=False,gravity_C0_added=False,
        measured_rigid_beam_axial_zero=abs(cut['N_tension_kip'])<=1e-7)
    projection['normal_derivative_per_in']=cut['normal_derivative_per_in']
    return projection


def physical_web_loads(floor_row,case_index):
    """Exact signed distributed web load for local equilibrium recovery."""
    floor=floor_row['assembly'];record=floor_row['loads'][case_index]
    return {e['id']:-record['beam_weight_factor']*floor.gamma*e['native']['area_in2']
            for e in floor.elements if e['kind']=='web'}
