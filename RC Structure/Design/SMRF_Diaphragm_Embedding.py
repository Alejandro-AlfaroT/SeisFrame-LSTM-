"""Exact eccentric-web wrench audit of a planar diaphragm response.

The planar EA web is the physical eccentric web with out-of-plane kinematics
held zero. The omitted constraint moments below are real analysis reactions;
reference-axis transport alone does not verify their physical support path.
"""
from __future__ import annotations
import numpy as np

METHOD='exact_planar_web_eccentric_constraint_wrench_audit_v1'

def spatial_resultant(points,wrenches):
    points=np.asarray(points,float);wrenches=np.asarray(wrenches,float)
    return np.r_[wrenches[:,:3].sum(axis=0),(wrenches[:,3:]+np.cross(points,wrenches[:,:3])).sum(axis=0)]

def map_planar_nodal_forces(floor,mesh,forces):
    """Exact T'F map onto the actual FloorAssembly retained physical DOFs."""
    from Design.SMRF_Floor_Substructure import rigid_arm_matrix
    xy=np.asarray(mesh['nodes'],float);force=np.asarray(forces,float)
    points=np.asarray(floor.points,float)
    if force.shape!=(len(xy),2) or not np.isfinite(force).all():raise ValueError('One finite actual planar force per physical node required')
    if points.shape!=(len(xy),3) or not np.array_equal(points[:,:2],xy) or np.any(points[:,2]!=0):
        raise ValueError('P5 and physical floor must have exactly the same slab-plane mesh')
    if len(floor.node_maps)!=len(points) or len(floor.dof_ids)!=6*len(floor.node_ids):raise ValueError('Complete physical floor DOF map required')
    retained=np.zeros(len(floor.dof_ids));physical=np.zeros((len(points),6));physical[:,:2]=force
    for node,(indices,transform) in enumerate(floor.node_maps):
        expected=rigid_arm_matrix(points[node]-points[floor.owner[node]])
        if np.asarray(indices).shape!=(6,) or np.asarray(transform).shape!=(6,6) or not np.allclose(transform,expected,rtol=0,atol=1e-13):
            raise ValueError('Physical floor node map differs from its actual rigid-arm geometry')
        np.add.at(retained,indices,np.asarray(transform).T@physical[node])
    retained_points=np.array([floor.node_positions[n] for n in floor.node_ids])
    before=spatial_resultant(points,physical);after=spatial_resultant(retained_points,retained.reshape(-1,6))
    error=float(np.max(abs(after-before),initial=0.))
    if error>1e-9*max(1.,float(np.max(abs(before))),float(np.sum(abs(force)))):
        raise ValueError('Retained physical floor load map failed spatial wrench balance')
    return retained,dict(physical_wrench_kip_kipin=before.tolist(),retained_wrench_kip_kipin=after.tolist(),maximum_mapping_error_kip_kipin=error)

def _same_floor_geometry(adapter,floor):
    def support_rows(rows):return sorted((str(r['id']),float(r['x']),float(r['y']),float(r['b_in']),float(r['h_in'])) for r in rows)
    def beam_rows(rows):return sorted((str(r['id']),r['axis'],float(r['line']),float(r['start']),float(r['end']),float(r['b_in']),float(r['h_in'])) for r in rows)
    if support_rows(floor.supports)!=support_rows(adapter.description['supports']) or beam_rows(floor.beams)!=beam_rows(adapter.description['beams']):
        raise ValueError('Coupled floor support/web geometry differs from frozen physical geometry')

def prepare_external_inertia_cases(adapters_by_floor,floors_by_floor,cases):
    """Map complete-building external inertia to actual physical floor DOFs.

    Dictionaries use actual one-based floor levels. Each case has a unique
    id, loads[nf,3] in kip/kip-in, and reference_xy_in[2] (common) or [nf,2].
    Mz is explicitly about that reference; accidental torque and actual-CM
    translation are supplied by the ELF caller, never silently recomputed.
    The output loads feed CoupledFloorSystem.solve_additional_floor_loads.
    No primary column tractions, P5 redistribution, or primary force addition
    is present. All physical column reactions must be recovered by the full
    coupled model. Code-method qualification is a separate decision.
    """
    from Design.SMRF_Physical_Diaphragm import identity
    from Design.SMRF_Coupled_Floor_Design import validate_definition
    if not adapters_by_floor:raise ValueError('Complete actual building floor adapters required')
    first=next(iter(adapters_by_floor.values()));nf=first.definition['geometry']['nf'];levels=set(range(1,nf+1))
    if set(adapters_by_floor)!=levels or set(floors_by_floor)!=levels:
        raise ValueError('External inertia requires every actual floor exactly once')
    case_ids=[str(r['id']) for r in cases]
    if not case_ids or any(not key for key in case_ids) or len(set(case_ids))!=len(case_ids):
        raise ValueError('Nonempty unique external load case identities required')
    validated=[]
    for row in cases:
        loads=np.asarray(row['loads'],float);references=np.asarray(row['reference_xy_in'],float)
        if references.shape==(2,):references=np.tile(references,(nf,1))
        if loads.shape!=(nf,3) or references.shape!=(nf,2) or not np.isfinite(loads).all() or not np.isfinite(references).all():
            raise ValueError('Each case requires finite complete floor loads and explicit wrench references')
        validated.append((loads,references))
    matrices={};floor_records=[];wrenches=np.zeros((len(cases),6))
    identity_keys=('definition_sha256','state_identity','source_identity','nodal_mass_sha256')
    for level in sorted(levels):
        adapter=adapters_by_floor[level];floor=floors_by_floor[level]
        validate_definition(adapter.definition)
        if (adapter.floor!=level or any(adapter.context[k]!=first.context[k] for k in identity_keys)
            or adapter.context['definition_sha256']!=adapter.definition['identity_sha256']
            or adapter.context['nodal_mass_sha256']!=identity(adapter.grid_mass_by_floor)
            or adapter.context['mesh_sha256']!=identity(adapter.mesh)):
            raise ValueError('Stale or mixed physical definition, mass, source, or mesh identity')
        _same_floor_geometry(adapter,floor)
        columns=[];audits=[]
        for key,(loads,references) in zip(case_ids,validated):
            load,reference=loads[level-1],references[level-1]
            force=adapter.external_inertia_forces(load,reference)
            vector,audit=map_planar_nodal_forces(floor,adapter.mesh,force)
            expected=np.array([load[0],load[1],0.,0.,0.,load[2]+reference[0]*load[1]-reference[1]*load[0]])
            error=float(np.max(abs(np.asarray(audit['retained_wrench_kip_kipin'])-expected)))
            if error>1e-9*max(1.,float(np.max(abs(expected))),float(np.sum(abs(force)))):
                raise ValueError('Physical external inertia failed the requested floor wrench')
            columns.append(vector);wrenches[case_ids.index(key)]+=expected
            audits.append(dict(case_id=key,requested_load_kip_kipin=load.tolist(),reference_xy_in=reference.tolist(),
                requested_wrench_at_slab_origin=expected.tolist(),maximum_requested_wrench_error_kip_kipin=error,
                physical_nodal_force_sha256=identity(force.tolist()),**audit))
        matrix=np.column_stack(columns)
        if floor.id in matrices:raise ValueError('Actual floor IDs must be unique')
        matrices[floor.id]=matrix.tolist()
        floor_records.append(dict(level=level,floor_id=floor.id,adapter_context=adapter.context,mass_audit=adapter.mass_audit,
            retained_dof_ids=floor.dof_ids,retained_node_positions=[{'id':key,'xyz':floor.node_positions[key]} for key in floor.node_ids],
            load_matrix_sha256=identity(matrix.tolist()),cases=audits))
    mapping=dict(method='complete_physical_external_inertia_TtransposeF_v1',case_ids=case_ids,floors=floor_records,
        load_datum='actual slab midplane z=0; floor Mz about explicit caller reference',
        external_inertia_only=True,column_traction_loads_included=False,all_physical_column_six_forces_required=True,
        primary_force_composition_authorized=False,qualification_asserted=False)
    mapping['input_identity']=identity(mapping)
    return dict(loads=matrices,case_ids=case_ids,input_identity=mapping['input_identity'],mapping=mapping)

def prepare_coupled_floor_cases(adapter,floor,influences,witness_id,*,conventional_forces,
                                selected_case_ids=None,include_residual_sign_sensitivity=False):
    """Prepare signed conventional P5 forces for the actual coupled floor.

    No 2D membrane response is solved or imported. Full actual coupled A/B
    stiffness and column restraint must recover N,M,Q and added column forces.
    All inertia/traction forces act at the declared slab-midplane load datum.
    """
    from Design.SMRF_Physical_Diaphragm import identity
    _same_floor_geometry(adapter,floor)
    prepared=adapter.prepare_load_cases(influences,witness_id,conventional_forces=conventional_forces,
        include_residual_sign_sensitivity=include_residual_sign_sensitivity)
    components={};component_records=[]
    for row in prepared['components']:
        f=sum(np.asarray(row[k]) for k in ('inertia','columns_above_on_floor','columns_below_on_floor'))
        vector,audit=map_planar_nodal_forces(floor,prepared['mesh'],f);components[row['id']]=vector
        component_records.append(dict(id=row['id'],role=row['role'],action_origin=row['action_origin'],
            provenance=row['provenance'],physical_nodal_force_sha256=identity(f.tolist()),**audit))
    all_ids=[row['id'] for row in prepared['force_case_definitions']]
    if selected_case_ids is not None and (not selected_case_ids or len(set(selected_case_ids))!=len(selected_case_ids) or not set(selected_case_ids)<=set(all_ids)):
        raise ValueError('Requested P5 cases must be a nonempty unique subset of actual declared cases')
    selected=set(all_ids if selected_case_ids is None else selected_case_ids)
    rows=[r for r in prepared['force_case_definitions'] if r['id'] in selected]
    vectors=np.column_stack([sum(r['factors'].get(key,0.)*value for key,value in components.items()) for r in rows])
    case_ids=[f'{witness_id}::{r["id"]}' for r in rows]
    mapping=dict(floor_id=floor.id,retained_dof_ids=floor.dof_ids,
        retained_node_positions=[{'id':key,'xyz':floor.node_positions[key]} for key in floor.node_ids],
        adapter_context=adapter.context,influences_sha256=prepared['influences_sha256'],witness_id=witness_id,
        component_records=component_records,case_definitions=[{**r,'coupled_case_id':key} for r,key in zip(rows,case_ids)],
        omitted_case_ids=[key for key in all_ids if key not in selected],load_datum='actual slab midplane z=0; no invented centroid strong-axis couple',
        load_matrix_sha256=identity(vectors.tolist()),physical_nodal_reconstruction_max_abs_kip=prepared['physical_nodal_reconstruction_max_abs_kip'],
        linearization_replay=prepared['linearization_replay'],all_additional_column_six_forces_required=True,qualification_asserted=False)
    mapping['input_identity']=identity(mapping)
    return dict(floor_id=floor.id,loads=vectors.tolist(),case_ids=case_ids,input_identity=mapping['input_identity'],mapping=mapping)

def eccentric_web_embedding(mesh,case,member_geometry,*,slab_thickness_in):
    """Recover physical web end wrenches and out-of-plane constraint reactions.

    Geometry rows are actual member_id -> {h_in,b_in}; z=0 is slab midplane.
    Web centroid z=-h/2 follows from slab top +hs/2 and clear web below slab.
    Reaction is K*u on held rotations. Releasing those rotations requires its
    NEGATIVE as the correction RHS, and the complete coupled operator to
    recover changes in both membrane and bending fields. No closure claimed.
    """
    xy=np.asarray(mesh['nodes'],float);rows={str(r['id']):r for r in case['beam_axial']}
    if len(rows)!=len(case['beam_axial']) or set(rows)!={str(b['id']) for b in mesh['beams']}:
        raise ValueError('Complete unique actual web force roster required')
    reaction=np.zeros((len(xy),6));physical=np.zeros_like(reaction);segments=[]
    for b in mesh['beams']:
        ident=str(b.get('member_id',b['id']));geometry=member_geometry[ident]
        height,width=float(geometry['h_in']),float(geometry['b_in'])
        if height<=slab_thickness_in or width<=0 or not np.isfinite([height,width]).all():raise ValueError('Positive actual disjoint web required')
        if not np.isclose(b['area_web_in2'],width*(height-slab_thickness_in),rtol=1e-12,atol=1e-12):raise ValueError('Actual web area differs from physical geometry')
        i,j=b['i'],b['j'];delta=xy[j]-xy[i];length=float(np.linalg.norm(delta));direction=delta/length
        force=float(rows[str(b['id'])]['N_kip'])*np.r_[direction,0.];offset=np.array([0.,0.,-height/2])
        positive=np.r_[force,np.cross(offset,force)]
        np.add.at(physical,[i,j],[-positive,positive])
        moment=np.r_[np.zeros(3),positive[3:]]
        np.add.at(reaction,[i,j],[-moment,moment])
        segments.append(dict(segment_id=str(b['id']),member_id=ident,i=i,j=j,centroid_z_in=float(offset[2]),
                             N_kip=float(rows[str(b['id'])]['N_kip']),positive_end_wrench_at_slab_plane=positive.tolist()))
    return dict(method=METHOD,segments=segments,physical_web_internal_nodal_wrenches=physical.tolist(),
        held_rotation_reactions= reaction.tolist(),released_rotation_correction_loads=(-reaction).tolist(),
        maximum_nodal_constraint_moment_kipin=float(np.max(np.abs(reaction[:,3:5]),initial=0.)),
        summed_constraint_moments_kipin=reaction[:,3:].sum(axis=0).tolist(),
        actual_support_path_verified=False,released_physical_field_verified=False,qualification_asserted=False)

def eccentric_web_cut(mesh,embedding,axis,at):
    """Actual positive-face web force and extra Mx/My from the physical z."""
    normal={'x':0,'y':1}.get(axis)
    if normal is None:raise ValueError('Cut axis must be x or y')
    xy=np.asarray(mesh['nodes']);total=np.zeros(6);members=[]
    for row in embedding['segments']:
        p,q=xy[row['i']],xy[row['j']]
        if abs(q[1-normal]-p[1-normal])>1e-10 or not min(p[normal],q[normal])<at<max(p[normal],q[normal]):continue
        wrench=np.asarray(row['positive_end_wrench_at_slab_plane'])*np.sign(q[normal]-p[normal])
        point=p+(at-p[normal])/(q[normal]-p[normal])*(q-p)
        wrench[5]+=point[0]*wrench[1]-point[1]*wrench[0]
        total+=wrench;members.append(row['member_id'])
    return dict(axis=axis,coordinate_in=at,physical_web_wrench_at_slab_origin=total.tolist(),member_ids=members,
                missing_out_of_plane_Mx_My_kipin=total[3:5].tolist(),qualification_asserted=False)
