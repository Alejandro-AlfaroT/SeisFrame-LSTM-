"""Complete source-bound prescribed-primary P5 floor/load-family batching.

One factorization per actual floor mesh, including every signed original
seismic combination and its exact gravity-specific primary influence basis.
No column traction is invented, no physical G is added here, and no planar
force state is presented as a three-dimensional displacement solution.
"""
from __future__ import annotations
import math
import time
import gzip
import hashlib
import json
import numpy as np
from pathlib import Path
from Design.SMRF_Coupled_Floor_Design import validate_definition,physical_grid
from Design.SMRF_Diaphragm_Actions import PreparedDiaphragmOperator
from Design.SMRF_Physical_Diaphragm import (PhysicalDiaphragm,identity,conventional_force_inputs,
    physical_membrane_strip_cut)

METHOD='complete_prescribed_primary_floor_gravity_witness_batches_v2_base_elf'


def _witness_forces(factors,combination):
    return dict(factors,witness_strength_factor=max(abs(float(combination['ex'])),abs(float(combination['ey']))))


def _inventory(common):
    definition=common['definition'];validate_definition(definition)
    if (common.get('definition_sha256')!=definition['identity_sha256'] or common.get('state_identity')!=definition['state_identity'] or
        not common.get('input_signature') or common.get('analysis_source_identity')!=identity(common['source_identity'])):
        raise ValueError('Common physical definition/state/source/input binding differs')
    combinations=common['combinations'];ids=[c['id'] for c in combinations]
    if not ids or len(set(ids))!=len(ids) or set(common['gravity_case_mapping'])!=set(ids):
        raise ValueError('Complete unique original combination/gravity mapping required')
    seismic=[c for c in combinations if c['ex'] or c['ey']];groups={}
    for c in seismic:groups.setdefault(common['gravity_case_mapping'][c['id']],[]).append(c)
    if not seismic or set(groups)!=set(common['p5_influences']):raise ValueError('Every actual seismic gravity family needs its own prescribed-primary influence record')
    mass_hash=identity(common['grid_masses'])
    weights=common['elf']['story_weights_kip'];nf=definition['geometry']['nf']
    if len(weights)!=nf or any(not math.isclose(float(weights[k-1]),sum(float(r['weight_kip']) for r in common['grid_masses'][str(k)]),rel_tol=1e-10,abs_tol=1e-10) for k in range(1,nf+1)):
        raise ValueError('ELF floor seismic weights differ from the actual physical mass snapshot')
    for gid,cases in groups.items():
        record=common['p5_influences'][gid]
        actual=[w['id'] for w in record['witnesses']]
        if sorted(actual)!=sorted(c['id'] for c in cases) or len(set(actual))!=len(actual):
            raise ValueError('P5 witness roster differs from original signed combinations')
        by_id={w['id']:w for w in record['witnesses']}
        for case in cases:
            loads=np.asarray(by_id[case['id']]['loads'],float)
            required=np.asarray(common['elf']['story_forces_kip'],float)[:,None]*np.array([case['ex'],case['ey']])
            if loads.shape!=(nf,3) or not np.isfinite(loads).all() or not np.allclose(loads[:,:2],required,rtol=1e-11,atol=1e-10):
                raise ValueError('P5 witness forces differ from actual signed original ELF combination')
            if 'primary' in common:
                increment=common['primary'][gid]['increments'][case['id']]
                if not np.array_equal(loads,np.asarray(increment['floor_loads'],float)):
                    raise ValueError('P5 witness wrench differs from its actual primary frame increment')
        if (record.get('protocol_id')!=common['input_signature'] or record.get('physical_gravity_case_id')!=gid or
            record.get('source_identity')!=common['analysis_source_identity'] or record.get('nodal_mass_sha256')!=mass_hash or
            record.get('physical_definition_sha256')!=definition['identity_sha256'] or record.get('state_identity')!=definition['state_identity']):
            raise ValueError('P5 influence gravity/source/physical mass/definition binding differs')
        if identity(record['grid_mass_by_floor'])!=mass_hash:raise ValueError('P5 influence actual mass payload differs from its declared binding')
        gravity=record['gravity_combination']
        if any((float(c['dead']),float(c['live']),c.get('live_pattern','all'))!=
               (float(gravity['dead']),float(gravity['live']),gravity.get('live_pattern','all')) for c in cases):
            raise ValueError('P5 witness was computed under a different simultaneous gravity family')
    return definition,seismic,groups


def estimate_primary_diaphragm_batch(common,*,subdivisions_per_bay,maximum_nodes_per_floor,
                                     mesh_axes_by_floor=None,include_residual_sign_sensitivity=False):
    """No matrix assembly or solve: count exact mesh/case/output dimensions.

    The retained-array estimate includes Python float/list overhead at48bytes
    per scalar and omits redundant GP clouds and component nodal arrays.
    Sparse LU memory is explicitly unknown until the actual mesh is measured;
    this estimate must never be advertised as a complete memory bound.
    """
    definition,seismic,groups=_inventory(common);g=definition['geometry'];rows=[];count=10 if include_residual_sign_sensitivity else 7
    if type(maximum_nodes_per_floor) is not int or maximum_nodes_per_floor<4:raise ValueError('Explicit positive physical mesh node bound required')
    if mesh_axes_by_floor is not None and set(mesh_axes_by_floor)!=set(range(1,g['nf']+1)):
        raise ValueError('Explicit physical axes must cover every floor')
    for desc in definition['floors']:
        x,y=physical_grid(desc,g,subdivisions_per_bay=subdivisions_per_bay) if mesh_axes_by_floor is None else mesh_axes_by_floor[desc['level']]
        nodes=len(x)*len(y);cells=(len(x)-1)*(len(y)-1)
        if nodes>maximum_nodes_per_floor:raise ValueError('Actual P5 mesh exceeds the declared node bound')
        # An upper estimate for truss segments includes every aligned grid line.
        web_segments=len(x)*(len(y)-1)+len(y)*(len(x)-1)
        scalars=len(seismic)*count*(2*nodes+web_segments)+4*nodes+4*cells
        rows.append(dict(floor=desc['level'],nodes=nodes,cells=cells,signed_witnesses=len(seismic),force_cases=len(seismic)*count,
            estimated_retained_scalar_count=scalars,estimated_retained_python_bytes=48*scalars,axes_sha256=identity([list(x),list(y)])))
    return dict(method=METHOD,common_input_signature=common['input_signature'],definition_sha256=definition['identity_sha256'],
        floors=rows,total_force_cases=sum(r['force_cases'] for r in rows),factorizations=len(rows),
        estimated_retained_python_bytes=sum(r['estimated_retained_python_bytes'] for r in rows),
        sparse_factor_memory_included=False,scope='dimension estimate only; measured factor memory and time still required')


def analyze_primary_diaphragms(common,*,sds,importance_factor,omega0,seismic_live_load_fraction,
        subdivisions_per_bay,maximum_nodes_per_floor,demand_factors_by_floor,
        max_witness_floor_pairs,max_wall_seconds,mesh_axes_by_floor=None,
        membrane_stiffness_factor=1.,web_axial_stiffness_factor=1.,include_residual_sign_sensitivity=False,
        progress=None,artifact_directory=None):
    """Return every prescribed-primary signed floor field for common strength."""
    started=time.monotonic();definition,seismic,groups=_inventory(common);nf=definition['geometry']['nf']
    if (type(max_witness_floor_pairs) is not int or max_witness_floor_pairs<nf*len(seismic) or
        not math.isfinite(float(max_wall_seconds)) or max_wall_seconds<=0):raise ValueError('Explicit sufficient batch pair count and positive finite wall bound required')
    if set(demand_factors_by_floor)!=set(range(1,nf+1)):raise ValueError('Actual transfer/irregularity force factors required on every floor')
    for row in demand_factors_by_floor.values():
        if set(row)!={'transfer_factor','irregularity_factor'} or any(not math.isfinite(float(v)) or float(v)<1 for v in row.values()):
            raise ValueError('Explicit valid conventional force amplification factors required')
    estimate=estimate_primary_diaphragm_batch(common,subdivisions_per_bay=subdivisions_per_bay,
        maximum_nodes_per_floor=maximum_nodes_per_floor,mesh_axes_by_floor=mesh_axes_by_floor,
        include_residual_sign_sensitivity=include_residual_sign_sensitivity)
    binding=dict(common_input_signature=common['input_signature'],source_identity=common['analysis_source_identity'],
        state_identity=definition['state_identity'],definition_sha256=definition['identity_sha256'],
        nodal_mass_sha256=identity(common['grid_masses']),combination_inventory_sha256=identity(common['combinations']),
        p5_influences_sha256={gid:identity(r) for gid,r in common['p5_influences'].items()})
    result=dict(method=METHOD,**binding,floors={},estimate=estimate,qualification_asserted=False,official_pilot_started=False,
        prescribed_primary_load_path_retained=True,physical_out_of_plane_compatibility_claimed=False,
        mesh_qualification_inferred=False,analysis_parameters=dict(sds=sds,importance_factor=importance_factor,omega0=omega0,
            seismic_live_load_fraction=seismic_live_load_fraction,subdivisions_per_bay=subdivisions_per_bay,
            maximum_nodes_per_floor=maximum_nodes_per_floor,membrane_stiffness_factor=membrane_stiffness_factor,
            web_axial_stiffness_factor=web_axial_stiffness_factor,
            include_residual_sign_sensitivity=include_residual_sign_sensitivity,equilibrium_tolerance=1e-8))
    destination=None if artifact_directory is None else Path(artifact_directory).resolve()
    if destination is not None:destination.mkdir(parents=True,exist_ok=True)
    completed=0
    for floor in range(1,nf+1):
        if time.monotonic()-started>max_wall_seconds:raise RuntimeError('Declared P5 batch wall bound reached before the next floor; numerical outcome unresolved')
        adapter=PhysicalDiaphragm(definition,floor,grid_mass_by_floor=common['grid_masses'],
            seismic_live_load_fraction=seismic_live_load_fraction,source_identity=common['analysis_source_identity'],
            subdivisions_per_bay=subdivisions_per_bay,mesh_axes=None if mesh_axes_by_floor is None else mesh_axes_by_floor[floor],
            maximum_nodes=maximum_nodes_per_floor,membrane_stiffness_factor=membrane_stiffness_factor,
            web_axial_stiffness_factor=web_axial_stiffness_factor)
        operator=PreparedDiaphragmOperator(adapter.mesh);witnesses={}
        factors=conventional_force_inputs(common['elf'],floor,sds=sds,importance_factor=importance_factor,omega0=omega0,
            **demand_factors_by_floor[floor])
        for combination in seismic:
            if time.monotonic()-started>max_wall_seconds:raise RuntimeError('Declared P5 batch wall bound reached before the next signed witness; numerical outcome unresolved')
            cid=combination['id'];gid=common['gravity_case_mapping'][cid]
            value=adapter.solve(common['p5_influences'][gid],cid,conventional_forces=_witness_forces(factors,combination),operator=operator,
                retain_slab_samples=False,include_residual_sign_sensitivity=include_residual_sign_sensitivity)
            value['mapped_component_arrays_sha256']=identity(value.pop('components'))
            value.pop('mesh');value['gravity_case_id']=gid;value['combination']=combination
            witnesses[cid]=value;completed+=1
            if progress:progress(dict(stage='prescribed_primary_floor_fields',floor=floor,combination_id=cid,
                completed_witness_floor_pairs=completed,total_witness_floor_pairs=nf*len(seismic),elapsed_seconds=time.monotonic()-started))
        widths={str(b['id']):float(b['b_in']) for b in adapter.description['beams']}
        record=dict(mesh=adapter.mesh,adapter=adapter.context,web_widths_in=widths,
            witnesses=witnesses,prepared_operator=operator.record(),force_factors=factors)
        record['payload_sha256']=identity(record)
        if destination is None:result['floors'][str(floor)]=record
        else:
            path=destination/f'primary_p5_floor_{floor:03d}.json.gz'
            if path.exists():raise ValueError('P5 artifact paths must be fresh; an existing field is never overwritten')
            with gzip.open(path,'wt',encoding='utf-8') as stream:json.dump(record,stream,allow_nan=False,separators=(',',':'))
            result['floors'][str(floor)]=dict(artifact_file=str(path),artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                payload_sha256=record['payload_sha256'],floor=floor,adapter=adapter.context)
        # The sparse factor belongs only to this floor; it is not accumulated
        # while the next floor is assembled. Results retain no operator object.
        del operator,record,witnesses
    result.update(completed_witness_floor_pairs=completed,elapsed_seconds=time.monotonic()-started,status='responses_computed')
    result['payload_sha256']=identity(result);return result


def _cut(batch,row,combination_id,branch_id,*,axis,station_in,interval_in,trace,transverse_reference_in):
    """Actual P5 core/remote/full-flange source for the common strength adapter."""
    if batch.get('method')!=METHOD or batch.get('status')!='responses_computed':raise ValueError('Complete actual prescribed-primary batch required')
    witness=row['witnesses'][combination_id]
    context=witness['adapter']
    if context!=row['adapter'] or context['definition_sha256']!=batch['definition_sha256'] or context['state_identity']!=batch['state_identity']:
        raise ValueError('P5 floor/witness physical identity differs')
    case=next((c for c in witness['cases'] if c['id']==branch_id),None)
    if case is None:raise ValueError('Requested P5 force branch is absent from its signed original combination')
    cut=physical_membrane_strip_cut({**witness,'mesh':row['mesh']},branch_id,axis=axis,station_in=station_in,
        interval_in=interval_in,trace=trace,transverse_reference_in=transverse_reference_in,web_widths_in=row['web_widths_in'])
    residual=case['force_moment_balance'];tolerance=case['force_moment_tolerance'];cut.update(combination_id=combination_id,gravity_case_id=witness['gravity_case_id'],
        branch_id=branch_id,common_input_signature=batch['common_input_signature'],
        floor_freebody_residual=[residual[0],residual[1],0.,0.,0.,residual[2]],
        floor_freebody_tolerance=[tolerance[0],tolerance[1],0.,0.,0.,tolerance[2]],
        floor_freebody_tolerance_basis=case['force_moment_tolerance_basis'],
        floor_freebody_equilibrium_audits=witness['component_equilibrium'],
        normalized_nodal_residual=case['normalized_nodal_residual'],prepared_operator=row['prepared_operator'])
    cut['identity_sha256']=identity({k:v for k,v in cut.items() if k!='identity_sha256'});return cut


class PrimaryDiaphragmReadout:
    """Integrity-checked common-cut reader caching only one floor at a time."""
    def __init__(self,batch):
        if batch.get('payload_sha256')!=identity({k:v for k,v in batch.items() if k!='payload_sha256'}):
            raise ValueError('Prescribed-primary batch payload identity differs')
        self.batch=batch;self._floor=None;self._record=None;self._axial_action_windows={}

    def floor(self,floor):
        floor=int(floor)
        if floor==self._floor:return self._record
        value=self.batch['floors'][str(floor)]
        if 'artifact_file' in value:
            path=Path(value['artifact_file'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=value['artifact_sha256']:raise ValueError('Physical P5 floor artifact content hash differs')
            with gzip.open(path,'rt',encoding='utf-8') as stream:record=json.load(stream)
        else:record=value
        expected=record.get('payload_sha256')
        if expected!=value.get('payload_sha256') or expected!=identity({k:v for k,v in record.items() if k!='payload_sha256'}):
            raise ValueError('Physical P5 floor payload hash differs')
        if record['adapter']['floor']!=floor:raise ValueError('Physical P5 floor artifact identity differs')
        self._floor=floor;self._record=record;self._axial_action_windows={};return record

    def cut(self,floor,combination_id,branch_id,**cut):return _cut(self.batch,self.floor(floor),combination_id,branch_id,**cut)


def primary_diaphragm_cut(batch,floor,combination_id,branch_id,**cut):
    """One-off verified readout; reuse PrimaryDiaphragmReadout for many cuts."""
    return PrimaryDiaphragmReadout(batch).cut(floor,combination_id,branch_id,**cut)


def replay_primary_diaphragms(batch,common,*,sds,importance_factor,omega0,seismic_live_load_fraction,
                             demand_factors_by_floor):
    """Rebuild actual signed free bodies and K@u without a factorization/solve.

    Source influences, actual geometry, mass, every original signed case and
    every prescribed branch are checked independently of saved pass flags.
    Recovered web N is recomputed from the complete displacement field. This
    numerical certificate does not establish mesh convergence or strength.
    """
    from Design import SMRF_Diaphragm_Actions as mechanics
    definition,seismic,groups=_inventory(common);nf=definition['geometry']['nf']
    reader=PrimaryDiaphragmReadout(batch);parameters=batch['analysis_parameters']
    expected=dict(common_input_signature=common['input_signature'],source_identity=common['analysis_source_identity'],
        state_identity=definition['state_identity'],definition_sha256=definition['identity_sha256'],
        nodal_mass_sha256=identity(common['grid_masses']),combination_inventory_sha256=identity(common['combinations']),
        p5_influences_sha256={gid:identity(r) for gid,r in common['p5_influences'].items()})
    if (batch.get('method')!=METHOD or batch.get('status')!='responses_computed'
        or any(batch.get(k)!=v for k,v in expected.items())
        or set(batch['floors'])!={str(k) for k in range(1,nf+1)}):
        raise ValueError('P5 batch complete actual source/input/floor binding differs')
    for key,value in dict(sds=sds,importance_factor=importance_factor,omega0=omega0,seismic_live_load_fraction=seismic_live_load_fraction).items():
        if parameters.get(key)!=value:raise ValueError('P5 batch prescribed force/mass policy differs')
    tolerance=parameters['equilibrium_tolerance']
    if tolerance!=1e-8 or set(demand_factors_by_floor)!=set(range(1,nf+1)):
        raise ValueError('Original P5 numerical tolerance and complete floor force factors required')
    rows=[];maximum_residual=0.;maximum_web_error=0.
    for floor in range(1,nf+1):
        stored=reader.floor(floor);mesh=stored['mesh']
        physical=PhysicalDiaphragm(definition,floor,grid_mass_by_floor=common['grid_masses'],
            seismic_live_load_fraction=seismic_live_load_fraction,source_identity=common['analysis_source_identity'],
            subdivisions_per_bay=parameters['subdivisions_per_bay'],mesh_axes=[mesh['x_in'],mesh['y_in']],
            maximum_nodes=parameters['maximum_nodes_per_floor'],membrane_stiffness_factor=parameters['membrane_stiffness_factor'],
            web_axial_stiffness_factor=parameters['web_axial_stiffness_factor'])
        if identity(physical.mesh)!=identity(mesh) or physical.context!=stored['adapter']:
            raise ValueError('P5 saved operator differs from rebuilt actual physical mesh/source')
        operator=stored['prepared_operator']
        if (operator.get('mesh_sha256')!=identity(mesh)
            or operator.get('source_sha256')!=hashlib.sha256(Path(mechanics.__file__).read_bytes()).hexdigest()):
            raise ValueError('P5 prepared operator source binding is stale')
        factors=conventional_force_inputs(common['elf'],floor,sds=sds,importance_factor=importance_factor,omega0=omega0,
            **demand_factors_by_floor[floor])
        if factors!=stored['force_factors']:raise ValueError('P5 saved conventional floor force factors differ')
        if set(stored['witnesses'])!={c['id'] for c in seismic}:raise ValueError('P5 saved signed witness inventory is incomplete')
        k,_=mechanics._assemble(mesh);xy=np.asarray(mesh['nodes'],float);n=len(xy)
        for combination in seismic:
            cid=combination['id'];gid=common['gravity_case_mapping'][cid];witness=stored['witnesses'][cid]
            rebuilt=physical.prepare_load_cases(common['p5_influences'][gid],cid,conventional_forces=_witness_forces(factors,combination),
                include_residual_sign_sensitivity=parameters['include_residual_sign_sensitivity'])
            if rebuilt['conventional_force_basis']!=witness.get('conventional_force_basis'):
                raise ValueError('P5 conventional base-force reconstruction differs')
            if (witness.get('combination')!=combination or witness.get('gravity_case_id')!=gid
                or witness.get('mapped_component_arrays_sha256')!=identity(rebuilt['components'])
                or witness.get('influences_sha256')!=identity(common['p5_influences'][gid])
                or witness.get('force_case_definitions')!=rebuilt['force_case_definitions']):
                raise ValueError('P5 saved original combination/component force decomposition differs')
            vectors={};audits={}
            for component in rebuilt['components']:
                vector,audit=mechanics._component(mesh,component,tolerance);vectors[component['id']]=vector;audits[component['id']]=audit
            definitions={c['id']:c for c in rebuilt['force_case_definitions']};cases={c['id']:c for c in witness['cases']}
            if len(cases)!=len(witness['cases']) or set(cases)!=set(definitions):
                raise ValueError('P5 saved conventional/direct force branch inventory is incomplete')
            for branch_id,description in definitions.items():
                case=cases[branch_id]
                if case['factors']!=description['factors']:raise ValueError('P5 saved branch factors differ')
                force=sum(float(multiplier)*vectors[key] for key,multiplier in description['factors'].items())
                u=np.asarray(case['displacements_in'],float)
                if u.shape!=(n,2) or not np.isfinite(u).all():raise ValueError('P5 saved displacement field is incomplete/nonfinite')
                residual=k@u.ravel()-force.ravel();scale=max(float(np.max(abs(force))),1.)
                error=float(np.max(abs(residual)));relative=error/scale
                balance=mechanics.resultant(xy,force)
                limits=np.array([sum(abs(multiplier)*audits[key]['equilibrium_tolerance']*audits[key]['force_normalization_kip']
                    *(1. if j<2 else audits[key]['moment_normalization_length_in']+sum(abs(v) for v in audits[key]['resultant_reference_xy_in']))
                    for key,multiplier in description['factors'].items()) for j in range(3)])
                if relative>tolerance or np.any(abs(balance)>limits):
                    raise ValueError(f'P5 numeric replay failed floor {floor}/{cid}/{branch_id}: Ku-F {relative:g}, free body {balance.tolist()}')
                if not np.allclose(case['force_moment_balance'],balance,rtol=0.,atol=1e-12) or not np.allclose(case['force_moment_tolerance'],limits,rtol=1e-14,atol=0.):
                    raise ValueError('P5 saved free-body residual/tolerance differs from raw reconstruction')
                beams={b['id']:b for b in case['beam_axial']}
                if len(beams)!=len(case['beam_axial']) or set(beams)!={b['id'] for b in mesh['beams']}:
                    raise ValueError('P5 saved complete web segment inventory differs')
                web_error=0.
                for b in mesh['beams']:
                    delta=xy[b['j']]-xy[b['i']];length=float(np.linalg.norm(delta))
                    axial=b['E_ksi']*b['stiffness_factor']*b['area_web_in2']/length*float(delta/length@(u[b['j']]-u[b['i']]))
                    old=beams[b['id']]
                    if (old['member_id']!=b.get('member_id',b['id']) or old['i']!=b['i'] or old['j']!=b['j']
                        or not math.isfinite(float(old['N_kip']))):raise ValueError('P5 saved actual web identity/force differs')
                    mismatch=abs(axial-old['N_kip']);web_error=max(web_error,mismatch)
                    if mismatch>1e-10*max(1.,abs(axial)):raise ValueError('P5 saved web force does not reproduce displacement field')
                maximum_residual=max(maximum_residual,relative);maximum_web_error=max(maximum_web_error,web_error)
                rows.append(dict(floor=floor,combination_id=cid,gravity_case_id=gid,branch_id=branch_id,
                    maximum_nodal_residual_kip=error,nodal_force_normalization_kip=scale,normalized_nodal_residual=relative,
                    equilibrium_tolerance=tolerance,force_moment_balance=balance.tolist(),force_moment_tolerance=limits.tolist(),
                    maximum_recovered_web_error_kip=web_error,web_segments=len(beams)))
        del k,physical
    expected_cases=nf*len(seismic)*(10 if parameters['include_residual_sign_sensitivity'] else 7)
    if len(rows)!=expected_cases:raise ValueError('Incomplete P5 replay case coverage')
    result=dict(method='reconstructed_actual_P5_load_operator_displacement_equilibrium_v1',batch_payload_sha256=batch['payload_sha256'],
        **expected,rows=rows,force_cases_checked=len(rows),assemblies=nf,factorizations=0,new_solves=0,
        maximum_normalized_nodal_residual=maximum_residual,maximum_recovered_web_error_kip=maximum_web_error,
        numerical_equilibrium_verified=True,mesh_convergence_verified=False,strength_verified=False)
    result['payload_sha256']=identity(result);return result
