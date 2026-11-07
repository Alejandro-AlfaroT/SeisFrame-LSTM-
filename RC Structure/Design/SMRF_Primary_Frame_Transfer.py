"""Retained released ELF frame with an explicit, force-verified gravity baseline.

ACI318-19 6.3.1.1 consistent-stiffness/model premise and8.4.1.9 separate
gravity/lateral analysis support this declared analytical transfer. It is not
a code-prescribed load formula. The physical coupled gravity demands remain
separate. Auxiliary baseline loads initialize the retained primary frame's
actual column axial state; they cancel from every reported ELF increment and
must never be credited as extra physical gravity or diaphragm design loads.

Primary member sections/releases are frozen directly from SMRF_Elastic's
installed inputs. Beam stiffness is linear; columns use the independently
verified native PDelta law. Full physical column force verification, rigid
projection, auxiliary-load wrench and tangent comparison remain explicit.
"""
from __future__ import annotations
import hashlib
import json
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from Design.SMRF_Substructure_Frame import ElasticColumn,SubstructureFrame

METHOD='retained_ELF_force_verified_gravity_initial_stress_transfer_v1'


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def installed_primary_definition(physical_definition,gravity_case):
    from Design.SMRF_Coupled_Floor_Design import validate_definition
    from Design.SMRF_Elastic import physical_members,beam_end_releases
    from Design import Perimeter_Systems as systems
    from Model import Member_Properties as mp,Member_Groups as mg
    import Structure_Parameters as sp
    validate_definition(physical_definition)
    state=mg.active()
    if state is None or state.identity()!=physical_definition['state_identity']:
        raise ValueError('Primary and physical gravity definitions must describe the same installed candidate')
    beams=[];columns=[];model=systems.design_model_for()
    previous=sp.LATERAL_DESIGN_MODEL
    try:
        sp.LATERAL_DESIGN_MODEL=model
        for tag,ni,nj,kind in physical_members():
            member=mg.resolve(tag);p=mp.elastic_section(member);factor=p['stiffness_modifier']
            row=dict(id=str(tag),i=ni,j=nj,A=p['area'],E=p['e'],G=p['g'],J=factor*p['j'],
                     Iy=factor*p['iy'],Iz=factor*p['iz'],vecxz=[1.,0.,0.] if kind=='column' else [0.,0.,1.],
                     kind=kind,level=member.story_or_floor,group_id=member.group_id)
            if kind=='column':
                row['axial_load_kip_per_in']=-float(gravity_case['dead'])*mp.column_self_weight_kip_per_in(member)
                columns.append(row)
            else:
                release=beam_end_releases(tag,kind)
                row.update(releasey=3 if '-releasey' in release else 0,releasez=3 if '-releasez' in release else 0)
                beams.append(row)
    finally:sp.LATERAL_DESIGN_MODEL=previous
    result=dict(method=METHOD,physical_definition_sha256=physical_definition['identity_sha256'],
                state_identity=state.identity(),lateral_design_model=model,nodes=physical_definition['nodes'],
                geometry=physical_definition['geometry'],beams=beams,columns=columns,gravity_case=dict(gravity_case))
    result['identity_sha256']=_hash(result)
    return result


class PrimaryFrame:
    def __init__(self,definition,*,source_identity):
        if not source_identity or definition.get('identity_sha256')!=_hash({k:v for k,v in definition.items() if k!='identity_sha256'}):
            raise ValueError('Primary frame needs exact frozen content/source identities')
        self.definition=definition;self.source_identity=source_identity;self.nodes=definition['nodes'];g=definition['geometry']
        byid={n['id']:n for n in self.nodes};self.dof_ids=[(n['id'],d) for n in self.nodes for d in range(1,7)]
        self.index={v:i for i,v in enumerate(self.dof_ids)};self.interfaces=[];self.beams=[]
        self.master_xy=[g['nx']*g['lx']/2,g['ny']*g['ly']/2]
        for floor in range(1,g['nf']+1):
            nodes=[n for n in self.nodes if n['xyz'][2]==floor*g['story_h']]
            dofs=[(n['id'],d) for n in nodes for d in range(1,7)];index={v:i for i,v in enumerate(dofs)}
            k=sparse.lil_matrix((len(dofs),len(dofs)))
            for b in definition['beams']:
                if b['level']!=floor:continue
                element=ElasticColumn(b,byid[b['i']]['xyz'],byid[b['j']]['xyz'])
                local=np.array([index[(b[end],d)] for end in ('i','j') for d in range(1,7)])
                k[np.ix_(local,local)]+=element.stiffness
                global_indices=np.array([self.index[(b[end],d)] for end in ('i','j') for d in range(1,7)])
                self.beams.append((element,global_indices))
            self.interfaces.append(dict(id=f'primary_floor_{floor}',dof_ids=dofs,stiffness=k.tocsr(),
                loads=np.zeros(len(dofs)),input_identity={'primary':definition['identity_sha256'],'floor':floor}))
        # Exactly the existing rigid floor ux,uy,rz kinematics; every physical
        # vertical displacement and rotation about X/Y remains independent.
        reduced=[];rr=[];cc=[];vv=[]
        def add(column,terms):
            for dof,value in terms:
                if value:rr.append(self.index[dof]);cc.append(column);vv.append(value)
        for floor in range(1,g['nf']+1):
            nodes=[n for n in self.nodes if n['xyz'][2]==floor*g['story_h']]
            for component in (1,2,6):
                at=len(reduced);reduced.append((f'master_{floor}',component));terms=[]
                for n in nodes:
                    x,y,_=n['xyz'];tag=n['id']
                    if component in (1,2):terms.append(((tag,component),1.))
                    else:terms.extend([((tag,1),-(y-self.master_xy[1])),((tag,2),x-self.master_xy[0]),((tag,6),1.)])
                add(at,terms)
            for n in nodes:
                for d in (3,4,5):
                    at=len(reduced);reduced.append((n['id'],d));add(at,[((n['id'],d),1.)])
        self.transform=sparse.csr_matrix((vv,(rr,cc)),shape=(len(self.dof_ids),len(reduced)))
        self.reduced_dof_ids=reduced;self._zero=self._frame(np.zeros(len(self.dof_ids)),identity='unloaded_geometry')
        self.baseline=None

    def _frame(self,nodal,*,identity):
        return SubstructureFrame(self.nodes,self.interfaces,self.definition['columns'],
            input_identity={'method':METHOD,'definition':self.definition['identity_sha256'],'source':self.source_identity,'state':identity},
            loads=nodal,kinematic_transform=self.transform,reduced_dof_ids=self.reduced_dof_ids)

    def transfer_gravity(self,physical_result,*,case_id,absolute_axial_tolerance_kip=1e-7,relative_axial_tolerance=1e-8,
                         saved_input_binding=None,saved_transfer=None):
        """Transfer actual P; an older saved result needs an explicit input hash.

        Default production use requires the same source identity. Reusing a
        completed diagnostic is supported only as a separately attributed,
        content-bound input: the prior source is never relabeled as current.
        This proves input integrity, not current-method qualification; all
        geometry, load factors, DOFs and both column-force checks still apply.
        """
        if physical_result.get('definition_sha256')!=self.definition['physical_definition_sha256']:
            raise ValueError('Physical gravity result belongs to a different frozen candidate')
        if saved_input_binding is not None:
            if (saved_input_binding.get('source_identity')!=physical_result.get('source_identity') or
                saved_input_binding.get('payload_sha256')!=_hash(physical_result) or
                not saved_input_binding.get('input_manifest_sha256')):
                raise ValueError('Saved gravity input binding does not match its actual content/source')
        elif physical_result.get('source_identity')!=self.source_identity:
            raise ValueError('Physical gravity result and primary transfer source identities differ')
        matches=[i for i,c in enumerate(physical_result['cases']) if c['id']==case_id]
        if len(matches)!=1 or physical_result.get('status') not in ('equilibrated','constitutive_equilibrated'):
            raise ValueError('One complete equilibrated physical simultaneous gravity case required')
        at=matches[0];target_state=physical_result['frame']['cases'][at]
        chosen=physical_result['cases'][at];declared=self.definition['gravity_case']
        if any(float(chosen[k])!=float(declared[k]) for k in ('dead','live')) or chosen.get('live_pattern','all')!=declared.get('live_pattern','all'):
            raise ValueError('Primary self-weight/gravity declaration differs from simultaneous physical case')
        mapping=dict(zip(map(tuple,physical_result['frame']['physical_dof_ids']),target_state['physical_displacements']))
        if set(mapping)!=set(self.dof_ids):raise ValueError('Physical/primary column node inventories differ')
        original=np.array([mapping[d] for d in self.dof_ids]);target=original.copy();g=self.definition['geometry'];projection=[]
        # Equal-node least squares is an explicit kinematic projection, not a
        # claimed physical mass redistribution. Vertical DOFs are untouched.
        for floor in range(1,g['nf']+1):
            nodes=[n for n in self.nodes if n['xyz'][2]==floor*g['story_h']]
            rows=[];values=[]
            for n in nodes:
                x,y,_=n['xyz'];rows.extend([[1,0,-(y-self.master_xy[1])],[0,1,x-self.master_xy[0]]])
                values.extend([mapping[(n['id'],1)],mapping[(n['id'],2)]])
            q=np.linalg.lstsq(np.array(rows),values,rcond=None)[0]
            for n in nodes:
                x,y,_=n['xyz'];target[self.index[(n['id'],1)]]=q[0]-(y-self.master_xy[1])*q[2]
                target[self.index[(n['id'],2)]]=q[1]+(x-self.master_xy[0])*q[2]
                target[self.index[(n['id'],6)]]=q[2]
            projection.append(dict(floor=floor,master_in_rad=q.tolist(),
                max_translation_projection_in=float(np.max(abs(np.array(rows)@q-values)))))
        geo=self._zero._geometric(target,0,tangent=False)[0]
        auxiliary=self._zero.stiffness@target+geo-self._zero.loads[:,0]
        for n in self.nodes:
            if n['xyz'][2]==0:
                for d in range(1,7):auxiliary[self.index[(n['id'],d)]]=0.
        frame=self._frame(auxiliary,identity={'physical_case':case_id,'physical_frame':physical_result['frame']['identity_sha256']})
        if saved_transfer is None:baseline=frame.solve(pdelta=True)
        else:
            frame.verify_saved(saved_transfer['result'],pdelta=True);baseline=saved_transfer['result']
        if baseline['status']!='equilibrated':raise ValueError('Transferred primary gravity baseline did not solve')
        state=baseline['cases'][0];ledgers=[];passed=True
        for tag,actual in state['columns'].items():
            if tag not in target_state['columns']:raise ValueError('Physical column missing from initial-stress ledger')
            expected=target_state['columns'][tag]
            a=np.asarray(actual['local_force'])[[0,6]];b=np.asarray(expected['local_force'])[[0,6]]
            tolerance=absolute_axial_tolerance_kip+relative_axial_tolerance*np.maximum(abs(a),abs(b))
            good=bool(np.all(abs(a-b)<=tolerance));passed &=good
            ledgers.append(dict(column_id=tag,target_signed_local_end_axials_kip=b.tolist(),
                                actual_signed_local_end_axials_kip=a.tolist(),error_kip=(a-b).tolist(),
                                tolerance_kip=tolerance.tolist(),passed=good))
        if not passed:raise ValueError('Solved primary baseline does not reproduce every actual physical column axial end force')
        wrench=np.zeros(6)
        for n in self.nodes:
            i=self.index[(n['id'],1)];f=auxiliary[i:i+6];wrench[:3]+=f[:3];wrench[3:]+=f[3:]+np.cross(n['xyz'],f[:3])
        self.baseline=dict(frame=frame,result=baseline,auxiliary_nodal_loads=auxiliary,
            physical_case_id=case_id,column_axial_ledger=ledgers,projection=projection,
            physical_result_source_identity=physical_result['source_identity'],
            saved_input_binding=saved_input_binding,
            auxiliary_global_wrench_kip_kipin=wrench.tolist(),
            target_displacement_error_in_rad=(np.asarray(state['physical_displacements'])-target).tolist())
        return {k:v for k,v in self.baseline.items() if k not in ('frame','auxiliary_nodal_loads')}

    def floor_load_vector(self,floor_loads):
        loads=np.asarray(floor_loads,float);g=self.definition['geometry']
        if loads.shape!=(g['nf'],3) or not np.isfinite(loads).all():raise ValueError('Actual signed floor Fx/Fy/Mz at existing master reference required')
        external=np.zeros(len(self.dof_ids))
        for floor,force in enumerate(loads,1):
            n=next(n for n in self.nodes if n['xyz'][2]==floor*g['story_h']);x,y,_=n['xyz']
            i=self.index[(n['id'],1)];external[i:i+2]=force[:2]
            external[i+5]=force[2]+(self.master_xy[0]-x)*force[1]-(self.master_xy[1]-y)*force[0]
        return external

    def incremental_reference(self,floor_loads):
        """Compare exact gravity-state tangent with a frozen-actual-P reference.

The latter suppresses deltaP times baseline lateral displacement, not the
actual initial P. Both are reported; no equality or safety bound is assumed.
Actual plus/minus direct solves independently verify the exact tangent.
"""
        if self.baseline is None:raise ValueError('An actual force-verified primary gravity baseline is required')
        frame=self.baseline['frame'];u=np.asarray(self.baseline['result']['cases'][0]['physical_displacements'])
        _,tangent=frame._geometric(u,0,tangent=True);fixed=sparse.lil_matrix(frame.stiffness.shape);ledger=[]
        for element,indices,weights in frame.columns:
            n=float(element.axial_derivative@u[indices]-weights[0]*element.length/2)
            fixed[np.ix_(indices,indices)]+=n*element.geometric_unit
            ledger.append(dict(column_id=str(element.id),actual_initial_basic_axial_tension_kip=n))
        rhs=np.asarray(self.transform.T@self.floor_load_vector(floor_loads)).ravel()
        exact=spsolve((self.transform.T@(frame.stiffness+tangent)@self.transform).tocsc(),rhs)
        frozen=spsolve((self.transform.T@(frame.stiffness+fixed.tocsr())@self.transform).tocsc(),rhs)
        return dict(method=METHOD,exact_tangent_displacement=(self.transform@exact).tolist(),
            frozen_actual_P_displacement=(self.transform@frozen).tolist(),initial_column_P=ledger,
            difference_displacement=(self.transform@(exact-frozen)).tolist(),
            qualification_asserted=False)

    def solve_increment(self,floor_loads,*,saved_increment=None):
        if self.baseline is None:raise ValueError('An actual force-verified primary gravity baseline is required')
        loads=np.asarray(floor_loads,float);external=self.floor_load_vector(loads)
        frame=self._frame(self.baseline['auxiliary_nodal_loads']+external,
                          identity={'physical_case':self.baseline['physical_case_id'],'floor_loads':loads.tolist()})
        if saved_increment is None:result=frame.solve(pdelta=True)
        else:
            frame.verify_saved(saved_increment['frame'],pdelta=True);result=saved_increment['frame']
            if not np.array_equal(np.asarray(saved_increment['floor_loads']),loads):raise ValueError('Saved primary signed floor loads differ')
        if result['status']!='equilibrated':return dict(status='unresolved',frame=result)
        state=result['cases'][0];base=self.baseline['result']['cases'][0]
        u=np.asarray(state['physical_displacements']);u0=np.asarray(base['physical_displacements'])
        columns={tag:{'global_force_increment':(np.asarray(r['global_resisting_force'])-base['columns'][tag]['global_resisting_force']).tolist(),
                      'local_force_increment':(np.asarray(r['local_force'])-base['columns'][tag]['local_force']).tolist(),
                      'initial_basic_axial_tension_kip':base['columns'][tag]['pdelta_basic_axial_tension_positive'],
                      'loaded_basic_axial_tension_kip':r['pdelta_basic_axial_tension_positive'],
                      'delta_basic_axial_tension_kip':r['pdelta_basic_axial_tension_positive']-base['columns'][tag]['pdelta_basic_axial_tension_positive']}
                 for tag,r in state['columns'].items()}
        beams={str(e.id):dict(global_force_increment=(e.stiffness@(u[ix]-u0[ix])).tolist(),
                             local_force_increment=(e.local_stiffness@(e.transform@(u[ix]-u0[ix]))).tolist()) for e,ix in self.beams}
        output=dict(method=METHOD,status='equilibrated',frame=result,columns=columns,beams=beams,
                    primary_definition_sha256=self.definition['identity_sha256'],source_identity=self.source_identity,
                    displacement_increment=(u-u0).tolist(),floor_loads=loads.tolist(),
                    physical_gravity_case_id=self.baseline['physical_case_id'],
                    auxiliary_load_increment=np.zeros(len(external)).tolist(),qualification_asserted=False)
        if saved_increment is not None:
            for collection in ('columns','beams'):
                if set(output[collection])!=set(saved_increment[collection]):raise ValueError('Saved primary member inventory differs')
                for tag,row in output[collection].items():
                    if set(row)!=set(saved_increment[collection][tag]):raise ValueError('Saved primary member action keys differ')
                    for key,value in row.items():
                        if not np.allclose(value,saved_increment[collection][tag][key],rtol=1e-10,atol=1e-8):
                            raise ValueError('Saved primary member increment differs from current native field')
            if not np.allclose(output['displacement_increment'],saved_increment['displacement_increment'],rtol=1e-11,atol=1e-12):
                raise ValueError('Saved primary displacement increment differs')
        return output

    def influence_response(self,increment):
        """Existing audited Floor_Influences format, from an actual increment."""
        if increment.get('status')!='equilibrated':raise ValueError('Complete primary force increment required')
        byid={n['id']:n for n in self.nodes};g=self.definition['geometry'];columns=[]
        for c in self.definition['columns']:
            ni,nj=byid[c['i']],byid[c['j']]
            columns.append(dict(tag=str(c['id']),i_node=c['i'],j_node=c['j'],i_xyz=ni['xyz'],j_xyz=nj['xyz'],
                i_level=int(round(ni['xyz'][2]/g['story_h'])),j_level=int(round(nj['xyz'][2]/g['story_h'])),
                global_force=increment['columns'][str(c['id'])]['global_force_increment']))
        dq=np.asarray(increment['frame']['cases'][0]['reduced_displacements'])-self.baseline['result']['cases'][0]['reduced_displacements']
        index={tuple(d):i for i,d in enumerate(self.reduced_dof_ids)}
        masters=[dict(node=100000+floor,xyz=[*self.master_xy,floor*g['story_h']]) for floor in range(1,g['nf']+1)]
        disp=[[float(dq[index[(f'master_{floor}',d)]]) for d in (1,2,6)] for floor in range(1,g['nf']+1)]
        return dict(columns=columns,floor_masters=masters,master_displacements=disp)


def transferred_floor_influences(primary,physical_definition,grid_mass_by_floor,*,
                                 witness_cases,protocol_id,max_direct_solves,max_wall_seconds,
                                 force_amplitude_kip=1.,torque_amplitude_kipin=1.,
                                 linearization_tolerances=None):
    """Replayable central floor basis about the verified physical gravity P.

    All solves use the independently native-verified primary element law.
    The direct second-order residual is retained by the existing P5 ledger;
    exact linear superposition is never a prerequisite or silently asserted.
    No new OpenSees model or analyze call is made by this function.
    """
    from Design import SMRF_Floor_Influences as api
    from Design.SMRF_Physical_Diaphragm import identity
    if primary.baseline is None or not protocol_id or not witness_cases:
        raise ValueError('Verified physical gravity baseline, protocol and direct witnesses required')
    binding=api.validate_physical_binding(physical_definition,grid_mass_by_floor,
        physical_definition_sha256=primary.definition['physical_definition_sha256'],
        state_identity=primary.definition['state_identity'],nodal_mass_sha256=identity(grid_mass_by_floor))
    factors=[float(w.get('strength_factor',1.)) for w in witness_cases]
    if any(not np.isfinite(f) or f<1. for f in factors):raise ValueError('Valid witness strength factors required')
    n=primary.definition['geometry']['nf'];required=12*n+len(witness_cases)+sum(f!=1. for f in factors)
    if type(max_direct_solves) is not int or max_direct_solves<required:
        raise ValueError('Declared transferred influence solve bound is insufficient')
    if not np.isfinite([max_wall_seconds,force_amplitude_kip,torque_amplitude_kipin]).all() or min(max_wall_seconds,force_amplitude_kip,torque_amplitude_kipin)<=0:
        raise ValueError('Positive finite analysis bounds and amplitudes required')
    if len({w['id'] for w in witness_cases})!=len(witness_cases):raise ValueError('Distinct witness IDs required')
    start=time.monotonic();solves=0
    def run(loads):
        nonlocal solves
        if solves>=max_direct_solves or time.monotonic()-start>max_wall_seconds:
            raise RuntimeError('Transferred floor influence bound exhausted')
        result=primary.solve_increment(loads);solves+=1
        out['raw_increment_witnesses'].append(dict(loads=np.asarray(loads).tolist(),increment=result))
        return primary.influence_response(result)
    out=dict(method=api.METHOD_VERSION,producer_method=METHOD,protocol_id=protocol_id,
        source_identity=primary.source_identity,**binding,grid_mass_by_floor=grid_mass_by_floor,
        lateral_design_model=primary.definition['lateral_design_model'],num_floor=n,
        gravity_combination=primary.definition['gravity_case'],basis=[],witnesses=[],raw_increment_witnesses=[],
        physical_gravity_case_id=primary.baseline['physical_case_id'],
        primary_definition_sha256=primary.definition['identity_sha256'],
        initial_column_axial_ledger=primary.baseline['column_axial_ledger'],
        auxiliary_global_wrench_kip_kipin=primary.baseline['auxiliary_global_wrench_kip_kipin'],
        auxiliary_loads_included_as_design_demand=False)
    for floor in range(1,n+1):
        for dof in api.DOFS:
            amplitude=torque_amplitude_kipin if dof=='rz' else force_amplitude_kip
            loads=np.zeros((n,3));loads[floor-1,api.DOFS.index(dof)]=amplitude
            coarse=api.sum_responses([(.5,api.subtract_response(run(loads),run(-loads)))])
            response=api.subtract_response(run(loads/2),run(-loads/2))
            trial=dict(num_floor=n,basis=[dict(source_floor=floor,dof=dof,amplitude=amplitude,response=coarse)])
            check=api.verify_superposition(trial,loads,response,**(linearization_tolerances or {}))
            check.update(coarse_response=coarse,definition='central +/-h and +/-h/2, normalized to h, same force-verified physical gravity')
            bodies=api.floor_free_bodies(response,loads)
            if not all(r['equilibrium_passed'] for r in bodies):raise ValueError('Transferred floor basis violates equilibrium')
            out['basis'].append(dict(source_floor=floor,dof=dof,amplitude=amplitude,response=response,
                                     free_bodies=bodies,linearization_verification=check))
    for witness,factor in zip(witness_cases,factors):
        loads=api._loads(witness['loads'],n);direct=run(loads)
        out['witnesses'].append(dict(id=witness['id'],loads=loads.tolist(),direct_increment=direct,
            verification=api.verify_superposition(out,loads,direct),
            initial_tangent_reference=primary.incremental_reference(loads)))
        if factor!=1.:
            base_loads=loads/factor;base_direct=run(base_loads)
            out['witnesses'][-1]['conventional_base_witness']=dict(strength_factor=factor,
                loads=base_loads.tolist(),direct_increment=base_direct,
                verification=api.verify_superposition(out,base_loads,base_direct))
    out.update(model_solves=solves,wall_seconds=time.monotonic()-start,
        all_linearizations_converged=all(r['linearization_verification']['passed'] for r in out['basis']),
        all_witnesses_passed=all(r['verification']['passed'] for r in out['witnesses']),
        scope='matching actual gravity and signed witnesses; complete second-order residual retained',
        native_analyze_calls=0,qualification_asserted=False)
    return out


def verify_transferred_floor_influences(primary,physical_definition,grid_mass_by_floor,saved):
    """Rebuild the entire central/direct basis from raw native-frame witnesses.

    Every saved displacement/force increment is checked by K@u, constraints,
    PDelta forces and actual baseline subtraction. No frame solve, eigen call
    or linearized inverse is used. The optional tangent reference remains a
    diagnostic and does not supply a P5 load or equilibrium assertion.
    """
    from Design import SMRF_Floor_Influences as api
    from Design.SMRF_Physical_Diaphragm import identity
    binding=api.validate_physical_binding(physical_definition,grid_mass_by_floor,
        physical_definition_sha256=primary.definition['physical_definition_sha256'],
        state_identity=primary.definition['state_identity'],nodal_mass_sha256=identity(grid_mass_by_floor))
    if (saved['source_identity']!=primary.source_identity or saved['primary_definition_sha256']!=primary.definition['identity_sha256'] or
        any(saved[k]!=v for k,v in binding.items()) or saved['physical_gravity_case_id']!=primary.baseline['physical_case_id'] or
        saved['initial_column_axial_ledger']!=primary.baseline['column_axial_ledger']):raise ValueError('Saved influence physical/source/gravity binding differs')
    raw=saved.get('raw_increment_witnesses')
    if not raw or len(raw)!=saved['model_solves']:raise ValueError('Complete raw primary influence witness inventory required')
    rows={};used=set()
    for i,r in enumerate(raw):rows.setdefault(identity(r['loads']),[]).append((i,r))
    def recover(loads):
        choices=rows.get(identity(np.asarray(loads).tolist()),[])
        available=[r for r in choices if r[0] not in used]
        if not available:raise ValueError('Missing/duplicated raw primary influence excitation')
        i,r=available[0];used.add(i)
        inc=primary.solve_increment(loads,saved_increment=r['increment'])
        return primary.influence_response(inc)
    n=primary.definition['geometry']['nf'];expected={(k,d) for k in range(1,n+1) for d in api.DOFS}
    if {(b['source_floor'],b['dof']) for b in saved['basis']}!=expected or len(saved['basis'])!=len(expected):raise ValueError('Complete central primary influence basis required')
    for basis in saved['basis']:
        loads=np.zeros((n,3));loads[basis['source_floor']-1,api.DOFS.index(basis['dof'])]=basis['amplitude']
        coarse=api.sum_responses([(.5,api.subtract_response(recover(loads),recover(-loads)))])
        response=api.subtract_response(recover(loads/2),recover(-loads/2))
        if identity(coarse)!=identity(basis['linearization_verification']['coarse_response']) or identity(response)!=identity(basis['response']):
            raise ValueError('Saved central basis differs from native-frame raw response witnesses')
    if not saved['witnesses'] or len({w['id'] for w in saved['witnesses']})!=len(saved['witnesses']):raise ValueError('Unique actual signed direct witnesses required')
    for witness in saved['witnesses']:
        if identity(recover(witness['loads']))!=identity(witness['direct_increment']):raise ValueError('Saved original signed primary direct response differs')
        base=witness.get('conventional_base_witness')
        if base is not None:
            loads=np.asarray(witness['loads'])/base['strength_factor']
            if identity(loads.tolist())!=identity(base['loads']) or identity(recover(loads))!=identity(base['direct_increment']):
                raise ValueError('Saved base-ELF direct response differs from native-frame witness')
    if len(used)!=len(raw):raise ValueError('Unexpected unused raw primary influence witnesses')
    return dict(raw_primary_increments_verified=len(used),central_components=len(expected),signed_direct_witnesses=len(saved['witnesses']),
        frame_solves=0,native_analyze_calls=0,source_identity=primary.source_identity)
