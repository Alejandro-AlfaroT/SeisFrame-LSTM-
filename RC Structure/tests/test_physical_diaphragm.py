"""Pure hand equilibrium witnesses for actual physical diaphragm mapping."""
import copy
import unittest
import numpy as np
from Design.SMRF_Physical_Diaphragm import (PhysicalDiaphragm,identity,rectangle_weights,rigid_diaphragm_permission,
    conventional_force_inputs,physical_membrane_cut,rectangle_shape_measure,line_shape_measure,point_shape_measure,
    consistent_affine_forces,affine_submeasure_wrench)
from Design.SMRF_Diaphragm_Actions import resultant,rectangular_mesh
from Design.SMRF_Floor_Influences import verify_superposition,validate_physical_binding
from tests.test_coupled_floor_geometry import fixture


def masses(d):
    out={};g=d['geometry'];gamma=d['gamma_kip_in3']
    for desc in d['floors']:
        k=desc['level'];bw=2. if k==1 else 2.5;cb=4. if k==1 else 3.;bh=5. if k==1 else 6.
        w=(40+bw)*(16+bw)*(.15+.25*.04)/144
        w+=(4*(20-4)+3*(16-cb))*bw*(bh-1)*gamma
        for j in range(2):
            for i in range(3):
                x,y=i*20,j*16
                overlap=(min(x+2,40+bw/2)-max(x-2,-bw/2))*(min(y+cb/2,16+bw/2)-max(y-cb/2,-bw/2))
                w+=(4*cb*10-overlap)*gamma
        out[str(k)]=[{'node':s['id'],'xy_in':[s['x'],s['y']],'weight_kip':w/6} for s in desc['supports']]
    return out


def hand_response(d,loads,residual=0.):
    """Six lines/two stories with unequal changing shares; exact XY wrench.

    This is an analytical floor equilibrium fixture, not claimed to be a
    three-dimensional frame stiffness solution. It includes genuine signed
    transfer plus a separate balanced residual force pair at floor1.
    """
    bynode={n['id']:n['xyz'] for n in d['nodes']};rows=[];loads=np.asarray(loads)
    for story in (1,2):
        cols=[c for c in d['columns'] if c['story']==story]
        xy=np.array([bynode[c['j']][:2] for c in cols])
        shares=np.array([.25,.15,.1,.2,.2,.1] if story==1 else [.1,.2,.2,.1,.15,.25])
        total=loads[story-1:].sum(axis=0);center=shares@xy;r=xy-center
        arm=np.array([20.,8.])-center;torque=total[2]+arm[0]*total[1]-arm[1]*total[0]
        polar=np.sum(shares*np.sum(r*r,axis=1))
        force=shares[:,None]*total[:2]+torque/polar*shares[:,None]*np.c_[-r[:,1],r[:,0]]
        if story==1:force[0,0]+=residual;force[2,0]-=residual
        for c,f in zip(cols,force):
            vec=[0.]*12;vec[0:2]=(-f).tolist();vec[6:8]=f.tolist()
            rows.append(dict(tag=c['id'],i_level=story-1,j_level=story,i_node=c['i'],j_node=c['j'],
                             i_xyz=bynode[c['i']],j_xyz=bynode[c['j']],global_force=vec))
    return dict(columns=rows,floor_masters=[{'node':900+k,'xyz':[20.,8.,10.*k]} for k in (1,2)],master_displacements=[[0.,0.,0.],[0.,0.,0.]])


def inputs():
    d=fixture();mass=masses(d);source='independent_physical_force_mapping'
    inf=dict(num_floor=2,source_identity=source,state_identity=d['state_identity'],
             physical_definition_sha256=d['identity_sha256'],nodal_mass_sha256=identity(mass),
             all_linearizations_converged=True,basis=[],witnesses=[])
    for floor in (1,2):
        for dof in ('x','y','rz'):
            load=np.zeros((2,3));load[floor-1,('x','y','rz').index(dof)]=1.
            response=hand_response(d,load);basisrow=dict(source_floor=floor,dof=dof,amplitude=1.,response=response)
            trial=dict(num_floor=2,basis=[basisrow]);check=verify_superposition(trial,load,response)
            basisrow['linearization_verification']={**check,'coarse_response':copy.deepcopy(response)};inf['basis'].append(basisrow)
    loads=[[10.,-3.,4.],[20.,-6.,-7.]]
    direct=hand_response(d,loads,residual=.8)
    inf['witnesses']=[dict(id='signed_X_30Y',loads=loads,direct_increment=direct)]
    return d,mass,source,inf


def conventional():
    return dict(elf_floor_force_kip=10.,fpx_eq_kip=20.,fpx_min_kip=8.,fpx_max_kip=16.,omega0=3.,transfer_factor=4.,irregularity_factor=1.25)


class PhysicalDiaphragmTests(unittest.TestCase):
    def test_redundancy_uses_separate_nonlinear_base_force_witness(self):
        d,mass,source,inf=inputs()
        adapter=PhysicalDiaphragm(d,1,grid_mass_by_floor=mass,seismic_live_load_fraction=.25,source_identity=source)
        baseline=adapter.solve(inf,'signed_X_30Y',conventional_forces=conventional())
        base=copy.deepcopy(inf['witnesses'][0]);rho=1.3
        inf['witnesses'][0]['loads']=(rho*np.array(base['loads'])).tolist()
        # Deliberately nonlinear residual: dividing this response by rho
        # cannot reproduce the base witness and must never be the algorithm.
        inf['witnesses'][0]['direct_increment']=hand_response(d,inf['witnesses'][0]['loads'],residual=.8*rho*rho)
        forces=dict(conventional(),witness_strength_factor=rho)
        with self.assertRaisesRegex(ValueError,'Actual base-ELF'):adapter.solve(inf,'signed_X_30Y',conventional_forces=forces)
        inf['witnesses'][0]['conventional_base_witness']=dict(base,strength_factor=rho)
        result=adapter.solve(inf,'signed_X_30Y',conventional_forces=forces,
            cuts=[dict(axis='x',coordinate_in=11.7)])
        self.assertTrue(result['conventional_force_basis']['separately_solved_base_elf'])
        for expected,actual in zip(baseline['cases'][2:4],result['cases'][2:4]):
            self.assertEqual(expected['id'],actual['id'])
            np.testing.assert_allclose(expected['displacements_in'],actual['displacements_in'],atol=1e-12,rtol=1e-11)
        structural=result['cases'][0];fpx=result['cases'][1];minimum=result['cases'][4]
        self.assertTrue(all(not k.startswith('base_elf/') for k in structural['factors']))
        for row in result['components']:
            if row['id'].startswith('base_elf/'):continue
            if row['role']=='transfer':
                self.assertAlmostEqual(fpx['factors'][row['id']],5.)
                self.assertAlmostEqual(minimum['factors'][row['id']],5.)
                self.assertNotIn('base_elf/'+row['id'],fpx['factors'])
        original=next(c for c in result['cases'] if c['id']=='structural_original_direct')
        self.assertFalse(np.allclose(original['displacements_in'],rho*np.array(baseline['cases'][5]['displacements_in']),atol=1e-12,rtol=1e-8))
        inf['witnesses'][0]['conventional_base_witness']['loads'][0][2]+=.1
        with self.assertRaisesRegex(ValueError,'complete original signed wrench'):adapter.solve(inf,'signed_X_30Y',conventional_forces=forces)

    def test_exact_partial_rectangle_traction_area_and_first_moments(self):
        x=[-1.,0.,2.,10.];y=[-2.,0.,3.,8.]
        w=rectangle_weights(x,y,[-.8,1.3,-1.1,2.7]);xy=np.array([(a,b) for b in y for a in x])
        area=2.1*3.8
        self.assertTrue(np.all(w>=0));self.assertAlmostEqual(w.sum(),area)
        np.testing.assert_allclose(w@xy,area*np.array([.25,.8]),atol=1e-13)

    def test_actual_mass_web_faces_and_different_above_below_footprints(self):
        d,mass,source,_=inputs();a=PhysicalDiaphragm(d,1,grid_mass_by_floor=mass,seismic_live_load_fraction=.25,source_identity=source)
        self.assertLess(a.mass_audit['maximum_relative_error'],1e-13)
        np.testing.assert_allclose(a.mass_audit['center_xy_in'],[20.,8.],atol=1e-12)
        self.assertGreater(np.count_nonzero(a.mass_weights),6)
        lower=a.traction_metadata['column:1:0:0']['physical_overlap']
        upper=a.traction_metadata['column:2:0:0']['physical_overlap']
        self.assertEqual(lower['area_in2'],3*3)
        self.assertEqual(upper['area_in2'],3*2.5)
        self.assertNotEqual(lower['centroid_in'],upper['centroid_in'])
        for b in a.mesh['beams']:
            c=next(c for c in a.clear_beams if c['id']==b['member_id'])
            self.assertGreaterEqual(b['start_in'],c['start']);self.assertLessEqual(b['end_in'],c['end'])
            self.assertEqual(b['area_web_in2'],c['b_in']*(c['h_in']-1))

    def test_signed_full_direct_reconstruction_and_all_conventional_branches(self):
        d,mass,source,inf=inputs();a=PhysicalDiaphragm(d,1,grid_mass_by_floor=mass,seismic_live_load_fraction=.25,source_identity=source)
        result=a.solve(inf,'signed_X_30Y',conventional_forces=conventional(),cuts=[dict(axis='x',coordinate_in=11.7)],include_residual_sign_sensitivity=True)
        self.assertEqual(len(result['cases']),10)
        self.assertFalse(result['qualification_asserted']);self.assertFalse(result['nonlinear_branch_resolve_asserted'])
        self.assertLess(result['physical_nodal_reconstruction_max_abs_kip'],1e-12)
        local,other,residual=result['components']
        self.assertEqual(np.max(np.abs(residual['inertia'])),0.)
        self.assertGreater(np.max(np.abs(residual['columns_below_on_floor'])),0.)
        for c in result['components']:
            loads=sum(np.asarray(c[k]) for k in ('inertia','columns_above_on_floor','columns_below_on_floor'))
            np.testing.assert_allclose(resultant(result['mesh']['nodes'],loads),[0.,0.,0.],atol=1e-11)
        cases={c['id']:c for c in result['cases']}
        factors={c['id']:c['factors'] for c in result['force_case_definitions']}
        self.assertEqual(factors['diaphragm_fpx'][local['id']],2.)
        self.assertEqual(factors['diaphragm_fpx'][residual['id']],5.)
        self.assertEqual(factors['collector_minimum'][local['id']],1.)
        self.assertEqual(factors['collector_minimum'][residual['id']],5.)
        self.assertTrue(all(v==3. for v in factors['collector_original_direct_omega'].values()))
        for force_key in ('Nxx_kip_per_in','Nyy_kip_per_in','Nxy_kip_per_in'):
            np.testing.assert_allclose([s[force_key] for s in cases['collector_original_direct_omega']['slab_samples']],
                3*np.array([s[force_key] for s in cases['structural_original_direct']['slab_samples']]),rtol=1e-10,atol=1e-10)
        for c in result['cases']:
            self.assertLess(c['normalized_nodal_residual'],1e-10)
            self.assertLess(np.max(np.abs(c['cuts'][0]['balance_error'])),1e-10)
            self.assertTrue(c['member_external_axial'])
            audit=c['continuum_cut_equilibrium'][0]
            np.testing.assert_allclose(np.array(audit['applied_continuum_left_wrench_kip_kipin'])+audit['required_total_cut_wrench_kip_kipin'],0.,atol=1e-13)
            self.assertFalse(audit['web_slab_allocation_verified'])

    def test_roof_zero_above_and_stale_mass_source_topology_guards(self):
        d,mass,source,inf=inputs();a=PhysicalDiaphragm(d,2,grid_mass_by_floor=mass,seismic_live_load_fraction=.25,source_identity=source)
        cfg={**conventional(),'elf_floor_force_kip':20.}
        result=a.solve(inf,'signed_X_30Y',conventional_forces=cfg)
        self.assertEqual(len(result['cases']),7)
        for c in result['components']:self.assertEqual(np.max(np.abs(c['columns_above_on_floor'])),0.)
        for key in ('state_identity','physical_definition_sha256','nodal_mass_sha256','source_identity'):
            broken=copy.deepcopy(inf);broken[key]='stale'
            with self.assertRaises(ValueError):a.solve(broken,'signed_X_30Y',conventional_forces=cfg)
        broken=copy.deepcopy(inf);broken['basis'][0]['response']['columns'][0]['j_xyz']=[999,0,10]
        with self.assertRaisesRegex(ValueError,'coordinates'):a.solve(broken,'signed_X_30Y',conventional_forces=cfg)
        wrong=copy.deepcopy(mass);wrong['2'][0]['weight_kip']*=1.01
        with self.assertRaisesRegex(ValueError,'mass does not reproduce'):
            PhysicalDiaphragm(d,2,grid_mass_by_floor=wrong,seismic_live_load_fraction=.25,source_identity=source)

    def test_forged_linearization_flag_cannot_replace_numerical_replay(self):
        d,mass,source,inf=inputs();a=PhysicalDiaphragm(d,1,grid_mass_by_floor=mass,seismic_live_load_fraction=.25,source_identity=source)
        inf['basis'][0]['linearization_verification']['coarse_response']['columns'][0]['global_force'][6]+=.1
        with self.assertRaisesRegex(ValueError,'linearization'):
            a.solve(inf,'signed_X_30Y',conventional_forces=conventional())

    def test_rigid_permission_uses_complete_actual_lines_and_all_required_types(self):
        d=fixture()
        for floor in d['floors']:
            for b in floor['beams']:b['system']='sfrs'
        d['identity_sha256']=identity({k:v for k,v in d.items() if k!='identity_sha256'})
        checks={str(k):dict(status='complete',present=False,evidence_sha256=f'computed_fixture_{k}') for k in range(2,6)}
        result=rigid_diaphragm_permission(d,horizontal_irregularity_checks=checks)
        self.assertTrue(result['permitted']);self.assertTrue(result['zones'])
        missing=copy.deepcopy(checks);del missing['3']
        self.assertFalse(rigid_diaphragm_permission(d,horizontal_irregularity_checks=missing)['permitted'])
        present=copy.deepcopy(checks);present['4']['present']=True
        self.assertFalse(rigid_diaphragm_permission(d,horizontal_irregularity_checks=present)['permitted'])
        # One missing perimeter span cannot masquerade as a complete line.
        d['floors'][0]['beams'][0]['system']='gravity'
        d['identity_sha256']=identity({k:v for k,v in d.items() if k!='identity_sha256'})
        result=rigid_diaphragm_permission(d,horizontal_irregularity_checks=checks)
        self.assertFalse(result['permitted'])
        self.assertTrue(any('bounding SFRS' in gap for gap in result['technical_gaps']))

    def test_producer_binding_recomputes_definition_state_and_full_mass_snapshot(self):
        d,mass,source,inf=inputs()
        binding={k:inf[k] for k in ('physical_definition_sha256','state_identity','nodal_mass_sha256')}
        self.assertEqual(validate_physical_binding(d,mass,**binding),binding)
        for key in binding:
            for value in ('',None,'stale'):
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                    validate_physical_binding(d,mass,**{**binding,key:value})
        changed=copy.deepcopy(mass);changed['2'][0]['weight_kip']*=1.01
        with self.assertRaises(ValueError):validate_physical_binding(d,changed,**binding)

    def test_physical_stress_cut_includes_disjoint_web_and_exact_moment_arm(self):
        mesh=rectangular_mesh([0.,5.,10.],[0.,4.,8.],thickness_in=2.,elastic_modulus_ksi=100.,
                              beams=[dict(id='w',member_id='beam',i=3,j=4,axis='x',area_web_in2=2.,E_ksi=100.)])
        u=[[.01*x,-.002*y] for x,y in mesh['nodes']]
        result={'mesh':mesh,'cases':[dict(id='affine',displacements_in=u,beam_axial=[dict(id='w',N_kip=2.)])]}
        cut=physical_membrane_cut(result,'affine','x',2.)
        np.testing.assert_allclose(cut['slab_wrench_kip_kipin'],[16.,0.,-64.],atol=1e-12)
        np.testing.assert_allclose(cut['web_wrench_kip_kipin'],[2.,0.,-8.],atol=1e-12)
        np.testing.assert_allclose(cut['total_wrench_kip_kipin'],[18.,0.,-72.],atol=1e-12)
        with self.assertRaises(ValueError):physical_membrane_cut(result,'affine','x',5.)
        # Exactly the same geometric cut, explicitly different traces, with
        # continuous displacement and a strain jump across the mesh line.
        result['cases'][0]['displacements_in']=[[.01*min(x,5)+.02*max(x-5,0),-.002*y] for x,y in mesh['nodes']]
        minus=physical_membrane_cut(result,'affine','x',5.,side='minus')
        plus=physical_membrane_cut(result,'affine','x',5.,side='plus')
        np.testing.assert_allclose(minus['slab_wrench_kip_kipin'],[16.,0.,-64.],atol=1e-12)
        # Plane stress: t*E/(1-nu^2)*(ex+nu*ey)*8, nu=.2.
        right_force=2*100/(1-.2**2)*(.02+.2*(-.002))*8
        np.testing.assert_allclose(plus['slab_wrench_kip_kipin'],[right_force,0.,-4*right_force],atol=1e-12)
        self.assertEqual(minus['web_wrench_kip_kipin'],[2.,0.,-8.])
        self.assertEqual(plus['web_wrench_kip_kipin'],[0.,0.,0.])

    def test_fpx_uses_actual_full_sds_weights_and_uncapped_equation(self):
        result=conventional_force_inputs({'story_forces_kip':[10.,30.],'story_weights_kip':[100.,200.]},1,
                                         sds=1.2,importance_factor=1.25,omega0=3.)
        self.assertAlmostEqual(result['fpx_eq_kip'],40/300*100)
        self.assertEqual(result['fpx_min_kip'],30.)
        self.assertEqual(result['fpx_max_kip'],60.)

    def test_consistent_affine_pressure_exact_wrench_and_fixed_subset_invariant(self):
        for x,y in (([-1.,3.],[-2.,2.]),([-1.,-.2,0.,1.,3.],[-2.,-1.,.5,2.])):
            full=rectangle_shape_measure(x,y,[-1.,3.,-2.,2.]);left=rectangle_shape_measure(x,y,[-1.,0.,-2.,2.])
            force=consistent_affine_forces(full,[2.,-1.],3.)
            xy=np.array([(a,b) for b in y for a in x])
            np.testing.assert_allclose(resultant(xy,force),[2.,-1.,2.],atol=2e-14)
            # Independently integrated q=[2,-1]/16 +3/(128/3)*[-y,x-1]
            # over[-1,0]x[-2,2], including its physical moment about(0,0).
            np.testing.assert_allclose(affine_submeasure_wrench(full,left,[2.,-1.],3.),[.5,-.671875,.734375],atol=2e-14)
        self.assertAlmostEqual(full['second'],16*((16+16)/12+1.))

    def test_consistent_line_and_point_load_functions_reproduce_exact_moments(self):
        for x in ([0.,10.],[0.,2.,3.,7.,10.]):
            y=[0.,2.,5.];line=line_shape_measure(x,y,'x',2.,0.,10.)
            force=consistent_affine_forces(line,[7.,-3.],4.)
            xy=np.array([(a,b) for b in y for a in x])
            np.testing.assert_allclose(resultant(xy,force),[7.,-3.,-25.],atol=5e-14)
            self.assertAlmostEqual(line['second'],1000/3+40.)
            point=point_shape_measure(x,y,[2.5,3.])
            np.testing.assert_allclose(resultant(xy,consistent_affine_forces(point,[4.,5.])),[4.,5.,.5],atol=1e-14)

    def test_actual_affine_load_wrench_on_fixed_cut_is_subdivision_invariant(self):
        from Design.SMRF_Floor_Influences import decompose_with_second_order_residual
        d,mass,source,inf=inputs();components=decompose_with_second_order_residual(inf,'signed_X_30Y',1)['components']
        a=PhysicalDiaphragm(d,1,grid_mass_by_floor=mass,seismic_live_load_fraction=.25,source_identity=source,subdivisions_per_bay=2)
        b=PhysicalDiaphragm(d,1,grid_mass_by_floor=mass,seismic_live_load_fraction=.25,source_identity=source,subdivisions_per_bay=5)
        self.assertAlmostEqual(a.mass_audit['physical_polar_kip_in2'],b.mass_audit['physical_polar_kip_in2'],places=10)
        for row in components:
            # A cut at a lumped member centroid has no physical point load:
            # that force is now delivered through its finite joint patch.
            for axis,at in (('x',7.3),('y',.45),('x',a.member_inertia_sources[0]['application_xy_in'][0])):
                np.testing.assert_allclose(a.component_wrench_on_left(row,axis,at),b.component_wrench_on_left(row,axis,at),atol=1e-11)

    def test_same_joint_member_inertia_reaction_cancellation_and_offset_rigid_work(self):
        from Design.SMRF_Physical_Diaphragm import force_through_joint
        x=[-1.,0.,.7,2.,4.];y=[-2.,-.3,0.,1.,3.]
        patch=rectangle_shape_measure(x,y,[-1.,.7,-2.,1.]);xy=np.array([(a,b) for b in y for a in x])
        force=np.array([7.,-3.]);point=np.array([.15,-.25]);joint=np.array([0.,0.])
        inertia=force_through_joint(patch,force,point)
        reaction=force_through_joint(patch,-force,point)
        np.testing.assert_allclose(inertia+reaction,0.,atol=1e-15)
        # A shifted centroid is NOT silently moved onto the column center.
        residual=inertia+force_through_joint(patch,-force,joint)
        expected_couple=point[0]*force[1]-point[1]*force[0]
        np.testing.assert_allclose(resultant(xy,residual),[0.,0.,expected_couple],atol=1e-14)
        translation=np.array([.3,-.7]);rotation=.04
        rigid=translation+rotation*np.c_[-xy[:,1],xy[:,0]]
        exact=np.dot(force,translation+rotation*np.array([-point[1],point[0]]))
        self.assertAlmostEqual(float(np.sum(inertia*rigid)),exact,places=13)
        self.assertAlmostEqual(float(np.sum(residual*rigid)),rotation*expected_couple,places=13)

    def test_member_inertia_delivery_preserves_original_point_force_and_rotation(self):
        from Design.SMRF_Physical_Diaphragm import _measure_properties,force_through_joint
        d,mass,source,_=inputs();a=PhysicalDiaphragm(d,1,grid_mass_by_floor=mass,seismic_live_load_fraction=.25,source_identity=source)
        load=np.array([17.,-11.,29.]);master=np.array([20.,8.]);total,_,center,polar=_measure_properties(a.mass_measure)
        arm=master-center;torque=load[2]+arm[0]*load[1]-arm[1]*load[0]
        xy=np.asarray(a.mesh['nodes'])
        for member,force in a.member_inertia_forces(load,master):
            point=np.array(member['application_xy_in']);delta=point-center
            expected=member['weight_kip']*(load[:2]/total+torque/polar*np.array([-delta[1],delta[0]]))
            np.testing.assert_allclose(force,expected,atol=1e-15)
            delivered=force_through_joint(a.column_measures[member['column_id']],force,point)
            np.testing.assert_allclose(resultant(xy,delivered),[*expected,point[0]*expected[1]-point[1]*expected[0]],atol=1e-12)

    def test_raw_q4_cut_error_with_exact_nodal_solution_manufactured_body_load(self):
        from Design.SMRF_Diaphragm_Actions import solve_diaphragm_actions
        # Exact continuum: ux=.001*x^2, uy=0; E=100,nu=0,t=1.
        # Nxx=.2*x, body fx=-.2; x=10 boundary traction +2kip/in.
        # Q4 nodal displacements are exact modulo rigid motion, but the
        # within-element Nxx trace is the exact value at cell midpoint.
        x=[0.,2.,5.,10.];y=[0.,2.,4.]
        mesh=rectangular_mesh(x,y,thickness_in=1.,elastic_modulus_ksi=100.,poisson=0.)
        body=rectangle_shape_measure(x,y,[0.,10.,0.,4.])['weights'][:,None]*np.array([-.2,0.])
        boundary=line_shape_measure(x,y,'y',10.,0.,4.)['weights'][:,None]*np.array([2.,0.])
        result=solve_diaphragm_actions(mesh,[dict(id='exact',role='local_inertia',provenance='independent exact quadratic continuum solution',
             inertia=body.tolist(),columns_above_on_floor=np.zeros_like(body).tolist(),columns_below_on_floor=boundary.tolist())],
             [dict(id='exact',factors={'exact':1.})])
        u=np.asarray(result['cases'][0]['displacements_in']);xy=np.asarray(mesh['nodes'])
        for row in range(len(y)):
            indices=np.arange(row*len(x),(row+1)*len(x))
            np.testing.assert_allclose(np.diff(u[indices,0]),np.diff(.001*np.array(x)**2),atol=1e-14)
        trace=physical_membrane_cut(result,'exact','x',3.1)
        np.testing.assert_allclose(trace['total_wrench_kip_kipin'],[.2*3.5*4,0.,-.2*3.5*8],atol=1e-12)
        exact_cut=np.array([.2*3.1*4,0.,-.2*3.1*8])
        self.assertAlmostEqual(trace['total_wrench_kip_kipin'][0]-exact_cut[0],.32)
        centered=physical_membrane_cut(result,'exact','x',3.5)
        np.testing.assert_allclose(centered['total_wrench_kip_kipin'],[.2*3.5*4,0.,-.2*3.5*8],atol=1e-12)
        from Design.SMRF_Physical_Diaphragm import centered_target_axis
        # Keep x=3.1 fixed; center its containing element without changing
        # the continuum force or evaluating an averaged/extrapolated stress.
        refined,_=centered_target_axis(x,[3.1])
        mesh=rectangular_mesh(refined,y,thickness_in=1.,elastic_modulus_ksi=100.,poisson=0.)
        body=rectangle_shape_measure(refined,y,[0.,10.,0.,4.])['weights'][:,None]*np.array([-.2,0.])
        boundary=line_shape_measure(refined,y,'y',10.,0.,4.)['weights'][:,None]*np.array([2.,0.])
        centered_result=solve_diaphragm_actions(mesh,[dict(id='exact',role='local_inertia',provenance='same exact continuum with actual section-centered cells',
             inertia=body.tolist(),columns_above_on_floor=np.zeros_like(body).tolist(),columns_below_on_floor=boundary.tolist())],
             [dict(id='exact',factors={'exact':1.})])
        np.testing.assert_allclose(physical_membrane_cut(centered_result,'exact','x',3.1)['total_wrench_kip_kipin'],exact_cut,atol=1e-12)

    def test_centered_target_axis_retains_physical_lines_nested_refinement_and_points(self):
        from Design.SMRF_Physical_Diaphragm import centered_target_axis
        first,cells=centered_target_axis([0.,2.,5.,10.],[3.1,7.3])
        second,fine=centered_target_axis([0.,1.,2.,3.,5.,7.,10.],[3.1,7.3],previous_axis=first)
        self.assertTrue(set(first)<=set(second))
        for coarse,finer in zip(cells,fine):
            self.assertEqual(coarse['target_in'],finer['target_in'])
            self.assertAlmostEqual((finer['from_in']+finer['to_in'])/2,finer['target_in'])
            self.assertLess(coarse['from_in'],finer['from_in']);self.assertGreater(coarse['to_in'],finer['to_in'])
        with self.assertRaises(ValueError):centered_target_axis([0.,2.,5.,10.],[2.])


if __name__=='__main__':unittest.main()
