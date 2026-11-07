import unittest
import copy
from unittest import mock
from types import SimpleNamespace
import numpy as np
from Design.SMRF_Diaphragm_Actions import rectangular_mesh
from Design.SMRF_Diaphragm_Embedding import (eccentric_web_embedding,eccentric_web_cut,map_planar_nodal_forces,
    prepare_coupled_floor_cases,prepare_external_inertia_cases,spatial_resultant)
from Design.SMRF_Floor_Substructure import rigid_arm_matrix

def explicit_rigid_floor(adapter):
    points=np.c_[adapter.mesh['nodes'],np.zeros(len(adapter.mesh['nodes']))];owner=np.arange(len(points));centers={}
    for support in adapter.description['supports']:
        center=next(i for i,p in enumerate(points) if p[0]==support['x'] and p[1]==support['y']);centers[center]=support['id']
        for i,p in enumerate(points):
            if abs(p[0]-support['x'])<=support['h_in']/2 and abs(p[1]-support['y'])<=support['b_in']/2:owner[i]=center
    retained=sorted(set(owner));index={n:i for i,n in enumerate(retained)};ids=[centers.get(n,f'slab:{n}') for n in retained]
    maps=[(np.arange(6*index[own],6*index[own]+6),rigid_arm_matrix(points[n]-points[own])) for n,own in enumerate(owner)]
    return SimpleNamespace(id='physical_fixture_floor',points=points,owner=owner,node_maps=maps,node_ids=ids,
        dof_ids=[(key,d) for key in ids for d in range(1,7)],node_positions={key:points[n].tolist() for key,n in zip(ids,retained)},
        supports=adapter.description['supports'],beams=adapter.description['beams'])

class EccentricWebTests(unittest.TestCase):
    def test_complete_external_inertia_only_preserves_signed_floor_wrench_and_rigid_work(self):
        from tests.test_physical_diaphragm import inputs
        from Design.SMRF_Physical_Diaphragm import PhysicalDiaphragm
        d,mass,source,_=inputs();adapters={k:PhysicalDiaphragm(d,k,grid_mass_by_floor=mass,seismic_live_load_fraction=.25,source_identity=source) for k in (1,2)}
        floors={k:explicit_rigid_floor(a) for k,a in adapters.items()}
        for k,f in floors.items():f.id=f'actual_floor_{k}'
        cases=[dict(id='signed',loads=[[10.,-3.,17.],[-4.,8.,-19.]],reference_xy_in=[[2.,5.],[17.,-2.]]),
               dict(id='torque_only',loads=[[0.,0.,31.],[0.,0.,-23.]],reference_xy_in=[0.,0.]),
               dict(id='zero',loads=np.zeros((2,3)).tolist(),reference_xy_in=[12.,9.])]
        with mock.patch('Design.SMRF_Diaphragm_Actions.solve_diaphragm_actions',side_effect=AssertionError('No solve permitted')):
            result=prepare_external_inertia_cases(adapters,floors,cases)
        self.assertFalse(result['mapping']['column_traction_loads_included'])
        self.assertFalse(result['mapping']['primary_force_composition_authorized'])
        self.assertEqual(set(result['loads']),{'actual_floor_1','actual_floor_2'})
        for k,floor in floors.items():
            vectors=np.asarray(result['loads'][floor.id]);self.assertEqual(vectors.shape,(len(floor.dof_ids),3))
            for index,case in enumerate(cases):
                load=np.asarray(case['loads'][k-1]);ref=np.asarray(case['reference_xy_in']);ref=ref[k-1] if ref.ndim==2 else ref
                expected=np.r_[load[:2],0.,0.,0.,load[2]+ref[0]*load[1]-ref[1]*load[0]]
                got=spatial_resultant([floor.node_positions[n] for n in floor.node_ids],vectors[:,index].reshape(-1,6))
                np.testing.assert_allclose(got,expected,rtol=1e-13,atol=1e-11)
                q=np.sin(np.arange(len(floor.dof_ids)))*.003
                direct=adapters[k].external_inertia_forces(load,ref)
                work=sum(direct[n]@(t@q[ind])[:2] for n,(ind,t) in enumerate(floor.node_maps))
                self.assertAlmostEqual(work,float(vectors[:,index]@q),places=12)
            np.testing.assert_array_equal(vectors[:,-1],0.)

    def test_external_inertia_is_reference_invariant_and_matches_original_inertia_component(self):
        from tests.test_physical_diaphragm import inputs
        from Design.SMRF_Physical_Diaphragm import PhysicalDiaphragm
        d,mass,source,inf=inputs();a=PhysicalDiaphragm(d,1,grid_mass_by_floor=mass,seismic_live_load_fraction=.25,source_identity=source)
        force=np.array([7.,-4.]);r=np.array([20.,8.]);s=np.array([-9.,13.]);torque=19.
        arm=r-s;moved=torque+arm[0]*force[1]-arm[1]*force[0]
        np.testing.assert_allclose(a.external_inertia_forces([*force,torque],r),a.external_inertia_forces([*force,moved],s),atol=1e-14)
        raw=dict(id='inertia',role='local_inertia',provenance='independent zero-traction identity witness',
                 applied_load=[*force,torque],master_xy_in=r.tolist())
        for side in ('above','below'):
            raw[f'columns_{side}_on_floor']=[dict(column_tag=tag,xy_in=meta['application_xy_in'],Fx_kip=0.,Fy_kip=0.,Mz_kip_in=0.)
                for tag,meta in a.traction_metadata.items() if meta['side']==side]
        component=a.components([raw])[0]
        np.testing.assert_array_equal(component['inertia'],a.external_inertia_forces([*force,torque],r))
        for field in ('columns_above_on_floor','columns_below_on_floor'):np.testing.assert_array_equal(component[field],0.)

    def test_external_inertia_rejects_missing_floor_stale_mass_and_unstated_reference(self):
        from tests.test_physical_diaphragm import inputs
        from Design.SMRF_Physical_Diaphragm import PhysicalDiaphragm
        d,mass,source,_=inputs();adapters={k:PhysicalDiaphragm(d,k,grid_mass_by_floor=mass,seismic_live_load_fraction=.25,source_identity=source) for k in (1,2)}
        floors={k:explicit_rigid_floor(a) for k,a in adapters.items()}
        for k,f in floors.items():f.id=str(k)
        case=dict(id='external',loads=[[1.,0.,0.],[2.,0.,0.]],reference_xy_in=[0.,0.])
        with self.assertRaises(ValueError):prepare_external_inertia_cases({1:adapters[1]},floors,[case])
        with self.assertRaises(ValueError):prepare_external_inertia_cases(adapters,floors,[{**case,'reference_xy_in':[0.]}])
        with self.assertRaises(ValueError):prepare_external_inertia_cases(adapters,floors,[case,copy.deepcopy(case)])
        mass['1'][0]['weight_kip']+=.01
        with self.assertRaises(ValueError):prepare_external_inertia_cases(adapters,floors,[case])

    def test_actual_retained_force_map_preserves_spatial_wrench_and_arbitrary_virtual_work(self):
        points=np.array([[0.,0.,0.],[4.,0.,0.],[4.,3.,0.]])
        floor=SimpleNamespace(points=points,owner=[0,0,2],node_ids=[101,102],dof_ids=[(n,d) for n in (101,102) for d in range(1,7)],
            node_positions={101:points[0].tolist(),102:points[2].tolist()},node_maps=[(np.arange(6),np.eye(6)),
                (np.arange(6),rigid_arm_matrix([4.,0.,0.])),(np.arange(6,12),np.eye(6))])
        forces=np.array([[1.,2.],[3.,-4.],[5.,6.]])
        mapped,audit=map_planar_nodal_forces(floor,{'nodes':points[:,:2].tolist()},forces)
        np.testing.assert_allclose(mapped[:6],[4.,-2.,0.,0.,0.,-16.])
        np.testing.assert_allclose(audit['retained_wrench_kip_kipin'],[9.,4.,0.,0.,0.,-7.])
        q=np.arange(12)*.003;work=0.
        for n,(ind,t) in enumerate(floor.node_maps):work+=forces[n]@(t@q[ind])[:2]
        self.assertAlmostEqual(work,float(mapped@q),places=13)

    def test_coupled_case_producer_never_solves_or_imports_planar_response_fields(self):
        from tests.test_physical_diaphragm import inputs,conventional
        from Design.SMRF_Physical_Diaphragm import PhysicalDiaphragm
        d,mass,source,inf=inputs();adapter=PhysicalDiaphragm(d,1,grid_mass_by_floor=mass,seismic_live_load_fraction=.25,source_identity=source)
        floor=explicit_rigid_floor(adapter)
        with mock.patch('Design.SMRF_Diaphragm_Actions.solve_diaphragm_actions',side_effect=AssertionError('No planar solve permitted')):
            result=prepare_coupled_floor_cases(adapter,floor,inf,'signed_X_30Y',conventional_forces=conventional())
        self.assertEqual(len(result['case_ids']),7);self.assertEqual(result['mapping']['omitted_case_ids'],[])
        self.assertFalse(result['mapping']['qualification_asserted']);self.assertTrue(result['mapping']['all_additional_column_six_forces_required'])
        loads=np.asarray(result['loads']);self.assertEqual(loads.shape,(len(floor.dof_ids),7));self.assertGreater(np.max(abs(loads)),1.)
        for col in loads.T:
            np.testing.assert_allclose(spatial_resultant([floor.node_positions[k] for k in floor.node_ids],col.reshape(-1,6)),0.,atol=1e-10)
        with self.assertRaises(ValueError):prepare_coupled_floor_cases(adapter,floor,inf,'signed_X_30Y',conventional_forces=conventional(),selected_case_ids=['invented'])
    def test_actual_centroid_wrenches_and_rigid_offset_virtual_work(self):
        mesh=rectangular_mesh([0.,10.],[0.,8.],thickness_in=2.,elastic_modulus_ksi=100.,beams=[
            dict(id='x',member_id='a',i=0,j=1,area_web_in2=24.,axis='x'),
            dict(id='y',member_id='b',i=1,j=3,area_web_in2=40.,axis='y')])
        case=dict(beam_axial=[dict(id='x',N_kip=7.),dict(id='y',N_kip=-3.)])
        result=eccentric_web_embedding(mesh,case,{'a':dict(h_in=10.,b_in=3.),'b':dict(h_in=12.,b_in=4.)},slab_thickness_in=2.)
        np.testing.assert_allclose(result['segments'][0]['positive_end_wrench_at_slab_plane'],[7.,0.,0.,0.,-35.,0.])
        np.testing.assert_allclose(result['segments'][1]['positive_end_wrench_at_slab_plane'],[0.,-3.,0.,-18.,0.,0.])
        np.testing.assert_allclose(result['summed_constraint_moments_kipin'],0.,atol=1e-14)
        q=np.arange(24).reshape(4,6)*.001;direct=0.
        for row in result['segments']:
            i,j=row['i'],row['j'];r=np.array([0.,0.,row['centroid_z_in']]);p=np.array(mesh['nodes'])
            unit=np.r_[p[j]-p[i],0.];unit/=np.linalg.norm(unit)
            ui=q[i,:3]+np.cross(q[i,3:],r);uj=q[j,:3]+np.cross(q[j,3:],r)
            direct+=row['N_kip']*np.dot(unit,uj-ui)
        self.assertAlmostEqual(direct,float(np.sum(q*np.array(result['physical_web_internal_nodal_wrenches']))),places=13)
        np.testing.assert_allclose(np.array(result['held_rotation_reactions'])+result['released_rotation_correction_loads'],0.)
        self.assertFalse(result['actual_support_path_verified'])

    def test_physical_cut_keeps_planar_torsion_and_additional_strong_axis_moment_separate(self):
        mesh=rectangular_mesh([0.,10.],[0.,8.],thickness_in=2.,elastic_modulus_ksi=100.,beams=[dict(id='x',member_id='a',i=2,j=3,area_web_in2=24.,axis='x')])
        result=eccentric_web_embedding(mesh,dict(beam_axial=[dict(id='x',N_kip=7.)]),{'a':dict(h_in=10.,b_in=3.)},slab_thickness_in=2.)
        cut=eccentric_web_cut(mesh,result,'x',5.)
        np.testing.assert_allclose(cut['physical_web_wrench_at_slab_origin'],[7.,0.,0.,0.,-35.,-56.])
        with self.assertRaises(ValueError):eccentric_web_embedding(mesh,dict(beam_axial=[]),{'a':dict(h_in=10.,b_in=3.)},slab_thickness_in=2.)

if __name__=='__main__':unittest.main()
