"""Explicit meshes and factorization reuse compared to independent native solves."""
import unittest
import numpy as np
import openseespy.opensees as ops
from Design.SMRF_Floor_Compatibility import CompatibleFloor
from Design.SMRF_Coupled_Analysis import analyze_coupled_gravity
from Design.SMRF_Composite_Sections import recover_floor_cut
from tests.test_smrf_floor_compatibility import SLAB,GEOMETRY,SECTIONS,CASE

MESH=dict(x_offsets_in=[0.,9.,18.,120.,222.,231.,240.],y_offsets_in=[0.,11.,100.,189.,200.],max_shells=8192)

class CompatibleMeshTests(unittest.TestCase):
    def setUp(self):ops.wipe();self.addCleanup(ops.wipe)

    def test_reused_factorization_matches_independently_rebuilt_displacement_columns(self):
        for perimeter in ('centerlines','beam_outer_faces'):
            m=CompatibleFloor(SLAB,dict(GEOMETRY,num_bay_x=2),SECTIONS,2,slab_perimeter=perimeter,mesh_spec=MESH)
            fast,meta=m.stiffness();slow,_=m.stiffness(method='prescribed')
            scale=np.tile([240.]*3+[1.]*3,len(m.keys))
            a=fast*np.outer(scale,scale);b=slow*np.outer(scale,scale)
            self.assertLess(np.max(abs(a-b))/np.max(abs(b)),1e-10)
            self.assertLess(meta['interior_relative_residual'],1e-8)

    def test_nonuniform_native_model_matches_condensed_and_cut_balances(self):
        g=dict(GEOMETRY,num_bay_x=2)
        case=dict(CASE,live_pattern=[[1,0]])
        cases=[dict(CASE,dead_factor=0.,live_factor=0.),case]
        for restraint in ('finite_membrane','rigid_joints'):
            model=CompatibleFloor(SLAB,g,SECTIONS,2,slab_perimeter='beam_outer_faces',mesh_spec=MESH)
            reduced=model.solve(cases,inplane_restraint=restraint)
            native=analyze_coupled_gravity(SLAB,g,SECTIONS,cases,2,slab_perimeter='beam_outer_faces',mesh_spec=MESH,
                inplane_restraint=restraint,constraint_handler='Transformation' if restraint=='finite_membrane' else 'Lagrange')
            self.assertEqual(native['status'],'diagnostic_complete')
            u=[[j['displacement_rotation'] for j in f['column_joint_displacements']] for f in native['floors']]
            np.testing.assert_allclose(reduced['joint_displacements'],u,rtol=1e-8,atol=1e-10)
            for a,b in zip(reduced['column_actions'],native['column_actions']):
                np.testing.assert_allclose(a['global_force_kip_kip_in'],b['global_force_kip_kip_in'],rtol=1e-8,atol=1e-6)
            for axis,cut in (('x',18.),('y',11.)):
                self.assertTrue(recover_floor_cut(native,2,axis,cut)['numerical_balance_passed'])
            self.assertAlmostEqual(reduced['weight_ledger']['slab_pressure_kip'],native['weight_ledger']['slab_area_load_kip'],places=8)

if __name__=='__main__':unittest.main()
