"""Native building comparisons of the explicit-input condensed diagnostic."""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ.setdefault(name,'1')
import copy, sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import openseespy.opensees as ops
from Design.SMRF_Floor_Compatibility import CompatibleFloor
from Design.SMRF_Coupled_Analysis import analyze_coupled_gravity

SLAB=dict(thickness_in=5.,concrete_fc_ksi=4.,concrete_unit_weight_kcf=.15,superimposed_dead_load_ksf=.05)
GEOMETRY=dict(num_bay_x=1,num_bay_y=1,num_floor=2,bay_x_in=240.,bay_y_in=200.,story_h_in=144.)
SECTIONS=dict(b_beam_in=12.,h_beam_in=18.,fc_beam_ksi=4.,b_col_in=18.,h_col_in=22.,fc_col_ksi=5.)
CASE=dict(id='D+L',dead_factor=1.2,live_factor=1.6,live_load_ksf=.05,live_pattern='all')

class FloorCompatibilityTests(unittest.TestCase):
    def setUp(self):
        ops.wipe();self.addCleanup(ops.wipe)

    def compare(self,g=GEOMETRY,s=SECTIONS,restraint='finite_membrane'):
        cases=[{**CASE,'dead_factor':0.,'live_factor':0.},CASE]
        model=CompatibleFloor(SLAB,g,s,mesh_per_bay=2)
        reduced=model.solve(cases,inplane_restraint=restraint)
        # Full native reference is created only after the prediction exists.
        native=analyze_coupled_gravity(SLAB,g,s,cases,mesh_per_bay=2,
            inplane_restraint=restraint,constraint_handler='Transformation' if restraint=='finite_membrane' else 'Lagrange')
        self.assertEqual(native['status'],'diagnostic_complete')
        u=np.array([[j['displacement_rotation'] for j in floor['column_joint_displacements']] for floor in native['floors']])
        np.testing.assert_allclose(reduced['joint_displacements'],u,atol=1e-9,rtol=1e-9)
        cols={(c['story'],c['grid_i'],c['grid_j']):c for c in native['column_actions']}
        for c in reduced['column_actions']:
            ref=cols[c['story'],c['grid_i'],c['grid_j']]
            for key in ('global_force_kip_kip_in','local_force_kip_kip_in'):
                np.testing.assert_allclose(c[key],ref[key],atol=1e-6,rtol=1e-8)
        for floor in reduced['floors']:
            panels=[x for x in native['shell_resultants'] if x['floor']==floor['story']]
            for a,b in zip(floor['shells'],panels):
                np.testing.assert_allclose(a['gauss_resultants_raw'],b['gauss_resultants_raw'],atol=1e-8,rtol=1e-7)
            webs=[x for x in native['web_segment_actions'] if x['floor']==floor['story']]
            for a,b in zip(floor['webs'],webs):
                np.testing.assert_allclose(a['global_force_kip_kip_in'],b['global_force_kip_kip_in'],atol=1e-6,rtol=1e-8)
        self.assertFalse(reduced['applied_to_design'])
        self.assertLess(reduced['reduced_equilibrium_relative_residual'],1e-10)

    def test_loaded_roof_over_unloaded_floor_matches_native_finite_membrane(self):self.compare()
    def test_exact_rigid_joint_transform_matches_native_lagrange(self):self.compare(restraint='rigid_joints')
    def test_changed_geometry_and_member_stiffness_match_native(self):
        self.compare(g={**GEOMETRY,'num_bay_x':2,'bay_y_in':216.,'story_h_in':192.},
                     s={**SECTIONS,'b_beam_in':16.,'h_beam_in':26.,'beam_stiffness_modifier':.35,'column_stiffness_modifier':.7})

    def test_input_identity_changes_with_each_design_dependency(self):
        original=CompatibleFloor(SLAB,GEOMETRY,SECTIONS,2)._identity
        changes=[('sections','fc_col_ksi',8.),('sections','h_col_in',26.),('sections','h_beam_in',24.),
                 ('sections','fc_beam_ksi',6.),('slab','thickness_in',6.),('geometry','story_h_in',192.)]
        for group,key,value in changes:
            with self.subTest(group=group,key=key):
                data=dict(slab=copy.deepcopy(SLAB),geometry=copy.deepcopy(GEOMETRY),sections=copy.deepcopy(SECTIONS))
                data[group][key]=value
                self.assertNotEqual(CompatibleFloor(**data,mesh_per_bay=2)._identity,original)

    def test_mutated_inputs_cannot_reuse_stiffness(self):
        model=CompatibleFloor(SLAB,GEOMETRY,SECTIONS,2);model.sections['h_beam_in']=24.
        with self.assertRaisesRegex(ValueError,'Inputs changed'):model.solve([CASE,CASE])

    def test_existing_domain_is_preserved(self):
        model=CompatibleFloor(SLAB,GEOMETRY,SECTIONS,2)
        ops.model('basic','-ndm',3,'-ndf',6);ops.node(777,0,0,0)
        with self.assertRaisesRegex(RuntimeError,'empty'):model.solve([CASE,CASE])
        self.assertIn(777,ops.getNodeTags())

if __name__=='__main__':unittest.main()
