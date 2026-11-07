import unittest,json
from unittest.mock import patch
import numpy as np
from Design.SMRF_Floor_Substructure import GeneralizedFloorStiffness
from Design.SMRF_Beam_Shell_Rotation_Dofs import retain_independent_beam_rotations
from Design.SMRF_Paired_Floor_Operator import PairedFloorOperator
from Design.SMRF_Paired_Physical_Equilibrium import solve_paired_dirichlet
from test_floor_generalized_stiffness import floor_fixture

def model_fixture():
    floor=floor_fixture();retain_independent_beam_rotations(floor)
    base=GeneralizedFloorStiffness(floor,beam_bending_factor=.5,slab_bending_factors=.2,web_torsion_factor=.1,
        stiffness_basis='independent numerical fixture',input_identity='paired_floor_251')
    with patch('Design.SMRF_Implicit_Membrane_Energy.splu',side_effect=AssertionError('Unexpected repeated membrane factor')):
        model=PairedFloorOperator(base,input_identity='paired_floor_251_actual')
    return floor,base,model

class PairedFloorOperatorTests(unittest.TestCase):
    def test_one_factor_reused_and_local_partition_includes_joint_load_distinction(self):
        floor,base,model=model_fixture();support=floor.supports[0]['id']
        wrench=[.3,-.2,-.7,1.1,-.4,.05];loads=floor.loads(-.002,joint_wrenches={support:wrench})
        boundary=np.array([floor.dof_ids.index(d) for d in floor.boundary_dofs])
        solved=solve_paired_dirichlet(model.apply_pair,base.stiffness,boundary,np.zeros(len(boundary)),loads['loads'],input_identity='paired_loaded_floor')
        raw=model.recover_pair(solved['displacements_hi'],solved['displacements_lo'],loads)
        force=solved['physical_action_hi']+solved['physical_action_lo'];actual=np.array(raw['assembled_element_resistance']);joint=np.zeros(len(force))
        for dof,value in enumerate(wrench,1):joint[floor.dof_ids.index((support,dof))]=value
        np.testing.assert_allclose(actual-(force-loads['loads']),joint,rtol=1e-9,atol=1e-9)
        self.assertLess(raw['force_partition_relative_error'],1e-9)
        self.assertEqual(model.energy.metadata['membrane_factorizations'],0)
        self.assertIs(model.energy.factor,base._relaxation_factor)
        self.assertFalse(raw['equilibrium_inferred_from_recovery'])
        self.assertEqual(json.loads(json.dumps(raw['joint_wrenches'])),[dict(node_id=support,wrench_kip_kipin=wrench)])
        self.assertEqual(model.metadata['stiffness_role'],'preconditioner_only')
        with self.assertRaises(ValueError):model.recover(solved['displacements_hi'],loads)

    def test_independent_dense_physical_energy_matches_action_and_saved_recovery(self):
        floor,base,model=model_fixture();M=(model.energy.M+model.energy.Mlow).toarray();O=(model.energy.O+model.energy.Olow).toarray()
        m,b=model.energy.m,model.energy.b;free=model.energy.free;A=M[np.ix_(m,m)];B=M[np.ix_(m,b)];z=np.zeros_like(B)
        z[free]=np.linalg.solve(A[np.ix_(free,free)],B[free]);dense=M+O
        dense[np.ix_(b,b)]+=(base.beta-1)*(M[np.ix_(b,b)]-B.T@z)
        uh=np.random.default_rng(251).normal(size=len(floor.dof_ids))*1e-4;ul=np.spacing(uh)*.25
        ah,al=model.apply_pair(uh,ul);np.testing.assert_allclose(ah+al,dense@(uh+ul),rtol=1e-8,atol=1e-8)
        first=model.recover_pair(uh,ul,floor.loads(-.002));second=model.recover_pair(uh.copy(),ul.copy(),floor.loads(-.002))
        self.assertEqual(first,second)
        self.assertLess(first['force_partition_relative_error'],1e-9)

    def test_missing_physical_loads_and_invalid_factor_reuse_refused(self):
        floor,_,model=model_fixture();loads=floor.loads(-.002);loads['element_loads'].pop(next(iter(loads['element_loads'])))
        with self.assertRaises(ValueError):model.recover_pair(np.zeros(len(floor.dof_ids)),np.zeros(len(floor.dof_ids)),loads)
        with self.assertRaises(ValueError):PairedFloorOperator(object(),input_identity='incomplete')
        # Reusing a bad solve cannot manufacture physical acceptance.
        from types import SimpleNamespace
        model.energy.factor=SimpleNamespace(solve=lambda rhs:np.zeros_like(rhs))
        with self.assertRaisesRegex(ValueError,'equilibrium'):
            model.apply_pair(np.random.default_rng(252).normal(size=len(floor.dof_ids)),np.zeros(len(floor.dof_ids)))

if __name__=='__main__':unittest.main()
