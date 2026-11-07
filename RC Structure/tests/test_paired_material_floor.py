import unittest
import numpy as np
from Design.SMRF_Material_Floor import MaterialFloorOperator
from Design.SMRF_Paired_Material_Floor import PairedMaterialFloorOperator
from test_floor_generalized_stiffness import floor_fixture
from test_material_floor_operator import inputs


class PairedMaterialTests(unittest.TestCase):
    def test_actual_ABD_and_web_forces_match_native_full_material_reference(self):
        floor=floor_fixture();slab,webs,shear=inputs(floor)
        # Different positive definite full matrices at all Gauss points,
        # including membrane-bending coupling and a distinct shear factor.
        rng=np.random.default_rng(255)
        for key in slab:
            values=[]
            for gp in range(4):
                a=rng.normal(size=(6,6));values.append(a.T@a+np.eye(6)*3)
            slab[key]=values;shear[key]=.7
        reference=MaterialFloorOperator(floor,slab_ABD=slab,web_matrices=webs,transverse_shear_factors=shear,
            source_identity='independent_full_ABD',material_state_identity='four_distinct_GP')
        paired=PairedMaterialFloorOperator(reference,input_identity='paired_material_255')
        h=rng.normal(size=len(floor.dof_ids))*1e-4;l=np.spacing(h)*.3;loads=floor.loads(-.002,joint_wrenches={floor.supports[0]['id']:[0,0,0,1.2,-.4,.1]})
        ah,al=paired.apply_pair(h,l);np.testing.assert_allclose(ah+al,reference.stiffness@(h+l),rtol=1e-9,atol=1e-8)
        actual=paired.recover_pair(h,l,loads);expected=reference.recover(h+l,loads)
        for a,b in zip(actual['elements'],expected['elements']):
            np.testing.assert_allclose(a['global_resisting_force'],b['global_resisting_force'],rtol=1e-9,atol=1e-8)
            if a['kind']=='slab':
                np.testing.assert_allclose(a['raw_resultants'],b['raw_resultants'],rtol=1e-9,atol=1e-8)
                np.testing.assert_allclose(a['common_generalized_strains'],b['common_generalized_strains'],rtol=1e-9,atol=1e-12)
        self.assertLess(actual['force_partition_relative_error'],1e-9)
        self.assertEqual(actual,paired.recover_pair(h.copy(),l.copy(),loads))
        self.assertEqual(paired.metadata['material_state_identity'],'four_distinct_GP')
        with self.assertRaises(ValueError):paired.recover(h,loads)


if __name__=='__main__':unittest.main()
