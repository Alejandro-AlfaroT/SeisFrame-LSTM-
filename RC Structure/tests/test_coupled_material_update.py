"""A complete material replacement agrees with a separate native-gross solve."""
import unittest
import numpy as np
from Design.SMRF_Coupled_Floor_Design import CoupledFloorSystem
from Design.SMRF_Material_Floor import MaterialFloorOperator
from Design.SMRF_Constitutive_Floor_Iteration import solve_constitutive_case
from test_coupled_floor_geometry import fixture,build
from test_material_floor_operator import inputs


class CoupledMaterialUpdateTests(unittest.TestCase):
    def updater(self,ownership=True):
        def update(row,response,iteration):
            slab,webs,shear=inputs(row['assembly'])
            model=MaterialFloorOperator(row['assembly'],slab_ABD=slab,web_matrices=webs,
                transverse_shear_factors=shear,source_identity='independent_linear_fixture',material_state_identity='actual_gross_fixture')
            return dict(operator=model,audit=dict(method='explicit_nominal_elastic_test',state_identity='fixture',
                material_domain_valid=True,quadrature_converged=True,physical_material_ownership_verified=ownership))
        return update

    def test_complete_nominal_material_update_matches_separate_gross_solution(self):
        definition=fixture();cases=[dict(id='D',dead=1.,live=0.)];system=build(definition,cases)
        operators={}
        for row in system.floors:
            slab,webs,shear=inputs(row['assembly'])
            operators[row['description']['id']]=MaterialFloorOperator(row['assembly'],slab_ABD=slab,
                web_matrices=webs,transverse_shear_factors=shear,source_identity='test_source',material_state_identity='elastic_nominal')
        system.set_floor_operators(operators,iteration_identity='all_floors_elastic_state_1')
        result=system.solve(pdelta=True)
        independent=CoupledFloorSystem(definition,cases,source_identity='test_source',stiffness_basis='gross independentnativeoracle',
            subdivisions_per_bay=2,max_shells_per_floor=1024,max_coupled_dofs=1500,
            beam_bending_factor=1.,slab_bending_factor=1.,web_torsion_factor=1.,column_bending_factor=.7)
        expected=independent.solve(pdelta=True)
        self.assertEqual(result['status'],'equilibrated')
        np.testing.assert_allclose(result['frame']['cases'][0]['physical_displacements'],
                                   expected['frame']['cases'][0]['physical_displacements'],rtol=1e-9,atol=1e-10)
        for actual,reference in zip(result['floors'],expected['floors']):
            np.testing.assert_allclose(actual['responses'][0]['retained_displacements'],
                                       reference['responses'][0]['retained_displacements'],rtol=1e-9,atol=1e-10)
            self.assertEqual(actual['stiffness']['material_state_identity'],'elastic_nominal')

    def test_missing_or_different_geometry_cannot_partially_update_floors(self):
        system=build(fixture(),[dict(id='D',dead=1.,live=0.)]);old=[r['model'] for r in system.floors]
        with self.assertRaisesRegex(ValueError,'Complete'):
            system.set_floor_operators({},iteration_identity='bad')
        self.assertEqual(old,[r['model'] for r in system.floors])
        other=build(fixture(),[dict(id='D',dead=1.,live=0.)])
        operators={r['description']['id']:r['model'] for r in other.floors}
        with self.assertRaisesRegex(ValueError,'different physical geometry'):
            system.set_floor_operators(operators,iteration_identity='bad')
        self.assertEqual(old,[r['model'] for r in system.floors])

    def test_iteration_uses_actual_updated_force_residual_and_preserves_unknown(self):
        system=build(fixture(),[dict(id='D',dead=1.,live=0.)])
        result=solve_constitutive_case(system,self.updater(),iteration_identity='independent_linear_replacement',
            maximum_iterations=4,maximum_seconds=10.)
        self.assertEqual(result['status'],'constitutive_equilibrated')
        history=result['constitutive_iteration']['history']
        self.assertGreater(history[0]['interior_residual_ratio'],1.)
        self.assertTrue(history[-1]['actual_material_equilibrium_converged'])
        self.assertGreaterEqual(len(history),2)
        self.assertFalse(result['qualification_asserted'])
        invalid=solve_constitutive_case(system,self.updater(False),iteration_identity='unknown_material_inventory',
            maximum_iterations=2,maximum_seconds=10.)
        self.assertEqual(invalid['status'],'unresolved')
        self.assertIn('ownership_unverified',invalid['reason'])

    def test_iteration_bound_is_incomplete_not_physical_failure(self):
        system=build(fixture(),[dict(id='D',dead=1.,live=0.)])
        result=solve_constitutive_case(system,self.updater(),iteration_identity='one_iteration_bound',
            maximum_iterations=1,maximum_seconds=10.)
        self.assertEqual(result['status'],'unresolved')
        self.assertEqual(result['reason'],'declared_constitutive_iteration_count_limit')

    def test_explicit_initial_material_factory_avoids_unrelated_scalar_floor_factors(self):
        definition=fixture();cases=[dict(id='D',dead=1.,live=0.)]
        def factory(floor,desc,definition,case):
            slab,webs,shear=inputs(floor)
            return MaterialFloorOperator(floor,slab_ABD=slab,web_matrices=webs,transverse_shear_factors=shear,
                source_identity='test_source',material_state_identity=f'actual_elastic_{desc["id"]}_{case["id"]}')
        system=CoupledFloorSystem(definition,cases,source_identity='test_source',stiffness_basis='explicit actual material initialoperator',
            subdivisions_per_bay=2,max_shells_per_floor=1024,max_coupled_dofs=1,
            beam_bending_factor=None,slab_bending_factor=None,web_torsion_factor=None,column_bending_factor=.7,
            initial_operator_factory=factory)
        result=system.solve(pdelta=True)
        self.assertEqual(result['status'],'equilibrated')
        self.assertTrue(all(r['stiffness']['material_state_identity'].startswith('actual_elastic') for r in result['floors']))


if __name__=='__main__':unittest.main()
