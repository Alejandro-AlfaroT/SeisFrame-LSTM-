"""Complete-system signed inertia branches retain gravity and every column."""
import unittest
import copy
from unittest.mock import patch
import numpy as np
from Design.SMRF_Physical_Seismic_Design import PhysicalSeismicSystem
from test_coupled_floor_geometry import fixture
from test_physical_diaphragm import masses


class PhysicalSeismicTests(unittest.TestCase):
    def test_complete_signed_loads_reactions_and_unchanged_gravity(self):
        definition=fixture()
        with patch('Design.SMRF_Physical_Seismic_Design.installed_grid_masses',return_value=masses(definition)):
            model=PhysicalSeismicSystem(definition,[dict(id='G',dead=1.,live=1.)],source_identity='actual_fixture',
                seismic_live_load_fraction=.25,subdivisions_per_bay=2,max_shells_per_floor=1024,
                max_coupled_dofs=1500,max_cases=3)
        load=np.array([[.1,-.03,.2],[.2,-.06,-.1]])
        cases=[dict(id=key,gravity_case_id='G',loads=(load*sign).tolist(),reference_xy_in=[20.,8.],
                    demand_basis_identity='independent_signed_load') for key,sign in [('plus',1),('minus',-1),('zero',0)]]
        result=model.solve(cases,input_identity='finite_signed_fixture',pdelta=False)
        with patch('Design.SMRF_Substructure_Frame.SubstructureFrame.solve',side_effect=AssertionError('Raw field replay cannot solve')):
            replay=model.solve(cases,input_identity='finite_signed_fixture',pdelta=False,saved_result=result)
        self.assertEqual(replay['status'],'responses_computed')
        changed=copy.deepcopy(result);changed['groups']['G']['total_response']['floors'][0]['responses'][1]['elements'][0]['global_resisting_force'][0]+=.1
        with self.assertRaisesRegex(ValueError,'Saved physical field'):
            model.solve(cases,input_identity='finite_signed_fixture',pdelta=False,saved_result=changed)
        self.assertEqual(result['status'],'responses_computed')
        group=result['groups']['G'];physical=group['total_response'];columns=group['increments']['plus']['columns']
        np.testing.assert_allclose(group['increments']['plus']['displacement_increment'],
            -np.array(group['increments']['minus']['displacement_increment']),atol=1e-10)
        np.testing.assert_allclose(group['increments']['zero']['displacement_increment'],0.,atol=1e-10)
        # Independently sum base column nodal wrenches in global coordinates.
        reaction=np.zeros(6);byid={n['id']:np.array(n['xyz']) for n in definition['nodes']}
        for col in definition['columns']:
            if col['story']!=1:continue
            f=np.array(columns[col['id']]['global_force_increment'][:6]);p=byid[col['i']]
            reaction+=np.r_[f[:3],f[3:]+np.cross(p,f[:3])]
        external=np.zeros(6)
        for k,(fx,fy,mz) in enumerate(load,1):
            external+=np.array([fx,fy,0.,-k*10*fy,k*10*fx,mz+20*fy-8*fx])
        np.testing.assert_allclose(reaction,-external,atol=1e-8)
        self.assertEqual(len(columns),len(definition['columns']))
        self.assertFalse(result['qualification_asserted']);self.assertFalse(result['primary_moment_guard_complete'])
        with self.assertRaisesRegex(ValueError,'bounded'):
            model.solve(cases+[{**cases[0],'id':'fourth'}],input_identity='invalid')
        with self.assertRaisesRegex(ValueError,'simultaneous gravity'):
            model.solve([{**cases[0],'gravity_case_id':'other'}],input_identity='invalid')
