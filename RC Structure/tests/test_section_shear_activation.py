"""Inactive shear branch, reflection and unchanged bounded acceptance."""
import unittest
import json
from pathlib import Path
import numpy as np
from test_spatial_floor_section import section
from Design.SMRF_Spatial_Floor_Section import SOLVER_METHOD,SpatialFloorSection


class ShearActivation(unittest.TestCase):
    def test_signed_seed_is_domain_valid_and_no_acceptance_claim(self):
        s=section();q=np.array([.0005,.0004,0.,0.,0.,0.,0.])
        for sign in (-1.,1.):
            target=np.array([1.,1.,sign*.1,0.,0.,sign*.05,0.])
            trial,audit=s._inactive_shear_seed(q,target)
            self.assertEqual(np.sign(trial[2]),sign)
            self.assertTrue(audit['initialization_only']);self.assertNotIn('passed',audit)
            self.assertTrue(s.state(trial)['material_domain_valid'])
            slab=next(r for r in s.state(trial)['concrete_regions'] if r['id']=='slab')
            self.assertGreater(abs(slab['nominal_resultants_per_in'][2]),0.)
        self.assertIsNone(s._inactive_shear_seed(q,np.zeros(7)))
        active=-q;active[2]=.0001
        self.assertIsNone(s._inactive_shear_seed(active,[0,0,.1,0,0,0,0]))

    def test_reflected_known_feasible_roots_and_independent_final_replay(self):
        fixture=json.loads(Path(__file__).with_name('section_shear_activation_220_fixture.json').read_text())
        fixture['cases']+=json.loads(Path(__file__).with_name('section_shear_activation_225_fixture.json').read_text())['cases']
        for row in fixture['cases']:
            s=SpatialFloorSection(**row['constructor'])
            for sign in (-1.,1.):
                # Reflection changes only engineering shear/twist signs.
                demand=np.asarray(row['target']).copy();demand[[2,5]]*=sign
                start=np.asarray(row['initial_strain']).copy();start[[2,5]]*=sign
                out=s.solve(demand,initial_strain=start,max_evaluations=250)
                self.assertEqual(out['status'],'pass',out)
                self.assertEqual(out['numerical_solver_method'],SOLVER_METHOD)
                self.assertLessEqual(out['evaluations'],250)
                self.assertTrue(any('slab_shear_activation' in a['stage'] and a.get('initialization',{}).get('initialization_only') is True for a in out['solve_attempts']))
                final=s.state(out['generalized_strain'],quadrature_order=out['quadrature']['orders'][1])
                self.assertTrue(final['material_domain_valid'])
                self.assertTrue(np.all(abs(np.asarray(final['design_resultants'])-demand)<=1e-6+1e-6*abs(demand)))

    def test_exhaustion_remains_unknown_and_cannot_expand_budget(self):
        s=section();out=s.solve([0.,0.,1e6,0.,0.,0.,0.],initial_strain=[.0005,.0004,0.,0.,0.,0.,0.],max_evaluations=1)
        self.assertEqual(out['status'],'unknown');self.assertEqual(out['evaluations'],1)
        self.assertFalse(out['restricted_model_state_verified'])
        self.assertEqual(len(out['solve_attempts']),1)


if __name__=='__main__':unittest.main()
