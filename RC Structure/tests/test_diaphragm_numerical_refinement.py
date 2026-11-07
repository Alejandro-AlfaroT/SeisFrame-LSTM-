"""Independent physical backward-error checks of bounded same-LU correction."""
import unittest
from unittest import mock
import numpy as np
from Design import SMRF_Diaphragm_Actions as actions
from Design.SMRF_Physical_Diaphragm import rectangle_shape_measure,line_shape_measure,physical_membrane_cut

def manufactured_input():
    x=[0.,2.,5.,10.];y=[0.,2.,4.]
    mesh=actions.rectangular_mesh(x,y,thickness_in=1.,elastic_modulus_ksi=100.,poisson=0.)
    body=rectangle_shape_measure(x,y,[0.,10.,0.,4.])['weights'][:,None]*np.array([-.2,0.])
    edge=line_shape_measure(x,y,'y',10.,0.,4.)['weights'][:,None]*np.array([2.,0.])
    return mesh,[dict(id='quadratic',role='local_inertia',provenance='exact ux=.001x² continuum',
          inertia=body.tolist(),columns_below_on_floor=edge.tolist(),columns_above_on_floor=np.zeros_like(body).tolist())],[dict(id='quadratic',factors={'quadratic':1.})]

class NumericalRefinementTests(unittest.TestCase):
    def test_same_factor_correction_restores_independent_exact_displacement_and_cut(self):
        native_factor=actions.splu;calls=[]
        def perturbed_factor(matrix):
            factor=native_factor(matrix)
            class Perturbed:
                def solve(self,rhs):
                    out=factor.solve(rhs);calls.append(1)
                    if len(calls)==1:out[0]+=.001
                    return out
            return Perturbed()
        with mock.patch.object(actions,'splu',side_effect=perturbed_factor):
            result=actions.solve_diaphragm_actions(*manufactured_input(),equilibrium_tolerance=1e-10)
        row=result['cases'][0]
        self.assertGreater(row['initial_normalized_nodal_residual'],1e-10)
        self.assertEqual(row['iterative_refinement_corrections'],1)
        self.assertLess(row['normalized_nodal_residual'],1e-10)
        # Independent ux=.001x² differences are exact modulo rigid gauge.
        np.testing.assert_allclose(np.diff(np.array(row['displacements_in'])[:4,0]),np.diff(.001*np.array([0.,2.,5.,10.])**2),atol=1e-13)
        np.testing.assert_allclose(physical_membrane_cut(result,'quadratic','x',3.5)['total_wrench_kip_kipin'],[2.8,0.,-5.6],atol=1e-12)

    def test_nonconvergent_numerical_result_still_fails_original_guard_after_three(self):
        calls=[]
        class FailedFactor:
            def solve(self,rhs):calls.append(1);return np.zeros_like(rhs)
        with mock.patch.object(actions,'splu',return_value=FailedFactor()),self.assertRaisesRegex(ValueError,'non-equilibrium'):
            actions.solve_diaphragm_actions(*manufactured_input(),equilibrium_tolerance=1e-10)
        self.assertEqual(len(calls),4)

if __name__=='__main__':unittest.main()
