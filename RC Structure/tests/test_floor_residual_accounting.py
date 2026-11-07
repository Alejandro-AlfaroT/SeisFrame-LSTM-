"""Hand equilibrium of nonlinear redistribution and clipped footprint wrenches."""
import copy
import unittest
import numpy as np
from test_floor_influence_decomposition import basis, hand_response
from Design.SMRF_Floor_Influences import decompose_with_second_order_residual
from Design.SMRF_Diaphragm_Actions import rectangular_mesh, footprint_traction, resultant


class ResidualAndFootprint(unittest.TestCase):
    def inputs(self):
        inf = basis(); inf['all_linearizations_converged'] = True
        load = [[10, 0, 0], [20, 0, 0]]
        direct = hand_response(load)
        # Five kip shifted between frame lines, with no new floor inertia.
        for i, amount in ((0, 5.), (1, -5.)):
            direct['columns'][i]['global_force'][6] += amount
            direct['columns'][i]['global_force'][0] -= amount
        inf['witnesses'] = [{'id': 'direct', 'loads': load, 'direct_increment': direct}]
        return inf

    def test_residual_is_preserved_separately_and_reconstructs_exactly(self):
        r = decompose_with_second_order_residual(self.inputs(), 'direct', 1)
        self.assertEqual(r['direct_reconstruction_max_abs_kip_kipin'], 0.)
        self.assertFalse(r['residual_inertial_scaling_permitted'])
        local, transfer, residual = r['components']
        self.assertEqual(residual['applied_load'], [0., 0., 0.])
        self.assertEqual(residual['role'], 'transfer')
        np.testing.assert_allclose([c['Fx_kip'] for c in residual['columns_below_on_floor']], [-5., 5.])
        self.assertEqual(local['applied_load'], [10., 0., 0.])
        self.assertTrue(all(c['equilibrium_passed'] for c in r['components']))

    def test_missing_linearization_or_unbalanced_residual_is_refused(self):
        inf = self.inputs(); inf['all_linearizations_converged'] = False
        with self.assertRaisesRegex(ValueError, 'linearization'):
            decompose_with_second_order_residual(inf, 'direct', 1)
        inf = self.inputs(); inf['witnesses'][0]['direct_increment']['columns'][0]['global_force'][6] += .1
        with self.assertRaisesRegex(ValueError, 'free body'):
            decompose_with_second_order_residual(inf, 'direct', 1)

    def test_edge_corner_transport_preserves_force_and_arbitrary_reference_moment(self):
        mesh = rectangular_mesh([0., 6., 12.], [0., 4., 8.], thickness_in=6., elastic_modulus_ksi=3000.)
        weights = {0: 1., 1: 2., 2: 1., 3: 2., 4: 4., 5: 2., 6: 1., 7: 2., 8: 1.}
        force = [13., -7.]
        for application in ([0., 0.], [0., 4.], [6., 4.]):
            nodal = footprint_traction(mesh, weights, force, application_xy_in=application,
                                       moment_kip_in=11., allow_reference_transport=True)
            np.testing.assert_allclose(resultant(mesh['nodes'], nodal, application), [13., -7., 11.], atol=1e-12)
            at_origin = 11.+application[0]*force[1]-application[1]*force[0]
            np.testing.assert_allclose(resultant(mesh['nodes'], nodal), [13., -7., at_origin], atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'centroid'):
            footprint_traction(mesh, weights, force, application_xy_in=[0., 0.], moment_kip_in=11.)


if __name__ == '__main__': unittest.main()
