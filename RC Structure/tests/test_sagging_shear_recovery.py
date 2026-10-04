"""Mesh-independent manufactured extrema at a changing moment-sign boundary."""
import copy
import unittest

from test_smrf_slab_recovery import panel
from Design.SMRF_Slab_Recovery import recover_sagging_shear


class SaggingShearRecovery(unittest.TestCase):
    geometry = dict(num_bay_x=1, num_bay_y=1, bay_x_in=40., bay_y_in=40.)
    faces = dict(x=(14., 14.), y=(14., 14.))

    def test_moving_gauss_mask_recovers_same_continuous_boundary(self):
        for spacing in (20., 10., 5., 2.5):
            grid = [i * spacing for i in range(int(40 / spacing) + 1)]
            for twist, expected in ((0., 7.), (1., 6.5)):
                p = panel(grid, grid, lambda x, y, i, j: dict(
                    mx=x-13., my=2., mxy_raw=twist, qx_raw=20.-x, qy_raw=0.))
                # Use a positive shear throughout the region to put its maximum
                # at the moment-sign boundary rather than the far panel edge.
                for point in p['gauss_point_resultants']:
                    point['qx_raw'] += 20.
                before = copy.deepcopy(p)
                r = recover_sagging_shear(p, self.geometry, self.faces)['x']
                target = expected + 20.
                self.assertGreaterEqual(r['shear_upper_kip_per_in'], target-1e-10)
                self.assertLessEqual(r['shear_upper_kip_per_in'], target * 1.00051)
                self.assertLessEqual(r['shear_witness_kip_per_in'], target+1e-10)
                self.assertEqual(r['witness']['top_demand'], 0.)
                self.assertEqual(p, before)

    def test_curved_twisting_boundary_uses_tensor_before_wood_armer(self):
        g = dict(self.geometry, bay_x_in=20., bay_y_in=20.)
        p = panel([0., 10., 20.], [0., 10., 20.], lambda x,y,i,j: dict(
            mx=x-4., my=y, mxy_raw=4., qx_raw=20.-x, qy_raw=0.))
        r = recover_sagging_shear(p, g, dict(x=(2.,2.), y=(2.,2.)))['x']
        expected = 20. - (4. + 16./19.)
        self.assertGreaterEqual(r['shear_upper_kip_per_in'], expected-1e-10)
        self.assertLessEqual(r['shear_upper_kip_per_in'], expected*1.00051)

    def test_retains_element_side_peak_and_excludes_beam_widths(self):
        p = panel([0.,7.,20.,33.,40.], [0.,7.,33.,40.], lambda x,y,i,j: dict(
            mx=1., my=1., mxy_raw=0., qx_raw=1000. if i in (0,3) or j != 1 else (1. if i==1 else 5.),
            qy_raw=0.))
        r = recover_sagging_shear(p, self.geometry, self.faces)['x']
        self.assertAlmostEqual(r['shear_upper_kip_per_in'], 5.)

    def test_hogging_field_has_no_bottom_shear_and_bad_input_is_refused(self):
        p = panel([0.,20.,40.], [0.,20.,40.], lambda x,y,i,j: dict(
            mx=-1., my=-1., mxy_raw=0., qx_raw=3., qy_raw=4.))
        result = recover_sagging_shear(p, self.geometry, self.faces)
        self.assertTrue(all(r['shear_upper_kip_per_in']==0. for r in result.values()))
        p['gauss_point_resultants'].pop()
        with self.assertRaises(ValueError):
            recover_sagging_shear(p, self.geometry, self.faces)

    def test_unclosed_bound_is_an_error_not_a_false_pass(self):
        p = panel([0.,20.,40.], [0.,20.,40.], lambda x,y,i,j: dict(
            mx=x-13., my=-1., mxy_raw=0., qx_raw=40.-x, qy_raw=0.))
        with self.assertRaisesRegex(ValueError, 'did not close'):
            recover_sagging_shear(p, self.geometry, self.faces, max_subdivisions=1)

    def test_transpose_preserves_axis_specific_sagging_envelope(self):
        a = panel([0.,10.,40.], [0.,20.,40.], lambda x,y,i,j: dict(
            mx=x-13., my=2., mxy_raw=1., qx_raw=40.-x, qy_raw=0.))
        b = panel([0.,20.,40.], [0.,10.,40.], lambda x,y,i,j: dict(
            mx=2., my=y-13., mxy_raw=1., qx_raw=0., qy_raw=40.-y))
        ra = recover_sagging_shear(a, self.geometry, self.faces)['x']
        rb = recover_sagging_shear(b, self.geometry, self.faces)['y']
        self.assertAlmostEqual(ra['shear_upper_kip_per_in'], rb['shear_upper_kip_per_in'], places=9)


if __name__ == '__main__':
    unittest.main()
