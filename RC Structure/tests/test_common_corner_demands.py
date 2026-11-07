"""Independent polynomial extrema for complete-panel corner demand bounds."""
import unittest
import numpy as np
from Design.SMRF_Common_Corner_Demands import panel_moment_bound,projected_panel_bound
from tests.test_physical_floor_cuts import shell


class CornerBoundsTests(unittest.TestCase):
    def field(self):
        coefficients=np.zeros((4,8));coefficients[0,3]=2.;coefficients[0,4]=-3.
        return dict(elements=[shell('panel',0.,10.,0.,8.,coefficients)])

    def test_entire_panel_bilinear_absolute_bound(self):
        r=panel_moment_bound(self.field(),[0,10,0,8])
        self.assertAlmostEqual(r['maximum_absolute_moment_kipin_per_ft']['x'],24.,11)
        self.assertAlmostEqual(r['maximum_absolute_moment_kipin_per_ft']['y'],36.,11)
        with self.assertRaisesRegex(ValueError,'cover the entire'):panel_moment_bound(self.field(),[0,11,0,8])

    def test_interior_quadratic_extremum_and_flange_clip(self):
        # Mx=2+16t(1-t), maximum6 atx5, not the endpoint2.
        r=projected_panel_bound(self.field(),[0,10,0,8],axis='x',axial_interval_in=[0,10],flange_interval_in=[2,6],coefficients=[0,16,-16])
        self.assertAlmostEqual(r['maximum_absolute_moment_kipin_per_ft'],72.)
        self.assertAlmostEqual(r['witness']['axial_in'],5.)
        r=projected_panel_bound(self.field(),[0,2,0,8],axis='x',axial_interval_in=[0,10],flange_interval_in=[2,6],coefficients=[0,16,-16])
        self.assertAlmostEqual(r['maximum_absolute_moment_kipin_per_ft'],12*(2+16*.2*.8))
        r=projected_panel_bound(self.field(),[0,10,0,1],axis='x',axial_interval_in=[0,10],flange_interval_in=[2,6],coefficients=[0,16,-16])
        self.assertEqual(r['intersecting_cells'],0)

    def test_y_axis_negative_extremum_absolute_bound(self):
        # My=-3-20t(1-t); strongestnegative=-8at y4.
        r=projected_panel_bound(self.field(),[0,10,0,8],axis='y',axial_interval_in=[0,8],flange_interval_in=[2,6],coefficients=[0,-20,20])
        self.assertAlmostEqual(r['maximum_absolute_moment_kipin_per_ft'],96.)
        self.assertAlmostEqual(r['witness']['axial_in'],4.)

if __name__=='__main__':unittest.main()
