import unittest
import numpy as np
from test_physical_floor_cuts import shell,beam
from test_slab_local_shear import perimeter
from Design.SMRF_Slab_Equilibrium_Cuts import recover_support_free_body


class EquilibriumCutTests(unittest.TestCase):
    def test_hand_loaded_rectangle_interior_edge_corner(self):
        for kind in ('interior','edge','corner'):
            fp,p=perimeter(kind);e=shell('s',fp['x_min_in'],100.,fp['y_min_in'],100.,np.zeros((4,8)))
            reaction=np.array([0.,0.,100.,20.,-30.,0.]);q=-.002
            r=recover_support_free_body(dict(elements=[e]),p,support_wrench_kip_kipin=reaction,
                pressure_by_element={'s':q},web_load_by_element={},trace='inside')
            x0,x1,y0,y1=p['bounds_in'];cx,cy=p['centroid_in'];w=q*(x1-x0)*(y1-y0)
            expected=[-100-w,-20+cy*100-((y0+y1)/2-cy)*w,30-cx*100+((x0+x1)/2-cx)*w]
            np.testing.assert_allclose(r['slab_resultant_kip_kipin'],expected,atol=1e-12)
            self.assertFalse(r['qualification_asserted'])

    def test_pure_eccentric_web_leaves_no_fictitious_slab_transfer(self):
        _,p=perimeter()
        for axis in ('x','y'):
            e=shell('s',-100,200,-100,200,np.zeros((4,8)))
            w=beam(axis,n=7,q=-.2,length=100,m0=3,v0=10,z=-6)
            a=0 if axis=='x' else 1
            for point in w['positions_in']:point[a]+=10.
            w['global_resisting_force'][3+a]=-9.;w['global_resisting_force'][9+a]=9.
            # Independent end-force equilibrium: carry the actual left-end
            # applied beam wrench to the support, including its eccentricity.
            f=np.array(w['global_resisting_force'][:6]);point=np.array(w['positions_in'][0])
            joint=np.r_[f[:3],f[3:]+np.cross(point,f[:3])]
            r=recover_support_free_body(dict(elements=[e,w]),p,support_wrench_kip_kipin=joint,
                pressure_by_element={'s':0.},web_load_by_element={w['id']:-.2},trace='inside')
            np.testing.assert_allclose(r['slab_boundary_wrench'],np.zeros(6),atol=1e-12)
            self.assertEqual(len(r['web_cuts']),1)

    def test_missing_loads_and_unbalanced_web_are_refused(self):
        _,p=perimeter();e=shell('s',-100,100,-100,100,np.zeros((4,8)));w=beam(length=100)
        with self.assertRaisesRegex(ValueError,'Every actual'):
            recover_support_free_body(dict(elements=[e,w]),p,support_wrench_kip_kipin=np.zeros(6),pressure_by_element={'s':0},web_load_by_element={},trace='inside')
        w['global_resisting_force'][8]+=1.
        with self.assertRaisesRegex(ValueError,'equilibrium'):
            recover_support_free_body(dict(elements=[e,w]),p,support_wrench_kip_kipin=np.zeros(6),pressure_by_element={'s':0},web_load_by_element={w['id']:-2},trace='inside')


if __name__=='__main__':unittest.main()
