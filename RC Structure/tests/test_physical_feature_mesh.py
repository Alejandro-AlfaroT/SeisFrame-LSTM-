import unittest
from Design.SMRF_Physical_Floor_Mesh import feature_grid
from test_coupled_floor_geometry import fixture


class FeatureMeshTests(unittest.TestCase):
    def test_background_point_does_not_make_sliver_beside_physical_feature(self):
        g=dict(nx=1,ny=1,lx=336.,ly=336.,nf=1)
        supports=[dict(id=f'{i}{j}',x=i*336.,y=j*336.,b_in=30.,h_in=30.) for i in (0,1) for j in (0,1)]
        beams=[dict(id=f'{axis}{at}',axis=axis,line=at,start=0.,end=336.,b_in=30.,h_in=28.) for axis in ('x','y') for at in (0.,336.)]
        args=dict(base_subdivisions_per_bay=16,slab_thickness_in=7.5,critical_depth_in=6.25,flange_boundaries={'x':[], 'y':[]})
        a=feature_grid(dict(supports=supports,beams=beams),g,refinement_level=1,**args)
        b=feature_grid(dict(supports=supports,beams=beams),g,refinement_level=2,**args)
        for coarse,fine in zip(a,b):
            self.assertIn(21.25,coarse);self.assertNotIn(21.,coarse)
            self.assertTrue(set(coarse)<=set(fine))
            self.assertGreaterEqual(min(y-x for x,y in zip(coarse,coarse[1:])),6.25/4-1e-9)

    def test_nested_pair_retains_actual_faces_and_critical_offsets(self):
        actual=fixture();floor=actual['floors'][0];g=actual['geometry']
        args=dict(base_subdivisions_per_bay=4,slab_thickness_in=actual['slab_thickness_in'],
                  critical_depth_in=actual['slab_thickness_in']*.7,flange_boundaries={'x':[], 'y':[]})
        a=feature_grid(floor,g,refinement_level=1,**args);b=feature_grid(floor,g,refinement_level=2,**args)
        for coarse,fine in zip(a,b):
            self.assertTrue(set(coarse)<=set(fine));self.assertGreater(len(fine),len(coarse))
        for support in floor['supports']:
            for k,axis,dim in ((0,'x','h_in'),(1,'y','b_in')):
                for sign in (-1,1):
                    value=support[axis]+sign*(support[dim]/2+args['critical_depth_in']/2)
                    if a[k][0]<value<a[k][-1]:self.assertTrue(any(abs(v-value)<1e-8 for v in a[k]))
        self.assertEqual(actual,fixture())


if __name__=='__main__':unittest.main()
