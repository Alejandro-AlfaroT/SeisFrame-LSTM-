"""Closed-form interior/edge/corner strength and independent stress-cut integrals."""
import math
import unittest
import numpy as np
from Design.SMRF_Slab_Local_Shear import (critical_perimeter,recover_slab_perimeter,
    check_two_way_strength,local_minimum_steel)
from test_physical_floor_cuts import shell


def perimeter(kind='interior'):
    fp=dict(x_min_in=-100.,x_max_in=100.,y_min_in=-100.,y_max_in=100.)
    if kind in ('edge','corner'):fp['x_min_in']=-8.
    if kind=='corner':fp['y_min_in']=-8.
    p=critical_perimeter(fp,column_x_in=0.,column_y_in=0.,column_h_x_in=20.,column_b_y_in=30.,
                         effective_depth_x_in=5.,effective_depth_y_in=7.)
    return fp,p


class LocalShearTests(unittest.TestCase):
    def test_interior_geometry_matches_aci_closed_Jc_formula(self):
        _,p=perimeter();d,lx,ly=6.,26.,36.
        self.assertEqual(p['support_type'],'interior');self.assertEqual(p['bo_in'],124.)
        np.testing.assert_allclose(p['centroid_in'],[0,0],atol=1e-14)
        jx=d*lx**3/6+lx*d**3/6+d*ly*lx**2/2
        jy=d*ly**3/6+ly*d**3/6+d*lx*ly**2/2
        np.testing.assert_allclose(p['Jc_in4'],[[jx,0],[0,jy]],atol=1e-10)

    def test_edge_corner_exact_open_perimeters_and_general_Jc(self):
        for kind,bo,alpha in [('edge',78.,30.),('corner',47.,20.)]:
            _,p=perimeter(kind);self.assertEqual(p['support_type'],kind)
            self.assertEqual(p['bo_in'],bo);self.assertEqual(p['alpha_s'],alpha)
            # Independent 3-point Gaussian integral of the actual vertical
            # shear walls; no production covariance/formula called here.
            samples=[]
            xi,wi=np.polynomial.legendre.leggauss(3)
            for s in p['sides']:
                a,b=s['interval_in'];tangent=np.array([0,1]) if s['axis']=='x' else np.array([1,0])
                for v,w in zip(xi,wi):
                    t=(a+b)/2+v*(b-a)/2
                    xy=np.array([s['station_in'],t] if s['axis']=='x' else [t,s['station_in']])
                    samples.append((xy,w*(b-a)/2,tangent))
            center=sum(x*w for x,w,_ in samples)/bo
            j=sum(w*(6*np.outer(x-center,x-center)+6**3/12*np.outer(t,t)) for x,w,t in samples)
            np.testing.assert_allclose(p['centroid_in'],center,atol=1e-12)
            np.testing.assert_allclose(p['Jc_in4'],j,atol=1e-9)
            if kind=='corner':self.assertNotEqual(p['Jc_in4'][0][1],0.)

    def test_biaxial_linear_shear_and_concrete_minimum_expressions(self):
        _,p=perimeter();r=check_two_way_strength(p,fc_ksi=4.,slab_resultant_kip_kipin=[20.,60.,-90.],direct_shear_peak_ksi=0.)
        base=20/(124*6);gf_x=1/(1+2/3*math.sqrt(26/36));gf_y=1/(1+2/3*math.sqrt(36/26))
        expected=base+(1-gf_x)*90*13/p['Jc_in4'][0][0]+(1-gf_y)*60*18/p['Jc_in4'][1][1]
        self.assertAlmostEqual(r['code_peak_shear_ksi'],expected,places=13)
        expected_vc=min(4.,2+4/(30/20),2+40*6/124)*math.sqrt(4000)/1000
        self.assertAlmostEqual(r['vc_ksi'],expected_vc,places=13)
        self.assertLess(max(abs(x) for x in r['moment_transfer_residual_kipin']),1e-10)
        _,edge=perimeter('edge');high=check_two_way_strength(edge,fc_ksi=25.,slab_resultant_kip_kipin=[0,0,0],direct_shear_peak_ksi=.31)
        self.assertEqual(high['status'],'pass');self.assertEqual(high['conservative_peak_status'],'fail')
        self.assertAlmostEqual(high['phi_vc_ksi'],.3,places=13)

    def test_actual_perimeter_polynomial_shear_and_moment_balance(self):
        c=np.zeros((4,8));c[1,6]=.07;c[2,7]=.09 # Qx=.07x,Qy=.09y
        # One element spans each entire section, so inside/outside traces are
        # identical and no meshline recovery assumption enters this oracle.
        e=shell('physical',-100,100,-100,100,c)
        for kind in ('interior','edge','corner'):
            _,p=perimeter(kind);r=recover_slab_perimeter({'elements':[e]},p,trace='outside')
            direct=np.zeros(3);peak=0.
            for side in p['sides']:
                a,b=side['interval_in'];xi,wi=np.polynomial.legendre.leggauss(4)
                for v,w in zip(xi,wi):
                    t=(a+b)/2+v*(b-a)/2;x,y=(side['station_in'],t) if side['axis']=='x' else (t,side['station_in'])
                    nx,ny=side['outward_normal'];q=.07*x*nx+.09*y*ny
                    direct+=w*(b-a)/2*np.array([q,(y-p['centroid_in'][1])*q,-(x-p['centroid_in'][0])*q])
                for t in (a,b):
                    x,y=(side['station_in'],t) if side['axis']=='x' else (t,side['station_in']);nx,ny=side['outward_normal']
                    peak=max(peak,abs(.07*x*nx+.09*y*ny)/6)
            np.testing.assert_allclose(r['slab_resultant_kip_kipin'],direct,atol=1e-11)
            self.assertAlmostEqual(r['maximum_direct_slab_shear_ksi'],peak,places=12)
            if kind=='interior':self.assertAlmostEqual(direct[0],.16*26*36,places=10)

    def test_actual_local_minimum_steel_trigger_band_and_unknown(self):
        fp,p=perimeter('corner');fy=60.;v=.15;provided={n:2. for n in ('x_top','x_bottom','y_top','y_bottom')}
        r=local_minimum_steel(p,fp,slab_thickness_in=8.,fc_ksi=4.,fy_ksi=fy,vuv_ksi=v,provided_developed_area_in2=provided)
        self.assertTrue(r['triggered'])
        self.assertAlmostEqual(r['checks']['x_top']['bslab_in'],35.)
        self.assertAlmostEqual(r['checks']['x_top']['required_in2'],5*v*35*47/(.75*20*60))
        missing=local_minimum_steel(p,fp,slab_thickness_in=8.,fc_ksi=4.,fy_ksi=60.,vuv_ksi=0.,provided_developed_area_in2={})
        self.assertFalse(missing['triggered']);self.assertEqual(missing['status'],'pass')
        missing=local_minimum_steel(p,fp,slab_thickness_in=8.,fc_ksi=4.,fy_ksi=60.,vuv_ksi=v,provided_developed_area_in2={})
        self.assertEqual(missing['status'],'unknown')

    def test_native_shell_transverse_shear_sign_matches_virtual_work(self):
        import openseespy.opensees as ops
        from Design.SMRF_Floor_Substructure import native_shell_kernel
        try:
            k=native_shell_kernel(20.,30.,8.,.2);u=np.zeros((4,6));u[:,2]=k['points'][:,0]*.001
            f=(k['K']@u.ravel()).reshape(4,6);r=(k['raw_resultant']@u.ravel()).reshape(4,8)
            self.assertGreater(r[:,6].mean(),0.)
            self.assertAlmostEqual(f[[1,2],2].sum(),30*r[:,6].mean(),places=10)
            # Positive right-face Q is the positive-Z traction on that face.
            self.assertAlmostEqual(f[[0,3],2].sum(),-30*r[:,6].mean(),places=10)
        finally:ops.wipe()


if __name__=='__main__':unittest.main()
