"""Independent affine-field, line equilibrium and physical ownership witnesses."""
import copy
import math
import unittest
import numpy as np
from Design.SMRF_Physical_Floor_Cuts import (slab_tensor,slab_line_integral,web_section,
    common_strip_cut,disjoint_transverse_partition,bar_clear_interval)


def shell(name,x0,x1,y0,y1,coefficients):
    a=1/math.sqrt(3);xy=[(x0+(1+xi)*(x1-x0)/2,y0+(1+eta)*(y1-y0)/2)
                            for xi,eta in [(-a,-a),(a,-a),(a,a),(-a,a)]]
    values=[np.array([1,x,y,x*y])@coefficients for x,y in xy]
    return dict(id=name,kind='slab',positions_in=[[x0,y0,0],[x1,y0,0],[x1,y1,0],[x0,y1,0]],
                raw_resultants=np.array(values).tolist())


def beam(axis='x',n=7.,q=-2.,length=10.,m0=3.,v0=8.,z=-6.):
    a=0 if axis=='x' else 1;rot=4 if axis=='x' else 3;sign=-1 if axis=='x' else 1
    f=np.zeros(12);f[a]=-n;f[6+a]=n;f[2]=v0;f[8]=-q*length-v0
    f[rot]=sign*m0;f[6+rot]=-sign*(m0-v0*length-q*length*length/2)
    p=[[0.,0.,z],[0.,0.,z]];p[1][a]=length
    return dict(id='beam:0',kind='web',member_id='beam',axis=axis,positions_in=p,
                global_resisting_force=f.tolist(),section={'b_in':2.})


class PhysicalCutTests(unittest.TestCase):
    def test_gauss_order_and_exact_bilinear_tensor_line_integral(self):
        c=np.arange(32,dtype=float).reshape(4,8)/17
        e=shell('s',2.,9.,-3.,5.,c)
        for x,y in ((2,-3),(9,5),(3.7,.13)):
            np.testing.assert_allclose(slab_tensor(e,x,y),np.array([1,x,y,x*y])@c,atol=1e-12)
        for axis,s,lo,hi in (('x',3.,-2.,4.),('y',2.,3.,8.)):
            actual=slab_line_integral([e],axis=axis,station_in=s,interval_in=[lo,hi],trace='plus')
            mean=(lo+hi)/2;xy=(s,mean) if axis=='x' else (mean,s)
            expected=(hi-lo)*(np.array([1,*xy,xy[0]*xy[1]])@c)
            np.testing.assert_allclose(actual['integral'],expected,atol=1e-12)

    def test_one_sided_traces_preserve_jumps_and_reject_gaps_duplicates(self):
        c=np.zeros((4,8));c[0]=np.arange(8)
        left=shell('left',0,10,0,10,c);right=shell('right',10,20,0,10,3*c)
        for trace,factor in (('minus',1),('plus',3)):
            r=slab_line_integral([left,right],axis='x',station_in=10,interval_in=[0,10],trace=trace)
            np.testing.assert_allclose(r['integral'],10*factor*c[0],atol=1e-12)
        for rows in ([left,left],[],[shell('partial',0,10,0,9,c)]):
            with self.assertRaisesRegex(ValueError,'gap|overlap|cover'):
                slab_line_integral(rows,axis='x',station_in=5,interval_in=[0,10],trace='plus')

    def test_exact_web_section_gravity_moment_sign_and_endpoint_balance(self):
        # Independent simply supported downward load: tension at bottom means
        # common M=int(z*sigma) is negative, of magnitude w*L^2/8 at midspan.
        for axis in ('x','y'):
            row=beam(axis,n=7,q=-2,length=10,m0=0,v0=10)
            r=web_section(row,5,uniform_vertical_load_kip_in=-2)
            self.assertAlmostEqual(r['N_tension_kip'],7.)
            self.assertAlmostEqual(r['M_centroid_kipin'],-25.)
            self.assertAlmostEqual(r['Q_vertical_kip'],0.)
            for x in (0,3,10):
                r=web_section(row,x,uniform_vertical_load_kip_in=-2)
                self.assertAlmostEqual(r['M_centroid_kipin'],-10*x+x*x)
            bad=copy.deepcopy(row);bad['global_resisting_force'][8]+=1.
            with self.assertRaisesRegex(ValueError,'equilibrium'):web_section(bad,5,uniform_vertical_load_kip_in=-2)

    def test_common_cut_transports_only_actual_web_and_same_slab_strip(self):
        c=np.zeros((4,8));c[0]=[1,2,3,4,5,6,7,8]
        response=dict(elements=[shell('s',0,10,-5,5,c),beam()],force_partition_relative_error=1e-13)
        r=common_strip_cut(response,axis='x',station_in=5,interval_in=[-2,2],trace='plus',web_id='beam',
            uniform_web_loads={'beam:0':-2},state_identity='s',combination_id='c',location_id='l',recovery_id='r')
        # N slab4+web7. M slab16 + web(3-8*5+25) + eccentric(-6*7).
        np.testing.assert_allclose(r['resultants'],[11,8,12,-38,20,24],atol=1e-12)
        self.assertFalse(r['accounting']['common_width_strain_projection_verified'])
        self.assertFalse(r['accounting']['core_remote_coverage_verified'])

    def test_actual_width_partition_and_bar_clear_enlargement(self):
        a=bar_clear_interval([-2,2],[-10,10],[dict(db_in=1.,global_transverse_coordinates_in=[-2.,0.,2.])])
        self.assertLess(a[0],-2.5);self.assertGreater(a[1],2.5)
        p=disjoint_transverse_partition([-10,10],[('a',a[0],a[1]),('b',5.,7.)])
        self.assertAlmostEqual(p['coverage_residual_in'],0.)
        self.assertEqual(len(p['pieces']),5)
        with self.assertRaisesRegex(ValueError,'overlap'):disjoint_transverse_partition([-10,10],[('a',-2,2),('b',1,4)])


if __name__=='__main__':unittest.main()
