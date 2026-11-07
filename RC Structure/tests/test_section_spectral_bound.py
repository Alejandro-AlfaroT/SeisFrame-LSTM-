"""Independent spectral/convex support examples; no structural analyses."""
import copy
import unittest
from types import SimpleNamespace
import numpy as np
from numpy.polynomial.legendre import leggauss
from Design.SMRF_Section_Spectral_Bound import principal_stress_support_upper_bound as upper,principal_capacity_separation_certificate as certificate
from Design.SMRF_Layered_Slab_Section import concrete_response
from test_spatial_floor_section import section


def plain(axis=0):
    return SimpleNamespace(regions=[dict(id='slab',s=(-7.,11.),z=(-4.,4.))],h=8.,fc=5.,phi=.6,
        sref=2.,width=18.,j=axis,other=1-axis,longitudinal=[],transverse=[])


class SpectralBoundTests(unittest.TestCase):
    def test_constant_tensor_exact_support_eigen_oracle_and_attaining_stress(self):
        s=plain();rng=np.random.default_rng(158)
        for _ in range(24):
            d=np.r_[rng.normal(size=3),np.zeros(4)]
            w=np.array([[d[0],d[2]/2],[d[2]/2,d[1]]]);e,v=np.linalg.eigh(w)
            stress=-s.fc*(v@np.diag(e<0)@v.T)
            exact=float(np.sum(w*stress))*s.phi*s.h
            for n in (1,4,16):self.assertAlmostEqual(upper(s,d,subdivisions=n),exact,11)

    def test_convex_upper_nested_and_dense_independent_integration(self):
        x,w=leggauss(32);rng=np.random.default_rng(159)
        for axis in (0,1):
            s=plain(axis)
            for _ in range(5):
                d=rng.normal(size=7)/[1,1,1,4,4,4,9]
                bounds=[upper(s,d,subdivisions=n) for n in (1,2,4,8,16)]
                self.assertTrue(all(b<=a+1e-11 for a,b in zip(bounds,bounds[1:])))
                xx,zz=np.meshgrid(2+9*x,4*x,indexing='ij');mat=np.zeros((*xx.shape,2,2))
                mat[...,0,0]=d[0]+d[3]*zz;mat[...,1,1]=d[1]+d[4]*zz
                mat[...,axis,axis]+=d[6]*(xx-2)
                mat[...,0,1]=mat[...,1,0]=(d[2]+d[5]*zz)/2
                f=np.maximum(-np.linalg.eigvalsh(mat),0).sum(axis=-1)*s.fc
                integral=float(np.sum(f*np.outer(w,w)))*36*s.phi/18
                self.assertGreaterEqual(bounds[-1]+1e-9,integral)
                self.assertLess(bounds[-1]-integral,.02*max(1,integral))

    def test_material_principal_premise_matches_implemented_restricted_law(self):
        rng=np.random.default_rng(160)
        for angle in np.linspace(0,2*np.pi,13):
            a=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            principal=rng.uniform(-.0018,.003,size=(17,2));strain=np.einsum('ij,nj,kj->nik',a,principal,a)
            state=concrete_response(strain,5.,model='polak1993_thorenfeldt',peak_strain=.002)
            eig=np.linalg.eigvalsh(state['stress_ksi'])
            self.assertGreaterEqual(float(eig.min()),-5.-1e-12);self.assertLessEqual(float(eig.max()),1e-12)

    def test_actual_compatible_resultants_inside_upper_bound_both_axes(self):
        rng=np.random.default_rng(161)
        for axis in ('x','y'):
            s=section(axis)
            for _ in range(5):
                q=rng.normal(size=7)*[.0001,.0001,.0001,.000003,.000003,.000003,.0000005]
                state=s.state(q,quadrature_order=24);d=rng.normal(size=7)/[1,1,1,26,26,26,48]
                self.assertLessEqual(float(d@state['design_resultants']),upper(s,d)+1e-7)

    def test_coupling_proves_failure_that_independent_component_direction_does_not(self):
        s=section();bounds=s.component_capacity_upper_bounds()
        target=np.array([bounds['positive'][0],bounds['positive'][1],.1,0,0,0,0]);d=[1,1,1,0,0,0,0]
        self.assertGreater(s.support_upper_bound(d),float(np.array(d)@target))
        self.assertLess(upper(s,d),float(np.array(d)@target)-.09)
        result=certificate(s,target,maximum_iterations=50,initial_directions=[d])
        self.assertTrue(result['capacity_exceeded'],result)
        self.assertAlmostEqual(result['proved_design_upper_bound'],upper(s,result['direction'],subdivisions=32),12)
        zero=certificate(s,np.zeros(7),maximum_iterations=5)
        self.assertFalse(zero['capacity_exceeded']);self.assertFalse(zero['adequacy_inferred_when_no_separation'])

    def test_actual_strength_domains_and_external_void_omission(self):
        s=section();a=copy.deepcopy(s);d=[1,1,0,0,0,0,0]
        bar=a.longitudinal[0];bar['strength_credited']=False
        self.assertAlmostEqual(upper(s,d)-upper(a,d),s.phi*bar['area_in2']*bar['fy_ksi']/s.width,12)
        row=a.transverse[0];old_length=row['strength_run_interval_in'][1]-row['strength_run_interval_in'][0]
        before=upper(a,d);row['strength_run_interval_in']=None
        self.assertAlmostEqual(before-upper(a,d),s.phi*row['area_in2']*row['count_per_length_in']*old_length*row['fy_ksi']/s.width,12)
        a.slab_voids=[dict(s=[-6,6],z=[-1,1])]
        self.assertEqual(upper(a,d),upper(a,d,subdivisions=1))

    def test_malformed_inputs_refused(self):
        s=section()
        for d in ([0]*6,[float('nan')]*7):
            with self.assertRaises(ValueError):upper(s,d)
        for n in (0,1.2,True,129):
            with self.assertRaises(ValueError):upper(s,[1]*7,subdivisions=n)
        with self.assertRaises(ValueError):certificate(s,[0]*7,maximum_iterations=0)

if __name__=='__main__':unittest.main()
