"""Pure section algebra, physical partitions and cut signs; no OpenSees."""
import unittest
import numpy as np
from scipy.integrate import dblquad
from Design.SMRF_Full_T_Projection import (
    FullTIncrementProjector, released_increment, normal_cut_from_global_wrench,
)


def projector(axis='x',flange=(-24,24),web=(-8,8),ef=4000.,ew=4000.):
    return FullTIncrementProjector(slab_thickness_in=8,beam_depth_in=30,
        flange_interval_in=flange,web_interval_in=web,slab_E_ksi=ef,web_E_ksi=ew,axis=axis)


class FullTProjectionTests(unittest.TestCase):
    def test_full_T_own_flange_inertia_and_scalar_hand(self):
        p=projector();r=p.project(n_tension_kip=0,m_z_kip_in=10000,increment_id='Ex')
        area=48*8+16*22;zc=16*22*(-15)/area
        ig=48*8**3/12+48*8*zc**2+16*22**3/12+16*22*(-15-zc)**2
        self.assertAlmostEqual(p.properties()['zero_N_EI_kip_in2'],4000*ig,5)
        self.assertAlmostEqual(r['generalized_normal_strain'][1],10000/(4000*ig),14)
        self.assertAlmostEqual(r['generalized_normal_strain'][0],-zc*r['generalized_normal_strain'][1],14)
        flange=r['regions'][0];web=r['regions'][1]
        self.assertGreater(abs(flange['m_about_own_z_centroid_kip_in']),1.)
        self.assertAlmostEqual(flange['m_z_kip_in']+web['m_about_own_z_centroid_kip_in']+web['m_z_axial_couple_kip_in'],10000,9)
        self.assertAlmostEqual(flange['n_tension_kip']+web['n_tension_kip'],0,10)
        self.assertFalse(r['nominal_probable_capacity_inventory_modified'])

    def test_asymmetric_requires_actual_weak_moment_and_closes_three_axes(self):
        p=projector(flange=(-8,40),ef=4200,ew=3800)
        with self.assertRaisesRegex(ValueError,'actual weak-axis'):
            p.project(n_tension_kip=23,m_z_kip_in=-9000,increment_id='Ex')
        diagnostic=p.project(n_tension_kip=23,m_z_kip_in=-9000,increment_id='Ex',allow_unclosed_transverse_diagnostic=True)
        self.assertTrue(diagnostic['requires_transverse_action_accounting'])
        self.assertGreater(abs(diagnostic['incidental_transverse_first_moment_kip_in']),100.)
        target=[23,-9000,425.]
        r=p.project(n_tension_kip=target[0],m_z_kip_in=target[1],m_s_kip_in=target[2],increment_id='Ex')
        q=r['generalized_normal_strain'];actual=[]
        # Independent direct2D integration of stress and its two first moments.
        for j in range(3):
            val=0.
            for lo,hi,zlo,zhi,e in [(-8,40,-4,4,4200),(-8,8,-26,-4,3800)]:
                val+=dblquad(lambda z,s:e*(q[0]+q[1]*z+q[2]*s)*[1,z,s][j],lo,hi,
                             lambda _:zlo,lambda _:zhi,epsabs=1e-8)[0]
            actual.append(val)
        np.testing.assert_allclose(actual,target,atol=1e-8)
        self.assertFalse(r['requires_transverse_action_accounting'])

    def test_physical_partition_and_linear_slab_field(self):
        p=projector(flange=(-8,40));r=p.project(n_tension_kip=-30,m_z_kip_in=8000,m_s_kip_in=0,increment_id='Ex')
        parts=p.partition(r,[dict(id='core',interval_in=[-8,12]),dict(id='remote',interval_in=[12,40])])
        np.testing.assert_allclose(parts['sum_n_mz_ms'],[-30,8000,0],atol=1e-8)
        self.assertTrue(parts['whole_web_owner_verified'])
        core=parts['strips'][0]
        web=r['regions'][1]
        np.testing.assert_allclose(core['region_n_mz_ms']['web'],[web['n_tension_kip'],web['m_z_kip_in'],web['m_s_kip_in']],atol=1e-9)
        self.assertEqual(parts['strips'][1]['region_n_mz_ms']['web'],[0,0,0])
        f0=np.array(p.slab_field(r,12));f1=np.array(p.slab_field(r,40));fm=np.array(p.slab_field(r,26))
        np.testing.assert_allclose(fm,(f0+f1)/2,atol=1e-11)
        self.assertNotEqual(f0[0],f1[0]);self.assertEqual(f0[3],f1[3])
        self.assertEqual(p.slab_field(r,41),[0]*6)
        with self.assertRaisesRegex(ValueError,'gap or overlap'):
            p.partition(r,[dict(id='a',interval_in=[-8,12]),dict(id='b',interval_in=[13,40])])
        with self.assertRaisesRegex(ValueError,'split'):
            p.partition(r,[dict(id='a',interval_in=[-8,0]),dict(id='b',interval_in=[0,40])])

    def test_global_coordinate_translation_and_xy_rotation(self):
        p=projector(flange=(-8,40));q=projector(axis='y',flange=(192,240),web=(192,208))
        args=dict(n_tension_kip=31,m_z_kip_in=-3000,m_s_kip_in=270,increment_id='E')
        a,b=p.project(**args),q.project(**args)
        np.testing.assert_allclose(a['generalized_normal_strain'],b['generalized_normal_strain'],atol=1e-12)
        np.testing.assert_allclose(np.array(p.slab_field(a,10))[[1,0,2,4,3,5]],q.slab_field(b,210),atol=1e-10)
        self.assertEqual(q.properties()['reference_s_in'],200)

    def test_cut_wrench_signs_independent_cross_product(self):
        force=13.;z=-9.;s=7.
        for axis in ('x','y'):
            f=np.array([force,0,0] if axis=='x' else [0,force,0])
            point=np.array([0,s,z] if axis=='x' else [s,0,z])
            m=np.cross(point,f)
            out=normal_cut_from_global_wrench(np.r_[f,m],axis=axis)
            self.assertEqual(out['n_tension_kip'],force)
            self.assertEqual(out['m_z_kip_in'],z*force)
            self.assertEqual(out['m_s_kip_in'],s*force)

    def test_only_identical_released_gravity_subtracted_and_reference_transport(self):
        g=dict(model_fingerprint='released001',state_identity='state1',cut_id='B7/2',gravity_load_id='1.2D+L',
               analysis_id='gravity1',reference_z_in=-15,n_tension_kip=-100,m_z_kip_in=400,m_s_kip_in=40,reference_s_in=200)
        e=dict(g,analysis_id='gravity+Ex',n_tension_kip=-75,m_z_kip_in=1300,m_s_kip_in=90)
        out=released_increment(e,g,increment_id='Ex1')
        self.assertEqual(out['n_tension_kip'],25)
        self.assertEqual(out['m_z_kip_in'],900-15*25)
        self.assertEqual(out['m_s_kip_in'],50);self.assertEqual(out['reference_s_in'],200)
        with self.assertRaisesRegex(ValueError,'identical model_fingerprint'):
            released_increment(e,dict(g,model_fingerprint='continuousgravity'),increment_id='wrong')


if __name__=='__main__':unittest.main()
