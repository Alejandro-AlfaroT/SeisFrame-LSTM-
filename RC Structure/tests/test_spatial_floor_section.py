"""Independent seven-resultant/actual-position integrals; no structural model."""
import math
import unittest
import numpy as np
from numpy.polynomial.legendre import leggauss
from Design.SMRF_Spatial_Floor_Section import SpatialFloorSection,_positive_affine_rectangle
from Design.SMRF_Layered_Slab_Section import LayeredSlabSection
from Design.SMRF_Shared_Floor_Strength import SpatialSharedFloorSection


def inventory():
    bars=[dict(id=f'x/{face}/{i}',s_in=s,z_in=z,area_in2=.31,fy_ksi=60.)
        for face,z in [('top',3.),('bottom',-3.)] for i,s in enumerate([-4,4,12,20,28,36])]
    bars += [dict(id=f'beam/{face}/{i}',s_in=s,z_in=z,area_in2=1.,fy_ksi=60.)
        for face,z in [('top',.9),('bottom',-22.)] for i,s in enumerate([-4,4])]
    trans=[dict(id='y/'+face,physical_bar_ids=['y/'+face+'/0','y/'+face+'/1'],
        accounting_length_in=16,area_in2=.31,z_in=z,run_interval_in=[-8,40],fy_ksi=60.)
        for face,z in [('top',2.25),('bottom',-2.25)]]
    return bars,trans


def section(axis='x',bars=None,trans=None,phi=.65):
    b,t=inventory()
    return SpatialFloorSection(8,5,ec_ksi=4000,axis=axis,slab_interval_in=[-8,40],
        web=dict(id='web',interval_in=[-8,8],z_interval_in=[-26,-4]),reference_s_in=0,
        longitudinal_bars=b if bars is None else bars,transverse_rows=t if trans is None else trans,
        phi_requirements={'N_M':.65,'diaphragm_shear':phi},concrete_peak_strain=.002)


def independent_reference(q):
    # Fixed high-order tensor product and POLAR circle integration are
    # independent of production segmented rectangular/z-cosine quadrature.
    bars,trans=inventory();n=.8+5*6.894757293168/17
    def density(s,z,web=False):
        x=q[0]+q[3]*z+q[6]*s;y=q[1]+q[4]*z;g=q[2]+q[5]*z
        if web:
            r=-x/.002;v=-5*n*r/(n-1+r**n)
            stress=np.stack([v,np.zeros_like(v),np.zeros_like(v)],axis=-1)
        else:
            mid=(x+y)/2;rad=np.hypot((x-y)/2,g/2);lo=mid-rad;hi=mid+rad
            assert np.max(hi)<0
            r1,r2=-lo/.002,-hi/.002
            f1,f2=-5*n*r1/(n-1+r1**n),-5*n*r2/(n-1+r2**n)
            stress=np.stack([f2+(f1-f2)*(hi-x)/(hi-lo),f2+(f1-f2)*(hi-y)/(hi-lo),
                (f1-f2)*(-g/2)/(hi-lo)],axis=-1)
        return np.concatenate([stress,stress*z[...,None],(stress[...,0]*s)[...,None]],axis=-1)
    x,w=leggauss(32);result=np.zeros(7)
    for sa,sb,za,zb,web in [(-8,40,-4,4,False),(-8,8,-26,-4,True)]:
        s,z=np.meshgrid((sa+sb)/2+(sb-sa)/2*x,(za+zb)/2+(zb-za)/2*x)
        result+=np.sum(density(s,z,web)*((sb-sa)*(zb-za)/4*np.outer(w,w))[...,None],axis=(0,1))
    # Polar area dA=r dr dtheta, not production circle-depth integration.
    a,aw=leggauss(48);theta=(a+1)*math.pi;tw=aw*math.pi
    r,rw=leggauss(20)
    for bar in bars:
        radius=math.sqrt(bar['area_in2']/math.pi);rad=(r+1)*radius/2
        rr,tt=np.meshgrid(rad,theta);weights=np.outer(tw,rw*radius/2)*rr
        ss=bar['s_in']+rr*np.cos(tt);zz=bar['z_in']+rr*np.sin(tt)
        result-=np.sum(density(ss,zz,bar['z_in']<-4)*weights[...,None],axis=(0,1))
        strain=q[0]+q[3]*bar['z_in']+q[6]*bar['s_in'];f=bar['area_in2']*np.clip(29000*strain,-60,60)
        result[[0,3,6]]+=f*np.array([1,bar['z_in'],bar['s_in']])
    for row in trans:
        radius=math.sqrt(row['area_in2']/math.pi);z=row['z_in']+radius*np.sin((a+1)*math.pi/2-math.pi/2)
        # Independent theta reduction of transverse circular void, uniform
        # actual integer count/longitudinal accounting length.
        th=a*math.pi/2;z=row['z_in']+radius*np.sin(th)
        zw=aw*math.pi/2*2*radius**2*np.cos(th)**2*(2/16)
        ss,zz=np.meshgrid(16+24*x,z)
        result-=np.sum(density(ss,zz)*np.outer(zw,24*w)[...,None],axis=(0,1))
        f=.31*(2/16)*48*np.clip(29000*(q[1]+q[4]*row['z_in']),-60,60)
        result[[1,4]]+=f*np.array([1,row['z_in']])
    return result/48


class SpatialSectionTests(unittest.TestCase):
    def test_exact_combined_support_bound_and_independent_triangle(self):
        self.assertAlmostEqual(_positive_affine_rectangle(0,2,0,2,-2,1,1),4/3,12)
        s=section();rng=np.random.default_rng(719)
        for _ in range(16):
            q=rng.normal(size=7)*np.array([.0001,.0001,.0001,.000003,.000003,.000003,.0000005])
            actual=np.array(s.state(q)['design_resultants']);d=rng.normal(size=7)/np.array([1,1,1,26,26,26,48])
            self.assertLessEqual(float(d@actual),s.support_upper_bound(d)+1e-7)
        b=s.component_capacity_upper_bounds();target=np.zeros(7)
        target[0]=.99*b['positive'][0];target[3]=-.8*b['negative_magnitude'][3]
        certificate=s.capacity_separation_certificate(target)
        self.assertTrue(certificate['capacity_exceeded'],certificate)
        self.assertGreater(certificate['demand'],s.support_upper_bound(certificate['direction'])+certificate['floating_point_allowance'])
        self.assertFalse(certificate['failure_inferred_from_optimizer'])

    def test_batched_auxiliary_quadrature_and_verified_inverse(self):
        s=section();q=np.array([-.0002,-.00015,.00002,.000001,.000002,.0000003,.0000001])
        batch=s.fixed_quadrature_resultants_batch([q,.5*q,np.zeros(7)],order=24)
        for got,trial in zip(batch,[q,.5*q,np.zeros(7)]):
            np.testing.assert_allclose(got,s.state(trial,quadrature_order=24)['design_resultants'],rtol=2e-6,atol=1e-7)
        demand=s.state(q)['design_resultants']
        result=s.solve(demand,initial_strain=.8*q,max_evaluations=100,use_batched_tangent=True)
        self.assertTrue(result['restricted_model_state_verified'],result)
        self.assertEqual(result['solve_attempts'][0]['jacobian'],'auxiliary_vectorized_fixed_quadrature')

    def test_atomic_seven_adapter_and_proved_bound_classification(self):
        s=section();adapter=SpatialSharedFloorSection(s,state_identity='state1',material_inventory_id='physical01')
        identity=dict(state_identity='state1',combination_id='1.2D+L+Ex',location_id='B7/cut3',recovery_id='coupled01')
        q=np.array([-.0003,-.0002,.0001,.000004,-.000003,.000002,.000002]);target=.65*independent_reference(q)
        cut=dict(identity=identity,source_identities=[dict(identity)],axis='x',cut_width_in=48,
            reference_z_in=0,reference_s_in=0,units='per_unit_width',resultants=target.tolist(),
            accounting=dict(method='disjoint_actual_fields',evidence_id='sevenledger01',
                equilibrium_residual=[0]*7,equilibrium_tolerance=[1e-7]*7,
                weak_first_moment_residual_kip_in=0,weak_first_moment_tolerance_kip_in=1e-7,
                spatial_seven_field_projection_verified=True,core_remote_coverage_verified=True))
        out=adapter.evaluate(cut,initial_strain=q,max_evaluations=10)
        self.assertEqual(out['status'],'pass',out)
        cut['accounting']['spatial_seven_field_projection_verified']=False
        self.assertEqual(adapter.evaluate(cut,initial_strain=q,max_evaluations=10)['status'],'unknown')
        cut['accounting']['spatial_seven_field_projection_verified']=True
        cut['resultants']=[1e6,0,0,0,0,0,0]
        self.assertEqual(adapter.evaluate(cut)['status'],'capacity_exceeded')
        bounds=s.component_capacity_upper_bounds()
        self.assertAlmostEqual(bounds['positive'][0],.65*(12*.31+4)*60/48,12)
        cut['resultants']=[0]*6
        with self.assertRaises(ValueError):adapter.evaluate(cut)

    def test_web_nominal_secant_exact_circle_inertia_and_ownership_ledger(self):
        s=section();beamids=[b['id'] for b in s.longitudinal if b['id'].startswith('beam/')]
        p=np.array([-.0001,0,0])
        noledger=s.web_nominal_secant(p,beam_bar_ids=beamids)
        self.assertEqual(noledger['status'],'unknown')
        self.assertEqual(set(noledger['missing_slab_void_ids']),{'beam/top/0','beam/top/1'})
        out=s.web_nominal_secant(p,beam_bar_ids=beamids,slab_accounting=dict(evidence_id='disjoint01',
            void_subtraction_ids=['beam/top/0','beam/top/1'],steel_owner_ids=['mat/top']))
        n=.8+5*6.894757293168/17;r=.05;sigma=-5*n*r/(n-1+r**n);ecsec=sigma/(-.0001)
        gross=np.array([[16*22,16*22*(-15),0],[16*22*(-15),16*(26**3-4**3)/3,0],
                        [0,0,22*16**3/12]],dtype=float)
        ks=np.zeros((3,3))
        for b in s.longitudinal:
            if not b['id'].startswith('beam/'):continue
            l=np.array([1,b['z_in'],b['s_in']]);ks+=29000*b['area_in2']*np.outer(l,l)
            if b['z_in']<-4:
                removal=b['area_in2']*np.outer(l,l)
                removal[1,1]+=b['area_in2']*b['radius']**2/4
                removal[2,2]+=b['area_in2']*b['radius']**2/4
                gross-=removal
        expected=ecsec*gross+ks
        np.testing.assert_allclose(out['ABD_normal'],expected,atol=1e-5,rtol=1e-9)
        np.testing.assert_allclose(out['nominal_resultants'],expected@p,atol=1e-8)
        self.assertEqual(out['status'],'pass',out)
        self.assertFalse(out['phi_applied']);self.assertFalse(out['slab_concrete_included'])
        bad=s.web_nominal_secant(p,beam_bar_ids=beamids,slab_accounting=dict(evidence_id='bad',
            void_subtraction_ids=['beam/top/0','beam/top/1'],steel_owner_ids=['beam/top/0']))
        self.assertIn('beam_steel_also_credited_to_slab_operator',bad['domain_and_accounting_errors'])

    def test_pure_tension_actual_positions_and_weak_moment_hand(self):
        s=section();q=np.array([.001,.0005,0,0,0,0,.00001]);r=s.state(q)
        bars,trans=inventory();expected=np.zeros(7)
        for b in bars:
            f=b['area_in2']*29000*(.001+.00001*b['s_in'])
            expected[[0,3,6]]+=f*np.array([1,b['z_in'],b['s_in']])/48
        expected[1]=2*.31/8*29000*.0005
        np.testing.assert_allclose(r['nominal_resultants'],expected,atol=1e-10)
        np.testing.assert_allclose(r['concrete_nominal_resultants'],0,atol=1e-10)
        top=[v for v in r['steel'] if v['id'].startswith('beam/top')]
        self.assertNotEqual(top[0]['strain'],top[1]['strain'])
        self.assertTrue(s.verify_quadrature(q)['passed'])

    def test_all_seven_independent_two_dimensional_integrals_and_inverse(self):
        s=section();q=np.array([-.0003,-.0002,.0001,.000004,-.000003,.000002,.000002])
        reference=independent_reference(q)
        self.assertTrue(np.all(abs(reference)>1e-5))
        np.testing.assert_allclose(s.state(q)['nominal_resultants'],reference,atol=2e-7,rtol=1e-7)
        solved=s.solve(.65*reference,initial_strain=q*.92,max_evaluations=40)
        self.assertEqual(solved['status'],'pass',solved)
        np.testing.assert_allclose(solved['generalized_strain'],q,atol=2e-9)
        self.assertLess(max(abs(v) for v in solved['equilibrium_residual']),1e-6)

    def test_six_resultant_limit_and_xy_exchange(self):
        bars,trans=inventory();rows=[]
        for b in bars:
            rows.append(dict(id=b['id'],axis='x',z_in=b['z_in'],area_in2_per_in=b['area_in2']/48,
                bar_area_in2=b['area_in2'],fy_ksi=60,concrete_owner='web' if b['z_in']<-4 else 'slab'))
        for row in trans:
            rows.append(dict(id=row['id'],axis='y',z_in=row['z_in'],area_in2_per_in=.31/8,
                bar_area_in2=.31,fy_ksi=60))
        six=LayeredSlabSection(8,5,rows,cut_width_in=48,drop_webs=[dict(id='web',axis='x',width_in=16,z_bottom_in=-26,z_top_in=-4)],
            concrete_peak_strain=.002,compression_strain_limit=.002,concrete_model='polak1993_thorenfeldt')
        q=np.array([-.0002,.0003,.0002,.000003,.000002,-.000002,0.])
        a=section().state(q);old=six.state(q[:6])
        np.testing.assert_allclose(a['nominal_resultants'][:6],old['nominal_resultants'],atol=2e-7,rtol=1e-6)
        idx=[1,0,2,4,3,5,6];b=section(axis='y').state(q[idx])
        np.testing.assert_allclose(np.array(a['nominal_resultants'])[idx],b['nominal_resultants'],atol=1e-8)

    def test_bar_crossing_slab_web_interface_has_one_finite_volume(self):
        # Uniform compression gives a source-independent exact area ledger.
        bars=[dict(id='crossing',s_in=0,z_in=-4,area_in2=1,fy_ksi=60)]
        s=section(bars=bars,trans=[]);q=[-.0002,0,0,0,0,0,0]
        n=.8+5*6.894757293168/17;r=.1;stress=-5*n*r/(n-1+r**n)
        expected=((48*8+16*22-1)*stress+1*29000*(-.0002))/48
        state=s.state(q)
        self.assertAlmostEqual(state['nominal_resultants'][0],expected,7)
        self.assertEqual(len(state['steel']),1)

    def test_actual_geometry_and_domain_failures_are_not_silenced(self):
        bars,trans=inventory();bars[1]=dict(bars[0],id='overlap')
        with self.assertRaisesRegex(ValueError,'overlap'):section(bars=bars)
        bars,trans=inventory();bars[0]['s_in']=-8
        with self.assertRaisesRegex(ValueError,'outside'):section(bars=bars)
        failed=section().state([-.0003,0,0,0,0,0,-.001])
        self.assertFalse(failed['material_domain_valid'])
        self.assertIn('concrete_compression_exceeds_declared_prepeak_limit',failed['material_domain_errors'])
        with self.assertRaises(ValueError):section().solve([0]*6)
        self.assertEqual(section(phi=.6).phi,.6)


if __name__=='__main__':unittest.main()
