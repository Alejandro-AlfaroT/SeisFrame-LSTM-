"""Independent hand force integrals and invariance checks; no OpenSees."""
import math
import unittest

import numpy as np
from scipy.integrate import quad

from Design.SMRF_Layered_Slab_Section import (
    LayeredSlabSection, common_field_ownership, compose_strain_fields,
    concrete_response, strain_tensors,
    design_concrete_peak_strain,
)


def inventory():
    return [dict(id=f'{axis}{face}',axis=axis,z_in=sign*z,
                 area_in2_per_in=.31/8,bar_area_in2=.31,fy_ksi=60.)
            for axis,z in [('x',3.),('y',2.25)]
            for face,sign in [('top',1),('bottom',-1)]]


class LayeredSlabTests(unittest.TestCase):
    def setUp(self):
        self.s=LayeredSlabSection(8.,5.,inventory())

    def test_mcft_hand_softening_and_prepeak(self):
        r=concrete_response(np.array([[-.001,0.],[0.,.004]]),5.)
        expected=-5/(.8+170*.004)*.75
        self.assertAlmostEqual(r['stress_ksi'][0,0],expected,12)
        self.assertEqual(r['stress_ksi'][1,1],0.)
        self.assertTrue(r['prepeak_valid'])
        r2=concrete_response(np.array([[-.0021,0.],[0.,.004]]),5.)
        self.assertFalse(r2['prepeak_valid'])
        # Constant .55fc would overstate capacity once tensile strain is large.
        weak=concrete_response(np.diag([-.002,.01]),5.)
        self.assertLess(abs(weak['stress_ksi'][0,0]),.55*5)
        # A .002 peak does not pointwise bound a .0026 backbone. Preserve
        # this known counterexample to automatic 'prepeak is conservative'.
        slower=concrete_response(np.diag([-.001,0]),5.,peak_strain=.0026)
        self.assertAlmostEqual(abs(slower['stress_ksi'][0,0])/5,2/2.6-1/2.6**2)
        self.assertLess(abs(slower['stress_ksi'][0,0]),.75*5)
        # Original Eq.18b uses the measured cylinder peak, including its
        # effect on transverse softening; 170 is not universal.
        measured=concrete_response(np.diag([-.001,.004]),5.,peak_strain=.0026)
        expected=-5/(.8+.34*.004/.0026)*(2/2.6-1/2.6**2)
        self.assertAlmostEqual(measured['stress_ksi'][0,0],expected,12)

    def test_published_thorenfeldt_biaxial_form_and_initial_slope(self):
        # Polak/Vecchio1993 p3446 explicitly evaluates the second compressed
        # principal direction with Eq17-22 too. There is no confinement bonus.
        n=.8+5*6.894757293168/17
        r=concrete_response(np.diag([-.001,-.0005]),5,
                            model='polak1993_thorenfeldt')
        expected=[-5*n*v/(n-1+v**n) for v in [.5,.25]]
        np.testing.assert_allclose(np.diag(r['stress_ksi']),expected,atol=1e-12)
        self.assertTrue(r['biaxial_compression_formulation_supported'])
        self.assertTrue(r['backbone_valid'])
        atpeak=concrete_response(np.diag([-.002,-.002]),5,model='polak1993_thorenfeldt')
        np.testing.assert_allclose(np.diag(atpeak['stress_ksi']),[-5,-5],atol=1e-12)
        ec=57*math.sqrt(5000)
        design=design_concrete_peak_strain(5,ec)
        small=concrete_response(np.diag([-1e-9,0]),5,peak_strain=design['peak_strain'],
                               model='polak1993_thorenfeldt')
        self.assertAlmostEqual(small['stress_ksi'][0,0]/(-1e-9),ec,5)
        self.assertFalse(design['measured'])
        supported=LayeredSlabSection(8,5,inventory(),concrete_model='polak1993_thorenfeldt')
        self.assertTrue(supported.state([-.0002,-.0001,0,0,0,0])['material_domain_valid'])
        self.assertFalse(supported.state([-.0002,-.0001,0,0,0,0])['code_qualified'])
        weak=concrete_response(np.diag([-.001,.02]),2.5,model='polak1993_thorenfeldt')
        self.assertTrue(weak['backbone_valid'])
        invalid=concrete_response(np.diag([-.001,.04]),2.5,model='polak1993_thorenfeldt')
        self.assertFalse(invalid['backbone_valid'])

    def test_pure_tension_exact_inventory_and_finite_void(self):
        r=self.s.state([.001,.0005,0,0,0,0])
        expected=[2*.31/8*29.,2*.31/8*14.5,0,0,0,0]
        np.testing.assert_allclose(r['nominal_resultants'],expected,atol=1e-12)
        np.testing.assert_allclose(r['concrete_nominal_resultants'],0.,atol=1e-12)
        self.assertAlmostEqual(r['concrete_net_volume_per_unit_area_in'],8-4*.31/8,10)
        self.assertTrue(r['material_domain_valid'])

    def test_proportional_path_hand_tension_and_failure_audit(self):
        # Both axes in tension: actual bar areas and moduli give exact linear
        # path; no concrete contribution or manufactured section-state target.
        target=.65*np.array([2*.31/8*29,2*.31/8*14.5,0,0,0,0])
        result=self.s.solve_load_path(target,required_shear_phi=.65,steps=4)
        self.assertTrue(result['proportional_force_path_verified'],result)
        for record in result['path']:
            f=record['load_fraction']
            np.testing.assert_allclose(record['generalized_strain'],[f*.001,f*.0005,0,0,0,0],atol=2e-9)
            self.assertGreater(record['incremental_work_slope'],0)
        self.assertFalse(result['continuous_stability_proven'])
        failure=self.s.solve_load_path(target*1e4,required_shear_phi=.65,steps=4,
                                       max_evaluations_per_step=20)
        self.assertFalse(failure['proportional_force_path_verified'])
        self.assertGreater(failure['steps_completed'],0)
        with self.assertRaises(ValueError):self.s.solve_load_path(target,required_shear_phi=.65,steps=0)

    def test_finite_compression_zone_independent_integrals(self):
        # epsilon_x=k*z: lower half compresses, upper half has zero concrete
        # stress. Integrate the polynomial analytically over gross concrete.
        k=.0004
        c=4.
        a=k/.002
        nc=-5*(a*c*c-a*a*c**3/3)
        mc=5*(2*a*c**3/3-a*a*c**4/4)
        # Circular void integrals use a distinct theta parameter and adaptive
        # scalar quadrature, independent of the production segmented rule.
        for row in inventory():
            radius=math.sqrt(.31/math.pi)
            def integrand(theta,moment):
                z=row['z_in']+radius*math.sin(theta)
                r=max(0.,-k*z/.002)
                stress=-5*(2*r-r*r) if z<0 else 0.
                return stress*(z if moment else 1.)*2*radius**2*math.cos(theta)**2/8
            nc-=quad(integrand,-math.pi/2,math.pi/2,args=(False,),epsabs=1e-12)[0]
            mc-=quad(integrand,-math.pi/2,math.pi/2,args=(True,),epsabs=1e-12)[0]
        state=self.s.state([0,0,0,k,0,0])
        self.assertAlmostEqual(state['concrete_nominal_resultants'][0],nc,9)
        self.assertAlmostEqual(state['concrete_nominal_resultants'][3],mc,9)
        self.assertTrue(self.s.verify_quadrature([0,0,0,k,0,0])['passed'])

    def test_uniform_shear_closed_form_and_inverse(self):
        # Principal strains +.0012,-.0008, axes +/-45 degrees. Compression
        # diagonal carries equal normal compression and positive shear.
        # Uniform density means exact net concrete volume handles every void.
        sigma=-5/(.8+170*.0012)*(2*.4-.4**2)
        net=8-4*.31/8
        normal=sigma/2*net+2*.31/8*29000*.0002
        target=.65*np.array([normal,normal,-sigma/2*net,0,0,0])
        state=self.s.solve(target,required_shear_phi=.65,
                           initial_strain=[.00015,.00015,.0018,0,0,0])
        self.assertTrue(state['equilibrium_verified'],state)
        self.assertTrue(state['restricted_model_state_verified'],state)
        self.assertFalse(state['code_qualified'])
        self.assertFalse(state['transverse_shear_verified'])

    def test_axis_exchange_and_shear_reversal(self):
        q=np.array([-.0004,.0011,.0012,.00003,-.00002,.00001])
        a=self.s.state(q)
        swapped=inventory()
        for row in swapped:row['axis']='y' if row['axis']=='x' else 'x'
        b=LayeredSlabSection(8,5,swapped).state(q[[1,0,2,4,3,5]])
        np.testing.assert_allclose(np.array(a['nominal_resultants'])[[1,0,2,4,3,5]],
                                   b['nominal_resultants'],rtol=1e-11,atol=1e-11)
        q[[2,5]]*=-1
        c=self.s.state(q)
        expected=np.array(a['nominal_resultants']);expected[[2,5]]*=-1
        np.testing.assert_allclose(c['nominal_resultants'],expected,atol=1e-10)

    def test_physical_common_strain_not_separate_strength_addition(self):
        global_beam=[.0001,0,0,-.0001,0,0]
        local=[0,.0003,0,.00002,0,0]
        total=compose_strain_fields(global_beam,local)
        r=common_field_ownership(self.s,total,plane_reference_id='z=slab-mid',
                                 action_projection_id='test-projection',material_owner_id='physical-slab-1')
        for row in r['steel']:
            axis=0 if row['axis']=='x' else 1
            self.assertAlmostEqual(row['strain'],total[axis]+row['z_in']*total[axis+3])
        separate=np.array(self.s.state(global_beam)['nominal_resultants'])+self.s.state(local)['nominal_resultants']
        self.assertGreater(np.linalg.norm(separate-r['nominal_resultants']),.01)
        self.assertFalse(r['ownership']['action_projection_verified'])

    def test_domain_failures_not_silent_capacity(self):
        self.assertFalse(self.s.state([-.0001,-.0001,0,0,0,0])['material_domain_valid'])
        self.assertFalse(self.s.state([0,0,0,.00051,0,0])['material_domain_valid'])
        self.assertFalse(self.s.state([.011,.001,0,0,0,0])['material_domain_valid'])
        q=[.0002,.0002,.002,0,0,0]
        target=self.s.state(q)['design_resultants']
        r=self.s.solve(target,required_shear_phi=.60,initial_strain=q)
        self.assertTrue(r['equilibrium_verified'])
        self.assertFalse(r['shear_phi_valid'])
        self.assertFalse(r['restricted_model_state_verified'])

    def test_inventory_input_validation(self):
        bad=inventory();bad[1]['id']=bad[0]['id']
        with self.assertRaises(ValueError):LayeredSlabSection(8,5,bad)
        bad=inventory();bad[2]['z_in']=bad[0]['z_in']
        with self.assertRaises(ValueError):LayeredSlabSection(8,5,bad)
        with self.assertRaises(ValueError):self.s.state([0]*5)

    def test_actual_catalog_diameter_controls_touching_mat_geometry(self):
        # ACI nominal#5 area.31 and physicaldb.625 are independently rounded.
        # Equivalent-area diameter.6283 must not fabricate a physical clash.
        rows=inventory()
        for row in rows:
            row['db_in']=.625
            if row['axis']=='y':row['z_in']=math.copysign(2.375,row['z_in'])
        s=LayeredSlabSection(8,5,rows)
        self.assertGreater(s.layers[0]['radius'],s.layers[0]['physical_radius'])
        self.assertTrue(s.state([.0001,.0001,0,0,0,0])['material_domain_valid'])
        rows[2]['z_in']=2.376
        with self.assertRaisesRegex(ValueError,'intersect physically'):LayeredSlabSection(8,5,rows)
        with self.assertRaises(ValueError):self.s.state([0,0,float('nan'),0,0,0])
        with self.assertRaises(ValueError):concrete_response([[0,1],[0,0]],5)
        with self.assertRaises(ValueError):LayeredSlabSection(8,5,inventory(),phi=.75)

    def test_composite_drop_web_independent_full_section_cut(self):
        # Physical 48-in strip: 8-in slab over a 12x20-in drop web. Beam top
        # steel displaces slab concrete, bottom steel displaces only web.
        rows=inventory()+[
            dict(id='beam-top',axis='x',z_in=.75,area_in2_per_in=4/48,bar_area_in2=1,fy_ksi=60),
            dict(id='beam-bottom',axis='x',z_in=-22,area_in2_per_in=4/48,bar_area_in2=1,fy_ksi=60,concrete_owner='web')]
        web=dict(id='web',axis='x',z_bottom_in=-24,z_top_in=-4,width_in=12)
        s=LayeredSlabSection(8,5,rows,cut_width_in=48,drop_webs=[web])
        q=np.array([-.0004,.0005,0,-.00006,0,0])
        out=s.state(q)
        # Direct full-width geometric integral, including all actual round
        # steel voids and each bar's centroid force. No production recovery or
        # constitutive function is used by this reference.
        def stress(z):
            ratio=max(0.,-(q[0]+z*q[3])/.002)
            return -5*(2*ratio-ratio**2)
        n=m=0.
        for lo,hi,width in [(-24,-4,12),(-4,4,48)]:
            n+=quad(lambda z:width*stress(z),lo,hi,epsabs=1e-10,points=[-q[0]/q[3]] if lo<-q[0]/q[3]<hi else None)[0]
            m+=quad(lambda z:width*z*stress(z),lo,hi,epsabs=1e-10,points=[-q[0]/q[3]] if lo<-q[0]/q[3]<hi else None)[0]
        for row in rows:
            radius=math.sqrt(row['bar_area_in2']/math.pi)
            count=row['area_in2_per_in']*48/row['bar_area_in2']
            def removed(theta,moment):
                z=row['z_in']+radius*math.sin(theta)
                return stress(z)*2*radius**2*math.cos(theta)**2*count*(z if moment else 1.)
            n-=quad(removed,-math.pi/2,math.pi/2,args=(False,),epsabs=1e-11)[0]
            m-=quad(removed,-math.pi/2,math.pi/2,args=(True,),epsabs=1e-11)[0]
            if row['axis']=='x':
                force=row['area_in2_per_in']*48*np.clip(29000*(q[0]+row['z_in']*q[3]),-60,60)
                n+=force;m+=row['z_in']*force
        self.assertAlmostEqual(out['full_strip_nominal_resultants'][0],n,7)
        self.assertAlmostEqual(out['full_strip_nominal_resultants'][3],m,7)
        # Recover the web's centroid moment and hidden axial couple, retaining
        # signed quantities. Their recombination equals its common-plane M.
        rc=np.array(out['drop_webs'][0]['concrete_nominal_resultants'])*48
        bottom=next(v for v in out['steel'] if v['id']=='beam-bottom')
        nw=rc[0]+bottom['force_kip_per_in']*48
        mw0=rc[3]+bottom['force_kip_per_in']*48*bottom['z_in']
        zw=-14.
        mw_centroid=mw0-zw*nw
        self.assertAlmostEqual(mw_centroid+zw*nw,mw0,10)
        self.assertGreater(abs(zw*nw),1.)
        self.assertTrue(s.verify_quadrature(q)['passed'])

        # Rotate the physical strip/web and all bars together; scalar cut
        # geometry and force accounting must be identical in the y frame.
        rotated=[dict(r,axis='y' if r['axis']=='x' else 'x') for r in rows]
        sy=LayeredSlabSection(8,5,rotated,cut_width_in=48,drop_webs=[dict(web,axis='y')])
        ry=sy.state(q[[1,0,2,4,3,5]])
        np.testing.assert_allclose(np.array(out['nominal_resultants'])[[1,0,2,4,3,5]],ry['nominal_resultants'],atol=1e-9)

    def test_composite_concurrent_membrane_and_twist_inverse(self):
        rows=inventory()+[dict(id='beam-bottom',axis='x',z_in=-22,
                              area_in2_per_in=4/48,bar_area_in2=1,fy_ksi=60,concrete_owner='web')]
        web=dict(id='web',axis='x',z_bottom_in=-24,z_top_in=-4,width_in=12)
        s=LayeredSlabSection(8,5,rows,cut_width_in=48,drop_webs=[web])
        # Concurrent nonzero Nxx,Nyy,Nxy,Mxx,Myy,Mxy. This inverse check tests
        # nonlinear coupling and ownership; it is not experimental validation.
        q=np.array([-.0003,.0005,.0006,-.00005,.00001,.00002])
        # Independent explicit 2x2 spectral projector and adaptive scalar
        # quadrature: this target does not call the production constitutive
        # law, eigenvalue routine, state recovery, or segmented integration.
        def tensor_stress(z):
            x,y,g=q[:3]+z*q[3:]
            mid=(x+y)/2
            radius=math.hypot((x-y)/2,g/2)
            lo,hi=mid-radius,mid+radius
            if lo>=0:return np.zeros(3)
            self.assertGreater(hi,0)
            r=-lo/.002
            stress=-5/max(1,.8+.34*hi/.002)*(2*r-r*r)
            # Projector onto the compressive principal direction.
            return stress*np.array([(hi-x)/(hi-lo),(hi-y)/(hi-lo),-g/2/(hi-lo)])
        reference=np.zeros(6)
        for i in range(6):
            reference[i]=quad(lambda z:tensor_stress(z)[i%3]*(z if i>=3 else 1),-4,4,epsabs=1e-11)[0]
        for row in rows:
            radius=math.sqrt(row['bar_area_in2']/math.pi)
            count=row['area_in2_per_in']/row['bar_area_in2']
            for i in range(6):
                if row.get('concrete_owner','slab')!='slab':
                    continue
                def void(theta):
                    z=row['z_in']+radius*math.sin(theta)
                    return tensor_stress(z)[i%3]*(z if i>=3 else 1)*2*radius**2*math.cos(theta)**2*count
                reference[i]-=quad(void,-math.pi/2,math.pi/2,epsabs=1e-11)[0]
            axis=0 if row['axis']=='x' else 1
            force=row['area_in2_per_in']*np.clip(29000*(q[axis]+row['z_in']*q[axis+3]),-60,60)
            reference[axis]+=force;reference[axis+3]+=row['z_in']*force
        def webstress(z):
            r=max(0,-(q[0]+z*q[3])/.002)
            return -5*(2*r-r*r)
        for moment in (False,True):
            result=quad(lambda z:webstress(z)*(z if moment else 1)*12/48,-24,-4,points=[-q[0]/q[3]],epsabs=1e-11)[0]
            row=rows[-1];radius=math.sqrt(row['bar_area_in2']/math.pi)
            def void(theta):
                z=row['z_in']+radius*math.sin(theta)
                return webstress(z)*(z if moment else 1)*2*radius**2*math.cos(theta)**2*row['area_in2_per_in']/row['bar_area_in2']
            result-=quad(void,-math.pi/2,math.pi/2,epsabs=1e-11)[0]
            reference[3 if moment else 0]+=result
        np.testing.assert_allclose(s.state(q)['nominal_resultants'],reference,atol=1e-9,rtol=1e-9)
        target=.65*reference
        self.assertTrue(np.all(np.abs(target)>1e-6))
        solved=s.solve(target,required_shear_phi=.65,initial_strain=q*.9)
        self.assertTrue(solved['restricted_model_state_verified'],solved)
        np.testing.assert_allclose(solved['generalized_strain'],q,atol=1e-9)
        with self.assertRaises(ValueError):
            LayeredSlabSection(8,5,rows,cut_width_in=48,
                               drop_webs=[web,dict(web,id='cross',axis='y')])


if __name__=='__main__':unittest.main()
