"""Independent axial integrals of the signed primary target, both floor axes."""
import copy
import unittest
import numpy as np
from Design.SMRF_Axial_Action_Accounting import (make_window_evidence,
    primary_moment_window_evidence,verify_point_accounting,linear_combination_window)
from Design.SMRF_Full_T_Projection import FullTIncrementProjector
from Design.SMRF_Physical_Floor_Cuts import identity


def action(axis,coeff,case,recovery,*,reference=1.):
    # Physical coordinate x is represented on [2,6], not a unit window.
    coeff=np.asarray(coeff,float);at=np.array([1.,.5,.25])@coeff
    row=dict(identity=dict(state_identity='actual',combination_id=case,recovery_id=recovery),
        material_inventory_id='cage',analysis_input_identity='same-input',axis=axis,
        cut_width_in=6.,reference_s_in=reference,reference_z_in=0.,units='integrated_strip',
        resultants=at.tolist(),source_response_sha256=recovery+'-source')
    p=dict(interval_in=[2.,6.],coefficients=coeff.tolist(),
        sample_values=(np.array([[1.,0.,0.],[1.,.5,.25],[1.,1.,1.]])@coeff).tolist(),
        quarter_witness=(np.array([1.,.25,.0625])@coeff).tolist())
    row['axial_action_window']=make_window_evidence(row,[2.,6.],[p],source_response_sha256=row['source_response_sha256'])
    return row


class PrimaryAxialWindowTests(unittest.TestCase):
    def fixture(self,axis,component,increment_only=False):
        j=0 if axis=='x' else 1
        p=FullTIncrementProjector(slab_thickness_in=1.,beam_depth_in=5.,flange_interval_in=[-1.,5.],
            web_interval_in=[-1.,1.],slab_E_ksi=200.,web_E_ksi=200.,axis=axis,transverse_reference_in=0.)
        c=np.array([[2.,3.,4.,5.,6.,7.,8.],[.7,-.3,.6,.2,-.5,.1,.9],[.1,.2,-.3,.4,.6,-.2,.5]])
        g=np.array([[.4,.5,.6,.7,.8,.9,1.],[.1,.2,.3,.4,.5,.6,.7],[.7,.6,.5,.4,.3,.2,.1]])
        physical=action(axis,c,'E','physical');gravity=action(axis,g,'G','gravity')
        affine=dict(axis=axis,station_in=4.,physical_member_interval_in=[0.,10.],state_identity='actual',
            physical_gravity_case_id='G',normal_derivative_per_in=[0.,-.2,.3],
            Q_vertical_kip=.2,V_inplane_kip=.3,M_centroid_kipin=-3.,Ms_transverse_first_moment_kipin=2.)
        affine['identity_sha256']=identity(affine)
        inc=dict(actual_primary_cut=affine,station_in=4.,m_z_kip_in=-3.,m_s_kip_in=2.)
        def exact(t):
            # Whole flange+web: no projected steel/material reassignment.
            v=np.array([1.,t,t*t])@c;gv=np.array([1.,t,t*t])@g
            if component in ('strong','both'):v[j+3]=gv[j+3]-3.-.2*(4*t-2.)
            if component in ('weak','both'):
                # References differ from the projector. External N is kept,
                # so converting target Ms back needs the physical N, not G's.
                v[6]=(v[6]+2.+.3*(4*t-2.) if increment_only else
                       gv[6]+gv[j]+2.+.3*(4*t-2.)-v[j])
            return v
        guard=copy.deepcopy(physical);guard.pop('axial_action_window')
        guard['identity']['recovery_id']+='|'+component;guard['resultants']=exact(.5).tolist()
        kw=dict(projector=p,primary_increment=inc,component=component,strip_interval=[-1.,5.],owns_web=True)
        if increment_only:kw['primary_weak_increment_only']=True
        return guard,physical,gravity,physical,kw,exact

    def test_nonuniform_target_all_seven_means_and_first_moments_both_axes(self):
        for axis in ('x','y'):
            for component in ('strong','weak','both'):
                guard,physical,g,pe,kw,exact=self.fixture(axis,component)
                result=primary_moment_window_evidence(guard,physical,g,pe,**kw)
                self.assertEqual(result['status'],'computed')
                # Independent five-point integration of explicit target
                # equations; no window coefficients or projection calls.
                nodes,weights=np.polynomial.legendre.leggauss(5);t=(nodes+1)/2
                vals=np.array([exact(x) for x in t])
                np.testing.assert_allclose(result['average_resultants'],weights@vals/2,rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(result['first_axial_moment_about_window_start'],8*(weights*t)@vals,rtol=1e-12,atol=1e-12)
                self.assertFalse(result['point_demand_replaced']);self.assertTrue(result['derivation']['original_point_reproduced'])
                guard['axial_action_window']=result
                # Nonuniform quadratic action remains unresolved by the
                # separate material averaging prerequisite.
                self.assertEqual(verify_point_accounting(guard,dict(axial_accounting_interval_in=[2.,6.]))['status'],'unknown')

    def test_incomplete_domain_and_resealed_affine_law_do_not_gain_credit(self):
        guard,p,g,pe,kw,_=self.fixture('x','both')
        bad=copy.deepcopy(p);bad.pop('axial_action_window')
        self.assertEqual(primary_moment_window_evidence(guard,bad,g,pe,**kw)['status'],'not_evaluated')
        inc=kw['primary_increment'];cut=inc['actual_primary_cut'];cut['physical_member_interval_in']=[3.,6.]
        cut['identity_sha256']=identity({k:v for k,v in cut.items() if k!='identity_sha256'})
        self.assertEqual(primary_moment_window_evidence(guard,p,g,pe,**kw)['status'],'not_evaluated')
        cut['physical_member_interval_in']=[0.,10.];cut['normal_derivative_per_in'][1]=.8
        cut['identity_sha256']=identity({k:v for k,v in cut.items() if k!='identity_sha256'})
        with self.assertRaisesRegex(ValueError,'signed guard target'):primary_moment_window_evidence(guard,p,g,pe,**kw)

    def test_changed_point_or_window_payload_is_refused(self):
        guard,p,g,pe,kw,_=self.fixture('y','strong')
        guard['resultants'][4]+=.01
        with self.assertRaisesRegex(ValueError,'original point'):primary_moment_window_evidence(guard,p,g,pe,**kw)
        guard,p,g,pe,kw,_=self.fixture('y','strong')
        p['axial_action_window']['average_resultants'][0]+=1.
        with self.assertRaisesRegex(ValueError,'payload differs'):primary_moment_window_evidence(guard,p,g,pe,**kw)

    def test_different_source_partitions_retain_exact_integrals(self):
        guard,p,g,pe,kw,exact=self.fixture('x','both')
        for source,breaks in ((p,[2.,3.1,6.]),(g,[2.,2.7,5.3,6.])):
            c=np.asarray(source['axial_action_window']['pieces'][0]['coefficients']);pieces=[]
            for lo,hi in zip(breaks[:-1],breaks[1:]):
                t=(lo-2)/4;dt=(hi-lo)/4
                local=np.array([c[0]+t*c[1]+t*t*c[2],dt*c[1]+2*t*dt*c[2],dt*dt*c[2]])
                pieces.append(dict(interval_in=[lo,hi],coefficients=local.tolist(),
                    sample_values=(np.array([[1,0,0],[1,.5,.25],[1,1,1]])@local).tolist(),
                    quarter_witness=(np.array([1,.25,.0625])@local).tolist()))
            source['axial_action_window']=make_window_evidence(source,[2.,6.],pieces,source_response_sha256=source['source_response_sha256'])
        result=primary_moment_window_evidence(guard,p,g,pe,**kw)
        self.assertEqual([row['interval_in'] for row in result['pieces']],[[2.,2.7],[2.7,3.1],[3.1,5.3],[5.3,6.]])
        nodes,weights=np.polynomial.legendre.leggauss(5);t=(nodes+1)/2
        expected=weights@np.array([exact(v) for v in t])/2
        np.testing.assert_allclose(result['average_resultants'],expected,atol=1e-12,rtol=1e-12)

    def test_P5_combination_keeps_physical_bending_and_signed_external_membrane(self):
        for axis in ('x','y'):
            j=0 if axis=='x' else 1
            for mode in ('strong','both'):
                guard,p,g,pe,kw,exact=self.fixture(axis,mode,increment_only=True)
                result=primary_moment_window_evidence(guard,p,g,pe,**kw)
                nodes,weights=np.polynomial.legendre.leggauss(5);t=(nodes+1)/2
                np.testing.assert_allclose(result['average_resultants'],weights@np.array([exact(v) for v in t])/2,atol=1e-12)
            # Independent polynomial addition for G+named P5, with the
            # physical bending retained instead of a second G bending term.
            g=copy.deepcopy(g);g['reference_s_in']=2.
            g['axial_action_window']['binding']['reference_s_in']=2.
            from Design.SMRF_Common_Floor_Actions import _hash
            g['axial_action_window']['payload_sha256']=_hash({k:v for k,v in g['axial_action_window'].items() if k!='payload_sha256'})
            primary=copy.deepcopy(p);primary['identity']['recovery_id']='P5'
            primary['axial_action_window']['binding']['recovery_id']='P5'
            primary['axial_action_window']['payload_sha256']=_hash({k:v for k,v in primary['axial_action_window'].items() if k!='payload_sha256'})
            keep=np.diag([0.,0.,0.,1.,1.,1.,0.]);gm=np.diag([1.,1.,1.,0.,0.,0.,1.]);gm[6,j]=1.
            pm=np.diag([1.,1.,1.,0.,0.,0.,1.])
            target=copy.deepcopy(p);target.pop('axial_action_window');target['identity']['recovery_id']='guard_P5'
            def expected(vg,vp):
                out=vp.copy();out[:3]=vg[:3]+vp[:3];out[6]=vg[6]+vg[j]+vp[6];return out
            target['resultants']=expected(np.array(g['resultants']),np.array(p['resultants'])).tolist()
            result=linear_combination_window(target,[p,g,primary],[keep,gm,pm],derivation_method='independent_test_P5')
            ag=np.asarray(g['axial_action_window']['average_resultants']);ap=np.asarray(p['axial_action_window']['average_resultants'])
            np.testing.assert_allclose(result['average_resultants'],expected(ag,ap),atol=1e-12)
            broken=copy.deepcopy(primary);broken['material_inventory_id']='another cage'
            with self.assertRaisesRegex(ValueError,'material/state'):linear_combination_window(target,[p,g,broken],[keep,gm,pm],derivation_method='test')
