"""Exact virtual-work integrals and refusal of unmatched bar/action lengths."""
import copy
import unittest
import numpy as np
from Design.SMRF_Axial_Action_Accounting import (integrate_window,make_window_evidence,
    verify_point_accounting,physical_window_evidence,select_axial_window,connected_web_domain)
from Design.SMRF_Physical_Floor_Strength import assess_physical_cut_family,verify_physical_cut_assessment
from tests.test_physical_floor_strength import PhysicalStrengthTests


def piece(a,b,c):
    c=np.asarray(c,float)
    return dict(interval_in=[a,b],coefficients=c.tolist(),
        sample_values=(np.array([[1,0,0],[1,.5,.25],[1,1,1]])@c).tolist(),
        quarter_witness=(np.array([1,.25,.0625])@c).tolist())


class AxialAccountingTests(unittest.TestCase):
    def test_full_window_translates_at_both_faces_and_does_not_shorten(self):
        for station,expected in ((10.,[10.,34.]),(90.,[66.,90.]),(50.,[38.,62.])):
            actual=select_axial_window(station,24.,[10.,90.],[])
            self.assertEqual(actual['status'],'computed');self.assertEqual(actual['interval_in'],expected)
            self.assertFalse(actual['length_shortened'])
        short=select_axial_window(5.,24.,[0.,10.],[])
        self.assertEqual(short['status'],'not_evaluated');self.assertNotIn('interval_in',short)
        # The end face cuts a real transverse bar: no clipping or deleting it.
        bar=[dict(db_in=1.,global_transverse_coordinates_in=[10.])]
        blocked=select_axial_window(10.,24.,[10.,90.],bar)
        self.assertEqual(blocked['status'],'not_evaluated')
        enlarged=select_axial_window(50.,24.,[10.,90.],[dict(db_in=1.,global_transverse_coordinates_in=[38.])])
        self.assertLess(enlarged['interval_in'][0],37.5)
        self.assertGreater(enlarged['actual_length_in'],24.)

    def test_connected_web_domain_does_not_bridge_joint_or_ambiguous_trace(self):
        from tests.test_physical_floor_cuts import beam
        first=beam('x',length=10.);second=copy.deepcopy(first);second['id']='other'
        second['positions_in']=[[10.,0.,-6.],[20.,0.,-6.]]
        response=dict(elements=[second,first])
        self.assertEqual(connected_web_domain(response,'x','beam',10.,'plus'),[0.,20.])
        self.assertEqual(connected_web_domain(response,'x','beam',20.,'minus'),[0.,20.])
        second['positions_in'][0][0]=11.
        self.assertEqual(connected_web_domain(response,'x','beam',10.,'minus'),[0.,10.])
        with self.assertRaises(ValueError):connected_web_domain(response,'x','beam',10.,'plus')
        with self.assertRaises(ValueError):connected_web_domain(response,'x','beam',10.5,'plus')
        second['positions_in'][0][0]=9.
        with self.assertRaises(ValueError):connected_web_domain(response,'x','beam',5.,'plus')

    def test_changed_recovery_cannot_reuse_physical_window(self):
        _,spec,action,_=PhysicalStrengthTests().fixture()
        action['identity']['recovery_id']+='|additional_guard'
        with self.assertRaises(ValueError):verify_point_accounting(action,spec)

    def test_refused_or_changed_selection_cannot_borrow_matching_window(self):
        _,spec,action,_=PhysicalStrengthTests().fixture()
        spec['axial_window_selection']=select_axial_window(5.,4.,[0.,10.],[])
        self.assertEqual(verify_point_accounting(action,spec)['status'],'pass')
        spec['axial_window_selection']['requested_length_in']=24.
        with self.assertRaises(ValueError):verify_point_accounting(action,spec)
        spec['axial_window_selection']=select_axial_window(5.,24.,[0.,10.],[])
        self.assertEqual(verify_point_accounting(action,spec)['status'],'unknown')

    def test_cached_physical_window_keeps_selection_scope(self):
        from tests.test_common_local_flexure import LocalFlexureTests
        from Design.SMRF_Common_Guard_Plan import _PhysicalSource
        common,bundle=LocalFlexureTests().fixture();source=_PhysicalSource(bundle,common)
        spec=dict(floor=1,axis='x',station_in=0.,interval_in=[-22.,22.],trace='plus',
            axial_accounting_interval_in=[-12.,12.],cut_id='window',member_id=None)
        spec['axial_window_selection']=select_axial_window(0.,24.,[-50.,50.],[])
        self.assertEqual(source.cut(spec,'G')['axial_action_window']['status'],'computed')
        # Same interval/field, but the new requested length does not fit.
        spec['axial_window_selection']=select_axial_window(0.,200.,[-50.,50.],[])
        self.assertEqual(source.cut(spec,'G')['axial_action_window']['status'],'not_evaluated')

    def test_all_seven_independent_polynomial_work_integrals(self):
        c=np.array([[2,-3,4,-5,6,-7,8],[3,5,-7,9,-11,13,-15],[-1,2,3,-4,5,6,-7]],float)
        result=integrate_window([piece(10,14,c)],[10,14])
        np.testing.assert_allclose(result['average_resultants'],c[0]+c[1]/2+c[2]/3,atol=1e-14)
        np.testing.assert_allclose(result['first_axial_moment_about_window_start'],16*(c[0]/2+c[1]/3+c[2]/4),atol=1e-14)
        # Independently integrate the polynomial work against q0+q1*x.
        q0=np.arange(1,8);q1=np.arange(8,15)/10
        exact=4*q0@(c[0]+c[1]/2+c[2]/3)+16*q1@(c[0]/2+c[1]/3+c[2]/4)
        actual=q0@result['integral_resultants']+q1@result['first_axial_moment_about_window_start']
        self.assertAlmostEqual(actual,exact,11)

    def test_piecewise_window_gap_overlap_and_bad_witness_refused(self):
        c=np.zeros((3,7));c[0]=1
        result=integrate_window([piece(0,1,c),piece(1,3,2*c)],[0,3])
        np.testing.assert_allclose(result['average_resultants'],5/3)
        for rows in ([piece(0,1,c),piece(2,3,c)],[piece(0,2,c),piece(1,3,c)],[piece(0,1,c)]):
            with self.assertRaises(ValueError):integrate_window(rows,[0,3])
        bad=piece(0,3,c);bad['quarter_witness'][4]+=.01
        with self.assertRaises(ValueError):integrate_window([bad],[0,3])

    def assess(self,action):
        inv,spec,_,dev=PhysicalStrengthTests().fixture()
        return assess_physical_cut_family(None,inv,[spec],{'c':action},slab_fc_ksi=5,ec_ksi=4000,phi_requirements={'NM':.65},
            required_cut_ids=['c'],required_case_ids=[action['identity']['combination_id']],assessment_identity='accounting',development_result=dev)

    def test_missing_accounting_cannot_pass_even_zero_material_solve(self):
        _,_,action,_=PhysicalStrengthTests().fixture();action.pop('axial_action_window')
        result=self.assess(action);self.assertEqual(result['status'],'unknown')
        self.assertIn('transverse_bar_length_and_concurrent_action_basis_not_established',result['records'][0]['pending'])
        replay=verify_physical_cut_assessment(result);self.assertEqual(replay['status'],'unknown')
        self.assertTrue(replay['saved_claims_verified'])

    def test_point_zero_with_nonzero_quadratic_average_remains_unknown(self):
        _,spec,action,_=PhysicalStrengthTests().fixture()
        # Nyy=(x-5)^2 over[3,7]; its midpoint is zero but its average is4/3.
        c=np.zeros((3,7));c[:,1]=[4,-16,16]
        action['axial_action_window']=make_window_evidence(action,[3,7],[piece(3,7,c)],source_response_sha256=action['source_response_sha256'])
        proof=verify_point_accounting(action,spec)
        self.assertEqual(proof['status'],'unknown');self.assertAlmostEqual(proof['average_resultants'][1],4/3,14)
        result=self.assess(action);self.assertEqual(result['status'],'unknown')
        self.assertEqual(action['resultants'],[0.]*7) # Original point never silently averaged away.
        self.assertEqual(verify_physical_cut_assessment(result)['status'],'unknown')

    def test_wrong_case_window_source_or_resealed_integral_refused(self):
        _,spec,action,_=PhysicalStrengthTests().fixture()
        for key,value in (('combination_id','other'),('state_identity','other')):
            bad=copy.deepcopy(action);bad['identity'][key]=value
            with self.assertRaises(ValueError):verify_point_accounting(bad,spec)
        badspec=dict(spec,axial_accounting_interval_in=[2,7])
        with self.assertRaises(ValueError):verify_point_accounting(action,badspec)
        bad=copy.deepcopy(action);bad['axial_action_window']['average_resultants'][0]=1
        from Design.SMRF_Common_Floor_Actions import _hash
        e=bad['axial_action_window'];e['payload_sha256']=_hash({k:v for k,v in e.items() if k!='payload_sha256'})
        with self.assertRaises(ValueError):verify_point_accounting(bad,spec)

    def test_actual_slab_and_web_gravity_integral_both_axes(self):
        from tests.test_physical_floor_cuts import shell,beam
        from Design.SMRF_Physical_Floor_Cuts import common_strip_cut
        from Design.SMRF_Primary_Moment_Guards import physical_cut_states
        for axis in ('x','y'):
            j=0 if axis=='x' else 1;c=np.zeros((4,8));c[0]=[1.,2.,.1,3.,4.,.2,.3,.4]
            elements=[shell('slab',-5,15,-5,15,c),beam(axis,n=7,q=-2,length=10,m0=0,v0=10,z=-6)]
            response=dict(elements=elements,force_partition_relative_error=0.)
            spec=dict(axis=axis,station_in=5.,interval_in=[-2.,2.],axial_accounting_interval_in=[3.,7.],member_id='beam')
            raw=common_strip_cut(response,axis=axis,station_in=5.,interval_in=[-2.,2.],trace='plus',web_id='beam',
                uniform_web_loads={'beam:0':-2},state_identity='s',combination_id='g',location_id='mid',recovery_id='actual_element_fields')
            raw.update(material_inventory_id='actual_bars',analysis_input_identity='raw_field_fixture',source_response_sha256='actual_field_digest')
            normalized,_=physical_cut_states([raw],outer_interval_in=[-2,2]);action=normalized[0]
            proof=physical_window_evidence(response,spec,action,uniform_web_loads={'beam:0':-2})
            expected=np.r_[4*c[0,:6],0.];expected[j]+=7.
            # Independent exact average of x^2-10x over[3,7], plus -6*N.
            expected[j+3]+=(7**3-3**3)/12-5*(7**2-3**2)/4-42
            np.testing.assert_allclose(proof['average_resultants'],expected,atol=1e-12)
            action['axial_action_window']=proof
            result=verify_point_accounting(action,spec);self.assertEqual(result['status'],'unknown')
            self.assertAlmostEqual(result['residual'][j+3],-4/3,12)
            missing=dict(spec,axial_accounting_interval_in=[-1.,7.])
            gap=physical_window_evidence(response,missing,action,uniform_web_loads={'beam:0':-2})
            self.assertEqual(gap['status'],'not_evaluated')

    def test_material_ends_and_development_changes_inside_window_stay_unknown(self):
        from Design.SMRF_Physical_Floor_Materials import spatial_section_from_inventory
        inv,spec,action,dev=PhysicalStrengthTests().fixture()
        args={k:spec[k] for k in ('floor','axis','station_in','interval_in','axial_accounting_interval_in')}
        def build():
            return spatial_section_from_inventory(inv,**args,fc_ksi=5,ec_ksi=4000,phi_requirements={'NM':.65},
                development_evidence=dev['cuts'][0]['evidence'],apply_development_domains=True)
        self.assertEqual(build()['axial_material_uniformity']['status'],'pass')
        extra=copy.deepcopy(inv['floors']['1']['bars'][0]);extra.update(id='ending_before_point',straight_run_interval_in=[0,4])
        inv['floors']['1']['bars'].append(extra)
        result=build();self.assertEqual(result['axial_material_uniformity']['status'],'unknown')
        self.assertNotIn('ending_before_point',result['physical_bar_ids'])
        self.assertIn('ending_before_point',{r['bar_id'] for r in result['axial_material_uniformity']['issues']})
        inv['floors']['1']['bars'].pop()
        dev['cuts'][0]['evidence'][0]['full_fy_development_domain']['interval_in']=[4,8]
        # Point5 is fully credited; part of window[3,7] is not.
        result=build();self.assertEqual(result['development']['status'],'pass')
        self.assertEqual(result['axial_material_uniformity']['status'],'unknown')
        self.assertIn('longitudinal_strength_credit_changes_inside_axial_window',
            {r['reason'] for r in result['axial_material_uniformity']['issues']})


if __name__=='__main__':unittest.main()
