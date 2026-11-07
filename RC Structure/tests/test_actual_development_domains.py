"""Full-fy domains keep absolute minimum lengths and actual steel ownership."""
import unittest
from Design.SMRF_Actual_Bar_Development import full_fy_developed_domain,side_development
from Design.SMRF_Transition_Physical import development_transverse_index

class Domains(unittest.TestCase):
    def fixture(self):
        lengths={'status':'pass','ld_tension_in':20.,'ld_compression_in':9.,'mandatory_confinement_passes':True}
        hooks=[{'status':'fail','available_in':2.,'ldh_required_in':6.,'standard_hook_geometry_passes':True,
            'mandatory_enclosing_ties_unproved':False},
            {'status':'pass','available_in':102.,'ldh_required_in':6.,'standard_hook_geometry_passes':True,
            'mandatory_enclosing_ties_unproved':False}]
        return side_development(0.,[0.,100.],lengths,hooks)
    def test_absolute_minimum_and_compression_govern_end_retreat(self):
        check=self.fixture();self.assertEqual(check['status'],'fail')
        domain=full_fy_developed_domain(check)
        self.assertEqual(domain['interval_in'],[9.,91.]);self.assertEqual(domain['physical_straight_run_interval_in'],[0.,100.])
        self.assertEqual(domain['endpoint_requirements'][0]['tension_retreat_in'],4.)
        self.assertFalse(domain['actual_demand_failure_established'])
    def test_unknown_hook_can_still_have_supported_straight_core(self):
        check=self.fixture()
        for side in check['sides']:side['hook_tension']={'status':'unknown'}
        domain=full_fy_developed_domain(check);self.assertEqual(domain['interval_in'],[20.,80.])
    def test_actual_physical_failure_cannot_become_credited_domain(self):
        check=self.fixture();check['physical_detail_status']='fail'
        self.assertEqual(full_fy_developed_domain(check)['status'],'fail')
    def test_no_undeclared_stress_reduction_or_phi_input(self):
        check=self.fixture();check['actual_stress_ksi']=.001;check['phi']=.65
        self.assertEqual(full_fy_developed_domain(check)['interval_in'],[9.,91.])
    def test_no_positive_domain_for_bar_shorter_than_two_required_ends(self):
        check=self.fixture();check['straight_run_interval_in']=[0.,10.]
        check['sides'][1]['straight_available_in']=10.;check['sides'][1]['hook_tension']['available_in']=12.
        self.assertEqual(full_fy_developed_domain(check)['status'],'fail')
    def test_nominal_lap_legs_do_not_pass_actual_projected_threshold(self):
        cage={'layout':{'legs':{'across_b_face':2,'across_h_face':4}},'hoop_leg_horizontal_projection_factor':.9}
        result=development_transverse_index(cage,.2,4.,8,1.)
        self.assertAlmostEqual(result['actual_crossing_area_in2'],.36)
        self.assertAlmostEqual(result['Ktr_in'],.45);self.assertFalse(result['passes'])
        self.assertEqual(40*2*.2/(4*8),.5) # old nominal shortcut passed this boundary

if __name__=='__main__':unittest.main()
