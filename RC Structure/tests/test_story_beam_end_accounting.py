import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Design.Grouped_Design import beam_end_sway_sums

class BeamEndSwayAccounting(unittest.TestCase):
    def test_undeveloped_end_is_not_replaced_by_member_wide_slab_strength(self):
        values={'hogging':200.,'sagging':120.,'basis':{
            'sizing_hogging_i_kip_in':100.,'sizing_hogging_j_kip_in':200.,
            'sizing_sagging_i_kip_in':80.,'sizing_sagging_j_kip_in':120.}}
        self.assertEqual(beam_end_sway_sums(values),(220.,280.))
        self.assertLess(max(beam_end_sway_sums(values)),values['hogging']+values['sagging'])

    def test_line_sway_must_be_summed_before_taking_the_weaker_sense(self):
        left={'hogging':200.,'sagging':120.,'basis':{
            'sizing_hogging_i_kip_in':100.,'sizing_hogging_j_kip_in':200.,
            'sizing_sagging_i_kip_in':80.,'sizing_sagging_j_kip_in':120.}}
        right={'hogging':200.,'sagging':120.,'basis':{
            'sizing_hogging_i_kip_in':200.,'sizing_hogging_j_kip_in':100.,
            'sizing_sagging_i_kip_in':120.,'sizing_sagging_j_kip_in':80.}}
        a,b=beam_end_sway_sums(left),beam_end_sway_sums(right)
        self.assertEqual(min(a[0]+b[0],a[1]+b[1]),500.)
        self.assertNotEqual(min(a)+min(b),500.)

    def test_no_slab_adapter_is_unchanged(self):
        self.assertEqual(beam_end_sway_sums({'hogging':200.,'sagging':120.}),(320.,320.))

if __name__=='__main__':unittest.main()
