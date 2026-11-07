"""Independent section equilibrium examples: concrete and steel occupy disjoint area."""
import unittest
from Design.SMRF_Beam_Slab_Strength import section_moment

class SectionConcreteDisplacement(unittest.TestCase):
    def test_doubly_reinforced_closed_form_with_both_layers_yielded(self):
        # b=12, h=24, fc=4, fy=60, A's=4 at 2, As=8 at 22.
        # T=480; C's=4*(60-3.4); Cc=T-C's; a=Cc/(3.4*12).
        compression_steel=4*(60-3.4)
        concrete=480-compression_steel
        a=concrete/(3.4*12)
        expected=480*22-compression_steel*2-concrete*a/2
        result=section_moment([(2,4),(22,8)],4,60,24,12)
        self.assertAlmostEqual(result['mn_kip_in'],expected,places=8)
        self.assertAlmostEqual(result['neutral_axis_in'],a/.85,places=9)
        self.assertAlmostEqual(result['steel_forces_kip'][0][1],240,places=9)
        self.assertEqual(result['displaced_concrete_forces_kip'],[(2.,13.6)])
        self.assertAlmostEqual(result['axial_equilibrium_residual_kip'],0,places=8)

    def test_singly_reinforced_section_is_unchanged(self):
        result=section_moment([(21.5,3.16)],6,60,24,20)
        a=3.16*60/(.85*6*20)
        self.assertAlmostEqual(result['mn_kip_in'],3.16*60*(21.5-a/2),places=8)
        self.assertEqual(result['displaced_concrete_forces_kip'],[])

    def test_mixed_grade_flange_section_balances_separate_material_forces(self):
        layers=[(2.5,3.16),(27.5,6.32),(1,1.5,60),(7,1.5,60)]
        result=section_moment(layers,5,80,30,16,60,8)
        a=result['stress_block_in']
        area=60*min(a,8)+16*max(0,a-8)
        steel=sum(f for _,f in result['steel_forces_kip'])
        removed=4.25*sum(area for y,area,*_ in layers if y<=a)
        self.assertAlmostEqual(4.25*area+steel-removed,0,places=8)
        self.assertAlmostEqual(result['gross_concrete_force_kip'],4.25*area,places=8)

    def test_hogging_block_can_reach_flange_of_shallow_web(self):
        layers=[(2,1.58),(14,2.37),(2.75,1.6,60),(15.25,1.6,60)]
        result=section_moment(layers,5,60,16,10,46,14,flange_at_far_face=True)
        a=result['stress_block_in'];self.assertGreater(a,2)
        self.assertAlmostEqual(result['gross_concrete_force_kip'],4.25*(10*a+36*(a-2)),places=8)
        self.assertAlmostEqual(result['axial_equilibrium_residual_kip'],0,places=8)

    def test_far_flange_has_no_effect_before_block_reaches_it(self):
        plain=section_moment([(27.5,3.16)],5,60,30,16)
        flange=section_moment([(27.5,3.16)],5,60,30,16,60,8,flange_at_far_face=True)
        self.assertAlmostEqual(plain['mn_kip_in'],flange['mn_kip_in'],places=8)

if __name__=='__main__':unittest.main()
