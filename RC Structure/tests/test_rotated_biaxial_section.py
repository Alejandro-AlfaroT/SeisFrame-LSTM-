"""Independent triangle equilibrium, phi and biaxial failure counterexample."""
import math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Design.SMRF_Biaxial_Section import BiaxialSection
from Design import Group_Checks as checks
from Design.ACI_Checks import pm_diagram_for,_interpolate_pm_capacity

BARS=[(x,y,.79) for x,y in ((-4,-4),(0,-4),(4,-4),(-4,0),(4,0),(-4,4),(0,4),(4,4))]

class RotatedBiaxialSection(unittest.TestCase):
    def test_closed_form_triangular_block_and_separate_bar_forces(self):
        section=BiaxialSection(14,14,10,80,BARS)
        c=5.;a=.65*c
        # At 45 degrees the block is a right triangle of area a^2.
        # Its edge lies outside every bar circle here, so no displacement term.
        cc=.85*10*a*a;centroid=7-a*math.sqrt(2)/3
        forces=[(x,y,area*max(-80,min(80,87*(1-(14-x-y)/(math.sqrt(2)*c))))) for x,y,area in BARS]
        pn=cc+sum(f for x,y,f in forces)
        mh=cc*centroid+sum(y*f for x,y,f in forces)
        mb=cc*centroid+sum(x*f for x,y,f in forces)
        state=section.state(c,math.pi/4)
        self.assertEqual(float(state['phi']),.9)
        self.assertAlmostEqual(float(state['p']),.9*pn,places=9)
        self.assertAlmostEqual(float(state['mh']),.9*mh,places=9)
        self.assertAlmostEqual(float(state['mb']),.9*mb,places=9)
        inverse=section.ray_capacity([.9*pn],[.9*mh],[.9*mb])
        self.assertAlmostEqual(float(inverse['dcr'][0]),1.,places=7)

    def test_uniaxial_phi_cannot_be_reused_for_a_biaxial_state(self):
        section=BiaxialSection(14,14,10,80,BARS)
        rotated=section.at_axial(0.,math.pi/4)
        uniaxial=section.at_axial(0.,0.)
        self.assertLess(float(rotated['phi']),.72)
        self.assertEqual(float(uniaxial['phi']),.9)
        # A point 5% outside the true diagonal strength looked safe under alpha=1.5.
        demand=1.05*float(rotated['mh'])
        layers=[(3*.79,3.),(2*.79,7.),(3*.79,11.)]
        old=_interpolate_pm_capacity(0.,pm_diagram_for(14,14,10,80,29000,layers))
        old_dcr=2**(1/1.5)*demand/old
        self.assertLess(old_dcr,.8)
        new=section.ray_capacity([0.],[demand],[demand])
        self.assertAlmostEqual(float(new['dcr'][0]),1.05,places=7)

    def test_batched_capacity_is_the_scalar_capacity_in_both_quadrants(self):
        section=BiaxialSection(14,14,10,80,BARS)
        p=[0.,100.,300.];mh=[500.,-500.,700.];mb=[200.,400.,-100.]
        batch=section.ray_capacity(p,mh,mb)['capacity']
        for i in range(3):
            scalar=section.ray_capacity([p[i]],[mh[i]],[mb[i]])['capacity'][0]
            self.assertAlmostEqual(float(batch[i]/scalar),1.,places=9)

    def test_domain_failures_remain_explicit_in_group_selection(self):
        section=BiaxialSection(14,14,10,80,BARS)
        diagrams={'biaxial_section':section}
        rows=[(1,'compression',section.compression_cap+1,5,5,0,0),
              (1,'tension',section.tension_cap-1,0,0,0,0),
              (1,'tension_moment',section.tension_cap,1,0,0,0)]
        result=checks.column_pm_envelope(diagrams,rows)
        self.assertTrue(result['outside_domain'])
        self.assertEqual(result['domain_failures'],{'compression':1,'tension':1,'moment_at_axial_end':1})

    def test_asymmetric_cage_is_refused_instead_of_using_absolute_moments(self):
        with self.assertRaisesRegex(ValueError,'doubly symmetric'):
            BiaxialSection(14,14,10,80,BARS[:-1])

    def test_selection_acceptance_requires_every_row_and_rejection_has_an_exact_witness(self):
        section=BiaxialSection(14,14,10,80,BARS)
        layers=[(3*.79,3.),(2*.79,7.),(3*.79,11.)]
        diagram=pm_diagram_for(14,14,10,80,29000,layers)
        diagrams={'y':diagram,'z':diagram,'biaxial_section':section}
        passing=[(1,str(i),float(i),100.,200.,0.,0.) for i in range(12)]
        result=checks.column_pm_selection_screen(diagrams,passing,1.)
        self.assertTrue(result['all_rows_checked'])
        self.assertEqual(len(result['values']),len(passing))
        failing=passing+[(1,'governing',0.,2000.,2000.,0.,0.)]
        result=checks.column_pm_selection_screen(diagrams,failing,1.)
        self.assertFalse(result['all_rows_checked'])
        self.assertGreater(result['dcr'],1.)
        witness=failing[result['index']]
        self.assertGreater(checks.column_pm_dcr(diagrams,*witness[2:5])['dcr'],1.)

if __name__=='__main__':unittest.main()
