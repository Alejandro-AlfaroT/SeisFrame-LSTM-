"""Both axial ends must be checked even when the larger compression looks safer."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Design import Group_Checks as checks

class ColumnEndAxialEnvelope(unittest.TestCase):
    def test_rising_pm_branch_can_be_governed_by_the_smaller_top_axial_force(self):
        # Independent piecewise-linear reference surface: at P=0, M=100;
        # at P=100, M=200. Bottom P=100 passes, top P=80 fails at M=190.
        diagrams={axis:[(-50.,0.),(0.,100.),(100.,200.),(200.,0.)] for axis in ('y','z')}
        f=[100.,3.,4.,0.,190.,0.,-80.,-3.,-4.,0.,-185.,0.]
        rows=checks.column_demand_rows([1],[{'id':'D+E','members':{'1':{'local_force_kip_kipin':f}}}])
        self.assertEqual([row[2] for row in rows],[100.,80.])
        self.assertEqual([row[7] for row in rows],['i','j'])
        value=checks.column_pm_envelope(diagrams,rows)
        self.assertAlmostEqual(value['dcr'],190./180.)
        self.assertEqual(rows[value['index']][7],'j')
        self.assertLess(checks.column_pm_dcr(diagrams,100.,190.,0.)['dcr'],1.)

    def test_top_tension_is_not_hidden_by_bottom_compression(self):
        diagrams={axis:[(-50.,0.),(0.,100.),(100.,200.),(200.,0.)] for axis in ('y','z')}
        f=[10.,0.,0.,0.,0.,0.,60.,0.,0.,0.,0.,0.]
        rows=checks.column_demand_rows([7],[{'id':'signed','members':{'7':{'local_force_kip_kipin':f}}}])
        value=checks.column_pm_envelope(diagrams,rows)
        self.assertEqual(value['domain_failures']['tension'],1)
        self.assertAlmostEqual(value['dcr'],1.2)

    def test_component_envelope_is_preserved_at_each_axial_end(self):
        f=[100.,3.,4.,0.,9.,2.,-80.,-6.,-2.,0.,-7.,-5.]
        rows=checks.column_demand_rows([1],[{'id':'case','members':{'1':{'local_force_kip_kipin':f}}}])
        self.assertEqual([row[3:7] for row in rows],[(9.,5.,6.,4.),(9.,5.,6.,4.)])

if __name__=='__main__':unittest.main()
