import copy,sys,unittest
from pathlib import Path
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Model import Member_Groups as mg
from Design.SMRF_Plan_Symmetry import strength_distribution_bound,bound_from_record
from Design.SMRF_Demands import classify_torsional_irregularity

COLUMN=mg.MemberDesign('column',30.,30.,6.,8,4,4,2,4,4,4.,(4,4))
BEAM=mg.MemberDesign('beam_x',16.,30.,5.,8,4,4,0,4,2,4.)

def fixture(nx=3,ny=4,nz=6,policy=mg.SYSTEM_BANDED_POLICY_ID,column=COLUMN):
    state=mg.GroupedDesign.uniform(nx,ny,nz,column,BEAM,policy)
    geom=dict(num_bay_x=nx,num_bay_y=ny,num_floor=nz,bay_x_in=282.,bay_y_in=336.,story_h_in=186.)
    proof=strength_distribution_bound(state,geom,clear_cover_in=1.5,grid_in=.5)
    return state,geom,proof

class PlanStrengthSymmetry(unittest.TestCase):
    def test_all_three_layouts_odd_and_even_counts_have_no_centerline_strength(self):
        for policy in (mg.SYSTEM_BANDED_POLICY_ID,mg.SYSTEM_CORE_BANDED_POLICY_ID,mg.SYSTEM_CORE_LINES_BANDED_POLICY_ID):
            for nx,ny in ((3,3),(3,4),(4,5),(6,6)):
                state,geom,proof=fixture(nx,ny,policy=policy)
                self.assertTrue(proof['verified'],proof)
                self.assertEqual(proof['one_side_fraction'],.5)
                self.assertEqual(proof['member_reflection_checks'],2*len(state.members()))

    def test_fraction_does_not_depend_on_provisional_m1_numbers(self):
        proof=fixture()[2]
        result=classify_torsional_irregularity(1.1,.99,False,proof)
        self.assertEqual(result['label'],'none')
        self.assertEqual(result['strength_criterion_status'],'verified_by_reflection')
        self.assertEqual(classify_torsional_irregularity(1.3,.5,False,proof)['label'],'type_1')
        self.assertEqual(classify_torsional_irregularity(1.1,.5,False)['label'],'unresolved')

    def test_asymmetric_actual_column_cage_is_outside_the_proof(self):
        proof=fixture(column=replace(COLUMN,bot_bars=3))[2]
        self.assertFalse(proof['verified'])
        self.assertTrue(any('not doubly symmetric' in p for p in proof['problems']))

    def test_all_frame_policy_does_not_get_a_perimeter_proof(self):
        self.assertFalse(fixture(policy=mg.POLICY_ID)[2]['verified'])

    def test_saved_flag_cannot_replace_reconstruction_of_physical_members(self):
        state,geom,proof=fixture()
        record=dict(member_groups=state.to_record(),geometry=geom,materials={'col_clear_cover_in':1.5},
                    request_identity={'policy':{'rebar':{'bar_position_grid_in':.5}}})
        self.assertEqual(bound_from_record(record),proof)
        bad=copy.deepcopy(record)
        next(iter(bad['member_groups']['designs'].values()))['h_in']+=1
        self.assertFalse(bound_from_record(bad)['verified'])

if __name__=='__main__':unittest.main()
