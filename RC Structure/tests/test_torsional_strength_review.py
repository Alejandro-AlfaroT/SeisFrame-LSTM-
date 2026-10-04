"""Independent portal mechanics and fail-closed candidate-review decisions."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools._story_strength_reference import benchmark, column_interaction
from tools.review_torsional_strength import compare_refinement, validate_input


def fractions(value, floors=2):
    return [dict(axis=a,gravity=g,pattern=p,story=s,sway=d,fraction=value,numerical_upper=value)
            for a in ('x','y') for g in ('high','low')
            for p in ('story_couple','uniform','height')
            for s in (range(1,floors+1) if p=='story_couple' else [1]) for d in (1,-1)]


class ReferenceMechanics(unittest.TestCase):
    def test_beam_and_column_controlled_portals(self):
        rows=benchmark()
        self.assertEqual(len(rows),2)
        for row in rows:
            self.assertAlmostEqual(row['static_capacity_kip'],row['analytical_capacity_kip'],places=9)
            self.assertAlmostEqual(row['kinematic_capacity_kip'],row['analytical_capacity_kip'],places=9)

    def test_column_curve_has_correct_steel_and_axial_cap(self):
        record={'sections':dict(b_col_in=24.,h_col_in=24.,fc_col_ksi=4.),
                'reinforcement':dict(col_longitudinal_centroid_offset_in=2.5,col_bar_area_in2=1.,
                                     col_top_bars=4,col_bot_bars=4,col_side_bars=2),
                'materials':dict(fy_ksi=60.,es_ksi=29000.)}
        x=column_interaction(record,'x');y=column_interaction(record,'y')
        self.assertEqual(x['bar_count'],12)
        self.assertAlmostEqual(x['N'][0],-720.)
        self.assertAlmostEqual(x['nominal_axial_cap_kip'],.8*(.85*4*(24*24-12)+720))
        self.assertLessEqual(x['max_dense_polygon_excess_kip_in'],1e-6)
        self.assertEqual(x['N'],y['N'])
        for a,b in zip(x['M'],y['M']):self.assertAlmostEqual(a,b,places=8)


class ReviewDecision(unittest.TestCase):
    def test_stable_result_below_threshold(self):
        result=compare_refinement(fractions(.68),fractions(.6801),2)
        self.assertTrue(result['strength_absence_supported'])
        self.assertAlmostEqual(result['maximum_refinement_envelope'],.6802)

    def test_refinement_crossing_threshold_stays_open(self):
        result=compare_refinement(fractions(.7496),fractions(.7499),2)
        self.assertTrue(result['converged'])
        self.assertFalse(result['strength_absence_supported'])

    def test_large_change_stays_open_even_far_below_threshold(self):
        self.assertFalse(compare_refinement(fractions(.60),fractions(.61),2)['strength_absence_supported'])

    def test_missing_or_duplicate_story_and_sway_are_rejected(self):
        rows=fractions(.68)
        for bad in (rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):compare_refinement(rows,bad,2)

    def test_nonfinite_or_inverted_bounds_are_rejected(self):
        for value,upper in [(float('nan'),.7),(.7,float('inf')),(.7,.6),(-.1,.7)]:
            rows=fractions(.68);bad=deepcopy(rows);bad[0].update(fraction=value,numerical_upper=upper)
            with self.assertRaises(ValueError):compare_refinement(rows,bad,2)

    def test_grouped_input_is_not_silently_uniform(self):
        with self.assertRaisesRegex(ValueError,'Grouped'):
            validate_input({'member_groups':{}})


if __name__=='__main__':unittest.main()
