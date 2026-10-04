from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.assert_torsional_strength import CHECK_ID, METHOD, checked_fraction_rows, replace_m1, validate_assertion


class AssertionPolicy(unittest.TestCase):
    def setUp(self):
        self.hashes={'review_torsional_strength.py':'a','_story_strength_reference.py':'b'}
        self.assertion=dict(asserted_by='research owner',assertion_date='2026-10-04',assertion_basis='scoped method review',
                            method=METHOD,adopted_for_v2_research=True,requires_per_design_evidence=True,
                            production_release_authorized=False,reviewer_sha256=self.hashes)

    def test_named_scoped_adoption(self):
        validate_assertion(self.assertion,self.hashes)

    def test_missing_provenance_and_blanket_or_changed_method_are_rejected(self):
        for field,value in [('asserted_by',''),('assertion_date','PROBE'),('requires_per_design_evidence',False),
                            ('method','beam_only'),('production_release_authorized',True),('reviewer_sha256',{})]:
            bad={**self.assertion,field:value}
            with self.subTest(field=field),self.assertRaises(ValueError):validate_assertion(bad,self.hashes)

    def test_replaces_only_m1_and_preserves_other_failures(self):
        original={'checks':[{'id':CHECK_ID,'clause':'torsion','status':'not_evaluated'},
                            {'id':'other','clause':'other check','status':'fail'}]}
        saved=deepcopy(original)
        result=replace_m1(original,{'id':CHECK_ID,'clause':'torsion','status':'pass'})
        self.assertEqual(result['counts'],{'pass':1,'fail':1,'not_evaluated':0})
        self.assertFalse(result['accepted'])
        self.assertEqual(original,saved)
        for checks in ([],[original['checks'][0]]*2,[{'id':CHECK_ID,'status':'fail'}]):
            with self.assertRaises(ValueError):replace_m1({'checks':checks},{'id':CHECK_ID,'status':'pass'})


class EvidenceCoverage(unittest.TestCase):
    def fixture(self,root):
        geometry=dict(num_floor=1,num_bay_x=2,num_bay_y=2)
        (root/'identity.json').write_text(json.dumps({'design_sha256':'design','knots':65,'kernel_sha256':'kernel'}))
        gravity=[]
        for axis in ('x','y'):
            for load in ('high','low'):
                for line in range(3):
                    rows=[]
                    for pattern in ('story_couple','uniform','height'):
                        for sway in (1,-1):
                            rows.append(dict(axis=axis,line=line,gravity=load,pattern=pattern,story=1,sway=sway,
                                             capacity_kip=100.,continuous_span_upper_bound_kip=100.,success=True,
                                             equilibrium_residual=0.,yield_violation=0.,compatibility_residual=0.,
                                             primal_dual_gap_kip=0.,maximum_beam_span_ratio=1.,maximum_column_dense_pm_ratio=1.,
                                             load_work=1.,mechanism=[]))
                    (root/f'{axis}_{load}_{line}.json').write_text(json.dumps(rows))
                    gravity.append(dict(axis=axis,line=line,gravity=load,success=True,reconstruction_residual=0.))
        (root/'gravity.json').write_text(json.dumps(gravity))
        return geometry

    def test_centerline_counts_on_both_sides_but_once_in_total(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);geometry=self.fixture(root)
            rows=checked_fraction_rows(root,geometry,'design',{'_story_strength_reference.py':'kernel'},65)
            self.assertEqual(len(rows),24)
            self.assertTrue(all(abs(r['fraction']-2/3)<1e-12 for r in rows))

    def test_missing_case_and_shear_mechanism_cannot_close_m1(self):
        for mode in ('missing','shear','nan','nan_work'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);geometry=self.fixture(root);path=root/'x_high_0.json';rows=json.loads(path.read_text())
                if mode=='missing':rows.pop()
                elif mode=='shear':rows[0]['mechanism']=[{'mode':'shear'}]
                elif mode=='nan':rows[0]['equilibrium_residual']=float('nan')
                else:rows[0]['load_work']=float('nan')
                path.write_text(json.dumps(rows))
                with self.assertRaises(ValueError):checked_fraction_rows(root,geometry,'design',{'_story_strength_reference.py':'kernel'},65)


if __name__=='__main__':unittest.main()
