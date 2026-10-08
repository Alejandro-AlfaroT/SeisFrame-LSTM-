"""Human mesh labels must preserve the completed-pair identity and raw evidence."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Design.SMRF_Slab_Refinement import refinement_diagnostics


def evidence_for(meshes):
    rows=[dict(panel_id='panel_x1_y1',axis='y',face='bottom',metric='vu_kip_per_ft',
               coarse=1.,fine=1.08,relative_change=.08,tolerance=.05,within_tolerance=False)]
    levels=[dict(index=i,requested_mesh=mesh,status='completed',
                 actions=dict(analysis_model_sha256=f'mesh{i}',strips=[])) for i,mesh in enumerate(meshes[:2])]
    levels.append(dict(index=2,requested_mesh=meshes[2],status='failed',error='saved solve error',
                       attempted_cases=[dict(loadcase={'id':'1.4D'},status='analysis_failed',analysis_return_code=-3)]))
    return dict(refinement=dict(status='analysis_failed',levels=levels,comparisons=[
        dict(coarse_analysis_sha256='mesh0',fine_analysis_sha256='mesh1',comparisons=rows)]))


class RefinementDiagnosticLabelTests(unittest.TestCase):
    def test_anisotropic_counts_label_failed_attempt_and_actual_completed_pair(self):
        meshes=[dict(subdivisions_per_bay=None,subdivisions_x_per_bay=x,subdivisions_y_per_bay=y)
                for x,y in ((12,18),(24,36),(48,72))]
        evidence=evidence_for(meshes)
        before=copy.deepcopy(evidence)
        report=refinement_diagnostics(evidence)
        self.assertIn('attempted level 2 (48x72 cells per bay) failed',report['detail'])
        self.assertIn('last completed comparison [12x18, 24x36] cells per bay',report['detail'])
        self.assertNotIn('None',report['detail'])
        self.assertIn('1 of 1 strip comparisons outside tolerance',report['detail'])
        self.assertEqual(evidence,before)
        self.assertEqual([level['requested_mesh'] for level in report['levels']],meshes)
        self.assertEqual(report['final_strip_comparisons'],before['refinement']['comparisons'][0]['comparisons'])
        self.assertEqual(report['failed_levels'][0]['attempted_cases'][0]['analysis_return_code'],-3)
        last=report['last_completed_comparison']
        self.assertEqual(last['coarse_level']['index'],0)
        self.assertEqual(last['fine_level']['index'],1)
        self.assertEqual(report['status'],'analysis_failed')

    def test_isotropic_scalar_labels_keep_existing_format_and_take_precedence(self):
        meshes=[dict(subdivisions_per_bay=n,subdivisions_x_per_bay=999,subdivisions_y_per_bay=888)
                for n in (24,48,60)]
        report=refinement_diagnostics(evidence_for(meshes))
        self.assertIn('attempted level 2 (60 cells per bay) failed',report['detail'])
        self.assertIn('last completed comparison [24, 48] cells per bay',report['detail'])
        self.assertNotIn('999',report['detail'])

    def test_missing_axis_counts_are_unknown_without_changing_raw_evidence(self):
        meshes=[{},dict(subdivisions_per_bay=48),dict(subdivisions_per_bay=None,subdivisions_x_per_bay=60)]
        report=refinement_diagnostics(evidence_for(meshes))
        self.assertIn('attempted level 2 (unknown cells per bay) failed',report['detail'])
        self.assertIn('last completed comparison [unknown, 48] cells per bay',report['detail'])
        self.assertEqual([level['requested_mesh'] for level in report['levels']],meshes)


if __name__=='__main__':
    unittest.main()
