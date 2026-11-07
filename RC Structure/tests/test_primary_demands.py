"""Actual-CM wrench transport and complete primary torsion/drift records."""
import unittest
import copy
from unittest.mock import patch
import numpy as np
from Design.SMRF_Primary_Frame_Transfer import PrimaryFrame
from Design.SMRF_Primary_Demands import floor_loads,torsion_assessment,drift_screen
from Design.SMRF_Demands import validate_torsion_assessment
from test_coupled_floor_geometry import fixture,build
from test_primary_frame_transfer import primary_fixture
from test_physical_diaphragm import masses


class PrimaryDemandTests(unittest.TestCase):
    def test_actual_mass_transport_and_physical_outside_face_torsion(self):
        d=fixture();mass=masses(d);elf=dict(story_forces_kip=[10.,20.],story_weights_kip=[sum(r['weight_kip'] for r in mass[str(k)]) for k in (1,2)])
        # Move a little actual mass from left to right; preserve floor weight.
        moved=mass['1'][0]['weight_kip']/4
        mass['1'][0]['weight_kip']-=moved;mass['1'][2]['weight_kip']+=moved
        out=floor_loads(d,elf,mass,ex=1.3,ey=-.39,accidental_ratio=.05,amplification=2.,eccentricity_sign=-1.)
        cx=sum(r['weight_kip']*r['xy_in'][0] for r in mass['1'])/elf['story_weights_kip'][0]
        expected=-.1*(18*13+42*(-3.9))+(cx-20)*(-3.9)
        np.testing.assert_allclose(out['loads'][0],[13.,-3.9,expected],atol=1e-12)
        stale={**elf,'story_weights_kip':[1.,2.]}
        with self.assertRaisesRegex(ValueError,'floor weight differs'):
            floor_loads(d,stale,mass,ex=1.,ey=0.,accidental_ratio=.05)

    def test_actual_P_torsion_and_both_sign_drift_are_replayable(self):
        d=fixture();system=build(d,[dict(id='DL',dead=1.,live=1.)]);gravity=system.solve()
        primary=PrimaryFrame(primary_fixture(system),source_identity=system.source_identity)
        primary.transfer_gravity(gravity,case_id='DL');mass=masses(d)
        elf=dict(story_forces_kip=[.01,.02],story_weights_kip=[sum(r['weight_kip'] for r in mass[str(k)]) for k in (1,2)])
        torsion,source=torsion_assessment(primary,d,elf,mass,accidental_ratio=.05,strength_distribution={'one_side_fraction':None})
        self.assertTrue(validate_torsion_assessment(torsion,2,policy_ratio=.05)['valid'])
        self.assertEqual(set(source),{'x+','x-','y+','y-'})
        self.assertGreaterEqual(torsion['amplification'],1.)
        with patch('Design.SMRF_Substructure_Frame.SubstructureFrame.solve',side_effect=AssertionError('No frame solve during TIR readback')):
            repeated,_=torsion_assessment(primary,d,elf,mass,accidental_ratio=.05,strength_distribution={'one_side_fraction':None},saved_responses=source)
        self.assertEqual(repeated,torsion)
        tamper=copy.deepcopy(source);tamper['x+']['load']['loads'][0][0]+=.01
        with self.assertRaisesRegex(ValueError,'Saved torsion load'):
            torsion_assessment(primary,d,elf,mass,accidental_ratio=.05,strength_distribution={'one_side_fraction':None},saved_responses=tamper)
        basis=dict(importance_factor=1.,risk_category='II',allowable_story_drift_ratio=.02,redundancy_factor=1.3,
                   effective_story_drift_ratio=.02/1.3)
        result,source=drift_screen(primary,d,elf,mass,accidental_ratio=.05,amplification=torsion['amplification'],
            basis=basis,sdc='D',cd=5.5,gravity_above_by_story={1:5.,2:2.})
        self.assertEqual(len(result['stories']),4);self.assertEqual(len(source),4)
        with patch('Design.SMRF_Substructure_Frame.SubstructureFrame.solve',side_effect=AssertionError('No frame solve during drift readback')):
            replay,_=drift_screen(primary,d,elf,mass,accidental_ratio=.05,amplification=torsion['amplification'],
                basis=basis,sdc='D',cd=5.5,gravity_above_by_story={1:5.,2:2.},saved_responses=source)
        self.assertEqual(result,replay)
        for row in result['stories']:
            self.assertAlmostEqual(row['design_drift_in'],5.5*row['elastic_drift_in'],places=12)
            self.assertAlmostEqual(row['allowable_drift_ratio'],.02/1.3,places=12)
