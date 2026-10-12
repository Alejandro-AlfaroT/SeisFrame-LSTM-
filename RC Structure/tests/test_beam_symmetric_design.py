"""Beam bar policy: independent top and bottom counts by default (2026-10-11), the symmetric option explicit.

No structural analyses are launched. Without a slab reinforcement layout nothing is credited to the top bars, so
these cases exercise the bands on the web bars alone (the slab credit has its own tests).
"""
import contextlib
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
import Redesign as rd
from Design.Config import DesignConfig


class BeamBarPolicyTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for key, value in {
            'SLAB_THICKNESS_IN': 6.0, 'B_BEAM': 10.0, 'H_BEAM': 18.0,
            'B_COL': 20.0, 'H_COL': 20.0, 'NUM_BAY_X': 2, 'NUM_BAY_Y': 6,
            'FC_BEAM_KSI': 4.0, 'FY_KSI': 60.0, 'BEAM_CLEAR_COVER_IN': 1.5,
            'BEAM_STIRRUP_BAR_SIZE': 4, 'AGGREGATE_MAX_SIZE_IN': 0.75,
            'BEAM_BAR_SIZE': 6, 'BEAM_TOP_BARS': 3, 'BEAM_BOT_BARS': 2,
            'SLAB_REINFORCEMENT': None,
        }.items():
            self.stack.enter_context(mock.patch.object(sp, key, value))
        self.cfg = DesignConfig()
        self.cfg.rebar.bar_sizes_beam = [6]
        self.cfg.rebar.beam_n_range = (2, 3)

    def test_independent_faces_are_the_default(self):
        self.assertFalse(DesignConfig().rebar.beam_symmetric)
        self.assertEqual(rd._beam_candidates(1.2, 1.4, .8, 1.0, self.cfg), [(6, 3, 2)])

    def test_the_symmetric_option_lets_the_larger_demand_govern_both_faces(self):
        self.cfg.rebar.beam_symmetric = True
        self.assertEqual(rd._beam_candidates(1.2, 1.4, .8, 1.0, self.cfg), [(6, 3, 3)])

    def test_spacing_still_excludes_impossible_cages(self):
        sp.B_BEAM = 7.0
        self.assertEqual(rd._beam_candidates(1.2, 1.4, .8, 1.0, self.cfg), [])

    def test_bottom_bars_follow_the_sagging_demand_within_the_reversal_floor(self):
        result = {1: {'type': 'beam', 'Mu_neg': 900., 'Mu_pos': 500.,
                      'dcr_neg': .85, 'dcr_pos': .70, 'dcr_V': .2}}
        _, update, _converged, _ = rd.redesign_steel(result, self.cfg)
        self.assertIsNotNone(update)
        self.assertLess(update['n_bot'], update['n_top'])
        # ACI 318-19 18.6.3.2 on the web bars: the sagging strength is at least half the hogging strength
        self.assertGreaterEqual(2 * update['n_bot'], update['n_top'])

    def test_an_in_band_asymmetric_cage_is_not_converged_under_the_symmetric_option(self):
        self.cfg.rebar.beam_symmetric = True
        result = {1: {'type': 'beam', 'Mu_neg': 900., 'Mu_pos': 500.,
                      'dcr_neg': .85, 'dcr_pos': .70, 'dcr_V': .2}}
        _, update, converged, _ = rd.redesign_steel(result, self.cfg)
        self.assertFalse(converged)
        self.assertIsNotNone(update)
        self.assertEqual(update['n_top'], update['n_bot'])

    def test_fallback_preserves_demand_minimum(self):
        # An impossible demand must not fall back to a small code-min cage, symmetric or not.
        result = {1: {'type': 'beam', 'Mu_neg': 10000., 'Mu_pos': 500.,
                      'dcr_neg': 9., 'dcr_pos': .70, 'dcr_V': 1.2}}
        for symmetric in (False, True):
            self.cfg.rebar.beam_symmetric = symmetric
            _, update, converged, _ = rd.redesign_steel(result, self.cfg)
            self.assertFalse(converged)
            if update is not None:
                # only the installed bars with a tighter stirrup spacing (shear DCR 1.2); under the symmetric
                # option the unequal installed faces block even that
                self.assertFalse(symmetric)
                self.assertEqual((update['bar_size'], update['n_top'], update['n_bot']), (6, 3, 2))
                self.assertIn('stirrup_spacing', update)


if __name__ == '__main__':
    unittest.main()
