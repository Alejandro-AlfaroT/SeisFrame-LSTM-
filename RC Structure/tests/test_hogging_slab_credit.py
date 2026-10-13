"""The slab credit to the beam top bars (2026-10-11) and the two V4 repairs of 2026-10-12.

No structural analyses are launched: a hand-written slab layout on a small grid.
"""
import contextlib
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
import Redesign as rd
from Design.Config import DesignConfig
from Design.SMRF_Beam_Slab_Strength import (credited_hogging_layers, family_hogging_layers, hogging_slab_credit,
                                            hogging_strength_kip_in, web_top_steel_for_hogging)
from Design.SMRF_Design_Evidence import analysis_input_signature

LAYOUT = {"bar_size": 4, "layers": {f"{axis}_{face}": {"bar_area_in2": 0.20, "spacing_in": 12.0, "effective_depth_in": 6.0}
                                    for axis in ("x", "y") for face in ("top", "bottom")}}
GEOMETRY = {"num_bay_x": 2, "num_bay_y": 3, "num_floor": 4, "bay_x_in": 288.0, "bay_y_in": 264.0, "h_col_in": 24.0, "b_col_in": 24.0}


def credit(beam_width_in=20.0):
    return hogging_slab_credit(LAYOUT, 7.0, beam_width_in, GEOMETRY, 1.5, 0.5, 6.0, 60.0)


class HoggingSlabCreditTests(unittest.TestCase):
    def test_member_keys_are_strings_and_survive_a_json_round_trip(self):
        c = credit()
        self.assertTrue(c["members"])
        self.assertTrue(all(type(k) is str for k in c["members"]))
        self.assertEqual(json.loads(json.dumps(c)), c)
        first = int(next(iter(c["members"])))
        self.assertEqual(credited_hogging_layers(c, first, 24.0), credited_hogging_layers(c, str(first), 24.0))
        self.assertTrue(credited_hogging_layers(c, first, 24.0))

    def test_the_analysis_input_signature_is_the_same_before_and_after_reloading(self):
        record = {"geometry": GEOMETRY, "sections": {"b_beam_in": 20.0}, "reinforcement": {"hogging_slab_credit": credit()},
                  "materials": {"fy_ksi": 60.0}, "floor_loads": {}, "seismic": {"sds": 1.0}, "demand": {}}
        self.assertEqual(analysis_input_signature(record), analysis_input_signature(json.loads(json.dumps(record))))

    def test_families_and_the_least_credited_end(self):
        c = credit()
        self.assertEqual(set(c["families"]), {"x_edge", "x_interior", "y_edge", "y_interior"})
        self.assertGreater(c["families"]["x_interior"]["slab_steel_in_flange_in2"], c["families"]["x_edge"]["slab_steel_in_flange_in2"])
        self.assertTrue(all(m["credited"] for m in c["members"].values()))          # 20 in perimeter beams develop a #4 hook
        narrow = credit(beam_width_in=7.0)                                            # 5 in of embedment: the hook does not fit
        self.assertFalse(narrow["anchorage"]["x"]["developed"])
        uncredited = [k for k, m in narrow["members"].items() if not m["credited"]]
        self.assertTrue(uncredited)
        self.assertTrue(all("exterior end" in narrow["members"][k]["reason"] for k in uncredited))
        self.assertEqual(credited_hogging_layers(narrow, uncredited[0], 24.0), [])
        self.assertTrue(family_hogging_layers(narrow, "x_interior", 24.0))

    def test_sizing_with_the_credit_closes_on_the_checks_solve(self):
        layers = family_hogging_layers(credit(), "x_interior", 24.0)
        mn_web = hogging_strength_kip_in(3.16, 21.5, 24.0, 20.0, 6.0, 60.0)          # four #8 on the web alone
        need = web_top_steel_for_hogging(mn_web, 21.5, 24.0, 20.0, 6.0, 60.0, layers)
        self.assertLess(need, 3.16)
        self.assertAlmostEqual(hogging_strength_kip_in(need, 21.5, 24.0, 20.0, 6.0, 60.0, layers) / mn_web, 1.0, places=3)
        self.assertEqual(web_top_steel_for_hogging(0.0, 21.5, 24.0, 20.0, 6.0, 60.0, layers), 0.0)


class MinimumSteelOnTheNominalDepth(unittest.TestCase):
    """V4 cases 0022 and 0087: two #7 on an 18x18 8 ksi beam passed the sizing's stacked-depth minimum (1.17 in2)
    and failed the detailing check's nominal-depth minimum (1.25 in2)."""

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for key, value in {
            'SLAB_THICKNESS_IN': 7.0, 'B_BEAM': 18.0, 'H_BEAM': 18.0, 'B_COL': 24.0, 'H_COL': 24.0,
            'NUM_BAY_X': 5, 'NUM_BAY_Y': 3, 'FC_BEAM_KSI': 8.0, 'FY_KSI': 60.0, 'BEAM_CLEAR_COVER_IN': 1.5,
            'BEAM_STIRRUP_BAR_SIZE': 4, 'AGGREGATE_MAX_SIZE_IN': 0.75, 'BEAM_BAR_SIZE': 7, 'BEAM_TOP_BARS': 2,
            'BEAM_BOT_BARS': 3, 'SLAB_REINFORCEMENT': None,
        }.items():
            self.stack.enter_context(mock.patch.object(sp, key, value))
        self.cfg = DesignConfig()
        self.cfg.rebar.bar_sizes_beam = [7]
        self.cfg.rebar.beam_n_range = (2, 4)

    def test_two_bars_below_the_nominal_minimum_are_not_offered(self):
        nominal = rd._beam_As_min(8.0, 60.0, 18.0, 18.0 - sp.longitudinal_cover_in("beam", 7))
        stacked = rd._beam_As_min(8.0, 60.0, 18.0, 18.0 - sp.beam_worst_centroid_offset_in(7, 2, 2))
        self.assertLess(stacked, 2 * 0.60)
        self.assertLess(2 * 0.60, nominal)
        candidates = rd._beam_candidates(0.0, 100.0, 0.0, 100.0, self.cfg)
        self.assertTrue(candidates)
        self.assertTrue(all(n_top * 0.60 >= nominal and n_bot * 0.60 >= nominal for _, n_top, n_bot in candidates))
        self.assertNotIn((7, 2, 3), candidates)
        self.assertIn((7, 3, 3), candidates)


if __name__ == '__main__':
    unittest.main()
