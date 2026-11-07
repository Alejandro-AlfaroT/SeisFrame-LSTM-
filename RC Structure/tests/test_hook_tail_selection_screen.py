"""Selection refuses a beam cage whose standard hook cannot fit the member depth, by the same rule the joint context
applies (2026-11-05). No solve."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp


class HookTailRule(unittest.TestCase):
    def test_numbers_of_the_rule(self):
        from Design.SMRF_Cage_Geometry import standard_hook_tail_fits as fits
        # No. 10: bend radius to the bar axis (10.16 + 1.27) / 2 = 5.715 in, extension 15.24 in, so the tail ends
        # 20.955 in beyond the row; with 1.5 in cover a 30 in beam admits a top row down to 30 - 1.5 - 0.635 - 20.955.
        self.assertTrue(fits(10, "top", 6.9, 30.0, 1.5))
        self.assertFalse(fits(10, "top", 6.92, 30.0, 1.5))
        self.assertFalse(fits(10, "top", 8.5, 30.0, 1.5))
        self.assertTrue(fits(9, "top", 8.5, 30.0, 1.5))
        self.assertTrue(fits(8, "top", 8.5, 30.0, 1.5))
        self.assertFalse(fits(8, "top", 8.5, 24.0, 1.5))
        self.assertEqual(fits(10, "bottom", 6.9, 30.0, 1.5), fits(10, "top", 6.9, 30.0, 1.5))
        self.assertFalse(fits(8, "top", 0.9, 30.0, 1.5))                 # the row itself is inside the cover

    def test_the_joint_context_and_the_helper_agree(self):
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg, Member_Bar_Layers as bar_layers
        from Design import SMRF_Transition_Context as tc
        from Design.Group_Capacity import joint_inventory
        from Design.SMRF_Cage_Geometry import standard_hook_tail_fits as fits
        compared, outcomes = 0, set()
        with gf.frame():
            base = gf.uniform_state()
            for size in (8, 10, 11):
                beams = {g: replace(d, bar_size=size, top_bars=4, bot_bars=4) for g, d in base.designs.items()
                         if base.groups[g]["member_type"] != "column"}
                state = mg.install(base.with_designs(beams))
                rows = bar_layers.arrangement()
                for entry in joint_inventory(state)["joints"][:6]:
                    context = tc._beam_context(entry, None, None)
                    for check in context["beam_geometry_checks"]:
                        if check["rule"] != "standard_hook_tail_fits_member_depth":
                            continue
                        design = state.designs[check["group"]]
                        offset = rows[check["group"]][check["face"]]["offsets_in"][check["row"]]
                        self.assertEqual(check["passes"], fits(design.bar_size, check["face"], offset, design.h_in,
                                                               sp.BEAM_CLEAR_COVER_IN), (size, check))
                        compared += 1
                        outcomes.add(check["passes"])
        self.assertGreater(compared, 0)
        self.assertEqual(outcomes, {True, False})                        # the fixture exercises both outcomes

    def test_selection_refuses_the_cage_with_its_own_code(self):
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg, Member_Bar_Layers as bar_layers
        from Design import Group_Selection as selection
        from Design.Config import DesignConfig
        from Design.SMRF_Cage_Geometry import standard_hook_tail_fits as fits
        self.assertEqual(selection.BEAM_REFUSALS["hook_tail_depth"][0], "hard_section")
        with gf.frame():
            state = mg.install(gf.uniform_state())
            gid = next(g for g, group in sorted(state.groups.items()) if group["member_type"] != "column")
            self.assertTrue(selection._has_terminating_end(state, gid))
            design = state.designs[gid]
            cfg = DesignConfig()
            # Without an installed slab mat layout the joint context is not evaluated, and the screen is off.
            self.assertIsNone((sp.SLAB_REINFORCEMENT or {}).get("layout"))
            for size in (6, 8, 9, 10, 11):
                result = selection.evaluate_beam_candidate(state, gid, replace(design, bar_size=size, top_bars=2, bot_bars=2),
                                                           (1.0, 1.0), cfg)
                self.assertNotEqual(result.get("reason_code"), "hook_tail_depth", size)
            sp.SLAB_REINFORCEMENT = {"layout": {"bar_size": 4, "outer_axis": "x", "layers": {
                f"{axis}_{face}": {"bar_size": 4, "bar_area_in2": .2, "spacing_in": 12.0,
                                   "effective_depth_in": 6. if face == "top" else 1., "axis": axis,
                                   "layer": "outer" if axis == "x" else "inner", "clear_cover_outer_mat_in": .75}
                for axis in ("x", "y") for face in ("top", "bottom")}}}
            bar_layers._CACHE.clear()
            seen = set()
            for size in (6, 8, 9, 10, 11):
                candidate = replace(design, bar_size=size, top_bars=2, bot_bars=2)
                result = selection.evaluate_beam_candidate(state, gid, candidate, (1.0, 1.0), cfg)
                with mg.installed(selection._trial_state(state, gid, candidate)):
                    rows = bar_layers.arrangement()[gid]
                fit = all(fits(size, face, offset, design.h_in, sp.BEAM_CLEAR_COVER_IN)
                          for face in ("top", "bottom") for offset in rows[face]["offsets_in"])
                if result.get("reason_code") == "hook_tail_depth":
                    self.assertFalse(fit, size)
                    self.assertEqual(result["reason_class"], "hard_section")
                    seen.add("refused")
                elif result["admissible"]:
                    self.assertTrue(fit, size)                           # nothing admissible fails the hook rule
                    seen.add("admitted")
            self.assertIn("refused", seen)


def setUpModule():
    # These tests verify the 3D physical detail route, which the computed common-action route keeps (2026-11-07).
    global _DETAIL_BASIS
    _DETAIL_BASIS = sp.DETAIL_BASIS
    sp.DETAIL_BASIS = "physical_3d_v2"


def tearDownModule():
    sp.DETAIL_BASIS = _DETAIL_BASIS


if __name__ == "__main__":
    unittest.main()
