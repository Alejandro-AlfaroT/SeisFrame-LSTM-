"""The repeated-work reductions of 2026-11-07 give the same numbers cold and warm, and never serve a stale entry. No solve."""
import unittest

import numpy as np

import Structure_Parameters as sp
from Design import SMRF_Beam_Axial as axial
from Design.SMRF_Uniaxial_Section import LayeredSection
from Model import Member_Groups as mg
from Model.Member_Groups import MemberDesign


def section(top=6.0, bottom=6.0, h=30.0, flange=96.0):
    regions = [(0.0, 8.0, flange, 5.0), (8.0, h, 16.0, 5.0)]
    layers = [(3.0, top, 1.0, 60.0), (h - 3.0, bottom, 1.0, 60.0), (2.0, 2.5, 0.31, 60.0), (6.0, 2.5, 0.31, 60.0)]
    return LayeredSection(h, regions, layers)


def reference_at_axial(sec, p, face="top", phi_mode="auto", fy_factor=1.0):
    """The bisection with the full state at every step, as it was before the axial-only path."""
    p = np.asarray(p, dtype=float)
    lo, hi = np.full(p.shape, 1e-10), np.full(p.shape, sec.h * 1000)
    for _ in range(64):
        mid = (lo + hi) / 2
        value = sec.state(mid, face, phi_mode, fy_factor)["p"]
        hi = np.where(value > p, mid, hi)
        lo = np.where(value <= p, mid, lo)
    return sec.state((lo + hi) / 2, face, phi_mode, fy_factor)


class AxialOnly(unittest.TestCase):
    AXIAL = np.array([-300.0, -50.0, -1.5, 0.0, 12.25, 90.0, 400.0])

    def test_same_bits_as_the_full_state_bisection(self):
        sec = section()
        for face in ("top", "bottom"):
            ours = sec.at_axial(self.AXIAL, face)
            reference = reference_at_axial(sec, self.AXIAL, face)
            for key in ("m", "p", "phi", "eps_t", "c"):
                self.assertTrue(np.array_equal(ours[key], reference[key]), f"{face} {key}")

    def test_axial_only_equals_state_p(self):
        sec = section()
        c = np.array([0.5, 2.0, 5.0, 9.0, 20.0, 60.0])
        for face in ("top", "bottom"):
            for mode, factor in (("auto", 1.0), ("nominal", 1.0), ("nominal", 1.25)):
                self.assertTrue(np.array_equal(sec._axial_only(c, face, mode, factor), sec.state(c, face, mode, factor)["p"]))


class IntervalCache(unittest.TestCase):
    def setUp(self):
        axial._INTERVALS.clear()
        axial._INTERVAL_BYTES[0] = 0

    def _family(self, sec):
        return {"i": sec, "j": sec, "span": sec}

    def test_cold_and_warm_are_equal(self):
        p = np.array([-20.0, 0.0, 15.0])
        cold = axial.capacity_at_axial(self._family(section()), p)
        self.assertEqual(len(axial._INTERVALS), 1)
        warm = axial.capacity_at_axial(self._family(section()), p)
        self.assertEqual(len(axial._INTERVALS), 1)
        for key in ("positive_kip_in", "negative_kip_in", "domain_failure", "tension_controlled"):
            self.assertTrue(np.array_equal(cold[key], warm[key]), key)
        for sign in ("positive", "negative"):
            self.assertTrue(np.array_equal(cold["phi"][sign], warm["phi"][sign]))

    def test_changed_steel_is_not_served_stale(self):
        p = np.array([0.0])
        first = axial.capacity_at_axial(self._family(section(top=6.0)), p)
        second = axial.capacity_at_axial(self._family(section(top=8.0)), p)
        self.assertEqual(len(axial._INTERVALS), 2)
        self.assertGreater(second["negative_kip_in"][0], first["negative_kip_in"][0])

    def test_changed_axial_is_not_served_stale(self):
        first = axial.capacity_at_axial(self._family(section()), np.array([0.0]))
        second = axial.capacity_at_axial(self._family(section()), np.array([-100.0]))
        self.assertEqual(len(axial._INTERVALS), 2)
        self.assertNotEqual(first["positive_kip_in"][0], second["positive_kip_in"][0])

    def test_changed_depth_and_flange_are_not_served_stale(self):
        p = np.array([0.0])
        axial.capacity_at_axial(self._family(section()), p)
        axial.capacity_at_axial(self._family(section(h=32.0)), p)
        axial.capacity_at_axial(self._family(section(flange=80.0)), p)
        self.assertEqual(len(axial._INTERVALS), 3)

    def test_the_limit_clears_the_cache(self):
        limit = axial.INTERVAL_CACHE_LIMIT_BYTES
        try:
            axial.INTERVAL_CACHE_LIMIT_BYTES = 1
            axial.capacity_at_axial(self._family(section()), np.array([0.0]))
            axial.capacity_at_axial(self._family(section(top=7.0)), np.array([0.0]))
            self.assertEqual(len(axial._INTERVALS), 1)
        finally:
            axial.INTERVAL_CACHE_LIMIT_BYTES = limit


class DerivedStates(unittest.TestCase):
    def _state(self):
        column = MemberDesign("column", 30.0, 30.0, 6.0, 9, 4, 4, 2, 4, 3, 4.0, (3, 3))
        beam = MemberDesign("beam_x", 16.0, 30.0, 5.0, 9, 4, 4, 0, 4, 2, 4.0)
        return mg.GroupedDesign.uniform(3, 3, 6, column, beam, policy_id=mg.SYSTEM_POLICY_ID)

    def test_derived_identity_equals_a_fresh_build(self):
        from dataclasses import replace
        state = self._state()
        gid = next(g for g, d in state.designs.items() if not d.is_column)
        design = replace(state.designs[gid], top_bars=5)
        ties = state.ties()
        tie = next((name for name, groups in ties.items() if gid in groups), None)
        derived = state.with_tied_designs({tie: design}) if tie else state.with_designs({gid: design})
        fresh = mg.GroupedDesign(state.assignments, derived.designs)
        self.assertEqual(derived.identity(), fresh.identity())
        self.assertNotEqual(derived.identity(), state.identity())

    def test_derived_member_carries_the_new_design(self):
        from dataclasses import replace
        state = self._state()
        tag = next(iter(state._members))
        before = state.member(tag).design
        gid = state.member(tag).group_id
        tie = state.tie_of(gid)
        changed = replace(before, top_bars=before.top_bars + 1)
        derived = state.with_tied_designs({tie: changed}) if tie else state.with_designs({gid: changed})
        self.assertEqual(derived.member(tag).design.top_bars, before.top_bars + 1)
        self.assertEqual(state.member(tag).design.top_bars, before.top_bars)

    def test_a_bad_design_is_still_refused(self):
        from dataclasses import replace
        state = self._state()
        gid = next(iter(state.designs))
        with self.assertRaises(mg.GroupedStateError):
            state.with_designs({gid: replace(state.designs[gid], top_bars=0)})

    def test_foreign_assignments_are_still_validated(self):
        state = self._state()
        broken = {**state.assignments, "counts": {"num_bay_x": 9, "num_bay_y": 3, "num_floor": 6}}
        with self.assertRaises(mg.GroupedStateError):
            mg.GroupedDesign(broken, state.designs)

    def test_parent_with_other_assignments_is_not_trusted(self):
        state = self._state()
        broken = {**state.assignments, "counts": {"num_bay_x": 9, "num_bay_y": 3, "num_floor": 6}}
        with self.assertRaises(mg.GroupedStateError):
            mg.GroupedDesign(broken, state.designs, _parent=state)


class Crossing(unittest.TestCase):
    def test_orthogonal_layers_may_touch_on_the_conventional_basis(self):
        from Model import Member_Bar_Layers as layers
        a = MemberDesign("beam_x", 16.0, 30.0, 5.0, 9, 4, 4, 0, 4, 2, 4.0)
        b = MemberDesign("beam_y", 16.0, 30.0, 5.0, 8, 4, 4, 0, 4, 2, 4.0)
        previous = sp.DETAIL_BASIS
        try:
            sp.DETAIL_BASIS = "conventional_detailing_v1"
            self.assertAlmostEqual(layers.crossing_pitch_in(a, b), sp.snap_up_to_grid(0.5 * (1.128 + 1.0)))
            sp.DETAIL_BASIS = "physical_3d_v2"
            self.assertAlmostEqual(layers.crossing_pitch_in(a, b), layers.pitch_in(a, b))
        finally:
            sp.DETAIL_BASIS = previous

    def test_parallel_layers_keep_their_clear_spacing(self):
        from Model import Member_Bar_Layers as layers
        a = MemberDesign("beam_x", 16.0, 30.0, 5.0, 9, 4, 4, 0, 4, 2, 4.0)
        self.assertGreaterEqual(layers.pitch_in(a, a), 1.128 + 1.128 - 1e-9)


if __name__ == "__main__":
    unittest.main()
