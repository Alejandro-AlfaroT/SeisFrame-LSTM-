"""Corrections from the review of 2026-11-05: a refused story premise on a one-story frame, the refusal evidence
counts, the same-lane hook condition, the lane-clear layout walk and the torsion stream readback. No solve."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp


class StoryPremiseOnAnyNumberOfStories(unittest.TestCase):
    def rows(self, stories):
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg
        from Design import Grouped_Design as gd, SMRF_Story_Adapter as adapter
        from Design.Config import DesignConfig
        with gf.frame():
            state = mg.install(gf.uniform_state())

            def refuse(*a, **k):
                raise adapter.StoryStrengthPremiseUnavailable("Joint nominal strength lacks topology/confinement evidence",
                                                              joint_id="joint_1")
            with mock.patch.object(adapter, "calculate", refuse):
                record = gd.story_strength_distribution(DesignConfig(), state=state, actions=[], expected_ids=[],
                                                        capacity={}, gravity_actions=[])
            with mock.patch.object(sp, "NUM_FLOOR", stories):
                ledger = {"floors": [{"floor": k, "seismic_weight_kip": 100.0} for k in range(1, stories + 1)]}
                vertical = gd.vertical_regularity(None, None, record, ledger)
                checks = {c["id"]: c for c in gd.regularity_checks(vertical, "D", strength_model_verified=False)}
            return record, checks["demands.vertical_strength_regularity"]

    def test_the_row_is_unknown_with_one_story_and_with_several(self):
        for stories in (1, 2, 4):
            record, row = self.rows(stories)
            self.assertEqual(row["status"], "not_evaluated", stories)
            self.assertNotIn("review_dependency", row.get("details") or {}, stories)
            self.assertIn("not calculated", (row["details"].get("reason") or ""), stories)

    def test_the_refusal_record_names_its_method_and_is_not_a_verified_strength(self):
        from Design import Design_Driver as driver
        from Design.Config import DesignConfig
        from Design.SMRF_Story_Adapter import ADAPTER_METHOD, METHOD, computed_verification_valid
        record, _row = self.rows(2)
        self.assertEqual((record["method"], record["adapter_method"]), (METHOD, ADAPTER_METHOD))
        self.assertFalse(computed_verification_valid(record))
        self.assertFalse(driver._strength_model_verified(DesignConfig(), record))


class RefusalEvidence(unittest.TestCase):
    def test_one_of_every_rule_is_kept_before_any_repeat(self):
        from Design.SMRF_Physical_Floor_Materials import _brief_failures
        failures = [{"rule": "a", "k": i} for i in range(8)] + [{"rule": "b", "collisions": list(range(9))}]
        brief = _brief_failures(failures)
        self.assertEqual(len(brief), 8)
        self.assertEqual([c["rule"] for c in brief[:2]], ["a", "b"])        # the late rule is not cut off
        self.assertEqual(len(brief[1]["collisions"]), 4)
        self.assertEqual(_brief_failures([]), [])


class SameLaneHooks(unittest.TestCase):
    def test_required_separation(self):
        from Design.SMRF_Cage_Geometry import same_lane_hook_separation_in as need, same_lane_hooks_clear as clear
        self.assertAlmostEqual(need(7), 14.4375, places=9)         # (5.25 + 0.875) / 2 + 10.5 + 0.875
        self.assertAlmostEqual(need(8), 16.5, places=9)            # (6 + 1) / 2 + 12 + 1
        self.assertAlmostEqual(need(9), 19.74, places=9)           # (9.024 + 1.128) / 2 + 13.536 + 1.128
        self.assertTrue(clear(8, 8.5, 3.0, 30.0))                  # 18.5 in apart
        self.assertFalse(clear(9, 8.5, 3.0, 30.0))                 # 18.5 < 19.74
        self.assertFalse(clear(8, 10.5, 8.0, 30.0))                # second rows, 11.5 in apart
        self.assertTrue(clear(8, 5.0, 3.0, 24.5))                  # exactly 16.5 in

    def test_selection_refuses_two_rows_that_share_a_lane_and_keeps_one_row(self):
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg, Member_Bar_Layers as bar_layers
        from Design import Group_Selection as selection
        from Design.Config import DesignConfig
        from Design.SMRF_Cage_Geometry import same_lane_hooks_clear as clear, standard_hook_tail_fits as fits
        from Design.SMRF_Splice_Physical import group_beam_lanes
        self.assertEqual(selection.BEAM_REFUSALS["hook_rows_same_lane"][0], "construction_policy")
        with gf.frame():
            state = mg.install(gf.uniform_state())
            sp.SLAB_REINFORCEMENT = {"layout": {"bar_size": 4, "outer_axis": "x", "layers": {
                f"{axis}_{face}": {"bar_size": 4, "bar_area_in2": .2, "spacing_in": 12.0,
                                   "effective_depth_in": 6. if face == "top" else 1., "axis": axis,
                                   "layer": "outer" if axis == "x" else "inner", "clear_cover_outer_mat_in": .75}
                for axis in ("x", "y") for face in ("top", "bottom")}}}
            bar_layers._CACHE.clear()
            gid = next(g for g, group in sorted(state.groups.items()) if group["member_type"] != "column")
            design, cfg, seen = state.designs[gid], DesignConfig(), set()
            for size in (6, 7, 8):
                for count in (2, 3, 4, 5, 6, 8):
                    candidate = replace(design, bar_size=size, top_bars=count, bot_bars=count)
                    result = selection.evaluate_beam_candidate(state, gid, candidate, (1.0, 1.0), cfg)
                    with mg.installed(selection._trial_state(state, gid, candidate)):
                        try:
                            rows = bar_layers.arrangement()[gid]
                        except mg.GroupedStateError:
                            continue
                        lanes = group_beam_lanes()[gid]
                    if lanes.get("status") != "pass":
                        continue
                    by = {}
                    for p in lanes["coordinates"]:
                        by.setdefault((p["source_face"], p["row"]), set()).add(round(p["x_in"], 6))
                    crossing = any(by.get(("top", i), set()) & by.get(("bottom", j), set()) and not clear(size, a, b, design.h_in)
                                   for i, a in enumerate(rows["top"]["offsets_in"]) for j, b in enumerate(rows["bottom"]["offsets_in"]))
                    tails = all(fits(size, face, o, design.h_in, sp.BEAM_CLEAR_COVER_IN)
                                for face in ("top", "bottom") for o in rows[face]["offsets_in"])
                    if result.get("reason_code") == "hook_rows_same_lane":
                        self.assertTrue(crossing and tails, (size, count))
                        seen.add("refused")
                    elif result["admissible"]:
                        self.assertFalse(crossing, (size, count))        # nothing admissible has crossing hooks
                        seen.add("admitted")
            self.assertEqual(seen, {"refused", "admitted"})


class SameLaneHooksOfUnequalRows(unittest.TestCase):
    def test_every_top_row_is_checked_against_every_bottom_row(self):
        """Cages with different top and bottom counts have rows of different index that share a lane: a rule that
        paired only rows of equal index would admit them (review of 2026-11-06)."""
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg, Member_Bar_Layers as bar_layers
        from Design import Group_Selection as selection
        from Design.Config import DesignConfig
        from Design.SMRF_Cage_Geometry import same_lane_hooks_clear as clear, standard_hook_tail_fits as fits
        from Design.SMRF_Splice_Physical import group_beam_lanes
        with gf.frame():
            state = mg.install(gf.uniform_state())
            sp.SLAB_REINFORCEMENT = {"layout": {"bar_size": 4, "outer_axis": "x", "layers": {
                f"{axis}_{face}": {"bar_size": 4, "bar_area_in2": .2, "spacing_in": 12.0,
                                   "effective_depth_in": 6. if face == "top" else 1., "axis": axis,
                                   "layer": "outer" if axis == "x" else "inner", "clear_cover_outer_mat_in": .75}
                for axis in ("x", "y") for face in ("top", "bottom")}}}
            bar_layers._CACHE.clear()
            gid = next(g for g, group in sorted(state.groups.items()) if group["member_type"] != "column")
            design, cfg, seen = state.designs[gid], DesignConfig(), set()
            for size in (5, 6, 7):
                for top in (2, 3, 4):
                    for bottom in (2, 3, 4, 5, 6, 8):
                        if top == bottom:
                            continue
                        candidate = replace(design, bar_size=size, top_bars=top, bot_bars=bottom)
                        result = selection.evaluate_beam_candidate(state, gid, candidate, (1.0, 1.0), cfg)
                        with mg.installed(selection._trial_state(state, gid, candidate)):
                            try:
                                rows = bar_layers.arrangement()[gid]
                            except mg.GroupedStateError:
                                continue
                            lanes = group_beam_lanes()[gid]
                        if lanes.get("status") != "pass":
                            continue
                        by = {}
                        for p in lanes["coordinates"]:
                            by.setdefault((p["source_face"], p["row"]), set()).add(round(p["x_in"], 6))
                        crossing = [(i, j) for i, a in enumerate(rows["top"]["offsets_in"]) for j, b in enumerate(rows["bottom"]["offsets_in"])
                                    if by.get(("top", i), set()) & by.get(("bottom", j), set()) and not clear(size, a, b, design.h_in)]
                        tails = all(fits(size, face, o, design.h_in, sp.BEAM_CLEAR_COVER_IN)
                                    for face in ("top", "bottom") for o in rows[face]["offsets_in"])
                        if result["admissible"]:
                            self.assertEqual(crossing, [], (size, top, bottom))
                        if result.get("reason_code") == "hook_rows_same_lane":
                            self.assertTrue(crossing and tails, (size, top, bottom))
                            if all(i != j for i, j in crossing):
                                seen.add("only_rows_of_different_index_cross")
                        elif crossing and tails and result.get("reason_code") is None:
                            self.fail(f"crossing hooks admitted: {(size, top, bottom, crossing)}")
            self.assertIn("only_rows_of_different_index_cross", seen)


class LaneClearLayoutWalk(unittest.TestCase):
    NAMES = ("x_top", "x_bottom", "y_top", "y_bottom")

    def items(self):
        rows = [(11, 11, 11, 11), (10, 10, 10, 10), (9, 9, 10, 10), (9, 9, 9, 9), (9, 9, 6, 6)]
        return [{"candidate_id": "/".join(map(str, r)),
                 "layout": {"bar_size": 4, "layers": {n: {"spacing_in": float(s)} for n, s in zip(self.NAMES, r)}}} for r in rows]

    def place(self, log):
        """11 in is unplaceable; x layers are lane-clear at 9 in, y layers only at 6 in; everything else columns only."""
        def run(item):
            log.append(item["candidate_id"])
            status = {}
            for n, r in item["layout"]["layers"].items():
                s = r["spacing_in"]
                status[n] = "fail" if s == 11 else "pass" if (n[0] == "x" and s == 9) or (n[0] == "y" and s == 6) else "columns_only"
            return ("fail" if "fail" in status.values() else "pass"), status, {"candidate_id": item["candidate_id"]}
        return run

    def test_first_walk_takes_the_first_placeable_layout_as_before(self):
        from Design.SMRF_Computed_Coordination import bounded_layout_proposals as walk
        log = []
        chosen, _a, passed_over, searches, unusable = walk(self.items(), self.place(log))
        self.assertEqual((chosen["candidate_id"], searches, passed_over), ("10/10/10/10", 2, 0))
        self.assertEqual({k[2] for k in unusable}, {11.0})               # only unplaceable layers are remembered

    def test_lane_clear_walk_passes_over_columns_only_layers_and_carries_known_layers(self):
        from Design.SMRF_Computed_Coordination import bounded_layout_proposals as walk
        log = []
        _c, _a, _p, _s, known = walk(self.items(), self.place(log))
        log.clear()
        chosen, _a, passed_over, searches, unusable = walk(self.items(), self.place(log), lane_clear_only=True, known=known)
        self.assertEqual(chosen["candidate_id"], "9/9/6/6")
        self.assertEqual(log, ["10/10/10/10", "9/9/9/9", "9/9/6/6"])     # 11 in known; 9/9/10/10 passed over after 10 in
        self.assertEqual((searches, passed_over), (3, 2))
        self.assertIn(("y_top", 4, 9.0, 4), unusable)

    def test_lane_clear_walk_returns_nothing_when_no_layout_is_lane_clear(self):
        from Design.SMRF_Computed_Coordination import bounded_layout_proposals as walk
        chosen, attempts, _p, searches, _u = walk(self.items()[:4], self.place([]), lane_clear_only=True)
        self.assertIsNone(chosen)
        self.assertEqual(searches, len(attempts))


class TorsionStreamReadback(unittest.TestCase):
    def test_a_package_of_the_current_stream_method_is_read_back(self):
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg
        from Design import SMRF_Beam_Torsion_Adapter as adapter
        from Design.SMRF_Common_Torsion_Stream import METHOD
        with gf.frame():
            state = mg.install(gf.uniform_state())
            current = adapter.evidence(state, ["c1"], {}, {"method": METHOD, "source_id": "s"}, common_actions=None)
            self.assertEqual(current["method"], "all_actual_beam_case_cut_torsion_v2_common_readback")
            self.assertEqual(current["status"], "not_evaluated")          # no common actions: unknown, with the reason
            self.assertTrue(current["inventory_problems"])
            self.assertTrue(all(row["status"] == "not_evaluated" for row in current["checks"]))


def setUpModule():
    # These tests verify the 3D physical detail route, which the computed common-action route keeps (2026-11-07).
    global _DETAIL_BASIS
    _DETAIL_BASIS = sp.DETAIL_BASIS
    sp.DETAIL_BASIS = "physical_3d_v2"


def tearDownModule():
    sp.DETAIL_BASIS = _DETAIL_BASIS


if __name__ == "__main__":
    unittest.main()
