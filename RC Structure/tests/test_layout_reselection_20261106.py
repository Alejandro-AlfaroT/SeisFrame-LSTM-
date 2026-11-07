"""After a cage change the slab mat layout is chosen again from the original catalog, never thinner than the installed
one; and a saved refused story strength replays by asking the adapter again (2026-11-06). No solve."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp

NAMES = ("x_top", "x_bottom", "y_top", "y_bottom")


def candidate(spacings):
    layers = {n: {"spacing_in": float(s), "area_in2_per_ft": 0.2 * 12.0 / s} for n, s in zip(NAMES, spacings)}
    layout = {"bar_size": 4, "layers": layers}
    ident = "/".join(f"{s:g}" for s in spacings)
    return {"candidate_id": ident, "layout": layout, "layout_sha256": "sha-" + ident}


CATALOG = {"enumerated_count": 6, "candidates": [candidate(s) for s in
           ((12, 12, 12, 12), (12, 12, 10, 10), (10, 10, 12, 12), (10, 10, 10, 10), (10, 10, 9, 9), (9, 9, 6, 6))]}


class State:
    def identity(self):
        return "state-after-cage-change"


class Reselection(unittest.TestCase):
    """The helper is driven with stand-ins for the placement, the row audit and the inventory: its own logic is the
    order of the walk, the not-weaker rule, the lane-clear retry and what it leaves installed."""

    def run_helper(self, current, placement_of, inventory_fails=(), reason="Authoritative installed beam straight intervals are unresolved",
                   plain_error=False, **options):
        from Design import SMRF_Computed_Coordination as coordination
        from Design import SMRF_Joint_Mat_Physical as mats, SMRF_Physical_Floor_Materials as materials
        tried, built = [], []

        def placement(_state):
            layout = sp.SLAB_REINFORCEMENT["layout"]
            ident = "/".join(f"{layout['layers'][n]['spacing_in']:g}" for n in NAMES)
            tried.append(ident)
            return placement_of(ident, layout)

        def inventory(_state):
            layout = sp.SLAB_REINFORCEMENT["layout"]
            ident = "/".join(f"{layout['layers'][n]['spacing_in']:g}" for n in NAMES)
            built.append(ident)
            if ident in inventory_fails:
                if plain_error:
                    raise ValueError("Authoritative beam intervals must cover the exact actual cage lane inventory")
                raise materials.PhysicalInventoryUnresolved(reason, {"unresolved_member_count": 3, "members": [1, 2, 3]})
            return {"identity_sha256": "inv"}
        saved = sp.SLAB_REINFORCEMENT
        try:
            sp.SLAB_REINFORCEMENT = {"layout": current["layout"], "accepted": True, "stage": "installed"}
            with mock.patch.object(mats, "global_mat_placement", placement), \
                    mock.patch.object(materials, "installed_inventory", inventory), \
                    mock.patch.object(coordination.bar_layers, "row_audit", lambda state=None: {"valid": True}), \
                    mock.patch.object(coordination.bar_layers, "raise_arrangement_errors", lambda audit: None):
                try:
                    chosen, evidence = coordination.reselect_layout(State(), CATALOG, current["layout"], **options)
                except ValueError as exc:
                    if not plain_error:
                        raise
                    chosen, evidence = None, {"raised": str(exc)}
            return chosen, evidence, tried, built, sp.SLAB_REINFORCEMENT
        finally:
            sp.SLAB_REINFORCEMENT = saved

    def test_the_lane_clear_walk_follows_only_the_refusal_it_can_repair(self):
        current = CATALOG["candidates"][3]

        def placement(ident, layout):
            return self.layers({n: ("pass" if ident == "9/9/6/6" else "columns_only") for n in NAMES})
        chosen, evidence, tried, built, installed = self.run_helper(
            current, placement, inventory_fails={"10/10/10/10"}, reason="Actual beam cage lanes are unresolved")
        self.assertIsNone(chosen)                                   # unresolved lanes: no layout can help
        self.assertEqual(len(evidence["walks"]), 1)
        self.assertEqual(tried, ["10/10/10/10"])
        self.assertIn("members", evidence["walks"][0]["inventory"]["evidence"])   # the ending walk keeps the whole refusal
        self.assertEqual(installed["stage"], "installed")

    def test_lane_clear_only_skips_the_plain_walk(self):
        current = CATALOG["candidates"][3]

        def placement(ident, layout):
            return self.layers({n: ("pass" if ident == "9/9/6/6" else "columns_only") for n in NAMES})
        chosen, evidence, tried, built, installed = self.run_helper(current, placement, lane_clear_only=True)
        self.assertEqual(chosen["candidate_id"], "9/9/6/6")
        self.assertEqual([w["lane_clear_only"] for w in evidence["walks"]], [True])
        self.assertEqual(built, ["9/9/6/6"])

    def test_an_error_that_is_not_a_refusal_leaves_the_earlier_layout_installed(self):
        current = CATALOG["candidates"][3]
        chosen, evidence, tried, built, installed = self.run_helper(
            current, lambda ident, layout: self.layers({n: "pass" for n in NAMES}), inventory_fails={"10/10/10/10"},
            plain_error=True)
        self.assertIn("raised", evidence)
        self.assertEqual((installed["layout"], installed["stage"]), (current["layout"], "installed"))

    @staticmethod
    def layers(status):
        return {"status": "pass" if all(v != "fail" for v in status.values()) else "fail",
                "layers": {n: {"status": "fail" if v == "fail" else "pass",
                               "phase_rule": "columns_only" if v == "columns_only" else "columns_and_swept_parallel_lanes"}
                           for n, v in status.items()}}

    def test_takes_the_first_not_weaker_layout_that_places(self):
        current = CATALOG["candidates"][1]                         # 12/12/10/10 installed; its y layers no longer place

        def placement(ident, layout):
            return self.layers({n: ("fail" if layout["layers"][n]["spacing_in"] == 10 and n[0] == "y" else "pass") for n in NAMES})
        chosen, evidence, tried, built, installed = self.run_helper(current, placement)
        # Not weaker than 12/12/10/10 in any layer: 12/12/10/10 itself, 10/10/10/10, 10/10/9/9, 9/9/6/6.
        self.assertEqual(evidence["eligible_not_weaker_count"], 4)
        self.assertEqual(chosen["candidate_id"], "10/10/9/9")
        self.assertEqual(tried, ["12/12/10/10", "10/10/9/9"])      # 10/10/10/10 passed over: its y layers are known to fail
        self.assertEqual(built, ["10/10/9/9"])
        self.assertEqual(installed["layout"], chosen["layout"])
        self.assertEqual(installed["stage"], "computed_cage_change_layout_reselection")
        self.assertEqual(evidence["walks"][0]["inventory"], {"status": "established"})
        self.assertEqual(evidence["selected_layout_sha256"], "sha-10/10/9/9")
        self.assertNotIn("12/12/12/12", tried)                     # a thinner mat is never proposed

    def test_lane_clear_walk_follows_a_refused_inventory_of_a_columns_only_layout(self):
        current = CATALOG["candidates"][3]                         # 10/10/10/10 installed

        def placement(ident, layout):
            return self.layers({n: ("pass" if layout["layers"][n]["spacing_in"] in (9, 6) and ident == "9/9/6/6" else
                                    "columns_only") for n in NAMES})
        chosen, evidence, tried, built, installed = self.run_helper(current, placement, inventory_fails={"10/10/10/10"})
        self.assertEqual(chosen["candidate_id"], "9/9/6/6")
        self.assertEqual([w["lane_clear_only"] for w in evidence["walks"]], [False, True])
        self.assertEqual(evidence["walks"][0]["inventory"]["status"], "unresolved")
        self.assertNotIn("members", evidence["walks"][0]["inventory"]["evidence"])
        self.assertEqual(built, ["10/10/10/10", "9/9/6/6"])
        self.assertEqual(installed["layout"], chosen["layout"])

    def test_nothing_established_leaves_the_earlier_layout_installed(self):
        current = CATALOG["candidates"][3]
        chosen, evidence, tried, built, installed = self.run_helper(
            current, lambda ident, layout: self.layers({n: "fail" for n in NAMES}))
        self.assertIsNone(chosen)
        self.assertIsNone(evidence["selected_candidate_id"])
        self.assertEqual(built, [])
        self.assertEqual((installed["layout"], installed["stage"]), (current["layout"], "installed"))
        self.assertEqual(len(evidence["walks"]), 1)

    def test_a_refused_inventory_without_a_columns_only_layer_ends_the_reselection(self):
        current = CATALOG["candidates"][3]
        chosen, evidence, tried, built, installed = self.run_helper(
            current, lambda ident, layout: self.layers({n: "pass" for n in NAMES}), inventory_fails={"10/10/10/10"})
        self.assertIsNone(chosen)
        self.assertEqual(len(evidence["walks"]), 1)
        self.assertEqual(installed["stage"], "installed")


class CommonLanesInTheThreadingVerdict(unittest.TestCase):
    """A cage whose common lane set is established to fail does not thread, where a slab layout is installed."""

    def price(self, lane_status, layout=True):
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg, Member_Bar_Layers as bar_layers
        from Design import Group_Selection as selection, SMRF_Splice_Physical as splice
        from Design.Config import DesignConfig
        with gf.frame():
            state = mg.install(gf.uniform_state())
            if layout:
                sp.SLAB_REINFORCEMENT = {"layout": {"bar_size": 4, "outer_axis": "x", "layers": {
                    f"{axis}_{face}": {"bar_size": 4, "bar_area_in2": .2, "spacing_in": 12.0,
                                       "effective_depth_in": 6. if face == "top" else 1., "axis": axis,
                                       "layer": "outer" if axis == "x" else "inner", "clear_cover_outer_mat_in": .75}
                    for axis in ("x", "y") for face in ("top", "bottom")}}}
                bar_layers._CACHE.clear()
            gid = next(g for g, group in sorted(state.groups.items()) if group["member_type"] != "column")
            real = splice.group_beam_lanes

            def lanes(state_=None):
                return {g: {**r, "status": lane_status, "pool_from_left_in": [], "requested_maximum": 2} for g, r in real(state_).items()}
            # The first cage every other rule admits, so the verdict under test is reached.
            with mock.patch.object(splice, "group_beam_lanes", lanes):
                for size in (6, 7, 8):
                    for count in (3, 4, 5, 6):
                        design = replace(state.designs[gid], bar_size=size, top_bars=count, bot_bars=count)
                        result = selection.evaluate_beam_candidate(state, gid, design, (1.0, 1.0), DesignConfig())
                        if result["admissible"]:
                            return result, selection
            self.fail("the fixture admits none of the cages tried")

    def test_established_failure_does_not_thread_and_says_why(self):
        result, selection = self.price("fail")
        self.assertTrue(result["admissible"])                       # it stays admissible, as any nonthreading cage does
        self.assertFalse(result["threads"])
        self.assertEqual(result["lanes"]["blocked_by"], "common_lane")
        self.assertIn("common_lane", selection.THREADING_BLOCKS)
        self.assertEqual((result["lanes"]["common_lanes_found"], result["lanes"]["common_lanes_requested"]), (0, 2))

    def test_an_inventory_not_evaluated_or_passing_changes_nothing(self):
        for status in ("not_evaluated", "pass"):
            result, _selection = self.price(status)
            self.assertTrue(result["admissible"], status)
            self.assertTrue(result["threads"], status)
            self.assertIsNone(result["lanes"]["blocked_by"], status)

    def test_without_a_slab_layout_the_physical_rule_is_not_read(self):
        result, _selection = self.price("fail", layout=False)
        self.assertTrue(result["threads"])


class RefusedStoryStrengthReplay(unittest.TestCase):
    REFUSAL = {"reason": "Joint nominal strength lacks topology/confinement evidence",
               "joint_id": "joint_7", "joint_type": "T1", "axis": "x"}

    def replay(self, behaviour, labels="current"):
        from Design import Grouped_Record as record, SMRF_Story_Adapter as adapter
        saved = {"premise_refusal": dict(self.REFUSAL)}
        if labels == "current":
            saved.update(method=adapter.METHOD, adapter_method=adapter.ADAPTER_METHOD)
        elif labels == "stale":
            saved.update(method=adapter.METHOD, adapter_method="an_earlier_adapter_method")
        with mock.patch.object(adapter, "calculate", behaviour):
            return record.replay_story_premise_refusal(saved, object(), [], [], {}, [])

    def test_a_refusal_saved_under_another_or_no_method_label_does_not_reproduce(self):
        from Design.SMRF_Story_Adapter import StoryStrengthPremiseUnavailable

        def refuse(*a, **k):
            raise StoryStrengthPremiseUnavailable("Joint nominal strength lacks topology/confinement evidence",
                                                  joint_id="joint_7", joint_type="T1", axis="x")
        self.assertEqual(self.replay(refuse), [])
        self.assertEqual(len(self.replay(refuse, labels="stale")), 1)
        self.assertEqual(len(self.replay(refuse, labels="none")), 2)

    def test_the_same_refusal_reproduces(self):
        from Design.SMRF_Story_Adapter import StoryStrengthPremiseUnavailable

        def refuse(*a, **k):
            raise StoryStrengthPremiseUnavailable("Joint nominal strength lacks topology/confinement evidence",
                                                  joint_id="joint_7", joint_type="T1", axis="x")
        self.assertEqual(self.replay(refuse), [])

    def test_another_reason_or_joint_is_a_difference(self):
        from Design.SMRF_Story_Adapter import StoryStrengthPremiseUnavailable

        def other_joint(*a, **k):
            raise StoryStrengthPremiseUnavailable("Joint nominal strength lacks topology/confinement evidence",
                                                  joint_id="joint_9", joint_type="T1", axis="x")

        def other_reason(*a, **k):
            raise StoryStrengthPremiseUnavailable("No physical positive nominal transverse-steel strength", member_tag=4, axis="x")
        self.assertTrue(self.replay(other_joint))
        self.assertTrue(self.replay(other_reason))

    def test_a_design_that_is_no_longer_refused_does_not_reproduce(self):
        found = self.replay(lambda *a, **k: {"stories": []})
        self.assertEqual(len(found), 1)
        self.assertIn("does not refuse", found[0])

    def test_binding_errors_are_not_swallowed(self):
        def broken(*a, **k):
            raise ValueError("Story strength action inventory: differs")
        with self.assertRaises(ValueError):
            self.replay(broken)


def setUpModule():
    # These tests verify the 3D physical detail route, which the computed common-action route keeps (2026-11-07).
    global _DETAIL_BASIS
    _DETAIL_BASIS = sp.DETAIL_BASIS
    sp.DETAIL_BASIS = "physical_3d_v2"


def tearDownModule():
    sp.DETAIL_BASIS = _DETAIL_BASIS


if __name__ == "__main__":
    unittest.main()
