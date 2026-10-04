"""Physical clashes, candidate rollback, and analysis rebuilding for layout repair."""
import copy
import os
import sys
import unittest
from contextlib import ExitStack
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Structure_Parameters as sp
from Design import Design_Driver as driver
from Design.Config import DesignConfig
from Design.SMRF_Cage_Geometry import joint_assembly
from Design.SMRF_Beam_Slab_Strength import beam_slab_strengths


class LayoutRepair(unittest.TestCase):
    def setUp(self):
        self.seed = driver._capture_state()
        self.cfg = DesignConfig.from_structure_parameters()
        self.actions = [{"id": "one", "members": []}, {"id": "two", "members": []}]
        self.worst = {"beam": .8, "column": .5}
        self.joint = {"evaluated": True, "all_pass": True, "counts": {"pass": 2}}
        self.capacity = {"accepted": False, "bar_threading": {"slab_clashes": ["overlap"]},
                         "checks": [{"id": "beam.bar_stacking_clear_of_slab_mats", "status": "fail"}]}
        sp.BEAM_BAR_STACKING, sp.BEAM_BAR_LAYER_ORDER = "y_over_x", "interleaved"

    def tearDown(self):
        driver._restore_state(self.seed)

    def mocked_checks(self, stack, dcr=.85, joint=None):
        stack.enter_context(patch.object(driver, "_capacity_state", return_value={}))
        stack.enter_context(patch("Design.SMRF_Capacity_Design.design_bar_threading",
                                  return_value={"passes": True, "slab_clashes": []}))
        stack.enter_context(patch.object(driver, "_capacity_design", return_value={"accepted": True, "checks": []}))
        check = stack.enter_context(patch.object(driver, "_governing_dcrs", return_value=(.55, dcr, {})))
        stack.enter_context(patch.object(driver, "_joint_scwb_state", return_value=joint or self.joint))
        stack.enter_context(patch.object(driver, "_beam_flexure_demand", return_value=100.))
        return check

    def test_acceptance_rechecks_every_action_and_installs_rows(self):
        with ExitStack() as stack:
            check = self.mocked_checks(stack)
            worst, joint, capacity, report = driver._repair_bar_stacking(
                self.cfg, self.actions, self.worst, self.joint, self.capacity)
        self.assertEqual(check.call_count, len(self.actions))
        self.assertEqual(worst["beam"], .85)
        self.assertTrue(capacity["accepted"])
        self.assertEqual(report["status"], "resolved")
        self.assertEqual(sp.BEAM_BAR_STACKING, "x_over_y")

    def test_geometric_fit_cannot_bypass_strength_or_incomplete_joint_evidence(self):
        for dcr, joint in [(1.2, self.joint), (.8, {**self.joint, "counts": {"not_evaluated": 1}})]:
            with self.subTest(dcr=dcr, joint=joint), ExitStack() as stack:
                self.mocked_checks(stack, dcr=dcr, joint=joint)
                before = driver._capture_state()
                result = driver._repair_bar_stacking(self.cfg, self.actions, self.worst, self.joint, self.capacity)
                self.assertEqual(result[-1]["status"], "arrangements_exhausted")
                self.assertEqual(driver._capture_state(), before)
                self.assertIs(result[2], self.capacity)

    def test_other_failures_do_not_trigger_slab_growth(self):
        self.capacity["checks"].append({"id": "joint.shear", "status": "fail"})
        with patch.object(driver, "_capacity_state") as check:
            report = driver._repair_bar_stacking(self.cfg, self.actions, self.worst, self.joint, self.capacity)[-1]
        check.assert_not_called()
        self.assertEqual(report["status"], "other_constraints_fail")

    def test_hoop_change_reintroducing_clash_is_rejected_and_rolled_back(self):
        before = driver._capture_state()
        def capacity(*args):
            sp.BEAM_STIRRUP_BAR_SIZE = 5
            return self.capacity
        with ExitStack() as stack:
            self.mocked_checks(stack)
            stack.enter_context(patch.object(driver, "_capacity_design", side_effect=capacity))
            result = driver._repair_bar_stacking(self.cfg, self.actions, self.worst, self.joint, self.capacity)
        self.assertEqual(result[-1]["status"], "arrangements_exhausted")
        self.assertEqual(driver._capture_state(), before)

    def test_numerical_error_restores_state_and_propagates(self):
        before = driver._capture_state()
        def fail(*args):
            sp.COL_STIRRUP_BAR_SIZE = 6
            raise RuntimeError("Capacity hoop selection cycled")
        with ExitStack() as stack:
            self.mocked_checks(stack)
            stack.enter_context(patch.object(driver, "_capacity_design", side_effect=fail))
            with self.assertRaisesRegex(RuntimeError, "cycled"):
                driver._repair_bar_stacking(self.cfg, self.actions, self.worst, self.joint, self.capacity)
        self.assertEqual(driver._capture_state(), before)

    def test_thicker_slab_rebuilds_transfer_strip_and_frame_actions_from_same_cage(self):
        self.cfg.demands.accidental_torsion_ratio = 0.
        slabs = [{"thickness_in": 7.}, {"thickness_in": 7.5}]
        search = {"steps": 0, "trials": [], "thickness_in": 7.}
        initial_bar = sp.BEAM_BAR_SIZE
        seen = []
        def steel(*args):
            seen.append(sp.BEAM_BAR_SIZE)
            sp.BEAM_BAR_SIZE = 11
            return self.worst, {}, self.actions, self.joint
        with ExitStack() as stack:
            period = stack.enter_context(patch.object(driver, "_model_period", return_value=1.))
            stack.enter_context(patch.object(driver, "_steel_pass", side_effect=steel))
            stack.enter_context(patch.object(driver, "_capacity_design", return_value=self.capacity))
            stack.enter_context(patch.object(driver, "_repair_bar_stacking", side_effect=[
                (self.worst, self.joint, self.capacity, {"status": "arrangements_exhausted"}),
                (self.worst, self.joint, self.capacity, {"status": "resolved"})]))
            stack.enter_context(patch.object(driver, "_thicker_slab", return_value=(slabs[1], None)))
            transfer = stack.enter_context(patch.object(driver, "_update_floor_transfer"))
            strips = stack.enter_context(patch.object(driver, "_slab_thickness_search",
                return_value=(slabs[1], {"steps": 0, "trials": []})))
            result = driver._analyze_bar_layout(self.cfg, slabs[0], search, 3)
        self.assertEqual(seen, [initial_bar, initial_bar])
        self.assertEqual(period.call_count, 2)
        transfer.assert_called_once_with(self.cfg, slabs[1])
        strips.assert_called_once_with(self.cfg, slabs[1])
        self.assertEqual(result[0], slabs[1])
        self.assertEqual(search["steps"], 1)
        self.assertEqual(len(search["bar_layout_trials"]), 2)

    def test_slab_ladder_exhaustion_retains_failed_candidate(self):
        self.cfg.demands.accidental_torsion_ratio = 0.
        with ExitStack() as stack:
            stack.enter_context(patch.object(driver, "_model_period", return_value=1.))
            stack.enter_context(patch.object(driver, "_steel_pass", return_value=(self.worst, {}, self.actions, self.joint)))
            stack.enter_context(patch.object(driver, "_capacity_design", return_value=self.capacity))
            stack.enter_context(patch.object(driver, "_repair_bar_stacking", return_value=(
                self.worst, self.joint, self.capacity, {"status": "arrangements_exhausted"})))
            stack.enter_context(patch.object(driver, "_thicker_slab", return_value=(None, "ladder exhausted")))
            transfer = stack.enter_context(patch.object(driver, "_update_floor_transfer"))
            search = {"steps": 0, "trials": []}
            result = driver._analyze_bar_layout(self.cfg, {"thickness_in": 14.}, search, 3)
        transfer.assert_not_called()
        self.assertFalse(result[-1]["accepted"])
        self.assertEqual(search["bar_layout_trials"][-1]["next_thickness_unavailable"], "ladder exhausted")

    def test_pilot_clash_coordinates_and_strengths_follow_installed_stacking(self):
        # Portable reproduction of pilot 0005: 0.06 in bottom-mat overlap.
        values = dict(B_COL=38., H_COL=38., FC_COL_KSI=8., B_BEAM=18., H_BEAM=36., FC_BEAM_KSI=5.,
                      COL_BAR_SIZE=8, COL_TOP_BARS=6, COL_BOT_BARS=6, COL_SIDE_BARS=4, COL_BAR_AREA=.79,
                      BEAM_BAR_SIZE=10, BEAM_TOP_BARS=5, BEAM_BOT_BARS=5, BEAM_BAR_AREA=1.27,
                      COL_STIRRUP_BAR_SIZE=5, BEAM_STIRRUP_BAR_SIZE=4, COL_CLEAR_COVER_IN=1.5,
                      BEAM_CLEAR_COVER_IN=1.5, AGGREGATE_MAX_SIZE_IN=.75, BEAM_BAR_MAX_LAYERS=2,
                      SLAB_THICKNESS_IN=7.5)
        for key, value in values.items():
            setattr(sp, key, value)
        sp.SLAB_REINFORCEMENT = {"layout": {"outer_axis": "x", "layers": {
            f"{axis}_{face}": {"bar_size": 4, "bar_area_in2": .2, "spacing_in": 12., "axis": axis,
                "layer": "outer" if axis == "x" else "inner", "clear_cover_outer_mat_in": .75,
                "effective_depth_in": 6.5 if axis == "x" else 6.}
            for axis in ("x", "y") for face in ("top", "bottom")}}}
        def record():
            return {**driver._state_record_core(), "slab": {"thickness_in": 7.5},
                    "slab_reinforcement": copy.deepcopy(sp.SLAB_REINFORCEMENT)}
        old = record()
        self.assertAlmostEqual(joint_assembly(old, "y_over_x")["slab_clashes"][0]["overlap_in"], .06)
        _, old_strengths = beam_slab_strengths(old)
        sp.BEAM_BAR_STACKING = "x_over_y"
        new = record()
        assembly = joint_assembly(new, "x_over_y")
        self.assertFalse(assembly["slab_clashes"])
        self.assertEqual(old["slab_reinforcement"], new["slab_reinforcement"])
        _, strengths = beam_slab_strengths(new)
        for axis in ("x", "y"):
            self.assertEqual(assembly["stacking"]["layer_offsets_from_face_in"][axis],
                             sp.beam_bar_layers()[axis]["top"]["offsets_in"])
            self.assertEqual(strengths[f"{axis}_interior"]["centroid_offsets_in"],
                             sp.beam_bar_stacking_offsets_in()[axis])
        self.assertNotEqual(old_strengths["x_interior"]["mn_negative_kip_in"],
                            strengths["x_interior"]["mn_negative_kip_in"])

        # Pilot 0006: no supported order avoids the crossing at 7 in;
        # the next slab rung and x-over-y do, with the same member sections.
        sp.B_BEAM, sp.H_BEAM, sp.FC_BEAM_KSI = 20., 30., 6.
        sp.COL_BAR_SIZE, sp.COL_BAR_AREA = 9, 1.
        sp.BEAM_BAR_SIZE, sp.BEAM_BAR_AREA = 11, 1.56
        sp.BEAM_TOP_BARS = sp.BEAM_BOT_BARS = 6
        for thickness in (7., 7.5):
            for axis in ("x_over_y", "y_over_x"):
                for order in ("interleaved", "blocked"):
                    sp.BEAM_BAR_STACKING, sp.BEAM_BAR_LAYER_ORDER = axis, order
                    trial = record()
                    trial["slab"]["thickness_in"] = thickness
                    clashes = joint_assembly(trial, axis)["slab_clashes"]
                    if thickness == 7.:
                        self.assertTrue(clashes, (axis, order))
                    elif axis == "x_over_y" and order == "interleaved":
                        self.assertFalse(clashes)


if __name__ == "__main__":
    unittest.main()
