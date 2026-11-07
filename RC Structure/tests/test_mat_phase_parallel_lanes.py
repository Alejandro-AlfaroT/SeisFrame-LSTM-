"""The global slab-mat phase prefers to clear the bar lanes of parallel beams that terminate at a joint (2026-11-05).

A mat bar in the vertical plane of a parallel beam bar drops its end hook through that bar where both terminate (a
perimeter joint) when the hook sweeps the bar's elevation. The phase search takes the first column-clear phase that
also clears those lanes; without one it keeps the first column-clear phase (the earlier rule) and the hook-family
trials decide. No solve.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp


def layout(spacing=12.0):
    return {"bar_size": 4, "outer_axis": "x", "layers": {
        f"{axis}_{face}": {"bar_size": 4, "bar_area_in2": .2, "spacing_in": spacing,
                           "effective_depth_in": 6. if face == "top" else 1., "axis": axis,
                           "layer": "outer" if axis == "x" else "inner", "clear_cover_outer_mat_in": .75}
        for axis in ("x", "y") for face in ("top", "bottom")}}


class ParallelLanes(unittest.TestCase):
    def placement(self, spacing=12.0):
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg
        from Design import SMRF_Joint_Mat_Physical as mats
        from Design import Group_Capacity as gc
        from Design.SMRF_Splice_Physical import group_beam_lanes
        with gf.frame():
            sp.SLAB_REINFORCEMENT = {"layout": layout(spacing)}
            state = mg.install(gf.uniform_state())
            mats._CACHE.clear()
            placement = mats.global_mat_placement(state)
            lanes = group_beam_lanes(state)
            entries = gc.joint_inventory(state)["joints"]
            terminating = []
            for entry in entries:
                for axis in ("x", "y"):
                    members = entry["beams"][axis]
                    if members and not (len(members) == 2 and not (entry.get("mixed") or {}).get(axis, False)):
                        terminating.append((entry, axis, members))
            return placement, lanes, terminating, (sp.BAY_X, sp.BAY_Y)

    def test_no_mat_bar_shares_the_plane_of_a_terminating_parallel_beam_bar(self):
        placement, lanes, terminating, (bay_x, bay_y) = self.placement()
        self.assertEqual(placement["status"], "pass")
        self.assertTrue(terminating)
        checked = 0
        for entry, axis, members in terminating:
            origin = entry["grid_j"] * bay_y if axis == "x" else entry["grid_i"] * bay_x
            for name, mat in placement["floors"][str(entry["level"])]["layers"].items():
                if mat["runs_along"] != axis or placement["layers"][name]["phase_rule"] != "columns_and_swept_parallel_lanes":
                    continue
                z_top = -mat["centroid_from_top_in"]
                z_tail = z_top - mat["hook_radius_in"] - mat["hook_extension_in"]
                for member in members:
                    beam_db = sp.rebar_diameter(member.design.bar_size)
                    reach = (mat["db_in"] + beam_db) / 2
                    for lane in lanes[member.group_id]["coordinates"]:
                        z_lane = -member.design.h_in / 2 + lane["y_in"]
                        if not z_tail - reach < z_lane < z_top + reach:
                            continue                               # the hooked mat end never reaches this lane's elevation
                        for v in mat["global_transverse_coordinates_in"]:
                            checked += 1
                            self.assertGreaterEqual(abs(v - origin - lane["x_in"]), reach - 1e-6,
                                                    (entry["id"], name, lane["id"], v))
        self.assertGreater(checked, 0)

    def test_a_phase_on_a_lane_is_refused_and_another_is_taken(self):
        from Design import SMRF_Joint_Mat_Physical as mats
        from Design import SMRF_Splice_Physical as splice
        placement, lanes, terminating, _bays = self.placement()
        entry, axis, members = terminating[0]
        name = next(n for n, m in placement["floors"][str(entry["level"])]["layers"].items() if m["runs_along"] == axis)
        phase = placement["layers"][name]["phase_in"]
        # Put every lane of every beam on the mat bar nearest its grid line under the phase just chosen (the fixture's
        # bays are multiples of the 12 in spacing, so that bar is at the same offset from every grid line): that phase
        # must then be refused.
        on_bar = ((phase + 6.0) % 12.0) - 6.0
        moved = {gid: {**record, "coordinates": [{**p, "x_in": on_bar} for p in record["coordinates"]]}
                 for gid, record in lanes.items()}
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg
        from unittest import mock
        with gf.frame():
            sp.SLAB_REINFORCEMENT = {"layout": layout()}
            state = mg.install(gf.uniform_state())
            mats._CACHE.clear()
            with mock.patch.object(splice, "group_beam_lanes", lambda state=None: moved):
                again = mats.global_mat_placement(state)
            mats._CACHE.clear()
        self.assertIn("parallel_lanes", again["method"])
        self.assertEqual(again["layers"][name]["status"], "pass")
        if again["layers"][name]["phase_rule"] == "columns_and_swept_parallel_lanes":
            self.assertNotEqual(again["layers"][name]["phase_in"], phase)
        else:
            # No lane-clear phase: the earlier rule stands (the first column-clear phase), never a failure.
            self.assertEqual(again["layers"][name]["phase_rule"], "columns_only")


def setUpModule():
    # These tests verify the 3D physical detail route, which the computed common-action route keeps (2026-11-07).
    global _DETAIL_BASIS
    _DETAIL_BASIS = sp.DETAIL_BASIS
    sp.DETAIL_BASIS = "physical_3d_v2"


def tearDownModule():
    sp.DETAIL_BASIS = _DETAIL_BASIS


if __name__ == "__main__":
    unittest.main()
