"""Member-resolved geometry, section properties, weights and masses (Model/Member_Properties, 2026-10-02).

In the uniform mode every function equals the one-section helper of Structure_Parameters it replaces. In
the grouped mode a changed group changes exactly its own members' properties and the ledgers that depend on
them, checked against independent hand totals.
"""
import contextlib
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

RC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_DIR))

import Structure_Parameters as sp                                              # noqa: E402
from Model import Member_Groups as mg                                           # noqa: E402
from Model import Member_Properties as mp                                       # noqa: E402

GEOMETRY = dict(NUM_BAY_X=2, NUM_BAY_Y=3, NUM_FLOOR=4, BAY_X=240.0, BAY_Y=216.0, STORY_H=168.0,
                B_COL=26.0, H_COL=30.0, B_BEAM=18.0, H_BEAM=28.0)


@contextlib.contextmanager
def frame(slab=7.0, **overrides):
    snapshot = dict(vars(sp))
    try:
        for key, value in {**GEOMETRY, **overrides}.items():
            setattr(sp, key, value)
        sp.SLAB_THICKNESS_IN = slab
        yield
    finally:
        mg.clear()
        for key in set(vars(sp)) - set(snapshot):
            delattr(sp, key)
        vars(sp).update(snapshot)


def close(test, a, b, rel=1e-12):
    test.assertTrue(math.isclose(a, b, rel_tol=rel, abs_tol=1e-12), (a, b))


class UniformEquivalence(unittest.TestCase):
    def test_every_member_property_equals_the_one_section_helper(self):
        for slab in (None, 7.0):
            with frame(slab):
                for member in mg.all_members():
                    if member.is_column:
                        close(self, mp.column_self_weight_kip_per_in(member), sp.col_self_weight_kip_per_in())
                        section = mp.elastic_section(member)
                        close(self, section["iy"], sp.rect_iy(sp.B_COL, sp.H_COL))
                        close(self, section["iz"], sp.rect_iz(sp.B_COL, sp.H_COL))
                        close(self, section["area"], sp.B_COL * sp.H_COL)
                        continue
                    axis, position = mp.beam_axis(member), member.location_class
                    legacy = sp.beam_flexural_section(axis, position)
                    ours = mp.beam_flexural_section(member)
                    close(self, ours["iy_in4"], legacy["iy_in4"])
                    close(self, ours["flange_width_in"], legacy["flange_width_in"])
                    close(self, ours["overhang_in"], legacy["overhang_in"])
                    self.assertEqual(ours["basis"], legacy["basis"])
                    close(self, mp.beam_self_weight_kip_per_in(member), sp.beam_self_weight_kip_per_in(axis))
                    close(self, mp.beam_drop_weight_kip_per_in(member), sp.beam_drop_weight_kip_per_in())
                    close(self, mp.elastic_section(member)["iy"], sp.beam_flexural_inertia_in4(axis, position))
                    if slab is not None:
                        clear_span, clear_web, sides = sp.beam_flange_geometry(axis, position)
                        geometry = mp.beam_flange_geometry(member)
                        close(self, geometry["clear_span_in"], clear_span)
                        self.assertEqual(geometry["slab_sides"], sides)
                        for side in geometry["sides"]:
                            close(self, side["clear_to_adjacent_web_in"], clear_web)

    def test_every_floor_ledger_equals_the_one_floor_helper(self):
        for slab in (None, 7.0):
            with frame(slab):
                for floor in range(1, sp.NUM_FLOOR + 1):
                    close(self, mp.floor_structural_self_weight_kip(floor), sp.total_structural_self_weight_per_floor())
                    close(self, mp.floor_seismic_weight_kip(floor), sp.total_floor_seismic_weight())
                    close(self, mp.story_gravity_weight_kip(floor), sp.total_floor_gravity_load() + sp.total_structural_self_weight_per_floor())
                    for j in range(sp.NUM_BAY_Y + 1):
                        for i in range(sp.NUM_BAY_X + 1):
                            close(self, mp.node_seismic_mass(floor, i, j), sp.node_seismic_mass(i, j))
                            close(self, mp.node_structural_self_weight_kip(floor, i, j), sp.node_structural_self_weight_kip(i, j))
                ledger = mp.weight_ledger()
                close(self, ledger["total_seismic_weight_kip"], sp.NUM_FLOOR * sp.total_floor_seismic_weight())
                for row in ledger["floors"]:
                    close(self, row["mass_sum_kip"], row["seismic_weight_kip"])           # the mass IS the seismic weight
                metadata = sp.floor_load_metadata()
                close(self, ledger["floors"][0]["column_self_weight_kip"], metadata["column_self_weight_per_floor_kip"])
                close(self, ledger["floors"][0]["beam_x_self_weight_kip"], metadata["beam_x_self_weight_per_floor_kip"])
                close(self, ledger["floors"][0]["beam_y_self_weight_kip"], metadata["beam_y_self_weight_per_floor_kip"])

    def test_column_axial_estimate_equals_the_legacy_tributary_estimate(self):
        from Model.IMK_Calibration import column_gravity_axial
        for slab in (None, 7.0):
            with frame(slab):
                for story in range(1, sp.NUM_FLOOR + 1):
                    for j in range(sp.NUM_BAY_Y + 1):
                        for i in range(sp.NUM_BAY_X + 1):
                            close(self, mp.column_gravity_axial(story, i, j), column_gravity_axial(story, i, j))


class GroupedHandTotals(unittest.TestCase):
    """One group changed in an isolated fixture: only its members change, and the ledgers follow by hand arithmetic."""

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(frame(7.0))
        seed = mg.GroupedDesign.uniform()
        column, beam = mg.uniform_member_design("column"), mg.uniform_member_design("beam_x")
        self.seed_column, self.seed_beam = column, beam
        self.state = seed.with_designs({
            "s03_04__column__interior": replace(column, b_in=20.0, h_in=22.0),
            "s01_02__beam_y__edge": replace(beam, member_type="beam_y", b_in=24.0, h_in=32.0),
        })
        mg.install(self.state)

    def tearDown(self):
        self.stack.close()

    def test_only_the_changed_groups_members_change(self):
        changed_columns = set(self.state.groups["s03_04__column__interior"]["member_tags"])
        changed_beams = set(self.state.groups["s01_02__beam_y__edge"]["member_tags"])
        self.assertEqual((len(changed_columns), len(changed_beams)), (2 * 1 * 2, 2 * 2 * 3))   # 2 stories x interior; 2 floors x 2 lines x 3 spans
        gamma = sp.CONCRETE_UNIT_WEIGHT_KCF / 1728.0
        for member in mg.all_members():
            if member.is_column:
                b, h = (20.0, 22.0) if member.member_tag in changed_columns else (26.0, 30.0)
                self.assertEqual((member.design.b_in, member.design.h_in), (b, h))
                close(self, mp.column_self_weight_kip_per_in(member), gamma * b * h * (168.0 - 7.0) / 168.0)
            else:
                b, h = (24.0, 32.0) if member.member_tag in changed_beams else (18.0, 28.0)
                self.assertEqual((member.design.b_in, member.design.h_in), (b, h))
                close(self, mp.beam_drop_weight_kip_per_in(member), gamma * b * (h - 7.0))

    def test_clear_spans_use_half_of_each_adjoining_joint_and_the_joint_takes_the_column_below(self):
        # X beams at floor 3 and 4 along the interior line j = 1 or 2 frame into the smaller interior column at i = 1
        # (h = 22 in along X) and into the perimeter column at i = 0 or 2 (h = 30 in): unequal faces.
        beam = mg.beam_at("beam_x", 3, 0, 1)
        self.assertEqual(mp.beam_end_joint_depths(beam), (30.0, 22.0))
        close(self, mp.beam_clear_span_in(beam), 240.0 - 15.0 - 11.0)
        self.assertEqual(mp.beam_face_distances_in(beam), (15.0, 11.0))
        mirrored = mg.beam_at("beam_x", 3, 1, 1)
        self.assertEqual(mp.beam_end_joint_depths(mirrored), (22.0, 30.0))
        # floor 2 is the top of band 1-2: its joint core is the column BELOW (story 2, still 26 x 30), not the smaller one above
        below = mg.beam_at("beam_x", 2, 0, 1)
        self.assertEqual(mp.beam_end_joint_depths(below), (30.0, 30.0))
        self.assertEqual(mg.joint_members(2, 1, 1)["column_above"].design.h_in, 22.0)
        self.assertEqual(mp.joint_core_dimensions(2, 1, 1), (30.0, 26.0))
        self.assertEqual(mp.joint_core_dimensions(3, 1, 1), (22.0, 20.0))
        # a Y beam into the changed interior column uses its b (20 in along Y)
        y_beam = mg.beam_at("beam_y", 4, 1, 0)
        self.assertEqual(mp.beam_end_joint_depths(y_beam), (26.0, 20.0))
        close(self, mp.beam_clear_span_in(y_beam), 216.0 - 13.0 - 10.0)
        gamma = sp.CONCRETE_UNIT_WEIGHT_KCF / 1728.0
        close(self, mp.beam_self_weight_kip_per_in(beam), gamma * 18.0 * 21.0 * (240.0 - 26.0) / 240.0)
        self.assertEqual(mp.JOINT_CORE_CONVENTION, "joint_core_takes_the_column_below_v1")

    def test_flange_uses_the_actual_neighbouring_webs(self):
        # floor 1: the edge Y beams (i = 0, 2) are 24 in wide, the interior Y beam (i = 1) is 18 in wide
        interior = mg.beam_at("beam_y", 1, 1, 0)
        geometry = mp.beam_flange_geometry(interior)
        self.assertEqual(geometry["slab_sides"], 2)
        for side in geometry["sides"]:
            close(self, side["clear_to_adjacent_web_in"], 240.0 - 9.0 - 12.0)
        edge = mg.beam_at("beam_y", 1, 0, 0)
        edge_geometry = mp.beam_flange_geometry(edge)
        self.assertEqual(edge_geometry["slab_sides"], 1)
        close(self, edge_geometry["sides"][0]["clear_to_adjacent_web_in"], 240.0 - 12.0 - 9.0)
        flange = mp.effective_flange(edge)
        clear_span = 216.0 - 13.0 - 13.0
        close(self, flange["flange_width_in"], 24.0 + min(6.0 * 7.0, (240.0 - 21.0) / 2.0, clear_span / 12.0))
        section = mp.beam_flexural_section(edge)
        close(self, section["iy_in4"], sp.t_section_inertia_in4(24.0, 32.0, 7.0, flange["flange_width_in"]))
        # an unchanged X beam next to nothing changed keeps the uniform flange
        with_uniform = mg.beam_at("beam_x", 1, 0, 1)
        close(self, mp.beam_flexural_section(with_uniform)["flange_width_in"],
              18.0 + 2.0 * min(8.0 * 7.0, (216.0 - 18.0) / 2.0, (240.0 - 30.0) / 8.0))

    def test_floor_weights_masses_and_axial_loads_follow_by_hand(self):
        gamma = sp.CONCRETE_UNIT_WEIGHT_KCF / 1728.0
        nx, ny = sp.NUM_BAY_X, sp.NUM_BAY_Y

        def hand_floor(floor):
            total = 0.0
            for j in range(ny + 1):
                for i in range(nx + 1):
                    interior = 0 < i < nx and 0 < j < ny
                    b, h = (20.0, 22.0) if (interior and floor in (3, 4)) else (26.0, 30.0)
                    total += gamma * b * h * (168.0 - 7.0)
            def core(level, i, j):                      # the joint core is the column below the floor
                interior = 0 < i < nx and 0 < j < ny
                return (22.0, 20.0) if (interior and level in (3, 4)) else (30.0, 26.0)
            for j in range(ny + 1):
                for i in range(nx):
                    total += gamma * 18.0 * (28.0 - 7.0) * (240.0 - 0.5 * core(floor, i, j)[0] - 0.5 * core(floor, i + 1, j)[0])
            for j in range(ny):
                for i in range(nx + 1):
                    edge = i in (0, nx)
                    b, h = (24.0, 32.0) if (edge and floor in (1, 2)) else (18.0, 28.0)
                    total += gamma * b * (h - 7.0) * (216.0 - 0.5 * core(floor, i, j)[1] - 0.5 * core(floor, i, j + 1)[1])
            return total

        uniform_floor = None
        for floor in range(1, sp.NUM_FLOOR + 1):
            close(self, mp.floor_structural_self_weight_kip(floor), hand_floor(floor), rel=1e-11)
        self.assertNotAlmostEqual(mp.floor_structural_self_weight_kip(1), mp.floor_structural_self_weight_kip(3), places=3)
        ledger = mp.weight_ledger()
        self.assertTrue(ledger["grouped"])
        area_weight = (sp.floor_dead_load_ksf() + sp.SEISMIC_LIVE_LOAD_FRACTION * sp.FLOOR_LIVE_LOAD_KSF) * (240.0 * 2 * 216.0 * 3) / 144.0
        for row in ledger["floors"]:
            close(self, row["seismic_weight_kip"], area_weight + hand_floor(row["floor"]), rel=1e-11)
            close(self, row["mass_sum_kip"], row["seismic_weight_kip"], rel=1e-11)      # nodal masses sum to the floor's seismic weight
        close(self, ledger["total_member_self_weight_kip"], sum(hand_floor(f) for f in range(1, 5)), rel=1e-11)
        # the interior base column carries its own (26 x 30) weight for stories 1-2 and the lighter column for 3-4
        axial = mp.column_gravity_axial(1, 1, 1)
        hand = sp.node_gravity_load_kip(1, 1) * 4 + sum(mp.node_structural_self_weight_kip(f, 1, 1) for f in (1, 2, 3, 4))
        close(self, axial, hand)
        self.assertLess(mp.node_structural_self_weight_kip(3, 1, 1), mp.node_structural_self_weight_kip(1, 1, 1))
        close(self, mp.gravity_above_kip(3), sum(mp.story_gravity_weight_kip(f) for f in (3, 4)))

    def test_the_one_section_helpers_refuse_to_answer_for_a_grouped_design(self):
        for helper in (sp.col_self_weight_kip_per_in, lambda: sp.beam_self_weight_kip_per_in("x"), sp.total_structural_self_weight_per_floor,
                       sp.total_floor_seismic_weight, lambda: sp.beam_flexural_section("x", "edge"), lambda: sp.node_seismic_mass(0, 0),
                       sp.floor_load_metadata, lambda: sp.beam_flange_geometry("x", "interior")):
            with self.assertRaises(mg.GroupedStateError):
                helper()


# These hand totals describe the framed members; the column extension above the roof is verified on its own
# (tests/test_roof_extension.py), so this module runs with the terminating roof column.
def setUpModule():
    global _ROOF_PATCH
    from unittest import mock as _mock
    import Structure_Parameters as _sp
    _ROOF_PATCH = _mock.patch.object(_sp, "ROOF_COLUMN_EXTENSION", False)
    _ROOF_PATCH.start()


def tearDownModule():
    _ROOF_PATCH.stop()


if __name__ == "__main__":
    unittest.main()
