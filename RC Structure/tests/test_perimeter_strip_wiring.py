"""Wiring tests for the perimeter strip (2026-11-06, no structural solve).

  record install and replay give the validator the plate load, unequal faces survive the frame call (Mx, My),
  the pushover inventory follows a live pattern, and the coupled model's column line uses the clipped volume with
  non-square perimeter columns on the floor's own widths.
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402
import test_perimeter_strip_loads as t                                           # noqa: E402

import Structure_Parameters as sp                                               # noqa: E402
from Design import Grouped_Record as gr                                         # noqa: E402
from Design import SMRF_Coupled_Analysis as coupled                             # noqa: E402
from Design import SMRF_Floor_Sections as floor_sections                        # noqa: E402
from Design import SMRF_Floor_Transfer as ft                                    # noqa: E402
from Design.SMRF_Coupled_Comparison import coupled_sections_by_member           # noqa: E402
from Loads import Gravity_Loads as loads                                        # noqa: E402
from Loads import Perimeter_Strip as strip                                      # noqa: E402
from Model import Floor_Footprint as fp                                         # noqa: E402
from Model import Member_Groups as mg                                           # noqa: E402
from Model import Member_Properties as mp                                       # noqa: E402
from Model.Build_Model import node_tag                                          # noqa: E402


class Reached(Exception):
    pass


class RecordWiring(unittest.TestCase):
    def setUp(self):
        self.stack = gf.frame(**t.LOADS)
        self.stack.__enter__()
        self.state = gf.changed_state()
        mg.install(self.state)

    def tearDown(self):
        self.stack.__exit__(None, None, None)

    def test_replay_validates_against_the_plate(self):
        record = {"materials": {}, "design_actions": {"expected_combination_ids": ["a"]}, "gravity_load_model": "slab_transfer",
                  "floor_transfer": {}, "slab": {"thickness_in": gf.SLAB}, "geometry": t.GEOMETRY,
                  "floor_loads": {"floor_dead_load_ksf": 0.1375, "floor_live_load_ksf": 0.04}}
        with mock.patch.object(gr, "_common_action_checks", return_value=[]), \
             mock.patch("Design.SMRF_Qualification.reinforcement_consistency_checks",
                        return_value=[{"id": "reinforcement.declared_material_scope"}]), \
             mock.patch("Design.SMRF_Design_Evidence.validated_combinations", return_value=([{"id": "a"}], ["a"])), \
             mock.patch.object(gr, "_cfg_view", return_value=None), \
             mock.patch.object(ft, "validate_floor_transfers_by_floor", side_effect=Reached) as validator:
            with self.assertRaises(Reached):
                gr._qualification_checks(record, self.state)
        self.assertAlmostEqual(validator.call_args.args[4], 297.0, places=10)
        self.assertAlmostEqual(validator.call_args.args[5], 86.4, places=10)

    def test_install_validates_against_the_plate(self):
        description = floor_sections.by_floor()
        record = {"schema_version": gr.SCHEMA, "design_mode": "grouped", "geometry": dict(gr.driver._slab_geometry()),
                  "materials": {"es_ksi": sp.ES_KSI, "fyt_ksi": sp.FYT_KSI, "aggregate_size_in": 1.0, "beam_clear_cover_in": 1.5,
                                "col_clear_cover_in": 1.5, "reinforcement_specification": sp.REINFORCEMENT_SPECIFICATION,
                                "exposure": sp.MATERIAL_EXPOSURE},
                  "member_groups": {}, "floor_sections": description, "gravity_load_model": "slab_transfer", "floor_transfer": {}}
        with mock.patch.object(gr, "_validated_record_grade", return_value=sp.FY_KSI), \
             mock.patch.object(gr, "_validated_slab", return_value={}), \
             mock.patch.object(gr, "_check_record_grades"), \
             mock.patch.object(gr.driver, "_apply_slab"), \
             mock.patch.object(sp, "require_a706"), \
             mock.patch.object(sp, "apply_reinforcement_grade"), \
             mock.patch.object(sp, "beam_bar_stacking_convention", return_value=None), \
             mock.patch.object(gr.mg.GroupedDesign, "from_record", return_value=self.state), \
             mock.patch.object(ft, "validate_floor_transfers_by_floor", side_effect=Reached) as validator:
            with self.assertRaises(Reached):
                gr.apply_grouped_design(record)
        self.assertAlmostEqual(validator.call_args.args[4], 297.0, places=10)
        self.assertAlmostEqual(validator.call_args.args[5], 86.4, places=10)


class FrameCallAndInventory(unittest.TestCase):
    def setUp(self):
        self.stack = gf.frame(**t.LOADS)
        self.stack.__enter__()

    def tearDown(self):
        self.stack.__exit__(None, None, None)

    def test_unequal_faces_keep_both_moments_at_the_frame_call(self):
        mg.install(gf.changed_state())
        transfer = t.synthetic_transfer(floor_sections.for_floor(1))
        faces = (-7.0, 492.0, -8.0, 662.0)
        positions = {node_tag(1, i, j): (i * t.LX, j * t.LY) for j in range(t.NY + 1) for i in range(t.NX + 1)}
        with mock.patch.object(fp, "physical_floor", return_value=fp.rectangle(*faces)), \
             mock.patch.object(loads.ops, "load") as nodal, mock.patch.object(loads.ops, "eleLoad") as element:
            loads._apply_perimeter_strip(1, transfer, 1.0, 0.0, "all")
        force = about_x = about_y = 0.0
        for call in nodal.call_args_list:
            tag, _fx, _fy, fz, mx, my, _mz = call.args
            x, y = positions[tag]
            force -= fz
            about_x += -(-fz) * y + mx
            about_y += (-fz) * x + my
        for call in element.call_args_list:
            member = mg.resolve(call.args[1])
            length = mp.member_length_in(member)
            load = -call.args[5] * length
            x = member.grid_i * t.LX + (0.5 * length if member.member_type == "beam_x" else 0.0)
            y = member.grid_j * t.LY + (0.5 * length if member.member_type == "beam_y" else 0.0)
            force += load
            about_x += -load * y
            about_y += load * x
        q = 0.1375 / 144.0
        foot, plate = fp.rectangle(*faces), fp.rectangle(0.0, t.NX * t.LX, 0.0, t.NY * t.LY)
        self.assertAlmostEqual(force, q * (foot["area_in2"] - plate["area_in2"]), places=9)
        self.assertAlmostEqual(about_x, -q * (foot["first_moment_y_in3"] - plate["first_moment_y_in3"]), places=6)
        self.assertAlmostEqual(about_y, q * (foot["first_moment_x_in3"] - plate["first_moment_x_in3"]), places=6)

    def test_pushover_inventory_follows_the_live_pattern(self):
        from Analysis import Pushover_Diagnostic as pushover
        sp.FLOOR_TRANSFER = t.synthetic_transfer(t.uniform_sections())
        settings = SimpleNamespace(gravity_self_weight_factor=0.0, gravity_dead_factor=0.0, gravity_live_factor=1.0, live_pattern="two_panels")
        total, _ = pushover.expected_gravity_total_kip(settings)
        force = t.applied(floor_factor=1.0, self_weight_factor=0.0, dead_factor=0.0, live_factor=1.0, roof_extension=False,
                          live_pattern="two_panels")[0]
        self.assertAlmostEqual(total, force, places=8)


class CoupledColumnWeight(unittest.TestCase):
    def test_coupled_column_line_is_the_clipped_volume_with_non_square_columns(self):
        with gf.frame():
            state = gf.uniform_state(column=replace(gf.COLUMN, b_in=22.0, h_in=34.0)).with_designs({"s01_02__beam_y__edge": gf.DEEP_BEAM})
            mg.install(state)
            slab = {"thickness_in": gf.SLAB, "concrete_fc_ksi": gf.BEAM.fc_ksi, "concrete_unit_weight_kcf": 0.15,
                    "superimposed_dead_load_ksf": 0.05}
            geometry = {"num_bay_x": gf.NX, "num_bay_y": gf.NY, "num_floor": gf.NZ, "bay_x_in": gf.BAY_X, "bay_y_in": gf.BAY_Y,
                        "story_h_in": gf.STORY_H}
            case = {"id": "g", "dead_factor": 1.2, "live_factor": 0.0, "live_load_ksf": 0.05, "live_pattern": "none"}
            calls, real = [], coupled.ops.eleLoad

            def spy(*a, **k):
                calls.append(a)
                return real(*a, **k)

            with mock.patch.object(coupled.ops, "eleLoad", spy), mock.patch.object(coupled.ops, "analyze", side_effect=Reached):
                with self.assertRaises(Reached):
                    coupled.analyze_coupled_gravity(slab, geometry, coupled_sections_by_member(), [case] * gf.NZ, mesh_per_bay=2)
            coupled.ops.wipe()
            got = [-a[6] * sp.STORY_H for a in calls if a[6] != 0.0]
            frame = {(m.story_or_floor, m.grid_j, m.grid_i): 1.2 * mp.column_self_weight_kip_per_in(m) * sp.STORY_H
                     for m in mg.all_members("column")}
            self.assertEqual(len(got), len(frame))
            for value, key in zip(got, sorted(frame)):
                self.assertAlmostEqual(value, frame[key], places=10, msg=key)


if __name__ == "__main__":
    unittest.main()
