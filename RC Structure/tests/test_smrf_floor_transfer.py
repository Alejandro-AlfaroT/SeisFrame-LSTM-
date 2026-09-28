"""Slab-to-frame gravity transfer: floor model, packing, and frame application.

Numerical checks only. They pin the load path (floor pressure reaches the
frame once, through the transfer), the sign and location of what each beam
receives, and the invariants that stop the tributary and transfer paths from
drifting apart. They do not certify the floor idealization.
"""
import contextlib
import copy
import json
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openseespy.opensees as ops  # noqa: E402

import Structure_Parameters as sp  # noqa: E402
from Design.SMRF_Floor_Analysis import analyze_floor, transfer_mesh_per_bay  # noqa: E402
from Design.SMRF_Floor_Transfer import (build_floor_transfer, transfer_totals,  # noqa: E402
                                        validate_floor_transfer)


SLAB = {"thickness_in": 5.0, "concrete_fc_ksi": 4.0,
        "concrete_unit_weight_kcf": 0.15, "superimposed_dead_load_ksf": 0.05}
GEOMETRY = {"num_bay_x": 3, "num_bay_y": 2, "bay_x_in": 240.0, "bay_y_in": 200.0}
SECTIONS = {"b_beam_in": 12.0, "h_beam_in": 18.0, "fc_beam_ksi": 4.0, "b_col_in": 18.0, "h_col_in": 18.0}
DEAD = {"id": "unit_dead", "dead_factor": 1.0, "live_factor": 0.0, "live_load_ksf": 0.05, "live_pattern": "none"}


def _by_key(result):
    return {(b["axis"], b["line_index"], b["span_index"]): b for b in result["beam_transfer"]}


class FlexibleFloorModelTests(unittest.TestCase):
    def test_exported_forces_and_couples_balance_asymmetric_patterns_at_every_mesh(self):
        """The production export IS the solved interface: forces plus couples, nothing redistributed."""
        from Design.SMRF_Floor_Transfer import global_couple
        for mesh in (4, 8, 12):
            for panels in ([[0, 0]], [[2, 1]], [[0, 0], [1, 0]]):
                with self.subTest(mesh=mesh, panels=panels):
                    case = {**DEAD, "id": "asymmetric", "dead_factor": 0., "live_factor": 1., "live_pattern": panels}
                    result = analyze_floor(SLAB, GEOMETRY, SECTIONS, case, mesh_per_bay=mesh, support_model="flexible_beams")
                    self.assertEqual(result["status"], "transfer_complete")
                    audit = result["transfer_equilibrium"]
                    self.assertTrue(audit["numerical_balance_passed"])
                    self.assertLess(max(audit["x_first_moment_relative_error"], audit["y_first_moment_relative_error"]), 1e-10)
                    # Recompute the balance from the exported inventory itself.
                    force = mx = my = cx = cy = 0.0
                    for beam in result["beam_transfer"]:
                        for load, couple in zip(beam["node_loads"], beam["node_couples"]):
                            f = load["x_fraction"]
                            x = (beam["span_index"] + f) * 240 if beam["axis"] == "x" else beam["line_index"] * 240
                            y = beam["line_index"] * 200 if beam["axis"] == "x" else (beam["span_index"] + f) * 200
                            force += load["load_kip"]; mx += load["load_kip"] * x; my += load["load_kip"] * y
                            gx, gy = global_couple(beam["axis"], couple["local_x_kip_in"], couple["local_y_kip_in"])
                            cx += gx; cy += gy
                    for column in result["column_direct_loads"]:
                        force += column["direct_load_kip"]
                        mx += column["direct_load_kip"] * column["grid_i"] * 240
                        my += column["direct_load_kip"] * column["grid_j"] * 200
                        cx += column["couple_global_mx_kip_in"]; cy += column["couple_global_my_kip_in"]
                    panel_load = .05 * 240 * 200 / 144
                    self.assertAlmostEqual(force, panel_load * len(panels), places=7)
                    self.assertAlmostEqual(mx + cy, sum(panel_load * (i + .5) * 240 for i, j in panels), places=6)
                    self.assertAlmostEqual(my - cx, sum(panel_load * (j + .5) * 200 for i, j in panels), places=6)

    def test_couples_are_what_close_the_moment_balance_under_asymmetric_load(self):
        case = {**DEAD, "id": "corner_live", "dead_factor": 0., "live_factor": 1., "live_pattern": [[0, 0]]}
        result = analyze_floor(SLAB, GEOMETRY, SECTIONS, case, mesh_per_bay=8, support_model="flexible_beams")
        self.assertTrue(result["equilibrium"]["numerical_balance_passed"])
        audit = result["transfer_equilibrium"]
        self.assertTrue(audit["vertical_force_balance_passed"])
        # Vertical loads alone do not balance the first moments; with the
        # exported couples they do, to solver precision.
        self.assertGreater(audit["vertical_load_only_y_first_moment_relative_error"], 1e-4)
        self.assertLess(audit["x_first_moment_relative_error"], 1e-10)
        self.assertLess(audit["y_first_moment_relative_error"], 1e-10)
        self.assertTrue(audit["numerical_balance_passed"])
        self.assertEqual(result["status"], "transfer_complete")
        self.assertFalse(result["verified"])
        self.assertFalse(ops.getNodeTags())

    def test_transfer_balances_with_raw_column_loads_and_couples(self):
        result = analyze_floor(SLAB, GEOMETRY, SECTIONS, DEAD, mesh_per_bay=8, support_model="flexible_beams")
        self.assertEqual(result["status"], "transfer_complete")
        balance = result["transfer_equilibrium"]
        applied = (0.15 * 5 / 12 + 0.05) / 144 * 720 * 400
        self.assertAlmostEqual(balance["applied_downward_kip"], applied, places=9)
        self.assertAlmostEqual(balance["beam_node_loads_kip"] + balance["column_direct_loads_kip"], applied, places=9)
        self.assertGreater(balance["column_direct_fraction"], 0.05)   # a coarse mesh sends more straight to the column
        for column in result["column_direct_loads"]:
            self.assertGreater(column["footprint_share_kip_informational"], 0.0)
            self.assertNotIn("reassigned_to_beams_kip", column)          # nothing is redistributed
        for beam in result["beam_transfer"]:
            self.assertEqual(len(beam["node_couples"]), len(beam["node_loads"]))
            for load in beam["node_loads"]:
                self.assertGreater(load["load_kip"], 0.0)          # downward
                self.assertTrue(0 < load["x_fraction"] < 1)

    def test_stiff_free_torsion_limit_reproduces_rigid_line_reactions(self):
        rigid = analyze_floor(SLAB, GEOMETRY, SECTIONS, DEAD, mesh_per_bay=6)
        stiff = analyze_floor(SLAB, GEOMETRY, {**SECTIONS, "beam_stiffness_modifier": 1e5,
                                               "beam_torsion_modifier": 1e-9},
                              DEAD, mesh_per_bay=6, support_model="flexible_beams")
        lines = {(r["axis"], r["line_index"], r["span_index"]): r["downward_load_to_line_kip"]
                 for r in rigid["beam_line_reactions"]}
        largest = max(lines.values())
        for key, beam in _by_key(stiff).items():
            self.assertLess(abs(beam["total_kip"] - lines[key]), 0.05 * largest, key)

    def test_symmetric_floor_gives_mirrored_transfers(self):
        beams = _by_key(analyze_floor(SLAB, GEOMETRY, SECTIONS, DEAD, mesh_per_bay=8, support_model="flexible_beams"))
        self.assertAlmostEqual(beams[("x", 0, 0)]["total_kip"], beams[("x", 2, 2)]["total_kip"], places=9)
        self.assertAlmostEqual(beams[("y", 0, 0)]["total_kip"], beams[("y", 3, 1)]["total_kip"], places=9)
        self.assertGreater(beams[("x", 1, 1)]["total_kip"], 1.5 * beams[("x", 0, 1)]["total_kip"])
        mirrored = [round(v["load_kip"], 9) for v in reversed(beams[("x", 0, 2)]["node_loads"])]
        self.assertEqual([round(v["load_kip"], 9) for v in beams[("x", 0, 0)]["node_loads"]], mirrored)

    def test_point_support_share_shrinks_with_mesh_and_beam_totals_converge(self):
        """Nothing is redistributed, so the direct-to-column share is the model's own
        point-support artifact: it must fall with refinement while the total stays exact."""
        results = {mesh: analyze_floor(SLAB, GEOMETRY, SECTIONS, DEAD, mesh_per_bay=mesh, support_model="flexible_beams")
                   for mesh in (8, 12, 16)}
        direct = [results[m]["transfer_equilibrium"]["column_direct_fraction"] for m in (8, 12, 16)]
        self.assertGreater(direct[0], direct[1])
        self.assertGreater(direct[1], direct[2])
        self.assertLess(direct[2], 0.04)
        coarse, fine = _by_key(results[12]), _by_key(results[16])
        for key in (("x", 1, 1), ("y", 1, 0), ("x", 0, 0)):
            self.assertLess(abs(coarse[key]["total_kip"] - fine[key]["total_kip"]) / fine[key]["total_kip"], 0.06, key)

    def test_beam_basis_is_the_slab_screen_t_section(self):
        from Design.SMRF_Slab import _beam_inertia
        result = analyze_floor(SLAB, GEOMETRY, SECTIONS, DEAD, mesh_per_bay=4, support_model="flexible_beams")
        inertia = result["beam_model"]["line_inertia"]
        gross, _, flange = _beam_inertia(12.0, 18.0, 5.0, 2, 200.0)
        self.assertAlmostEqual(inertia["x_interior"]["t_section_gross_in4"], gross)
        self.assertAlmostEqual(inertia["x_interior"]["iy_in4"], gross - flange * 5.0 ** 3 / 12.0)
        self.assertLess(inertia["x_edge"]["iy_in4"], inertia["x_interior"]["iy_in4"])

    def test_inputs_are_validated(self):
        with self.assertRaisesRegex(ValueError, "support_model"):
            analyze_floor(SLAB, GEOMETRY, SECTIONS, DEAD, support_model="springs")
        with self.assertRaisesRegex(ValueError, "sections.b_beam_in"):
            analyze_floor(SLAB, GEOMETRY, {}, DEAD, support_model="flexible_beams")
        with self.assertRaisesRegex(ValueError, "deeper than the slab"):
            analyze_floor(SLAB, GEOMETRY, {**SECTIONS, "h_beam_in": 5.0}, DEAD, support_model="flexible_beams")
        self.assertEqual(transfer_mesh_per_bay(2, 2), 16)
        self.assertEqual(transfer_mesh_per_bay(6, 6), 14)
        self.assertFalse(ops.getNodeTags())


class TransferRecordTests(unittest.TestCase):
    def test_builder_accepts_asymmetric_patterns_and_validation_catches_a_tampered_couple(self):
        record = build_floor_transfer(SLAB, GEOMETRY, SECTIONS, .05, mesh_per_bay=8,
                                      live_patterns=[{"id": "corner", "panels": [[0, 0]]}])
        area = 720 * 400 / 144
        validate_floor_transfer(record, GEOMETRY, SECTIONS, 5., .1125 * area, .05 * area)
        case = record["unit_cases"]["live_pattern_corner"]
        self.assertTrue(case["equilibrium"]["numerical_balance_passed"])
        case["beams"][0]["node_couples"][0][1] += 5.0            # an invented torsion couple
        with self.assertRaisesRegex(ValueError, "moment balance"):
            validate_floor_transfer(record, GEOMETRY, SECTIONS, 5., .1125 * area, .05 * area)
        del case["beams"][0]["node_couples"][0]                  # a missing couple row
        with self.assertRaisesRegex(ValueError, "load positions"):
            validate_floor_transfer(record, GEOMETRY, SECTIONS, 5., .1125 * area, .05 * area)

    def test_actual_load_inventory_is_checked_not_just_saved_pass_flags(self):
        original = build_floor_transfer(SLAB, GEOMETRY, SECTIONS, .05, mesh_per_bay=8)
        area = 720 * 400 / 144
        def validate(record):
            return validate_floor_transfer(record, GEOMETRY, SECTIONS, 5., .1125 * area, .05 * area)
        mutations = {
            "changed beam load": lambda c: c["beams"][0]["node_loads"][0].__setitem__(1, 999),
            "nonfinite column": lambda c: c["columns"][0].__setitem__("direct_load_kip", float("nan")),
            "duplicate beam": lambda c: c["beams"].append(copy.deepcopy(c["beams"][0])),
            "duplicate column": lambda c: c["columns"].append(copy.deepcopy(c["columns"][0])),
            "missing beam": lambda c: c["beams"].pop(),
            "missing column": lambda c: c["columns"].pop(),
            "bad axis": lambda c: c["beams"][0].__setitem__("axis", "z"),
            "bad index": lambda c: c["columns"][0].__setitem__("grid_i", True),
            "bad factor": lambda c: c["loadcase"].__setitem__("dead_factor", True),
            "bad pattern": lambda c: c["loadcase"].__setitem__("live_pattern", "all"),
            "bad summary": lambda c: c.__setitem__("beam_kip", 0.),
        }
        for name, mutate in mutations.items():
            value = copy.deepcopy(original)
            mutate(value["unit_cases"]["dead"])
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate(value)

    def test_relocated_load_with_identical_totals_is_rejected(self):
        value = build_floor_transfer(SLAB, GEOMETRY, SECTIONS, .05, mesh_per_bay=8)
        value["unit_cases"]["dead"]["beams"][0]["node_loads"][0][0] += .01
        value["unit_cases"]["dead"]["beams"][0]["node_couples"][0][0] += .01   # keep the couple rows aligned
        with self.assertRaisesRegex(ValueError, "moment balance"):
            validate_floor_transfer(value, GEOMETRY, SECTIONS, 5., .1125 * 2000, .05 * 2000)

    def test_pattern_cases_receive_the_same_validation_as_full_live(self):
        # Full-floor pattern is mechanically identical to the live unit case,
        # and lets this test exercise record integrity independently of the
        # separate known asymmetric-transfer mechanics failure.
        pattern = {"id": "all_panels", "panels": [[i, j] for i in range(3) for j in range(2)]}
        record = build_floor_transfer(SLAB, GEOMETRY, SECTIONS, .05, mesh_per_bay=8, live_patterns=[pattern])
        validate_floor_transfer(record, GEOMETRY, SECTIONS, 5., 225., 100.)
        record["unit_cases"]["live_pattern_all_panels"]["beams"][0]["node_loads"][0][1] += 1.
        with self.assertRaises(ValueError):
            validate_floor_transfer(record, GEOMETRY, SECTIONS, 5., 225., 100.)

    def test_zero_live_load_and_stale_transfer_method(self):
        record = build_floor_transfer(SLAB, GEOMETRY, SECTIONS, 0., mesh_per_bay=8)
        validate_floor_transfer(record, GEOMETRY, SECTIONS, 5., 225., 0.)
        self.assertIsNone(record["unit_cases"]["live"])
        record["method_version"] = "old_force_only_method"
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_floor_transfer(record, GEOMETRY, SECTIONS, 5., 225., 0.)

    def test_record_carries_unit_cases_that_match_pressure_times_area(self):
        record = build_floor_transfer(SLAB, GEOMETRY, SECTIONS, 0.05, mesh_per_bay=8)
        area = 720 * 400 / 144
        dead, live = (0.15 * 5 / 12 + 0.05) * area, 0.05 * area
        validate_floor_transfer(record, GEOMETRY, SECTIONS, 5.0, dead, live)
        self.assertAlmostEqual(transfer_totals(record, 1.2, 1.6), 1.2 * dead + 1.6 * live, places=9)
        round_trip = json.loads(json.dumps(record))
        validate_floor_transfer(round_trip, GEOMETRY, SECTIONS, 5.0, dead, live)
        self.assertEqual(round_trip["load_model"], "slab_transfer")

    def test_validation_refuses_foreign_or_unbalanced_records(self):
        record = build_floor_transfer(SLAB, GEOMETRY, SECTIONS, 0.05, mesh_per_bay=8)
        area = 720 * 400 / 144
        dead, live = (0.15 * 5 / 12 + 0.05) * area, 0.05 * area
        with self.assertRaisesRegex(ValueError, "geometry.num_bay_x"):
            validate_floor_transfer(record, {**GEOMETRY, "num_bay_x": 4}, SECTIONS, 5.0, dead, live)
        with self.assertRaisesRegex(ValueError, "sections.h_beam_in"):
            validate_floor_transfer(record, GEOMETRY, {**SECTIONS, "h_beam_in": 20.0}, 5.0, dead, live)
        with self.assertRaisesRegex(ValueError, "slab thickness"):
            validate_floor_transfer(record, GEOMETRY, SECTIONS, 6.0, dead, live)
        with self.assertRaisesRegex(ValueError, "expected"):
            validate_floor_transfer(record, GEOMETRY, SECTIONS, 5.0, dead * 1.01, live)
        broken = copy.deepcopy(record)
        broken["unit_cases"]["live"] = None
        with self.assertRaisesRegex(ValueError, "lacks the live"):
            validate_floor_transfer(broken, GEOMETRY, SECTIONS, 5.0, dead, live)


class FrameApplicationTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        values = {
            "NUM_BAY_X": 3, "NUM_BAY_Y": 2, "NUM_FLOOR": 2,
            "BAY_X": 240.0, "BAY_Y": 200.0, "STORY_H": 144.0,
            "B_COL": 18.0, "H_COL": 18.0, "B_BEAM": 12.0, "H_BEAM": 18.0,
            "FC_COL_KSI": 4.0, "FC_BEAM_KSI": 4.0,
            "SLAB_THICKNESS_IN": 5.0, "FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF": 0.05,
            "FLOOR_LIVE_LOAD_KSF": 0.05, "SEISMIC_LIVE_LOAD_FRACTION": 0.0,
            "CONCRETE_UNIT_WEIGHT_KCF": 0.150, "CONCRETE_UNIT_WEIGHT_KCI": 0.150 / 1728.0,
            "GRAVITY_LOAD_MODEL": "nodal", "FLOOR_TRANSFER": None,
            "NUM_MODES": sp.NUM_MODES, "GEOMETRY_VARIANT_NAME": getattr(sp, "GEOMETRY_VARIANT_NAME", "baseline"),
        }
        for name, value in values.items():
            self.stack.enter_context(mock.patch.object(sp, name, value, create=True))
        self.transfer = build_floor_transfer(SLAB, GEOMETRY, SECTIONS, 0.05, mesh_per_bay=8)

    def tearDown(self):
        self.stack.close()
        ops.wipe()

    def test_transfer_mode_applies_the_same_floor_total_as_nodal_mode(self):
        """The no-double-count invariant: same total, only the path differs."""
        from Loads import Gravity_Loads as loads
        n_col = sp.NUM_FLOOR * (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1)
        n_bx = sp.NUM_FLOOR * sp.NUM_BAY_X * (sp.NUM_BAY_Y + 1)

        def applied(mode_transfer, **factors):
            with mock.patch.object(sp, "FLOOR_TRANSFER", mode_transfer), \
                 mock.patch.object(loads.ops, "timeSeries"), mock.patch.object(loads.ops, "pattern"), \
                 mock.patch.object(loads.ops, "load") as nodal, mock.patch.object(loads.ops, "eleLoad") as element:
                loads.apply_gravity_loads(**factors)
            floor = sum(-call.args[3] for call in nodal.call_args_list)
            weight = 0.0
            for call in element.call_args_list:
                args = call.args
                if args[3] == "-beamPoint":
                    floor -= args[5]
                    self.assertTrue(0 < args[6] < 1)
                    self.assertGreater(args[1], n_col)
                else:
                    tag = args[1]
                    length = sp.STORY_H if tag <= n_col else sp.BAY_X if tag <= n_col + n_bx else sp.BAY_Y
                    weight -= (args[5] + args[6]) * length
            return floor, weight

        floor_nodal, weight_nodal = applied(None, floor_factor=1.0)
        floor_transfer, weight_transfer = applied(self.transfer, floor_factor=1.0)
        self.assertAlmostEqual(floor_transfer, floor_nodal, places=9)
        self.assertAlmostEqual(floor_nodal, sp.NUM_FLOOR * sp.total_floor_gravity_load(), places=9)
        self.assertAlmostEqual(weight_transfer, weight_nodal, places=9)
        # Separate dead/live factors are honoured by the transfer path.
        area = sp.BAY_X * sp.NUM_BAY_X * sp.BAY_Y * sp.NUM_BAY_Y / 144.0
        floor_factored, _ = applied(self.transfer, floor_factor=1.0, self_weight_factor=1.2,
                                    dead_factor=1.2, live_factor=1.6)
        self.assertAlmostEqual(floor_factored, sp.NUM_FLOOR * (1.2 * sp.floor_dead_load_ksf() + 1.6 * 0.05) * area, places=9)

    def test_beam_loads_land_on_the_right_elements(self):
        """Interior x-beams get more than edge x-beams, on every floor."""
        from Loads import Gravity_Loads as loads
        with mock.patch.object(sp, "FLOOR_TRANSFER", self.transfer), \
             mock.patch.object(loads.ops, "timeSeries"), mock.patch.object(loads.ops, "pattern"), \
             mock.patch.object(loads.ops, "load"), mock.patch.object(loads.ops, "eleLoad") as element:
            loads.apply_gravity_loads()
        per_element = {}
        for call in element.call_args_list:
            if call.args[3] == "-beamPoint":
                per_element[call.args[1]] = per_element.get(call.args[1], 0.0) - call.args[5]
        n_col = sp.NUM_FLOOR * (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1)
        for k in (1, 2):
            edge = per_element[loads._beam_element_tag(k, "x", 0, 1)]
            interior = per_element[loads._beam_element_tag(k, "x", 1, 1)]
            self.assertGreater(interior, 1.5 * edge)
            self.assertGreater(loads._beam_element_tag(k, "x", 0, 0), n_col)
        self.assertEqual(len(per_element), sp.NUM_FLOOR * (3 * 3 + 4 * 2))

    def test_elastic_frame_base_reactions_balance_in_transfer_mode(self):
        from Design import Design_Driver as driver
        from Design.SMRF_Elastic import build_design_model
        from Loads.Gravity_Loads import apply_gravity_loads
        from Analysis.Gravity import run_gravity_analysis
        from Model.Build_Model import node_tag
        expected = sp.NUM_FLOOR * (sp.total_floor_gravity_load() + sp.total_structural_self_weight_per_floor())
        with mock.patch.object(sp, "FLOOR_TRANSFER", self.transfer):
            self.assertEqual(sp.effective_gravity_load_model(), "slab_transfer")
            with driver._quiet():
                ops.wipe()
                build_design_model()
                apply_gravity_loads()
                run_gravity_analysis()
                ops.reactions()
            upward = sum(ops.nodeReaction(node_tag(0, i, j), 3)
                         for i in range(sp.NUM_BAY_X + 1) for j in range(sp.NUM_BAY_Y + 1))
            self.assertAlmostEqual(upward, expected, places=6)
            # Beams now carry gravity hogging; under nodal loads they carry almost none.
            from RC_Design_Check import get_element_tags, _extract_forces, beam_flexure_demands
            _, bx, _ = get_element_tags()
            hogging = max(beam_flexure_demands(*_extract_forces(tag)[5:7])[1] for tag in bx)
            self.assertGreater(hogging, 100.0)

    def test_geometry_change_drops_the_transfer(self):
        from Geometry_Overrides import apply_geometry_overrides
        with mock.patch.object(sp, "FLOOR_TRANSFER", self.transfer):
            apply_geometry_overrides({"NUM_BAY_X": 4}, emit=False)
            self.assertIsNone(sp.FLOOR_TRANSFER)
            self.assertIsNone(sp.SLAB_THICKNESS_IN)


if __name__ == "__main__":
    unittest.main()
