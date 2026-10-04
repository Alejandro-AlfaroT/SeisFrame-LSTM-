"""The floor path under a grouped design: by-line floor models, one transfer per distinct floor (2026-10-02).

  * a floor described line by line with equal lines solves to the one-beam floor's transfer;
  * unlike lines change the load path, keep force and first-moment equilibrium, and are graded to their
    own beam faces with the same number of cells in every bay;
  * floors are matched by the signature of their mechanical inputs (two distinct floors in the fixture);
  * the frame receives each floor's own transfer, and the base reaction is the hand total;
  * a uniform transfer, a transfer of another design, an edited transfer and a missing floor are refused.
"""
import copy
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402

import openseespy.opensees as ops                                               # noqa: E402
import Structure_Parameters as sp                                               # noqa: E402
from Analysis.Gravity import run_gravity_analysis                                # noqa: E402
from Design import SMRF_Floor_Sections as fs                                     # noqa: E402
from Design.SMRF_Elastic import build_design_model                               # noqa: E402
from Design.SMRF_Floor_Analysis import analyze_floor                             # noqa: E402
from Design.SMRF_Floor_Mesh import (floor_mesh, graded_face_levels, graded_face_levels_between,  # noqa: E402
                                    nested_refinement, resolve_recipe_plan, span_offsets)
from Design.SMRF_Floor_Transfer import (build_floor_transfer, build_floor_transfers_by_floor,  # noqa: E402
                                        validate_floor_transfers_by_floor)
from Loads.Gravity_Loads import apply_gravity_loads                              # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402
from Model.Build_Model import node_tag                                           # noqa: E402

SLAB = {"thickness_in": gf.SLAB, "concrete_fc_ksi": gf.BEAM.fc_ksi, "concrete_unit_weight_kcf": 0.15,
        "superimposed_dead_load_ksf": 0.05}
GEOMETRY = {"num_bay_x": gf.NX, "num_bay_y": gf.NY, "num_floor": gf.NZ, "bay_x_in": gf.BAY_X, "bay_y_in": gf.BAY_Y,
            "story_h_in": gf.STORY_H}
UNIFORM = {"b_beam_in": gf.BEAM.b_in, "h_beam_in": gf.BEAM.h_in, "fc_beam_ksi": gf.BEAM.fc_ksi,
           "b_col_in": gf.COLUMN.b_in, "h_col_in": gf.COLUMN.h_in}
MESH = 8
AREA_SQFT = gf.BAY_X * gf.NX * gf.BAY_Y * gf.NY / 144.0


def unit_case(dead=1.0, live=0.0):
    return {"id": "unit", "dead_factor": dead, "live_factor": live, "live_load_ksf": 0.05,
            "live_pattern": "all" if live else "none"}


def line_loads(result):
    return {(b["axis"], b["line_index"], b["span_index"]): b["total_kip"] for b in result["beam_transfer"]}


class Mesh(unittest.TestCase):
    def test_equal_widths_give_the_symmetric_recipe_and_unlike_widths_keep_the_cell_count(self):
        self.assertEqual(graded_face_levels_between(240.0, 14.0, 14.0), graded_face_levels(240.0, 14.0))
        symmetric = graded_face_levels(240.0, 18.0)
        unlike = graded_face_levels_between(240.0, 24.0, 18.0)
        for level, (a, b) in enumerate(zip(symmetric, unlike)):
            self.assertEqual(len(a), len(b), level)                       # same cells per bay at every level
            self.assertEqual((b[0], b[-1]), (0.0, 240.0))
            self.assertTrue(all(y > x for x, y in zip(b, b[1:])))
            self.assertIn(12.0, b)                                         # the left beam's face (24 / 2)
            self.assertIn(240.0 - 9.0, b)                                  # the right beam's face (18 / 2)
            self.assertIn(120.0, b)
        for coarse, fine in zip(unlike, unlike[1:]):
            self.assertTrue(set(coarse) <= set(fine))                      # nested

    def test_a_grid_given_bay_by_bay_keeps_the_index_arithmetic(self):
        bays = [graded_face_levels_between(240.0, 24.0, 18.0)[0], graded_face_levels_between(240.0, 18.0, 24.0)[0]]
        ys = graded_face_levels(216.0, 18.0)[0]
        grid = floor_mesh(2, 3, 240.0, 216.0, 4, mesh_spec={"x_offsets_in": bays, "y_offsets_in": ys, "max_shells": 45000})
        mx = grid["subdivisions_x_per_bay"]
        self.assertEqual(mx, len(bays[0]) - 1)
        self.assertEqual(grid["x_coordinates_in"][mx], 240.0)              # the interior beam line is still at index mx
        self.assertEqual(grid["x_coordinates_in"][2 * mx], 480.0)
        self.assertIsNone(grid["x_offsets_in"])                            # nothing reads one bay's offsets for another
        self.assertEqual(span_offsets(grid, "x", 1), bays[1])
        self.assertEqual(span_offsets(grid, "y", 2), ys)
        self.assertIn(240.0 + 9.0, grid["x_coordinates_in"])               # the interior beam's face in the second bay
        with self.assertRaises(ValueError):
            floor_mesh(2, 3, 240.0, 216.0, 4, mesh_spec={"x_offsets_in": [bays[0]], "y_offsets_in": ys, "max_shells": 45000})
        with self.assertRaises(ValueError):
            floor_mesh(2, 3, 240.0, 216.0, 4, mesh_spec={"x_offsets_in": [bays[0], bays[1][:-2] + [240.0]],
                                                         "y_offsets_in": ys, "max_shells": 45000})

    def test_the_recipe_resolves_from_a_by_line_floor_and_from_its_recorded_widths(self):
        with gf.frame():
            mg.install(gf.changed_state())
            sections = fs.for_floor(1)
        policy = {"recipe": "graded_face_v1", "levels": 3, "max_shells": 45000}
        plan = resolve_recipe_plan(GEOMETRY, sections, policy)
        self.assertEqual(plan["status"], "resolved")
        self.assertEqual(plan["inputs"]["face_widths_in"], {"x": [24.0, 18.0, 24.0], "y": [18.0] * 4})
        again = resolve_recipe_plan(GEOMETRY, {"face_widths_in": plan["inputs"]["face_widths_in"]}, policy)
        self.assertEqual(again["meshes"], plan["meshes"])
        grids = [floor_mesh(gf.NX, gf.NY, gf.BAY_X, gf.BAY_Y, 4, mesh_spec=spec) for spec in plan["meshes"]]
        for a, b in zip(grids, grids[1:]):
            nested_refinement(a, b)
        # equal lines: the by-line plan has the coordinates of the one-width plan
        with gf.frame():
            mg.install(gf.uniform_state())
            same = resolve_recipe_plan(GEOMETRY, fs.for_floor(1), policy)
        legacy = resolve_recipe_plan(GEOMETRY, UNIFORM, policy)
        for spec_a, spec_b in zip(same["meshes"], legacy["meshes"]):
            a = floor_mesh(gf.NX, gf.NY, gf.BAY_X, gf.BAY_Y, 4, mesh_spec=spec_a)
            b = floor_mesh(gf.NX, gf.NY, gf.BAY_X, gf.BAY_Y, 4, mesh_spec=spec_b)
            self.assertEqual(a["coordinate_sha256"], b["coordinate_sha256"])


class FloorModel(unittest.TestCase):
    def test_equal_lines_solve_to_the_one_beam_floor(self):
        with gf.frame():
            mg.install(gf.uniform_state())
            by_line = fs.for_floor(2)
        self.assertEqual(fs.uniform_equivalent(by_line), UNIFORM)
        ops.wipe()
        a = analyze_floor(SLAB, GEOMETRY, UNIFORM, unit_case(), mesh_per_bay=MESH, support_model="flexible_beams")
        b = analyze_floor(SLAB, GEOMETRY, by_line, unit_case(), mesh_per_bay=MESH, support_model="flexible_beams")
        self.assertEqual((a["status"], b["status"]), ("transfer_complete", "transfer_complete"))
        for key, load in line_loads(a).items():
            self.assertTrue(math.isclose(line_loads(b)[key], load, rel_tol=1e-9, abs_tol=1e-9), key)
        for ca, cb in zip(a["column_direct_loads"], b["column_direct_loads"]):
            self.assertTrue(math.isclose(ca["direct_load_kip"], cb["direct_load_kip"], rel_tol=1e-9, abs_tol=1e-9))
            self.assertTrue(math.isclose(ca["footprint_share_kip_informational"], cb["footprint_share_kip_informational"],
                                         rel_tol=1e-12))

    def test_unlike_lines_change_the_load_path_and_keep_equilibrium(self):
        with gf.frame():
            mg.install(gf.changed_state())
            low, high = fs.for_floor(1), fs.for_floor(3)
        self.assertEqual(low["beam_lines"]["y"], [{"b_in": 24.0, "h_in": 32.0, "fc_ksi": 4.0}, {"b_in": 18.0, "h_in": 28.0, "fc_ksi": 4.0},
                                                  {"b_in": 24.0, "h_in": 32.0, "fc_ksi": 4.0}])
        self.assertEqual(high["supports"][1][1], {"b_in": 22.0, "h_in": 22.0})          # the smaller interior column below floor 3
        self.assertEqual(high["supports"][0][0], {"b_in": 26.0, "h_in": 26.0})
        self.assertNotEqual(fs.signature(low), fs.signature(high))
        ops.wipe()
        uniform = analyze_floor(SLAB, GEOMETRY, UNIFORM, unit_case(), mesh_per_bay=MESH, support_model="flexible_beams")
        changed = analyze_floor(SLAB, GEOMETRY, low, unit_case(), mesh_per_bay=MESH, support_model="flexible_beams")
        self.assertEqual(changed["status"], "transfer_complete")
        audit = changed["transfer_equilibrium"]
        self.assertLess(max(audit["relative_error"], audit["x_first_moment_relative_error"], audit["y_first_moment_relative_error"]), 1e-8)
        dead = (0.15 * gf.SLAB / 12.0 + 0.05) * AREA_SQFT
        self.assertTrue(math.isclose(audit["applied_downward_kip"], dead, rel_tol=1e-12))
        # the deeper, wider edge Y beams are stiffer supports: they draw more of the slab load than before
        edge = lambda r: sum(v for (axis, line, _span), v in line_loads(r).items() if axis == "y" and line in (0, gf.NX))  # noqa: E731
        self.assertGreater(edge(changed), 1.02 * edge(uniform))
        lines = changed["beam_model"]["lines"]
        self.assertEqual((lines["y"][0]["position"], lines["y"][1]["position"]), ("edge", "interior"))
        self.assertGreater(lines["y"][0]["iy_in4"], lines["y"][1]["iy_in4"])
        self.assertEqual(changed["beam_model"]["floor_sections_sha256"], fs.signature(low))
        # the smaller column under floor 3 changes only the informational footprint share, not the point-support solution
        upper = analyze_floor(SLAB, GEOMETRY, high, unit_case(), mesh_per_bay=MESH, support_model="flexible_beams")
        for key, load in line_loads(uniform).items():
            self.assertTrue(math.isclose(line_loads(upper)[key], load, rel_tol=1e-9, abs_tol=1e-9), key)
        interior = next(c for c in upper["column_direct_loads"] if (c["grid_i"], c["grid_j"]) == (1, 1))
        self.assertTrue(math.isclose(interior["footprint_share_kip_informational"],
                                     (0.15 * gf.SLAB / 12.0 + 0.05) / 144.0 * 22.0 * 22.0, rel_tol=1e-12))


class TransferByFloor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gf.frame():
            mg.install(gf.changed_state())
            cls.description = fs.by_floor()
            ops.wipe()
            cls.record = build_floor_transfers_by_floor(SLAB, GEOMETRY, cls.description, sp.FLOOR_LIVE_LOAD_KSF, mesh_per_bay=MESH)
            mg.install(gf.uniform_state())
            cls.uniform_description = fs.by_floor()
            ops.wipe()
            cls.uniform_record = build_floor_transfers_by_floor(SLAB, GEOMETRY, cls.uniform_description,
                                                                sp.FLOOR_LIVE_LOAD_KSF, mesh_per_bay=MESH)
            ops.wipe()
            cls.legacy = build_floor_transfer(SLAB, GEOMETRY, UNIFORM, sp.FLOOR_LIVE_LOAD_KSF, mesh_per_bay=MESH)
            cls.dead_kip = sp.floor_dead_load_ksf() * AREA_SQFT if False else (0.15 * gf.SLAB / 12.0 + 0.05) * AREA_SQFT
            cls.live_kip = sp.FLOOR_LIVE_LOAD_KSF * AREA_SQFT

    def frame_forces(self, state, record):
        """Gravity (1.2D + 1.6L) on the elastic frame with the given transfer; base reaction and beam forces."""
        with gf.frame():
            if state is not None:
                mg.install(state)
            sp.FLOOR_TRANSFER = record
            ops.wipe()
            build_design_model()
            apply_gravity_loads(floor_factor=1.0, self_weight_factor=1.2, dead_factor=1.2, live_factor=1.6)
            run_gravity_analysis()
            ops.reactions()
            reaction = sum(ops.nodeReaction(node_tag(0, i, j), 3) for j in range(gf.NY + 1) for i in range(gf.NX + 1))
            forces = {tag: list(ops.eleResponse(tag, "localForce")) for tag, *_ in gf.hand_members()}
            return reaction, forces

    def test_floors_are_matched_by_signature_not_by_band(self):
        self.assertEqual(self.record["distinct_floor_count"], 2)
        floors = self.record["floors"]
        self.assertEqual(floors["1"], floors["2"])
        self.assertEqual(floors["3"], floors["4"])
        self.assertNotEqual(floors["1"], floors["3"])
        self.assertEqual(self.uniform_record["distinct_floor_count"], 1)        # four floors, one mechanical floor
        self.assertEqual(len(set(self.uniform_record["floors"].values())), 1)

    def test_every_floor_gets_its_own_transfer_and_the_base_reaction_is_the_hand_total(self):
        reaction, forces = self.frame_forces(gf.changed_state(), self.record)
        members = sum(gf.hand_weight_kip(kind, k, i, j) for _tag, kind, k, i, j in gf.hand_members())
        hand = gf.NZ * (1.2 * self.dead_kip + 1.6 * self.live_kip) + 1.2 * members
        self.assertTrue(math.isclose(reaction, hand, rel_tol=1e-8), (reaction, hand))
        tags = {(kind, k, i, j): tag for tag, kind, k, i, j in gf.hand_members()}
        # Statics of every beam: its two end shears carry exactly what ITS floor's transfer put on it plus its
        # own weight (the bending-couple pairs add no net force). Floors 1 and 3 use different transfers.
        transfers = validate_floor_transfers_by_floor(self.record, GEOMETRY, self.description, gf.SLAB, self.dead_kip, self.live_kip)
        totals = {}
        for (kind, k, i, j), tag in tags.items():
            if kind == "column":
                continue
            key = ("x", j, i) if kind == "beam_x" else ("y", i, j)
            cases = transfers[k]["unit_cases"]
            load = sum(factor * next(b["total_kip"] for b in cases[name]["beams"] if (b["axis"], b["line_index"], b["span_index"]) == key)
                       for name, factor in (("dead", 1.2), ("live", 1.6)))
            expected = load + 1.2 * gf.hand_weight_kip(kind, k, i, j)
            f = forces[tag]
            self.assertTrue(math.isclose(f[2] + f[8], expected, rel_tol=1e-8, abs_tol=1e-8), (kind, k, i, j, f[2] + f[8], expected))
            totals[(kind, k, i, j)] = load
        self.assertGreater(totals[("beam_y", 1, 0, 1)], 1.02 * totals[("beam_y", 3, 0, 1)])     # the stiffer edge line of floors 1-2
        self.assertTrue(math.isclose(totals[("beam_y", 1, 0, 1)], totals[("beam_y", 2, 0, 1)], rel_tol=1e-12))
        # floors that share a signature receive the same transfer
        self.assertIs(validate_floor_transfers_by_floor(self.record, GEOMETRY, self.description, gf.SLAB, self.dead_kip,
                                                        self.live_kip)[1],
                      validate_floor_transfers_by_floor(self.record, GEOMETRY, self.description, gf.SLAB, self.dead_kip,
                                                        self.live_kip)[2])

    def test_uniform_expansion_reproduces_the_uniform_frame_under_the_transfer(self):
        legacy_reaction, legacy = self.frame_forces(None, self.legacy)
        grouped_reaction, grouped = self.frame_forces(gf.uniform_state(), self.uniform_record)
        self.assertTrue(math.isclose(grouped_reaction, legacy_reaction, rel_tol=1e-9))
        for tag, values in legacy.items():
            for a, b in zip(grouped[tag], values):
                self.assertTrue(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-5), (tag, a, b))

    def test_stale_and_partial_transfers_are_refused(self):
        def apply(state, record):
            with gf.frame():
                mg.install(state)
                sp.FLOOR_TRANSFER = record
                ops.wipe()
                build_design_model()
                apply_gravity_loads()

        with self.assertRaises(mg.GroupedStateError):                # one common floor's transfer under a grouped design
            apply(gf.changed_state(), self.legacy)
        with self.assertRaisesRegex(ValueError, "other floors"):     # the transfer of another design
            apply(gf.changed_state(), self.uniform_record)
        missing = copy.deepcopy(self.record)
        del missing["floors"]["4"]
        with self.assertRaisesRegex(ValueError, "other floors"):
            apply(gf.changed_state(), missing)
        edited = copy.deepcopy(self.record)
        sha = edited["floors"]["1"]
        edited["transfers"][sha]["unit_cases"]["dead"]["beams"][0]["node_loads"][0][1] += 0.5
        with self.assertRaises(ValueError):                          # an edited load no longer balances its own summary
            apply(gf.changed_state(), edited)
        relabelled = copy.deepcopy(self.record)
        low, high = relabelled["floors"]["1"], relabelled["floors"]["3"]
        relabelled["transfers"][low], relabelled["transfers"][high] = relabelled["transfers"][high], relabelled["transfers"][low]
        with self.assertRaisesRegex(ValueError, "other beam lines or supports"):
            apply(gf.changed_state(), relabelled)
        another_slab = copy.deepcopy(self.record)
        with self.assertRaises(ValueError):
            with gf.frame(slab=8.0):
                mg.install(gf.changed_state())
                sp.FLOOR_TRANSFER = another_slab
                ops.wipe()
                build_design_model()
                apply_gravity_loads()

    def test_unchanged_floors_are_reused_and_changed_floors_are_solved_again(self):
        with gf.frame():
            mg.install(gf.changed_state())
            ops.wipe()
            again = build_floor_transfers_by_floor(SLAB, GEOMETRY, fs.by_floor(), sp.FLOOR_LIVE_LOAD_KSF, mesh_per_bay=MESH,
                                                   reuse=self.record)
            self.assertEqual(sorted(again["reused_signatures"]), sorted(self.record["transfers"]))
            thicker = {**SLAB, "thickness_in": 7.5}
            ops.wipe()
            fresh = build_floor_transfers_by_floor(thicker, GEOMETRY, fs.by_floor(), sp.FLOOR_LIVE_LOAD_KSF, mesh_per_bay=MESH,
                                                   reuse=self.record)
            self.assertEqual(fresh["reused_signatures"], [])


SLAB_POLICY = {"superimposed_dead_load_ksf": 0.05, "fy_ksi": 60.0, "concrete_unit_weight_kcf": 0.15}
ASSERTIONS = {**{flag: True for flag in ("analysis_applicability_verified", "all_floors_enveloped", "load_pattern_envelope_verified",
                                         "spatial_envelope_per_unit_width", "twisting_moment_resolution_verified",
                                         "zero_membrane_force_verified", "verified", "two_way_shear_path_assessed")},
              "asserted_by": "implementation fixture", "assertion_date": "2026-10-02", "assertion_basis": "implementation fixture"}
# Two levels of the graded recipe and a loose numerical screen: the fixture exercises the plumbing, not convergence.
REFINEMENT = {"recipe": "graded_face_v1", "levels": 2, "max_shells": 45000, "moment_tolerance": 0.9, "shear_tolerance": 0.9,
              "tolerance_basis": "implementation fixture: plumbing only, not a convergence criterion"}


class SlabThickness(unittest.TestCase):
    def descriptions(self):
        with gf.frame():
            mg.install(gf.uniform_state())
            same = fs.by_floor()
            mg.install(gf.changed_state())
            changed = fs.by_floor()
        return same, changed

    def test_equal_floors_give_the_one_beam_screen(self):
        from Design.SMRF_Slab import choose_slab, evaluate_slab
        same, _changed = self.descriptions()
        legacy = choose_slab(GEOMETRY, {k: UNIFORM[k] for k in ("b_beam_in", "h_beam_in", "fc_beam_ksi")}, SLAB_POLICY)
        grouped = choose_slab(GEOMETRY, same, SLAB_POLICY)
        self.assertEqual(grouped["thickness_in"], legacy["thickness_in"])
        self.assertTrue(math.isclose(grouped["required_thickness_in"], legacy["required_thickness_in"], rel_tol=1e-12))
        self.assertEqual(grouped["distinct_floor_count"], 1)
        for a, b in zip(grouped["panels"], legacy["panels"]):
            self.assertEqual(a["panel_id"], b["panel_id"])
            self.assertTrue(math.isclose(a["alpha_fm"], b["alpha_fm"], rel_tol=1e-12))
            self.assertTrue(math.isclose(a["clear_span_x_in"], b["clear_span_x_in"], rel_tol=1e-12))
        self.assertTrue(all(check["status"] == "pass" for check in evaluate_slab(grouped)))

    def test_one_thickness_passes_every_panel_of_every_distinct_floor(self):
        from Design.SMRF_Slab import _beam_inertia, choose_slab, evaluate_slab
        _same, changed = self.descriptions()
        record = choose_slab(GEOMETRY, changed, SLAB_POLICY)
        self.assertEqual(record["distinct_floor_count"], 2)
        self.assertEqual(sorted(k for entry in record["floor_panels"] for k in entry["floors"]), [1, 2, 3, 4])
        for entry in record["floor_panels"]:
            self.assertTrue(all(panel["status"] == "pass" for panel in entry["panels"]))
        low = next(entry for entry in record["floor_panels"] if entry["floors"] == [1, 2])
        corner = next(panel for panel in low["panels"] if panel["panel_id"] == "panel_x1_y1")
        self.assertTrue(math.isclose(corner["clear_span_x_in"], gf.BAY_X - 12.0 - 9.0))      # 24 in edge beam, 18 in interior beam
        self.assertTrue(math.isclose(corner["clear_span_y_in"], gf.BAY_Y - 18.0))
        edge = next(e for e in corner["edges"] if e["side"] == "x_min")
        inertia, _projection, _flange = _beam_inertia(24.0, 32.0, record["thickness_in"], 1, gf.BAY_X)
        self.assertTrue(math.isclose(edge["beam_inertia_in4"], inertia, rel_tol=1e-12))
        self.assertEqual((edge["beam_width_in"], edge["beam_depth_in"]), (24.0, 32.0))
        self.assertTrue(all(check["status"] == "pass" for check in evaluate_slab(record)))
        edited = copy.deepcopy(record)
        edited["floor_panels"][0]["panels"][0]["alpha_fm"] *= 1.01
        self.assertNotEqual(evaluate_slab(edited)[0]["status"], "pass")
        mixed = copy.deepcopy(changed)
        sha = next(iter(mixed["distinct"]))
        mixed["distinct"][sha]["sections"]["fc_beam_ksi"] = 5.0                              # no longer its own signature
        with self.assertRaises(ValueError):
            choose_slab(GEOMETRY, mixed, SLAB_POLICY)


class StripDemandsByFloor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from Design.SMRF_Slab import choose_slab
        from Design.SMRF_Slab_Refinement import build_refined_slab_action_evidence, build_refined_slab_action_evidence_by_floor
        with gf.frame():
            mg.install(gf.changed_state())
            cls.changed = fs.by_floor()
            mg.install(gf.uniform_state())
            cls.same = fs.by_floor()
        cls.slab = choose_slab(GEOMETRY, cls.changed, SLAB_POLICY)
        cls.inputs = {"thickness_in": cls.slab["thickness_in"], "fc_ksi": cls.slab["concrete_fc_ksi"], "fy_ksi": 60.0,
                      "max_aggregate_size_in": 0.75, "exposure": "sheltered_interior", "steel_specification": "ASTM A706",
                      "concrete_type": "normalweight", "panel_ids": sorted(p["panel_id"] for p in cls.slab["panels"]),
                      "num_floor": gf.NZ}
        ops.wipe()
        cls.combined = build_refined_slab_action_evidence_by_floor(cls.slab, GEOMETRY, cls.changed, 0.05, cls.inputs, REFINEMENT,
                                                                   assertions=ASSERTIONS)
        ops.wipe()
        cls.same_combined = build_refined_slab_action_evidence_by_floor(cls.slab, GEOMETRY, cls.same, 0.05, cls.inputs, REFINEMENT,
                                                                        assertions=ASSERTIONS)
        ops.wipe()
        cls.legacy = build_refined_slab_action_evidence(cls.slab, GEOMETRY, UNIFORM, 0.05, cls.inputs, REFINEMENT, assertions=ASSERTIONS)

    def test_the_combination_is_the_envelope_of_every_distinct_floor_and_is_verified_only_as_a_whole(self):
        from Design.SMRF_Slab_Actions import _FLAGS
        from Design.SMRF_Slab_Refinement import refinement_verified
        combined = self.combined
        self.assertEqual((combined["distinct_floor_count"], combined["floors_served"]), (2, [1, 2, 3, 4]))
        self.assertEqual(combined["refinement"]["status"], "passed")
        self.assertTrue(refinement_verified(combined))
        self.assertTrue(all(combined[flag] is True for flag in _FLAGS))
        for entry in combined["by_floor"]:
            single = entry["evidence"]
            self.assertIs(single["all_floors_enveloped"], False)          # one floor never claims to represent the building
            self.assertIn("face_widths_in", single["refinement"]["recipe_inputs"])
            rows = {(r["panel_id"], r["axis"], r["face"]): r for r in single["strips"]}
            for row in combined["strips"]:
                key = (row["panel_id"], row["axis"], row["face"])
                self.assertGreaterEqual(row["mu_kip_in_per_ft"], rows[key]["mu_kip_in_per_ft"])
                self.assertGreaterEqual(row["vu_kip_per_ft"], rows[key]["vu_kip_per_ft"])
        sources = {tuple(row["moment_source_floor"]["floors"]) for row in combined["strips"]}
        self.assertEqual(sources, {(1, 2), (3, 4)})                       # both floors govern somewhere

    def test_equal_floors_reproduce_the_one_floor_demands(self):
        self.assertEqual(self.same_combined["distinct_floor_count"], 1)
        legacy = {(r["panel_id"], r["axis"], r["face"]): r for r in self.legacy["strips"]}
        for row in self.same_combined["strips"]:
            other = legacy[(row["panel_id"], row["axis"], row["face"])]
            self.assertTrue(math.isclose(row["mu_kip_in_per_ft"], other["mu_kip_in_per_ft"], rel_tol=1e-8, abs_tol=1e-9))
            self.assertTrue(math.isclose(row["vu_kip_per_ft"], other["vu_kip_per_ft"], rel_tol=1e-8, abs_tol=1e-9))

    def test_the_strip_routine_takes_the_combination_and_refuses_a_partial_or_edited_one(self):
        from Design.SMRF_Slab_Refinement import combine_floor_evidence, refinement_verified
        from Design.SMRF_Slab_Reinforcement import design_slab_reinforcement
        record = design_slab_reinforcement(self.inputs, self.combined)
        self.assertIsNotNone(record["layout"])
        one_floor = self.combined["by_floor"][0]["evidence"]
        refused = design_slab_reinforcement(self.inputs, one_floor)
        self.assertIsNone(refused["layout"])
        self.assertIn("all_floors_enveloped", refused["checks"][0]["details"]["reason"])
        partial = combine_floor_evidence(self.combined["by_floor"][:1], gf.NZ)
        self.assertIs(partial["all_floors_enveloped"], False)
        self.assertFalse(refinement_verified(partial))
        self.assertIsNone(design_slab_reinforcement(self.inputs, partial)["layout"])
        edited = copy.deepcopy(self.combined)
        edited["strips"][0]["mu_kip_in_per_ft"] *= 0.5
        self.assertFalse(refinement_verified(edited))
        self.assertIsNone(design_slab_reinforcement(self.inputs, edited)["layout"])
        twice = combine_floor_evidence(self.combined["by_floor"] + self.combined["by_floor"][:1], gf.NZ)
        self.assertFalse(refinement_verified(twice))


class CoupledReference(unittest.TestCase):
    """The monolithic slab/web/column model resolves every floor's beam lines and every story's columns."""
    CASE = {"id": "gravity", "dead_factor": 1.2, "live_factor": 1.6, "live_load_ksf": 0.05, "live_pattern": "all"}

    def by_member(self, state):
        from Design.SMRF_Coupled_Comparison import coupled_sections_by_member
        with gf.frame():
            mg.install(state)
            return coupled_sections_by_member()

    def solve(self, sections):
        from Design.SMRF_Coupled_Analysis import analyze_coupled_gravity
        ops.wipe()
        result = analyze_coupled_gravity(SLAB, GEOMETRY, sections, [self.CASE] * gf.NZ, mesh_per_bay=4)
        ops.wipe()
        self.assertEqual(result["status"], "diagnostic_complete")
        return result

    def test_equal_members_solve_to_the_one_section_model(self):
        legacy = self.solve({**UNIFORM, "fc_col_ksi": gf.COLUMN.fc_ksi})
        grouped = self.solve(self.by_member(gf.uniform_state()))
        a = {(r["grid_i"], r["grid_j"]): r["force_moment"] for r in legacy["base_reactions"]}
        b = {(r["grid_i"], r["grid_j"]): r["force_moment"] for r in grouped["base_reactions"]}
        for key, values in a.items():
            for x, y in zip(b[key], values):
                self.assertTrue(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-7), (key, x, y))
        for key, value in legacy["weight_ledger"].items():
            self.assertTrue(math.isclose(grouped["weight_ledger"][key], value, rel_tol=1e-12))

    def test_changed_groups_reach_the_webs_the_columns_and_the_weight_ledger(self):
        sections = self.by_member(gf.changed_state())
        self.assertEqual(sections["floors"][0]["beam_lines"]["y"][0], {"b_in": 24.0, "h_in": 32.0, "fc_ksi": 4.0})
        self.assertEqual(sections["floors"][2]["supports"][1][1], {"b_in": 22.0, "h_in": 22.0})       # story 3 interior column
        self.assertEqual(sections["floors"][1]["supports"][1][1], {"b_in": 26.0, "h_in": 26.0})       # story 2: the column BELOW floor 2
        result = self.solve(sections)
        self.assertTrue(result["equilibrium"]["numerical_balance_passed"])
        columns = sum(gf.hand_weight_kip(kind, k, i, j) for _t, kind, k, i, j in gf.hand_members() if kind == "column")
        beams = sum(gf.hand_weight_kip(kind, k, i, j) for _t, kind, k, i, j in gf.hand_members() if kind != "column")
        self.assertTrue(math.isclose(result["weight_ledger"]["column_weight_kip"], 1.2 * columns, rel_tol=1e-12))
        self.assertTrue(math.isclose(result["weight_ledger"]["beam_drop_weight_kip"], 1.2 * beams, rel_tol=1e-12))
        with self.assertRaises(ValueError):                       # a floor missing from the description
            self.solve({**sections, "floors": sections["floors"][:-1]})

    def test_the_frame_with_its_per_floor_transfer_carries_the_coupled_models_total(self):
        from Design.SMRF_Coupled_Comparison import compare_transfer_to_coupled
        with gf.frame():
            mg.install(gf.changed_state())
            ops.wipe()
            sp.FLOOR_TRANSFER = build_floor_transfers_by_floor(SLAB, GEOMETRY, fs.by_floor(), sp.FLOOR_LIVE_LOAD_KSF, mesh_per_bay=MESH)
            comparison = compare_transfer_to_coupled(SLAB, mesh_per_bay=4)
        # the two models hold the same slab pressure and the same member-by-member weight ledger
        self.assertLess(comparison["total_relative_error"], 1e-9)
        self.assertEqual(len(comparison["columns"]), (gf.NX + 1) * (gf.NY + 1))
        self.assertLess(comparison["max_column_vertical_relative_difference"], 0.10)


class DiagnosticModulesRefuse(unittest.TestCase):
    """Floor diagnostics that model one beam and one column say so instead of taking one line's section for all."""

    def test_by_line_floors_are_refused(self):
        from Design import SMRF_Cut_Regions, SMRF_Floor_Compatibility
        with gf.frame():
            mg.install(gf.changed_state())
            by_line = fs.for_floor(1)
        with self.assertRaisesRegex(ValueError, "not wired for a grouped design"):
            SMRF_Floor_Compatibility.CompatibleFloor(SLAB, GEOMETRY, by_line)
        with self.assertRaisesRegex(ValueError, "not wired for a grouped design"):
            SMRF_Cut_Regions.effective_flange_regions(GEOMETRY, by_line, SLAB, axis="x", slab_perimeter="centerlines")


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
