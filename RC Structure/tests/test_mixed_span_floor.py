"""Mixed sections on one beam line: actual solver/load-path and case-0003 regressions.

Software diagnostics, not structural design qualification. Small asymmetric floors exercise
both directions, centroid steps, physical slab faces, conservation and stale-transfer refusal.
"""
import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf
import openseespy.opensees as ops
from Model import Member_Groups as mg
from Design import SMRF_Floor_Sections as fs
from Design.SMRF_Floor_Analysis import analyze_floor
from Design.SMRF_Floor_Mesh import floor_mesh, resolve_recipe_plan, nested_refinement, witness_face_band_extension
from Design.SMRF_Floor_Transfer import build_floor_transfer, validate_floor_transfer
from Design.SMRF_Slab import _panels_by_line, choose_slab
from Design.SMRF_Coupled_Analysis import analyze_coupled_gravity, BY_MEMBER_SCHEMA

G = dict(num_bay_x=2, num_bay_y=2, num_floor=1, bay_x_in=240., bay_y_in=216., story_h_in=168.)
SLAB = dict(thickness_in=7., concrete_fc_ksi=4., concrete_unit_weight_kcf=.15, superimposed_dead_load_ksf=.05)
CASE = dict(id="gravity", dead_factor=1., live_factor=0., live_load_ksf=.05, live_pattern="none")


def floor(mixed=True):
    beam = dict(b_in=16., h_in=30., fc_ksi=4.)
    result = dict(schema=fs.SPAN_SCHEMA, fc_beam_ksi=4.,
                  beam_lines={a: [[dict(beam) for _ in range(2)] for _ in range(3)] for a in ("x", "y")},
                  supports=[[dict(b_in=24., h_in=24.) for _ in range(3)] for _ in range(3)])
    if mixed:
        result["beam_lines"]["x"][1][0].update(b_in=22., h_in=36.)
        result["beam_lines"]["y"][1][1].update(b_in=20., h_in=34.)
    return result


def mirror(s):
    s = copy.deepcopy(s)
    for row in s["beam_lines"]["x"]:
        row.reverse()
    s["beam_lines"]["y"].reverse()
    for row in s["supports"]:
        row.reverse()
    return s


class MixedSpans(unittest.TestCase):
    def tearDown(self):
        ops.wipe()

    def test_case0003_core_ring_keeps_actual_spans_and_story_signatures(self):
        # The geometry/material/section pattern of the failed candidate (no expensive design search).
        col = replace(gf.COLUMN, b_in=22., h_in=22., fc_ksi=9.)
        beam = replace(gf.BEAM, b_in=16., h_in=30., fc_ksi=8.)
        with gf.frame(NUM_BAY_X=7, NUM_BAY_Y=4, NUM_FLOOR=5, BAY_X=252., BAY_Y=342., STORY_H=192.):
            state = mg.GroupedDesign.uniform(7, 4, 5, col, beam, policy_id=mg.SYSTEM_CORE_BANDED_POLICY_ID)
            updates = {tie: replace(d, h_in=32.) for tie, d in state.tie_designs().items() if "sfrs_beam" in tie}
            updates.update({tie: replace(d, b_in=23., h_in=23.) for tie, d in state.tie_designs().items() if "sfrs_column" in tie})
            mg.install(state.with_tied_designs(updates))
            desc = fs.by_floor()
            s = fs.for_floor(1)
            self.assertEqual(s["schema"], fs.SPAN_SCHEMA)
            self.assertEqual({b["h_in"] for b in s["beam_lines"]["x"][1]}, {30., 32.})
            for axis, count, spans in (("x", 5, 7), ("y", 8, 4)):
                for line in range(count):
                    for span in range(spans):
                        member = mg.beam_at("beam_" + axis, 1, span if axis == "x" else line,
                                            line if axis == "x" else span)
                        actual = fs.beam_line(s, axis, line, span)
                        self.assertEqual((actual["b_in"], actual["h_in"]), (member.design.b_in, member.design.h_in))
            self.assertEqual(len(desc["distinct"]), 1)
            geom = dict(G, num_bay_x=7, num_bay_y=4, num_floor=5, bay_x_in=252., bay_y_in=342., story_h_in=192.)
            self.assertTrue(choose_slab(geom, desc, {})["thickness_screen_passed"])

    def test_span_identity_validation_and_panel_edges_are_local(self):
        s = floor()
        fs.validate(s, 2, 2)
        with self.assertRaisesRegex(ValueError, "span_index"):
            fs.beam_line(s, "x", 1)
        self.assertIsNone(fs.uniform_equivalent(s))
        self.assertNotEqual(fs.signature(s), fs.signature(mirror(s)))
        self.assertEqual(fs.panel_face_widths(s, 0, 0), {"x": (16., 16.), "y": (16., 22.)})
        self.assertEqual(fs.panel_face_widths(s, 1, 1), {"x": (20., 16.), "y": (16., 16.)})
        panels = {(p["bay_x_index"], p["bay_y_index"]): p for p in _panels_by_line(G, s, {"fy_ksi": 60.}, 7.)}
        edge = lambda p: next(e for e in p["edges"] if e["side"] == "y_max")
        self.assertEqual(edge(panels[1, 1])["beam_depth_in"], 36.)
        self.assertEqual(edge(panels[2, 1])["beam_depth_in"], 30.)
        self.assertEqual(panels[1, 1]["clear_span_y_in"], 216. - (16.+22.)/2)
        bad = copy.deepcopy(s)
        bad["beam_lines"]["x"][1].pop()
        with self.assertRaises(ValueError): fs.validate(bad, 2, 2)
        bad = floor()
        bad["beam_lines"]["y"][1][0]["h_in"] = float("inf")
        with self.assertRaises(ValueError): fs.validate(bad, 2, 2)

    def test_mixed_stiffness_changes_load_path_and_preserves_reflected_equilibrium(self):
        results = [analyze_floor(SLAB, G, s, CASE, mesh_per_bay=4, support_model="flexible_beams")
                   for s in (floor(False), floor(), mirror(floor()))]
        for result in results:
            self.assertEqual(result["status"], "transfer_complete")
            self.assertTrue(result["equilibrium"]["numerical_balance_passed"])
            self.assertTrue(result["transfer_equilibrium"]["numerical_balance_passed"])
        loads = lambda r: {(b["axis"], b["line_index"], b["span_index"]): b["total_kip"] for b in r["beam_transfer"]}
        a, b, c = map(loads, results)
        self.assertGreater(max(abs(b[k]-a[k]) for k in a), .1)
        for (axis, line, span), value in b.items():
            reflected = (axis, line, 1-span) if axis == "x" else (axis, 2-line, span)
            self.assertAlmostEqual(value, c[reflected], places=7)
        properties = results[1]["beam_model"]["lines"]["x"][1]
        self.assertGreater(properties[0]["iy_in4"], properties[1]["iy_in4"])
        self.assertEqual([p["area_in2"] for p in properties], [22.*36., 16.*30.])

    def test_mesh_retains_every_width_face_and_reports_real_budget(self):
        s = floor()
        policy = dict(recipe="graded_face_v1", levels=3, max_shells=130000)
        plan = resolve_recipe_plan(G, s, policy)
        replay = resolve_recipe_plan(G, {"face_widths_in": plan["inputs"]["face_widths_in"]}, policy)
        self.assertEqual(plan, replay)
        grids = [floor_mesh(2, 2, 240., 216., 4, mesh_spec=m) for m in plan["meshes"]]
        for grid in grids:
            for axis, count, length in (("x", 2, 240.), ("y", 2, 216.)):
                coords = grid[axis + "_coordinates_in"]
                for line in range(3):
                    for span in range(2):
                        w = fs.beam_width(s, "y" if axis == "x" else "x", line, span)
                        if line: self.assertIn(line * length - w / 2, coords)
                        if line < count: self.assertIn(line * length + w / 2, coords)
        for a, b in zip(grids, grids[1:]): nested_refinement(a, b)
        limited = resolve_recipe_plan(G, s, dict(policy, max_shells=grids[0]["shell_count"]))
        self.assertEqual(limited["status"], "unresolved_budget")
        self.assertEqual(len(limited["meshes"]), 1)
        self.assertEqual(limited["meshes"][0]["x_offsets_in"], plan["meshes"][0]["x_offsets_in"])

    def test_saved_transfer_rejects_one_changed_span(self):
        s = floor()
        transfer = build_floor_transfer(SLAB, G, s, .05, mesh_per_bay=4)
        area = 2*2*240.*216./144.
        args = (7., (.15*7./12.+.05)*area, .05*area)
        validate_floor_transfer(transfer, G, s, *args)
        edited = copy.deepcopy(s)
        edited["beam_lines"]["x"][1][1]["h_in"] += 1.
        with self.assertRaisesRegex(ValueError, "other beam lines"):
            validate_floor_transfer(transfer, G, edited, *args)

    def test_strip_demands_recover_local_faces_without_claiming_qualification(self):
        from Design.SMRF_Slab_Actions import build_slab_action_evidence
        inputs = dict(thickness_in=7., fc_ksi=4., fy_ksi=60., max_aggregate_size_in=.75,
                      exposure="sheltered_interior", steel_specification="ASTM A706", concrete_type="normalweight",
                      num_floor=1, panel_ids=[f"panel_x{i}_y{j}" for i in (1, 2) for j in (1, 2)])
        results = []
        for sections in (floor(), mirror(floor())):
            evidence = build_slab_action_evidence(SLAB, G, sections, .05, inputs, mesh_per_bay=4,
                                                  uniform_all_floors=False)
            self.assertFalse(evidence["verified"])
            self.assertTrue(all(r["numerical_balance_passed"] for r in evidence["equilibrium"]))
            self.assertTrue(all(r["complete"] for r in evidence["physical_face_coverage"]))
            self.assertEqual(len(evidence["strips"]), 16)
            results.append({(r["panel_id"], r["axis"], r["face"]): r for r in evidence["strips"]})
        for (panel, axis, face), row in results[0].items():
            i, j = int(panel[7]), int(panel[-1])
            other = results[1][f"panel_x{3-i}_y{j}", axis, face]
            self.assertAlmostEqual(row["mu_kip_in_per_ft"], other["mu_kip_in_per_ft"], places=7)

    def test_mixed_width_local_refinement_retains_faces_and_nested_nodes(self):
        from test_slab_local_refinement import action
        from Design.SMRF_Slab_Actions import compare_slab_action_refinement
        plan = resolve_recipe_plan(G, floor(), dict(recipe="graded_face_v1", levels=2, max_shells=130000))
        grid = floor_mesh(2, 2, 240., 216., 4, mesh_spec=plan["meshes"][-1])
        location = dict(x_in=120., y_in=205.)  # face of the 22-in beam bounding panel (0,0)
        a, b = action(dict(grid, solver="coarse"), 1., location), action(grid, 1.1, location)
        comparison = compare_slab_action_refinement(a, b, moment_tolerance=.05, shear_tolerance=.05)
        extension = witness_face_band_extension(G, plan["inputs"], a, b, comparison, grid)
        self.assertEqual(extension["status"], "proposed")
        self.assertEqual(extension["selected_axis"], "y")
        nested_refinement(grid, floor_mesh(2, 2, 240., 216., 4, mesh_spec=extension["mesh_spec"]))

    def test_coupled_centroid_steps_have_hand_weight_and_reflected_reactions(self):
        results = []
        for s in (floor(), mirror(floor())):
            result = analyze_coupled_gravity(SLAB, G, dict(schema=BY_MEMBER_SCHEMA, fc_col_ksi=5., floors=[s]),
                                             [CASE], mesh_per_bay=3)
            self.assertEqual(result["status"], "diagnostic_complete")
            self.assertTrue(result["equilibrium"]["numerical_balance_passed"])
            # Beam drops between faces of identical 24-inch columns, counted once.
            weight = sum(b["b_in"]*(b["h_in"]-7.)*((240. if a == "x" else 216.)-24.)*.15/1728.
                         for a in ("x", "y") for row in s["beam_lines"][a] for b in row)
            self.assertAlmostEqual(result["weight_ledger"]["beam_drop_weight_kip"], weight, places=10)
            results.append({(r["grid_i"], r["grid_j"]): r["force_moment"][2] for r in result["base_reactions"]})
        for (i, j), value in results[0].items(): self.assertAlmostEqual(value, results[1][2-i, j], places=7)


if __name__ == "__main__":
    unittest.main()
