"""System rules of the perimeter-SFRS basis (2026-10-16): tied reinforcement selection, the strong-column rule in the
perimeter-frame directions only, Chapter 15 joint shear outside the SFRS, tie-governing hoops, SFRS-only story
strength lines and the ACI 318-19 18.14 gravity-frame evidence of the capacity design."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402
import Structure_Parameters as sp                                               # noqa: E402
from Design import Gravity_Frame_Members as gravity_rules                        # noqa: E402
from Design import Group_Capacity as capacity_design                             # noqa: E402
from Design import Group_Selection as selection                                  # noqa: E402
from Design import Grouped_Design as grouped                                     # noqa: E402
from Design import Perimeter_Systems as systems                                  # noqa: E402
from Design.Config import DesignConfig                                           # noqa: E402
from Design.SMRF_Demands import strength_load_combinations                       # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402

GRAVITY_GROUPS = {"s01_04__column__interior", "s01_04__beam_x__interior", "s01_04__beam_y__interior"}
KEPT = {"gravity_1.4D", "gravity_1.2D_1.6L", "seismic_high_gravity_X_x+1_y+1", "seismic_high_gravity_Y_x+1_y+1",
        "seismic_low_gravity_X_x-1_y-1", "seismic_low_gravity_Y_x-1_y-1"}


# The perimeter frames carry the whole lateral load under this basis. At the fixture's SDS of 1.0 the two perimeter
# frames of this 2 x 3 bay frame cannot thread enough beam bars through any column within the 32 in cap, so the
# rules test runs the fixture sections (26 in columns, 18 x 28 beams, grades at the first rungs of their ladders)
# at a seismic demand the perimeter carries: SDS 0.6, SD1 0.36 (still SDC D). Nothing here sizes a structure.
RULES_SEISMIC = {"ASCE_SDS": 0.6, "ASCE_SD1": 0.36, "ASCE_S1": 0.36}
COLUMN = replace(gf.COLUMN, fc_ksi=6.0)
BEAM = replace(gf.BEAM, fc_ksi=5.0)


def system_state():
    return mg.GroupedDesign.uniform(gf.NX, gf.NY, gf.NZ, COLUMN, BEAM, policy_id=mg.SYSTEM_POLICY_ID)


def few_combinations():
    """A subset of the strength combinations (both gravity families, both axes, both signs) to keep the tests short."""
    return [c for c in strength_load_combinations(sp.ASCE_SDS, redundancy_factor=1.3) if c["id"] in KEPT]


def solve_actions(state, combinations):
    period = systems.model_period(state)
    solver = systems.CombinationSolver(period, None, state)
    actions = []
    for combination in combinations:
        solved = solver.analyze(combination)
        actions.append({**combination, "analysis_succeeded": True, "axial_reference": "joint_faces",
                        "analysis_models": solved["models"], "members": solved["members"]})
    return period, actions


def one_design_per_tie(state):
    return all(len({replace(state.designs[g], member_type="x").section_key() for g in gids}) == 1 for gids in state.ties().values())


class SystemRules(unittest.TestCase):
    def test_ties_scwb_directions_joint_rules_hoops_and_gravity_evidence(self):
        with gf.frame(**RULES_SEISMIC):
            state = system_state()
            mg.install(state)
            cfg = DesignConfig()
            combinations = few_combinations()
            period, actions = solve_actions(state, combinations)
            expected_ids = [c["id"] for c in combinations]
            compatibility = systems.compatibility_run(period, None, cfg, state)
            # selection by tie
            selected, report = selection.select_reinforcement(state, actions, cfg)
            self.assertEqual(report["rules"], selection.SYSTEM_RULES_ID)
            self.assertEqual(report["units"], {"beams": ["gravity_beam", "sfrs_beam"], "columns": ["sfrs_column", "gravity_column"]})
            self.assertTrue(one_design_per_tie(selected))
            final = report["passes"][-1]
            self.assertEqual(set(final["beams"]), {"gravity_beam", "sfrs_beam"})
            self.assertEqual(final["beams"]["sfrs_beam"]["groups"], ["s01_04__beam_x__edge", "s01_04__beam_y__edge"])
            self.assertEqual(set(final["columns"]), {"sfrs_column", "gravity_column"})
            self.assertEqual(final["columns"]["sfrs_column"]["chain"],
                             [["s01_04__column__corner", "s01_04__column__x_boundary_edge", "s01_04__column__y_boundary_edge"]])
            self.assertEqual(final["columns"]["gravity_column"]["chain"], [["s01_04__column__interior"]])
            self.assertEqual(report["exhausted_groups"], [])
            self.assertEqual(grouped.invalid_beam_row_groups(selected), [])
            # capacity design: joint rules by joint direction
            mg.install(selected)
            capacity = capacity_design.build_group_capacity_design(actions, expected_ids, cfg, None, selected,
                                                                   compatibility=compatibility)
            by_core = {}
            for type_id, kind in capacity["joint_types"].items():
                by_core.setdefault(kind["core_group"], set()).add(tuple(sorted(capacity["joint_rules"][type_id].items())))
            self.assertEqual(by_core["s01_04__column__interior"], {(("x", "chapter_15"), ("y", "chapter_15"))})
            self.assertEqual(by_core["s01_04__column__corner"], {(("x", "18.8"), ("y", "18.8"))})
            self.assertEqual(by_core["s01_04__column__x_boundary_edge"], {(("x", "chapter_15"), ("y", "18.8"))})
            self.assertEqual(by_core["s01_04__column__y_boundary_edge"], {(("x", "18.8"), ("y", "chapter_15"))})
            for kind in capacity["joint_types"].values():
                for item in kind["shear"].values():
                    if item["rule"] == "chapter_15":
                        self.assertEqual(item["steel_stress_factor"], 1.0)
                        self.assertIn("15.4.2", item["clause"])
                        self.assertEqual(item["beam_moment_basis"], "nominal (fy)")
                    else:
                        self.assertEqual(item["steel_stress_factor"], 1.25)
                        self.assertEqual(item["clause"], "ACI 318-19 18.8.4")
            shear_checks = [c for c in capacity["checks"] if c["id"] == "joint.shear_screen"]
            self.assertTrue(any(c["details"]["rule"] == "chapter_15" for c in shear_checks))
            self.assertTrue(any(c["details"]["rule"] == "18.8" for c in shear_checks))
            # the strong-column rule is required only where the column is part of a perimeter frame
            required = {(j["grid_i"], j["grid_j"], axis): states["required"]
                        for j in capacity["scwb"]["joints"] for axis, states in j["directions"].items()}
            self.assertFalse(required[(1, 1, "x")])
            self.assertFalse(required[(1, 1, "y")])
            self.assertTrue(required[(0, 0, "x")])
            self.assertTrue(required[(0, 0, "y")])
            self.assertTrue(required[(0, 1, "y")])          # x-boundary edge column: frame along y
            self.assertFalse(required[(0, 1, "x")])
            self.assertTrue(required[(1, 0, "x")])          # y-boundary edge column: frame along x
            self.assertFalse(required[(1, 0, "y")])
            inputs = grouped.joint_check_inputs(capacity, selected)
            self.assertTrue(inputs["scwb_not_required"])
            interior_joint = next(j for j in inputs["joints"] if (j["grid_i"], j["grid_j"]) == (1, 1))
            self.assertEqual({axis: states["required"] for axis, states in interior_joint["directions"].items()}, {"x": False, "y": False})
            # the joint evaluator skips those directions: no strong-column check, open or otherwise, at an interior joint
            from Design.SMRF_Joints import evaluate_joints
            scwb_locations = {c["location"] for c in evaluate_joints(inputs) if c["id"] == "scwb"}
            self.assertTrue(scwb_locations)
            self.assertFalse(any(loc.startswith(interior_joint["id"] + "/") for loc in scwb_locations))
            self.assertFalse(any(loc.endswith("/x/positive") and loc.startswith(j["id"] + "/") for j in inputs["joints"]
                                 if (j["grid_i"], j["grid_j"]) == (0, 1) for loc in scwb_locations))      # edge column: x not required
            # the gravity-frame evidence
            gravity = capacity["gravity_frame"]
            self.assertTrue(gravity["compatibility_calculated"])
            self.assertEqual(set(gravity["groups"]), GRAVITY_GROUPS)
            for gid, row in gravity["groups"].items():
                self.assertIn(row["branch"], gravity_rules.BRANCHES)
                self.assertEqual(row["induced"]["branch"], row["branch"])
                kind = "column" if row["member_type"] == "column" else "beam"
                self.assertTrue(any(c["id"] == f"gravity.{kind}_compatibility_branch" and c["status"] == "pass" for c in row["checks"]))
                self.assertGreater(row["induced"]["moment_ratio"], 0.0)
            ids = {c["id"] for c in capacity["checks"]}
            self.assertIn("gravity.column_hoop_spacing_full_length", ids)
            self.assertIn("gravity.beam_transverse_spacing", ids)
            self.assertIn("gravity.column_capacity_shear", ids)
            self.assertEqual(set(systems.gravity_induced_from_capacity(capacity)), GRAVITY_GROUPS)
            stage = systems.compatibility_stage(compatibility, systems.gravity_induced_from_capacity(capacity))
            self.assertEqual(set(stage["groups"]), GRAVITY_GROUPS)
            # hoop updates keep one design per tie
            updates = capacity_design.hoop_updates(capacity, selected)
            for gid, design in updates.items():
                self.assertEqual(design.member_type, selected.groups[gid]["member_type"])
            after = selected.with_designs(updates) if updates else selected
            self.assertTrue(one_design_per_tie(after))
            # story strength: the perimeter lines only
            strength = grouped.story_strength_distribution(cfg)
            x = strength["stories"][0]["by_direction"]["x"]
            self.assertEqual(x["line_systems"], ["sfrs", "gravity", "gravity", "sfrs"])
            self.assertEqual(x["line_story_strength_kip"][1], 0.0)
            self.assertGreater(x["line_story_strength_kip"][0], 0.0)
            self.assertLessEqual(x["one_side_fraction"], 0.5 + 1e-9)
            y = strength["stories"][0]["by_direction"]["y"]
            self.assertEqual(y["line_systems"], ["sfrs", "gravity", "sfrs"])
            self.assertIn("perimeter", strength["lines_counted"])

    def test_the_band_policy_selection_is_unchanged(self):
        with gf.frame(**RULES_SEISMIC):
            state = gf.uniform_state()
            mg.install(state)
            cfg = DesignConfig()
            combinations = few_combinations()
            period, actions = solve_actions(state, combinations)
            selected, report = selection.select_reinforcement(state, actions, cfg)
            self.assertEqual(report["rules"], selection.RULES_ID)
            self.assertEqual(report["units"]["columns"], list(mg.COLUMN_LOCATIONS))
            self.assertTrue(all(gid in state.groups for gid in report["units"]["beams"]))
            final = report["passes"][-1]
            self.assertEqual(set(final["columns"]), set(mg.COLUMN_LOCATIONS))
            self.assertEqual(final["columns"]["corner"]["chain"], [["s01_02__column__corner"], ["s03_04__column__corner"]])
            self.assertEqual(final["beams"]["s01_02__beam_x__edge"]["group_id"], "s01_02__beam_x__edge")
            mg.install(selected)
            capacity = capacity_design.build_group_capacity_design(actions, [c["id"] for c in combinations], cfg, None, selected)
            self.assertIsNone(capacity["gravity_frame"])
            self.assertTrue(all(rule == "18.8" for rules in capacity["joint_rules"].values() for rule in rules.values()))
            self.assertTrue(all(states["required"] for j in capacity["scwb"]["joints"] for states in j["directions"].values()))
            self.assertEqual(grouped.joint_check_inputs(capacity, selected)["scwb_not_required"], [])
            self.assertEqual(capacity_design.hoop_updates(capacity, selected),
                             capacity_design._tied_hoop_updates(capacity_design.hoop_updates(capacity, selected), selected))
            strength = grouped.story_strength_distribution(cfg)
            self.assertEqual(strength["lines_counted"], "every frame line")
            self.assertTrue(all(v > 0 for v in strength["stories"][0]["by_direction"]["x"]["line_story_strength_kip"]))


if __name__ == "__main__":
    unittest.main()
