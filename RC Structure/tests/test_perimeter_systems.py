"""The two-system analysis of the perimeter-SFRS basis (2026-10-16): released elastic design model, superposition
by member system, the compatibility run and the 18.14.3 branches of the gravity groups."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402
import openseespy.opensees as ops                                               # noqa: E402
import Structure_Parameters as sp                                               # noqa: E402
from Analysis.Gravity import run_gravity_analysis                                # noqa: E402
from Design import Design_Driver as driver                                       # noqa: E402
from Design import Gravity_Frame_Members as gravity_rules                        # noqa: E402
from Design import Group_Checks as checks                                        # noqa: E402
from Design import Perimeter_Systems as systems                                  # noqa: E402
from Design.Config import DesignConfig                                           # noqa: E402
from Design.SMRF_Demands import strength_load_combinations                       # noqa: E402
from Design.SMRF_Elastic import build_design_model                               # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402
from Model.Build_Model import floor_master_node                                  # noqa: E402

GRAVITY_GROUPS = {"s01_04__column__interior", "s01_04__beam_x__interior", "s01_04__beam_y__interior"}


def system_state():
    return mg.GroupedDesign.uniform(gf.NX, gf.NY, gf.NZ, gf.COLUMN, gf.BEAM, policy_id=mg.SYSTEM_POLICY_ID)


def _end_moment(forces):
    """The larger end moment of a member (kip-in), either bending axis, from its local force vector."""
    return max(abs(forces[4]), abs(forces[10]), abs(forces[5]), abs(forces[11]))


def _push_roof(force_kip):
    ops.timeSeries("Linear", 901)
    ops.pattern("Plain", 901, 901)
    ops.load(floor_master_node(sp.NUM_FLOOR), force_kip, 0.0, 0.0, 0.0, 0.0, 0.0)
    run_gravity_analysis()


class ReleasedDesignModel(unittest.TestCase):
    def test_interior_beams_are_pinned_only_in_the_perimeter_model(self):
        with gf.frame():
            mg.install(system_state())
            interior = mg.beam_at("beam_x", 2, 0, 1).member_tag            # an x beam on the interior line j = 1
            perimeter = mg.beam_at("beam_x", 2, 0, 0).member_tag           # the same bay on the perimeter line j = 0
            moments = {}
            for model in ("all_frames", "perimeter_sfrs"):
                with systems.lateral_model(model):
                    ops.wipe()
                    build_design_model()
                    _push_roof(50.0)
                    moments[model] = {"interior": _end_moment(ops.eleResponse(interior, "localForce")),
                                      "perimeter": _end_moment(ops.eleResponse(perimeter, "localForce"))}
                    ops.wipe()
            self.assertEqual(sp.LATERAL_DESIGN_MODEL, "all_frames")
            self.assertGreater(moments["all_frames"]["interior"], 0.2 * moments["all_frames"]["perimeter"])
            # a release code pins the end exactly: the released beam reads zero moment
            self.assertLess(moments["perimeter_sfrs"]["interior"], 1e-9 * moments["perimeter_sfrs"]["perimeter"])
            self.assertGreater(moments["perimeter_sfrs"]["perimeter"], moments["all_frames"]["perimeter"])

    def test_the_band_policy_keeps_the_continuous_model(self):
        with gf.frame():
            state = gf.uniform_state()
            self.assertFalse(systems.two_system(state))
            self.assertEqual(systems.design_model_for(state), "all_frames")
            mg.install(system_state())
            self.assertTrue(systems.two_system())
            self.assertEqual(systems.design_model_for(), "perimeter_sfrs")
        with self.assertRaises(ValueError):
            with systems.lateral_model("no_such_model"):
                pass


class Superposition(unittest.TestCase):
    def test_seismic_actions_superpose_by_member_system(self):
        with gf.frame():
            state = system_state()
            mg.install(state)
            period = systems.model_period(state)
            combinations = strength_load_combinations(sp.ASCE_SDS, redundancy_factor=1.3)
            seismic = next(c for c in combinations if c["id"] == "seismic_high_gravity_X_x+1_y+1")
            solver = systems.CombinationSolver(period, None, state)
            solved = solver.analyze(seismic)
            self.assertEqual(solver.solves, 3)
            self.assertEqual(solved["models"], {mg.SYSTEM_SFRS: systems.MODEL_SFRS, mg.SYSTEM_GRAVITY: systems.MODEL_GRAVITY})
            self.assertIsNotNone(solved["elf"])
            # the three solves by hand
            gravity_only = {**seismic, "ex": 0.0, "ey": 0.0}
            with systems.lateral_model("all_frames"):
                driver._analyze_combination(gravity_only, period, None)
                continuous = checks.capture_member_actions(seismic["dead"])
            with systems.lateral_model("perimeter_sfrs"):
                driver._analyze_combination(gravity_only, period, None)
                released_gravity = checks.capture_member_actions(seismic["dead"])
                driver._analyze_combination(seismic, period, None)
                released = checks.capture_member_actions(seismic["dead"])
            systems_of = systems.member_systems()
            lateral_increment = {mg.SYSTEM_SFRS: 0.0, mg.SYSTEM_GRAVITY: 0.0}
            for tag, row in solved["members"].items():
                system = systems_of[tag]
                if system == mg.SYSTEM_GRAVITY:
                    expected = continuous[tag]["local_force_kip_kipin"]
                    self.assertEqual(row["analysis_model"], systems.MODEL_GRAVITY)
                else:
                    expected = [c + e - g for c, e, g in zip(continuous[tag]["local_force_kip_kipin"],
                                                              released[tag]["local_force_kip_kipin"],
                                                              released_gravity[tag]["local_force_kip_kipin"])]
                    self.assertEqual(row["analysis_model"], systems.MODEL_SFRS)
                    self.assertAlmostEqual(row["axial_i_kip"], continuous[tag]["axial_i_kip"] + released[tag]["axial_i_kip"]
                                           - released_gravity[tag]["axial_i_kip"], places=6)
                for value, target in zip(row["local_force_kip_kipin"], expected):
                    self.assertAlmostEqual(value, target, places=6)
                if row["member_type"] != "column":
                    self.assertTrue(row["span_bending"]["equilibrium"]["passed"])
                    lateral_increment[system] = max(lateral_increment[system], _end_moment(
                        [e - g for e, g in zip(released[tag]["local_force_kip_kipin"], released_gravity[tag]["local_force_kip_kipin"])]))
            # the lateral increment lives in the perimeter beams; the released interior beams carry none
            self.assertGreater(lateral_increment[mg.SYSTEM_SFRS], 0.0)
            self.assertLess(lateral_increment[mg.SYSTEM_GRAVITY], 1e-9 * lateral_increment[mg.SYSTEM_SFRS])
            # a second combination of the same gravity family reuses both cached gravity solves
            other = next(c for c in combinations if c["id"] == "seismic_high_gravity_X_x-1_y+1")
            before = solver.solves
            solver.analyze(other)
            self.assertEqual(solver.solves, before + 1)
            # a gravity combination is the continuous frame, as before
            gravity = next(c for c in combinations if c["id"] == "gravity_1.2D_1.6L")
            plain = solver.analyze(gravity)
            self.assertEqual(plain["models"], {"all": "all_frames"})
            self.assertIsNone(plain["elf"])
            driver._analyze_combination(gravity, period, None)
            reference = checks.capture_member_actions(gravity["dead"])
            for tag, row in plain["members"].items():
                for value, target in zip(row["local_force_kip_kipin"], reference[tag]["local_force_kip_kipin"]):
                    self.assertAlmostEqual(value, target, places=6)
            self.assertEqual(sp.LATERAL_DESIGN_MODEL, "all_frames")

    def test_the_band_policy_solves_one_model_per_combination(self):
        with gf.frame():
            state = gf.uniform_state()
            mg.install(state)
            period = systems.model_period(state)
            combination = strength_load_combinations(sp.ASCE_SDS)[2]
            solver = systems.CombinationSolver(period, None, state)
            solved = solver.analyze(combination)
            self.assertEqual(solver.solves, 1)
            self.assertEqual(solved["models"], {"all": "all_frames"})
            self.assertNotIn("analysis_model", next(iter(solved["members"].values())))
            self.assertIsNone(systems.compatibility_run(period, None, DesignConfig(), state))
            self.assertEqual(systems.gravity_group_induced(None, DesignConfig(), state), {})
            self.assertFalse(systems.compatibility_stage(None, {})["evaluated"])


class Compatibility(unittest.TestCase):
    def test_the_design_displacement_is_imposed_and_induces_gravity_member_actions(self):
        with gf.frame():
            state = system_state()
            mg.install(state)
            cfg = DesignConfig()
            period = systems.model_period(state)
            displacements = systems.design_displacements(period, None, cfg, state)
            self.assertEqual(displacements["cd"], sp.ASCE_CD)
            self.assertEqual(displacements["importance_factor"], sp.ASCE_IE)
            self.assertEqual(displacements["lateral_design_model"], "perimeter_sfrs")
            roof = displacements["axes"]["x"]["levels"][-1]
            self.assertGreater(roof["elastic_in"]["ux"], 0.0)
            self.assertAlmostEqual(roof["design_in"]["ux"], sp.ASCE_CD * roof["elastic_in"]["ux"] / sp.ASCE_IE)
            self.assertLess(abs(roof["elastic_in"]["uy"]), 1e-6 * roof["elastic_in"]["ux"])          # doubly symmetric plan
            self.assertEqual(displacements["axes"]["x"]["roof_design_displacement_in"], roof["design_in"]["ux"])
            compatibility = systems.compatibility_run(period, None, cfg, state, displacements=displacements)
            self.assertEqual(len(compatibility["cases"]), 8)
            self.assertEqual(compatibility["solves"], 10)
            self.assertEqual({c["family"] for c in compatibility["cases"]},
                             {"compatibility_high_gravity", "compatibility_low_gravity"})
            case = next(c for c in compatibility["cases"] if c["id"] == "compatibility_high_gravity_x+")
            self.assertAlmostEqual(case["dead"], 1.2 + 0.2 * sp.ASCE_SDS)
            self.assertEqual(case["live"], 1.0)
            low = next(c for c in compatibility["cases"] if c["id"] == "compatibility_low_gravity_x-")
            self.assertAlmostEqual(low["dead"], 0.9 - 0.2 * sp.ASCE_SDS)
            self.assertEqual(low["live"], 0.0)
            # the imposed displacement is reached at every level (penalty constraints)
            for level in case["imposed"]:
                target = level["target_in"]["ux"]
                self.assertGreater(target, 0.0)
                self.assertAlmostEqual(level["achieved_in"]["ux"], target, delta=1e-3 * target)
            for level in low["imposed"]:
                self.assertLess(level["target_in"]["ux"], 0.0)
            # the continuous frame now puts drift moments into the interior (gravity) beams
            interior = str(mg.beam_at("beam_x", 2, 0, 1).member_tag)
            perimeter = str(mg.beam_at("beam_x", 2, 0, 0).member_tag)
            self.assertGreater(_end_moment(case["members"][interior]["local_force_kip_kipin"]),
                               0.1 * _end_moment(case["members"][perimeter]["local_force_kip_kipin"]))
            self.assertEqual(case["members"][interior]["member_type"], "beam_x")
            induced = systems.gravity_group_induced(compatibility, cfg, state)
            self.assertEqual(set(induced), GRAVITY_GROUPS)
            for gid, row in induced.items():
                self.assertIn(row["branch"], gravity_rules.BRANCHES)
                self.assertGreater(row["moment_ratio"], 0.0)
                self.assertGreater(row["shear_ratio"], 0.0)
                self.assertEqual(row["tie"], state.tie_of(gid))
                self.assertTrue(row["governing"]["shear"]["combination"].startswith("compatibility_"))
            self.assertEqual(induced["s01_04__column__interior"]["member_type"], "column")
            stage = systems.compatibility_stage(compatibility, induced)
            self.assertTrue(stage["evaluated"])
            self.assertEqual(set(stage["groups"]), GRAVITY_GROUPS)
            self.assertLess(stage["largest_residual_in"], 1e-3 * roof["design_in"]["ux"])
            self.assertEqual(sp.LATERAL_DESIGN_MODEL, "all_frames")


class BranchFromRatios(unittest.TestCase):
    def test_ratio_form_matches_the_action_form(self):
        by_actions = gravity_rules.compatibility_branch(1200.0, 40.0, 1500.0, 60.0)
        by_ratios = gravity_rules.branch_from_ratios(0.8, 40.0 / 60.0)
        self.assertEqual(by_actions["branch"], gravity_rules.BRANCH_WITHIN_STRENGTH)
        self.assertEqual(by_actions, by_ratios)
        self.assertEqual(gravity_rules.branch_from_ratios(1.05, 0.2)["branch"], gravity_rules.BRANCH_YIELDING)
        self.assertEqual(gravity_rules.induced_branch({"moment_ratio": 0.3, "shear_ratio": 1.2})["branch"],
                         gravity_rules.BRANCH_YIELDING)
        self.assertEqual(gravity_rules.induced_branch({"mu_kip_in": 10.0, "vu_kip": 1.0},
                                                      {"phi_mn_kip_in": 20.0, "phi_vn_kip": 2.0})["branch"],
                         gravity_rules.BRANCH_WITHIN_STRENGTH)
        self.assertEqual(gravity_rules.induced_branch({"moment_ratio": 0.3, "shear_ratio": 0.2, "calculated": False})["branch"],
                         gravity_rules.BRANCH_YIELDING)
        with self.assertRaises(ValueError):
            gravity_rules.branch_from_ratios(-0.1, 0.2)


if __name__ == "__main__":
    unittest.main()
