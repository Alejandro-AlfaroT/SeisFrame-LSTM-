"""The perimeter-SFRS design model (2026-10-16): interior beams released in bending, perimeter frames carry the lateral load."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402
import openseespy.opensees as ops                                               # noqa: E402
import Structure_Parameters as sp                                               # noqa: E402
from Analysis.Gravity import run_gravity_analysis                                # noqa: E402
from Model import IMK_Hinges as ih                                               # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402
from Model.Build_Model import build_model, floor_master_node                     # noqa: E402


def _end_moment(tag):
    """The larger strong-axis end moment of a member (kip-in) from its local force vector."""
    forces = ops.eleResponse(tag, "localForce")
    return max(abs(forces[4]), abs(forces[10]), abs(forces[5]), abs(forces[11]))


def _push_roof(force_kip):
    ops.timeSeries("Linear", 901)
    ops.pattern("Plain", 901, 901)
    ops.load(floor_master_node(sp.NUM_FLOOR), force_kip, 0.0, 0.0, 0.0, 0.0, 0.0)
    run_gravity_analysis()


class ReleasedInteriorBeams(unittest.TestCase):
    def test_interior_beams_are_released_only_in_the_perimeter_model(self):
        with gf.frame():
            interior = mg.beam_at("beam_x", 2, 0, 1).member_tag            # an x beam on an interior line (j = 1 of ny = 3)
            perimeter = mg.beam_at("beam_x", 2, 0, 0).member_tag           # the same bay on the perimeter line j = 0
            self.assertEqual(mg.member_system("beam_x", 0, 1, sp.NUM_BAY_X, sp.NUM_BAY_Y), mg.SYSTEM_GRAVITY)
            self.assertEqual(mg.member_system("beam_x", 0, 0, sp.NUM_BAY_X, sp.NUM_BAY_Y), mg.SYSTEM_SFRS)
            moments = {}
            for model in ("all_frames", "perimeter_sfrs"):
                sp.LATERAL_DESIGN_MODEL = model
                try:
                    ops.wipe()
                    build_model()
                    registry = ih._HINGE_REGISTRY
                    self.assertEqual(registry[interior]["ends_released"], model == "perimeter_sfrs")
                    self.assertFalse(registry[perimeter]["ends_released"])
                    self.assertFalse(any(e["ends_released"] for e in registry.values() if e["member_type"] == "column"))
                    if model == "perimeter_sfrs":
                        self.assertTrue(registry[interior]["installed_materials"]["i"]["y"]["released"])
                        self.assertEqual(registry[interior]["installed_materials"]["j"]["z"]["material_type"], "Elastic")
                        self.assertNotIn("released", registry[perimeter]["installed_materials"]["i"]["y"])
                    _push_roof(50.0)
                    moments[model] = {"interior": _end_moment(interior), "perimeter": _end_moment(perimeter)}
                finally:
                    sp.LATERAL_DESIGN_MODEL = "all_frames"
                    ops.wipe()
            # with every frame engaged the interior beam carries lateral moment; released, it carries none worth the name
            self.assertGreater(moments["all_frames"]["interior"], 0.2 * moments["all_frames"]["perimeter"])
            self.assertLess(moments["perimeter_sfrs"]["interior"], 1e-3 * moments["perimeter_sfrs"]["perimeter"])
            # the perimeter beams pick up what the interior ones no longer carry
            self.assertGreater(moments["perimeter_sfrs"]["perimeter"], moments["all_frames"]["perimeter"])

    def test_the_flag_is_part_of_the_design_state(self):
        from Design import Design_Driver as driver
        self.assertIn("LATERAL_DESIGN_MODEL", driver._STATE_KEYS)
        self.assertEqual(sp.LATERAL_DESIGN_MODEL, "all_frames")
        self.assertIn("perimeter_sfrs", sp.LATERAL_DESIGN_MODELS)


if __name__ == "__main__":
    unittest.main()
