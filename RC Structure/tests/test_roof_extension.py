"""The column extension above the roof joint, as weight and mass, by hand (2026-10-04).

The joint classification credits a column carried one column depth above the roof (ACI 318-19 15.2.6). These
tests show that the same extension is in the applied gravity, the base reaction, the nodal mass, the seismic
weight, the column axial estimate and the story stability weight, once each, in the uniform and the grouped
route, and that switching the declaration off removes all of it.
"""
import contextlib
import sys
import unittest
from pathlib import Path
from unittest import mock

import openseespy.opensees as ops

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import Structure_Parameters as sp                                                # noqa: E402
import grouped_fixture as gf                                                     # noqa: E402
from Analysis.Gravity import run_gravity_analysis                                # noqa: E402
from Design import Design_Driver as driver                                       # noqa: E402
from Design.SMRF_Elastic import build_design_model, gravity_weight_above_story, gravity_weight_per_story  # noqa: E402
from Loads.Gravity_Loads import apply_gravity_loads                              # noqa: E402
from Loads.Seismic_ELF import seismic_weight_per_floor                           # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402
from Model import Member_Properties as mp                                        # noqa: E402
from Model import Roof_Extension as roof                                         # noqa: E402
from Model.Build_Model import installed_node_seismic_mass, node_tag             # noqa: E402
from Model.IMK_Calibration import column_gravity_axial                           # noqa: E402

UNIT = 0.150 / 1728.0


class UniformRoofExtension(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        values = {"NUM_BAY_X": 2, "NUM_BAY_Y": 3, "NUM_FLOOR": 3, "BAY_X": 240.0, "BAY_Y": 300.0, "STORY_H": 144.0,
                  "B_COL": 24.0, "H_COL": 30.0, "B_BEAM": 18.0, "H_BEAM": 28.0, "SLAB_THICKNESS_IN": 8.0,
                  "FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF": 0.05, "FLOOR_DEAD_LOAD_KSF": 0.15, "FLOOR_LIVE_LOAD_KSF": 0.05,
                  "SEISMIC_LIVE_LOAD_FRACTION": 0.25, "CONCRETE_UNIT_WEIGHT_KCF": 0.150, "CONCRETE_UNIT_WEIGHT_KCI": UNIT,
                  "ROOF_COLUMN_EXTENSION": True, "NUM_MODES": sp.NUM_MODES,
                  "GEOMETRY_VARIANT_NAME": getattr(sp, "GEOMETRY_VARIANT_NAME", "baseline")}
        for name, value in values.items():
            self.stack.enter_context(mock.patch.object(sp, name, value, create=True))
        self.stub = UNIT * 24.0 * 30.0 * 30.0                       # one column depth (the larger dimension) of full section
        self.columns = 3 * 4

    def tearDown(self):
        ops.wipe()
        self.stack.close()

    def nodes(self):
        return [(i, j) for i in range(sp.NUM_BAY_X + 1) for j in range(sp.NUM_BAY_Y + 1)]

    def test_length_and_weight_by_hand(self):
        self.assertEqual(roof.length_in(), 30.0)
        self.assertAlmostEqual(roof.weight_kip(1, 2), self.stub, places=12)
        self.assertAlmostEqual(self.stub, 1.875, places=12)
        self.assertAlmostEqual(roof.total_weight_kip(), self.columns * self.stub, places=10)
        self.assertEqual(driver._joint_continuity_declaration()["roof_column_extension_in"], roof.length_in())

    def test_mass_seismic_weight_and_their_agreement(self):
        for i, j in self.nodes():
            self.assertAlmostEqual(installed_node_seismic_mass(3, i, j), sp.node_seismic_mass(i, j) + self.stub / sp.G, places=14)
            for floor in (1, 2):
                self.assertEqual(installed_node_seismic_mass(floor, i, j), sp.node_seismic_mass(i, j))
        weights = seismic_weight_per_floor()
        floor = sp.total_floor_seismic_weight()
        self.assertEqual(weights[:2], [floor, floor])
        self.assertAlmostEqual(weights[2], floor + self.columns * self.stub, places=9)
        for k, weight in enumerate(weights, start=1):
            self.assertAlmostEqual(sum(installed_node_seismic_mass(k, i, j) for i, j in self.nodes()) * sp.G, weight, places=9)
        metadata = sp.floor_load_metadata()
        self.assertAlmostEqual(metadata["total_seismic_weight_kip"], sum(weights), places=9)
        self.assertEqual(metadata["roof_column_extension_in"], 30.0)

    def test_base_reaction_carries_the_factored_extension_weight(self):
        for mode in ("nodal", "beam_uniform"):
            for factor in (1.0, 1.2):
                with self.subTest(mode=mode, factor=factor), mock.patch.object(sp, "GRAVITY_LOAD_MODEL", mode):
                    with driver._quiet():
                        build_design_model()
                        apply_gravity_loads(floor_factor=1.0, self_weight_factor=factor)
                        run_gravity_analysis()
                        ops.reactions()
                    upward = sum(ops.nodeReaction(node_tag(0, i, j), 3) for i, j in self.nodes())
                    framed = 3 * (sp.total_floor_gravity_load() + factor * sp.total_structural_self_weight_per_floor())
                    self.assertAlmostEqual(upward, framed + factor * self.columns * self.stub, places=6)
                    # every column line carries its own extension: the roof-story column's top axial force
                    ops.wipe()

    def test_column_axial_and_stability_weight(self):
        floor = sp.node_gravity_load_kip(1, 1) + sp.node_structural_self_weight_kip(1, 1)
        self.assertAlmostEqual(column_gravity_axial(3, 1, 1), floor + self.stub, places=9)
        self.assertAlmostEqual(column_gravity_axial(1, 1, 1), 3 * floor + self.stub, places=9)
        self.assertAlmostEqual(gravity_weight_above_story(3), gravity_weight_per_story() + self.columns * self.stub, places=9)
        self.assertAlmostEqual(gravity_weight_above_story(1), 3 * gravity_weight_per_story() + self.columns * self.stub, places=9)

    def test_declaration_off_removes_all_of_it(self):
        with mock.patch.object(sp, "ROOF_COLUMN_EXTENSION", False):
            self.assertEqual(roof.length_in(), 0.0)
            self.assertEqual(roof.total_weight_kip(), 0.0)
            self.assertEqual(seismic_weight_per_floor(), [sp.total_floor_seismic_weight()] * 3)
            self.assertEqual(installed_node_seismic_mass(3, 1, 1), sp.node_seismic_mass(1, 1))
            self.assertNotIn("roof_column_extension_in", driver._joint_continuity_declaration())
            self.assertAlmostEqual(gravity_weight_above_story(1), 3 * gravity_weight_per_story(), places=9)

    def test_legacy_bundled_mass_omits_it_and_gravity_keeps_it(self):
        with mock.patch.object(sp, "SLAB_THICKNESS_IN", None):
            self.assertEqual(roof.seismic_weight_kip(), 0.0)
            self.assertEqual(installed_node_seismic_mass(3, 1, 1), sp.node_seismic_mass(1, 1))
            self.assertAlmostEqual(roof.total_weight_kip(), self.columns * self.stub, places=10)


class GroupedRoofExtension(unittest.TestCase):
    def setUp(self):
        self.stack = gf.frame()
        self.stack.__enter__()
        self.addCleanup(self.stack.__exit__, None, None, None)
        self.extension = mock.patch.object(sp, "ROOF_COLUMN_EXTENSION", True)
        self.extension.start()
        self.addCleanup(self.extension.stop)
        mg.install(gf.changed_state())
        self.addCleanup(mg.clear)
        self.addCleanup(ops.wipe)

    def lines(self):
        return [(i, j) for j in range(gf.NY + 1) for i in range(gf.NX + 1)]

    def hand_stub(self, i, j):
        design = gf.hand_design("column", gf.NZ, i, j)               # each line's own roof-story column
        return UNIT * design.b_in * design.h_in * max(design.b_in, design.h_in)

    def test_each_line_uses_its_own_roof_column_and_the_ledger_carries_it_once(self):
        stubs = {line: self.hand_stub(*line) for line in self.lines()}
        self.assertGreater(len({round(v, 9) for v in stubs.values()}), 1)          # the changed groups differ at the roof
        for (i, j), stub in stubs.items():
            self.assertAlmostEqual(roof.weight_kip(i, j), stub, places=12)
            self.assertAlmostEqual(installed_node_seismic_mass(gf.NZ, i, j), gf.hand_node_mass(gf.NZ, i, j) + stub / sp.G, places=13)
            self.assertAlmostEqual(installed_node_seismic_mass(1, i, j), gf.hand_node_mass(1, i, j), places=13)
        total = sum(stubs.values())
        ledger = mp.weight_ledger()
        self.assertAlmostEqual(ledger["total_roof_column_extension_weight_kip"], total, places=10)
        self.assertEqual([row["roof_column_extension_weight_kip"] for row in ledger["floors"][:-1]], [0.0] * (gf.NZ - 1))
        for row in ledger["floors"]:
            self.assertAlmostEqual(row["seismic_weight_kip"], row["mass_sum_kip"], places=9)
        self.assertEqual(seismic_weight_per_floor(), [row["seismic_weight_kip"] for row in ledger["floors"]])
        self.assertAlmostEqual(seismic_weight_per_floor()[-1],
                               sum(gf.hand_node_mass(gf.NZ, i, j) for i, j in self.lines()) * sp.G + total, places=9)

    def test_base_reaction_is_the_hand_total_plus_the_extensions(self):
        with driver._quiet():
            build_design_model()
            apply_gravity_loads()
            run_gravity_analysis()
            ops.reactions()
        upward = sum(ops.nodeReaction(node_tag(0, i, j), 3) for i, j in self.lines())
        members = sum(gf.hand_weight_kip(kind, k, i, j) for _tag, kind, k, i, j in gf.hand_members())
        floor_load = ((sp.FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF + sp.CONCRETE_UNIT_WEIGHT_KCF * gf.SLAB / 12.0 + sp.FLOOR_LIVE_LOAD_KSF)
                      * gf.BAY_X * gf.NX * gf.BAY_Y * gf.NY / 144.0) * gf.NZ
        total = sum(self.hand_stub(i, j) for i, j in self.lines())
        self.assertAlmostEqual(upward, floor_load + members + total, places=6)
        self.assertAlmostEqual(mp.weight_ledger()["total_gravity_kip"], floor_load + members + total, places=6)

    def test_column_axial_carries_the_extension_of_its_own_line(self):
        with mock.patch.object(sp, "ROOF_COLUMN_EXTENSION", False):
            without = column_gravity_axial(1, 1, 1)
        self.assertAlmostEqual(column_gravity_axial(1, 1, 1), without + self.hand_stub(1, 1), places=9)


if __name__ == "__main__":
    unittest.main()
