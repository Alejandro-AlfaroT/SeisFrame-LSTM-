"""The single-element transfer application reproduces an explicit-node reference.

The exported interface couples act at interior beam nodes the design frame
does not have. The production path carries each vertical-plane bending
couple as a force pair inside the element and splits the torsion couple to
the end joints. These tests compare that path, on the real design frame,
against a reference frame whose beams are subdivided at every transfer
node and loaded there with the forces and both couples as nodal loads:
reactions, joint displacements and rotations, beam end forces and the
bending diagram along each beam. Global equilibrium alone is not enough --
the split the pair replaced passed it while giving no beam response.
"""
import contextlib
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openseespy.opensees as ops  # noqa: E402

import Structure_Parameters as sp  # noqa: E402
from Design import Design_Driver as driver  # noqa: E402
from Design.SMRF_Floor_Transfer import build_floor_transfer, global_couple  # noqa: E402
from Design.SMRF_Elastic import beam_line_family, build_design_model, physical_members  # noqa: E402
from Design.SMRF_Beam_Actions import recover_beam_bending, applied_element_loads  # noqa: E402
from Loads import Gravity_Loads as loads  # noqa: E402
from Analysis.Gravity import run_gravity_analysis  # noqa: E402
from Model.Build_Model import node_tag  # noqa: E402

SLAB = {"thickness_in": 5.0, "concrete_fc_ksi": 4.0,
        "concrete_unit_weight_kcf": 0.15, "superimposed_dead_load_ksf": 0.05}
GEOMETRY = {"num_bay_x": 2, "num_bay_y": 2, "bay_x_in": 180.0, "bay_y_in": 150.0}
SECTIONS = {"b_beam_in": 12.0, "h_beam_in": 20.0, "fc_beam_ksi": 4.0, "b_col_in": 20.0, "h_col_in": 20.0}
INTERIOR_NODE_BASE = 1_000_000
SEGMENT_BASE = 2_000_000


class FixedBeamCoupleTests(unittest.TestCase):
    """The reviewer's counterexample: 240-in fixed-fixed beam, 120 kip-in couple at midspan."""

    def _beam(self, explicit_node):
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        ops.node(1, 0., 0., 0.)
        ops.node(2, 240., 0., 0.)
        ops.fix(1, 1, 1, 1, 1, 1, 1)
        ops.fix(2, 0, 1, 1, 1, 1, 1)             # one free DOF keeps the system non-empty
        ops.geomTransf("Linear", 1, 0, 0, 1)
        props = (100., 4000., 1700., 1000., 5000., 5000.)
        if explicit_node:
            ops.node(3, 120., 0., 0.)
            ops.element("elasticBeamColumn", 1, 1, 3, *props, 1)
            ops.element("elasticBeamColumn", 2, 3, 2, *props, 1)
        else:
            ops.element("elasticBeamColumn", 1, 1, 2, *props, 1)
        ops.timeSeries("Constant", 1)
        ops.pattern("Plain", 1, 1)

    def _solve(self):
        ops.constraints("Plain"); ops.numberer("Plain"); ops.system("FullGeneral")
        ops.integrator("LoadControl", 1.0); ops.algorithm("Linear"); ops.analysis("Static")
        self.assertEqual(ops.analyze(1), 0)
        ops.reactions()

    def tearDown(self):
        ops.wipe()

    def test_force_pair_reproduces_the_explicit_couple(self):
        self._beam(explicit_node=True)
        ops.load(3, 0, 0, 0, 0, 120., 0)
        self._solve()
        self.assertAlmostEqual(ops.nodeReaction(1, 3), -0.75, places=9)       # 1.5 M / L
        self.assertAlmostEqual(ops.nodeReaction(1, 5), 30.0, places=9)        # 0.25 M
        reference = (ops.nodeReaction(1, 3), ops.nodeReaction(2, 3), ops.nodeReaction(1, 5), ops.nodeReaction(2, 5))
        self._beam(explicit_node=False)
        for fraction, force in loads._bending_couple_pair(0.5, 120., 240.):
            ops.eleLoad("-ele", 1, "-type", "-beamPoint", 0., force, fraction)
        self._solve()
        pair = (ops.nodeReaction(1, 3), ops.nodeReaction(2, 3), ops.nodeReaction(1, 5), ops.nodeReaction(2, 5))
        for a, b in zip(reference, pair):
            self.assertAlmostEqual(a, b, delta=2e-4 * 120.)                     # O((arm/L)^2) of M
        f = ops.eleResponse(1, "localForce")
        self.assertAlmostEqual(f[2], -0.75, delta=1e-4)
        self.assertAlmostEqual(f[4], 30.0, delta=0.02)

    def test_end_split_gives_no_beam_response(self):
        """What the pair replaced: the (1 - x)/x split of the couple to fixed ends does nothing."""
        self._beam(explicit_node=False)
        ops.load(1, 0, 0, 0, 0, 60., 0)
        ops.load(2, 0, 0, 0, 0, 60., 0)
        self._solve()
        self.assertEqual(ops.nodeReaction(1, 3), 0.0)
        self.assertEqual(ops.eleResponse(1, "localForce")[4], 0.0)

    def test_pair_geometry(self):
        rows = loads._bending_couple_pair(0.5, 120., 240.)
        self.assertEqual(rows, [((120. - 1.) / 240., 60.), ((120. + 1.) / 240., -60.)])
        near = loads._bending_couple_pair(0.005, 10., 240.)      # 1.2 in from the end: the arm shrinks to 0.6 in
        self.assertAlmostEqual(near[0][0] * 240., 0.6)
        self.assertAlmostEqual(near[1][0] * 240., 1.8)
        self.assertAlmostEqual(near[0][1] * 2 * 0.6, 10.)
        self.assertEqual(loads._bending_couple_pair(0.3, 0., 240.), [])


class FrameReferenceTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        values = {
            "NUM_BAY_X": 2, "NUM_BAY_Y": 2, "NUM_FLOOR": 2,
            "BAY_X": 180.0, "BAY_Y": 150.0, "STORY_H": 144.0,
            "B_COL": 20.0, "H_COL": 20.0, "B_BEAM": 12.0, "H_BEAM": 20.0,
            "FC_COL_KSI": 4.0, "FC_BEAM_KSI": 4.0,
            "SLAB_THICKNESS_IN": 5.0, "FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF": 0.05,
            "FLOOR_LIVE_LOAD_KSF": 0.05, "SEISMIC_LIVE_LOAD_FRACTION": 0.0,
            "CONCRETE_UNIT_WEIGHT_KCF": 0.150, "CONCRETE_UNIT_WEIGHT_KCI": 0.150 / 1728.0,
            "GRAVITY_LOAD_MODEL": "nodal", "FLOOR_TRANSFER": None,
            "NUM_MODES": sp.NUM_MODES, "GEOMETRY_VARIANT_NAME": getattr(sp, "GEOMETRY_VARIANT_NAME", "baseline"),
        }
        for name, value in values.items():
            self.stack.enter_context(mock.patch.object(sp, name, value, create=True))
        from Design.SMRF_Demands import live_load_patterns
        self.patterns = live_load_patterns(2, 2)
        self.transfer = build_floor_transfer(SLAB, GEOMETRY, SECTIONS, 0.05, mesh_per_bay=6,
                                             live_patterns=self.patterns)
        self.stack.enter_context(mock.patch.object(sp, "FLOOR_TRANSFER", self.transfer))

    def tearDown(self):
        self.stack.close()
        ops.wipe()

    # -- reference frame: beams subdivided at every transfer node, loads applied there --
    def _reference(self, dead_factor, live_factor, live_pattern):
        from Model.Build_Model import create_nodes, fix_base_nodes, create_rigid_diaphragms
        from Model.IMK_Hinges import reset_hinge_registry
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        reset_hinge_registry()
        create_nodes()
        fix_base_nodes()
        ops.geomTransf("PDelta", sp.COL_TRANSF_TAG, 1, 0, 0)
        ops.geomTransf("Linear", sp.BEAM_X_TRANSF_TAG, 0, 0, 1)
        ops.geomTransf("Linear", sp.BEAM_Y_TRANSF_TAG, 0, 0, 1)
        live_case = "live" if live_pattern == "all" else f"live_pattern_{live_pattern}"
        cases = [(self.transfer["unit_cases"]["dead"], dead_factor)]
        if live_factor:
            cases.append((self.transfer["unit_cases"][live_case], live_factor))
        fractions = {}
        for case, _factor in cases:
            for beam in case["beams"]:
                key = (beam["axis"], beam["line_index"], beam["span_index"])
                fractions.setdefault(key, set()).update(x for x, _p in beam["node_loads"])
                fractions[key].update(x for x, _cx, _cy in beam.get("node_couples", []))
        segments = {}            # physical beam tag -> [segment tags in order]
        interior = {}            # (k, axis, line, span, fraction) -> node tag
        next_node, next_element = INTERIOR_NODE_BASE, SEGMENT_BASE
        for tag, ni, nj, kind in physical_members():
            column = kind == "column"
            b, h, fc = ((sp.B_COL, sp.H_COL, sp.FC_COL_KSI) if column else (sp.B_BEAM, sp.H_BEAM, sp.FC_BEAM_KSI))
            modifier = sp.section_stiffness_modifier("column" if column else "beam")
            e = sp.concrete_ec_ksi(fc)
            # Same section as the production frame: rectangular columns, the
            # line's T/L section for beams (Structure_Parameters.beam_flexural_section).
            iy = sp.rect_iy(b, h) if column else sp.beam_flexural_inertia_in4(*beam_line_family(ni, kind))
            props = (b * h, e, sp.concrete_shear_modulus_ksi(e), modifier * sp.approx_rect_j(b, h),
                     modifier * iy, modifier * sp.rect_iz(b, h))
            transform = (sp.COL_TRANSF_TAG if column else
                         sp.BEAM_X_TRANSF_TAG if kind == "beam_x" else sp.BEAM_Y_TRANSF_TAG)
            if column:
                ops.element("elasticBeamColumn", tag, ni, nj, *props, transform)
                continue
            axis = "x" if kind == "beam_x" else "y"
            k, line, span = self._beam_key(tag, axis)
            a, c = ops.nodeCoord(ni), ops.nodeCoord(nj)
            chain = [ni]
            for f in sorted(fractions.get((axis, line, span), ())):
                ops.node(next_node, *(a[d] + f * (c[d] - a[d]) for d in range(3)))
                interior[(k, axis, line, span, f)] = next_node
                chain.append(next_node)
                next_node += 1
            chain.append(nj)
            segments[tag] = []
            for left, right in zip(chain, chain[1:]):
                ops.element("elasticBeamColumn", next_element, left, right, *props, transform)
                segments[tag].append(next_element)
                next_element += 1
        create_rigid_diaphragms()
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for k in range(1, sp.NUM_FLOOR + 1):
            nodal = {}
            for case, factor in cases:
                for beam in case["beams"]:
                    axis, line, span = beam["axis"], beam["line_index"], beam["span_index"]
                    for x, p in beam["node_loads"]:
                        row = nodal.setdefault(interior[(k, axis, line, span, x)], [0., 0., 0.])
                        row[0] -= factor * p
                    for x, cx, cy in beam.get("node_couples", []):
                        gx, gy = global_couple(axis, factor * cx, factor * cy)
                        row = nodal.setdefault(interior[(k, axis, line, span, x)], [0., 0., 0.])
                        row[1] += gx
                        row[2] += gy
                for column in case["columns"]:
                    node = node_tag(k, column["grid_i"], column["grid_j"])
                    row = nodal.setdefault(node, [0., 0., 0.])
                    row[0] -= factor * column["direct_load_kip"]
                    row[1] += factor * column.get("couple_global_mx_kip_in", 0.)
                    row[2] += factor * column.get("couple_global_my_kip_in", 0.)
            for node, (fz, mx, my) in nodal.items():
                ops.load(node, 0., 0., fz, mx, my, 0.)
        with driver._quiet():
            run_gravity_analysis()
            ops.reactions()
        return segments

    @staticmethod
    def _beam_key(tag, axis):
        """(floor, line, span) of a physical beam tag, from the production tag map."""
        lines = range(sp.NUM_BAY_Y + 1) if axis == "x" else range(sp.NUM_BAY_X + 1)
        spans = range(sp.NUM_BAY_X) if axis == "x" else range(sp.NUM_BAY_Y)
        for k in range(1, sp.NUM_FLOOR + 1):
            for line in lines:
                for span in spans:
                    if loads._beam_element_tag(k, axis, line, span) == tag:
                        return k, line, span
        raise KeyError(tag)

    def _production(self, dead_factor, live_factor, live_pattern):
        with driver._quiet():
            ops.wipe()
            build_design_model()
            loads.apply_gravity_loads(floor_factor=1.0, self_weight_factor=0.0, dead_factor=dead_factor,
                                      live_factor=live_factor, live_pattern=live_pattern)
            run_gravity_analysis()
            ops.reactions()

    def _snapshot(self):
        reactions = {(i, j): [ops.nodeReaction(node_tag(0, i, j), d) for d in (3, 4, 5)]
                     for i in range(sp.NUM_BAY_X + 1) for j in range(sp.NUM_BAY_Y + 1)}
        joints = {(k, i, j): [ops.nodeDisp(node_tag(k, i, j), d) for d in (3, 4, 5)]
                  for k in range(1, sp.NUM_FLOOR + 1) for i in range(sp.NUM_BAY_X + 1) for j in range(sp.NUM_BAY_Y + 1)}
        return reactions, joints

    def _compare_case(self, dead_factor, live_factor, live_pattern):
        self._production(dead_factor, live_factor, live_pattern)
        reactions_p, joints_p = self._snapshot()
        beams = [(tag, kind) for tag, _ni, _nj, kind in physical_members() if kind != "column"]
        element_loads = applied_element_loads()
        production = {}
        for tag, kind in beams:
            force = list(ops.eleResponse(tag, "localForce"))
            length = sp.BAY_X if kind == "beam_x" else sp.BAY_Y
            item = element_loads.get(tag, {"uniform_z_kip_per_in": 0., "point_z_loads": []})
            recovered = recover_beam_bending(length, force, **item)
            production[tag] = (force, length, recovered)
        segments = self._reference(dead_factor, live_factor, live_pattern)
        reactions_r, joints_r = self._snapshot()

        load_scale = max(abs(v[0]) for v in reactions_r.values())
        moment_scale = max(abs(v[1]) + abs(v[2]) for v in reactions_r.values())
        for key in reactions_r:
            self.assertAlmostEqual(reactions_p[key][0], reactions_r[key][0], delta=1e-4 * load_scale, msg=("Rz", key))
            for d in (1, 2):
                self.assertAlmostEqual(reactions_p[key][d], reactions_r[key][d], delta=1e-3 * moment_scale, msg=("M", key, d))
        disp_scale = max(abs(v[0]) for v in joints_r.values())
        rot_scale = max(abs(v[1]) + abs(v[2]) for v in joints_r.values())
        for key in joints_r:
            self.assertAlmostEqual(joints_p[key][0], joints_r[key][0], delta=1e-3 * disp_scale, msg=("uz", key))
            for d in (1, 2):
                self.assertAlmostEqual(joints_p[key][d], joints_r[key][d], delta=1e-3 * rot_scale, msg=("rot", key, d))

        # Beam end forces and the bending diagram at every interior transfer node.
        for tag, kind in beams:
            force, length, recovered = production[tag]
            chain = segments[tag]
            first = list(ops.eleResponse(chain[0], "localForce"))
            last = list(ops.eleResponse(chain[-1], "localForce"))
            end_scale = max(1.0, abs(first[4]), abs(last[10]))
            self.assertAlmostEqual(force[2], first[2], delta=1e-3 * max(1.0, abs(first[2])), msg=("Vz_i", tag))
            self.assertAlmostEqual(force[4], first[4], delta=2e-3 * end_scale, msg=("My_i", tag))
            self.assertAlmostEqual(force[8], last[8], delta=1e-3 * max(1.0, abs(last[8])), msg=("Vz_j", tag))
            self.assertAlmostEqual(force[10], last[10], delta=2e-3 * end_scale, msg=("My_j", tag))
            # Bending diagram just left and just right of every interior node: the
            # production diagram from the single element's end forces and loads,
            # the reference from the segment end forces (linear inside a segment).
            # The pair ramps the couple over 2 arm; outside that window both
            # diagrams are exact, so the samples sit one tenth of an inch beyond it.
            s = loads.TRANSFER_COUPLE_ARM_IN + 0.1
            x = 0.0
            for segment, following in zip(chain[:-1], chain[1:]):
                seg = list(ops.eleResponse(segment, "localForce"))
                nxt = list(ops.eleResponse(following, "localForce"))
                span = self._segment_length(segment)
                x += span
                left_reference = seg[4] + seg[2] * (span - s)       # M(x - s) from the left segment
                right_reference = nxt[4] + nxt[2] * s               # M(x + s) from the right segment
                self.assertAlmostEqual(self._moment_at(recovered, force, x - s), left_reference,
                                       delta=3e-3 * end_scale, msg=("M(x-)", tag, x))
                self.assertAlmostEqual(self._moment_at(recovered, force, x + s), right_reference,
                                       delta=3e-3 * end_scale, msg=("M(x+)", tag, x))

    @staticmethod
    def _segment_length(segment):
        a, b = [ops.nodeCoord(n) for n in ops.eleNodes(segment)]
        return math.dist(a, b)

    @staticmethod
    def _moment_at(recovered, force, x):
        w = recovered["uniform_z_kip_per_in"]
        length = recovered["length_in"]
        points = [(f * length, p) for f, p in recovered["point_z_loads"]]
        return force[4] + force[2] * x + w * x * x / 2 + math.fsum(p * (x - a) for a, p in points if a < x)

    def test_response_is_insensitive_to_the_pair_arm(self):
        """The pair is an approximation to the point couple; halving or doubling its arm
        changes the frame response by O((arm/L)^2), and nothing reads the shear inside
        the pair window as a demand (end forces, face reactions and the moment envelope
        are the consumers)."""
        results = {}
        for arm in (0.5, 1.0, 2.0):
            with mock.patch.object(loads, "TRANSFER_COUPLE_ARM_IN", arm):
                self._production(1.2, 1.6, "all")
                reactions, joints = self._snapshot()
                _cols, bx, by = __import__("RC_Design_Check").get_element_tags()
                ends = {tag: list(ops.eleResponse(tag, "localForce")) for tag in bx + by}
                results[arm] = (reactions, joints, ends)
        base_r, base_j, base_e = results[1.0]
        load_scale = max(abs(v[0]) for v in base_r.values())
        rot_scale = max(abs(v[1]) + abs(v[2]) for v in base_j.values())
        end_scale = max(abs(f[4]) + abs(f[10]) for f in base_e.values())
        for arm in (0.5, 2.0):
            r, j, e = results[arm]
            for key in base_r:
                self.assertAlmostEqual(r[key][0], base_r[key][0], delta=1e-5 * load_scale, msg=("Rz", arm, key))
            for key in base_j:
                for d in (1, 2):
                    self.assertAlmostEqual(j[key][d], base_j[key][d], delta=1e-4 * rot_scale, msg=("rot", arm, key))
            for tag in base_e:
                for d in (4, 10):
                    self.assertAlmostEqual(e[tag][d], base_e[tag][d], delta=1e-4 * end_scale, msg=("M_end", arm, tag))

    def test_full_live_matches_the_explicit_node_reference(self):
        self._compare_case(1.2, 1.6, "all")

    def test_asymmetric_pattern_matches_the_explicit_node_reference(self):
        self._compare_case(1.2, 1.6, self.patterns[0]["id"])

    def test_dead_only_matches_the_explicit_node_reference(self):
        self._compare_case(1.4, 0.0, "all")


if __name__ == "__main__":
    unittest.main()
