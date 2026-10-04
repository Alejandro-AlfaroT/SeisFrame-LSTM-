import contextlib
import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Design.SMRF_Joint_Adapter import (
    build_joint_evidence, local_force_axial_actions, nominal_rectangular_capacity,
    physical_joint_inventory, record_section_capacity, section_bar_coordinates,
)
from Design.SMRF_Joints import evaluate_joints


def record():
    return {"geometry": {"num_bay_x": 2, "num_bay_y": 2, "num_floor": 2},
            "sections": {"b_col_in": 24., "h_col_in": 30., "fc_col_ksi": 5.,
                         "b_beam_in": 16., "h_beam_in": 24., "fc_beam_ksi": 5.},
            "reinforcement": {"col_top_bars": 4, "col_bot_bars": 4, "col_side_bars": 3,
                              "col_bar_area_in2": 1., "col_longitudinal_centroid_offset_in": 2.5,
                              "beam_top_bars": 4, "beam_bot_bars": 3, "beam_side_bars": 1,
                              "beam_bar_area_in2": .79, "beam_longitudinal_centroid_offset_in": 2.25,
                              "beam_bar_diameter_in": 1.},
            "materials": {"fy_ksi": 60., "es_ksi": 29000., "normalweight": True}}


def actions(data):
    inventory = physical_joint_inventory(data["geometry"])
    return [{"id": cid, "analysis_succeeded": True, "axial_reference": "joint_faces",
             "members": {str(m["tag"]): {"axial_i_kip": pi, "axial_j_kip": pj}
                         for m in inventory["members"] if m["kind"] == "column"}}
            for cid, pi, pj in (("D+L+Ex", 100., 90.), ("D+L-Ex", 250., 230.))]


def evaluated(data=None, **kwargs):
    data = data or record()
    return build_joint_evidence(data, actions(data),
                                expected_combination_ids=["D+L+Ex", "D+L-Ex"], **kwargs)


class JointInventoryTests(unittest.TestCase):
    def test_tag_order_matches_actual_elastic_builder(self):
        import Structure_Parameters as sp
        from Design.SMRF_Elastic import physical_members
        data = record()
        with contextlib.ExitStack() as stack:
            for key, value in (("NUM_BAY_X", 2), ("NUM_BAY_Y", 2), ("NUM_FLOOR", 2)):
                stack.enter_context(mock.patch.object(sp, key, value))
            expected = list(physical_members())
        got = physical_joint_inventory(data["geometry"])
        self.assertEqual([(m["tag"], m["node_i"], m["node_j"], m["kind"]) for m in got["members"]], expected)

    def test_roof_has_one_actual_column_not_two(self):
        got = physical_joint_inventory(record()["geometry"])
        self.assertEqual(got["elevated_joint_count"], 18)
        for joint in got["joints"]:
            self.assertEqual(len(joint["columns"]), 1 if joint["is_roof"] else 2)
        middle = next(j for j in got["joints"] if j["floor"] == 1 and j["grid_i"] == j["grid_j"] == 1)
        self.assertEqual(middle["plan_type"], "interior")
        self.assertEqual((len(middle["beams_x"]), len(middle["beams_y"])), (2, 2))

    def test_malformed_geometry_fails(self):
        for bad in (0, -1, 1.5, True, float("nan")):
            geometry = dict(record()["geometry"], num_bay_x=bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                physical_joint_inventory(geometry)


class SectionCompatibilityTests(unittest.TestCase):
    def test_bar_inventory_and_per_member_covers(self):
        data = record()
        bars = section_bar_coordinates(data, "column")
        self.assertEqual(len(bars), 14)
        self.assertEqual(sum(a for a, _, _ in bars), 14)
        self.assertEqual(len({(y, z) for _, y, z in bars}), 14)
        self.assertEqual(sorted({z for _, _, z in bars}), [-12.5, -6.25, 0, 6.25, 12.5])
        self.assertEqual(max(z for _, _, z in section_bar_coordinates(data, "beam")), 9.75)

    def test_singly_reinforced_hand_solution(self):
        # a=As fy/(.85 fc b) = 2.941176..., a<d, steel yielded.
        got = nominal_rectangular_capacity(width_in=12, depth_in=24, fc_ksi=4,
                                            fy_ksi=60, es_ksi=29000,
                                            layers=[(2., 21.)], axial_kip=0.)
        a = 2 * 60 / (.85 * 4 * 12)
        self.assertAlmostEqual(got["mn_kip_in"], 2 * 60 * (21 - a / 2), places=4)
        self.assertAlmostEqual(got["equilibrium_residual_kip"], 0, places=4)

    def test_rectangular_column_axes_are_distinct_and_sign_symmetric(self):
        data = record()
        x = record_section_capacity(data, "column", "x", "positive", 200)
        y = record_section_capacity(data, "column", "y", "positive", 200)
        negative = record_section_capacity(data, "column", "x", "negative", 200)
        self.assertGreater(x["mn_kip_in"], y["mn_kip_in"])
        self.assertAlmostEqual(x["mn_kip_in"], negative["mn_kip_in"])

    def test_known_neutral_axis_with_compression_steel_replacement(self):
        # Independent hand equilibrium at c=8, beta=.8 for fc=5 ksi.
        b, h, fc, fy, es, c = 20., 28., 5., 60., 29000., 8.
        steel = [(3., 2.5), (2., 14.), (3., 25.5)]
        a = .8 * c
        p = .85 * fc * b * a
        moment = p * (h / 2 - a / 2)
        for area, d in steel:
            stress = max(-fy, min(fy, es * .003 * (c - d) / c))
            force = area * (stress - (.85 * fc if d <= a else 0))
            p += force
            moment += force * (h / 2 - d)
        got = nominal_rectangular_capacity(width_in=b, depth_in=h, fc_ksi=fc,
                                            fy_ksi=fy, es_ksi=es, layers=steel,
                                            axial_kip=p)
        self.assertAlmostEqual(got["neutral_axis_in"], c, places=6)
        self.assertAlmostEqual(got["mn_kip_in"], abs(moment), places=4)

    def test_beam_positive_and_negative_use_actual_unequal_steel(self):
        data = record()
        positive = record_section_capacity(data, "beam", "x", "positive", 0)
        negative = record_section_capacity(data, "beam", "x", "negative", 0)
        self.assertGreater(negative["mn_kip_in"], positive["mn_kip_in"])

    def test_no_out_of_domain_clamping_and_no_missing_material_defaults(self):
        data = record()
        for axial in (-100000, 100000, float("nan")):
            with self.subTest(axial=axial), self.assertRaises(ValueError):
                record_section_capacity(data, "column", "x", "positive", axial)
        data["materials"].pop("es_ksi")
        with self.assertRaises(ValueError):
            record_section_capacity(data, "column", "x", "positive", 100)


class EvidenceAdapterTests(unittest.TestCase):
    def test_actual_column_end_is_enveloped_over_all_cases(self):
        got = evaluated()
        self.assertTrue(got["action_inventory_complete"])
        first = got["joints"][0]["directions"]["x"]["positive"]["column_capacities"]
        below = next(c for c in first if c["end"] == "j")
        above = next(c for c in first if c["end"] == "i")
        self.assertEqual((below["axial_min_kip"], below["axial_max_kip"]), (90, 230))
        self.assertEqual((above["axial_min_kip"], above["axial_max_kip"]), (100, 250))
        self.assertEqual(below["mn_kip_in"], min(o["mn_kip_in"] for o in below["observations"]))
        self.assertTrue(below["axial_envelope_checked"])

    def test_centerline_or_missing_reference_cannot_complete_joint_face_envelope(self):
        for reference in (None, "centerline"):
            data = record()
            force_cases = actions(data)
            force_cases[0]["axial_reference"] = reference
            got = build_joint_evidence(data, force_cases, expected_combination_ids=["D+L+Ex", "D+L-Ex"])
            with self.subTest(reference=reference):
                self.assertTrue(got["action_inventory_complete"])
                self.assertFalse(got["joint_face_reference_complete"])
                cap = got["joints"][0]["directions"]["x"]["positive"]["column_capacities"][0]
                self.assertFalse(cap["axial_envelope_checked"])
                self.assertGreater(cap["mn_kip_in"], 0)

    def test_asymmetric_column_uses_weaker_compression_face_for_both_sways(self):
        data = record()
        data["reinforcement"]["col_top_bars"] = 3
        expected_mn = min(record_section_capacity(data, "column", "x", face, p)["mn_kip_in"]
                          for face in ("positive", "negative") for p in (90, 230))
        got = evaluated(data)
        for state in got["joints"][0]["directions"]["x"].values():
            below = next(c for c in state["column_capacities"] if c["end"] == "j")
            self.assertAlmostEqual(below["mn_kip_in"], expected_mn)

    def test_missing_or_failed_load_case_never_completes_envelope(self):
        for mode in ("missing", "failed", "duplicate", "missing_column", "nan"):
            data = record()
            force_cases = actions(data)
            if mode == "missing":
                force_cases.pop()
            elif mode == "failed":
                force_cases[0]["analysis_succeeded"] = False
            elif mode == "duplicate":
                force_cases.append(copy.deepcopy(force_cases[0]))
            elif mode == "missing_column":
                force_cases[0]["members"].pop("1")
            else:
                force_cases[0]["members"]["1"]["axial_j_kip"] = float("nan")
            got = build_joint_evidence(data, force_cases, expected_combination_ids=["D+L+Ex", "D+L-Ex"])
            with self.subTest(mode=mode):
                self.assertFalse(got["action_inventory_complete"])
                self.assertFalse(got["joints"][0]["directions"]["x"]["positive"]["column_capacities"][0]["axial_envelope_checked"])

    def test_unknown_slab_does_not_become_zero_contribution(self):
        got = evaluated()
        beam = got["joints"][0]["directions"]["x"]["positive"]["beam_capacities"][0]
        self.assertNotIn("slab_mn_kip_in", beam)
        checks = evaluate_joints(got)
        self.assertTrue(all(c["status"] == "not_evaluated" for c in checks if c["id"] == "scwb"))

    def test_explicit_slab_evidence_is_specific_to_beam_end_and_flexure(self):
        data = record()
        inventory = physical_joint_inventory(data["geometry"])
        first = inventory["joints"][0]["beams_x"][0]
        data["beam_slab_strengths"] = {f"{first['tag']}/{first['end']}/negative": {
            "slab_basis": "developed_effective_width", "slab_mn_kip_in": 100,
            "section_compatibility_verified": True}}
        got = evaluated(data)
        states = got["joints"][0]["directions"]["x"]
        self.assertEqual(states["positive"]["beam_capacities"][0]["slab_mn_kip_in"], 100)
        self.assertNotIn("slab_mn_kip_in", states["negative"]["beam_capacities"][0])

    def test_through_bar_and_terminating_edges_are_distinct(self):
        got = evaluated()
        self.assertEqual(sum(a["bars_pass_through"] for a in got["through_bar_anchorage"]), 12)
        checks = evaluate_joints(got)
        through = [c for c in checks if c["id"] == "joint_through_bar_depth"]
        self.assertEqual(sum(c["status"] == "pass" for c in through), 12)
        self.assertEqual(sum(c["status"] == "not_evaluated" for c in through), 24)

    def test_unsolved_section_capacity_cannot_pass(self):
        data = record()
        data["materials"].pop("es_ksi")
        got = evaluated(data)
        for state in got["joints"][0]["directions"]["x"].values():
            self.assertFalse(state["column_capacities"][0]["axial_envelope_checked"])


class LocalForceConventionTests(unittest.TestCase):
    def test_compression_and_tension_not_absolute_values(self):
        f = [0.] * 12
        f[0], f[6] = 11, -10
        self.assertEqual(local_force_axial_actions(f), {"axial_i_kip": 11, "axial_j_kip": 10})
        f[0], f[6] = -11, 10
        self.assertEqual(local_force_axial_actions(f), {"axial_i_kip": -11, "axial_j_kip": -10})

    def test_in_memory_opensees_gravity_sign(self):
        import openseespy.opensees as ops
        ops.wipe()
        try:
            ops.model("basic", "-ndm", 3, "-ndf", 6)
            ops.node(1, 0, 0, 0)
            ops.node(2, 0, 0, 144)
            ops.fix(1, 1, 1, 1, 1, 1, 1)
            ops.geomTransf("Linear", 1, 1, 0, 0)
            ops.element("elasticBeamColumn", 1, 1, 2, 400., 4000., 1600., 10000., 13000., 13000., 1)
            ops.timeSeries("Linear", 1)
            ops.pattern("Plain", 1, 1)
            ops.load(2, 0, 0, -10, 0, 0, 0)
            ops.system("BandGeneral")
            ops.numberer("RCM")
            ops.constraints("Plain")
            ops.integrator("LoadControl", 1)
            ops.algorithm("Linear")
            ops.analysis("Static")
            self.assertEqual(ops.analyze(1), 0)
            forces = ops.eleResponse(1, "localForce")
            got = local_force_axial_actions(forces)
            self.assertAlmostEqual(got["axial_i_kip"], 10)
            self.assertAlmostEqual(got["axial_j_kip"], 10)
        finally:
            ops.wipe()



class StaggeredElevationTests(unittest.TestCase):
    """The joint's beam capacities use the staggered bar elevations the record carries (2026-09-27), so the
    beam and joint checks price the same section."""

    def staggered(self):
        data = record()
        data["sections"]["fc_beam_ksi"] = 5.
        data["reinforcement"]["beam_bar_size"] = 8
        data["reinforcement"]["beam_side_bars"] = 0      # the strength module prices top and bottom bars only
        data["reinforcement"]["beam_bar_stacking"] = {
            "convention": "y_over_x", "max_layers": 2, "layer_order": "interleaved",
            "layers": {"x": {"top": {"layers": 1, "per_layer": [4], "offsets_in": [5.07], "centroid_in": 5.07},
                             "bottom": {"layers": 1, "per_layer": [3], "offsets_in": [2.25], "centroid_in": 2.25}},
                       "y": {"top": {"layers": 1, "per_layer": [4], "offsets_in": [2.25], "centroid_in": 2.25},
                             "bottom": {"layers": 1, "per_layer": [3], "offsets_in": [5.07], "centroid_in": 5.07}}}}
        return data

    def test_the_lower_direction_loses_hogging_and_the_nominal_record_does_not(self):
        data = self.staggered()
        x_neg = record_section_capacity(data, "beam", "x", "negative", 0)["mn_kip_in"]
        y_neg = record_section_capacity(data, "beam", "y", "negative", 0)["mn_kip_in"]
        self.assertLess(x_neg, y_neg, "x top bars sit a layer lower at the joint")
        nominal = record()
        nominal["reinforcement"]["beam_side_bars"] = 0
        self.assertAlmostEqual(record_section_capacity(nominal, "beam", "x", "negative", 0)["mn_kip_in"],
                               record_section_capacity(nominal, "beam", "y", "negative", 0)["mn_kip_in"])
        # y keeps its top bars at the nominal elevation; only its bottom (compression) bars moved a layer
        self.assertAlmostEqual(y_neg / record_section_capacity(nominal, "beam", "y", "negative", 0)["mn_kip_in"], 1.0, delta=0.03)
        bars = section_bar_coordinates(data, "beam", "x")
        self.assertEqual(sorted(round(z, 2) for _, _, z in bars if z > 0), [24. / 2 - 5.07] * 4)

    def test_joint_adapter_agrees_with_the_record_strength_module(self):
        from Design.SMRF_Beam_Slab_Strength import composite_beam_strengths
        data = self.staggered()
        beam = {"b_in": 16., "h_in": 24., "fc_ksi": 5., "fy_ksi": 60., "bar_size": 8, "top_bars": 4, "bot_bars": 3,
                "centroid_offset_in": {"top": 5.07, "bottom": 2.25}}
        family = composite_beam_strengths(beam, {"thickness_in": 0.}, None, {"bay_x_in": 240., "bay_y_in": 240., "h_col_in": 30., "b_col_in": 24.}, "x", "interior")
        for sign in ("negative", "positive"):
            adapter = record_section_capacity(data, "beam", "x", sign, 0)["mn_kip_in"]
            module = family["rectangular"][sign]["mn_kip_in"]
            self.assertAlmostEqual(adapter / module, 1.0, delta=0.02, msg=f"{sign}: adapter {adapter:.0f} vs module {module:.0f}")

    def test_layers_must_add_up(self):
        data = self.staggered()
        data["reinforcement"]["beam_bar_stacking"]["layers"]["x"]["top"]["per_layer"] = [3]
        with self.assertRaisesRegex(ValueError, "not the 4"):
            section_bar_coordinates(data, "beam", "x")

    def test_unequal_row_arrays_are_rejected_not_truncated(self):
        """Review item 3: [3, 1] bars with one elevation must not quietly become three bars."""
        data = self.staggered()
        rows = data["reinforcement"]["beam_bar_stacking"]["layers"]["x"]["top"]
        rows["layers"], rows["per_layer"], rows["offsets_in"], rows["centroid_in"] = 2, [3, 1], [5.07], 5.07
        with self.assertRaisesRegex(ValueError, "must match"):
            section_bar_coordinates(data, "beam", "x")
        rows["offsets_in"] = [5.07, 7.89]
        with self.assertRaisesRegex(ValueError, "centroid"):
            section_bar_coordinates(data, "beam", "x")
        rows["centroid_in"] = (3 * 5.07 + 1 * 7.89) / 4
        self.assertEqual(len([z for _, _, z in section_bar_coordinates(data, "beam", "x") if z > 0]), 4)


if __name__ == "__main__":
    unittest.main()
