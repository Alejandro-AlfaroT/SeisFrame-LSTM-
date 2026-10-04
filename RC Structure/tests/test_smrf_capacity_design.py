"""Capacity design: probable strengths, member shear, joint shear, anchorage, splices.

Hand checks against ACI 318-19 Chapter 18 expressions on an explicit state,
plus the behaviours that matter for the design search (what grows the
column, what the hoops are chosen from, how joint confinement is read).
"""
import math
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Design.SMRF_Capacity_Design import (build_capacity_design, design_anchorage,  # noqa: E402
                                         design_beam_shear, design_joint_shear,
                                         design_splices, probable_beam_strengths,
                                         straight_development_length)
from Design.SMRF_Beam_Slab_Strength import section_moment  # noqa: E402


def layout(spacing=10.0, bar_area=0.20):
    return {"layers": {f"{axis}_{face}": {"bar_area_in2": bar_area, "spacing_in": spacing,
                                          "effective_depth_in": 4.0 if axis == "x" else 3.5}
                       for axis in ("x", "y") for face in ("top", "bottom")}}


def state(**changes):
    result = {
        "geometry": {"num_bay_x": 3, "num_bay_y": 3, "num_floor": 8, "bay_x_in": 120.0, "bay_y_in": 120.0,
                     "story_h_in": 120.0},
        "sections": {"b_col_in": 26.0, "h_col_in": 26.0, "fc_col_ksi": 4.0,
                     "b_beam_in": 10.0, "h_beam_in": 16.0, "fc_beam_ksi": 8.0},
        "materials": {"fy_ksi": 60.0, "es_ksi": 29000.0, "normalweight": True},
        "beam": {"bar_size": 8, "top_bars": 2, "bot_bars": 2, "centroid_offset_in": 2.5, "clear_cover_in": 1.5,
                 "self_weight_kip_per_in": {"x": 0.01, "y": 0.01}},
        "column": {"bar_size": 6, "top_bars": 4, "bot_bars": 4, "side_bars": 4, "centroid_offset_in": 2.375,
                   "clear_cover_in": 1.5, "stirrup_bar_size": 4,
                   "layers": [(1.76, 2.375), (0.88, 6.7), (0.88, 11.03), (0.88, 15.37), (0.88, 19.7), (1.76, 23.625)]},
        "slab": {"thickness_in": 5.0, "layout": layout()},
        "transfer": None, "sds": 1.0,
        "column_axial_envelope": {s: (10.0 * (9 - s), 40.0 * (9 - s)) for s in range(1, 9)},
        "column_shear_demand": {s: 12.0 for s in range(1, 9)},
    }
    for key, value in changes.items():
        result[key] = {**result[key], **value} if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def single_load_transfer(load_kip=10.0, x_fraction=0.5):
    """One node load at x on every beam of a 3x3 floor, dead only."""
    beams = []
    for axis, lines, spans in (("x", 4, 3), ("y", 4, 3)):
        for line in range(lines):
            for span in range(spans):
                beams.append({"axis": axis, "line_index": line, "span_index": span,
                              "node_loads": [[x_fraction, load_kip]]})
    return {"unit_cases": {"dead": {"beams": beams}, "live": None}}


class ProbableStrengthTests(unittest.TestCase):
    def test_rectangular_column_y_bending_rotates_concrete_and_steel(self):
        from Design import SMRF_Capacity_Design as capacity
        st = state(sections={"b_col_in": 20., "h_col_in": 26.},
                   column={"layers_about_z": [(2.64, 2.375), (1.76, 10.), (2.64, 17.625)]})
        strengths = {f"{axis}_{position}": capacity.probable_beam_strengths(st, position, axis)
                     for axis in ("x", "y") for position in ("interior", "edge")}
        with patch.object(capacity, "column_probable_pm", wraps=capacity.column_probable_pm) as pm:
            columns = capacity.design_column_shear(st, strengths)
        x, y = [call.args[0] for call in pm.call_args_list]
        self.assertEqual((x["sections"]["b_col_in"], x["sections"]["h_col_in"]), (20., 26.))
        self.assertEqual((y["sections"]["b_col_in"], y["sections"]["h_col_in"]), (26., 20.))
        self.assertEqual(y["column"]["layers"], st["column"]["layers_about_z"])
        p_min, p_max = st["column_axial_envelope"][1]
        diagram = capacity.column_probable_pm(y)
        expected = max(capacity._moment_at(diagram, p_min + (p_max - p_min) * k / 8) for k in range(9))
        self.assertAlmostEqual(columns["stories"]["y"][1]["mpr_column_kip_in"], expected)

    def test_rectangular_column_without_orthogonal_layers_fails_closed(self):
        from Design import SMRF_Capacity_Design as capacity
        with self.assertRaisesRegex(ValueError, "layers_about_z"):
            capacity.design_column_shear(state(sections={"b_col_in": 20.}), {})

    def test_mpr_uses_1_25_fy_on_the_composite_section(self):
        s = probable_beam_strengths(state(), "interior", "x")
        self.assertGreater(s["mpr_negative_kip_in"], 1.15 * s["mn_negative_kip_in"])
        self.assertLess(s["mpr_negative_kip_in"], 1.30 * s["mn_negative_kip_in"])
        self.assertGreater(s["mn_negative_kip_in"], s["mn_positive_kip_in"])       # slab in tension
        self.assertAlmostEqual(s["tension_steel_hogging_in2"], 2 * 0.79 + 0.20 * s["effective_flange_width_in"] / 10.0 * 2)
        self.assertAlmostEqual(s["tension_steel_sagging_in2"], 2 * 0.79)

    def test_rectangular_mpr_matches_hand_value(self):
        bare = state(slab={"thickness_in": 5.0, "layout": None})
        s = probable_beam_strengths(bare, "edge", "x")
        # 2#8 at 1.25 fy = 75 ksi, d = 13.5, fc 8, singly reinforced.
        a = 1.58 * 75.0 / (0.85 * 8.0 * 10.0)
        hand = 1.58 * 75.0 * (13.5 - a / 2.0)
        self.assertAlmostEqual(s["mpr_positive_kip_in"], hand, delta=0.02 * hand)


class BeamShearTests(unittest.TestCase):
    def test_ve_is_mpr_equilibrium_plus_factored_gravity_reactions(self):
        st = state(transfer=single_load_transfer(10.0, 0.5))
        strengths = {f"{a}_{p}": probable_beam_strengths(st, p, a) for a in ("x", "y") for p in ("edge", "interior")}
        result = design_beam_shear(st, strengths, st["transfer"])
        fam = strengths["x_interior"]
        ln = fam["clear_span_in"]
        mechanism = (fam["mpr_negative_kip_in"] + fam["mpr_positive_kip_in"]) / ln
        # (1.2 + 0.2 * 1.0) D on a 10-kip midspan load -> 7 kip per face, plus the
        # beam's own weight between the joint faces. The fixture supplies the smeared
        # centerline line weight (0.01 kip/in over 120 in); the free body uses the
        # physical drop weight over the 94-in clear span, and the two agree:
        # smeared * L / 2 == physical * ln / 2 (load ledger identity).
        reaction = 1.4 * 5.0 + 1.4 * 0.01 * 120.0 / 2.0
        self.assertAlmostEqual(result["mechanism_shear_kip"], mechanism, places=6)
        self.assertAlmostEqual(result["ve_kip"], mechanism + reaction, places=6)
        self.assertTrue(result["vc_zero_hinge_zone"])          # mechanism dominates
        hoops = result["hoops"]
        d = 16.0 - 2.5
        self.assertLessEqual(hoops["spacing_in"], min(d / 4.0, 6.0 * 1.0, 6.0))
        self.assertGreaterEqual(hoops["phi_vn_kip"], result["ve_kip"])
        self.assertEqual(result["section_adequate"], result["vs_required_kip"] <= result["vs_limit_kip"])
        entry = result["families"]["x_interior"][0]
        self.assertEqual(entry["mpr_basis"], "fy_at_least_1.25_phi_1.0")
        self.assertTrue(entry["shear_capacity_requirements_checked"])


class FaceReactionTests(unittest.TestCase):
    """The reviewer's counterexample: the span with the largest total is not the span with the largest end."""

    def _transfer(self, cases):
        beams = []
        for axis, lines, spans in (("x", 4, 3), ("y", 4, 3)):
            for line in range(lines):
                for span in range(spans):
                    beams.append({"axis": axis, "line_index": line, "span_index": span, "node_loads": [], "node_couples": []})
        transfer = {"unit_cases": {}}
        for name, per_span in cases.items():
            case = {"beams": [dict(b, node_loads=list(per_span.get((b["axis"], b["line_index"], b["span_index"]), {}).get("loads", [])),
                                   node_couples=list(per_span.get((b["axis"], b["line_index"], b["span_index"]), {}).get("couples", [])))
                              for b in beams]}
            transfer["unit_cases"][name] = case
        return transfer

    def test_each_end_is_governed_by_its_own_worst_span(self):
        from Design.SMRF_Capacity_Design import _face_reactions
        geometry = {"num_bay_x": 3, "num_bay_y": 3, "bay_x_in": 120.0, "bay_y_in": 120.0}
        c = 20.0                                   # column depth: clear span 100 in, faces at 10 and 110
        # Span A: 20 kip at midspan -> 10 / 10.  Span B: 19 kip at 10% of the clear span -> 17.1 / 1.9.
        transfer = self._transfer({"dead": {("x", 1, 0): {"loads": [[0.5, 20.0]]},
                                            ("x", 1, 1): {"loads": [[(10.0 + 10.0) / 120.0, 19.0]]}}})
        reactions, basis = _face_reactions(transfer, "x", "interior", geometry, 1.0, 0.0, 0.0, column_depth_in=c)
        self.assertAlmostEqual(reactions[0], 17.1, places=9)
        self.assertAlmostEqual(reactions[1], 10.0, places=9)
        self.assertIn("span (1, 1)", basis)
        self.assertIn("span (1, 0)", basis)

    def test_loads_inside_the_column_footprint_go_to_the_column(self):
        from Design.SMRF_Capacity_Design import _face_reactions
        geometry = {"num_bay_x": 3, "num_bay_y": 3, "bay_x_in": 120.0, "bay_y_in": 120.0}
        transfer = self._transfer({"dead": {("x", 1, 0): {"loads": [[0.05, 30.0], [0.5, 10.0]]}}})   # 6 in from the centerline: inside a 20-in column
        reactions, _ = _face_reactions(transfer, "x", "interior", geometry, 1.0, 0.0, 0.0, column_depth_in=20.0)
        self.assertAlmostEqual(reactions[0], 5.0)
        self.assertAlmostEqual(reactions[1], 5.0)

    def test_couples_and_patterns_enter_the_envelope(self):
        from Design.SMRF_Capacity_Design import _face_reactions
        geometry = {"num_bay_x": 3, "num_bay_y": 3, "bay_x_in": 120.0, "bay_y_in": 120.0}
        # A clockwise 50 kip-in couple at midspan of the clear span: -0.5 / +0.5 kip.
        transfer = self._transfer({"dead": {("x", 1, 0): {"couples": [[0.5, 0.0, 50.0]]}},
                                   "live": {("x", 1, 0): {"loads": [[0.5, 4.0]]}},
                                   "live_pattern_a": {("x", 1, 0): {"loads": [[0.5, 6.0]]}}})
        reactions, basis = _face_reactions(transfer, "x", "interior", geometry, 1.0, 1.0, 0.0, column_depth_in=20.0)
        self.assertAlmostEqual(reactions[0], -0.5 + 3.0)       # the pattern loads this span more than the full floor
        self.assertAlmostEqual(reactions[1], +0.5 + 3.0)
        self.assertIn("2 unit case(s)", basis)
        zero_live, _ = _face_reactions(transfer, "x", "interior", geometry, 1.0, 0.0, 0.0, column_depth_in=20.0)
        # Per-end envelope over spans: the couple pulls this span's left face down, so an
        # unloaded span (0) governs the left end while this span governs the right.
        self.assertAlmostEqual(zero_live[0], 0.0)
        self.assertAlmostEqual(zero_live[1], +0.5)


class JointKindTests(unittest.TestCase):
    def test_both_edge_orientations_and_both_directions_are_enumerated(self):
        from Design.SMRF_Capacity_Design import joint_kinds, design_joint_shear, design_column_shear
        kinds = joint_kinds(3, 3)
        self.assertEqual(set(kinds), {"corner", "edge_x", "edge_y", "interior"})
        # A column on the y-perimeter line: two edge x-beams, one terminating interior y-beam.
        self.assertEqual(kinds["edge_x"]["x"], ("x_edge", 2, 1))
        self.assertEqual(kinds["edge_x"]["y"], ("y_interior", 1, 2))
        self.assertEqual(kinds["edge_y"]["x"], ("x_interior", 1, 2))
        self.assertEqual(kinds["edge_y"]["y"], ("y_edge", 2, 1))
        self.assertEqual(set(joint_kinds(1, 3)), {"corner", "edge_y"})
        self.assertEqual(set(joint_kinds(1, 1)), {"corner"})
        st = state()
        strengths = {f"{a}_{p}": probable_beam_strengths(st, p, a) for a in ("x", "y") for p in ("edge", "interior")}
        joints = design_joint_shear(st, strengths, design_column_shear(st, strengths))["joints"]
        ids = {k for k in joints}
        for kind in ("corner", "edge_x", "edge_y", "interior"):
            for axis in ("x", "y"):
                self.assertIn(f"joint_shear/floor/{kind}/{axis}", ids)
                self.assertIn(f"joint_shear/roof/{kind}/{axis}", ids)
        # The terminating interior beam at an x-perimeter column uses the interior family alone.
        entry = joints["joint_shear/floor/edge_x/y"]
        self.assertEqual(entry["beam_family"], "y_interior")
        self.assertEqual(len(entry["beam_face_forces_kip"]), 1)
        fam = strengths["y_interior"]
        self.assertAlmostEqual(entry["beam_face_forces_kip"][0],
                               1.25 * 60.0 * max(fam["tension_steel_hogging_in2"], fam["tension_steel_sagging_in2"]))


class SelfWeightLedgerTests(unittest.TestCase):
    def test_free_body_weight_equals_the_smeared_element_weight(self):
        """The joint-face free body carries the physical drop weight on the clear span;
        the frame element carries the same weight smeared over the centerline length."""
        from Design.SMRF_Capacity_Design import _face_reactions, _drop_weight
        geometry = {"num_bay_x": 3, "num_bay_y": 3, "bay_x_in": 120.0, "bay_y_in": 120.0}
        sections = {"h_col_in": 18.0, "b_col_in": 18.0}
        physical = 0.15 / 1728.0 * 10.0 * (16.0 - 5.0)                    # drop below a 5-in slab
        smeared = physical * (1.0 - 18.0 / 120.0)
        beam = {"self_weight_kip_per_in": {"x": smeared, "y": smeared}, "drop_weight_kip_per_in": physical}
        self.assertAlmostEqual(_drop_weight(beam, "x", geometry, sections), physical)
        self.assertAlmostEqual(_drop_weight({"self_weight_kip_per_in": {"x": smeared}}, "x", geometry, sections), physical)
        reactions, _ = _face_reactions(None, "x", "interior", geometry, 1.0, 0.0, physical, column_depth_in=18.0)
        self.assertAlmostEqual(reactions[0] + reactions[1], physical * (120.0 - 18.0))
        self.assertAlmostEqual(reactions[0] + reactions[1], smeared * 120.0)


class ColumnAndJointTests(unittest.TestCase):
    def test_column_shear_is_joint_limited_and_hoops_are_what_the_cage_can_hold(self):
        st = state()
        result = build_capacity_design(st)
        columns = result["columns"]
        top = columns["stories"]["x"][8]
        self.assertTrue(top["roof_story"])
        self.assertLessEqual(top["ve_kip"], max(top["ve_own_kip"], top["vu_analysis_kip"]))
        self.assertGreaterEqual(top["ve_kip"], top["vu_analysis_kip"])
        self.assertTrue(top["vc_zero"])                       # P_min = 10 kip << Ag fc / 20
        hoops, cage = columns["hoops"], columns["cage"]
        # 4 top bars: at most 4 legs across that face (hoop + 2 crossties); the
        # 6-bar side face needs alternate support (2 crossties) -> 4 legs is the
        # least in both directions, and the legs are chosen per direction.
        self.assertEqual(hoops["legs"], {"across_b_face": 4, "across_h_face": 4})
        self.assertEqual(hoops["legs_model"], 4)
        self.assertEqual({axis: v["legs_key"] for axis, v in hoops["by_direction"].items()},
                         {"x": "across_b_face", "y": "across_h_face"})
        self.assertEqual(columns["stories"]["y"][8]["axis"], "y")
        self.assertEqual(cage["legs_max"], {"across_b_face": 4, "across_h_face": 6})
        self.assertEqual(cage["legs_min"], {"across_b_face": 4, "across_h_face": 4})
        self.assertEqual(cage["constructible_legs"], [4])
        self.assertTrue(all(c["passes"] for c in cage["checks"]))
        self.assertEqual(cage["arrangement"]["h_face"]["supported"], [True, False, True, False, True, True])
        self.assertAlmostEqual(hoops["av_in2"], 4 * 0.20)
        self.assertIn("column.cage_layout", {c["id"] for c in result["checks"]})
        conf = columns["confinement"]
        self.assertAlmostEqual(conf["ash_ratio_required"], max(0.3 * (26 * 26 / (23 * 23) - 1) * 4 / 60, 0.09 * 4 / 60))
        self.assertGreaterEqual(hoops["ash_provided_per_in"], conf["ash_ratio_required"] * conf["bc_in"])
        self.assertLessEqual(hoops["spacing_in"], min(26 / 4.0, 6 * 0.75, conf["so_in"]))

    def test_column_legs_are_chosen_per_direction(self):
        """The dv150 case: 28x28 with #10 bars, 3 top and 4 side per face.

        3 top bars allow at most 3 legs across the b faces; the 6-bar side
        faces need 4. No single count fits both, so a common-count hoop was
        never constructible. Per direction (18.7.5.4 Ash is per direction) the
        cage is 3 legs one way and 4 the other, and Av, Ash and phi Vn are
        evaluated with each direction's own legs and depth.
        """
        st = state(sections={"b_col_in": 28.0, "h_col_in": 28.0, "fc_col_ksi": 8.0},
                   column={"bar_size": 10, "top_bars": 3, "bot_bars": 3, "side_bars": 4, "centroid_offset_in": 2.76,
                           "layers": [(3.81, 2.76), (2.54, 7.26), (2.54, 11.75), (2.54, 16.25), (2.54, 20.74), (3.81, 25.24)],
                           "layers_about_z": [(7.62, 2.76), (2.54, 14.0), (7.62, 25.24)]})
        columns = build_capacity_design(st)["columns"]
        hoops, cage = columns["hoops"], columns["cage"]
        self.assertIsNotNone(hoops)
        self.assertEqual(cage["constructible_legs"], [])
        self.assertEqual(cage["constructible_legs_by_direction"], {"across_b_face": [3], "across_h_face": [4, 5, 6]})
        self.assertEqual(hoops["legs"]["across_b_face"], 3)
        self.assertGreaterEqual(hoops["legs"]["across_h_face"], 4)
        self.assertEqual(hoops["legs_model"], 3)
        self.assertTrue(cage["constructible"] and all(c["passes"] for c in cage["checks"]))
        self.assertEqual(cage["arrangement"]["b_face"]["crossties"], 1)
        self.assertEqual(cage["arrangement"]["h_face"]["crossties"], hoops["legs"]["across_h_face"] - 2)
        ab = {4: 0.20, 5: 0.31}[hoops["bar_size"]]
        x, y = hoops["by_direction"]["x"], hoops["by_direction"]["y"]
        self.assertEqual((x["legs_key"], y["legs_key"]), ("across_b_face", "across_h_face"))
        self.assertAlmostEqual(x["av_in2"], 3 * ab)
        self.assertAlmostEqual(y["av_in2"], hoops["legs"]["across_h_face"] * ab)
        for v in (x, y):
            self.assertGreaterEqual(v["ash_provided_per_in"], v["ash_required_per_in"] - 1e-9)
            self.assertAlmostEqual(v["ash_provided_per_in"], v["av_in2"] / hoops["spacing_in"])
        self.assertAlmostEqual(hoops["ash_provided_per_in"], min(x["ash_provided_per_in"], y["ash_provided_per_in"]))
        # The y frame bends the column through b with its own probable moment.
        self.assertIn("y", columns["stories"])
        self.assertEqual(columns["stories"]["y"][1]["axis"], "y")
        self.assertNotAlmostEqual(columns["stories"]["x"][1]["mpr_column_kip_in"],
                                  columns["stories"]["y"][1]["mpr_column_kip_in"])

    # ACI 318-19 Table 18.8.4.3, read from the standard (printed p. 313) on
    # 2026-09-18 -- gamma by (column, beam in the direction of Vu, confinement
    # by transverse beams per 15.2.8), independent of the production lookup:
    #   continuous column / continuous beam: 20 confined, 15 not confined
    #   continuous column / other beam:      15 confined, 12 not confined
    #   other column / continuous beam:      15 confined, 12 not confined
    #   other column / other beam:           12 confined,  8 not confined
    # Column "continuous or meets 15.2.6" (extends >= h above with bars and
    # hoops continued); beam "continuous or meets 15.2.7" (extends >= h beyond
    # the face with the opposite beam's bars and hoops continued); confined
    # per 15.2.8 by two transverse beams each >= 3/4 of the face width,
    # extending >= h, with >= 2 continuous top and bottom bars and >= No. 3
    # stirrups.
    CONTINUITY = {"beam_reinforcement_continuous_through_interior_joints": True,
                  "column_reinforcement_continuous_through_floor_joints": True, "basis": "test declaration"}

    @staticmethod
    def _joints(st, beams=None):
        strengths = {f"{a}_{p}": probable_beam_strengths(st, p, a) for a in ("x", "y") for p in ("edge", "interior")}
        return design_joint_shear(st, strengths, None, beams)["joints"], strengths

    def test_joint_gamma_follows_the_three_table_inputs_with_narrow_beams(self):
        """10-in beams on 26-in columns: no transverse confinement (0.38 < 3/4); continuity still counts."""
        st = state(joint_continuity=self.CONTINUITY)
        joints, strengths = self._joints(st, beams={"hoops": {"bar_size": 4}})
        expected = {"floor/interior": 15.0, "floor/edge_x": {"x": 15.0, "y": 12.0}, "floor/edge_y": {"x": 12.0, "y": 15.0},
                    "floor/corner": 12.0, "roof/interior": 12.0, "roof/edge_x": {"x": 12.0, "y": 8.0},
                    "roof/edge_y": {"x": 8.0, "y": 12.0}, "roof/corner": 8.0}
        for key, value in expected.items():
            for axis in ("x", "y"):
                gamma = value[axis] if isinstance(value, dict) else value
                entry = joints[f"joint_shear/{key}/{axis}"]
                self.assertEqual(entry["gamma"], gamma, (key, axis, entry["gamma_basis"]))
                self.assertFalse(entry["transverse_confinement"])
                self.assertTrue(entry["evidence_complete"], entry["unevaluated_evidence"])
        interior = joints["joint_shear/floor/interior/x"]
        self.assertEqual((interior["classification"]["column"]["state"], interior["classification"]["beam"]["state"]),
                         ("continuous", "continuous"))
        self.assertEqual(joints["joint_shear/roof/corner/x"]["classification"]["column"]["state"], "other")
        self.assertEqual(joints["joint_shear/floor/corner/x"]["classification"]["beam"]["state"], "other")
        self.assertEqual(interior["aj_in2"], 26.0 * 26.0)
        fam = strengths["x_interior"]
        t_hog = 1.25 * 60.0 * fam["tension_steel_hogging_in2"]
        t_sag = 1.25 * 60.0 * fam["tension_steel_sagging_in2"]
        vcol = (fam["mpr_negative_kip_in"] + fam["mpr_positive_kip_in"]) / 120.0
        self.assertAlmostEqual(interior["vj_kip"], t_hog + t_sag - vcol, places=6)
        self.assertAlmostEqual(joints["joint_shear/roof/interior/x"]["column_shear_kip"], 2.0 * vcol, places=6)
        self.assertAlmostEqual(interior["phi_vn_kip"], 0.85 * 15.0 * math.sqrt(4000.0) * 676.0 / 1000.0, places=6)

    def test_joint_gamma_with_confining_transverse_beams(self):
        """20-in beams on 26-in columns (0.77 >= 3/4): interior joints are confined by their two transverse beams."""
        st = state(sections={"b_beam_in": 20.0}, joint_continuity=self.CONTINUITY)
        joints, _ = self._joints(st, beams={"hoops": {"bar_size": 4}})
        self.assertEqual(joints["joint_shear/floor/interior/x"]["gamma"], 20.0)   # continuous / continuous / confined
        self.assertTrue(joints["joint_shear/floor/interior/x"]["transverse_confinement"])
        self.assertEqual(joints["joint_shear/roof/interior/x"]["gamma"], 15.0)    # other / continuous / confined
        self.assertEqual(joints["joint_shear/floor/edge_x/x"]["gamma"], 15.0)     # one transverse beam: not confined
        self.assertFalse(joints["joint_shear/floor/edge_x/x"]["transverse_confinement"])
        self.assertEqual(joints["joint_shear/floor/edge_x/y"]["gamma"], 15.0)     # terminating beam, two transverse: confined
        self.assertTrue(joints["joint_shear/floor/edge_x/y"]["transverse_confinement"])
        self.assertEqual(joints["joint_shear/roof/edge_x/y"]["gamma"], 12.0)      # other / other / confined
        self.assertEqual(joints["joint_shear/floor/corner/x"]["gamma"], 12.0)     # one transverse beam
        self.assertEqual(joints["joint_shear/roof/corner/x"]["gamma"], 8.0)

    def test_inadequate_transverse_width_never_confines(self):
        st = state(sections={"b_beam_in": 19.0}, joint_continuity=self.CONTINUITY)   # 19/26 = 0.73 < 3/4
        joints, _ = self._joints(st, beams={"hoops": {"bar_size": 4}})
        entry = joints["joint_shear/floor/interior/x"]
        self.assertEqual(entry["gamma"], 15.0)
        self.assertFalse(entry["classification"]["confinement"]["width_ok"])
        self.assertTrue(entry["evidence_complete"])                                   # nothing unknown: it simply fails

    def test_sufficient_width_without_reinforcement_evidence_stays_unconfined(self):
        st = state(sections={"b_beam_in": 20.0}, joint_continuity=self.CONTINUITY)
        joints, _ = self._joints(st, beams=None)                                      # no beam hoops selected yet
        entry = joints["joint_shear/floor/interior/x"]
        self.assertEqual(entry["gamma"], 15.0)
        self.assertFalse(entry["transverse_confinement"])
        self.assertIn("transverse_beam_stirrup_bar_size", entry["unevaluated_evidence"])
        self.assertFalse(entry["evidence_complete"])
        # One continuous top bar only: 15.2.8(c) needs two.
        st = state(sections={"b_beam_in": 20.0}, beam={"top_bars": 1}, joint_continuity=self.CONTINUITY)
        joints, _ = self._joints(st, beams={"hoops": {"bar_size": 4}})
        self.assertEqual(joints["joint_shear/floor/interior/x"]["gamma"], 15.0)
        self.assertFalse(joints["joint_shear/floor/interior/x"]["classification"]["confinement"]["continuous_top_and_bottom_bars"])

    def test_missing_continuity_declaration_keeps_the_conservative_rows(self):
        st = state(sections={"b_beam_in": 20.0})                                      # no joint_continuity at all
        joints, _ = self._joints(st, beams={"hoops": {"bar_size": 4}})
        entry = joints["joint_shear/floor/interior/x"]
        self.assertEqual(entry["gamma"], 8.0)                                         # other / other / not confined
        self.assertEqual({entry["classification"]["column"]["state"], entry["classification"]["beam"]["state"]}, {"other"})
        self.assertFalse(entry["evidence_complete"])
        self.assertEqual(set(entry["unevaluated_evidence"]),
                         {"column_reinforcement_continuous_through_floor_joints",
                          "beam_reinforcement_continuous_through_interior_joints"})
        # The conservative arithmetic never reads as a checked classification: the
        # downstream check stays open and names what is missing (R5).
        self.assertFalse(entry["capacity_topology_and_confinement_checked"])
        from Design.SMRF_Joints import joint_shear_check
        downstream = joint_shear_check(entry, location=entry["id"])
        self.assertEqual(downstream["status"], "not_evaluated")
        self.assertIn("Unevaluated: column_reinforcement", downstream["details"]["reason"])
        # With the declaration and designed hoops the flag is earned, not assumed.
        declared = state(sections={"b_beam_in": 20.0}, joint_continuity=self.CONTINUITY)
        strengths = {f"{a}_{p}": probable_beam_strengths(declared, p, a) for a in ("x", "y") for p in ("edge", "interior")}
        with_hoops = design_joint_shear(declared, strengths, {"hoops": {"bar_size": 4, "legs": {"across_b_face": 4, "across_h_face": 4}}},
                                        {"hoops": {"bar_size": 4}})["joints"]
        self.assertTrue(with_hoops["joint_shear/floor/interior/x"]["capacity_topology_and_confinement_checked"])
        self.assertEqual(joint_shear_check(with_hoops["joint_shear/floor/interior/x"])["status"], "pass")
        without_hoops = design_joint_shear(declared, strengths, None, {"hoops": {"bar_size": 4}})["joints"]
        self.assertFalse(without_hoops["joint_shear/floor/interior/x"]["capacity_topology_and_confinement_checked"])
        # A declaration that says False is a known no, not an unknown.
        st = state(joint_continuity={"beam_reinforcement_continuous_through_interior_joints": False,
                                     "column_reinforcement_continuous_through_floor_joints": True})
        joints, _ = self._joints(st, beams={"hoops": {"bar_size": 4}})
        entry = joints["joint_shear/floor/interior/x"]
        self.assertEqual(entry["gamma"], 12.0)                                        # continuous column / other beam
        self.assertTrue(entry["evidence_complete"])

    def test_dv150_v10_joint_entries_hand_values(self):
        """Case 0003 sections (32x32 fc-4 columns, 16x22 beams): the corrected coefficients on fixed Aj.

        phi Vn = 0.85 gamma sqrt(4000) 1024 / 1000: floor interior 825.734 kip
        (gamma 15), roof interior and roof edge in the two-beam direction
        660.587 (12), floor corner 660.587 (12), roof corner 440.391 (8).
        """
        st = state(sections={"b_col_in": 32.0, "h_col_in": 32.0, "fc_col_ksi": 4.0, "b_beam_in": 16.0, "h_beam_in": 22.0},
                   joint_continuity=self.CONTINUITY)
        joints, _ = self._joints(st, beams={"hoops": {"bar_size": 4}})
        unit = 0.85 * math.sqrt(4000.0) * 1024.0 / 1000.0
        for joint_id, gamma in (("floor/interior/x", 15.0), ("roof/interior/x", 12.0), ("roof/edge_x/x", 12.0),
                                ("floor/corner/x", 12.0), ("roof/corner/x", 8.0), ("floor/edge_x/y", 12.0)):
            entry = joints[f"joint_shear/{joint_id}"]
            self.assertEqual(entry["gamma"], gamma, joint_id)
            self.assertAlmostEqual(entry["phi_vn_kip"], gamma * unit, places=6)
        self.assertAlmostEqual(joints["joint_shear/floor/interior/x"]["phi_vn_kip"], 825.7339, places=3)
        self.assertAlmostEqual(joints["joint_shear/roof/corner/x"]["phi_vn_kip"], 440.3914, places=3)

    def test_joint_hoop_relaxation_is_recorded_per_joint_and_not_applied(self):
        st = state(sections={"b_beam_in": 20.0}, joint_continuity=self.CONTINUITY)
        strengths = {f"{a}_{p}": probable_beam_strengths(st, p, a) for a in ("x", "y") for p in ("edge", "interior")}
        transverse = design_joint_shear(st, strengths, None, {"hoops": {"bar_size": 4}})["joint_transverse"]
        by_joint = transverse["relaxation_18_8_3_2_by_joint"]
        self.assertTrue(by_joint["joint_shear/floor/interior/x"])                    # four beams >= 3/4 width
        self.assertFalse(by_joint["joint_shear/floor/corner/x"])
        self.assertFalse(transverse["relaxation_18_8_3_2_applicable"])
        self.assertFalse(transverse["relaxation_18_8_3_2_applied"])

    def test_hooked_anchorage_and_splices(self):
        st = state()
        anchorage = design_anchorage(st)["directions"]["x"]
        self.assertAlmostEqual(anchorage["ldh_required_in"], 60000.0 * 1.0 / (65.0 * math.sqrt(4000.0)), places=6)
        self.assertAlmostEqual(anchorage["embedment_available_in"], 26.0 - 1.5 - 0.5)
        self.assertTrue(anchorage["passes"])
        splices = design_splices(st)
        # #8 in 8-ksi concrete: ld = 60000 / (20 sqrt(8000)) * 1.0 = 33.5 in; 1.3 ld = 43.6 in
        # against 94 - 2 * 32 = 30 in between the hinge zones: mechanical splices.
        self.assertAlmostEqual(straight_development_length(8, 8.0, 60.0), 60000.0 / (20.0 * math.sqrt(8000.0)), places=6)
        self.assertFalse(splices["beam"]["lap_splice_feasible"])
        self.assertEqual(splices["beam"]["splice_type"], "type_2_mechanical_18.2.7")
        self.assertTrue(splices["column"]["lap_splice_feasible"])
        self.assertEqual(splices["column"]["splice_type"], "class_B_lap_center_half")

    def test_build_reports_checks_and_evidence_groups(self):
        result = build_capacity_design(state(transfer=single_load_transfer()))
        ids = {c["id"] for c in result["checks"]}
        for check_id in ("beam.capacity_shear_section", "column.hoops_selected", "joint.shear_screen",
                         "joint.classification_evidence", "joint.terminating_bar_hook", "detailing.splices_designed"):
            self.assertIn(check_id, ids)
        self.assertEqual(len(result["joint_evidence"]["joint_shear"]), 16)   # 4 kinds x 2 axes x floor/roof
        self.assertEqual(len(result["joint_evidence"]["beam_capacity_shear"]), 8)   # 4 families x 2 gravity cases
        self.assertEqual(result["accepted"], all(c["status"] == "pass" for c in result["checks"]))
        # The fixture carries no continuity declaration: every joint whose classification
        # needs continuity evidence (all floor joints: column continuity; roof joints with two
        # beams in the direction: beam continuity) is open, none failed, the build is not
        # accepted, and the numeric screens are still reported. A roof corner joint, or a
        # roof edge joint in its terminating direction, is Other/Other by topology alone
        # and needs no declaration: its evidence item is complete.
        evidence_items = {c["location"]: c["status"] for c in result["checks"] if c["id"] == "joint.classification_evidence"}
        self.assertEqual(len(evidence_items), 16)
        self.assertEqual(sum(1 for s in evidence_items.values() if s == "not_evaluated"), 12)
        self.assertEqual({k for k, s in evidence_items.items() if s == "pass"},
                         {"joint_shear/roof/corner/x", "joint_shear/roof/corner/y",
                          "joint_shear/roof/edge_x/y", "joint_shear/roof/edge_y/x"})
        self.assertTrue(all(s == "not_evaluated" for k, s in evidence_items.items() if "/floor/" in k))
        self.assertFalse(result["joints"]["evidence_complete"])
        self.assertFalse(result["accepted"])
        declared = build_capacity_design(state(transfer=single_load_transfer(),
                                               joint_continuity=self.CONTINUITY))
        self.assertTrue(all(c["status"] == "pass" for c in declared["checks"] if c["id"] == "joint.classification_evidence"))
        self.assertTrue(declared["joints"]["evidence_complete"])


if __name__ == "__main__":
    unittest.main()
