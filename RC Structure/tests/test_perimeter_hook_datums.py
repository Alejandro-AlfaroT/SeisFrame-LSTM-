"""The perimeter hook of the slab mats at two datums and its location factor (2026-11-01).

  P  psi_o by ACI 318-19 Table 25.4.3.2 on the declared corner convention: the fixture's corners (26 in columns,
     18 in beams) are inside the column core with covers of 6 in (psi_o 1.0); narrow 16 in columns leave covers of
     2 in, so psi_o is 1.25 and 25.4.3.4 asks for ties the mats do not have; an end at an edge column keeps 1.0;
     missing geometry raises; the #6 at 3 in hook at 5 ksi is 18.70 in at psi_o 1.25 against 16.0 in (the row
     fails, the screen rejects that spacing) and 14.96 in at 1.0; a corner needing ties leaves no layout;
  S/D2  an end that develops at the support face but not at the joint face (16 in column, 18 in perimeter beam,
     #6 at 4 in: 15.53 in against 16.0 and 15.0 in) is undeveloped for sizing, developed for the strong-column rule,
     and the hinges keep the support face; D4: 18.6.3.2 reads the larger negative at the face;
  I  per end and sign: sizing <= hinge <= strong-column <= composite, and Mpr >= the strong-column value; on a
     layout developed at the support face everywhere the strong-column values equal the hinge values;
  K  a sentinel in the strong-column keys changes nothing in sizing, the hinge properties or the saved end
     strengths; the strength cache key follows each exterior end's column;
  U  the uniform route's context rebuild reproduces the corner hooks and flags a changed column.

The fixture frame is not a designed structure. No solve.
"""
import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402
from test_group_composite_strength import slab_layout                            # noqa: E402
from test_smrf_slab_reinforcement import CORNERS, evidence, inputs              # noqa: E402

import Structure_Parameters as sp                                               # noqa: E402
from Design import Group_Checks as checks                                        # noqa: E402
from Design import Group_Selection as selection                                  # noqa: E402
from Design import Grouped_Design as gd                                          # noqa: E402
from Design import Grouped_Record as gr                                          # noqa: E402
from Design.SMRF_Beam_Slab_Strength import (HOOK_CORNER_CONVENTION, corner_hook_geometry,  # noqa: E402
                                            hook_development_length_in, hook_location_factor,
                                            perimeter_slab_bar_anchorage, uniform_perimeter_hook_corners)
from Design.SMRF_Qualification import uniform_slab_context_check                 # noqa: E402
from Design.SMRF_Slab_Reinforcement import _hook_rejected_spacings, design_slab_reinforcement  # noqa: E402
from Model import IMK_Hinges                                                     # noqa: E402
from Model import Member_Bar_Layers as bl                                        # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402
from Model import Member_Properties as mp                                        # noqa: E402


def corner(side, in_plane, core):
    return {"side_cover_in": side, "in_plane_cover_in": in_plane, "in_column_core": core,
            "convention": HOOK_CORNER_CONVENTION}


class LocationFactor(unittest.TestCase):
    def test_hand_values_of_the_corner_case(self):
        self.assertAlmostEqual(hook_development_length_in(6, 5.0, 60.0, 3.0, 1.25)[0], 18.705, places=2)
        self.assertAlmostEqual(hook_development_length_in(6, 5.0, 60.0, 3.0)[0], 14.964, places=2)
        with self.assertRaises(ValueError):
            hook_development_length_in(6, 5.0, 60.0, 3.0, 1.1)

    def test_table_25_4_3_2_conditions(self):
        self.assertEqual(hook_location_factor(6)["psi_o"], 1.0)                         # an edge column's end
        self.assertEqual(hook_location_factor(6, corner(3.0, 3.0, True))["psi_o"], 1.0)  # in the core, >= 2.5 in
        self.assertEqual(hook_location_factor(6, corner(4.5, 3.0, False))["psi_o"], 1.0)  # >= 6 db
        self.assertEqual(hook_location_factor(6, corner(3.0, 3.0, False))["psi_o"], 1.25)
        self.assertEqual(hook_location_factor(6, corner(2.4, 3.0, True))["psi_o"], 1.25)  # in the core, < 2.5 in
        self.assertFalse(hook_location_factor(6, corner(2.0, 3.0, False))["confinement_required_25_4_3_4"])
        self.assertTrue(hook_location_factor(6, corner(2.0, 2.0, False))["confinement_required_25_4_3_4"])

    def test_missing_geometry_is_never_psi_o_one(self):
        for bad in ({"side_cover_in": 7.0}, corner(None, 7.0, True), corner(7.0, 7.0, 1),
                    {**corner(7.0, 7.0, True), "convention": "another"}, corner(float("nan"), 7.0, True)):
            with self.subTest(corner=bad), self.assertRaises((ValueError, TypeError)):
                hook_location_factor(6, bad)

    def test_the_row_fails_and_the_screen_rejects_at_psi_o_1_25(self):
        layout = slab_layout(6, 3.0)
        low = perimeter_slab_bar_anchorage(layout, "x", 18.0, 1.5, 0.5, 5.0, 60.0, corner=corner(3.0, 3.0, False))
        self.assertEqual((low["psi_o"], low["developed"]), (1.25, False))
        self.assertAlmostEqual(low["ldh_required_in"], 18.705, places=2)
        self.assertAlmostEqual(low["embedment_available_in"], 16.0)
        self.assertTrue(perimeter_slab_bar_anchorage(layout, "x", 18.0, 1.5, 0.5, 5.0, 60.0, corner=CORNERS["x"])["developed"])
        context = {"beam_width_in": 18.0, "beam_clear_cover_in": 1.5, "beam_hoop_diameter_in": 0.5, "fc_beam_ksi": 5.0,
                   "perimeter_hook_corners": {"x": CORNERS["x"], "y": corner(3.0, 3.0, False)}}
        rejected, embedment = _hook_rejected_spacings(6, {"fy_ksi": 60.0}, {"spacing_options_in": [3.0, 6.0]}, context)
        self.assertEqual((sorted(rejected), embedment), ([3.0], 16.0))
        context["perimeter_hook_corners"]["y"] = CORNERS["y"]
        self.assertEqual(_hook_rejected_spacings(6, {"fy_ksi": 60.0}, {"spacing_options_in": [3.0, 6.0]}, context)[0], {})

    def test_a_corner_that_needs_ties_leaves_no_layout(self):
        from test_slab_structural_gates import CONTEXT
        tied = {**CONTEXT, "perimeter_hook_corners": {"x": corner(2.0, 2.0, False), "y": CORNERS["y"]}}
        result = design_slab_reinforcement(inputs(), evidence(), context=tied)
        self.assertIsNone(result["layout"])
        self.assertTrue(all("25.4.3.4" in trial["reason"] for trial in result["trial_history"] if "reason" in trial))


class HookDatums(unittest.TestCase):
    def setUp(self):
        self.frame = gf.frame(BEAM_BAR_STACKING="x_over_y")
        self.frame.__enter__()
        self.state = mg.install(gf.uniform_state())

    def tearDown(self):
        self.frame.__exit__(None, None, None)

    def narrow_columns(self, size=16.0, groups=None):
        designs = {gid: replace(d, b_in=size, h_in=size) for gid, d in self.state.designs.items()
                   if d.is_column and (groups is None or gid in groups)}
        self.state = mg.install(self.state.with_designs(designs))

    def beams(self):
        return [m for m in self.state.members() if not m.is_column]

    def test_the_corner_geometry_of_the_fixture(self):
        self.assertEqual(mp.perimeter_hook_corner(1, 0, 0, "x"), corner(6.0, 6.0, True))
        self.assertIsNone(mp.perimeter_hook_corner(1, 1, 0, "x"))
        self.assertEqual(gd._perimeter_hook_corners(), {"x": corner(6.0, 6.0, True), "y": corner(6.0, 6.0, True)})
        self.narrow_columns()
        self.assertEqual(mp.perimeter_hook_corner(1, 0, 0, "x"), corner(2.0, 2.0, False))
        self.assertEqual(corner_hook_geometry(30.0, 30.0, 1.5, 0.625, 20.0, 0.5, 26.0, 0.5, 1.5), corner(7.0, 4.0, True))

    def test_a_reverse_geometry_end_develops_at_the_support_face_only(self):
        self.narrow_columns()
        sp.SLAB_REINFORCEMENT = {"layout": slab_layout(6, 4.0)}
        member = mg.beam_at("beam_x", 1, 0, 1)                 # an interior frame line: end i at an edge column
        strengths = mp.beam_strengths(member)
        basis, family = strengths["basis"], strengths["family"]
        hook = basis["exterior_anchorage_by_end"]["i"]
        self.assertIsNone(mp.exterior_end_anchorage_inputs(member)["i"]["corner"])
        self.assertEqual((hook["developed_support_face"], hook["developed_joint_face"]), (True, False))
        self.assertAlmostEqual(hook["support_face"]["ldh_required_in"], 15.535, places=2)
        self.assertAlmostEqual(hook["support_face"]["embedment_available_in"], 16.0)
        self.assertAlmostEqual(hook["joint_face"]["embedment_available_in"], 15.0)
        composite, rectangular = family["mn_negative_kip_in"], family["mn_undeveloped_negative_kip_in"]
        self.assertGreater(composite, rectangular)
        self.assertEqual(basis["hogging_i_kip_in"], composite)              # the hinges: the support face
        self.assertEqual(basis["scwb_hogging_i_kip_in"], composite)         # the strong-column rule: either datum
        self.assertEqual(basis["sizing_hogging_i_kip_in"], rectangular)     # beam sizing: both datums
        rows = bl.member_rows(member)
        resolved = checks.resolved_beam_flexural_strengths(member, rows)
        self.assertAlmostEqual(resolved["nominal_sections"]["i"]["negative"], rectangular)
        self.assertAlmostEqual(resolved["nominal_negative_faces_18_6_3_2"]["i"], composite)
        detail = checks.beam_detailing_strengths(member, rows)
        self.assertAlmostEqual(detail["mn_negative_left_kipin"], composite)                         # D4
        self.assertAlmostEqual(detail["mn_positive_left_kipin"], resolved["nominal_sections"]["i"]["positive"])
        self.assertEqual(gr.beam_end_strengths(self.state)[str(member.member_tag)]["hogging_i_kip_in"], composite)

    def test_a_corner_that_needs_ties_is_developed_at_neither_datum(self):
        self.narrow_columns()
        sp.SLAB_REINFORCEMENT = {"layout": slab_layout(4, 12.0)}
        hook = mp.beam_strengths(mg.beam_at("beam_x", 1, 0, 0))["basis"]["exterior_anchorage_by_end"]["i"]
        self.assertTrue(hook["support_face"]["confinement_required_25_4_3_4"])
        self.assertEqual(hook["support_face"]["psi_o"], 1.25)
        self.assertFalse(hook["developed_either"])

    def test_the_end_strength_order_holds_on_every_beam(self):
        # The datums nest (both => support face => either). In hogging the mats only add strength, so sizing <= hinge
        # <= strong-column <= composite. In sagging each per-end value is one of the two sections the datum rule picks:
        # the composite section is not always the stronger there (this fixture: 4275.39 composite against 4276.57
        # kip-in without mats, the mats sitting near the neutral axis), so no order is claimed in sagging.
        for narrow, layout in ((False, slab_layout(4, 12.0)), (False, slab_layout(8, 5.0)), (True, slab_layout(6, 4.0)),
                               (True, slab_layout(4, 12.0))):
            with self.subTest(narrow=narrow, bar=layout["bar_size"]):
                if narrow:
                    self.narrow_columns()
                sp.SLAB_REINFORCEMENT = {"layout": layout}
                for member in self.beams():
                    strengths, probable = mp.beam_strengths(member), mp.beam_strengths(member, 1.25)
                    basis, family = strengths["basis"], strengths["family"]
                    for end in ("i", "j"):
                        hook = basis["exterior_anchorage_by_end"].get(end)
                        rules = {"sizing_": True, "": True, "scwb_": True}
                        if hook is not None:
                            self.assertTrue(not hook["developed_both"] or hook["developed_support_face"])
                            self.assertTrue(not hook["developed_support_face"] or hook["developed_either"])
                            rules = {"sizing_": hook["developed_both"], "": hook["developed_support_face"],
                                     "scwb_": hook["developed_either"]}
                        hogging = [basis[f"{prefix}hogging_{end}_kip_in"] for prefix in rules] + [strengths["hogging"]]
                        self.assertEqual(hogging, sorted(hogging), (member.member_tag, end))
                        for sign, short in (("hogging", "negative"), ("sagging", "positive")):
                            for prefix, developed in rules.items():
                                expected = family[f"mn_{short}_kip_in" if developed else f"mn_undeveloped_{short}_kip_in"]
                                self.assertEqual(basis[f"{prefix}{sign}_{end}_kip_in"], expected,
                                                 (member.member_tag, prefix, sign, end))
                            self.assertGreaterEqual(probable[sign], basis[f"scwb_{sign}_{end}_kip_in"])
                self.state = mg.install(gf.uniform_state())

    def test_where_the_support_face_develops_the_strong_column_values_are_the_hinge_values(self):
        sp.SLAB_REINFORCEMENT = {"layout": slab_layout(4, 12.0)}
        for member in self.beams():
            basis = mp.beam_strengths(member)["basis"]
            self.assertTrue(all(r["developed_support_face"] for r in basis["exterior_anchorage_by_end"].values()))
            for sign in ("hogging", "sagging"):
                for end in ("i", "j"):
                    self.assertEqual(basis[f"scwb_{sign}_{end}_kip_in"], basis[f"{sign}_{end}_kip_in"])

    def test_a_sentinel_in_the_strong_column_keys_reaches_no_other_consumer(self):
        self.narrow_columns()
        sp.SLAB_REINFORCEMENT = {"layout": slab_layout(6, 4.0)}
        member = mg.beam_at("beam_x", 1, 0, 1)
        rows = bl.member_rows(member)

        def numbers(resolved):
            return {key: value for key, value in resolved.items() if key != "basis"}   # the basis is echoed, not read
        before = (numbers(checks.resolved_beam_flexural_strengths(member, rows)),
                  IMK_Hinges._resolved_member_properties(member)[0], gr.beam_end_strengths(self.state))
        real = mp.beam_strengths

        def inflated(*args, **kwargs):
            result = copy.deepcopy(real(*args, **kwargs))
            for key in list(result["basis"]):
                if key.startswith("scwb_"):
                    result["basis"][key] = 1.0e12
            return result
        with mock.patch.object(mp, "beam_strengths", inflated):
            after = (numbers(checks.resolved_beam_flexural_strengths(member, rows)),
                     IMK_Hinges._resolved_member_properties(member)[0], gr.beam_end_strengths(self.state))
        self.assertEqual(before, after)

    def test_the_selection_requirement_sums_both_sways_on_the_strong_column_basis(self):
        # At a joint on the i = 0 line the only beam is on the plus side: positive sway hogs it (its larger strength),
        # negative sway sags it; the requirement is 1.2 times the larger of the two sway sums.
        sp.SLAB_REINFORCEMENT = {"layout": slab_layout(4, 12.0)}
        requirements = selection._joint_requirements(self.state, True)
        for (level, i, j), entry in requirements.items():
            joint = mg.joint_members(level, i, j)
            for axis in ("x", "y"):
                sums = {"positive": 0.0, "negative": 0.0}
                for side, end in (("minus", "j"), ("plus", "i")):
                    beam = joint[f"beam_{axis}_{side}"]
                    if beam is None:
                        continue
                    basis = mp.beam_strengths(beam)["basis"]
                    sums["positive"] += basis[f"scwb_{'sagging' if end == 'j' else 'hogging'}_{end}_kip_in"]
                    sums["negative"] += basis[f"scwb_{'hogging' if end == 'j' else 'sagging'}_{end}_kip_in"]
                self.assertAlmostEqual(entry[axis], sp.SCWB_RATIO_MIN * max(sums.values()), places=9,
                                       msg=(level, i, j, axis))
        hog = mp.beam_strengths(mg.beam_at("beam_x", 1, 0, 1))["basis"]
        self.assertGreater(hog["scwb_hogging_i_kip_in"], hog["scwb_sagging_i_kip_in"])
        self.assertAlmostEqual(requirements[(1, 0, 1)]["x"], sp.SCWB_RATIO_MIN * hog["scwb_hogging_i_kip_in"], places=9)

    def test_the_cache_key_follows_each_exterior_end_s_column(self):
        sp.SLAB_REINFORCEMENT = {"layout": slab_layout(6, 4.0)}
        interior, edge = mg.beam_at("beam_x", 1, 0, 1), mg.beam_at("beam_x", 1, 0, 0)
        keys = (checks.beam_section_context(interior), checks.beam_section_context(edge))
        boundary = [gid for gid in self.state.designs if "__column__" in gid and "boundary_edge" in gid]
        self.narrow_columns(groups=[gid for gid in boundary
                                    if gid == mp.joint_core_column(1, 0, 1).group_id])
        self.assertNotEqual(checks.beam_section_context(mg.beam_at("beam_x", 1, 0, 1)), keys[0])
        self.assertEqual(checks.beam_section_context(mg.beam_at("beam_x", 1, 0, 0)), keys[1])


class UniformContextRebuild(unittest.TestCase):
    def record(self, column=26.0):
        corners = uniform_perimeter_hook_corners(26.0, 26.0, 1.5, 0.5, 18.0, 0.5, 1.5)
        context = {"clear_span_x_in": 214.0, "clear_span_y_in": 190.0, "beam_width_in": 18.0,
                   "column_core_width_in": 26.0 - 2.0 * 2.5, "beam_clear_cover_in": 1.5, "beam_hoop_diameter_in": 0.5,
                   "fc_beam_ksi": 4.0, "perimeter_hook_corners": corners}
        return {"slab_reinforcement": {"inputs": {"context": context}},
                "sections": {"h_col_in": column, "b_col_in": column, "b_beam_in": 18.0, "fc_beam_ksi": 4.0},
                "reinforcement": {"col_longitudinal_centroid_offset_in": 2.5, "beam_clear_cover_in": 1.5,
                                  "beam_stirrup_diameter_in": 0.5, "col_clear_cover_in": 1.5,
                                  "col_stirrup_diameter_in": 0.5},
                "geometry": {"bay_x_in": 214.0 + column, "bay_y_in": 190.0 + column}}

    def test_the_rebuild_reproduces_the_corner_hooks_and_flags_a_changed_column(self):
        self.assertEqual(uniform_slab_context_check(self.record())["status"], "pass")
        changed = self.record()
        changed["sections"].update(h_col_in=16.0, b_col_in=16.0)
        changed["geometry"] = {"bay_x_in": 214.0 + 26.0, "bay_y_in": 190.0 + 26.0}
        check = uniform_slab_context_check(changed)
        self.assertEqual(check["status"], "fail")
        self.assertIn("perimeter_hook_corners", check["details"]["fields_that_differ"])


if __name__ == "__main__":
    unittest.main()
