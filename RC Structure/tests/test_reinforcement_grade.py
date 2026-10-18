"""Longitudinal reinforcement grade by story height (design decision of 2026-10-17).

ASTM A706 Grade 80 longitudinal bars at story heights of 16 ft and more, Grade 60 below; the specification is locked to
A706 of the grade (A615 is refused, ACI 318-19 20.2.2.5); hoops, crossties and stirrups stay A706 Grade 60 (fyt = 60 ksi,
Table 20.2.2.4(a)); the Grade 80 clauses follow the grade: 18.6.3.1 (rho <= 0.02), 18.6.4.4 / 18.7.5.3 (5 db hoop
spacing term), 18.8.2.3 (26 db joint depth), Table 21.2.2 (strain limits at eps_ty = fy / Es) and Table 25.4.2.5
(psi_g = 1.15). Bars larger than No. 11 take Type 2 mechanical splices (25.5.1.1, 18.2.7) at every story height.
"""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp

_KEYS = ("STORY_H", "NUM_MODES", "GEOMETRY_VARIANT_NAME", "SLAB_THICKNESS_IN", "FLOOR_TRANSFER", "SLAB_REINFORCEMENT",
         "SLAB_ACTIONS", "FY_KSI", "REINFORCEMENT_SPECIFICATION")


def _detailing_example(fy_ksi):
    common = {"b_in": 24, "h_in": 24, "fc_ksi": 5, "clear_cover_in": 1.5, "stirrup_db_in": 0.5, "bar_db_in": 1,
              "bar_area_in2": 0.79, "n_top": 4, "n_bottom": 4, "aggregate_size_in": 0.75, "hoop_spacing_in": 6,
              "end_zone_length_in": 48}
    beam = dict(common, b_in=12, n_top=3, n_bottom=3, continuous_top_bars=3, continuous_bottom_bars=3, first_hoop_distance_in=2)
    for key in ("positive_left", "negative_left", "positive_right", "negative_right", "positive_min_along", "negative_min_along"):
        beam[f"mn_{key}_kipin"] = 1000
    return {"material": {"fy_ksi": fy_ksi, "normalweight": True}, "geometry": {"span_x_in": 150, "span_y_in": 180},
            "beam": beam, "column": dict(common, n_side_per_face=2, clear_height_in=120, hx_in=12)}


class GradeStateCase(unittest.TestCase):
    def setUp(self):
        self._saved = {key: copy.deepcopy(getattr(sp, key, None)) for key in _KEYS}

    def tearDown(self):
        for key, value in self._saved.items():
            setattr(sp, key, value)


class GradeByStoryHeightTests(GradeStateCase):
    def test_the_grade_follows_the_story_height(self):
        self.assertEqual(sp.longitudinal_grade_for_story(12.0 * 15.5), 60.0)
        self.assertEqual(sp.longitudinal_grade_for_story(12.0 * 16.0), 80.0)
        self.assertEqual(sp.longitudinal_grade_for_story(12.0 * 17.0), 80.0)
        self.assertEqual(sp.LONGITUDINAL_GRADE_STORY_HEIGHT_FT, 16.0)

    def test_apply_reinforcement_grade_locks_the_specification_to_a706(self):
        self.assertEqual(sp.apply_reinforcement_grade(story_h_in=192.0), 80.0)
        self.assertEqual(sp.FY_KSI, 80.0)
        self.assertEqual(sp.REINFORCEMENT_SPECIFICATION, "ASTM A706 Grade 80")
        self.assertEqual(sp.apply_reinforcement_grade(story_h_in=144.0), 60.0)
        self.assertEqual(sp.REINFORCEMENT_SPECIFICATION, "ASTM A706 Grade 60")
        with self.assertRaises(ValueError):
            sp.apply_reinforcement_grade(fy_ksi=75.0)                   # not an A706 grade of 20.2.2.5
        # the transverse steel never moves
        self.assertEqual(sp.FYT_KSI, 60.0)
        self.assertEqual(sp.TRANSVERSE_SPECIFICATION, "ASTM A706 Grade 60")

    def test_a615_and_a_mismatched_grade_are_refused(self):
        self.assertEqual(sp.require_a706("ASTM A706 Grade 80", 80.0), "ASTM A706 Grade 80")
        for spec, grade in (("ASTM A615 Grade 80", 80.0), ("ASTM A615 Grade 60", 60.0), ("ASTM A706 Grade 60", 80.0),
                            ("ASTM A706 Grade 80", 60.0), (None, 60.0), ("ASTM A706 Grade 100", 100.0)):
            with self.assertRaises(ValueError, msg=f"{spec} at {grade}"):
                sp.require_a706(spec, grade)

    def test_geometry_overrides_switch_the_grade_and_the_design_config_follows(self):
        import Geometry_Overrides as go
        from Design.Config import DesignConfig
        go.apply_geometry_overrides({"STORY_H": 192.0}, emit=False)
        self.assertEqual(sp.FY_KSI, 80.0)
        self.assertEqual(sp.REINFORCEMENT_SPECIFICATION, "ASTM A706 Grade 80")
        cfg = DesignConfig.from_structure_parameters()
        self.assertEqual(cfg.materials.fy_ksi, 80.0)
        self.assertEqual(cfg.materials.reinforcement_specification, "ASTM A706 Grade 80")
        go.apply_geometry_overrides({"STORY_H": 156.0}, emit=False)
        self.assertEqual(sp.FY_KSI, 60.0)
        self.assertEqual(sp.REINFORCEMENT_SPECIFICATION, "ASTM A706 Grade 60")

    def test_the_grade_dependent_rules(self):
        self.assertEqual((sp.beam_rho_max_code(60.0), sp.beam_rho_max_code(80.0)), (0.025, 0.02))        # 18.6.3.1
        self.assertEqual((sp.through_joint_bar_depth_factor(60.0), sp.through_joint_bar_depth_factor(80.0)), (20.0, 26.0))   # 18.8.2.3
        self.assertEqual((sp.hoop_spacing_bar_factor(60.0), sp.hoop_spacing_bar_factor(80.0)), (6.0, 5.0))   # 18.6.4.4, 18.7.5.3
        self.assertAlmostEqual(sp.steel_yield_strain(60.0), 60.0 / 29000.0)                                # 21.2.2.1
        self.assertAlmostEqual(sp.steel_yield_strain(80.0), 80.0 / 29000.0)
        sp.apply_reinforcement_grade(fy_ksi=80.0)                                                          # the defaults follow FY_KSI
        self.assertEqual((sp.beam_rho_max_code(), sp.through_joint_bar_depth_factor(), sp.hoop_spacing_bar_factor()), (0.02, 26.0, 5.0))


class GradeInTheChecksTests(GradeStateCase):
    def test_phi_strain_limits_follow_the_grade(self):
        from Design import ACI_Checks
        import RC_Design_Check
        for phi in (ACI_Checks._phi_PM, RC_Design_Check._phi_PM):
            self.assertGreater(phi(0.0025, 60.0), 0.65)        # above eps_ty = 0.00207 at Grade 60
            self.assertEqual(phi(0.0025, 80.0), 0.65)          # below eps_ty = 0.00276 at Grade 80
            self.assertEqual(phi(0.0055, 60.0), 0.90)          # above 0.00507
            self.assertLess(phi(0.0055, 80.0), 0.90)           # below 0.00576
            self.assertAlmostEqual(phi(80.0 / 29000.0 + 0.0015, 80.0), 0.775)

    def test_development_length_carries_psi_g(self):
        from Design.SMRF_Capacity_Design import straight_development_length
        ld60 = straight_development_length(8, 5.0, 60.0)
        ld80 = straight_development_length(8, 5.0, 80.0)
        self.assertAlmostEqual(ld80 / ld60, 80.0 * 1.15 / 60.0)                                           # Table 25.4.2.5

    def test_transverse_geometry_uses_five_bar_diameters_at_grade_80(self):
        from Design.SMRF_Detailing import select_transverse_geometry, evaluate_detailing
        sixty, eighty = select_transverse_geometry(_detailing_example(60)), select_transverse_geometry(_detailing_example(80))
        self.assertEqual(sixty["hoop_spacing_bar_diameters"], 6.0)
        self.assertEqual(eighty["hoop_spacing_bar_diameters"], 5.0)
        self.assertEqual(sixty["beam"]["spacing_bounds"]["longitudinal_diameters_in"], 6.0)                # 1 in bars
        self.assertEqual(eighty["beam"]["spacing_bounds"]["longitudinal_diameters_in"], 5.0)
        self.assertEqual(eighty["beam"]["hoop_spacing_in"], 5)                                             # min(d/4 = 5.375, 5 db, 6)
        self.assertEqual(eighty["longitudinal_grade_ksi"], 80)
        with self.assertRaises(ValueError):
            select_transverse_geometry(_detailing_example(100))
        # the evaluation accepts Grade 80 and tightens the end-hoop bound: d/4 = 5.375 in, 5 db = 5 in with 1 in bars
        data = _detailing_example(80)
        data["beam"]["hoop_spacing_in"] = 5.2
        checks = {c["id"]: c for c in evaluate_detailing(data) if c["id"] in ("detailing.material_scope", "beam.end_hoop_spacing")}
        self.assertEqual(checks["detailing.material_scope"]["status"], "pass")
        self.assertEqual(checks["beam.end_hoop_spacing"]["status"], "fail")                               # 5.2 > 5 db = 5 in
        data60 = _detailing_example(60)
        data60["beam"]["hoop_spacing_in"] = 5.2
        sixty_checks = {c["id"]: c for c in evaluate_detailing(data60)}
        self.assertEqual(sixty_checks["beam.end_hoop_spacing"]["status"], "pass")                         # 5.2 <= min(5.375, 6, 6)

    def test_beam_ratio_ceiling_and_joint_depth_follow_the_grade(self):
        from Design.Config import DesignConfig
        from Design import Group_Selection as selection
        cfg = DesignConfig.from_structure_parameters()
        cfg.rebar.rho_beam_practical_max = 0.025                      # expose the code limit
        sp.apply_reinforcement_grade(fy_ksi=80.0)
        self.assertEqual(min(sp.beam_rho_max_code(), cfg.rebar.rho_beam_practical_max), 0.02)
        # a 24 in through depth takes a No. 9 bar at Grade 60 (20 x 1.128 = 22.6) and refuses it at Grade 80 (26 x 1.128 = 29.3)
        self.assertLessEqual(20.0 * sp.rebar_diameter(9), 24.0)
        self.assertGreater(sp.through_joint_bar_depth_factor() * sp.rebar_diameter(9), 24.0)
        sp.apply_reinforcement_grade(fy_ksi=60.0)
        self.assertLessEqual(sp.through_joint_bar_depth_factor() * sp.rebar_diameter(9), 24.0)
        self.assertTrue(hasattr(selection, "evaluate_beam_candidate"))

    def test_slab_mats_keep_grade_60_beside_grade_80_beam_bars(self):
        from Design.SMRF_Beam_Slab_Strength import section_moment, hogging_strength_kip_in
        self.assertEqual(sp.SLAB_FY_KSI, 60.0)
        self.assertEqual(sp.SLAB_REINFORCEMENT_SPECIFICATION, "ASTM A706 Grade 60")
        # hogging: the compression face is the beam bottom; the top bars sit at 25.5 in and the slab mats at 27 in from it
        web, slab = (25.5, 3.0), (27.0, 1.2)
        all_80 = section_moment([web, slab], 5.0, 80.0, 28.0, 18.0)["mn_kip_in"]
        mixed = section_moment([web, slab + (60.0,)], 5.0, 80.0, 28.0, 18.0)["mn_kip_in"]
        all_60 = section_moment([web, slab], 5.0, 60.0, 28.0, 18.0)["mn_kip_in"]
        self.assertLess(all_60, mixed)
        self.assertLess(mixed, all_80)
        # credited (depth, area) slab layers are priced at SLAB_FY_KSI by default
        credited = hogging_strength_kip_in(3.0, 25.5, 28.0, 18.0, 5.0, 80.0, [(27.0, 1.2)])
        self.assertAlmostEqual(credited, mixed)
        # a face string in the third slot (the layout triples) is not a grade
        self.assertAlmostEqual(section_moment([web, (27.0, 1.2, "top")], 5.0, 80.0, 28.0, 18.0)["mn_kip_in"], all_80)

    def test_records_name_the_transverse_grade_and_the_rule(self):
        from types import SimpleNamespace
        from Design import Grouped_Record as gr
        from Design.Config import DesignConfig
        sp.apply_reinforcement_grade(story_h_in=192.0)
        cfg = DesignConfig.from_structure_parameters()
        state = SimpleNamespace(groups={"c": {"group_id": "c", "member_type": "column"}, "b": {"group_id": "b", "member_type": "beam"}},
                                designs={"c": SimpleNamespace(fc_ksi=6.0), "b": SimpleNamespace(fc_ksi=5.0)})
        material = gr._materials(cfg, state)
        self.assertEqual((material["fy_ksi"], material["fyt_ksi"]), (80.0, 60.0))
        self.assertEqual(material["reinforcement_specification"], "ASTM A706 Grade 80")
        self.assertEqual(material["transverse_specification"], "ASTM A706 Grade 60")
        self.assertEqual(material["longitudinal_grade_rule"]["story_height_ft_for_grade_80"], 16.0)

    def test_the_qualification_scope_accepts_a706_of_either_grade_only(self):
        from Design.SMRF_Qualification import reinforcement_consistency_checks
        record = {"materials": {"reinforcement_specification": "ASTM A706 Grade 80", "exposure": "sheltered_interior",
                                "fy_ksi": 80, "fyt_ksi": 60, "aggregate_size_in": 0.75}}
        checks = {c["id"]: c for c in reinforcement_consistency_checks(record)}
        self.assertEqual(checks["reinforcement.declared_material_scope"]["status"], "pass")
        for spec, grade in (("ASTM A615 Grade 80", 80), ("ASTM A706 Grade 60", 80), ("ASTM A706 Grade 80", 60)):
            record["materials"].update(reinforcement_specification=spec, fy_ksi=grade)
            checks = {c["id"]: c for c in reinforcement_consistency_checks(record)}
            self.assertEqual(checks["reinforcement.declared_material_scope"]["status"], "fail", msg=f"{spec} at {grade}")


if __name__ == "__main__":
    unittest.main()
