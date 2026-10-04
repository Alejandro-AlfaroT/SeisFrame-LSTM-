"""Column sizing rules of 2 October 2026 (user decision): the ACI 318-19 18.7.3.1 exception where the column is
discontinuous above the connection and Pu < Ag f'c / 10; the beam-delivery-limited column design shear for the uniform
design; the column step-down pass after the first passing candidate."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Design import Design_Driver as driver                                     # noqa: E402
from Design import SMRF_Joints as joints                                       # noqa: E402
from Design.Config import DesignConfig                                         # noqa: E402
from Design.Section_Design import column_ladder                                # noqa: E402
from Design.SMRF_Capacity_Design import COLUMN_SHEAR_METHOD_COLUMN_OWN, COLUMN_SHEAR_METHOD_JOINT_LIMITED  # noqa: E402


def sway(column_mn=5000.0, axial_max=50.0, discontinuous=True, area=576.0, fc=5.0, columns=1, connection=True):
    column = {"axial_envelope_checked": True, "factored_axial_kip": 20.0, "mn_kip_in": column_mn,
              "axial_min_kip": 10.0, "axial_max_kip": axial_max}
    beam = {"slab_basis": "no_slab", "slab_mn_kip_in": 0.0, "mn_kip_in": 4000.0}
    state = {"nominal_strengths": True, "column_capacities": [dict(column) for _ in range(columns)],
             "beam_capacities": [dict(beam), dict(beam)]}
    if connection:
        state["connection"] = {"column_discontinuous_above": discontinuous, "column_gross_area_in2": area, "column_fc_ksi": fc}
    return state


class DiscontinuousColumnException(unittest.TestCase):
    def test_a_roof_column_with_low_axial_load_is_excepted_and_recorded(self):
        check = joints.scwb_check(sway(), location="joint_9/x/positive")
        self.assertEqual(check["status"], "pass")
        self.assertEqual(check["clause"], joints.SCWB_EXCEPTION_CLAUSE)
        self.assertEqual((check["demand"], check["capacity"], check["units"]), (50.0, 576.0 * 5.0 / 10.0, "kip"))
        details = check["details"]
        self.assertTrue(details["roof_exemption_applied"])
        self.assertAlmostEqual(details["ratio_provided"], 5000.0 / 8000.0)      # recorded, below 1.2, not required
        self.assertIsNone(details["ratio_required"])
        self.assertEqual(details["exception"]["clause"], "ACI 318-19 18.7.3.1")

    def test_the_rule_applies_when_any_condition_is_missing(self):
        for label, state in (("axial at the limit", sway(axial_max=288.0)),
                             ("axial above the limit", sway(axial_max=400.0)),
                             ("column continues above", sway(discontinuous=False)),
                             ("two columns at the joint", sway(columns=2, column_mn=2500.0)),
                             ("no connection evidence", sway(connection=False))):
            check = joints.scwb_check(state)
            self.assertEqual(check["clause"], joints.SCWB_CLAUSE, label)
            self.assertEqual(check["status"], "fail", label)
            self.assertFalse(check["details"]["roof_exemption_applied"], label)
            self.assertEqual(check["units"], "kip-in", label)

    def test_a_missing_axial_envelope_keeps_the_rule_and_bad_evidence_is_not_a_pass(self):
        state = sway()
        del state["column_capacities"][0]["axial_max_kip"]
        self.assertEqual(joints.scwb_check(state)["clause"], joints.SCWB_CLAUSE)
        broken = sway(area=0.0)
        self.assertEqual(joints.scwb_check(broken)["status"], "not_evaluated")

    def test_the_switch_restores_the_rule_everywhere(self):
        original = joints.SCWB_DISCONTINUOUS_COLUMN_EXCEPTION
        try:
            joints.SCWB_DISCONTINUOUS_COLUMN_EXCEPTION = False
            self.assertEqual(joints.scwb_check(sway())["status"], "fail")
        finally:
            joints.SCWB_DISCONTINUOUS_COLUMN_EXCEPTION = original

    def test_the_adapter_marks_roof_connections_only(self):
        from Design.SMRF_Joint_Adapter import build_joint_evidence
        record = {"geometry": {"num_bay_x": 1, "num_bay_y": 1, "num_floor": 2},
                  "sections": {"b_col_in": 24.0, "h_col_in": 24.0, "fc_col_ksi": 5.0}}
        evidence = build_joint_evidence(record, [], expected_combination_ids=["c1"])
        flags = {(joint["floor"], joint["directions"]["x"]["positive"]["connection"]["column_discontinuous_above"])
                 for joint in evidence["joints"]}
        self.assertEqual(flags, {(1, False), (2, True)})
        connection = evidence["joints"][0]["directions"]["y"]["negative"]["connection"]
        self.assertEqual((connection["column_gross_area_in2"], connection["column_fc_ksi"]), (576.0, 5.0))


class UniformColumnShearRule(unittest.TestCase):
    def test_the_uniform_design_reads_the_beam_limited_rule_and_the_grouped_rule_is_unchanged(self):
        from Design.SMRF_Capacity_Design import COLUMN_SHEAR_METHOD_ANALYSIS_SPLIT
        cfg = DesignConfig()
        self.assertEqual(cfg.capacity.uniform_column_shear_method, COLUMN_SHEAR_METHOD_ANALYSIS_SPLIT)   # since 2026-10-04
        self.assertNotEqual(COLUMN_SHEAR_METHOD_ANALYSIS_SPLIT, COLUMN_SHEAR_METHOD_JOINT_LIMITED)
        self.assertEqual(cfg.capacity.column_shear_method, COLUMN_SHEAR_METHOD_COLUMN_OWN)




class RoofColumnExtension(unittest.TestCase):
    """Joint shear option 1 (user decision 2026-10-03): a declared column extension of one column depth above the
    roof joint, reinforcement continued, makes the roof column continuous for Table 18.8.4.3 (ACI 318-19 15.2.6)."""

    def classify(self, level, continuity, beam_width=20.0):
        from Design.SMRF_Capacity_Design import classify_joint_shear_inputs
        return classify_joint_shear_inputs(level, 2, 2, beam_width, 28.0, 30.0, 28.0, 200.0, 200.0, 126.0, (5, 5), 4, continuity)

    def test_the_declared_extension_moves_the_roof_joint_off_the_other_row(self):
        base = {"beam_reinforcement_continuous_through_interior_joints": True,
                "column_reinforcement_continuous_through_floor_joints": True}
        stub = {**base, "roof_column_extension_in": 28.0, "column_reinforcement_continued_through_roof_extension": True}
        without, with_stub = self.classify("roof", base), self.classify("roof", stub)
        self.assertEqual((without["column"]["state"], without["gamma"]), ("other", 12.0))
        self.assertEqual((with_stub["column"]["state"], with_stub["gamma"]), ("continuous", 15.0))
        self.assertIn("15.2.6", with_stub["column"]["basis"])
        self.assertEqual(self.classify("roof", stub, beam_width=24.0)["gamma"], 20.0)       # and confined by wide beams
        self.assertEqual(self.classify("floor", stub)["gamma"], self.classify("floor", base)["gamma"])  # floors unchanged

    def test_a_short_extension_or_undeclared_reinforcement_stays_other(self):
        base = {"beam_reinforcement_continuous_through_interior_joints": True,
                "column_reinforcement_continuous_through_floor_joints": True}
        for extra in ({"roof_column_extension_in": 27.0, "column_reinforcement_continued_through_roof_extension": True},
                      {"roof_column_extension_in": 28.0, "column_reinforcement_continued_through_roof_extension": False},
                      {"roof_column_extension_in": 28.0}):
            result = self.classify("roof", {**base, **extra})
            self.assertEqual((result["column"]["state"], result["gamma"]), ("other", 12.0), extra)

    def test_the_design_declares_one_column_depth_and_the_switch_removes_it(self):
        import Structure_Parameters as sp
        from unittest import mock
        declared = driver._joint_continuity_declaration()
        self.assertEqual(declared["roof_column_extension_in"], max(sp.B_COL, sp.H_COL))
        self.assertTrue(declared["column_reinforcement_continued_through_roof_extension"])
        with mock.patch.object(sp, "ROOF_COLUMN_EXTENSION", False):
            self.assertNotIn("roof_column_extension_in", driver._joint_continuity_declaration())


PASSING = {"drift_ok": True, "scwb_ok": True, "joint_scwb_failed": False, "capacity_accepted": True,
           "beam_section_adequate": True, "beam_hoops_selected": True, "column_section_adequate": True,
           "column_hoops_selected": True, "joints_all_pass": True, "anchorage_all_pass": True, "joint_shear_ratio": None}


class SearchRepairsAfterTheSecondScreen(unittest.TestCase):
    """The r2 screen of 2 October: an overstressed beam that the moment rule would leave in place widens; a joint
    short only of confinement width widens the beam before the column grows; bar rows past mid-depth are a named
    error the search can step on."""

    def setUp(self):
        from Design.Section_Design import beam_ladder
        self.columns = column_ladder()
        self.beams = beam_ladder(span_in=216.0, story_height_in=144.0)
        self.ci = self.columns.index((26.0, 26.0, 5.0))
        self.bi = self.beams.index((12.0, 22.0, 5.0))

    def plan(self, worst, **flags):
        return driver._plan_next_rungs(self.columns, self.beams, self.ci, self.bi, worst, 0.85, 1.0, self.ci, {**PASSING, **flags})

    def test_an_overstressed_beam_the_moment_rule_leaves_in_place_widens(self):
        column, beam, reasons = self.plan({"beam": 2.5, "column": 0.5, "beam_flexure_demand_kip_in": 100.0})
        self.assertIn("beam_strength", reasons)
        self.assertGreater(beam, self.bi)
        self.assertEqual(self.beams[beam][1], 22.0)                              # same depth,
        self.assertGreater(self.beams[beam][0], 12.0)                            # wider web
        self.assertEqual(column, self.ci)

    def test_a_wide_beam_that_cannot_be_reinforced_grows_the_column_in_size_not_grade(self):
        """case_0010 of the r4 screen: the section carries the moment, the beam is already three quarters of the
        column, and it sits on its seed bars: the column lanes are the limit, so the column takes the next size at
        the same grade and the beam is held."""
        ci = self.columns.index((22.0, 22.0, 10.0))
        bi = self.beams.index((20.0, 22.0, 5.0))
        column, beam, reasons = driver._plan_next_rungs(self.columns, self.beams, ci, bi,
                                                        {"beam": 6.1, "column": 0.5, "beam_flexure_demand_kip_in": 100.0},
                                                        0.85, 1.0, ci, dict(PASSING))
        self.assertEqual(reasons, ["beam_strength", "beam_bars_need_column_room"])
        self.assertEqual(self.columns[column], (23.0, 23.0, 10.0))
        self.assertEqual(beam, bi)
        self.assertIn("beam_bars_need_column_room", driver.COLUMN_STEP_REASONS)
        self.assertIsNone(driver._next_larger_column_same_grade(self.columns, len(self.columns) - 1))

    def test_a_joint_short_of_confinement_width_widens_the_beam_first(self):
        worst = {"beam": 0.8, "column": 0.4}
        column, beam, reasons = self.plan(worst, joints_all_pass=False, capacity_accepted=False, joint_shear_ratio=1.1,
                                          joint_confinement_width_in=0.75 * 26.0)
        self.assertEqual(reasons, ["joint_shear_or_anchorage", "joint_confinement_width"])
        self.assertEqual(column, self.ci)                                        # the column waits
        self.assertEqual(self.beams[beam][1:], (22.0, 5.0))
        self.assertGreaterEqual(self.beams[beam][0], 19.5)
        # no variant that wide at this depth, or an anchorage failure: the column grows as before
        for flags in ({"joint_confinement_width_in": 40.0}, {"joint_confinement_width_in": None},
                      {"joint_confinement_width_in": 19.5, "anchorage_all_pass": False}):
            column, beam, reasons = self.plan(worst, joints_all_pass=False, capacity_accepted=False, joint_shear_ratio=1.1, **flags)
            self.assertNotIn("joint_confinement_width", reasons)
            # strength before geometry (2026-10-03): the same size at the top grade, or a larger column where
            # the joint-shear ratio asks for more than the top grade gives; never the same rung
            self.assertGreater(column, self.ci)
            self.assertEqual(self.columns[column][2], 10.0)
        self.assertIn("joint_confinement_width", driver.BEAM_STEP_REASONS)

    def test_bar_rows_past_mid_depth_raise_the_named_error(self):
        from Design.SMRF_Beam_Slab_Strength import BeamBarRowsError, validate_bar_rows
        rows = {"per_layer": [2, 2], "offsets_in": [2.7, 8.5]}
        with self.assertRaises(BeamBarRowsError) as caught:
            validate_bar_rows(rows, 4, "top bars", 16.0)
        self.assertIsInstance(caught.exception, ValueError)
        self.assertIn("past mid-depth of a 16 in section", str(caught.exception))
        validate_bar_rows(rows, 4, "top bars", 24.0)                              # deep enough: accepted


if __name__ == "__main__":
    unittest.main()
