"""Repairs from the review of the grouped design (2 October 2026), each pinned by its counterexample.

  R1  vertical regularity follows ASCE 7-22 Table 12.3-2: soft and extreme soft story (Types 1a, 1b) with
      the 12.3.2.2 drift-ratio exception, weak and extreme weak story (Types 4a, 4b); no weight (mass)
      irregularity, whose ratios stay as a description without a check;
  R2  a regularity check never passes on missing, duplicated or unusable story evidence;
  R4  an axial load is placed against both ends of the strength surface before any shortcut, in the
      grouped routines (scalar and vectorised) and in the two shared uniform helpers;
  R5  the reinforcement pass is settled only when the longitudinal selection and the hoops stop changing
      in the same pass, and the evidence it returns describes the design left installed.

The last class holds what reading ASCE/SEI 7-22 itself added afterwards (12.3.2.2 exception 2, an exception 1 that
is never shown by an empty list, 12.3.3.3). R3 (bar coordinates through a column transition) is in test_smrf_transitions. The inputs here are synthetic
bookkeeping; none of them is a frame result.
"""
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp                                              # noqa: E402
import RC_Design_Check as legacy_check                                          # noqa: E402
from Design import ACI_Checks as aci                                            # noqa: E402
from Design import Group_Checks as checks                                       # noqa: E402
from Design import Grouped_Design as gd                                         # noqa: E402
from Design.Config import DesignConfig                                          # noqa: E402

NF, HEIGHT = 6, 168.0
FORCES = [10.0] * NF                                                           # story shears 60, 50, 40, 30, 20, 10 kip


def drift_screen(stiffness, drift_ratio=None, load_factor=1.0, skip=(), extra=()):
    """Drift rows whose elastic drift gives the stated story stiffness (kip/in) under FORCES, both directions."""
    rows = []
    for axis in ("x", "y"):
        for k in range(1, NF + 1):
            if (k, axis) in skip:
                continue
            shear = sum(FORCES[k - 1:])
            drift = shear / stiffness[k - 1]
            rows.append({"location": f"story:{k}/Q{axis}/{axis}", "elastic_drift_in": drift,
                         "drift_ratio": (drift * 5.5 / HEIGHT) if drift_ratio is None else drift_ratio[k - 1]})
    return {"stories": rows + list(extra), "assumptions": {"rho_for_drift_load": load_factor}}


def strength(values=(100.0,) * NF, status="provisional"):
    return {"applicability": {"status": status},
            "stories": [{"story": k, "by_direction": {axis: {"story_strength_kip": values[k - 1]} for axis in ("x", "y")}}
                        for k in range(1, NF + 1)]}


def ledger(weights=(100.0,) * NF):
    return {"floors": [{"floor": k, "seismic_weight_kip": weights[k - 1]} for k in range(1, NF + 1)]}


ELF = {"story_forces_kip": FORCES, "base_shear_kip": sum(FORCES)}
UNIFORM = [100.0] * NF


def evaluate(screen=None, elf=ELF, strengths=None, weights=None, sdc="D", verified=False):
    with mock.patch.object(sp, "NUM_FLOOR", NF):
        vertical = gd.vertical_regularity(drift_screen(UNIFORM) if screen is None else screen, elf,
                                          strength() if strengths is None else strengths, ledger() if weights is None else weights)
    result = {c["id"]: c for c in gd.regularity_checks(vertical, sdc, strength_model_verified=verified)}
    return vertical, result


STIFFNESS, STRENGTH = "demands.vertical_stiffness_regularity", "demands.vertical_strength_regularity"


class VerticalRegularityEdition(unittest.TestCase):
    def test_the_table_is_the_2022_one(self):
        limits = gd.VERTICAL_THRESHOLDS
        self.assertEqual(limits["edition"], "ASCE 7-22")
        self.assertEqual(limits["soft_story_1a"], {"ratio_to_story_above": 0.70, "ratio_to_average_of_three_above": 0.80})
        self.assertEqual(limits["extreme_soft_story_1b"], {"ratio_to_story_above": 0.60, "ratio_to_average_of_three_above": 0.70})
        self.assertEqual((limits["drift_ratio_exception"]["limit"], limits["drift_ratio_exception"]["applies_to"]), (1.30, ["1a", "1b"]))
        self.assertEqual(limits["weak_story_4a"]["ratio_to_story_above"], 1.00)
        self.assertEqual(limits["weak_story_4a"]["sdc_e_f_permitted_at_or_above"], 0.80)
        self.assertEqual(limits["extreme_weak_story_4b"]["ratio_to_story_above"], 0.65)
        text = repr(limits)
        for stale in ("5a", "5b", "1.5", "mass_ratio"):
            self.assertNotIn(stale, text)

    def test_there_is_no_mass_check_and_no_mass_threshold(self):
        heavy = (100.0, 100.0, 400.0, 100.0, 100.0, 100.0)                     # one story four times its neighbours
        vertical, result = evaluate(weights=ledger(heavy))
        self.assertEqual(set(result), {STIFFNESS, STRENGTH})
        self.assertEqual(result[STIFFNESS]["status"], "pass")                  # a heavy story does not touch the stiffness check
        weights = vertical["story_weights"]
        self.assertIn("no threshold and no check", weights["status"])
        self.assertAlmostEqual(weights["rows"][2]["ratio_to_story_above"], 4.0, places=12)
        self.assertNotIn("mass", vertical["indicated"])
        self.assertNotIn("mass", vertical)

    def test_a_regular_frame_passes_stiffness_and_keeps_strength_open(self):
        vertical, result = evaluate()
        self.assertEqual(result[STIFFNESS]["status"], "pass")
        self.assertIn("Types 1a, 1b", result[STIFFNESS]["clause"])
        self.assertEqual(result[STRENGTH]["status"], "not_evaluated")
        self.assertIn("Types 4a, 4b", result[STRENGTH]["clause"])
        self.assertEqual(len(vertical["stiffness"]), 2 * (NF - 1))
        self.assertTrue(all(row["indicated_type"] is None for row in vertical["stiffness"]))

    def test_a_soft_story_is_type_1a_and_stays_open_for_review_when_the_exception_does_not_hold(self):
        soft = [100.0, 75.0, 100.0, 100.0, 100.0, 100.0]                       # story 2 at 75% of the three stories above
        vertical, result = evaluate(drift_screen(soft))
        rows = vertical["indicated"]["soft_story_1a"]
        self.assertEqual({(r["story"], r["direction"]) for r in rows}, {(2, "x"), (2, "y")})
        self.assertEqual(vertical["indicated"]["extreme_soft_story_1b"], [])
        self.assertFalse(vertical["indicated"]["drift_ratio_exception_holds"])  # story 2 drifts 1.67 times story 3
        self.assertEqual(result[STIFFNESS]["status"], "not_evaluated")
        self.assertIn("soft story (Type 1a)", repr(result[STIFFNESS]))
        self.assertEqual(result[STIFFNESS]["details"]["category"], "stiffness irregularity indicated")

    def test_the_three_story_average_criterion_is_applied_where_three_stories_exist_above(self):
        stiff_above = [100.0, 100.0, 100.0, 130.0, 130.0, 130.0]               # story 3 at 77% of the three above
        vertical, _result = evaluate(drift_screen(stiff_above))
        row = next(r for r in vertical["stiffness"] if (r["story"], r["direction"]) == (3, "x"))
        self.assertAlmostEqual(row["ratio_to_story_above"], 100.0 / 130.0, places=12)
        self.assertAlmostEqual(row["ratio_to_average_of_three_above"], 100.0 / 130.0, places=12)
        self.assertEqual(row["indicated_type"], "1a")                          # above 0.70 of the next story, below 0.80 of the average
        top = next(r for r in vertical["stiffness"] if (r["story"], r["direction"]) == (4, "x"))
        self.assertIsNone(top["ratio_to_average_of_three_above"])              # only two stories above story 4

    def test_an_extreme_soft_story_is_type_1b_prohibited_in_sdc_e_and_f_only(self):
        extreme = [100.0, 55.0, 100.0, 100.0, 100.0, 100.0]
        vertical, in_d = evaluate(drift_screen(extreme), sdc="D")
        self.assertEqual({r["story"] for r in vertical["indicated"]["extreme_soft_story_1b"]}, {2})
        self.assertEqual(vertical["indicated"]["soft_story_1a"], [])
        self.assertEqual(in_d[STIFFNESS]["status"], "not_evaluated")
        for sdc in ("E", "F"):
            _vertical, result = evaluate(drift_screen(extreme), sdc=sdc)
            self.assertEqual(result[STIFFNESS]["status"], "fail")
            self.assertIn("12.3.3.1", result[STIFFNESS]["clause"])

    def test_the_drift_ratio_exception_removes_types_1a_and_1b_and_skips_the_top_two_stories(self):
        soft = [100.0, 65.0, 100.0, 100.0, 100.0, 100.0]
        even = [0.010] * NF                                                    # no story drifts more than 1.3 times the one above
        vertical, result = evaluate(drift_screen(soft, drift_ratio=even))
        self.assertTrue(vertical["indicated"]["drift_ratio_exception_holds"])
        self.assertEqual(result[STIFFNESS]["status"], "pass")
        self.assertIn("exception holds", result[STIFFNESS]["details"]["basis"])
        self.assertEqual({r["story"] for r in vertical["indicated"]["drift_ratio_exception_rows"]}, {1, 2, 3, 4})
        top_heavy = [0.010, 0.010, 0.010, 0.010, 0.030, 0.010]                 # only story 5 against story 6 exceeds 1.3
        vertical, result = evaluate(drift_screen(soft, drift_ratio=top_heavy))
        self.assertTrue(vertical["indicated"]["drift_ratio_exception_holds"])
        self.assertEqual(result[STIFFNESS]["status"], "pass")
        lower = [0.010, 0.020, 0.010, 0.010, 0.010, 0.010]                     # story 2 at twice story 3
        vertical, result = evaluate(drift_screen(soft, drift_ratio=lower))
        self.assertFalse(vertical["indicated"]["drift_ratio_exception_holds"])
        self.assertEqual(result[STIFFNESS]["status"], "not_evaluated")
        # the exception is about stiffness: a mass or strength ratio is never excused by it
        _vertical, result = evaluate(drift_screen(UNIFORM, drift_ratio=even), strengths=strength((50.0, 100.0, 100.0, 100.0, 100.0, 100.0)))
        self.assertEqual(result[STRENGTH]["status"], "not_evaluated")

    def test_weak_story_rows_are_types_4a_and_4b_and_stay_open_under_the_provisional_model(self):
        values = (90.0, 100.0, 60.0, 100.0, 100.0, 100.0)                      # story 1 at 90%, story 3 at 60% of the story above
        vertical, result = evaluate(strengths=strength(values))
        weak = {(r["story"], r["direction"]): r for r in vertical["indicated"]["weak_story_4a_provisional"]}
        extreme = {(r["story"], r["direction"]) for r in vertical["indicated"]["extreme_weak_story_4b_provisional"]}
        self.assertEqual(set(weak), {(1, "x"), (1, "y")})
        self.assertTrue(weak[(1, "x")]["at_or_above_80_percent"])
        self.assertEqual(extreme, {(3, "x"), (3, "y")})
        self.assertEqual(result[STRENGTH]["status"], "not_evaluated")
        self.assertEqual(result[STRENGTH]["details"]["category"], "awaiting review item M1")
        self.assertIn("neither established nor excluded", result[STRENGTH]["details"]["reason"])
        self.assertEqual(len(result[STRENGTH]["details"]["extreme_weak_story_4b_rows"]), 2)
        # the block calling itself verified changes nothing without the record's assertion
        _vertical, result = evaluate(strengths=strength(values, status="verified"))
        self.assertEqual(result[STRENGTH]["status"], "not_evaluated")

    def test_with_an_asserted_strength_model_the_prohibitions_of_12_3_3_follow_the_category(self):
        self.assertEqual(evaluate(verified=True)[1][STRENGTH]["status"], "pass")
        extreme = strength((100.0, 100.0, 60.0, 100.0, 100.0, 100.0))
        for sdc in ("D", "E", "F"):
            self.assertEqual(evaluate(strengths=extreme, sdc=sdc, verified=True)[1][STRENGTH]["status"], "fail", sdc)   # Type 4b
        self.assertEqual(evaluate(strengths=extreme, sdc="C", verified=True)[1][STRENGTH]["status"], "not_evaluated")
        weak_70 = strength((100.0, 70.0, 100.0, 100.0, 100.0, 100.0))          # Type 4a below 80%
        weak_90 = strength((100.0, 90.0, 100.0, 100.0, 100.0, 100.0))          # Type 4a at or above 80%
        self.assertEqual(evaluate(strengths=weak_70, sdc="E", verified=True)[1][STRENGTH]["status"], "fail")
        self.assertEqual(evaluate(strengths=weak_90, sdc="E", verified=True)[1][STRENGTH]["status"], "not_evaluated")
        self.assertEqual(evaluate(strengths=weak_70, sdc="D", verified=True)[1][STRENGTH]["status"], "not_evaluated")


class MissingStoryEvidence(unittest.TestCase):
    def assert_open(self, screen=None, elf=ELF, fragment=""):
        vertical, result = evaluate(screen, elf)
        self.assertFalse(vertical["evidence"]["stiffness_complete"])
        self.assertEqual(vertical["stiffness"], [])                            # no ratio is formed from unusable evidence
        self.assertIsNone(vertical["indicated"]["soft_story_1a"])
        self.assertIsNone(vertical["indicated"]["drift_ratio_exception_holds"])
        self.assertEqual(result[STIFFNESS]["status"], "not_evaluated")
        self.assertIn(fragment, " | ".join(vertical["evidence"]["stiffness_problems"]))
        return vertical

    def test_the_reviewed_probe_no_drift_rows_is_not_evaluated(self):
        vertical = self.assert_open({"stories": []}, {"story_forces_kip": FORCES}, "no drift row")
        self.assertIn("load basis", " | ".join(vertical["evidence"]["stiffness_problems"]))

    def test_one_missing_row_is_enough(self):
        self.assert_open(drift_screen(UNIFORM, skip={(3, "y")}), fragment="no drift row for story/direction 3/y")

    def test_a_duplicated_row_is_refused(self):
        twin = {"location": "story:2/Qx/x", "elastic_drift_in": 0.4, "drift_ratio": 0.01}
        self.assert_open(drift_screen(UNIFORM, extra=[twin]), fragment="more than one drift row")

    def test_nan_zero_and_negative_values_are_refused(self):
        for bad in (float("nan"), 0.0, -0.2, None, True):
            screen = drift_screen(UNIFORM)
            screen["stories"][0]["elastic_drift_in"] = bad
            self.assert_open(screen, fragment="not a finite positive number")
            screen = drift_screen(UNIFORM)
            screen["stories"][0]["drift_ratio"] = bad
            self.assert_open(screen, fragment="not a finite positive number")

    def test_an_unreadable_or_foreign_row_is_refused(self):
        self.assert_open(drift_screen(UNIFORM, extra=[{"location": "roof"}]), fragment="no readable story")
        self.assert_open(drift_screen(UNIFORM, extra=[{"location": "story:9/Qx/x", "elastic_drift_in": 0.1, "drift_ratio": 0.01}]),
                         fragment="outside the frame")

    def test_the_elf_shear_basis_must_be_coherent(self):
        self.assert_open(elf={"story_forces_kip": FORCES[:-1]}, fragment="not 6 finite non-negative values")
        self.assert_open(elf={"story_forces_kip": [10.0, 10.0, float("nan"), 10.0, 10.0, 10.0]}, fragment="not 6 finite")
        self.assert_open(elf={"story_forces_kip": [0.0] * NF}, fragment="base shear is not positive")
        self.assert_open(elf={"story_forces_kip": FORCES, "base_shear_kip": 75.0}, fragment="recorded base shear")
        self.assert_open(elf={"story_forces_kip": [10.0, 10.0, 10.0, 10.0, 10.0, 0.0]}, fragment="carries no ELF shear")
        self.assert_open(elf=None, fragment="not 6 finite")
        self.assert_open(drift_screen(UNIFORM, load_factor=1.3), fragment="load basis")

    def test_missing_story_strength_leaves_the_strength_check_open_as_incomplete(self):
        partial = strength()
        partial["stories"].pop(2)
        vertical, result = evaluate(strengths=partial, verified=True)
        self.assertFalse(vertical["evidence"]["strength_complete"])
        self.assertEqual(result[STRENGTH]["status"], "not_evaluated")
        self.assertEqual(result[STRENGTH]["details"]["category"], "incomplete story evidence")
        broken = strength((100.0, float("nan"), 100.0, 100.0, 100.0, 100.0))
        self.assertEqual(evaluate(strengths=broken, verified=True)[1][STRENGTH]["status"], "not_evaluated")


SURFACE = [(-100.0, 0.0), (0.0, 1000.0), (100.0, 0.0)]                         # phi P from -100 (tension) to 100 kip; phi M 1000 at P = 0
DIAGRAMS = {"y": SURFACE, "z": SURFACE}


class AxialDomain(unittest.TestCase):
    CASES = (  # (P, M, expected DCR, basis, domain failure)
        (-1000.0, 0.0, 10.0, "axial_tension_outside_surface", "tension"),      # the reviewed probe: it returned -10
        (-1000.0, 5.0, 10.0, "axial_tension_outside_surface", "tension"),
        (-100.0, 0.0, 1.0, "axial", None),                                     # at the tension end
        (-100.0, 5.0, 999.0, "no_flexural_strength_at_axial_load", "moment_at_axial_end"),
        (-50.0, 0.0, 0.5, "axial", None),                                      # inside, in tension
        (-50.0, 250.0, 0.5, "biaxial_load_contour", None),                     # moment 250 of 500; axial ratio 0.5
        (0.0, 500.0, 0.5, "biaxial_load_contour", None),
        (50.0, 0.0, 0.5, "axial", None),
        (100.0, 0.0, 1.0, "axial", None),                                      # at the compression cap
        (100.0, 5.0, 999.0, "no_flexural_strength_at_axial_load", "moment_at_axial_end"),
        (150.0, 0.0, 1.5, "axial_compression_above_cap", "compression"),
        (150.0, 5.0, 1.5, "axial_compression_above_cap", "compression"),
    )

    def test_scalar_ratios_are_non_negative_and_named_at_inside_and_outside_both_ends(self):
        for p, m, expected, basis, failure in self.CASES:
            result = checks.column_pm_dcr(DIAGRAMS, p, m, 0.0)
            self.assertAlmostEqual(result["dcr"], expected, places=9, msg=(p, m))
            self.assertEqual((result["basis"], result["domain_failure"]), (basis, failure), (p, m))
            self.assertGreaterEqual(result["dcr"], 0.0)

    def test_the_vectorised_routine_is_the_scalar_one_row_by_row(self):
        rows = [(i, "c", p, m, 0.0) for i, (p, m, *_rest) in enumerate(self.CASES)]
        envelope = checks.column_pm_envelope(DIAGRAMS, rows)
        for value, (p, m, expected, *_rest) in zip(envelope["values"], self.CASES):
            self.assertAlmostEqual(value, expected, places=9, msg=(p, m))
        self.assertEqual(envelope["domain_failures"], {"compression": 2, "tension": 2, "moment_at_axial_end": 2})
        self.assertTrue(envelope["outside_domain"])
        self.assertEqual(checks.column_pm_dcr_max(DIAGRAMS, rows), (envelope["dcr"], envelope["index"]))
        inside = checks.column_pm_envelope(DIAGRAMS, [(1, "c", -50.0, 250.0, 0.0), (2, "c", 50.0, 0.0, 0.0)])
        self.assertFalse(inside["outside_domain"])
        self.assertAlmostEqual(inside["dcr"], 0.5, places=9)

    def test_a_surface_without_a_tension_branch_gives_the_named_failure_for_any_tension(self):
        compression_only = {"y": [(0.0, 500.0), (100.0, 0.0)], "z": [(0.0, 500.0), (100.0, 0.0)]}
        result = checks.column_pm_dcr(compression_only, -5.0, 0.0, 0.0)
        self.assertEqual((result["dcr"], result["domain_failure"]), (aci.DOMAIN_FAILURE_DCR, "tension"))
        self.assertEqual(checks.column_pm_envelope(compression_only, [(1, "c", -5.0, 0.0, 0.0)])["dcr"], aci.DOMAIN_FAILURE_DCR)

    def test_the_axial_helpers(self):
        self.assertEqual(aci.axial_limits(SURFACE), (100.0, -100.0))
        self.assertEqual(aci.axial_limits(SURFACE, [(-60.0, 0.0), (0.0, 1.0), (140.0, 0.0)]), (100.0, -60.0))   # the common domain
        self.assertEqual(aci.axial_demand_ratio(-50.0, 100.0, -100.0), 0.5)
        self.assertEqual(aci.axial_demand_ratio(50.0, 100.0, -100.0), 0.5)
        self.assertEqual(aci.axial_demand_ratio(-1000.0, 100.0, -100.0), 10.0)

    def test_the_shared_uniform_check_places_tension_against_the_tension_end(self):
        cfg = DesignConfig.from_structure_parameters()
        pure = aci.check_column_pm(-1000.0, 0.0, 0.0, DIAGRAMS, cfg)
        self.assertAlmostEqual(pure.dcr, 10.0, places=12)
        self.assertFalse(pure.ok)
        self.assertEqual(pure.capacity, -100.0)
        with_moment = aci.check_column_pm(-1000.0, 5.0, 5.0, DIAGRAMS, cfg)
        self.assertAlmostEqual(with_moment.dcr, 10.0, places=12)
        self.assertFalse(with_moment.ok)
        inside = aci.check_column_pm(-50.0, 0.0, 0.0, DIAGRAMS, cfg)
        self.assertAlmostEqual(inside.dcr, 0.5, places=12)
        self.assertTrue(inside.ok)
        at_end = aci.check_column_pm(-100.0, 0.0, 0.0, DIAGRAMS, cfg)
        self.assertAlmostEqual(at_end.dcr, 1.0, places=12)
        self.assertTrue(at_end.ok)
        # compression is unchanged
        self.assertAlmostEqual(aci.check_column_pm(50.0, 0.0, 0.0, DIAGRAMS, cfg).dcr, 0.5, places=12)
        self.assertAlmostEqual(aci.check_column_pm(150.0, 0.0, 5.0, DIAGRAMS, cfg).dcr, 1.5, places=12)
        self.assertAlmostEqual(aci.check_column_pm(0.0, 0.0, 500.0, DIAGRAMS, cfg).dcr, 0.5, places=12)
        # and the grouped routine agrees with it on every case
        for p, m, *_rest in AxialDomain.CASES:
            self.assertAlmostEqual(aci.check_column_pm(p, 0.0, m, DIAGRAMS, cfg).dcr, checks.column_pm_dcr(DIAGRAMS, p, m, 0.0)["dcr"],
                                   places=9, msg=(p, m))

    def test_the_older_resultant_moment_check_does_the_same(self):
        dcr, ok, _moment, _capacity = legacy_check.check_column_PM(-1000.0, 0.0, 0.0, SURFACE)
        self.assertAlmostEqual(dcr, 10.0, places=12)
        self.assertFalse(ok)
        self.assertEqual(legacy_check.check_column_PM(-1000.0, 3.0, 4.0, SURFACE)[:2], (10.0, False))
        self.assertEqual(legacy_check.check_column_PM(-50.0, 0.0, 0.0, SURFACE)[:2], (0.5, True))
        self.assertEqual(legacy_check.check_column_PM(50.0, 0.0, 0.0, SURFACE)[:2], (0.5, True))
        self.assertEqual(legacy_check.check_column_PM(150.0, 0.0, 0.0, SURFACE)[:2], (1.5, False))
        self.assertEqual(legacy_check.check_column_PM(0.0, 300.0, 400.0, SURFACE)[:2], (0.5, True))


class FakeState:
    """A stand-in for a grouped design that records the hoop updates it was given."""

    def __init__(self, hoops=()):
        self.hoops = tuple(hoops)

    def with_designs(self, updates):
        return FakeState(self.hoops + tuple(sorted(updates)))


class SettlementContract(unittest.TestCase):
    def run_pass(self, selection_results, hoop_results, margins=(1.0,), max_passes=6):
        """reinforcement_pass with scripted selection reports and hoop updates; returns (result, capacities built)."""
        selections, hoops, built = iter(selection_results), iter(hoop_results), []

        def select(state, actions, cfg, scwb_margin=1.0):
            return state, {"settled": next(selections), "exhausted_groups": []}

        def build(actions, expected_ids, cfg, transfers, state):
            built.append(state)
            return {"built_for": state}

        import Design.SMRF_Joints as joints
        with mock.patch.object(gd.selection, "select_reinforcement", select), mock.patch.object(gd.mg, "install"), \
                mock.patch.object(gd.capacity_design, "build_group_capacity_design", build), \
                mock.patch.object(gd.capacity_design, "hoop_updates", lambda capacity, state: next(hoops)), \
                mock.patch.object(gd, "joint_check_inputs", return_value={}), mock.patch.object(joints, "evaluate_joints", return_value=[]):
            return gd.reinforcement_pass(FakeState(), [], [], None, None, max_passes=max_passes, scwb_margins=margins), built

    def test_the_reviewed_probe_unsettled_selection_with_still_hoops_is_not_settled(self):
        result, built = self.run_pass([False] * 6, [{}] * 6)
        self.assertFalse(result["settled"])
        self.assertFalse(result["selection_settled"])
        self.assertTrue(result["hoops_settled"])
        self.assertEqual(len(result["log"]), 6)                                # every pass was spent; nothing was relaxed
        self.assertEqual({entry["selection_settled"] for entry in result["log"]}, {False})
        self.assertEqual(len(built), 6)

    def test_both_must_settle_in_the_same_pass(self):
        # pass 1: selection settled, hoops change; pass 2: hoops still, selection not; pass 3: both
        result, built = self.run_pass([True, False, True], [{"g1": 1}, {}, {}])
        self.assertTrue(result["settled"])
        self.assertEqual([(e["selection_settled"], e["hoops_settled"]) for e in result["log"]],
                         [(True, False), (False, True), (True, True)])
        self.assertEqual(result["state"].hoops, ("g1",))
        self.assertIs(result["capacity"]["built_for"], result["state"])        # the evidence is of the settled design
        self.assertFalse(result["evidence_rebuilt_on_final_design"])

    def test_hoops_that_keep_changing_are_a_search_outcome_with_evidence_of_the_final_design(self):
        result, built = self.run_pass([True] * 6, [{f"g{k}": 1} for k in range(6)])
        self.assertFalse(result["settled"])
        self.assertTrue(result["selection_settled"])
        self.assertFalse(result["hoops_settled"])
        self.assertTrue(result["evidence_rebuilt_on_final_design"])
        self.assertEqual(len(result["state"].hoops), 6)                        # all six updates are installed
        self.assertEqual(len(built), 7)                                        # six passes and one rebuild
        self.assertIs(result["capacity"]["built_for"], result["state"])        # and the capacity design describes them
        self.assertIsNot(built[5], result["state"])                            # the last in-loop capacity design did not

    def test_the_candidate_constraints_carry_both_conditions(self):
        import inspect
        source = inspect.getsource(gd.evaluate_candidate)
        for key in ("longitudinal_selection_settled", "hoops_settled", "slab_layout_settled", "slab_layout_kept"):
            self.assertIn(f'"{key}"', source)
        self.assertIn('constraints["slab_layout_settled"] and layout_kept', source)


class LowRiseExceptionAndPrimarySource(unittest.TestCase):
    """ASCE/SEI 7-22 p. 120, read from the standard: 12.3.2.2 exceptions 1 and 2, 12.3.3.3, and the table's source."""

    @staticmethod
    def frame(nf, stiffness, drift_ratio=None, sdc="D", rows_to_drop=()):
        forces = [10.0] * nf
        rows = []
        for axis in ("x", "y"):
            for k in range(1, nf + 1):
                if (k, axis) in rows_to_drop:
                    continue
                drift = sum(forces[k - 1:]) / stiffness[k - 1]
                rows.append({"location": f"story:{k}/Q{axis}/{axis}", "elastic_drift_in": drift,
                             "drift_ratio": (drift * 5.5 / HEIGHT) if drift_ratio is None else drift_ratio[k - 1]})
        screen = {"stories": rows, "assumptions": {"rho_for_drift_load": 1.0}}
        elf = {"story_forces_kip": forces, "base_shear_kip": sum(forces)}
        strengths = {"applicability": {"status": "provisional"},
                     "stories": [{"story": k, "by_direction": {axis: {"story_strength_kip": 100.0} for axis in ("x", "y")}}
                                 for k in range(1, nf + 1)]}
        weights = {"floors": [{"floor": k, "seismic_weight_kip": 100.0} for k in range(1, nf + 1)]}
        with mock.patch.object(sp, "NUM_FLOOR", nf):
            vertical = gd.vertical_regularity(screen, elf, strengths, weights)
        return vertical, {c["id"]: c for c in gd.regularity_checks(vertical, sdc)}

    def test_the_table_names_its_source_and_both_exceptions(self):
        limits = gd.VERTICAL_THRESHOLDS
        self.assertIn("p. 120", limits["source"])
        self.assertEqual(limits["drift_ratio_exception"]["section"], "12.3.2.2 exception 1")
        self.assertEqual(limits["low_rise_exception"]["section"], "12.3.2.2 exception 2")
        self.assertEqual(limits["low_rise_exception"]["two_story_sdc"], ["B", "C", "D"])
        self.assertEqual(limits["low_rise_exception"]["applies_to"], ["1a", "1b"])
        self.assertEqual(set(limits["prohibitions"]), {"12.3.3.1", "12.3.3.2", "12.3.3.3"})
        self.assertIn("less than that in the story above", limits["weak_story_4a"]["reading"])

    def test_a_two_story_frame_in_sdc_b_c_or_d_is_covered_by_exception_2_and_says_so(self):
        for sdc in ("B", "C", "D"):
            vertical, result = self.frame(2, [50.0, 100.0], sdc=sdc)             # first story at half the second
            self.assertEqual(vertical["num_stories"], 2)
            self.assertEqual({r["story"] for r in vertical["indicated"]["extreme_soft_story_1b"]}, {1})   # the ratio is still recorded
            self.assertEqual(result[STIFFNESS]["status"], "pass", sdc)
            self.assertIn("exception 2", result[STIFFNESS]["details"]["basis"])
            self.assertIn("Types 1a, 1b", result[STIFFNESS]["clause"])
            self.assertEqual(result[STIFFNESS]["details"]["num_stories"], 2)

    def test_exception_1_is_not_shown_by_an_empty_list_of_story_relationships(self):
        # two stories: the only relationship is that of the top two stories, which exception 1 leaves out
        vertical, _result = self.frame(2, [50.0, 100.0], drift_ratio=[0.010, 0.010])
        self.assertEqual(vertical["indicated"]["drift_ratio_exception_rows"], [])
        self.assertFalse(vertical["indicated"]["drift_ratio_exception_evaluable"])
        self.assertIs(vertical["indicated"]["drift_ratio_exception_holds"], False)

    def test_a_two_story_frame_in_sdc_e_or_f_is_not_covered(self):
        for sdc in ("E", "F"):
            _vertical, result = self.frame(2, [50.0, 100.0], sdc=sdc)            # Type 1b
            self.assertEqual(result[STIFFNESS]["status"], "fail", sdc)
            self.assertIn("12.3.3.1", result[STIFFNESS]["clause"])
            self.assertIn("no story relationship to evaluate", result[STIFFNESS]["details"]["basis"])
            _vertical, result = self.frame(2, [65.0, 100.0], sdc=sdc)            # Type 1a
            self.assertEqual(result[STIFFNESS]["status"], "not_evaluated", sdc)
            self.assertIn("no story relationship to evaluate", result[STIFFNESS]["details"]["reason"])
            _vertical, result = self.frame(2, [100.0, 100.0], sdc=sdc)           # nothing indicated
            self.assertEqual(result[STIFFNESS]["status"], "pass", sdc)
            self.assertIn("no story is below", result[STIFFNESS]["details"]["basis"])

    def test_an_unknown_category_does_not_get_the_two_story_exception(self):
        _vertical, result = self.frame(2, [50.0, 100.0], sdc=None)
        self.assertEqual(result[STIFFNESS]["status"], "not_evaluated")

    def test_a_one_story_frame_is_covered_in_every_category(self):
        for sdc in ("B", "D", "F", None):
            vertical, result = self.frame(1, [100.0], sdc=sdc)
            self.assertEqual(vertical["stiffness"], [])
            self.assertEqual(result[STIFFNESS]["status"], "pass", sdc)
            self.assertIn("exception 2", result[STIFFNESS]["details"]["basis"])

    def test_missing_evidence_still_comes_first_for_a_two_story_frame(self):
        vertical, result = self.frame(2, [50.0, 100.0], rows_to_drop={(1, "x")})
        self.assertFalse(vertical["evidence"]["stiffness_complete"])
        self.assertIsNone(vertical["indicated"]["drift_ratio_exception_evaluable"])
        self.assertEqual(result[STIFFNESS]["status"], "not_evaluated")
        self.assertEqual(result[STIFFNESS]["details"]["category"], "incomplete story evidence")

    def test_three_stories_have_one_relationship_and_exception_1_can_hold_on_it(self):
        vertical, result = self.frame(3, [65.0, 100.0, 100.0], drift_ratio=[0.010, 0.010, 0.010])
        self.assertEqual([r["story"] for r in vertical["indicated"]["drift_ratio_exception_rows"]], [1, 1])   # x and y
        self.assertTrue(vertical["indicated"]["drift_ratio_exception_evaluable"])
        self.assertTrue(vertical["indicated"]["drift_ratio_exception_holds"])
        self.assertEqual(result[STIFFNESS]["status"], "pass")
        self.assertIn("exception holds", result[STIFFNESS]["details"]["basis"])
        vertical, result = self.frame(3, [65.0, 100.0, 100.0], drift_ratio=[0.020, 0.010, 0.010])
        self.assertIs(vertical["indicated"]["drift_ratio_exception_holds"], False)
        self.assertEqual(result[STIFFNESS]["status"], "not_evaluated")
        self.assertIn("does not hold", result[STIFFNESS]["details"]["reason"])

    def test_an_extreme_weak_story_in_sdc_b_or_c_names_the_height_limit_of_12_3_3_3(self):
        extreme = strength((100.0, 100.0, 60.0, 100.0, 100.0, 100.0))
        for sdc in ("B", "C"):
            check = evaluate(strengths=extreme, sdc=sdc, verified=True)[1][STRENGTH]
            self.assertEqual(check["status"], "not_evaluated")
            self.assertIn("12.3.3.3", check["clause"])
            self.assertIn("two stories or 30 ft", check["details"]["reason"])
        in_d = evaluate(strengths=extreme, sdc="D", verified=True)[1][STRENGTH]
        self.assertEqual(in_d["status"], "fail")                                # 12.3.3.2

    def test_the_redundancy_note_states_12_3_4_2_1_instead_of_calling_it_unavailable(self):
        from Design import SMRF_Demands as demands
        self.assertNotIn("not available", " ".join(demands.redundancy_requirement.__doc__.split()))
        self.assertNotIn("was not available", " ".join(Path(demands.__file__).read_text(encoding="utf-8").split()))
        self.assertEqual(demands.redundancy_requirement("D")["required"], 1.3)


if __name__ == "__main__":
    unittest.main()
