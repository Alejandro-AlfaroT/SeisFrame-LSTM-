"""Demand basis: SDC, ELF eligibility, live-load arrangements, torsion, declarations."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import Structure_Parameters as sp  # noqa: E402
from Design.SMRF_Demands import (amplification_from_level_displacements, classify_torsional_irregularity,  # noqa: E402
                                 elf_eligibility, evaluate_demand_basis, live_load_patterns,
                                 one_sided_strength_fraction, redundancy_requirement, seismic_design_category,
                                 story_drift_ratio, strength_distribution_fraction, strength_load_combinations,
                                 torsional_irregularity_limit, validate_torsion_assessment)
from Design.SMRF_Qualification import _apply_verification_assertions  # noqa: E402
from Design.SMRF_Common import not_evaluated  # noqa: E402
from Loads.Seismic_ELF import accidental_torsion_moment  # noqa: E402


class SeismicDesignCategoryTests(unittest.TestCase):
    def test_tables_11_6_1_and_11_6_2(self):
        self.assertEqual(seismic_design_category(0.10, 0.05, 0.05), "A")
        self.assertEqual(seismic_design_category(0.20, 0.05, 0.05), "B")
        self.assertEqual(seismic_design_category(0.20, 0.15, 0.10), "C")    # SD1 governs
        self.assertEqual(seismic_design_category(0.45, 0.18, 0.18), "C")
        self.assertEqual(seismic_design_category(0.50, 0.25, 0.25), "D")    # threshold is inclusive
        self.assertEqual(seismic_design_category(1.00, 0.60, 0.60), "D")
        self.assertEqual(seismic_design_category(1.25, 0.75, 0.75), "E")    # S1 >= 0.75, RC II
        self.assertEqual(seismic_design_category(1.25, 0.75, 0.75, "IV"), "F")
        self.assertEqual(seismic_design_category(0.20, 0.05, 0.05, "IV"), "C")
        with self.assertRaises(ValueError):
            seismic_design_category(1.0, 0.6, 0.6, "V")

    def test_elf_permission_is_edition_explicit(self):
        """ASCE 7-22 12.6 permits ELF for any structure; the 7-16 Table 12.6-1 rule survives only under its own label."""
        for sdc, height, regular, period in (("D", 126.0, False, 1.0), ("E", 200.0, True, 3.0), ("F", 400.0, False, 6.0)):
            permitted, reason = elf_eligibility(sdc, height, regular, period, 0.6)          # default edition: 7-22
            self.assertTrue(permitted)
            self.assertIn("ASCE 7-22 12.6", reason)
            self.assertIn("Table 12.6-1", reason)
        self.assertTrue(elf_eligibility("D", 126.0, True, 1.0, 0.6, edition="ASCE 7-16")[0])
        self.assertFalse(elf_eligibility("D", 126.0, False, 1.0, 0.6, edition="ASCE 7-16")[0])   # any irregularity
        self.assertTrue(elf_eligibility("D", 200.0, True, 1.5, 0.6, edition="ASCE 7-16")[0])     # T < 3.5 Ts
        self.assertFalse(elf_eligibility("E", 200.0, True, 3.0, 0.6, edition="ASCE 7-16")[0])
        self.assertTrue(elf_eligibility("C", 300.0, False, 5.0, 0.6, edition="ASCE 7-16")[0])
        self.assertIn("ASCE 7-16", elf_eligibility("D", 126.0, False, 1.0, 0.6, edition="ASCE 7-16")[1])
        with self.assertRaises(ValueError):
            elf_eligibility("D", 126.0, True, 1.0, 0.6, edition="ASCE 7-10")


class TorsionalIrregularityRuleTests(unittest.TestCase):
    def test_type_1_from_tir_or_strength_distribution(self):
        none = classify_torsional_irregularity(1.2, 0.5, strength_model_verified=True)
        self.assertEqual((none["type_1"], none["label"]), (False, "none"))
        self.assertTrue(none["strength_criterion_evaluated"] and none["strength_criterion_resolved"])
        self.assertEqual(none["strength_criterion_status"], "verified")
        self.assertNotIn("amplification_required", none)                      # Ax is not a function of the TIR
        by_tir = classify_torsional_irregularity(1.3, 0.5)
        self.assertTrue(by_tir["type_1"] and by_tir["by_tir"] and not by_tir["by_strength_distribution"])
        self.assertFalse(by_tir["tir_exceeds_1_4"])
        by_strength = classify_torsional_irregularity(1.05, 0.80)             # provisional model, still establishes Type 1
        self.assertTrue(by_strength["type_1"] and by_strength["by_strength_distribution"] and not by_strength["by_tir"])
        self.assertEqual(by_strength["strength_criterion_status"], "provisional")
        self.assertTrue(classify_torsional_irregularity(2.5, 0.5)["tir_exceeds_1_4"])
        unknown = classify_torsional_irregularity(1.1)                        # strength split not evaluated
        self.assertIsNone(unknown["by_strength_distribution"])
        self.assertFalse(unknown["strength_criterion_evaluated"])
        self.assertFalse(unknown["type_1"])                                   # provisional: the evaluator keeps it open
        self.assertEqual((unknown["label"], unknown["strength_criterion_status"]), ("unresolved", "unevaluated"))
        with self.assertRaises(ValueError):
            classify_torsional_irregularity(0.9)

    def test_provisional_strength_model_cannot_certify_absence(self):
        """Review item M1: complete arithmetic on an unverified model establishes Type 1 but never its absence."""
        provisional = classify_torsional_irregularity(1.15, 0.69)              # 0074/0144: TIR <= 1.2, fraction < 0.75
        self.assertFalse(provisional["type_1"])
        self.assertFalse(provisional["absence_established"])
        self.assertEqual(provisional["label"], "unresolved")
        self.assertEqual(provisional["strength_criterion_status"], "provisional")
        self.assertFalse(provisional["strength_criterion_resolved"])
        self.assertEqual(provisional["strength_model_review_item"], "M1")
        verified = classify_torsional_irregularity(1.15, 0.69, strength_model_verified=True)
        self.assertEqual((verified["label"], verified["absence_established"]), ("none", True))
        # A positive TIR establishes Type 1 whatever the strength branch says.
        self.assertEqual(classify_torsional_irregularity(1.25, 0.69)["label"], "type_1")
        self.assertEqual(classify_torsional_irregularity(1.25)["label"], "type_1")
        # The verified flag means nothing without a fraction.
        self.assertEqual(classify_torsional_irregularity(1.1, None, strength_model_verified=True)["strength_criterion_status"],
                         "unevaluated")

    def test_tir_ceiling_is_project_policy_in_every_sdc(self):
        """ASCE 7-22 removed the 7-16 12.3.3.1 E/F prohibition (FEMA P-2192 1.4.5; CA Title 24 1617.12.6)."""
        for sdc in ("B", "C", "D", "E", "F", None):
            limit, kind, basis = torsional_irregularity_limit(sdc)
            self.assertEqual((limit, kind), (1.4, "project_policy"), sdc)
            self.assertIn("removed", basis)
            self.assertIn("not a code provision", basis)

    def test_amplification_uses_level_displacements_not_story_drifts(self):
        """The review example: edge displacements (0.10, 0.10) at level 1 and (0.30, 0.20) at level 2."""
        self.assertAlmostEqual(story_drift_ratio(0.30 - 0.10, 0.20 - 0.10), 4.0 / 3.0)   # story 2 drifts 0.20 / 0.10
        level_2 = amplification_from_level_displacements(0.30, 0.20)
        self.assertAlmostEqual(level_2["ratio"], 1.2)
        self.assertEqual(level_2["ax"], 1.0)                                  # (1.2/1.2)^2 = 1, not (1.333/1.2)^2 = 1.235
        self.assertAlmostEqual(amplification_from_level_displacements(-0.30, -0.15)["ax"], (4.0 / 3.0 / 1.2) ** 2)
        self.assertAlmostEqual(amplification_from_level_displacements(0.9, 0.1)["ax"], (1.8 / 1.2) ** 2)     # 2.25
        # Two edges bound the ratio at 2.0 (one edge still), so Ax tops out at (2/1.2)^2 = 2.78 below the 3.0 cap.
        self.assertAlmostEqual(amplification_from_level_displacements(1.0, 0.0)["ax"], (2.0 / 1.2) ** 2)
        self.assertEqual(amplification_from_level_displacements(0.0, 0.0)["ax"], 1.0)

    def test_one_sided_strength_fraction_counts_lines_at_the_center(self):
        equal_three, detail = one_sided_strength_fraction([-1.0, 0.0, 1.0], [1.0, 1.0, 1.0], 0.0)
        self.assertAlmostEqual(equal_three, 2.0 / 3.0)                        # not 1/2: the center line is "at" the COM
        self.assertEqual(detail["lines_at_center"], 1)
        self.assertAlmostEqual(one_sided_strength_fraction([0.0, 1.0, 2.0, 3.0], [1.0] * 4, 1.5)[0], 0.5)   # even lines
        self.assertAlmostEqual(one_sided_strength_fraction([0.0, 1.0], [1.0, 1.0], 0.5)[0], 0.5)           # one bay
        self.assertAlmostEqual(one_sided_strength_fraction([0.0, 1.0, 2.0], [0.5, 1.0, 0.5], 1.0)[0], 0.75)  # weak edges: exactly 3/4
        weaker = one_sided_strength_fraction([0.0, 1.0, 2.0], [0.4, 1.0, 0.4], 1.0)[0]
        self.assertAlmostEqual(weaker, 1.4 / 1.8)                             # 0.778 > 0.75: Type 1 by strength
        self.assertTrue(classify_torsional_irregularity(1.05, weaker)["by_strength_distribution"])
        self.assertFalse(classify_torsional_irregularity(1.05, 0.75)["by_strength_distribution"])   # "more than" 75%
        self.assertAlmostEqual(one_sided_strength_fraction([-1.0, 0.0, 1.0], [2.0, 1.0, 1.0], 0.0)[0], 0.75)  # unequal lines
        with self.assertRaises(ValueError):
            one_sided_strength_fraction([0.0, 1.0], [0.0, 0.0], 0.5)
        with self.assertRaises(ValueError):
            one_sided_strength_fraction([0.0, 1.0], [1.0], 0.5)

    def test_saved_strength_evidence_needs_the_full_line_contract(self):
        """Review item F1: a named scalar, a single direction or a short roster is unknown, never a fraction."""
        import copy
        geometry = {"num_bay_x": 2, "num_bay_y": 3, "num_floor": 4, "story_h_in": 120.0, "bay_x_in": 120.0, "bay_y_in": 144.0}
        families = {"x_edge": {"mn_negative_kip_in": 500.0, "mn_positive_kip_in": 400.0},
                    "x_interior": {"mn_negative_kip_in": 1000.0, "mn_positive_kip_in": 800.0},
                    "y_edge": {"mn_negative_kip_in": 500.0, "mn_positive_kip_in": 400.0},
                    "y_interior": {"mn_negative_kip_in": 1000.0, "mn_positive_kip_in": 800.0}}
        block = _strength_block(families, geometry)
        value, reason = strength_distribution_fraction({"lateral_strength_distribution": block}, geometry, families)
        self.assertIsNone(reason)
        # y is resisted by the 3 x-grid lines (weak, strong, weak): (1800 + 900) / (1800 + 1800) = 0.75 exactly.
        self.assertAlmostEqual(block["by_direction"]["y"]["one_side_fraction"], 0.75)
        self.assertAlmostEqual(block["by_direction"]["x"]["one_side_fraction"], 0.5)      # 4 y-grid lines, no center line
        self.assertAlmostEqual(value, 0.75)

        def rejected(mutate, expect):
            broken = copy.deepcopy(block)
            mutate(broken)
            got, why = strength_distribution_fraction({"lateral_strength_distribution": broken}, geometry, families)
            self.assertIsNone(got, expect)
            self.assertIn(expect, why)

        rejected(lambda b: b["by_direction"].pop("y"), "exactly directions x and y")
        rejected(lambda b: b.pop("by_direction"), "exactly directions x and y")
        rejected(lambda b: b.pop("strength_inputs"), "not recorded")
        rejected(lambda b: b.pop("uniform_over_height"), "story coverage")
        rejected(lambda b: b["by_direction"]["y"]["line_positions_in"].pop(), "frame lines")
        rejected(lambda b: b["by_direction"]["y"]["line_story_strength_kip"].pop(), "frame lines")
        rejected(lambda b: b["by_direction"]["y"]["line_positions_in"].__setitem__(1, 100.0), "grid positions")
        rejected(lambda b: b["by_direction"]["y"]["line_story_strength_kip"].__setitem__(0, 99.0), "do not reproduce")
        rejected(lambda b: b["strength_inputs"]["families"]["x_edge"].__setitem__("mn_negative_kip_in", 501.0), "stale")
        rejected(lambda b: b["strength_inputs"]["bays"].__setitem__("x", 3), "bays")
        rejected(lambda b: b.__setitem__("one_side_fraction", 0.5), "largest direction value")
        rejected(lambda b: b["by_direction"]["y"].__setitem__("one_side_fraction", 0.5), "stored fraction")
        rejected(lambda b: b.pop("applicability"), "applicability status")
        rejected(lambda b: b["applicability"].__setitem__("status", "reviewed"), "applicability status")
        # No geometry to check the roster against, or a scalar with only a model name: unknown.
        self.assertIsNone(strength_distribution_fraction({"lateral_strength_distribution": block}, None, families)[0])
        self.assertIsNone(strength_distribution_fraction({"lateral_strength_distribution": {"one_side_fraction": 0.5, "model": "named"}},
                                                         geometry, families)[0])
        self.assertIsNone(strength_distribution_fraction({}, geometry, families)[0])
        # Without the rebuilt families the link cannot be checked but the roster still is.
        self.assertAlmostEqual(strength_distribution_fraction({"lateral_strength_distribution": block}, geometry, None)[0], 0.75)

    def test_redundancy_requirement_by_sdc(self):
        self.assertEqual(redundancy_requirement("C")["required"], 1.0)
        self.assertEqual(redundancy_requirement("D")["required"], 1.3)
        self.assertEqual(redundancy_requirement("E", conditions_12_3_4_2_verified=True)["required"], 1.0)
        self.assertIn("12.3.4.2", redundancy_requirement("F")["basis"])


class LiveLoadPatternTests(unittest.TestCase):
    def test_arrangements_cover_alternate_and_adjacent_spans(self):
        patterns = {p["id"]: p for p in live_load_patterns(3, 2)}
        self.assertEqual(set(patterns), {"alternate_x_even", "alternate_x_odd", "alternate_y_even", "alternate_y_odd",
                                         "adjacent_x_support_1", "adjacent_x_support_2", "adjacent_y_support_1"})
        self.assertEqual(sorted(patterns["alternate_x_even"]["panels"]), [[0, 0], [0, 1], [2, 0], [2, 1]])
        self.assertEqual(sorted(patterns["adjacent_x_support_2"]["panels"]), [[1, 0], [1, 1], [2, 0], [2, 1]])
        # A single bay has no odd spans and no interior supports.
        self.assertEqual([p["id"] for p in live_load_patterns(1, 1)], ["alternate_x_even", "alternate_y_even"])

    def test_pattern_combinations_extend_the_canonical_rule(self):
        base = strength_load_combinations(1.0)
        self.assertEqual(len(base), 18)
        self.assertTrue(all(c["live_pattern"] == "all" for c in base))
        patterned = strength_load_combinations(1.0, live_patterns=live_load_patterns(3, 3))
        extra = [c for c in patterned if c["family"] == "gravity_pattern"]
        self.assertEqual(len(extra), 8)
        self.assertTrue(all(c["dead"] == 1.2 and c["live"] == 1.6 and c["ex"] == c["ey"] == 0.0 for c in extra))
        self.assertEqual(len({c["id"] for c in patterned}), 26)


class TorsionTests(unittest.TestCase):
    def test_accidental_torsion_moment_uses_the_perpendicular_plan_dimension(self):
        with mock.patch.object(sp, "NUM_BAY_X", 3), mock.patch.object(sp, "NUM_BAY_Y", 2), \
             mock.patch.object(sp, "BAY_X", 240.0), mock.patch.object(sp, "BAY_Y", 200.0):
            self.assertAlmostEqual(accidental_torsion_moment("x", 10.0), 0.05 * 400.0 * 10.0)
            self.assertAlmostEqual(accidental_torsion_moment("y", 10.0), 0.05 * 720.0 * 10.0)
            self.assertAlmostEqual(accidental_torsion_moment("x", 10.0, 0.05, 2.0), 2.0 * 0.05 * 400.0 * 10.0)


# Beam families "rebuilt from the record's final cage" for the synthetic fixture (kip-in): equal on every
# line, so the 3x3 fixture's four frame lines per direction give a one-sided fraction of exactly 1/2.
FAMILIES = {name: {"mn_negative_kip_in": 1000.0, "mn_positive_kip_in": 800.0}
            for name in ("x_edge", "x_interior", "y_edge", "y_interior")}
GEOMETRY = {"num_bay_x": 3, "num_bay_y": 3, "num_floor": 8, "story_h_in": 120.0, "bay_x_in": 120.0, "bay_y_in": 120.0}


def evaluate(record):
    """The evaluator with the fixture's rebuilt families (a real record supplies its own)."""
    return evaluate_demand_basis(record, strength_families=FAMILIES)


def _strength_block(families=FAMILIES, geometry=GEOMETRY, status="verified"):
    from Design.SMRF_Demands import line_story_strengths
    by_direction = {}
    for direction in ("x", "y"):
        positions, strengths, names = line_story_strengths(direction, geometry, families, geometry["story_h_in"])
        fraction, detail = one_sided_strength_fraction(positions, strengths, 0.5 * positions[-1])
        by_direction[direction] = {"line_positions_in": positions, "line_story_strength_kip": strengths,
                                   "line_families": names, "one_side_fraction": fraction, **detail}
    return {"one_side_fraction": max(v["one_side_fraction"] for v in by_direction.values()),
            "by_direction": by_direction, "model": "fixture: beam-sway line strengths",
            "applicability": {"status": status, "review_item": "M1", "basis": "fixture"},
            "uniform_over_height": {"claim": True, "basis": "fixture: one cage over the height"},
            "strength_inputs": {"story_h_in": geometry["story_h_in"],
                                "bays": {"x": geometry["num_bay_x"], "y": geometry["num_bay_y"]},
                                "families": {k: dict(v) for k, v in families.items()}}}


def _verification(story_strength_model_verified=True):
    """The record's IndependentVerification assertion, with provenance (SMRF_Common.assertion_provenance_valid)."""
    return {"asserted_by": "test", "assertion_date": "2026-09-13", "assertion_basis": "fixture",
            "story_strength_model_verified": story_strength_model_verified}


def _record_from_levels(u_a, u_b, amplification=None):
    """A fixture whose torsion rows are built from edge level displacements (fixed base), every case alike."""
    record = _record()
    rows, per_case, ax_by_level = [], {}, {}
    for axis in ("x", "y"):
        for sign in (1.0, -1.0):
            case = f"{axis}{'+' if sign > 0 else '-'}"
            prev_a = prev_b = 0.0
            for k, (la, lb) in enumerate(zip(u_a, u_b), 1):
                drift_a, drift_b = abs(la - prev_a), abs(lb - prev_b)
                ratio = story_drift_ratio(drift_a, drift_b)
                level = amplification_from_level_displacements(sign * la, sign * lb)
                rows.append({"story": k, "direction": axis, "case": case, "eccentricity_sign": sign,
                             "delta_end_a_in": drift_a, "delta_end_b_in": drift_b, "delta_max_over_avg": ratio,
                             "delta_level_a_in": sign * la, "delta_level_b_in": sign * lb,
                             "level_max_over_avg": level["ratio"], "ax_level": level["ax"]})
                per_case[case] = max(per_case.get(case, 0.0), ratio)
                ax_by_level[k] = max(ax_by_level.get(k, 0.0), level["ax"])
                prev_a, prev_b = la, lb
    tir = max(per_case.values())
    envelope = max(ax_by_level.values())
    required = envelope if tir > 1.2 else 1.0
    torsion = record["demand_basis"]["torsion"]
    torsion.update(stories=rows, tir_by_case=per_case, tir=tir, max_drift_ratio=tir, amplification_by_level=ax_by_level,
                   amplification_required=required, amplification=required if amplification is None else amplification,
                   torsional_irregularity="type_1" if tir > 1.2 else "none")
    torsion.pop("classification", None)
    return record


def _record(declared=True, torsion_ratio=1.067, patterns=True, amplification=None, rho=1.3, seismic=None,
            strength_model_verified=True):
    """A complete fixture. By default the story-strength model is asserted verified (both the evidence's
    applicability status and the record's assertion), so a TIR at or below 1.2 resolves to 'none'."""
    policy = {"risk_category": "II", "site_class": "C", "occupancy": "office", "partition_allowance_ksf": 0.010,
              "storage_live_fraction_in_weight": 0.0, "roof_live_load_ksf": 0.020, "snow_load_ksf": 0.0,
              "wind_governs": False, "rain_ponding_excluded": True, "accidental_torsion_ratio": 0.05,
              "live_load_patterning": True, "declared_by": "test" if declared else "",
              "declaration_date": "2026-09-13", "declaration_basis": "fixture" if declared else ""}
    # Edge drifts a, b with max/avg = torsion_ratio; level displacements grow with the story at the same ratio,
    # so the level-displacement Ax equals what the story-drift ratio would give (the two agree on this fixture).
    a = torsion_ratio / (2.0 - torsion_ratio)
    ax = min(3.0, max(1.0, (torsion_ratio / 1.2) ** 2))
    required = ax if torsion_ratio > 1.2 else 1.0
    if amplification is None:
        amplification = required
    stories = [{"story": k, "direction": axis, "case": f"{axis}{'+' if sign > 0 else '-'}", "eccentricity_sign": sign,
                "delta_end_a_in": a, "delta_end_b_in": 1.0, "delta_max_over_avg": torsion_ratio,
                "delta_level_a_in": sign * k * a, "delta_level_b_in": sign * k * 1.0,
                "level_max_over_avg": torsion_ratio, "ax_level": ax}
               for axis in ("x", "y") for sign in (1.0, -1.0) for k in range(1, 9)]
    return {
        "seismic": seismic or {"sds": 1.0, "sd1": 0.6, "s1": 0.6, "site_label": "sdc_d_high"},
        "geometry": dict(GEOMETRY),
        "floor_loads": {"floor_live_load_ksf": 0.05, "floor_superimposed_dead_load_ksf": 0.05,
                        "seismic_live_load_fraction": 0.0, "total_floor_seismic_weight_kip": 200.0},
        "demand_basis": {"policy": policy, "code_edition": "ASCE 7-22", "redundancy_factor": rho,
                         "torsion": {"ratio": 0.05, "assessment_amplification": 1.0, "base": "fixed",
                                     "amplification": amplification,
                                     "amplification_required": required, "amplification_by_level": {k: ax for k in range(1, 9)},
                                     "max_drift_ratio": torsion_ratio, "tir": torsion_ratio,
                                     "tir_by_case": {c: torsion_ratio for c in ("x+", "x-", "y+", "y-")},
                                     "cases": ["x+", "x-", "y+", "y-"], "stories": stories,
                                     "torsional_irregularity": ("type_1" if torsion_ratio > 1.2 else
                                                                "none" if strength_model_verified else "unresolved")},
                         "verification": _verification(strength_model_verified),
                         "regularity": {"regular": True, "lateral_strength_distribution": _strength_block(
                             status="verified" if strength_model_verified else "provisional")},
                         "design_period_sec": 1.0, "ts_sec": 0.6,
                         "period_basis": {"capped_at_cu_ta": False},
                         "live_load_patterns": live_load_patterns(3, 3) if patterns else [],
                         "patterns_in_strength_envelope": patterns,
                         "drift": {"cd": 5.5, "rho_for_drift_load": 1.0, "beam_stiffness_modifier": 0.35,
                                   "column_stiffness_modifier": 0.70, "second_order_included": True}},
    }


class DemandBasisEvaluationTests(unittest.TestCase):
    def test_declared_record_evaluates_every_item(self):
        statuses = {c["id"]: c for c in evaluate(_record())}
        for key in ("site_hazard", "elf_eligibility", "accidental_torsion", "torsional_irregularity", "redundancy",
                    "load_scope", "live_load_patterning", "effective_seismic_weight", "drift_analysis_basis"):
            self.assertEqual(statuses[f"demands.{key}"]["status"], "pass", key)
        self.assertEqual(statuses["demands.site_hazard"]["details"]["derived_sdc"], "D")
        elf = statuses["demands.elf_eligibility"]
        self.assertEqual(elf["clause"], "ASCE 7-22 12.6 (analysis procedure selection)")
        self.assertEqual(elf["details"]["code_edition"], "ASCE 7-22")
        self.assertTrue(elf["details"]["asce_7_16_table_12_6_1"]["permitted"])
        self.assertEqual(statuses["demands.torsional_irregularity"]["details"]["rows_validated"], 32)
        self.assertEqual(statuses["demands.accidental_torsion"]["details"]["limit_kind"], "project_policy")   # SDC D
        self.assertEqual(statuses["demands.redundancy"]["capacity"], 1.3)                                    # 12.3.4.2

    def test_undeclared_policy_keeps_declarations_open_but_computed_items_evaluated(self):
        statuses = {c["id"]: c["status"] for c in evaluate(_record(declared=False))}
        for key in ("site_hazard", "elf_eligibility", "load_scope", "effective_seismic_weight"):
            self.assertEqual(statuses[f"demands.{key}"], "not_evaluated", key)
        for key in ("accidental_torsion", "torsional_irregularity", "redundancy", "live_load_patterning",
                    "drift_analysis_basis"):
            self.assertEqual(statuses[f"demands.{key}"], "pass", key)

    def test_invalid_declarations_are_rejected_explicitly(self):
        """The cross-check's reproductions: blank/whitespace provenance and an invalid site class."""
        from Design.SMRF_Demands import demand_policy_problems
        record = _record(declared=True)
        base = {c["id"]: c for c in evaluate(record)}
        self.assertTrue(all(c["status"] != "not_evaluated" for cid, c in base.items() if cid.startswith("demands.site_hazard")))
        for mutate, expect in (
            (lambda p: p.update(declared_by="", declaration_date="", declaration_basis="   "), "declaration_basis is blank"),
            (lambda p: p.update(site_class="QZ"), "site_class"),
            (lambda p: p.update(site_class=" D "), "site_class"),        # noncanonical: the hazard check reads it verbatim
            (lambda p: p.update(site_class="d"), "site_class"),
            (lambda p: p.update(risk_category=" II"), "risk_category"),
            (lambda p: p.update(declaration_date="2026-13-40"), "ISO calendar date"),
            (lambda p: p.update(risk_category="V"), "risk_category"),
            (lambda p: p.update(accidental_torsion_ratio=0.0), "accidental_torsion_ratio"),
            (lambda p: p.update(roof_live_load_ksf=-1.0), "roof_live_load_ksf"),
            (lambda p: p.update(wind_governs="no"), "wind_governs"),
        ):
            bad = _record(declared=True)
            mutate(bad["demand_basis"]["policy"])
            problems = demand_policy_problems(bad["demand_basis"]["policy"])
            self.assertTrue(any(expect in x for x in problems), (expect, problems))
            result = {c["id"]: c for c in evaluate(bad)}
            hazard = result["demands.site_hazard"]
            self.assertEqual(hazard["status"], "not_evaluated", expect)
            self.assertIn("Rejected", hazard["details"]["reason"])
            self.assertTrue(all(c["status"] == "not_evaluated" for cid, c in result.items()
                                if cid in ("demands.site_hazard", "demands.load_scope", "demands.effective_seismic_weight")))
        # Design time refuses a partly filled declaration outright.
        from Design.Config import DemandPolicy
        self.assertEqual(DemandPolicy().problems(), ["declared_by is blank", "declaration_date is blank", "declaration_basis is blank"])
        self.assertTrue(DemandPolicy(declared_by="a", declaration_date="2026-09-14", declaration_basis="b").declared())
        self.assertFalse(DemandPolicy(declared_by="a", declaration_date="2026-09-14", declaration_basis="b", site_class="Z").declared())

    def test_site_specific_analysis_flag_cannot_be_dodged_by_whitespace(self):
        """Fourth cross-check: a padded spelling once dodged the flag. Under ASCE 7-22 the flag is Site Class F
        only (11.4.7); Site Class D with S1 >= 0.2 passes (the 7-16 rule is not in this edition)."""
        record = _record()
        record["demand_basis"]["policy"]["site_class"] = "F"
        canonical = {c["id"]: c for c in evaluate(record)}["demands.site_hazard"]
        self.assertEqual(canonical["status"], "fail")
        self.assertTrue(canonical["details"]["site_specific_ground_motion_required"])
        self.assertIn("11.4.7", canonical["clause"])
        self.assertNotIn("11.4.8", canonical["clause"])
        record["demand_basis"]["policy"]["site_class"] = "D"
        self.assertGreaterEqual(record["seismic"]["s1"], 0.2)
        in_d = {c["id"]: c for c in evaluate(record)}["demands.site_hazard"]
        self.assertEqual(in_d["status"], "pass")
        self.assertFalse(in_d["details"]["site_specific_ground_motion_required"])
        for spelling in (" F ", "f", "F\n"):
            record["demand_basis"]["policy"]["site_class"] = spelling
            padded = {c["id"]: c for c in evaluate(record)}["demands.site_hazard"]
            self.assertEqual(padded["status"], "not_evaluated", spelling)
            self.assertIn("not exactly one of", padded["details"]["reason"])

    def test_design_refuses_a_partly_filled_or_invalid_declaration(self):
        from Design import Design_Driver as driver
        from Design.Config import DesignConfig, DemandPolicy
        with self.assertRaisesRegex(ValueError, "DemandPolicy is not a valid declaration"):
            driver.design_structure(cfg=DesignConfig(demands=DemandPolicy(declared_by="someone")), verbose=False)
        with self.assertRaisesRegex(ValueError, "site_class"):
            driver.design_structure(cfg=DesignConfig(demands=DemandPolicy(site_class="Q")), verbose=False)

    def test_mislabelled_site_fails_the_hazard_item(self):
        record = _record()
        record["seismic"] = {"sds": 0.50, "sd1": 0.25, "s1": 0.25, "site_label": "sdc_c"}
        statuses = {c["id"]: c for c in evaluate(record)}
        self.assertEqual(statuses["demands.site_hazard"]["status"], "fail")
        self.assertEqual(statuses["demands.site_hazard"]["details"]["derived_sdc"], "D")

    def test_type_1_irregularity_no_longer_bars_elf_but_must_be_amplified(self):
        """The cases 0074/0144 situation: TIR just over 1.2 in SDC D under ASCE 7-22."""
        statuses = {c["id"]: c for c in evaluate(_record(torsion_ratio=1.3))}
        elf = statuses["demands.elf_eligibility"]
        self.assertEqual(elf["status"], "pass")                                   # 7-22 12.6: ELF for any structure
        self.assertFalse(elf["details"]["regular"])
        self.assertFalse(elf["details"]["asce_7_16_table_12_6_1"]["permitted"])   # the superseded verdict, recorded
        self.assertEqual(statuses["demands.accidental_torsion"]["status"], "pass")   # 1.3 <= 1.4 policy ceiling
        torsion = statuses["demands.torsional_irregularity"]
        self.assertEqual(torsion["status"], "pass")
        self.assertTrue(torsion["details"]["type_1"] and torsion["details"]["by_tir"])
        self.assertAlmostEqual(torsion["capacity"], (1.3 / 1.2) ** 2)             # Ax required, 12.8.4.3
        # Ax not applied to the strength combinations: the irregularity item fails, the procedure item does not.
        under = {c["id"]: c for c in evaluate(_record(torsion_ratio=1.3, amplification=1.0))}
        self.assertEqual(under["demands.torsional_irregularity"]["status"], "fail")
        self.assertEqual(under["demands.elf_eligibility"]["status"], "pass")

    def test_tir_above_1_4_is_a_policy_failure_in_every_sdc(self):
        in_d = {c["id"]: c for c in evaluate(_record(torsion_ratio=1.45))}
        self.assertEqual(in_d["demands.accidental_torsion"]["status"], "fail")
        self.assertEqual(in_d["demands.accidental_torsion"]["details"]["limit_kind"], "project_policy")
        self.assertIn("project policy", in_d["demands.accidental_torsion"]["clause"])
        self.assertEqual(in_d["demands.elf_eligibility"]["status"], "pass")
        sdc_e = {"sds": 1.25, "sd1": 0.75, "s1": 0.75, "site_label": "sdc_e"}
        in_e = {c["id"]: c for c in evaluate(_record(torsion_ratio=1.45, seismic=sdc_e))}
        self.assertEqual(in_e["demands.accidental_torsion"]["status"], "fail")
        self.assertEqual(in_e["demands.accidental_torsion"]["details"]["limit_kind"], "project_policy")   # not code in E either
        self.assertNotIn("12.3.3.1 (SDC E/F prohibition)", in_e["demands.accidental_torsion"]["clause"])
        self.assertIn("removed", in_e["demands.accidental_torsion"]["details"]["limit_basis"])
        self.assertAlmostEqual(in_e["demands.accidental_torsion"]["demand"], 1.45)    # TIR recomputed from the rows
        self.assertTrue(in_e["demands.redundancy"]["details"]["tir_exceeds_1_4"])

    def test_torsion_evidence_fails_closed(self):
        """The review counterexamples plus duplicates, missing cases, tampered rows, legacy rows and stale scalars."""
        def torsion_statuses(record):
            checks = {c["id"]: c for c in evaluate(record)}
            return checks["demands.torsional_irregularity"], checks["demands.accidental_torsion"]

        # 1. Every case and story row removed, scalar TIR/Ax retained.
        record = _record(torsion_ratio=1.3)
        record["demand_basis"]["torsion"]["stories"] = []
        record["demand_basis"]["torsion"]["cases"] = []
        irregularity, ceiling = torsion_statuses(record)
        self.assertEqual((irregularity["status"], ceiling["status"]), ("not_evaluated", "not_evaluated"))
        self.assertIn("per-story rows", irregularity["details"]["reason"])
        # 2. Scalar TIR/Ax changed, rows and cached classification untouched: the scalars do not reproduce.
        record = _record(torsion_ratio=1.3)
        record["demand_basis"]["torsion"]["classification"] = classify_torsional_irregularity(1.3, 0.5)
        record["demand_basis"]["torsion"]["tir"] = record["demand_basis"]["torsion"]["max_drift_ratio"] = 1.35
        record["demand_basis"]["torsion"]["amplification"] = 1.01
        irregularity, ceiling = torsion_statuses(record)
        self.assertEqual((irregularity["status"], ceiling["status"]), ("not_evaluated", "not_evaluated"))
        self.assertIn("stored tir", irregularity["details"]["reason"])
        # 2b. Consistent rows at 1.35 but a cached classification from another frame: the cache is not trusted.
        record = _record(torsion_ratio=1.35)
        record["demand_basis"]["torsion"]["classification"] = classify_torsional_irregularity(1.1, 0.5)
        irregularity, ceiling = torsion_statuses(record)
        self.assertEqual(irregularity["status"], "not_evaluated")
        self.assertIn("stored classification", irregularity["details"]["reason"])
        # 3. Strength-distribution evidence removed with a low TIR: the classification is open, the ceiling still evaluates.
        record = _record(torsion_ratio=1.1)
        del record["demand_basis"]["regularity"]["lateral_strength_distribution"]
        record["demand_basis"]["torsion"].pop("classification", None)
        irregularity, ceiling = torsion_statuses(record)
        self.assertEqual(irregularity["status"], "not_evaluated")
        self.assertIn("strength-distribution", irregularity["details"]["reason"])
        self.assertEqual(ceiling["status"], "pass")
        # 4. A duplicated row.
        record = _record()
        record["demand_basis"]["torsion"]["stories"].append(dict(record["demand_basis"]["torsion"]["stories"][0]))
        self.assertEqual(torsion_statuses(record)[0]["status"], "not_evaluated")
        # 5. A missing accidental-torsion case (all y- rows dropped, cases list trimmed to match).
        record = _record()
        record["demand_basis"]["torsion"]["stories"] = [r for r in record["demand_basis"]["torsion"]["stories"] if r["case"] != "y-"]
        record["demand_basis"]["torsion"]["cases"] = ["x+", "x-", "y+"]
        del record["demand_basis"]["torsion"]["tir_by_case"]["y-"]
        irregularity, _ceiling = torsion_statuses(record)
        self.assertEqual(irregularity["status"], "not_evaluated")
        self.assertIn("exactly once", irregularity["details"]["reason"])
        # 6. A tampered row ratio, a tampered level Ax, and a tampered per-level envelope.
        for field, value in (("delta_max_over_avg", 1.0), ("ax_level", 1.0)):
            record = _record(torsion_ratio=1.3)
            record["demand_basis"]["torsion"]["stories"][5][field] = value
            self.assertEqual(torsion_statuses(record)[0]["status"], "not_evaluated", field)
        record = _record(torsion_ratio=1.3)
        record["demand_basis"]["torsion"]["amplification_by_level"][3] = 1.0
        self.assertEqual(torsion_statuses(record)[0]["status"], "not_evaluated")
        # 7. Legacy single-sign rows (no case field): valid evidence of what it was, not of the 7-22 case set.
        record = _record()
        record["demand_basis"]["torsion"] = {"ratio": 0.05, "amplification": 1.0, "max_drift_ratio": 1.067,
                                             "torsional_irregularity": "none",
                                             "stories": [{"story": k, "direction": d, "delta_end_a_in": 1.0, "delta_end_b_in": 1.0,
                                                          "delta_max_over_avg": 1.0} for d in ("x", "y") for k in range(1, 9)]}
        irregularity, ceiling = torsion_statuses(record)
        self.assertEqual((irregularity["status"], ceiling["status"]), ("not_evaluated", "not_evaluated"))
        self.assertIn("legacy", irregularity["details"]["reason"])
        self.assertTrue(validate_torsion_assessment(record["demand_basis"]["torsion"], 8)["legacy"])
        # 8. The assessment did not run with Ax = 1, or did not say so.
        for value in (1.5, None):
            record = _record()
            record["demand_basis"]["torsion"]["assessment_amplification"] = value
            self.assertEqual(torsion_statuses(record)[0]["status"], "not_evaluated")
        # 9. A stored required Ax that does not reproduce from the level displacements.
        record = _record(torsion_ratio=1.3)
        record["demand_basis"]["torsion"]["amplification_required"] = 1.0
        irregularity, _ceiling = torsion_statuses(record)
        self.assertEqual(irregularity["status"], "not_evaluated")
        self.assertIn("amplification_required", irregularity["details"]["reason"])
        # 10. The eccentricity differs from the declared policy value.
        record = _record()
        record["demand_basis"]["torsion"]["ratio"] = 0.06
        self.assertEqual(torsion_statuses(record)[0]["status"], "not_evaluated")
        # And the untouched fixture still evaluates, with the TIR read from the rows.
        irregularity, ceiling = torsion_statuses(_record(torsion_ratio=1.3))
        self.assertEqual((irregularity["status"], ceiling["status"]), ("pass", "pass"))
        self.assertAlmostEqual(ceiling["demand"], 1.3)
        self.assertEqual(len(irregularity["details"]["ax_by_level"]), 8)

    def test_strength_evidence_gaps_leave_the_classification_open(self):
        """Review item F1 through the evaluator: X-only lines, a named scalar, a short roster, stale inputs."""
        def status(mutate):
            record = _record()
            block = record["demand_basis"]["regularity"]["lateral_strength_distribution"]
            mutate(block)
            record["demand_basis"]["torsion"].pop("classification", None)
            check = {c["id"]: c for c in evaluate(record)}["demands.torsional_irregularity"]
            return check["status"], check["details"].get("reason", "")

        for mutate, expect in (
                (lambda b: b["by_direction"].pop("y"), "exactly directions x and y"),
                (lambda b: b.pop("by_direction"), "exactly directions x and y"),
                (lambda b: b["by_direction"]["x"]["line_positions_in"].pop(), "frame lines"),
                (lambda b: b["strength_inputs"]["families"]["y_interior"].__setitem__("mn_positive_kip_in", 1.0), "stale"),
                (lambda b: b.pop("uniform_over_height"), "story coverage")):
            self.assertEqual(status(mutate), ("not_evaluated", status(mutate)[1]))
            self.assertIn(expect, status(mutate)[1])
        # A scalar whose aggregate was "updated" to hide a removed direction: still open.
        def hide_y(b):
            b["by_direction"].pop("y")
            b["one_side_fraction"] = b["by_direction"]["x"]["one_side_fraction"]
        self.assertEqual(status(hide_y)[0], "not_evaluated")
        # Control: the complete fixture passes.
        self.assertEqual({c["id"]: c["status"] for c in evaluate(_record())}["demands.torsional_irregularity"], "pass")

    def test_provisional_strength_model_keeps_a_low_tir_classification_open(self):
        """Review item M1 through the evaluator: unverified line arithmetic cannot certify 'no Type 1'."""
        checks = {c["id"]: c for c in evaluate(_record(torsion_ratio=1.1, strength_model_verified=False))}
        item = checks["demands.torsional_irregularity"]
        self.assertEqual(item["status"], "not_evaluated")
        self.assertIn("provisional", item["details"]["reason"])
        self.assertIn("M1", item["details"]["reason"])
        self.assertEqual(item["details"]["strength_criterion_status"], "provisional")
        self.assertEqual(item["details"]["classification"]["label"], "unresolved")
        self.assertEqual(checks["demands.accidental_torsion"]["status"], "pass")          # the TIR ceiling still evaluates
        # A TIR above 1.2 establishes Type 1 on its own; the strength branch is reported as provisional.
        by_tir = {c["id"]: c for c in evaluate(_record(torsion_ratio=1.3, strength_model_verified=False))}["demands.torsional_irregularity"]
        self.assertEqual(by_tir["status"], "pass")
        self.assertTrue(by_tir["details"]["by_tir"])
        self.assertIn("provisional", by_tir["details"]["strength_branch_note"])
        # The verified flag needs provenance, and the evidence's status must match the record's assertion.
        record = _record(torsion_ratio=1.1, strength_model_verified=True)
        record["demand_basis"]["verification"]["assertion_basis"] = ""
        item = {c["id"]: c for c in evaluate(record)}["demands.torsional_irregularity"]
        self.assertEqual(item["status"], "not_evaluated")
        self.assertIn("no story_strength_model_verified assertion", item["details"]["reason"])
        record = _record(torsion_ratio=1.1, strength_model_verified=True)
        record["demand_basis"]["regularity"]["lateral_strength_distribution"]["applicability"]["status"] = "provisional"
        record["demand_basis"]["torsion"].pop("classification", None)
        item = {c["id"]: c for c in evaluate(record)}["demands.torsional_irregularity"]
        self.assertEqual(item["status"], "not_evaluated")
        self.assertIn("stale", item["details"]["reason"])
        # A verified model with more than 75% on one side is Type 1 by strength and needs its Ax.
        record = _record(torsion_ratio=1.1, strength_model_verified=True)
        weak = {name: {"mn_negative_kip_in": 400.0 if "edge" in name else 1000.0,
                       "mn_positive_kip_in": 320.0 if "edge" in name else 800.0}
                for name in FAMILIES}
        geometry = {**GEOMETRY, "num_bay_x": 2, "num_bay_y": 2}
        record["geometry"] = geometry
        record["demand_basis"]["regularity"]["lateral_strength_distribution"] = _strength_block(weak, geometry)
        record["demand_basis"]["torsion"].pop("classification", None)
        item = {c["id"]: c for c in evaluate_demand_basis(record, strength_families=weak)}["demands.torsional_irregularity"]
        self.assertEqual(item["status"], "pass")                  # Type 1 by strength; level ratios < 1.2 need Ax = 1.0
        self.assertTrue(item["details"]["type_1"] and item["details"]["by_strength_distribution"])
        self.assertEqual(item["details"]["label"], "type_1")
        self.assertAlmostEqual(item["details"]["one_side_strength_fraction"], (720.0 + 1800.0) / (720.0 * 2 + 1800.0))
        # The same weak-perimeter frame on a provisional model is still Type 1 (conservative), not 'unresolved'.
        record["demand_basis"]["verification"]["story_strength_model_verified"] = False
        record["demand_basis"]["regularity"]["lateral_strength_distribution"]["applicability"]["status"] = "provisional"
        item = {c["id"]: c for c in evaluate_demand_basis(record, strength_families=weak)}["demands.torsional_irregularity"]
        self.assertEqual((item["status"], item["details"]["label"]), ("pass", "type_1"))

    def test_drift_primitives_must_match_displacement_primitives(self):
        """Review item F2: story drifts are the differences of the same edge's level displacements (fixed base)."""
        # Consistent non-proportional response: edge a levels 0.1, 0.3, 0.5, ...; edge b 0.1, 0.2, 0.3, ...
        # Story drifts (0.1, 0.1) then (0.2, 0.1): TIR = 1.333 (by drift); level ratios 1.0, 1.2, 1.25, ... 1.304.
        record = _record_from_levels([0.1 + 0.2 * k for k in range(8)], [0.1 + 0.1 * k for k in range(8)])
        validated = validate_torsion_assessment(record["demand_basis"]["torsion"], 8, 0.05)
        self.assertTrue(validated["valid"], validated.get("reason"))
        self.assertAlmostEqual(validated["tir"], 4.0 / 3.0)
        top = amplification_from_level_displacements(1.5, 0.8)                  # level 8: ratio 1.304, Ax 1.181
        self.assertAlmostEqual(validated["ax_required_envelope"], top["ax"])
        self.assertLess(validated["ax_required_envelope"], (4.0 / 3.0 / 1.2) ** 2)     # not the TIR-based value
        checks = {c["id"]: c for c in evaluate(record)}
        self.assertEqual(checks["demands.torsional_irregularity"]["status"], "pass")
        self.assertAlmostEqual(checks["demands.torsional_irregularity"]["capacity"], top["ax"])
        self.assertAlmostEqual(checks["demands.accidental_torsion"]["demand"], 4.0 / 3.0)
        # The inconsistent record: drift magnitudes 2.0769 / 1.0 with displacements that imply 1.0 / 1.0.
        broken = _record_from_levels([1.0 * (k + 1) for k in range(8)], [1.0 * (k + 1) for k in range(8)])
        for row in broken["demand_basis"]["torsion"]["stories"]:
            row["delta_end_a_in"], row["delta_max_over_avg"] = 1.35 / (2.0 - 1.35), 1.35
        torsion = broken["demand_basis"]["torsion"]
        torsion["tir"] = torsion["max_drift_ratio"] = 1.35
        torsion["tir_by_case"] = {c: 1.35 for c in torsion["cases"]}
        validated = validate_torsion_assessment(torsion, 8, 0.05)
        self.assertFalse(validated["valid"])
        self.assertIn("recorded story drift", validated["reason"])
        checks = {c["id"]: c for c in evaluate(broken)}
        self.assertEqual(checks["demands.torsional_irregularity"]["status"], "not_evaluated")
        self.assertEqual(checks["demands.accidental_torsion"]["status"], "not_evaluated")
        # A single displaced level (edge b, story 3 shifted by 1e-3 in) is caught; a 1e-9 in wobble is tolerated.
        for shift, expected in ((1e-3, "not_evaluated"), (1e-10, "pass")):
            record = _record_from_levels([0.1 + 0.2 * k for k in range(8)], [0.1 + 0.1 * k for k in range(8)])
            for row in record["demand_basis"]["torsion"]["stories"]:
                if row["story"] == 3:
                    row["delta_level_b_in"] += shift * row["eccentricity_sign"]
            self.assertEqual({c["id"]: c["status"] for c in evaluate(record)}["demands.torsional_irregularity"], expected, shift)
        # An assessment that is not fixed-base cannot be checked this way.
        record = _record()
        record["demand_basis"]["torsion"]["base"] = "flexible"
        self.assertEqual({c["id"]: c["status"] for c in evaluate(record)}["demands.torsional_irregularity"], "not_evaluated")

    def test_redundancy_factor_against_the_sdc_requirement(self):
        low = {c["id"]: c for c in evaluate(_record(rho=1.0))}
        self.assertEqual(low["demands.redundancy"]["status"], "fail")            # SDC D needs 1.3 without (a)/(b)
        sdc_c = {"sds": 0.40, "sd1": 0.19, "s1": 0.19, "site_label": "sdc_c"}
        in_c = {c["id"]: c for c in evaluate(_record(rho=1.0, seismic=sdc_c))}
        self.assertEqual(in_c["demands.redundancy"]["status"], "pass")
        self.assertEqual(in_c["demands.redundancy"]["capacity"], 1.0)
        # A record that did not declare rho: read it from the saved seismic action factors.
        legacy = _record()
        del legacy["demand_basis"]["redundancy_factor"]
        legacy["design_actions"] = {"combinations": [{"family": "seismic_high_gravity_X", "ex": 1.3, "ey": 0.39},
                                                     {"family": "gravity", "ex": 0.0, "ey": 0.0}]}
        derived = {c["id"]: c for c in evaluate(legacy)}
        self.assertEqual(derived["demands.redundancy"]["status"], "pass")
        self.assertEqual(derived["demands.redundancy"]["demand"], 1.3)
        del legacy["design_actions"]
        self.assertEqual({c["id"]: c for c in evaluate(legacy)}["demands.redundancy"]["status"], "not_evaluated")

    def test_empty_record_is_all_not_evaluated(self):
        self.assertTrue(all(c["status"] == "not_evaluated" for c in evaluate_demand_basis({})))


class VerificationAssertionTests(unittest.TestCase):
    def test_assertions_need_an_author_and_a_basis(self):
        checks = [not_evaluated("floor.independent_hand_verification", "review", "open"),
                  not_evaluated("slab_fire_resistance", "4.11", "open")]
        unsigned = {"demand_basis": {"verification": {"floor_hand_check_verified": True, "asserted_by": "", "assertion_basis": ""}}}
        self.assertTrue(all(c["status"] == "not_evaluated" for c in _apply_verification_assertions(unsigned, checks)))
        signed = {"demand_basis": {"verification": {"floor_hand_check_verified": True, "fire_resistance_scope_accepted": False,
                                                    "asserted_by": "engineer", "assertion_date": "2026-09-13",
                                                    "assertion_basis": "hand check of case 7"}}}
        result = {c["id"]: c for c in _apply_verification_assertions(signed, checks)}
        self.assertEqual(result["floor.independent_hand_verification"]["status"], "pass")
        self.assertEqual(result["floor.independent_hand_verification"]["details"]["asserted_by"], "engineer")
        self.assertEqual(result["slab_fire_resistance"]["status"], "not_evaluated")


if __name__ == "__main__":
    unittest.main()
