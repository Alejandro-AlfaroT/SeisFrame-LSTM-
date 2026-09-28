"""Codex review follow-ups (2026-09-18) verified through final qualification, not helper returns.

R2: torsion evidence fails closed in ``qualify_design``. R3: TIR (story
drifts) and Ax (level displacements) are separate physical inputs in the
saved assessment. R4: the one-sided strength fraction comes from the frame
topology and beam families. R5: missing joint continuity evidence stays
open through ``qualify_design`` and the downstream joint-shear check.
One small frame is designed once for the class; every counterexample is an
in-memory mutation of its JSON round-tripped record.
"""
import contextlib
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp                                            # noqa: E402
from Design import Design_Driver as driver                                   # noqa: E402
from Design.SMRF_Capacity_Design import build_capacity_design                # noqa: E402
from Design.SMRF_Demands import validate_torsion_assessment                  # noqa: E402
from Design.SMRF_Design_Evidence import capacity_state_from_record           # noqa: E402
from Design.SMRF_Qualification import qualify_design                         # noqa: E402


def statuses(record):
    return {(c["id"] + ("@" + c["location"] if c.get("location") else "")): c for c in qualify_design(record)["checks"]}


class FinalQualificationCounterexamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack = contextlib.ExitStack()
        for name in set(driver._STATE_KEYS) | {"COVER", "NUM_BAY_X", "NUM_BAY_Y", "NUM_FLOOR", "NUM_MODES", "ASCE_SDS"}:
            cls.stack.enter_context(mock.patch.object(sp, name, getattr(sp, name)))
        sp.NUM_BAY_X = sp.NUM_BAY_Y = 1
        sp.NUM_FLOOR = 2
        sp.NUM_MODES = 3 * sp.NUM_FLOOR
        # The toy 10-in beam meets an 18-in column with four #8 per face: one bar passes per layer, so
        # its two bars would take two layers (allowed since 2026-09-27) and the layered depth leaves the
        # one-iteration search no capacity-shear section. The fixture starts at the 18-in variant of
        # its depth, where three bars pass per layer and the record fails nothing of its own.
        sp.B_BEAM = 18.0
        try:
            record = driver.design_structure(max_section_iter=1, max_steel_iter=1, verbose=False)
        finally:
            driver.ops.wipe()
        cls.record = json.loads(json.dumps(record, allow_nan=False))       # as read back from disk

    @classmethod
    def tearDownClass(cls):
        cls.stack.close()

    def fresh(self):
        return copy.deepcopy(self.record)

    def verified(self):
        """The designed record with the story-strength model asserted verified, consistently: the record's
        IndependentVerification assertion (with provenance) and the evidence's applicability status."""
        record = self.fresh()
        stamp = {"asserted_by": "test", "assertion_date": "2026-09-20", "assertion_basis": "fixture: review item M1 closed",
                 "story_strength_model_verified": True}
        # The in-memory record carries the policy under demand_basis; a written one also under request_identity.
        for policy in (((record.get("request_identity") or {}).get("policy") or {}).get("verification"),
                       record["demand_basis"].get("verification")):
            if isinstance(policy, dict):
                policy.update(stamp)
        block = record["demand_basis"]["regularity"]["lateral_strength_distribution"]
        block["applicability"] = {**block["applicability"], "status": "verified"}
        record["demand_basis"]["torsion"].pop("classification", None)
        return record

    # ---- R3: the saved assessment carries both physical inputs ------------------------------------
    def test_assessment_records_story_drifts_and_level_displacements_separately(self):
        torsion = self.record["demand_basis"]["torsion"]
        self.assertLessEqual(torsion["tir"], 1.2)                                # the symmetric fixture frame
        self.assertEqual(sorted(torsion["cases"]), ["x+", "x-", "y+", "y-"])
        self.assertEqual(len(torsion["stories"]), 4 * sp.NUM_FLOOR)
        for row in torsion["stories"]:
            for key in ("delta_end_a_in", "delta_end_b_in", "delta_max_over_avg", "delta_level_a_in", "delta_level_b_in",
                        "level_max_over_avg", "ax_level"):
                self.assertIn(key, row)
        self.assertEqual(torsion["assessment_amplification"], 1.0)
        validated = validate_torsion_assessment(torsion, sp.NUM_FLOOR, 0.05)
        self.assertTrue(validated["valid"], validated.get("reason"))
        self.assertEqual(set(validated["ax_by_level"]), set(range(1, sp.NUM_FLOOR + 1)))
        self.assertAlmostEqual(torsion["amplification_envelope_12_8_4_3"], validated["ax_required_envelope"])
        self.assertGreaterEqual(torsion["amplification"], torsion["amplification_required"])
        # The strength-distribution evidence names its model and reproduces from its lines (R4).
        strength = self.record["demand_basis"]["regularity"]["lateral_strength_distribution"]
        self.assertIn("beam-sway mechanism", strength["model"])
        self.assertEqual(set(strength["by_direction"]), {"x", "y"})
        self.assertAlmostEqual(strength["one_side_fraction"], 0.5)             # one bay per direction: two lines, no center line
        self.assertEqual(strength["by_direction"]["x"]["lines_at_center"], 0)
        # Its applicability is a distinct, provisional status (review item M1), never implied by the arithmetic.
        self.assertEqual(strength["applicability"]["status"], "provisional")
        self.assertEqual(strength["applicability"]["review_item"], "M1")
        self.assertEqual(torsion["torsional_irregularity"], "unresolved")
        self.assertEqual(torsion["classification"]["strength_criterion_status"], "provisional")
        self.assertEqual(torsion["base"], "fixed")

    def test_designed_record_evaluates_the_torsion_items(self):
        """Unasserted story-strength model: the TIR ceiling evaluates, the Type 1 item stays open (M1)."""
        checks = statuses(self.record)
        item = checks["demands.torsional_irregularity"]
        self.assertEqual(item["status"], "not_evaluated")
        self.assertIn("provisional", item["details"]["reason"])
        self.assertIn("M1", item["details"]["reason"])
        self.assertEqual(item["details"]["classification"]["label"], "unresolved")
        self.assertEqual(checks["demands.accidental_torsion"]["status"], "pass")
        self.assertEqual(checks["demands.accidental_torsion"]["details"]["limit_kind"], "project_policy")
        # With the model asserted verified (assertion and evidence status together) the item evaluates.
        verified = statuses(self.verified())["demands.torsional_irregularity"]
        self.assertEqual(verified["status"], "pass")
        self.assertEqual(verified["details"]["rows_validated"], 4 * sp.NUM_FLOOR)
        self.assertEqual(verified["details"]["strength_criterion_status"], "verified")
        # The assertion alone (evidence still priced as provisional) is stale; the evidence claim alone is unbacked.
        record = self.verified()
        record["demand_basis"]["regularity"]["lateral_strength_distribution"]["applicability"]["status"] = "provisional"
        self.assertIn("stale", statuses(record)["demands.torsional_irregularity"]["details"]["reason"])
        record = self.fresh()
        record["demand_basis"]["regularity"]["lateral_strength_distribution"]["applicability"]["status"] = "verified"
        record["demand_basis"]["torsion"].pop("classification", None)
        self.assertIn("no story_strength_model_verified assertion", statuses(record)["demands.torsional_irregularity"]["details"]["reason"])

    # ---- R2: Codex's counterexamples through qualify_design ----------------------------------------
    def test_removing_every_case_and_row_leaves_the_items_open(self):
        record = self.fresh()
        record["demand_basis"]["torsion"]["stories"] = []
        record["demand_basis"]["torsion"]["cases"] = []
        checks = statuses(record)
        self.assertEqual(checks["demands.torsional_irregularity"]["status"], "not_evaluated")
        self.assertEqual(checks["demands.accidental_torsion"]["status"], "not_evaluated")
        self.assertIn("rows", checks["demands.torsional_irregularity"]["details"]["reason"])

    def test_changed_scalars_with_a_cached_classification_are_rejected(self):
        record = self.fresh()
        torsion = record["demand_basis"]["torsion"]
        torsion["tir"] = torsion["max_drift_ratio"] = 1.35
        torsion["amplification"] = 1.01
        checks = statuses(record)
        self.assertEqual(checks["demands.torsional_irregularity"]["status"], "not_evaluated")
        self.assertEqual(checks["demands.accidental_torsion"]["status"], "not_evaluated")
        self.assertIn("stored tir", checks["demands.torsional_irregularity"]["details"]["reason"])

    def test_removed_strength_evidence_keeps_the_classification_open(self):
        record = self.fresh()
        del record["demand_basis"]["regularity"]["lateral_strength_distribution"]
        del record["demand_basis"]["torsion"]["classification"]
        checks = statuses(record)
        self.assertEqual(checks["demands.torsional_irregularity"]["status"], "not_evaluated")
        self.assertIn("strength-distribution", checks["demands.torsional_irregularity"]["details"]["reason"])
        self.assertEqual(checks["demands.accidental_torsion"]["status"], "pass")    # the TIR ceiling needs no strength evidence

    def test_duplicate_and_missing_rows_are_rejected(self):
        record = self.fresh()
        record["demand_basis"]["torsion"]["stories"].append(dict(record["demand_basis"]["torsion"]["stories"][0]))
        self.assertEqual(statuses(record)["demands.torsional_irregularity"]["status"], "not_evaluated")
        record = self.fresh()
        record["demand_basis"]["torsion"]["stories"].pop()
        self.assertEqual(statuses(record)["demands.torsional_irregularity"]["status"], "not_evaluated")

    def test_inconsistent_row_and_scalar_values_are_rejected(self):
        record = self.fresh()
        record["demand_basis"]["torsion"]["stories"][0]["delta_max_over_avg"] = 1.5
        self.assertEqual(statuses(record)["demands.torsional_irregularity"]["status"], "not_evaluated")
        record = self.fresh()
        record["demand_basis"]["torsion"]["tir_by_case"]["x+"] = 1.5
        self.assertEqual(statuses(record)["demands.torsional_irregularity"]["status"], "not_evaluated")
        record = self.fresh()
        record["demand_basis"]["torsion"]["stories"][1]["ax_level"] = 2.0
        self.assertEqual(statuses(record)["demands.torsional_irregularity"]["status"], "not_evaluated")

    def test_legacy_single_sign_assessment_is_open_not_manufactured(self):
        record = self.fresh()
        torsion = record["demand_basis"]["torsion"]
        legacy_rows = [{k: r[k] for k in ("story", "direction", "delta_end_a_in", "delta_end_b_in", "delta_max_over_avg")}
                       for r in torsion["stories"] if r["eccentricity_sign"] > 0]
        record["demand_basis"]["torsion"] = {"ratio": 0.05, "amplification": torsion["amplification"],
                                             "max_drift_ratio": torsion["max_drift_ratio"],
                                             "torsional_irregularity": "none", "stories": legacy_rows}
        checks = statuses(record)
        self.assertEqual(checks["demands.torsional_irregularity"]["status"], "not_evaluated")
        self.assertIn("legacy", checks["demands.torsional_irregularity"]["details"]["reason"])
        self.assertEqual(checks["demands.accidental_torsion"]["status"], "not_evaluated")

    # ---- F1 (second review): strength-distribution evidence must be complete ----------------------
    def test_strength_evidence_gaps_stay_open_through_qualification(self):
        def irregularity(mutate):
            record = self.fresh()
            block = record["demand_basis"]["regularity"]["lateral_strength_distribution"]
            mutate(block)
            record["demand_basis"]["torsion"].pop("classification", None)
            return statuses(record)["demands.torsional_irregularity"]

        for mutate, expect in (
                (lambda b: b["by_direction"].pop("y"), "exactly directions x and y"),
                (lambda b: b.pop("by_direction"), "exactly directions x and y"),
                (lambda b: b["by_direction"]["x"]["line_positions_in"].append(999.0), "frame lines"),
                (lambda b: b["by_direction"]["x"]["line_positions_in"].__setitem__(0, 5.0), "grid positions"),
                (lambda b: b["strength_inputs"]["families"]["x_interior"].__setitem__("mn_negative_kip_in", 1.0), "stale"),
                (lambda b: b.pop("strength_inputs"), "not recorded")):
            check = irregularity(mutate)
            self.assertEqual(check["status"], "not_evaluated", expect)
            self.assertIn(expect, check["details"]["reason"])
        # Hiding a removed direction behind an "updated" aggregate does not help.
        def hide_y(b):
            b["by_direction"].pop("y")
            b["one_side_fraction"] = b["by_direction"]["x"]["one_side_fraction"]
        self.assertEqual(irregularity(hide_y)["status"], "not_evaluated")
        # Control: the designed record's complete evidence, once its model is asserted, passes and reproduces
        # from the record's own cage; unasserted it is open for the model, not for the evidence.
        control = statuses(self.verified())["demands.torsional_irregularity"]
        self.assertEqual(control["status"], "pass")
        self.assertIn("by_direction", control["details"]["strength_distribution"])
        unasserted = statuses(self.record)["demands.torsional_irregularity"]
        self.assertEqual(unasserted["status"], "not_evaluated")
        self.assertIn("provisional", unasserted["details"]["reason"])

    # ---- F2 (second review): drift and displacement primitives must agree -------------------------
    def test_drift_displacement_consistency_through_qualification(self):
        from Design.SMRF_Demands import story_drift_ratio, amplification_from_level_displacements
        # Inconsistent: one recorded story drift no longer equals the difference of its edge's level displacements.
        record = self.fresh()
        row = record["demand_basis"]["torsion"]["stories"][0]
        row["delta_end_a_in"] *= 1.01
        row["delta_max_over_avg"] = story_drift_ratio(row["delta_end_a_in"], row["delta_end_b_in"])
        checks = statuses(record)
        self.assertEqual(checks["demands.torsional_irregularity"]["status"], "not_evaluated")
        self.assertIn("recorded story drift", checks["demands.torsional_irregularity"]["details"]["reason"])
        # Consistent non-proportional response on the designed two-story frame: levels (0.10, 0.10) then
        # (0.30, 0.20) in; story drifts (0.10, 0.10) and (0.20, 0.10): TIR 1.333, level ratios 1.0 and 1.2, Ax 1.0.
        record = self.fresh()
        torsion = record["demand_basis"]["torsion"]
        levels = {1: (0.10, 0.10), 2: (0.30, 0.20)}
        for row in torsion["stories"]:
            sign, k = row["eccentricity_sign"], row["story"]
            la, lb = levels[k]
            pa, pb = levels.get(k - 1, (0.0, 0.0))
            row.update(delta_level_a_in=sign * la, delta_level_b_in=sign * lb,
                       delta_end_a_in=abs(la - pa), delta_end_b_in=abs(lb - pb),
                       delta_max_over_avg=story_drift_ratio(la - pa, lb - pb))
            level = amplification_from_level_displacements(la, lb)
            row.update(level_max_over_avg=level["ratio"], ax_level=level["ax"])
        torsion["tir_by_case"] = {c: 4.0 / 3.0 for c in torsion["cases"]}
        torsion["tir"] = torsion["max_drift_ratio"] = 4.0 / 3.0
        torsion["amplification_by_level"] = {"1": 1.0, "2": 1.0}
        torsion["amplification_required"] = torsion["amplification"] = 1.0
        torsion.pop("classification", None)
        checks = statuses(record)
        self.assertEqual(checks["demands.torsional_irregularity"]["status"], "pass")
        self.assertTrue(checks["demands.torsional_irregularity"]["details"]["by_tir"])
        self.assertEqual(checks["demands.torsional_irregularity"]["capacity"], 1.0)          # level ratio 1.2 -> Ax 1
        self.assertAlmostEqual(checks["demands.accidental_torsion"]["demand"], 4.0 / 3.0)   # TIR from story drifts

    # ---- R5: missing joint continuity evidence through final qualification --------------------------
    def test_missing_joint_continuity_stays_open_through_qualification(self):
        """A record designed without the declaration: conservative gammas, but nothing reads as checked."""
        record = self.fresh()
        state = capacity_state_from_record(record)
        self.assertIsNotNone(state["joint_continuity"])                      # the driver declared it
        state["joint_continuity"] = None
        record["capacity_design"] = json.loads(json.dumps(build_capacity_design(state), allow_nan=False))
        record["detailing"]["joint_continuity"] = None
        result = qualify_design(record)
        checks = {(c["id"] + ("@" + c["location"] if c.get("location") else "")): c for c in result["checks"]}
        self.assertEqual(checks["qualification.capacity_evidence_recomputed"]["status"], "pass")   # the record is self-consistent
        self.assertEqual(checks["qualification.joint_capacity_completion"]["status"], "not_evaluated")
        # Floor corner joints need the column-continuity declaration: open, naming what is missing.
        # Roof corner joints are Other/Other by topology alone: evaluated on complete evidence.
        joint_items = {k: v for k, v in checks.items() if k.startswith("joint_shear@")}
        floor_items = {k: v for k, v in joint_items.items() if "/floor/" in k}
        roof_items = {k: v for k, v in joint_items.items() if "/roof/" in k}
        self.assertTrue(floor_items and roof_items)
        self.assertTrue(all(v["status"] == "not_evaluated" for v in floor_items.values()), {k: v["status"] for k, v in floor_items.items()})
        self.assertTrue(all("Unevaluated: column_reinforcement" in v["details"]["reason"] for v in floor_items.values()))
        self.assertTrue(all(v["status"] in ("pass", "fail") for v in roof_items.values()))
        evidence_items = {k: v for k, v in checks.items() if k.startswith("joint.classification_evidence@")}
        self.assertTrue(all(v["status"] == "not_evaluated" for k, v in evidence_items.items() if "/floor/" in k))
        self.assertTrue(all(v["status"] == "pass" for k, v in evidence_items.items() if "/roof/" in k))
        self.assertFalse(result["accepted"])
        # The declared record keeps every joint item evaluated.
        declared = statuses(self.record)
        self.assertTrue(all(v["status"] != "not_evaluated" for k, v in declared.items() if k.startswith("joint_shear@")))
        self.assertTrue(all(v["status"] == "pass" for k, v in declared.items() if k.startswith("joint.classification_evidence@")))
        self.assertNotEqual(declared["qualification.joint_capacity_completion"]["status"], "not_evaluated")


    # ---- one slab convention on both sides of the design (2026-09-21) ------------------------------
    def test_hinge_strengths_equal_the_record_families_without_a_slab_layout(self):
        """This frame was designed without the PROBE assertions, so no slab reinforcement layout exists.
        The record's families then price hogging on the bare rectangle and sagging with the chosen slab's
        flange concrete and no mats; the installed hinge must yield at exactly those values (it used to
        price sagging on a thickness-0 section, weaker than the design assumed)."""
        from Model.Build_Model import build_model
        from Model.IMK_Hinges import hinge_registry
        from Design.SMRF_Beam_Slab_Strength import beam_slab_strengths
        record = self.fresh()
        self.assertIsNone((record.get("slab_reinforcement") or {}).get("layout"))
        self.assertIsNotNone(record["slab"]["thickness_in"])
        entries, families = beam_slab_strengths(record)
        self.assertTrue(all(e["slab_basis"] == "unknown" for e in entries.values()))     # joints stay not_evaluated
        driver.apply_design(record)
        try:
            build_model()
            registry = hinge_registry()
        finally:
            driver.ops.wipe()
        beams = [e for e in registry.values() if e["member_type"] != "column"]
        self.assertTrue(beams)
        for entry in beams:
            fam = families[entry["beam_family"]]
            self.assertAlmostEqual(entry["yield_moment_y_hogging_kip_in"], fam["mn_negative_kip_in"], delta=1e-6)
            self.assertAlmostEqual(entry["yield_moment_y_sagging_kip_in"], fam["mn_positive_kip_in"], delta=1e-6)
            # Sagging counts the flange concrete (above the rectangle); hogging counts no mats (equals the rectangle).
            self.assertGreater(entry["yield_moment_y_sagging_kip_in"], fam["rectangular"]["positive"]["mn_kip_in"] + 1.0)
            self.assertAlmostEqual(entry["yield_moment_y_hogging_kip_in"], fam["rectangular"]["negative"]["mn_kip_in"], delta=1e-6)
            self.assertIn("flange concrete", entry["strength_basis"])
            for end in ("i", "j"):                      # no layout: no per-end undeveloped reduction either
                self.assertAlmostEqual(entry[f"yield_moment_y_hogging_{end}_kip_in"], fam["mn_negative_kip_in"], delta=1e-6)
                self.assertAlmostEqual(entry[f"yield_moment_y_sagging_{end}_kip_in"], fam["mn_positive_kip_in"], delta=1e-6)
        # The pilot's installed-design verification agrees, with no evidence gap left to report.
        from Analysis.Hinge_Hysteresis_Diagnostic import verify_installed_design
        driver.apply_design(record)
        try:
            build_model()
            verification = verify_installed_design(record)
        finally:
            driver.ops.wipe()
        self.assertTrue(verification["consistent"], verification["differences"])
        self.assertEqual(verification["design_evidence_gaps"], [])


class StrengthDistributionTopologyTests(unittest.TestCase):
    """R4 on the live driver state: odd and even bay counts, edge/interior families."""

    def setUp(self):
        self.stack = contextlib.ExitStack()
        for name in set(driver._STATE_KEYS) | {"NUM_BAY_X", "NUM_BAY_Y", "NUM_FLOOR", "BAY_X", "BAY_Y", "STORY_H"}:
            self.stack.enter_context(mock.patch.object(sp, name, getattr(sp, name)))
        sp.SLAB_THICKNESS_IN = None
        sp.SLAB_REINFORCEMENT = None
        sp.FLOOR_TRANSFER = None

    def tearDown(self):
        self.stack.close()

    def test_center_line_counts_on_both_sides(self):
        sp.NUM_BAY_X, sp.NUM_BAY_Y = 2, 3                                    # x lines: 4 (no center); y lines: 3 (center)
        block = driver._lateral_strength_distribution()
        self.assertIsNotNone(block["one_side_fraction"])
        x, y = block["by_direction"]["x"], block["by_direction"]["y"]
        # Direction x: frames on the 4 y-grid lines, all equal on the rectangular proxy -> 1/2.
        self.assertEqual(x["lines_at_center"], 0)
        self.assertAlmostEqual(x["one_side_fraction"], 0.5)
        # Direction y: frames on the 3 x-grid lines, the middle one at the center of mass -> 2/3.
        self.assertEqual(y["lines_at_center"], 1)
        self.assertAlmostEqual(y["one_side_fraction"], 2.0 / 3.0)
        self.assertAlmostEqual(block["one_side_fraction"], 2.0 / 3.0)
        self.assertEqual(len(y["line_story_strength_kip"]), 3)
        self.assertTrue(all(s > 0 for s in y["line_story_strength_kip"]))
        self.assertIn("beam-sway mechanism", block["model"])

    def test_weaker_perimeter_lines_raise_the_three_line_fraction(self):
        import Design.SMRF_Beam_Slab_Strength as strengths
        sp.NUM_BAY_X, sp.NUM_BAY_Y = 2, 2
        block = driver._lateral_strength_distribution()
        families = block["strength_inputs"]["families"]
        weak_edge = {**families, "x_edge": {"mn_negative_kip_in": 0.4 * families["x_interior"]["mn_negative_kip_in"],
                                            "mn_positive_kip_in": 0.4 * families["x_interior"]["mn_positive_kip_in"]}}
        with mock.patch.object(strengths, "beam_slab_strengths", return_value=({}, weak_edge)):
            weak = driver._lateral_strength_distribution()
        self.assertAlmostEqual(weak["by_direction"]["x"]["one_side_fraction"], 1.4 / 1.8)   # (1 + 0.4) / (1 + 0.8)
        self.assertGreater(weak["one_side_fraction"], 0.75)
        self.assertEqual(weak["strength_inputs"]["families"]["x_edge"], weak_edge["x_edge"])  # the inputs travel with the lines
        with mock.patch.object(strengths, "beam_slab_strengths", side_effect=RuntimeError("no families")):
            unknown = driver._lateral_strength_distribution()
        self.assertIsNone(unknown["one_side_fraction"])
        self.assertIn("not evaluated", unknown["basis"])


if __name__ == "__main__":
    unittest.main()
