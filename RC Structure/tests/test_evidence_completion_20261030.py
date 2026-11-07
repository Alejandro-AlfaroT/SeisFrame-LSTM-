"""Evidence repairs of the code-check audit of the genetic search (2026-10-30): findings F03-F08.

  F03  slab credit: a selected layout is credited as developed only where the slab record's own development
       check passes on the current floor context; the exterior perimeter-hook treatment is kept;
  F04  frame-beam axial applicability (ACI 318-19 18.6.4.7) from the signed envelope of the solved actions;
  F05  splice placement by bar grade (18.2.7.2: Type 2 anywhere only for Grade 60);
  F06  saved drift checks rebuilt from their numbers instead of trusted;
  F07  grouped hoop identity: missing selection, selected mismatch and selected match kept apart;
  F08  the qualification scope follows the system actually designed.

Rows and records are small and written by hand; the fixture frame is not a designed structure.
"""
import copy
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402
from test_smrf_slab_reinforcement import CORNERS                                 # noqa: E402

import openseespy.opensees as ops                                               # noqa: E402
import Structure_Parameters as sp                                               # noqa: E402
from Design import Code_Check_Contract as contract                               # noqa: E402
from Design import Group_Capacity as gc                                          # noqa: E402
from Design import Grouped_Record as gr                                          # noqa: E402
from Design import SMRF_Capacity_Design as cd                                    # noqa: E402
from Design.SMRF_Common import make_check, not_evaluated                         # noqa: E402
from Design.SMRF_Demands import evaluate_drift_and_stability, reconstruct_drift_checks  # noqa: E402
from Design.SMRF_Joints import evaluate_joints                                   # noqa: E402
from Design.SMRF_Qualification import qualification_scope                        # noqa: E402
from Design.SMRF_Slab_Reinforcement import slab_strength_eligibility             # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402

CONTEXT = {"clear_span_x_in": 214.0, "clear_span_y_in": 190.0, "beam_width_in": 18.0, "alpha_f_min": 2.0,
           "alpha_f_l2_l1_min": 2.0, "thickness_screen_passed": True, "column_core_width_in": 23.0,
           "two_way_shear_path_assessed": False, "slab_column_local_steel_assessed": False,
           "columns_at_beam_intersections": True, "beam_clear_cover_in": 1.5,
           "beam_hoop_diameter_in": 0.5, "fc_beam_ksi": 4.0, "perimeter_hook_corners": CORNERS}


def slab_layout(size=3, spacing=12.0):
    return {"bar_size": size, "outer_axis": "x", "layers": {
        f"{axis}_{face}": {"bar_size": size, "bar_area_in2": sp.rebar_area(size), "spacing_in": spacing,
                           "effective_depth_in": gf.SLAB - 1.25 if face == "top" else 1.25, "axis": axis,
                           "layer": "outer" if axis == "x" else "inner", "clear_cover_outer_mat_in": 1.0}
        for axis in ("x", "y") for face in ("top", "bottom")}}


def slab_record(layout=True, development="pass", anchorage=("pass", "pass"), context=CONTEXT, size=3):
    """A slab reinforcement record with the completion rows the eligibility reads."""
    if not layout:
        return {"layout": None, "inputs": {"slab": {"thickness_in": gf.SLAB}, "context": context,
                                           "demand_evidence": {"verified": False}},
                "checks": [not_evaluated("slab_verified_strip_action_inputs", "ACI 318-19 8.4",
                                         "Upstream slab analysis verification is incomplete: ['verified'].")]}
    rows = []
    if development is not None:
        rows.append(make_check("slab_bar_development", "ACI 318-19 25.4.2.4", 12.0, 95.0 if development == "pass" else 6.0))
    for axis, status in zip(("x", "y"), anchorage):
        rows.append(make_check("slab_perimeter_bar_anchorage", "ACI 318-19 25.4.3.1", 6.0, 16.0 if status == "pass" else 3.0,
                               location=axis))
    return {"layout": slab_layout(size), "inputs": {"slab": {"thickness_in": gf.SLAB}, "context": context}, "checks": rows,
            "method_version": "fixture"}


# ---- F03 ----------------------------------------------------------------------------------------------------------
class SlabCreditEligibility(unittest.TestCase):
    def test_an_absent_layout_stays_unknown_with_its_upstream_reason(self):
        result = slab_strength_eligibility(slab_record(layout=False))
        self.assertEqual((result["credit"], result["layout_selected"]), ("not_established", False))
        self.assertEqual(result["reason_code"], "slab_action_review_missing")
        # No record and the placeholder a record keeps without demand evidence (transfer disabled) say the same thing,
        # so an evaluation and its qualification agree.
        placeholder = {"layout": None, "inputs": {"slab": {"thickness_in": gf.SLAB}, "demand_evidence": None},
                       "checks": [not_evaluated("slab_verified_strip_action_inputs", "ACI 318-19 8.4",
                                                "Verified slab action evidence is missing.")]}
        keys = ("credit", "reason_code", "reason", "layout_selected", "signature")
        self.assertEqual({k: slab_strength_eligibility(None)[k] for k in keys},
                         {k: slab_strength_eligibility(placeholder)[k] for k in keys})
        self.assertEqual(slab_strength_eligibility(None)["reason_code"], "no_slab_action_evidence")

    def test_a_selected_layout_with_failed_development_is_not_developed(self):
        result = slab_strength_eligibility(slab_record(development="fail"))
        self.assertTrue(result["layout_selected"])
        self.assertEqual((result["credit"], result["reason_code"]), ("not_established", "slab_bar_development_failed"))
        self.assertFalse(result["slab_checks_passed"])
        self.assertEqual([c["id"] for c in result["failed_checks"]], ["slab_bar_development"])

    def test_a_missing_development_check_is_unknown_not_developed(self):
        result = slab_strength_eligibility(slab_record(development=None))
        self.assertEqual((result["credit"], result["reason_code"]), ("not_established", "slab_bar_development_unknown"))

    def test_a_passing_development_check_on_the_current_context_is_developed(self):
        result = slab_strength_eligibility(slab_record(), current_context=dict(CONTEXT))
        self.assertEqual((result["credit"], result["context_current"]), ("developed", True))

    def test_a_cage_or_hoop_change_after_pricing_makes_the_record_stale(self):
        changed = {**CONTEXT, "beam_hoop_diameter_in": 0.625}                      # the perimeter hoop grew
        result = slab_strength_eligibility(slab_record(), current_context=changed)
        self.assertEqual((result["credit"], result["reason_code"]), ("not_established", "slab_evidence_stale"))

    def test_review_items_are_listed_apart_from_unknown_checks(self):
        from Design.SMRF_Common import awaiting_review
        record = slab_record()
        # A neutral review item: the fire row is a declared scope exclusion since 2026-11-01, not a review.
        record["checks"].append(awaiting_review("slab_review_fixture", "fixture clause", "fixture review",
                                                "fixture_review"))
        result = slab_strength_eligibility(record)
        self.assertEqual([r["item"] for r in result["review_dependencies"]], ["fixture_review"])
        self.assertEqual(result["unknown_checks"], [])

    def test_the_review_bundle_keeps_computed_preconditions_apart_from_assertions(self):
        from Design.Grouped_Design import slab_review_bundle
        self.assertEqual(slab_review_bundle(None)["status"], "no_slab_action_evidence")
        evidence = {"method": "plate_finite_element", "cases": [{"id": "1.2D+1.6L"}], "pattern_rule": "fixture",
                    "numerical_preconditions": {"physical_recovery_valid": True, "mesh_refinement_verified": False},
                    "engineering_assertions": {}, "assertion_provenance_valid": False,
                    "equilibrium": [{"numerical_balance_passed": True}], "shear_recovery": {"top_rows_recovered_at_face": True}}
        bundle = slab_review_bundle(evidence)
        self.assertEqual(bundle["status"], "blocked")
        self.assertIn("computed precondition not met: mesh_refinement_verified", bundle["blocking"])
        self.assertIn("not asserted: verified", bundle["blocking"])
        self.assertTrue(bundle["computed"]["equilibrium"]["all_balanced"])
        self.assertIn("floor.independent_hand_verification", bundle["engineering_review"]["independent_reference_calculation"])

    def test_changing_the_slab_bars_changes_the_realized_identity(self):
        identity = gf.uniform_state().identity()
        a = contract.realized_identity(identity, gf.SLAB, slab_record(size=3))
        b = contract.realized_identity(identity, gf.SLAB, slab_record(size=4))
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, contract.realized_identity(gf.changed_state().identity(), gf.SLAB, slab_record(size=3)))


class StrongColumnSlabCredit(unittest.TestCase):
    """The joint-face beam strengths the strong-column rule reads, at interior, exterior and corner joints, both axes,
    both sway signs, with a column band transition (the changed state's smaller upper interior columns)."""

    def setUp(self):
        self.stack = gf.frame()
        self.stack.__enter__()
        self.state = mg.install(gf.changed_state())
        self.inventory = gc.joint_inventory(self.state)
        tags = sorted({c.member_tag for e in self.inventory["joints"] for c in (e["below"], e["above"]) if c is not None})
        self.actions = [{"id": "c1", "analysis_succeeded": True, "axial_reference": "joint_faces",
                         "members": {str(t): {"axial_i_kip": 80.0, "axial_j_kip": 75.0} for t in tags}}]

    def tearDown(self):
        self.stack.__exit__(None, None, None)

    def rows(self, record, anchorage_developed=True):
        sp.SLAB_REINFORCEMENT = record
        eligibility = slab_strength_eligibility(record)
        real = gc.mp.perimeter_slab_bar_anchorage if hasattr(gc.mp, "perimeter_slab_bar_anchorage") else None
        from Design import SMRF_Beam_Slab_Strength as bss
        original = bss.perimeter_slab_bar_anchorage

        def anchorage(*args, **kwargs):
            result = original(*args, **kwargs)
            return result if anchorage_developed or result is None else {**result, "developed": False}
        with mock.patch.object(bss, "perimeter_slab_bar_anchorage", anchorage):
            evidence = gc.scwb_evidence(self.inventory, self.actions, ["c1"], state=self.state, slab_eligibility=eligibility)
        del real
        checks = [c for c in evaluate_joints({"joints": evidence["joints"]}) if c["id"] == "scwb"]
        return evidence, {c["location"]: c for c in checks}

    def joint(self, i, j, level=1):
        return next(e["id"] for e in self.inventory["joints"] if (e["level"], e["grid_i"], e["grid_j"]) == (level, i, j))

    def test_every_joint_axis_and_sign_has_a_row(self):
        _evidence, rows = self.rows(slab_record())
        self.assertEqual(len(rows), len(self.inventory["joints"]) * 2 * 2)

    def test_developed_mats_evaluate_every_joint_including_a_transition(self):
        evidence, rows = self.rows(slab_record())
        self.assertFalse([loc for loc, c in rows.items() if c["status"] == "not_evaluated"])
        transition = next(e for e in self.inventory["joints"] if e["above"] is not None
                          and e["above"].group_id != e["core"].group_id)
        self.assertIn(rows[f"{transition['id']}/x/positive"]["status"], ("pass", "fail"))
        bases = {cap.get("slab_basis") for j in evidence["joints"] for d in j["directions"].values()
                 for s in ("positive", "negative") for cap in d[s]["beam_capacities"]}
        self.assertEqual(bases, {"developed_effective_width"})

    def test_failed_interior_development_leaves_every_credited_end_unknown_with_its_reason(self):
        _evidence, rows = self.rows(slab_record(development="fail"))
        self.assertTrue(all(c["status"] == "not_evaluated" for c in rows.values()))
        self.assertTrue(all("slab_bar_development_failed" in c["details"]["reason_codes"] for c in rows.values()))

    def test_an_exterior_hook_that_does_not_fit_keeps_its_own_known_basis(self):
        # Interior development failed, perimeter hooks do not fit: the corner joint (both beams end at the perimeter)
        # is evaluated on the known undeveloped strengths; an interior joint is not.
        evidence, rows = self.rows(slab_record(development="fail"), anchorage_developed=False)
        corner, interior = self.joint(0, 0), self.joint(1, 1)
        for axis in ("x", "y"):
            for sign in ("positive", "negative"):
                self.assertIn(rows[f"{corner}/{axis}/{sign}"]["status"], ("pass", "fail"))
                self.assertEqual(rows[f"{interior}/{axis}/{sign}"]["status"], "not_evaluated")
        caps = [cap for j in evidence["joints"] if j["id"] == corner for d in j["directions"].values()
                for s in ("positive", "negative") for cap in d[s]["beam_capacities"]]
        self.assertEqual({cap["slab_basis"] for cap in caps}, {"terminated_undeveloped", "flange_concrete_undeveloped_bars"})
        self.assertTrue(all(cap["slab_mn_kip_in"] == 0.0 for cap in caps if cap["flexure_sign"] == "negative"))

    def test_no_layout_is_unknown_with_the_upstream_reason_never_zero(self):
        _evidence, rows = self.rows(slab_record(layout=False))
        self.assertTrue(all(c["status"] == "not_evaluated" for c in rows.values()))
        self.assertTrue(all("slab_action_review_missing" in c["details"]["reason_codes"] for c in rows.values()))
        self.assertTrue(all(c["demand"] is None for c in rows.values()))

    def test_the_saved_evidence_does_not_depend_on_the_context_comparison(self):
        # The evaluation may read the record with its floor context checked; the qualification recomputes the capacity
        # design from the saved record alone. When the context is current both must give the same saved evidence.
        record = slab_record()
        sp.SLAB_REINFORCEMENT = record
        checked = gc.scwb_evidence(self.inventory, self.actions, ["c1"], state=self.state,
                                   slab_eligibility=slab_strength_eligibility(record, current_context=dict(CONTEXT)))
        self.assertEqual(checked, gc.scwb_evidence(self.inventory, self.actions, ["c1"], state=self.state))

    def test_stale_slab_evidence_is_unknown(self):
        sp.SLAB_REINFORCEMENT = slab_record()
        stale = slab_strength_eligibility(slab_record(), current_context={**CONTEXT, "column_core_width_in": 19.0})
        evidence = gc.scwb_evidence(self.inventory, self.actions, ["c1"], state=self.state, slab_eligibility=stale)
        rows = [c for c in evaluate_joints({"joints": evidence["joints"]}) if c["id"] == "scwb"]
        self.assertTrue(all("slab_evidence_stale" in c["details"]["reason_codes"] for c in rows))

    def test_an_invalid_action_inventory_names_the_columns_cause(self):
        sp.SLAB_REINFORCEMENT = slab_record()
        evidence = gc.scwb_evidence(self.inventory, self.actions + [dict(self.actions[0])], ["c1"], state=self.state)
        rows = [c for c in evaluate_joints({"joints": evidence["joints"]}) if c["id"] == "scwb"]
        self.assertTrue(all("action_inventory_invalid" in c["details"]["reason_codes"] for c in rows))
        self.assertFalse(evidence["action_inventory_complete"])


class UniformSearchReactsToKnownFailuresOnly(unittest.TestCase):
    """all_pass now means complete and passing; the uniform confining-beam retry reacts to a known strong-column
    failure, not to unknown strong-column evidence (the summary the driver writes, Design_Driver._joint_scwb_state)."""

    def test_unknown_strong_column_evidence_does_not_block_the_confining_beam_retry(self):
        from Design import Uniform_Search as search
        columns = [(34.0, 34.0, 10.0)]
        beams = [(24.0, 30.0, 5.0), (28.0, 30.0, 5.0)]
        unknown = contract.scwb_summary([not_evaluated("scwb", "c", "slab", "joint_1/x/positive")], ["joint_1/x/positive"])
        failed = contract.scwb_summary([make_check("scwb", "c", 3.0, 1.0, location="joint_1/x/positive")],
                                       ["joint_1/x/positive"])
        base = {"strength_within_ceiling": True, "drift_accepted": True,
                "capacity_design_failed_checks": [{"id": "joint.shear_screen"}]}
        self.assertFalse(unknown["all_pass"])
        self.assertEqual(search.confining_beam_retry(columns, beams, 0, 0, {**base, "joint_scwb": unknown}), 1)
        self.assertIsNone(search.confining_beam_retry(columns, beams, 0, 0, {**base, "joint_scwb": failed}))
        self.assertEqual(search.confining_beam_retry(columns, beams, 0, 0, {**base, "joint_scwb": {"all_pass": True}}), 1)


# ---- F04 ----------------------------------------------------------------------------------------------------------
def case(cid, values, source="strength", succeeded=True):
    return (source, {"id": cid, "analysis_succeeded": succeeded,
                     "members": {str(tag): {"axial_i_kip": a, "axial_j_kip": b} for tag, (a, b) in values.items()}})


class BeamAxialApplicability(unittest.TestCase):
    B, H, FC = 18.0, 28.0, 4.0
    LIMIT = 18.0 * 28.0 * 4.0 / 10.0                                                 # 201.6 kip

    def check(self, cases, tags=(7, 8)):
        return cd.beam_axial_check(cd.beam_axial_envelope(cases, tags, self.B, self.H, self.FC, "g"))

    def test_below_at_and_above_the_threshold(self):
        self.assertEqual(self.check([case("c1", {7: (10.0, 9.0), 8: (5.0, 5.0)})])["status"], "pass")
        self.assertEqual(self.check([case("c1", {7: (self.LIMIT, self.LIMIT), 8: (0.0, 0.0)})])["status"], "pass")
        above = self.check([case("c1", {7: (250.0, 250.0), 8: (0.0, 0.0)})])
        self.assertEqual(above["status"], "fail")
        self.assertIn("unsupported", above["details"]["branch"])
        self.assertAlmostEqual(above["capacity"], self.LIMIT)

    def test_missing_failed_or_nonfinite_axial_is_unknown_never_zero(self):
        for cases in ([case("c1", {7: (10.0, 9.0)})],                                   # member 8 absent
                      [case("c1", {7: (10.0, math.nan), 8: (1.0, 1.0)})],
                      [case("c1", {7: (10.0, 9.0), 8: (1.0, 1.0)}, succeeded=None)], []):
            with self.subTest(cases=cases):
                result = self.check(cases)
                self.assertEqual(result["status"], "not_evaluated")
                self.assertEqual(result["details"]["reason_code"], "beam_axial_actions_missing")

    def test_a_missing_or_repeated_combination_leaves_the_envelope_unknown(self):
        cases = [case("c1", {7: (10.0, 9.0), 8: (5.0, 5.0)})]
        envelope = cd.beam_axial_envelope(cases, (7, 8), self.B, self.H, self.FC, "g", expected_ids=["c1", "c2"])
        self.assertEqual(envelope["status"], "unknown")
        repeated = cd.beam_axial_envelope(cases + cases, (7, 8), self.B, self.H, self.FC, "g", expected_ids=["c1"])
        self.assertEqual(repeated["status"], "unknown")
        self.assertEqual(cd.beam_axial_envelope(cases, (7, 8), self.B, self.H, self.FC, "g", expected_ids=["c1"])["status"],
                         "evaluated")

    def test_large_tension_is_not_read_as_compression(self):
        result = self.check([case("c1", {7: (-500.0, -500.0), 8: (50.0, 40.0)})])
        self.assertEqual((result["status"], result["demand"]), ("pass", 50.0))
        self.assertEqual(result["details"]["envelope"]["governing_tension"]["axial_kip"], -500.0)

    def test_the_axial_governing_case_is_named_even_when_another_case_governs_flexure(self):
        envelope = cd.beam_axial_envelope([case("flexure_governs", {7: (20.0, 20.0), 8: (0.0, 0.0)}),
                                           case("axial_governs", {7: (180.0, 175.0), 8: (0.0, 0.0)})],
                                          (7, 8), self.B, self.H, self.FC, "g")
        self.assertEqual(envelope["governing_compression"]["case"], "axial_governs")

    def test_a_gravity_compatibility_case_can_govern_its_envelope(self):
        envelope = cd.beam_axial_envelope([case("s1", {7: (20.0, 20.0)}), case("dx+", {7: (90.0, 90.0)}, "compatibility")],
                                          (7,), self.B, self.H, self.FC, "gravity")
        self.assertEqual((envelope["governing_compression"]["source"], envelope["pu_max_kip"]), ("compatibility", 90.0))

    def test_the_model_basis_states_the_rigid_diaphragm_limitation(self):
        envelope = cd.beam_axial_envelope([case("c1", {7: (1.0, 1.0)})], (7,), self.B, self.H, self.FC, "g")
        self.assertIn("rigid-diaphragm", envelope["model_basis"])

    def test_the_sign_convention_on_a_controlled_opensees_element(self):
        # One elastic member along global X fixed at node 1: a load pushing node 2 toward node 1 compresses it. The
        # captured actions read axial_i = localForce[0] and axial_j = -localForce[6]: both must be +P in compression.
        for load, expected in ((-10.0, 10.0), (10.0, -10.0)):
            ops.wipe()
            ops.model("basic", "-ndm", 3, "-ndf", 6)
            ops.node(1, 0.0, 0.0, 0.0)
            ops.node(2, 100.0, 0.0, 0.0)
            ops.fix(1, 1, 1, 1, 1, 1, 1)
            ops.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
            ops.element("elasticBeamColumn", 1, 1, 2, 100.0, 3000.0, 1200.0, 1000.0, 500.0, 500.0, 1)
            ops.timeSeries("Linear", 1)
            ops.pattern("Plain", 1, 1)
            ops.load(2, load, 0.0, 0.0, 0.0, 0.0, 0.0)
            ops.system("BandGeneral")
            ops.numberer("Plain")
            ops.constraints("Plain")
            ops.integrator("LoadControl", 1.0)
            ops.algorithm("Linear")
            ops.analysis("Static")
            self.assertEqual(ops.analyze(1), 0)
            forces = ops.eleResponse(1, "localForce")
            self.assertAlmostEqual(forces[0], expected, places=9)
            self.assertAlmostEqual(-forces[6], expected, places=9)
        ops.wipe()


class FrameBeamAxialInGroupedCapacity(unittest.TestCase):
    def setUp(self):
        self.stack = gf.frame()
        self.stack.__enter__()

    def tearDown(self):
        self.stack.__exit__(None, None, None)

    def test_the_search_has_a_move_for_the_new_failures(self):
        from Design import Grouped_Search as gs
        state = gf.uniform_state()
        beam = next(g for g in sorted(state.groups) if not state.designs[g].is_column)
        for check in ("beam.axial_force_applicability", "detailing.splice_location"):
            evaluation = {"status": "evaluated", "constraints": {"capacity_design_failed_checks": [
                {"id": check, "location": beam, "status": "fail"}]}, "evidence": {}}
            self.assertTrue(gs.propose_growth(evaluation, state), check)

    def test_the_grouped_completion_row_consumes_the_axial_check_not_a_statement(self):
        state = mg.install(gf.uniform_state())
        gid = next(g for g in sorted(state.groups) if not state.designs[g].is_column)
        hoops = {"bar_size": 4, "legs": 2, "spacing_in": 5.0}
        for axial_status, expected in (("pass", "pass"), ("fail", "fail"), ("not_evaluated", "not_evaluated")):
            axial = (make_check("beam.axial_force_applicability", "c", 10.0 if axial_status == "pass" else 300.0, 201.6,
                                location=gid, details={"branch": "fixture"}) if axial_status != "not_evaluated" else
                     not_evaluated("beam.axial_force_applicability", "c", "unknown envelope", gid))
            capacity = {"columns": {}, "beams": {gid: {"hoops": hoops, "section_adequate": True}}, "checks": [axial],
                        "splices": {}, "accepted": False}
            rows = [c for c in gr._capacity_completion_checks(capacity, state)
                    if c["id"] == "beam.hoop_layout_and_axial_applicability"]
            self.assertEqual([r["status"] for r in rows], [expected])
            self.assertNotIn("carry no factored axial load", str(rows[0]["details"]))


# ---- F05 ----------------------------------------------------------------------------------------------------------
class SplicePlacement(unittest.TestCase):
    def test_hand_checked_development_lengths(self):
        # ACI 318-19 Table 25.4.2.3, No. 8 bottom bar, f'c 5 ksi: 60000 / (20 sqrt(5000)) db = 42.43 in; Grade 80 with
        # psi_g = 1.15 (Table 25.4.2.5): 80000 x 1.15 / (20 sqrt(5000)) = 65.05 in.
        self.assertAlmostEqual(cd.straight_development_length(8, 5.0, 60.0), 42.426, places=2)
        self.assertAlmostEqual(cd.straight_development_length(8, 5.0, 80.0), 65.054, places=2)

    def test_grade_60_type_2_is_permitted_anywhere_grade_80_is_not(self):
        g60 = cd.splice_placement("beam", 100.0, 28.0, 60.0, "type_2_mechanical_18.2.7", None, 1.0)
        g80 = cd.splice_placement("beam", 100.0, 28.0, 80.0, "type_2_mechanical_18.2.7", None, 1.0)
        self.assertTrue(g60["legal_placement"])
        self.assertEqual(g60["excluded_from_each_end_in"], 0.0)
        self.assertFalse(g80["legal_placement"])                                       # [56, 44]: no interval left
        self.assertEqual(g80["permitted_interval_in"], [56.0, 44.0])

    def test_a_legal_grade_80_location_lies_outside_both_exclusion_zones(self):
        result = cd.splice_placement("beam", 216.0, 28.0, 80.0, "type_2_mechanical_18.2.7", None, 1.0)
        self.assertTrue(result["legal_placement"])
        self.assertEqual(result["permitted_interval_in"], [56.0, 160.0])
        self.assertEqual(result["proposed_interval_in"], [104.0, 112.0])
        self.assertIsNotNone(result["product_qualification"])

    def test_an_interval_too_short_for_the_coupler_is_illegal(self):
        result = cd.splice_placement("column", 140.0, 32.0, 80.0, "type_2_mechanical_18.2.7", None, 1.693)
        self.assertEqual(result["permitted_interval_in"], [64.0, 76.0])
        self.assertFalse(result["legal_placement"])                                    # 12 in for a 13.5 in envelope

    def test_lap_placements_follow_their_own_rules(self):
        column = cd.splice_placement("column", 140.0, 26.0, 60.0, "class_B_lap_center_half", 62.2, 1.128)
        self.assertEqual(column["permitted_interval_in"], [35.0, 105.0])
        self.assertTrue(column["legal_placement"])
        beam = cd.splice_placement("beam", 190.0, 28.0, 60.0, "class_B_lap_outside_hinge_zones", 80.0, 1.0)
        self.assertFalse(beam["legal_placement"])                                      # [56, 134] cannot hold 80 in

    def test_a_gravity_member_is_outside_18_2_7_2(self):
        result = cd.splice_placement("beam", 100.0, 28.0, 80.0, "type_2_mechanical_18.2.7", None, 1.0, sfrs=False)
        self.assertTrue(result["legal_placement"])
        self.assertIn("18.14", result["rule"])

    def test_grouped_splices_carry_a_location_check_per_group_at_each_grade(self):
        for grade in (60.0, 80.0):
            with self.subTest(grade=grade), gf.frame(FY_KSI=grade):
                state = mg.install(gf.uniform_state())
                splices = gc.group_splices(state)
                self.assertEqual(set(splices), set(state.groups))
                for gid, splice in splices.items():
                    placement = splice["placement"]
                    self.assertEqual(placement["grade_ksi"], grade)
                    check = gc.splice_location_check(gid, splice)
                    self.assertEqual(check["status"], "pass" if placement["legal_placement"] else "fail")

    def test_a_beam_lap_keeps_its_hoops_at_d_over_4_on_the_conventional_basis(self):
        # 2026-11-07: a lap whose groups keep their 25.2.1 spacing is used, with the hoops over it at min(d/4, 4 in)
        # (18.6.3.3); where the member's own hoops are wider the extra sets over the lap are counted.
        with gf.frame(FY_KSI=60.0):
            state = mg.install(gf.uniform_state())
            gid = next(g for g in sorted(state.groups) if g.endswith("beam_x__edge"))
            loose = gc.group_splices(state, beam_hoops={gid: 5.0})[gid]
            tight = gc.group_splices(state, beam_hoops={gid: 3.5})[gid]
        self.assertTrue(loose["splice_type"].startswith("class_B_lap"))
        details = [c["physical_detail"] for c in loose["member_contexts"]]
        self.assertTrue(all(d["splice"] == "class_b_lap" for d in details))
        self.assertTrue(all(d["additional_hoop_sets_over_lap"] > 0 for d in details))
        self.assertTrue(all("additional_hoop_sets_over_lap" not in c["physical_detail"] for c in tight["member_contexts"]))

    def test_a_beam_lap_needs_hoops_over_it_at_d_over_4_and_4_in(self):
        # The length/enclosure alternative still applies 18.6.3.3. Actual
        # production selects its documented coaxial Type 2 device; a passing
        # length-only lap must not imply an unrepresented physical lap cage.
        previous, sp.DETAIL_BASIS = sp.DETAIL_BASIS, "physical_3d_v2"
        self.addCleanup(setattr, sp, "DETAIL_BASIS", previous)
        with gf.frame(FY_KSI=60.0):
            state = mg.install(gf.uniform_state())
            gid = next(g for g in sorted(state.groups) if g.endswith("beam_x__edge"))
            loose = gc.group_splices(state, beam_hoops={gid: 5.0})[gid]
            tight = gc.group_splices(state, beam_hoops={gid: 3.5})[gid]
        self.assertEqual(tight["splice_type"], "type_2_mechanical_18.2.7")
        self.assertEqual(tight["lap_alternative"]["splice_type"], "class_B_lap_outside_hinge_zones")
        self.assertTrue(tight["lap_enclosure"]["satisfied"])
        self.assertEqual(loose["splice_type"], "type_2_mechanical_18.2.7")
        self.assertFalse(loose["lap_enclosure"]["satisfied"])
        self.assertAlmostEqual(loose["lap_enclosure"]["required_spacing_in"], 4.0)
        self.assertIn("18.6.3.3", loose["lap_alternative"]["splice_rule"])

    def test_a_transition_coupler_gets_its_own_location_check(self):
        with gf.frame(FY_KSI=80.0):
            state = mg.install(gf.uniform_state())
            lower, upper = "s01_02__column__interior", "s03_04__column__interior"
            transitions = {f"{lower}>{upper}": {"path_counts": {"mechanically_spliced": 12}, "joints": ["joint_17"],
                                                "splice": {"center_half_clear_height_in": 78.0}}}
            result = gc.transition_splices(state, transitions)[f"{lower}>{upper}"]
            # 156 in clear, 26 in deep: Grade 80 couplers 52 in clear of each face -> [52, 104]
            self.assertEqual(result["placement"]["permitted_interval_in"], [52.0, 104.0])
            self.assertTrue(result["placement"]["legal_placement"])
            short = gc.transition_splices(state, {f"{lower}>{upper}": {**transitions[f"{lower}>{upper}"],
                                                                       "splice": {"center_half_clear_height_in": 55.0}}})
            self.assertFalse(short[f"{lower}>{upper}"]["placement"]["legal_placement"])     # [52, 58] < 8 db
            self.assertEqual(gc.transition_splices(state, {f"{lower}>{upper}": {"path_counts": {}}}), {})

    def test_gravity_groups_take_the_special_splice_rules_unless_within_strength(self):
        from Design.Gravity_Frame_Members import BRANCH_WITHIN_STRENGTH, BRANCH_YIELDING
        with gf.frame():
            state, _ = gr.seed_from_sections((26, 26, 6), (16, 30, 5), policy_id="system_three_story_v1")
            gravity = next(g for g in sorted(state.groups) if state.system_of(g) == mg.SYSTEM_GRAVITY)
            sfrs = next(g for g in sorted(state.groups) if state.system_of(g) == mg.SYSTEM_SFRS)
            self.assertTrue(gc.special_splice_rules(state, sfrs))
            self.assertTrue(gc.special_splice_rules(state, gravity))                        # branch unknown: 18.14.3.3
            self.assertTrue(gc.special_splice_rules(state, gravity, {gravity: BRANCH_YIELDING}))
            self.assertFalse(gc.special_splice_rules(state, gravity, {gravity: BRANCH_WITHIN_STRENGTH}))

    def test_uniform_splices_designed_requires_legal_placements(self):
        state = {"sections": {"h_col_in": 26.0, "b_col_in": 26.0, "h_beam_in": 28.0, "fc_beam_ksi": 4.0, "fc_col_ksi": 5.0},
                 "beam": {"bar_size": 8, "centroid_offset_in": 2.5}, "column": {"bar_size": 14},
                 "materials": {"fy_ksi": 80.0}, "geometry": {"bay_x_in": 100.0, "bay_y_in": 100.0, "story_h_in": 120.0}}
        result = cd.design_splices(state)
        self.assertFalse(result["all_designed"])
        self.assertEqual(result["beam"]["splice_type"], "type_2_mechanical_18.2.7")
        self.assertFalse(result["beam"]["placement"]["legal_placement"])


# ---- F06 ----------------------------------------------------------------------------------------------------------
def drift_screen(fail_story=None):
    """A saved drift screen as Design_Driver._drift_screen writes it: 2 stories, 2 x 1 bays, both directions."""
    nodes = [f"{i},{j}" for j in range(2) for i in range(3)]
    checks, rows = [], []
    for axis in ("x", "y"):
        stories = []
        for k in (1, 2):
            delta = 0.8 if k == fail_story else 0.2
            stories.append({"id": f"{k}/Q{axis}", "height_in": 168.0, "node_deltas_in": {n: {"x": delta, "y": delta} for n in nodes},
                            "expected_node_ids": nodes, "directions": [axis], "story_shear_kip": {axis: 100.0 * (3 - k)},
                            "gravity_above_kip": 800.0 * (3 - k)})
        result = evaluate_drift_and_stability(stories, importance_factor=1.0, risk_category="II", drift_limit_ratio=0.02,
                                              redundancy_factor=1.3, seismic_design_category="D", second_order_included=True)
        checks += result["checks"]
        rows += result["stories"]
    return {"checks": checks, "stories": rows,
            "assumptions": {"cd": 5.5, "rho_for_drift_load": 1.0, "rho_for_limit": 1.3, "risk_category": "II",
                            "importance_factor": 1.0, "base_drift_limit_ratio": 0.02, "effective_drift_limit_ratio": 0.02 / 1.3,
                            "limit_seismic_design_category": "D"}}


class SavedDriftReconstruction(unittest.TestCase):
    def rebuild(self, screen):
        return reconstruct_drift_checks(screen, 2, 168.0, 2, 1)

    def test_an_unchanged_screen_reproduces_exactly(self):
        screen = drift_screen(fail_story=1)
        result = self.rebuild(screen)
        self.assertEqual(result["integrity"]["status"], "pass", result["differences"])
        rebuilt = {(c["id"], c["location"]): (c["status"], c["demand"], c["capacity"]) for c in result["checks"]}
        saved = {(c["id"], c["location"]): (c["status"], c["demand"], c["capacity"]) for c in screen["checks"]}
        self.assertEqual(rebuilt, saved)
        self.assertIn("fail", {status for status, _d, _c in rebuilt.values()})

    def test_a_deleted_failed_check_is_found_and_the_failure_is_still_consumed(self):
        screen = drift_screen(fail_story=1)
        failed = next(c for c in screen["checks"] if c["status"] == "fail")
        screen["checks"].remove(failed)
        result = self.rebuild(screen)
        self.assertEqual(result["integrity"]["status"], "fail")
        self.assertIn((failed["id"], failed["location"], "fail"),
                      {(c["id"], c["location"], c["status"]) for c in result["checks"]})

    def test_an_edited_status_demand_or_limit_is_found(self):
        for edit in ("status", "demand", "limit"):
            with self.subTest(edit=edit):
                screen = drift_screen(fail_story=1)
                target = next(c for c in screen["checks"] if c["id"] == "demands.story_drift" and c["status"] == "fail")
                if edit == "status":
                    target["status"] = "pass"
                elif edit == "demand":
                    target["demand"] = 0.001
                else:
                    screen["assumptions"]["effective_drift_limit_ratio"] = 0.05
                result = self.rebuild(screen)
                self.assertEqual(result["integrity"]["status"], "fail")

    def test_an_omitted_direction_is_not_borrowed_from_the_other(self):
        screen = drift_screen()
        screen["stories"] = [r for r in screen["stories"] if not r["location"].endswith("/y")]
        screen["checks"] = [c for c in screen["checks"] if "/Qy" not in c["location"]]
        result = self.rebuild(screen)
        self.assertEqual(result["integrity"]["status"], "fail")
        y_rows = [c for c in result["checks"] if "/Qy" in c["location"]]
        self.assertTrue(y_rows)
        self.assertTrue(all(c["status"] == "not_evaluated" for c in y_rows))

    def test_a_self_consistent_limit_rewrite_inside_the_screen_is_refused_by_the_declared_basis(self):
        # The screen's assumptions are outside the input digest; the record's demand_basis.drift is inside it. Relaxing
        # rho (or the category) consistently inside the screen alone, labels included, must not erase a failure.
        screen = drift_screen(fail_story=1)
        declared = copy.deepcopy(screen["assumptions"])
        forged = copy.deepcopy(screen)
        forged["assumptions"].update(rho_for_limit=1.0, effective_drift_limit_ratio=0.02)
        for check in forged["checks"]:
            if check["id"] == "demands.story_drift":
                check["capacity"] = 0.02
                check["status"] = "pass" if check["demand"] <= 0.02 else "fail"
        for row in forged["stories"]:
            row["allowable_drift_ratio"] = 0.02
        result = reconstruct_drift_checks(forged, 2, 168.0, 2, 1, declared=declared, risk_category="II")
        self.assertEqual(result["integrity"]["status"], "fail")
        self.assertTrue(any("declared basis" in p for p in result["problems"]))
        self.assertFalse([c for c in result["checks"] if c["status"] == "pass" and c["id"] == "demands.story_drift"])
        # Without the declaration the same forgery is still caught by the screen's own fixed rho of 1.3.
        self.assertEqual(reconstruct_drift_checks(forged, 2, 168.0, 2, 1)["integrity"]["status"], "fail")

    def test_the_declared_basis_and_the_record_risk_category_must_agree(self):
        screen = drift_screen()
        declared = copy.deepcopy(screen["assumptions"])
        self.assertEqual(reconstruct_drift_checks(screen, 2, 168.0, 2, 1, declared=declared, risk_category="II")
                         ["integrity"]["status"], "pass")
        self.assertEqual(reconstruct_drift_checks(screen, 2, 168.0, 2, 1, declared=declared, risk_category="III")
                         ["integrity"]["status"], "fail")
        other = {**declared, "limit_seismic_design_category": "E"}
        self.assertEqual(reconstruct_drift_checks(screen, 2, 168.0, 2, 1, declared=other)["integrity"]["status"], "fail")

    def test_a_deleted_drift_row_with_its_labels_kept_is_a_difference(self):
        screen = drift_screen(fail_story=1)
        screen["stories"] = [r for r in screen["stories"] if r["location"] != "story:1/Qx/x"]
        result = self.rebuild(screen)
        self.assertEqual(result["integrity"]["status"], "fail")
        self.assertTrue(any("without its saved drift row" in d for d in result["differences"]))

    def test_a_screen_saved_before_the_risk_basis_is_unknown_not_tampered(self):
        screen = drift_screen()
        screen["assumptions"] = {"cd": 5.5, "rho_for_drift_load": 1.0, "rho_for_limit": 1.3}
        result = self.rebuild(screen)
        self.assertEqual(result["integrity"]["status"], "not_evaluated")
        self.assertEqual(result["integrity"]["details"]["reason_code"], "legacy_drift_schema")
        self.assertTrue(all(c["status"] == "not_evaluated" for c in result["checks"]))

    def test_a_row_saved_before_its_inputs_were_kept_rebuilds_drift_but_not_stability(self):
        screen = drift_screen()
        for row in screen["stories"]:
            for key in ("node_drifts_in", "expected_node_ids", "story_shear_kip", "gravity_above_kip", "height_in"):
                row.pop(key, None)
        result = self.rebuild(screen)
        by_id = {}
        for c in result["checks"]:
            by_id.setdefault(c["id"], set()).add(c["status"])
        self.assertEqual(by_id["demands.story_drift"], {"pass"})
        self.assertEqual(by_id["demands.stability"], {"not_evaluated"})
        self.assertEqual(result["integrity"]["status"], "pass")                     # not tampered, only not rebuildable
        self.assertTrue(result["not_rebuildable"])


# ---- F07 ----------------------------------------------------------------------------------------------------------
class GroupedHoopIdentity(unittest.TestCase):
    def setUp(self):
        self.stack = gf.frame()
        self.stack.__enter__()
        self.state = mg.install(gf.uniform_state())
        self.columns = sorted(g for g in self.state.groups if self.state.designs[g].is_column)
        self.beams = sorted(g for g in self.state.groups if not self.state.designs[g].is_column)

    def tearDown(self):
        self.stack.__exit__(None, None, None)

    def hoops(self, gid, spacing=None):
        d = self.state.designs[gid]
        if d.is_column:
            return {"bar_size": d.stirrup_bar_size, "legs_model": d.stirrup_legs, "spacing_in": spacing or d.stirrup_spacing_in,
                    "legs": dict(d.legs_by_direction)}
        return {"bar_size": d.stirrup_bar_size, "legs": d.stirrup_legs, "spacing_in": spacing or d.stirrup_spacing_in}

    def identity(self, transverse):
        return gr.hoop_identity_checks({"transverse": transverse}, self.state)

    def test_all_selected_and_matching(self):
        result = self.identity({g: self.hoops(g) for g in self.columns + self.beams})
        status = {c["id"]: c["status"] for c in result["checks"]}
        self.assertEqual(status, {"qualification.hoop_selection_complete": "pass", "qualification.hoops_match_design": "pass"})
        self.assertTrue(result["capacity_usable"])

    def test_a_missing_cage_is_incomplete_and_unknown_and_keeps_the_known_failures_usable(self):
        transverse = {g: self.hoops(g) for g in self.columns + self.beams}
        transverse[self.columns[0]] = None
        result = self.identity(transverse)
        status = {c["id"]: c["status"] for c in result["checks"]}
        self.assertEqual(status, {"qualification.hoop_selection_complete": "fail",
                                  "qualification.hoops_match_design": "not_evaluated"})
        self.assertEqual(result["states"][self.columns[0]], "not_selected")
        self.assertTrue(result["capacity_usable"])                  # the recomputed capacity failures are still read

    def test_an_actual_mismatch_beside_a_missing_cage_fails_the_identity(self):
        transverse = {g: self.hoops(g) for g in self.columns + self.beams}
        transverse[self.columns[0]] = None
        transverse[self.beams[0]] = self.hoops(self.beams[0], spacing=3.0)
        result = self.identity(transverse)
        status = {c["id"]: c["status"] for c in result["checks"]}
        self.assertEqual(status["qualification.hoops_match_design"], "fail")
        self.assertEqual(result["states"][self.beams[0]], "installed_mismatch")
        self.assertFalse(result["capacity_usable"])


# ---- F08 ----------------------------------------------------------------------------------------------------------
class QualificationScope(unittest.TestCase):
    def test_uniform_scope_names_the_diaphragm_check_it_runs(self):
        scope = qualification_scope()
        self.assertIn("every grid line", scope["system"])
        self.assertIn("SMRF_Diaphragm", scope["diaphragm"])

    def test_grouped_perimeter_scope_does_not_borrow_the_all_grid_line_premise(self):
        scope = gr.grouped_scope({"member_groups": {"policy": {"id": "system_three_story_v1"}}})
        self.assertIn("perimeter", scope["system"])
        self.assertIn("gravity frames", scope["system"])
        self.assertIn("NOT evaluated", scope["diaphragm"])
        check = gr._diaphragm_check({"member_groups": {"policy": {"id": "system_three_story_v1"}}})
        self.assertEqual(check["status"], "not_evaluated")
        self.assertIn("collectors", check["details"]["reason"])

    def test_the_grouped_regularity_text_names_the_frames_of_the_installed_policy(self):
        from Design.Grouped_Design import grouped_regularity
        with gf.frame():
            system, _ = gr.seed_from_sections((26, 26, 6), (16, 30, 5), policy_id="system_three_story_v1")
            self.assertIn("perimeter", grouped_regularity({}, {}, system)["horizontal"])
            self.assertNotIn("every grid line", grouped_regularity({}, {}, system)["horizontal"])
            self.assertIn("every grid line", grouped_regularity({}, {}, gf.uniform_state())["horizontal"])

    def test_grouped_band_scope_is_every_grid_line_and_still_has_no_diaphragm_check(self):
        scope = gr.grouped_scope({"member_groups": {"policy": {"id": mg.POLICY_ID}}})
        self.assertIn("every grid line", scope["system"])
        self.assertIn("NOT evaluated", scope["diaphragm"])
        self.assertEqual(gr._diaphragm_check({"member_groups": {"policy": {"id": mg.POLICY_ID}}})["status"], "not_evaluated")


if __name__ == "__main__":
    unittest.main()
