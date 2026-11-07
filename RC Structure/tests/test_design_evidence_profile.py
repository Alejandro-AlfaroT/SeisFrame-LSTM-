"""The per-run design-evidence profile (Design.Design_Evidence_Profile, 2026-11-01).

  L  loading: the unreviewed example declares numerical refinement with every assertion false; the test fixture is
     refused unless a test allows it; unknown or missing keys, non-boolean flags, NaN, undated or blank provenance, a
     malformed plan, a plan without its basis are refused; assertions need a matching review, a plan and the covered
     floor sections, never a synthesized approver; review evidence is hashed on load; PROBE assertions are refused
     outside a test fixture;
  I  identity: every declared field changes the profile identity, and the configuration it implies enters the
     evaluation identity;
  R  the runner: the profile enters the request, the basis and the evaluation identity before any evaluation; a scope
     that differs from the run, and a test fixture, are refused; without a profile the unreviewed defaults are kept;
  S  scope: a candidate whose floor sections the review does not cover is evaluated unasserted;
  2  review round 2: the slab ladder does not step past a reviewed thickness; a partial review is passed to the slab
     evidence of in-scope candidates (and the defaults to the others) and is not effective; a malformed explicit offset
     is a profile error; the PROBE stamp is the word, not a substring; a plan whose shell budget does not resolve on the
     seed's floors is refused;
  3  review round 3: a profile without a plan says single-mesh; a PROBE stamp in any review block is refused, asserted
     or not; a reviewed thickness off the declared ladder is refused.

No solve: the runner is prepared (plan only) and the slab stage runs on patched solvers.
"""
import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402

import Structure_Parameters as sp                                               # noqa: E402
from Design import Design_Driver as driver                                       # noqa: E402
from Design import Design_Evidence_Profile as dep                                # noqa: E402
from Design import Grouped_Design as gd                                          # noqa: E402
from Design import SMRF_Floor_Sections as floor_sections                         # noqa: E402
from Design.Config import DesignConfig, SlabActionAssertions                     # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402

RC = Path(__file__).resolve().parents[1]
EXAMPLE = RC / "tools/genetic_design_search/evidence_profiles/case_0007_slab_refinement_unreviewed.json"
FIXTURE = RC / "tests/fixtures/design_evidence_test_fixture.json"
FIXTURE_EVIDENCE = RC / "tests/fixtures/design_evidence_fixture_review.txt"
CASE_0007 = {"case_id": "case_0007", "label": "lhs_007", "num_bay_x": 3, "num_bay_y": 3, "num_floor": 6,
             "bay_x_width_ft": 23.5, "bay_y_width_ft": 28.0, "story_height_ft": 15.5, "seismic_site": "sdc_d_high",
             "geometry_name": "lhs_bx3_by3_s6_sh15p5_bwx23p5_bwy28_sdc_d_high",
             "purpose": "Latin hypercube point 7 of 10, seed 20261017."}


def data(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, value, name="profile.json", raw=None):
        path = self.tmp / name
        path.write_text(raw if raw is not None else json.dumps(value), encoding="utf-8")
        return path

    def fixture_copy(self, mutate=None, kind="test_fixture"):
        """The test fixture with its evidence file beside it in the temporary folder, optionally changed."""
        shutil.copyfile(FIXTURE_EVIDENCE, self.tmp / FIXTURE_EVIDENCE.name)
        value = data(FIXTURE)
        value["kind"] = kind
        if mutate:
            mutate(value)
        return self.write(value)


# ---- L ---------------------------------------------------------------------------------------------------------------
class Loading(TempDir):
    def test_the_unreviewed_example_declares_numerical_refinement_with_every_assertion_false(self):
        profile = dep.load(EXAMPLE)
        self.assertFalse(profile.asserted)
        cfg = profile.apply(DesignConfig())
        self.assertEqual(cfg.floor_analysis.slab_refinement, data(EXAMPLE)["slab_refinement"])
        self.assertEqual(cfg.slab_actions, SlabActionAssertions())
        self.assertIsNone(cfg.floor_analysis.slab_assertion_floor_sections)
        self.assertFalse(cfg.slab_actions.all_asserted())
        self.assertIn("every engineering assertion false", profile.summary()["effect"])

    def test_the_test_fixture_is_refused_unless_a_test_allows_it(self):
        with self.assertRaisesRegex(dep.EvidenceProfileError, "test-fixture"):
            dep.load(FIXTURE)
        profile = dep.load(FIXTURE, allow_test_fixture=True)
        self.assertTrue(profile.asserted)
        cfg = profile.apply(DesignConfig())
        self.assertTrue(cfg.slab_actions.all_asserted())
        self.assertEqual(cfg.floor_analysis.slab_assertion_floor_sections, ("a" * 64,))

    def test_unknown_or_missing_keys_are_refused_at_every_level(self):
        base = data(EXAMPLE)
        mutations = [lambda d: d.update(extra=1), lambda d: d.pop("review"),
                     lambda d: d["provenance"].update(approved=True), lambda d: d["scope"]["geometry"].pop("story_h_in"),
                     lambda d: d["scope"]["loads"].update(roof_live_load_ksf=0.02),
                     lambda d: d["slab_action_assertions"].update(asserted=True),
                     lambda d: d["slab_action_assertions"].pop("verified")]
        for index, mutate in enumerate(mutations):
            value = copy.deepcopy(base)
            mutate(value)
            with self.subTest(index=index), self.assertRaises(dep.EvidenceProfileError):
                dep.load(self.write(value))

    def test_a_flag_must_be_a_json_boolean_and_a_number_finite(self):
        for flag in ("true", 1, None):
            value = data(EXAMPLE)
            value["slab_action_assertions"]["verified"] = flag
            with self.subTest(flag=flag), self.assertRaisesRegex(dep.EvidenceProfileError, "JSON boolean"):
                dep.load(self.write(value))
        text = EXAMPLE.read_text(encoding="utf-8").replace('"story_h_in": 186.0', '"story_h_in": NaN')
        with self.assertRaises(dep.EvidenceProfileError):
            dep.load(self.write(None, raw=text))

    def test_provenance_must_be_named_and_dated(self):
        for key, bad in (("prepared_by", " "), ("prepared_date", "11/01/2026"), ("purpose", "")):
            value = data(EXAMPLE)
            value["provenance"][key] = bad
            with self.subTest(key=key), self.assertRaises(dep.EvidenceProfileError):
                dep.load(self.write(value))

    def test_a_plan_is_validated_without_solving_and_needs_its_basis(self):
        value = data(EXAMPLE)
        value["slab_refinement"]["moment_tolerance"] = 1.5
        with self.assertRaisesRegex(dep.EvidenceProfileError, "slab_refinement"):
            dep.load(self.write(value))
        value = data(EXAMPLE)
        value["slab_refinement_basis"] = None
        with self.assertRaises(dep.EvidenceProfileError):
            dep.load(self.write(value))
        value = data(EXAMPLE)
        value["slab_refinement"] = None
        with self.assertRaisesRegex(dep.EvidenceProfileError, "without a slab_refinement"):
            dep.load(self.write(value))

    def test_assertions_need_a_matching_review_a_plan_and_covered_floors_never_a_synthesized_approver(self):
        cases = {
            "reviewed without review": (lambda d: d.update(review=None), "reviewed"),
            "assertion provenance not the review's": (lambda d: d["slab_action_assertions"].update(asserted_by="someone else"),
                                                      "test_fixture"),
            "asserted without a plan": (lambda d: d.update(slab_refinement=None, slab_refinement_basis=None), "test_fixture"),
            "asserted without covered floors": (lambda d: d["scope"].update(floor_sections_sha256=None), "test_fixture"),
            "asserted while unreviewed": (lambda d: d.update(review_status="under_review"), "test_fixture"),
            "an example asserting": (lambda d: d.update(review_status="unreviewed"), "unreviewed_example"),
        }
        for name, (mutate, kind) in cases.items():
            with self.subTest(name=name), self.assertRaises(dep.EvidenceProfileError):
                dep.load(self.fixture_copy(mutate, kind=kind), allow_test_fixture=True)
        orphan = data(EXAMPLE)
        orphan["slab_action_assertions"]["asserted_by"] = "a reviewer"
        with self.assertRaisesRegex(dep.EvidenceProfileError, "without asserting"):
            dep.load(self.write(orphan))

    def test_review_evidence_is_hashed_on_load(self):
        path = self.fixture_copy()
        dep.load(path, allow_test_fixture=True)
        (self.tmp / FIXTURE_EVIDENCE.name).write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(dep.EvidenceProfileError, "differs from its recorded sha256"):
            dep.load(path, allow_test_fixture=True)
        (self.tmp / FIXTURE_EVIDENCE.name).unlink()
        with self.assertRaisesRegex(dep.EvidenceProfileError, "does not exist"):
            dep.load(path, allow_test_fixture=True)

    def test_probe_assertions_are_refused_outside_a_test_fixture(self):
        def probe(d):
            for key in ("asserted_by",):
                d["slab_action_assertions"][key] = "PROBE"
            d["review"]["reviewed_by"] = "PROBE"
        with self.assertRaisesRegex(dep.EvidenceProfileError, "PROBE"):
            dep.load(self.fixture_copy(probe, kind="reviewed"))


class ReviewRoundOne(TempDir):
    def test_repeated_keys_a_mistyped_plan_and_an_asserted_profile_without_its_thickness_are_refused(self):
        text = EXAMPLE.read_text(encoding="utf-8").replace('"kind": "unreviewed_example",',
                                                           '"kind": "reviewed", "kind": "unreviewed_example",')
        with self.assertRaisesRegex(dep.EvidenceProfileError, "repeated"):
            dep.load(self.write(None, raw=text))
        for change in ({"max_shells": True}, {"max_shells": 3}, {"levels": 99}, {"levels": 4.0}, {"recipe": "nonexistent"}):
            value = data(EXAMPLE)
            value["slab_refinement"].pop("local_extension")
            value["slab_refinement"].update(change)
            with self.subTest(change=change), self.assertRaises(dep.EvidenceProfileError):
                dep.load(self.write(value))
        with self.assertRaisesRegex(dep.EvidenceProfileError, "slab_thickness_in"):
            dep.load(self.fixture_copy(lambda d: d["scope"].update(slab_thickness_in=None)), allow_test_fixture=True)

    def test_the_reviewed_thickness_bounds_the_assertion_scope(self):
        profile = dep.load(FIXTURE, allow_test_fixture=True)
        with gf.frame():
            with mg.installed(gf.changed_state()):
                description = floor_sections.by_floor()
                cfg = profile.apply(DesignConfig())
                cfg.floor_analysis.slab_assertion_floor_sections = tuple(
                    sha for sha, _sections, _served in floor_sections.distinct_floors(description))
                self.assertEqual(cfg.floor_analysis.slab_assertion_thickness_in, 7.0)
                self.assertTrue(gd.slab_assertion_scope(cfg, description)["effective"])       # the fixture slab is 7 in
                sp.SLAB_THICKNESS_IN = 7.5
                scope = gd.slab_assertion_scope(cfg, description)
                self.assertEqual((scope["floors_in_scope"], scope["thickness_in_scope"], scope["effective"]),
                                 (True, False, False))


# ---- I ---------------------------------------------------------------------------------------------------------------
class Identity(TempDir):
    def test_every_declared_field_changes_the_identity(self):
        base = dep.load(EXAMPLE).identity()
        changes = [lambda d: d["slab_refinement"].update(moment_tolerance=0.04),
                   lambda d: d["provenance"].update(purpose="another purpose"),
                   lambda d: d["scope"]["loads"].update(floor_live_load_ksf=0.08),
                   lambda d: d.update(review_status="under_review"),
                   lambda d: d.update(slab_refinement_basis=d["slab_refinement_basis"] + " Revised.")]
        seen = {base}
        for index, change in enumerate(changes):
            value = data(EXAMPLE)
            change(value)
            identity = dep.load(self.write(value, f"p{index}.json")).identity()
            self.assertNotIn(identity, seen)
            seen.add(identity)

    def test_the_configuration_it_implies_enters_the_evaluation_identity(self):
        with gf.frame():
            plain = driver.design_request_identity(DesignConfig())["sha256"]
            profiled = driver.design_request_identity(dep.load(EXAMPLE).apply(DesignConfig()))["sha256"]
            asserted = driver.design_request_identity(dep.load(FIXTURE, allow_test_fixture=True).apply(DesignConfig()))["sha256"]
        self.assertEqual(len({plain, profiled, asserted}), 3)


# ---- R ---------------------------------------------------------------------------------------------------------------
class Runner(TempDir):
    def prepare(self, profile=None):
        import importlib.util
        spec = importlib.util.spec_from_file_location("evidence_runner", RC / "tools/genetic_design_search/run.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        case = self.write(CASE_0007, "case_0007.json")
        args = ["--engine", "generation_v3", "--case-file", str(case), "--profile", "v2_nonlinear_flexure_screening_v1",
                "--grouping", "system_three_story_v1", "--seed-column", "30", "30", "6", "--seed-beam", "16", "30", "5"]
        if profile is not None:
            args += ["--design-evidence", str(profile)]
        snapshot = dict(vars(sp))
        try:
            return runner.prepare(runner.parser().parse_args(args))
        finally:
            mg.clear()
            for key in set(vars(sp)) - set(snapshot):
                delattr(sp, key)
            vars(sp).update(snapshot)

    def scoped_example(self, mutate=None):
        value = data(EXAMPLE)
        value["scope"]["case_sha256"] = hashlib.sha256(json.dumps(CASE_0007).encode("utf-8")).hexdigest()
        if mutate:
            mutate(value)
        return self.write(value, "profile.json")

    def test_the_profile_enters_the_request_the_basis_and_the_evaluation_identity_before_any_evaluation(self):
        _state, cfg, _config, request = self.prepare(self.scoped_example())
        profile = dep.load(self.tmp / "profile.json")
        self.assertEqual(request["design_evidence"]["identity_sha256"], profile.identity())
        self.assertEqual(request["basis"]["design_evidence"]["identity_sha256"], profile.identity())
        self.assertFalse(request["basis"]["design_evidence"]["asserted"])
        policy = request["evaluation_identity"]["policy"]
        self.assertEqual(policy["floor_analysis"]["slab_refinement"], data(EXAMPLE)["slab_refinement"])
        self.assertFalse(any(v for k, v in policy["slab_actions"].items() if isinstance(v, bool)))
        self.assertEqual(cfg.floor_analysis.slab_refinement, data(EXAMPLE)["slab_refinement"])
        _s, plain_cfg, _c, plain = self.prepare()
        self.assertIsNone(plain["design_evidence"]["profile_id"])
        self.assertIsNone(plain["basis"]["design_evidence"]["identity_sha256"])
        self.assertIsNone(plain_cfg.floor_analysis.slab_refinement)
        self.assertNotEqual(plain["sha256"], request["sha256"])
        self.assertNotEqual(plain["evaluation_identity"]["sha256"], request["evaluation_identity"]["sha256"])
        self.assertNotEqual(plain["basis"], request["basis"])

    def test_a_slab_policy_that_differs_or_a_plan_that_does_not_resolve_on_the_seed_floors_is_refused(self):
        with self.assertRaisesRegex(ValueError, "live_load_patterning"):
            self.prepare(self.scoped_example(lambda d: d["scope"]["slab_policy"].update(live_load_patterning=False)))

        def explicit(d):
            d["slab_refinement"] = {"meshes": [{"x_offsets_in": [0.0, 10.0, 20.0], "y_offsets_in": [0.0, 10.0, 20.0],
                                                "max_shells": 4000}] * 2, "moment_tolerance": 0.05,
                                    "shear_tolerance": 0.05, "tolerance_basis": "a plan that does not span the bays"}
        with self.assertRaisesRegex(ValueError, "does not resolve on the seed's floors"):
            self.prepare(self.scoped_example(explicit))

    def test_a_scope_that_differs_from_the_run_or_a_test_fixture_is_refused(self):
        with self.assertRaisesRegex(ValueError, "does not cover this run"):
            self.prepare(self.scoped_example(lambda d: d["scope"]["geometry"].update(num_floor=5)))
        with self.assertRaisesRegex(ValueError, "case_sha256"):
            self.prepare(EXAMPLE)                    # the example names the case file of the research run
        with self.assertRaisesRegex(dep.EvidenceProfileError, "test-fixture"):
            self.prepare(FIXTURE)

    def test_a_plan_whose_shell_budget_does_not_resolve_on_the_seed_floors_is_refused(self):
        def small(d):
            d["slab_refinement"].pop("local_extension")
            d["slab_refinement"]["max_shells"] = 1000
        with self.assertRaisesRegex(ValueError, "unresolved_budget"):
            self.prepare(self.scoped_example(small))
        _state, _cfg, _config, request = self.prepare(self.scoped_example())
        rows = request["design_evidence"]["seed_plan_resolution"]
        self.assertTrue(rows)
        self.assertTrue(all(row["status"] == "resolved" and len(row["levels"]) == 4 for row in rows))


# ---- S ---------------------------------------------------------------------------------------------------------------
class AssertionScope(unittest.TestCase):
    def test_a_candidate_whose_floor_sections_the_review_does_not_cover_is_evaluated_unasserted(self):
        profile = dep.load(FIXTURE, allow_test_fixture=True)
        with gf.frame():
            with mg.installed(gf.changed_state()):
                description = floor_sections.by_floor()
                floors = sorted(sha for sha, _sections, _served in floor_sections.distinct_floors(description))
                cfg = profile.apply(DesignConfig())
                cfg.floor_analysis.slab_evidence_basis = "engineering_assertions_v0"     # the basis under test
                scope = gd.slab_assertion_scope(cfg, description)
                self.assertEqual((scope["asserted"], scope["in_scope"], scope["effective"]), (True, False, False))
                self.assertFalse(gd._effective_assertions(cfg, description).all_asserted())
                self.assertFalse(gd._two_way_shear_path_assessed(cfg))
                covered = profile.apply(DesignConfig())
                covered.floor_analysis.slab_assertion_floor_sections = tuple(floors)
                scope = gd.slab_assertion_scope(covered, description)
                self.assertTrue(scope["effective"])
                self.assertTrue(gd._effective_assertions(covered, description).all_asserted())
                unrestricted = DesignConfig()
                self.assertEqual(gd.slab_assertion_scope(unrestricted, description)["in_scope"], True)
                self.assertFalse(gd.slab_assertion_scope(unrestricted, description)["effective"])


# ---- 2 ---------------------------------------------------------------------------------------------------------------
class ReviewRoundTwo(TempDir):
    @staticmethod
    def covered(cfg, description):
        cfg.floor_analysis.slab_assertion_floor_sections = tuple(
            sha for sha, _sections, _served in floor_sections.distinct_floors(description))
        return cfg

    def run_ladder(self, bound):
        """slab_stage on patched solvers: refined evidence accepted wherever the assertions apply, and no admissible
        layout at any thickness. Returns (CandidateInfeasible, calls, thickness in force)."""
        from unittest import mock
        calls = []

        def choose(geometry, description, policy):
            return {"thickness_in": max(7.0, policy["minimum_thickness_in"]), "required_thickness_in": 7.0,
                    "governing_floors": [1], "superimposed_dead_load_ksf": 0.05, "live_load_mass_fraction": 0.0,
                    "thickness_screen_passed": True}

        def transfer(*_args, **_kwargs):
            calls.append(("floor_transfer", sp.SLAB_THICKNESS_IN))
            return {"transfers": [1], "reused_signatures": []}

        def evidence(cfg, slab, description, inputs, context):
            asserted = gd._effective_assertions(cfg, description).all_asserted()
            calls.append(("refined_evidence", slab["thickness_in"], asserted))
            return {"refinement": {"status": "passed"}, "asserted": asserted}

        def design(inputs, evidence, _existing, _context):
            if evidence["asserted"]:
                return {"layout": None, "checks": [{"id": "slab_flexure", "status": "fail",
                                                    "details": {"reason": "no admissible layout"}}]}
            return {"layout": None, "checks": [{"id": "slab_verified_strip_action_inputs", "status": "not_evaluated",
                                                "details": {"reason": "not asserted"}}]}

        profile = dep.load(FIXTURE, allow_test_fixture=True)
        with gf.frame():
            with mg.installed(gf.changed_state()):
                cfg = self.covered(profile.apply(DesignConfig()), floor_sections.by_floor())
                cfg.floor_analysis.slab_assertion_thickness_in = bound
                with mock.patch("Design.SMRF_Slab.choose_slab", choose), \
                        mock.patch("Design.SMRF_Floor_Transfer.build_floor_transfers_by_floor", transfer), \
                        mock.patch.object(gd, "_slab_evidence", evidence), \
                        mock.patch("Design.SMRF_Slab_Reinforcement.design_slab_reinforcement", design), \
                        mock.patch.object(gd.driver, "_slab_strength_inputs_from_state", lambda s, c: {}), \
                        mock.patch.object(gd, "slab_completion_context", lambda s, c, st=None: {}), \
                        self.assertRaises(gd.CandidateInfeasible) as caught:
                    gd.slab_stage(cfg, {})
                return caught.exception, calls, sp.SLAB_THICKNESS_IN

    def test_the_slab_ladder_does_not_step_past_the_reviewed_thickness(self):
        error, calls, thickness = self.run_ladder(7.0)
        self.assertEqual(error.stage, "slab_reinforcement")
        self.assertIn("reviewed slab thickness", error.reason)
        self.assertEqual(calls, [("floor_transfer", 7.0), ("refined_evidence", 7.0, True)])
        self.assertEqual(thickness, 7.0)
        self.assertEqual([(a["thickness_in"], a["status"]) for a in error.evidence["attempts"]],
                         [(7.0, "no_admissible_layout")])
        # Without a reviewed thickness the earlier rule holds: the ladder steps through the declared range.
        error, calls, _thickness = self.run_ladder(None)
        self.assertIn("any thickness of the declared range", error.reason)
        self.assertGreater(len(error.evidence["attempts"]), 1)
        self.assertTrue(all(call[2] for call in calls if call[0] == "refined_evidence"))

    def test_a_partial_review_reaches_the_slab_evidence_of_in_scope_candidates_and_is_not_effective(self):
        from unittest import mock
        profile = dep.load(self.fixture_copy(lambda d: d["slab_action_assertions"].update(verified=False)),
                           allow_test_fixture=True)
        self.assertTrue(profile.asserted)
        self.assertIn("partially asserted", profile.summary()["effect"])
        self.assertNotIn("partially", dep.load(FIXTURE, allow_test_fixture=True).summary()["effect"])
        passed = []

        def builder(*_args, assertions=None, **_kwargs):
            passed.append(assertions)
            return {"refinement": {"status": "passed"}}

        slab = {"thickness_in": 7.0, "concrete_fc_ksi": 5.0, "concrete_unit_weight_kcf": 0.15,
                "superimposed_dead_load_ksf": 0.05}
        with gf.frame():
            with mg.installed(gf.changed_state()):
                description = floor_sections.by_floor()
                cfg = self.covered(profile.apply(DesignConfig()), description)
                cfg.floor_analysis.slab_evidence_basis = "engineering_assertions_v0"     # the basis under test
                scope = gd.slab_assertion_scope(cfg, description)
                self.assertEqual((scope["in_scope"], scope["effective"]), (True, False))
                self.assertFalse(gd._two_way_shear_path_assessed(cfg))
                with mock.patch("Design.SMRF_Slab_Refinement.build_refined_slab_action_evidence", builder), \
                        mock.patch("Design.SMRF_Slab_Refinement.combine_floor_evidence", lambda floors, n: floors):
                    gd._slab_evidence(cfg, slab, description, {}, {})
                    self.assertTrue(passed)
                    # the reviewed flags and provenance reach the evidence builder of every in-scope floor
                    self.assertTrue(all(a == asdict(cfg.slab_actions) for a in passed))
                    self.assertFalse(SlabActionAssertions(**passed[0]).all_asserted())     # so no layout is selected
                    passed.clear()
                    cfg.floor_analysis.slab_assertion_floor_sections = ("b" * 64,)
                    gd._slab_evidence(cfg, slab, description, {}, {})
                    self.assertTrue(passed)
                    self.assertTrue(all(a == asdict(SlabActionAssertions()) for a in passed))   # out of scope: none

    def test_a_malformed_explicit_offset_is_a_profile_error(self):
        good = [0.0, 10.0, 20.0]
        for bad in (5, True, 5.0, {"a": 1}, [], "abc"):
            value = data(EXAMPLE)
            value["slab_refinement"] = {"meshes": [{"x_offsets_in": bad, "y_offsets_in": good, "max_shells": 4000},
                                                   {"x_offsets_in": [0.0, 5.0, 10.0, 20.0], "y_offsets_in": good,
                                                    "max_shells": 4000}],
                                        "moment_tolerance": 0.05, "shear_tolerance": 0.05,
                                        "tolerance_basis": "a test plan"}
            with self.subTest(bad=bad), self.assertRaises(dep.EvidenceProfileError):
                dep.load(self.write(value))

    def test_the_probe_stamp_is_the_word_not_a_substring(self):
        def named(d):
            d["review"]["reviewed_by"] = d["slab_action_assertions"]["asserted_by"] = "A. Probert"
            d["review"]["basis"] = "probed the nested meshes and the strip peaks"
        self.assertFalse(dep._probe_stamp("A. Probert", "probed the nested meshes and the strip peaks"))
        self.assertTrue(dep.load(self.fixture_copy(named), allow_test_fixture=True).asserted)
        # Its remaining software-fixture warnings still prohibit production loading.
        with self.assertRaisesRegex(dep.EvidenceProfileError, "TEST FIXTURE"):
            dep.load(self.fixture_copy(named, kind="reviewed"))
        with self.assertRaisesRegex(dep.EvidenceProfileError, "PROBE"):
            dep.load(self.fixture_copy(lambda d: d["review"].update(
                basis="PROBE -- design verification run, not a certification"), kind="reviewed"))
        from Design.Config import PROBE_SLAB_REFINEMENT
        value = data(EXAMPLE)
        value["slab_refinement"] = dict(PROBE_SLAB_REFINEMENT, tolerance_basis="the inherited screen")
        value["slab_refinement_basis"] = "probed the nested meshes; tolerance 5 percent"
        with self.assertRaisesRegex(dep.EvidenceProfileError, "investigation"):
            dep.load(self.write(value))
        value["slab_refinement_basis"] += ", an investigation criterion"
        dep.load(self.write(value))


# ---- 3 ---------------------------------------------------------------------------------------------------------------
class ReviewRoundThree(TempDir):
    def test_a_profile_without_a_plan_says_single_mesh(self):
        value = data(EXAMPLE)
        value["slab_refinement"] = value["slab_refinement_basis"] = None
        effect = dep.load(self.write(value)).summary()["effect"]
        self.assertIn("single-mesh slab actions (no refinement plan)", effect)
        self.assertIn("numerical refinement of the slab actions", dep.load(EXAMPLE).summary()["effect"])

    def test_a_probe_stamp_in_any_review_block_is_refused_asserted_or_not(self):
        stamp = "PROBE -- design verification run, not a certification"

        def unasserted(d):
            d["slab_action_assertions"].update({flag: False for flag in dep.ASSERTION_FLAGS})
            d["slab_action_assertions"].update(asserted_by="", assertion_date="", assertion_basis="")
            d["review"]["reviewed_by"] = stamp
        for mutate in (unasserted, lambda d: d["review"].update(decision=stamp),
                       lambda d: d["review"]["evidence"][0].update(description=stamp)):
            with self.subTest(mutate=mutate), self.assertRaisesRegex(dep.EvidenceProfileError, "PROBE"):
                dep.load(self.fixture_copy(mutate, kind="reviewed"))
        # Changing only kind cannot turn the explicit software-fixture warnings into engineering evidence.
        with self.assertRaisesRegex(dep.EvidenceProfileError, "TEST FIXTURE"):
            dep.load(self.fixture_copy(kind="reviewed"))

    def test_a_reviewed_thickness_off_the_declared_ladder_is_refused(self):
        for thickness in (7.25, 20.0, 4.0):
            with self.subTest(thickness=thickness), self.assertRaisesRegex(dep.EvidenceProfileError, "ladder"):
                dep.load(self.fixture_copy(lambda d, t=thickness: d["scope"].update(slab_thickness_in=t)),
                         allow_test_fixture=True)
        self.assertEqual(dep.load(FIXTURE, allow_test_fixture=True).data["scope"]["slab_thickness_in"], 7.0)


if __name__ == "__main__":
    unittest.main()
