"""The procedure-based slab evidence path, P1 (Design.SMRF_Slab_Procedure; profile schema v2), 2026-11-01.

  L  loading: a v2 example loads inactive_unreviewed and asserts nothing; a changed method file makes it
     inactive_method_changed; a v2 profile keeps the domain instead of the floor list and the single thickness, and
     declares its plan; a reviewed procedure refuses TEST FIXTURE and PROBE text; an asserted flag whose dispositions
     are not all accepted is refused; the v1 example keeps its identity;
  A  apply: the procedure is carried under every status, the assertions only under an active one;
  D  the domain: a floor's beams, grade and supports, and the slab thickness on the ladder; malformed input is outside;
  I  instance evidence: the load-case inventory recomputed from the slab's own loads; the checks pass on complete
     evidence and fail on a missing piece; masking keeps the numbers; an edited record fails the replay;
  S  the slab stage: the scope fails closed under an inactive procedure and outside the domain; a plan that does not
     resolve on a floor is recorded as that floor's unverified evidence, not raised; each floor's instance is kept and
     the flags of a floor that fails are masked; the uniform route refuses a procedure.

No solve: the slab builders are patched.
"""
import copy
import hashlib
import json
import re
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402

import Structure_Parameters as sp                                               # noqa: E402
from Design import Design_Driver as driver                                       # noqa: E402
from Design import Design_Evidence_Profile as dep                                # noqa: E402
from Design import Grouped_Design as gd                                          # noqa: E402
from Design import SMRF_Floor_Sections as floor_sections                         # noqa: E402
from Design import SMRF_Slab_Procedure as proc                                   # noqa: E402
from Design.Config import DesignConfig, SlabActionAssertions                     # noqa: E402
from Design.SMRF_Slab_Actions import _FLAGS                                      # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402

RC = Path(__file__).resolve().parents[1]
EXAMPLE = RC / "tools/genetic_design_search/evidence_profiles/case_0007_slab_refinement_unreviewed.json"
FIXTURE = RC / "tests/fixtures/design_evidence_test_fixture.json"
FIXTURE_EVIDENCE = RC / "tests/fixtures/design_evidence_fixture_review.txt"
SLAB = {"thickness_in": 7.0, "concrete_fc_ksi": 5.0, "concrete_unit_weight_kcf": 0.15, "superimposed_dead_load_ksf": 0.05}


def procedure_block(evidence=None, accepted=False, plan=None):
    source, information = proc.method_sha256()
    decision = "accepted" if accepted else "open"
    return {"schema": proc.PROCEDURE_SCHEMA,
            "method": {"id": "fixture_method", "source_sha256": source, "information_sha256": information},
            "domain": {"beam_sections_in": [[18.0, 28.0]], "beam_fc_ksi": [4.0], "column_sections_in": [[26.0, 26.0]],
                       "thickness_ladder_in": {"minimum": 5.0, "maximum": 14.0, "increment": 0.5}},
            "protocol": {"expected_resolution": {"status": "resolved", "base_levels": 2},
                         "instance_checks": list(proc.INSTANCE_CHECKS),
                         "plan_sha256": proc.plan_sha256(plan) if plan is not None else "0" * 64},
            "dispositions": {name: {"decision": decision, "consequence": f"fixture disposition {name}",
                                    "evidence": list(evidence or [])} for name in proc.DISPOSITION_IDS},
            "flag_basis": {flag: list(proc.REQUIRED_DISPOSITIONS[flag]) for flag in dep.ASSERTION_FLAGS}}


def configuration(status="test_fixture_active", **domain):
    block = procedure_block()
    block["domain"].update(domain)
    return replace(DesignConfig(), floor_analysis=replace(DesignConfig().floor_analysis, slab_procedure={
        "status": status, "identity": proc.identity(block), "procedure": block}))


class Loading(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, value, name="profile.json"):
        path = self.tmp / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def v2(self, base=EXAMPLE, mutate=None, **block):
        value = json.loads(Path(base).read_text(encoding="utf-8"))
        value.update(schema=dep.SCHEMA_V2, profile_id="v2_fixture")
        value["scope"].update(floor_sections_sha256=None, slab_thickness_in=None)
        value["procedure"] = procedure_block(plan=value["slab_refinement"], **block)
        if mutate:
            mutate(value)
        return value

    def fixture_v2(self, kind="test_fixture", accepted=True, mutate=None):
        shutil.copyfile(FIXTURE_EVIDENCE, self.tmp / FIXTURE_EVIDENCE.name)
        evidence = [{"path": FIXTURE_EVIDENCE.name, "sha256": hashlib.sha256(FIXTURE_EVIDENCE.read_bytes()).hexdigest()}]
        value = self.v2(FIXTURE, evidence=evidence, accepted=accepted)
        value["kind"] = kind
        if mutate:
            mutate(value)
        return self.write(value)

    def test_a_v2_example_loads_inactive_and_asserts_nothing(self):
        profile = dep.load(self.write(self.v2()))
        self.assertEqual(profile.procedure_status, "inactive_unreviewed")
        cfg = profile.apply(DesignConfig())
        self.assertEqual(cfg.slab_actions, SlabActionAssertions())
        procedure = cfg.floor_analysis.slab_procedure
        self.assertEqual((procedure["status"], procedure["identity"]),
                         ("inactive_unreviewed", proc.identity(profile.data["procedure"])))
        self.assertIsNone(cfg.floor_analysis.slab_assertion_floor_sections)
        summary = profile.summary()
        self.assertEqual((summary["schema"], summary["procedure_status"], summary["asserted"]),
                         (dep.SCHEMA_V2, "inactive_unreviewed", False))
        self.assertIn("nothing is asserted", summary["effect"])

    def test_a_changed_method_file_makes_the_procedure_inactive(self):
        value = self.v2()
        value["procedure"]["method"]["source_sha256"]["Design/SMRF_Slab_Actions.py"] = "0" * 64
        profile = dep.load(self.write(value))
        self.assertEqual(profile.procedure_status, "inactive_method_changed")
        self.assertEqual(profile.summary()["method_files_changed"], ["Design/SMRF_Slab_Actions.py"])
        fixture = self.fixture_v2(mutate=lambda d: d["procedure"]["method"]["source_sha256"].update(
            {"Design/SMRF_Floor_Mesh.py": "1" * 64}))
        changed = dep.load(fixture, allow_test_fixture=True)
        self.assertEqual(changed.procedure_status, "inactive_method_changed")
        self.assertEqual(changed.apply(DesignConfig()).slab_actions, SlabActionAssertions())   # nothing applies

    def test_the_domain_replaces_the_floor_list_and_the_single_thickness_and_a_plan_is_declared(self):
        for mutate in (lambda d: d["scope"].update(floor_sections_sha256=["a" * 64]),
                       lambda d: d["scope"].update(slab_thickness_in=7.0),
                       lambda d: d.update(slab_refinement=None, slab_refinement_basis=None),
                       lambda d: d["procedure"].pop("flag_basis"),
                       lambda d: d["procedure"]["protocol"].update(instance_checks=["refinement_passed"])):
            with self.subTest(mutate=mutate), self.assertRaises(dep.EvidenceProfileError):
                dep.load(self.write(self.v2(mutate=mutate)))

    def test_a_test_fixture_procedure_is_labelled_and_never_a_review(self):
        profile = dep.load(self.fixture_v2(), allow_test_fixture=True)
        self.assertEqual(profile.procedure_status, "test_fixture_active")
        self.assertTrue(profile.apply(DesignConfig()).slab_actions.all_asserted())
        with self.assertRaisesRegex(dep.EvidenceProfileError, "test-fixture"):
            dep.load(self.fixture_v2())                                         # the runner's load
        # Relabelled as a reviewed procedure, its TEST FIXTURE text is refused.
        with self.assertRaisesRegex(dep.EvidenceProfileError, "TEST FIXTURE or PROBE"):
            dep.load(self.fixture_v2(kind="reviewed_procedure"))

    def test_each_flag_rests_on_its_own_dispositions(self):
        # A gate flag cannot rest on an unrelated disposition ...
        with self.assertRaisesRegex(dep.EvidenceProfileError, "its own dispositions"):
            dep.load(self.write(self.v2(mutate=lambda d: d["procedure"]["flag_basis"].update(
                two_way_shear_path_assessed=["tolerance"]))))
        # ... and an asserted gate whose own disposition is rejected is refused, the others accepted.
        def reject(d):
            d["procedure"]["dispositions"]["two_way_shear_path"]["decision"] = "rejected"
            d["slab_action_assertions"]["two_way_shear_path_assessed"] = True
        with self.assertRaisesRegex(dep.EvidenceProfileError, "two_way_shear_path_assessed"):
            dep.load(self.fixture_v2(mutate=reject), allow_test_fixture=True)

    def test_the_protocol_binds_the_reviewed_plan(self):
        def loosen(d):
            d["slab_refinement"]["moment_tolerance"] = 0.5                        # after the protocol hashed it
        with self.assertRaisesRegex(dep.EvidenceProfileError, "plan_sha256"):
            dep.load(self.write(self.v2(mutate=loosen)))

    def test_fixture_wording_anywhere_is_not_a_review(self):
        def clean(value):
            if isinstance(value, str):
                return re.sub(r"(?i)test\s+fixture", "a named reviewer", value)
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, list):
                return [clean(v) for v in value]
            return value

        def relabel(d, extra=None):
            d.update(clean(d))
            d["procedure"]["protocol"]["plan_sha256"] = proc.plan_sha256(d["slab_refinement"])   # the cleaned plan
            if extra:
                extra(d)
        self.assertEqual(dep.load(self.fixture_v2(kind="reviewed_procedure", mutate=relabel)).procedure_status, "active")
        for where in (lambda d: d.update(slab_refinement_basis="TEST FIXTURE plan"),
                      lambda d: d["provenance"].update(purpose="TEST FIXTURE only"),
                      lambda d: d["review"]["evidence"][0].update(description="TEST FIXTURE file")):
            with self.subTest(where=where), self.assertRaisesRegex(dep.EvidenceProfileError, "TEST FIXTURE or PROBE"):
                dep.load(self.fixture_v2(kind="reviewed_procedure", mutate=lambda d, w=where: relabel(d, w)))

    def test_an_asserted_flag_needs_accepted_dispositions(self):
        with self.assertRaisesRegex(dep.EvidenceProfileError, "no permission to set a flag true"):
            dep.load(self.fixture_v2(accepted=False), allow_test_fixture=True)

    def test_the_v1_example_keeps_its_identity(self):
        self.assertEqual(dep.load(EXAMPLE).identity(), "59d60994a8d2ca71d61774c55e7c87089c550f5cc898311aef0e3454a3e3bdb5")
        self.assertIsNone(dep.load(EXAMPLE).procedure_status)


class DomainAndInstances(unittest.TestCase):
    def description(self):
        return {"schema": floor_sections.SCHEMA, "fc_beam_ksi": 4.0,
                "beam_lines": {"x": [{"b_in": 18.0, "h_in": 28.0, "fc_ksi": 4.0}] * 3,
                               "y": [{"b_in": 18.0, "h_in": 28.0, "fc_ksi": 4.0}] * 3},
                "supports": [[{"b_in": 26.0, "h_in": 26.0}] * 3] * 3}

    def evidence(self, nx=2, ny=2):
        cases = [{"id": "1.4D", "dead_factor": 1.4, "live_factor": 0.0, "live_pattern": "none"},
                 {"id": "1.2D+1.6L_all", "dead_factor": 1.2, "live_factor": 1.6, "live_pattern": "all"}]
        return {**{flag: True for flag in _FLAGS}, "cases": cases, "strips": [{"m": 1.0}],
                "equilibrium": [{"case_id": c["id"], "numerical_balance_passed": True} for c in cases],
                "numerical_preconditions": {"zero_membrane_force_verified": True},
                "shear_recovery": {"top_rows_recovered_at_face": True},
                "refinement": {"status": "passed", "resolution": {"status": "resolved", "meshes": [1, 2]}}}

    def test_the_domain(self):
        procedure = configuration().floor_analysis.slab_procedure
        self.assertEqual(proc.floor_domain_problems(procedure, self.description(), 7.0), [])
        self.assertIn("slab thickness 7.25 in is not on the domain's ladder",
                      proc.floor_domain_problems(procedure, self.description(), 7.25))
        wide = copy.deepcopy(self.description())
        wide["beam_lines"]["x"][0] = {"b_in": 20.0, "h_in": 28.0, "fc_ksi": 4.0}
        self.assertEqual(proc.floor_domain_problems(procedure, wide, 7.0), ["beam section 20x28 in outside the domain"])
        self.assertTrue(proc.floor_domain_problems(procedure, {"beam_lines": {}}, 7.0))           # malformed: outside

    def test_the_load_case_inventory_is_recomputed_from_the_slab(self):
        light = proc.expected_load_cases({**SLAB, "thickness_in": 5.0}, 0.05, 3, 3)      # 0.05 <= 0.75 (0.0625 + 0.05)
        self.assertEqual(sorted(light), ["1.2D+1.6L_all", "1.4D"])
        heavy = proc.expected_load_cases({**SLAB, "thickness_in": 5.0}, 0.10, 3, 3)
        self.assertEqual(len(heavy), 2 + 2 + 2 + 2)
        self.assertEqual(heavy["1.2D+1.2L_even"][0], 1.2)
        self.assertAlmostEqual(heavy["1.2D+1.2L_even"][1], 1.2, places=12)          # 3/4 of 1.6

    def test_instance_checks_pass_fail_mask_and_replay(self):
        procedure = configuration().floor_analysis.slab_procedure
        record = proc.instance_record(procedure, "a" * 64, [1, 2], self.evidence(), self.description(), SLAB, 0.05, 2, 2)
        self.assertTrue(record["passed"], record)
        for change in (lambda e: e["refinement"].update(status="comparison_failed"),
                       lambda e: e["refinement"]["resolution"].update(meshes=[1, 2, 3]),
                       lambda e: e["equilibrium"][0].update(numerical_balance_passed=False),
                       lambda e: e["shear_recovery"].update(top_rows_recovered_at_face=False),
                       lambda e: e["cases"].pop()):
            evidence = self.evidence()
            change(evidence)
            with self.subTest(change=change):
                self.assertFalse(proc.instance_record(procedure, "a" * 64, [1], evidence, self.description(), SLAB,
                                                      0.05, 2, 2)["passed"])
        masked = proc.masked(self.evidence(), "fixture")
        self.assertFalse(any(masked[flag] for flag in _FLAGS))
        self.assertEqual(masked["strips"], self.evidence()["strips"])
        combined = {"by_floor": [{"floor_sections_sha256": "a" * 64, "floors": [1, 2], "evidence": self.evidence()}],
                    "procedure_instances": [record]}
        description = {"distinct": {"a" * 64: {"sections": self.description()}}}
        self.assertEqual(proc.replay_problems(procedure, combined, description, SLAB, 0.05, 2, 2), [])
        combined["by_floor"][0]["evidence"]["refinement"]["status"] = "comparison_failed"     # an edited pass
        self.assertTrue(proc.replay_problems(procedure, combined, description, SLAB, 0.05, 2, 2))


class SlabStage(unittest.TestCase):
    def setUp(self):
        self.frame = gf.frame()
        self.frame.__enter__()
        self.state = mg.install(gf.uniform_state())
        self.description = floor_sections.by_floor()

    def tearDown(self):
        self.frame.__exit__(None, None, None)

    def test_the_scope_fails_closed(self):
        self.assertTrue(gd.slab_assertion_scope(configuration(), self.description)["in_scope"])
        inactive = gd.slab_assertion_scope(configuration("inactive_unreviewed"), self.description)
        self.assertFalse(inactive["in_scope"])
        self.assertEqual(inactive["procedure"]["status"], "inactive_unreviewed")
        outside = gd.slab_assertion_scope(configuration(beam_sections_in=[[20.0, 28.0]]), self.description)
        self.assertFalse(outside["in_scope"])
        self.assertTrue(all(row["problems"] for row in outside["procedure"]["floors"]))
        sp.SLAB_THICKNESS_IN = 7.25
        self.assertFalse(gd.slab_assertion_scope(configuration(), self.description)["thickness_in_scope"])

    def test_an_unresolvable_plan_is_recorded_and_each_floor_keeps_its_instance(self):
        cfg = configuration()
        cfg = replace(cfg, floor_analysis=replace(cfg.floor_analysis, slab_refinement={"recipe": "graded_face_v1"}))

        def refuse(*_args, **_kwargs):
            raise ValueError("the face lies outside the first quarter of the bay")
        with mock.patch("Design.SMRF_Slab_Refinement.build_refined_slab_action_evidence", refuse):
            combined = gd._slab_evidence(cfg, SLAB, self.description, {}, {})
        self.assertEqual(combined["refinement"]["status"], "plan_unresolvable")
        self.assertFalse(any(combined[flag] for flag in _FLAGS))
        instances = combined["procedure_instances"]
        self.assertEqual(len(instances), len(self.description["distinct"]))
        self.assertTrue(all(not row["passed"] and row["refinement_status"] == "plan_unresolvable" for row in instances))
        self.assertTrue(all("procedure_mask" in entry["evidence"] for entry in combined["by_floor"]))
        self.assertEqual(combined["procedure"]["status"], "test_fixture_active")

    def test_the_uniform_route_refuses_a_procedure(self):
        with self.assertRaisesRegex(ValueError, "grouped route only"):
            driver._update_slab_reinforcement(configuration(), SLAB)


if __name__ == "__main__":
    unittest.main()
