"""Acceptance/persistence oracles only: synthetic replay, never engineering proof."""
import copy
import hashlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Design import SMRF_Common_Action_Contract as ca
from Design import Code_Check_Contract as cc
from Design.SMRF_Common import make_check, not_evaluated


class CommonActionContract(unittest.TestCase):
    def setUp(self):
        self.state = types.SimpleNamespace(identity=lambda: "actual-fixture-state")
        self.file = "Design/SMRF_Common_Action_Contract.py"
        self.source = {self.file: hashlib.sha256((ca.SOURCE_ROOT/self.file).read_bytes()).hexdigest()}
        self.patch_sources = patch.object(ca, "REQUIRED_SOURCE_FILES", (self.file,))
        self.patch_sources.start()
        self.addCleanup(self.patch_sources.stop)
        self.package = dict(method="SYNTHETIC UNIT FIXTURE ONLY", state_identity=self.state.identity(),
                            definition_sha256="definition", source_identity=self.source,
                            analysis_source_identity=ca.sha256(self.source), actual_response={"force": 12.})
        self.seal()
        self.rows = [make_check(name, "synthetic arithmetic", 12., 20.) for name in ca.ROW_IDS]
        self.producer = types.ModuleType("Design.SMRF_Common_Action_Analysis")
        def replay(state, package):
            return {**{k: package[k] for k in ("state_identity", "source_identity", "definition_sha256", "payload_sha256")},
                    "checks": copy.deepcopy(self.rows)}
        self.producer.replay_assessment = replay
        self.patch_module = patch.dict(sys.modules, {self.producer.__name__: self.producer})
        self.patch_module.start()
        self.addCleanup(self.patch_module.stop)

    def seal(self):
        self.package["payload_sha256"] = ca.payload_sha256(self.package)

    def record(self):
        record = {name: {} for name in ("geometry", "materials", "floor_loads", "seismic", "demand")}
        record.update(member_groups={"sha256": self.state.identity()}, slab={"thickness_in": 7.},
                      slab_reinforcement={"layout": {"bar_size": 4}}, bar_layers={"actual": 1},
                      common_actions=copy.deepcopy(self.package))
        record["common_actions_input_binding"] = ca.saved_binding(record, record["common_actions"])
        return record

    def test_thirteen_rows_and_all_seven_individually_required(self):
        self.assertEqual(len(ca.ROW_IDS), 13)
        self.assertEqual(len({r for r in ca.ROW_IDS if r.startswith("common.disposition.")}), 7)
        self.assertEqual(len(ca.required_inventory()), 13)

    def test_exact_replayed_arithmetic_passes_contract_only(self):
        result = ca.assess_common_actions(self.state, self.package)
        self.assertTrue(result["accepted"])
        self.assertTrue(result["replay_completed"])

    def test_missing_package_is_thirteen_unknowns(self):
        result = ca.assess_common_actions(self.state, None)
        self.assertFalse(result["accepted"])
        self.assertTrue(all(r["status"] == "not_evaluated" for r in result["checks"]))

    def test_bound_historical_diagnostic_cannot_qualify_current_candidate(self):
        self.package['saved_physical_input_binding']=dict(prior_source_identity='old',input_manifest_sha256='input')
        self.seal()
        with self.assertRaisesRegex(ValueError,'diagnostic evidence'):ca.validate_package(self.state.identity(),self.package)

    def test_saved_pass_flag_does_not_replace_replay(self):
        self.package.update(status="pass", accepted=True, checks=self.rows)
        self.seal()
        self.rows = []
        self.assertFalse(ca.assess_common_actions(self.state, self.package)["accepted"])

    def test_each_missing_disposition_blocks(self):
        original = copy.deepcopy(self.rows)
        for name in ca.ROW_IDS[6:]:
            with self.subTest(name=name):
                self.rows = [r for r in original if r["id"] != name]
                result = ca.assess_common_actions(self.state, self.package)
                self.assertFalse(result["accepted"])
                self.assertEqual([r["id"] for r in result["checks"] if r["status"] == "not_evaluated"], [name])

    def test_computed_failure_stays_beside_unknown(self):
        self.rows[0] = make_check(ca.ROW_IDS[0], "synthetic", 25., 20.)
        self.rows.pop()
        result = ca.assess_common_actions(self.state, self.package)
        self.assertEqual(result["checks"][0]["status"], "fail")
        self.assertEqual(result["checks"][-1]["status"], "not_evaluated")

    def test_duplicate_replay_refused(self):
        self.rows.append(self.rows[0])
        self.assertFalse(ca.assess_common_actions(self.state, self.package)["replay_completed"])

    def test_arithmetic_tamper_refused(self):
        self.rows[0]["demand"] = 50.
        self.assertFalse(ca.assess_common_actions(self.state, self.package)["replay_completed"])

    def test_review_flag_cannot_replace_computation(self):
        self.rows[0] = not_evaluated(ca.ROW_IDS[0], "synthetic", "reviewed")
        self.rows[0]["details"]["review_dependency"] = {"item": "invented"}
        self.assertFalse(ca.assess_common_actions(self.state, self.package)["replay_completed"])

    def test_replay_identity_mismatch_refused(self):
        original = self.producer.replay_assessment
        self.producer.replay_assessment = lambda state, package: {**original(state, package), "definition_sha256": "other"}
        self.assertFalse(ca.assess_common_actions(self.state, self.package)["replay_completed"])

    def test_payload_change_refused(self):
        self.package["actual_response"]["force"] = 19.
        self.assertFalse(ca.assess_common_actions(self.state, self.package)["replay_completed"])

    def test_other_state_refused(self):
        self.package["state_identity"] = "other"
        self.seal()
        self.assertFalse(ca.assess_common_actions(self.state, self.package)["replay_completed"])

    def test_source_change_refused(self):
        self.package["source_identity"][self.file] = "0"*64
        self.package["analysis_source_identity"] = ca.sha256(self.package["source_identity"])
        self.seal()
        self.assertFalse(ca.assess_common_actions(self.state, self.package)["replay_completed"])

    def test_source_path_escape_refused(self):
        self.assertTrue(ca.source_problems({"../outside.py": "0"*64}))

    def test_mechanical_source_identity_refused(self):
        self.package["analysis_source_identity"] = "other"
        self.seal()
        self.assertFalse(ca.assess_common_actions(self.state, self.package)["replay_completed"])

    def test_roundtrip_uses_exact_raw_payload(self):
        record = json.loads(json.dumps(self.record()))
        self.assertEqual(ca.bound_saved_package(record), self.package)

    def test_legacy_absence_not_upgraded(self):
        self.assertIsNone(ca.bound_saved_package({}))

    def test_mat_slab_and_bar_coordinate_tampering_refused(self):
        changes = [("slab", {"thickness_in": 8.}), ("slab_reinforcement", {"layout": {"bar_size": 5}}),
                   ("bar_layers", {"actual": 2}), ("request_identity", {"policy": {"changed": True}})]
        for key, value in changes:
            with self.subTest(key=key):
                record = self.record(); record[key] = value
                with self.assertRaises(ValueError): ca.bound_saved_package(record)

    def test_raw_tamper_not_fixed_by_resealing_only_payload(self):
        record = self.record()
        record["common_actions"]["actual_response"]["force"] = 30.
        record["common_actions"]["payload_sha256"] = ca.payload_sha256(record["common_actions"])
        with self.assertRaises(ValueError): ca.bound_saved_package(record)

    def test_required_common_scope_retains_foundation_boundary(self):
        new = cc.scope_for_common_actions(True); old = cc.scope_for_common_actions(False)
        self.assertNotIn("diaphragm", new["excluded"])
        self.assertIn("foundation", new["excluded"])
        self.assertIn("diaphragm", old["excluded"])

    def test_qualification_replay_and_scope_do_not_infer_review(self):
        from Design import Grouped_Record as gr
        record = self.record()
        checks = gr._common_action_checks(record, self.state)
        self.assertEqual(len(checks), 14)
        self.assertEqual(gr._diaphragm_check(record, checks)["status"], "pass")
        record["code_check_rows"] = {"rows": [cc.compact_row(ca.FAMILY, r) for r in self.rows]}
        self.assertIn("Required computed", gr.grouped_scope(record)["diaphragm"])
        record["code_check_rows"]["rows"].pop()
        self.assertIn("NOT evaluated", gr.grouped_scope(record)["diaphragm"])
        self.assertIn("legacy_saved_text", gr.grouped_scope(record)["saved_limitations_status"])
        record["code_check_rows"]["contract_version"] = cc.CONTRACT_VERSION
        self.assertIn("current_contract_scope", gr.grouped_scope(record)["saved_limitations_status"])

    def test_actual_inventory_requires_common_rows_and_missing_blocks_acceptance(self):
        from tests import grouped_fixture as gf, contract_fixture as cf
        from Design.Config import DesignConfig
        cfg = DesignConfig()
        cfg.floor_analysis.common_action_procedure = {"kind": "computed_common_actions_v1"}
        with gf.frame():
            state = gf.uniform_state()
            expected = cc.expected_inventory(state, cfg)
            self.assertLessEqual(ca.required_inventory(), expected["exact"])
            rows = [r for r in cf.rows_for(state) if r["family"] != ca.FAMILY]
            result = cc.code_check_result(rows, expected)
            self.assertFalse(result["accepted"])
            missing = {(r["family"], r["id"], r["location"]) for r in result["missing"]}
            self.assertLessEqual(ca.required_inventory(), missing)

    def test_mandatory_common_rows_cannot_claim_review(self):
        expected = {"exact": ca.required_inventory(), "patterns": [], "conditional": [], "excluded": []}
        rows = [cc.compact_row(ca.FAMILY, r) for r in self.rows]
        rows[0] = cc.compact_row(ca.FAMILY, not_evaluated(ca.ROW_IDS[0], "synthetic", "not calculated"))
        rows[0]["details"]["review_dependency"] = {"item": "invented"}
        result = cc.code_check_result(rows, expected)
        self.assertEqual(result["status"], "invalid")
        self.assertFalse(result["accepted"])


if __name__ == "__main__": unittest.main()
