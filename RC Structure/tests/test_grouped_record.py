"""A grouped design end to end: search, record, qualification, installation, nonlinear build and exports (2026-10-02).

One grouped record is produced on the fixture frame by the real search (slab-to-frame transfer on, slab
actions not asserted, a seed whose upper interior columns are smaller than the lower ones) and then used by
every test:

  the record       no uniform sections or reinforcement block; member groups, candidate log, quantities,
                   margins and the separate stop reason; the log file beside it has the same candidates;
  qualification    reproduces from the file on disk; edited evidence is caught by recomputation; reviews
                   that have not been made stay open (strength model, story-strength model M1);
  fail closed      an edited member row, a stale floor transfer, another request identity and a uniform
                   consumer are all refused, and a refused installation leaves the model as it was;
  the model        the nonlinear frame built from the record matches it member by member, and the element
                   rows, hinge tables and global parameters carry every member's own group and section.

The fixture frame is not a designed structure and the time history is a short synthetic pulse on it: these
are implementation checks of the code path, not screening results.
"""
import argparse
import copy
import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402

RC = Path(__file__).resolve().parents[1]
if str(RC / "Data_Generation") not in sys.path:
    sys.path.insert(0, str(RC / "Data_Generation"))

import openseespy.opensees as ops                                               # noqa: E402
import Structure_Parameters as sp                                               # noqa: E402
from Analysis import Hinge_Hysteresis_Diagnostic as hd                           # noqa: E402
from Analysis import Hinge_Moment_Rotation as hmr                                # noqa: E402
from Data_Generation import Graph_Exporter as graph                              # noqa: E402
from Design import Design_Driver as driver                                       # noqa: E402
from Design import Grouped_Record as gr                                          # noqa: E402
from Design import Grouped_Search as gs                                          # noqa: E402
from Design import Screening_Manifest as screening                               # noqa: E402
from Design.Config import DesignConfig, FloorAnalysisConfig                      # noqa: E402
from Design.SMRF_Qualification import GENERATION_RELEASE_READY                   # noqa: E402
from Loads.Ground_Motion import GroundMotionRecord                               # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402
from Model.Build_Model import build_model                                        # noqa: E402

TOP_INTERIOR, BASE_INTERIOR = "s03_04__column__interior", "s01_02__column__interior"
POLICY = gs.SearchPolicy(max_feasibility_trials=6, max_reduction_trials=1)
RECOMPUTED = ("grouped.record_installs", "joint.reproducible_actions", "qualification.capacity_evidence_recomputed",
              "qualification.hoops_match_design", "qualification.member_strength_recomputed", "grouped.weight_ledger_recomputed",
              "demands.story_evidence_recomputed", "grouped.beam_end_strengths_recomputed", "grouped.quantities_recomputed",
              "floor.transfer_load_integrity", "slab_thickness_evidence")


def status_of(qualification, name):
    return {c["status"] for c in qualification["checks"] if c["id"] == name}


def pulse(scale, n=60, dt=0.01, phase=0.0):
    t = np.arange(n) * dt
    return scale * np.sin(2.0 * np.pi * t / 0.4 + phase) * np.exp(-((t - 0.3) / 0.2) ** 2)


class GroupedRecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack = gf.frame(slab=None)
        cls.stack.__enter__()
        cls.folder = Path(tempfile.mkdtemp(prefix="grouped_record_test_"))
        cls.cfg = DesignConfig(floor_analysis=FloorAnalysisConfig(transfer_to_frame=True, transfer_mesh_per_bay=8))
        # 32 in since the bond rule of ACI 318-19 18.7.4.3 (2026-10-02): at 30 in the interior columns needed No. 14
        # bars, whose 1.25 ld (80 in at f'c 5 ksi) does not fit the 70-in half clear height, so that seed is no longer
        # feasible and the search would grow it to this pair anyway.
        uniform, described = gr.seed_from_sections((32.0, 32.0, 5.0), (18.0, 28.0, 4.0))
        # A seed that is already grouped: the upper interior columns one ladder step smaller (1 in per face).
        cls.seed_state = gs.resized(uniform, TOP_INTERIOR, (30.0, 30.0))
        cls.seed = {**described, "kind": "test_seed_with_a_smaller_upper_interior_column", "changed": {TOP_INTERIOR: [30.0, 30.0]}}
        cls.path = cls.folder / gr.ARTIFACT_NAME
        cls.created_record, cls.created = gr.load_or_create_grouped_design(cls.path, cls.cfg, cls.seed_state, cls.seed, POLICY,
                                                                           verbose=False)
        cls.record = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.state = mg.GroupedDesign.from_record(cls.record["member_groups"])

    @classmethod
    def tearDownClass(cls):
        cls.stack.__exit__(None, None, None)
        shutil.rmtree(cls.folder, ignore_errors=True)

    def setUp(self):
        ops.wipe()
        mg.clear()
        self.baseline = {key: getattr(sp, key, None) for key in gr._STATE_KEYS}

    # ---- the record ---------------------------------------------------------------------------------------
    def test_the_record_is_grouped_and_carries_no_uniform_section_or_cage(self):
        record = self.record
        self.assertTrue(self.created)
        self.assertEqual((record["schema_version"], record["design_mode"]), (gr.SCHEMA, "grouped"))
        self.assertNotIn("sections", record)
        self.assertNotIn("reinforcement", record)
        self.assertEqual(self.created_record, record)                                          # what was returned is what is on disk
        self.assertEqual(len(record["member_groups"]["members"]), len(gf.hand_members()))
        self.assertEqual(set(record["member_groups"]["designs"]), set(self.state.groups))
        self.assertFalse(record["member_groups"]["uniform"])
        self.assertEqual((self.state.designs[BASE_INTERIOR].b_in, self.state.designs[TOP_INTERIOR].b_in), (32.0, 30.0))
        self.assertIn("summary only", record["legacy_uniform_summary"]["label"])
        self.assertEqual(record["legacy_uniform_summary"]["column_size_range_in"], [30.0, 32.0])
        self.assertEqual(record["quantities"]["distinct_form_sizes"]["column"], [[30.0, 30.0], [32.0, 32.0]])
        self.assertEqual(record["floor_loads"]["ledger"]["grouped"], True)
        self.assertEqual(len(record["floor_sections"]["distinct"]), 2)                         # floors 1-2 and 3-4 differ at the interior supports
        self.assertEqual(set(record["floor_transfer"]["transfers"]), set(record["floor_sections"]["distinct"]))
        self.assertTrue(record["limitations"])

    def test_the_search_block_keeps_every_candidate_its_decision_and_a_separate_stop_reason(self):
        search = self.record["search"]
        self.assertEqual(search["policy"]["max_reduction_trials"], 1)
        self.assertEqual(search["seed"], self.seed)
        self.assertEqual(search["stop_reason"], "reduction_budget_exhausted")
        self.assertIn("smaller sections may remain untried", search["stop_detail"])
        decisions = [(c["phase"], c["decision"]) for c in search["candidates"]]
        self.assertEqual(decisions[0], ("feasibility", "feasible"))
        self.assertEqual(search["counts"]["reduction_candidates"], 1)
        evaluated = [c for c in search["candidates"] if c["evaluation"] is not None]
        for candidate in evaluated:
            for key in ("constraints", "quantities", "margins", "sections", "stages", "solves"):
                self.assertIn(key, candidate["evaluation"])
            self.assertNotIn("evidence", candidate["evaluation"])
        # no slab layout is established (slab actions are not asserted): total steel is unknown, so no move is adopted
        self.assertIsNone(self.record["quantities"]["total_steel_in3"])
        self.assertIsNone(self.record["quantities"]["slab_steel_in3"])
        self.assertEqual({t["decision"] for t in search["tradeoffs"]}, {"unresolved"})
        self.assertEqual(self.record["member_groups"]["sha256"], evaluated[0]["evaluation"]["result_identity"])
        lines = [json.loads(line) for line in (self.folder / gr.CANDIDATE_LOG_NAME).read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(line["index"], line["decision"]) for line in lines], [(c["index"], c["decision"]) for c in search["candidates"]])

    def test_quantities_are_worked_from_the_groups(self):
        q = self.record["quantities"]
        slab = self.record["slab"]["thickness_in"]
        frame = 0.0
        for gid, design in self.state.designs.items():
            members = len(self.state.groups[gid]["member_tags"])
            if design.is_column:
                frame += design.b_in * design.h_in * (gf.STORY_H - slab) * members
        columns = sum(q["groups"][gid]["concrete_in3"] for gid, d in self.state.designs.items() if d.is_column)
        self.assertAlmostEqual(columns, frame, places=6)
        self.assertAlmostEqual(q["concrete_slab_in3"], slab * gf.NX * gf.BAY_X * gf.NY * gf.BAY_Y * gf.NZ, places=6)
        self.assertAlmostEqual(q["concrete_total_in3"], q["concrete_frame_in3"] + q["concrete_slab_in3"], places=6)
        self.assertAlmostEqual(q["frame_steel_in3"], q["longitudinal_steel_in3"] + q["transverse_steel_in3"] + q["transition_extra_steel_in3"]
                               + q["transition_extra_transverse_steel_in3"], places=6)
        design = self.state.designs[BASE_INTERIOR]
        self.assertAlmostEqual(q["groups"][BASE_INTERIOR]["longitudinal_steel_in3"],
                               design.longitudinal_area_in2 * gf.STORY_H * len(self.state.groups[BASE_INTERIOR]["member_tags"]), places=6)

    # ---- qualification ------------------------------------------------------------------------------------
    def test_qualification_reproduces_from_the_file_and_restores_the_model(self):
        saved = self.record["qualification"]
        again = gr.qualify_grouped_design(self.record)
        self.assertEqual(again["counts"], saved["counts"])
        self.assertEqual([(c["id"], c.get("location", ""), c["status"]) for c in again["checks"]],
                         [(c["id"], c.get("location", ""), c["status"]) for c in saved["checks"]])
        self.assertIsNone(mg.active())                                                         # qualification installed nothing for good
        self.assertEqual({key: getattr(sp, key, None) for key in gr._STATE_KEYS}, self.baseline)
        for name in RECOMPUTED:
            self.assertEqual(status_of(again, name), {"pass"}, name)
        self.assertEqual(again["counts"]["fail"], 0)

    def test_reviews_that_have_not_been_made_stay_open_and_nothing_is_released(self):
        qualification = self.record["qualification"]
        self.assertFalse(qualification["accepted"])
        self.assertFalse(self.record["dcr"]["accepted"])
        for name in ("qualification.strength_model_verification", "demands.vertical_strength_regularity",
                     "qualification.detailing_model_consistency", "detailing.congestion_and_placement",
                     "floor.compatibility_idealization_reviewed", "qualification.slab_contribution"):
            self.assertEqual(status_of(qualification, name), {"not_evaluated"}, name)
        strength = self.record["demand_basis"]["regularity"]["lateral_strength_distribution"]
        self.assertEqual(strength["applicability"]["status"], "provisional")
        self.assertEqual(strength["uniform_over_height"]["claim"], False)
        self.assertEqual([entry["story"] for entry in strength["stories"]], [1, 2, 3, 4])
        self.assertIsNone(self.record["demand_basis"]["regularity"]["regular"])
        self.assertFalse(GENERATION_RELEASE_READY)
        with self.assertRaises(RuntimeError):
            gr.require_accepted_grouped_design(self.record)

    def test_the_band_boundary_transition_is_declared_and_awaits_review_when_the_cage_changes(self):
        transitions = self.record["capacity_design"]["transitions"]
        key = f"{BASE_INTERIOR}>{TOP_INTERIOR}"
        boundary = transitions[key]
        self.assertEqual((boundary["kind"], boundary["supported"]), ("offset_bends", True))                    # 32 to 30: 1 in per face
        self.assertEqual(boundary["rules"], "column_transition_rules_v2")
        # every bar of the lower column has one path; none runs straight, because the faces step in
        self.assertEqual(sum(boundary["path_counts"].values()), len(boundary["paths"]))
        self.assertEqual(boundary["path_counts"]["straight"], 0)
        self.assertGreaterEqual(boundary["offset_bend"]["bar_offset_in"], 1.0)
        self.assertLessEqual(boundary["offset_bend"]["inclined_length_in"], 28.0)
        self.assertTrue(boundary["not_established"])
        review = next(c for c in self.record["qualification"]["checks"] if c["id"] == "detailing.column_transition_rules_reviewed")
        self.assertIn(key, review["details"]["transitions"])
        self.assertTrue(review["details"]["not_established"])
        # a boundary whose columns differ only in their hoop bar is an offset transition too, not a straight one
        for other, transition in transitions.items():
            lower, upper = (self.state.designs[gid] for gid in other.split(">"))
            same = (lower.b_in, lower.bar_size, lower.top_bars, lower.side_bars, lower.stirrup_bar_size) ==                    (upper.b_in, upper.bar_size, upper.top_bars, upper.side_bars, upper.stirrup_bar_size)
            self.assertEqual(transition["kind"] == "same_cage", same, other)
        self.assertEqual(status_of(self.record["qualification"], "detailing.column_transition_rules_reviewed"), {"not_evaluated"})
        self.assertEqual(status_of(self.record["qualification"], "column.transition"), {"pass"})

    def test_vertical_regularity_is_on_the_2022_table_with_no_mass_check(self):
        ids = {c["id"] for c in self.record["qualification"]["checks"]}
        self.assertNotIn("demands.vertical_mass_regularity", ids)
        self.assertIn("demands.vertical_stiffness_regularity", ids)
        vertical = self.record["demand_basis"]["regularity"]["vertical_evidence"]
        self.assertEqual(vertical["thresholds"]["edition"], "ASCE 7-22")
        self.assertTrue(vertical["evidence"]["stiffness_complete"])
        self.assertEqual(len(vertical["stiffness"]), 2 * (gf.NZ - 1))
        self.assertIn("no threshold and no check", vertical["story_weights"]["status"])
        self.assertEqual(len(vertical["story_weights"]["rows"]), gf.NZ)
        stiffness = next(c for c in self.record["qualification"]["checks"] if c["id"] == "demands.vertical_stiffness_regularity")
        self.assertIn("Types 1a, 1b", stiffness["clause"])
        strength = next(c for c in self.record["qualification"]["checks"] if c["id"] == "demands.vertical_strength_regularity")
        self.assertIn("Types 4a, 4b", strength["clause"])
        self.assertEqual(strength["details"]["category"], "awaiting review item M1")
        # drift rows removed from the record: the evidence no longer reproduces and nothing passes on it
        record = copy.deepcopy(self.record)
        record["drift_screen"]["stories"] = record["drift_screen"]["stories"][:-1]
        qualification = gr.qualify_grouped_design(record)
        self.assertEqual(status_of(qualification, "demands.story_evidence_recomputed"), {"fail"})
        self.assertEqual(status_of(qualification, "demands.vertical_stiffness_regularity"), {"not_evaluated"})

    def test_the_candidate_settled_in_cages_hoops_and_slab_together(self):
        constraints = self.record["constraints"]
        for key in ("reinforcement_settled", "longitudinal_selection_settled", "hoops_settled", "slab_layout_settled", "slab_layout_kept"):
            self.assertIs(constraints[key], True, key)
        stage = self.record["stages"]["reinforcement"]
        self.assertTrue(stage["settled"] and stage["selection_settled"] and stage["hoops_settled"])
        self.assertFalse(stage["evidence_rebuilt_on_final_design"])
        last = stage["log"][-1]
        self.assertEqual((last["selection_settled"], last["hoops_settled"], last["hoops_changed"]), (True, True, []))

    def test_transition_hoops_are_priced_in_the_quantities(self):
        q = self.record["quantities"]
        self.assertIn("transition_extra_transverse_steel_in3", q)
        self.assertGreaterEqual(q["transition_extra_transverse_steel_in3"], 0.0)
        self.assertAlmostEqual(q["frame_steel_in3"], q["longitudinal_steel_in3"] + q["transverse_steel_in3"]
                               + q["transition_extra_steel_in3"] + q["transition_extra_transverse_steel_in3"], places=6)
        # above the reduced interior column no continuing-hoop detail is declared, and that is said, not priced as zero
        self.assertIn(f"{BASE_INTERIOR}>{TOP_INTERIOR}", q["transitions_without_priced_confinement"])

    def test_per_group_detailing_checks_carry_their_group(self):
        checks = [c for c in self.record["qualification"]["checks"] if c["id"] in ("beam.minimum_steel", "column.minimum_rho",
                                                                                 "beam.clear_span", "column.confinement_area_and_support")]
        locations = {c["location"].split("/")[0] for c in checks}
        self.assertEqual(locations, set(self.state.groups))
        clear = {c["location"]: c for c in self.record["qualification"]["checks"] if c["id"] == "beam.clear_span"}
        # the interior x beams of floors 1-2 frame between 32-in columns: 240 - 32
        self.assertAlmostEqual(clear["s01_02__beam_x__interior"]["demand"], gf.BAY_X - 32.0, places=9)
        self.assertAlmostEqual(clear["s03_04__beam_x__interior"]["demand"], gf.BAY_X - 0.5 * 32.0 - 0.5 * 30.0, places=9)

    def test_edited_evidence_is_caught_by_recomputation(self):
        def qualify(mutate):
            record = copy.deepcopy(self.record)
            mutate(record)
            return gr.qualify_grouped_design(record)

        def edit_capacity(record):
            check = next(c for c in record["capacity_design"]["checks"] if c["id"] == "joint.shear_screen")
            check["demand"] *= 0.5

        self.assertEqual(status_of(qualify(edit_capacity), "qualification.capacity_evidence_recomputed"), {"fail"})
        edited = qualify(edit_capacity)
        self.assertEqual(status_of(edited, "qualification.joint_capacity_completion"), {"not_evaluated"})   # nothing nested is consumed

        def edit_quantities(record):
            record["quantities"]["concrete_total_in3"] *= 0.9
        self.assertEqual(status_of(qualify(edit_quantities), "grouped.quantities_recomputed"), {"fail"})

        def edit_ledger(record):
            record["floor_loads"]["ledger"]["floors"][0]["seismic_weight_kip"] *= 1.1
        self.assertEqual(status_of(qualify(edit_ledger), "grouped.weight_ledger_recomputed"), {"fail"})

        def edit_story_strength(record):
            entry = record["demand_basis"]["regularity"]["lateral_strength_distribution"]["stories"][0]["by_direction"]["x"]
            entry["line_story_strength_kip"][0] *= 1.2
        self.assertEqual(status_of(qualify(edit_story_strength), "demands.story_evidence_recomputed"), {"fail"})

        def edit_factor(record):
            record["design_actions"]["combinations"][0]["dead"] += 0.1
        self.assertEqual(status_of(qualify(edit_factor), "joint.reproducible_actions"), {"not_evaluated"})

        def edit_beam_strength(record):
            tag = next(iter(record["beam_end_strengths"]))
            record["beam_end_strengths"][tag]["hogging_i_kip_in"] *= 1.05
        self.assertEqual(status_of(qualify(edit_beam_strength), "grouped.beam_end_strengths_recomputed"), {"fail"})
        self.assertIsNone(mg.active())

    # ---- fail closed ----------------------------------------------------------------------------------------
    def assert_refused(self, record, exception, pattern=None):
        with (self.assertRaisesRegex(exception, pattern) if pattern else self.assertRaises(exception)):
            gr.apply_grouped_design(record)
        self.assertIsNone(mg.active())
        self.assertEqual({key: getattr(sp, key, None) for key in gr._STATE_KEYS}, self.baseline)
        self.assertEqual(sp.B_COL, gf.COLUMN.b_in)                                              # the uniform values are back, untouched
        qualification = gr.qualify_grouped_design(record)
        self.assertEqual(status_of(qualification, "grouped.record_installs"), {"fail"})
        self.assertEqual(len(qualification["checks"]), 1)
        self.assertFalse(qualification["accepted"])

    def test_an_edited_member_row_or_group_design_is_refused(self):
        record = copy.deepcopy(self.record)
        record["member_groups"]["members"][0]["b_in"] = 24.0
        self.assert_refused(record, mg.GroupedStateError, "disagrees with its group")
        record = copy.deepcopy(self.record)
        record["member_groups"]["designs"][TOP_INTERIOR]["top_bars"] += 1
        self.assert_refused(record, mg.GroupedStateError, "digest")
        record = copy.deepcopy(self.record)
        del record["member_groups"]["assignments"]["members"][-1]
        self.assert_refused(record, mg.GroupedStateError)

    def test_a_floor_transfer_of_another_design_is_refused(self):
        # The same record with its upper interior columns changed consistently (rows and digest rebuilt), but the old
        # floor descriptions and transfer left in place: the floors it was solved for are no longer the design's.
        other = self.state.with_designs({TOP_INTERIOR: gs.resized(self.state, TOP_INTERIOR, (26.0, 26.0)).designs[TOP_INTERIOR]})
        record = copy.deepcopy(self.record)
        record["member_groups"] = json.loads(json.dumps(other.to_record()))
        self.assert_refused(record, ValueError, "floor descriptions")
        with mg.installed(other):
            from Design import SMRF_Floor_Sections as fs
            record["floor_sections"] = json.loads(json.dumps(fs.by_floor()))
        self.assert_refused(record, ValueError, "other floors")
        record = copy.deepcopy(self.record)
        record["floor_transfer"]["transfers"].pop(next(iter(record["floor_transfer"]["transfers"])))
        self.assert_refused(record, ValueError, "one solution per distinct floor")

    def test_a_record_with_a_uniform_block_or_another_schema_or_geometry_is_refused(self):
        record = copy.deepcopy(self.record)
        record["sections"] = {"b_col_in": 30.0}
        self.assert_refused(record, ValueError, "no uniform sections")
        record = copy.deepcopy(self.record)
        record["schema_version"] = driver.DESIGN_SCHEMA_VERSION
        self.assert_refused(record, ValueError, "Not a grouped design record")
        record = copy.deepcopy(self.record)
        record["geometry"]["bay_x_in"] += 12.0
        self.assert_refused(record, ValueError, "geometry")
        record = copy.deepcopy(self.record)
        record["slab"]["thickness_in"] += 0.5
        self.assert_refused(record, ValueError, "slab thickness evidence")

    def test_a_uniform_consumer_cannot_read_a_grouped_record(self):
        with self.assertRaises(KeyError):
            driver.apply_design(self.record)
        self.assertEqual(sp.B_COL, gf.COLUMN.b_in)
        self.assertTrue(gr.is_grouped_record(self.record))
        self.assertFalse(gr.is_grouped_record({"sections": {}, "reinforcement": {}}))

    def test_the_request_identity_covers_seed_budget_grouping_and_rules(self):
        base = gr.grouped_request_identity(self.cfg, self.seed, POLICY)
        self.assertEqual(base["sha256"], self.record["request_identity"]["sha256"])
        self.assertEqual(base["grouping"]["group_count"], len(self.state.groups))
        self.assertEqual(base["search"]["max_reduction_trials"], 1)
        for key in ("evaluation", "reinforcement_selection", "column_transitions", "bar_layers", "capacity_design", "qualification"):
            self.assertIn(key, base["rules"])
        other_budget = gr.grouped_request_identity(self.cfg, self.seed, gs.SearchPolicy(max_feasibility_trials=6, max_reduction_trials=2))
        other_seed = gr.grouped_request_identity(self.cfg, {**self.seed, "column": [36.0, 36.0, 5.0]}, POLICY)
        self.assertEqual(len({base["sha256"], other_budget["sha256"], other_seed["sha256"]}), 3)
        self.assertNotEqual(base["sha256"], base["uniform_request_sha256"])
        # the cached record is refused for another request, and reused for its own
        with self.assertRaisesRegex(RuntimeError, "different inputs"):
            gr.load_or_create_grouped_design(self.path, self.cfg, self.seed_state, self.seed,
                                             gs.SearchPolicy(max_feasibility_trials=6, max_reduction_trials=2), verbose=False)
        self.assertIsNone(mg.active())
        record, created = gr.load_or_create_grouped_design(self.path, self.cfg, self.seed_state, self.seed, POLICY, verbose=False)
        self.assertFalse(created)
        self.assertEqual(mg.active().identity(), self.record["member_groups"]["sha256"])
        with mg.installed(self.state):
            with self.assertRaises(mg.GroupedStateError):                                       # built before a design is installed
                gr.grouped_request_identity(self.cfg, self.seed, POLICY)

    def test_seeds_are_square_and_a_uniform_record_expands_into_its_groups(self):
        with self.assertRaisesRegex(ValueError, "square"):
            gr.seed_from_sections((30.0, 26.0, 5.0), (18.0, 28.0, 4.0))
        uniform = {"schema_version": driver.DESIGN_SCHEMA_VERSION, "request_identity": {"sha256": "abc"},
                   "sections": {"b_col_in": 26.0, "h_col_in": 26.0, "fc_col_ksi": 5.0, "b_beam_in": 18.0, "h_beam_in": 28.0, "fc_beam_ksi": 4.0},
                   "reinforcement": {"col_bar_size": 9, "col_top_bars": 4, "col_bot_bars": 4, "col_side_bars": 2,
                                     "col_stirrup_bar_size": 4, "col_stirrup_legs": 4, "col_stirrup_spacing_in": 4.0,
                                     "col_stirrup_legs_by_direction": {"across_b_face": 4, "across_h_face": 4},
                                     "beam_bar_size": 8, "beam_top_bars": 4, "beam_bot_bars": 3, "beam_side_bars": 0,
                                     "beam_stirrup_bar_size": 4, "beam_stirrup_legs": 2, "beam_stirrup_spacing_in": 5.0}}
        state, seed = gr.seed_from_uniform_record(uniform)
        self.assertEqual(state.identity(), gf.uniform_state().identity())
        self.assertTrue(state.is_uniform())
        self.assertEqual((seed["kind"], seed["source_request_sha256"]), ("uniform_design_record", "abc"))

    # ---- the nonlinear model and its exports ------------------------------------------------------------------
    def test_the_gravity_modal_stage_builds_the_recorded_design_member_by_member(self):
        stage = screening.gravity_modal_stage(self.record, None, self.folder / "stage")
        self.assertEqual(stage["status"], "completed", stage.get("error"))
        self.assertTrue(all(stage["checks"].values()), stage["checks"])
        verification = stage["installed_design_verification"]
        self.assertTrue(verification["consistent"], verification["differences"])
        items = {c["item"]: c for c in verification["checks"]}
        members = len(gf.hand_members())
        self.assertEqual(items["registry.member_designs_vs_record_rows"]["installed"], f"{members} match")
        self.assertEqual(items["registry.beam_end_strengths_vs_record"]["saved"], f"{4 * (members - gf.NZ * (gf.NX + 1) * (gf.NY + 1))} entries")
        self.assertEqual(set(verification["registry"]["groups"]), set(self.state.groups))
        audit = stage["model_audit"]
        self.assertEqual(audit["sections"]["design_mode"], "grouped")
        self.assertEqual(audit["sections"]["by_group"][TOP_INTERIOR]["b_in"], 30.0)
        self.assertEqual(set(audit["beam_hinge_strengths_by_family"]), {g for g, d in self.state.designs.items() if not d.is_column})
        json.dumps(stage, default=str)                                                          # nothing withdrawn slipped into the result
        # the design period (elastic cracked frame) and the nonlinear model's elastic reference agree closely
        self.assertAlmostEqual(stage["modal"]["periods_elastic_reference_sec"][0], self.record["demand"]["model_period_sec"], delta=0.02)
        with (self.folder / "stage" / "gravity_modal" / hmr.SPRINGS_NAME).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0])[:len(hmr.SPRING_COLUMNS)], list(hmr.SPRING_COLUMNS))
        self.assertEqual(list(rows[0])[len(hmr.SPRING_COLUMNS):], list(hmr.GROUP_COLUMNS))
        saved = {str(row["member_tag"]): row["group_id"] for row in self.record["member_groups"]["members"]}
        for row in rows:
            self.assertEqual(row["group_id"], saved[row["physical_member_tag"]])

    def test_the_pushover_diagnostic_resolves_members_and_balances_gravity(self):
        from Analysis import Pushover_Diagnostic as diag
        gr.apply_grouped_design(self.record)
        settings = diag.DiagnosticSettings(direction="x", sign=1.0, du_in=0.05, max_steps=6, target_roof_drift_ratio=0.02)
        summary = diag.run_diagnostic(settings, self.folder / "pushover")
        self.assertTrue(summary["checks"]["all_pass"], summary["checks"])
        gravity = summary["gravity"]
        # the expected total is each floor's own transfer plus every member's own weight; the reaction balances it
        self.assertAlmostEqual(gravity["expected_vertical_load_kip"], gravity["vertical_reaction_kip"], places=6)
        self.assertLess(abs(gravity["vertical_equilibrium_residual_kip"]), 1e-6)
        self.assertEqual(summary["model_audit"]["sections"]["design_mode"], "grouped")
        self.assertEqual(summary["model_audit"]["sections"]["by_group"][TOP_INTERIOR]["b_in"], 30.0)
        json.dumps(summary, default=str)
        steps = [json.loads(line) for line in (self.folder / "pushover" / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
        columns = {row["tag"]: row for row in steps[-1]["columns"]} if "columns" in steps[-1] else {}
        for row in columns.values():                                                            # face offsets from each column's own joints
            self.assertEqual(row["face_offsets_in"], [0.0 if row["story"] == 1 else 14.0, 14.0])  # 28-in beams at every joint

    def test_a_tampered_registry_is_reported_by_the_installed_design_check(self):
        gr.apply_grouped_design(self.record)
        build_model()
        self.assertTrue(hd.verify_installed_design(self.record)["consistent"])
        record = copy.deepcopy(self.record)
        row = next(r for r in record["member_groups"]["members"] if r["group_id"] == TOP_INTERIOR)
        row["top_bars"] += 1
        tag = next(iter(record["beam_end_strengths"]))
        record["beam_end_strengths"][tag]["sagging_j_kip_in"] *= 1.01
        result = hd.verify_installed_design(record)
        self.assertFalse(result["consistent"])
        text = " | ".join(result["differences"])
        self.assertIn(f"member {row['member_tag']}", text)
        self.assertIn(f"beam {tag} end j sagging", text)

    def test_element_rows_and_global_parameters_resolve_every_member(self):
        gr.apply_grouped_design(self.record)
        build_model()
        rows = graph.collect_element_rows()
        saved = {row["member_tag"]: row for row in self.record["member_groups"]["members"]}
        self.assertEqual(len(rows), len(saved))
        first = list(rows[0])
        self.assertEqual(first[-len(graph.GROUP_ELEMENT_COLUMNS):], list(graph.GROUP_ELEMENT_COLUMNS))
        self.assertEqual(first[first.index("dir_z") + 1], "story_or_floor")                    # appended after the uniform columns
        for row in rows:
            member = saved[row["ele_tag"]]
            for key in ("b_in", "h_in", "fc_ksi", "bar_size", "top_bars", "bot_bars", "side_bars", "group_id", "stirrup_spacing_in"):
                self.assertEqual(row[key], member[key], (row["ele_tag"], key))
            self.assertEqual((row["node_i"], row["node_j"], row["element_type"]), (member["node_i"], member["node_j"], member["member_type"]))
        sizes = {row["b_in"] for row in rows if row["group_id"] in (BASE_INTERIOR, TOP_INTERIOR)}
        self.assertEqual(sizes, {30.0, 32.0})
        _index, attributes = graph._edge_arrays(graph.collect_graph_edge_rows(rows))
        self.assertEqual(attributes.shape, (2 * len(rows), 14))                                 # the edge feature contract is unchanged
        parameters = graph.collect_global_parameters()
        for key in graph.UNIFORM_SECTION_PARAMETER_KEYS:
            self.assertIsNone(parameters[key], key)
        self.assertEqual((parameters["design_mode"], parameters["member_groups_sha256"]), ("grouped", self.state.identity()))
        self.assertEqual(parameters["member_groups"][TOP_INTERIOR]["b_in"], 30.0)
        self.assertEqual(parameters["reinforcement_geometry"]["schema_version"], "smrf_reinforcement_geometry_by_group_v1")
        self.assertEqual(set(parameters["reinforcement_geometry"]["groups"]), set(self.state.groups))
        self.assertEqual(parameters["floor_loads"]["ledger"]["grouped"], True)
        json.dumps(parameters)
        mg.clear()
        uniform = graph.collect_global_parameters()                                             # the uniform mode writes what it always did
        self.assertEqual(uniform["b_col_in"], gf.COLUMN.b_in)
        self.assertNotIn("design_mode", uniform)
        self.assertEqual([k for k in parameters if k not in uniform],
                         ["design_mode", "member_groups_sha256", "member_groups", "uniform_section_parameters"])

    def test_a_short_time_history_exports_group_columns_and_the_hybrid_vector_refuses_the_case(self):
        import Ground_Motion_Main as gm
        import Hybrid_Exporter
        gr.apply_grouped_design(self.record)
        out = self.folder / "ntha"
        out.mkdir()
        accel_x, accel_y = pulse(0.25), pulse(0.2, phase=0.7)
        np.savetxt(self.folder / "pulse_x.txt", accel_x)
        np.savetxt(self.folder / "pulse_y.txt", accel_y)
        record_x = GroundMotionRecord("PULSE_X", 0.01, accel_x, units="g", source_path=str(self.folder / "pulse_x.txt"))
        record_y = GroundMotionRecord("PULSE_Y", 0.01, accel_y, units="g", source_path=str(self.folder / "pulse_y.txt"))
        summary = gm.run_one(record_x, record_y, argparse.Namespace(damping_ratio=0.05, rayleigh_mode_i=0, rayleigh_mode_j=2, dt_factor=1.0), out)
        self.assertFalse(summary["status"]["failed"])
        parameters = json.loads((out / "global_parameters.json").read_text(encoding="utf-8"))
        self.assertEqual((parameters["design_mode"], parameters["member_groups_sha256"], parameters["b_col_in"]),
                         ("grouped", self.state.identity(), None))
        with (out / "hinge_backbone.csv").open(newline="", encoding="utf-8") as handle:
            hinges = list(csv.DictReader(handle))
        saved = {str(row["member_tag"]): row["group_id"] for row in self.record["member_groups"]["members"]}
        self.assertEqual(list(hinges[0])[-6:], ["group_id", "band", "location_class", "story_or_floor", "grid_i", "grid_j"])
        for row in hinges:
            self.assertEqual(row["group_id"], saved[row["ele_tag"]])
        _arrays, springs, schema = hmr.read_hinge_moment_rotation(out)
        self.assertTrue(schema["member_groups"]["present"])
        self.assertEqual({row["group_id"] for row in springs}, set(self.state.groups))
        for key in ("design_mode", "member_groups_sha256"):
            self.assertIn(key, gm.OUTPUT_IDENTITY_KEYS)
        gm.validate_ntha_output_compatibility(out)                                              # the same design may resume
        with mg.installed(gs.resized(self.state, TOP_INTERIOR, (32.0, 32.0))):
            with self.assertRaisesRegex(RuntimeError, "member_groups_sha256"):
                gm.validate_ntha_output_compatibility(out)                                      # another grouped design may not
        with self.assertRaisesRegex(ValueError, "grouped design"):
            Hybrid_Exporter.compile_hybrid_sample(out)


if __name__ == "__main__":
    unittest.main()
