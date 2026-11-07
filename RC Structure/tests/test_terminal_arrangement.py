"""The terminal cage arrangement (2026-11-01, the four bar-row rejections of the 15-minute search run).

  A  the complete-arrangement row audit names every face whose rows do not fit, with what set them, and the capacity
     design refuses such a state before any calculation;
  C  a selection's terminal arrangement is closed: a column change that invalidates another beam unit's rows is closed
     by selecting the beams again on the final columns; where no beam cage fits those columns, the responsible column
     lines are restored to their last geometrically valid cages; a slab-mat layout and a hoop change that move rows are
     caught and closed before the capacity design is rebuilt; a closed selection stays unsettled and every unit's cage is
     the one its selection gives on the final state;
  I  where nothing closes it, the outcome is incomplete and structured: a genuine no-fit domain, alternating beam units
     and a repeated state, each classified, with the rejected and the last geometrically valid state apart and nothing
     priced on either; the evaluation record keeps that outcome, its call site and, with keep_actions, what replays it;
  R  review round 1: select_reinforcement closes the arrangement its own column pass invalidated; a beam step whose
     units still move is never adopted; the closure row carries the column reports in force; a restoration gives back
     longitudinal cages only, keeping the installed hoops and the line's exhaustion; a hoop change alone offers none;
     the seed's arrangement under the selected mats stops the candidate before any strength is priced;
     review round 2: a second closure on the same report keeps a restored line's exhaustion (taken from the last
     geometrically valid state's reports); an arrangement that does not settle, and a malformed row record, are raised
     at every gate as the errors they are, never a bar-row outcome.

Synthetic demands on the fixture frame (no analysis): nothing here is a designed structure or an accepted design.
"""
import copy
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402
from test_beam_threading_evidence import synthetic_actions                       # noqa: E402

import Structure_Parameters as sp                                               # noqa: E402
from Design import Group_Capacity as gc                                          # noqa: E402
from Design import Group_Selection as selection                                  # noqa: E402
from Design import Grouped_Design as gd                                          # noqa: E402
from Design.Config import DesignConfig                                           # noqa: E402
from Design.SMRF_Beam_Slab_Strength import BeamBarRowsError, CageArrangementError  # noqa: E402
from Model import Member_Bar_Layers as bar_layers                                # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402

RULES = {"ASCE_SDS": 0.6, "ASCE_SD1": 0.36, "ASCE_S1": 0.36}
LIGHT = replace(gf.COLUMN, fc_ksi=6.0)                                          # four lanes for the #8 beam bars
DENSE = replace(gf.COLUMN, fc_ksi=6.0, bar_size=9, top_bars=5, bot_bars=5, side_bars=3)   # two lanes
SHALLOW = replace(gf.BEAM, fc_ksi=5.0, h_in=16.0, top_bars=4, bot_bars=4)       # one layer beside LIGHT, two beside DENSE
# Four #8 bars a face and nothing else: the beam units cannot leave two layers beside the dense cage.
ONLY_FOUR_NO8 = replace(DesignConfig(), rebar=replace(DesignConfig().rebar, beam_n_range=(4, 4), bar_sizes_beam=[8]))


def system_state(column, beam):
    return mg.GroupedDesign.uniform(gf.NX, gf.NY, gf.NZ, column, beam, policy_id=mg.SYSTEM_POLICY_ID)


def slab_layout(size=6, spacing=12.0):
    """Slab mats on the fixture's 7 in slab (the record form the layering and the strengths read)."""
    return {"bar_size": size, "outer_axis": "x", "layers": {
        f"{axis}_{face}": {"bar_size": size, "bar_area_in2": sp.rebar_area(size), "spacing_in": spacing,
                           "effective_depth_in": gf.SLAB - 1.25 if face == "top" else 1.25, "axis": axis,
                           "layer": "outer" if axis == "x" else "inner", "clear_cover_outer_mat_in": 0.75}
        for axis in ("x", "y") for face in ("top", "bottom")}}


def selection_report(reference=None):
    """A selection report ending on one pass (its column units as selected), with ``reference`` audited as its last
    geometrically valid state when given."""
    report = {"passes": [{"pass": 1, "changed_groups": [], "beams": {}, "columns": {}}], "settled": False,
              "reserved_cores": []}
    if reference is not None:
        selection._note_geometry(reference, report, {"pass": 1, "phase": "columns"}, {})
    return report


class Fixture(unittest.TestCase):
    def setUp(self):
        self.frame = gf.frame(**RULES)
        self.frame.__enter__()
        self.valid = system_state(LIGHT, SHALLOW)
        self.actions = synthetic_actions(self.valid, {gid: (300.0, 300.0) for gid in self.valid.groups
                                                      if "__beam_" in gid})

    def tearDown(self):
        self.frame.__exit__(None, None, None)


# ---- A ---------------------------------------------------------------------------------------------------------------
class RowAudit(Fixture):
    def test_the_audit_names_each_face_that_does_not_fit_with_what_set_its_rows(self):
        self.assertTrue(bar_layers.row_audit(self.valid)["valid"])
        dense = system_state(DENSE, SHALLOW)
        audit = bar_layers.row_audit(dense)
        self.assertFalse(audit["valid"])
        self.assertEqual(audit["state_identity"], dense.identity())
        for item in audit["invalid"]:
            self.assertEqual(item["error"], "past_mid_depth")
            self.assertGreaterEqual(item["offsets_in"][-1], 0.5 * item["section_in"][1])
            raised = [c for c in item["controls"] if c]
            self.assertTrue(raised and all(c["joint_type"]["column"] for c in raised))

    def test_the_capacity_design_refuses_known_invalid_rows_before_any_calculation(self):
        dense = system_state(DENSE, SHALLOW)
        with mock.patch.object(gc, "beam_group_capacity", side_effect=AssertionError("priced")) as priced, \
                mg.installed(dense):
            with self.assertRaises(CageArrangementError) as caught:
                gc.build_group_capacity_design([], [], DesignConfig(), None, dense)
        self.assertEqual(priced.call_count, 0)
        self.assertEqual(caught.exception.evidence["outcome"], "capacity_refused_invalid_rows")
        self.assertTrue(caught.exception.evidence["group_ids"])


# ---- C ---------------------------------------------------------------------------------------------------------------
class TerminalClosure(Fixture):
    def test_a_column_change_that_invalidates_another_beam_unit_is_closed_on_the_final_columns(self):
        # Only the SFRS column cage changes; the gravity (interior) beams that frame into those columns lose their lanes.
        changed = self.valid.with_tied_designs({"sfrs_column": DENSE})
        invalid = {i["group"] for i in bar_layers.row_audit(changed)["invalid"]}
        self.assertIn("s01_04__beam_x__interior", invalid)
        report = selection_report(self.valid)
        closed = selection.close_terminal_arrangement(changed, self.actions, DesignConfig(), 1.0, report)
        log = report["terminal_arrangements"][-1]
        self.assertEqual((log["status"], log["adopted_route"]), ("closed", "beams_on_final_columns"))
        self.assertTrue(bar_layers.row_audit(closed)["valid"])
        # the column cages are the final ones, untouched; the dependencies name the changed SFRS columns
        self.assertEqual({g: d for g, d in closed.designs.items() if d.is_column},
                         {g: d for g, d in changed.designs.items() if d.is_column})
        self.assertTrue(set(log["dependencies"]["responsible_column_groups"])
                        & {g for g in changed.ties()["sfrs_column"]})
        row = report["passes"][-1]
        self.assertEqual(row["closure_route"], "beams_on_final_columns")
        self.assertFalse(report["settled"])
        # no stale unit result: every beam unit's own selection on the final state gives the cage it holds
        for label, gids in selection._beam_units(closed):
            design, _unit = selection.select_beam_groups(closed, gids, self.actions, DesignConfig())
            self.assertEqual(replace(design, member_type="x"), replace(closed.designs[gids[0]], member_type="x"))

    def test_when_no_beam_cage_fits_the_final_columns_the_responsible_lines_are_restored(self):
        changed = self.valid.with_tied_designs({"sfrs_column": DENSE, "gravity_column": DENSE})
        report = selection_report(self.valid)
        closed = selection.close_terminal_arrangement(changed, self.actions, ONLY_FOUR_NO8, 1.0, report)
        log = report["terminal_arrangements"][-1]
        first, second = log["routes"]
        self.assertEqual((first["route"], first["status"]), ("beams_on_final_columns", "failed"))
        self.assertTrue(first["units_without_admissible_cage"])
        self.assertEqual((second["route"], second["status"]), ("responsible_columns_restored", "adopted"))
        self.assertTrue(bar_layers.row_audit(closed)["valid"])
        self.assertEqual({g: d for g, d in closed.designs.items() if d.is_column},
                         {g: d for g, d in self.valid.designs.items() if d.is_column})
        self.assertTrue(all(item.get("restored") for item in report["passes"][-1]["columns"].values()))
        self.assertFalse(report["settled"])

    def test_a_slab_mat_layout_that_moves_rows_past_mid_depth_is_caught_and_closed(self):
        # Six #8 a face beside the light cage fit in 20 in without mats; #4 mats move the crossing rows past mid-depth.
        deep = system_state(LIGHT, replace(SHALLOW, h_in=20.0, top_bars=6, bot_bars=6))
        self.assertTrue(bar_layers.row_audit(deep)["valid"])
        sp.SLAB_REINFORCEMENT = {"layout": slab_layout(size=4)}
        audit = bar_layers.row_audit(deep)
        self.assertFalse(audit["valid"])
        self.assertTrue(any(c and c["by"] == "slab_mat" for item in audit["invalid"] for c in item["controls"]))
        report = selection_report()
        # 2026-11-05: with a mat layout installed, selection also screens the standard hooks of terminating rows, and no
        # hooked cage fits a 20 in beam whose rows sit below the mats. This test is about closing the row arrangement,
        # so the hook screens (tested in test_hook_tail_selection_screen and test_review_corrections_20261105) are off.
        with mock.patch.object(selection, "_has_terminating_end", return_value=False):
            closed = selection.close_terminal_arrangement(deep, self.actions, DesignConfig(), 1.0, report)
        self.assertEqual(report["terminal_arrangements"][-1]["status"], "closed")
        self.assertTrue(bar_layers.row_audit(closed)["valid"])

    def test_a_hoop_change_that_moves_rows_is_closed_before_the_capacity_design_is_rebuilt(self):
        # #6 bars, four a face, beside the dense cage: #5 hoops fit at 8.5 in, #6 hoops put the last row at mid-depth.
        fits = system_state(DENSE, replace(SHALLOW, h_in=18.0, bar_size=6, stirrup_bar_size=5))
        self.assertTrue(bar_layers.row_audit(fits)["valid"])
        hooped = {gid: replace(design, stirrup_bar_size=6) for gid, design in fits.designs.items() if not design.is_column}
        self.assertFalse(bar_layers.row_audit(fits.with_designs(hooped))["valid"])
        report = selection_report(fits)
        report["settled"], report["exhausted_groups"] = True, []
        seen = []

        def capacity(actions, expected_ids, cfg, transfers, state, **kwargs):
            seen.append((state.identity(), bar_layers.row_audit(state)["valid"]))
            return {"checks": [], "accepted": True, "scwb": {"joints": []}, "transverse": {}, "joint_types": {},
                    "joint_evidence": {}}

        with mock.patch.object(selection, "select_reinforcement", return_value=(fits, report)), \
                mock.patch.object(gc, "build_group_capacity_design", side_effect=capacity), \
                mock.patch.object(gc, "hoop_updates", return_value=hooped), \
                mock.patch("Design.SMRF_Joints.evaluate_joints", return_value=[]):
            result = gd.reinforcement_pass(fits, self.actions, [], DesignConfig(), None, max_passes=1, scwb_margins=(1.0,))
        self.assertTrue(all(valid for _identity, valid in seen))
        self.assertEqual(seen[-1][0], result["state"].identity())
        self.assertTrue(bar_layers.row_audit(result["state"])["valid"])
        log = report["terminal_arrangements"][-1]
        self.assertEqual((log["context"], log["status"]), ("reinforcement_pass_after_hoops", "closed"))
        self.assertTrue(result["evidence_rebuilt_on_final_design"])
        self.assertFalse(result["settled"])
        self.assertFalse(result["selection_settled"])


# ---- I ---------------------------------------------------------------------------------------------------------------
class IncompleteOutcome(Fixture):
    def test_a_genuine_no_fit_domain_is_an_incomplete_outcome_with_both_states_apart(self):
        changed = self.valid.with_tied_designs({"sfrs_column": DENSE, "gravity_column": DENSE})
        # No geometrically valid state at all; then one (deeper beams) whose column cages are the final ones, so no
        # column line is responsible and nothing can be restored.
        deeper = system_state(DENSE, replace(SHALLOW, h_in=24.0))
        for reference in (None, deeper):
            report = selection_report(reference)
            with self.assertRaises(CageArrangementError) as caught:
                selection.close_terminal_arrangement(changed, self.actions, ONLY_FOUR_NO8, 1.0, report,
                                                     inputs={"call": "select_reinforcement", "input_state": changed})
            evidence = caught.exception.evidence
            self.assertEqual(evidence["classification"], "no_fitting_beam_cage")
            self.assertIn("not proof", evidence["reason"])
            self.assertEqual(evidence["rejected_state"]["state_identity"], changed.identity())
            self.assertTrue(all(row["controls"] for row in evidence["rejected_state"]["invalid_rows"]))
            if reference is None:
                self.assertIsNone(evidence["last_geometrically_valid_state"])
            else:
                self.assertNotEqual(evidence["last_geometrically_valid_state"]["state_identity"], changed.identity())
            self.assertEqual([r["route"] for r in evidence["routes"]],
                             ["responsible_columns_restored", "beams_on_final_columns"])
            self.assertEqual(evidence["routes"][0]["status"], "not_available")
            self.assertIs(caught.exception.detail["rejected_state"], changed)
            json.dumps(evidence)                                    # JSON-safe as it stands

    def test_alternating_beam_units_and_a_repeated_state_are_classified(self):
        changed = self.valid.with_tied_designs({"sfrs_column": DENSE, "gravity_column": DENSE})
        report = selection_report(self.valid)
        with mock.patch.object(selection, "_beam_sweeps", return_value=(changed, False)):
            with self.assertRaises(CageArrangementError) as caught:
                selection.close_terminal_arrangement(changed, self.actions, DesignConfig(), 1.0, report)
        self.assertEqual(caught.exception.evidence["classification"], "beam_units_alternate")
        report = selection_report(self.valid)
        with mock.patch.object(selection, "_beam_sweeps", return_value=(changed, True)):
            with self.assertRaises(CageArrangementError) as caught:
                selection.close_terminal_arrangement(changed, self.actions, DesignConfig(), 1.0, report)
        self.assertEqual(caught.exception.evidence["classification"], "repeated_state")
        self.assertTrue(any(str(r.get("reason", "")).startswith("repeated_state") for r in caught.exception.evidence["routes"]))

    def test_the_record_keeps_the_incomplete_outcome_its_call_site_and_what_replays_it(self):
        changed = self.valid.with_tied_designs({"sfrs_column": DENSE, "gravity_column": DENSE})
        try:
            selection.close_terminal_arrangement(changed, self.actions, ONLY_FOUR_NO8, 1.0, selection_report(),
                                                 inputs={"call": "select_reinforcement", "input_state": self.valid})
        except CageArrangementError as exc:
            error = exc
        kept = {"actions": self.actions, "expected_ids": ["synthetic_1", "synthetic_2"]}
        for keep in (True, False):
            record = gd.bar_rows_outcome({"status": "evaluated"}, error, self.valid, kept, keep)
            self.assertEqual((record["status"], record["failed_stage"]), ("infeasible", "bar_rows"))
            failure = record["failure_evidence"]
            self.assertEqual((failure["exception"], failure["outcome"]), ("CageArrangementError",
                                                                         "terminal_arrangement_incomplete"))
            self.assertTrue(failure["call_site"]["raised_at"].startswith("Group_Selection.close_terminal_arrangement"))
            self.assertIs(record["state"], changed)
            json.dumps(failure)
            if keep:
                evidence = record["evidence"]
                self.assertTrue(evidence["incomplete"])
                self.assertIs(evidence["actions"], self.actions)
                self.assertIs(evidence["selection_input_state"], self.valid)
                self.assertIs(evidence["rejected_state"], changed)
            else:
                self.assertNotIn("evidence", record)


# ---- R ---------------------------------------------------------------------------------------------------------------
class ReviewRoundOne(Fixture):
    def test_select_reinforcement_closes_the_arrangement_its_own_column_pass_invalidated(self):
        # 18 in beams on light columns; a large column moment makes the one pass's column phase take cages with fewer
        # lanes, and the beams it priced before no longer fit beside them.
        state = system_state(LIGHT, replace(SHALLOW, h_in=18.0))
        actions = synthetic_actions(state, {gid: (1200.0, 1200.0) for gid in state.groups if "__beam_" in gid},
                                    column_axial=200.0, column_moment=8000.0)
        with mg.installed(state):
            closed, report = selection.select_reinforcement(state, actions, DesignConfig(), max_passes=1, coordinate=False)
        phases = {(row["phase"], row["pass"]): row["valid"] for row in report["geometry"]}
        self.assertTrue(phases[("beams", 1)])
        self.assertFalse(phases[("columns", 1)])                      # the column phase invalidated the beams' rows
        log = report["terminal_arrangements"][-1]
        self.assertEqual((log["status"], log["adopted_route"]), ("closed", "beams_on_final_columns"))
        self.assertTrue(bar_layers.row_audit(closed)["valid"])
        self.assertEqual(report["passes"][-1]["closure_route"], "beams_on_final_columns")
        self.assertFalse(report["settled"])
        self.assertEqual(report["pass_budget"], {"max_passes": 1, "used": 1, "exhausted": True})

    def test_a_beam_step_whose_units_still_move_is_never_adopted(self):
        changed = self.valid.with_tied_designs({"sfrs_column": DENSE})
        with mock.patch.object(selection, "_beam_sweeps", return_value=(self.valid, False)):
            with self.assertRaises(CageArrangementError) as caught:
                selection.close_terminal_arrangement(changed, self.actions, DesignConfig(), 1.0, selection_report(self.valid))
        self.assertEqual(caught.exception.evidence["classification"], "beam_units_alternate")
        self.assertTrue(all(r["status"] != "adopted" for r in caught.exception.evidence["routes"]))

    def test_the_closure_row_carries_the_column_reports_in_force(self):
        changed = self.valid.with_tied_designs({"sfrs_column": DENSE})
        report = selection_report(self.valid)
        chain = [[gid for gid in changed.groups if "__column__" in gid and "interior" not in gid]]
        report["columns_in_force"] = {"sfrs_column": {"location": "sfrs_column", "status": "no_feasible_chain",
                                                      "exhausted": True, "chain": chain}}
        report["passes"].append({"pass": 2, "changed_groups": [], "beams": {}, "columns": {},
                                 "beam_rows_problem": "a layer past mid-depth"})
        selection.close_terminal_arrangement(changed, self.actions, DesignConfig(), 1.0, report)
        self.assertTrue(report["passes"][-1]["columns"]["sfrs_column"]["exhausted"])
        self.assertTrue(set(chain[0]) <= set(report["exhausted_groups"]))

    def test_a_restoration_gives_back_longitudinal_cages_and_keeps_the_installed_hoops_and_exhaustion(self):
        changed = self.valid.with_tied_designs({"sfrs_column": replace(DENSE, stirrup_bar_size=5),
                                                "gravity_column": replace(DENSE, stirrup_bar_size=5)})
        report = selection_report(self.valid)
        report["columns_in_force"] = {"gravity_column": {"location": "gravity_column", "status": "no_feasible_chain",
                                                         "exhausted": True, "chain": [["s01_04__column__interior"]]}}
        closed = selection.close_terminal_arrangement(changed, self.actions, ONLY_FOUR_NO8, 1.0, report)
        self.assertEqual(report["terminal_arrangements"][-1]["adopted_route"], "responsible_columns_restored")
        for gid, design in closed.designs.items():
            if design.is_column:
                self.assertEqual((design.bar_size, design.top_bars, design.side_bars),
                                 (LIGHT.bar_size, LIGHT.top_bars, LIGHT.side_bars))
                self.assertEqual(design.stirrup_bar_size, 5)               # the installed hoops are kept
        self.assertTrue(report["passes"][-1]["columns"]["gravity_column"]["exhausted"])
        self.assertIn("installed_chain_on_final_beams", report["passes"][-1]["columns"]["gravity_column"])

    def test_a_hoop_change_alone_offers_no_restoration(self):
        invalid = system_state(DENSE, SHALLOW)
        audit = bar_layers.row_audit(invalid)
        reference = invalid.with_tied_designs({"sfrs_column": replace(DENSE, stirrup_bar_size=5),
                                               "gravity_column": replace(DENSE, stirrup_bar_size=5)})
        report = selection_report()
        report["last_geometrically_valid"] = {"state_identity": reference.identity(), "where": {"pass": 1},
                                              "designs": {g: mg.design_record(d) for g, d in reference.designs.items()},
                                              "columns": {}}
        dependencies = selection._row_dependencies(invalid, audit, report["last_geometrically_valid"])
        self.assertEqual((dependencies["responsible_column_groups"], dependencies["restore"]), ([], {}))
        self.assertTrue(dependencies["column_groups_with_only_other_changes"])

    def test_the_seed_arrangement_under_the_selected_mats_stops_before_any_strength(self):
        seed = system_state(LIGHT, replace(SHALLOW, h_in=20.0, top_bars=6, bot_bars=6))
        self.assertTrue(bar_layers.row_audit(seed)["valid"])
        layout = slab_layout(size=4)

        def slab_stage(cfg, context):
            sp.SLAB_REINFORCEMENT = {"layout": layout}
            return {"slab": {"thickness_in": gf.SLAB}, "attempts": [], "description": {"distinct": {}},
                    "reinforcement": {"layout": layout}, "evidence": None, "transfer": None}

        with mock.patch.object(gd, "slab_stage", slab_stage), mock.patch.object(gd, "_require_family_grades"), \
                mock.patch.object(gd, "story_strength_distribution", side_effect=AssertionError("priced")) as strength:
            cfg = DesignConfig()
            cfg.floor_analysis.detail_basis = "physical_3d_v2"            # the route whose crossing clearance pushes the rows
            record = gd.evaluate_candidate(seed, cfg, {}, keep_actions=True)
        self.assertEqual(strength.call_count, 0)
        self.assertEqual((record["status"], record["failed_stage"]), ("infeasible", "bar_rows"))
        failure = record["failure_evidence"]
        self.assertEqual((failure["outcome"], failure["classification"]),
                         ("seed_arrangement_invalid", "seed_rows_do_not_fit_the_selected_slab_mats"))
        self.assertFalse(record["stages"]["slab"]["seed_rows_fit"])
        self.assertIn("not proof", failure["reason"])


class ReviewRoundTwo(Fixture):
    def test_a_second_closure_on_the_same_report_keeps_a_restored_lines_exhaustion(self):
        changed = self.valid.with_tied_designs({"sfrs_column": DENSE, "gravity_column": DENSE})
        report = selection_report(self.valid)
        # The report in force says selected; only the reports of the last geometrically valid state carry the exhaustion
        # the restoration takes over (a second closure must not fall back to the stale report in force).
        report["columns_in_force"] = {"gravity_column": {"location": "gravity_column", "status": "selected",
                                                         "exhausted": False, "chain": [["s01_04__column__interior"]]}}
        report["last_geometrically_valid"]["columns"] = {
            "gravity_column": {"location": "gravity_column", "status": "no_feasible_chain", "exhausted": True,
                               "chain": [["s01_04__column__interior"]]}}
        closed = selection.close_terminal_arrangement(changed, self.actions, ONLY_FOUR_NO8, 1.0, report)
        self.assertEqual(report["terminal_arrangements"][-1]["adopted_route"], "responsible_columns_restored")
        self.assertEqual(report["columns_in_force"]["gravity_column"]["status"], "restored")
        chain = {gid for band in report["columns_in_force"]["gravity_column"]["chain"] for gid in band}
        self.assertTrue(chain and chain <= set(report["exhausted_groups"]))
        # A later closure on the same report (the after-hoops closure); an SFRS column change stands in for the hoops.
        trial = closed.with_tied_designs({"sfrs_column": DENSE})
        self.assertFalse(bar_layers.row_audit(trial)["valid"])
        selection.close_terminal_arrangement(trial, self.actions, DesignConfig(), 1.0, report,
                                             inputs={"call": "reinforcement_pass_after_hoops"})
        self.assertEqual(report["terminal_arrangements"][-1]["adopted_route"], "beams_on_final_columns")
        line = report["passes"][-1]["columns"]["gravity_column"]
        self.assertEqual((line["status"], line["exhausted"]), ("restored", True))
        self.assertTrue(chain <= set(report["exhausted_groups"]))

    def test_an_arrangement_that_does_not_settle_is_raised_at_every_gate(self):
        unsettled = mg.GroupedStateError("The top beam bar layers did not settle in 8 sweeps over the joints")

        def slab_stage(cfg, context):
            return {"slab": {"thickness_in": gf.SLAB}, "attempts": [], "description": {"distinct": {}},
                    "reinforcement": None, "evidence": None, "transfer": None}

        with mock.patch.object(bar_layers, "arrangement", side_effect=unsettled):
            audit = bar_layers.row_audit(self.valid)
            self.assertEqual((audit["valid"], audit["invalid"]), (False, []))
            self.assertIn("did not settle", audit["settle_error"])
            with self.assertRaisesRegex(mg.GroupedStateError, "did not settle"):
                selection.close_terminal_arrangement(self.valid, self.actions, DesignConfig(), 1.0, selection_report())
            with mg.installed(self.valid), self.assertRaisesRegex(mg.GroupedStateError, "did not settle"):
                gc.build_group_capacity_design([], [], DesignConfig(), None, self.valid)
            with mock.patch.object(gd, "slab_stage", slab_stage), mock.patch.object(gd, "_require_family_grades"), \
                    self.assertRaisesRegex(mg.GroupedStateError, "did not settle"):
                gd.evaluate_candidate(self.valid, DesignConfig(), {})

    def test_a_malformed_row_record_is_raised_at_every_gate_as_a_code_error(self):
        real = bar_layers.arrangement

        def malformed(*args, **kwargs):
            rows = copy.deepcopy(real(*args, **kwargs))
            rows["s01_04__beam_x__edge"]["top"]["centroid_in"] += 1.0      # a centroid the rows do not give
            return rows

        def slab_stage(cfg, context):
            return {"slab": {"thickness_in": gf.SLAB}, "attempts": [], "description": {"distinct": {}},
                    "reinforcement": None, "evidence": None, "transfer": None}

        def raised(call):
            with self.assertRaisesRegex(ValueError, "malformed beam bar rows") as caught:
                call()
            self.assertNotIsInstance(caught.exception, BeamBarRowsError)        # never a bar-row design outcome

        with mock.patch.object(bar_layers, "arrangement", side_effect=malformed):
            audit = bar_layers.row_audit(self.valid)
            self.assertEqual([(i["group"], i["face"], i["error"]) for i in audit["invalid"]],
                             [("s01_04__beam_x__edge", "top", "rows_invalid")])
            raised(lambda: selection.close_terminal_arrangement(self.valid, self.actions, DesignConfig(), 1.0,
                                                                selection_report()))
            with mg.installed(self.valid):
                raised(lambda: gc.build_group_capacity_design([], [], DesignConfig(), None, self.valid))
            with mock.patch.object(gd, "slab_stage", slab_stage), mock.patch.object(gd, "_require_family_grades"):
                raised(lambda: gd.evaluate_candidate(self.valid, DesignConfig(), {}))


def setUpModule():
    # These tests verify the 3D physical detail route, which the computed common-action route keeps (2026-11-07).
    global _DETAIL_BASIS
    _DETAIL_BASIS = sp.DETAIL_BASIS
    sp.DETAIL_BASIS = "physical_3d_v2"


def tearDownModule():
    sp.DETAIL_BASIS = _DETAIL_BASIS


if __name__ == "__main__":
    unittest.main()
