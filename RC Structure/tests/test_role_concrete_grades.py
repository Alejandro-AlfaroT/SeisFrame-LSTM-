"""Concrete grades by structural role (Design.Concrete_Grades, decision DEC-SFV6-GRAVITY-CONCRETE-20261023, 2026-10-23).

Gravity columns of the two-system policies take their own ladder, 5, 6 and 7 ksi from 5 ksi, separate from the
special-frame columns (6-10 ksi); the beams keep one shared grade on the special-frame beam ladder until a slab,
pour-boundary and flange-concrete rule for two beam grades on one floor is decided. Checked here: the families of
each policy, the starting grades of a seed, the classification under the core ring and the core lines (a member the
layout brings into the SFRS takes the special-frame grade), grade moves that touch only the failing group's family,
the 7 ksi ceiling, the validator of new and legacy records, and the family summary a record and its reports carry.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf
import Structure_Parameters as sp
from Design import Concrete_Grades as cg
from Design import Grouped_Record as gr
from Design import Grouped_Search as gs
from Design.Section_Design import (BEAM_CONCRETE_STRENGTHS_KSI, COLUMN_CONCRETE_STRENGTHS_KSI,
                                   GRAVITY_BEAM_CONCRETE_STRENGTHS_KSI, GRAVITY_COLUMN_CONCRETE_STRENGTHS_KSI)
from Model import Member_Groups as mg

SEED_COLUMN, SEED_BEAM = (20.0, 20.0, 6.0), (16.0, 24.0, 5.0)


def column_grades(state):
    """{system: {grades}} of the column groups."""
    out = {}
    for gid, group in state.groups.items():
        if group["member_type"] == "column":
            out.setdefault(state.system_of(gid), set()).add(state.designs[gid].fc_ksi)
    return out


def evaluation(**constraints):
    base = {"exhausted_groups": [], "exhausted_column_lines": {}, "capacity_design_failed_checks": [],
            "joint_checks_failed": [], "drift_failed_checks": [], "joint_scwb": {"all_pass": True}}
    return {"status": "evaluated", "constraints": {**base, **constraints}}


class Ladders(unittest.TestCase):
    def test_the_decision_and_the_unchanged_special_frame_ladders(self):
        self.assertEqual(GRAVITY_COLUMN_CONCRETE_STRENGTHS_KSI, (5.0, 6.0, 7.0))
        self.assertEqual(GRAVITY_BEAM_CONCRETE_STRENGTHS_KSI, (5.0, 6.0, 7.0))
        self.assertEqual(COLUMN_CONCRETE_STRENGTHS_KSI, (6.0, 7.0, 8.0, 9.0, 10.0))
        self.assertEqual(BEAM_CONCRETE_STRENGTHS_KSI, (5.0, 6.0, 8.0))
        self.assertEqual(gs.GRADE_LADDERS, {"column": COLUMN_CONCRETE_STRENGTHS_KSI, "beam": BEAM_CONCRETE_STRENGTHS_KSI})

    def test_families_by_policy(self):
        band = cg.families(mg.POLICY_ID)
        self.assertEqual(set(band), {"column", "beam"})
        self.assertIsNone(band["column"]["start_ksi"])
        for policy in (mg.SYSTEM_POLICY_ID, mg.SYSTEM_BANDED_POLICY_ID, mg.SYSTEM_CORE_BANDED_POLICY_ID,
                       mg.SYSTEM_CORE_LINES_BANDED_POLICY_ID):
            families = cg.families(policy)
            self.assertEqual(set(families), {"sfrs_column", "gravity_column", "beam"}, policy)
            self.assertEqual(families["gravity_column"]["ladder"], (5.0, 6.0, 7.0))
            self.assertEqual(families["gravity_column"]["start_ksi"], 5.0)
            self.assertEqual(families["sfrs_column"]["ladder"], COLUMN_CONCRETE_STRENGTHS_KSI)
            self.assertEqual(families["beam"]["ladder"], BEAM_CONCRETE_STRENGTHS_KSI)
            self.assertEqual(families["beam"]["systems"], (mg.SYSTEM_SFRS, mg.SYSTEM_GRAVITY))
        self.assertFalse(cg.SEPARATE_GRAVITY_BEAMS)
        self.assertIn("slab", cg.GRAVITY_BEAM_RULE)


class Seeds(unittest.TestCase):
    def setUp(self):
        self.frame = gf.frame(NUM_BAY_X=4, NUM_BAY_Y=4, NUM_FLOOR=4)
        self.frame.__enter__()
        self.addCleanup(self.frame.__exit__, None, None, None)

    def test_gravity_columns_start_at_5_and_the_special_frame_at_the_seed(self):
        state, seed = gr.seed_from_sections(SEED_COLUMN, SEED_BEAM, policy_id=mg.SYSTEM_BANDED_POLICY_ID)
        self.assertEqual(column_grades(state), {mg.SYSTEM_SFRS: {6.0}, mg.SYSTEM_GRAVITY: {5.0}})
        self.assertEqual({d.fc_ksi for d in state.designs.values() if not d.is_column}, {5.0})
        self.assertEqual(seed["concrete_grades"]["families"], {"sfrs_column": 6.0, "gravity_column": 5.0, "beam": 5.0})
        self.assertEqual(seed["concrete_grades"]["policy"], cg.POLICY_ID)
        # sizes stay the seed's: only the grade differs by role
        self.assertEqual({(d.b_in, d.h_in) for d in state.designs.values() if d.is_column}, {(20.0, 20.0)})

    def test_no_silent_seed_grade_for_gravity_columns(self):
        """A 4 ksi or a 10 ksi seed column changes the special-frame start, never the gravity start."""
        for fc in (4.0, 10.0):
            state, _seed = gr.seed_from_sections((20.0, 20.0, fc), SEED_BEAM, policy_id=mg.SYSTEM_BANDED_POLICY_ID)
            self.assertEqual(column_grades(state), {mg.SYSTEM_SFRS: {fc}, mg.SYSTEM_GRAVITY: {5.0}})

    def test_the_band_policy_keeps_the_seed_grades(self):
        state, seed = gr.seed_from_sections(SEED_COLUMN, SEED_BEAM, policy_id=mg.POLICY_ID)
        self.assertEqual({d.fc_ksi for d in state.designs.values() if d.is_column}, {6.0})
        self.assertEqual(seed["concrete_grades"]["families"], {"column": 6.0, "beam": 5.0})


class Classification(unittest.TestCase):
    def _seed(self, nx, ny, policy):
        with gf.frame(NUM_BAY_X=nx, NUM_BAY_Y=ny, NUM_FLOOR=3):
            return gr.seed_from_sections(SEED_COLUMN, SEED_BEAM, policy_id=policy)[0]

    def test_core_ring_promotes_interior_columns_to_the_special_frame_grade(self):
        perimeter = self._seed(5, 3, mg.SYSTEM_BANDED_POLICY_ID)
        ring = self._seed(5, 3, mg.SYSTEM_CORE_BANDED_POLICY_ID)
        for state in (perimeter, ring):
            for gid, group in state.groups.items():
                if group["member_type"] == "column":
                    expected = 5.0 if state.system_of(gid) == mg.SYSTEM_GRAVITY else 6.0
                    self.assertEqual(state.designs[gid].fc_ksi, expected, (state.policy_id, gid))
        # the ring takes interior columns into the SFRS: fewer gravity columns, every promoted one at 6 ksi
        def gravity_members(s):
            return [m for m in s.members("column") if s.system_of(m.group_id) == mg.SYSTEM_GRAVITY]
        self.assertLess(len(gravity_members(ring)), len(gravity_members(perimeter)))
        self.assertTrue(gravity_members(ring), "a 5 x 3 core ring keeps gravity columns")
        promoted = [g for g in ring.groups if ring.groups[g]["location_class"] in ("core_corner", "core_edge")]
        self.assertTrue(promoted)
        self.assertTrue(all(cg.family_of(ring, g) == "sfrs_column" and ring.designs[g].fc_ksi == 6.0 for g in promoted))
        self.assertTrue(all(m.design.fc_ksi == 5.0 for m in gravity_members(ring)))

    def test_a_layout_without_gravity_columns(self):
        lines = self._seed(3, 3, mg.SYSTEM_CORE_LINES_BANDED_POLICY_ID)
        self.assertEqual(cg.groups_of(lines, "gravity_column"), [])
        self.assertIsNone(cg.grade(lines, "gravity_column"))
        self.assertIsNone(cg.next_grade(lines, "gravity_column"))
        self.assertIs(cg.regraded(lines, "gravity_column", 6.0), lines)
        self.assertEqual(cg.describe(lines)["families"]["gravity_column"]["groups"], 0)
        self.assertEqual(cg.problems(lines), [])

    def test_every_group_has_exactly_one_family_and_members_carry_their_group_grade(self):
        with gf.frame(NUM_BAY_X=5, NUM_BAY_Y=3, NUM_FLOOR=4):
            for policy in (mg.SYSTEM_BANDED_POLICY_ID, mg.SYSTEM_CORE_BANDED_POLICY_ID, mg.SYSTEM_CORE_LINES_BANDED_POLICY_ID):
                state = gr.seed_from_sections(SEED_COLUMN, SEED_BEAM, policy_id=policy)[0]
                members = {f: set(cg.groups_of(state, f)) for f in cg.families(policy)}
                self.assertEqual(sum(len(v) for v in members.values()), len(state.groups))
                self.assertEqual(set().union(*members.values()), set(state.groups))
                for member in state.members():
                    self.assertEqual(member.design.fc_ksi, state.designs[member.group_id].fc_ksi)


class GradeMoves(unittest.TestCase):
    def setUp(self):
        self.frame = gf.frame(NUM_BAY_X=4, NUM_BAY_Y=4, NUM_FLOOR=6)
        self.frame.__enter__()
        self.addCleanup(self.frame.__exit__, None, None, None)
        self.state = gr.seed_from_sections(SEED_COLUMN, SEED_BEAM, policy_id=mg.SYSTEM_BANDED_POLICY_ID)[0]
        self.gravity_column = next(g for g in sorted(self.state.groups) if cg.family_of(self.state, g) == "gravity_column")
        self.sfrs_column = next(g for g in sorted(self.state.groups) if cg.family_of(self.state, g) == "sfrs_column")

    def test_a_gravity_failure_steps_only_the_gravity_columns(self):
        moves = gs.propose_growth(evaluation(capacity_design_failed_checks=[
            {"id": "gravity.column_capacity_shear", "location": self.gravity_column}]), self.state)
        self.assertEqual(set(moves), {"gravity_column_grade"})
        self.assertEqual(moves["gravity_column_grade"][0], 6.0)
        moved = gs.apply_moves(self.state, moves)
        self.assertEqual(column_grades(moved), {mg.SYSTEM_SFRS: {6.0}, mg.SYSTEM_GRAVITY: {6.0}})
        self.assertEqual(cg.grade(moved, "beam"), 5.0)
        # every band's gravity columns, not only the failing band's
        self.assertEqual({moved.designs[g].fc_ksi for g in cg.groups_of(moved, "gravity_column")}, {6.0})

    def test_a_special_frame_failure_does_not_raise_the_gravity_columns(self):
        moves = gs.propose_growth(evaluation(capacity_design_failed_checks=[
            {"id": "column.pm_strength", "location": self.sfrs_column}]), self.state)
        self.assertEqual(set(moves), {"sfrs_column_grade"})
        moved = gs.apply_moves(self.state, moves)
        self.assertEqual(column_grades(moved), {mg.SYSTEM_SFRS: {7.0}, mg.SYSTEM_GRAVITY: {5.0}})
        self.assertEqual(gs._move_rows(moves)[0]["kind"], "grade")
        self.assertIn("sfrs_column grade -> 7 ksi", gs._moves_summary(moves))

    def test_both_families_move_independently_in_one_proposal(self):
        moves = gs.propose_growth(evaluation(capacity_design_failed_checks=[
            {"id": "gravity.column_capacity_shear", "location": self.gravity_column},
            {"id": "column.pm_strength", "location": self.sfrs_column}]), self.state)
        self.assertEqual(set(moves), {"gravity_column_grade", "sfrs_column_grade"})
        moved = gs.apply_moves(self.state, moves)
        self.assertEqual(column_grades(moved), {mg.SYSTEM_SFRS: {7.0}, mg.SYSTEM_GRAVITY: {6.0}})

    def test_the_gravity_ceiling_is_7_ksi_and_then_the_size_grows(self):
        top = gs.regraded(self.state, "gravity_column", 7.0)
        self.assertIsNone(gs.next_grade(top, "gravity_column"))
        moves = gs.propose_growth(evaluation(capacity_design_failed_checks=[
            {"id": "gravity.column_capacity_shear", "location": self.gravity_column}]), top)
        self.assertNotIn("gravity_column_grade", moves)
        self.assertIn(top.tie_of(self.gravity_column), moves)        # the size, never an 8 ksi gravity column
        self.assertEqual(cg.grade(gs.apply_moves(top, moves), "gravity_column"), 7.0)

    def test_the_special_frame_keeps_its_own_ladder(self):
        state = gs.regraded(self.state, "sfrs_column", 9.0)
        self.assertEqual(gs.next_grade(state, "sfrs_column"), 10.0)
        self.assertEqual(gs.next_grade(state, "gravity_column"), 6.0)
        self.assertEqual(gs.next_grade(gs.regraded(state, "sfrs_column", 10.0), "sfrs_column"), None)

    def test_an_unknown_family_is_refused(self):
        with self.assertRaises(mg.GroupedStateError):
            gs.regraded(self.state, "column", 7.0)                  # a member type is not a two-system family
        with self.assertRaises(mg.GroupedStateError):
            gs.common_grade(self.state, "gravity_beam")              # not a family while the gravity beams share the grade


class Validator(unittest.TestCase):
    def setUp(self):
        self.frame = gf.frame(NUM_BAY_X=4, NUM_BAY_Y=4, NUM_FLOOR=4)
        self.frame.__enter__()
        self.addCleanup(self.frame.__exit__, None, None, None)
        self.state = gr.seed_from_sections(SEED_COLUMN, SEED_BEAM, policy_id=mg.SYSTEM_BANDED_POLICY_ID)[0]

    def test_a_role_mixed_frame_passes_the_role_policy_and_fails_the_legacy_rule(self):
        self.assertEqual(cg.problems(self.state), [])
        legacy = cg.problems(self.state, cg.LEGACY_POLICY_ID)
        self.assertTrue(legacy and "column groups use different concrete grades" in legacy[0])
        with self.assertRaises(mg.GroupedStateError):
            cg.require(self.state, cg.LEGACY_POLICY_ID)

    def test_a_gravity_grade_off_its_ladder_is_refused_with_the_reason(self):
        for bad in (8.0, 4.0, 10.0):
            state = gs.regraded(self.state, "gravity_column", bad)
            issues = cg.problems(state)
            self.assertEqual(len(issues), 1, bad)
            self.assertIn("outside its ladder", issues[0])
            self.assertIn(cg.DECISION_ID, issues[0])
            with self.assertRaisesRegex(mg.GroupedStateError, "gravity_column grade"):
                cg.require(state)

    def test_two_grades_inside_one_family_are_refused(self):
        gids = cg.groups_of(self.state, "gravity_column")
        self.assertGreaterEqual(len(gids), 2)
        from dataclasses import replace
        split = self.state.with_designs({gid: replace(self.state.designs[gid], fc_ksi=6.0)
                                         for gid in gids if self.state.groups[gid]["band"] == self.state.groups[gids[0]]["band"]})
        issues = cg.problems(split)
        self.assertTrue(any("different concrete grades" in i for i in issues))

    def test_unknown_policy_and_record_policy(self):
        self.assertTrue(cg.problems(self.state, "made_up_v9"))
        self.assertEqual(cg.policy_of_record({"materials": {}}), cg.LEGACY_POLICY_ID)
        self.assertEqual(cg.policy_of_record({"materials": {"concrete_grade_policy": cg.POLICY_ID}}), cg.POLICY_ID)

    def test_the_design_guard_uses_the_policy(self):
        from Design import Grouped_Design as gd
        gd._require_family_grades(self.state)                       # the role policy: passes
        with self.assertRaises(mg.GroupedStateError):
            gd._require_family_grades(self.state, cg.LEGACY_POLICY_ID)


class Summary(unittest.TestCase):
    def setUp(self):
        self.frame = gf.frame(NUM_BAY_X=4, NUM_BAY_Y=4, NUM_FLOOR=4)
        self.frame.__enter__()
        self.addCleanup(self.frame.__exit__, None, None, None)
        self.state = gr.seed_from_sections(SEED_COLUMN, SEED_BEAM, policy_id=mg.SYSTEM_BANDED_POLICY_ID)[0]

    def test_the_record_reports_each_family_and_the_gravity_beams(self):
        state = gs.regraded(gs.regraded(self.state, "sfrs_column", 10.0), "beam", 8.0)
        summary = cg.describe(state)
        families = summary["families"]
        self.assertEqual({f: row["fc_ksi"] for f, row in families.items()}, {"sfrs_column": 10.0, "gravity_column": 5.0, "beam": 8.0})
        self.assertTrue(families["sfrs_column"]["at_top_rung"])
        self.assertFalse(families["gravity_column"]["at_top_rung"])
        self.assertEqual(families["gravity_column"]["groups"], len(cg.groups_of(state, "gravity_column")))
        gravity_beams = summary["gravity_beams"]
        self.assertEqual(gravity_beams["fc_ksi"], 8.0)
        self.assertFalse(gravity_beams["within_gravity_ladder"])     # stated, not hidden: the shared grade is above 7 ksi
        self.assertIn("slab", gravity_beams["rule"])
        self.assertTrue(cg.describe(self.state)["gravity_beams"]["within_gravity_ladder"])
        self.assertIn("15.5(a)", summary["joint_concrete_basis"])

    def test_identity_block_and_last_candidate_grades(self):
        block = cg.identity_block()
        self.assertEqual(block["gravity_column_ladder_ksi"], [5.0, 6.0, 7.0])
        self.assertFalse(block["separate_gravity_beams"])
        sections = {gid: {"fc_ksi": d.fc_ksi} for gid, d in self.state.designs.items()}
        self.assertEqual(cg.family_grades(mg.SYSTEM_BANDED_POLICY_ID, sections),
                         {"beam": [5.0], "gravity_column": [5.0], "sfrs_column": [6.0]})

    def test_the_materials_block_names_no_single_column_grade_for_a_mixed_frame(self):
        from Design.Config import DesignConfig
        materials = gr._materials(DesignConfig.from_structure_parameters(), self.state)
        self.assertIsNone(materials["fc_col_ksi"])                   # 6 and 5 ksi columns: no first-grade pick
        self.assertEqual(materials["fc_beam_ksi"], 5.0)
        self.assertEqual(materials["concrete_grade_policy"], cg.POLICY_ID)
        self.assertEqual(materials["concrete_grades"]["families"]["gravity_column"]["fc_ksi"], 5.0)


class RecordGrades(unittest.TestCase):
    """A saved record is read under the grade policy it names; the reload path routes by that policy."""

    def setUp(self):
        self.frame = gf.frame(NUM_BAY_X=4, NUM_BAY_Y=4, NUM_FLOOR=4)
        self.frame.__enter__()
        self.addCleanup(self.frame.__exit__, None, None, None)
        self.state = gr.seed_from_sections(SEED_COLUMN, SEED_BEAM, policy_id=mg.SYSTEM_BANDED_POLICY_ID)[0]

    def record(self, state, policy=cg.POLICY_ID, block=None):
        import json
        materials = {"fy_ksi": sp.FY_KSI, "es_ksi": sp.ES_KSI, "fyt_ksi": sp.FYT_KSI,
                     "reinforcement_specification": sp.REINFORCEMENT_SPECIFICATION, "exposure": sp.MATERIAL_EXPOSURE,
                     "aggregate_size_in": 0.75, "beam_clear_cover_in": 1.5, "col_clear_cover_in": 1.5, "beam_bar_stacking": {}}
        if policy is not None:
            materials["concrete_grade_policy"] = policy
            materials["concrete_grades"] = cg.describe(state) if block is None else block
        record = {"schema_version": gr.SCHEMA, "design_mode": "grouped", "geometry": gr.driver._slab_geometry(),
                  "materials": materials, "member_groups": state.to_record(), "search": {}}
        return json.loads(json.dumps(gr._canonical(record)))          # as saved and read back

    def test_the_known_policy_pins_its_definition(self):
        self.assertEqual(cg.POLICY_ID, "role_concrete_grades_v1")
        self.assertEqual(cg.KNOWN_POLICIES[cg.POLICY_ID], cg._current_definition())
        from unittest import mock
        with mock.patch.object(cg, "SEPARATE_GRAVITY_BEAMS", True):     # the planned switch needs a new policy id
            self.assertNotIn(cg._current_definition(), list(cg.KNOWN_POLICIES.values()))

    def test_a_current_record_reads_back_and_its_wording_is_not_state(self):
        gr._check_record_grades(self.record(self.state), self.state)
        block = cg.describe(self.state)
        block["joint_concrete_basis"] += " (reworded)"
        block["families"]["gravity_column"]["basis"] = "reworded"
        gr._check_record_grades(self.record(self.state, block=block), self.state)

    def test_a_saved_grade_that_disagrees_names_the_field(self):
        block = cg.describe(self.state)
        block["families"]["gravity_column"]["fc_ksi"] = 7.0
        with self.assertRaisesRegex(ValueError, "families/gravity_column/fc_ksi"):
            gr._check_record_grades(self.record(self.state, block=block), self.state)

    def test_legacy_and_unknown_policies(self):
        common = gs.regraded(self.state, "gravity_column", 6.0)        # one column grade: a legacy record may hold it
        gr._check_record_grades(self.record(common, policy=None), common)
        with self.assertRaisesRegex(mg.GroupedStateError, "column groups use different concrete grades"):
            gr._check_record_grades(self.record(self.state, policy=None), self.state)
        with self.assertRaisesRegex(ValueError, "cannot be read: the record was written under concrete grade policy"):
            gr._check_record_grades(self.record(self.state, policy="role_concrete_grades_v9"), self.state)

    def test_the_reload_path_routes_by_the_record_policy(self):
        from unittest import mock

        class Reached(Exception):
            pass

        with mock.patch.object(gr, "_validated_slab", return_value={"thickness_in": 8.0}), \
                mock.patch.object(gr.driver, "_apply_slab", side_effect=Reached):
            with self.assertRaises(Reached):                             # the grade checks passed
                gr.apply_grouped_design(self.record(self.state))
            with self.assertRaisesRegex(mg.GroupedStateError, "column groups use different concrete grades"):
                gr.apply_grouped_design(self.record(self.state, policy=None))
            with self.assertRaisesRegex(ValueError, "cannot be read"):
                gr.apply_grouped_design(self.record(self.state, policy="role_concrete_grades_v9"))
        self.assertFalse(mg.is_grouped())                                # a refused install leaves nothing installed


if __name__ == "__main__":
    unittest.main()
