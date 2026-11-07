"""Beam cage threading evidence, member-paired tied pricing, the joint-aware cage coordination and factor separation
(the lower-perimeter cage audit of case_0007 / o0013, 2026-10-31).

  G  the lanes of the audited 32 in core with twenty #10 bars (three #9 per layer, no #10 or #11) against an independent
     grid enumeration and the audit's recorded geometry; the 34/36 in and sixteen-#11 look-ahead geometry; the passage
     clearance reported as the detailing policy it is;
  P  a tied X/Y cage priced on each member's own demand and strength (the final check's pairing): the union of the tie's
     demands is a bound only and does not eliminate a cage the final check accepts;
  T  the threading-constrained deficit detected while the best nonthreading cage is still above 1, apart from
     lane_limited; every refusal named by code and class, the practical steel ceiling a policy and not a code limit,
     nonthreading cages counted by cause and never refused or admitted for it;
  C  the coordinated repair: a column cage change that opens lanes is adopted only with a stronger, settled selection;
     a change the strong-column rule refuses stays refused; a repeated state is refused; an alternating selection is
     recorded as a cycle; a failed coordination leaves the selection as it was and is not run again; the selected state
     is restored after every trial; caches keyed on the bar grid and the layer policy; with two height bands the band
     that meets the blocked one at the boundary joints keeps the lanes its beams thread today, and the chain report says
     how many cage pairs each boundary rule cut;
  R  R changes the ELF coefficient between its bounds (the Cs lower bounds included), never the fixed-section probable
     strengths or the joint mechanism.

Synthetic demands on the fixture frame (no analysis): nothing here is a designed structure or an accepted design.
"""
import json
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402

import Structure_Parameters as sp                                               # noqa: E402
from Design import Group_Capacity as gc                                          # noqa: E402
from Design import Group_Checks as checks                                        # noqa: E402
from Design import Group_Selection as selection                                  # noqa: E402
from Design import SMRF_Cage_Geometry as geometry                                # noqa: E402
from Design import SMRF_Joints as joints                                         # noqa: E402
from Design.Config import DesignConfig                                           # noqa: E402
from Loads import Seismic_ELF as elf                                             # noqa: E402
from Model import Member_Bar_Layers as bar_layers                                # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402

# The audit's recorded geometry of the 32 in core (GEOMETRY_AND_R.json, case baseline_32in_20no10, both directions).
RECORDED_COLUMN_BARS_IN_BAND = [8.0, 13.5, 18.5, 24.0]
RECORDED_NO9_GRID_POSITIONS = [10.5, 11.0, 16.0, 21.0, 21.5]
RECORDED_NO9_LANES = [10.5, 16.0, 21.0]
# The perimeter frames carry the lateral load of the system fixture at this seismic demand (test_system_rules).
RULES_SEISMIC = {"ASCE_SDS": 0.6, "ASCE_SD1": 0.36, "ASCE_S1": 0.36}
BEAM = replace(gf.BEAM, fc_ksi=5.0, bot_bars=4)
# A congested column cage (sixteen #9): the tie's stronger cages do not thread it.
DENSE_COLUMN = replace(gf.COLUMN, fc_ksi=6.0, bar_size=9, top_bars=5, bot_bars=5, side_bars=3)
# A 32 in core with twenty #10 bars (the audited cage) at 8 ksi: some cages that open its lanes keep the strong column.
COORDINATION_COLUMN = replace(gf.COLUMN, b_in=32.0, h_in=32.0, fc_ksi=8.0, bar_size=10, top_bars=6, bot_bars=6, side_bars=4)
X, Y = "s01_04__beam_x__edge", "s01_04__beam_y__edge"


def audited_core(size_in, beam_bar, bars_per_face=8, column_bar=10, top=6, side=4):
    """column_cage_admits_beam_bars for the audited joint: a square core of twenty #10 (six per b face, four interior
    per h face) behind 1.5 in cover and #4 hoops, a 22 in beam with #4 hoops, 3/4 in aggregate, two layers."""
    return geometry.column_cage_admits_beam_bars(size_in, size_in, 1.5, 0.5, sp.rebar_diameter(column_bar), top, side,
                                                 22.0, 1.5, 0.5, sp.rebar_diameter(beam_bar), bars_per_face, 0.75,
                                                 max_layers=2)


def enumerated_lanes(direction, beam_db, column_db, grid):
    """Independent of thread_bars: every grid point within the centre limits clear of the column bars, then the longest
    sequence of mutually clear points."""
    lo, hi = direction["bar_centre_limits_in"]
    clearance = direction["clearance_used_in"]
    gap = 0.5 * (beam_db + column_db) + clearance
    points = [i * grid for i in range(math.ceil(lo / grid - 1e-9), math.floor(hi / grid + 1e-9) + 1)
              if all(abs(i * grid - o) >= gap - 1e-9 for o in direction["column_bar_lanes_blocked_in"])]
    paths = []
    for point in points:
        before = [p for p in paths if point - p[-1] >= beam_db + clearance - 1e-9]
        paths.append((max(before, key=len) if before else []) + [point])
    return points, max(paths, key=len, default=[])


def system_state(column=DENSE_COLUMN, beam=BEAM):
    return mg.GroupedDesign.uniform(gf.NX, gf.NY, gf.NZ, column, beam, policy_id=mg.SYSTEM_POLICY_ID)


def synthetic_actions(state, beams=None, column_axial=100.0, column_moment=50.0):
    """Two combinations without an analysis: each beam group its (sagging, hogging) envelope from ``beams`` (100 kip-in
    otherwise), each column a small concurrent axial load and moment."""
    actions = []
    for cid in ("synthetic_1", "synthetic_2"):
        members = {}
        for member in state.members():
            if member.is_column:
                m = column_moment
                members[str(member.member_tag)] = {
                    "local_force_kip_kipin": [column_axial, 1.0, 1.0, 0.0, m, m, -column_axial, -1.0, -1.0, 0.0, m, m],
                    "axial_i_kip": column_axial, "axial_j_kip": column_axial}
            else:
                positive, negative = (beams or {}).get(member.group_id, (100.0, 100.0))
                members[str(member.member_tag)] = {
                    "local_force_kip_kipin": [0.0] * 12,
                    "span_bending": {"full_span": {"mu_positive_kip_in": positive, "mu_negative_kip_in": negative}}}
        actions.append({"id": cid, "members": members})
    return actions


class AuditedCoreGeometry(unittest.TestCase):
    def setUp(self):
        self.frame = gf.frame()
        self.frame.__enter__()

    def tearDown(self):
        self.frame.__exit__(None, None, None)

    def test_three_no9_lanes_and_none_for_no10_or_no11_match_the_independent_enumeration(self):
        for bar, lanes in ((9, 3), (10, 0), (11, 0)):
            result = audited_core(32.0, bar)
            self.assertEqual(result["bars_per_layer"], {"x": lanes, "y": lanes})
            for direction in result["directions"].values():
                self.assertEqual(direction["column_bar_lanes_blocked_in"], RECORDED_COLUMN_BARS_IN_BAND)
                points, best = enumerated_lanes(direction, sp.rebar_diameter(bar), sp.rebar_diameter(10), 0.5)
                self.assertEqual(len(best), lanes)
                if bar == 9:
                    self.assertEqual(points, RECORDED_NO9_GRID_POSITIONS)
                    self.assertEqual(direction["threaded_positions_in"], RECORDED_NO9_LANES)
        # Two layers of three: six #9 per face is the most that threads; a seventh needs a third layer.
        self.assertTrue(audited_core(32.0, 9, bars_per_face=6)["passes"])
        self.assertFalse(audited_core(32.0, 9, bars_per_face=7)["passes"])

    def test_larger_cores_and_fewer_larger_column_bars_are_geometry_only(self):
        # Same twenty-#10 cage: 34 in admits one #10 or #11 per layer, 36 in three; sixteen #11 at 32 in two of each.
        for size_in, lanes in ((34.0, {9: 3, 10: 1, 11: 1}), (36.0, {9: 3, 10: 3, 11: 3})):
            for bar, count in lanes.items():
                self.assertEqual(audited_core(size_in, bar)["bars_per_layer"], {"x": count, "y": count}, (size_in, bar))
        for bar in (9, 10, 11):
            self.assertEqual(audited_core(32.0, bar, column_bar=11, top=5, side=3)["bars_per_layer"], {"x": 2, "y": 2})

    def test_the_passage_clearance_is_reported_as_the_detailing_policy_it_is(self):
        policy = geometry.passage_policy(sp.rebar_diameter(9))
        self.assertEqual((policy["kind"], policy["relaxed"]), ("construction_policy", False))
        self.assertIn("not a code minimum", policy["passage_clearance"]["basis"])
        self.assertEqual((policy["bar_position_grid_in"], policy["max_layers_per_face"]), (0.5, sp.BEAM_BAR_MAX_LAYERS))
        # the value the lanes were threaded with
        self.assertAlmostEqual(audited_core(32.0, 9)["directions"]["x"]["clearance_used_in"],
                               policy["passage_clearance"]["value_in"], places=12)


class MemberPairedPricing(unittest.TestCase):
    def test_a_cage_the_union_bound_rejects_passes_on_each_members_own_demand_and_strength(self):
        with gf.frame(**RULES_SEISMIC):
            state = mg.install(system_state(column=replace(gf.COLUMN, fc_ksi=6.0)))
            cfg = _cfg()
            rows = bar_layers.arrangement()
            weakest = {gid: {sign: min(checks.resolved_beam_flexural_strengths(state.member(tag), rows[gid])[sign]["phi_mn_kip_in"]
                                       for tag in checks.beam_group_representatives(state, gid))
                             for sign in ("positive", "negative")} for gid in (X, Y)}
            sign = max(("positive", "negative"),
                       key=lambda s: abs(weakest[X][s] - weakest[Y][s]) / max(weakest[X][s], weakest[Y][s]))
            strong, weak = (X, Y) if weakest[X][sign] > weakest[Y][sign] else (Y, X)
            self.assertLess(weakest[weak][sign], 0.97 * weakest[strong][sign])        # the two groups really differ
            index = 0 if sign == "positive" else 1
            envelopes = {strong: [1.0, 1.0], weak: [1.0, 1.0]}
            envelopes[strong][index] = 0.98 * weakest[strong][sign]
            envelopes[weak][index] = 0.5 * weakest[weak][sign]
            actions = synthetic_actions(state, {gid: tuple(v) for gid, v in envelopes.items()})
            union = tuple(max(v[i] for v in envelopes.values()) for i in (0, 1))
            for gid in (X, Y):
                demand_rows = checks.beam_demand_rows(state.groups[gid]["member_tags"], actions)
                tags, paired = selection._paired_beam_demand(state, gid, demand_rows)
                priced = selection.evaluate_beam_candidate(state, gid, state.designs[gid], union, _cfg(), tags,
                                                           paired_demand=paired)
                final = checks.beam_group_strength(state.designs[gid], demand_rows, rows[gid], state)
                authoritative = max(final["flexure_positive"]["dcr"], final["flexure_negative"]["dcr"])
                self.assertEqual(priced["dcr_basis"], "member_paired")
                self.assertAlmostEqual(priced["dcr"], authoritative, places=12)
                self.assertLessEqual(authoritative, 1.0)
                if gid == weak:
                    # the union pairs the strong group's demand with this group's strength: a bound, not a member DCR
                    self.assertGreater(priced["dcr_union_bound"], 1.0)
            _design, report = selection.select_beam_groups(state, (X, Y), actions, cfg)
            self.assertEqual((report["status"], report["dcr_basis"]), ("selected", "member_paired"))
            self.assertLessEqual(report["dcr"], cfg.dcr.dcr_hard_max)
            self.assertGreaterEqual(report["dcr_union_bound"], report["dcr"] - 1e-12)


class ThreadingConstrainedDeficit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gf.frame(**RULES_SEISMIC):
            cls.state = system_state()
            with mg.installed(cls.state):
                cls.actions = synthetic_actions(cls.state, {X: (8000.0, 8000.0), Y: (8000.0, 8000.0)})
                cls.design, cls.report = selection.select_beam_groups(cls.state, (X, Y), cls.actions, _cfg())
            cls.light = system_state(column=replace(gf.COLUMN, fc_ksi=6.0))
            with mg.installed(cls.light):
                cls.light_report = selection.select_beam_groups(cls.light, (X, Y), cls.actions, _cfg())[1]

    def test_the_mixed_deficit_is_detected_while_the_best_nonthreading_cage_is_still_short(self):
        report, evidence = self.report, self.report["threading"]
        self.assertEqual(report["status"], "strength_short")
        self.assertFalse(report["lane_limited"])                     # no cage reaches the demand, threading or not ...
        self.assertTrue(evidence["threading_constrained_deficit"])   # ... yet threading keeps a stronger cage out
        self.assertTrue(evidence["authenticated"])
        best_t, best_n = evidence["best_threading"], evidence["best_nonthreading"]
        self.assertGreater(best_n["dcr"], 1.0)
        self.assertLess(best_n["dcr"], best_t["dcr"])
        self.assertEqual((best_t["threads"], best_n["threads"]), (True, False))
        self.assertEqual(selection._cage(self.design), {k: best_t[k] for k in ("bar_size", "top_bars", "bot_bars")})
        self.assertAlmostEqual(report["dcr"], best_t["dcr"], places=12)
        self.assertTrue(evidence["blocking_cores"])
        self.assertTrue(all(self.state.designs[core].is_column for core in evidence["blocking_cores"]))
        self.assertTrue(set(evidence["blocking_cores"]) <= set(evidence["core_groups"]))
        self.assertTrue(evidence["prospects"])
        self.assertTrue(all(p["dcr"] < best_t["dcr"] for p in evidence["prospects"]))
        for gid in (X, Y):
            lanes = evidence["lanes"]["strongest_nonthreading"][gid]
            self.assertFalse(lanes["threads"])
            self.assertTrue(set(lanes["governing_cores"]) <= set(evidence["blocking_cores"]))
            # the installed rows say what set every raised offset, with the joint type where it did
            for face in ("top", "bottom"):
                rows = evidence["rows"][gid][face]
                self.assertEqual(len(rows["set_by"]), len(rows["per_layer"]))
                for control in rows["set_by"]:
                    if control is not None:
                        self.assertIn(control["by"], ("own_layer_pitch", "crossing_layer", "slab_mat"))
                        self.assertTrue(control["joints"])
        # The same demand with a lighter column cage: threading constrains nothing, the deficit is strength alone.
        light = self.light_report["threading"]
        self.assertEqual(self.light_report["status"], "strength_short")
        self.assertFalse(light["threading_constrained_deficit"])
        self.assertNotIn("prospects", light)

    def test_refusals_name_code_and_class_and_nonthreading_cages_are_counted_by_cause(self):
        evidence = self.report["threading"]
        for gid in (X, Y):
            codes = evidence["first_refusals"][gid]
            self.assertTrue(set(codes) <= set(selection.BEAM_REFUSALS), codes)
            classes = {}
            for code, count in codes.items():
                kind = selection.BEAM_REFUSALS[code][0]
                classes[kind] = classes.get(kind, 0) + count
            self.assertEqual(evidence["refusal_classes"][gid], classes)
        self.assertEqual(sum(evidence["nonthreading_by_cause"].values()),
                         evidence["admissible_cages"] - evidence["threading_cages"])
        self.assertTrue(set(evidence["nonthreading_by_cause"]) <= set(selection.THREADING_BLOCKS))
        # The classes: section code limits, the declared construction rules, missing evidence.
        kinds = {code: value[0] for code, value in selection.BEAM_REFUSALS.items()}
        for code in ("minimum_steel", "steel_ratio_code", "tension_control", "strength_balance", "width_fit",
                     "through_joint_depth"):
            self.assertEqual(kinds[code], "hard_section", code)
        for code in ("steel_ratio_practical", "row_past_mid_depth"):
            self.assertEqual(kinds[code], "construction_policy", code)
        for code in ("bar_layers_unsettled", "rows_invalid"):
            self.assertEqual(kinds[code], "missing_evidence", code)

    def test_the_practical_steel_ceiling_is_a_policy_refusal_not_the_code_limit(self):
        with gf.frame(**RULES_SEISMIC), mg.installed(self.state):
            cfg = _cfg()
            design = self.state.designs[X]
            refused = None
            for candidate in selection.beam_cage_candidates(design, cfg):
                result = selection.evaluate_beam_candidate(self.state, X, candidate, (1.0, 1.0), cfg)
                if result.get("reason_code") == "steel_ratio_practical":
                    refused = candidate
                    break
            self.assertIsNotNone(refused, "the fixture has a cage between the practical ceiling and the code limit")
            self.assertEqual(result["reason_class"], "construction_policy")
            relaxed = _cfg()
            relaxed.rebar.rho_beam_practical_max = 1.0
            with self.assertRaisesRegex(ValueError, "authorized"):
                selection.evaluate_beam_candidate(self.state, X, refused, (1.0, 1.0), relaxed)
            relaxed.rebar.rho_beam_practical_max = 0.0175              # a permissible relaxation of the base default
            again = selection.evaluate_beam_candidate(self.state, X, refused, (1.0, 1.0), relaxed)
            self.assertNotIn(again.get("reason_code"), ("steel_ratio_practical", "steel_ratio_code"))
            minimum = selection.evaluate_beam_candidate(self.state, X, replace(design, bar_size=4, top_bars=2, bot_bars=2),
                                                        (1.0, 1.0), cfg)
            self.assertEqual((minimum["reason_code"], minimum["reason_class"]), ("minimum_steel", "hard_section"))


class CachesAndRestoration(unittest.TestCase):
    def test_lanes_are_never_served_from_another_bar_grid(self):
        with gf.frame():
            column = mg.MemberDesign("column", 32.0, 32.0, 6.0, 10, 6, 6, 4, 4, 5, 3.0, (5, 5))
            beam = mg.MemberDesign("beam_x", 22.0, 36.0, 5.0, 10, 6, 6, 0, 4, 4, 6.0)
            on_grid = bar_layers.bars_per_layer(column, beam, "x")
            sp.BAR_POSITION_GRID_IN = None
            even = bar_layers.bars_per_layer(column, beam, "x")
            fresh = geometry.column_cage_admits_beam_bars(32.0, 32.0, sp.COL_CLEAR_COVER_IN, 0.5, sp.rebar_diameter(10), 6, 4,
                                                          22.0, sp.BEAM_CLEAR_COVER_IN, 0.5, sp.rebar_diameter(10), 6,
                                                          sp.AGGREGATE_MAX_SIZE_IN, grid_in=None)["bars_per_layer"]["x"]
            self.assertEqual(on_grid, 0)
            self.assertEqual(even, fresh)
            self.assertNotEqual(even, on_grid)
            sp.BAR_POSITION_GRID_IN = 0.5
            self.assertEqual(bar_layers.bars_per_layer(column, beam, "x"), on_grid)

    def test_rows_follow_the_layer_policy_and_the_row_controls_do_not_change_them(self):
        with gf.frame(**RULES_SEISMIC):
            # One bar more than a layer holds at the tie's cores: two layers under the two-layer rule, one row under one.
            lanes = bar_layers.bars_per_layer(DENSE_COLUMN, replace(BEAM, top_bars=12, bot_bars=12), "x")
            self.assertGreaterEqual(lanes, 2)
            state = mg.install(system_state(beam=replace(BEAM, top_bars=lanes + 1, bot_bars=lanes + 1)))
            two = bar_layers.arrangement()
            self.assertEqual((two[X]["top"]["layers"], two["_basis"]["max_layers"]), (2, 2))
            explained = bar_layers.row_controls()
            self.assertEqual(bar_layers.arrangement(), two)               # explaining the rows changed nothing
            self.assertEqual({face: len(v) for face, v in explained[X].items()},
                             {face: two[X][face]["layers"] for face in ("top", "bottom")})
            sp.BEAM_BAR_MAX_LAYERS = 1
            one = bar_layers.arrangement()
            self.assertEqual(one["_basis"]["max_layers"], 1)
            self.assertNotEqual(one[X], two[X])                           # not the two-layer rows from the cache
            sp.BEAM_BAR_MAX_LAYERS = 2
            self.assertEqual(bar_layers.arrangement(), two)
            self.assertIs(mg.active(), state)

    def test_a_grade_change_reprices_the_same_cage(self):
        with gf.frame(**RULES_SEISMIC):
            state = mg.install(system_state())
            cfg, design = _cfg(), state.designs[X]
            first = selection.evaluate_beam_candidate(state, X, design, (4000.0, 4000.0), cfg)
            stronger = state.with_tied_designs({state.tie_of(X): replace(design, fc_ksi=8.0)})
            other = selection.evaluate_beam_candidate(stronger, X, replace(design, fc_ksi=8.0), (4000.0, 4000.0), cfg)
            again = selection.evaluate_beam_candidate(state, X, design, (4000.0, 4000.0), cfg)
            self.assertLess(other["dcr"], first["dcr"])
            self.assertEqual(again["dcr"], first["dcr"])
            self.assertIs(mg.active(), state)


class RuleAtTheRoof(unittest.TestCase):
    """The fixture's strong-column refusals are at the roof, whose synthetic axial loads are below Ag f'c / 10: these
    tests exercise coordination under 18.7.3.2 there, so the 18.7.3.1 exception is held off (tested on its own in
    tests/test_roof_exception_grouped.py)."""

    @classmethod
    def setUpClass(cls):
        cls._rule = mock.patch.object(joints, "SCWB_DISCONTINUOUS_COLUMN_EXCEPTION", False)
        cls._rule.start()
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls._rule.stop()


class CoordinatedTrial(RuleAtTheRoof):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with gf.frame(**RULES_SEISMIC):
            cls.state = system_state(column=COORDINATION_COLUMN)
            cls.cfg = _cfg()
            with mg.installed(cls.state):
                cls.actions = synthetic_actions(cls.state, {X: (6000.0, 6000.0), Y: (6000.0, 6000.0)})
                cls.unit = selection.select_beam_groups(cls.state, (X, Y), cls.actions, cls.cfg)[1]

    def prospect(self, bar, count):
        return next(p for p in self.unit["threading"]["prospects"] if (p["bar_size"], p["top_bars"]) == (bar, count))

    def trial(self, prospect, margin=1.0, visited=None):
        with gf.frame(**RULES_SEISMIC), mg.installed(self.state):
            identity = self.state.identity()
            outcome = selection._coordinated_trial(self.state, self.state.tie_of(X), self.unit, prospect, self.actions,
                                                   self.cfg, margin, 4, {identity} if visited is None else visited, {X, Y})
            self.assertIs(mg.active(), self.state)                      # the selected state is back in force
            self.assertEqual(self.state.identity(), identity)           # and unchanged
        return outcome

    def test_a_column_cage_change_that_opens_lanes_is_adopted_with_a_stronger_settled_selection(self):
        self.assertTrue(self.unit["threading"]["threading_constrained_deficit"])
        prospect = self.prospect(10, 4)
        outcome = self.trial(prospect)
        log = outcome["log"]
        self.assertTrue(outcome["adopted"], log)
        self.assertLess(log["dcr_after"], log["dcr_before"])
        self.assertEqual(outcome["reserved"], sorted(prospect["blocking_cores"]))
        new = outcome["state"]
        domain = {(c.bar_size, c.top_bars, c.side_bars)
                  for c in selection.column_cage_candidates(COORDINATION_COLUMN, self.cfg, layout=mg.layout_of(new.policy_id))}
        for core in outcome["reserved"]:
            design = new.designs[core]
            self.assertNotEqual(design, self.state.designs[core])               # the cage changed ...
            self.assertEqual((design.b_in, design.h_in, design.fc_ksi), (32.0, 32.0, 8.0))    # ... never the section
            self.assertIn((design.bar_size, design.top_bars, design.side_bars), domain)
        with gf.frame(**RULES_SEISMIC), mg.installed(new):
            lanes = selection._lane_governors(new, (X, Y), new.designs[X])
        self.assertTrue(all(lanes[gid]["threads"] for gid in (X, Y)))          # the installed beam cage threads

    def test_a_change_the_strong_column_rule_refuses_stays_refused(self):
        outcome = self.trial(self.prospect(8, 7))
        log = outcome["log"]
        self.assertFalse(outcome["adopted"])
        self.assertEqual(log["status"], "failed")
        self.assertIn("no cage chain", log["reason"])
        bands = [band for line in log["column_lines"].values() for band in line["bands"].values()]
        self.assertTrue(any(band["rejected"].get("scwb", 0) > 0 for band in bands))
        self.assertIn("column_size_lookahead", log)
        # A larger strong-column margin refuses even the change that opens lanes at the plain margin.
        self.assertFalse(self.trial(self.prospect(10, 4), margin=4.0)["adopted"])

    def test_a_repeated_state_is_refused(self):
        visited = {self.state.identity()}
        self.assertTrue(self.trial(self.prospect(10, 4), visited=visited)["adopted"])
        again = self.trial(self.prospect(10, 4), visited=visited)
        self.assertFalse(again["adopted"])
        self.assertTrue(again["log"]["reason"].startswith("repeated_state"), again["log"]["reason"])


class CoupledBands(unittest.TestCase):
    """Two height bands (six stories, bands of three under the three-story system policy)."""
    LOWER = ("s01_03__beam_x__edge", "s01_03__beam_y__edge")
    UPPER = ("s04_06__beam_x__edge", "s04_06__beam_y__edge")
    FRAME = {"NUM_FLOOR": 6, **RULES_SEISMIC}

    @classmethod
    def setUpClass(cls):
        with gf.frame(**cls.FRAME):
            cls.state = mg.GroupedDesign.uniform(gf.NX, gf.NY, 6, COORDINATION_COLUMN, BEAM, policy_id="system_three_story_v1")
            cls.cfg = _cfg()
            with mg.installed(cls.state):
                cls.actions = synthetic_actions(cls.state, {gid: (6000.0, 6000.0) for gid in cls.LOWER})
                cls.unit = selection.select_beam_groups(cls.state, cls.LOWER, cls.actions, cls.cfg)[1]
        cls.lower_cores = sorted(g for g in cls.state.groups if g.startswith("s01_03__column__") and "interior" not in g)
        cls.upper_cores = sorted(g for g in cls.state.groups if g.startswith("s04_06__column__") and "interior" not in g)

    def test_the_band_above_is_coupled_only_where_its_beams_thread_today(self):
        with gf.frame(**self.FRAME):
            self.assertEqual(selection._adjacent_band_cores(self.state, set(self.lower_cores)), (self.upper_cores, self.upper_cores))
            self.assertEqual(selection._adjacent_band_cores(self.state, set(self.upper_cores)), (self.lower_cores, self.lower_cores))
            # Upper beams with more bars than two layers of lanes carry: there is no open lane to keep there.
            lanes = bar_layers.bars_per_layer(COORDINATION_COLUMN, replace(BEAM, top_bars=14, bot_bars=14), "x")
            crowded = self.state.with_tied_designs({self.state.tie_of(self.UPPER[0]): replace(
                self.state.designs[self.UPPER[0]], top_bars=2 * lanes + 1, bot_bars=2 * lanes + 1)})
            self.assertEqual(selection._adjacent_band_cores(crowded, set(self.lower_cores)), (self.upper_cores, []))

    def test_a_coupled_trial_keeps_the_lanes_of_the_band_above(self):
        self.assertTrue(self.unit["threading"]["threading_constrained_deficit"])
        prospect = next(p for p in self.unit["threading"]["prospects"] if (p["bar_size"], p["top_bars"]) == (8, 7))
        with gf.frame(**self.FRAME), mg.installed(self.state):
            outcome = selection._coordinated_trial(self.state, self.state.tie_of(self.LOWER[0]), self.unit, prospect,
                                                   self.actions, self.cfg, 1.0, 4, {self.state.identity()}, set(self.LOWER))
            self.assertIs(mg.active(), self.state)
        log = outcome["log"]
        self.assertTrue(outcome["adopted"], log)
        self.assertEqual(log["adjacent_lanes_kept"], self.upper_cores)
        self.assertTrue(set(self.upper_cores) <= set(outcome["reserved"]))
        self.assertLess(log["dcr_after"], log["dcr_before"])
        line = log["column_lines"]["sfrs_column"]
        upper_band = next(band for key, band in line["bands"].items() if key.startswith("s04_06"))
        self.assertIn("beam_lanes", upper_band["rejected"])                      # threading was required up there too
        (boundary, counts), = line["links"].items()
        self.assertTrue(boundary.startswith("s01_03") and ">s04_06" in boundary)
        self.assertGreaterEqual(counts["pairs"], counts["transition"] + counts["strong_column"] + 1)
        new = outcome["state"]
        with gf.frame(**self.FRAME), mg.installed(new):
            for gids in (self.LOWER, self.UPPER):
                lanes = selection._lane_governors(new, gids, new.designs[gids[0]])
                self.assertTrue(all(lanes[gid]["threads"] for gid in gids), (gids, lanes))

    def test_the_chain_report_names_the_boundary_rule_that_cut_each_pair(self):
        # Another concrete grade above: no joint material rule is declared, so every cage pair is cut by the transition.
        with gf.frame():
            gid = "s03_04__column__interior"
            state = mg.install(gf.uniform_state().with_designs({gid: replace(gf.uniform_state().designs[gid], fc_ksi=6.0)}))
            _selected, report = selection.select_column_chain(state, "interior", synthetic_actions(state), _cfg())
            self.assertEqual(report["status"], "no_feasible_chain")
            counts = report["links"][f"s01_02__column__interior>{gid}"]
            self.assertGreater(counts["pairs"], 0)
            self.assertEqual((counts["transition"], counts["strong_column"]), (counts["pairs"], 0))
            self.assertTrue(counts["examples"])
            self.assertTrue(all(e["cut_by"] == "transition" and e["detail"] for e in counts["examples"]))


class SelectionWithCoordination(RuleAtTheRoof):
    def test_an_alternating_selection_is_recorded_as_a_cycle(self):
        with gf.frame(**RULES_SEISMIC):
            state = mg.install(system_state())
            actions = synthetic_actions(state, {X: (6000.0, 6000.0), Y: (6000.0, 6000.0)})
            _selected, report = selection.select_reinforcement(state, actions, _cfg())
            cycle = report["cycle"]
            self.assertFalse(report["settled"])
            self.assertLess(cycle["repeats_pass"], cycle["pass"])
            self.assertEqual(len(report["passes"]), cycle["pass"])              # the passes stopped at the proof
            self.assertIn("unsettled", report["coordination"]["basis"])
            self.assertIs(mg.active(), state)

    def test_a_failed_coordination_leaves_the_selection_as_it_was_and_is_not_run_again(self):
        with gf.frame(**RULES_SEISMIC):
            state = mg.install(system_state())
            actions = synthetic_actions(state, {X: (6000.0, 6000.0), Y: (6000.0, 6000.0)})
            cfg = _cfg()
            plain, base = selection.select_reinforcement(state, actions, cfg, scwb_margin=4.0, coordinate=False)
            selected, report = selection.select_reinforcement(state, actions, cfg, scwb_margin=4.0)
            log = report["coordination"]
            self.assertEqual(log["status"], "no_coordinated_change")
            self.assertTrue(log["trials"])
            self.assertLessEqual(len(log["trials"]), selection.COORDINATION_MAX_TRIALS)
            self.assertTrue(all(trial["status"] == "failed" and trial["reason"] for trial in log["trials"]))
            self.assertEqual(selected.identity(), plain.identity())
            self.assertEqual((report["exhausted_groups"], report["reserved_cores"]), (base["exhausted_groups"], []))
            again, second = selection.select_reinforcement(selected, actions, cfg, scwb_margin=4.0)
            self.assertTrue(second["coordination"]["reused"])
            self.assertEqual(second["coordination"]["trials"], log["trials"])
            self.assertEqual(again.identity(), selected.identity())
            self.assertIs(mg.active(), state)
            # what the evaluation record keeps is plain JSON
            json.dumps({"threading": selection.threading_constraint_summary(report),
                        "coordination": selection.coordination_summary(report)}, allow_nan=False)


class FactorSeparation(unittest.TestCase):
    def coefficient(self, r, period):
        sp.ASCE_R = r
        return elf.seismic_response_coefficient(period)

    def test_r_scales_the_elf_coefficient_only_between_its_bounds(self):
        with gf.frame(ASCE_SDS=1.0, ASCE_SD1=0.6, ASCE_S1=0.5, ASCE_IE=1.25, ASCE_TL=8.0):
            # Cs_max (12.8-3) governs at both R: exactly proportional to 1/R.
            a, b = self.coefficient(8.0, 1.0), self.coefficient(1.0, 1.0)
            self.assertEqual((a["governing_bound"], b["governing_bound"]), ("Cs_max (12.8-3)", "Cs_max (12.8-3)"))
            self.assertAlmostEqual(b["cs"] / a["cs"], 8.0, places=12)
            # At a long period Cs_min = 0.044 SDS Ie (12.8-5) governs at R = 8 and does not scale with R.
            a, b = self.coefficient(8.0, 6.0), self.coefficient(1.0, 6.0)
            self.assertEqual(a["governing_bound"], "Cs_min (12.8-5/6)")
            self.assertAlmostEqual(a["cs"], 0.044 * 1.0 * 1.25, places=12)
            self.assertLess(b["cs"] / a["cs"], 8.0)
            # With S1 >= 0.6 the 12.8-6 bound 0.5 S1 / (R / Ie) scales with R itself.
            sp.ASCE_S1 = 0.75
            a, b = self.coefficient(8.0, 6.0), self.coefficient(1.0, 6.0)
            self.assertAlmostEqual(a["cs_min"], 0.5 * 0.75 / (8.0 / 1.25), places=12)
            self.assertAlmostEqual(b["cs_min"] / a["cs_min"], 8.0, places=12)

    def test_r_leaves_the_fixed_section_probable_strengths_and_joint_mechanism_unchanged(self):
        with gf.frame(**RULES_SEISMIC):
            state = mg.install(system_state(column=replace(gf.COLUMN, fc_ksi=6.0)))
            entry = next(e for e in gc.joint_inventory(state)["joints"]
                         if e["level"] == 1 and e["grid_j"] == 0 and 0 < e["grid_i"] < gf.NX)
            members = {m.member_tag: m for m in entry["beams"]["x"] + entry["beams"]["y"]}
            results = {}
            for r in (8.0, 1.0):
                sp.ASCE_R = r
                strengths = {tag: gc.beam_probable_strengths(member) for tag, member in members.items()}
                joint = gc.joint_shear(entry, "x", strengths, None, {}, gc.JOINT_RULE_SFRS)
                results[r] = (strengths, {k: joint[k] for k in ("vj_kip", "column_shear_kip", "beam_face_forces_kip",
                                                                "steel_stress_factor", "beam_moment_basis")})
            self.assertEqual(results[8.0], results[1.0])
            self.assertEqual(results[8.0][1]["steel_stress_factor"], 1.25)


def _cfg():
    # The deficits these tests build exist under the earlier 0.015 practical ceiling, set explicitly (2026-11-07).
    from Design.Config import with_beam_practical_ratio
    return with_beam_practical_ratio(DesignConfig(), 0.015)


def setUpModule():
    # These tests verify the 3D physical detail route, which the computed common-action route keeps (2026-11-07).
    global _DETAIL_BASIS
    _DETAIL_BASIS = sp.DETAIL_BASIS
    sp.DETAIL_BASIS = "physical_3d_v2"


def tearDownModule():
    sp.DETAIL_BASIS = _DETAIL_BASIS


if __name__ == "__main__":
    unittest.main()
