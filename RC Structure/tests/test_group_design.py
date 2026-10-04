"""Member strength, capacity design and reinforcement selection by design group (2026-10-02).

  A2  on the uniform design expanded into its groups, the group routines give the uniform routines' numbers:
      the solved actions, the governing member DCRs, the hoops, the capacity shears and the joint shears;
  A4  with one column group changed above a band boundary, the transition, the joint types and the
      strong-column evidence follow the physical members (two unlike columns at the boundary joints, one
      column at the roof, no roof exemption);
  A5  a factored axial load outside a section's strength domain is an error that names the group, never a
      clamped value, and the vectorised interaction used to price candidate cages is the scalar one.

The fixture frame is not a designed structure; its numbers are compared, not accepted.
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402

import Structure_Parameters as sp                                               # noqa: E402
from Design import Design_Driver as driver                                       # noqa: E402
from Design import Group_Capacity as capacity_design                             # noqa: E402
from Design import Group_Checks as checks                                        # noqa: E402
from Design import Group_Selection as selection                                  # noqa: E402
from Design.Config import DesignConfig                                           # noqa: E402
from Design.SMRF_Capacity_Design import build_capacity_design                    # noqa: E402
from Design.SMRF_Common import SectionAxialDomainError                           # noqa: E402
from Design.SMRF_Demands import strength_load_combinations                       # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402

BOUNDARY = "s01_02__column__interior>s03_04__column__interior"


def solve(capture, period=None):
    """Every strength combination on the model as it stands; ``capture`` reads the member actions.

    The eigen solver is repeatable to about 3e-9 only, so two solves that are to be compared share one period.
    """
    combinations = strength_load_combinations(sp.ASCE_SDS)
    period = driver._model_period() if period is None else period
    actions = []
    for combination in combinations:
        driver._analyze_combination(combination, period)
        actions.append({**combination, "analysis_succeeded": True, "axial_reference": "joint_faces",
                        "members": capture(combination["dead"])})
    return actions, [c["id"] for c in combinations], period


class UniformEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack = gf.frame()
        cls.stack.__enter__()
        cls.cfg = DesignConfig.from_structure_parameters()
        # The equivalence is between the two paths under one Ve rule: the grouped design is defined for the
        # column-own rule only, so the uniform reference reads it too (its own default is the joint-limited rule).
        cls.cfg.capacity.uniform_column_shear_method = None
        cls.legacy_actions, cls.ids, cls.period = solve(driver._capture_element_actions)
        column = beam = 0.0
        for action in cls.legacy_actions:
            a, b, _results = driver._governing_dcrs(cls.cfg, action["members"])
            column, beam = max(column, a), max(beam, b)
        cls.legacy_dcr = {"column": column, "beam": beam}
        # The grouped design does not declare the roof column extension (uniform design only, 2026-10-03), so
        # the uniform reference of this equivalence is built without it: one joint classification on both sides.
        from unittest import mock
        with mock.patch.object(sp, "ROOF_COLUMN_EXTENSION", False):
            cls.legacy = build_capacity_design(driver._capacity_state(cls.cfg, cls.legacy_actions))
        cls.state = mg.install(gf.uniform_state())
        cls.actions, _ids, _period = solve(checks.capture_member_actions, cls.period)
        cls.strengths = checks.strength_checks(cls.actions, cls.cfg)
        cls.capacity = capacity_design.build_group_capacity_design(cls.actions, cls.ids, cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.stack.__exit__(None, None, None)

    def test_solved_actions_are_the_uniform_ones_member_by_member(self):
        worst = 0.0
        for legacy, grouped in zip(self.legacy_actions, self.actions):
            self.assertEqual(set(legacy["members"]), set(grouped["members"]))
            for tag, member in legacy["members"].items():
                other = grouped["members"][tag]
                worst = max(worst, abs(member["axial_i_kip"] - other["axial_i_kip"]), abs(member["axial_j_kip"] - other["axial_j_kip"]),
                            max(abs(a - b) for a, b in zip(member["local_force_kip_kipin"], other["local_force_kip_kipin"])))
                self.assertEqual(other["group_id"], mg.resolve(int(tag)).group_id)
        self.assertLess(worst, 1e-9)

    def test_governing_member_dcrs_are_the_uniform_ones(self):
        for kind in ("column", "beam"):
            self.assertAlmostEqual(self.strengths["worst"][kind], self.legacy_dcr[kind], places=10)
        self.assertEqual(set(self.strengths["groups"]), set(self.state.groups))
        # every group names the member and combination that govern it
        for gid, result in self.strengths["groups"].items():
            governing = result["pm"] if result["member_type"] == "column" else result["flexure_negative"]
            self.assertIn(governing["member_tag"], self.state.groups[gid]["member_tags"])
            self.assertIn(governing["combination"], self.ids)

    def test_every_group_selects_the_uniform_hoops(self):
        beam, column = self.legacy["transverse"]["beam"], self.legacy["transverse"]["column"]
        for gid, hoops in self.capacity["transverse"].items():
            expected = column if self.state.designs[gid].is_column else beam
            self.assertEqual((hoops["bar_size"], hoops["legs"], hoops["spacing_in"]),
                             (expected["bar_size"], expected["legs"], expected["spacing_in"]), gid)
        # The fixture's written-down hoops are not the designed ones: the update installs exactly the selected hoops.
        for gid, design in capacity_design.hoop_updates(self.capacity, self.state).items():
            hoops = self.capacity["transverse"][gid]
            self.assertEqual((design.stirrup_bar_size, design.stirrup_spacing_in), (hoops["bar_size"], hoops["spacing_in"]))
            self.assertEqual(design.stirrup_legs, hoops["legs_model"] if design.is_column else hoops["legs"])

    def test_capacity_shears_envelope_to_the_uniform_values(self):
        self.assertAlmostEqual(max(data["ve_kip"] for data in self.capacity["beams"].values()), self.legacy["beams"]["ve_kip"], places=9)
        self.assertAlmostEqual(max(data["governing"]["ve_kip"] for data in self.capacity["columns"].values()),
                               self.legacy["columns"]["governing"]["ve_kip"], places=9)
        for data in self.capacity["columns"].values():
            self.assertAlmostEqual(data["vs_limit_kip"], self.legacy["columns"]["vs_limit_kip"], places=9)
            self.assertEqual(data["column_shear_method"], "column_own_probable_envelope_v3")

    def test_joint_shears_are_the_uniform_values(self):
        legacy = {(round(j["vj_kip"], 6), round(j["phi_vn_kip"], 6)) for j in self.legacy["joints"]["joints"].values()}
        grouped = {(round(item["vj_kip"], 6), round(item["phi_vn_kip"], 6))
                   for kind in self.capacity["joint_types"].values() for item in kind["shear"].values()}
        self.assertEqual(grouped, legacy)

    def test_every_transition_is_the_same_cage_and_the_design_is_accepted_like_the_uniform_one(self):
        self.assertEqual({t["kind"] for t in self.capacity["transitions"].values()}, {"same_cage"})
        self.assertEqual(self.capacity["accepted"], self.legacy["accepted"])
        self.assertEqual([c for c in self.capacity["checks"] if c["status"] == "fail"], [])

    def test_joint_inventory_covers_every_physical_joint_once(self):
        joints = [joint for kind in self.capacity["joint_types"].values() for joint in kind["joints"]]
        self.assertEqual(len(joints), len(set(joints)))
        self.assertEqual(len(joints), gf.NZ * (gf.NX + 1) * (gf.NY + 1))
        self.assertEqual(len(self.capacity["scwb"]["joints"]), len(joints))

    def test_vectorised_interaction_is_the_scalar_one(self):
        for gid in ("s01_02__column__interior", "s03_04__column__corner"):
            rows = checks.column_demand_rows(self.state.groups[gid]["member_tags"], self.actions)
            for design in (gf.COLUMN, gf.SMALL_COLUMN, replace(gf.COLUMN, b_in=14.0, h_in=14.0, top_bars=2, bot_bars=2, side_bars=0)):
                diagrams = checks.column_pm_diagrams(design)
                scalar = [checks.column_pm_dcr(diagrams, p, my, mz)["dcr"] for _t, _c, p, my, mz, _vy, _vz in rows]
                value, index = checks.column_pm_dcr_max(diagrams, rows)
                self.assertAlmostEqual(value, max(scalar), places=9)
                self.assertAlmostEqual(scalar[index], max(scalar), places=9)

    def test_a_tension_load_beyond_the_surface_is_a_named_failure_with_a_ratio_above_one_in_both_forms(self):
        diagrams = checks.column_pm_diagrams(gf.COLUMN)
        end = min(p for p, _m in diagrams["y"])
        tension = end - 50.0
        for my, mz in ((0.0, 0.0), (100.0, 100.0)):
            scalar = checks.column_pm_dcr(diagrams, tension, my, mz)
            self.assertEqual((scalar["basis"], scalar["domain_failure"]), ("axial_tension_outside_surface", "tension"))
            self.assertAlmostEqual(scalar["dcr"], tension / end, places=12)                      # positive and above one
            envelope = checks.column_pm_envelope(diagrams, [(1, "c", tension, my, mz, 0.0, 0.0)])
            self.assertAlmostEqual(envelope["dcr"], scalar["dcr"], places=12)
            self.assertEqual(envelope["domain_failures"], {"compression": 0, "tension": 1, "moment_at_axial_end": 0})
            self.assertTrue(envelope["outside_domain"])

    def test_a_cage_whose_axial_domain_a_demand_leaves_is_rejected_by_name_in_selection(self):
        # the group's own columns pulled 3000 kip in one combination: no cage of the section covers it
        gid = "s03_04__column__corner"
        tags = {str(tag) for tag in self.state.groups[gid]["member_tags"]}
        actions = []
        for index, action in enumerate(self.actions):
            members = dict(action["members"])
            if index == 0:
                for tag in tags:
                    force = list(members[tag]["local_force_kip_kipin"])
                    force[0] = -3000.0
                    members[tag] = {**members[tag], "local_force_kip_kipin": force}
            actions.append({**action, "members": members})
        _selected, report = selection.select_column_chain(self.state, "corner", actions, self.cfg)
        band = report["bands"][gid]
        self.assertEqual(band["candidates_admissible"], 0)
        self.assertGreater(band["rejected"]["axial_domain"], 0)
        self.assertEqual(band["rejected"]["strength"], 0)                                        # named as a domain failure, not as strength
        self.assertEqual((report["status"], report["blocked_at"]), ("no_feasible_chain", gid))

    def test_an_axial_load_outside_the_section_domain_names_its_group_and_is_not_clamped(self):
        gid = "s01_02__column__interior"
        slender = replace(gf.COLUMN, b_in=12.0, h_in=12.0, fc_ksi=1.5, bar_size=5, top_bars=2, bot_bars=2, side_bars=0)
        state = self.state.with_designs({gid: slender})
        with mg.installed(state):
            with self.assertRaises(SectionAxialDomainError) as caught:
                capacity_design.build_group_capacity_design(self.actions, self.ids, self.cfg, None, state)
        self.assertEqual(caught.exception.group_id, gid)
        self.assertIs(mg.active(), self.state)                                               # the trial did not stay installed

    def test_selection_settles_on_admissible_cages_and_is_a_fixed_point(self):
        state, report = selection.select_reinforcement(self.state, self.actions, self.cfg)
        self.assertTrue(report["settled"])
        self.assertEqual(report["exhausted_groups"], [])
        for gid, design in state.designs.items():
            original = self.state.designs[gid]
            self.assertEqual((design.b_in, design.h_in, design.fc_ksi), (original.b_in, original.h_in, original.fc_ksi))
            if design.is_column:
                cages = {(c.bar_size, c.top_bars, c.bot_bars, c.side_bars) for c in selection.column_cage_candidates(original, self.cfg)}
                self.assertIn((design.bar_size, design.top_bars, design.bot_bars, design.side_bars), cages)
                self.assertLessEqual(design.longitudinal_area_in2 / design.gross_area_in2, self.cfg.rebar.rho_col_practical_max + 1e-12)
        columns = report["passes"][-1]["columns"]
        self.assertEqual(set(columns), set(mg.COLUMN_LOCATIONS))
        for item in columns.values():
            self.assertEqual(item["status"], "selected")
            self.assertTrue(item["scwb_in_selection"])                                        # a slab is installed: the screen runs
            self.assertIn("proxy", item["scwb_screen"])                                       # but without a slab layout it is a proxy
        again, second = selection.select_reinforcement(state, self.actions, self.cfg)
        self.assertEqual(again.identity(), state.identity())
        self.assertEqual(second["passes"][0]["changed_groups"], [])
        with mg.installed(state):
            strengths = checks.strength_checks(self.actions, self.cfg, state)
            capacity = capacity_design.build_group_capacity_design(self.actions, self.ids, self.cfg, None, state)
        self.assertLessEqual(max(strengths["worst"].values()), self.cfg.dcr.dcr_hard_max + 1e-9)
        self.assertTrue(all(t["supported"] for t in capacity["transitions"].values()))

    def test_a_smaller_column_above_is_reached_by_an_offset_bend_chain_and_the_cage_pool_is_recorded(self):
        # 26 below, 24 above on the interior lines (1 in per face): the two bands must select cages with the same
        # counts on every face and no larger bars above, which the per-band layer preference alone would not give.
        gid = "s03_04__column__interior"
        stepped = self.state.with_designs({gid: replace(self.state.designs[gid], b_in=24.0, h_in=24.0)})
        state, report = selection.select_reinforcement(stepped, self.actions, self.cfg)
        line = report["passes"][-1]["columns"]["interior"]
        self.assertEqual(line["status"], "selected")
        self.assertIn(line["cage_pool"], ("preferred_beam_layers", "cages_the_beam_bars_thread", "all_admissible_cages"))
        lower, upper = state.designs["s01_02__column__interior"], state.designs[gid]
        self.assertEqual((upper.top_bars, upper.bot_bars, upper.side_bars), (lower.top_bars, lower.bot_bars, lower.side_bars))
        self.assertLessEqual(upper.bar_size, lower.bar_size)
        with mg.installed(state):
            capacity = capacity_design.build_group_capacity_design(self.actions, self.ids, self.cfg, None, state)
        transition = capacity["transitions"][f"s01_02__column__interior>{gid}"]
        self.assertEqual((transition["kind"], transition["supported"]), ("offset_bends", True))

    def test_a_line_without_a_chain_says_where_it_breaks_and_carries_each_bands_own_cage(self):
        # 26 below, 20 above: a 3 in face step has no declared transition, so no chain exists whatever the cages
        gid = "s03_04__column__interior"
        stepped = self.state.with_designs({gid: replace(self.state.designs[gid], b_in=20.0, h_in=20.0)})
        selected, report = selection.select_column_chain(stepped, "interior", self.actions, self.cfg)
        self.assertEqual((report["status"], report["exhausted"], report["blocked_at"]), ("no_feasible_chain", True, gid))
        # either the small band has no cage for its own demands, or none of its cages continues the chain from below
        self.assertTrue("continues a feasible chain" in report["blocked_by"] or "its own strength" in report["blocked_by"])
        self.assertEqual(report["cage_pool"], "all_admissible_cages")                          # every pool was tried
        heaviest = max(c.longitudinal_area_in2 for c in selection.column_cage_candidates(stepped.designs["s01_02__column__interior"], self.cfg))
        self.assertLess(selected["s01_02__column__interior"].longitudinal_area_in2, heaviest)   # not the heaviest cage

    def test_a_larger_strong_column_margin_never_selects_less_column_steel(self):
        def column_steel(margin):
            state, _report = selection.select_reinforcement(self.state, self.actions, self.cfg, scwb_margin=margin)
            return sum(d.longitudinal_area_in2 * len(state.groups[gid]["member_tags"]) for gid, d in state.designs.items() if d.is_column)
        self.assertGreaterEqual(column_steel(1.3) + 1e-9, column_steel(1.0))


class ChangedGroupAtABandBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack = gf.frame()
        cls.stack.__enter__()
        cls.cfg = DesignConfig.from_structure_parameters()
        cls.state = mg.install(gf.changed_state())
        cls.actions, cls.ids, _period = solve(checks.capture_member_actions)
        cls.capacity = capacity_design.build_group_capacity_design(cls.actions, cls.ids, cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.stack.__exit__(None, None, None)

    def test_the_unlike_columns_meet_in_a_declared_transition_that_is_not_supported(self):
        transitions = self.capacity["transitions"]
        self.assertEqual(len(transitions), len(mg.COLUMN_LOCATIONS))                          # one per column line at the boundary
        changed = transitions[BOUNDARY]
        # 26 x 26 with 4/4/2 below, 22 x 22 with 3/3/1 above: bars stop where the column steps in, and they lie outside
        # the upper cage
        self.assertEqual((changed["kind"], changed["supported"], changed["column_reinforcement_continuous"]), ("bar_count_reduction", False, False))
        self.assertEqual(changed["rules"], "column_transition_rules_v2")
        self.assertGreater(changed["path_counts"]["terminated"], 0)
        self.assertIn("terminated lower bars stay inside the upper cage without an offset",
                      [item["rule"] for item in changed["items"] if not item["passes"]])
        self.assertEqual(len(changed["joints"]), (gf.NX - 1) * (gf.NY - 1))                   # the interior column lines
        for key, transition in transitions.items():
            if key != BOUNDARY:
                self.assertEqual((transition["kind"], transition["supported"]), ("same_cage", True))

    def test_the_unsupported_transition_fails_the_capacity_design_and_nothing_else_does(self):
        failed = [(c["id"], c["location"]) for c in self.capacity["checks"] if c["status"] == "fail"]
        self.assertEqual(failed, [("column.transition", BOUNDARY)])
        self.assertFalse(self.capacity["accepted"])

    def test_boundary_joints_use_the_column_below_as_core_and_the_other_group_above(self):
        kinds = [kind for kind in self.capacity["joint_types"].values()
                 if kind["core_group"] == "s01_02__column__interior" and kind["column_above_group"] == "s03_04__column__interior"]
        self.assertEqual(len(kinds), 1)
        self.assertEqual(kinds[0]["level"], 2)
        self.assertEqual((kinds[0]["transition"]["kind"], kinds[0]["transition"]["supported"]), ("bar_count_reduction", False))
        # the joint hoops are those of the column below (the core)
        self.assertEqual(kinds[0]["joint_hoops"], self.capacity["columns"]["s01_02__column__interior"]["hoops"])

    def test_strong_column_evidence_is_per_physical_joint_with_one_column_at_the_roof(self):
        scwb = self.capacity["scwb"]
        self.assertFalse(scwb["roof_exemption_applied"])
        roof = [joint for joint in scwb["joints"] if joint["is_roof"]]
        self.assertEqual(len(roof), (gf.NX + 1) * (gf.NY + 1))
        self.assertEqual({joint["floor"] for joint in roof}, {gf.NZ})
        self.assertEqual(sorted(scwb["expected_combination_ids"]), sorted(self.ids))

    def test_each_column_group_is_designed_on_its_own_members_envelope(self):
        for gid, data in self.capacity["columns"].items():
            self.assertEqual(data["group_id"], gid)
            self.assertEqual(data["member_count"], len(self.state.groups[gid]["member_tags"]))
            self.assertIn(data["governing"]["story"], self.state.groups[gid]["stories_or_floors"])
        small = self.capacity["columns"]["s03_04__column__interior"]
        self.assertAlmostEqual(small["vs_limit_kip"] / self.capacity["columns"]["s01_02__column__interior"]["vs_limit_kip"],
                               (22.0 * (22.0 - mp_cover(gf.SMALL_COLUMN))) / (26.0 * (26.0 - mp_cover(gf.COLUMN))), places=9)

    def test_the_changed_beam_group_gets_its_own_capacity_shear_and_hoops(self):
        deep, plain = self.capacity["beams"]["s01_02__beam_y__edge"], self.capacity["beams"]["s03_04__beam_y__edge"]
        self.assertGreater(deep["ve_kip"], plain["ve_kip"])                                     # 24 x 32 with 5 + 4 bars against 18 x 28 with 4 + 3
        self.assertEqual(deep["hoops"] is not None, True)
        updates = capacity_design.hoop_updates(self.capacity, self.state)
        for gid, design in updates.items():                                                      # only the hoop fields move
            before = self.state.designs[gid]
            self.assertEqual((design.b_in, design.h_in, design.bar_size, design.top_bars, design.bot_bars, design.side_bars),
                             (before.b_in, before.h_in, before.bar_size, before.top_bars, before.bot_bars, before.side_bars))


def mp_cover(design):
    from Model import Member_Properties as mp
    return mp.longitudinal_cover_in(design)


if __name__ == "__main__":
    unittest.main()
