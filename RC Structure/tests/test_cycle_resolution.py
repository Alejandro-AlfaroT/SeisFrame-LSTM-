"""Reinforcement-pass cycles resolved only on verification (2026-11-01 closure review).

  V  verify_installed_selection checks the installed cages against the selection's own rules without selecting: the
     selection's settled output is verified, an arrangement that does not fit fails on its rows before any rule, a cage
     the rules refuse is named, and a beam unit is priced on its installed cage alone;
  R  resolve_cycle tries the distinct states of a cycle strong columns first, at most the declared number, and adopts
     one only when its rules are met, its hoops are a fixed point and its capacity design is accepted;
  P  the reinforcement pass adopts such a state with its own verification and capacity design, and never calls it
     settled; without one the cycle stays unresolved;
  C  the contract completes the two selection flags for a verified resolution of the same final state only;
  S  the seed arrangement outcome names the slab mats as the cause only when the arrangement fits with them masked
     (a mat that moved rows which do not fit without the mats either is contributing), records a masked audit that
     gives no answer instead of failing on it, and keeps its declared call site beside where it was raised;
  W  verification round (2026-11-01): a proposal whose verification or capacity design raises a named failure is
     recorded and not adopted; a cycle after a lead-in pass is tried on the inputs it repeats; an adopted real fixture
     state is installed and described by its own capacity design; the contract binds a resolution to the cycle record
     that adopted it; a resolved record is accepted by assess_evaluation; the search adapter keeps it unsettled.

Synthetic demands on the fixture frame (no analysis) and stand-in states: nothing here is a designed structure or an
accepted design.
"""
import copy
import sys
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract_fixture as cf                                                    # noqa: E402
import grouped_fixture as gf                                                     # noqa: E402
from test_beam_threading_evidence import synthetic_actions                       # noqa: E402
from test_terminal_arrangement import DENSE, LIGHT, RULES, SHALLOW, system_state  # noqa: E402

import Structure_Parameters as sp                                               # noqa: E402
from Design import Code_Check_Contract as contract                               # noqa: E402
from Design import Genetic_Diagnostics as genetic_diagnostics                    # noqa: E402
from Design import Group_Capacity as gc                                          # noqa: E402
from Design import Group_Selection as selection                                  # noqa: E402
from Design import Grouped_Design as gd                                          # noqa: E402
from Design.Config import DesignConfig                                           # noqa: E402
from Model import Member_Bar_Layers as bar_layers                                # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402


# ---- V ---------------------------------------------------------------------------------------------------------------
class InstalledSelectionVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = gf.frame(**RULES)
        cls.frame.__enter__()
        cls.valid = system_state(LIGHT, SHALLOW)
        cls.actions = synthetic_actions(cls.valid, {gid: (300.0, 300.0) for gid in cls.valid.groups if "__beam_" in gid})
        with mg.installed(cls.valid):
            cls.selected, cls.report = selection.select_reinforcement(cls.valid, cls.actions, DesignConfig())

    @classmethod
    def tearDownClass(cls):
        cls.frame.__exit__(None, None, None)

    def test_the_selections_own_settled_output_is_verified(self):
        self.assertTrue(self.report["settled"])
        self.assertEqual(self.report["exhausted_groups"], [])
        verification = selection.verify_installed_selection(self.selected, self.actions, DesignConfig())
        self.assertTrue(verification["verified"], verification["failing"])
        self.assertEqual(verification["state_identity"], self.selected.identity())
        self.assertEqual({u["status"] for u in verification["beams"].values()}, {"selected"})
        self.assertEqual({u["status"] for u in verification["columns"].values()}, {"selected"})
        # It selects nothing: the state is what it was, and a report built on it is the verification alone.
        report = selection.verified_selection_report(self.report, verification)
        self.assertEqual([p["pass"] for p in report["passes"]], ["verification"])
        self.assertEqual((report["settled"], report["exhausted_groups"]), (False, []))
        self.assertEqual(selection.terminal_arrangement_summary(report)[0]["status"], "valid")
        self.assertIsNone(selection.coordination_summary(report))

    def test_an_arrangement_that_does_not_fit_fails_on_its_rows_before_any_rule(self):
        dense = system_state(DENSE, SHALLOW)
        with mock.patch.object(selection, "select_beam_groups", side_effect=AssertionError("priced")):
            verification = selection.verify_installed_selection(dense, self.actions, DesignConfig())
        self.assertFalse(verification["verified"])
        self.assertEqual((verification["failing"], verification["beams"]), (["rows"], {}))
        self.assertTrue(verification["invalid_rows"])

    def test_a_cage_the_rules_refuse_is_named(self):
        tie = next(t for t in self.selected.ties() if "sfrs_beam" in t)
        weak = self.selected.with_tied_designs({tie: replace(self.selected.designs[self.selected.ties()[tie][0]],
                                                             bar_size=4, top_bars=2, bot_bars=2)})
        verification = selection.verify_installed_selection(weak, self.actions, DesignConfig())
        self.assertFalse(verification["verified"])
        self.assertEqual([f for f in verification["failing"] if f.startswith("beam:")],
                         ["beam:sfrs_beam:no_admissible_cage"])

    def test_a_beam_unit_is_priced_on_its_installed_cage_alone(self):
        tie = next(t for t in self.selected.ties() if "sfrs_beam" in t)
        gids = self.selected.ties()[tie]
        with mock.patch.object(selection, "evaluate_beam_candidate", wraps=selection.evaluate_beam_candidate) as priced:
            design, unit = selection.select_beam_groups(self.selected, gids, self.actions, DesignConfig(),
                                                        installed_only=True)
        installed = self.selected.designs[gids[0]]
        self.assertEqual({(c.args[2].bar_size, c.args[2].top_bars, c.args[2].bot_bars) for c in priced.call_args_list},
                         {(installed.bar_size, installed.top_bars, installed.bot_bars)})
        self.assertEqual((unit["installed_only"], unit["candidates_admissible"], unit["status"]), (True, 1, "selected"))
        self.assertEqual(design, installed)


# ---- R ---------------------------------------------------------------------------------------------------------------
def stand_in(name, column_area=0.0, beam_area=0.0):
    """A state with one column group and one beam group of two members each, and the given bar areas."""
    designs = {"c": SimpleNamespace(is_column=True, longitudinal_area_in2=column_area),
               "b": SimpleNamespace(is_column=False, longitudinal_area_in2=beam_area)}
    return SimpleNamespace(designs=designs, groups={"c": {"member_tags": [1, 2]}, "b": {"member_tags": [3, 4]}},
                           identity=lambda: name)


def verification_of(state, verified):
    return {"rules": selection.VERIFICATION_ID, "state_identity": state.identity(), "verified": verified,
            "failing": [] if verified else ["beam:stand_in:strength_short"], "beams": {}, "columns": {}}


class ResolveCycle(unittest.TestCase):
    def resolve(self, window, verified, hoops=None, accepted=None):
        """resolve_cycle with the verification, capacity design and hoop updates scripted by state identity."""
        tried = []

        def verify(state, actions, cfg, scwb_margin=1.0, reserve=None):
            tried.append(state.identity())
            return verification_of(state, state.identity() in verified)

        def build(actions, expected_ids, cfg, transfers, state, compatibility=None):
            return {"built_for": state, "accepted": (accepted or {}).get(state.identity(), True),
                    "checks": [] if (accepted or {}).get(state.identity(), True) else [{"id": "beam.strength", "status": "fail"}]}

        with mock.patch.object(gd.selection, "verify_installed_selection", verify), mock.patch.object(gd.mg, "install"), \
                mock.patch.object(gd.capacity_design, "build_group_capacity_design", build), \
                mock.patch.object(gd.capacity_design, "hoop_updates",
                                  lambda capacity, state: (hoops or {}).get(state.identity(), {})):
            return gd.resolve_cycle(window, [], [], None, None, 1.0), tried

    def test_the_states_are_tried_strong_columns_first_and_each_once(self):
        light, heavy, heavier_beams = stand_in("light", 10.0, 50.0), stand_in("heavy", 20.0, 1.0), stand_in("b", 10.0, 60.0)
        order = [s.identity() for s in gd._cycle_candidates([light, heavy, heavier_beams, light])]
        self.assertEqual(order, ["heavy", "b", "light"])            # column steel, then all steel; a repeat once

    def test_a_verified_state_with_fixed_hoops_and_an_accepted_capacity_design_is_adopted(self):
        a, b = stand_in("a", 10.0), stand_in("b", 20.0)
        (record, adopted, capacity, verification), tried = self.resolve([a, b], verified={"a"})
        self.assertEqual(tried, ["b", "a"])                           # b first (more column steel), not verified
        self.assertIs(adopted, a)
        self.assertIs(capacity["built_for"], a)                        # its own capacity design describes it
        self.assertEqual(verification["state_identity"], "a")
        self.assertEqual(record["adopted_state_identity"], "a")
        self.assertEqual([(c["state_identity"], c["selection_rules_met"]) for c in record["candidates"]],
                         [("b", False), ("a", True)])
        self.assertEqual(record["candidates"][0]["unmet_rules"], ["beam:stand_in:strength_short"])
        self.assertTrue(record["candidates"][1]["hoops_fixed"] and record["candidates"][1]["capacity_accepted"])

    def test_a_state_whose_hoops_move_or_whose_capacity_design_fails_is_not_adopted(self):
        a, b = stand_in("a", 10.0), stand_in("b", 20.0)
        (record, adopted, capacity, _v), _tried = self.resolve([a, b], verified={"a", "b"}, hoops={"b": {"c": 1}},
                                                               accepted={"a": False})
        self.assertIsNone(adopted)
        self.assertIsNone(capacity)
        self.assertIsNone(record["adopted_state_identity"])
        rows = {c["state_identity"]: c for c in record["candidates"]}
        self.assertEqual((rows["b"]["hoops_fixed"], rows["b"]["hoops_asked"]), (False, ["c"]))
        self.assertEqual((rows["a"]["capacity_accepted"], rows["a"]["capacity_not_passing"]), (False, ["beam.strength"]))
        self.assertIn("stay unsettled", record["reason"])

    def test_no_more_than_the_declared_candidates_are_tried(self):
        window = [stand_in(f"s{k}", float(k)) for k in range(gd.CYCLE_RESOLUTION_MAX_CANDIDATES + 2)]
        (record, adopted, _c, _v), tried = self.resolve(window, verified=set())
        self.assertIsNone(adopted)
        self.assertEqual(len(tried), gd.CYCLE_RESOLUTION_MAX_CANDIDATES)
        self.assertEqual((record["distinct_states"], record["not_tried"]), (len(window), 2))


# ---- P ---------------------------------------------------------------------------------------------------------------
class Toggle:
    """A stand-in for a grouped design whose scripted selection flips it between two states (a two-pass cycle)."""
    designs, groups = {}, {}

    def __init__(self, tick=0):
        self.tick = tick

    def with_designs(self, updates):
        return Toggle(self.tick)

    def identity(self):
        return f"toggle|{self.tick}"


class ReinforcementPassCycle(unittest.TestCase):
    def run_pass(self, verified):
        built, closed = [], []

        def select(state, actions, cfg, scwb_margin=1.0):
            return Toggle(1 - state.tick), {"settled": False, "exhausted_groups": []}

        def build(actions, expected_ids, cfg, transfers, state, compatibility=None):
            built.append(state)
            return {"built_for": state, "accepted": True, "checks": []}

        import Design.SMRF_Joints as joints
        with mock.patch.object(gd.selection, "select_reinforcement", select), mock.patch.object(gd.mg, "install"), \
                mock.patch.object(gd.capacity_design, "build_group_capacity_design", build), \
                mock.patch.object(gd.capacity_design, "hoop_updates", lambda capacity, state: {}), \
                mock.patch.object(gd.selection, "verify_installed_selection",
                                  lambda state, *a, **k: verification_of(state, state.identity() in verified)), \
                mock.patch.object(gd, "joint_check_inputs", return_value={}), \
                mock.patch.object(joints, "evaluate_joints", return_value=[]), \
                mock.patch.object(gd.selection, "close_terminal_arrangement",
                                  lambda state, *a, **k: closed.append(state) or state):
            result = gd.reinforcement_pass(Toggle(), [], [], None, None, max_passes=6, scwb_margins=(1.0,))
        return result, built, closed

    def test_an_adopted_state_is_described_by_its_own_verification_and_capacity_and_is_never_called_settled(self):
        result, _built, closed = self.run_pass(verified={"toggle|1"})
        self.assertEqual((result["stop"], result["cycle"]["pass"], result["cycle"]["repeats_pass"]), ("cycle", 3, 1))
        self.assertEqual(result["state"].identity(), "toggle|1")
        self.assertIs(result["capacity"]["built_for"], result["state"])
        self.assertEqual(result["resolution"], contract.VERIFIED_CYCLE_RESOLUTION)
        self.assertEqual((result["settled"], result["selection_settled"], result["hoops_settled"]), (False, False, True))
        self.assertEqual(result["selection"]["verification"]["state_identity"], "toggle|1")
        self.assertEqual([p["pass"] for p in result["selection"]["passes"]], ["verification"])
        self.assertEqual(result["cycle"]["resolution"]["adopted_state_identity"], "toggle|1")
        self.assertEqual(closed, [])                                  # nothing to close: its hoops are a fixed point

    def test_an_adopted_states_hoops_are_settled_although_the_last_pass_asked_for_hoops(self):
        asked = iter([{"g": 1}, {"g": 1}])                            # both passes ask for hoops; the adopted state does not
        import Design.SMRF_Joints as joints

        def select(state, actions, cfg, scwb_margin=1.0):
            return Toggle(1 - state.tick), {"settled": False, "exhausted_groups": []}
        with mock.patch.object(gd.selection, "select_reinforcement", select), mock.patch.object(gd.mg, "install"), \
                mock.patch.object(gd.capacity_design, "build_group_capacity_design",
                                  lambda *a, **k: {"accepted": True, "checks": []}), \
                mock.patch.object(gd.capacity_design, "hoop_updates", lambda capacity, state: next(asked, {})), \
                mock.patch.object(gd.selection, "verify_installed_selection",
                                  lambda state, *a, **k: verification_of(state, state.identity() == "toggle|1")), \
                mock.patch.object(gd, "joint_check_inputs", return_value={}), \
                mock.patch.object(joints, "evaluate_joints", return_value=[]), \
                mock.patch.object(gd.selection, "close_terminal_arrangement", lambda state, *a, **k: state):
            result = gd.reinforcement_pass(Toggle(), [], [], None, None, max_passes=6, scwb_margins=(1.0,))
        self.assertFalse(result["log"][-1]["hoops_settled"])            # the last pass asked for hoops
        self.assertEqual((result["resolution"], result["hoops_settled"]), (contract.VERIFIED_CYCLE_RESOLUTION, True))

    def test_without_a_verified_state_the_cycle_stays_unresolved(self):
        result, _built, _closed = self.run_pass(verified=set())
        self.assertEqual(result["resolution"], "cycle_unresolved")
        self.assertFalse(result["settled"])
        self.assertIsNone(result["cycle"]["resolution"]["adopted_state_identity"])
        self.assertEqual([c["state_identity"] for c in result["cycle"]["resolution"]["candidates"]],
                         ["toggle|0", "toggle|1"])
        self.assertEqual(result["state"].identity(), "toggle|0")       # the state the repeated pass would start from


# ---- C ---------------------------------------------------------------------------------------------------------------
class ContractCompletion(unittest.TestCase):
    BASE = {flag: True for flag in contract.COMPLETION_FLAGS}

    def completion(self, **changes):
        return contract.numerical_completion({**self.BASE, **changes}, "final")

    UNSETTLED = {"reinforcement_settled": False, "longitudinal_selection_settled": False}
    VERIFIED = {"reinforcement_resolution": {"kind": contract.VERIFIED_CYCLE_RESOLUTION, "state_identity": "final"},
                "reinforcement_stop": "cycle", "reinforcement_cycle": {"resolution": {"adopted_state_identity": "final"}}}

    def test_a_verified_resolution_completes_only_the_selection_flags_of_its_own_state(self):
        self.assertTrue(self.completion()[2])
        flags, problems, complete, resolved, _m = self.completion(**self.UNSETTLED)
        self.assertEqual((complete, resolved, len(problems)), (False, False, 2))
        flags, problems, complete, resolved, _m = self.completion(**self.UNSETTLED, **self.VERIFIED)
        self.assertEqual((complete, resolved, problems), (True, True, []))
        self.assertEqual((flags["reinforcement_settled"], flags["longitudinal_selection_settled"]), (False, False))
        tag = self.VERIFIED["reinforcement_resolution"]
        for other in ({**tag, "state_identity": "another state"}, {**tag, "kind": "cycle_unresolved"}):
            self.assertFalse(self.completion(**self.UNSETTLED, **{**self.VERIFIED, "reinforcement_resolution": other})[2])
        unhooped = self.completion(**self.UNSETTLED, **self.VERIFIED, hoops_settled=False)
        self.assertEqual((unhooped[2], unhooped[3]), (False, False))        # not resolved without settled hoops
        self.assertIn("reinforcement_settled is false", unhooped[1])
        self.assertEqual(self.completion(**self.VERIFIED, slab_layout_settled=False)[1], ["slab_layout_settled is false"])
        malformed = self.completion(**self.VERIFIED, hoops_settled=None)
        self.assertEqual((malformed[2], malformed[4]), (False, {"hoops_settled": None}))

    def test_the_resolution_is_bound_to_the_cycle_record_that_adopted_this_state(self):
        # A tag alone does not complete: the passes must have stopped on a cycle whose own record adopted this state.
        for change in ({"reinforcement_stop": "pass_budget"}, {"reinforcement_stop": "settled"},
                       {"reinforcement_cycle": None},
                       {"reinforcement_cycle": {"resolution": {"adopted_state_identity": "another state"}}},
                       {"reinforcement_cycle": {"resolution": {"adopted_state_identity": None}}}):
            self.assertFalse(self.completion(**self.UNSETTLED, **{**self.VERIFIED, **change})[3], change)
        # A record that names no state is never resolved, whatever the tag and the cycle record say.
        nameless = {**self.BASE, **self.UNSETTLED,
                    "reinforcement_resolution": {"kind": contract.VERIFIED_CYCLE_RESOLUTION},
                    "reinforcement_stop": "cycle", "reinforcement_cycle": {"resolution": {}}}
        for identity in (None, ""):
            self.assertFalse(contract.numerical_completion(nameless, identity)[3])


# ---- S ---------------------------------------------------------------------------------------------------------------
class SeedOutcome(unittest.TestCase):
    AUDIT = {"state_identity": "seed", "invalid": [{"group": "g1", "face": "bottom", "error": "past_mid_depth",
                                                     "controls": [{"by": "crossing_beam"}]}]}
    LAYOUT = {"outer_axis": "x", "layers": {"x_bottom": {"bar_size": 4}}}

    def outcome(self, fits_masked):
        record = {"layout": self.LAYOUT}
        before = sp.SLAB_REINFORCEMENT
        sp.SLAB_REINFORCEMENT = record
        try:
            with mock.patch.object(gd.mg, "installed", lambda state: nullcontext()), \
                    mock.patch.object(gd.bar_layers, "row_audit",
                                      lambda state: {"valid": fits_masked and not sp.SLAB_REINFORCEMENT["layout"]["layers"]}):
                _reason, evidence, _detail = gd._seed_arrangement_outcome(object(), self.AUDIT, {"reinforcement": record})
            self.assertIs(sp.SLAB_REINFORCEMENT, record)              # the mats are back in force afterwards
        finally:
            sp.SLAB_REINFORCEMENT = before
        return evidence

    def test_a_mat_that_moved_a_crossing_layer_is_named_by_the_masked_audit(self):
        evidence = self.outcome(fits_masked=True)
        self.assertEqual(evidence["classification"], "seed_rows_do_not_fit_the_selected_slab_mats")
        self.assertEqual((evidence["mats_named_by_failing_rows"], evidence["rows_fit_with_mats_masked"]), (False, True))
        self.assertIn("selected slab mats", evidence["reason"])
        evidence = self.outcome(fits_masked=False)
        self.assertEqual(evidence["classification"], "seed_rows_do_not_fit")
        self.assertFalse(evidence["rows_fit_with_mats_masked"])

    def test_the_declared_call_site_is_kept_beside_where_it_was_raised(self):
        class Stopped(Exception):
            pass
        try:
            raise Stopped("rows")
        except Stopped as exc:
            exc.evidence = {"outcome": "seed_arrangement_invalid", "call_site": "the declared call site"}
            exc.detail = {"rejected_state": "seed"}
            record = gd.bar_rows_outcome({"status": "evaluated"}, exc, "seed", None, False)
        call_site = record["failure_evidence"]["call_site"]
        self.assertEqual(call_site["declared"], "the declared call site")
        self.assertIn("raised_at", call_site)


# ---- W ---------------------------------------------------------------------------------------------------------------
class VerificationRound(unittest.TestCase):
    """The findings of the 2026-11-01 verification round, each pinned."""

    @classmethod
    def setUpClass(cls):
        cls.frame = gf.frame(**RULES)
        cls.frame.__enter__()
        cls.valid = system_state(LIGHT, SHALLOW)
        cls.actions = synthetic_actions(cls.valid, {gid: (300.0, 300.0) for gid in cls.valid.groups if "__beam_" in gid})
        cls.ids = [a["id"] for a in cls.actions]
        with mg.installed(cls.valid):
            selected, _report = selection.select_reinforcement(cls.valid, cls.actions, DesignConfig())
        # S1: the selection's output with the hoops its capacity design asks for (a hoop fixed point).
        s1 = selected
        for _ in range(4):
            mg.install(s1)
            updates = gc.hoop_updates(gc.build_group_capacity_design(cls.actions, cls.ids, DesignConfig(), None, s1), s1)
            if not updates:
                break
            s1 = s1.with_designs(updates)
        tie = s1.ties()["gravity_beam"][0]
        cls.s1 = s1
        cls.s2 = s1.with_tied_designs({"gravity_beam": replace(s1.designs[tie], top_bars=s1.designs[tie].top_bars + 1,
                                                               bot_bars=s1.designs[tie].bot_bars + 1)})

    @classmethod
    def tearDownClass(cls):
        cls.frame.__exit__(None, None, None)

    def test_a_proposal_whose_verification_or_capacity_design_raises_is_recorded_and_not_adopted(self):
        a, b = stand_in("a", 10.0), stand_in("b", 20.0)

        def verify(state, *args, **kwargs):
            if state.identity() == "b":
                raise RuntimeError("layers do not settle")
            return verification_of(state, True)

        def build(*args, **kwargs):
            raise ValueError("no equilibrium root of the section model")

        with mock.patch.object(gd.selection, "verify_installed_selection", verify), mock.patch.object(gd.mg, "install"), \
                mock.patch.object(gd.capacity_design, "build_group_capacity_design", build):
            record, adopted, _c, _v = gd.resolve_cycle([a, b], [], [], None, None, 1.0)
        self.assertIsNone(adopted)
        rows = {row["state_identity"]: row for row in record["candidates"]}
        self.assertEqual(rows["b"]["verification_error"], "RuntimeError: layers do not settle")
        self.assertEqual(rows["a"]["capacity_error"], "ValueError: no equilibrium root of the section model")
        # A programming error is not a design outcome: it propagates.
        with mock.patch.object(gd.selection, "verify_installed_selection", side_effect=TypeError("bug")), \
                mock.patch.object(gd.mg, "install"):
            with self.assertRaises(TypeError):
                gd.resolve_cycle([a], [], [], None, None, 1.0)

    def test_a_cycle_after_a_lead_in_pass_is_tried_on_the_inputs_it_repeats(self):
        sequence = {"A": "B", "B": "C", "C": "B"}                       # A -> B -> C -> B: the cycle repeats pass 2
        windows = []

        class Named(Toggle):
            def __init__(self, name):
                self.name = name

            def identity(self):
                return self.name

        def select(state, actions, cfg, scwb_margin=1.0):
            return Named(sequence[state.identity()]), {"settled": False, "exhausted_groups": []}

        def resolve(window, *args, **kwargs):
            windows.append([s.identity() for s in window])
            return {"candidates": [], "adopted_state_identity": None}, None, None, None

        import Design.SMRF_Joints as joints
        with mock.patch.object(gd.selection, "select_reinforcement", select), mock.patch.object(gd.mg, "install"), \
                mock.patch.object(gd.capacity_design, "build_group_capacity_design", lambda *a, **k: {"accepted": True}), \
                mock.patch.object(gd.capacity_design, "hoop_updates", lambda capacity, state: {}), \
                mock.patch.object(gd, "resolve_cycle", resolve), mock.patch.object(gd, "joint_check_inputs", return_value={}), \
                mock.patch.object(joints, "evaluate_joints", return_value=[]):
            result = gd.reinforcement_pass(Named("A"), [], [], None, None, max_passes=6, scwb_margins=(1.0,))
        self.assertEqual((result["cycle"]["pass"], result["cycle"]["repeats_pass"]), (4, 2))
        self.assertEqual(windows, [["B", "C"]])

    def test_an_adopted_real_state_is_installed_and_described_by_its_own_capacity_design(self):
        flip = {self.s1.identity(): self.s2, self.s2.identity(): self.s1}

        def select(state, actions, cfg, scwb_margin=1.0, **kwargs):
            stale = {"rules": selection.SYSTEM_RULES_ID, "settled": False, "exhausted_groups": ["STALE"],
                     "passes": [{"pass": 1, "changed_groups": ["x"],
                                 "beams": {"stale": {"status": "strength_short", "lane_limited": True, "groups": ["STALE"]}},
                                 "columns": {"stale": {"exhausted": True, "chain": [["STALE_COLUMN"]]}}}],
                     "coordination": {"rules": "stale", "status": "adopted", "trials": []}, "reserved_cores": []}
            return flip.get(state.identity(), self.s1), stale

        real_build = gc.build_group_capacity_design

        def accepted(*args, **kwargs):
            # The fixture's synthetic demands leave some capacity checks unknown; acceptance is forced only to reach the
            # adopted path on real objects (verification, hoops and joints stay real).
            capacity = real_build(*args, **kwargs)
            capacity["accepted"] = True
            return capacity

        with mock.patch.object(gd.selection, "select_reinforcement", select), \
                mock.patch.object(gd.capacity_design, "build_group_capacity_design", accepted):
            mg.install(self.s1)
            result = gd.reinforcement_pass(self.s1, self.actions, self.ids, DesignConfig(), None, max_passes=6,
                                           scwb_margins=(1.0,))
        self.assertEqual(result["resolution"], contract.VERIFIED_CYCLE_RESOLUTION)
        adopted = result["cycle"]["resolution"]["adopted_state_identity"]
        self.assertEqual(adopted, result["state"].identity())
        self.assertIn(adopted, (self.s1.identity(), self.s2.identity()))
        self.assertEqual(mg.active().identity(), adopted)                    # installed for everything that follows
        self.assertFalse(gc.hoop_updates(result["capacity"], result["state"]))
        self.assertTrue(bar_layers.row_audit(result["state"])["valid"])
        report = result["selection"]
        self.assertEqual(report["exhausted_groups"], [])                      # nothing stale from the last selection
        self.assertIsNone(selection.coordination_summary(report))
        self.assertEqual(selection.threading_constraint_summary(report), {})
        self.assertEqual(selection.terminal_arrangement_summary(report)[0]["final_state_identity"], adopted)
        self.assertFalse(result["settled"] or result["selection_settled"])

    def test_a_unit_that_does_not_thread_or_a_column_chain_the_rules_refuse_fails_the_verification(self):
        real_beams, real_columns = selection.select_beam_groups, selection.select_column_chain

        def not_threading(state, gids, actions, cfg, installed_only=False):
            design, unit = real_beams(state, gids, actions, cfg, installed_only=installed_only)
            return design, ({**unit, "threads": False} if "gravity_beam" in str(gids) or "interior" in gids[0] else unit)

        def refused(state, location, actions, cfg, scwb_margin=1.0, **kwargs):
            selected, chain = real_columns(state, location, actions, cfg, scwb_margin, **kwargs)
            return selected, ({**chain, "status": "no_feasible_chain"} if "interior" in str(location) else chain)

        with mock.patch.object(selection, "select_beam_groups", not_threading):
            verification = selection.verify_installed_selection(self.s1, self.actions, DesignConfig())
        self.assertFalse(verification["verified"])
        self.assertEqual(verification["failing"], ["beam:gravity_beam:not_threading"])
        with mock.patch.object(selection, "select_column_chain", refused):
            verification = selection.verify_installed_selection(self.s1, self.actions, DesignConfig())
        self.assertFalse(verification["verified"])
        self.assertEqual(verification["failing"], ["column:gravity_column:no_feasible_chain"])

    def test_the_evaluators_feasible_reads_the_contracts_completion(self):
        # evaluate_candidate needs frame solves, so its feasible expression is pinned at the source: the contract's
        # numerical_completion of the final state, not the raw flags.
        import inspect
        source = inspect.getsource(gd.evaluate_candidate)
        self.assertIn("complete = contract.numerical_completion(constraints, state.identity())[2]", source)
        self.assertIn("contract.accepted_for_code_design(code_check, complete,", source)
        self.assertNotIn("all(completion.values())", source)
        # The preliminary screen's settle term follows the same rule, so the screen and the code check agree.
        self.assertIn('selection_complete = constraints["reinforcement_settled"] or contract.numerical_completion(', source)
        self.assertIn("not exhausted and selection_complete", source)

    def test_an_unsettled_arrangement_is_named_by_the_verification(self):
        with mock.patch.object(selection.bar_layers, "row_audit",
                               return_value={"valid": False, "invalid": [], "settle_error": "the layers do not settle"}):
            verification = selection.verify_installed_selection(self.s1, self.actions, DesignConfig())
        self.assertEqual((verification["failing"], verification["settle_error"]),
                         (["rows_unsettled"], "the layers do not settle"))


class ResolvedRecord(unittest.TestCase):
    """A record completed through a verified cycle resolution, through the shared contract fixture."""

    def setUp(self):
        self.frame = gf.frame()
        self.frame.__enter__()

    def tearDown(self):
        self.frame.__exit__(None, None, None)

    def test_a_resolved_record_is_accepted_by_the_contract_and_kept_unsettled_by_the_search_adapter(self):
        state = gf.uniform_state()
        identity = state.identity()
        resolution = {"reinforcement_settled": False, "longitudinal_selection_settled": False,
                      "reinforcement_stop": "cycle",
                      "reinforcement_resolution": {"kind": contract.VERIFIED_CYCLE_RESOLUTION, "state_identity": identity},
                      "reinforcement_cycle": {"resolution": {"adopted_state_identity": identity}}}
        record = cf.evaluation_record(state, constraints=resolution)
        assessed = contract.assess_evaluation(record, submitted_identity=identity)
        self.assertEqual((record["feasible"], assessed["valid"], assessed["classification"]), (True, True, "accepted"))
        self.assertEqual(assessed["numerical_completion"]["selection_flags_met_by"], contract.VERIFIED_CYCLE_RESOLUTION)
        bound_elsewhere = copy.deepcopy(resolution)
        bound_elsewhere["reinforcement_cycle"]["resolution"]["adopted_state_identity"] = "another state"
        unbound = cf.evaluation_record(state, constraints=bound_elsewhere)
        self.assertFalse(unbound["feasible"])
        self.assertEqual(contract.assess_evaluation(unbound, submitted_identity=identity)["classification"], "unresolved")
        # The search adapter reads the selection's own convergence: a verified resolution is not settled there.
        self.assertFalse(genetic_diagnostics._settled({**resolution, "hoops_settled": True}))


class SeedOutcomeOnRealArrangements(unittest.TestCase):
    """The seed outcome's classes on the fixture frame's own arrangements, under the fixture's #4 slab mats."""

    def setUp(self):
        self.frame = gf.frame(**RULES)
        self.frame.__enter__()
        from test_terminal_arrangement import slab_layout
        self.record = {"layout": slab_layout(size=4)}
        self.before = sp.SLAB_REINFORCEMENT

    def tearDown(self):
        sp.SLAB_REINFORCEMENT = self.before
        self.frame.__exit__(None, None, None)

    def outcome(self, state):
        sp.SLAB_REINFORCEMENT = self.record
        with mg.installed(state):
            audit = bar_layers.row_audit(state)
            self.assertFalse(audit["valid"])
            _reason, evidence, _detail = gd._seed_arrangement_outcome(state, audit, {"reinforcement": self.record})
        self.assertIs(sp.SLAB_REINFORCEMENT, self.record)
        return evidence

    def test_a_mat_that_pushed_a_crossing_layer_is_named_by_the_masked_audit_alone(self):
        # Band policy: the y beams' top rows are raised by crossing layers that a mat pushed; no failing row names it.
        band = mg.GroupedDesign.uniform(gf.NX, gf.NY, gf.NZ, LIGHT, replace(SHALLOW, top_bars=4, bot_bars=4, bar_size=7))
        y_beams = replace(SHALLOW, member_type="beam_y", top_bars=5, bot_bars=5, bar_size=7)
        state = band.with_designs({g: y_beams for g, group in band.groups.items() if group["member_type"] == "beam_y"})
        evidence = self.outcome(state)
        self.assertEqual((evidence["mats_named_by_failing_rows"], evidence["rows_fit_with_mats_masked"]), (False, True))
        self.assertEqual(evidence["classification"], "seed_rows_do_not_fit_the_selected_slab_mats")

    def test_a_named_mat_on_rows_that_do_not_fit_without_the_mats_either_is_contributing(self):
        evidence = self.outcome(system_state(DENSE, SHALLOW))
        self.assertEqual((evidence["mats_named_by_failing_rows"], evidence["rows_fit_with_mats_masked"]), (True, False))
        self.assertEqual(evidence["classification"], "seed_rows_do_not_fit_with_or_without_the_slab_mats")
        self.assertIn("do not fit without the mats either", evidence["reason"])

    def test_a_masked_audit_that_raises_is_recorded_and_the_outcome_stands(self):
        state = system_state(DENSE, SHALLOW)
        real = bar_layers.row_audit

        def audit(candidate):
            if not (sp.SLAB_REINFORCEMENT or {}).get("layout", {}).get("layers"):
                raise KeyError("x_top")
            return real(candidate)
        with mock.patch.object(gd.bar_layers, "row_audit", audit):
            evidence = self.outcome(state)
        self.assertIsNone(evidence["rows_fit_with_mats_masked"])
        self.assertIn("raised KeyError", evidence["masked_audit_problem"])
        self.assertEqual(evidence["classification"], "seed_rows_do_not_fit_with_or_without_the_slab_mats")


def setUpModule():
    # These tests verify the 3D physical detail route, which the computed common-action route keeps (2026-11-07).
    global _DETAIL_BASIS
    _DETAIL_BASIS = sp.DETAIL_BASIS
    sp.DETAIL_BASIS = "physical_3d_v2"


def tearDownModule():
    sp.DETAIL_BASIS = _DETAIL_BASIS


if __name__ == "__main__":
    unittest.main()
