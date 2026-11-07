"""End-to-end fail-closed fixtures of the two conditional product branches (conditional branch coverage, 2026-11-05).

A candidate that needs evidence the project does not hold must end unknown: not accepted, and not turned into a
computed failure either.

  (a) A column transition whose bars change size through mechanical splices between two sizes for which the pinned
      coupler report publishes no product (here No. 10 below No. 8).
  (b) Column bars larger than No. 11 ending at the roof (here No. 14): ACI 318-19 18.8.5.3 gives the seismic
      straight-bar development only through No. 11.

Each branch is driven through the production chain without an analysis:

  Group_Capacity.build_group_capacity_design      joint_transition, SMRF_Transitions.column_transition,
                                                  SMRF_Transition_Physical.verify_transition,
                                                  SMRF_Coupler_Catalog.lookup_coupler,
                                                  SMRF_Roof_Anchorage.verify_declared_roof and the row emission
  Grouped_Record._capacity_completion_checks      the mechanical splice product row
  Grouped_Design.joint_check_inputs + SMRF_Joints.evaluate_joints     the joint shear rows (the cascade)
  Code_Check_Contract.classify_row, code_check_result, assess_evaluation and Genetic_Search.assessment

What is written down instead of computed:

  * The member actions are two synthetic combinations (a small axial load and moment in every column, a small span
    moment in every beam). No analysis, eigen solve or floor solve is run.
  * Design.SMRF_Transition_Context.beam_context is replaced by a complete context without beam bars. The fixture
    frame's own beam cages do not place in its joints, which would make every joint detail a computed failure and
    hide the branch under test (a failure outranks an unknown in the three-valued status); with the real context one
    joint type also costs most of a minute. Everything downstream of the context is production code.
  * The contract needs every required row of the design. The rows this fixture cannot produce without an analysis
    (strength, drift, regularity, slab) are the passing rows of contract_fixture; the rows under test are the
    production rows, laid over them. The same construction with a published pair, or with No. 11 bars, passes, so
    each test separates the branch from the fixture.
"""
import contextlib
import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
for _path in (TESTS_DIR.parent, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import openseespy.opensees as ops                                                # noqa: E402
import contract_fixture as cf                                                    # noqa: E402
import grouped_fixture as gf                                                     # noqa: E402
import Structure_Parameters as sp                                                # noqa: E402
from Design import Code_Check_Contract as contract                               # noqa: E402
from Design import Genetic_Search as ga                                          # noqa: E402
from Design import Group_Capacity as gc                                          # noqa: E402
from Design import Grouped_Design as gd                                          # noqa: E402
from Design import Grouped_Record as gr                                          # noqa: E402
from Design import SMRF_Joints as joint_rules                                    # noqa: E402
from Design import SMRF_Transition_Context as context_module                     # noqa: E402
from Design import SMRF_Transitions as transitions                               # noqa: E402
from Design.Config import DesignConfig                                           # noqa: E402
from Design.SMRF_Common import make_check                                        # noqa: E402
from Design.SMRF_Coupler_Catalog import lookup_coupler                           # noqa: E402
from Design.SMRF_Roof_Anchorage import verify_declared_roof                      # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402

NX, NY, NZ = 1, 1, gf.NZ                                # four corner columns, bands s01_02 and s03_04
LOWER_BAND, UPPER_BAND = "s01_02", "s03_04"
BOUNDARY_LEVEL = 2
FY_KSI, SPECIFICATION = 60.0, "ASTM A706 Grade 60"
BEAM = replace(gf.BEAM, fc_ksi=5.0)
# 30 in column, twelve No. 10, No. 4 hoops with four legs at 4 in: the local hoop at a coupler can be chosen from the
# production ladder, so a published pair on this cage passes its physical detail. 4.5 ksi keeps the confinement
# spacing of both bands at 4 in (at 5 ksi the No. 8 band is selected at 3 in and its roof extension then fails the
# end-interval rule of its transverse sets, a computed failure that has nothing to do with the branch under test).
TRANSITION_COLUMN = mg.MemberDesign("column", 30.0, 30.0, 4.5, 10, 4, 4, 2, 4, 4, 4.0, (4, 4))
# 36 in column, twelve bars, No. 5 hoops with four legs at 4 in: the general Chapter 25 straight development of a
# No. 14 bar is met in the joint and the declared roof extension, so nothing but the seismic expression is open.
ROOF_COLUMN = mg.MemberDesign("column", 36.0, 36.0, 6.0, 14, 4, 4, 2, 5, 4, 4.0, (4, 4))

TRANSITION = "column.transition"
TRANSITION_DETAIL = "column.transition_physical_detail"
ROOF_DETAIL = "column.roof_anchorage_physical_detail"
PRODUCT_ROW = "detailing.mechanical_splice_product_qualification"
SPLICE_DETAIL = "detailing.splice_physical_detail"
PRODUCT_RULE = "mechanical_splice_product"
ROOF_RULE = "roof_straight_bar_tension_and_compression_development"
# The production rows laid over the passing rows of the contract fixture.
WATCHED = {("capacity", TRANSITION), ("capacity", TRANSITION_DETAIL), ("capacity", ROOF_DETAIL), ("joints", "joint_shear"),
           ("detailing", PRODUCT_ROW)}


def complete_beam_context(entry, transition=None, column_hoops=None):
    """The stand-in for SMRF_Transition_Context.beam_context: complete beam geometry, no beam bar in the joint."""
    above = entry["above"] or entry["core"]
    return {"method": "fixture_complete_context_without_beam_bars", "beam_segments": [], "beam_geometry_complete": True,
            "beam_geometry_checks": [], "beam_hook_families": [], "status": "pass",
            "upper_hoop_spacing_in": (column_hoops or {}).get(above.group_id, above.design.stirrup_spacing_in),
            "transverse_fy_ksi": sp.FYT_KSI, "aggregate_max_size_in": sp.AGGREGATE_MAX_SIZE_IN,
            "longitudinal_specification": sp.REINFORCEMENT_SPECIFICATION}


def helper_context(spacing_in):
    """The same complete context as a plain record, for the functions called below the capacity design."""
    return {"beam_segments": [], "beam_geometry_complete": True, "upper_hoop_spacing_in": spacing_in,
            "transverse_fy_ksi": 60.0, "aggregate_max_size_in": 0.75, "longitudinal_specification": SPECIFICATION,
            "axial_max_kip": 0.0}


def synthetic_actions(state):
    """Two combinations written down, not solved: 100 kip and 50 kip-in in every column, 100 kip-in in every span."""
    actions = []
    for cid in ("synthetic_1", "synthetic_2"):
        members = {}
        for member in state.members():
            if member.is_column:
                members[str(member.member_tag)] = {
                    "local_force_kip_kipin": [100.0, 1.0, 1.0, 0.0, 50.0, 50.0, -100.0, -1.0, -1.0, 0.0, 50.0, 50.0],
                    "axial_i_kip": 100.0, "axial_j_kip": 100.0}
            else:
                members[str(member.member_tag)] = {
                    "local_force_kip_kipin": [0.0] * 12,
                    "span_bending": {"full_span": {"mu_positive_kip_in": 100.0, "mu_negative_kip_in": 100.0}}}
        actions.append({"id": cid, "members": members})
    return actions


def banded_state(column, upper_bar_size):
    """The frame with ``column`` in both bands and bars of ``upper_bar_size`` in the columns of the upper band."""
    state = mg.GroupedDesign.uniform(NX, NY, NZ, column, BEAM)
    upper = {gid: replace(design, bar_size=upper_bar_size) for gid, design in state.designs.items()
             if design.is_column and gid.startswith(UPPER_BAND)}
    if not upper:
        raise AssertionError("the fixture frame has no column group in the upper band")
    return state.with_designs(upper)


def key_of(row):
    return row["family"], row["id"], row["location"]


def overlay(baseline, produced, also=()):
    """``baseline`` with every watched production row (WATCHED, and the exact keys of ``also``) in place of its passing
    stand-in (a watched row the inventory does not list is appended)."""
    watched = {key_of(row): row for row in produced if key_of(row)[:2] in WATCHED or key_of(row) in also}
    return [watched.pop(key_of(row), row) for row in baseline] + list(watched.values())


def capacity_design(state):
    actions = synthetic_actions(state)
    return gc.build_group_capacity_design(actions, [a["id"] for a in actions], DesignConfig(), None, state)


def evaluate(state, also=()):
    """What the grouped evaluator derives from the capacity design of ``state`` and how the contract and the search
    classify it. Call inside the fixture frame."""
    identity = state.identity()
    with mg.installed(state):
        capacity = capacity_design(state)
        completion = gr._capacity_completion_checks(capacity, state)
        joint_checks = joint_rules.evaluate_joints(gd.joint_check_inputs(capacity, state))
        # The families as Grouped_Design assembles them for the contract.
        produced = ([contract.compact_row("capacity", check) for check in capacity["checks"]]
                    + [contract.compact_row("joints", check) for check in joint_checks]
                    + [contract.compact_row("detailing", check) for check in completion if check["id"] == PRODUCT_ROW])
        expected = contract.expected_inventory(state)
        baseline = cf.rows_for(state)
        rows = overlay(baseline, produced, also)
        record = cf.evaluation_record(state, rows=rows)
        assessed = contract.assess_evaluation(record, submitted_identity=identity, submitted_state=state)
        search = ga.assessment(record, None, submitted_identity=identity, submitted_state=state)
    return {"state": state, "identity": identity, "capacity": capacity, "completion": completion, "produced": produced,
            "expected": expected, "baseline": baseline, "rows": rows, "record": record, "assessed": assessed,
            "search": search, "production_only": contract.code_check_result(produced, expected)}


def joint_types(capacity, level=None, roof=False):
    return sorted(name for name, kind in capacity["joint_types"].items()
                  if (kind["column_above_group"] is None) == roof and (level is None or kind["level"] == level))


def unknown_keys(result):
    return {(item["family"], item["id"], item["location"]) for item in result["unknown"]}


class FrameFixture(unittest.TestCase):
    """The 1 x 1 bay, four-story fixture frame held for a whole class (each capacity design costs about half a
    minute, so it is built once per state)."""
    COLUMN = None

    @classmethod
    def open_frame(cls):
        stack = contextlib.ExitStack()
        cls.addClassCleanup(stack.close)
        stack.enter_context(gf.frame(column=cls.COLUMN, beam=BEAM, FY_KSI=FY_KSI, REINFORCEMENT_SPECIFICATION=SPECIFICATION,
                                     NUM_BAY_X=NX, NUM_BAY_Y=NY))
        stack.enter_context(mock.patch.object(context_module, "beam_context", complete_beam_context))
        for solve in ("analyze", "eigen"):
            stack.enter_context(mock.patch.object(ops, solve, side_effect=AssertionError("no solve in these fixtures")))

    def row(self, case, family, check_id, location):
        found = [row for row in case["produced"] if key_of(row) == (family, check_id, location)]
        self.assertEqual(len(found), 1, f"{family} {check_id} @ {location}: {len(found)} production rows")
        return found[0]

    def result_of(self, case, rows):
        return contract.code_check_result(rows, case["expected"])

    def assess(self, case, record):
        with mg.installed(case["state"]):
            return contract.assess_evaluation(record, submitted_identity=case["identity"], submitted_state=case["state"])

    def record_of(self, case, rows):
        """The evaluation record whose own reported result follows from ``rows`` (an evaluator that reports them)."""
        with mg.installed(case["state"]):
            return cf.evaluation_record(case["state"], rows=rows)


# ---- the catalog and the two helpers, below the capacity design ------------------------------------------------------
class ProductStatusAtItsSource(unittest.TestCase):
    def test_every_unpublished_pair_of_the_column_ladder_is_unknown_never_a_failure_or_an_invented_product(self):
        ladder = list(DesignConfig().rebar.bar_sizes_col)
        self.assertEqual(ladder, [5, 6, 7, 8, 9, 10, 11, 14], "the column bar ladder changed: review the pairs below")
        published = {(5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 14)}
        unpublished = 0
        for fy in (60.0, 80.0):
            specification = f"ASTM A706 Grade {int(fy)}"
            for index, upper in enumerate(ladder):
                for lower in ladder[index + 1:]:
                    product = lookup_coupler(lower, upper, fy, specification)
                    with self.subTest(lower=lower, upper=upper, fy=fy):
                        if (upper, lower) in published:
                            self.assertEqual(product["status"], "pass")
                            self.assertTrue(product["part_number"])
                        else:
                            unpublished += 1
                            self.assertEqual(product["status"], "not_evaluated")
                            self.assertIn("No published", product["reason"])
                            for invented in ("part_number", "envelope_diameter_in", "length_in"):
                                self.assertNotIn(invented, product)
        self.assertEqual(unpublished, 2 * 21)

    def test_a_coaxial_size_change_is_spliced_mechanically_and_only_the_published_pair_is_verified(self):
        cases = {}
        for upper_size in (8, 9):
            cases[upper_size] = transitions.column_transition(
                TRANSITION_COLUMN, replace(TRANSITION_COLUMN, bar_size=upper_size), fy_ksi=FY_KSI, clear_cover_in=1.5,
                upper_clear_height_in=140.0, joint_depth_in=28.0, lower_hoop_spacing_in=4.0,
                physical_context=helper_context(4.0))
        for upper_size, result in cases.items():
            # The same path decision for both pairs: twelve coaxial bars, no lap (the lap bars would overlap).
            self.assertEqual(result["path_counts"]["mechanically_spliced"], 12, upper_size)
            self.assertEqual(result["path_counts"]["lap_spliced"], 0, upper_size)
            self.assertTrue(result["geometry_supported"], upper_size)
        unknown, known = cases[8]["physical_detail"], cases[9]["physical_detail"]
        self.assertEqual((unknown["status"], unknown["unknown_rules"], unknown["failure_rules"]),
                         ("not_evaluated", [PRODUCT_RULE], []))
        self.assertEqual(unknown["coupler_product"]["status"], "not_evaluated")
        self.assertNotIn("part_number", unknown["coupler_product"])
        self.assertEqual(unknown["couplers"], [])                       # no device body is placed for an unknown product
        self.assertFalse(cases[8]["physically_verified"])
        self.assertEqual(cases[8]["not_established"], [PRODUCT_RULE])
        self.assertTrue(cases[8]["supported"])                           # the preliminary screen alone does not refuse it
        self.assertEqual((known["status"], known["unknown_rules"], known["failure_rules"]), ("pass", [], []))
        self.assertEqual(known["coupler_product"]["part_number"], "EL3228A2")
        self.assertEqual(len(known["couplers"]), 12)
        self.assertTrue(cases[9]["physically_verified"])

    def roof(self, design):
        return verify_declared_roof(design, extension_in=36.0, joint_depth_in=28.0, fy_ksi=FY_KSI, clear_cover_in=1.5,
                                    spacing_in=4.0, specification=SPECIFICATION, physical_context=helper_context(4.0),
                                    axial_max_kip=0.0)

    def rule(self, result, name):
        found = [check for check in result["checks"] if check["rule"] == name]
        self.assertEqual(len(found), 1, name)
        return found[0]

    def test_no14_at_the_roof_is_unknown_with_its_reason_and_no11_is_evaluated(self):
        large = self.roof(ROOF_COLUMN)
        self.assertEqual((large["status"], large["unknown_rules"], large["failure_rules"]), ("not_evaluated", [ROOF_RULE], []))
        open_rule = self.rule(large, ROOF_RULE)
        self.assertIsNone(open_rule["passes"])
        self.assertIn("18.8.5.3", open_rule["detail"])
        self.assertIn("No11", open_rule["detail"])
        # Not a disguised geometric failure: the general straight development of the same bar is computed and met.
        self.assertEqual(self.rule(large, "roof_general_straight_development")["status"], "pass")
        self.assertEqual(large["termination_type"], "straight_bar_at_declared_extension_top")
        self.assertNotIn("product", large)                               # the straight route claims no head product
        evaluated = self.roof(replace(ROOF_COLUMN, bar_size=11))
        self.assertEqual((evaluated["status"], evaluated["unknown_rules"], evaluated["failure_rules"]), ("pass", [], []))
        self.assertEqual(self.rule(evaluated, ROOF_RULE)["status"], "pass")

    def test_a_no11_roof_detail_that_does_not_work_is_a_computed_failure_and_no14_never_passes(self):
        sparse = replace(ROOF_COLUMN, top_bars=3, bot_bars=3, side_bars=1, stirrup_legs=3, stirrup_legs_by_direction=(3, 3))
        evaluated = self.roof(replace(sparse, bar_size=11))
        self.assertEqual(evaluated["status"], "fail")
        self.assertEqual(evaluated["unknown_rules"], [])
        self.assertIn(self.rule(evaluated, ROOF_RULE)["status"], ("pass", "fail"))
        large = self.roof(sparse)
        # A real failure beside the unknown: the detail fails, the seismic development stays listed as not established.
        self.assertEqual(large["status"], "fail")
        self.assertEqual(large["unknown_rules"], [ROOF_RULE])


# ---- branch (a): a mechanically spliced transition between sizes without a published coupler -------------------------
class UnpublishedCouplerPairAtAColumnTransition(FrameFixture):
    COLUMN = TRANSITION_COLUMN
    KEY = f"{LOWER_BAND}__column__corner>{UPPER_BAND}__column__corner"

    @classmethod
    def setUpClass(cls):
        cls.open_frame()
        # The splice inventory rows of the two column groups are production rows here as well (the beam groups' are
        # not: the fixture beams' own splice devices do not fit, which is not the branch under test).
        cls.upper, cls.lower = f"{UPPER_BAND}__column__corner", f"{LOWER_BAND}__column__corner"
        splices = {("capacity", SPLICE_DETAIL, gid) for gid in (cls.upper, cls.lower)}
        cls.candidate = evaluate(banded_state(TRANSITION_COLUMN, 8), splices)     # No. 10 below No. 8: no published pair
        cls.control = evaluate(banded_state(TRANSITION_COLUMN, 9), splices)       # No. 10 below No. 9: published
        cls.boundary = joint_types(cls.candidate["capacity"], level=BOUNDARY_LEVEL)
        cls.shear_keys = {("joints", "joint_shear", f"joint_shear/{name}/{axis}") for name in cls.boundary for axis in "xy"}
        cls.detail_keys = {("capacity", TRANSITION_DETAIL, name) for name in cls.boundary}
        cls.splice_keys = {("capacity", SPLICE_DETAIL, cls.upper)}
        cls.open_keys = cls.detail_keys | cls.shear_keys | cls.splice_keys | {("detailing", PRODUCT_ROW, "")}

    def test_the_fixture_reaches_the_branch_at_every_band_boundary_joint_type(self):
        self.assertEqual(len(self.boundary), 4)
        self.assertEqual(joint_types(self.control["capacity"], level=BOUNDARY_LEVEL), self.boundary)
        for case, size in ((self.candidate, 8), (self.control, 9)):
            aggregate = case["capacity"]["transitions"][self.KEY]
            self.assertEqual(sorted(aggregate["contexts"]), self.boundary)
            for name, context in aggregate["contexts"].items():
                self.assertEqual(context["path_counts"]["mechanically_spliced"], 12, (size, name))
                product = context["physical_detail"]["coupler_product"]
                self.assertEqual((product["lower_bar_size"], product["upper_bar_size"]), (10, size))
                self.assertEqual(product, lookup_coupler(10, size, FY_KSI, SPECIFICATION))

    def test_the_product_status_is_not_evaluated_and_nothing_else_is_open_or_failed(self):
        aggregate = self.candidate["capacity"]["transitions"][self.KEY]
        self.assertFalse(aggregate["physically_verified"])
        for name in self.boundary:
            detail = self.candidate["capacity"]["joint_types"][name]["physical_transition"]
            self.assertEqual((detail["status"], detail["unknown_rules"], detail["failure_rules"]),
                             ("not_evaluated", [PRODUCT_RULE], []), name)
            self.assertEqual(detail["coupler_product"]["status"], "not_evaluated")
            self.assertNotIn("part_number", detail["coupler_product"])
            self.assertEqual(detail["couplers"], [])
            published = self.control["capacity"]["joint_types"][name]["physical_transition"]
            self.assertEqual((published["status"], published["unknown_rules"], published["failure_rules"]), ("pass", [], []), name)
        self.assertTrue(self.control["capacity"]["transitions"][self.KEY]["physically_verified"])

    def test_the_transition_detail_row_is_unknown_and_the_preliminary_row_does_not_stand_in_for_it(self):
        for name in self.boundary:
            row = self.row(self.candidate, "capacity", TRANSITION_DETAIL, name)
            self.assertEqual(row["status"], "not_evaluated")
            self.assertEqual(contract.classify_row(row), ("unknown", []))
            self.assertEqual(contract.classify_row(self.row(self.control, "capacity", TRANSITION_DETAIL, name)), ("pass", []))
        # The fast-path row passes for both pairs: it is not the evidence, and the contract asks for the detail row.
        for case in (self.candidate, self.control):
            self.assertEqual(self.row(case, "capacity", TRANSITION, self.KEY)["status"], "pass")
        self.assertLessEqual(self.detail_keys, self.candidate["expected"]["exact"])
        # Only the band boundary is open: every other transition detail of the candidate is evaluated and passes.
        others = [row for row in self.candidate["produced"] if row["id"] == TRANSITION_DETAIL and row["location"] not in self.boundary]
        self.assertEqual(len(others), 8)
        self.assertEqual({row["status"] for row in others}, {"pass"})

    def test_the_joint_shear_strength_at_those_joints_is_not_usable_and_its_row_is_unknown(self):
        for name in self.boundary:
            for axis in "xy":
                location = f"joint_shear/{name}/{axis}"
                self.assertIs(self.candidate["capacity"]["joint_types"][name]["shear"][axis]
                              ["capacity_topology_and_confinement_checked"], False)
                self.assertIs(self.control["capacity"]["joint_types"][name]["shear"][axis]
                              ["capacity_topology_and_confinement_checked"], True)
                row = self.row(self.candidate, "joints", "joint_shear", location)
                self.assertEqual(contract.classify_row(row), ("unknown", []))
                self.assertEqual(contract.classify_row(self.row(self.control, "joints", "joint_shear", location)), ("pass", []))
        self.assertLessEqual(self.shear_keys, self.candidate["expected"]["exact"])

    def test_the_product_qualification_row_is_unknown_for_the_transition_devices_only(self):
        row = self.row(self.candidate, "detailing", PRODUCT_ROW, "")
        self.assertEqual(contract.classify_row(row), ("unknown", []))
        self.assertEqual(contract.classify_row(self.row(self.control, "detailing", PRODUCT_ROW, "")), ("pass", []))
        check = next(c for c in self.candidate["completion"] if c["id"] == PRODUCT_ROW)
        open_products = sorted(name for name, product in check["details"]["products"].items() if product.get("status") != "pass")
        self.assertEqual(open_products, [f"transition/{self.KEY}/{name}" for name in self.boundary])
        self.assertTrue(all(p.get("status") == "not_evaluated" for n, p in check["details"]["products"].items() if n in open_products))

    def test_the_spliced_upper_members_make_their_groups_splice_inventory_row_unknown(self):
        state = self.candidate["state"]
        with mg.installed(state):
            above_boundary = sorted(tag for tag in state.groups[self.upper]["member_tags"]
                                    if state.member(tag).story_or_floor == BOUNDARY_LEVEL + 1)
        self.assertEqual(len(above_boundary), 4)
        inventory = self.candidate["capacity"]["splices"][self.upper]["physical_detail"]
        self.assertEqual((inventory["status"], sorted(inventory["unknown_members"]), inventory["failed_members"]),
                         ("not_evaluated", above_boundary, []))
        self.assertEqual(contract.classify_row(self.row(self.candidate, "capacity", SPLICE_DETAIL, self.upper)), ("unknown", []))
        # The lower group's own same-size devices are published and placed; so are both groups of the published pair.
        self.assertEqual(contract.classify_row(self.row(self.candidate, "capacity", SPLICE_DETAIL, self.lower)), ("pass", []))
        for gid in (self.upper, self.lower):
            self.assertEqual(contract.classify_row(self.row(self.control, "capacity", SPLICE_DETAIL, gid)), ("pass", []))
        self.assertLessEqual(self.splice_keys, self.candidate["expected"]["exact"])

    def test_a_code_check_holding_the_rows_is_incomplete_and_the_published_pair_passes(self):
        result = self.result_of(self.candidate, self.candidate["rows"])
        self.assertEqual((result["status"], result["accepted"], result["complete"]), ("incomplete", False, False))
        self.assertEqual(unknown_keys(result), self.open_keys)
        self.assertEqual((result["counts"]["fail"], result["counts"]["invalid"], result["counts"]["missing"]), (0, 0, 0))
        control = self.result_of(self.control, self.control["rows"])
        self.assertEqual((control["status"], control["accepted"], control["counts"]["unknown"]), ("passed", True, 0))
        # Each of the four row kinds blocks on its own, the product row although no inventory lists it.
        self.assertNotIn(("detailing", PRODUCT_ROW, ""), self.candidate["expected"]["exact"])
        for kept in (self.detail_keys, self.shear_keys, self.splice_keys, {("detailing", PRODUCT_ROW, "")}):
            rows = [row for row in self.candidate["rows"] if key_of(row) not in self.open_keys - kept]
            rows += [row for row in self.candidate["baseline"] if key_of(row) in self.open_keys - kept]
            single = self.result_of(self.candidate, rows)
            self.assertEqual((single["status"], single["accepted"]), ("incomplete", False))
            self.assertEqual(unknown_keys(single), kept)

    def test_the_production_rows_alone_are_never_a_passed_code_check(self):
        result = self.candidate["production_only"]
        self.assertNotEqual(result["status"], "passed")
        self.assertFalse(result["accepted"])
        self.assertLessEqual(self.open_keys, unknown_keys(result))

    def test_the_candidate_is_unresolved_for_the_contract_and_the_search_and_never_accepted(self):
        assessed = self.candidate["assessed"]
        self.assertEqual((assessed["valid"], assessed["classification"], assessed["accepted"]), (True, "unresolved", False))
        self.assertEqual(assessed["code_check"]["status"], "incomplete")
        self.assertIs(self.candidate["record"]["feasible"], False)
        classification, fitness = self.candidate["search"]
        self.assertEqual(classification, "unresolved")
        self.assertEqual(fitness, (2, len(self.open_keys), 0.0, 0.0))
        self.assertEqual((self.control["assessed"]["classification"], self.control["assessed"]["accepted"]), ("accepted", True))
        self.assertEqual(self.control["search"], ("feasible", (0, 0, 0.0, 0.0)))

    def test_roof_bars_up_to_no11_on_the_same_frame_are_evaluated(self):
        for case in (self.candidate, self.control):
            roofs = joint_types(case["capacity"], roof=True)
            self.assertEqual(len(roofs), 4)
            for name in roofs:
                self.assertEqual(contract.classify_row(self.row(case, "capacity", ROOF_DETAIL, name)), ("pass", []))

    # ---- a record that claims more than its evidence ----
    def test_a_record_that_claims_feasible_or_reports_a_passed_check_over_the_same_rows_is_invalid(self):
        claimed = {**self.candidate["record"], "feasible": True}
        assessed = self.assess(self.candidate, claimed)
        self.assertEqual((assessed["valid"], assessed["classification"], assessed["accepted"]), (False, "invalid", False))
        reported = {**self.candidate["record"], "feasible": True, "code_check": self.control["record"]["code_check"]}
        assessed = self.assess(self.candidate, reported)
        self.assertEqual((assessed["valid"], assessed["classification"], assessed["accepted"]), (False, "invalid", False))
        self.assertTrue(any("does not follow from its rows" in problem for problem in assessed["problems"]))

    def test_a_status_rewritten_to_pass_without_numbers_makes_the_rows_invalid(self):
        rows = [({**row, "status": "pass"} if key_of(row) in self.open_keys else row) for row in self.candidate["rows"]]
        for row in rows:
            if key_of(row) in self.open_keys:
                self.assertEqual(contract.classify_row(row)[0], "invalid")
        result = self.result_of(self.candidate, rows)
        self.assertEqual((result["status"], result["accepted"]), ("invalid", False))
        self.assertEqual(len(result["invalid"]), len(self.open_keys))
        # Even when the evaluator's own report agrees with the rewritten rows.
        assessed = self.assess(self.candidate, self.record_of(self.candidate, rows))
        self.assertEqual((assessed["valid"], assessed["classification"], assessed["accepted"]), (False, "invalid", False))

    def test_forging_the_transition_rows_alone_leaves_the_other_rows_of_the_branch_open(self):
        forged = {key: contract.compact_row(key[0], make_check(key[1], "forged", 1, 1, "==", location=key[2]))
                  for key in self.detail_keys}
        rows = [forged.get(key_of(row), row) for row in self.candidate["rows"]]
        assessed = self.assess(self.candidate, self.record_of(self.candidate, rows))
        self.assertEqual((assessed["classification"], assessed["accepted"]), ("unresolved", False))
        self.assertEqual(unknown_keys(assessed["code_check"]), self.open_keys - self.detail_keys)

    def test_saved_capacity_evidence_edited_to_pass_differs_from_its_recomputation(self):
        """The comparison Grouped_Record.qualify_grouped_design makes before it reads any capacity row: the capacity
        design rebuilt from the saved actions against the saved evidence (only the rebuilt one is then read)."""
        state = self.candidate["state"]
        with mg.installed(state):
            recomputed = gr._canonical(capacity_design(state))
        saved = gr._canonical(self.candidate["capacity"])
        self.assertEqual(gr.differences(saved, recomputed), [])
        name = self.boundary[0]
        edited = copy.deepcopy(saved)
        detail = edited["joint_types"][name]["physical_transition"]
        detail.update(status="pass", unknown_rules=[])
        detail["coupler_product"]["status"] = "pass"
        found = gr.differences(edited, recomputed)
        self.assertTrue(found)
        self.assertTrue(all(path.startswith(f"/joint_types/{name}/physical_transition") for path in found), found)
        edited = copy.deepcopy(saved)
        index = next(i for i, check in enumerate(edited["checks"])
                     if check["id"] == TRANSITION_DETAIL and check["location"] == name)
        edited["checks"][index] = gr._canonical(make_check(TRANSITION_DETAIL, saved["checks"][index]["clause"], 1, 1, "==",
                                                           location=name))
        found = gr.differences(edited, recomputed)
        self.assertTrue(found)
        self.assertTrue(all(path.startswith(f"/checks[{index}]") for path in found), found)
        # What is read after the comparison is the rebuilt evidence, and it is still unknown.
        rebuilt = [check for check in recomputed["checks"] if check["id"] == TRANSITION_DETAIL and check["location"] in self.boundary]
        self.assertEqual([check["status"] for check in rebuilt], ["not_evaluated"] * len(self.boundary))


# ---- branch (b): No. 14 column bars ending at the roof ----------------------------------------------------------------
class No14ColumnBarsAtTheRoof(FrameFixture):
    COLUMN = ROOF_COLUMN

    @classmethod
    def setUpClass(cls):
        cls.open_frame()
        cls.candidate = evaluate(banded_state(ROOF_COLUMN, 14))                              # No. 14 in every band
        cls.control = evaluate(banded_state(replace(ROOF_COLUMN, bar_size=11), 11))          # the same cage in No. 11
        cls.roofs = joint_types(cls.candidate["capacity"], roof=True)
        cls.detail_keys = {("capacity", ROOF_DETAIL, name) for name in cls.roofs}
        cls.shear_keys = {("joints", "joint_shear", f"joint_shear/{name}/{axis}") for name in cls.roofs for axis in "xy"}
        cls.open_keys = cls.detail_keys | cls.shear_keys

    def test_the_roof_anchorage_is_not_evaluated_for_the_seismic_development_alone(self):
        self.assertEqual(len(self.roofs), 4)
        self.assertEqual(joint_types(self.control["capacity"], roof=True), self.roofs)
        for name in self.roofs:
            detail = self.candidate["capacity"]["joint_types"][name]["physical_roof_anchorage"]
            self.assertEqual((detail["status"], detail["unknown_rules"], detail["failure_rules"]),
                             ("not_evaluated", [ROOF_RULE], []), name)
            open_rule = next(check for check in detail["checks"] if check["rule"] == ROOF_RULE)
            self.assertIn("18.8.5.3", open_rule["detail"])
            self.assertIn("No11", open_rule["detail"])
            general = next(check for check in detail["checks"] if check["rule"] == "roof_general_straight_development")
            self.assertEqual(general["status"], "pass")
            evaluated = self.control["capacity"]["joint_types"][name]["physical_roof_anchorage"]
            self.assertEqual((evaluated["status"], evaluated["unknown_rules"], evaluated["failure_rules"]), ("pass", [], []), name)

    def test_the_roof_row_is_unknown_and_the_same_cage_in_no11_is_evaluated(self):
        for name in self.roofs:
            row = self.row(self.candidate, "capacity", ROOF_DETAIL, name)
            self.assertEqual(row["status"], "not_evaluated")
            self.assertEqual(contract.classify_row(row), ("unknown", []))
            self.assertEqual(contract.classify_row(self.row(self.control, "capacity", ROOF_DETAIL, name)), ("pass", []))
        self.assertLessEqual(self.detail_keys, self.candidate["expected"]["exact"])

    def test_nothing_else_of_the_no14_columns_is_open(self):
        # Same cage in both bands: every transition detail is evaluated and passes, and no coupler pair is missing.
        details = [row for row in self.candidate["produced"] if row["id"] == TRANSITION_DETAIL]
        self.assertEqual(len(details), 12)
        self.assertEqual({row["status"] for row in details}, {"pass"})
        self.assertEqual(contract.classify_row(self.row(self.candidate, "detailing", PRODUCT_ROW, "")), ("pass", []))

    def test_the_joint_shear_strength_at_the_roof_joints_is_not_usable_and_its_row_is_unknown(self):
        for name in self.roofs:
            for axis in "xy":
                location = f"joint_shear/{name}/{axis}"
                self.assertIs(self.candidate["capacity"]["joint_types"][name]["shear"][axis]
                              ["capacity_topology_and_confinement_checked"], False)
                self.assertEqual(contract.classify_row(self.row(self.candidate, "joints", "joint_shear", location)), ("unknown", []))
                self.assertEqual(contract.classify_row(self.row(self.control, "joints", "joint_shear", location)), ("pass", []))
        self.assertLessEqual(self.shear_keys, self.candidate["expected"]["exact"])

    def test_a_code_check_holding_the_rows_is_incomplete_and_no11_passes(self):
        result = self.result_of(self.candidate, self.candidate["rows"])
        self.assertEqual((result["status"], result["accepted"], result["complete"]), ("incomplete", False, False))
        self.assertEqual(unknown_keys(result), self.open_keys)
        self.assertEqual((result["counts"]["fail"], result["counts"]["invalid"], result["counts"]["missing"]), (0, 0, 0))
        control = self.result_of(self.control, self.control["rows"])
        self.assertEqual((control["status"], control["accepted"], control["counts"]["unknown"]), ("passed", True, 0))
        for kept in (self.detail_keys, self.shear_keys):
            rows = [row for row in self.candidate["rows"] if key_of(row) not in self.open_keys - kept]
            rows += [row for row in self.candidate["baseline"] if key_of(row) in self.open_keys - kept]
            single = self.result_of(self.candidate, rows)
            self.assertEqual((single["status"], single["accepted"]), ("incomplete", False))
            self.assertEqual(unknown_keys(single), kept)
        production = self.candidate["production_only"]
        self.assertNotEqual(production["status"], "passed")
        self.assertFalse(production["accepted"])
        self.assertLessEqual(self.open_keys, unknown_keys(production))

    def test_the_candidate_is_unresolved_for_the_contract_and_the_search_and_never_accepted(self):
        assessed = self.candidate["assessed"]
        self.assertEqual((assessed["valid"], assessed["classification"], assessed["accepted"]), (True, "unresolved", False))
        self.assertIs(self.candidate["record"]["feasible"], False)
        self.assertEqual(self.candidate["search"], ("unresolved", (2, len(self.open_keys), 0.0, 0.0)))
        self.assertEqual((self.control["assessed"]["classification"], self.control["assessed"]["accepted"]), ("accepted", True))
        self.assertEqual(self.control["search"], ("feasible", (0, 0, 0.0, 0.0)))

    def test_a_record_that_claims_more_than_the_roof_rows_support_is_invalid(self):
        claimed = {**self.candidate["record"], "feasible": True, "code_check": self.control["record"]["code_check"]}
        assessed = self.assess(self.candidate, claimed)
        self.assertEqual((assessed["valid"], assessed["classification"], assessed["accepted"]), (False, "invalid", False))
        rows = [({**row, "status": "pass"} if key_of(row) in self.open_keys else row) for row in self.candidate["rows"]]
        assessed = self.assess(self.candidate, self.record_of(self.candidate, rows))
        self.assertEqual((assessed["valid"], assessed["classification"], assessed["accepted"]), (False, "invalid", False))
        self.assertEqual(len(assessed["code_check"]["invalid"]), len(self.open_keys))


def setUpModule():
    # These tests verify the 3D physical detail route, which the computed common-action route keeps (2026-11-07).
    global _DETAIL_BASIS
    _DETAIL_BASIS = sp.DETAIL_BASIS
    sp.DETAIL_BASIS = "physical_3d_v2"


def tearDownModule():
    sp.DETAIL_BASIS = _DETAIL_BASIS


if __name__ == "__main__":
    unittest.main()
