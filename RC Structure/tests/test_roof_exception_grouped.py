"""The ACI 318-19 18.7.3.1 exception per connection on the grouped route (R1, 2026-11-01).

  E  evidence: every sway state carries the connection (discontinuous above at the roof only, the roof column's own
     Ag and f'c); a roof row whose column's largest joint-face compression is below its own Ag f'c / 10 is an
     exception row with Pu and the limit, the others keep 18.7.3.2; the limit is per section (Pu between two roof
     limits splits them); at the limit the rule applies; the all-combination maximum decides and the E-only maximum
     is recorded beside it; the switch off restores the rule; the rows below the roof do not change;
  M  missing evidence is never an exemption: an invalid action inventory, or no slab layout, leaves the roof rows
     not_evaluated (the beam evidence is still required at an exempt connection);
  S  the selection's roof exemption reads the same helper on the same axial maximum and section;
  C  the contract: an exception row stands only at a roof connection of the design, passing, against that column's
     own Ag f'c / 10; the compact row keeps the exception basis; the summary counts it; the margins leave it out.

The fixture frame is not a designed structure. No solve.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract_fixture as cf                                                    # noqa: E402
import grouped_fixture as gf                                                     # noqa: E402
from test_evidence_completion_20261030 import slab_record                        # noqa: E402

import Structure_Parameters as sp                                               # noqa: E402
from Design import Code_Check_Contract as contract                               # noqa: E402
from Design import Group_Capacity as gc                                          # noqa: E402
from Design import Group_Selection as selection                                  # noqa: E402
from Design import Grouped_Design as gd                                          # noqa: E402
from Design import SMRF_Joints as joints                                         # noqa: E402
from Design.SMRF_Common import make_check                                        # noqa: E402
from Design.SMRF_Joints import SCWB_CLAUSE, SCWB_EXCEPTION_CLAUSE, discontinuous_column_exempt, evaluate_joints  # noqa: E402
from Design.SMRF_Slab_Reinforcement import slab_strength_eligibility             # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402

FLAG = "per connection: ACI 318-19 18.7.3.1 (SMRF_Joints.scwb_check)"


class GroupedRoofException(unittest.TestCase):
    def setUp(self):
        self.stack = gf.frame()
        self.stack.__enter__()
        self.state = mg.install(gf.changed_state())
        self.inventory = gc.joint_inventory(self.state)
        self.tags = sorted({c.member_tag for e in self.inventory["joints"] for c in (e["below"], e["above"])
                            if c is not None})
        self.roof = {e["below"].member_tag: e for e in self.inventory["joints"] if e["above"] is None}
        sp.SLAB_REINFORCEMENT = slab_record()

    def tearDown(self):
        self.stack.__exit__(None, None, None)

    def limit(self, tag):
        design = self.roof[tag]["below"].design
        return design.b_in * design.h_in * design.fc_ksi / 10.0

    def actions(self, axial=75.0, roof=None, cid="c1", **factors):
        """One solved combination: every column end at ``axial`` (a roof column's top end at roof(tag) if given)."""
        members = {}
        for tag in self.tags:
            top = roof(tag) if roof is not None and tag in self.roof else axial
            members[str(tag)] = {"axial_i_kip": axial, "axial_j_kip": top}
        return {"id": cid, "analysis_succeeded": True, "axial_reference": "joint_faces", "members": members, **factors}

    def rows(self, actions, record=None):
        record = slab_record() if record is None else record
        sp.SLAB_REINFORCEMENT = record
        evidence = gc.scwb_evidence(self.inventory, actions, [a["id"] for a in actions], state=self.state,
                                    slab_eligibility=slab_strength_eligibility(record))
        return evidence, {c["location"]: c for c in evaluate_joints({"joints": evidence["joints"]}) if c["id"] == "scwb"}

    def roof_rows(self, rows):
        return {loc: row for loc, row in rows.items() if loc.split("/")[0] in {e["id"] for e in self.roof.values()}}

    def test_every_sway_state_carries_the_connection_and_the_flag_says_per_connection(self):
        evidence, _rows = self.rows([self.actions()])
        self.assertEqual(evidence["roof_exemption_applied"], FLAG)
        for joint in evidence["joints"]:
            for states in joint["directions"].values():
                for sign in ("positive", "negative"):
                    connection = states[sign]["connection"]
                    self.assertIs(connection["column_discontinuous_above"], joint["is_roof"])

    def test_a_roof_row_below_its_own_limit_is_an_exception_row_with_its_evidence(self):
        _evidence, rows = self.rows([self.actions(axial=75.0)])
        roof = self.roof_rows(rows)
        self.assertTrue(roof)
        for loc, row in roof.items():
            tag = next(t for t, e in self.roof.items() if e["id"] == loc.split("/")[0])
            self.assertEqual((row["clause"], row["status"], row["units"]), (SCWB_EXCEPTION_CLAUSE, "pass", "kip"))
            self.assertEqual((row["demand"], row["capacity"]), (75.0, self.limit(tag)))
            self.assertTrue(row["details"]["roof_exemption_applied"])
            self.assertIsNone(row["details"]["ratio_required"])
        for loc, row in rows.items():
            if loc not in roof:
                self.assertEqual(row["clause"], SCWB_CLAUSE)

    def test_the_limit_is_per_section(self):
        limits = sorted({self.limit(tag) for tag in self.roof})
        self.assertGreaterEqual(len(limits), 2, "the fixture has roof columns of two sections")
        pu = 0.5 * (limits[0] + limits[-1])
        _evidence, rows = self.rows([self.actions(axial=pu)])
        for loc, row in self.roof_rows(rows).items():
            tag = next(t for t, e in self.roof.items() if e["id"] == loc.split("/")[0])
            expected = SCWB_EXCEPTION_CLAUSE if pu < self.limit(tag) else SCWB_CLAUSE
            self.assertEqual(row["clause"], expected, loc)
        self.assertEqual({row["clause"] for row in self.roof_rows(rows).values()}, {SCWB_EXCEPTION_CLAUSE, SCWB_CLAUSE})

    def test_at_the_limit_the_rule_applies(self):
        _evidence, rows = self.rows([self.actions(roof=self.limit)])
        self.assertEqual({row["clause"] for row in self.roof_rows(rows).values()}, {SCWB_CLAUSE})

    def test_the_all_combination_maximum_decides_and_the_e_only_maximum_is_recorded(self):
        high = 1.1 * max(self.limit(tag) for tag in self.roof)
        actions = [self.actions(cid="gravity", roof=lambda tag: high, ex=0.0, ey=0.0),
                   self.actions(cid="seismic", axial=75.0, ex=1.3, ey=0.39)]
        evidence, rows = self.rows(actions)
        self.assertEqual({row["clause"] for row in self.roof_rows(rows).values()}, {SCWB_CLAUSE})
        roof_ids = {e["id"] for e in self.roof.values()}
        caps = [cap for joint in evidence["joints"] if joint["id"] in roof_ids
                for states in joint["directions"].values() for cap in states["positive"]["column_capacities"]]
        self.assertTrue(caps)
        for cap in caps:
            self.assertEqual((cap["axial_max_kip"], cap["axial_max_combination_id"]), (high, "gravity"))
            self.assertEqual((cap["axial_max_e_kip"], cap["axial_max_e_combination_id"]), (75.0, "seismic"))
        # Actions that do not carry the factors leave the E-only maximum undetermined, never guessed.
        evidence, _rows = self.rows([self.actions()])
        cap = evidence["joints"][0]["directions"]["x"]["positive"]["column_capacities"][0]
        self.assertIsNone(cap["axial_max_e_kip"])

    def test_the_switch_off_restores_the_rule(self):
        with mock.patch.object(joints, "SCWB_DISCONTINUOUS_COLUMN_EXCEPTION", False):
            _evidence, rows = self.rows([self.actions(axial=75.0)])
        self.assertEqual({row["clause"] for row in rows.values()}, {SCWB_CLAUSE})

    def test_the_rows_below_the_roof_do_not_change(self):
        evidence, rows = self.rows([self.actions(axial=75.0)])
        for joint in evidence["joints"]:
            for states in joint["directions"].values():
                for sign in ("positive", "negative"):
                    del states[sign]["connection"]
        without = {c["location"]: c for c in evaluate_joints({"joints": evidence["joints"]}) if c["id"] == "scwb"}
        roof = self.roof_rows(rows)
        self.assertEqual({k: v for k, v in rows.items() if k not in roof}, {k: v for k, v in without.items() if k not in roof})
        self.assertEqual({row["clause"] for row in self.roof_rows(without).values()}, {SCWB_CLAUSE})

    def test_missing_evidence_is_never_an_exemption(self):
        action = self.actions(axial=75.0)
        _evidence, rows = self.rows([action, dict(action)])                    # a duplicated combination id
        self.assertTrue(all(row["status"] == "not_evaluated" for row in self.roof_rows(rows).values()))
        _evidence, rows = self.rows([action], record=slab_record(layout=False))   # no slab layout: beams unknown
        roof = self.roof_rows(rows)
        self.assertTrue(roof and all(row["status"] == "not_evaluated" for row in roof.values()))
        self.assertTrue(all(row["clause"] == SCWB_CLAUSE for row in roof.values()))

    def test_the_selection_reads_the_same_helper_on_the_same_inputs(self):
        actions = [self.actions(cid="a", axial=75.0), self.actions(cid="b", roof=lambda tag: 0.5 * self.limit(tag))]
        evidence, rows = self.rows(actions)
        ranges = selection._axial_ranges(sorted(self.roof), actions)
        for tag, entry in self.roof.items():
            joint = next(j for j in evidence["joints"] if j["id"] == entry["id"])
            states = joint["directions"]["x"]["positive"]
            cap, connection = states["column_capacities"][0], states["connection"]
            self.assertEqual(ranges[tag]["j"][1], cap["axial_max_kip"])
            design = self.state.member(tag).design
            self.assertEqual(design.b_in * design.h_in, connection["column_gross_area_in2"])
            self.assertEqual(design.fc_ksi, connection["column_fc_ksi"])
            exempt = discontinuous_column_exempt(ranges[tag]["j"][1], design.b_in * design.h_in, design.fc_ksi)[0]
            self.assertEqual(exempt, rows[f"{entry['id']}/x/positive"]["clause"] == SCWB_EXCEPTION_CLAUSE)


class ContractExceptionRows(unittest.TestCase):
    def setUp(self):
        self.stack = gf.frame()
        self.stack.__enter__()
        self.state = gf.uniform_state()
        self.expected = contract.expected_inventory(self.state)
        self.limits = self.expected["scwb_exception_limits"]

    def tearDown(self):
        self.stack.__exit__(None, None, None)

    def exception_row(self, location, capacity, demand=75.0):
        check = make_check("scwb", SCWB_EXCEPTION_CLAUSE, demand, capacity, units="kip", location=location,
                           details={"ratio_provided": 0.9, "ratio_required": None, "roof_exemption_applied": True,
                                    "exception": {"clause": "ACI 318-19 18.7.3.1", "pu_max_kip": demand,
                                                  "limit_kip": capacity}})
        return contract.compact_row("joints", check)

    def rows_with(self, row):
        key = ("joints", "scwb", row["location"])
        return [r for r in cf.rows_for(self.state) if (r["family"], r["id"], r["location"]) != key] + [row]

    def test_the_inventory_lists_the_roof_connections_only(self):
        self.assertTrue(self.limits)
        with mg.installed(self.state):
            roof = {e["id"] for e in gc.joint_inventory(self.state)["joints"] if e["above"] is None}
        self.assertEqual({loc.split("/")[0] for loc in self.limits}, roof)
        self.assertTrue(set(self.limits) <= {loc for _f, _n, loc in self.expected["exact"] if _n == "scwb"})

    def test_an_exception_row_at_its_own_limit_completes_the_inventory_and_keeps_its_basis(self):
        location, limit = sorted(self.limits.items())[0]
        row = self.exception_row(location, limit)
        self.assertEqual(row["details"]["exception"]["limit_kip"], limit)
        self.assertTrue(row["details"]["roof_exemption_applied"])
        result = contract.code_check_result(self.rows_with(row), self.expected)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(contract.scwb_summary([row])["exceptions_18_7_3_1"], 1)

    def test_an_exception_row_elsewhere_or_at_another_limit_is_invalid(self):
        location, limit = sorted(self.limits.items())[0]
        other = next(loc for _f, n, loc in sorted(self.expected["exact"]) if n == "scwb" and loc not in self.limits)
        for row in (self.exception_row(location, 1.5 * limit), self.exception_row(other, limit),
                    self.exception_row(location, limit, demand=2.0 * limit)):
            with self.subTest(location=row["location"], capacity=row["capacity"], demand=row["demand"]):
                self.assertEqual(contract.code_check_result(self.rows_with(row), self.expected)["status"], "invalid")

    def test_an_exception_row_must_compare_strictly_below_without_a_tolerance(self):
        location, limit = sorted(self.limits.items())[0]
        reversed_row = self.exception_row(location, limit)
        reversed_row.update(comparison=">=", demand=2.0 * limit)                   # passes as '>=' by its numbers
        tolerant = self.exception_row(location, limit, demand=limit)
        tolerant["details"]["numerical_tolerance"] = 1e-6
        at_limit = self.exception_row(location, limit, demand=limit)                 # '<=', no tolerance: the rule applies
        for row in (reversed_row, tolerant, at_limit):
            with self.subTest(comparison=row["comparison"], demand=row["demand"]):
                self.assertEqual(contract.code_check_result(self.rows_with(row), self.expected)["status"], "invalid")

    def test_with_the_exception_switched_off_no_location_may_claim_it(self):
        with mock.patch.object(joints, "SCWB_DISCONTINUOUS_COLUMN_EXCEPTION", False):
            expected = contract.expected_inventory(self.state)
        self.assertEqual(expected["scwb_exception_limits"], {})
        location, limit = sorted(self.limits.items())[0]
        self.assertEqual(contract.code_check_result(self.rows_with(self.exception_row(location, limit)), expected)["status"],
                         "invalid")

    def test_the_margins_leave_exception_rows_out(self):
        exception = make_check("scwb", SCWB_EXCEPTION_CLAUSE, 300.0, 310.0, units="kip", location="roof/x/positive",
                               details={"roof_exemption_applied": True})
        rule = make_check("scwb", SCWB_CLAUSE, 100.0, 200.0, units="kip-in", location="floor/x/positive",
                          details={"roof_exemption_applied": False})
        margins = gd._margins({"groups": {}}, {"checks": []}, [exception, rule], {"checks": []})
        self.assertEqual((margins["scwb"]["at"], margins["scwb"]["demand_over_capacity"]), ("floor/x/positive", 0.5))


if __name__ == "__main__":
    unittest.main()
