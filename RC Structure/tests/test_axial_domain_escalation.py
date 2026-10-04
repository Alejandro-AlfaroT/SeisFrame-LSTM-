"""Uniform design, screening repair item 2 (case_0006 of the 2 October screens): a factored axial load outside a
section's strength domain is an infeasible cage in the strong-column escalation and an infeasible section in the
search, never an exception out of design_structure. The demand and the domain stay in the record."""
import contextlib
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Structure_Parameters as sp                                              # noqa: E402
from Design import Design_Driver as driver                                     # noqa: E402
from Design import Verify_Designs as verify                                    # noqa: E402
from Design.Config import DesignConfig                                         # noqa: E402
from Design.SMRF_Common import SectionAxialDomainError                         # noqa: E402

MESSAGE = "Axial force 1989.61 outside nominal section domain [-211.2, 1573.24]."


@contextlib.contextmanager
def restored_parameters():
    snapshot = dict(vars(sp))
    try:
        yield
    finally:
        for key in set(vars(sp)) - set(snapshot):
            delattr(sp, key)
        vars(sp).update(snapshot)


class StrongColumnEscalation(unittest.TestCase):
    """_scwb_column_steel with the joint adapter mocked: light cages raise, heavier ones price."""

    def run_escalation(self, raising_below_ast, satisfied_at_ast=None):
        check = {"demand": 1000.0, "details": {"ratio_provided": 0.5}}
        sway = {"column_capacities": [{"factored_axial_kip": 1989.61}]}
        joint_scwb = {"_failing": [(check, sway, "x")], "_state": {"reinforcement": {}}}
        pool = [(9, 3, 3, 1, 8.0), (11, 3, 3, 1, 12.5), (11, 4, 4, 2, 18.7), (14, 4, 4, 2, 27.0)]

        def capacity(record, member, axis, face, p):
            cage = (record["reinforcement"]["col_bar_size"], record["reinforcement"]["col_top_bars"],
                    record["reinforcement"]["col_bot_bars"], record["reinforcement"]["col_side_bars"])
            ast = next(c[4] for c in pool if c[:4] == cage)
            if ast < raising_below_ast:
                raise SectionAxialDomainError(MESSAGE)
            return {"mn_kip_in": 2000.0 if satisfied_at_ast is not None and ast >= satisfied_at_ast else 500.0 + ast}

        with mock.patch("Redesign.col_candidates_for_beam_bars", return_value=(pool, True)), \
             mock.patch("Redesign.scwb_cage_order", side_effect=lambda c, cfg=None: (0, c[4], c[0])), \
             mock.patch("Design.SMRF_Joint_Adapter.record_section_capacity", side_effect=capacity):
            with restored_parameters():
                sp.B_COL = sp.H_COL = 18.0
                update, exhausted = driver._scwb_column_steel(DesignConfig(), joint_scwb)
        return update, exhausted, joint_scwb["axial_domain"]

    def test_a_cage_outside_its_domain_is_rejected_by_name_and_a_heavier_one_is_installed(self):
        update, exhausted, domain = self.run_escalation(raising_below_ast=15.0, satisfied_at_ast=18.0)
        self.assertEqual((update, exhausted), ({"bar_size": 11, "n_top": 4, "n_bot": 4, "n_side": 2}, False))
        self.assertEqual((domain["rejected_cages"], domain["cages_offered"], domain["all_cages_rejected"]), (2, 4, False))
        self.assertIn("1989.61", domain["first_rejection"])
        self.assertIn("[-211.2, 1573.24]", domain["first_rejection"])

    def test_the_fallback_prices_only_cages_inside_their_domain(self):
        update, exhausted, domain = self.run_escalation(raising_below_ast=15.0)          # none satisfies 1000 kip-in
        self.assertTrue(exhausted)
        self.assertEqual(update, {"bar_size": 14, "n_top": 4, "n_bot": 4, "n_side": 2})     # the heaviest priced cage
        self.assertEqual(domain["rejected_cages"], 2)

    def test_every_cage_outside_its_domain_exhausts_the_escalation_without_an_update(self):
        update, exhausted, domain = self.run_escalation(raising_below_ast=100.0)
        self.assertEqual((update, exhausted), (None, True))
        self.assertTrue(domain["all_cages_rejected"])
        self.assertEqual(domain["rejected_cages"], 4)


class SectionEscalation(unittest.TestCase):
    def test_the_search_steps_the_column_and_keeps_the_demand_instead_of_raising(self):
        """The first steel pass raises for the first rung (as case_0006 did at 18 in); the design goes on at the
        next column, the escalation is recorded with its demand and domain, and no design iteration is spent."""
        real = driver._steel_pass
        calls = []

        def first_rung_outside_domain(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise SectionAxialDomainError(MESSAGE)
            return real(*args, **kwargs)

        with restored_parameters():
            sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR = 1, 1, 2
            sp.BAY_X = sp.BAY_Y = 180.0
            sp.STORY_H = 144.0
            sp.NUM_MODES = 3
            sp.apply_seismic_site("sdc_d_low")
            with mock.patch.object(driver, "_steel_pass", side_effect=first_rung_outside_domain):
                record = driver.design_structure(cfg=verify.probe_config("2026-10-02"), max_section_iter=2, max_steel_iter=2, verbose=False)
        escalations = record["axial_domain_escalations"]
        self.assertEqual(len(escalations), 1)
        first = escalations[0]
        self.assertEqual((first["escalation"], first["design_iterations_used"]), (1, 0))
        self.assertIn("1989.61", first["error"])
        self.assertIn("[-211.2, 1573.24]", first["error"])
        self.assertEqual(first["column_section"][:2], [18.0, 18.0])                                 # the fixture's starting rung
        # strength before geometry (2026-10-03): the same size at the top concrete grade, not a larger column
        self.assertEqual(record["history"][0]["column_section"][:2], first["column_section"][:2])
        self.assertGreater(record["history"][0]["column_section"][2], first["column_section"][2])
        self.assertEqual(record["history"][0]["column_section"][2], 10.0)
        self.assertEqual(record["history"][0]["iteration"], 1)                                     # no iteration was spent
        self.assertEqual(record["iterations"], len(record["history"]))

    def test_a_ladder_with_no_larger_or_stronger_column_is_a_precise_exhaustion(self):
        with restored_parameters():
            sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR = 1, 1, 2
            sp.BAY_X = sp.BAY_Y = 180.0
            sp.STORY_H = 144.0
            sp.NUM_MODES = 3
            sp.apply_seismic_site("sdc_d_low")
            top = driver.column_ladder()[-1]
            with mock.patch.object(driver, "column_ladder", return_value=[top]), \
                 mock.patch.object(driver, "_steel_pass", side_effect=SectionAxialDomainError(MESSAGE)):
                with self.assertRaisesRegex(RuntimeError, "No column in the ladder carries this factored axial load"):
                    driver.design_structure(cfg=verify.probe_config("2026-10-02"), max_section_iter=2, max_steel_iter=2, verbose=False)


if __name__ == "__main__":
    unittest.main()
