"""The bounded slab-layout proposal stage passes over a layer spacing it has already found unplaceable, keeps the
catalog order and still bounds the phase searches it makes (2026-11-05). No solve."""
import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NAMES = ("x_top", "x_bottom", "y_top", "y_bottom")


def candidate(spacings, bar_size=4):
    return {"candidate_id": f"No{bar_size}/" + ",".join(f"{n}:{s:g}" for n, s in zip(NAMES, spacings)),
            "layout": {"bar_size": bar_size, "layers": {n: {"spacing_in": float(s)} for n, s in zip(NAMES, spacings)}}}


def catalog(values=(11, 10, 9), bar_size=4):
    """Layouts ordered by total steel like the production catalog: the widest spacings first."""
    rows = sorted(itertools.product(values, repeat=4), key=lambda s: (sum(1.0 / v for v in s), [-v for v in s]))
    return [candidate(s, bar_size) for s in rows]


def placer(unplaceable, log):
    """A phase search that fails exactly the layers whose (name, spacing) is in ``unplaceable``."""
    def place(item):
        log.append(item["candidate_id"])
        status = {n: ("fail" if (n, r["spacing_in"]) in unplaceable else "pass") for n, r in item["layout"]["layers"].items()}
        overall = "pass" if all(v == "pass" for v in status.values()) else "fail"
        return overall, status, {"candidate_id": item["candidate_id"], "status": overall}
    return place


class BoundedLayoutProposals(unittest.TestCase):
    def setUp(self):
        from Design.SMRF_Computed_Coordination import bounded_layout_proposals
        self.walk = bounded_layout_proposals

    def test_a_spacing_no_layer_can_take_is_searched_once_per_layer(self):
        items, log = catalog(), []
        bad = {(n, 11.0) for n in NAMES}
        first_good = next(k for k, c in enumerate(items) if all(r["spacing_in"] != 11.0 for r in c["layout"]["layers"].values()))
        self.assertGreater(first_good, 8)                         # a walk of the first 8 layouts alone never reaches it
        chosen, attempts, passed_over, searches, unplaceable = self.walk(items, placer(bad, log), limit=8)
        self.assertEqual(chosen["candidate_id"], items[first_good]["candidate_id"])
        self.assertEqual(searches, 2)                             # the all-11 layout, then the first without an 11
        self.assertEqual(passed_over, first_good - 1)
        self.assertEqual(len(attempts), 2)
        self.assertEqual({k[0] for k in unplaceable}, set(NAMES))
        self.assertEqual(log, [items[0]["candidate_id"], items[first_good]["candidate_id"]])

    def test_catalog_order_is_kept_and_the_first_layout_wins_when_it_places(self):
        items, log = catalog(), []
        chosen, attempts, passed_over, searches, unplaceable = self.walk(items, placer(set(), log))
        self.assertEqual((chosen["candidate_id"], searches, passed_over, unplaceable), (items[0]["candidate_id"], 1, 0, []))

    def test_only_the_failed_layer_is_remembered(self):
        items, log = catalog(), []
        bad = {("y_top", 11.0)}
        chosen, _attempts, passed_over, searches, unplaceable = self.walk(items, placer(bad, log))
        self.assertEqual(chosen["layout"]["layers"]["y_top"]["spacing_in"], 10.0)
        self.assertEqual([r["spacing_in"] for n, r in chosen["layout"]["layers"].items() if n != "y_top"], [11.0] * 3)
        self.assertEqual(searches, 2)
        self.assertEqual(unplaceable, [("y_top", 4, 11.0, 4)])
        for name in log:                                          # nothing searched twice
            self.assertEqual(log.count(name), 1)
        self.assertGreaterEqual(passed_over, 0)

    def test_the_bound_counts_searches_and_a_search_without_layer_evidence_passes_nothing_over(self):
        items, log = catalog(), []

        def rows_do_not_fit(item):
            log.append(item["candidate_id"])
            return "actual_beam_rows_do_not_fit", None, {"candidate_id": item["candidate_id"], "status": "actual_beam_rows_do_not_fit"}
        chosen, attempts, passed_over, searches, unplaceable = self.walk(items, rows_do_not_fit, limit=5)
        self.assertIsNone(chosen)
        self.assertEqual((searches, len(attempts), passed_over, unplaceable), (5, 5, 0, []))
        self.assertEqual(log, [c["candidate_id"] for c in items[:5]])

    def test_a_layer_is_keyed_with_its_bar_and_the_largest_bar_of_the_layout(self):
        mixed = [candidate((11, 11, 11, 11), 4), candidate((11, 11, 11, 11), 5)]
        log = []

        def place(item):
            log.append(item["candidate_id"])
            size = item["layout"]["bar_size"]
            status = {n: ("fail" if size == 4 else "pass") for n in NAMES}
            return ("fail" if size == 4 else "pass"), status, {"candidate_id": item["candidate_id"]}
        chosen, _attempts, passed_over, searches, _unplaceable = self.walk(mixed, place)
        self.assertEqual((chosen["layout"]["bar_size"], passed_over, searches), (5, 0, 2))   # another bar is another search


if __name__ == "__main__":
    unittest.main()
