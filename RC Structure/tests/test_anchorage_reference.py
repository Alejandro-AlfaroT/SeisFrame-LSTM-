"""Section-equilibrium anchorage reference (Analysis/Anchorage_Reference v3, 2026-09-30): the section analysis
reproduces Park and Ruitong's published Unit 1 nominal strengths; events are individual layer strain crossings
with interpolation; the through-bar pairing is by layer identity and invariant to the order the layers are
listed in; a deliberately unequal two-layer section exposes the difference between first-layer and all-layers
yield; the strain-profile treatments and the far-face states are reported, not assumed; the fully mobilised
two-face profile is accepted only inside its domain (repair review of 2026-09-30)."""
import copy
import math
import sys
import unittest
from pathlib import Path

RC_DIR = Path(__file__).resolve().parents[1]
if str(RC_DIR) not in sys.path:
    sys.path.insert(0, str(RC_DIR))

from Analysis import Anchorage_Reference as ar  # noqa: E402


def two_layer_section(order="natural"):
    """A synthetic section with an unequal two-layer tension group: 3 #9 at 2.5 in and 2 #9 at 5.5 in from the
    top, 2 #8 at the bottom, through a 30-in column."""
    a9, a8 = 1.0, 0.79
    layers = [{"depth_from_top": 2.5, "area": 3 * a9, "bars": 3, "db": 1.128, "group": "top"},
              {"depth_from_top": 5.5, "area": 2 * a9, "bars": 2, "db": 1.128, "group": "top"},
              {"depth_from_top": 24.0 - 2.5, "area": 2 * a8, "bars": 2, "db": 1.0, "group": "bottom"}]
    if order == "reversed":
        layers = list(reversed(layers))
    steel = ar.Steel(fy=60.0, es=29000.0, fu=90.0, eps_sh=0.010, esh=1000.0, eps_su=0.12)
    return ar.Section(b=16.0, h=24.0, layers=copy.deepcopy(layers), concrete=ar.Concrete(fc=5.0), steel=steel, anchorage_length=30.0, fc_anchorage=5.0, label="two_layer")


class Unit1Section(unittest.TestCase):
    def test_nominal_strengths_match_the_paper_table_4(self):
        sec = ar.unit1_section()
        hog = ar.aci_nominal(sec, "hogging")["Mn"] / 1e6      # kN m
        sag = ar.aci_nominal(sec, "sagging")["Mn"] / 1e6
        self.assertAlmostEqual(hog, 114.8, delta=0.02 * 114.8)     # paper Table 4, M2u
        self.assertAlmostEqual(sag, 54.8, delta=0.02 * 54.8)       # M1u

    def test_first_layer_yield_is_the_extreme_layer_at_its_yield_strain(self):
        sec = ar.unit1_section()
        bond = ar.Bond.sezen(sec.fc_anchorage, "MPa")
        ref = ar.reference(sec, bond, "hogging")
        ev = ref["events"]
        f = ev["first_layer_yield"]
        self.assertEqual(f["layer"], "top1")                                   # the outer layer yields first
        self.assertAlmostEqual(f["layers"]["top1"]["eps"], sec.steel.eps_y, delta=1e-9)
        self.assertLess(f["M"], ev["all_tension_layers_yielded"]["M"])
        self.assertLess(ev["all_tension_layers_yielded"]["M"], ev["hardening_onset"]["M"])
        self.assertLess(f["M"], ev["aci_nominal_Mn"]["Mn"])
        # the outer layer's slip at yield is Sezen's closed form
        s_closed = sec.steel.fy ** 2 * 16.0 / (8.0 * bond.u_elastic * sec.steel.es)
        self.assertAlmostEqual(f["layers"]["top1"]["slip_sezen_bilinear"], s_closed, delta=1e-6 * s_closed)
        self.assertTrue(f["valid"])
        self.assertGreater(f["theta_slip"], 0.0)

    def test_bond_values_are_the_sezen_psi_forms(self):
        bond = ar.Bond.sezen(45.9, "MPa")
        fc_psi = 45.9 * ar.PSI_PER_MPA
        self.assertAlmostEqual(bond.u_elastic, 12.0 * math.sqrt(fc_psi) / ar.PSI_PER_MPA, places=9)
        self.assertAlmostEqual(bond.u_yielded, 6.0 * math.sqrt(fc_psi) / ar.PSI_PER_MPA, places=9)
        bond_us = ar.Bond.sezen(5.0, "ksi")
        self.assertAlmostEqual(bond_us.u_elastic, 12.0 * math.sqrt(5000.0) / 1000.0, places=9)


class LayerBookkeeping(unittest.TestCase):
    def test_layer_order_invariance(self):
        a = ar.reference(two_layer_section("natural"), ar.Bond.sezen(5.0, "ksi"), "hogging")
        b = ar.reference(two_layer_section("reversed"), ar.Bond.sezen(5.0, "ksi"), "hogging")
        for name in ("first_layer_yield", "all_tension_layers_yielded", "hardening_onset"):
            self.assertEqual(a["events"][name]["layer"], b["events"][name]["layer"])
            self.assertAlmostEqual(a["events"][name]["M"], b["events"][name]["M"], places=9)
            self.assertAlmostEqual(a["events"][name]["theta_slip"], b["events"][name]["theta_slip"], places=12)
        for ra, rb in zip(a["rows"][::25], b["rows"][::25]):
            self.assertAlmostEqual(ra["anchorage_ratio_max"], rb["anchorage_ratio_max"], places=12)
            for lid in ra["layers"]:
                self.assertAlmostEqual(ra["layers"][lid]["sigma"], rb["layers"][lid]["sigma"], places=9)
                if ra["layers"][lid]["in_tension"]:
                    self.assertAlmostEqual(ra["layers"][lid]["slip_sezen_bilinear"], rb["layers"][lid]["slip_sezen_bilinear"], places=12)

    def test_unequal_two_layer_section_separates_the_yield_events(self):
        ref = ar.reference(two_layer_section(), ar.Bond.sezen(5.0, "ksi"), "hogging")
        ev = ref["events"]
        self.assertEqual(ev["first_layer_yield"]["layer"], "top1")
        self.assertEqual(ev["all_tension_layers_yielded"]["layer"], "top2")
        self.assertGreater(ev["all_tension_layers_yielded"]["M"], 1.02 * ev["first_layer_yield"]["M"])
        f = ev["first_layer_yield"]
        # the inner layer is still elastic at first-layer yield and slips less
        self.assertLess(f["layers"]["top2"]["eps"], 0.9 * f["layers"]["top1"]["eps"])
        self.assertLess(f["layers"]["top2"]["slip_sezen_bilinear"], f["layers"]["top1"]["slip_sezen_bilinear"])
        # the compatible face rotation lies between the two layers' implied rotations
        pl = f["face_rotation"]["sezen_bilinear"]["per_layer"]
        lo, hi = sorted(v["implied_rotation"] for v in pl.values())
        self.assertLessEqual(lo, f["theta_slip"]); self.assertLessEqual(f["theta_slip"], hi)
        # events are strain crossings, so a plateau stored a rounding below fy cannot defer them
        self.assertAlmostEqual(f["layers"]["top1"]["eps"], 60.0 / 29000.0, delta=1e-9)

    def test_strain_profiles_and_states_are_reported_not_assumed(self):
        sec = two_layer_section()
        steel, bond = sec.steel, ar.Bond.sezen(5.0, "ksi")
        db = 1.128
        # elastic bar: both profiles coincide with the closed form
        e = ar.layer_anchorage(steel, bond, db, 40.0, 40.0 / 29000.0, -10.0, 30.0)
        self.assertAlmostEqual(e["slip_sezen_bilinear"], 40.0 ** 2 * db / (8.0 * bond.u_elastic * steel.es), places=12)
        self.assertAlmostEqual(e["slip_inverse_law"], e["slip_sezen_bilinear"], places=12)
        self.assertEqual(e["state"], "zero-stress zone inside the joint"); self.assertTrue(e["valid"])
        # hardened bar: the two profiles differ and both are reported
        sig = float(steel.stress(0.011)); y = ar.layer_anchorage(steel, bond, db, sig, 0.011, -10.0, 30.0)
        self.assertGreater(y["slip_inverse_law"], y["slip_sezen_bilinear"])
        self.assertGreater(y["L_yielded"], 0.0)
        # a compressive far face that does not fit is flagged invalid, not silently clipped
        x = ar.layer_anchorage(steel, bond, db, sig, 0.011, -55.0, 12.0)
        self.assertFalse(x["valid"]); self.assertIn("capacity exceeded", x["state"]); self.assertGreater(x["anchorage_ratio"], 1.0)
        # tension at both faces: fully mobilised profile with a positive minimum stress inside the joint
        t = ar.layer_anchorage(steel, bond, db, 55.0, 55.0 / 29000.0, 50.0, 10.0)
        self.assertEqual(t["state"], "bond fully mobilised: tension at both faces"); self.assertTrue(t["valid"])
        self.assertGreater(t["sigma_min_inside_joint"], 0.0); self.assertAlmostEqual(t["anchorage_ratio"], 1.0)
        self.assertLess(t["slip_sezen_bilinear"], e["slip_sezen_bilinear"] * (55.0 / 40.0) ** 2)     # shorter than the free-end elongation


def domain_errors(r, length, fy):
    """The domain probe (repair review of 2026-09-30), reproduced: a two-face result must keep both segments
    inside the available length, its interior minimum below both endpoints and the elastic cap, and its yielded
    length within the near segment."""
    if "sigma_min_inside_joint" not in r:
        return []
    errors = []
    if not 0.0 <= r["L_near"] <= length:
        errors.append("near segment outside available length")
    if not 0.0 <= r["L_far"] <= length:
        errors.append("far segment outside available length")
    if not 0.0 <= r["sigma_min_inside_joint"] <= min(r["sigma_near"], r["sigma_far"], fy) + 1e-9:
        errors.append("internal minimum exceeds an endpoint or elastic-domain cap")
    if r["L_yielded"] > r["L_near"] + 1e-9:
        errors.append("yielded segment longer than near segment")
    return errors


class TwoFaceDomain(unittest.TestCase):
    def test_zero_far_stress_counterexample_is_capacity_exceeded(self):
        # The reproduced state from the default section (18-in column, #6 bar, far face unstressed): revision 2
        # returned a fully mobilised profile with a negative far segment and an interior minimum above the zero endpoint
        sec = ar.frame_section(); bond = ar.Bond.sezen(sec.fc_anchorage, "ksi")
        r = ar.layer_anchorage(sec.steel, bond, sec.layers[0]["db"], 77.7515, 0.0338428, 0.0, sec.anchorage_length)
        self.assertEqual(r["status"], "invalid"); self.assertFalse(r["valid"]); self.assertIn("capacity exceeded", r["state"])
        self.assertNotIn("sigma_min_inside_joint", r)
        self.assertGreater(r["anchorage_ratio"], 1.0); self.assertAlmostEqual(r["anchorage_ratio"], r["required_length_ratio"])
        # the required-length screen is retained: near length = yielded + elastic development to zero (21.10 in > 18 in), no far length
        self.assertAlmostEqual(r["L_near"], r["L_yielded"] + 60.0 * sec.layers[0]["db"] / (4.0 * bond.u_elastic), places=9)
        self.assertAlmostEqual(r["L_near"], 21.1034, delta=5e-4); self.assertEqual(r["L_far"], 0.0)
        self.assertAlmostEqual(r["anchorage_ratio"], r["L_near"] / sec.anchorage_length, places=12)
        self.assertGreater(r["sigma_far_required"], 0.0)                 # bond would need tension at the far face
        self.assertEqual(domain_errors(r, sec.anchorage_length, sec.steel.fy), [])

    def test_two_face_profile_is_accepted_only_inside_its_domain(self):
        steel, bond, db = two_layer_section().steel, ar.Bond.sezen(5.0, "ksi"), 1.128
        ey = 1.0 / 29000.0
        # inside the domain: tension at both faces, the zero-stress zone does not fit, both segments nonnegative and summing to L
        t = ar.layer_anchorage(steel, bond, db, 55.0, 55.0 * ey, 50.0, 10.0)
        self.assertEqual(t["status"], "valid"); self.assertIn("fully mobilised", t["state"]); self.assertEqual(t["anchorage_ratio"], 1.0)
        self.assertAlmostEqual(t["L_near"] + t["L_far"], 10.0, places=9); self.assertGreaterEqual(t["L_far"], 0.0); self.assertGreaterEqual(t["L_near"], t["L_yielded"])
        self.assertLessEqual(t["sigma_min_inside_joint"], 50.0); self.assertGreater(t["sigma_min_inside_joint"], 0.0)
        self.assertGreater(t["required_length_ratio"], 1.0)              # the imposed ratio of 1 is not the zero-stress requirement
        self.assertEqual(domain_errors(t, 10.0, steel.fy), [])
        # the far segment would be negative: capacity exceeded, the required-length screen retained, nothing clamped
        x = ar.layer_anchorage(steel, bond, db, 55.0, 55.0 * ey, 5.0, 10.0)
        self.assertEqual(x["status"], "invalid"); self.assertIn("capacity exceeded", x["state"]); self.assertNotIn("sigma_min_inside_joint", x)
        self.assertGreater(x["anchorage_ratio"], 1.0); self.assertGreater(x["sigma_far_required"], 5.0)
        # a far face yielded in tension is outside the elastic far-segment formula: unsupported, not accepted
        u = ar.layer_anchorage(steel, bond, db, 55.0, 55.0 * ey, 65.0, 10.0)
        self.assertEqual(u["status"], "unsupported"); self.assertFalse(u["valid"]); self.assertNotIn("sigma_min_inside_joint", u)
        # a far face carrying more tension than the near face: unsupported
        g = ar.layer_anchorage(steel, bond, db, 20.0, 20.0 * ey, 55.0, 10.0)
        self.assertEqual(g["status"], "unsupported"); self.assertFalse(g["valid"])
        # the yielded length alone longer than the joint: capacity exceeded
        sig = float(steel.stress(0.05)); y = ar.layer_anchorage(steel, bond, db, sig, 0.05, 30.0, 2.0)
        self.assertEqual(y["status"], "invalid"); self.assertIn("yielded length", y["state"])
        # every branch reports the same bookkeeping
        for r in (t, x, u, g, y):
            for key in ("L_near", "L_yielded", "L_far", "anchorage_ratio", "required_length_ratio", "slip_sezen_bilinear", "slip_inverse_law", "state", "status", "valid"):
                self.assertIn(key, r)
            self.assertEqual(r["valid"], r["status"] == "valid")

    def test_no_two_face_state_leaves_its_domain_in_the_reference_curves(self):
        for sec, unit in ((ar.frame_section(), "ksi"), (ar.unit1_section(), "MPa"), (two_layer_section(), "ksi")):
            bond = ar.Bond.sezen(sec.fc_anchorage, unit)
            for direction in ("hogging", "sagging"):
                ref = ar.reference(sec, bond, direction)
                for r in ref["rows"]:
                    for lr in r["layers"].values():
                        if not lr["in_tension"]:
                            continue
                        self.assertEqual(domain_errors(lr, sec.anchorage_length, sec.steel.fy), [], (sec.label, direction, r["M"]))
                        alt = ar.layer_anchorage(sec.steel, bond, lr["db"], lr["sigma"], lr["eps"], 0.0, sec.anchorage_length)
                        self.assertEqual(domain_errors(alt, sec.anchorage_length, sec.steel.fy), [])
                        self.assertNotIn("fully mobilised", alt["state"])          # a zero far stress never yields a two-face profile
                        self.assertEqual(lr["valid_far_unstressed"], alt["valid"]); self.assertEqual(lr["status_far_unstressed"], alt["status"])
                        self.assertEqual(alt["valid"], alt["anchorage_ratio"] <= 1.0)
                self.assertEqual(ref["unsupported_rows"], sum(1 for r in ref["rows"] if r["unsupported"]))
                self.assertEqual(ref["far_unstressed_invalid_rows"], sum(1 for r in ref["rows"] if not r["valid_far_unstressed"]))


class StatusContract(unittest.TestCase):
    """The two-face review of 2026-09-30, bookkeeping follow-ups: every branch returns the complete record,
    the non-tension branch included, and the export keeps everything the far-unstressed alternative holds."""
    KEYS = ("sigma_near", "eps_near", "sigma_far", "L_near", "L_yielded", "L_far", "anchorage_ratio", "required_length_ratio",
            "slip_sezen_bilinear", "slip_inverse_law", "state", "status", "valid")

    def test_every_branch_returns_the_complete_record(self):
        steel, bond, db = two_layer_section().steel, ar.Bond.sezen(5.0, "ksi"), 1.128
        ey = 1.0 / 29000.0
        hard, harder = float(steel.stress(0.011)), float(steel.stress(0.05))
        cases = {"not_applicable": [(0.0, 0.0, 0.0, 30.0), (-1.0, -1.0 * ey, 0.0, 30.0), (-1.0, -1.0 * ey, 40.0, 30.0)],
                 "valid": [(40.0, 40.0 * ey, -10.0, 30.0), (40.0, 40.0 * ey, 0.0, 30.0), (55.0, 55.0 * ey, 50.0, 10.0)],
                 "invalid": [(hard, 0.011, -55.0, 12.0), (55.0, 55.0 * ey, 5.0, 10.0), (55.0, 55.0 * ey, 0.0, 10.0), (harder, 0.05, 30.0, 2.0)],
                 "unsupported": [(55.0, 55.0 * ey, 65.0, 10.0), (20.0, 20.0 * ey, 55.0, 10.0)]}
        for status, states in cases.items():
            for sig, eps, far, length in states:
                r = ar.layer_anchorage(steel, bond, db, sig, eps, far, length)
                self.assertEqual(set(self.KEYS) - set(r), set(), (status, sig, far))
                self.assertEqual(r["status"], status, (sig, far, r["state"])); self.assertIn(r["status"], ar.STATUSES)
                self.assertEqual(r["valid"], status in ("valid", "not_applicable"))
                self.assertEqual(domain_errors(r, length, steel.fy), [])

    def test_zero_and_compressive_near_face_is_a_complete_zero_demand_record(self):
        steel, bond = two_layer_section().steel, ar.Bond.sezen(5.0, "ksi")
        for sig in (0.0, -1.0):                                             # direct calls at 0 and -1 ksi
            r = ar.layer_anchorage(steel, bond, 1.128, sig, sig / 29000.0, 0.0, 30.0)
            for key in ("L_near", "L_yielded", "L_far", "anchorage_ratio", "required_length_ratio", "slip_sezen_bilinear", "slip_inverse_law"):
                self.assertEqual(r[key], 0.0, key)
            self.assertEqual(r["status"], "not_applicable"); self.assertTrue(r["valid"])
            self.assertIn("compression-bar slip not modelled", r["state"])   # zero slip is bookkeeping, not a compression-slip model
            self.assertNotIn("sigma_min_inside_joint", r)

    def test_a_tension_group_layer_in_compression_does_not_break_the_reference(self):
        # a third 'top' layer placed near the bottom face: under hogging it belongs to the nominal tension group but is compressed
        sec = two_layer_section()
        layers = [{k: v for k, v in ly.items() if k != "id"} for ly in sec.layers] + [{"depth_from_top": 22.5, "area": 1.0, "bars": 1, "db": 1.128, "group": "top"}]
        sec = ar.Section(b=16.0, h=24.0, layers=layers, concrete=ar.Concrete(fc=5.0), steel=sec.steel, anchorage_length=30.0, fc_anchorage=5.0, label="deep_top_layer")
        deep = next(ly["id"] for ly in sec.layers if ly["depth_from_top"] == 22.5)
        ref = ar.reference(sec, ar.Bond.sezen(5.0, "ksi"), "hogging")          # raised KeyError before the complete record
        compressed = [r["layers"][deep] for r in ref["rows"] if r["layers"][deep]["sigma"] <= 0.0]
        self.assertGreater(len(compressed), 0)
        for lr in compressed:
            self.assertEqual(lr["status"], "not_applicable"); self.assertEqual(lr["status_far_unstressed"], "not_applicable")
            self.assertEqual(lr["slip_sezen_bilinear"], 0.0); self.assertEqual(lr["required_length_ratio"], 0.0)
        for r in ref["rows"]:
            tension = [lr for lr in r["layers"].values() if lr["in_tension"]]
            self.assertEqual(r["valid"], all(lr["status"] in ("valid", "not_applicable") for lr in tension))
            if r["layers"][deep]["sigma"] <= 0.0:                               # the compressed layer is not in the rotation fit
                self.assertNotIn(deep, r["face_rotation"]["sezen_bilinear"]["per_layer"])
        self.assertIsNotNone(ref["events"]["first_layer_yield"]); self.assertGreater(ref["events"]["first_layer_yield"]["theta_slip"], 0.0)

    def test_export_keeps_every_layer_quantity_including_the_far_unstressed_alternative(self):
        import csv, tempfile
        sec = ar.frame_section(); ref = ar.reference(sec, ar.Bond.sezen(sec.fc_anchorage, "ksi"), "hogging")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"; ar.export_rows(ref, path)
            with path.open(newline="", encoding="utf-8") as handle:
                exported = list(csv.DictReader(handle))
        self.assertEqual(len(exported), len(ref["rows"]))
        first_bad = next(i for i, r in enumerate(ref["rows"]) if not r["valid_far_unstressed"])       # a reclassified far-unstressed state
        checked = 0
        for i in sorted({0, len(ref["rows"]) // 2, first_bad, len(ref["rows"]) - 1}):
            for lid, lr in ref["rows"][i]["layers"].items():
                if not lr["in_tension"]:
                    continue
                # everything the alternative holds in memory has a column
                self.assertEqual({k for k in lr if k.endswith("_far_unstressed")} - set(ar.LAYER_COLUMNS), set())
                for col in ar.LAYER_COLUMNS:
                    cell, value = exported[i][f"{lid}_{col}"], lr.get(col, "")
                    if isinstance(value, (bool, str)):
                        self.assertEqual(cell, str(value), (i, lid, col))
                    else:
                        self.assertEqual(float(cell), float(value), (i, lid, col))
                    checked += 1
            for col in ("anchorage_ratio_max", "required_length_ratio_max"):
                self.assertEqual(float(exported[i][col]), float(ref["rows"][i][col]))
        self.assertGreater(checked, 0)
        bad = ref["rows"][first_bad]["layers"]; lid = next(k for k, v in bad.items() if v["in_tension"] and not v["valid_far_unstressed"])
        self.assertEqual(exported[first_bad][f"{lid}_status_far_unstressed"], "invalid")
        self.assertGreater(float(exported[first_bad][f"{lid}_required_length_ratio_far_unstressed"]), 1.0)


class FrameSection(unittest.TestCase):
    def test_default_section_reports_events_and_flags_the_anchorage(self):
        sec = ar.frame_section()
        bond = ar.Bond.sezen(sec.fc_anchorage, "ksi")
        for direction in ("hogging", "sagging"):
            ref = ar.reference(sec, bond, direction)
            ev = ref["events"]
            for name in ("first_layer_yield", "all_tension_layers_yielded", "hardening_onset"):
                self.assertIsNotNone(ev[name])
            self.assertGreater(ev["hardening_onset"]["M"], ev["first_layer_yield"]["M"])
            self.assertGreater(ev["aci_nominal_Mn"]["Mn"], 0.0)
            self.assertEqual(ref["rows"][-1]["valid"], ref["rows"][-1]["anchorage_ratio_max"] <= 1.0)
            self.assertEqual(ref["invalid_rows"], sum(1 for r in ref["rows"] if not r["valid"]))
            self.assertEqual(ref["bond"]["basis"].split(":")[0], "Sezen and Setzler 2008")
            self.assertIn("anchorage_model", ar.ASSUMPTIONS)

    def test_symmetric_section_gives_identical_directions(self):
        sec = ar.frame_section()
        bond = ar.Bond.sezen(sec.fc_anchorage, "ksi")
        h, s = (ar.reference(sec, bond, d)["events"]["first_layer_yield"] for d in ("hogging", "sagging"))
        self.assertAlmostEqual(h["M"], s["M"], places=9)
        self.assertAlmostEqual(h["theta_slip"], s["theta_slip"], places=12)


if __name__ == "__main__":
    unittest.main()
