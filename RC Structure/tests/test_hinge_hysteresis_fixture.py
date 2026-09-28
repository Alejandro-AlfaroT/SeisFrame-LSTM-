"""Cyclic yielding fixture for the full-rate hinge measurement path (ground-motion pilot, 2026-09-21).

A small designed frame (2 x 1 bays, 2 stories, with its slab layout so the
beam hinges are asymmetric) is pushed at the roof in +Y past yield, reversed
past yield in -Y, and pushed back, with the pilot's recorders attached the
same way the real-case runner attaches them. The checks are on the
measurement path only: every recorder row is the spring's own conjugate
(moment, rotation) pair at the committed step (equal to eleResponse
basicForce / deformation at that step, time column equal to the analysis
time), the initial slope of every loop is the installed Ke of that hinge,
the positive and negative yield plateaus are the installed Fy of that end
and sign (hogging at end i is the positive plateau, sagging the negative one;
the reverse at end j), yield is identified from the full-rate extrema by
both criteria in both signs, and a reversed loop encloses positive area.
"""
import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openseespy.opensees as ops                                              # noqa: E402
import Structure_Parameters as sp                                              # noqa: E402
from Analysis import Hinge_Hysteresis_Diagnostic as hd                         # noqa: E402
from Model.Build_Model import apply_analysis_constraints, build_model, roof_master_node  # noqa: E402
from Analysis.Gravity import run_gravity_analysis                              # noqa: E402
from Design import Design_Driver as driver                                      # noqa: E402
from Loads.Gravity_Loads import apply_gravity_loads                            # noqa: E402
from Model.IMK_Hinges import hinge_registry                                     # noqa: E402


class CyclicMeasurementFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack = contextlib.ExitStack()
        for name in set(driver._STATE_KEYS) | {"COVER", "NUM_BAY_X", "NUM_BAY_Y", "NUM_FLOOR", "NUM_MODES", "ASCE_SDS"}:
            cls.stack.enter_context(mock.patch.object(sp, name, getattr(sp, name)))
        sp.NUM_BAY_X, sp.NUM_BAY_Y = 2, 1
        sp.NUM_FLOOR = 2
        sp.NUM_MODES = 3 * sp.NUM_FLOOR
        # This fixture measures installed strengths with the diagnostic's exact
        # Bilin tangent-based yield and plastic-rotation accounting; under
        # IMKPeakOriented (the production member material since 2026-09-24)
        # the diagnostic reports virgin-envelope exceedance only, with no
        # accumulated plastic rotation, so the both-sign plateau checks below
        # are not defined for it. The material is pinned here; a
        # PeakOriented cyclic measurement fixture is separate, open work.
        # Likewise the deterioration mode: the plateau checks compare the
        # first-excursion strength with the installed Fy, and the Haselton
        # capacities (Lambda ~ 0.6 rad) already take a few tenths of a percent
        # off a spring that yields only after the reversal. The translation
        # itself is tested in tests/test_imk_deterioration_mapping.py.
        cls.stack.enter_context(mock.patch.object(sp, "IMK_MATERIAL_TYPE", "IMKBilin"))
        cls.stack.enter_context(mock.patch.object(sp, "IMK_DETERIORATION_MODE", "direct"))
        # The PROBE configuration asserts the slab-action evidence, so the design establishes a slab
        # reinforcement layout and the beam hinges carry the composite, per-end asymmetric strengths
        # of a qualified record (hogging with the slab mats, sagging with the flange, lower exterior ends).
        from Design.Verify_Designs import probe_config
        try:
            record = driver.design_structure(cfg=probe_config("2026-09-21"), max_section_iter=1, max_steel_iter=1, verbose=False)
        except BaseException:
            # A failed fixture must not leave the patched geometry behind for
            # every later test module (it did: 2x1x2 leaked into the slab-load,
            # energy-mapping and integration tests of the same discover run).
            cls.stack.close()
            raise
        finally:
            ops.wipe()
        cls.record = json.loads(json.dumps(record, allow_nan=False))
        driver.apply_design(cls.record)
        cls.tmp = tempfile.TemporaryDirectory()
        # Build, gravity (production convention), recorders, cyclic roof push in Y.
        ops.wipe()
        build_model()
        apply_gravity_loads()
        run_gravity_analysis()
        cls.verification = hd.verify_installed_design(cls.record)
        cls.audit, cls.inventory, cls.hinges = hd.run_audit()
        cls.coverage = hd.attach_hinge_recorders(Path(cls.tmp.name) / "recorders", cls.hinges)
        roof = roof_master_node()
        ops.timeSeries("Linear", 31)
        ops.pattern("Plain", 31, 31)
        ops.load(roof, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        ops.wipeAnalysis()
        ops.system("BandGeneral")
        apply_analysis_constraints()
        ops.numberer("RCM")
        ops.test("NormDispIncr", 1e-6, 100)
        ops.algorithm("Newton")
        ops.analysis("Static")
        height = sp.NUM_FLOOR * sp.STORY_H
        cls.samples = []                     # (time, uy, {hinge_tag: (basicForce, deformation)}) at every committed step
        cls.step_count = 0

        def push_to(target, du):
            current = ops.nodeDisp(roof, 2)
            n = max(1, int(round(abs(target - current) / du)))
            ops.integrator("DisplacementControl", roof, 2, (target - current) / n)
            for _ in range(n):
                if ops.analyze(1) != 0:
                    raise RuntimeError("fixture push did not converge")
                cls.step_count += 1
                cls.samples.append((ops.getTime(), ops.nodeDisp(roof, 2),
                                    {int(h["hinge_tag"]): (list(ops.eleResponse(h["hinge_tag"], "basicForce")),
                                                           list(ops.eleResponse(h["hinge_tag"], "deformation")))
                                     for h in cls.hinges}))

        push_to(+0.025 * height, 0.05)       # 2.5% roof drift in +Y
        push_to(-0.025 * height, 0.05)       # through zero to -2.5%
        push_to(+0.015 * height, 0.05)       # back to +1.5%: closes a loop
        hd.close_recorders()
        cls.recorded = hd.read_hinge_recorders(cls.coverage)
        cls.rows = hd.evaluate_hinge_histories(cls.recorded, cls.hinges)
        cls.summary = hd.yield_summary(cls.rows)
        ops.wipe()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        cls.stack.close()

    def test_installed_design_and_registry_verify_against_the_record(self):
        self.assertTrue(self.verification["consistent"], self.verification["differences"])
        families = self.verification["registry"]["beam_families"]
        self.assertTrue(families)
        for fam, data in families.items():
            self.assertGreater(max(data["hogging_kip_in"]), max(data["sagging_kip_in"]))       # slab makes the hinges asymmetric
        self.assertEqual(self.coverage["hinge_count"], len(self.hinges))
        self.assertEqual(self.verification["registry"]["hinged_members"], len(self.inventory["columns"]) + len(self.inventory["beams"]))
        self.assertTrue(bool((self.record.get("slab_reinforcement") or {}).get("layout")))   # composite hinges, as in a qualified record
        self.assertEqual(self.verification["design_evidence_gaps"], [])

    def test_recorder_rows_are_the_springs_conjugate_pair_at_every_committed_step(self):
        for axis in ("y", "z"):
            block = self.recorded[axis]
            self.assertEqual(block["rows"], self.step_count)                                # static: no sub-steps
            np.testing.assert_allclose(block["time"], [s[0] for s in self.samples], rtol=0, atol=1e-9)   # time alignment
        index = {tag: k for k, tag in enumerate(self.recorded["hinge_tag_order"])}
        for row, (_t, _uy, per_hinge) in enumerate(self.samples):
            for tag, (basic, deformation) in per_hinge.items():
                k = index[tag]
                for axis, j in (("y", 0), ("z", 1)):
                    self.assertAlmostEqual(self.recorded[axis]["moment"][row, k], basic[j], delta=1e-8 * max(1.0, abs(basic[j])))
                    self.assertAlmostEqual(self.recorded[axis]["rotation"][row, k], deformation[j], delta=1e-12)

    def test_initial_slope_is_the_installed_ke_of_each_hinge(self):
        index = {tag: k for k, tag in enumerate(self.recorded["hinge_tag_order"])}
        checked = 0
        for h in self.hinges:
            k = index[int(h["hinge_tag"])]
            for axis in ("y", "z"):
                block = self.recorded[axis]
                m, r = block["moment"][:4, k], block["rotation"][:4, k]
                if abs(r[-1]) < 1e-9 or abs(r[-1]) > 0.5 * (h["springs"][axis]["theta_y_positive"] or 1.0):
                    continue                                                                     # unloaded axis, or past the elastic range already
                slope = (m[-1] - m[0]) / (r[-1] - r[0])
                self.assertAlmostEqual(slope, h["springs"][axis]["ke"], delta=1e-6 * h["springs"][axis]["ke"], msg=(h["hinge_tag"], axis))
                checked += 1
        self.assertGreater(checked, 4)

    def test_yield_plateaus_match_the_installed_asymmetric_strengths_per_end_and_sign(self):
        yielded_both = [r for r in self.rows if r["member_type"].startswith("beam") and r["spring_axis"] == "y"
                        and r["yielded_positive"] and r["yielded_negative"]]
        self.assertGreaterEqual(len(yielded_both), 2, self.summary)
        registry = hinge_registry()
        fmax_pos, fmax_neg = sp.IMK_FMAXFY_POS, sp.IMK_FMAXFY_NEG
        for r in yielded_both:
            entry = registry[r["member_tag"]]
            hog = entry[f"yield_moment_y_hogging_{r['end']}_kip_in"]
            sag = entry[f"yield_moment_y_sagging_{r['end']}_kip_in"]
            expected_pos, expected_neg = (hog, sag) if r["end"] == "i" else (sag, hog)
            self.assertAlmostEqual(r["fy_positive_kip_in"], expected_pos, places=6)
            self.assertAlmostEqual(r["fy_negative_kip_in"], expected_neg, places=6)
            self.assertNotAlmostEqual(expected_pos, expected_neg, delta=1.0)                     # asymmetric
            # The plateau reached lies between Fy and Fmax/Fy x Fy of its own sign.
            self.assertGreaterEqual(r["moment_max_kip_in"], (1 - 1e-3) * expected_pos)
            self.assertLessEqual(r["moment_max_kip_in"], fmax_pos * expected_pos * (1 + 1e-6))
            self.assertGreaterEqual(-r["moment_min_kip_in"], (1 - 1e-3) * expected_neg)
            self.assertLessEqual(-r["moment_min_kip_in"], fmax_neg * expected_neg * (1 + 1e-6))
            self.assertFalse(r["criteria_disagree"])
            self.assertTrue(r["virgin_yielded"])
            self.assertGreater(r["plastic_rotation_positive_accumulated"], 0.0)
            self.assertGreater(r["plastic_rotation_negative_accumulated"], 0.0)
            self.assertGreater(r["plastic_rotation_peak_positive"], 0.0)
            self.assertGreater(r["plastic_rotation_peak_negative"], 0.0)

    def test_reversed_loops_enclose_positive_area_and_elastic_springs_do_not(self):
        yielded = [r for r in self.rows if r["yielded"]]
        elastic = [r for r in self.rows if not r["yielded"] and r["plastic_steps"] == 0]
        self.assertTrue(yielded and elastic)
        for r in yielded:
            self.assertGreater(r["dissipated_energy_kip_in"], 0.0, r["hinge_tag"])
        for r in elastic:
            scale = max(1.0, abs(r["moment_max_kip_in"]) * max(abs(r["rotation_max"]), abs(r["rotation_min"])))
            self.assertLess(abs(r["dissipated_energy_kip_in"]), 1e-6 * scale, r["hinge_tag"])
        self.assertGreater(self.summary["total_dissipated_energy_kip_in"], 0.0)
        # The summary counts what the rows say.
        self.assertEqual(self.summary["yielded_springs"], len(yielded))

    def test_plots_are_written_for_the_yielded_springs(self):
        written = hd.plot_hinge_loops(self.recorded, self.hinges, self.rows, Path(self.tmp.name) / "loops", limit=6)
        self.assertTrue(written)
        self.assertTrue(all(Path(p).exists() for p in written))
        path = hd.write_histories_npz(self.recorded, Path(self.tmp.name) / "hist.npz")
        loaded = np.load(path)
        self.assertEqual(loaded["moment_y"].shape, (self.step_count, len(self.hinges)))


    def test_kinematic_reversal_is_identified_per_sign(self):
        """A spring pushed past yield in +, then reversed only far enough to yield on the shifted branch (rotation still
        positive, moment below that sign's Fy): the path-aware call sees the negative-direction yielding, the virgin
        criteria cannot (review finding, 2026-09-21)."""
        ke, fy_pos, fy_neg, theta_p = 4.0e6, 3000.0, 2000.0, 0.05
        ty_pos, ty_neg = fy_pos / ke, fy_neg / ke
        hardening = 0.1 * fy_pos / theta_p
        theta_a = 0.010
        m_a = fy_pos + hardening * (theta_a - ty_pos)
        theta_b = theta_a - (fy_pos + fy_neg) / ke                 # elastic unloading range of a kinematic model
        m_b = m_a - (fy_pos + fy_neg)
        theta_c = 0.003                                             # still on the positive side
        m_c = m_b - hardening * (theta_b - theta_c)
        leg1 = np.linspace(0.0, theta_a, 60)
        rotation = np.concatenate([leg1, np.linspace(theta_a, theta_b, 40)[1:], np.linspace(theta_b, theta_c, 40)[1:]])
        moment = np.concatenate([np.where(leg1 <= ty_pos, ke * leg1, fy_pos + hardening * (leg1 - ty_pos)),
                                 np.linspace(m_a, m_b, 40)[1:], np.linspace(m_b, m_c, 40)[1:]])
        recorded = {"hinge_tag_order": [1], "y": {"time": np.arange(len(rotation), dtype=float), "moment": moment[:, None],
                                                  "rotation": rotation[:, None], "rows": len(rotation)}}
        hinge = {"hinge_tag": 1, "member_tag": 1, "member_type": "beam_x", "end": "i", "joint_node": 1, "beam_family": "x_edge",
                 "theta_p": theta_p, "theta_pc": 0.1, "theta_u": 0.2,
                 "springs": {"y": {"index": 0, "fy_positive_kip_in": fy_pos, "fy_negative_kip_in": fy_neg, "ke": ke,
                                   "theta_y_positive": ty_pos, "theta_y_negative": ty_neg}}}
        row = hd.evaluate_hinge_histories(recorded, [hinge])[0]
        self.assertTrue(row["yielded_positive"] and row["yielded_negative"])                   # both directions, path-aware
        self.assertFalse(row["virgin_yield_negative_by_rotation"] or row["virgin_yield_negative_by_moment"])   # origin-referenced misses it
        self.assertFalse(row["criteria_disagree"])                                             # the spring did yield: both calls agree on that
        # Accumulated plastic rotation excludes the elastic part of the moment change along the hardening branch (dM / Ke).
        self.assertAlmostEqual(row["plastic_rotation_positive_accumulated"], (theta_a - ty_pos) - (m_a - fy_pos) / ke, delta=1e-9)
        self.assertAlmostEqual(row["plastic_rotation_negative_accumulated"], (theta_b - theta_c) - (m_b - m_c) / ke, delta=1e-9)
        self.assertAlmostEqual(row["plastic_rotation_peak_negative"], 0.0)                      # the production one-sided measure
        self.assertGreater(row["dissipated_energy_kip_in"], 0.0)
        rot = np.linspace(0.0, 0.5 * ty_pos, 30)
        rec = {"hinge_tag_order": [1], "y": {"time": np.arange(30.0), "moment": (ke * rot)[:, None], "rotation": rot[:, None], "rows": 30}}
        row = hd.evaluate_hinge_histories(rec, [hinge])[0]
        self.assertFalse(row["yielded"] or row["virgin_yielded"])
        self.assertEqual(row["plastic_steps"], 0)
        self.assertAlmostEqual(row["dissipated_energy_kip_in"], 0.0, delta=1e-12)

    def test_recorder_reader_handles_recovery_substeps_and_failed_attempts(self):
        """Rows outnumber steps when NTHA subdivides a step; a failed attempt commits nothing; a window limit drops trailing rows."""
        import tempfile as _tf
        d = Path(_tf.mkdtemp())
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        ops.node(1, 0, 0, 0)
        ops.node(2, 0, 0, 0)
        ops.fix(1, 1, 1, 1, 1, 1, 1)
        ops.fix(2, 1, 1, 1, 1, 0, 0)
        ke = 1.0e5
        for tag in (11, 12):
            ops.uniaxialMaterial("IMKBilin", tag, ke, 0.02, 0.06, 0.12, 100.0, 1.1, 0.2, 0.02, 0.06, 0.12, 60.0, 1.1, 0.2, 10, 10, 10, 1, 1, 1, 1, 1)
        hinge_tag = sp.IMK_HINGE_ELEMENT_TAG_BASE + 11
        ops.element("zeroLength", hinge_tag, 1, 2, "-mat", 11, 12, "-dir", 5, 6, "-orient", 1, 0, 0, 0, 1, 0)
        ops.mass(2, 0, 0, 0, 0, 1.0e-2, 1.0e-2)
        spring = {"index": 0, "fy_positive_kip_in": 100.0, "fy_negative_kip_in": 60.0, "ke": ke, "theta_y_positive": 1e-3, "theta_y_negative": 6e-4}
        hinge = {"hinge_tag": hinge_tag, "member_tag": 11, "member_type": "beam_x", "end": "i", "joint_node": 2, "beam_family": None,
                 "theta_p": 0.02, "theta_pc": 0.06, "theta_u": 0.12, "springs": {"y": dict(spring), "z": {**spring, "index": 1}}}
        coverage = hd.attach_hinge_recorders(d, [hinge])
        ops.timeSeries("Trig", 1, 0.0, 100.0, 0.5)
        ops.pattern("Plain", 1, 1)
        ops.load(2, 0, 0, 0, 0, 50.0, 0)                        # below both yield strengths: this test is about row bookkeeping
        ops.constraints("Plain")
        ops.numberer("Plain")
        ops.system("BandGeneral")
        ops.test("NormDispIncr", 1e-6, 100)
        ops.algorithm("Newton")
        ops.integrator("Newmark", 0.5, 0.25)
        ops.analysis("Transient")
        dt, committed, samples = 0.01, 0, []
        for step in range(40):
            if step in (10, 25):                                  # a deliberately failing attempt: nothing is committed
                ops.test("NormDispIncr", 1e-30, 1)
                ok = ops.analyze(1, dt)
                ops.test("NormDispIncr", 1e-6, 100)
                self.assertNotEqual(ok, 0)
            if step in (15, 30):                                  # a subdivided recovery: ten commits
                self.assertEqual(ops.analyze(10, dt / 10.0), 0)
                committed += 10
            else:
                self.assertEqual(ops.analyze(1, dt), 0)
                committed += 1
            samples.append((ops.getTime(), list(ops.eleResponse(hinge_tag, "basicForce"))[0], list(ops.eleResponse(hinge_tag, "deformation"))[0]))
        hd.close_recorders()
        rec = hd.read_hinge_recorders(coverage)
        self.assertEqual(rec["y"]["rows"], committed)                                       # 38 + 20
        self.assertTrue(rec["y"]["time_strictly_increasing"])
        for t_s, m_s, r_s in samples:                                                        # aligned by time, not by row index
            k = int(np.argmin(np.abs(rec["y"]["time"] - t_s)))
            self.assertAlmostEqual(rec["y"]["time"][k], t_s, delta=1e-9)
            self.assertAlmostEqual(rec["y"]["moment"][k, 0], m_s, delta=1e-8 * max(1.0, abs(m_s)))
            self.assertAlmostEqual(rec["y"]["rotation"][k, 0], r_s, delta=1e-12)
        limited = hd.read_hinge_recorders(coverage, time_limit=samples[19][0])
        self.assertEqual(limited["y"]["rows"] + limited["y"]["rows_dropped_beyond_window"], committed)
        self.assertLessEqual(limited["y"]["time"][-1], samples[19][0] + 1e-9)
        rows = hd.evaluate_hinge_histories(rec, [hinge])
        self.assertEqual(len(rows), 2)
        ops.wipe()

    def test_installed_design_verification_detects_a_mismatch(self):
        tampered = json.loads(json.dumps(self.record))
        tampered["reinforcement"]["col_top_bars"] += 1
        tampered["beam_slab_families"]["x_edge"]["mn_negative_kip_in"] *= 1.01
        ops.wipe()
        build_model()
        result = hd.verify_installed_design(tampered)
        ops.wipe()
        self.assertFalse(result["consistent"])
        self.assertTrue(any("col_top_bars" in d for d in result["differences"]))
        self.assertTrue(any("x_edge.hogging" in d for d in result["differences"]))
        self.assertTrue(any("beam end strengths per member" in v for v in result["verified"]))


if __name__ == "__main__":
    unittest.main()
