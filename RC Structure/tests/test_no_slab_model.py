"""The floor as load and rigid diaphragm only (NoSlab branch, design decision of 2026-10-21).

Structure_Parameters.SLAB_STRUCTURAL_PARTICIPATION = False makes every beam a rectangular section in strength and in
stiffness, takes the floor load to the beams by the two-way 45-degree tributary rule in the transfer record format,
puts the 'no_slab' basis in the strong-column evidence, prices no slab steel and refuses records of the other floor
model. True restores the composite slab model (tests/slab_model.py runs its tests that way).
"""
import contextlib
import copy
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402
import Structure_Parameters as sp                                               # noqa: E402
from Design import Grouped_Record as gr                                          # noqa: E402
from Design import SMRF_Floor_Transfer as ft                                     # noqa: E402
from Design.SMRF_Beam_Slab_Strength import composite_beam_strengths              # noqa: E402
from Design.SMRF_Demands import live_load_patterns                               # noqa: E402
from Design.SMRF_Joints import scwb_check                                        # noqa: E402

BEAM = {"b_in": 18.0, "h_in": 28.0, "fc_ksi": 5.0, "fy_ksi": 60.0, "bar_size": 8, "top_bars": 4, "bot_bars": 3,
        "centroid_offset_in": 2.5}
GEOMETRY = {"bay_x_in": 360.0, "bay_y_in": 240.0, "h_col_in": 24.0, "b_col_in": 24.0}
LAYOUT = {"layers": {f"{a}_{f}": {"bar_area_in2": 0.20, "spacing_in": 12.0, "effective_depth_in": 6.5 if f == "top" else 1.25}
                     for a in ("x", "y") for f in ("top", "bottom")}}
SLAB = {"thickness_in": 8.0, "concrete_unit_weight_kcf": 0.15, "superimposed_dead_load_ksf": 0.05}
SECTIONS = {"b_beam_in": 16.0, "h_beam_in": 28.0, "fc_beam_ksi": 5.0, "b_col_in": 24.0, "h_col_in": 24.0}


def composite(value):
    return mock.patch.object(sp, "SLAB_STRUCTURAL_PARTICIPATION", value)


class Switch(unittest.TestCase):
    def test_the_branch_default_is_no_participation(self):
        self.assertFalse(sp.SLAB_STRUCTURAL_PARTICIPATION)
        record = sp.floor_model_record()
        self.assertEqual(record["id"], "floor_load_and_rigid_diaphragm_only_v1")
        self.assertIs(record["slab_structural_participation"], False)
        with composite(True):
            self.assertEqual(sp.floor_model_record()["id"], "composite_slab_participating_v1")


class RectangularBeams(unittest.TestCase):
    def test_strength_ignores_the_slab_thickness_layout_and_flange(self):
        family = composite_beam_strengths(BEAM, {"thickness_in": 8.0}, LAYOUT, GEOMETRY, "x", "interior")
        self.assertFalse(family["slab_participation"])
        self.assertEqual(family["effective_flange_width_in"], BEAM["b_in"])
        self.assertEqual(family["slab_steel_in_flange_in2"], 0.0)
        for sign in ("positive", "negative"):
            self.assertAlmostEqual(family[f"mn_{sign}_kip_in"], family["rectangular"][sign]["mn_kip_in"], places=9)
        with composite(True):
            slab = composite_beam_strengths(BEAM, {"thickness_in": 8.0}, LAYOUT, GEOMETRY, "x", "interior")
        self.assertGreater(slab["effective_flange_width_in"], BEAM["b_in"])
        self.assertGreater(slab["mn_negative_kip_in"], 1.2 * family["mn_negative_kip_in"])       # the mats in hogging
        self.assertGreater(slab["mn_positive_kip_in"], family["mn_positive_kip_in"])             # the flange in sagging
        self.assertAlmostEqual(slab["rectangular"]["negative"]["mn_kip_in"], family["mn_negative_kip_in"], places=9)

    def test_stiffness_is_the_rectangular_web(self):
        with gf.frame(slab=7.0):
            rectangular = sp.rect_iy(sp.B_BEAM, sp.H_BEAM)
            for axis in ("x", "y"):
                for position in ("edge", "interior"):
                    section = sp.beam_flexural_section(axis, position)
                    self.assertEqual(section["iy_in4"], rectangular)
                    self.assertIn("no slab participation", section["basis"])
            with composite(True):
                self.assertGreater(sp.beam_flexural_section("x", "interior")["iy_in4"], 1.3 * rectangular)


class TributaryTransfer(unittest.TestCase):
    def check_rule(self, lx, ly):
        geometry = {"num_bay_x": 3, "num_bay_y": 2, "bay_x_in": lx, "bay_y_in": ly}
        transfer = ft.build_floor_transfer(SLAB, geometry, SECTIONS, 0.05, live_patterns=live_load_patterns(3, 2))
        self.assertEqual(transfer["method_version"], ft.METHOD_VERSION_TRIBUTARY)
        area = 3 * 2 * lx * ly / 144.0
        pressure = 0.15 * 8 / 12 + 0.05
        ft.validate_floor_transfer(transfer, geometry, SECTIONS, 8.0, pressure * area, 0.05 * area)
        dead = transfer["unit_cases"]["dead"]
        totals = {(b["axis"], b["line_index"], b["span_index"]): b["total_kip"] for b in dead["beams"]}
        q = pressure / 144.0
        a, b = min(lx, ly), max(lx, ly)
        long_edge, short_edge = q * a * (2 * b - a) / 4.0, q * a * a / 4.0
        x_edge, y_edge = (long_edge, short_edge) if lx >= ly else (short_edge, long_edge)
        # the 45-degree areas start at the grid points; within half a column (12 in < either cap) of a centerline the
        # depth is s, so each span end hands q * 12^2 / 2 to its column and the beam keeps the rest
        end = q * 12.0 ** 2 / 2.0
        self.assertAlmostEqual(totals[("x", 0, 1)], x_edge - 2 * end, places=9)              # perimeter x line, one panel
        self.assertAlmostEqual(totals[("x", 1, 1)], 2 * (x_edge - 2 * end), places=9)        # interior x line, two panels
        self.assertAlmostEqual(totals[("y", 0, 0)], y_edge - 2 * end, places=9)
        self.assertAlmostEqual(totals[("y", 1, 1)], 2 * (y_edge - 2 * end), places=9)
        direct = {(c["grid_i"], c["grid_j"]): c["direct_load_kip"] for c in dead["columns"]}
        self.assertAlmostEqual(direct[(1, 1)], q * 24.0 * 24.0, places=12)       # an interior column: its footprint
        self.assertAlmostEqual(direct[(1, 0)], q * 24.0 * 12.0, places=12)       # an edge column: half of it in the floor
        self.assertAlmostEqual(direct[(0, 0)], q * 12.0 * 12.0, places=12)       # a corner column: a quarter
        self.assertAlmostEqual(dead["beam_kip"] + dead["column_direct_kip"], dead["applied_kip"], places=9)
        self.assertTrue(all(c[1] == 0.0 and c[2] == 0.0 for beam in dead["beams"] for c in beam["node_couples"]))
        for beam in dead["beams"]:                                                  # every point on the clear span
            length = lx if beam["axis"] == "x" else ly
            self.assertTrue(all(12.0 < x * length < length - 12.0 for x, _p in beam["node_loads"]))
        for beam in dead["beams"]:                                                  # symmetric about midspan
            loads = [p for _x, p in beam["node_loads"]]
            self.assertTrue(all(math.isclose(p, q_, rel_tol=1e-12, abs_tol=1e-12) for p, q_ in zip(loads, reversed(loads))))
        return transfer

    def test_the_45_degree_rule_in_both_aspects_and_square_bays(self):
        self.check_rule(360.0, 240.0)
        self.check_rule(240.0, 360.0)
        self.check_rule(300.0, 300.0)

    def test_the_face_reactions_are_the_exact_clear_span_reactions(self):
        from Design.SMRF_Capacity_Design import _span_face_reactions
        for span, other, column in ((240.0, 240.0, 28.0), (240.0, 240.0, 20.0), (360.0, 240.0, 32.0), (240.0, 360.0, 16.0)):
            geometry = {"num_bay_x": 3, "num_bay_y": 2, "bay_x_in": span, "bay_y_in": other}
            sections = {**SECTIONS, "b_col_in": column, "h_col_in": column}
            dead = ft.build_floor_transfer(SLAB, geometry, sections, 0.05)["unit_cases"]["dead"]
            beam = next(b for b in dead["beams"] if (b["axis"], b["line_index"], b["span_index"]) == ("x", 0, 1))
            q, face, clear, cap = (0.15 * 8 / 12 + 0.05) / 144.0, column / 2.0, span - column, other / 2.0
            steps = 20000
            width = clear / steps
            exact = sum(q * min(s, span - s, cap) * (1.0 - (s - face) / clear) * width
                        for s in (face + (k + 0.5) * width for k in range(steps)))
            left, right = _span_face_reactions(beam, span, column, 1.0)
            with self.subTest(span=span, column=column):
                self.assertAlmostEqual(left / exact, 1.0, places=6)
                self.assertAlmostEqual(left, right, places=9)                  # one panel, symmetric span

    def test_unequal_columns_balance_and_keep_the_points_off_the_columns(self):
        from Design import SMRF_Floor_Sections as fs
        from Model import Member_Groups as mg
        import test_grouped_floor as tgf
        with gf.frame():
            mg.install(gf.changed_state())
            description = fs.by_floor()
        container = ft.build_floor_transfers_by_floor(tgf.SLAB, tgf.GEOMETRY, description, sp.FLOOR_LIVE_LOAD_KSF)
        self.assertEqual(container["method_version"], ft.METHOD_VERSION_TRIBUTARY)
        self.assertIsNone(container["mesh_per_bay"])
        dead_kip = (0.15 * gf.SLAB / 12.0 + 0.05) * tgf.AREA_SQFT
        live_kip = sp.FLOOR_LIVE_LOAD_KSF * tgf.AREA_SQFT
        args = (tgf.GEOMETRY, description, gf.SLAB, dead_kip, live_kip)
        transfers = ft.validate_floor_transfers_by_floor(container, *args)
        for floor_id, transfer in transfers.items():
            sections = transfer["sections"]
            for beam in transfer["unit_cases"]["dead"]["beams"]:
                axis, line, span = beam["axis"], beam["line_index"], beam["span_index"]
                face_i, face_j = ft._tributary_faces(sections, axis, line, span)
                length = tgf.GEOMETRY["bay_x_in"] if axis == "x" else tgf.GEOMETRY["bay_y_in"]
                with self.subTest(floor=floor_id, beam=(axis, line, span)):
                    self.assertTrue(all(face_i < x * length < length - face_j for x, _p in beam["node_loads"]))
        with composite(True), self.assertRaisesRegex(ValueError, "stale or unknown"):
            ft.validate_floor_transfers_by_floor(container, *args)
        plate = {**container, "method_version": ft.METHOD_VERSION_FLEXIBLE}
        with self.assertRaisesRegex(ValueError, "not the live floor model"):
            ft.validate_floor_transfers_by_floor(plate, *args)

    def test_a_pattern_case_loads_only_its_panels(self):
        geometry = {"num_bay_x": 3, "num_bay_y": 2, "bay_x_in": 360.0, "bay_y_in": 240.0}
        patterns = live_load_patterns(3, 2)
        transfer = ft.build_floor_transfer(SLAB, geometry, SECTIONS, 0.05, live_patterns=patterns)
        for pattern in patterns:
            case = transfer["unit_cases"][f"live_pattern_{pattern['id']}"]
            self.assertAlmostEqual(case["applied_kip"], 0.05 / 144.0 * 360.0 * 240.0 * len(pattern["panels"]), places=9)

    def test_the_validators_accept_only_the_live_floor_models_method(self):
        geometry = {"num_bay_x": 3, "num_bay_y": 2, "bay_x_in": 360.0, "bay_y_in": 240.0}
        transfer = ft.build_floor_transfer(SLAB, geometry, SECTIONS, 0.05)
        area = 3 * 2 * 360.0 * 240.0 / 144.0
        args = (geometry, SECTIONS, 8.0, (0.15 * 8 / 12 + 0.05) * area, 0.05 * area)
        ft.validate_floor_transfer(transfer, *args)
        with composite(True), self.assertRaisesRegex(ValueError, r"^Floor transfer method or load model is stale or unknown\.$"):
            ft.validate_floor_transfer(transfer, *args)                  # the Verification1 text under the composite model
        shell = copy.deepcopy(transfer)
        shell["method_version"] = ft.METHOD_VERSION_FLEXIBLE
        with self.assertRaisesRegex(ValueError, "not the live floor model"):
            ft.validate_floor_transfer(shell, *args)


class StrongColumnBasis(unittest.TestCase):
    def test_no_slab_basis_is_evaluated_not_left_open(self):
        column = {"mn_kip_in": 5000.0, "factored_axial_kip": 10.0, "axial_envelope_checked": True}
        state = {"nominal_strengths": True, "column_capacities": [column, dict(column)],
                 "beam_capacities": [{"mn_kip_in": 3000.0, "slab_basis": "no_slab", "slab_mn_kip_in": 0.0}]}
        check = scwb_check(state)
        self.assertIn(check["status"], ("pass", "fail"))
        self.assertAlmostEqual(check["demand"], 1.2 * 3000.0)


class Records(unittest.TestCase):
    def test_a_record_of_the_composite_model_is_refused(self):
        with gf.frame():
            state = gf.uniform_state()
            record = {"schema_version": gr.SCHEMA, "design_mode": "grouped", "geometry": gr.driver._slab_geometry(),
                      "member_groups": state.to_record(), "search": {},
                      "materials": {"fy_ksi": sp.FY_KSI, "es_ksi": sp.ES_KSI, "fyt_ksi": sp.FYT_KSI, "aggregate_size_in": 0.75,
                                    "beam_clear_cover_in": 1.5, "col_clear_cover_in": 1.5,
                                    "reinforcement_specification": sp.REINFORCEMENT_SPECIFICATION_BY_GRADE[sp.FY_KSI]}}
            reached = ValueError("reached the slab validation")
            for floor_model in (None, {"slab_structural_participation": True}):
                altered = dict(record) if floor_model is None else {**record, "floor_model": floor_model}
                with self.subTest(floor_model=floor_model), mock.patch.object(gr, "_validated_slab", side_effect=reached), \
                        self.assertRaisesRegex(ValueError, "another floor model"):
                    gr.apply_grouped_design(altered)
            with mock.patch.object(gr, "_validated_slab", side_effect=reached), \
                    self.assertRaisesRegex(ValueError, "reached the slab validation"):
                gr.apply_grouped_design({**record, "floor_model": sp.floor_model_record()})


class UniformRoute(unittest.TestCase):
    """The uniform design route takes the floor model from the record, as the grouped route does."""

    def setUp(self):
        from Design import Design_Driver as driver
        self.driver = driver
        self.stack = contextlib.ExitStack()
        for name in set(driver._STATE_KEYS) | {"COVER", "NUM_BAY_X", "NUM_BAY_Y", "NUM_FLOOR", "NUM_MODES", "ASCE_SDS"}:
            self.stack.enter_context(mock.patch.object(sp, name, getattr(sp, name)))

    def tearDown(self):
        self.stack.close()

    @staticmethod
    def artifact(floor_model):
        from test_smrf_integration import artifact
        record = artifact()
        record.pop("floor_model", None)
        if floor_model is not None:
            record["floor_model"] = floor_model
        return record

    def test_qualification_declares_the_scope_and_drops_the_slab_items(self):
        from Design.SMRF_Qualification import qualify_design

        def statuses(record):
            return {c["id"]: c["status"] for c in qualify_design(record)["checks"]}
        no_slab = statuses(self.artifact(sp.floor_model_record()))
        with composite(True):
            slab = statuses(self.artifact(sp.floor_model_record()))
        self.assertEqual(no_slab["floor.no_slab_participation_scope"], "pass")
        self.assertEqual(no_slab["qualification.slab_contribution"], "pass")
        self.assertEqual(slab["qualification.slab_contribution"], "not_evaluated")
        self.assertNotIn("floor.no_slab_participation_scope", slab)
        for name in ("slab_verified_strip_action_inputs", "floor.qualified_slab_actions", "diaphragm.scope"):
            self.assertIn(name, slab)
            self.assertNotIn(name, no_slab)
        # a record without the block is the composite model's
        self.assertEqual(statuses(self.artifact(None)).keys(), slab.keys())

    def test_the_scope_check_fails_when_a_slab_layout_rides_along(self):
        from Design.SMRF_Qualification import qualify_design
        record = self.artifact(sp.floor_model_record())
        record["slab_reinforcement"] = {"layout": {"layers": {}}}
        check = next(c for c in qualify_design(record)["checks"] if c["id"] == "floor.no_slab_participation_scope")
        self.assertEqual(check["status"], "fail")

    def test_apply_refuses_the_other_floor_model_before_any_assignment(self):
        before = sp.B_COL
        with composite(True):
            composite_model = sp.floor_model_record()
        for record in (self.artifact(composite_model), self.artifact(None)):   # a record without the block is composite
            record["sections"]["b_col_in"] = before + 4.0
            with self.subTest(floor_model=record.get("floor_model")), \
                    self.assertRaisesRegex(ValueError, "another floor model"):
                self.driver.apply_design(record)
            self.assertEqual(sp.B_COL, before)
        self.driver.apply_design(self.artifact(sp.floor_model_record()))
        no_slab = self.artifact(sp.floor_model_record())
        with composite(True), self.assertRaisesRegex(ValueError, "another floor model"):
            self.driver.apply_design(no_slab)

    def test_a_no_slab_install_drops_any_slab_layout_and_credits_no_hogging_steel(self):
        from Design.ACI_Checks import hogging_slab_credit_from_state
        from Design.Config import DesignConfig
        record = self.artifact(sp.floor_model_record())
        record["slab_reinforcement"] = {"layout": LAYOUT}
        self.driver.apply_design(record)
        self.assertIsNone(sp.SLAB_REINFORCEMENT)
        sp.SLAB_REINFORCEMENT = {"layout": LAYOUT}                       # even when a layout is put there by hand
        cfg = DesignConfig.from_structure_parameters()
        cfg.rebar.beam_slab_steel_in_hogging = True
        self.assertIsNone(hogging_slab_credit_from_state(cfg))

    def test_evidence_designs_no_slab_reinforcement_and_solves_no_plate(self):
        from Design import SMRF_Design_Evidence as evidence
        from Design import SMRF_Beam_Slab_Strength as strength
        from Design.Config import DesignConfig
        stubs = {"slab_strength_inputs": mock.Mock(return_value={"slab": "inputs"}),
                 "design_slab_reinforcement": mock.Mock(return_value={"layout": "designed"}),
                 "joint_evidence": mock.Mock(return_value={}),
                 "floor_diagnostics": mock.Mock(return_value={"status": "solved"})}
        with contextlib.ExitStack() as stack:
            for name, stub in stubs.items():
                stack.enter_context(mock.patch.object(evidence, name, stub))
            stack.enter_context(mock.patch.object(strength, "beam_slab_strengths", return_value=({}, {})))
            record = evidence.attach_design_evidence({"floor_model": sp.floor_model_record()}, DesignConfig())
            self.assertIsNone(record.get("slab_reinforcement"))
            self.assertEqual(record["floor_analysis"]["status"], "not_applicable_no_slab_participation")
            stubs["design_slab_reinforcement"].assert_not_called()
            stubs["floor_diagnostics"].assert_not_called()
            record = evidence.attach_design_evidence({}, DesignConfig())       # composite: the record has no block
            self.assertEqual(record["slab_reinforcement"], {"layout": "designed"})
            self.assertEqual(record["floor_analysis"]["status"], "solved")


class UniformDesignedRecord(unittest.TestCase):
    """A uniform design made end to end under no-slab (the pushover fixture's 2 x 1 bay, two-story frame)."""

    @classmethod
    def setUpClass(cls):
        import json
        import openseespy.opensees as ops
        from Design import Design_Driver as driver
        cls.driver = driver
        cls.stack = contextlib.ExitStack()
        for name in set(driver._STATE_KEYS) | {"COVER", "NUM_BAY_X", "NUM_BAY_Y", "NUM_FLOOR", "NUM_MODES", "ASCE_SDS"}:
            cls.stack.enter_context(mock.patch.object(sp, name, getattr(sp, name)))
        sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR = 2, 1, 2
        sp.NUM_MODES = 3 * sp.NUM_FLOOR
        try:
            record = driver.design_structure(max_section_iter=1, max_steel_iter=1, verbose=False)
            cls.record = json.loads(json.dumps(record, allow_nan=False))
        except BaseException:
            cls.stack.close()                    # tearDownClass does not run after a failed setUpClass
            raise
        finally:
            ops.wipe()

    @classmethod
    def tearDownClass(cls):
        cls.stack.close()

    def test_the_record_is_the_floor_load_and_diaphragm_model(self):
        record = self.record
        self.assertEqual(record["floor_model"]["id"], "floor_load_and_rigid_diaphragm_only_v1")
        self.assertEqual(record["floor_transfer"]["method_version"], ft.METHOD_VERSION_TRIBUTARY)
        self.assertIsNone(record.get("slab_reinforcement"))
        self.assertEqual(record["floor_analysis"]["status"], "not_applicable_no_slab_participation")
        self.assertEqual({v["slab_basis"] for v in record["beam_slab_strengths"].values()}, {"no_slab"})
        for name, family in record["beam_slab_families"].items():
            with self.subTest(family=name):
                self.assertIs(family["slab_participation"], False)
                self.assertEqual(family["slab_steel_in_flange_in2"], 0.0)
                self.assertEqual(family["flange_overhang_in"], 0.0)

    def test_qualification_evaluates_the_strong_column_rule_and_declares_the_scope(self):
        from Design.SMRF_Qualification import qualify_design
        statuses = {}
        for check in qualify_design(self.record)["checks"]:
            statuses.setdefault(check["id"], set()).add(check["status"])
        for name in ("floor.no_slab_participation_scope", "qualification.slab_contribution", "floor.transfer_load_integrity"):
            self.assertEqual(statuses.get(name), {"pass"}, name)
        self.assertTrue(statuses.get("scwb") and "not_evaluated" not in statuses["scwb"])
        for name in ("slab_verified_strip_action_inputs", "floor.qualified_slab_actions", "diaphragm.scope"):
            self.assertNotIn(name, statuses)
        self.assertFalse([name for name in statuses if "coupled" in name])

    def test_no_text_of_the_record_or_its_qualification_describes_the_composite_slab(self):
        from Design.SMRF_Qualification import qualify_design
        phrases = ("slab bars in flange", "developed slab", "T-section", "flange-concrete", "slab mats in",
                   "bars, slab layout", "bars and slab layout", "the common slab layout", "beam-plus-slab")

        def texts(value, path=""):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield from texts(item, f"{path}/{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    yield from texts(item, f"{path}[{index}]")
            elif isinstance(value, str):
                yield path, value
        found = [(where, phrase) for source in (self.record, qualify_design(copy.deepcopy(self.record)))
                 for where, text in texts(source) for phrase in phrases if phrase in text]
        self.assertEqual(found, [])

    def test_the_record_installs_here_and_is_refused_by_the_composite_model(self):
        before = sp.B_COL
        with composite(True), self.assertRaisesRegex(ValueError, "another floor model"):
            self.driver.apply_design(copy.deepcopy(self.record))
        self.assertEqual(sp.B_COL, before)
        self.driver.apply_design(copy.deepcopy(self.record))
        self.assertIsNone(sp.SLAB_REINFORCEMENT)


class CompositeParity(unittest.TestCase):
    """With the switch True, nothing the NoSlab work added shows in a record: the outputs stay as on Verification1."""

    def test_the_new_keys_appear_only_without_slab_participation(self):
        from Design import Design_Driver as driver
        family = composite_beam_strengths(BEAM, {"thickness_in": 8.0}, LAYOUT, GEOMETRY, "x", "interior")
        self.assertIs(family["slab_participation"], False)
        self.assertEqual(sp.floor_model_entry()["floor_model"]["slab_structural_participation"], False)
        self.assertIs(sp.floor_load_metadata()["slab_structural_participation"], False)
        self.assertIs(driver.design_request_identity()["inputs"]["SLAB_STRUCTURAL_PARTICIPATION"], False)
        with composite(True):
            family = composite_beam_strengths(BEAM, {"thickness_in": 8.0}, LAYOUT, GEOMETRY, "x", "interior")
            self.assertNotIn("slab_participation", family)
            self.assertEqual(sp.floor_model_entry(), {})
            self.assertNotIn("slab_structural_participation", sp.floor_load_metadata())
            self.assertNotIn("SLAB_STRUCTURAL_PARTICIPATION", driver.design_request_identity()["inputs"])
            self.assertTrue(family["basis"].startswith("ACI 318-19 6.3.2.1 effective flange"))

    def test_the_grouped_strength_label_follows_the_switch(self):
        from Design import Group_Checks
        self.assertEqual(Group_Checks.beam_strength_basis(), "rectangular_no_slab")
        with composite(True):
            self.assertEqual(Group_Checks.beam_strength_basis(), Group_Checks.BEAM_STRENGTH_BASIS)


if __name__ == "__main__":
    unittest.main()
