import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


RC_STRUCTURE_DIR = Path(__file__).resolve().parents[1]
if str(RC_STRUCTURE_DIR) not in sys.path:
    sys.path.insert(0, str(RC_STRUCTURE_DIR))

import Structure_Parameters as sp
from Analysis.NTHA import _analysis_step_schedule
from Data_Generation import Generate_Hybrid_Dataset
from Data_Generation import Generate_Parameterized_Dataset
from Data_Generation import Hybrid_Exporter
from Data_Generation.Graph_Exporter import collect_global_parameters
from Ground_Motion_Main import (
    analysis_run_name,
    validate_ntha_output_compatibility,
)
from Loads import Ground_Motion
from Loads.Ground_Motion import GroundMotionRecord
from Model import IMK_Hinges


class ColumnCapacityRegressionTests(unittest.TestCase):
    """Guards for the column flexural capacity used by the design loop.

    The design side used to size columns with a singly-reinforced BEAM
    formula -- max(top, bottom) bars and zero axial load -- while the model
    that got analysed took hinge capacity off the nominal P-M surface. That
    understated column Mn by roughly 2x to 4x, so strong-column/weak-beam
    demanded columns several times larger than ACI 318-19 18.7.3.2 requires
    and reported failures for frames that comply.
    """

    def setUp(self):
        self._saved = {
            name: getattr(sp, name)
            for name in (
                "B_COL", "H_COL", "FC_COL_KSI", "COVER", "COL_BAR_AREA",
                "COL_TOP_BARS", "COL_BOT_BARS", "COL_SIDE_BARS",
            )
        }
        sp.B_COL = sp.H_COL = 18.0
        sp.FC_COL_KSI = 5.0
        sp.COVER = 1.5
        sp.COL_BAR_AREA = 0.44
        sp.COL_TOP_BARS = sp.COL_BOT_BARS = sp.COL_SIDE_BARS = 2

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(sp, name, value)

    def test_pm_diagram_counts_side_bars_on_both_faces(self):
        # Model/Sections.py lays COL_SIDE_BARS down EACH side face, so the
        # section holds 2 x that count. Counting them once left the P-M
        # surface short a quarter of the longitudinal steel.
        from RC_Design_Check import _col_steel_layers

        expected = (
            sp.COL_TOP_BARS + sp.COL_BOT_BARS + 2 * sp.COL_SIDE_BARS
        ) * sp.COL_BAR_AREA
        self.assertAlmostEqual(
            sum(area for area, _ in _col_steel_layers()), expected, places=6
        )

    def test_layer_builder_accepts_a_candidate_depth(self):
        from RC_Design_Check import _col_steel_layers

        layers = _col_steel_layers(h=30.0)
        self.assertAlmostEqual(max(depth for _, depth in layers), 30.0 - sp.COVER)
        # The globals must be untouched: the ladder prices candidates without
        # disturbing the section currently installed in the model.
        self.assertEqual(sp.H_COL, 18.0)

    def test_pm_capacity_exceeds_the_beam_formula(self):
        from Design.Design_Driver import _column_nominal_moment

        beam_formula = sp.column_nominal_moment_y()
        pm_based = _column_nominal_moment()
        self.assertGreater(pm_based, beam_formula)
        # Measured ~2x at near-zero axial for this section; the guard is loose
        # enough to survive detailing changes but would catch a regression to
        # the beam formula.
        self.assertGreater(pm_based / beam_formula, 1.5)

    def test_history_ratio_and_flag_share_one_capacity_basis(self):
        # The recorded ratio and the pass/fail flag must be computed from the
        # same column capacity. They were not, so a history row could show the
        # old beam-formula ratio beside the new P-M pass/fail decision.
        import inspect
        from Design import Design_Driver

        source = inspect.getsource(Design_Driver.design_structure)
        head, _, tail = source.partition('"scwb_ratio"')
        self.assertTrue(tail, "scwb_ratio disappeared from the history entry")
        expression = tail[:220]
        self.assertIn("_column_nominal_moment()", expression)
        self.assertNotIn("column_nominal_moment_y()", expression)

    def test_scwb_search_reports_when_the_ladder_is_exhausted(self):
        from Design.Design_Driver import _smallest_scwb_column_index

        tiny = [(10.0, 10.0, 3.0)]
        index, satisfied = _smallest_scwb_column_index(tiny)
        self.assertEqual(index, 0)
        self.assertFalse(satisfied)

        generous = [(10.0, 10.0, 3.0), (60.0, 60.0, 8.0)]
        index, satisfied = _smallest_scwb_column_index(generous)
        self.assertTrue(satisfied)
        self.assertEqual(index, 1)


class HingeYieldRotationTests(unittest.TestCase):
    """Plastic rotation is measured from the spring's own elastic limit.

    The recorded hinge rotation is the zeroLength SPRING's rotation, but the
    registry's theta_y_target is a member-level nominal held fixed at 0.004
    (columns) / 0.005 (beams). Because the spring is deliberately
    IMK_HINGE_STIFFNESS_FACTOR times the member stiffness it yields at My/Ke,
    which across the pilot is 6.2e-5 to 1.55e-4 -- 26x to 65x smaller.
    Subtracting the nominal understated plastic rotation and damage_ratio,
    and damage_ratio is the binding criterion in the inelasticity gate.
    """

    def test_registry_records_the_spring_yield_rotation(self):
        import inspect
        source = inspect.getsource(IMK_Hinges.create_imk_member)
        self.assertIn("theta_y_spring_y", source)
        self.assertIn("theta_y_spring_z", source)

    def test_spring_limit_is_far_below_the_member_nominal(self):
        # Guards the premise: if these ever coincide the distinction is moot
        # and something about the stiffness model has changed.
        self.assertGreater(getattr(sp, "IMK_HINGE_STIFFNESS_FACTOR", 1.0), 1.0)
        self.assertGreater(sp.IMK_BEAM_THETA_Y, 0.0)
        self.assertGreater(sp.IMK_COLUMN_THETA_Y, 0.0)

    def test_backbone_rows_subtract_the_spring_limit(self):
        import inspect
        from Ground_Motion_Main import _hinge_backbone_rows
        source = inspect.getsource(_hinge_backbone_rows)
        self.assertIn("spring_theta_y", source)
        self.assertIn("rotation - spring_theta_y", source)
        # The member nominal must still be reported -- it is what the
        # Haselton backbone was calibrated against.
        self.assertIn('"theta_y": theta_y', source)

    def test_yielded_flag_uses_the_spring_limit(self):
        import inspect
        from Ground_Motion_Main import _hinge_backbone_rows
        source = inspect.getsource(_hinge_backbone_rows)
        self.assertIn("rotation > spring_theta_y", source)
        self.assertNotIn("rotation > theta_y else", source)


class TargetDriftBandTests(unittest.TestCase):
    """The band shares encode a deliberate bias toward nonlinear response.

    This dataset exists to train a surrogate on the regime that is hard to
    analyse, so it over-samples damage relative to real seismic hazard. The
    shares are set from measurement: across 88 pilot runs the is_inelastic
    gate passed 0% below 1.25% drift and 17 of 17 above 2.0%, because the gate
    is dominated by hinge damage ratio, which crosses its threshold near 2%.
    """

    def test_shares_sum_to_one(self):
        from Data_Generation.Calibrate_Intensity import TARGET_DRIFT_BANDS
        self.assertAlmostEqual(sum(b[3] for b in TARGET_DRIFT_BANDS), 1.0, places=6)

    def test_bands_are_contiguous_and_ordered(self):
        from Data_Generation.Calibrate_Intensity import TARGET_DRIFT_BANDS
        for (_n1, _lo1, hi1, _s1), (_n2, lo2, _hi2, _s2) in zip(
            TARGET_DRIFT_BANDS, TARGET_DRIFT_BANDS[1:]
        ):
            self.assertAlmostEqual(hi1, lo2, places=6)

    def test_majority_of_targets_reach_the_gate_threshold(self):
        # 2% drift is where the inelasticity gate empirically passes; the
        # dataset's whole purpose is that most samples clear it.
        from Data_Generation.Calibrate_Intensity import target_drift_sequence
        targets = target_drift_sequence(2000, seed=0)
        share = sum(1 for t in targets if t >= 0.020) / len(targets)
        self.assertGreater(share, 0.55)

    def test_elastic_coverage_is_retained(self):
        # Skewing hard is the point, but the surrogate still needs to see the
        # linear regime or it cannot learn the transition into yielding.
        from Data_Generation.Calibrate_Intensity import target_drift_sequence
        targets = target_drift_sequence(2000, seed=0)
        share = sum(1 for t in targets if t < 0.010) / len(targets)
        self.assertGreater(share, 0.10)

    def test_no_target_exceeds_what_the_scale_cap_can_chase(self):
        from Data_Generation.Calibrate_Intensity import (
            SCALE_FACTOR_MAX, TARGET_DRIFT_BANDS,
        )
        self.assertLessEqual(SCALE_FACTOR_MAX, 3.0)
        self.assertLessEqual(max(b[2] for b in TARGET_DRIFT_BANDS), 0.10)


class GravityEscalationTests(unittest.TestCase):
    """Column ladder helpers and the distinction between solver failure and strength failure."""

    def test_escalation_skips_concrete_strength_variants(self):
        # The ladder interleaves f'c within a size, and f'c is the wrong lever
        # for a stiffness problem: E*I goes as sqrt(f'c) but h^4.
        from Design.Design_Driver import _next_larger_column_index

        ladder = [
            (18.0, 18.0, 4.0), (18.0, 18.0, 5.0), (18.0, 18.0, 6.0),
            (20.0, 20.0, 4.0), (20.0, 20.0, 5.0),
            (22.0, 22.0, 4.0),
        ]
        self.assertEqual(_next_larger_column_index(ladder, 0), 3)
        self.assertEqual(_next_larger_column_index(ladder, 1), 3)
        self.assertEqual(_next_larger_column_index(ladder, 2), 3)
        self.assertEqual(_next_larger_column_index(ladder, 3), 5)

    def test_no_larger_section_reports_none(self):
        from Design.Design_Driver import _next_larger_column_index

        ladder = [(18.0, 18.0, 4.0), (18.0, 18.0, 8.0)]
        self.assertIsNone(_next_larger_column_index(ladder, 0))
        self.assertIsNone(_next_larger_column_index(ladder, 1))

    def test_escalation_reaches_the_real_ladder_top(self):
        from Design.Design_Driver import _next_larger_column_index
        from Design.Section_Design import column_ladder, nearest_rung_index

        ladder = column_ladder()
        index = nearest_rung_index(ladder, 18.0, 18.0, 5.0)
        sizes = [ladder[index][0]]
        while (index := _next_larger_column_index(ladder, index)) is not None:
            sizes.append(ladder[index][0])
        self.assertEqual(sizes[0], 18.0)
        self.assertEqual(sizes[-1], max(s for s, _, _ in ladder))
        # Strictly increasing, one entry per distinct size above the start.
        self.assertEqual(sizes, sorted(set(sizes)))

    def test_gravity_nonconvergence_is_unresolved_and_does_not_grow_the_column(self):
        from Design import Design_Driver as driver
        baseline = driver._capture_state()
        try:
            search = {"first_fitted_beam_index": 0, "beam_rung_steps": [], "steps": 0}
            with mock.patch.object(driver, "_slab_search", return_value=(0, {"thickness_in": 5.0}, [], search)), \
                 mock.patch.object(driver, "_model_period", side_effect=RuntimeError("Gravity analysis failed: injected")), \
                 mock.patch.object(driver, "_next_larger_column_index") as grow, \
                 mock.patch.object(driver, "_steel_pass") as steel:
                with self.assertRaisesRegex(RuntimeError, "analysis unresolved.*No strength failure inferred"):
                    driver.design_structure(max_section_iter=2, verbose=False)
                grow.assert_not_called()
                steel.assert_not_called()
        finally:
            driver._restore_state(baseline)


class DesignConfigSyncTests(unittest.TestCase):
    """cfg must track sp whenever the ladder changes a section.

    The ACI checks read section dimensions and f'c from cfg while the ladder
    writes them to sp. cfg was built once at entry and never refreshed, so
    after an escalation the checks compared the new larger model against the
    capacity of the starting section -- demand growing, capacity frozen. Column
    DCR then rose with section size instead of falling (case_0013: 1.104 at
    26x26 to 2.005 at 36x36), and no case needing an escalation could converge.
    """

    def _saved(self):
        return {k: getattr(sp, k) for k in (
            "B_COL", "H_COL", "FC_COL_KSI", "B_BEAM", "H_BEAM", "FC_BEAM_KSI")}

    def setUp(self):
        self._restore = self._saved()

    def tearDown(self):
        for k, v in self._restore.items():
            setattr(sp, k, v)

    def test_sync_copies_every_section_field(self):
        from Design.Config import DesignConfig
        from Design.Design_Driver import _sync_cfg_to_sp

        sp.B_COL, sp.H_COL, sp.FC_COL_KSI = 18.0, 18.0, 5.0
        sp.B_BEAM, sp.H_BEAM, sp.FC_BEAM_KSI = 10.0, 18.0, 4.0
        cfg = DesignConfig.from_structure_parameters()

        # Ladder escalates: sp moves, cfg must follow.
        sp.B_COL, sp.H_COL, sp.FC_COL_KSI = 36.0, 36.0, 8.0
        sp.B_BEAM, sp.H_BEAM, sp.FC_BEAM_KSI = 12.0, 22.0, 6.0
        self.assertEqual(cfg.sections.b_col_in, 18.0)   # stale before sync
        _sync_cfg_to_sp(cfg)

        self.assertEqual(cfg.sections.b_col_in, 36.0)
        self.assertEqual(cfg.sections.h_col_in, 36.0)
        self.assertEqual(cfg.sections.b_beam_in, 12.0)
        self.assertEqual(cfg.sections.h_beam_in, 22.0)
        self.assertEqual(cfg.materials.fc_col_ksi, 8.0)
        self.assertEqual(cfg.materials.fc_beam_ksi, 6.0)

    def test_sync_tolerates_no_config(self):
        from Design.Design_Driver import _sync_cfg_to_sp
        _sync_cfg_to_sp(None)

    def test_capacity_rises_with_section_after_sync(self):
        # The observable consequence: a bigger column must report more P-M
        # capacity. With a stale cfg the diagram never changed.
        from Design.ACI_Checks import build_pm_diagram
        from Design.Config import DesignConfig
        from Design.Design_Driver import _sync_cfg_to_sp

        sp.B_COL, sp.H_COL, sp.FC_COL_KSI = 18.0, 18.0, 5.0
        cfg = DesignConfig.from_structure_parameters()
        small = max(m for _p, m in build_pm_diagram(cfg))

        sp.B_COL, sp.H_COL = 36.0, 36.0
        stale = max(m for _p, m in build_pm_diagram(cfg))
        self.assertAlmostEqual(stale, small, places=6)  # the bug

        _sync_cfg_to_sp(cfg)
        synced = max(m for _p, m in build_pm_diagram(cfg))
        self.assertGreater(synced, small * 2.0)

    def test_every_rung_application_is_followed_by_a_sync(self):
        import inspect
        from Design import Design_Driver

        source = inspect.getsource(Design_Driver.design_structure)
        self.assertIn("_sync_cfg_to_sp(cfg)", source)
        # The sync must come after the rungs are applied, before any checking.
        rung_at = source.index('_apply_rung(beams[beam_index], "beam")')
        sync_at = source.index("_sync_cfg_to_sp(cfg)")
        self.assertGreater(sync_at, rung_at)


class RecordTargetMatchingTests(unittest.TestCase):
    """Records are chosen to suit each case's target drift, not round robin.

    Assignment used to happen before the target was known, so a case aiming at
    4% drift could draw a weak motion needing 6x scaling while a strong record
    sat unused on an elastic target. Over a 500-case plan matching halves the
    median scale factor and removes ceiling clipping.
    """

    def _calibration(self):
        from Data_Generation.Calibrate_Intensity import SPECTRUM_PERIODS_SEC

        n = len(SPECTRUM_PERIODS_SEC)
        # Two pairs an order of magnitude apart in spectral demand.
        return {
            "coefficients": {"a": -3.58, "b": 0.89, "c": 1.24},
            "period_ratio": 1.88,
            "record_spectra_g": {
                "WEAK_X": [0.02] * n, "WEAK_Y": [0.02] * n,
                "STRONG_X": [0.60] * n, "STRONG_Y": [0.60] * n,
            },
        }

    def _pairs(self):
        return {1: ("WEAK_X", "WEAK_Y"), 2: ("STRONG_X", "STRONG_Y")}

    def test_a_severe_target_takes_the_stronger_record(self):
        chosen = Generate_Parameterized_Dataset._matched_case_records(
            self._calibration(),
            {"num_floor": 6, "story_height_in": 144.0},
            self._pairs(),
            [1, 2],
            [0.040],          # near-collapse target
            [1],              # round robin would have given the weak pair
            case_index=1, seed=0,
        )
        self.assertEqual(chosen, [2])

    def test_an_elastic_target_takes_the_weaker_record(self):
        chosen = Generate_Parameterized_Dataset._matched_case_records(
            self._calibration(),
            {"num_floor": 6, "story_height_in": 144.0},
            self._pairs(),
            [1, 2],
            [0.003],          # elastic target
            [2],
            case_index=1, seed=0,
        )
        self.assertEqual(chosen, [1])

    def test_unpriceable_records_fall_back_to_round_robin(self):
        # No spectra for this pair, so it cannot be ranked.
        chosen = Generate_Parameterized_Dataset._matched_case_records(
            self._calibration(),
            {"num_floor": 6, "story_height_in": 144.0},
            {},               # empty pair table
            [7],
            [0.020],
            [7],
            case_index=3, seed=0,
        )
        self.assertEqual(chosen, [7])

    def test_matching_is_deterministic_for_a_seed(self):
        args = (
            self._calibration(), {"num_floor": 6, "story_height_in": 144.0},
            self._pairs(), [1, 2], [0.020], [1],
        )
        first = Generate_Parameterized_Dataset._matched_case_records(
            *args, case_index=5, seed=11
        )
        second = Generate_Parameterized_Dataset._matched_case_records(
            *args, case_index=5, seed=11
        )
        self.assertEqual(first, second)

    def test_unclamped_factor_separates_over_ceiling_records(self):
        # scale_factor_for clamps at SCALE_FACTOR_MAX, which makes every
        # over-ceiling record tie. Ranking needs them ordered.
        from Data_Generation.Calibrate_Intensity import scale_factor_for

        co = {"a": -3.58, "b": 0.89, "c": 1.24}
        weak = Generate_Parameterized_Dataset._unclamped_scale_factor(0.04, 1.5, 0.02, co)
        weaker = Generate_Parameterized_Dataset._unclamped_scale_factor(0.04, 1.5, 0.002, co)
        self.assertGreater(weaker, weak)
        self.assertEqual(
            scale_factor_for(0.04, 1.5, 0.02, co),
            scale_factor_for(0.04, 1.5, 0.002, co),
        )

    def test_record_selection_is_fingerprinted_only_for_calibrated_plans(self):
        # Adding the key unconditionally would make every existing
        # uniform-ladder plan mismatch on reload and refuse to resume.
        import inspect

        source = inspect.getsource(Generate_Parameterized_Dataset.load_plan)
        self.assertIn("record_selection", source)
        self.assertIn('if calibration else {}', source)


class GenerationRegressionTests(unittest.TestCase):
    def test_scheduler_sha256_is_uppercase_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "plan.csv"
            path.write_bytes(b"deterministic plan\n")

            digest = Generate_Parameterized_Dataset.sha256_file(path)

            self.assertEqual(
                digest,
                "E70AA1A390E16921CD8F5E1328E38406BA88590FEBC5D66545FDB4DF4A4F2A2E",
            )

    def test_expansion_record_set_enables_only_pairs_up_to_30000_points(self):
        hybrid_pairs = Generate_Hybrid_Dataset._all_pair_keys(
            "peer_expansion_30k", None, 30000
        )
        hybrid_ids = [item[1] for item in hybrid_pairs]
        scheduler_ids = Generate_Parameterized_Dataset.eligible_record_ids(
            "peer_expansion_30k", 30000
        )

        self.assertEqual(len(hybrid_ids), 26)
        self.assertEqual(scheduler_ids, hybrid_ids)
        self.assertNotIn(34, hybrid_ids)
        self.assertNotIn(51, hybrid_ids)
        self.assertLessEqual(max(item[2] for item in hybrid_pairs), 30000)

    def test_expansion_plan_uses_new_case_ids_and_unused_geometries(self):
        base = Generate_Parameterized_Dataset.build_plan(
            2500, [1], seed=Generate_Parameterized_Dataset.SEED
        )
        expansion = Generate_Parameterized_Dataset.build_plan(
            1300,
            list(range(1, 27)),
            seed=Generate_Parameterized_Dataset.SEED,
            geometry_offset=2500,
            case_id_offset=2500,
        )
        geometry_keys = (
            "num_bay_x",
            "num_bay_y",
            "num_floor",
            "story_height_ft",
            "bay_x_width_ft",
            "bay_y_width_ft",
        )
        base_geometries = {
            tuple(case[key] for key in geometry_keys) for case in base
        }
        expansion_geometries = {
            tuple(case[key] for key in geometry_keys) for case in expansion
        }

        self.assertEqual(expansion[0]["case_id"], "case_2501")
        self.assertEqual(expansion[-1]["case_id"], "case_3800")
        self.assertEqual(len(expansion_geometries), 1300)
        self.assertFalse(base_geometries & expansion_geometries)

    def test_parameterized_scheduler_uses_lightweight_record_catalog(self):
        expected = [
            item[1]
            for item in Generate_Hybrid_Dataset._all_pair_keys(
                "peer_mle_all", None, 15000
            )
        ]

        actual = Generate_Parameterized_Dataset.eligible_record_ids(
            "peer_mle_all", 15000
        )

        self.assertEqual(actual, expected)

    def test_parameterized_progress_uses_cached_completion_state(self):
        cases = [
            {"case_id": "case_0001", "result_id": 1},
            {"case_id": "case_0002", "result_id": 2},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with mock.patch.object(
                Generate_Parameterized_Dataset,
                "complete",
                side_effect=AssertionError("progress must not rescan case artifacts"),
            ):
                state = Generate_Parameterized_Dataset.progress(
                    root=root,
                    cases=cases,
                    results={},
                    active={"case_0002": {"pid": 123}},
                    status="running",
                    started="test-start",
                    completed_ids={"case_0001"},
                    write_manifest=True,
                    invocation_case_ids=["case_0001", "case_0002"],
                    case_start=1,
                    case_end=2,
                )

            self.assertEqual(state["completed_count"], 1)
            self.assertEqual(state["remaining_count"], 1)
            self.assertEqual(state["active_cases"], {"case_0002": {"pid": 123}})
            self.assertEqual(state["invocation_completed_count"], 1)
            self.assertEqual(state["invocation_remaining_case_ids"], ["case_0002"])

    def test_device_case_ranges_are_non_overlapping(self):
        cases = [{"case_id": f"case_{index:04d}"} for index in range(1, 11)]
        completed = {"case_0001", "case_0006"}

        scope_a, selected_a = Generate_Parameterized_Dataset.select_pending_cases(
            cases, completed, case_start=1, case_end=5
        )
        scope_b, selected_b = Generate_Parameterized_Dataset.select_pending_cases(
            cases, completed, case_start=6, case_end=10
        )

        ids_a = {case["case_id"] for case in selected_a}
        ids_b = {case["case_id"] for case in selected_b}
        self.assertEqual(len(scope_a), 5)
        self.assertEqual(len(scope_b), 5)
        self.assertFalse(ids_a & ids_b)
        self.assertEqual(ids_a, {"case_0002", "case_0003", "case_0004", "case_0005"})
        self.assertEqual(ids_b, {"case_0007", "case_0008", "case_0009", "case_0010"})

        with self.assertRaises(ValueError):
            Generate_Parameterized_Dataset.select_pending_cases(
                cases, completed, case_start=0, case_end=5
            )

    def test_refused_designs_are_terminal_and_reported(self):
        """A saved design that qualification refused is skipped on resume, with its reasons."""
        cases = [{"case_id": f"case_{index:04d}", "runs": [{"run_name": "r1"}]} for index in range(1, 4)]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            refused_dir = root / "cases" / "case_0002"
            refused_dir.mkdir(parents=True)
            (refused_dir / "design.json").write_text(json.dumps({
                "schema_version": "x", "qualification": {"accepted": False, "counts": {"fail": 1, "not_evaluated": 2},
                "checks": [{"id": "scwb:joint_1/x/positive", "status": "fail"},
                           {"id": "floor.independent_hand_verification", "status": "not_evaluated"},
                           {"id": "slab_fire_resistance", "status": "not_evaluated"}]}}), encoding="utf-8")
            ok_dir = root / "cases" / "case_0003"
            ok_dir.mkdir(parents=True)
            (ok_dir / "design.json").write_text(json.dumps({"qualification": {"accepted": True, "checks": []}}), encoding="utf-8")

            refused = Generate_Parameterized_Dataset.scan_refused(root, cases)
            self.assertEqual(list(refused), ["case_0002"])
            self.assertEqual(refused["case_0002"]["failed"], ["scwb"])
            self.assertEqual(refused["case_0002"]["not_evaluated"], ["floor.independent_hand_verification", "slab_fire_resistance"])
            _scope, selected = Generate_Parameterized_Dataset.select_pending_cases(cases, set(), refused_ids=refused)
            self.assertEqual([c["case_id"] for c in selected], ["case_0001", "case_0003"])
            _scope, retried = Generate_Parameterized_Dataset.select_pending_cases(cases, set(), refused_ids=())
            self.assertEqual(len(retried), 3)
            results = {"case_0002": {"case_id": "case_0002", "status": "design_refused", "design_refusal": refused["case_0002"]}}
            state = Generate_Parameterized_Dataset.progress(root, cases, results, {}, "running", "now", set())
            self.assertEqual(state["design_refused_case_ids"], ["case_0002"])
            self.assertEqual(state["remaining_case_ids"], ["case_0001", "case_0003"])
            self.assertEqual(state["design_refusals"]["case_0002"]["failed"], ["scwb"])

    def test_stop_request_is_atomic_and_detectable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stop_path = Path(temp_dir) / "STOP_GENERATION.json"

            payload = Generate_Hybrid_Dataset._request_stop(stop_path)

            self.assertTrue(Generate_Hybrid_Dataset._stop_requested(stop_path))
            self.assertEqual(
                json.loads(stop_path.read_text(encoding="utf-8")),
                payload,
            )
            self.assertFalse((stop_path.parent / f".{stop_path.name}.tmp").exists())

    def test_atomic_checkpoint_retries_transient_windows_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "generation_state.json"
            real_replace = Generate_Hybrid_Dataset.os.replace
            attempts = []

            def transient_lock(source, destination):
                attempts.append((source, destination))
                if len(attempts) < 3:
                    raise PermissionError(5, "Access is denied")
                return real_replace(source, destination)

            with (
                mock.patch.object(
                    Generate_Hybrid_Dataset.os,
                    "replace",
                    side_effect=transient_lock,
                ),
                mock.patch.object(Generate_Hybrid_Dataset.time, "sleep") as sleep,
            ):
                Generate_Hybrid_Dataset._write_json_atomic(
                    output_path,
                    {"status": "completed"},
                )

            self.assertEqual(len(attempts), 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                {"status": "completed"},
            )

    def test_running_child_is_terminated_when_stop_is_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            stop_path = temp_path / "STOP_GENERATION.json"
            log_path = temp_path / "child.log"
            Generate_Hybrid_Dataset._request_stop(stop_path)

            result = Generate_Hybrid_Dataset._run_command(
                [sys.executable, "-B", "-c", "import time; time.sleep(30)"],
                cwd=temp_path,
                log_path=log_path,
                stop_path=stop_path,
                poll_interval_sec=0.01,
            )

            self.assertTrue(result["interrupted"])
            self.assertEqual(result["stop_reason"], "stop_file")
            self.assertNotEqual(result["returncode"], 0)
            self.assertIn("will be retried on resume", log_path.read_text(encoding="utf-8"))

    def test_resume_reuses_successful_ntha_when_sample_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            args = SimpleNamespace(
                ntha_root=str(temp_path / "ntha"),
                dataset_dir=str(temp_path / "dataset"),
                max_npts=15000,
                result_id=[1],
                set_name="peer_mle_all",
                split=None,
                limit=1,
                start_index=0,
                _stop_path=str(temp_path / "dataset" / "STOP_GENERATION.json"),
                x_only=False,
                scale_factor=None,
                damping_ratio=0.05,
                rayleigh_mode_i=0,
                rayleigh_mode_j=2,
                dt_factor=1.0,
                skip_existing=True,
                python_exe=sys.executable,
                catalog_summary=False,
            )

            with (
                mock.patch.object(
                    Generate_Hybrid_Dataset,
                    "_all_pair_keys",
                    return_value=[("peer_result_id:1", 1, 100)],
                ),
                mock.patch.object(
                    Generate_Hybrid_Dataset,
                    "_status_success",
                    return_value=True,
                ),
                mock.patch.object(Generate_Hybrid_Dataset, "_run_command") as run_command,
            ):
                summaries = Generate_Hybrid_Dataset.run_ntha_batch(args)

            run_command.assert_not_called()
            self.assertTrue(summaries[0]["skipped"])
            self.assertIn("sample will be compiled", summaries[0]["reason"])

    def test_time_series_uses_scaled_values_without_rewriting_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "record.txt"
            source.write_text("1.0\n2.0\n", encoding="utf-8")
            before = source.read_bytes()
            record = GroundMotionRecord(
                record_id="TEST",
                dt_sec=0.01,
                acceleration=np.asarray([1.0, 2.0]),
                units="g",
                scale_factor=2.0,
                source_path=str(source),
            )

            with mock.patch.object(Ground_Motion.ops, "timeSeries") as time_series:
                returned = Ground_Motion.define_path_time_series(10, record)

            self.assertIsNone(returned)
            self.assertEqual(before, source.read_bytes())
            args = time_series.call_args.args
            self.assertIn("-values", args)
            values_index = args.index("-values") + 1
            self.assertAlmostEqual(args[values_index], 2.0 * 386.4)
            self.assertAlmostEqual(args[values_index + 1], 4.0 * 386.4)
            self.assertNotIn("-filePath", args)

    def test_analysis_substeps_preserve_full_excitation_duration(self):
        x = GroundMotionRecord("X", 0.01, np.zeros(100))
        y = GroundMotionRecord("Y", 0.01, np.zeros(120))

        dt, steps = _analysis_step_schedule(x, dt_factor=0.5)
        self.assertAlmostEqual(dt, 0.005)
        self.assertEqual(steps, 200)

        dt, steps = _analysis_step_schedule(x, y, dt_factor=1.0)
        self.assertAlmostEqual(dt, 0.01)
        self.assertEqual(steps, 120)

        with self.assertRaises(ValueError):
            _analysis_step_schedule(x, dt_factor=0.0)

    def test_exporter_recovers_stale_path_and_applies_scale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ground_motion_dir = Path(temp_dir) / "Ground_Motions"
            processed_dir = ground_motion_dir / "processed"
            metadata_dir = ground_motion_dir / "metadata"
            processed_dir.mkdir(parents=True)
            metadata_dir.mkdir(parents=True)
            processed_file = processed_dir / "TEST_in_per_sec2.txt"
            np.savetxt(processed_file, np.asarray([1.0, 2.0, 3.0]))
            manifest = metadata_dir / "record_manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "record_id",
                        "processed_file",
                        "processed_units",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "record_id": "TEST",
                        "processed_file": "processed/TEST_in_per_sec2.txt",
                        "processed_units": "in/sec^2",
                    }
                )

            summary = {
                "record_id": "TEST",
                "dt_sec": 1.0,
                "analysis_dt_sec": 0.5,
                "units": "in/sec^2",
                "scale_factor": 2.0,
                "source_path": r"C:\old\desktop\processed\TEST_in_per_sec2.txt",
            }
            with (
                mock.patch.object(
                    Hybrid_Exporter,
                    "GROUND_MOTION_DIR",
                    ground_motion_dir,
                ),
                mock.patch.object(
                    Hybrid_Exporter,
                    "GROUND_MOTION_MANIFEST",
                    manifest,
                ),
            ):
                acceleration = Hybrid_Exporter._load_acceleration_from_summary(
                    summary,
                    target_steps=5,
                )

            np.testing.assert_allclose(
                acceleration,
                np.asarray([2.0, 3.0, 4.0, 5.0, 6.0]),
            )

    def test_manifest_paths_are_relative_and_resolvable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir) / "dataset"
            target = Path(temp_dir) / "ntha" / "peer_1"
            base.mkdir()
            target.mkdir(parents=True)

            portable = Hybrid_Exporter._portable_path(target, base)

            self.assertFalse(Path(portable).is_absolute())
            self.assertEqual((base / portable).resolve(), target.resolve())

    def test_output_identity_rejects_mixed_model_configuration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "peer_1"
            output_dir.mkdir()
            current = collect_global_parameters()
            current["num_floor"] = int(current["num_floor"]) + 1
            (output_dir / "global_parameters.json").write_text(
                json.dumps(current),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "different model configurations"):
                validate_ntha_output_compatibility(output_dir)

    def test_analysis_variants_get_distinct_output_names(self):
        self.assertEqual(analysis_run_name("peer_1"), "peer_1")
        self.assertEqual(
            analysis_run_name("peer_1", scale_factor=2.0),
            "peer_1__sf_2",
        )
        self.assertEqual(
            analysis_run_name("peer_1", x_only=True, dt_factor=0.5),
            "peer_1__x_only__dtf_0p5",
        )

    def test_output_identity_rejects_changed_imk_energy_parameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            current = collect_global_parameters()
            (output_dir / "global_parameters.json").write_text(json.dumps(current), encoding="utf-8")
            with mock.patch.object(sp, "IMK_LAMBDA_S", sp.IMK_LAMBDA_S * 2):
                with self.assertRaisesRegex(RuntimeError, "imk_cyclic_parameters"):
                    validate_ntha_output_compatibility(output_dir)

    def test_imk_material_definition_uses_documented_argument_order(self):
        # Regression guard: this call has previously shipped with a wrong
        # OpenSees IMKBilin argument order (a nonexistent "A" deterioration
        # mode standing in for the FmaxFy/FresFy strength ratios). Pin the
        # exact positional order the current IMKBilin signature requires.
        with mock.patch.object(sp, "IMK_MATERIAL_TYPE", "IMKBilin"), mock.patch.object(IMK_Hinges.ops, "uniaxialMaterial") as uniaxial_material:
            IMK_Hinges._define_imk_peak_material(
                mat_tag=101,
                elastic_stiffness=5000.0,
                yield_moment=250.0,
            )

        uniaxial_material.assert_called_once()
        args = uniaxial_material.call_args.args
        self.assertEqual(len(args), 23)
        self.assertEqual(
            args,
            (
                "IMKBilin",
                101,
                5000.0,
                sp.IMK_THETA_P_POS,
                sp.IMK_THETA_PC_POS,
                sp.IMK_THETA_U_POS,
                250.0,
                getattr(sp, "IMK_FMAXFY_POS", 1.10),
                getattr(sp, "IMK_FRESFY_POS", sp.IMK_RES_POS),
                sp.IMK_THETA_P_NEG,
                sp.IMK_THETA_PC_NEG,
                sp.IMK_THETA_U_NEG,
                250.0,
                getattr(sp, "IMK_FMAXFY_NEG", 1.10),
                getattr(sp, "IMK_FRESFY_NEG", sp.IMK_RES_NEG),
                sp.IMK_LAMBDA_S,
                sp.IMK_LAMBDA_C,
                sp.IMK_LAMBDA_K,
                sp.IMK_C_S,
                sp.IMK_C_C,
                sp.IMK_C_K,
                sp.IMK_D_POS,
                sp.IMK_D_NEG,
            ),
        )

    def test_hinge_stiffness_mode_selects_correct_basis(self):
        # Regression guard: hinge calibration previously used a
        # "yield_rotation" basis that was later changed to
        # "member_stiffness_factor". Pin the behavior of every supported
        # mode so a future edit can't silently change which basis is used.
        length = 120.0

        with mock.patch.object(sp, "IMK_HINGE_STIFFNESS_MODE", "yield_rotation"):
            components = IMK_Hinges.imk_hinge_stiffness_components(
                "column", "rot_z", length
            )
        self.assertEqual(components["mode"], "yield_rotation")
        self.assertAlmostEqual(
            components["selected_stiffness"], components["yield_based_stiffness"]
        )

        with mock.patch.object(sp, "IMK_HINGE_STIFFNESS_MODE", "member_stiffness_factor"):
            components = IMK_Hinges.imk_hinge_stiffness_components(
                "column", "rot_z", length
            )
        self.assertEqual(components["mode"], "member_stiffness_factor")
        self.assertAlmostEqual(
            components["selected_stiffness"], components["member_based_stiffness"]
        )

        with mock.patch.object(sp, "IMK_HINGE_STIFFNESS_MODE", "max"):
            components = IMK_Hinges.imk_hinge_stiffness_components(
                "column", "rot_z", length
            )
        self.assertAlmostEqual(
            components["selected_stiffness"],
            max(
                components["yield_based_stiffness"],
                components["member_based_stiffness"],
            ),
        )

        with mock.patch.object(sp, "IMK_HINGE_STIFFNESS_MODE", "bogus"):
            with self.assertRaises(ValueError):
                IMK_Hinges.imk_hinge_stiffness_components("column", "rot_z", length)

    def test_member_orientation_ties_expected_dofs(self):
        self.assertEqual(IMK_Hinges._orientation("column")[1], (1, 2, 3, 6))
        self.assertEqual(IMK_Hinges._orientation("beam_x")[1], (1, 2, 3, 4))
        self.assertEqual(IMK_Hinges._orientation("beam_y")[1], (1, 2, 3, 5))

        with self.assertRaises(ValueError):
            IMK_Hinges._orientation("brace")



class CollapseCompletionTests(unittest.TestCase):
    """A collapsed run is finished, not failed.

    Hybrid_Exporter keeps and labels a run that stopped early because the
    structure collapsed, and Calibrate_Intensity keeps it as an ordinary
    observation below its drift ceiling (it does not censor). The scheduler
    judged the same run only by step count, so it stayed in
    the remaining list and was retried on every invocation -- forever, since
    the analysis is deterministic.
    """

    def _make_run(self, root, *, failed, collapse, with_sample=True):
        case = {
            "case_id": "case_0001",
            "runs": [{"run_index": 1, "result_id": 7, "scale_factor": 3.0,
                      "run_name": "peer_7__sf_3"}],
        }
        run = case["runs"][0]
        paths = Generate_Parameterized_Dataset.run_paths_for(root, case, run)
        paths["status"].parent.mkdir(parents=True, exist_ok=True)
        paths["status"].write_text(json.dumps({
            "npts_requested": 6000,
            "completed_steps": 4588 if failed else 6000,
            "failed": failed,
        }), encoding="utf-8")
        if with_sample:
            paths["sample"].parent.mkdir(parents=True, exist_ok=True)
            paths["sample"].write_bytes(b"")
            paths["sample"].with_name("hybrid_metadata.json").write_text(
                json.dumps({"collapse": collapse, "npts_requested": 6000,
                            "completed_steps": 4588 if failed else 6000}), encoding="utf-8"
            )
        return case, run

    def test_collapsed_run_counts_as_complete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case, run = self._make_run(root, failed=True, collapse=True)
            self.assertTrue(Generate_Parameterized_Dataset.complete_run(root, case, run))
            self.assertTrue(Generate_Parameterized_Dataset.complete(root, case))

    def test_failed_run_without_collapse_label_is_not_complete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case, run = self._make_run(root, failed=True, collapse=False)
            self.assertFalse(Generate_Parameterized_Dataset.complete_run(root, case, run))

    def test_collapse_without_a_sample_is_not_complete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case, run = self._make_run(root, failed=True, collapse=True, with_sample=False)
            self.assertFalse(Generate_Parameterized_Dataset.complete_run(root, case, run))

    def test_clean_run_still_complete_without_any_collapse_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case, run = self._make_run(root, failed=False, collapse=False)
            self.assertTrue(Generate_Parameterized_Dataset.complete_run(root, case, run))

    def test_previous_collapse_does_not_finish_a_new_failed_attempt(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case, run = self._make_run(root, failed=True, collapse=True)
            paths = Generate_Parameterized_Dataset.run_paths_for(root, case, run)
            metadata_path = paths["sample"].with_name("hybrid_metadata.json")
            metadata = json.loads(metadata_path.read_text())
            metadata["analysis_status_sha256"] = hashlib.sha256(paths["status"].read_bytes()).hexdigest()
            metadata_path.write_text(json.dumps(metadata))
            old = paths["status"].stat()
            status = json.loads(paths["status"].read_text())
            # Same counts and timestamps: only the content binding catches it.
            status["retry_reason"] = "new failed attempt"
            paths["status"].write_text(json.dumps(status))
            os.utime(paths["status"], ns=(old.st_atime_ns, old.st_mtime_ns))
            self.assertFalse(Generate_Parameterized_Dataset.complete_run(root, case, run))

    def test_legacy_collapse_with_different_step_count_is_not_complete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case, run = self._make_run(root, failed=True, collapse=True)
            paths = Generate_Parameterized_Dataset.run_paths_for(root, case, run)
            old = paths["status"].stat()
            status = json.loads(paths["status"].read_text())
            status["completed_steps"] = 3
            paths["status"].write_text(json.dumps(status))
            os.utime(paths["status"], ns=(old.st_atime_ns, old.st_mtime_ns))
            self.assertFalse(Generate_Parameterized_Dataset.complete_run(root, case, run))


class StaleSampleTests(unittest.TestCase):
    """A retried NTHA must not be served the previous attempt's sample."""

    def _layout(self, root):
        ntha = root / "ntha" / "peer_1"
        dataset = root / "dataset" / "peer_1"
        ntha.mkdir(parents=True)
        dataset.mkdir(parents=True)
        status = ntha / "status.json"
        sample = dataset / "hybrid_sample.npz"
        status.write_text("{}", encoding="utf-8")
        sample.write_bytes(b"")
        return status, sample, ntha

    def test_sample_newer_than_analysis_is_fresh(self):
        from Data_Generation import Hybrid_Exporter
        with tempfile.TemporaryDirectory() as temp_dir:
            status, sample, ntha = self._layout(Path(temp_dir))
            os.utime(status, (1_000, 1_000))
            os.utime(sample, (2_000, 2_000))
            self.assertFalse(Hybrid_Exporter._sample_is_stale(ntha, sample))

    def test_analysis_newer_than_sample_is_stale(self):
        from Data_Generation import Hybrid_Exporter
        with tempfile.TemporaryDirectory() as temp_dir:
            status, sample, ntha = self._layout(Path(temp_dir))
            os.utime(sample, (1_000, 1_000))
            os.utime(status, (2_000, 2_000))
            self.assertTrue(Hybrid_Exporter._sample_is_stale(ntha, sample))

    def test_missing_status_is_not_reusable(self):
        from Data_Generation import Hybrid_Exporter
        with tempfile.TemporaryDirectory() as temp_dir:
            status, sample, ntha = self._layout(Path(temp_dir))
            status.unlink()
            self.assertTrue(Hybrid_Exporter._sample_is_stale(ntha, sample))

    def test_status_digest_detects_retry_even_with_preserved_timestamps(self):
        import hashlib
        from Data_Generation import Hybrid_Exporter
        with tempfile.TemporaryDirectory() as temp_dir:
            status, sample, ntha = self._layout(Path(temp_dir))
            digest = hashlib.sha256(status.read_bytes()).hexdigest()
            sample.with_name("hybrid_metadata.json").write_text(json.dumps({"analysis_status_sha256": digest}))
            status.write_text('{"failed":true}')
            os.utime(status, (1000, 1000))
            os.utime(sample, (2000, 2000))
            self.assertTrue(Hybrid_Exporter._sample_is_stale(ntha, sample))

if __name__ == "__main__":
    unittest.main()
