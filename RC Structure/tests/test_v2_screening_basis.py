"""V2 screening basis (2026-10-01): Risk Category III as one consistent basis, the half-foot geometry
population, the analysis profile in the design identity, manifest validation, and the real worker path.

The worker tests design a tiny off-population fixture frame (2 x 1 bays, 2 stories, 15-ft bays, one
section iteration) in a fresh interpreter through the runner's own worker entry point. That is an
implementation fixture: it is not one of the screening cases and no ground motion is run.
"""
import contextlib
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path
import random
from unittest import mock

RC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_DIR))
sys.path.insert(0, str(RC_DIR / "Data_Generation"))

import Structure_Parameters as sp                                              # noqa: E402
import Generate_Parameterized_Dataset as gen                                    # noqa: E402
from Design import Design_Driver as driver                                      # noqa: E402
from Design import Screening_Manifest as sm                                     # noqa: E402
from Design import Verify_Designs as verify                                     # noqa: E402
from Design.Config import DemandPolicy, DesignConfig                            # noqa: E402
from Design.SMRF_Demands import (evaluate_drift_and_stability, risk_basis_check, seismic_design_category,  # noqa: E402
                                 story_drift_basis)
from Design.Verification_Integrity import expected_identity                     # noqa: E402
from Loads.Seismic_ELF import seismic_response_coefficient                      # noqa: E402
from Model import Analysis_Profile as ap                                        # noqa: E402

V2 = ap.V2_FLEXURE
FIXTURE_CASE = {"case_id": "case_9002", "num_bay_x": 2, "num_bay_y": 1, "num_floor": 2, "story_height_ft": 12,
                "bay_x_width_ft": 15, "bay_y_width_ft": 15, "seismic_site": "sdc_d_low",
                "geometry_name": "fixture_bx2_by1_s2_sh12ft_bwx15ft_bwy15ft_sdc_d_low"}
PROBE_DATE = "2026-10-01"


@contextlib.contextmanager
def restored_parameters():
    """Every Structure_Parameters name as it was, whatever the block changes or adds."""
    snapshot = dict(vars(sp))
    try:
        yield
    finally:
        for key in set(vars(sp)) - set(snapshot):
            delattr(sp, key)
        vars(sp).update(snapshot)


def story(delta=0.2, height=144.0):
    return {"id": 1, "height_in": height, "node_deltas_in": {"0,0": {"x": delta, "y": 0.0}}, "expected_node_ids": ["0,0"],
            "directions": ["x"], "story_shear_kip": {"x": 100.0}, "gravity_above_kip": 500.0}


def manifest_dict():
    return {
        "schema": sm.MANIFEST_SCHEMA, "profile_id": V2, "status": "prepared_for_implementation_not_executed",
        "scope": "test manifest",
        "design_basis": {"bay_and_story_height_increment_ft": 0.5, "risk_category": "III", "importance_factor": 1.25,
                         "bay_x_width_ft": [18, 30], "bay_y_width_ft": [18, 30], "story_height_ft": [12, 18],
                         "num_bay_x": [2, 6], "num_bay_y": [2, 6], "num_floor": [4, 9]},
        "nonlinear_profile": {"element_formulation": "imk", "member_material": "IMKPeakOriented", "apply_to_beams": True,
                              "apply_to_columns": True, "joint_model": "rigid_centerline", "explicit_face_slip_interfaces": False},
        "execution": {"initial_case_id": "case_0001", "initial_workers": 1, "subsequent_max_workers": 2, "max_section_iter": 10,
                      "output_root": None, "fresh_root_required": True},
        "cases": [
            {"case_id": "case_0001", "label": "reference", "num_bay_x": 2, "num_bay_y": 6, "num_floor": 6, "bay_x_width_ft": 20,
             "bay_y_width_ft": 20, "story_height_ft": 14, "seismic_site": "sdc_d_low", "geometry_name": "v2_reference", "purpose": "p"},
            {"case_id": "case_0002", "label": "fractional", "num_bay_x": 2, "num_bay_y": 2, "num_floor": 4, "bay_x_width_ft": 18.5,
             "bay_y_width_ft": 30, "story_height_ft": 12.5, "seismic_site": "sdc_d_high", "geometry_name": "v2_fractional", "purpose": "p"},
        ],
        "acceptance_policy": {"probe_is_not_acceptance": True},
    }


class RiskBasis(unittest.TestCase):
    def test_category_three_is_the_one_configured_basis(self):
        self.assertEqual(sp.ASCE_RISK_CATEGORY, "III")
        self.assertEqual(sp.ASCE_IE, 1.25)
        self.assertEqual(sp.SEISMIC_IMPORTANCE_FACTOR_BY_RISK_CATEGORY, {"I": 1.0, "II": 1.0, "III": 1.25, "IV": 1.5})
        self.assertEqual(sp.ALLOWABLE_STORY_DRIFT_RATIO_BY_RISK_CATEGORY, {"I": 0.020, "II": 0.020, "III": 0.015, "IV": 0.010})
        basis = sp.seismic_design_basis(redundancy_factor=1.3, seismic_design_category="D")
        self.assertEqual((basis["risk_category"], basis["importance_factor"], basis["allowable_story_drift_ratio"]), ("III", 1.25, 0.015))
        self.assertAlmostEqual(basis["effective_story_drift_ratio"], 0.015 / 1.3, places=15)
        self.assertNotAlmostEqual(basis["effective_story_drift_ratio"], 0.012, places=4)     # 0.01154, not a rounded 0.012
        self.assertFalse(basis["low_rise_allowance_asserted"])
        self.assertEqual(sp.seismic_design_basis(redundancy_factor=1.3, seismic_design_category="C")["effective_story_drift_ratio"], 0.015)
        self.assertIn("not a nonlinear intensity target", basis["scope_note"])

    def test_site_class_c_is_the_one_declared_basis(self):
        """User decision 2026-10-02: Site Class C (ASCE 7-22 Table 20.2-1) for the whole population, one source."""
        self.assertEqual(sp.ASCE_SITE_CLASS, "C")
        self.assertEqual(sp.ASCE_SITE_CLASSES, ("A", "B", "BC", "C", "CD", "D", "DE", "E", "F"))
        self.assertEqual(DemandPolicy().site_class, "C")
        self.assertEqual(DesignConfig.from_structure_parameters().demands.site_class, "C")
        self.assertEqual(verify.probe_config(PROBE_DATE).demands.site_class, "C")
        basis = sp.seismic_design_basis(site_class="C")
        self.assertEqual(basis["site_class"], "C")
        self.assertIn("Site Class F only", basis["site_response_analysis"])
        self.assertIn("read from ASCE/SEI 7-22", basis["source_access_note"])
        with self.assertRaisesRegex(ValueError, "contradicts the declared basis"):
            sp.seismic_design_basis(site_class="D")
        with self.assertRaisesRegex(ValueError, "contradicts the declared basis"):
            driver.design_structure(cfg=DesignConfig(demands=DemandPolicy(declared_by="a", declaration_date=PROBE_DATE,
                                                                          declaration_basis="b", site_class="D")), verbose=False)

    def test_policy_and_both_probe_factories_inherit_the_category(self):
        from Design import Evidence_Summary as evidence
        self.assertEqual(DemandPolicy().risk_category, "III")
        self.assertEqual(DesignConfig.from_structure_parameters().demands.risk_category, "III")
        self.assertEqual(verify.probe_config(PROBE_DATE).demands.risk_category, "III")
        self.assertEqual(evidence.probe_config(PROBE_DATE).demands.risk_category, "III")
        with restored_parameters():
            sp.apply_risk_category("II")
            self.assertEqual((sp.ASCE_RISK_CATEGORY, sp.ASCE_IE), ("II", 1.0))                # set together, never one alone
            self.assertEqual(DemandPolicy().risk_category, "II")
            self.assertEqual(verify.probe_config(PROBE_DATE).demands.risk_category, "II")

    def test_contradictory_category_and_factor_are_refused_everywhere(self):
        with self.assertRaises(ValueError):
            sp.seismic_design_basis(risk_category="II")                                       # a policy naming another category
        with self.assertRaises(ValueError):
            sp.seismic_design_basis(importance_factor=1.0)                                    # a stale factor
        with self.assertRaises(ValueError):
            sp.apply_risk_category("V")
        with restored_parameters():
            sp.ASCE_IE = 1.0                                                                  # a stale module default
            with self.assertRaises(ValueError):
                sp.seismic_design_basis()
            with self.assertRaises(ValueError):
                driver.design_request_identity()
            with self.assertRaises(ValueError):
                driver.design_structure(cfg=verify.probe_config(PROBE_DATE), max_section_iter=1, max_steel_iter=1, verbose=False)
        with self.assertRaises(ValueError):
            driver.design_request_identity(DesignConfig(demands=DemandPolicy(risk_category="II")))
        for args in (("III", 1.0, None), ("II", 1.25, None), ("III", 1.25, 0.020), ("IV", 1.5, 0.015), ("V", 1.0, None)):
            with self.assertRaises(ValueError, msg=str(args)):
                story_drift_basis(*args)
        self.assertEqual(story_drift_basis("III", 1.25, 0.012)["limit"], 0.012)               # stricter than the row is allowed

    def test_drift_evaluator_uses_the_category_row_and_keeps_its_own_arguments_consistent(self):
        result = evaluate_drift_and_stability([story()], importance_factor=1.25, risk_category="III")
        row = result["stories"][0]
        self.assertAlmostEqual(row["allowable_drift_ratio"], 0.015 / 1.3, places=15)
        self.assertAlmostEqual(row["design_drift_in"], 0.2 * 5.5 / 1.25, places=12)             # Cd / Ie
        assumptions = result["assumptions"]
        self.assertEqual((assumptions["risk_category"], assumptions["importance_factor"]), ("III", 1.25))
        self.assertEqual(assumptions["base_drift_limit_ratio"], 0.015)
        self.assertAlmostEqual(assumptions["effective_drift_limit_ratio"], 0.015 / 1.3, places=15)
        self.assertFalse(assumptions["low_rise_allowance_asserted"])
        in_c = evaluate_drift_and_stability([story()], importance_factor=1.25, risk_category="III", seismic_design_category="C")
        self.assertEqual(in_c["stories"][0]["allowable_drift_ratio"], 0.015)
        legacy = evaluate_drift_and_stability([story()])                                      # the self-consistent Risk II default
        self.assertAlmostEqual(legacy["stories"][0]["allowable_drift_ratio"], 0.02 / 1.3)
        self.assertEqual(legacy["assumptions"]["risk_category"], "II")
        with self.assertRaises(ValueError):
            evaluate_drift_and_stability([story()], importance_factor=1.25)                   # Ie of III with the default category
        with self.assertRaises(ValueError):
            evaluate_drift_and_stability([story()], importance_factor=1.25, risk_category="III", drift_limit_ratio=0.02)
        # the same story count never changes the limit: no low-rise allowance for a four-story frame
        self.assertNotIn("num_floor", evaluate_drift_and_stability.__code__.co_varnames)

    def test_importance_factor_reaches_the_base_shear_coefficient(self):
        with restored_parameters():
            sp.apply_seismic_site("sdc_d_low")
            cs_three = seismic_response_coefficient(0.3)
            self.assertAlmostEqual(cs_three["cs_basic"], sp.ASCE_SDS / (sp.ASCE_R / 1.25), places=15)
            sp.apply_risk_category("II")
            cs_two = seismic_response_coefficient(0.3)
            self.assertAlmostEqual(cs_three["cs"] / cs_two["cs"], 1.25, places=12)
        # Risk III derives the same seismic design category as Risk II for every site of the population
        for _label, sds, sd1, s1 in sp.SEISMIC_SITE_OPTIONS:
            self.assertEqual(seismic_design_category(sds, sd1, s1, "III"), seismic_design_category(sds, sd1, s1, "II"))

    def test_saved_risk_basis_check_accepts_a_consistent_record_and_rejects_a_relabelled_one(self):
        policy = {"risk_category": "III"}
        seismic = {"risk_category": "III", "importance_factor": 1.25}
        drift = {"risk_category": "III", "importance_factor": 1.25, "base_drift_limit_ratio": 0.015,
                 "effective_drift_limit_ratio": 0.015 / 1.3, "low_rise_allowance_asserted": False}
        inputs = {"ASCE_IE": 1.25, "ASCE_RISK_CATEGORY": "III"}
        self.assertEqual(risk_basis_check(policy, seismic, drift, inputs)["status"], "pass")
        self.assertEqual(risk_basis_check(policy, {"sds": 1.0}, drift, inputs)["status"], "not_evaluated")      # a legacy record
        for change in (dict(policy={"risk_category": "II"}), dict(seismic={"risk_category": "III", "importance_factor": 1.0}),
                       dict(drift={**drift, "base_drift_limit_ratio": 0.02}), dict(drift={**drift, "risk_category": "II"}),
                       dict(drift={**drift, "importance_factor": 1.0}), dict(drift={k: v for k, v in drift.items() if k != "base_drift_limit_ratio"}),
                       dict(drift={**drift, "low_rise_allowance_asserted": True}), dict(inputs={"ASCE_IE": 1.0}),
                       dict(inputs={"ASCE_IE": 1.25, "ASCE_RISK_CATEGORY": "II"})):
            arguments = {"policy": policy, "seismic": seismic, "drift": drift, "inputs": inputs, **change}
            check = risk_basis_check(arguments["policy"], arguments["seismic"], arguments["drift"], arguments["inputs"])
            self.assertEqual(check["status"], "fail", change)
            self.assertTrue(check["details"]["problems"])


class GeometryPopulation(unittest.TestCase):
    def test_range_endpoints_levels_and_unchanged_counts(self):
        ranges = gen.RANGES
        self.assertEqual(gen.GEOMETRY_INCREMENT_FT, 0.5)
        for key in ("bay_x_width_ft", "bay_y_width_ft"):
            self.assertEqual((ranges[key][0], ranges[key][-1], len(ranges[key])), (18.0, 30.0, 25))
            self.assertEqual(ranges[key], tuple(tick / 2 for tick in range(36, 61)))
        self.assertEqual((ranges["story_height_ft"][0], ranges["story_height_ft"][-1], len(ranges["story_height_ft"])), (12.0, 18.0, 13))
        self.assertEqual(ranges["story_height_ft"], tuple(tick / 2 for tick in range(24, 37)))
        for key in ("bay_x_width_ft", "bay_y_width_ft", "story_height_ft"):
            steps = {round(b - a, 12) for a, b in zip(ranges[key], ranges[key][1:])}
            self.assertEqual(steps, {0.5})
        self.assertEqual(ranges["num_bay_x"], (2, 3, 4, 5, 6)); self.assertEqual(ranges["num_bay_y"], (2, 3, 4, 5, 6))
        self.assertEqual(ranges["num_floor"], (4, 5, 6, 7, 8, 9))
        for key in ("num_bay_x", "num_bay_y", "num_floor"):
            self.assertTrue(all(type(v) is int for v in ranges[key]))
        self.assertIn("half_foot", gen.POPULATION_BASIS)

    def test_fractional_geometry_round_trips_through_names_json_and_inches(self):
        self.assertEqual(gen.feet_to_inches(18.5), 222.0)
        self.assertEqual(gen.feet_to_inches(12.5), 150.0)
        self.assertEqual(12.0 * 18.5, 222.0); self.assertEqual(12.0 * 12.5, 150.0)             # the worker's own conversion is exact too
        self.assertEqual([gen.feet_label(v) for v in (14, 14.0, 12.5, 18.5, 30.0)], ["14", "14", "12.5", "18.5", "30"])
        for off_grid in (18.25, 12.1, 29.99):
            with self.assertRaises(ValueError):
                gen.feet_to_inches(off_grid)
        for value in gen.RANGES["bay_x_width_ft"] + gen.RANGES["story_height_ft"]:
            self.assertEqual(gen.feet_to_inches(value), 12.0 * value)
            self.assertEqual(float(json.loads(json.dumps(value))), value)
        plan = gen.build_plan(40, [11, 12, 13])
        fractional = [c for c in plan if c["story_height_ft"] % 1 or c["bay_x_width_ft"] % 1 or c["bay_y_width_ft"] % 1]
        self.assertTrue(fractional, "forty cases of a half-foot population include fractional lengths")
        for case in plan:
            self.assertEqual(case["story_height_in"], 12.0 * case["story_height_ft"])
            self.assertEqual(case["bay_x_in"], 12.0 * case["bay_x_width_ft"])
            self.assertIn(f"sh{gen.feet_label(case['story_height_ft'])}ft_bwx{gen.feet_label(case['bay_x_width_ft'])}ft_"
                          f"bwy{gen.feet_label(case['bay_y_width_ft'])}ft", case["geometry_name"])
            self.assertNotIn(".0ft", case["geometry_name"])
            self.assertEqual(json.loads(json.dumps(case))["story_height_ft"], case["story_height_ft"])
        with restored_parameters():
            case = {**FIXTURE_CASE, "bay_x_width_ft": 18.5, "story_height_ft": 12.5}
            verify.configure_case(case)
            self.assertEqual((sp.BAY_X, sp.STORY_H, sp.BAY_Y), (222.0, 150.0, 180.0))
            self.assertEqual(driver.design_request_identity()["inputs"]["BAY_X"], 222.0)

    def test_candidate_planning_is_named_seeded_replaceable_and_serialised(self):
        expected = list(product(*gen.RANGES.values()))
        random.Random(gen.SEED).shuffle(expected)
        self.assertEqual(gen.candidate_geometries(gen.SEED)[:50], expected[:50])                # the same order as before the interface
        self.assertEqual([tuple(c[k] for k in gen.RANGES) for c in gen.build_plan(5, [1])], expected[:5])
        self.assertEqual([tuple(c[k] for k in gen.RANGES) for c in verify.plan_cases(5)], expected[:5])
        provenance = gen.sampling_provenance(gen.SEED)
        self.assertEqual(provenance["method"], "seeded_shuffled_cartesian_prefix_v1")
        self.assertEqual(provenance["seed"], gen.SEED)
        self.assertEqual(provenance["pool_size"], 5 * 5 * 6 * 13 * 25 * 25)
        self.assertEqual(provenance["bounds"]["bay_x_width_ft"], [18.0, 30.0])
        self.assertEqual(provenance["level_counts"], {"num_bay_x": 5, "num_bay_y": 5, "num_floor": 6, "story_height_ft": 13,
                                                      "bay_x_width_ft": 25, "bay_y_width_ft": 25})
        self.assertEqual(len(provenance["source_sha256"]), 64)
        self.assertNotIn("source_sha256", gen.sampling_provenance(gen.SEED, include_source=False))
        self.assertIn("latin_hypercube", provenance["candidate_methods_not_implemented"][0])
        with self.assertRaises(ValueError):
            gen.candidate_geometries(gen.SEED, "latin_hypercube_v1")                             # not implemented, not silently substituted
        fake = [(2, 2, 4, 12.0, 18.0, 18.0), (6, 6, 9, 18.0, 30.0, 30.0)]
        with mock.patch.dict(gen.CANDIDATE_PLANNERS, {"fixed_pair_test": lambda seed: list(fake)}):
            plan = gen.build_plan(2, [1], sampling_method="fixed_pair_test")
            self.assertEqual([tuple(c[k] for k in gen.RANGES) for c in plan], fake)


class IdentityAndCache(unittest.TestCase):
    def test_identity_carries_the_basis_and_the_profile_and_changes_with_either(self):
        with restored_parameters():
            base = driver.design_request_identity()
            self.assertEqual(base["schema"], "rc_smrf_candidate_v11_risk_basis_profile")
            self.assertEqual((base["inputs"]["ASCE_RISK_CATEGORY"], base["inputs"]["ASCE_IE"]), ("III", 1.25))
            self.assertEqual(base["policy"]["demands"]["risk_category"], "III")
            self.assertEqual(base["policy"]["model_profile"]["id"], ap.UNPROFILED)
            self.assertEqual(base["policy"]["model_profile"]["settings"]["JOINT_MODEL"], sp.JOINT_MODEL)
            profile = ap.apply_profile(V2)
            profiled = driver.design_request_identity()
            self.assertEqual(profiled["policy"]["model_profile"]["id"], V2)
            self.assertEqual(profiled["policy"]["model_profile"]["sha256"], profile["sha256"])
            self.assertEqual(profiled["policy"]["model_profile"]["settings"]["JOINT_MODEL"], "rigid_centerline")
            self.assertNotEqual(base["sha256"], profiled["sha256"])
            self.assertNotEqual(verify.methodology_sha256(base), verify.methodology_sha256(profiled))
        with restored_parameters():
            sp.apply_risk_category("II")
            other = driver.design_request_identity()
            self.assertEqual(other["inputs"]["ASCE_IE"], 1.0)
            self.assertNotEqual(other["sha256"], base["sha256"])
            self.assertNotEqual(verify.methodology_sha256(other), verify.methodology_sha256(base))

    def test_a_cached_design_of_another_basis_or_profile_is_refused(self):
        with tempfile.TemporaryDirectory() as folder, restored_parameters():
            path = Path(folder) / "design.json"
            unprofiled = driver.design_request_identity()
            ap.apply_profile(V2)
            current = driver.design_request_identity()
            for stale in (unprofiled, {**current, "sha256": "0" * 64}):
                path.write_text(json.dumps({"schema_version": driver.DESIGN_SCHEMA_VERSION, "request_identity": stale}), encoding="utf-8")
                before = path.read_bytes()
                with self.assertRaises(RuntimeError) as caught:
                    driver.load_or_create_design(path)
                self.assertIn("different inputs or methodology", str(caught.exception))
                self.assertEqual(path.read_bytes(), before)                                    # preserved, not overwritten
            path.write_text(json.dumps({"schema_version": "rc_smrf_candidate_v10_edition_joint_search", "request_identity": current}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                driver.load_or_create_design(path)                                             # the Risk II schema is not reused

    def test_launcher_identity_reproduces_the_worker_and_restores_the_parameters(self):
        before = (sp.JOINT_MODEL, sp.ANALYSIS_PROFILE_ID, sp.NTHA_HINGE_HISTORY_STRIDE, sp.NUM_BAY_X, sp.BAY_X)
        cfg = verify.probe_config(PROBE_DATE)
        profiled = expected_identity(FIXTURE_CASE, cfg, V2)
        plain = expected_identity(FIXTURE_CASE, cfg)
        self.assertEqual((sp.JOINT_MODEL, sp.ANALYSIS_PROFILE_ID, sp.NTHA_HINGE_HISTORY_STRIDE, sp.NUM_BAY_X, sp.BAY_X), before)
        self.assertEqual(profiled["policy"]["model_profile"]["id"], V2)
        self.assertEqual(plain["policy"]["model_profile"]["id"], ap.UNPROFILED)
        self.assertNotEqual(profiled["sha256"], plain["sha256"])
        self.assertEqual(profiled["inputs"]["BAY_X"], 180.0)
        # a fresh interpreter configured the way a worker configures itself builds the same identity
        code = ("import json, sys; sys.path.insert(0, '.'); sys.path.insert(0, 'Data_Generation');"
                "from Design import Verify_Designs as v; from Design import Design_Driver as d; import Structure_Parameters as sp;"
                f"case = json.loads({json.dumps(json.dumps(FIXTURE_CASE))});"
                f"v.configure_case(case, {V2!r});"
                f"i = d.design_request_identity(v.probe_config({PROBE_DATE!r}));"
                "print(json.dumps({'sha256': i['sha256'], 'ie': sp.ASCE_IE, 'risk': sp.ASCE_RISK_CATEGORY, 'joint': sp.JOINT_MODEL,"
                " 'stride': sp.NTHA_HINGE_HISTORY_STRIDE, 'profile': sp.ANALYSIS_PROFILE_ID, 'policy_risk': i['policy']['demands']['risk_category']}))")
        out = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(RC_DIR), capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        worker = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(worker, {"sha256": profiled["sha256"], "ie": 1.25, "risk": "III", "joint": "rigid_centerline", "stride": 1,
                                  "profile": V2, "policy_risk": "III"})


class ProfileAndTopology(unittest.TestCase):
    def test_profile_sets_only_its_settings_and_requires_its_state(self):
        with restored_parameters():
            identity = ap.apply_profile(V2)
            self.assertEqual((sp.ELEMENT_FORMULATION, sp.IMK_MATERIAL_TYPE, sp.IMK_APPLY_TO_BEAMS, sp.IMK_APPLY_TO_COLUMNS, sp.JOINT_MODEL),
                             ("imk", "IMKPeakOriented", True, True, "rigid_centerline"))
            self.assertEqual(sp.NTHA_HINGE_HISTORY_STRIDE, 1)
            self.assertEqual(sp.IMK_ENERGY_MAPPING_MODE, "legacy_unmapped")                     # required, not changed
            self.assertEqual(identity["state"]["IMK_CYCLIC_CALIBRATION_STATUS"], "provisional_not_experimentally_calibrated")
            self.assertFalse(identity["joint_springs"]); self.assertFalse(identity["explicit_face_slip_interfaces"])
            self.assertIn("no clear spans", identity["declarations"]["joint_idealisation"])
            self.assertIn("not isolated plastic flexure", identity["declarations"]["slip"])
            self.assertEqual(ap.configuration_problems(V2), [])
            sp.JOINT_MODEL = "imk_pinching_scissors"                                            # drift away from the profile
            self.assertTrue(ap.configuration_problems(V2))
            with self.assertRaises(ValueError):
                ap.profile_identity(V2)
        with restored_parameters():
            sp.IMK_ENERGY_MAPPING_MODE = "explicit_reference_energy_v1"
            with self.assertRaises(ValueError):
                ap.apply_profile(V2)                                                            # the corrected mapping is not enabled by a profile
        with restored_parameters():
            sp.apply_risk_category("II")
            with self.assertRaises(ValueError):
                ap.apply_profile(V2)
        with self.assertRaises(ValueError):
            ap.apply_profile("no_such_profile")
        self.assertEqual(sp.ANALYSIS_PROFILE_ID, ap.UNPROFILED)

    def test_installed_topology_is_read_from_the_domain_not_from_the_text(self):
        import openseespy.opensees as ops
        from Model import Deformation_Ownership as own
        from Model.Build_Model import build_model
        from Model.IMK_Hinges import hinge_element_tag
        small = dict(NUM_BAY_X=1, NUM_BAY_Y=1, NUM_FLOOR=2)
        try:
            with restored_parameters():
                for key, value in small.items():
                    setattr(sp, key, value)
                sp.NUM_MODES = 6
                ap.apply_profile(V2)
                build_model()
                good = ap.verify_installed_domain(V2)
                self.assertTrue(good["consistent"], good["problems"])
                self.assertEqual(good["members_with_hinges"], 2 * 4 + 2 * 2 + 2 * 2)             # columns, x beams, y beams
                self.assertEqual(good["hinge_spring_elements_in_domain"], 2 * good["members_with_hinges"])
                self.assertEqual((good["joint_spring_elements_in_domain"], good["joint_registry_entries"], good["joint_beam_core_nodes_in_domain"]), (0, 0, 0))
                self.assertEqual(good["face_slip_interfaces_registered"], 0)
                self.assertFalse(good["joint_springs_installed"])
                self.assertEqual(set(good["bar_slip_owner_by_end"]), {"member_hinge"})
                self.assertEqual(set(good["bond_slip_indicator_by_end"]), {"1.0"})
                # a face interface registered in this domain is found, whatever the configuration says
                member = next(iter(good and __import__("Model.IMK_Hinges", fromlist=["hinge_registry"]).hinge_registry()))
                own.register_slip_interface(member, "i", calibration_id="test", status="provisional_diagnostic_only",
                                            element_tag=hinge_element_tag(member, 1))
                flagged = ap.verify_installed_domain(V2)
                self.assertFalse(flagged["consistent"])
                self.assertTrue(any("face slip interface" in p for p in flagged["problems"]))
                with self.assertRaises(RuntimeError):
                    ap.verify_installed_domain(V2, raise_on_failure=True)
                own.reset_slip_interfaces()
                # the repository default joint model installs pinching springs: the same check reports them
                sp.JOINT_MODEL = "imk_pinching_scissors"
                build_model()
                pinching = ap.verify_installed_domain(V2)
                self.assertFalse(pinching["consistent"])
                self.assertGreater(pinching["joint_spring_elements_in_domain"], 0)
                self.assertTrue(pinching["joint_springs_installed"])
                self.assertTrue(any("joint spring" in p for p in pinching["problems"]))
                unprofiled = ap.installed_topology()
                self.assertEqual(unprofiled["joint_spring_elements_in_domain"], pinching["joint_spring_elements_in_domain"])
        finally:
            ops.wipe()
            own.reset_slip_interfaces()


class Manifest(unittest.TestCase):
    def write(self, folder, manifest):
        path = Path(folder) / "screening_plan.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_a_valid_manifest_is_loaded_with_its_cases_fractions_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            loaded = sm.load_manifest(self.write(folder, manifest_dict()))
        self.assertEqual(loaded["profile_id"], V2)
        self.assertEqual([c["case_id"] for c in loaded["cases"]], ["case_0001", "case_0002"])
        self.assertEqual((loaded["cases"][1]["bay_x_width_ft"], loaded["cases"][1]["story_height_ft"]), (18.5, 12.5))   # fractions preserved
        self.assertEqual(loaded["design_basis"]["risk_category"], "III")
        self.assertEqual(loaded["sampling"]["method"], "explicit_manifest_cases_v1")
        self.assertEqual(loaded["sampling"]["manifest_sha256"], loaded["sha256"])
        self.assertEqual((loaded["initial_case_id"], loaded["initial_workers"], loaded["subsequent_max_workers"], loaded["max_section_iter"]),
                         ("case_0001", 1, 2, 10))
        self.assertTrue(verify.plan_sha256(loaded["cases"]))

    def test_contradictions_with_the_running_code_stop_the_launch(self):
        def changed(path, value):
            manifest = manifest_dict()
            target = manifest
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            return manifest
        cases = {
            "risk category": changed(("design_basis", "risk_category"), "II"),
            "importance factor": changed(("design_basis", "importance_factor"), 1.0),
            "increment": changed(("design_basis", "bay_and_story_height_increment_ft"), 1.0),
            "bay range": changed(("design_basis", "bay_x_width_ft"), [10, 15]),
            "joint model": changed(("nonlinear_profile", "joint_model"), "imk_pinching_scissors"),
            "member material": changed(("nonlinear_profile", "member_material"), "IMKBilin"),
            "face interfaces": changed(("nonlinear_profile", "explicit_face_slip_interfaces"), True),
            "unknown profile": changed(("profile_id",), "v9"),
            "schema": changed(("schema",), "other"),
            "off the grid": changed(("cases", 1, "bay_x_width_ft"), 18.25),
            "outside the range": changed(("cases", 1, "bay_y_width_ft"), 31),
            "too many floors": changed(("cases", 0, "num_floor"), 10),
            "fractional count": changed(("cases", 0, "num_bay_x"), 2.0),
            "unknown site": changed(("cases", 0, "seismic_site"), "sdc_z"),
            "duplicate id": changed(("cases", 1, "case_id"), "case_0001"),
            "unsafe name": changed(("cases", 1, "geometry_name"), "a b/c"),
            "initial case": changed(("execution", "initial_case_id"), "case_0042"),
            "budget": changed(("execution", "max_section_iter"), 0),
        }
        for label, manifest in cases.items():
            with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError, msg=label):
                sm.load_manifest(self.write(folder, manifest))

    def test_a_manifest_run_launches_nothing_by_default_and_writes_nothing_when_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write(folder, manifest_dict())
            root = Path(folder) / "root"
            base = ["--manifest", str(path), "--output-root", str(root), "--probe-assertions", "--probe-date", PROBE_DATE]
            for extra, expected in (([], "launches nothing by default"), (["--case-ids", "case_0002"], "runs first and alone"),
                                    (["--case-ids", "case_0001", "--workers", "2"], "exceeds the manifest's limit"),
                                    (["--remaining"], "has no result yet"),
                                    (["--case-ids", "case_0001", "--max-section-iter", "20"], "contradicts the manifest's budget"),
                                    (["--case-ids", "case_0001", "--profile", "other"], "contradicts the manifest's profile"),
                                    (["--case-ids", "case_0001", "--sites", "sdc_c"], "names its cases and sites")):
                with self.assertRaises(SystemExit) as caught:
                    verify.main(base + extra)
                self.assertIn(expected, str(caught.exception), extra)
            self.assertFalse((root / "plan.json").exists())
            self.assertEqual([p.name for p in root.iterdir() if p.name != ".launcher.lease"] if root.exists() else [], [])
            self.assertEqual(verify.main(base + ["--plan-only"]), 0)

    def test_the_four_statuses_are_separate_and_probe_never_accepts(self):
        record = {"qualification": {"accepted": True, "counts": {"pass": 3, "fail": 0, "not_evaluated": 0}, "failed": [], "not_evaluated": []},
                  "search": {"stop_reason": "candidate_screen_passed"},
                  "request_identity": {"policy": {"verification": {"strength_model_verified": True, "story_strength_model_verified": False,
                                                                   "asserted_by": "PROBE"}}}}
        probe = sm.status_fields(record, True, PROBE_DATE, ("design",))
        self.assertEqual(probe["numerical_completion"]["design"], "completed")
        self.assertTrue(probe["design_checks"]["accepted_by_checklist"])                        # the checklist can pass ...
        self.assertFalse(probe["independent_verification"]["verified_by_a_person"])             # ... a PROBE verifies nothing ...
        self.assertFalse(probe["production_acceptance"]["accepted"])                            # ... and nothing is accepted for production
        self.assertFalse(probe["production_acceptance"]["generation_release_ready"])
        self.assertEqual(probe["numerical_completion"]["gravity_modal"], "not_run")
        exhausted = sm.status_fields({**record, "search": {"stop_reason": "iteration_budget_exhausted"},
                                      "qualification": {"accepted": False, "counts": {"fail": 2}, "failed": ["joint_shear:1", "demands.story_drift:2/x"],
                                                        "not_evaluated": ["demands.torsional_irregularity"]}}, False, None, ("design", "gravity_modal"),
                                     {"gravity_modal": {"status": "error"}})
        self.assertEqual(exhausted["numerical_completion"]["design"], "completed")              # an exhausted budget is a completed search
        self.assertTrue(exhausted["numerical_completion"]["design_search_budget_exhausted"])
        self.assertEqual(exhausted["numerical_completion"]["gravity_modal"], "error")
        self.assertEqual(exhausted["design_checks"]["failed"], {"joint": ["joint_shear"], "drift_and_stability": ["demands.story_drift"]})
        self.assertFalse(exhausted["production_acceptance"]["accepted"])
        self.assertEqual(sm.group_ids(["beam.bars_thread_column", "scwb.joint", "slab.strip", "x.y"]),
                         {"strong_column_weak_beam": ["scwb.joint"], "anchorage_and_bar_fit": ["beam.bars_thread_column"],
                          "slab_and_floor": ["slab.strip"], "other": ["x.y"]})


class WorkerFixture(unittest.TestCase):
    """The runner's own worker entry point on a tiny off-population frame, in a fresh interpreter."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.case_dir = cls.root / FIXTURE_CASE["case_id"]
        command = verify._worker_command(sys.executable, FIXTURE_CASE, cls.case_dir, True, PROBE_DATE, 1, V2, ("design",),
                                         ["--attempt-id", "fixture"])
        cls.proc = subprocess.run(command, cwd=str(RC_DIR), capture_output=True, text=True, timeout=1800)
        cls.result = verify.saved_result(cls.case_dir)
        cls.record = json.loads((cls.case_dir / "design.json").read_text(encoding="utf-8")) if (cls.case_dir / "design.json").exists() else None

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_risk_three_and_the_profile_reach_the_worker_the_record_and_its_identity(self):
        self.assertIsNotNone(self.result, self.proc.stderr[-2000:])
        self.assertEqual(self.result["status"], "designed", self.result.get("error"))
        self.assertEqual(self.result["configured_basis"], {"risk_category": "III", "importance_factor": 1.25, "joint_model": "rigid_centerline",
                                                           "member_material": "IMKPeakOriented", "analysis_profile_id": V2})
        seismic = self.record["seismic"]
        self.assertEqual((seismic["risk_category"], seismic["importance_factor"]), ("III", 1.25))
        self.assertEqual(seismic["design_basis"]["allowable_story_drift_ratio"], 0.015)
        assumptions = self.record["drift_screen"]["assumptions"]
        self.assertEqual((assumptions["risk_category"], assumptions["importance_factor"], assumptions["base_drift_limit_ratio"]), ("III", 1.25, 0.015))
        self.assertAlmostEqual(assumptions["effective_drift_limit_ratio"], 0.015 / 1.3, places=15)
        self.assertFalse(assumptions["low_rise_allowance_asserted"])
        for row in self.record["drift_screen"]["stories"]:
            self.assertAlmostEqual(row["allowable_drift_ratio"], 0.015 / 1.3, places=15)
        self.assertEqual(self.record["demand_basis"]["drift"]["risk_category"], "III")
        identity = self.record["request_identity"]
        self.assertEqual((identity["inputs"]["ASCE_RISK_CATEGORY"], identity["inputs"]["ASCE_IE"]), ("III", 1.25))
        self.assertEqual(identity["policy"]["demands"]["risk_category"], "III")
        self.assertEqual(identity["policy"]["model_profile"]["id"], V2)
        self.assertEqual(self.record["schema_version"], driver.DESIGN_SCHEMA_VERSION)
        # base shear coefficient: SDS / (R / Ie) for this short frame, with Ie = 1.25
        overview = self.result["overview"]["period_and_base_shear"]
        self.assertAlmostEqual(overview["cs_base_shear_over_weight"], 0.75 / (8.0 / 1.25), places=9)
        check = [c for c in self.record["qualification"]["checks"] if c["id"] == "demands.risk_basis"]
        self.assertEqual([c["status"] for c in check], ["pass"])

    def test_the_result_row_keeps_the_four_statuses_apart(self):
        completion = self.result["numerical_completion"]
        self.assertEqual(completion["design"], "completed")
        self.assertEqual(completion["design_search_stop_reason"], self.record["search"]["stop_reason"])
        self.assertEqual(self.result["design_checks"]["accepted_by_checklist"], self.record["qualification"]["accepted"])
        self.assertIn("PROBE", self.result["independent_verification"]["mode"])
        self.assertFalse(self.result["independent_verification"]["verified_by_a_person"])
        self.assertFalse(self.result["independent_verification"]["story_strength_model_verified"])
        self.assertFalse(self.result["production_acceptance"]["accepted"])
        self.assertEqual(self.result["max_section_iter"], 1)
        self.assertEqual(self.record["search"]["max_section_iter"], 1)
        self.assertGreater(self.result["elapsed_s"], 0.0)
        self.assertEqual(self.result["overview"]["geometry"]["bay_x_ft"], 15.0)

    def test_the_cached_design_is_reused_only_under_its_own_profile_and_probe_date(self):
        same = verify.validate_saved(FIXTURE_CASE, self.case_dir, True, PROBE_DATE, V2)
        self.assertEqual(same["status"], "designed", same.get("error"))
        for label, args in (("no profile", (True, PROBE_DATE, None)), ("other date", (True, "2026-09-30", V2))):
            refused = verify.validate_saved(FIXTURE_CASE, self.case_dir, *args)
            self.assertEqual(refused["status"], "error", label)
            self.assertIn("Untrusted cached design", refused.get("error", "")) if label == "no profile" else None
        self.assertTrue((self.case_dir / "design.json").exists())                               # refused, never removed

    def test_a_later_stage_runs_on_the_existing_design_without_rewriting_it(self):
        from Design.Verification_Integrity import file_sha256
        design_before, result_before = file_sha256(self.case_dir / "design.json"), file_sha256(self.case_dir / "result.json")
        row, cached = verify._launch(sys.executable, FIXTURE_CASE, self.case_dir, True, PROBE_DATE, max_section_iter=1, profile_id=V2,
                                     stages=("design", "gravity_modal"))
        self.assertTrue(cached)
        self.assertEqual(file_sha256(self.case_dir / "design.json"), design_before)
        self.assertEqual(file_sha256(self.case_dir / "result.json"), result_before)
        stage = json.loads((self.case_dir / "gravity_modal.json").read_text(encoding="utf-8"))
        self.assertEqual(stage["status"], "completed", stage.get("error"))
        self.assertTrue(stage["all_checks_pass"], stage["checks"])
        topology = stage["installed_topology"]
        self.assertTrue(topology["consistent"], topology["problems"])
        self.assertEqual(topology["joint_spring_elements_in_domain"], 0)
        self.assertEqual(topology["face_slip_interfaces_registered"], 0)
        self.assertEqual(topology["hinge_spring_elements_in_domain"], 2 * topology["members_with_hinges"])
        export = stage["hinge_export_check"]
        self.assertEqual(export["springs_in_table"], 2 * export["hinges_in_domain"])
        self.assertEqual(export["missing_values_at_gravity"], 0)
        self.assertTrue(export["all_gravity_values_finite"])
        self.assertEqual(export["hinge_moment_rotation_rows"], 0)                               # no ground motion: no transient rows
        self.assertTrue((self.case_dir / "gravity_modal" / "hinge_springs.csv").exists())
        self.assertEqual(row["numerical_completion"]["gravity_modal"], "completed")
        self.assertEqual(row["stage_results"]["gravity_modal"]["status"], "completed")
        # the summary of the root reports the stage and the separate statuses
        plan = {"mode": "manifest", "profile_id": V2, "design_basis": sp.seismic_design_basis(), "population_basis": gen.POPULATION_BASIS,
                "sampling": {"method": "explicit_manifest_cases_v1"}}
        lines = verify.summarize([verify.saved_result(self.case_dir)], self.root, True, [FIXTURE_CASE], PROBE_DATE, V2, plan)
        text = "\n".join(lines)
        self.assertIn("Status of each case", text); self.assertIn("PROBE (unverified)", text); self.assertIn("not accepted", text)
        self.assertIn("Gravity and modal stage", text)
        summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["counts"]["production_accepted"], 0)
        self.assertEqual(summary["cases"][0]["stage_results"]["gravity_modal"]["status"], "completed")
        self.assertTrue((self.case_dir / "case_summary.md").exists())

    def test_fixed_design_install_requires_the_records_own_basis_and_profile(self):
        from Analysis.Fixed_Design_Diagnostics import install_record
        with restored_parameters():
            _overrides, check = install_record(copy.deepcopy(self.record), "fixture", V2)
            self.assertTrue(check["risk_basis"]["match"]); self.assertTrue(check["analysis_profile"]["match"])
            self.assertEqual((sp.JOINT_MODEL, sp.ANALYSIS_PROFILE_ID), ("rigid_centerline", V2))
        relabelled = copy.deepcopy(self.record)
        relabelled["seismic"].update(risk_category="II", importance_factor=1.0)
        legacy = copy.deepcopy(self.record)
        for key in ("risk_category", "importance_factor", "design_basis"):
            legacy["seismic"].pop(key)
        other_profile = copy.deepcopy(self.record)
        other_profile["request_identity"]["policy"]["model_profile"]["id"] = ap.UNPROFILED
        for label, record in (("relabelled Risk II", relabelled), ("legacy without a basis", legacy), ("designed for another profile", other_profile)):
            with restored_parameters(), self.assertRaises(RuntimeError, msg=label):
                install_record(record, "fixture", V2)
        with restored_parameters():
            _overrides, check = install_record(legacy, "fixture")                               # a legacy diagnostic stays possible, and says so
            self.assertIsNone(check["risk_basis"]["match"])


class FailurePreservation(unittest.TestCase):
    def test_a_worker_that_fails_leaves_a_complete_row_and_no_accepted_design(self):
        bad = {**FIXTURE_CASE, "case_id": "case_9003", "seismic_site": "sdc_unknown"}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / bad["case_id"]
            row, cached = verify._launch(sys.executable, bad, out, True, PROBE_DATE, max_section_iter=1, profile_id=V2)
            self.assertFalse(cached)
            self.assertEqual(row["status"], "error")
            self.assertIn("Unknown seismic site", row["error"])
            self.assertTrue(row.get("traceback"))
            self.assertEqual(row["numerical_completion"]["design"], "error")
            self.assertFalse(row["design_checks"]["accepted_by_checklist"])
            self.assertFalse(row["production_acceptance"]["accepted"])
            self.assertGreaterEqual(row["elapsed_s"], 0.0)
            self.assertEqual(row["profile_id"], V2)
            self.assertFalse((out / "design.json").exists())
            self.assertTrue((out / "result.json").exists()); self.assertTrue((out / "stderr.txt").exists())
            lines = verify.summarize([row], Path(folder), True, [bad], PROBE_DATE, V2, {"mode": "manifest"})
            self.assertIn("1 errors", "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
