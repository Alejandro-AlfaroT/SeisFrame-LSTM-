"""The V2 population after the 2026-10-14 decisions: bays 3-6 and height over least plan width at most 2.25."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Data_Generation"))

from Generate_Parameterized_Dataset import (GEOMETRY_INCREMENT_FT, MAX_HEIGHT_TO_WIDTH_RATIO, POPULATION_BASIS, RANGES,  # noqa: E402
                                            SAMPLING_METHOD, SEISMIC_SITES, candidate_geometries, geometry_admissible,
                                            height_to_width_ratio, sampling_provenance)
from Design.Screening_Manifest import _problem_list  # noqa: E402


def manifest_problems(manifest):
    return _problem_list(manifest, RANGES, SEISMIC_SITES, GEOMETRY_INCREMENT_FT)


class Population(unittest.TestCase):
    def test_ranges_and_cap(self):
        self.assertEqual(RANGES["num_bay_x"], (3, 4, 5, 6))
        self.assertEqual(RANGES["num_bay_y"], (3, 4, 5, 6))
        self.assertEqual(MAX_HEIGHT_TO_WIDTH_RATIO, 2.25)
        self.assertIn("bays3_6", POPULATION_BASIS)
        self.assertIn("aspect_le_2p25", POPULATION_BASIS)

    def test_admissibility(self):
        self.assertAlmostEqual(height_to_width_ratio(3, 3, 9, 18.0, 18.0, 18.0), 162.0 / 54.0)
        self.assertFalse(geometry_admissible({"num_bay_x": 3, "num_bay_y": 3, "num_floor": 9, "story_height_ft": 18.0,
                                              "bay_x_width_ft": 18.0, "bay_y_width_ft": 18.0}))
        self.assertTrue(geometry_admissible({"num_bay_x": 3, "num_bay_y": 3, "num_floor": 4, "story_height_ft": 12.0,
                                             "bay_x_width_ft": 18.0, "bay_y_width_ft": 18.0}))
        self.assertFalse(geometry_admissible((2, 3, 4, 12.0, 18.0, 18.0)))          # two bays: not a member
        self.assertTrue(geometry_admissible((6, 6, 9, 18.0, 30.0, 30.0)))

    def test_candidate_pool_respects_the_cap_and_records_it(self):
        pool = candidate_geometries(20261004, SAMPLING_METHOD)
        self.assertTrue(all(geometry_admissible(g) for g in pool))
        self.assertTrue(all(g[0] >= 3 and g[1] >= 3 for g in pool))
        worst = max(height_to_width_ratio(*g) for g in pool)
        self.assertLessEqual(worst, MAX_HEIGHT_TO_WIDTH_RATIO + 1e-12)
        provenance = sampling_provenance(20261004, include_source=False)
        self.assertEqual(provenance["max_height_to_width_ratio"], MAX_HEIGHT_TO_WIDTH_RATIO)
        self.assertEqual(provenance["pool_size"], len(pool))
        self.assertLess(provenance["pool_size"], provenance["cartesian_size"])
        self.assertEqual(provenance["cartesian_size"], 4 * 4 * 6 * 13 * 25 * 25)
        inadmissible = sum(1 for g in product(*RANGES.values()) if not geometry_admissible(g))
        self.assertEqual(provenance["cartesian_size"] - provenance["pool_size"], inadmissible)


class ManifestValidation(unittest.TestCase):
    def manifest(self, case):
        return {"schema": "design_screening_manifest_v1", "profile_id": "v2_nonlinear_flexure_screening_v1",
                "design_basis": {"bay_and_story_height_increment_ft": 0.5, "risk_category": "III", "importance_factor": 1.25,
                                 **{key: [min(v), max(v)] for key, v in RANGES.items()},
                                 "max_height_to_width_ratio": MAX_HEIGHT_TO_WIDTH_RATIO},
                "execution": {"initial_case_id": case["case_id"], "initial_workers": 1, "subsequent_max_workers": 2, "max_section_iter": 16},
                "stages": ["design"], "cases": [case]}

    def test_a_slender_or_two_bay_case_is_refused(self):
        slender = {"case_id": "case_0001", "geometry_name": "slender", "num_bay_x": 3, "num_bay_y": 3, "num_floor": 9,
                   "story_height_ft": 18.0, "bay_x_width_ft": 18.0, "bay_y_width_ft": 18.0, "seismic_site": "sdc_d_high"}
        problems = manifest_problems(self.manifest(slender))
        self.assertTrue(any("exceeds the population cap" in p for p in problems), problems)
        two_bay = {**slender, "num_bay_x": 2, "num_floor": 4, "story_height_ft": 12.0}
        problems = manifest_problems(self.manifest(two_bay))
        self.assertTrue(any("num_bay_x = 2 is outside the population" in p for p in problems), problems)
        fine = {**slender, "num_floor": 5, "story_height_ft": 12.0}
        self.assertEqual([p for p in manifest_problems(self.manifest(fine)) if "cap" in p or "outside" in p], [])


class LatinHypercubeTool(unittest.TestCase):
    def test_pilot_points_are_members_of_the_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pilot"
            result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(ROOT / "tools" / "build_lhs_pilot.py"),
                                     "--count", "40", "--seed", "20261014", "--shards", "2", "--name", "t", "--out", str(out)],
                                    cwd=str(ROOT), capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            plan = json.loads((out / "screening_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(len(plan["cases"]), 40)
            for case in plan["cases"]:
                self.assertTrue(geometry_admissible({key: case[key] for key in RANGES}), case)
            self.assertEqual(plan["design_basis"]["max_height_to_width_ratio"], MAX_HEIGHT_TO_WIDTH_RATIO)
            self.assertIn("redraws_for_inadmissible_structures", plan["sampling"])
            self.assertEqual(manifest_problems(plan), [])


if __name__ == "__main__":
    unittest.main()
