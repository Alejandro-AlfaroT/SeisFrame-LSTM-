"""The ``figures`` verification stage (2026-10-02): the structure in 3D, the member cross sections with their
reinforcement, and the joint detailing with the joint shear table, drawn from the design record alone."""
import contextlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Structure_Parameters as sp                                              # noqa: E402
from Design import Design_Driver as driver                                     # noqa: E402
from Design import Screening_Manifest as manifest                              # noqa: E402
from Design import Verify_Designs as verify                                    # noqa: E402
from Design.Design_Figures import FILES, draw_design_figures                   # noqa: E402

PROBE_DATE = "2026-10-02"


@contextlib.contextmanager
def restored_parameters():
    snapshot = dict(vars(sp))
    try:
        yield
    finally:
        for key in set(vars(sp)) - set(snapshot):
            delattr(sp, key)
        vars(sp).update(snapshot)


class FiguresStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """One small frame designed once (PROBE assertions, two section iterations) for every test here."""
        with restored_parameters():
            sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR = 1, 1, 2
            sp.BAY_X = sp.BAY_Y = 180.0
            sp.STORY_H = 144.0
            sp.NUM_MODES = 3
            sp.apply_seismic_site("sdc_d_low")
            cls.record = driver.design_structure(cfg=verify.probe_config(PROBE_DATE), max_section_iter=2, max_steel_iter=2, verbose=False)

    def test_the_stage_is_listed_after_gravity_modal_and_has_a_runner(self):
        self.assertEqual(manifest.STAGES, ("design", "gravity_modal", "figures"))
        self.assertEqual(set(manifest.STAGE_RUNNERS), {"gravity_modal", "figures"})

    def test_the_three_figures_are_drawn_from_the_record_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = manifest.figures_stage(self.record, None, tmp)
            self.assertEqual(result["status"], "completed", result.get("error"))
            self.assertEqual([Path(f).name for f in result["files"]], list(FILES))
            for name in FILES:
                path = Path(tmp) / "figures" / name
                self.assertTrue(path.exists(), name)
                self.assertGreater(path.stat().st_size, 20_000, name)
            self.assertEqual(result["notes"], [])
            self.assertLess(result["elapsed_s"], 60.0)
            # the stage result is JSON (it is written beside the design and summarised)
            json.dumps(result)

    def test_a_grouped_record_is_refused_and_the_stage_reports_it(self):
        with self.assertRaisesRegex(ValueError, "uniform design records only"):
            draw_design_figures({"design_mode": "grouped", "member_groups": {}}, tempfile.gettempdir())
        with tempfile.TemporaryDirectory() as tmp:
            result = manifest.figures_stage({"design_mode": "grouped", "member_groups": {}}, None, tmp)
            self.assertEqual(result["status"], "error")
            self.assertIn("uniform design records only", result["error"])

    def test_the_status_fields_name_the_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = manifest.figures_stage(self.record, None, tmp)
            fields = manifest.status_fields(self.record, True, PROBE_DATE, ("design", "figures"), {"figures": stage})
            self.assertEqual(fields["numerical_completion"]["figures"], "completed")
            self.assertEqual(fields["numerical_completion"]["gravity_modal"], "not_run")


if __name__ == "__main__":
    unittest.main()
