"""A saved evaluation record reloads as the record that was qualified (2026-11-07).

Grouped_Record._canonical keeps the insertion order of every mapping and the record is qualified in that form. Evidence
texts that embed mappings (the slab strip demand_location names its moment and shear locations) are rebuilt on replay
from the reloaded mappings, so the file must keep the same order: save_evaluation_record writes it unsorted.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from Design import Grouped_Record as gr  # noqa: E402


def _record():
    location = {"case_id": "1.2D+1.6L_all", "x_in": 126.74, "element": 5429, "gauss_point": 3,
                "recovery": {"method": "mitc4", "witness": [1, 2]}}
    strip = {"panel_id": "panel_x1_y1", "moment_location": location}
    strip["demand_location"] = f"{strip['panel_id']}: moment at {strip['moment_location']}"
    return {"zeta": 1, "alpha": {"b": 2, "a": 1}, "strips": [strip],
            "qualification": {"checks": [{"id": "grouped.record_installs", "clause": "Evidence integrity",
                                          "status": "pass"}]}}


class SavedRecordFileOrder(unittest.TestCase):
    def _save_and_load(self, record):
        evaluation = {"evidence": {"state": mock.Mock(policy_id="policy")}}
        with mock.patch.object(gr, "grouped_request_identity", return_value="identity"), \
                mock.patch.object(gr, "build_record_from_evaluation", return_value=record), \
                tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "record.json.gz"
            saved = gr.save_evaluation_record(evaluation, None, 1, path)
            return saved, gr.load_saved_record(path)

    def test_reloaded_mappings_keep_the_canonical_order(self):
        record = gr._canonical(_record())
        _saved, loaded = self._save_and_load(record)
        self.assertEqual(loaded, record)
        self.assertEqual(list(loaded), list(record))
        self.assertEqual(list(loaded["alpha"]), ["b", "a"])
        self.assertEqual(list(loaded["strips"][0]["moment_location"]), list(record["strips"][0]["moment_location"]))

    def test_a_text_built_from_a_reloaded_mapping_reads_the_same(self):
        _saved, loaded = self._save_and_load(gr._canonical(_record()))
        strip = loaded["strips"][0]
        self.assertEqual(strip["demand_location"], f"{strip['panel_id']}: moment at {strip['moment_location']}")

    def test_the_save_reports_the_file_hash_and_the_replay_summary(self):
        saved, _loaded = self._save_and_load(gr._canonical(_record()))
        self.assertEqual(len(saved["sha256"]), 64)
        self.assertTrue(saved["replay"]["replayed"])


if __name__ == "__main__":
    unittest.main()
