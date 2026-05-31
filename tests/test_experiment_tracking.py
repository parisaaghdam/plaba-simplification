from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from app.evaluate import save_experiment_run


class ExperimentTrackingTests(unittest.TestCase):
    def test_save_experiment_run_writes_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            metrics = {"n_samples": 2, "sari": 39.2, "avg_fk_grade": 8.3}
            rows = [
                {
                    "index": 0,
                    "source": "A",
                    "prediction": "B",
                    "reference_count": 2,
                    "fk_grade": 7.9,
                    "accepted": True,
                }
            ]

            paths = save_experiment_run(metrics=metrics, rows=rows, output_dir=out_dir)

            self.assertTrue(Path(paths["json"]).exists())
            self.assertTrue(Path(paths["csv"]).exists())

            payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["metrics"]["n_samples"], 2)
            self.assertEqual(len(payload["rows"]), 1)

            with Path(paths["csv"]).open("r", encoding="utf-8", newline="") as f:
                data = list(csv.DictReader(f))
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["source"], "A")


if __name__ == "__main__":
    unittest.main()
