import csv
import json
import tempfile
import unittest
from pathlib import Path

from soma_retargeter.tools.benchmark_retargeting import main


class TestBenchmarkRetargeting(unittest.TestCase):
    def test_benchmark_writes_required_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "bench"
            rc = main([
                "--robots",
                "roboparty_rpo",
                "--compare",
                "legacy",
                "v2",
                "--output",
                str(out),
            ])

            self.assertEqual(rc, 0)
            for rel in (
                "benchmark_summary.json",
                "benchmark_frames.csv",
                "environment.json",
                "commands.txt",
                "per_robot/roboparty_rpo.json",
            ):
                self.assertTrue((out / rel).exists(), rel)

            summary = json.loads((out / "benchmark_summary.json").read_text())
            self.assertEqual(summary["status"], "ok")
            self.assertIn("task_residual_by_type_priority", summary["metric_names"])
            self.assertEqual(summary["robots"], ["roboparty_rpo"])

            per_robot = json.loads((out / "per_robot" / "roboparty_rpo.json").read_text())
            self.assertEqual(per_robot["profile_schema_version"], 2)
            self.assertIn("task_summary", per_robot)
            self.assertIn("chain_summary", per_robot)
            self.assertEqual(per_robot["metrics"]["hand_position_rmse"]["status"], "not_run")

            with (out / "benchmark_frames.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertEqual(rows[0]["robot"], "roboparty_rpo")

    def test_benchmark_persists_failure_payload(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "bench"
            rc = main([
                "--robots",
                "does_not_exist",
                "--output",
                str(out),
            ])

            self.assertEqual(rc, 1)
            failure = json.loads((out / "failures" / "does_not_exist.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["robot"], "does_not_exist")
            self.assertIn("exception", failure)
            self.assertIn("stack", failure)


if __name__ == "__main__":
    unittest.main()
