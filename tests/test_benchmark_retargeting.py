import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from soma_retargeter.tools.benchmark_retargeting import _legacy_runtime_retargeter_config, main


class TestBenchmarkRetargeting(unittest.TestCase):
    def test_legacy_runtime_config_does_not_use_compiled_profile(self):
        cfg = _legacy_runtime_retargeter_config("roboparty_rpo")
        self.assertNotIn("compiled_retarget_profile", cfg)
        self.assertNotIn("direction_tasks", cfg)
        self.assertNotIn("pole_vector_tasks", cfg)
        self.assertEqual(cfg["ik_map"]["Hips"]["t_weight"], 10.0)
        self.assertEqual(cfg["ik_map"]["Hips"]["r_weight"], 2.0)
        self.assertEqual(cfg["ik_map"]["Chest"]["t_weight"], 0.5)
        self.assertEqual(cfg["ik_map"]["Chest"]["r_weight"], 0.5)

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
            self.assertIn("collision_summary", per_robot)
            self.assertIn("proxy_count", per_robot["collision_summary"])
            self.assertIn("pair_count", per_robot["collision_summary"])
            self.assertIn("root_ground_summary", per_robot)
            self.assertIn("ground_height_source", per_robot["root_ground_summary"])
            self.assertIn("horizontal_scale", per_robot["root_ground_summary"])
            self.assertEqual(per_robot["metrics"]["hand_position_rmse"]["status"], "not_run")
            self.assertEqual(per_robot["metrics"]["hand_position_rmse"].get("reason"), None)

            with (out / "benchmark_frames.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertEqual(rows[0]["robot"], "roboparty_rpo")

    def test_benchmark_records_runtime_motion_metrics_when_motions_are_requested(self):
        runtime_payload = {
            "status": "ok",
            "runtime_seconds": {"motion_runtime": 1.25, "motion_count": 1},
            "motions": [
                {
                    "motion": "/tmp/fixture.bvh",
                    "frames": 4,
                    "sample_rate": 60.0,
                    "metrics": {
                        "velocity_p95": {"status": "ok", "value": 2.5},
                        "penetration": {"status": "ok", "value": 0.0},
                    },
                }
            ],
            "metrics": {
                "velocity_p95": {"status": "ok", "value": 2.5, "motion_count": 1},
                "penetration": {"status": "ok", "value": 0.0, "motion_count": 1},
                "fallback_counts": {"status": "ok", "pole_vector": [0]},
            },
        }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            motion_dir = root / "motions"
            motion_dir.mkdir()
            (motion_dir / "fixture.bvh").write_text("HIERARCHY\n", encoding="utf-8")
            out = root / "bench"

            with mock.patch("soma_retargeter.tools.benchmark_retargeting._run_runtime_benchmark", return_value=runtime_payload):
                rc = main([
                    "--robots",
                    "roboparty_rpo",
                    "--motions",
                    str(motion_dir),
                    "--output",
                    str(out),
                ])

            self.assertEqual(rc, 0)
            per_robot = json.loads((out / "per_robot" / "roboparty_rpo.json").read_text())
            self.assertEqual(per_robot["motion_benchmark"]["status"], "ok")
            self.assertIn("legacy", per_robot["compare_results"])
            self.assertIn("v2", per_robot["compare_results"])
            self.assertEqual(per_robot["metrics"]["velocity_p95"]["status"], "ok")
            self.assertEqual(per_robot["metrics"]["velocity_p95"]["value"], 2.5)
            self.assertEqual(per_robot["metrics"]["runtime_seconds"]["motion_runtime"], 1.25)

            with (out / "benchmark_frames.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            runtime_rows = [row for row in rows if row["frame"] == "0" and row["metric"] == "velocity_p95"]
            self.assertEqual({row["compare_mode"] for row in runtime_rows}, {"legacy", "v2"})
            self.assertTrue(all(row["value"] == "2.5" for row in runtime_rows))

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
