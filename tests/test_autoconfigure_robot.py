import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from soma_retargeter.tools import autoconfigure_robot


class TestAutoconfigureRobot(unittest.TestCase):
    def _compiled(self, warnings=None):
        return type(
            "Compiled",
            (),
            {
                "schema_version": 2,
                "compiler_version": "test",
                "robot_fingerprint": "fingerprint",
                "source_config_hash": "config-hash",
                "confidence": 0.95,
                "warnings": warnings or [],
                "tasks": [object()],
                "semantic_sites": {"LeftHand": object()},
                "chains": {"left_arm": object()},
            },
        )()

    def _patch_compile_path(self, compiled):
        return (
            mock.patch("soma_retargeter.tools.autoconfigure_robot.resolve_robot_name", side_effect=lambda name: name),
            mock.patch(
                "soma_retargeter.tools.autoconfigure_robot.get_robot_profile",
                return_value={"retargeter_config": "/tmp/robot_config.json"},
            ),
            mock.patch("soma_retargeter.tools.autoconfigure_robot._load_raw_config", return_value={"ik_map": {}}),
            mock.patch("soma_retargeter.tools.autoconfigure_robot.get_robot_mjcf_path", return_value="/tmp/robot.xml"),
            mock.patch("soma_retargeter.tools.autoconfigure_robot.analyze_mjcf_morphology", return_value=object()),
            mock.patch("soma_retargeter.tools.autoconfigure_robot.compile_retarget_profile", return_value=compiled),
        )

    def test_dry_run_write_report_skips_profile_write(self):
        compiled = self._compiled()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "profile.json"
            patches = self._patch_compile_path(compiled)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], mock.patch(
                "soma_retargeter.tools.autoconfigure_robot.write_profile_json"
            ) as write_profile:
                rc = autoconfigure_robot.main(
                    [
                        "--robot",
                        "testbot",
                        "--dry-run",
                        "--write-report",
                        "--seed",
                        "17",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(rc, 0)
            write_profile.assert_not_called()
            report = json.loads(output.with_name("profile.autoconfig_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["robot"], "testbot")
            self.assertEqual(report["seed"], 17)
            self.assertTrue(report["dry_run"])
            self.assertFalse(report["wrote_profile"])
            self.assertEqual(report["compiled_profile"]["task_count"], 1)

    def test_strict_benchmark_gate_failure_returns_four_and_reports_args(self):
        compiled = self._compiled()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "profile.json"
            patches = self._patch_compile_path(compiled)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], mock.patch(
                "soma_retargeter.tools.autoconfigure_robot.write_profile_json"
            ), mock.patch("soma_retargeter.tools.benchmark_retargeting.main", return_value=4) as benchmark_main:
                rc = autoconfigure_robot.main(
                    [
                        "--robot",
                        "testbot",
                        "--force",
                        "--benchmark",
                        "--strict",
                        "--write-report",
                        "--seed",
                        "5",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(rc, 4)
            benchmark_args = benchmark_main.call_args.args[0]
            self.assertIn("--strict-gates", benchmark_args)
            self.assertIn("--seed", benchmark_args)
            self.assertIn("5", benchmark_args)
            report = json.loads(output.with_name("profile.autoconfig_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "benchmark_gate_failed")
            self.assertEqual(report["benchmark"]["return_code"], 4)
            self.assertIn("--strict-gates", report["benchmark"]["args"])

    def test_unknown_robot_returns_three_and_can_write_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "profile.json"
            with mock.patch("soma_retargeter.tools.autoconfigure_robot.resolve_robot_name", return_value="missing"), mock.patch(
                "soma_retargeter.tools.autoconfigure_robot.get_robot_profile", return_value=None
            ):
                rc = autoconfigure_robot.main(
                    ["--robot", "missing", "--write-report", "--output", str(output)]
                )

            self.assertEqual(rc, 3)
            report = json.loads(output.with_name("profile.autoconfig_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["error"]["error"], "unknown_robot")


if __name__ == "__main__":
    unittest.main()
