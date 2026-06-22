#!/usr/bin/env python3
"""Audit Step 2.2 Assets44 scope and validation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ALLOWED_DEFERRED = 2
IN_SCOPE = 44


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts/retargeting_v3_step2_assets44")
    parser.add_argument("--lock", default="assets/robot_zoo/robot_zoo_lock.json")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    lock_path = Path(args.lock)
    errors: list[str] = []

    if not lock_path.exists():
        raise SystemExit(f"missing lock: {lock_path}")
    lock = json.loads(lock_path.read_text())
    scope = lock.get("scope_decision", {})
    totals = lock.get("totals", {})
    _expect(errors, totals.get("entries") == 46, "lock totals.entries must be 46")
    _expect(errors, totals.get("source_available") == 46, "lock source_available must be 46")
    _expect(errors, totals.get("vendored") == 38, "lock vendored snapshots must be 38")
    _expect(errors, totals.get("fetch_only") == 5, "lock fetch_only count must be 5")
    _expect(errors, totals.get("local_existing") == 1, "lock local_existing count must be 1")
    _expect(errors, totals.get("snapshot_failed") == ALLOWED_DEFERRED, "lock deferred snapshot count must be 2")
    _expect(errors, scope.get("in_scope_source_count") == IN_SCOPE, "scope in_scope_source_count must be 44")
    _expect(errors, scope.get("deferred_snapshot_count") == ALLOWED_DEFERRED, "scope deferred_snapshot_count must be 2")
    _expect(errors, scope.get("blocking_count_for_current_scope") == 0, "scope blockers must be 0")

    snapshot_dirs = [path for path in (lock_path.parent / "snapshots").iterdir() if path.is_dir()]
    _expect(errors, len(snapshot_dirs) == 38, "committed snapshot directory count must be 38")
    for path in snapshot_dirs:
        source = path / "SOURCE.json"
        _expect(errors, source.exists(), f"snapshot missing SOURCE.json: {path.name}")
        if source.exists():
            payload = json.loads(source.read_text())
            for key in ("upstream_repository", "upstream_ref", "source_sha256", "snapshot_sha256", "license_files"):
                _expect(errors, bool(payload.get(key)), f"snapshot {path.name} missing {key}")

    if not artifact_dir.exists():
        errors.append(f"missing artifact dir: {artifact_dir}")
    else:
        summary_path = artifact_dir / "summary.json"
        _expect(errors, summary_path.exists(), "missing summary.json")
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            _audit_summary(errors, summary)
        environment_path = artifact_dir / "environment.json"
        _expect(errors, environment_path.exists(), "missing environment.json")
        if environment_path.exists():
            _audit_environment(errors, json.loads(environment_path.read_text()))
        required = [
            "environment.json",
            "commands.txt",
            "scope.json",
            "source_inventory.json",
            "load_matrix.json",
            "semantic_matrix.json",
            "deterministic_rerun.json",
            "cross_format.json",
            "deferred_snapshots.json",
        ]
        for name in required:
            _expect(errors, (artifact_dir / name).exists(), f"missing artifact {name}")
        required_test_results = [
            "acceptance_ledger.json",
            "test_results/pytest.txt",
            "test_results/junit.xml",
            "test_results/coverage.json",
        ]
        for name in required_test_results:
            _expect(errors, (artifact_dir / name).exists(), f"missing reproducibility artifact {name}")
        _audit_no_absolute_paths(errors, artifact_dir)

    if errors:
        raise SystemExit("Step 2.2 Assets44 audit FAIL\n" + "\n".join(f"- {error}" for error in errors))
    print("Step 2.2 Assets44 audit PASS")


def _audit_summary(errors: list[str], summary: dict) -> None:
    expected = {
        "manifest_total": 46,
        "in_scope_total": 44,
        "deferred_snapshot_count": 2,
        "vendored_snapshot_count": 38,
        "fetch_only_cached_count": 5,
        "project_local_count": 1,
        "source_unavailable": 0,
        "model_load_failed": 0,
        "semantic_failed": 0,
    }
    for key, value in expected.items():
        _expect(errors, summary.get(key) == value, f"summary {key} must be {value}, got {summary.get(key)!r}")
    _expect(errors, summary.get("load_passed_in_scope") == 44, "all 44 in-scope models must load")
    _expect(errors, summary.get("semantic_passed_in_scope") == 44, "all 44 in-scope semantics must pass or be honestly classified")
    _expect(errors, summary.get("deterministic_compared") == summary.get("deterministic_matched"), "deterministic rerun must match")
    _expect(errors, summary.get("deterministic_compared", 0) > 0, "deterministic rerun must compare at least one model")
    _expect(errors, summary.get("negative_control_passed", 0) > 0, "negative controls must be represented")
    _expect(errors, summary.get("partial_passed", 0) > 0, "structured partial humanoids must be represented")


def _audit_environment(errors: list[str], environment: dict) -> None:
    _expect(errors, bool(environment.get("source_code_commit")), "environment source_code_commit is required")
    _expect(errors, environment.get("git_status_short") == "", "environment git_status_short must be clean")
    _expect(
        errors,
        environment.get("source_code_commit_is_artifact_commit_ancestor") is True,
        "environment must record source_code_commit ancestry",
    )


def _audit_no_absolute_paths(errors: list[str], artifact_dir: Path) -> None:
    forbidden = ("/mnt/", "/home/", "/Users/", "/private/var/", "/tmp/")
    allowed_suffixes = {".xml"}
    for path in artifact_dir.rglob("*"):
        if not path.is_file() or path.suffix in allowed_suffixes:
            continue
        try:
            text = path.read_text(errors="ignore")
        except UnicodeDecodeError:
            continue
        for token in forbidden:
            if token in text:
                errors.append(f"absolute path leak {token!r} in {path}")
                break


def _expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


if __name__ == "__main__":
    main()
