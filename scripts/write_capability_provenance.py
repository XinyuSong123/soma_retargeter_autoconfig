#!/usr/bin/env python3
"""Write and validate Step 2.3.1 capability provenance metadata."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ARTIFACT_DIR = Path("artifacts/retargeting_v3_step2_capability")
CAPABILITY_AUDIT_SCRIPT = "scripts/audit_retargeting_v3_capability.py"
CORE_DIFF_PATHS = ("soma_retargeter", "tests", "scripts", ".github")
FORBIDDEN_ABSOLUTE_PATH_TOKENS = ("/mnt/", "/home/", "/Users/", "/private/var/", "/tmp/")
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"
REQUIRED_ARTIFACT_FILES = (
    "test_results/pytest.txt",
    "test_results/junit.xml",
    "test_results/pytest_summary.json",
    "lfs_state.json",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source-commit")
    parser.add_argument("--artifact-commit")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--source-branch")
    parser.add_argument("--source-clean-before", default="true")
    parser.add_argument("--source-clean-after", default="true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-lfs-fsck", action="store_true")
    parser.add_argument("--acceptance-command")
    parser.add_argument("--acceptance-returncode", type=int)
    parser.add_argument("--acceptance-stdout-file")
    parser.add_argument("--acceptance-stderr-file")
    parser.add_argument("--pytest-command")
    parser.add_argument("--pytest-returncode", type=int)
    parser.add_argument("--check", action="store_true", help="validate existing metadata instead of writing it")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = repo_root / artifact_dir

    if args.check:
        failures = validate_artifact_metadata(artifact_dir)
        return _print_failures(failures)

    source_commit = args.source_commit or _git_stdout(repo_root, "rev-parse", "HEAD")
    artifact_commit = args.artifact_commit or _git_stdout(repo_root, "rev-parse", "HEAD")
    source_branch = args.source_branch or _current_branch(repo_root)
    environment = build_environment_metadata(
        repo_root=repo_root,
        source_commit=source_commit,
        artifact_commit=artifact_commit,
        remote=args.remote,
        source_branch=source_branch,
        source_worktree_clean_before_run=_parse_bool(args.source_clean_before),
        source_worktree_clean_after_run=_parse_bool(args.source_clean_after),
        seed=args.seed,
    )
    lfs_state = collect_lfs_state(repo_root=repo_root, run_fsck=not args.skip_lfs_fsck)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json(artifact_dir / "environment.json", environment)
    _write_json(artifact_dir / "lfs_state.json", lfs_state)
    write_commands_file(
        artifact_dir / "commands.txt",
        source_commit=source_commit,
        pytest_command=args.pytest_command,
        acceptance_command=args.acceptance_command,
    )
    if args.acceptance_command is not None:
        ledger = build_acceptance_ledger(
            command=args.acceptance_command,
            returncode=args.acceptance_returncode,
            stdout=_read_text_arg(args.acceptance_stdout_file),
            stderr=_read_text_arg(args.acceptance_stderr_file),
            source_code_commit=source_commit,
        )
        _write_json(artifact_dir / "acceptance_ledger.json", ledger)

    failures: list[str] = []
    failures.extend(validate_environment_metadata(environment))
    failures.extend(validate_lfs_state(lfs_state))
    if args.acceptance_command is not None:
        failures.extend(validate_acceptance_ledger(ledger))
    return _print_failures(failures)


def validate_artifact_metadata(artifact_dir: Path) -> list[str]:
    failures: list[str] = []
    failures.extend(validate_required_artifact_files(artifact_dir))
    failures.extend(validate_environment_metadata(_read_json_if_exists(artifact_dir / "environment.json")))
    failures.extend(validate_lfs_state(_read_json_if_exists(artifact_dir / "lfs_state.json")))
    failures.extend(validate_acceptance_ledger(_read_json_if_exists(artifact_dir / "acceptance_ledger.json")))
    failures.extend(validate_no_absolute_paths(artifact_dir))
    return failures


def validate_environment_metadata(environment: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    source_commit = environment.get("source_code_commit")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        failures.append("environment.source_code_commit must be a 40-character git SHA")
    artifact_commit = environment.get("artifact_commit")
    if not isinstance(artifact_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", artifact_commit):
        failures.append("environment.artifact_commit must be a 40-character git SHA")
    if environment.get("source_code_commit_remote_resolvable") is not True:
        failures.append("environment.source_code_commit_remote_resolvable must be true")
    if environment.get("source_code_commit_is_artifact_commit_ancestor") is not True:
        failures.append("environment.source_code_commit_is_artifact_commit_ancestor must be true")
    if environment.get("source_worktree_clean_before_run") is not True:
        failures.append("environment.source_worktree_clean_before_run must be true")
    if environment.get("source_worktree_clean_after_run") is not True:
        failures.append("environment.source_worktree_clean_after_run must be true")
    core_diff = environment.get("core_diff_after_source_commit")
    if core_diff != []:
        failures.append(f"environment.core_diff_after_source_commit must be [], got {core_diff!r}")
    if environment.get("git_status_short") != "":
        failures.append("environment.git_status_short must be an empty string")
    failures.extend(_absolute_path_failures(environment, "environment"))
    return failures


def validate_lfs_state(state: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if not isinstance(state.get("git_lfs_version"), str) or not state.get("git_lfs_version"):
        failures.append("lfs_state.git_lfs_version missing")
    if state.get("fsck_returncode") != 0:
        failures.append(f"lfs_state.fsck_returncode must be 0, got {state.get('fsck_returncode')!r}")
    pointer_files = state.get("pointer_files_detected")
    if pointer_files != []:
        failures.append(f"lfs_state.pointer_files_detected must be [], got {pointer_files!r}")
    missing_objects = state.get("missing_lfs_objects")
    if missing_objects != []:
        failures.append(f"lfs_state.missing_lfs_objects must be [], got {missing_objects!r}")
    snapshot_count = state.get("materialized_snapshot_count")
    if not isinstance(snapshot_count, int) or snapshot_count <= 0:
        failures.append("lfs_state.materialized_snapshot_count must be a positive integer")
    tracked_paths = state.get("tracked_paths")
    if not isinstance(tracked_paths, list) or not tracked_paths:
        failures.append("lfs_state.tracked_paths must be a non-empty list")
    failures.extend(_absolute_path_failures(state, "lfs_state"))
    return failures


def validate_required_artifact_files(artifact_dir: Path) -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACT_FILES:
        path = artifact_dir / rel
        if not path.is_file():
            failures.append(f"required artifact file missing: {rel}")
    return failures


def validate_acceptance_ledger(ledger: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    command = str(ledger.get("command", ""))
    stdout = str(ledger.get("stdout", ""))
    audit_name = str(ledger.get("audit_name", ""))
    if CAPABILITY_AUDIT_SCRIPT not in command:
        failures.append(f"acceptance_ledger.command must run {CAPABILITY_AUDIT_SCRIPT}")
    if "audit_retargeting_v3_assets44.py" in command or "step2_assets44" in command:
        failures.append("acceptance_ledger.command must not reference stale Step 2.2 Assets44 audit")
    if "capability" not in audit_name:
        failures.append("acceptance_ledger.audit_name must identify the capability audit")
    if "Step 2.3" not in stdout and "Step2.3.1" not in stdout:
        failures.append("acceptance_ledger.stdout must come from the Step 2.3 capability audit")
    if ledger.get("returncode") != 0:
        failures.append(f"acceptance_ledger.returncode must be 0, got {ledger.get('returncode')!r}")
    if ledger.get("status") != "passed":
        failures.append(f"acceptance_ledger.status must be 'passed', got {ledger.get('status')!r}")
    if not ledger.get("source_code_commit"):
        failures.append("acceptance_ledger.source_code_commit missing")
    if not ledger.get("time_utc"):
        failures.append("acceptance_ledger.time_utc missing")
    failures.extend(_absolute_path_failures(ledger, "acceptance_ledger"))
    return failures


def validate_no_absolute_paths(artifact_dir: Path) -> list[str]:
    failures: list[str] = []
    if not artifact_dir.exists():
        return failures
    for path in sorted(artifact_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() == ".xml":
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if _contains_forbidden_absolute_path(text):
            failures.append(f"artifact file contains local absolute path: {_relative_or_str(path, artifact_dir)}")
    return failures


def build_environment_metadata(
    *,
    repo_root: Path,
    source_commit: str,
    artifact_commit: str,
    remote: str,
    source_branch: str | None,
    source_worktree_clean_before_run: bool,
    source_worktree_clean_after_run: bool,
    seed: int,
) -> dict[str, Any]:
    git_status_short = _git_stdout(repo_root, "status", "--short")
    core_diff = _core_diff_after_source(repo_root, source_commit, artifact_commit)
    return {
        "schema_version": 1,
        "time_utc": _utc_now(),
        "source_code_commit": source_commit,
        "artifact_commit": artifact_commit,
        "artifact_commit_observed": artifact_commit,
        "source_branch": source_branch,
        "source_code_commit_remote_resolvable": _remote_contains_commit(
            repo_root=repo_root,
            source_commit=source_commit,
            remote=remote,
            source_branch=source_branch,
        ),
        "source_code_commit_is_artifact_commit_ancestor": _is_ancestor(repo_root, source_commit, artifact_commit),
        "source_worktree_clean_before_run": source_worktree_clean_before_run,
        "source_worktree_clean_after_run": source_worktree_clean_after_run,
        "core_diff_after_source_commit": core_diff,
        "git_status_short": git_status_short,
        "package_versions": _package_versions(),
        "platform": platform.platform(),
        "python": sys.version,
        "seed": seed,
    }


def collect_lfs_state(*, repo_root: Path, run_fsck: bool = True) -> dict[str, Any]:
    tracked_paths = _git_lfs_paths(repo_root)
    pointer_files = detect_lfs_pointer_files(repo_root, [repo_root / rel for rel in tracked_paths])
    fsck_returncode = 0
    fsck_output = ""
    if run_fsck:
        fsck = _run(repo_root, "git", "lfs", "fsck")
        fsck_returncode = fsck.returncode
        fsck_output = (fsck.stdout or "") + (fsck.stderr or "")
    return {
        "schema_version": 1,
        "time_utc": _utc_now(),
        "git_lfs_version": _git_lfs_version(repo_root),
        "fsck_returncode": fsck_returncode,
        "fsck_output": fsck_output.strip(),
        "pointer_files_detected": pointer_files,
        "missing_lfs_objects": _missing_lfs_objects(fsck_output) if fsck_returncode else [],
        "materialized_snapshot_count": _materialized_snapshot_count(repo_root),
        "tracked_paths": tracked_paths,
    }


def detect_lfs_pointer_files(repo_root: Path, paths: Iterable[Path]) -> list[str]:
    repo_root = repo_root.resolve()
    pointer_files: list[str] = []
    for path in paths:
        candidate = path if path.is_absolute() else repo_root / path
        if _is_lfs_pointer(candidate):
            pointer_files.append(_relative_or_str(candidate, repo_root))
    return sorted(pointer_files)


def build_acceptance_ledger(
    *,
    command: str,
    returncode: int | None,
    stdout: str,
    stderr: str,
    source_code_commit: str,
) -> dict[str, Any]:
    rc = 1 if returncode is None else returncode
    return {
        "schema_version": 1,
        "audit_name": "retargeting_v3_step2_3_1_capability",
        "command": command,
        "returncode": rc,
        "status": "passed" if rc == 0 else "failed",
        "stdout": stdout,
        "stderr": stderr,
        "source_code_commit": source_code_commit,
        "time_utc": _utc_now(),
    }


def write_commands_file(
    path: Path,
    *,
    source_commit: str,
    pytest_command: str | None,
    acceptance_command: str | None,
) -> None:
    lines = [
        "# Step 2.3.1 clean capability artifact generation protocol",
        f"C={source_commit}",
        "git fetch origin retargeting-v3-step2-capability-acceptance-hardening",
        "git merge-base --is-ancestor \"$C\" FETCH_HEAD",
        "git worktree add --detach ../.soma-retargeter-worktrees/capability-hardening-source \"$C\"",
        "git status --short",
        "git lfs pull",
        "git lfs fsck",
        pytest_command
        or (
            "PYTHONPATH=. python -m pytest -q "
            "--junitxml \"$OUT/artifacts/test_results/junit.xml\" "
            "> \"$OUT/artifacts/test_results/pytest.txt\" 2>&1"
        ),
        (
            "PYTHONPATH=. python -m soma_retargeter.tools.validate_kinematic_profile_v3 "
            "--manifest assets/robot_zoo/robot_zoo_manifest.json "
            "--lock assets/robot_zoo/robot_zoo_lock.json "
            "--output-dir \"$OUT/artifacts\" "
            "--low-discrepancy-count 32 "
            "--deterministic-rerun"
        ),
        acceptance_command
        or (
            "PYTHONPATH=. python scripts/audit_retargeting_v3_capability.py "
            "--artifact-dir \"$OUT/artifacts\" "
            "--numerical-artifact-dir artifacts/retargeting_v3_step2_numerical "
            "--manifest assets/robot_zoo/robot_zoo_manifest.json "
            "--lock assets/robot_zoo/robot_zoo_lock.json "
            "--write-report \"$OUT/artifacts/capability_audit.json\""
        ),
        "git status --short",
        "git merge-base --is-ancestor \"$C\" \"$A\"",
        "git diff --name-only \"$C..$A\" -- soma_retargeter tests scripts .github",
    ]
    path.write_text("\n".join(lines) + "\n")


def _git_stdout(repo_root: Path, *args: str) -> str:
    result = _run(repo_root, "git", *args)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _current_branch(repo_root: Path) -> str | None:
    result = _run(repo_root, "git", "branch", "--show-current")
    branch = result.stdout.strip()
    return branch or None


def _core_diff_after_source(repo_root: Path, source_commit: str, artifact_commit: str) -> list[str]:
    result = _run(
        repo_root,
        "git",
        "diff",
        "--name-only",
        f"{source_commit}..{artifact_commit}",
        "--",
        *CORE_DIFF_PATHS,
    )
    if result.returncode != 0:
        return [f"<git diff failed: {result.stderr.strip() or result.stdout.strip()}>"]
    return [line for line in result.stdout.splitlines() if line.strip()]


def _remote_contains_commit(
    *,
    repo_root: Path,
    source_commit: str,
    remote: str,
    source_branch: str | None,
) -> bool:
    if not source_branch:
        return False
    fetch = _run(repo_root, "git", "fetch", "--quiet", remote, source_branch)
    if fetch.returncode != 0:
        return False
    remote_head = _git_stdout(repo_root, "rev-parse", "FETCH_HEAD")
    return _is_ancestor(repo_root, source_commit, remote_head)


def _is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    result = _run(repo_root, "git", "merge-base", "--is-ancestor", ancestor, descendant)
    return result.returncode == 0


def _git_lfs_paths(repo_root: Path) -> list[str]:
    result = _run(repo_root, "git", "lfs", "ls-files", "--long")
    if result.returncode != 0:
        return []
    paths = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            paths.append(parts[-1])
    return sorted(paths)


def _git_lfs_version(repo_root: Path) -> str:
    result = _run(repo_root, "git", "lfs", "version")
    return result.stdout.strip() if result.returncode == 0 else ""


def _materialized_snapshot_count(repo_root: Path) -> int:
    snapshot_root = repo_root / "assets/robot_zoo/snapshots"
    if not snapshot_root.exists():
        return 0
    count = 0
    for path in snapshot_root.glob("*/model.*"):
        if path.is_file() and not _is_lfs_pointer(path):
            count += 1
    return count


def _missing_lfs_objects(fsck_output: str) -> list[str]:
    missing = []
    for line in fsck_output.splitlines():
        lowered = line.lower()
        if "missing" in lowered or "not found" in lowered:
            missing.append(line.strip())
    return missing


def _is_lfs_pointer(path: Path) -> bool:
    try:
        return path.is_file() and path.read_bytes()[: len(LFS_POINTER_PREFIX)] == LFS_POINTER_PREFIX
    except OSError:
        return False


def _package_versions() -> dict[str, str]:
    packages = ("mujoco", "newton", "robot-descriptions", "soma-retargeter", "warp")
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_text_arg(path: str | None) -> str:
    if path is None:
        return ""
    return Path(path).read_text()


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes"}:
        return True
    if lowered in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _contains_forbidden_absolute_path(text: str) -> bool:
    return any(token in text for token in FORBIDDEN_ABSOLUTE_PATH_TOKENS)


def _absolute_path_failures(value: Any, label: str) -> list[str]:
    failures = []
    for path, child in _walk(value):
        if isinstance(child, str) and _contains_forbidden_absolute_path(child):
            failures.append(f"{label}.{'.'.join(path)} contains a local absolute path")
    return failures


def _walk(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = (*path, str(key))
            yield child_path, child
            yield from _walk(child, child_path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            child_path = (*path, str(index))
            yield child_path, child
            yield from _walk(child, child_path)


def _relative_or_str(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _print_failures(failures: list[str]) -> int:
    if not failures:
        print("Step 2.3.1 capability provenance PASS")
        return 0
    print("Step 2.3.1 capability provenance FAIL")
    for failure in failures:
        print(f"- {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
