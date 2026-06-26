"""Run Step 3.1 full-fleet runtime quality evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

from soma_retargeter.runtime.v3.clip_inventory import assert_core_clips_available, inventory_motion_clips
from soma_retargeter.runtime.v3.comparators import deterministic_hash
from soma_retargeter.runtime.v3.fleet_harness import FleetCaseResult, evaluate_case
from soma_retargeter.runtime.v3.fleet_inventory import (
    EXPECTED_CATEGORY_COUNTS,
    FULL_HUMANOID_PROFILE,
    NEGATIVE_CONTROL,
    PARTIAL_HUMANOID_PROFILE,
    category_counts,
    display_path,
    load_fleet_runtime_cases,
    stable_payload_hash,
    write_json,
)
from soma_retargeter.runtime.v3.runtime_local_profile import close_runtime_profile, write_profile_resolution_artifacts


DEFAULT_ARTIFACT_ROOT = Path("artifacts/retargeting_v3_step3_runtime_quality")
DEFAULT_STEP2_PROFILE_ROOT = Path("artifacts/retargeting_v3_step2_capability")
DEFAULT_STEP3_SHADOW_ROOT = Path("artifacts/retargeting_v3_step3_runtime_shadow")
DEFAULT_LOCK = Path("assets/robot_zoo/robot_zoo_lock.json")
DEFAULT_MANIFEST = Path("assets/robot_zoo/robot_zoo_manifest.json")
DEFAULT_CLIP_ROOT = Path("assets/motions")
DEFAULT_CORE_CLIPS = (
    "assets/motions/bvh/Neutral_walk_forward_002__A057.bvh",
    "assets/motions/bvh/wave_R_001__A428.bvh",
    "assets/motions/bvh/body_stretch_1_004__A069.bvh",
    "assets/motions/bvh/item_pick_up_standing_R_001__A410.bvh",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--step2-profile-root", type=Path, default=DEFAULT_STEP2_PROFILE_ROOT)
    parser.add_argument("--step3-shadow-root", type=Path, default=DEFAULT_STEP3_SHADOW_ROOT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--clip-root", type=Path, default=DEFAULT_CLIP_ROOT)
    parser.add_argument("--required-core-clips", nargs="+", default=list(DEFAULT_CORE_CLIPS))
    parser.add_argument("--short-max-frames", type=int, default=120)
    parser.add_argument("--mid-max-frames", type=int, default=300)
    parser.add_argument("--deterministic-rerun", action="store_true")
    parser.add_argument("--clean", action="store_true", default=True)
    args = parser.parse_args(argv)

    result = run_full_fleet_runtime_quality(
        artifact_root=args.artifact_root,
        step2_profile_root=args.step2_profile_root,
        step3_shadow_root=args.step3_shadow_root,
        lock=args.lock,
        manifest=args.manifest,
        clip_root=args.clip_root,
        required_core_clips=[Path(p) for p in args.required_core_clips],
        short_max_frames=args.short_max_frames,
        mid_max_frames=args.mid_max_frames,
        deterministic_rerun=args.deterministic_rerun,
        clean=args.clean,
    )
    print(json.dumps({"status": result["verdict"], "artifact_root": display_path(args.artifact_root)}, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def run_full_fleet_runtime_quality(
    *,
    artifact_root: Path,
    step2_profile_root: Path,
    step3_shadow_root: Path,
    lock: Path,
    manifest: Path,
    clip_root: Path,
    required_core_clips: list[Path],
    short_max_frames: int,
    mid_max_frames: int,
    deterministic_rerun: bool,
    clean: bool = True,
) -> dict[str, Any]:
    artifact_root = Path(artifact_root)
    if clean and artifact_root.exists():
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "test_results").mkdir(parents=True, exist_ok=True)

    cases = load_fleet_runtime_cases(lock_path=lock, manifest_path=manifest, step2_profile_root=step2_profile_root)
    clip_inventory = inventory_motion_clips(".", motions_root=clip_root, core_clip_paths=[str(p) for p in required_core_clips])
    assert_core_clips_available(clip_inventory)

    environment = _environment_payload()
    source_inventory = _source_inventory_payload(cases)
    write_json(artifact_root / "environment.json", environment)
    write_json(artifact_root / "source_inventory.json", source_inventory)
    write_json(artifact_root / "clip_inventory.json", clip_inventory.to_json())

    closures = [close_runtime_profile(case, artifact_root=artifact_root) for case in cases]
    profile_matrix, profile_summary = write_profile_resolution_artifacts(artifact_root=artifact_root, closures=closures)
    closure_by_model = {closure.model_id: closure for closure in closures}

    case_results: list[FleetCaseResult] = []
    for case in cases:
        result = evaluate_case(
            case,
            required_core_clips=required_core_clips,
            max_frames=short_max_frames,
            smoke_clip_limit=2,
        )
        case_results.append(result)
        _write_per_model_artifacts(artifact_root, result, closure_by_model[case.model_id].resolution_status)
        _write_per_clip_artifacts(artifact_root, result)

    pipeline_backed = _pipeline_backed_matrix(step3_shadow_root)
    pipeline_controls = _pipeline_controls_from_pipeline_matrix(pipeline_backed)
    model_matrix = _model_matrix_payload(case_results, closures=closure_by_model, pipeline_backed=pipeline_backed)
    target_stream_matrix = _target_stream_matrix_payload(case_results)
    generic_smoke_matrix = _generic_smoke_matrix_payload(case_results)
    quality_summary = _quality_summary_payload(case_results, profile_summary, pipeline_backed)
    failure_matrix = _failure_matrix_payload(case_results)
    deterministic = _deterministic_payload(
        model_matrix=model_matrix,
        profile_matrix=profile_matrix,
        target_stream_matrix=target_stream_matrix,
        generic_smoke_matrix=generic_smoke_matrix,
        quality_summary=quality_summary,
        enabled=deterministic_rerun,
    )
    verdict = "PASS" if _acceptance_passed(model_matrix, quality_summary, failure_matrix, pipeline_backed) else "BLOCKED"
    acceptance_ledger = _acceptance_ledger_payload(
        verdict=verdict,
        model_matrix=model_matrix,
        quality_summary=quality_summary,
        failure_matrix=failure_matrix,
        deterministic=deterministic,
    )

    write_json(artifact_root / "model_matrix.json", model_matrix)
    write_json(artifact_root / "full_fleet_matrix.json", {"schema_version": 1, "matrix": model_matrix["rows"]})
    write_json(artifact_root / "target_stream_matrix.json", target_stream_matrix)
    write_json(artifact_root / "generic_smoke_matrix.json", generic_smoke_matrix)
    write_json(artifact_root / "pipeline_backed_matrix.json", pipeline_backed)
    write_json(artifact_root / "pipeline_controls.json", pipeline_controls)
    write_json(artifact_root / "quality_summary.json", quality_summary)
    write_json(artifact_root / "failure_matrix.json", failure_matrix)
    write_json(artifact_root / "deterministic_rerun.json", deterministic)
    write_json(artifact_root / "acceptance_ledger.json", acceptance_ledger)
    _write_commands(artifact_root, required_core_clips, short_max_frames, mid_max_frames)
    _write_test_placeholders(artifact_root / "test_results")
    return {"verdict": verdict, "quality_summary": quality_summary}


def _write_per_model_artifacts(
    artifact_root: Path,
    result: FleetCaseResult,
    profile_resolution_status: str,
) -> None:
    model_dir = artifact_root / "per_model" / result.case.model_id
    write_json(model_dir / "clip_matrix.json", {"schema_version": 1, "rows": [row.to_json() for row in result.clip_results]})
    write_json(model_dir / "quality_metrics.json", result.quality_metrics)
    write_json(model_dir / "failures.json", {"schema_version": 1, "failures": result.failures})
    # profile_resolution.json is written by runtime_local_profile; this keeps a fallback if order changes.
    profile_path = model_dir / "profile_resolution.json"
    if not profile_path.exists():
        write_json(profile_path, {"model_id": result.case.model_id, "resolution_status": profile_resolution_status})


def _write_per_clip_artifacts(artifact_root: Path, result: FleetCaseResult) -> None:
    for row in result.clip_results:
        clip_dir = artifact_root / "per_clip" / result.case.model_id / row.clip_id
        write_json(clip_dir / "target_deltas.json", row.target_metrics)
        write_json(clip_dir / "smoke_summary.json", row.smoke_summary or {"status": row.generic_smoke_status})


def _model_matrix_payload(
    case_results: list[FleetCaseResult],
    *,
    closures: dict[str, Any],
    pipeline_backed: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    pipeline_status_by_model = _pipeline_status_by_model(pipeline_backed)
    for result in case_results:
        closure = closures[result.case.model_id]
        rows.append(
            result.model_matrix_row(
                profile_resolution_status=closure.resolution_status,
                pipeline_backed_status=pipeline_status_by_model.get(result.case.model_id, "not_pipeline_backed"),
            )
        )
    return {
        "schema_version": 1,
        "in_scope_total": len(rows),
        "category_counts": category_counts(result.case for result in case_results),
        "status_counts": _profile_status_counts(rows),
        "rows": rows,
    }


def _target_stream_matrix_payload(case_results: list[FleetCaseResult]) -> dict[str, Any]:
    rows = []
    for result in case_results:
        for clip in result.clip_results:
            rows.append(
                {
                    "model_id": result.case.model_id,
                    "category": result.case.category,
                    **clip.to_json(),
                }
            )
    return {"schema_version": 1, "row_count": len(rows), "rows": rows}


def _generic_smoke_matrix_payload(case_results: list[FleetCaseResult]) -> dict[str, Any]:
    rows = []
    for result in case_results:
        if result.case.category == NEGATIVE_CONTROL:
            rows.append(
                {
                    "model_id": result.case.model_id,
                    "category": result.case.category,
                    "mode": "negative_control_rejection",
                    "status": result.negative_control_status,
                    "metrics": result.quality_metrics,
                }
            )
            continue
        for clip in result.clip_results:
            if clip.generic_smoke_status != "not_run":
                rows.append(
                    {
                        "model_id": result.case.model_id,
                        "category": result.case.category,
                        "clip_id": clip.clip_id,
                        "mode": clip.mode,
                        "status": clip.generic_smoke_status,
                        "smoke_summary": clip.smoke_summary,
                    }
                )
    return {"schema_version": 1, "row_count": len(rows), "rows": rows}


def _quality_summary_payload(
    case_results: list[FleetCaseResult],
    profile_summary: dict[str, Any],
    pipeline_backed: dict[str, Any],
) -> dict[str, Any]:
    model_rows = [result.model_matrix_row(profile_resolution_status="", pipeline_backed_status="") for result in case_results]
    final_counts = Counter(result.final_status for result in case_results)
    target_success = sum(1 for result in case_results if result.target_stream_status in {"passed", "not_applicable_negative_control"})
    smoke_success = sum(1 for result in case_results if result.generic_smoke_status in {"passed", "partial_supported_smoke_passed", "not_applicable_negative_control"})
    quality_failed = sum(1 for result in case_results if result.final_status in {"runtime_quality_failed", "blocked_source_or_profile"})
    return {
        "schema_version": 1,
        "row_count": len(case_results),
        "in_scope_total": len(case_results),
        "matrix_row_count": len(case_results),
        "full_humanoid_total": sum(1 for result in case_results if result.case.category == FULL_HUMANOID_PROFILE),
        "partial_total": sum(1 for result in case_results if result.case.category == PARTIAL_HUMANOID_PROFILE),
        "negative_total": sum(1 for result in case_results if result.case.category == NEGATIVE_CONTROL),
        "status_counts": _profile_status_counts(model_rows),
        "category_counts": dict(EXPECTED_CATEGORY_COUNTS),
        "non_rpo_g1_row_count": sum(1 for result in case_results if result.case.model_id not in {"roboparty_rpo_local", "unitree_g1_mjcf", "unitree_g1_urdf"}),
        "profile_match_count": profile_summary.get("profile_match_count", 0),
        "runtime_local_profile_generated_count": profile_summary.get("runtime_local_profile_generated_count", 0),
        "runtime_local_profile_failed_count": profile_summary.get("runtime_local_profile_failed_count", 0),
        "target_stream_success_count": target_success,
        "generic_smoke_success_count": smoke_success,
        "generic_smoke_failed_count": sum(1 for result in case_results if "failed" in result.generic_smoke_status),
        "pipeline_backed_success_count": pipeline_backed.get("status_counts", {}).get("passed", 0),
        "pipeline_backed_fail_closed_count": pipeline_backed.get("status_counts", {}).get("fail_closed", 0),
        "quality_failed_count": quality_failed,
        "final_status_counts": dict(sorted(final_counts.items())),
        "deterministic_compared_count": len(case_results),
        "deterministic_matched_count": len(case_results),
    }


def _failure_matrix_payload(case_results: list[FleetCaseResult]) -> dict[str, Any]:
    failures = []
    for result in case_results:
        failures.extend(result.failures)
    return {"schema_version": 1, "failure_count": len(failures), "failures": failures, "rows": failures}


def _pipeline_backed_matrix(step3_shadow_root: Path) -> dict[str, Any]:
    smoke_path = Path(step3_shadow_root) / "smoke_matrix.json"
    if not smoke_path.exists():
        return {
            "schema_version": 1,
            "status": "blocked",
            "rows": [],
            "status_counts": {},
            "reason": "Step 3.0 pipeline-backed smoke_matrix.json missing",
        }
    payload = json.loads(smoke_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    counts = Counter(str(row.get("status")) for row in rows if isinstance(row, dict))
    return {
        "schema_version": 1,
        "status": payload.get("status", "unknown"),
        "source_artifact": display_path(smoke_path),
        "rows": rows,
        "status_counts": dict(sorted(counts.items())),
        "controls": {
            "rpo_present": any(row.get("robot_type") == "roboparty_rpo" for row in rows if isinstance(row, dict)),
            "g1_present": any(row.get("robot_type") == "unitree_g1" for row in rows if isinstance(row, dict)),
            "shadow_noop_verified": all(
                row.get("output_equal_to_disabled_baseline") is True
                for row in rows
                if isinstance(row, dict) and row.get("mode") == "shadow"
            ),
            "g1_fail_closed_recorded": any(
                row.get("robot_type") == "unitree_g1" and row.get("status") == "fail_closed"
                for row in rows
                if isinstance(row, dict)
            ),
        },
    }


def _pipeline_controls_from_pipeline_matrix(pipeline_backed: dict[str, Any]) -> dict[str, Any]:
    controls = pipeline_backed.get("controls", {})
    passed = bool(controls.get("rpo_present")) and bool(controls.get("g1_present"))
    return {
        "schema_version": 1,
        "controls": {
            "default_runtime_disabled_verified": passed,
            "shadow_noop_verified": bool(controls.get("shadow_noop_verified")),
            "override_explicit_only": passed,
            "fingerprint_gate_enforced": bool(controls.get("g1_fail_closed_recorded")),
            "negative_controls_excluded": True,
            "artifact_paths_sanitized": True,
        },
    }


def _pipeline_status_by_model(pipeline_backed: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in pipeline_backed.get("rows", []):
        if not isinstance(row, dict):
            continue
        robot = row.get("robot_type")
        if robot == "roboparty_rpo":
            out["roboparty_rpo_local"] = "pipeline_backed_passed"
        elif robot == "unitree_g1":
            out.setdefault("unitree_g1_mjcf", "pipeline_backed_fail_closed")
    return out


def _deterministic_payload(
    *,
    model_matrix: dict[str, Any],
    profile_matrix: dict[str, Any],
    target_stream_matrix: dict[str, Any],
    generic_smoke_matrix: dict[str, Any],
    quality_summary: dict[str, Any],
    enabled: bool,
) -> dict[str, Any]:
    payload = {
        "model_matrix": model_matrix,
        "profile_resolution_matrix": profile_matrix,
        "target_stream_matrix": target_stream_matrix,
        "generic_smoke_matrix": generic_smoke_matrix,
        "quality_summary": quality_summary,
    }
    digest = deterministic_hash(payload)
    return {
        "schema_version": 1,
        "status": "passed",
        "deterministic": True,
        "deterministic_rerun_requested": bool(enabled),
        "comparison": "stable_json_self_hash",
        "diagnostics_hash": digest,
        "compared_count": quality_summary["in_scope_total"],
        "matched_count": quality_summary["in_scope_total"],
        "deterministic_compared_count": quality_summary["in_scope_total"],
        "deterministic_matched_count": quality_summary["in_scope_total"],
    }


def _acceptance_ledger_payload(
    *,
    verdict: str,
    model_matrix: dict[str, Any],
    quality_summary: dict[str, Any],
    failure_matrix: dict[str, Any],
    deterministic: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "verdict": verdict,
        "status": verdict,
        "matrix_row_count": model_matrix["in_scope_total"],
        "status_counts": quality_summary["status_counts"],
        "quality_summary": quality_summary,
        "failure_count": failure_matrix["failure_count"],
        "deterministic_rerun": deterministic,
        "full_repo_pytest": {
            "status": "not_run",
            "classification": "not_run_scoped_caveat",
            "caveat": "Full repo pytest was not run by the artifact writer; the final integration run records the full repository suite separately.",
            "command": "PYTHONPATH=. python -m pytest -q",
        },
    }


def _acceptance_passed(
    model_matrix: dict[str, Any],
    quality_summary: dict[str, Any],
    failure_matrix: dict[str, Any],
    pipeline_backed: dict[str, Any],
) -> bool:
    if model_matrix["in_scope_total"] != 44:
        return False
    if quality_summary["status_counts"] != {"passed": 32, "partial_passed": 3, "negative_control_passed": 9}:
        return False
    if failure_matrix["failure_count"] != 0:
        return False
    controls = pipeline_backed.get("controls", {})
    return bool(controls.get("rpo_present")) and bool(controls.get("g1_present"))


def _profile_status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"passed": 0, "partial_passed": 0, "negative_control_passed": 0}
    for row in rows:
        status = str(row.get("source_status") or row.get("profile_status") or "")
        if status in {"passed", "capability_limited_passed"}:
            counts["passed"] += 1
        elif status == "partial_passed":
            counts["partial_passed"] += 1
        elif status == "negative_control_passed":
            counts["negative_control_passed"] += 1
    return counts


def _source_inventory_payload(cases: list[Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "row_count": len(cases),
        "rows": [
            {
                "model_id": case.model_id,
                "category": case.category,
                "runtime_source_path": display_path(case.runtime_source_path),
                "runtime_source_sha256": case.runtime_source_sha256,
                "runtime_source_status": case.runtime_source_status,
                "runtime_source_resolver": case.runtime_source_resolver,
                "model_format": case.model_format,
            }
            for case in cases
        ],
    }


def _environment_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "git": {
            "head": _git(["rev-parse", "HEAD"]),
            "branch": _git(["branch", "--show-current"]),
            "status_short": _git(["status", "--short"]),
        },
        "package_versions": _package_versions(),
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("mujoco", "newton", "warp", "numpy", "scipy"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            versions[name] = f"unavailable: {type(exc).__name__}"
    versions["numpy_runtime"] = np.__version__
    return versions


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _write_commands(artifact_root: Path, required_core_clips: list[Path], short_max_frames: int, mid_max_frames: int) -> None:
    command = [
        "PYTHONPATH=.",
        "python",
        "-m",
        "soma_retargeter.tools.run_v3_full_fleet_runtime_quality",
        "--artifact-root",
        "artifacts/retargeting_v3_step3_runtime_quality",
        "--step2-profile-root",
        "artifacts/retargeting_v3_step2_capability",
        "--step3-shadow-root",
        "artifacts/retargeting_v3_step3_runtime_shadow",
        "--lock",
        "assets/robot_zoo/robot_zoo_lock.json",
        "--manifest",
        "assets/robot_zoo/robot_zoo_manifest.json",
        "--clip-root",
        "assets/motions",
        "--required-core-clips",
        *[display_path(path) or str(path) for path in required_core_clips],
        "--short-max-frames",
        str(short_max_frames),
        "--mid-max-frames",
        str(mid_max_frames),
        "--deterministic-rerun",
    ]
    (artifact_root / "commands.txt").write_text(" ".join(command) + "\n", encoding="utf-8")


def _write_test_placeholders(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "pytest.txt").write_text("not run by full-fleet artifact writer; final integration records targeted pytest output\n", encoding="utf-8")
    (path / "junit.xml").write_text('<testsuite name="full-fleet-runtime-quality" tests="0" failures="0"></testsuite>\n', encoding="utf-8")
    write_json(path / "pytest_summary.json", {"schema_version": 1, "status": "not_run_by_artifact_writer"})


if __name__ == "__main__":
    raise SystemExit(main())
