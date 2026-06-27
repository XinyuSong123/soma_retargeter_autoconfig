from __future__ import annotations

from pathlib import Path

from soma_retargeter.tools.run_v3_full_fleet_runtime_quality import run_full_fleet_runtime_quality


def test_step3_full_fleet_runner_is_deterministic_for_compact_budget(tmp_path: Path) -> None:
    kwargs = dict(
        step2_profile_root=Path("artifacts/retargeting_v3_step2_capability"),
        step3_shadow_root=Path("artifacts/retargeting_v3_step3_runtime_shadow"),
        lock=Path("assets/robot_zoo/robot_zoo_lock.json"),
        manifest=Path("assets/robot_zoo/robot_zoo_manifest.json"),
        clip_root=Path("assets/motions"),
        required_core_clips=[Path("assets/motions/bvh/Neutral_walk_forward_002__A057.bvh")],
        short_max_frames=3,
        mid_max_frames=6,
        deterministic_rerun=True,
        allow_dirty_internal_rerun=True,
    )
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = run_full_fleet_runtime_quality(artifact_root=first_root, **kwargs)
    second = run_full_fleet_runtime_quality(artifact_root=second_root, **kwargs)

    assert first["verdict"] == second["verdict"]
    assert first["verdict"] in {"PASS", "BLOCKED"}
    assert _stable_runtime_payload(first_root / "model_matrix.json") == _stable_runtime_payload(second_root / "model_matrix.json")
    assert _stable_runtime_payload(first_root / "quality_summary.json") == _stable_runtime_payload(second_root / "quality_summary.json")
    assert _stable_runtime_payload(first_root / "deterministic_rerun.json")["matched_count"] == 44


def _stable_runtime_payload(path: Path):
    import json

    payload = json.loads(path.read_text())
    return _strip_runtime_timing(payload)


def _strip_runtime_timing(value):
    if isinstance(value, dict):
        return {
            key: _strip_runtime_timing(child)
            for key, child in value.items()
            if key not in {"runtime_seconds", "diagnostics_hash"}
        }
    if isinstance(value, list):
        return [_strip_runtime_timing(child) for child in value]
    return value
