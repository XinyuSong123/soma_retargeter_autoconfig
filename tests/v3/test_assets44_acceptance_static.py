from __future__ import annotations

import json
from pathlib import Path

from soma_retargeter.robotics.v3.robot_zoo import reproduction_validate_command, sha256_file


LOCK = Path("assets/robot_zoo/robot_zoo_lock.json")
SEMANTIC_MAPS = Path("assets/robot_zoo/semantic_maps")
SEMANTIC_EXPECTATIONS = Path("assets/robot_zoo/semantic_expectations")


def test_assets44_lock_scope_is_frozen() -> None:
    lock = json.loads(LOCK.read_text())
    totals = lock["totals"]
    scope = lock["scope_decision"]

    assert totals["entries"] == 46
    assert totals["source_available"] == 46
    assert totals["vendored"] == 38
    assert totals["fetch_only"] == 5
    assert totals["local_existing"] == 1
    assert totals["snapshot_failed"] == 2
    assert scope["decision"] == "accept_exactly_two_deferred_snapshots"
    assert scope["deferred_snapshot_ids"] == ["berkeley_humanoid_urdf", "romeo_urdf"]
    assert scope["deferred_snapshot_count"] == 2
    assert scope["in_scope_source_count"] == 44
    assert scope["blocking_count_for_current_scope"] == 0


def test_assets44_semantic_maps_are_source_bound() -> None:
    for path in sorted(SEMANTIC_MAPS.glob("*.json")):
        payload = json.loads(path.read_text())
        assert payload["verification_status"] == "verified", path
        assert payload.get("model_fingerprint"), path
        source = payload.get("model_source")
        assert isinstance(source, dict), path
        assert source.get("resolver"), path
        assert source.get("path"), path
        assert source.get("sha256"), path
        source_path = str(source["path"])
        if source_path.startswith("${ROBOT_DESCRIPTIONS_CACHE}/"):
            continue
        local = Path(source_path)
        assert local.exists(), path
        assert sha256_file(local) == source["sha256"], path


def test_assets44_structured_partials_are_explicit() -> None:
    expected = {
        "berkeley_humanoid_mjcf_direct": {"LeftHand", "RightHand"},
        "sigmaban_urdf": {"LeftHand", "RightHand"},
        "simple_humanoid_urdf": {"LeftHand", "RightHand", "LeftFoot", "RightFoot"},
    }
    for model_id, missing in expected.items():
        payload = json.loads((SEMANTIC_EXPECTATIONS / f"{model_id}.json").read_text())
        assert payload["expected_capability"] == "partial_humanoid"
        assert payload["verification_status"] == "partial_blocked"
        assert payload["coverage_status"] == "structurally_incomplete"
        assert payload["blocker"]["code"] == "structurally_incomplete"
        assert missing <= set(payload["missing_required_semantics"])


def test_assets44_reproduction_command_is_lock_aware() -> None:
    command = reproduction_validate_command(
        manifest_path=Path("${ROBOT_ZOO_MANIFEST}"),
        lock_path=Path("${ROBOT_ZOO_LOCK}"),
        output_dir=Path("${RETARGETING_V3_ARTIFACTS}"),
        low_discrepancy_count=1,
        deterministic_rerun=True,
    )

    assert "--lock" in command
    assert "${ROBOT_ZOO_LOCK}" in command
    assert "--deterministic-rerun" in command


def test_assets44_audit_summary_counts_use_deterministic_totals(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "environment.json").write_text("{}")
    (artifact_dir / "commands.txt").write_text("")
    for name in (
        "scope.json",
        "source_inventory.json",
        "load_matrix.json",
        "semantic_matrix.json",
        "deterministic_rerun.json",
        "cross_format.json",
        "deferred_snapshots.json",
    ):
        (artifact_dir / name).write_text("{}")
    summary = {
        "manifest_total": 46,
        "in_scope_total": 44,
        "deferred_snapshot_count": 2,
        "vendored_snapshot_count": 38,
        "fetch_only_cached_count": 5,
        "project_local_count": 1,
        "source_unavailable": 0,
        "model_load_failed": 0,
        "semantic_failed": 0,
        "load_passed_in_scope": 44,
        "semantic_passed_in_scope": 44,
        "deterministic_compared": 12,
        "deterministic_matched": 12,
    }
    (artifact_dir / "summary.json").write_text(json.dumps(summary))

    from scripts.audit_retargeting_v3_assets44 import _audit_summary

    errors: list[str] = []
    _audit_summary(errors, summary)
    assert errors == []
