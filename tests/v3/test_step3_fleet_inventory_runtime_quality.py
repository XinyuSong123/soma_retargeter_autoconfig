from __future__ import annotations

from pathlib import Path

from soma_retargeter.runtime.v3.fleet_inventory import (
    EXPECTED_CATEGORY_COUNTS,
    FULL_HUMANOID_PROFILE,
    NEGATIVE_CONTROL,
    PARTIAL_HUMANOID_PROFILE,
    category_counts,
    load_fleet_runtime_cases,
)


def test_step3_1_fleet_inventory_derives_all_44_rows_from_lock_and_profiles() -> None:
    cases = load_fleet_runtime_cases()

    assert len(cases) == 44
    assert category_counts(cases) == EXPECTED_CATEGORY_COUNTS
    assert {case.model_id for case in cases if case.category == PARTIAL_HUMANOID_PROFILE} == {
        "berkeley_humanoid_mjcf_direct",
        "sigmaban_urdf",
        "simple_humanoid_urdf",
    }
    assert {case.model_id for case in cases if case.category == NEGATIVE_CONTROL} == {
        "bolt_urdf",
        "cassie_mjcf",
        "cassie_urdf",
        "franka_panda_mjcf",
        "rhea_urdf",
        "stretch3_mjcf",
        "unitree_go2_mjcf",
        "unitree_go2_urdf",
        "upkie_urdf",
    }


def test_step3_1_fleet_inventory_resolves_materialized_runtime_sources() -> None:
    cases = load_fleet_runtime_cases()

    for case in cases:
        assert case.runtime_source_path.exists(), case.model_id
        assert case.runtime_source_path.is_file(), case.model_id
        assert case.runtime_source_sha256
        assert not case.lfs_pointer, case.model_id
        assert not Path(case.to_json()["runtime_source_path"]).is_absolute()
        assert case.profile_path.exists(), case.model_id


def test_step3_1_fleet_inventory_records_category_specific_semantic_contracts() -> None:
    cases = load_fleet_runtime_cases()

    full = [case for case in cases if case.category == FULL_HUMANOID_PROFILE]
    partial = [case for case in cases if case.category == PARTIAL_HUMANOID_PROFILE]
    negative = [case for case in cases if case.category == NEGATIVE_CONTROL]

    assert all(case.semantic_map_path and case.semantic_map_path.exists() for case in full)
    assert all(case.supported_semantics == ("Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot") for case in full)
    assert all(case.semantic_expectation_path and case.semantic_expectation_path.exists() for case in partial)
    assert all(case.supported_semantics for case in partial)
    assert all(case.missing_required_semantics for case in partial)
    assert all(not case.semantic_map_path for case in negative)
