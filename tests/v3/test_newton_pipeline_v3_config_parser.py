import pytest

from soma_retargeter.pipelines.v3_runtime_config import (
    RuntimeV3Mode,
    parse_v3_runtime_profile_config,
)


def test_missing_v3_runtime_profile_config_is_disabled_noop():
    cfg = parse_v3_runtime_profile_config({})

    assert cfg.enabled is False
    assert cfg.mode == RuntimeV3Mode.DISABLED
    assert cfg.should_compute_targets is False
    assert cfg.should_collect_diagnostics is False
    assert cfg.profile_model_id is None


def test_enabled_false_is_disabled_even_if_mode_is_present():
    cfg = parse_v3_runtime_profile_config(
        {
            "v3_runtime_profile": {
                "enabled": False,
                "mode": "shadow",
                "diagnostics_enabled": True,
            }
        }
    )

    assert cfg.enabled is False
    assert cfg.mode == RuntimeV3Mode.DISABLED
    assert cfg.should_compute_targets is False
    assert cfg.should_collect_diagnostics is False


def test_shadow_config_uses_contract_defaults():
    cfg = parse_v3_runtime_profile_config(
        {
            "v3_runtime_profile": {
                "enabled": True,
                "mode": "shadow",
            }
        }
    )

    assert cfg.enabled is True
    assert cfg.mode == RuntimeV3Mode.SHADOW
    assert cfg.target_policy == "shadow_only"
    assert cfg.profile_artifact_root == "artifacts/retargeting_v3_step2_capability"
    assert cfg.semantic_tasks == (
        "Hips",
        "Chest",
        "LeftHand",
        "RightHand",
        "LeftFoot",
        "RightFoot",
    )
    assert cfg.override_tasks == ()
    assert cfg.should_compute_targets is True
    assert cfg.should_collect_diagnostics is True


def test_override_experimental_requires_explicit_replace_policy_and_tasks():
    with pytest.raises(ValueError, match="override_tasks"):
        parse_v3_runtime_profile_config(
            {
                "v3_runtime_profile": {
                    "enabled": True,
                    "mode": "override_experimental",
                    "target_policy": "replace_configured_semantics",
                }
            }
        )

    cfg = parse_v3_runtime_profile_config(
        {
            "v3_runtime_profile": {
                "enabled": True,
                "mode": "override_experimental",
                "target_policy": "replace_configured_semantics",
                "override_tasks": ["LeftHand", "RightHand"],
            }
        }
    )

    assert cfg.mode == RuntimeV3Mode.OVERRIDE_EXPERIMENTAL
    assert cfg.override_tasks == ("LeftHand", "RightHand")


def test_invalid_v3_runtime_profile_values_fail_closed():
    with pytest.raises(ValueError, match="mode"):
        parse_v3_runtime_profile_config(
            {"v3_runtime_profile": {"enabled": True, "mode": "replace"}}
        )

    with pytest.raises(ValueError, match="diagnostics_max_frames"):
        parse_v3_runtime_profile_config(
            {
                "v3_runtime_profile": {
                    "enabled": True,
                    "mode": "shadow",
                    "diagnostics_max_frames": 0,
                }
            }
        )
