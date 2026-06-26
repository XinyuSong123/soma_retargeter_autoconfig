import json
from pathlib import Path

import pytest

try:
    from soma_retargeter.runtime.v3.override import (
        FingerprintMismatchError,
        OverridePolicyError,
        load_profile_artifact,
        validate_override_profile_policy,
    )
except ModuleNotFoundError as exc:
    if not str(exc.name).startswith("soma_retargeter.runtime.v3."):
        raise
    import importlib.util
    import sys

    module_path = Path("soma_retargeter/runtime/v3/override.py")
    spec = importlib.util.spec_from_file_location("_agent_d_override", module_path)
    override = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = override
    spec.loader.exec_module(override)
    FingerprintMismatchError = override.FingerprintMismatchError
    OverridePolicyError = override.OverridePolicyError
    load_profile_artifact = override.load_profile_artifact
    validate_override_profile_policy = override.validate_override_profile_policy


PROFILE_ROOT = Path("artifacts/retargeting_v3_step2_capability")


def _override_config(**extra):
    config = {
        "enabled": True,
        "mode": "override_experimental",
        "target_policy": "replace_configured_semantics",
        "override_tasks": ["Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot"],
        "allow_runtime_recompile_on_mismatch": False,
    }
    config.update(extra)
    return config


def test_g1_override_fail_closed_on_fingerprint_mismatch():
    profile = load_profile_artifact(PROFILE_ROOT, "unitree_g1_mjcf")

    with pytest.raises(FingerprintMismatchError, match="unitree_g1"):
        validate_override_profile_policy(
            "unitree_g1",
            profile,
            runtime_fingerprint="definitely-not-the-profile-fingerprint",
            config=_override_config(),
            runtime_local_profile_generated=False,
        )


def test_g1_override_mismatch_requires_runtime_local_profile_when_allowed():
    profile = load_profile_artifact(PROFILE_ROOT, "unitree_g1_mjcf")

    with pytest.raises(FingerprintMismatchError, match="runtime-local profile"):
        validate_override_profile_policy(
            "unitree_g1",
            profile,
            runtime_fingerprint="definitely-not-the-profile-fingerprint",
            config=_override_config(allow_runtime_recompile_on_mismatch=True),
            runtime_local_profile_generated=False,
        )

    decision = validate_override_profile_policy(
        "unitree_g1",
        profile,
        runtime_fingerprint="definitely-not-the-profile-fingerprint",
        config=_override_config(allow_runtime_recompile_on_mismatch=True),
        runtime_local_profile_generated=True,
    )
    assert decision.override_allowed is True
    assert decision.runtime_local_profile_required is True
    assert decision.fingerprint_match is False


def test_rpo_override_requires_strict_profile_match():
    profile = load_profile_artifact(PROFILE_ROOT, "roboparty_rpo_local")
    decision = validate_override_profile_policy(
        "roboparty_rpo",
        profile,
        runtime_fingerprint=profile["model"]["fingerprint"],
        config=_override_config(),
    )
    assert decision.strict_match_required is True
    assert decision.fingerprint_match is True
    assert decision.override_allowed is True

    with pytest.raises(FingerprintMismatchError, match="strict"):
        validate_override_profile_policy(
            "roboparty_rpo",
            profile,
            runtime_fingerprint="definitely-not-the-profile-fingerprint",
            config=_override_config(allow_runtime_recompile_on_mismatch=True),
            runtime_local_profile_generated=True,
        )


def test_override_rejects_non_passed_profile_status(tmp_path):
    profile = json.loads((PROFILE_ROOT / "per_robot" / "roboparty_rpo_local.json").read_text())
    profile["status"] = "partial_passed"

    with pytest.raises(OverridePolicyError, match="status"):
        validate_override_profile_policy(
            "roboparty_rpo",
            profile,
            runtime_fingerprint=profile["model"]["fingerprint"],
            config=_override_config(),
        )
