import numpy as np
import pytest

try:
    from soma_retargeter.runtime.v3.override import (
        OverridePolicyError,
        require_explicit_override_config,
        replace_configured_semantic_targets,
    )
except ModuleNotFoundError as exc:
    if not str(exc.name).startswith("soma_retargeter.runtime.v3."):
        raise
    import importlib.util
    import sys
    from pathlib import Path

    module_path = Path("soma_retargeter/runtime/v3/override.py")
    spec = importlib.util.spec_from_file_location("_agent_d_override", module_path)
    override = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = override
    spec.loader.exec_module(override)
    OverridePolicyError = override.OverridePolicyError
    require_explicit_override_config = override.require_explicit_override_config
    replace_configured_semantic_targets = override.replace_configured_semantic_targets


def _legacy_targets(frame_count=3):
    values = np.arange(frame_count * 4 * 7, dtype=np.float32)
    return values.reshape(frame_count, 4, 7)


def test_replace_configured_semantic_targets_preserves_effector_order_and_shape():
    legacy = _legacy_targets()
    effector_names = ["Hips", "Chest", "LeftHand", "RightHand"]
    v3_targets = {
        "Chest": np.full((3, 7), 100.0, dtype=np.float32),
        "RightHand": np.full((3, 7), 200.0, dtype=np.float32),
        "LeftHand": np.full((3, 7), 300.0, dtype=np.float32),
    }

    replaced = replace_configured_semantic_targets(
        legacy,
        v3_targets,
        effector_names,
        ["RightHand", "Chest"],
        config={
            "enabled": True,
            "mode": "override_experimental",
            "target_policy": "replace_configured_semantics",
            "override_tasks": ["RightHand", "Chest"],
        },
    )

    assert replaced.shape == legacy.shape
    assert np.array_equal(replaced[:, 0, :], legacy[:, 0, :])
    assert np.array_equal(replaced[:, 2, :], legacy[:, 2, :])
    assert np.array_equal(replaced[:, 1, :], v3_targets["Chest"])
    assert np.array_equal(replaced[:, 3, :], v3_targets["RightHand"])
    assert effector_names == ["Hips", "Chest", "LeftHand", "RightHand"]
    assert np.isfinite(replaced).all()


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"enabled": False, "mode": "override_experimental", "target_policy": "replace_configured_semantics"},
        {"enabled": True, "mode": "shadow", "target_policy": "replace_configured_semantics"},
        {"enabled": True, "mode": "override_experimental", "target_policy": "shadow_only"},
    ],
)
def test_override_requires_explicit_experimental_config(config):
    with pytest.raises(OverridePolicyError):
        require_explicit_override_config(config)


def test_replace_rejects_missing_semantic_in_v3_targets():
    with pytest.raises(OverridePolicyError, match="missing V3 target"):
        replace_configured_semantic_targets(
            _legacy_targets(),
            {"Chest": np.ones((3, 7), dtype=np.float32)},
            ["Hips", "Chest", "LeftHand", "RightHand"],
            ["LeftHand"],
            config={
                "enabled": True,
                "mode": "override_experimental",
                "target_policy": "replace_configured_semantics",
                "override_tasks": ["LeftHand"],
            },
        )


def test_replace_rejects_nonfinite_v3_targets():
    target = np.ones((3, 7), dtype=np.float32)
    target[1, 0] = np.nan
    with pytest.raises(OverridePolicyError, match="non-finite"):
        replace_configured_semantic_targets(
            _legacy_targets(),
            {"Chest": target},
            ["Hips", "Chest", "LeftHand", "RightHand"],
            ["Chest"],
            config={
                "enabled": True,
                "mode": "override_experimental",
                "target_policy": "replace_configured_semantics",
                "override_tasks": ["Chest"],
            },
        )
