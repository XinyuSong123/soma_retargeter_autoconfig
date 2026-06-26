import json
from pathlib import Path

import numpy as np

try:
    from soma_retargeter.runtime.v3.override import (
        finite_target_summary,
        load_profile_artifact,
        replace_configured_semantic_targets,
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
    finite_target_summary = override.finite_target_summary
    load_profile_artifact = override.load_profile_artifact
    replace_configured_semantic_targets = override.replace_configured_semantic_targets
    validate_override_profile_policy = override.validate_override_profile_policy


PROFILE_ROOT = Path("artifacts/retargeting_v3_step2_capability")
RPO_OVERRIDE_CONFIG = Path("configs/roboparty_rpo/soma_to_rpo_retargeter_config_v3_override.json")
RPO_SHADOW_CONFIG = Path("configs/roboparty_rpo/soma_to_rpo_retargeter_config_v3_shadow.json")


def _load_config(path):
    return json.loads(path.read_text())["v3_runtime_profile"]


def test_rpo_override_smoke_config_is_explicit_and_does_not_tune_weights():
    override = json.loads(RPO_OVERRIDE_CONFIG.read_text())
    shadow = json.loads(RPO_SHADOW_CONFIG.read_text())

    assert override["ik_map"] == shadow["ik_map"]
    assert "ik_iterations" not in override
    assert "joint_limit_weight" not in override
    assert "smooth_joint_filter_weight" not in override

    v3 = override["v3_runtime_profile"]
    assert v3["enabled"] is True
    assert v3["mode"] == "override_experimental"
    assert v3["target_policy"] == "replace_configured_semantics"
    assert v3["override_tasks"] == ["Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot"]
    assert v3["profile_model_id"] == "roboparty_rpo_local"


def test_rpo_override_replacement_smoke_outputs_finite_for_120_frames():
    config = _load_config(RPO_OVERRIDE_CONFIG)
    profile = load_profile_artifact(PROFILE_ROOT, "roboparty_rpo_local")
    validate_override_profile_policy(
        "roboparty_rpo",
        profile,
        runtime_fingerprint=profile["model"]["fingerprint"],
        config=config,
    )

    frame_count = 120
    effector_names = ["Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot", "LeftArm"]
    legacy = np.zeros((frame_count, len(effector_names), 7), dtype=np.float32)
    legacy[..., 6] = 1.0
    v3_targets = {}
    for semantic_index, semantic in enumerate(config["override_tasks"]):
        targets = np.zeros((frame_count, 7), dtype=np.float32)
        targets[:, 0] = np.linspace(0.0, 0.01 * (semantic_index + 1), frame_count)
        targets[:, 1] = semantic_index * 0.001
        targets[:, 6] = 1.0
        v3_targets[semantic] = targets

    replaced = replace_configured_semantic_targets(
        legacy,
        v3_targets,
        effector_names,
        config["override_tasks"],
        config=config,
    )
    summary = finite_target_summary(replaced)

    assert summary.frame_count == frame_count
    assert summary.target_count == len(effector_names)
    assert summary.nan_count == 0
    assert summary.inf_count == 0
    assert summary.finite is True
    assert np.array_equal(replaced[:, effector_names.index("LeftArm"), :], legacy[:, -1, :])
