import json
import math

import numpy as np
import pytest

from soma_retargeter.robotics.v3.spatial import so3_exp, transform
from soma_retargeter.runtime.v3.diagnostics import (
    build_target_delta_diagnostics,
    write_deterministic_json,
)


def test_target_delta_diagnostics_are_per_semantic_finite_and_deterministic(tmp_path):
    legacy = {
        "Hips": np.stack([transform([0.0, 0.0, 0.0]), transform([1.0, 0.0, 0.0])]),
        "Chest": np.stack([transform([0.0, 0.0, 1.0]), transform([1.0, 0.0, 1.0])]),
    }
    v3 = {
        "Hips": np.stack([transform([0.1, 0.0, 0.0]), transform([1.3, 0.0, 0.0])]),
        "Chest": np.stack(
            [
                transform([0.0, 0.0, 1.0], so3_exp(np.array([0.0, 0.0, math.pi / 2.0]))),
                transform([1.0, 0.0, 1.0], so3_exp(np.array([0.0, 0.0, math.pi / 2.0]))),
            ]
        ),
    }

    payload = build_target_delta_diagnostics(
        robot_type="testbot",
        mode="shadow",
        clip_name="clip",
        frame_count=2,
        semantic_names=["Chest", "Hips", "LeftHand"],
        legacy_transforms=legacy,
        v3_transforms=v3,
        capability_status={"Chest": "converged", "Hips": "root_reference", "LeftHand": "unsupported"},
        root_policy={"horizontal_scale": 2.0, "support_height_policy": "source_support_height_delta"},
    )

    assert payload["semantic_names"] == ["Chest", "Hips", "LeftHand"]
    hips = payload["per_semantic"]["Hips"]
    assert hips["translation_delta_mean"] == 0.2
    assert hips["translation_delta_max"] == 0.3
    assert hips["translation_delta_p95"] == 0.29
    assert hips["rotation_delta_max"] == 0.0
    assert hips["finite_count"] == 2
    assert hips["nan_count"] == 0

    chest = payload["per_semantic"]["Chest"]
    assert chest["translation_delta_max"] == 0.0
    assert chest["rotation_delta_mean"] == pytest.approx(math.pi / 2.0)
    assert chest["target_source"] == "target_builder.build_targets_from_source_semantic_frames"
    assert chest["capability_status"] == "converged"

    skipped = payload["per_semantic"]["LeftHand"]
    assert skipped["skipped_reason"] == "missing_legacy_target,missing_v3_target"
    assert skipped["finite_count"] == 0

    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    write_deterministic_json(path_a, payload)
    write_deterministic_json(path_b, json.loads(path_a.read_text()))
    assert path_a.read_text() == path_b.read_text()
