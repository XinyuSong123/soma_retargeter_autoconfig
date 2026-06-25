from __future__ import annotations

import soma_retargeter.robotics.v3.validation as validation_module


def test_negative_control_variant_pair_remains_not_eligible_even_when_loaded() -> None:
    pair = {
        "inputs": {
            "urdf_status": "negative_control_passed",
            "mjcf_status": "negative_control_passed",
            "urdf_summary": {
                "status": "negative_control_passed",
                "expected_capability": "negative_control",
                "robot_class": "non_humanoid",
            },
            "mjcf_summary": {
                "status": "negative_control_passed",
                "expected_capability": "negative_control",
                "robot_class": "non_humanoid",
            },
        }
    }

    eligibility = validation_module._variant_pair_eligibility(pair)

    assert eligibility is not None
    assert eligibility["status"] == "not_eligible"
    assert eligibility["urdf_status"] == "negative_control_passed"
    assert eligibility["mjcf_status"] == "negative_control_passed"
    assert "urdf:negative_control" in eligibility["reason"]
    assert "mjcf:negative_control" in eligibility["reason"]
