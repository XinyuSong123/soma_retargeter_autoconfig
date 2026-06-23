from __future__ import annotations

import json
from pathlib import Path

from soma_retargeter.robotics.v3.target_geometry_audit import (
    DEFAULT_AUDIT_MODEL_IDS,
    audit_target_geometry_matrix,
    write_target_geometry_matrix,
)


def test_default_target_geometry_scope_lists_requested_differential_groups():
    expected = {
        "booster_t1_urdf",
        "booster_t1_mjcf",
        "fourier_n1_urdf",
        "fourier_n1_mjcf",
        "talos_urdf",
        "pal_talos_mjcf_direct",
        "atlas_drc_urdf",
        "atlas_v4_urdf",
        "unitree_h1_urdf",
        "unitree_h1_mjcf",
        "unitree_g1_urdf",
        "unitree_g1_mjcf",
        "roboparty_rpo_local",
        "robotis_op3_mjcf",
        "jaxon_urdf",
        "valkyrie_urdf",
        "mujoco_humanoid_mjcf",
    }

    assert set(DEFAULT_AUDIT_MODEL_IDS) == expected


def test_target_geometry_matrix_writer_records_counts_and_artifact_shape(tmp_path: Path):
    output_path = tmp_path / "target_geometry_matrix.json"

    payload = write_target_geometry_matrix(
        output_path,
        model_ids=("roboparty_rpo_local",),
        backend="newton",
    )

    assert output_path.exists()
    assert json.loads(output_path.read_text()) == payload
    assert payload["schema_version"] == 1
    assert payload["counts"]["requested"] == 1
    assert payload["counts"]["runtime_loaded"] == 1
    assert payload["rows"][0]["model_id"] == "roboparty_rpo_local"
    assert payload["rows"][0]["source"]["source_path"]
    assert payload["rows"][0]["sites"]["LeftToe"]["target_frame_convention"] == "verified_body_local_offset"


def test_target_geometry_audit_marks_unavailable_runtime_without_map_rewrite():
    matrix = audit_target_geometry_matrix(model_ids=("jaxon_urdf",), backend="newton")
    row = matrix["rows"][0]

    assert row["model_id"] == "jaxon_urdf"
    assert row["status"] in {"runtime_loaded", "runtime_unavailable_artifact_fallback"}
    assert row["semantic_map"]["verification_status"] == "verified"
    assert row["map_change_recommendation"] in {"none", "source_unavailable_no_runtime_correction"}
