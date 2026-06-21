from __future__ import annotations

import json
from pathlib import Path


LOCK = Path("assets/robot_zoo/source_lock.json")

# These are the sixteen public sources that were source_unavailable in the
# previous Step-2 artifact because the local cache was incomplete.
EXPECTED_RECOVERY_SOURCE_IDS = {
    "atlas_drc_urdf",
    "atlas_v4_urdf",
    "bolt_urdf",
    "cassie_urdf",
    "draco3_urdf",
    "fourier_gr1_urdf",
    "jaxon_urdf",
    "mujoco_humanoid_mjcf",
    "rhea_urdf",
    "robonaut2_urdf",
    "romeo_urdf",
    "sigmaban_urdf",
    "simple_humanoid_urdf",
    "talos_urdf",
    "upkie_urdf",
    "valkyrie_urdf",
}


def test_previously_unavailable_sources_have_immutable_recovery_records():
    lock = json.loads(LOCK.read_text())
    recovery = lock["recovery_sources"]

    assert set(recovery) == EXPECTED_RECOVERY_SOURCE_IDS
    assert lock["counts"]["recovery_source_count"] == len(EXPECTED_RECOVERY_SOURCE_IDS)

    for source in recovery.values():
        assert source["repository"].startswith("https://github.com/")
        assert source["repository"].endswith(".git")
        assert len(source["commit"]) == 40
        assert source["model_path"]
        assert len(source["model_blob_sha"]) == 40
        assert source["redistribution"] in {"kinematic_snapshot", "fetch_only"}
        assert isinstance(source["required"], bool)
