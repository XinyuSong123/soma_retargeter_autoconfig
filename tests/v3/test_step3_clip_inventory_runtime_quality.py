from __future__ import annotations

from pathlib import Path

import pytest

from soma_retargeter.runtime.v3.clip_inventory import (
    CORE_CLIP_RELATIVE_PATHS,
    assert_core_clips_available,
    deterministic_frame_budget,
    inventory_motion_clips,
)


def _write_bvh(path: Path, *, frames: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "HIERARCHY",
                "ROOT Hips",
                "{",
                "  OFFSET 0 0 0",
                "  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation",
                "}",
                "MOTION",
                f"Frames: {frames}",
                "Frame Time: 0.008333",
                *("0 0 0 0 0 0" for _ in range(frames)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(path: Path, *, rows: int, header: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if header:
        lines.append("Frame,root_translateX")
    lines.extend(f"{index},0.0" for index in range(rows))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_clip_inventory_covers_all_motion_bvh_csv_and_enforces_core_clips(tmp_path: Path) -> None:
    for index, relative in enumerate(CORE_CLIP_RELATIVE_PATHS):
        _write_bvh(tmp_path / relative, frames=10 + index)
    _write_bvh(tmp_path / "assets/motions/bvh/extra_motion.bvh", frames=125)
    _write_csv(tmp_path / "assets/motions/csv/extra_motion.csv", rows=301, header=True)
    _write_csv(tmp_path / "assets/motions/bvh/legacy_headerless.csv", rows=5)

    inventory = inventory_motion_clips(tmp_path)

    assert inventory.status == "passed"
    assert inventory.format_counts == {"bvh": 5, "csv": 2}
    assert not inventory.missing_core_clips
    assert not inventory.unloadable_core_clips
    assert_core_clips_available(inventory)

    paths = {clip.path for clip in inventory.clips}
    assert paths == {
        *CORE_CLIP_RELATIVE_PATHS,
        "assets/motions/bvh/extra_motion.bvh",
        "assets/motions/csv/extra_motion.csv",
        "assets/motions/bvh/legacy_headerless.csv",
    }
    extra_csv = next(clip for clip in inventory.clips if clip.path.endswith("extra_motion.csv"))
    assert extra_csv.frame_count == 301
    assert extra_csv.frame_budget is not None
    assert extra_csv.frame_budget.policy == "strided_long_clip"


def test_clip_inventory_blocks_missing_or_unloadable_core_clip(tmp_path: Path) -> None:
    for relative in CORE_CLIP_RELATIVE_PATHS[:-1]:
        _write_bvh(tmp_path / relative, frames=8)
    broken = tmp_path / CORE_CLIP_RELATIVE_PATHS[0]
    broken.write_text("HIERARCHY\nROOT Hips\n", encoding="utf-8")

    inventory = inventory_motion_clips(tmp_path)

    assert inventory.status == "blocked"
    assert inventory.missing_core_clips == (CORE_CLIP_RELATIVE_PATHS[-1],)
    assert inventory.unloadable_core_clips == (CORE_CLIP_RELATIVE_PATHS[0],)
    with pytest.raises(RuntimeError, match="required core clip gate failed"):
        assert_core_clips_available(inventory)


def test_deterministic_frame_budget_records_stable_caps_and_stride() -> None:
    short = deterministic_frame_budget(12)
    mid = deterministic_frame_budget(240)
    long = deterministic_frame_budget(1000)

    assert short.policy == "full_short_clip"
    assert short.frame_indices == tuple(range(12))
    assert mid.policy == "full_mid_clip"
    assert mid.selected_frame_count == 240
    assert long.policy == "strided_long_clip"
    assert long.stride == 4
    assert long.selected_frame_count <= 300
    assert long.frame_indices[:5] == (0, 4, 8, 12, 16)
    assert long.to_json()["deterministic"] is True
