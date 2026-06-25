from __future__ import annotations

from soma_retargeter.robotics.v3.target_builder import CANONICAL_MOTION_NAMES

from soma_retargeter.robotics.v3.capability_status import (
    EXTENDED_MOTIONS,
    INVARIANCE_MOTIONS,
    ORDINARY_MOTIONS,
    STRESS_MOTIONS,
    motion_class_for,
    validate_motion_class_partition,
)


def test_fixed_motion_classes_partition_canonical_motion_order() -> None:
    classes = [INVARIANCE_MOTIONS, ORDINARY_MOTIONS, EXTENDED_MOTIONS, STRESS_MOTIONS]
    flattened = [motion for group in classes for motion in group]

    assert sorted(flattened) == sorted(CANONICAL_MOTION_NAMES)
    assert len(flattened) == len(set(flattened))
    assert validate_motion_class_partition(CANONICAL_MOTION_NAMES) == []

    for motion in CANONICAL_MOTION_NAMES:
        assert motion_class_for(motion) in {"invariance", "ordinary", "extended", "stress"}


def test_motion_class_partition_reports_missing_duplicates_and_unknowns() -> None:
    incomplete_order = [
        motion
        for motion in CANONICAL_MOTION_NAMES
        if motion != "crossed_body_reach"
    ]
    invalid_order = [*incomplete_order, "neutral", "single_step_target"]

    failures = validate_motion_class_partition(invalid_order)

    assert any("missing canonical motion class assignment: crossed_body_reach" in failure for failure in failures)
    assert any("duplicate canonical motion in order: neutral" in failure for failure in failures)
    assert any("unknown canonical motion in order: single_step_target" in failure for failure in failures)
