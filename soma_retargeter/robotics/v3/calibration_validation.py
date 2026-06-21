"""Independent validation helpers for v3 rest calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rest_frames import RestCalibration
from .spatial import rotation_error
from .target_builder import build_neutral_targets


@dataclass(frozen=True)
class NeutralReconstructionValidation:
    position_errors: dict[str, float]
    orientation_errors: dict[str, float]
    max_position_error: float
    max_orientation_error: float
    position_gate_m: float = 1e-3
    orientation_gate_rad: float = 1e-2
    source: str = "independent_target_builder_vs_runtime_neutral_fk"

    @property
    def passed(self) -> bool:
        return self.max_position_error < self.position_gate_m and self.max_orientation_error < self.orientation_gate_rad

    def to_json(self) -> dict:
        return {
            "position_errors": self.position_errors,
            "orientation_errors": self.orientation_errors,
            "max_position_error": self.max_position_error,
            "max_orientation_error": self.max_orientation_error,
            "position_gate_m": self.position_gate_m,
            "orientation_gate_rad": self.orientation_gate_rad,
            "passed": self.passed,
            "source": self.source,
        }


def validate_neutral_reconstruction(calibration: RestCalibration) -> NeutralReconstructionValidation:
    """Regenerate neutral targets and compare them with runtime neutral FK snapshots."""
    targets = build_neutral_targets(calibration)
    position_errors: dict[str, float] = {}
    orientation_errors: dict[str, float] = {}
    for name, neutral_t in calibration.robot_neutral_site_transforms.items():
        target_t = targets.transforms.get(name)
        if target_t is None:
            position_errors[name] = float("inf")
            orientation_errors[name] = float("inf")
            continue
        position_errors[name] = float(np.linalg.norm(target_t[:3, 3] - neutral_t[:3, 3]))
        orientation_errors[name] = rotation_error(target_t[:3, :3], neutral_t[:3, :3])
    return NeutralReconstructionValidation(
        position_errors=position_errors,
        orientation_errors=orientation_errors,
        max_position_error=max(position_errors.values(), default=0.0),
        max_orientation_error=max(orientation_errors.values(), default=0.0),
    )
