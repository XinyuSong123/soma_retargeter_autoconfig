"""Configuration contract for optional V3 runtime profile integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


V3_RUNTIME_CONFIG_KEY = "v3_runtime_profile"

V3_RUNTIME_MODES = frozenset({"disabled", "shadow", "override_experimental"})
V3_TARGET_POLICIES = frozenset({"shadow_only", "replace_configured_semantics"})
DEFAULT_SEMANTIC_TASKS = ("Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot")
DEFAULT_DIAGNOSTICS_DIR = "artifacts/retargeting_v3_step3_runtime_shadow"


class RuntimeV3Mode:
    DISABLED = "disabled"
    SHADOW = "shadow"
    OVERRIDE_EXPERIMENTAL = "override_experimental"


@dataclass(frozen=True)
class V3RuntimeProfileConfig:
    enabled: bool = False
    mode: str = "disabled"
    profile_artifact_root: str = "artifacts/retargeting_v3_step2_capability"
    profile_model_id: str | None = None
    target_policy: str = "shadow_only"
    fail_on_fingerprint_mismatch: bool = True
    allow_runtime_recompile_on_mismatch: bool = False
    semantic_tasks: tuple[str, ...] = DEFAULT_SEMANTIC_TASKS
    override_tasks: tuple[str, ...] = ()
    diagnostics_enabled: bool = True
    diagnostics_max_frames: int = 240
    diagnostics_output_dir: str = DEFAULT_DIAGNOSTICS_DIR
    write_per_frame_debug: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.enabled and self.mode != "disabled"

    @property
    def should_compute_targets(self) -> bool:
        return self.active

    @property
    def should_collect_diagnostics(self) -> bool:
        return self.active and self.diagnostics_enabled

    @property
    def shadow_enabled(self) -> bool:
        return self.active and self.mode == "shadow"

    @property
    def override_enabled(self) -> bool:
        return self.active and self.mode == "override_experimental"

    @property
    def diagnostics_dir(self) -> Path:
        return Path(self.diagnostics_output_dir)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["semantic_tasks"] = list(self.semantic_tasks)
        payload["override_tasks"] = list(self.override_tasks)
        return payload


RuntimeV3Config = V3RuntimeProfileConfig


def parse_v3_runtime_profile_config(raw_config: dict[str, Any] | None) -> V3RuntimeProfileConfig:
    """Parse the optional top-level ``v3_runtime_profile`` config block.

    Missing config and ``{"enabled": false}`` both produce an inert config. The
    caller can use this in ``NewtonPipeline`` without importing runtime V3 code.
    """

    if raw_config is None:
        return V3RuntimeProfileConfig()
    payload = raw_config.get(V3_RUNTIME_CONFIG_KEY) if isinstance(raw_config, dict) else None
    if payload is None:
        return V3RuntimeProfileConfig()
    if not isinstance(payload, dict):
        raise TypeError(f"{V3_RUNTIME_CONFIG_KEY} must be an object when present")

    enabled = bool(payload.get("enabled", False))
    mode = str(payload.get("mode", "disabled"))
    if not enabled:
        mode = "disabled"
    if mode not in V3_RUNTIME_MODES:
        allowed = ", ".join(sorted(V3_RUNTIME_MODES))
        raise ValueError(f"invalid v3 runtime mode {mode!r}; allowed: {allowed}")

    target_policy = str(payload.get("target_policy", "shadow_only"))
    if target_policy not in V3_TARGET_POLICIES:
        allowed = ", ".join(sorted(V3_TARGET_POLICIES))
        raise ValueError(f"invalid v3 runtime target_policy {target_policy!r}; allowed: {allowed}")
    if mode == "shadow" and target_policy != "shadow_only":
        raise ValueError("v3 shadow mode requires target_policy='shadow_only'")
    if mode == "override_experimental" and target_policy != "replace_configured_semantics":
        raise ValueError("v3 override_experimental mode requires target_policy='replace_configured_semantics'")

    semantic_tasks = _string_tuple(payload.get("semantic_tasks", DEFAULT_SEMANTIC_TASKS), "semantic_tasks")
    override_tasks = _string_tuple(payload.get("override_tasks", ()), "override_tasks")
    if mode == "override_experimental" and not override_tasks:
        raise ValueError("v3 override_experimental requires non-empty override_tasks")
    unknown_override = sorted(set(override_tasks) - set(semantic_tasks))
    if unknown_override:
        raise ValueError(f"override_tasks must be a subset of semantic_tasks; unknown: {unknown_override}")

    diagnostics_max_frames = int(payload.get("diagnostics_max_frames", 240))
    if diagnostics_max_frames <= 0:
        raise ValueError("diagnostics_max_frames must be positive")

    profile_model_id = payload.get("profile_model_id")
    if profile_model_id is not None:
        profile_model_id = str(profile_model_id)
        if not profile_model_id:
            raise ValueError("profile_model_id cannot be empty when provided")

    return V3RuntimeProfileConfig(
        enabled=enabled,
        mode=mode,
        profile_artifact_root=str(payload.get("profile_artifact_root", "artifacts/retargeting_v3_step2_capability")),
        profile_model_id=profile_model_id,
        target_policy=target_policy,
        fail_on_fingerprint_mismatch=bool(payload.get("fail_on_fingerprint_mismatch", True)),
        allow_runtime_recompile_on_mismatch=bool(payload.get("allow_runtime_recompile_on_mismatch", False)),
        semantic_tasks=semantic_tasks,
        override_tasks=override_tasks,
        diagnostics_enabled=bool(payload.get("diagnostics_enabled", True)),
        diagnostics_max_frames=diagnostics_max_frames,
        diagnostics_output_dir=str(payload.get("diagnostics_output_dir", DEFAULT_DIAGNOSTICS_DIR)),
        write_per_frame_debug=bool(payload.get("write_per_frame_debug", False)),
        raw=dict(payload),
    )


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{label} must be a list of semantic names")
    out = tuple(str(item) for item in value)
    if any(not item for item in out):
        raise ValueError(f"{label} cannot contain empty semantic names")
    if len(set(out)) != len(out):
        raise ValueError(f"{label} cannot contain duplicate semantic names")
    return out
