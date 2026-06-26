"""Explicit experimental V3 target override policy and array mapping."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class OverridePolicyError(RuntimeError):
    pass


class FingerprintMismatchError(OverridePolicyError):
    pass


@dataclass(frozen=True)
class OverridePolicyDecision:
    robot_type: str
    override_allowed: bool
    strict_match_required: bool
    fingerprint_match: bool
    runtime_local_profile_required: bool
    profile_status: str


@dataclass(frozen=True)
class FiniteTargetSummary:
    frame_count: int
    target_count: int
    nan_count: int
    inf_count: int
    finite: bool


def require_explicit_override_config(config: Mapping[str, Any] | None) -> None:
    payload = dict(config or {})
    if payload.get("enabled") is not True:
        raise OverridePolicyError("override_experimental requires enabled=true")
    if payload.get("mode") != "override_experimental":
        raise OverridePolicyError("override_experimental requires mode='override_experimental'")
    if payload.get("target_policy") != "replace_configured_semantics":
        raise OverridePolicyError("override_experimental requires target_policy='replace_configured_semantics'")
    override_tasks = payload.get("override_tasks")
    if not isinstance(override_tasks, list) or not override_tasks:
        raise OverridePolicyError("override_experimental requires explicit non-empty override_tasks")


def replace_configured_semantic_targets(
    legacy_targets: np.ndarray,
    v3_targets: Mapping[str, np.ndarray],
    effector_names: list[str],
    override_tasks: list[str],
    *,
    config: Mapping[str, Any] | None,
) -> np.ndarray:
    require_explicit_override_config(config)
    configured = set(str(name) for name in (config or {}).get("override_tasks", []))
    requested = [str(name) for name in override_tasks]
    if not set(requested) <= configured:
        raise OverridePolicyError("requested override task is not explicitly configured")

    legacy = np.asarray(legacy_targets, dtype=np.float32)
    if legacy.ndim != 3 or legacy.shape[2] != 7:
        raise OverridePolicyError(f"legacy targets must have shape [F,N,7], got {legacy.shape}")
    out = np.array(legacy, copy=True)
    effector_index = {name: index for index, name in enumerate(effector_names)}
    for semantic_name in requested:
        if semantic_name not in effector_index:
            raise OverridePolicyError(f"missing legacy effector for override task {semantic_name}")
        if semantic_name not in v3_targets:
            raise OverridePolicyError(f"missing V3 target for override task {semantic_name}")
        replacement = np.asarray(v3_targets[semantic_name], dtype=np.float32)
        if replacement.shape != (legacy.shape[0], 7):
            raise OverridePolicyError(
                f"V3 target for {semantic_name} must have shape ({legacy.shape[0]}, 7), got {replacement.shape}"
            )
        if not np.all(np.isfinite(replacement)):
            raise OverridePolicyError(f"V3 target for {semantic_name} contains non-finite values")
        out[:, effector_index[semantic_name], :] = replacement
    return out


def load_profile_artifact(profile_root: str | Path, profile_model_id: str) -> dict[str, Any]:
    path = Path(profile_root) / "per_robot" / f"{profile_model_id}.json"
    return json.loads(path.read_text())


def validate_override_profile_policy(
    robot_type: str,
    profile: Mapping[str, Any],
    *,
    runtime_fingerprint: str,
    config: Mapping[str, Any],
    runtime_local_profile_generated: bool = False,
) -> OverridePolicyDecision:
    require_explicit_override_config(config)
    status = str(profile.get("status", ""))
    if status != "passed":
        raise OverridePolicyError(f"override_experimental requires profile status 'passed', got {status!r}")
    profile_fingerprint = str(profile.get("model", {}).get("fingerprint", ""))
    fingerprint_match = bool(profile_fingerprint and runtime_fingerprint == profile_fingerprint)
    strict = str(robot_type) == "roboparty_rpo"
    allow_runtime_local = bool(config.get("allow_runtime_recompile_on_mismatch", False))
    if not fingerprint_match:
        if strict:
            raise FingerprintMismatchError("strict RPO runtime/profile fingerprint match is required for override")
        if allow_runtime_local:
            if not runtime_local_profile_generated:
                raise FingerprintMismatchError(
                    f"{robot_type} override requires a runtime-local profile before replacing targets"
                )
            return OverridePolicyDecision(
                robot_type=str(robot_type),
                override_allowed=True,
                strict_match_required=False,
                fingerprint_match=False,
                runtime_local_profile_required=True,
                profile_status=status,
            )
        raise FingerprintMismatchError(f"{robot_type} runtime/profile fingerprint mismatch; override is fail-closed")
    return OverridePolicyDecision(
        robot_type=str(robot_type),
        override_allowed=True,
        strict_match_required=strict,
        fingerprint_match=True,
        runtime_local_profile_required=False,
        profile_status=status,
    )


def finite_target_summary(targets: np.ndarray) -> FiniteTargetSummary:
    values = np.asarray(targets)
    nan_count = int(np.isnan(values).sum())
    inf_count = int(np.isinf(values).sum())
    frame_count = int(values.shape[0]) if values.ndim >= 1 else 0
    target_count = int(values.shape[1]) if values.ndim >= 2 else 0
    return FiniteTargetSummary(
        frame_count=frame_count,
        target_count=target_count,
        nan_count=nan_count,
        inf_count=inf_count,
        finite=nan_count == 0 and inf_count == 0,
    )

