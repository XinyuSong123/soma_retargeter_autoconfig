"""Canonical motion-class capability status policy.

This module intentionally has no Robot Zoo dependencies.  It evaluates a
canonical projection report as a motion/task matrix and returns only the profile
status policy outcome plus concrete gate failures.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping


INVARIANCE_MOTIONS = frozenset(
    {
        "neutral",
        "root_translation",
        "global_root_yaw",
    }
)

ORDINARY_MOTIONS = frozenset(
    {
        "torso_pitch",
        "torso_roll",
        "torso_yaw",
        "arms_forward",
        "elbow_bend",
        "squat",
        "single_step",
    }
)

EXTENDED_MOTIONS = frozenset(
    {
        "mixed_torso_rotation",
        "overhead_reach",
        "asymmetric_arm_reach",
        "crossed_body_reach",
    }
)

STRESS_MOTIONS = frozenset(
    {
        "extreme_but_valid_joint_limit_stress",
    }
)

PROFILE_PASSED = "passed"
PROFILE_CAPABILITY_LIMITED_PASSED = "capability_limited_passed"
PROFILE_ALGORITHM_FAILED = "algorithm_failed"

EXACT_CERTIFICATE_CLASS = "exact_reachable"
LIMITED_CERTIFICATE_CLASSES = frozenset(
    {
        "capability_limited_rank",
        "capability_limited_joint_limits",
        "capability_limited_mixed",
        "unsupported_rank_zero",
    }
)
FAILED_CERTIFICATE_CLASSES = frozenset(
    {
        "solver_failed",
        "numerical_invalid",
        "invalid_target_geometry",
    }
)
CERTIFICATE_CLASSES = frozenset(
    {
        EXACT_CERTIFICATE_CLASS,
        *LIMITED_CERTIFICATE_CLASSES,
        *FAILED_CERTIFICATE_CLASSES,
    }
)

ORTHOGONAL_RESIDUAL_FRACTION_GATE = 0.95
REACHABLE_RESIDUAL_FRACTION_GATE = 0.05
_TASK_DIMENSION = 3

_MOTION_CLASS_BY_NAME = {
    **{motion: "invariance" for motion in INVARIANCE_MOTIONS},
    **{motion: "ordinary" for motion in ORDINARY_MOTIONS},
    **{motion: "extended" for motion in EXTENDED_MOTIONS},
    **{motion: "stress" for motion in STRESS_MOTIONS},
}
_ALL_MOTIONS = frozenset(_MOTION_CLASS_BY_NAME)


@dataclass(frozen=True)
class CapabilityTaskDecision:
    motion: str
    task: str
    motion_class: str
    certificate_class: str | None
    exact: bool
    limited: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityStatusResult:
    status: str
    failures: tuple[str, ...]
    limited_tasks: tuple[str, ...]
    task_decisions: tuple[CapabilityTaskDecision, ...]


def motion_class_for(motion_name: str) -> str:
    try:
        return _MOTION_CLASS_BY_NAME[str(motion_name)]
    except KeyError as exc:
        raise ValueError(f"unknown canonical motion: {motion_name}") from exc


def validate_motion_class_partition(canonical_motion_order: Iterable[str]) -> list[str]:
    failures: list[str] = []
    class_counts = Counter(
        motion
        for motion_group in (INVARIANCE_MOTIONS, ORDINARY_MOTIONS, EXTENDED_MOTIONS, STRESS_MOTIONS)
        for motion in motion_group
    )
    for motion, count in sorted(class_counts.items()):
        if count > 1:
            failures.append(f"duplicate canonical motion class assignment: {motion}")
    assigned = set(class_counts)
    missing_assignments = sorted(_ALL_MOTIONS - assigned)
    for motion in missing_assignments:
        failures.append(f"missing canonical motion class assignment: {motion}")

    order = [str(motion) for motion in canonical_motion_order]
    order_counts = Counter(order)
    for motion, count in sorted(order_counts.items()):
        if count > 1:
            failures.append(f"duplicate canonical motion in order: {motion}")
    present = set(order)
    for motion in sorted(_ALL_MOTIONS - present):
        failures.append(f"missing canonical motion class assignment: {motion}")
    for motion in sorted(present - _ALL_MOTIONS):
        failures.append(f"unknown canonical motion in order: {motion}")
    return failures


def evaluate_profile_status(
    canonical_projection: Mapping[str, object],
    *,
    required_tasks: Iterable[str] | None = None,
) -> CapabilityStatusResult:
    failures: list[str] = []
    decisions: list[CapabilityTaskDecision] = []
    limited_tasks: list[str] = []

    if not isinstance(canonical_projection, Mapping):
        return CapabilityStatusResult(
            status=PROFILE_ALGORITHM_FAILED,
            failures=("canonical projection report missing or not a mapping",),
            limited_tasks=(),
            task_decisions=(),
        )

    motions = canonical_projection.get("motions", {})
    if not isinstance(motions, Mapping):
        return CapabilityStatusResult(
            status=PROFILE_ALGORITHM_FAILED,
            failures=("canonical projection motions missing or not a mapping",),
            limited_tasks=(),
            task_decisions=(),
        )

    motion_order = canonical_projection.get("motion_order")
    if isinstance(motion_order, (list, tuple)):
        ordered_motions = [str(motion) for motion in motion_order]
    else:
        ordered_motions = [str(motion) for motion in motions]
        failures.append("canonical projection motion_order missing or not a list")
    failures.extend(validate_motion_class_partition(ordered_motions))

    task_names = _required_task_names(motions, required_tasks=required_tasks)
    if not task_names:
        failures.append("required task set is empty")

    for motion_name in ordered_motions:
        motion_class = _MOTION_CLASS_BY_NAME.get(motion_name)
        if motion_class is None:
            continue
        motion = motions.get(motion_name)
        if not isinstance(motion, Mapping):
            failures.append(f"{motion_name}: canonical motion report missing or not a mapping")
            continue
        tasks = motion.get("tasks", {})
        if not isinstance(tasks, Mapping):
            failures.append(f"{motion_name}: tasks missing or not a mapping")
            continue
        if motion_class == "stress":
            continue
        for task_name in task_names:
            task_payload = tasks.get(task_name)
            if not isinstance(task_payload, Mapping):
                failures.append(f"{motion_name}.{task_name}: required task report missing")
                continue
            decision = evaluate_task_status(motion_name, task_name, task_payload)
            decisions.append(decision)
            failures.extend(f"{motion_name}.{task_name}: {failure}" for failure in decision.failures)
            if decision.limited:
                limited_tasks.append(f"{motion_name}.{task_name}")

    if failures:
        status = PROFILE_ALGORITHM_FAILED
    elif limited_tasks:
        status = PROFILE_CAPABILITY_LIMITED_PASSED
    else:
        status = PROFILE_PASSED
    return CapabilityStatusResult(
        status=status,
        failures=tuple(failures),
        limited_tasks=tuple(limited_tasks),
        task_decisions=tuple(decisions),
    )


def evaluate_task_status(
    motion_name: str,
    task_name: str,
    task_payload: Mapping[str, object],
) -> CapabilityTaskDecision:
    motion_class = motion_class_for(motion_name)
    threshold = projection_exact_threshold(task_name, motion_name)
    normalized = _finite_float(task_payload.get("normalized_residual"))
    exact_threshold_passed = normalized is not None and normalized <= threshold
    failures: list[str] = []
    if normalized is None:
        failures.append("normalized_residual missing or nonfinite")

    certificate = task_payload.get("capability_certificate")
    if not isinstance(certificate, Mapping):
        failures.append("capability_certificate missing")
        return CapabilityTaskDecision(
            motion=motion_name,
            task=task_name,
            motion_class=motion_class,
            certificate_class=None,
            exact=False,
            limited=False,
            failures=tuple(failures),
        )

    certificate_class = str(certificate.get("certificate_class", ""))
    if certificate_class not in CERTIFICATE_CLASSES:
        failures.append(f"unknown certificate_class={certificate_class!r}")
    elif certificate_class in FAILED_CERTIFICATE_CLASSES:
        failures.append(f"failed certificate_class={certificate_class}")
    elif motion_class == "invariance":
        if not exact_threshold_passed:
            failures.append(
                "invariance motion must be exact; "
                f"normalized_residual={normalized!r} threshold={threshold:g}"
            )
        if certificate_class != EXACT_CERTIFICATE_CLASS:
            failures.append(f"invariance motion cannot use certificate_class={certificate_class}")
        failures.extend(
            _exact_certificate_failures(
                certificate,
                task_payload,
                exact_threshold_passed=exact_threshold_passed,
            )
        )
    elif certificate_class == EXACT_CERTIFICATE_CLASS:
        failures.extend(
            _exact_certificate_failures(
                certificate,
                task_payload,
                exact_threshold_passed=exact_threshold_passed,
            )
        )
    elif certificate_class in LIMITED_CERTIFICATE_CLASSES:
        failures.extend(
            _limited_certificate_failures(
                certificate,
                task_payload,
                certificate_class=certificate_class,
                exact_threshold_passed=exact_threshold_passed,
                threshold=threshold,
                normalized_residual=normalized,
            )
        )

    limited = (
        certificate_class in LIMITED_CERTIFICATE_CLASSES
        and motion_class in {"ordinary", "extended"}
        and not failures
    )
    exact = certificate_class == EXACT_CERTIFICATE_CLASS and exact_threshold_passed and not failures
    return CapabilityTaskDecision(
        motion=motion_name,
        task=task_name,
        motion_class=motion_class,
        certificate_class=certificate_class or None,
        exact=exact,
        limited=limited,
        failures=tuple(failures),
    )


def projection_exact_threshold(task_name: str, motion_name: str) -> float:
    if motion_name == "neutral":
        return 1e-3
    if "foot" in task_name:
        return 0.06
    if "hand" in task_name:
        return 0.12
    if "torso" in task_name:
        return 0.08
    return 0.05


def profile_status_from_semantic_readiness(semantic_status: str, projection_status: str) -> str:
    if semantic_status != "full_humanoid_ready":
        return semantic_status
    if projection_status == PROFILE_PASSED:
        return "full_humanoid_ready"
    return projection_status


def _required_task_names(
    motions: Mapping[str, object],
    *,
    required_tasks: Iterable[str] | None,
) -> tuple[str, ...]:
    if required_tasks is not None:
        return tuple(str(task) for task in required_tasks)
    task_names: set[str] = set()
    for motion_name, motion in motions.items():
        if _MOTION_CLASS_BY_NAME.get(str(motion_name)) == "stress":
            continue
        if not isinstance(motion, Mapping):
            continue
        tasks = motion.get("tasks", {})
        if isinstance(tasks, Mapping):
            task_names.update(str(task) for task in tasks)
    return tuple(sorted(task_names))


def _exact_certificate_failures(
    certificate: Mapping[str, object],
    task_payload: Mapping[str, object],
    *,
    exact_threshold_passed: bool,
) -> list[str]:
    failures: list[str] = []
    if not exact_threshold_passed:
        failures.append("exact_reachable certificate failed exact_threshold_passed")
    failures.extend(
        _required_true_gate_failures(
            certificate,
            task_payload,
            ("projected_gradient_kkt", "seed_consensus"),
        )
    )
    failures.extend(
        _false_gate_failures(
            certificate,
            task_payload,
            ("primal_feasible", "continuation", "joint_limits", "numerical", "residual_explained"),
        )
    )
    gate_exact = _gate_value(certificate, task_payload, "exact_threshold_passed")
    if gate_exact is False:
        failures.append("certificate gate exact_threshold_passed=false")
    return failures


def _limited_certificate_failures(
    certificate: Mapping[str, object],
    task_payload: Mapping[str, object],
    *,
    certificate_class: str,
    exact_threshold_passed: bool,
    threshold: float,
    normalized_residual: float | None,
) -> list[str]:
    failures: list[str] = []
    if exact_threshold_passed:
        failures.append(
            "limited certificate requires exact_threshold_passed=false; "
            f"normalized_residual={normalized_residual!r} threshold={threshold:g}"
        )
    gate_exact = _gate_value(certificate, task_payload, "exact_threshold_passed")
    if gate_exact is True:
        failures.append("certificate gate exact_threshold_passed=true for limited certificate")
    failures.extend(
        _required_true_gate_failures(
            certificate,
            task_payload,
            ("projected_gradient_kkt", "seed_consensus", "residual_explained"),
        )
    )
    failures.extend(
        _false_gate_failures(
            certificate,
            task_payload,
            ("primal_feasible", "continuation", "joint_limits", "numerical"),
        )
    )

    if certificate_class in {"capability_limited_rank", "capability_limited_mixed"}:
        failures.extend(_rank_limited_failures(certificate))
    if certificate_class in {"capability_limited_joint_limits", "capability_limited_mixed"}:
        failures.extend(_joint_limit_failures(certificate))
    if certificate_class == "unsupported_rank_zero":
        failures.extend(_rank_zero_failures(certificate, task_payload))
    return failures


def _rank_limited_failures(certificate: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    decomposition = _mapping(certificate.get("decomposition"))
    rank = _finite_float(decomposition.get("rank"))
    rank_incompatible_fraction = _finite_float(decomposition.get("rank_incompatible_fraction"))
    rank_incompatible_residual = _finite_float(decomposition.get("rank_incompatible_residual_norm"))
    component_tolerance = _finite_float(decomposition.get("component_tolerance")) or 0.0
    if rank is None and rank_incompatible_residual is None:
        failures.append("rank evidence missing")
    elif not (
        (rank is not None and rank < _TASK_DIMENSION)
        or (rank_incompatible_residual is not None and rank_incompatible_residual > component_tolerance)
    ):
        failures.append("rank evidence does not show rank-incompatible residual")

    if rank_incompatible_fraction is None:
        failures.append("rank_incompatible_fraction missing")
    elif rank_incompatible_fraction < ORTHOGONAL_RESIDUAL_FRACTION_GATE:
        failures.append(
            "rank_incompatible_fraction below gate: "
            f"{rank_incompatible_fraction:g} < {ORTHOGONAL_RESIDUAL_FRACTION_GATE:g}"
        )
    elif 1.0 - rank_incompatible_fraction > REACHABLE_RESIDUAL_FRACTION_GATE:
        failures.append(
            "reachable_residual_fraction above gate: "
            f"{1.0 - rank_incompatible_fraction:g} > {REACHABLE_RESIDUAL_FRACTION_GATE:g}"
        )
    return failures


def _joint_limit_failures(certificate: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    active_limits = _mapping(certificate.get("active_limits"))
    decomposition = _mapping(certificate.get("decomposition"))
    active_count = _finite_float(active_limits.get("count"))
    active_residual = _finite_float(decomposition.get("active_limit_residual_norm"))
    component_tolerance = _finite_float(decomposition.get("component_tolerance")) or 0.0
    if not (
        (active_count is not None and active_count > 0)
        or (active_residual is not None and active_residual > component_tolerance)
    ):
        failures.append("active joint-limit evidence missing")
    return failures


def _rank_zero_failures(
    certificate: Mapping[str, object],
    task_payload: Mapping[str, object],
) -> list[str]:
    failures: list[str] = []
    decomposition = _mapping(certificate.get("decomposition"))
    rank = _finite_float(decomposition.get("rank"))
    if rank != 0:
        failures.append(f"unsupported_rank_zero requires rank=0, observed {rank!r}")
    active_coordinates = task_payload.get("active_coordinates", [])
    active_coordinate_count = _sequence_length(active_coordinates)
    if active_coordinate_count != 0:
        failures.append("unsupported_rank_zero requires active_coordinate_count=0")
    demand_norm = _finite_float(decomposition.get("demand_norm"))
    demand_residual = _finite_float(task_payload.get("demand_residual"))
    component_tolerance = _finite_float(decomposition.get("component_tolerance")) or 0.0
    unreachable_demand = task_payload.get("unreachable_demand")
    if not (
        unreachable_demand is True
        or (demand_norm is not None and demand_norm > component_tolerance)
        or (demand_residual is not None and demand_residual > component_tolerance)
    ):
        failures.append("unsupported_rank_zero missing preserved nonzero demand evidence")
    failures.extend(_false_gate_failures(certificate, task_payload, ("no_orthogonal_leakage", "numerical")))
    return failures


def _required_true_gate_failures(
    certificate: Mapping[str, object],
    task_payload: Mapping[str, object],
    gate_names: Iterable[str],
) -> list[str]:
    failures: list[str] = []
    for gate_name in gate_names:
        value = _gate_value(certificate, task_payload, gate_name)
        if value is True:
            continue
        if value is False:
            failures.append(f"certificate gate {gate_name}=false")
        else:
            failures.append(f"certificate gate {gate_name} missing")
    return failures


def _false_gate_failures(
    certificate: Mapping[str, object],
    task_payload: Mapping[str, object],
    gate_names: Iterable[str],
) -> list[str]:
    failures: list[str] = []
    for gate_name in gate_names:
        if _gate_value(certificate, task_payload, gate_name) is False:
            failures.append(f"certificate gate {gate_name}=false")
    return failures


def _gate_value(
    certificate: Mapping[str, object],
    task_payload: Mapping[str, object],
    gate_name: str,
) -> object:
    gates = _mapping(certificate.get("gates"))
    if gate_name in gates:
        return gates[gate_name]
    if gate_name in certificate:
        return certificate[gate_name]
    if gate_name in task_payload:
        return task_payload[gate_name]
    if gate_name == "seed_consensus":
        seed = _mapping(certificate.get("seed_consensus"))
        if "passed" in seed:
            return seed["passed"]
    if gate_name in {"projected_gradient_kkt", "primal_feasible"}:
        kkt = _mapping(certificate.get("kkt"))
        if gate_name == "projected_gradient_kkt":
            norm = _finite_float(kkt.get("projected_gradient_norm"))
            tolerance = _finite_float(kkt.get("tolerance"))
            if norm is not None and tolerance is not None:
                return norm <= tolerance
        kkt_certificate = _mapping(task_payload.get("kkt_certificate"))
        if gate_name in kkt_certificate:
            return kkt_certificate[gate_name]
    return None


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence_length(value: object) -> int:
    if value is None:
        return 0
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 1


def _finite_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric
