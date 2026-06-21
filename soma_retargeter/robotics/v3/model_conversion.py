"""Cross-format model conversion and equivalence scaffolding for Step 2."""

from __future__ import annotations

from pathlib import Path
import json

import mujoco
import numpy as np

from .model_adapter import MuJoCoRuntimeModelAdapter, SemanticSite
from .model_fingerprint import sha256_file
from .semantic_sites import build_semantic_sites
from .spatial import rotation_error


DEFAULT_CONVERSION_SETTINGS = {
    "converter": "mujoco.mj_saveLastXML",
    "canonical_format": "mjcf",
    "preserve_runtime_topology": True,
}


def convert_urdf_to_canonical_mjcf(
    urdf_path: str | Path,
    output_path: str | Path,
    *,
    settings: dict | None = None,
) -> dict:
    """Convert a URDF through MuJoCo's compiled loader to canonical MJCF.

    This is a same-source conversion primitive: the generated MJCF is intended
    for strict equivalence checks against the original URDF loaded by the same
    runtime adapter, not for comparing unrelated vendor and Menagerie variants.
    """

    source = Path(urdf_path)
    output = Path(output_path)
    merged_settings = dict(DEFAULT_CONVERSION_SETTINGS)
    merged_settings.update(settings or {})
    adapter = MuJoCoRuntimeModelAdapter(source, model_format="urdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(output), adapter.model)
    report = {
        "schema_version": 1,
        "source": str(source),
        "output": str(output),
        "settings": merged_settings,
        "source_sha256": sha256_file(source),
        "output_sha256": sha256_file(output),
        "source_fingerprint": adapter.fingerprint,
        "loader_provenance": adapter.loader_provenance,
        "runtime_signature": runtime_signature(adapter),
    }
    adapter.close()
    return report


def write_conversion_report(report: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def runtime_signature(adapter: MuJoCoRuntimeModelAdapter) -> dict:
    return {
        "backend": adapter.__class__.__name__,
        "format": adapter.model_format,
        "nq": adapter.nq,
        "nv": adapter.nv,
        "body_names": adapter.body_names,
        "coordinates": [coord.to_json() for coord in adapter.coordinate_info],
    }


def compare_runtime_models(
    left: MuJoCoRuntimeModelAdapter,
    right: MuJoCoRuntimeModelAdapter,
    *,
    semantic_map: dict[str, str | dict] | None = None,
    position_atol: float = 1e-7,
    rotation_atol: float = 1e-7,
) -> dict:
    """Compare same-source runtime kinematics with explicit tolerances."""

    failures: list[str] = []
    left_sig = runtime_signature(left)
    right_sig = runtime_signature(right)
    if left_sig["nq"] != right_sig["nq"] or left_sig["nv"] != right_sig["nv"]:
        failures.append("qpos_or_velocity_dimension_mismatch")
    if left_sig["body_names"] != right_sig["body_names"]:
        failures.append("body_name_order_mismatch")
    left_coords = _coordinate_signature(left)
    right_coords = _coordinate_signature(right)
    if left_coords != right_coords:
        failures.append("coordinate_signature_mismatch")

    semantic_fk = {}
    if semantic_map:
        left_sites = build_semantic_sites(left, semantic_map)
        right_sites = build_semantic_sites(right, semantic_map)
        semantic_fk = _compare_semantic_fk(
            left,
            right,
            left_sites,
            right_sites,
            position_atol=position_atol,
            rotation_atol=rotation_atol,
        )
        failures.extend(semantic_fk["failures"])

    return {
        "schema_version": 1,
        "comparison_mode": "same_source_strict",
        "strict_equivalent": not failures,
        "failures": failures,
        "left_fingerprint": left.fingerprint,
        "right_fingerprint": right.fingerprint,
        "left_signature": left_sig,
        "right_signature": right_sig,
        "semantic_fk": semantic_fk,
        "tolerances": {
            "position_atol": position_atol,
            "rotation_atol": rotation_atol,
        },
    }


def _coordinate_signature(adapter: MuJoCoRuntimeModelAdapter) -> list[dict]:
    return [
        {
            "label": coord.label,
            "joint_name": coord.joint_name,
            "joint_type": coord.joint_type,
            "limited": coord.limited,
            "lower": _finite_or_none(coord.lower),
            "upper": _finite_or_none(coord.upper),
        }
        for coord in adapter.coordinate_info
    ]


def _compare_semantic_fk(
    left: MuJoCoRuntimeModelAdapter,
    right: MuJoCoRuntimeModelAdapter,
    left_sites: dict[str, SemanticSite],
    right_sites: dict[str, SemanticSite],
    *,
    position_atol: float,
    rotation_atol: float,
) -> dict:
    left_state = left.forward_kinematics(left.neutral_q())
    right_state = right.forward_kinematics(right.neutral_q())
    per_site = {}
    failures: list[str] = []
    for name in sorted(set(left_sites) & set(right_sites)):
        left_t = left.site_transform(left_state, left_sites[name])
        right_t = right.site_transform(right_state, right_sites[name])
        position_error = float(np.linalg.norm(left_t[:3, 3] - right_t[:3, 3]))
        rot_error = rotation_error(left_t[:3, :3], right_t[:3, :3])
        per_site[name] = {
            "position_error": position_error,
            "rotation_error": rot_error,
            "passed": position_error <= position_atol and rot_error <= rotation_atol,
        }
        if not per_site[name]["passed"]:
            failures.append(f"semantic_fk_mismatch:{name}")
    return {"per_site": per_site, "failures": failures}


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None
