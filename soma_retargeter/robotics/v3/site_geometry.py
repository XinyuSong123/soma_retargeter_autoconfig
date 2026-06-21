"""Geometry helpers for distal semantic sites.

The helpers in this module intentionally return body-local positions plus
provenance metadata. They do not decide whether a robot is a full humanoid;
callers can use the nonzero-origin gate for positive humanoid hand and foot
sites.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_IDENTITY_QUAT_XYZW = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)


@dataclass(frozen=True)
class BodyGeometryBounds:
    body_name: str
    minimum: np.ndarray
    maximum: np.ndarray
    evidence: tuple[str, ...]

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.minimum + self.maximum)

    @property
    def span(self) -> np.ndarray:
        return self.maximum - self.minimum


@dataclass(frozen=True)
class GeometrySite:
    semantic_name: str
    body_name: str
    local_position: np.ndarray
    local_rotation_xyzw: np.ndarray
    source: str
    confidence: float
    evidence: tuple[str, ...]

    def to_semantic_map_entry(self) -> dict:
        return {
            "body": self.body_name,
            "local_position": self.local_position.tolist(),
            "local_rotation_xyzw": self.local_rotation_xyzw.tolist(),
            "source": self.source,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


def body_geometry_bounds(adapter, body_name: str) -> BodyGeometryBounds:
    """Return conservative body-local geom bounds for one compiled body."""

    resolved = adapter.resolve_body_name(body_name)
    model = getattr(adapter, "model", None)
    if model is None or not hasattr(model, "body_geomadr"):
        raise ValueError("adapter does not expose compiled body geometry")
    body_id = adapter.body_id(resolved)
    geom_adr = int(model.body_geomadr[body_id])
    geom_num = int(model.body_geomnum[body_id])
    if geom_adr < 0 or geom_num <= 0:
        raise ValueError(f"body {resolved!r} has no compiled geometry")

    minima = []
    maxima = []
    evidence = []
    for geom_id in range(geom_adr, geom_adr + geom_num):
        pos = np.asarray(model.geom_pos[geom_id], dtype=float)
        half_extents = _geom_half_extents(model, geom_id)
        minima.append(pos - half_extents)
        maxima.append(pos + half_extents)
        evidence.append(f"compiled_geom:{geom_id}:type={int(model.geom_type[geom_id])}")
    return BodyGeometryBounds(
        body_name=resolved,
        minimum=np.min(np.vstack(minima), axis=0),
        maximum=np.max(np.vstack(maxima), axis=0),
        evidence=tuple(evidence),
    )


def infer_distal_hand_site(
    adapter,
    body_name: str,
    *,
    semantic_name: str,
    axis: int = 0,
    direction: float = 1.0,
) -> GeometrySite:
    """Infer a distal hand position from the compiled body geometry bounds."""

    bounds = body_geometry_bounds(adapter, body_name)
    pos = bounds.center.copy()
    pos[axis] = bounds.maximum[axis] if direction >= 0.0 else bounds.minimum[axis]
    enforce_nonzero_origin(semantic_name, pos, source="geometry_bounds")
    return GeometrySite(
        semantic_name=semantic_name,
        body_name=bounds.body_name,
        local_position=pos,
        local_rotation_xyzw=_IDENTITY_QUAT_XYZW.copy(),
        source="geometry_bounds",
        confidence=0.85,
        evidence=("distal_hand_axis_bounds", *bounds.evidence),
    )


def infer_foot_sites(
    adapter,
    body_name: str,
    *,
    side_prefix: str,
    forward_axis: int = 0,
    up_axis: int = 2,
) -> dict[str, GeometrySite]:
    """Infer sole center, toe, and heel body-local sites from foot bounds."""

    bounds = body_geometry_bounds(adapter, body_name)
    sole = bounds.center.copy()
    sole[up_axis] = bounds.minimum[up_axis]
    toe = sole.copy()
    heel = sole.copy()
    toe[forward_axis] = bounds.maximum[forward_axis]
    heel[forward_axis] = bounds.minimum[forward_axis]

    names = {
        f"{side_prefix}Foot": sole,
        f"{side_prefix}Toe": toe,
        f"{side_prefix}Heel": heel,
    }
    sites: dict[str, GeometrySite] = {}
    for semantic_name, pos in names.items():
        enforce_nonzero_origin(semantic_name, pos, source="geometry_bounds")
        role = semantic_name.removeprefix(side_prefix).lower()
        sites[semantic_name] = GeometrySite(
            semantic_name=semantic_name,
            body_name=bounds.body_name,
            local_position=pos,
            local_rotation_xyzw=_IDENTITY_QUAT_XYZW.copy(),
            source="geometry_bounds",
            confidence=0.85,
            evidence=(f"{role}_bounds", *bounds.evidence),
        )
    return sites


def enforce_nonzero_origin(
    semantic_name: str,
    local_position,
    *,
    source: str,
    atol: float = 1e-9,
) -> None:
    pos = np.asarray(local_position, dtype=float)
    if float(np.linalg.norm(pos)) <= atol:
        raise ValueError(
            f"{semantic_name} site from {source} is at the body origin; "
            "distal hand/sole/toe/heel sites require nonzero local geometry evidence"
        )


def _geom_half_extents(model, geom_id: int) -> np.ndarray:
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    geom_type = int(model.geom_type[geom_id])
    sphere = _enum_value("mjGEOM_SPHERE", default=2)
    capsule = _enum_value("mjGEOM_CAPSULE", default=3)
    ellipsoid = _enum_value("mjGEOM_ELLIPSOID", default=4)
    cylinder = _enum_value("mjGEOM_CYLINDER", default=5)
    box = _enum_value("mjGEOM_BOX", default=6)

    if geom_type == sphere:
        return np.repeat(size[0], 3)
    if geom_type in {box, ellipsoid}:
        return np.maximum(size[:3], 0.0)
    if geom_type in {capsule, cylinder}:
        radius = max(float(size[0]), 0.0)
        half_length = max(float(size[1]), 0.0)
        return np.asarray([radius, radius, half_length + radius], dtype=float)
    # Mesh and plugin geoms do not expose a portable local AABB here. MuJoCo
    # still provides a compiled size vector; use it as a conservative extent.
    if np.any(size[:3] > 0.0):
        return np.maximum(size[:3], 0.0)
    radius = float(getattr(model, "geom_rbound", np.zeros(geom_id + 1))[geom_id])
    return np.repeat(max(radius, 0.0), 3)


def _enum_value(name: str, *, default: int) -> int:
    try:
        import mujoco

        return int(getattr(mujoco.mjtGeom, name))
    except Exception:
        return default
