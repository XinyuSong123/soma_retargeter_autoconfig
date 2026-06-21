"""Geometry helpers for distal semantic sites.

These helpers return body-local positions plus provenance metadata. They are
offline validation utilities; they do not decide whether a robot should be
classified as a complete humanoid.
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
    """Return conservative compiled body-local geometry bounds for one body."""

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
        minimum, maximum, geom_evidence = _compiled_geom_bounds(model, geom_id)
        minima.append(minimum)
        maxima.append(maximum)
        evidence.append(geom_evidence)
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
    """Infer a distal hand position from compiled body geometry bounds."""

    bounds = body_geometry_bounds(adapter, body_name)
    pos = bounds.center.copy()
    pos[axis] = bounds.maximum[axis] if direction >= 0.0 else bounds.minimum[axis]
    enforce_nonzero_origin(semantic_name, pos, source="compiled_geom_bounds")
    return GeometrySite(
        semantic_name=semantic_name,
        body_name=bounds.body_name,
        local_position=pos,
        local_rotation_xyzw=_IDENTITY_QUAT_XYZW.copy(),
        source="compiled_geom_bounds",
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

    positions = {
        f"{side_prefix}Foot": sole,
        f"{side_prefix}Toe": toe,
        f"{side_prefix}Heel": heel,
    }
    sites: dict[str, GeometrySite] = {}
    for semantic_name, pos in positions.items():
        enforce_nonzero_origin(semantic_name, pos, source="compiled_geom_bounds")
        role = semantic_name.removeprefix(side_prefix).lower()
        sites[semantic_name] = GeometrySite(
            semantic_name=semantic_name,
            body_name=bounds.body_name,
            local_position=pos,
            local_rotation_xyzw=_IDENTITY_QUAT_XYZW.copy(),
            source="compiled_geom_bounds",
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


def _compiled_geom_bounds(model, geom_id: int) -> tuple[np.ndarray, np.ndarray, str]:
    pos = np.asarray(model.geom_pos[geom_id], dtype=float)
    rot = _quat_wxyz_to_matrix(np.asarray(model.geom_quat[geom_id], dtype=float))
    vertices, evidence = _geom_vertices(model, geom_id)
    points = pos + (rot @ vertices.T).T
    return np.min(points, axis=0), np.max(points, axis=0), evidence


def _geom_vertices(model, geom_id: int) -> tuple[np.ndarray, str]:
    geom_type = int(model.geom_type[geom_id])
    geom_dataid = getattr(model, "geom_dataid", None)
    data_id = int(geom_dataid[geom_id]) if geom_dataid is not None else -1
    mesh_type = _enum_value("mjGEOM_MESH", default=7)
    if geom_type == mesh_type and data_id >= 0 and hasattr(model, "mesh_vert"):
        vert_adr = int(model.mesh_vertadr[data_id])
        vert_num = int(model.mesh_vertnum[data_id])
        if vert_num > 0:
            return (
                np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=float),
                f"compiled_mesh_vertices:{geom_id}",
            )

    half_extents = _geom_half_extents(model, geom_id)
    signs = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    return signs * half_extents, f"compiled_geom_extent:{geom_id}:type={geom_type}"


def _geom_half_extents(model, geom_id: int) -> np.ndarray:
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    geom_type = int(model.geom_type[geom_id])
    sphere = _enum_value("mjGEOM_SPHERE", default=2)
    capsule = _enum_value("mjGEOM_CAPSULE", default=3)
    ellipsoid = _enum_value("mjGEOM_ELLIPSOID", default=4)
    cylinder = _enum_value("mjGEOM_CYLINDER", default=5)
    box = _enum_value("mjGEOM_BOX", default=6)

    if geom_type == sphere:
        return np.repeat(max(float(size[0]), 0.0), 3)
    if geom_type in {box, ellipsoid}:
        return np.maximum(size[:3], 0.0)
    if geom_type in {capsule, cylinder}:
        radius = max(float(size[0]), 0.0)
        half_length = max(float(size[1]), 0.0)
        return np.asarray([radius, radius, half_length + radius], dtype=float)
    if np.any(size[:3] > 0.0):
        return np.maximum(size[:3], 0.0)
    radius = float(getattr(model, "geom_rbound", np.zeros(geom_id + 1))[geom_id])
    return np.repeat(max(radius, 0.0), 3)


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        return np.eye(3)
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _enum_value(name: str, *, default: int) -> int:
    try:
        import mujoco

        return int(getattr(mujoco.mjtGeom, name))
    except Exception:
        return default
