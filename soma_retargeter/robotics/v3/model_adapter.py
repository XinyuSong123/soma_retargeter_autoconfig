"""Runtime-model adapter backed by MuJoCo's compiled kinematics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
import hashlib
import re
import tempfile
import warnings
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .model_fingerprint import model_fingerprint_payload, sha256_file
from .spatial import relative_transform, transform


_JOINT_TYPES = {
    mujoco.mjtJoint.mjJNT_FREE: "free",
    mujoco.mjtJoint.mjJNT_BALL: "ball",
    mujoco.mjtJoint.mjJNT_SLIDE: "prismatic",
    mujoco.mjtJoint.mjJNT_HINGE: "revolute",
}

_NEWTON_JOINT_TYPES = {
    0: "prismatic",
    1: "revolute",
    2: "ball",
    3: "fixed",
    4: "free",
    5: "distance",
    6: "d6",
    7: "cable",
}
_NEWTON_WORLD_BODY_ID = -1


@dataclass(frozen=True)
class SemanticSite:
    semantic_name: str
    body_name: str
    local_position: np.ndarray
    local_rotation_xyzw: np.ndarray
    source: str = "explicit_semantic_override"
    confidence: float = 1.0
    reason: str = "configured"
    evidence: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "semantic_name": self.semantic_name,
            "body_name": self.body_name,
            "local_position": self.local_position.tolist(),
            "local_rotation_xyzw": self.local_rotation_xyzw.tolist(),
            "source": self.source,
            "confidence": self.confidence,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class RobotKinematicState:
    q: np.ndarray
    body_xpos: np.ndarray
    body_xmat: np.ndarray


@dataclass(frozen=True)
class CoordinateInfo:
    index: int
    label: str
    joint_name: str
    joint_type: str
    qpos_adr: int
    dof_adr: int
    limited: bool
    lower: float
    upper: float

    def to_json(self) -> dict:
        return self.__dict__.copy()


class RuntimeRobotModelAdapter(Protocol):
    model_format: str
    nq: int
    nv: int

    def neutral_q(self) -> np.ndarray: ...
    def integrate(self, q: np.ndarray, delta_v: np.ndarray) -> np.ndarray: ...
    def forward_kinematics(self, q: np.ndarray) -> RobotKinematicState: ...
    def body_transform(self, state: RobotKinematicState, body_name: str) -> np.ndarray: ...
    def site_transform(self, state: RobotKinematicState, site: SemanticSite) -> np.ndarray: ...
    def active_velocity_coordinates(self, reference: SemanticSite, target: SemanticSite) -> list[int]: ...


def normalize_label(name: str | None) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class MuJoCoRuntimeModelAdapter:
    """Adapter whose topology, dimensions, FK and integration come from MuJoCo."""

    def __init__(self, model_path: str | Path, model_format: str | None = None):
        self.model_path = Path(model_path)
        self.model_format = model_format or self.model_path.suffix.lstrip(".").lower()
        self.loader_provenance: dict = {
            "backend": "mujoco",
            "load_path": str(self.model_path),
            "temporary_load_copy": None,
            "patches": [],
        }
        self.site_frame_provenance: dict[str, dict] = {}
        self.model = self._load_model(self.model_path)
        self._data = mujoco.MjData(self.model)
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        self._body_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            for i in range(self.model.nbody)
        ]
        self._body_id_by_name = {name: i for i, name in enumerate(self._body_names)}
        self._norm_body_id = {normalize_label(name): i for i, name in enumerate(self._body_names)}
        self._site_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i) or f"site_{i}"
            for i in range(self.model.nsite)
        ]
        self._site_id_by_name = {name: i for i, name in enumerate(self._site_names)}
        self._norm_site_id = {normalize_label(name): i for i, name in enumerate(self._site_names)}
        self._coordinate_info = self._build_coordinate_info()
        self.fingerprint_details = model_fingerprint_payload(
            self.model_path,
            backend="mujoco",
            model_format=self.model_format,
            loader_name="mujoco",
            loader_version=getattr(mujoco, "__version__", None),
            loader_provenance=self.loader_provenance,
            compiled_summary=self._compiled_summary(),
        )

    @property
    def fingerprint(self) -> str:
        return str(self.fingerprint_details["sha256"])

    @property
    def body_names(self) -> list[str]:
        return list(self._body_names)

    @property
    def coordinate_info(self) -> list[CoordinateInfo]:
        return list(self._coordinate_info)

    def close(self) -> None:
        self._data = None
        self.model = None  # type: ignore[assignment]

    def neutral_q(self) -> np.ndarray:
        return np.asarray(self.model.qpos0, dtype=float).copy()

    def integrate(self, q: np.ndarray, delta_v: np.ndarray) -> np.ndarray:
        out = np.asarray(q, dtype=float).copy()
        delta = np.asarray(delta_v, dtype=float).copy()
        mujoco.mj_integratePos(self.model, out, delta, 1.0)
        return out

    def forward_kinematics(self, q: np.ndarray) -> RobotKinematicState:
        if self._data is None:
            self._data = mujoco.MjData(self.model)
        data = self._data
        data.qpos[:] = np.asarray(q, dtype=float)
        mujoco.mj_forward(self.model, data)
        return RobotKinematicState(
            q=np.asarray(q, dtype=float).copy(),
            body_xpos=np.asarray(data.xpos, dtype=float).copy(),
            body_xmat=np.asarray(data.xmat, dtype=float).reshape(self.model.nbody, 3, 3).copy(),
        )

    def resolve_body_name(self, body_name: str) -> str:
        if body_name in self._body_id_by_name:
            return body_name
        key = normalize_label(body_name)
        if key in self._norm_body_id:
            return self._body_names[self._norm_body_id[key]]
        raise KeyError(f"unknown body {body_name!r}")

    def body_id(self, body_name: str) -> int:
        return self._body_id_by_name[self.resolve_body_name(body_name)]

    def model_site_frame(self, site_name: str) -> tuple[str, np.ndarray, np.ndarray]:
        sid = self.site_id(site_name)
        body_id = int(self.model.site_bodyid[sid])
        site_quat_wxyz = np.asarray(self.model.site_quat[sid], dtype=float)
        self.site_frame_provenance[site_name] = {
            "source": "mujoco_compiled_site",
            "trusted_runtime_site": True,
            "body": self._body_names[body_id],
        }
        return (
            self._body_names[body_id],
            np.asarray(self.model.site_pos[sid], dtype=float).copy(),
            np.asarray([site_quat_wxyz[1], site_quat_wxyz[2], site_quat_wxyz[3], site_quat_wxyz[0]], dtype=float),
        )

    def site_id(self, site_name: str) -> int:
        if site_name in self._site_id_by_name:
            return self._site_id_by_name[site_name]
        key = normalize_label(site_name)
        if key in self._norm_site_id:
            return self._norm_site_id[key]
        raise KeyError(f"unknown model site {site_name!r}")

    def body_transform(self, state: RobotKinematicState, body_name: str) -> np.ndarray:
        bid = self.body_id(body_name)
        return transform(state.body_xpos[bid], state.body_xmat[bid])

    def site_transform(self, state: RobotKinematicState, site: SemanticSite) -> np.ndarray:
        body_t = self.body_transform(state, site.body_name)
        local_t = transform(site.local_position, _quat_xyzw_to_matrix(site.local_rotation_xyzw))
        return body_t @ local_t

    def relative_transform(self, state: RobotKinematicState, reference: SemanticSite, target: SemanticSite) -> np.ndarray:
        return relative_transform(self.site_transform(state, reference), self.site_transform(state, target))

    def active_velocity_coordinates(self, reference: SemanticSite, target: SemanticSite) -> list[int]:
        body_ids = self.body_path_ids(reference.body_name, target.body_name)
        lca_id = self.body_id(self.lca_body(reference.body_name, target.body_name))
        active: list[int] = []
        for bid in body_ids:
            if bid == lca_id:
                continue
            for jid in self._joints_on_body(bid):
                jtype = self.model.jnt_type[jid]
                adr = int(self.model.jnt_dofadr[jid])
                width = _joint_nv(jtype)
                active.extend(range(adr, adr + width))
        return sorted(set(active))

    def body_path_ids(self, reference_body: str, target_body: str) -> list[int]:
        ref = self.body_id(reference_body)
        tgt = self.body_id(target_body)
        ref_anc = self._ancestors(ref)
        tgt_anc = self._ancestors(tgt)
        tgt_set = set(tgt_anc)
        lca = next(b for b in ref_anc if b in tgt_set)
        ref_branch = ref_anc[: ref_anc.index(lca)]
        tgt_branch = tgt_anc[: tgt_anc.index(lca)]
        return ref_branch + [lca] + list(reversed(tgt_branch))

    def body_path(self, reference_body: str, target_body: str) -> list[str]:
        return [self._body_names[i] for i in self.body_path_ids(reference_body, target_body)]

    def lca_body(self, reference_body: str, target_body: str) -> str:
        ref_anc = self._ancestors(self.body_id(reference_body))
        tgt_anc = set(self._ancestors(self.body_id(target_body)))
        for bid in ref_anc:
            if bid in tgt_anc:
                return self._body_names[bid]
        return self._body_names[0]

    def coordinate(self, index: int) -> CoordinateInfo:
        return self._coordinate_info[index]

    def coordinate_limits(self, indices: list[int] | None = None) -> list[CoordinateInfo]:
        if indices is None:
            return self.coordinate_info
        return [self._coordinate_info[i] for i in indices]

    def set_velocity_coordinates(self, q: np.ndarray, indices: list[int], values: np.ndarray) -> np.ndarray:
        out = np.asarray(q, dtype=float).copy()
        delta = np.zeros(self.nv)
        for idx, value in zip(indices, values):
            current = self._velocity_coordinate_value(out, idx)
            delta[idx] = float(value) - current
        return self.integrate(out, delta)

    def _velocity_coordinate_value(self, q: np.ndarray, dof_idx: int) -> float:
        info = self._coordinate_info[dof_idx]
        if info.joint_type in {"revolute", "prismatic"}:
            return float(q[info.qpos_adr])
        return 0.0

    def _ancestors(self, body_id: int) -> list[int]:
        out = []
        cur = int(body_id)
        while cur >= 0:
            out.append(cur)
            parent = int(self.model.body_parentid[cur])
            if parent == cur or cur == 0:
                break
            cur = parent
        return out

    def _descendant_or_self_ids(self, body_id: int) -> set[int]:
        out = {body_id}
        changed = True
        while changed:
            changed = False
            for i in range(self.model.nbody):
                if i not in out and int(self.model.body_parentid[i]) in out:
                    out.add(i)
                    changed = True
        return out

    def _joints_on_body(self, body_id: int) -> list[int]:
        adr = int(self.model.body_jntadr[body_id])
        num = int(self.model.body_jntnum[body_id])
        if adr < 0 or num <= 0:
            return []
        return list(range(adr, adr + num))

    def _build_coordinate_info(self) -> list[CoordinateInfo]:
        infos: list[CoordinateInfo | None] = [None] * self.nv
        for jid in range(self.model.njnt):
            jtype = self.model.jnt_type[jid]
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"
            dof_adr = int(self.model.jnt_dofadr[jid])
            qpos_adr = int(self.model.jnt_qposadr[jid])
            width = _joint_nv(jtype)
            limited = bool(self.model.jnt_limited[jid])
            lower = float(self.model.jnt_range[jid, 0]) if limited else -np.inf
            upper = float(self.model.jnt_range[jid, 1]) if limited else np.inf
            for k in range(width):
                idx = dof_adr + k
                suffix = "" if width == 1 else f"[{k}]"
                infos[idx] = CoordinateInfo(
                    index=idx,
                    label=f"{joint_name}{suffix}",
                    joint_name=joint_name,
                    joint_type=_JOINT_TYPES[jtype],
                    qpos_adr=qpos_adr,
                    dof_adr=dof_adr,
                    limited=limited and width == 1,
                    lower=lower if width == 1 else -np.inf,
                    upper=upper if width == 1 else np.inf,
                )
        if any(x is None for x in infos):
            missing = [i for i, x in enumerate(infos) if x is None]
            raise RuntimeError(f"missing coordinate metadata for dofs {missing}")
        return [x for x in infos if x is not None]

    def _load_model(self, model_path: Path):
        try:
            return mujoco.MjModel.from_xml_path(str(model_path))
        except ValueError as exc:
            text = model_path.read_text()
            patch = _patch_xml_for_mujoco_loading(text, model_path)
            if patch.text == text:
                raise
            with tempfile.NamedTemporaryFile("w", suffix=model_path.suffix, dir=str(model_path.parent), delete=False) as f:
                f.write(patch.text)
                tmp = Path(f.name)
            self.loader_provenance["temporary_load_copy"] = {
                "used": True,
                "directory": "model_parent",
                "suffix": model_path.suffix,
                "note": "ephemeral path intentionally omitted from deterministic fingerprint",
            }
            self.loader_provenance["patches"] = patch.provenance
            try:
                return mujoco.MjModel.from_xml_path(str(tmp))
            except Exception:
                raise exc
            finally:
                tmp.unlink(missing_ok=True)

    def _compiled_summary(self) -> dict:
        return {
            "nq": self.nq,
            "nv": self.nv,
            "nbody": int(self.model.nbody),
            "njnt": int(self.model.njnt),
            "nsite": int(self.model.nsite),
            "body_names": self._body_names,
            "site_names": self._site_names,
            "coordinates": [info.to_json() for info in self._coordinate_info],
            "qpos0_sha256": hashlib.sha256(np.asarray(self.model.qpos0, dtype=float).tobytes()).hexdigest(),
        }


def _joint_nv(jtype: int) -> int:
    if jtype == mujoco.mjtJoint.mjJNT_FREE:
        return 6
    if jtype == mujoco.mjtJoint.mjJNT_BALL:
        return 3
    return 1


@dataclass(frozen=True)
class _XmlPatchResult:
    text: str
    provenance: list[dict]


def _patch_xml_for_mujoco_loading(text: str, model_path: Path) -> _XmlPatchResult:
    provenance: list[dict] = []
    patched = text.replace('solver="cg"', 'solver="CG"')
    if patched != text:
        provenance.append({"type": "mujoco_solver_case", "from": 'solver="cg"', "to": 'solver="CG"'})
    if model_path.suffix.lower() == ".urdf":
        patched, mesh_provenance = _absolutize_urdf_meshes(patched, model_path.parent)
        provenance.extend(mesh_provenance)
    return _XmlPatchResult(patched, provenance)


def _patch_xml_for_newton_loading(text: str, model_path: Path) -> _XmlPatchResult:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return _XmlPatchResult(text, [])
    provenance: list[dict] = []
    for option in root.findall(".//option"):
        solver = option.attrib.get("solver")
        if solver is None or solver.strip().lstrip("+-").isdigit():
            continue
        del option.attrib["solver"]
        provenance.append(
            {
                "type": "newton_unsupported_mjcf_option",
                "element": "option",
                "attribute": "solver",
                "value": solver,
                "note": "Newton's MJCF loader rejects named solver enums; solver selection is a dynamics option and is omitted only in the temporary kinematic load copy.",
            }
        )
    assetdir = root.find("compiler").attrib.get("assetdir") if root.find("compiler") is not None else None
    if assetdir:
        asset_root = model_path.parent / assetdir
        for asset in root.findall(".//asset/*"):
            filename = asset.attrib.get("file")
            if not filename or Path(filename).is_absolute() or Path(filename).parts[:1] == Path(assetdir).parts[:1]:
                continue
            direct = model_path.parent / filename
            declared = asset_root / filename
            if direct.exists() or not declared.exists():
                continue
            patched = Path(assetdir) / filename
            asset.set("file", patched.as_posix())
            provenance.append(
                {
                    "type": "newton_mjcf_assetdir_file",
                    "element": asset.tag,
                    "attribute": "file",
                    "from": filename,
                    "to": patched.as_posix(),
                    "note": "Newton's MJCF loader does not honor compiler assetdir; asset paths are rewritten only in the temporary kinematic load copy.",
                }
            )
    if not provenance:
        return _XmlPatchResult(text, [])
    return _XmlPatchResult(ET.tostring(root, encoding="unicode"), provenance)


def _absolutize_urdf_meshes(text: str, base_dir: Path) -> tuple[str, list[dict]]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return text, []
    changed = False
    provenance: list[dict] = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename or Path(filename).is_absolute():
            continue
        path = _resolve_urdf_mesh_path(filename, base_dir)
        if path.exists():
            mesh.set("filename", str(path))
            changed = True
            provenance.append(
                {
                    "type": "temporary_urdf_mesh_abspath",
                    "original": filename,
                    "resolved": str(path),
                    "sha256": sha256_file(path),
                    "note": "applied only to temporary load copy",
                }
            )
    if not changed:
        return text, provenance
    return ET.tostring(root, encoding="unicode"), provenance


def _resolve_urdf_mesh_path(filename: str, base_dir: Path) -> Path:
    if filename.startswith("package://"):
        rest = filename.removeprefix("package://")
        parts = Path(rest).parts
        if not parts:
            return base_dir / rest
        package_name = parts[0]
        rel = Path(*parts[1:]) if len(parts) > 1 else Path()
        for parent in [base_dir, *base_dir.parents]:
            if parent.name == package_name:
                return (parent / rel).resolve()
            candidate = parent / package_name / rel
            if candidate.exists():
                return candidate.resolve()
        return (base_dir.parent / rel).resolve()
    return (base_dir / filename).resolve()


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    from .spatial import quat_xyzw_to_matrix

    return quat_xyzw_to_matrix(q)


def _quat_xyzw_batch_to_matrix(quats: np.ndarray) -> np.ndarray:
    q = np.asarray(quats, dtype=float)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"expected quaternions shaped (n, 4), got {q.shape}")
    norm = np.linalg.norm(q, axis=1)
    safe = np.where(norm > 0.0, norm, 1.0)
    q = q / safe[:, None]
    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    mats = np.empty((q.shape[0], 3, 3), dtype=float)
    mats[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    mats[:, 0, 1] = 2.0 * (xy - wz)
    mats[:, 0, 2] = 2.0 * (xz + wy)
    mats[:, 1, 0] = 2.0 * (xy + wz)
    mats[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    mats[:, 1, 2] = 2.0 * (yz - wx)
    mats[:, 2, 0] = 2.0 * (xz - wy)
    mats[:, 2, 1] = 2.0 * (yz + wx)
    mats[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return mats


class NewtonRuntimeModelAdapter:
    """Adapter backed by Newton's compiled ModelBuilder model on CPU."""

    def __init__(self, model_path: str | Path, model_format: str | None = None):
        import newton

        self._newton = newton
        self.model_path = Path(model_path)
        self.model_format = model_format or self.model_path.suffix.lstrip(".").lower()
        self.loader_provenance: dict = {
            "backend": "newton",
            "load_path": str(self.model_path),
            "temporary_load_copy": None,
            "patches": [],
        }
        self.site_frame_provenance: dict[str, dict] = {}
        builder = self._load_builder()
        self.model = builder.finalize(device="cpu")
        self.nq = int(self.model.joint_coord_count)
        self.nv = int(self.model.joint_dof_count)
        self._body_labels = list(self.model.body_label)
        self._body_names = [_leaf_label(label) for label in self._body_labels]
        self._body_id_by_name = {name: i for i, name in enumerate(self._body_names)}
        self._body_id_by_name.update({label: i for i, label in enumerate(self._body_labels)})
        self._body_id_by_name["world"] = _NEWTON_WORLD_BODY_ID
        self._norm_body_id = {normalize_label(name): i for i, name in enumerate(self._body_names)}
        self._norm_body_id.update({normalize_label(label): i for i, label in enumerate(self._body_labels)})
        self._norm_body_id["world"] = _NEWTON_WORLD_BODY_ID
        self._joint_type = _to_numpy(self.model.joint_type).astype(int)
        self._joint_child = _to_numpy(self.model.joint_child).astype(int)
        self._joint_parent = _to_numpy(self.model.joint_parent).astype(int)
        self._joint_q_start = _to_numpy(self.model.joint_q_start).astype(int)
        self._joint_qd_start = _to_numpy(self.model.joint_qd_start).astype(int)
        self._joint_limit_lower = _to_numpy(self.model.joint_limit_lower)
        self._joint_limit_upper = _to_numpy(self.model.joint_limit_upper)
        self._coordinate_info = self._build_coordinate_info()
        self._fk_state = self.model.state()
        self._zero_qd_wp = None
        self.fingerprint_details = model_fingerprint_payload(
            self.model_path,
            backend="newton",
            model_format=self.model_format,
            loader_name="newton",
            loader_version=getattr(newton, "__version__", None),
            loader_provenance=self.loader_provenance,
            compiled_summary=self._compiled_summary(),
        )

    @property
    def fingerprint(self) -> str:
        return str(self.fingerprint_details["sha256"])

    @property
    def body_names(self) -> list[str]:
        return ["world", *self._body_names]

    @property
    def coordinate_info(self) -> list[CoordinateInfo]:
        return list(self._coordinate_info)

    def close(self) -> None:
        self._fk_state = None
        self._zero_qd_wp = None
        self.model = None  # type: ignore[assignment]

    def neutral_q(self) -> np.ndarray:
        return _to_numpy(self.model.joint_q).astype(float).copy()

    def integrate(self, q: np.ndarray, delta_v: np.ndarray) -> np.ndarray:
        out = np.asarray(q, dtype=float).copy()
        delta = np.asarray(delta_v, dtype=float)
        for jid in range(int(self.model.joint_count)):
            jtype = int(self._joint_type[jid])
            q_start = int(self._joint_q_start[jid])
            qd_start = int(self._joint_qd_start[jid])
            if jtype == 4:
                out[q_start : q_start + 3] += delta[qd_start : qd_start + 3]
                rot_delta = delta[qd_start + 3 : qd_start + 6]
                if np.linalg.norm(rot_delta) > 0.0:
                    current = Rotation.from_quat(out[q_start + 3 : q_start + 7])
                    out[q_start + 3 : q_start + 7] = (Rotation.from_rotvec(rot_delta) * current).as_quat()
                out[q_start + 3 : q_start + 7] = _normalize_quat_xyzw(out[q_start + 3 : q_start + 7])
            elif jtype in {0, 1}:
                out[q_start] += delta[qd_start]
            elif jtype == 2:
                rot_delta = delta[qd_start : qd_start + 3]
                if np.linalg.norm(rot_delta) > 0.0:
                    current = Rotation.from_quat(out[q_start : q_start + 4])
                    out[q_start : q_start + 4] = (Rotation.from_rotvec(rot_delta) * current).as_quat()
                out[q_start : q_start + 4] = _normalize_quat_xyzw(out[q_start : q_start + 4])
            elif jtype == 6:
                q_width = self._joint_position_width(jid)
                v_width = self._joint_velocity_width(jid)
                if q_width == v_width:
                    out[q_start : q_start + q_width] += delta[qd_start : qd_start + v_width]
        return out

    def forward_kinematics(self, q: np.ndarray) -> RobotKinematicState:
        import warp as wp

        if self._fk_state is None:
            self._fk_state = self.model.state()
        if self._zero_qd_wp is None:
            self._zero_qd_wp = wp.array(np.zeros(self.nv, dtype=np.float32), dtype=wp.float32, device="cpu")
        state = self._fk_state
        q_wp = wp.array(np.asarray(q, dtype=np.float32), dtype=wp.float32, device="cpu")
        self._newton.eval_fk(self.model, q_wp, self._zero_qd_wp, state)
        body_q = _to_numpy(state.body_q).astype(float)
        positions = body_q[:, :3]
        rotations = _quat_xyzw_batch_to_matrix(body_q[:, 3:7])
        return RobotKinematicState(q=np.asarray(q, dtype=float).copy(), body_xpos=positions, body_xmat=rotations)

    def resolve_body_name(self, body_name: str) -> str:
        if body_name in self._body_id_by_name:
            return self._body_name_from_id(self._body_id_by_name[body_name])
        key = normalize_label(body_name)
        if key in self._norm_body_id:
            return self._body_name_from_id(self._norm_body_id[key])
        raise KeyError(f"unknown body {body_name!r}")

    def body_id(self, body_name: str) -> int:
        return self._body_id_by_name[self.resolve_body_name(body_name)]

    def model_site_frame(self, site_name: str) -> tuple[str, np.ndarray, np.ndarray]:
        compiled = _mujoco_compiled_site_frame(self.model_path, site_name)
        if compiled is not None:
            body_name, pos, quat = compiled
            resolved_body = self.resolve_body_name(body_name)
            self.site_frame_provenance[site_name] = {
                "source": "mujoco_compiled_site_crosscheck",
                "trusted_runtime_site": True,
                "body": resolved_body,
                "note": "Newton does not expose compiled sites; MuJoCo compiled model is used as reference truth for the same source file.",
            }
            warnings.warn(
                "Newton does not expose compiled model sites; using a MuJoCo compiled site cross-check for site truth.",
                RuntimeWarning,
                stacklevel=2,
            )
            return resolved_body, pos, quat
        frame = _mjcf_site_frame_from_xml(self.model_path, site_name)
        if frame is None:
            raise KeyError(f"unknown model site {site_name!r}")
        body_name, pos, quat = frame
        resolved_body = self.resolve_body_name(body_name)
        self.site_frame_provenance[site_name] = {
            "source": "raw_xml_site_fallback",
            "trusted_runtime_site": False,
            "body": resolved_body,
            "note": "Newton and MuJoCo compiled site truth were unavailable; this is provenance only, not compiled FK truth.",
        }
        warnings.warn(
            "Newton does not expose compiled model sites and MuJoCo cross-check failed; using untrusted raw XML site provenance.",
            RuntimeWarning,
            stacklevel=2,
        )
        return resolved_body, pos, quat

    def body_transform(self, state: RobotKinematicState, body_name: str) -> np.ndarray:
        bid = self.body_id(body_name)
        if bid == _NEWTON_WORLD_BODY_ID:
            return np.eye(4)
        return transform(state.body_xpos[bid], state.body_xmat[bid])

    def site_transform(self, state: RobotKinematicState, site: SemanticSite) -> np.ndarray:
        body_t = self.body_transform(state, site.body_name)
        local_t = transform(site.local_position, _quat_xyzw_to_matrix(site.local_rotation_xyzw))
        return body_t @ local_t

    def relative_transform(self, state: RobotKinematicState, reference: SemanticSite, target: SemanticSite) -> np.ndarray:
        return relative_transform(self.site_transform(state, reference), self.site_transform(state, target))

    def active_velocity_coordinates(self, reference: SemanticSite, target: SemanticSite) -> list[int]:
        body_ids = self.body_path_ids(reference.body_name, target.body_name)
        lca_id = self.body_id(self.lca_body(reference.body_name, target.body_name))
        active: list[int] = []
        for bid in body_ids:
            if bid == lca_id:
                continue
            for jid in self._joints_on_body(bid):
                adr = int(self._joint_qd_start[jid])
                width = self._joint_velocity_width(jid)
                active.extend(range(adr, adr + width))
        return sorted(set(active))

    def body_path_ids(self, reference_body: str, target_body: str) -> list[int]:
        ref = self.body_id(reference_body)
        tgt = self.body_id(target_body)
        ref_anc = self._ancestors(ref)
        tgt_anc = self._ancestors(tgt)
        tgt_set = set(tgt_anc)
        lca = next(b for b in ref_anc if b in tgt_set)
        ref_branch = ref_anc[: ref_anc.index(lca)]
        tgt_branch = tgt_anc[: tgt_anc.index(lca)]
        return ref_branch + [lca] + list(reversed(tgt_branch))

    def body_path(self, reference_body: str, target_body: str) -> list[str]:
        return [self._body_name_from_id(i) for i in self.body_path_ids(reference_body, target_body)]

    def lca_body(self, reference_body: str, target_body: str) -> str:
        ref_anc = self._ancestors(self.body_id(reference_body))
        tgt_anc = set(self._ancestors(self.body_id(target_body)))
        for bid in ref_anc:
            if bid in tgt_anc:
                return self._body_name_from_id(bid)
        return "world"

    def coordinate(self, index: int) -> CoordinateInfo:
        return self._coordinate_info[index]

    def coordinate_limits(self, indices: list[int] | None = None) -> list[CoordinateInfo]:
        if indices is None:
            return self.coordinate_info
        return [self._coordinate_info[i] for i in indices]

    def set_velocity_coordinates(self, q: np.ndarray, indices: list[int], values: np.ndarray) -> np.ndarray:
        out = np.asarray(q, dtype=float).copy()
        delta = np.zeros(self.nv)
        for idx, value in zip(indices, values):
            current = self._velocity_coordinate_value(out, idx)
            delta[idx] = float(value) - current
        return self.integrate(out, delta)

    def _velocity_coordinate_value(self, q: np.ndarray, dof_idx: int) -> float:
        info = self._coordinate_info[dof_idx]
        if info.joint_type in {"revolute", "prismatic"}:
            return float(q[info.qpos_adr])
        return 0.0

    def _ancestors(self, body_id: int) -> list[int]:
        if body_id == _NEWTON_WORLD_BODY_ID:
            return [_NEWTON_WORLD_BODY_ID]
        out = []
        cur = int(body_id)
        while cur >= 0:
            out.append(cur)
            parent = self._parent_body(cur)
            if parent < 0:
                out.append(_NEWTON_WORLD_BODY_ID)
                break
            cur = parent
        return out

    def _parent_body(self, body_id: int) -> int:
        for jid in self._joints_on_body(body_id):
            return int(self._joint_parent[jid])
        return -1

    def _joints_on_body(self, body_id: int) -> list[int]:
        if body_id == _NEWTON_WORLD_BODY_ID:
            return []
        return [jid for jid, child in enumerate(self._joint_child) if int(child) == int(body_id)]

    def _body_name_from_id(self, body_id: int) -> str:
        if body_id == _NEWTON_WORLD_BODY_ID:
            return "world"
        return self._body_names[body_id]

    def _load_builder(self):
        def load(path: Path):
            builder = self._newton.ModelBuilder()
            if self.model_format == "urdf":
                builder.add_urdf(str(path))
            else:
                builder.add_mjcf(str(path))
            return builder

        try:
            return load(self.model_path)
        except ValueError as exc:
            if self.model_format == "urdf":
                raise
            text = self.model_path.read_text()
            patch = _patch_xml_for_newton_loading(text, self.model_path)
            if patch.text == text:
                raise
            with tempfile.NamedTemporaryFile("w", suffix=self.model_path.suffix, dir=str(self.model_path.parent), delete=False) as f:
                f.write(patch.text)
                tmp = Path(f.name)
            self.loader_provenance["temporary_load_copy"] = {
                "used": True,
                "directory": "model_parent",
                "suffix": self.model_path.suffix,
                "note": "ephemeral path intentionally omitted from deterministic fingerprint",
            }
            self.loader_provenance["patches"] = patch.provenance
            try:
                return load(tmp)
            except Exception:
                raise exc
            finally:
                tmp.unlink(missing_ok=True)

    def _build_coordinate_info(self) -> list[CoordinateInfo]:
        infos: list[CoordinateInfo | None] = [None] * self.nv
        for jid, label in enumerate(self.model.joint_label):
            jtype = int(self._joint_type[jid])
            joint_type = _NEWTON_JOINT_TYPES.get(jtype, f"joint_type_{jtype}")
            qpos_adr = int(self._joint_q_start[jid])
            dof_adr = int(self._joint_qd_start[jid])
            width = self._joint_velocity_width(jid)
            joint_name = _leaf_label(label)
            for k in range(width):
                idx = dof_adr + k
                if idx < 0:
                    continue
                limited = jtype in {0, 1} and abs(self._joint_limit_lower[idx]) < 1e9 and abs(self._joint_limit_upper[idx]) < 1e9
                suffix = "" if width == 1 else f"[{k}]"
                infos[idx] = CoordinateInfo(
                    index=idx,
                    label=f"{joint_name}{suffix}",
                    joint_name=joint_name,
                    joint_type=joint_type,
                    qpos_adr=qpos_adr,
                    dof_adr=dof_adr,
                    limited=bool(limited),
                    lower=float(self._joint_limit_lower[idx]) if limited else -np.inf,
                    upper=float(self._joint_limit_upper[idx]) if limited else np.inf,
                )
        if any(x is None for x in infos):
            missing = [i for i, x in enumerate(infos) if x is None]
            raise RuntimeError(f"missing Newton coordinate metadata for dofs {missing}")
        return [x for x in infos if x is not None]

    def _joint_velocity_width(self, joint_id: int) -> int:
        jtype = int(self._joint_type[joint_id])
        fixed_width = _newton_joint_nv(jtype)
        if fixed_width > 0 or jtype in {3, 5, 7}:
            return fixed_width
        start = int(self._joint_qd_start[joint_id])
        if start < 0:
            return 0
        later_starts = [int(value) for value in self._joint_qd_start if int(value) > start]
        end = min(later_starts) if later_starts else self.nv
        return max(0, end - start)

    def _joint_position_width(self, joint_id: int) -> int:
        start = int(self._joint_q_start[joint_id])
        if start < 0:
            return 0
        later_starts = [int(value) for value in self._joint_q_start if int(value) > start]
        end = min(later_starts) if later_starts else self.nq
        return max(0, end - start)

    def _compiled_summary(self) -> dict:
        return {
            "nq": self.nq,
            "nv": self.nv,
            "nbody": int(self.model.body_count),
            "njnt": int(self.model.joint_count),
            "body_names": self._body_names,
            "body_labels": self._body_labels,
            "coordinates": [info.to_json() for info in self._coordinate_info],
            "neutral_q_sha256": hashlib.sha256(np.asarray(self.neutral_q(), dtype=float).tobytes()).hexdigest(),
        }


def _newton_joint_nv(jtype: int) -> int:
    if jtype == 4:
        return 6
    if jtype == 2:
        return 3
    if jtype in {0, 1}:
        return 1
    return 0


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _leaf_label(label: str) -> str:
    return str(label).split("/")[-1]


def _normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    quat = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        raise ValueError("cannot normalize zero quaternion")
    quat = quat / norm
    if quat[3] < 0.0:
        quat = -quat
    return quat


def _mjcf_site_frame_from_xml(model_path: Path, site_name: str) -> tuple[str, np.ndarray, np.ndarray] | None:
    if model_path.suffix.lower() not in {".xml", ".mjcf"}:
        return None
    try:
        root = ET.fromstring(model_path.read_text())
    except ET.ParseError:
        return None

    def visit_body(body: ET.Element) -> tuple[str, np.ndarray, np.ndarray] | None:
        body_name = body.attrib.get("name")
        if body_name:
            for site in body.findall("site"):
                candidate = site.attrib.get("name")
                if candidate == site_name or normalize_label(candidate) == normalize_label(site_name):
                    return body_name, _xml_vec(site.attrib.get("pos"), 3), _xml_quat_xyzw(site)
        for child in body.findall("body"):
            found = visit_body(child)
            if found is not None:
                return found
        return None

    worldbody = root.find("worldbody")
    if worldbody is None:
        return None
    for body in worldbody.findall("body"):
        found = visit_body(body)
        if found is not None:
            return found
    return None


def _mujoco_compiled_site_frame(model_path: Path, site_name: str) -> tuple[str, np.ndarray, np.ndarray] | None:
    try:
        adapter = MuJoCoRuntimeModelAdapter(model_path)
        return adapter.model_site_frame(site_name)
    except Exception:
        return None
    finally:
        try:
            adapter.close()  # type: ignore[name-defined]
        except Exception:
            pass


def _xml_vec(text: str | None, width: int) -> np.ndarray:
    if not text:
        return np.zeros(width)
    values = np.fromstring(text, sep=" ", dtype=float)
    if values.size != width:
        raise ValueError(f"expected {width} XML vector values, got {values.size}")
    return values


def _xml_quat_xyzw(element: ET.Element) -> np.ndarray:
    quat = element.attrib.get("quat")
    if quat:
        values = _xml_vec(quat, 4)
        return np.asarray([values[1], values[2], values[3], values[0]], dtype=float)
    return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
