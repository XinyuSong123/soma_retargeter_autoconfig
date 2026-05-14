# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import pathlib
import sys
import traceback
from contextlib import redirect_stdout

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import newton
import numpy as np
import warp as wp

import soma_retargeter.robot_registry_parser as robot_registry_parser
from soma_retargeter.animation.skeleton import SkeletonInstance
import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.utils.newton_utils as newton_utils
import soma_retargeter.pipelines.utils as pipeline_utils
from soma_retargeter.foot_anchors import (
    find_capability_profile_path,
    generate_virtual_sole_anchors,
    load_capability_profile,
    resolve_capability_profile,
)
from soma_retargeter.tools.foot_grounding import FootGroundingOptions, ground_robot_pose_payload
from soma_retargeter.utils.warp_compat import install_cpu_pinned_memory_fallback
from app.optimize_scaler_config import optimize_scaler_from_pose_pairs
from soma_retargeter.renderers.skeleton_renderer import SkeletonRenderer
from soma_retargeter.tools.pose_io import (
    apply_robot_pose_to_model,
    create_reference_soma_skeleton_instance,
    load_human_pose_json,
    load_human_pose_payload,
    load_robot_pose_payload,
    save_human_pose_json,
)
from soma_retargeter.tools.pose_optimizer_core import (
    POSE_SLOT_LABELS,
    POSE_SLOT_ORDER,
    PairedPoseOptimizationDefaults,
    PoseSlotState,
    get_default_human_pose_files,
    get_default_output_config,
    get_default_pose_artifact_path,
    get_default_report_path,
    get_default_retargeter_config,
    get_default_robot_pose_files,
    get_default_scaler_config,
    get_default_source_reference_bvh,
    resolve_default_output_file,
    resolve_default_pose_artifact_file,
    resolve_default_report_file,
)


_UI_PANEL_MARGIN = 10
_UI_PANEL_ALPHA = 0.92
_UI_FONT_SIZE_PX = 18.0
_UI_FONT_CANDIDATES = (
    (pathlib.Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"), 0),
    (pathlib.Path("/usr/share/fonts/truetype/arphic/ukai.ttc"), 0),
)

_HUMAN_PREVIEW_COLOR = (58.0 / 255.0, 180.0 / 255.0, 168.0 / 255.0)
_HUMAN_SLOT_COLORS = {
    "t_pose": (0.96, 0.42, 0.42),
    "natural_down": (0.43, 0.83, 0.44),
    "both_arms_forward": (0.35, 0.60, 0.96),
    "both_elbows_forward_90": (0.94, 0.72, 0.28),
    "arms_forward_squat_hip_yaw_out_45": (0.72, 0.48, 0.90),
}
_HUMAN_SELECTED_SCENE_OFFSET = wp.vec3(0.0, 0.0, 0.0)
_ROBOT_SCENE_OFFSET = wp.vec3(2.4, 0.0, 0.0)
_VIRTUAL_ANCHOR_ORDER = ("sole_center", "toe", "heel", "inner_edge", "outer_edge")
_VIRTUAL_ANCHOR_LABELS = {
    "sole_center": "sole_center",
    "toe": "virtual toe",
    "heel": "virtual heel",
    "inner_edge": "inner_edge",
    "outer_edge": "outer_edge",
}
_VIRTUAL_ANCHOR_COLORS = {
    "sole_center": (1.00, 0.88, 0.15),
    "toe": (1.00, 0.20, 0.85),
    "heel": (0.10, 0.82, 1.00),
    "inner_edge": (0.20, 1.00, 0.35),
    "outer_edge": (1.00, 0.55, 0.12),
}
_CONTACT_ON_COLOR = (0.05, 1.00, 0.35)
_CONTACT_OFF_COLOR = (0.72, 0.72, 0.72)
_VIRTUAL_LINK_COLOR = (0.78, 0.60, 1.00)
_CONTACT_HEIGHT_THRESHOLD_M = 0.025
_SUPPORT_CONTACT_ANCHORS = ("toe", "heel", "inner_edge", "outer_edge")
_POSE_SLOT_LABELS_ZH = {
    "t_pose": "T姿态",
    "natural_down": "自然下垂",
    "both_arms_forward": "双臂前平举",
    "both_elbows_forward_90": "双肘前摆90度",
    "arms_forward_squat_hip_yaw_out_45": "双臂前伸下蹲",
}
_SLOT_STATUS_ZH = {
    "Empty": "空",
    "Preset": "标准模板",
    "Default": "默认",
    "Template": "待填写模板",
    "Loaded": "已加载",
    "Captured": "已采集",
}


def _robot_pose_payload_has_unfilled_values(payload: dict) -> bool:
    joint_positions = payload.get("joint_positions_rad")
    return isinstance(joint_positions, dict) and any(
        value is None for value in joint_positions.values()
    )


def _pose_slot_label_zh(slot_id: str) -> str:
    return _POSE_SLOT_LABELS_ZH.get(slot_id, POSE_SLOT_LABELS.get(slot_id, slot_id))


def _slot_status_zh(status: str) -> str:
    return _SLOT_STATUS_ZH.get(status, status)


def _find_ui_font_candidate() -> tuple[pathlib.Path, int] | None:
    for font_path, font_no in _UI_FONT_CANDIDATES:
        if font_path.exists():
            return font_path, font_no
    return None


def _transform_to_array(transform) -> np.ndarray:
    return np.asarray(transform, dtype=np.float32)


def _make_wp_transform(values) -> wp.transform:
    values = np.asarray(values, dtype=np.float32)
    return wp.transform(
        wp.vec3(*values[0:3].tolist()),
        wp.quat(*values[3:7].tolist()),
    )


def _quat_rotate_xyzw(quat_xyzw, point) -> np.ndarray:
    quat = normalize_np_quat_xyzw(quat_xyzw)
    q_vec = quat[0:3]
    point = np.asarray(point, dtype=np.float32)
    t = 2.0 * np.cross(q_vec, point)
    return point + quat[3] * t + np.cross(q_vec, t)


def normalize_np_quat_xyzw(quat_xyzw) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quat / norm


def _transform_point_xyzw(transform_values, local_point) -> np.ndarray:
    transform_values = np.asarray(transform_values, dtype=np.float32)
    return transform_values[0:3] + _quat_rotate_xyzw(transform_values[3:7], local_point)


def _clone_skeleton_instance(skeleton_instance: SkeletonInstance) -> SkeletonInstance:
    cloned = SkeletonInstance(
        skeleton_instance.skeleton,
        skeleton_instance.color,
        _make_wp_transform(_transform_to_array(skeleton_instance.xform)),
    )
    cloned.set_local_transforms(np.array(skeleton_instance.local_transforms, copy=True))
    return cloned


def _default_viewer_config_path(robot_type: str) -> pathlib.Path:
    path = robot_registry_parser.get_profile_path(robot_type, "bvh_converter_config")
    if path is not None:
        return path
    fallback = robot_registry_parser.get_generated_converter_config_path(robot_type)
    if fallback is not None:
        return fallback
    return io_utils.get_repo_root() / "soma_retargeter" / "configs" / robot_type / f"{robot_type}_bvh_to_csv_converter_config.json"


def _resolve_startup_paths(args):
    robot_type = robot_registry_parser.resolve_robot_name(args.robot)
    args.robot = robot_type
    args.config = str(args.config or _default_viewer_config_path(robot_type))
    default_scaler_config = get_default_scaler_config(robot_type)
    default_retargeter_config = get_default_retargeter_config(robot_type)
    default_source_reference_bvh = get_default_source_reference_bvh(robot_type)
    args.scaler_config = str(args.scaler_config or default_scaler_config or "")
    args.retargeter_config = str(args.retargeter_config or default_retargeter_config or "")
    args.source_reference_bvh = str(args.source_reference_bvh or default_source_reference_bvh or "")
    default_output = get_default_output_config(robot_type, args.scaler_config or None)
    args.output = str(args.output or default_output or "")
    args.report = str(args.report or (get_default_report_path(robot_type, args.output) if args.output else ""))
    args.pose_artifact = str(
        args.pose_artifact or (get_default_pose_artifact_path(robot_type, args.output) if args.output else "")
    )
    return args


class PoseOptimizerUI:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.viewer.vsync = True
        self.args = args
        self.robot_type = robot_registry_parser.resolve_robot_name(args.robot)
        self.robot_profile = robot_registry_parser.get_robot_profile(self.robot_type)
        self.robot_display_name = self.robot_type
        self.default_human_pose_files = get_default_human_pose_files(self.robot_type)
        self.default_robot_pose_files = get_default_robot_pose_files(self.robot_type)
        self.pose_registration_issues = robot_registry_parser.get_pose_pair_issues(self.robot_type)
        self.time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.logs: list[str] = []
        self.show_virtual_sole_overlay = True
        self.show_feet_contact_overlay = True
        self.virtual_overlay_status: dict[str, bool] = {"left": False, "right": False}
        self.virtual_overlay_messages: list[str] = []
        self.virtual_anchor_edit_side = "left"
        self.auto_ground_feet_preview = True
        self.foot_grounding_summary = ""

        self.config_path = str(args.config)
        self.scaler_config_path = str(args.scaler_config)
        self.retargeter_config_path = str(args.retargeter_config)
        self.source_reference_bvh = str(args.source_reference_bvh)

        self.output_file = str(
            args.output
            or (resolve_default_output_file(self.scaler_config_path) if self.scaler_config_path else "")
        )
        self.report_file = str(
            args.report
            or (resolve_default_report_file(self.output_file) if self.output_file else "")
        )
        self.pose_artifact_file = str(
            args.pose_artifact
            or (resolve_default_pose_artifact_file(self.output_file) if self.output_file else "")
        )

        defaults = PairedPoseOptimizationDefaults()
        self.iterations = int(defaults.iterations)
        self.learning_rate = float(defaults.learning_rate)
        self.rotation_iterations = int(defaults.iterations)
        self.rotation_learning_rate = float(defaults.learning_rate)
        self.rotation_position_weight = 1.0
        self.rotation_loss_weight = 1.0

        self.skeleton, self.human_editor = create_reference_soma_skeleton_instance(
            self.source_reference_bvh
        )
        self.skeleton_renderer = SkeletonRenderer(self.skeleton)

        self.human_slots = {
            slot_id: PoseSlotState(slot_id=slot_id, label=f"人体 {_pose_slot_label_zh(slot_id)}")
            for slot_id in POSE_SLOT_ORDER
        }
        self.robot_slots = {
            slot_id: PoseSlotState(slot_id=slot_id, label=f"机器人 {_pose_slot_label_zh(slot_id)}")
            for slot_id in POSE_SLOT_ORDER
        }
        self.training_slot_enabled = {slot_id: False for slot_id in POSE_SLOT_ORDER}
        self.display_slot_id: str | None = None
        self.preview_human_slot_id: str | None = None
        self.preview_robot_slot_id: str | None = None

        self._install_chinese_ui_font()
        self.initialize_default_human_slots()
        self._build_robot_model()
        self._initialize_virtual_foot_overlay()
        self.initialize_default_robot_slots()
        if self.scaler_config_path:
            self.append_log(f"已自动生成/使用 scaler 配置: {self.scaler_config_path}")
        if self.config_path:
            self.append_log(f"已自动生成/使用 converter 配置: {self.config_path}")
        if self.retargeter_config_path:
            self.append_log(f"IK 映射直接来自 retargeter 配置: {self.retargeter_config_path}")
        for issue in self.pose_registration_issues:
            self.append_log(f"[提示] {issue}")
        self.append_log(f"已加载人体参考骨架: {self.source_reference_bvh}")
        self.append_log(
            f"已加载 {self.robot_display_name} 运行时模型: "
            f"{pipeline_utils.get_robot_mjcf_path(self.robot_type)}"
        )

        self.viewer.renderer.set_title(f"{self.robot_display_name} 姿态优化器")
        self.viewer.register_ui_callback(lambda ui: self.gui(ui), position="free")

    def append_log(self, message: str) -> None:
        self.logs.append(message)
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]

    def _install_chinese_ui_font(self) -> None:
        ui = getattr(self.viewer, "ui", None)
        if ui is None or not getattr(ui, "is_available", False):
            return

        font_candidate = _find_ui_font_candidate()
        if font_candidate is None:
            self.append_log("[警告] 未找到可用中文字体，界面中文可能仍会显示异常。")
            return

        font_path, font_no = font_candidate
        try:
            font_cfg = ui.imgui.ImFontConfig()
            font_cfg.font_no = int(font_no)
            font = ui.io.fonts.add_font_from_file_ttf(str(font_path), _UI_FONT_SIZE_PX, font_cfg)
            if font is None:
                raise RuntimeError("字体加载返回空对象")
            ui.io.font_default = font
            self.append_log(f"已加载中文界面字体: {font_path}")
        except Exception as exc:
            self.append_log(f"[警告] 加载中文界面字体失败: {exc}")

    def _build_robot_model(self):
        robot_builder = newton.ModelBuilder()
        robot_builder.add_mjcf(str(pipeline_utils.get_robot_mjcf_path(self.robot_type)))

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_builder(robot_builder, wp.transform(_ROBOT_SCENE_OFFSET, wp.quat_identity()))

        self.robot_builder = robot_builder
        self.model = builder.finalize()
        self.state = self.model.state()
        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets([0, 0, 0])
        self.robot_default_joint_q = np.array(self.model.joint_q.numpy(), copy=True)

    def _initialize_virtual_foot_overlay(self) -> None:
        self.capability_profile = {}
        self.resolved_capability = {}
        self.virtual_sole_anchors = {"enabled": False}
        self.virtual_foot_link_names = {"left": None, "right": None}
        self.capability_profile_path = find_capability_profile_path(self.robot_type)
        if self.capability_profile_path is None:
            config_dir = robot_registry_parser.get_robot_config_dir(self.robot_type)
            if config_dir is not None:
                self.capability_profile_path = config_dir / "robot_capability.json"
        self.robot_body_name_to_index = {
            newton_utils.get_name_from_label(label): index
            for index, label in enumerate(self.robot_builder.body_label)
        }

        try:
            capability_profile, _ = load_capability_profile(self.robot_type)
            self.capability_profile = capability_profile
            resolution = resolve_capability_profile(self.robot_type, profile=capability_profile)
            resolution.raise_for_errors()
            self.resolved_capability = resolution.payload
            self.virtual_sole_anchors = generate_virtual_sole_anchors(
                self.robot_type,
                self.resolved_capability,
                profile=capability_profile,
            )
            links = self.resolved_capability.get("links", {})
            self.virtual_foot_link_names = {
                "left": links.get("left_foot"),
                "right": links.get("right_foot"),
            }
            self.virtual_overlay_messages = [
                *resolution.warnings,
                *self.virtual_sole_anchors.get("warnings", []),
            ]
            if self.virtual_sole_anchors.get("enabled"):
                self.append_log(
                    "已生成可视化虚拟足底锚点: sole_center, virtual toe, virtual heel, inner_edge, outer_edge"
                )
            else:
                self.append_log("[提示] 未生成虚拟足底锚点，可检查 robot_capability.json 的 foot 设置。")
        except Exception as exc:
            self.virtual_overlay_messages = [str(exc)]
            self.append_log(f"[警告] 虚拟足底/接触可视化初始化失败: {exc}")

    def _editable_anchor_payload(self) -> dict[str, dict[str, list[float]]]:
        return {
            side: {
                anchor_name: [
                    float(value)
                    for value in self.virtual_sole_anchors.get(side, {}).get(anchor_name, [0.0, 0.0, 0.0])
                ]
                for anchor_name in _VIRTUAL_ANCHOR_ORDER
            }
            for side in ("left", "right")
        }

    def _save_virtual_anchor_edits(self) -> None:
        if self.capability_profile_path is None:
            self.append_log("[错误] 无法解析 robot_capability.json 保存路径。")
            return

        profile_path = pathlib.Path(self.capability_profile_path)
        if profile_path.exists():
            profile = io_utils.load_json(profile_path)
        else:
            profile = {
                "schema_version": 1,
                "robot": self.robot_type,
                "capability_mode": "user_declared_with_auto_validation",
            }
        foot = profile.setdefault("foot", {})
        foot["sole_anchor_mode"] = "manual_anchors"
        foot["manual_anchors"] = self._editable_anchor_payload()
        io_utils.save_json(profile_path, profile, indent=4, ensure_ascii=False)
        self.append_log(f"已保存手动虚拟足底锚点: {profile_path}")
        self._initialize_virtual_foot_overlay()
        self._reground_loaded_robot_slots("manual virtual sole anchors")

    def _reset_virtual_anchors_to_auto(self) -> None:
        if self.capability_profile_path is None:
            self.append_log("[错误] 无法解析 robot_capability.json 保存路径。")
            return

        profile_path = pathlib.Path(self.capability_profile_path)
        if not profile_path.exists():
            self.append_log("[提示] 当前没有已保存的手动虚拟锚点。")
            self._initialize_virtual_foot_overlay()
            return

        profile = io_utils.load_json(profile_path)
        foot = profile.setdefault("foot", {})
        foot.pop("manual_anchors", None)
        if foot.get("sole_anchor_mode") == "manual_anchors":
            foot["sole_anchor_mode"] = "auto"
        io_utils.save_json(profile_path, profile, indent=4, ensure_ascii=False)
        self.append_log(f"已恢复自动虚拟足底锚点: {profile_path}")
        self._initialize_virtual_foot_overlay()
        self._reground_loaded_robot_slots("auto virtual sole anchors")

    def _render_virtual_anchor_editor(self, ui) -> None:
        if not self.virtual_sole_anchors.get("enabled"):
            ui.text_wrapped("当前没有可编辑的虚拟足底锚点。")
            return

        changed_side, side_index = ui.combo(
            "编辑脚",
            0 if self.virtual_anchor_edit_side == "left" else 1,
            ["left", "right"],
        )
        if changed_side:
            self.virtual_anchor_edit_side = "left" if side_index == 0 else "right"

        side = self.virtual_anchor_edit_side
        side_anchors = self.virtual_sole_anchors.setdefault(side, {})
        ui.text_wrapped("坐标是 foot link 局部坐标，单位 m。修改后场景里的点会立即移动。")
        for anchor_name in _VIRTUAL_ANCHOR_ORDER:
            color = _VIRTUAL_ANCHOR_COLORS[anchor_name]
            ui.text_colored(
                ui.ImVec4(color[0], color[1], color[2], 1.0),
                _VIRTUAL_ANCHOR_LABELS[anchor_name],
            )
            current = side_anchors.get(anchor_name, [0.0, 0.0, 0.0])
            current = [float(current[0]), float(current[1]), float(current[2])]
            ui.set_next_item_width(260)
            changed, updated = ui.input_float3(
                f"xyz##{side}_{anchor_name}",
                current,
                format="%.4f",
            )
            if changed:
                side_anchors[anchor_name] = [float(updated[0]), float(updated[1]), float(updated[2])]

        if ui.button("保存虚拟足底点"):
            self._save_virtual_anchor_edits()
        ui.same_line()
        if ui.button("恢复自动生成"):
            self._reset_virtual_anchors_to_auto()

    def initialize_default_human_slots(self):
        if not self.default_human_pose_files:
            self.append_log("[提示] 当前未加载到硬编码标准 human pose JSON，窗口仍会打开。")
            return

        loaded_slot_ids = []
        for slot_id in POSE_SLOT_ORDER:
            default_path = self.default_human_pose_files.get(slot_id)
            if default_path is None:
                continue
            default_path = pathlib.Path(default_path)
            if not default_path.exists():
                self.append_log(f"[警告] 缺少默认人体姿态文件 [{slot_id}]: {default_path}")
                continue

            try:
                record = load_human_pose_json(default_path, self.skeleton)
                slot = self.human_slots[slot_id]
                slot.status = "Preset"
                slot.path = str(default_path)
                slot.pose_name = record.pose_name
                slot.summary = f"{record.pose_name} | 关节={self.skeleton.num_joints}"
                slot.payload = record.payload
                loaded_slot_ids.append(slot_id)
            except Exception as exc:
                self.append_log(f"[警告] 加载默认人体姿态失败 [{slot_id}]: {exc}")

        if loaded_slot_ids:
            self.set_display_slot("t_pose" if "t_pose" in loaded_slot_ids else loaded_slot_ids[0], announce=False)
            loaded_labels = ", ".join(_pose_slot_label_zh(slot_id) for slot_id in loaded_slot_ids)
            self.append_log(f"已初始化标准人体姿态: {loaded_labels}")

    def _auto_ground_robot_pose_payload(
        self,
        slot_id: str,
        payload: dict,
        path: pathlib.Path | None,
    ) -> dict:
        if not self.auto_ground_feet_preview:
            return payload

        try:
            result = ground_robot_pose_payload(
                self.robot_type,
                payload,
                options=FootGroundingOptions(
                    contact_threshold_m=_CONTACT_HEIGHT_THRESHOLD_M,
                    max_joint_delta_rad=0.22,
                ),
                profile=self.capability_profile or None,
            )
        except Exception as exc:
            self.append_log(f"[警告] 自动足底贴地失败 [{_pose_slot_label_zh(slot_id)}]: {exc}")
            return payload

        status = "contact" if result.contact_success else "not contact"
        self.foot_grounding_summary = (
            f"{_pose_slot_label_zh(slot_id)} auto-ground: {status}, "
            f"max |z|={result.max_abs_support_anchor_z_m:.4f} m"
        )
        for message in result.messages:
            self.append_log(f"[提示] {message}")

        if not result.changed:
            return payload

        if path is not None:
            io_utils.save_json(path, result.payload, indent=2, ensure_ascii=False)
            self.append_log(
                f"已自动贴地并更新机器人姿态 [{_pose_slot_label_zh(slot_id)}]: {path} "
                f"(max |z|={result.max_abs_support_anchor_z_m:.4f} m, "
                f"tuned={', '.join(result.tuned_joint_deltas_rad.keys()) or 'root only'})"
            )
        else:
            self.append_log(
                f"已自动贴地机器人姿态 [{_pose_slot_label_zh(slot_id)}] "
                f"(max |z|={result.max_abs_support_anchor_z_m:.4f} m)"
            )
        return result.payload

    def _reground_loaded_robot_slots(self, reason: str) -> None:
        updated_slot_ids = []
        for slot_id, slot in self.robot_slots.items():
            if not slot.payload:
                continue
            path = pathlib.Path(slot.path) if slot.path else None
            payload = self._auto_ground_robot_pose_payload(slot_id, slot.payload, path)
            try:
                record = load_robot_pose_payload(payload, path)
            except Exception as exc:
                self.append_log(f"[警告] 重新加载贴地姿态失败 [{_pose_slot_label_zh(slot_id)}]: {exc}")
                continue
            slot.payload = record.payload
            slot.pose_name = record.pose_name
            slot.summary = f"{record.pose_name} | 关节={len(record.joint_positions_rad)}"
            updated_slot_ids.append(slot_id)

        if updated_slot_ids:
            labels = ", ".join(_pose_slot_label_zh(slot_id) for slot_id in updated_slot_ids)
            self.append_log(f"已用 {reason} 重新贴地已加载姿态: {labels}")
            if self.preview_robot_slot_id in updated_slot_ids:
                self._update_robot_preview_state()

    def initialize_default_robot_slots(self):
        if not self.default_robot_pose_files:
            self.append_log("[提示] params.py 中未注册任何机器人 pose JSON，窗口仍会打开。")
            return

        loaded_slot_ids = []
        for slot_id in POSE_SLOT_ORDER:
            default_path = self.default_robot_pose_files.get(slot_id)
            if default_path is None:
                continue
            default_path = pathlib.Path(default_path)
            if not default_path.exists():
                self.append_log(f"[警告] 缺少默认机器人姿态文件 [{slot_id}]: {default_path}")
                continue

            try:
                payload = io_utils.load_json(default_path)
                if _robot_pose_payload_has_unfilled_values(payload):
                    self._mark_robot_slot_as_template(slot_id, str(default_path), payload)
                    continue
                payload = self._auto_ground_robot_pose_payload(slot_id, payload, default_path)
                record = load_robot_pose_payload(payload, default_path)
                slot = self.robot_slots[slot_id]
                slot.status = "Default"
                slot.path = str(default_path)
                slot.pose_name = record.pose_name
                slot.summary = f"{record.pose_name} | 关节={len(record.joint_positions_rad)}"
                slot.payload = record.payload
                loaded_slot_ids.append(slot_id)
            except Exception as exc:
                self.append_log(f"[警告] 加载默认机器人姿态失败 [{slot_id}]: {exc}")

        if loaded_slot_ids:
            if self.display_slot_id in loaded_slot_ids:
                self.set_display_slot(self.display_slot_id, announce=False)
            else:
                self.set_display_slot(
                    "natural_down" if "natural_down" in loaded_slot_ids else loaded_slot_ids[0],
                    announce=False,
                )
            self._reset_training_selection_to_ready_slots()
            self.append_log(
                f"已加载默认 {self.robot_display_name} 机器人姿态: "
                + ", ".join(_pose_slot_label_zh(slot_id) for slot_id in loaded_slot_ids)
            )

    def _mark_robot_slot_as_template(self, slot_id: str, path: str, payload: dict) -> None:
        joint_positions = payload.get("joint_positions_rad", {})
        unfilled_count = sum(1 for value in joint_positions.values() if value is None)
        slot = self.robot_slots[slot_id]
        slot.status = "Template"
        slot.path = path
        slot.pose_name = payload.get("pose_name", pathlib.Path(path).stem)
        slot.summary = f"{slot.pose_name} | 待填写 {unfilled_count} 个机器人关节角"
        slot.payload = None
        self.training_slot_enabled[slot_id] = False
        if self.preview_robot_slot_id == slot_id:
            self.preview_robot_slot_id = None
        self.append_log(
            f"机器人姿态模板 [{_pose_slot_label_zh(slot_id)}] 还没填完，默认不参与训练: {path}"
        )

    def set_display_slot(self, slot_id: str, *, announce: bool = True) -> None:
        self.display_slot_id = slot_id
        self.preview_human_slot_id = slot_id if self.human_slots[slot_id].payload else None
        self.preview_robot_slot_id = slot_id if self.robot_slots[slot_id].payload else None
        if announce:
            self.append_log(f"当前显示分组已切换到: {_pose_slot_label_zh(slot_id)}")

    def _slot_is_ready_for_training(self, slot_id: str) -> bool:
        return bool(self.human_slots[slot_id].payload and self.robot_slots[slot_id].payload)

    def _selected_training_slot_ids(self) -> list[str]:
        return [slot_id for slot_id in POSE_SLOT_ORDER if self.training_slot_enabled.get(slot_id, False)]

    def _reset_training_selection_to_ready_slots(self) -> None:
        for slot_id in POSE_SLOT_ORDER:
            self.training_slot_enabled[slot_id] = self._slot_is_ready_for_training(slot_id)

    def _count_ready_slots(self) -> int:
        return sum(1 for slot_id in POSE_SLOT_ORDER if self._slot_is_ready_for_training(slot_id))

    def _training_summary_text(self) -> str:
        selected_count = len(self._selected_training_slot_ids())
        ready_count = self._count_ready_slots()
        return f"参与优化 {selected_count}/{len(POSE_SLOT_ORDER)} 组 | 已就绪 {ready_count}/{len(POSE_SLOT_ORDER)} 组"

    def _slot_compact_status(self, slot_id: str) -> str:
        human_status = _slot_status_zh(self.human_slots[slot_id].status)
        robot_status = _slot_status_zh(self.robot_slots[slot_id].status)
        ready_text = "就绪" if self._slot_is_ready_for_training(slot_id) else "未就绪"
        return f"Human {human_status} | Robot {robot_status} | {ready_text}"

    def _get_human_slot_record(self, slot_id: str):
        slot = self.human_slots[slot_id]
        if not slot.payload:
            return None
        if slot.path:
            return load_human_pose_json(slot.path, self.skeleton)
        return load_human_pose_payload(slot.payload, self.skeleton)

    def _get_robot_slot_record(self, slot_id: str):
        slot = self.robot_slots[slot_id]
        if not slot.payload:
            return None
        return load_robot_pose_payload(slot.payload, slot.path)

    def _update_robot_preview_state(self):
        wp.copy(self.model.joint_q, wp.array(self.robot_default_joint_q, dtype=wp.float32))
        if self.preview_robot_slot_id is not None:
            record = self._get_robot_slot_record(self.preview_robot_slot_id)
            if record is not None:
                try:
                    apply_robot_pose_to_model(self.model, self.robot_builder, record)
                except Exception as exc:
                    self.append_log(f"[错误] 机器人姿态预览失败 [{self.preview_robot_slot_id}]: {exc}")
                    self.preview_robot_slot_id = None
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

    def _compute_virtual_anchor_world_points(self) -> dict[str, dict[str, np.ndarray]]:
        if not self.virtual_sole_anchors.get("enabled"):
            return {}

        body_q = self.state.body_q.numpy()
        world_points: dict[str, dict[str, np.ndarray]] = {}
        for side in ("left", "right"):
            foot_link = self.virtual_foot_link_names.get(side)
            if foot_link not in self.robot_body_name_to_index:
                continue
            body_idx = self.robot_body_name_to_index[foot_link]
            foot_tx = body_q[body_idx]
            side_anchors = self.virtual_sole_anchors.get(side, {})
            points = {}
            for anchor_name in _VIRTUAL_ANCHOR_ORDER:
                local_point = side_anchors.get(anchor_name)
                if local_point is None:
                    continue
                points[anchor_name] = _transform_point_xyzw(foot_tx, local_point)
            if points:
                world_points[side] = points
        return world_points

    def _draw_virtual_foot_overlay(self):
        world_points = self._compute_virtual_anchor_world_points()
        if not world_points or not (self.show_virtual_sole_overlay or self.show_feet_contact_overlay):
            self.viewer.log_points("/contact_aware_foot_ik/virtual_sole_anchors", None)
            self.viewer.log_lines("/contact_aware_foot_ik/virtual_links", None, None, None)
            self.viewer.log_lines("/contact_aware_foot_ik/feet_contact_support", None, None, None)
            return

        point_positions = []
        point_colors = []
        virtual_link_starts = []
        virtual_link_ends = []
        support_starts = []
        support_ends = []
        support_colors = []
        body_q = self.state.body_q.numpy()
        self.virtual_overlay_status = {"left": False, "right": False}

        for side, points in world_points.items():
            foot_link = self.virtual_foot_link_names.get(side)
            body_idx = self.robot_body_name_to_index.get(foot_link) if foot_link else None
            foot_origin = body_q[body_idx][0:3] if body_idx is not None else None
            support_anchor_z = [
                abs(float(points[anchor_name][2]))
                for anchor_name in _SUPPORT_CONTACT_ANCHORS
                if anchor_name in points
            ]
            contact_on = bool(support_anchor_z) and max(support_anchor_z) <= _CONTACT_HEIGHT_THRESHOLD_M
            self.virtual_overlay_status[side] = contact_on

            if self.show_virtual_sole_overlay:
                for anchor_name in _VIRTUAL_ANCHOR_ORDER:
                    point = points.get(anchor_name)
                    if point is None:
                        continue
                    point_positions.append(point)
                    point_colors.append(_VIRTUAL_ANCHOR_COLORS[anchor_name])
                    if foot_origin is not None:
                        virtual_link_starts.append(foot_origin)
                        virtual_link_ends.append(point)

            if self.show_feet_contact_overlay:
                contact_color = _CONTACT_ON_COLOR if contact_on else _CONTACT_OFF_COLOR
                support_order = ("toe", "outer_edge", "heel", "inner_edge", "toe")
                for start_name, end_name in zip(support_order[:-1], support_order[1:], strict=False):
                    if start_name in points and end_name in points:
                        support_starts.append(points[start_name])
                        support_ends.append(points[end_name])
                        support_colors.append(contact_color)
                if "sole_center" in points:
                    ground_point = np.array(points["sole_center"], copy=True)
                    ground_point[2] = 0.0
                    support_starts.append(points["sole_center"])
                    support_ends.append(ground_point)
                    support_colors.append(contact_color)

        if self.show_virtual_sole_overlay and point_positions:
            anchor_points = wp.array(np.asarray(point_positions, dtype=np.float32), dtype=wp.vec3)
            self.viewer.log_points(
                "/contact_aware_foot_ik/virtual_sole_anchors",
                anchor_points,
                radii=wp.full(len(point_positions), 0.018, dtype=wp.float32, device=anchor_points.device),
                colors=wp.array(np.asarray(point_colors, dtype=np.float32), dtype=wp.vec3),
            )
            if virtual_link_starts:
                self.viewer.log_lines(
                    "/contact_aware_foot_ik/virtual_links",
                    wp.array(np.asarray(virtual_link_starts, dtype=np.float32), dtype=wp.vec3),
                    wp.array(np.asarray(virtual_link_ends, dtype=np.float32), dtype=wp.vec3),
                    _VIRTUAL_LINK_COLOR,
                    width=0.006,
                )
        else:
            self.viewer.log_points("/contact_aware_foot_ik/virtual_sole_anchors", None)
            self.viewer.log_lines("/contact_aware_foot_ik/virtual_links", None, None, None)

        if self.show_feet_contact_overlay and support_starts:
            self.viewer.log_lines(
                "/contact_aware_foot_ik/feet_contact_support",
                wp.array(np.asarray(support_starts, dtype=np.float32), dtype=wp.vec3),
                wp.array(np.asarray(support_ends, dtype=np.float32), dtype=wp.vec3),
                wp.array(np.asarray(support_colors, dtype=np.float32), dtype=wp.vec3),
                width=0.012,
            )
        else:
            self.viewer.log_lines("/contact_aware_foot_ik/feet_contact_support", None, None, None)

    def step(self):
        self.time += self.frame_dt
        self._update_robot_preview_state()

    def _draw_human_scene(self):
        if self.display_slot_id is None:
            return

        record = self._get_human_slot_record(self.display_slot_id)
        if record is None:
            return

        preview_instance = _clone_skeleton_instance(self.human_editor)
        preview_instance.xform = wp.transform(
            wp.vec3(*record.root_translation_m.tolist()),
            wp.quat(*record.root_rotation_xyzw.tolist()),
        )
        preview_instance.set_local_transforms(np.array(record.local_transforms, copy=True))
        preview_instance.xform = wp.mul(
            wp.transform(_HUMAN_SELECTED_SCENE_OFFSET, wp.quat_identity()),
            preview_instance.xform,
        )
        preview_instance.color = _HUMAN_SLOT_COLORS.get(self.display_slot_id, _HUMAN_PREVIEW_COLOR)
        self.skeleton_renderer.draw(self.viewer, preview_instance, 0)

    def render(self):
        self.viewer.begin_frame(self.time)
        self._draw_human_scene()
        self.viewer.log_state(self.state)
        self._draw_virtual_foot_overlay()
        self.viewer.end_frame()

    def _validate_slots_ready(self) -> tuple[bool, list[str]]:
        issues = []
        selected_slot_ids = self._selected_training_slot_ids()
        if not selected_slot_ids:
            if self._count_ready_slots() == 0:
                issues.append("当前没有就绪姿态组，请先在 params.py 的 POSE_PAIR_JSON_DICT 中注册机器人 pose JSON。")
            else:
                issues.append("请至少勾选 1 组训练姿态。")
        for slot_id in selected_slot_ids:
            if not self.human_slots[slot_id].payload:
                issues.append(f"人体槽位 [{_pose_slot_label_zh(slot_id)}] 为空。")
            if not self.robot_slots[slot_id].payload:
                issues.append(f"机器人槽位 [{_pose_slot_label_zh(slot_id)}] 为空。")
        if not self.scaler_config_path:
            issues.append("未能自动解析 scaler 配置路径。")
        elif not pathlib.Path(self.scaler_config_path).exists():
            issues.append("自动生成的 scaler 配置不存在，请检查 params.py 中的 retargeter 路径。")
        if not self.retargeter_config_path:
            issues.append("请在 params.py 的 RETARGETER_CONFIG_DICT 中注册 retargeter config 路径。")
        elif not pathlib.Path(self.retargeter_config_path).exists():
            issues.append("Retargeter 配置不存在，请检查 params.py 的 RETARGETER_CONFIG_DICT。")
        if not self.output_file:
            issues.append("未能自动解析输出 scaler config 路径。")
        if self.iterations <= 0:
            issues.append("迭代次数必须大于 0。")
        if self.learning_rate <= 0:
            issues.append("学习率必须大于 0。")
        return len(issues) == 0, issues

    def _materialize_pose_artifacts(self):
        output_dir = pathlib.Path(self.output_file).resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_root = output_dir / f"{pathlib.Path(self.output_file).stem}_pose_artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.pose_artifact_file).resolve().parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.report_file).resolve().parent.mkdir(parents=True, exist_ok=True)

        human_pose_paths = []
        robot_pose_paths = []
        selected_slot_ids = self._selected_training_slot_ids()
        for slot_id in selected_slot_ids:
            human_path = artifact_root / f"human_{slot_id}.json"
            record = self._get_human_slot_record(slot_id)
            temp_instance = _clone_skeleton_instance(self.human_editor)
            temp_instance.xform = wp.transform(
                wp.vec3(*record.root_translation_m.tolist()),
                wp.quat(*record.root_rotation_xyzw.tolist()),
            )
            temp_instance.set_local_transforms(np.array(record.local_transforms, copy=True))
            save_human_pose_json(
                human_path,
                temp_instance,
                record.pose_name,
                self.source_reference_bvh,
            )
            human_pose_paths.append(str(human_path))

            robot_path = artifact_root / f"robot_{slot_id}.json"
            with open(robot_path, "w", encoding="utf-8") as handle:
                json.dump(self.robot_slots[slot_id].payload, handle, indent=2, ensure_ascii=False)
            robot_pose_paths.append(str(robot_path))

        artifact_payload = {
            "source_reference_bvh": self.source_reference_bvh,
            "scaler_config_path": self.scaler_config_path,
            "retargeter_config_path": self.retargeter_config_path,
            "output_file": self.output_file,
            "report_file": self.report_file,
            "selected_pose_groups": [
                {
                    "slot_id": slot_id,
                    "label": _pose_slot_label_zh(slot_id),
                }
                for slot_id in selected_slot_ids
            ],
            "human_pose_files": human_pose_paths,
            "robot_pose_files": robot_pose_paths,
            "params": {
                "iterations": self.iterations,
                "learning_rate": self.learning_rate,
                "optimize_offset_rotation": True,
                "rotation_iterations": self.iterations,
                "rotation_learning_rate": self.learning_rate,
                "rotation_position_weight": 1.0,
                "rotation_loss_weight": 1.0,
                "rotation_joint_scope": "all_mapped_joints",
            },
        }
        with open(self.pose_artifact_file, "w", encoding="utf-8") as handle:
            json.dump(artifact_payload, handle, indent=2, ensure_ascii=False)
        return human_pose_paths, robot_pose_paths

    def run_optimization(self):
        ok, issues = self._validate_slots_ready()
        if not ok:
            for issue in issues:
                self.append_log(f"[错误] {issue}")
            return

        try:
            selected_slot_ids = self._selected_training_slot_ids()
            human_pose_paths, robot_pose_paths = self._materialize_pose_artifacts()
            self.append_log(
                "开始优化，使用姿态组: "
                + ", ".join(_pose_slot_label_zh(slot_id) for slot_id in selected_slot_ids)
            )
            stdout_buffer = io.StringIO()
            with redirect_stdout(stdout_buffer):
                result = optimize_scaler_from_pose_pairs(
                    config_file=self.scaler_config_path,
                    human_pose_files=human_pose_paths,
                    robot_pose_json_files=robot_pose_paths,
                    retargeter_config_file=self.retargeter_config_path,
                    source_reference_bvh=self.source_reference_bvh,
                    iterations=self.iterations,
                    learning_rate=self.learning_rate,
                    output_file=self.output_file,
                    optimize_offset_rotation=True,
                    rotation_iterations=self.iterations,
                    rotation_learning_rate=self.learning_rate,
                    rotation_position_weight=1.0,
                    rotation_loss_weight=1.0,
                    report_file=self.report_file,
                )
            captured_logs = stdout_buffer.getvalue().splitlines()
            for line in captured_logs[-80:]:
                self.append_log(line)
            scaler_reference = robot_registry_parser.link_scaler_config_to_retargeter(
                self.retargeter_config_path,
                result["output_file"],
            )
            self.scaler_config_path = result["output_file"]
            self.append_log(f"优化完成，输出配置: {result['output_file']}")
            self.append_log(f"运行时 scaler 引用: {scaler_reference}")
            self.append_log(f"运行报告: {result['report_file']}")
            self.append_log(f"姿态产物: {self.pose_artifact_file}")
        except Exception:
            self.append_log("[错误] 优化失败。")
            for line in traceback.format_exc().splitlines():
                self.append_log(line)

    def _render_human_pose_editor(self, ui, width: float, height: float):
        ui.set_next_window_pos(ui.ImVec2(_UI_PANEL_MARGIN, _UI_PANEL_MARGIN))
        ui.set_next_window_size(ui.ImVec2(width, height))
        ui.set_next_window_bg_alpha(_UI_PANEL_ALPHA)
        ui.begin("姿态预览", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))

        current_group_label = _pose_slot_label_zh(self.display_slot_id) if self.display_slot_id else "未选择"
        ui.text(f"{self.robot_display_name} 配置优化")
        ui.separator()
        ui.text(f"当前分组: {current_group_label}")
        ui.text(self._training_summary_text())
        ui.separator()
        ui.text_wrapped(
            f"左侧是标准 human 姿态，右侧是同一组 {self.robot_display_name} 机器人姿态。"
        )
        if self.display_slot_id is not None:
            human_slot = self.human_slots[self.display_slot_id]
            robot_slot = self.robot_slots[self.display_slot_id]
            ui.separator()
            ui.text(f"Human: {_slot_status_zh(human_slot.status)}")
            if human_slot.pose_name:
                ui.text_wrapped(f"人体姿态: {human_slot.pose_name}")
            ui.spacing()
            ui.text(f"Robot: {_slot_status_zh(robot_slot.status)}")
            if robot_slot.pose_name:
                ui.text_wrapped(f"机器人姿态: {robot_slot.pose_name}")

        ui.end()

    def _render_pose_slots_panel(self, ui, x: float, y: float, width: float, height: float):
        ui.set_next_window_pos(ui.ImVec2(x, y))
        ui.set_next_window_size(ui.ImVec2(width, height))
        ui.set_next_window_bg_alpha(_UI_PANEL_ALPHA)
        ui.begin("姿态槽位", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))

        ui.text(f"{self.robot_display_name} 姿态组")
        ui.text(self._training_summary_text())
        ui.separator()
        for slot_id in POSE_SLOT_ORDER:
            ui.push_id(f"pose_slot_{slot_id}")
            is_current = self.display_slot_id == slot_id
            title = _pose_slot_label_zh(slot_id)
            ui.text(f"{'当前' if is_current else '分组'}: {title}")
            ui.text_wrapped(self._slot_compact_status(slot_id))
            if ui.button("显示这一组"):
                self.set_display_slot(slot_id)
            ui.same_line()
            changed, self.training_slot_enabled[slot_id] = ui.checkbox(
                "参与优化",
                self.training_slot_enabled[slot_id],
            )
            if changed and self.training_slot_enabled[slot_id] and not self._slot_is_ready_for_training(slot_id):
                self.append_log(f"[提示] {_pose_slot_label_zh(slot_id)} 已勾选，但机器人姿态还没准备好。")
            robot_slot = self.robot_slots[slot_id]
            if robot_slot.pose_name:
                ui.text_wrapped(robot_slot.pose_name)
            ui.separator()
            ui.pop_id()

        ui.end()

    def _render_optimizer_panel(self, ui, x: float, y: float, width: float, height: float):
        ui.set_next_window_pos(ui.ImVec2(x, y))
        ui.set_next_window_size(ui.ImVec2(width, height))
        ui.set_next_window_bg_alpha(_UI_PANEL_ALPHA)
        ui.begin("优化设置", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))

        selected_slot_ids = self._selected_training_slot_ids()
        ui.text("训练数据")
        selected_text = "、".join(_pose_slot_label_zh(slot_id) for slot_id in selected_slot_ids) if selected_slot_ids else "未选择"
        ui.text_wrapped(f"当前勾选: {selected_text}")
        ui.text(self._training_summary_text())
        if self.pose_registration_issues:
            ui.separator()
            ui.text_wrapped("params.py 中的姿态注册还不完整，窗口可以继续使用。")
            for issue in self.pose_registration_issues[:5]:
                ui.text_wrapped(issue)
        ui.separator()

        ui.text("自动配置")
        ui.text_wrapped("Scaler、报告和中间姿态文件会自动写到机器人配置目录。路径问题请直接检查 params.py。")
        ui.separator()
        ui.text("足部接触 / 虚拟链接")
        _, self.show_virtual_sole_overlay = ui.checkbox(
            "显示 virtual sole anchors",
            self.show_virtual_sole_overlay,
        )
        _, self.show_feet_contact_overlay = ui.checkbox(
            "显示 feet contact 支撑线",
            self.show_feet_contact_overlay,
        )
        anchor_terms = " | ".join(
            f"{_VIRTUAL_ANCHOR_LABELS[name]}" for name in _VIRTUAL_ANCHOR_ORDER
        )
        ui.text_wrapped(f"锚点术语: {anchor_terms}")
        for anchor_name in _VIRTUAL_ANCHOR_ORDER:
            color = _VIRTUAL_ANCHOR_COLORS[anchor_name]
            ui.text_colored(
                ui.ImVec4(color[0], color[1], color[2], 1.0),
                f"{_VIRTUAL_ANCHOR_LABELS[anchor_name]}",
            )
        ui.text_wrapped("紫色细线: foot link -> auto virtual link / soft anchor")
        left_state = "contact" if self.virtual_overlay_status.get("left") else "not contact"
        right_state = "contact" if self.virtual_overlay_status.get("right") else "not contact"
        ui.text_wrapped(f"feet contact preview: left={left_state}, right={right_state}")
        if self.foot_grounding_summary:
            ui.text_wrapped(self.foot_grounding_summary)
        ui.text_wrapped(
            f"template: {self.resolved_capability.get('priority_template', 'unknown')} | "
            f"anchors: {self.virtual_sole_anchors.get('source', 'disabled')}"
        )
        for message in self.virtual_overlay_messages[:3]:
            ui.text_wrapped(f"[提示] {message}")
        if ui.collapsing_header("手动调整 5 个虚拟足底点"):
            self._render_virtual_anchor_editor(ui)
        ui.separator()
        ui.text("注册状态")
        for line in robot_registry_parser.get_robot_setup_status(self.robot_type):
            ui.text_wrapped(line)
        ui.separator()

        ui.text("优化参数")
        changed, self.iterations = ui.input_int("迭代次数", int(self.iterations))
        self.iterations = max(1, int(self.iterations))
        changed, self.learning_rate = ui.input_float("学习率", float(self.learning_rate), format="%.6f")
        self.learning_rate = max(1e-6, float(self.learning_rate))

        ui.separator()
        ready, issues = self._validate_slots_ready()
        if not ready:
            ui.text_wrapped("当前还不能开始优化:")
            for issue in issues[:4]:
                ui.text_wrapped(issue)
        if ui.button("开始优化配置"):
            self.run_optimization()

        ui.end()

    def _render_log_panel(self, ui, x: float, y: float, width: float, height: float):
        ui.set_next_window_pos(ui.ImVec2(x, y))
        ui.set_next_window_size(ui.ImVec2(width, height))
        ui.set_next_window_bg_alpha(_UI_PANEL_ALPHA)
        ui.begin("运行日志", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))
        for line in self.logs[-25:]:
            ui.text_wrapped(line)
        ui.end()

    def gui(self, ui):
        viewport = ui.get_main_viewport()
        left_width = max(360.0, min(460.0, viewport.size.x * 0.30))
        right_width = max(440.0, min(560.0, viewport.size.x * 0.34))
        log_height = 170.0
        top_height = max(420.0, viewport.size.y - log_height - 2 * _UI_PANEL_MARGIN)
        self._render_human_pose_editor(ui, left_width, top_height)
        right_top_height = top_height * 0.62
        self._render_pose_slots_panel(
            ui,
            viewport.size.x - right_width - _UI_PANEL_MARGIN,
            _UI_PANEL_MARGIN,
            right_width,
            right_top_height,
        )
        self._render_optimizer_panel(
            ui,
            viewport.size.x - right_width - _UI_PANEL_MARGIN,
            _UI_PANEL_MARGIN + right_top_height + _UI_PANEL_MARGIN,
            right_width,
            top_height - right_top_height - _UI_PANEL_MARGIN,
        )
        self._render_log_panel(
            ui,
            _UI_PANEL_MARGIN,
            viewport.size.y - log_height - _UI_PANEL_MARGIN,
            viewport.size.x - 2 * _UI_PANEL_MARGIN,
            log_height,
        )

    def run(self):
        while self.viewer.is_running():
            self.step()
            self.render()
        self.viewer.close()


def parse_args():
    import newton.examples

    parser = newton.examples.create_parser()
    parser.set_defaults(viewer="gl")
    parser.add_argument(
        "--robot",
        type=str,
        default=robot_registry_parser.get_active_robot_name(),
        help="Registered robot name or alias from params.py, for example g1 or rpo.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Viewer config path. Omit it to auto-generate beside the retargeter config.",
    )
    parser.add_argument(
        "--scaler-config",
        type=str,
        default=None,
        help="Scaler config optimized by the UI. Omit it to auto-generate beside the retargeter config.",
    )
    parser.add_argument(
        "--retargeter-config",
        type=str,
        default=None,
        help="Retargeter config used to extract robot target effectors.",
    )
    parser.add_argument(
        "--source-reference-bvh",
        type=str,
        default=None,
        help="Reference SOMA BVH used by the human pose editor.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output scaler config written after optimization.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Optional paired-pose run report path.",
    )
    parser.add_argument(
        "--pose-artifact",
        type=str,
        default=None,
        help="Optional paired-pose artifact path.",
    )
    parser.add_argument(
        "--auto-refine",
        action="store_true",
        help="Deprecated: run the legacy teacher-guided refinement flow before opening the optimizer UI.",
    )
    viewer, args = newton.examples.init(parser)
    return viewer, _resolve_startup_paths(args)


def main():
    viewer, args = parse_args()
    if isinstance(viewer, newton.viewer.ViewerNull):
        raise RuntimeError("pose_optimizer_ui.py requires a visual viewer backend such as --viewer gl.")

    if not pathlib.Path(args.config).exists():
        raise FileNotFoundError(f"Viewer config file not found: {args.config}")

    if args.auto_refine:
        from soma_retargeter.teacher_refinement import refine_registered_robot_config

        result = refine_registered_robot_config(args.robot, retargeter_config_path=args.retargeter_config)
        print(f"[INFO] Auto-refine decision for {args.robot}: {result['decision']} ({result['reason']})")

    if install_cpu_pinned_memory_fallback(wp):
        print("[INFO] CUDA is unavailable; using regular CPU viewer buffers instead of pinned buffers.")

    with wp.ScopedDevice(args.device):
        app = PoseOptimizerUI(viewer, args)
        app.run()


if __name__ == "__main__":
    main()
