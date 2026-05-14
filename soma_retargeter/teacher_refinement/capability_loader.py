# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import soma_retargeter.robot_registry_parser as robot_registry_parser
from soma_retargeter.teacher_refinement.capability_schema import default_capability_profile
from soma_retargeter.utils import io_utils


def _strip_inline_comment(line: str) -> str:
    in_quote: str | None = None
    for idx, char in enumerate(line):
        if char in {"'", '"'}:
            in_quote = None if in_quote == char else char
        elif char == "#" and in_quote is None:
            return line[:idx]
    return line


def _parse_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the small mapping-only YAML profile shape used by robot_capability.yaml."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_inline_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line in {path}: {raw_line}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid YAML indentation in {path}: {raw_line}")
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_yaml_scalar(value)

    return root


def find_capability_profile_path(robot_name: str, explicit_path: str | Path | None = None) -> Path | None:
    if explicit_path:
        path = io_utils.resolve_path(explicit_path)
        return path if path.exists() else path

    config_dir = robot_registry_parser.get_robot_config_dir(robot_name)
    if config_dir is None:
        return None

    for filename in ("robot_capability.json", "robot_capability.yaml", "robot_capability.yml"):
        candidate = config_dir / filename
        if candidate.exists():
            return candidate
    return None


def load_capability_profile(
    robot_name: str,
    explicit_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Load a capability profile or return a conservative all-auto profile."""

    robot_name = robot_registry_parser.resolve_robot_name(robot_name)
    path = find_capability_profile_path(robot_name, explicit_path)
    if path is None:
        return default_capability_profile(robot_name), None
    if not path.exists():
        raise FileNotFoundError(f"Robot capability profile not found: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in {".yaml", ".yml"}:
        payload = _load_simple_yaml(path)
    else:
        raise ValueError(f"Unsupported capability profile extension: {path.suffix}")

    if not isinstance(payload, dict):
        raise ValueError(f"Robot capability profile must contain an object: {path}")
    payload.setdefault("robot", robot_name)
    return payload, path
