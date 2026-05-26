# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from pathlib import Path
from typing import Union, Dict

# Assumes this file is in soma_retargeter/utils
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def get_package_root() -> Path:
    """Return the filesystem path to the package root."""
    return _PACKAGE_ROOT


def get_repo_root() -> Path:
    """Return the repository root directory."""
    return get_package_root().parent


def get_configs_dir() -> Path:
    """Return the configs directory path."""
    return get_package_root() / 'configs'


def resolve_path(path: Union[str, Path], base_dir: Union[str, Path, None] = None, default_root: Union[str, Path, None] = None) -> Path:
    """
    Resolve a path used by runtime configs.

    Relative paths are checked against ``base_dir`` (when supplied), the current
    working directory, the repository root, and finally ``soma_retargeter/configs``.
    If no candidate exists, the path is returned under ``default_root`` or the
    repository root.
    """
    path = Path(path).expanduser()
    if path.is_absolute():
        return path

    candidates = []
    if base_dir is not None:
        candidates.append(Path(base_dir).expanduser() / path)
    candidates.extend([
        Path.cwd() / path,
        get_repo_root() / path,
        get_configs_dir() / path,
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate

    root = Path(default_root).expanduser() if default_root is not None else get_repo_root()
    return root / path


def get_config_file(*relative_parts: str) -> Path:
    """Return a path to a config file, allowing absolute or repo-relative paths."""
    if len(relative_parts) == 1:
        return resolve_path(relative_parts[0], default_root=get_configs_dir())
    return get_configs_dir().joinpath(*relative_parts)


def load_json(path: Union[str, Path]) -> Dict:
    """
    Load a JSON file from the specified path.
    Args:
        path (Union[str, Path]): The file path to the JSON file. Can be a string or Path object.
    Returns:
        Dict: The parsed JSON content as a dictionary.
    Raises:
        FileNotFoundError: If the JSON file does not exist at the specified path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[ERROR]: JSON file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: Union[str, Path], payload: Dict, *, indent: int = 4, ensure_ascii: bool = False) -> Path:
    """Write a JSON payload, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=indent, ensure_ascii=ensure_ascii)
    return path
