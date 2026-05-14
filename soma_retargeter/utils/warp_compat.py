# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


def _cuda_is_available(wp_module: Any) -> bool:
    try:
        return bool(wp_module.is_cuda_available()) and int(wp_module.get_cuda_device_count()) > 0
    except Exception:
        return False


def _is_cpu_device(device: Any) -> bool:
    if device is None:
        return True
    if isinstance(device, str):
        return device == "cpu"
    if getattr(device, "is_cpu", False):
        return True
    return str(device) == "cpu"


def install_cpu_pinned_memory_fallback(wp_module: Any) -> bool:
    """Disable pinned CPU allocations when Warp has no usable CUDA driver.

    Newton's GL viewer allocates CPU staging buffers with ``pinned=True``.
    On machines without a CUDA-capable device or driver, Warp 1.12.0 can fail
    that allocation path even though a normal CPU buffer would work. This shim
    keeps the viewer usable on CPU-only systems by falling back to unpinned CPU
    buffers. It is intentionally narrow and leaves CUDA machines untouched.
    """

    if getattr(wp_module, "_soma_cpu_pinned_fallback_installed", False):
        return False
    if _cuda_is_available(wp_module):
        return False

    original_empty = wp_module.empty

    def empty_with_cpu_pinned_fallback(*args, **kwargs):
        if kwargs.get("pinned") is True and _is_cpu_device(kwargs.get("device")):
            kwargs = dict(kwargs)
            kwargs["pinned"] = False
        return original_empty(*args, **kwargs)

    wp_module.empty = empty_with_cpu_pinned_fallback
    wp_module._soma_cpu_pinned_fallback_installed = True
    return True
