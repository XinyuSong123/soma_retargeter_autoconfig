# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated compatibility wrapper for virtual sole anchor generation.

Use :mod:`soma_retargeter.foot_anchors` for new contact-aware foot IK
configuration paths. This module remains so older imports continue to work.
"""

from soma_retargeter.foot_anchors import (
    ANCHOR_NAMES,
    generate_virtual_sole_anchors,
    generate_virtual_sole_anchors_for_robot,
)

__all__ = [
    "ANCHOR_NAMES",
    "generate_virtual_sole_anchors",
    "generate_virtual_sole_anchors_for_robot",
]
