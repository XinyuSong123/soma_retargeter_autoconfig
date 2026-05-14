# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated teacher-guided retargeter config refinement utilities.

The normal workflow now generates contact-aware virtual sole anchors through
``soma_retargeter.foot_anchors`` during runtime config loading. This package is
kept for backward compatibility with older refinement scripts.
"""


def refine_registered_robot_config(*args, **kwargs):
    from soma_retargeter.teacher_refinement.refiner import refine_registered_robot_config as _impl

    return _impl(*args, **kwargs)

__all__ = ["refine_registered_robot_config"]
