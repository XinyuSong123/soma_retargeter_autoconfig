# Subagent Execution Guide

主 agent：Integrator。

## Worktree 建议

```bash
git worktree add ../v3-agent-a -b agent-a/runtime-kinematics
git worktree add ../v3-agent-b -b agent-b/calibration
git worktree add ../v3-agent-d -b agent-d/robot-zoo
```

并行上限：3。

## Agent A — Runtime Kinematics

Owned files建议：

```text
soma_retargeter/robotics/runtime_model_adapter.py
soma_retargeter/robotics/reachability_v3.py
tests/test_runtime_model_adapter.py
tests/test_full_chain_reachability.py
assets/robots/synthetic_v3/
```

必须先交付 synthetic tests，再接真实机器人。

## Agent B — Calibration & Targets

Owned files：

```text
soma_retargeter/robotics/rest_pose_calibration.py
soma_retargeter/robotics/target_builder_v3.py
tests/test_rest_pose_calibration.py
tests/test_target_builder_v3.py
```

禁止调用legacy scaler offsets。

## Agent C — IK Runtime

Owned files：

```text
soma_retargeter/pipelines/v3_objectives.py
soma_retargeter/pipelines/newton_pipeline_v3.py
tests/test_relative_torso_objective.py
tests/test_v3_batch_contact.py
```

在A/B接口稳定后启动。

## Agent D — Robot Zoo

Owned files：

```text
assets/robot_zoo/
soma_retargeter/tools/sync_robot_zoo.py
soma_retargeter/tools/audit_robot_zoo_licenses.py
soma_retargeter/tools/validate_robot_zoo.py
tests/test_robot_zoo_manifest.py
```

D1查许可证；D2做canonical snapshots；D3做semantic ground truth。

## Agent E — Benchmark

Owned files：

```text
soma_retargeter/tools/benchmark_retargeting_v3.py
soma_retargeter/metrics/semantic_evaluator.py
tests/test_semantic_evaluator.py
.github/workflows/
```

不得使用pipeline自己的input_targets作为最终横向真值。

## Agent F — Red Team

不得实现feature。检查：

- hidden manual tuning；
- robot-name branches；
- static XML FK；
- fake priority；
- rank0 false pass；
- asset license；
- unrun tests；
- dishonest completion。

## Handoff 模板

```markdown
# Agent X Handoff
- Branch/commit:
- Files:
- Commands:
- Tests:
- Numerical results:
- Assumptions:
- Known failures:
- Integration order:
- Review risks:
```
