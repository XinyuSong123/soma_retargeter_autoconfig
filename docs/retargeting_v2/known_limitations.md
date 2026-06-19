# Retargeting v2 Known Limitations

- Runtime motion benchmark metrics are not implemented yet. Current artifacts are compile-level profile checks with runtime metrics marked `not_run`.
- Profile validation now covers numeric health gates and left/right chain length mismatch thresholds, but true symmetry tying is still compile-time reporting rather than a separate constrained optimizer.
- Self-collision runtime barriers are optional and disabled by default through `collision_weight=0.0`; motion-level evidence is still needed before making them a default feasibility term.
- E3 v2 and OLI are not registered in the current workspace, so benchmark artifacts cover only `roboparty_rpo` and `unitree_g1`.
- Unitree G1 v2 profile currently reports zero chain ranks and placeholder chain lengths, indicating incomplete registry/morphology coverage for that asset path.
- Semantic auto-detection is still shallow; explicit `ik_map` remains the reliable path.
- Virtual hand and foot sites now prefer explicit MJCF sites and then use primitive geom or STL mesh distal bounds when available, but non-STL mesh formats and rotated/fromto geom bounds are still limited.
- Root/ground metadata is compile-time provenance only; runtime root-height stabilization still needs motion-level acceptance metrics.
- The priority residual guard currently protects joint-limit margin residuals, not all high-priority feasibility residuals.
