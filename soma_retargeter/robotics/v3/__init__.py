"""Offline runtime-kinematics compiler for retargeting v3 Step 2."""

__all__ = [
    "KinematicProfileV3",
    "MuJoCoRuntimeModelAdapter",
    "NewtonRuntimeModelAdapter",
    "RuntimeRobotModelAdapter",
    "SemanticSite",
    "compile_kinematic_profile_v3",
]


def __getattr__(name: str):
    if name in {"MuJoCoRuntimeModelAdapter", "NewtonRuntimeModelAdapter", "RuntimeRobotModelAdapter", "SemanticSite"}:
        from .model_adapter import MuJoCoRuntimeModelAdapter, NewtonRuntimeModelAdapter, RuntimeRobotModelAdapter, SemanticSite

        return {
            "MuJoCoRuntimeModelAdapter": MuJoCoRuntimeModelAdapter,
            "NewtonRuntimeModelAdapter": NewtonRuntimeModelAdapter,
            "RuntimeRobotModelAdapter": RuntimeRobotModelAdapter,
            "SemanticSite": SemanticSite,
        }[name]
    if name in {"KinematicProfileV3", "compile_kinematic_profile_v3"}:
        from .profile import KinematicProfileV3, compile_kinematic_profile_v3

        return {
            "KinematicProfileV3": KinematicProfileV3,
            "compile_kinematic_profile_v3": compile_kinematic_profile_v3,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
