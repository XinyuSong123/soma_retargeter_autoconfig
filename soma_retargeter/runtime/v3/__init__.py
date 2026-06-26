"""Runtime V3 profile shadow integration helpers."""

__all__ = [
    "RuntimeModelIdentity",
    "RuntimeProfileResolution",
    "RuntimeProfileStatusError",
    "RuntimeV3ProfileError",
    "RuntimeV3Profile",
    "RuntimeV3TargetAdapter",
    "SourceSemanticFrameBatch",
    "extract_source_semantic_frames",
    "load_runtime_v3_profile",
    "resolve_runtime_v3_profile_id",
]


def __getattr__(name: str):
    if name in {
        "RuntimeModelIdentity",
        "RuntimeProfileResolution",
        "RuntimeProfileStatusError",
        "RuntimeV3ProfileError",
        "RuntimeV3Profile",
        "load_runtime_v3_profile",
        "resolve_runtime_v3_profile_id",
    }:
        from . import profile_loader

        return getattr(profile_loader, name)
    if name in {"SourceSemanticFrameBatch", "extract_source_semantic_frames"}:
        from .source_frames import SourceSemanticFrameBatch, extract_source_semantic_frames

        return {
            "SourceSemanticFrameBatch": SourceSemanticFrameBatch,
            "extract_source_semantic_frames": extract_source_semantic_frames,
        }[name]
    if name == "RuntimeV3TargetAdapter":
        from .target_adapter import RuntimeV3TargetAdapter

        return RuntimeV3TargetAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
