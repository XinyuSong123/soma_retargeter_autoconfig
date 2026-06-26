"""Step 3.1 motion clip inventory and deterministic frame budgets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable


CORE_CLIP_RELATIVE_PATHS: tuple[str, ...] = (
    "assets/motions/bvh/Neutral_walk_forward_002__A057.bvh",
    "assets/motions/bvh/wave_R_001__A428.bvh",
    "assets/motions/bvh/body_stretch_1_004__A069.bvh",
    "assets/motions/bvh/item_pick_up_standing_R_001__A410.bvh",
)

MOTION_EXTENSIONS = frozenset({".bvh", ".csv"})
DEFAULT_SHORT_CLIP_MAX_FRAMES = 120
DEFAULT_MID_CLIP_MAX_FRAMES = 300
DEFAULT_CSV_SAMPLE_RATE = 120.0


class ClipInventoryError(RuntimeError):
    """Raised when required clip inventory gates fail."""


@dataclass(frozen=True)
class FrameBudget:
    source_frame_count: int
    selected_frame_count: int
    frame_indices: tuple[int, ...]
    stride: int
    short_clip_max_frames: int = DEFAULT_SHORT_CLIP_MAX_FRAMES
    mid_clip_max_frames: int = DEFAULT_MID_CLIP_MAX_FRAMES
    policy: str = "deterministic"

    def to_json(self) -> dict[str, Any]:
        return {
            "source_frame_count": self.source_frame_count,
            "selected_frame_count": self.selected_frame_count,
            "frame_indices": list(self.frame_indices),
            "stride": self.stride,
            "short_clip_max_frames": self.short_clip_max_frames,
            "mid_clip_max_frames": self.mid_clip_max_frames,
            "policy": self.policy,
            "deterministic": True,
        }


@dataclass(frozen=True)
class ClipInventoryEntry:
    path: str
    name: str
    format: str
    is_core: bool
    exists: bool
    load_status: str
    frame_count: int | None
    sample_rate: float | None
    frame_budget: FrameBudget | None
    sha256: str | None
    byte_count: int | None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "format": self.format,
            "is_core": self.is_core,
            "exists": self.exists,
            "load_status": self.load_status,
            "frame_count": self.frame_count,
            "sample_rate": self.sample_rate,
            "frame_budget": None if self.frame_budget is None else self.frame_budget.to_json(),
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "error": self.error,
        }


@dataclass(frozen=True)
class ClipInventory:
    schema_version: int
    motions_root: str
    status: str
    clips: tuple[ClipInventoryEntry, ...]
    core_clip_paths: tuple[str, ...]
    missing_core_clips: tuple[str, ...]
    unloadable_core_clips: tuple[str, ...]
    format_counts: dict[str, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "motions_root": self.motions_root,
            "status": self.status,
            "clips": [clip.to_json() for clip in self.clips],
            "core_clip_paths": list(self.core_clip_paths),
            "missing_core_clips": list(self.missing_core_clips),
            "unloadable_core_clips": list(self.unloadable_core_clips),
            "format_counts": dict(sorted(self.format_counts.items())),
        }


def deterministic_frame_budget(
    frame_count: int,
    *,
    short_clip_max_frames: int = DEFAULT_SHORT_CLIP_MAX_FRAMES,
    mid_clip_max_frames: int = DEFAULT_MID_CLIP_MAX_FRAMES,
) -> FrameBudget:
    """Return the deterministic Step 3.1 frame subset for a clip."""

    total = int(frame_count)
    short_limit = int(short_clip_max_frames)
    mid_limit = int(mid_clip_max_frames)
    if total < 0:
        raise ValueError("frame_count must be non-negative")
    if short_limit <= 0:
        raise ValueError("short_clip_max_frames must be positive")
    if mid_limit < short_limit:
        raise ValueError("mid_clip_max_frames must be greater than or equal to short_clip_max_frames")
    if total == 0:
        return FrameBudget(
            source_frame_count=0,
            selected_frame_count=0,
            frame_indices=(),
            stride=1,
            short_clip_max_frames=short_limit,
            mid_clip_max_frames=mid_limit,
            policy="empty_clip",
        )
    if total <= short_limit:
        indices = tuple(range(total))
        policy = "full_short_clip"
        stride = 1
    elif total <= mid_limit:
        indices = tuple(range(total))
        policy = "full_mid_clip"
        stride = 1
    else:
        stride = int(math.ceil(total / mid_limit))
        indices = tuple(range(0, total, stride))[:mid_limit]
        policy = "strided_long_clip"
    return FrameBudget(
        source_frame_count=total,
        selected_frame_count=len(indices),
        frame_indices=indices,
        stride=stride,
        short_clip_max_frames=short_limit,
        mid_clip_max_frames=mid_limit,
        policy=policy,
    )


def inventory_motion_clips(
    repo_root: str | Path = ".",
    *,
    motions_root: str | Path | None = None,
    core_clip_paths: Iterable[str] = CORE_CLIP_RELATIVE_PATHS,
    require_core_clips: bool = True,
) -> ClipInventory:
    """Inventory all BVH/CSV clips under ``assets/motions``."""

    root = Path(repo_root).resolve()
    motion_root = (root / "assets/motions") if motions_root is None else Path(motions_root)
    if not motion_root.is_absolute():
        motion_root = root / motion_root
    core_paths = tuple(_normalize_relative_path(path) for path in core_clip_paths)
    clip_paths = {
        path.resolve()
        for path in motion_root.rglob("*")
        if path.is_file() and path.suffix.lower() in MOTION_EXTENSIONS
    }
    for rel in core_paths:
        clip_paths.add((root / rel).resolve())

    clips = tuple(
        inspect_motion_clip(path, repo_root=root, core_clip_paths=core_paths)
        for path in sorted(clip_paths, key=lambda value: _display_path(value, root))
    )
    by_path = {clip.path: clip for clip in clips}
    missing_core = tuple(path for path in core_paths if path not in by_path or not by_path[path].exists)
    unloadable_core = tuple(
        path
        for path in core_paths
        if path in by_path and by_path[path].exists and by_path[path].load_status != "loaded"
    )
    format_counts: dict[str, int] = {}
    for clip in clips:
        if clip.exists:
            format_counts[clip.format] = format_counts.get(clip.format, 0) + 1

    status = "passed"
    if require_core_clips and (missing_core or unloadable_core):
        status = "blocked"
    return ClipInventory(
        schema_version=1,
        motions_root=_display_path(motion_root, root),
        status=status,
        clips=clips,
        core_clip_paths=core_paths,
        missing_core_clips=missing_core,
        unloadable_core_clips=unloadable_core,
        format_counts=format_counts,
    )


def inspect_motion_clip(
    path: str | Path,
    *,
    repo_root: str | Path = ".",
    core_clip_paths: Iterable[str] = CORE_CLIP_RELATIVE_PATHS,
) -> ClipInventoryEntry:
    """Inspect one BVH/CSV clip without changing its contents."""

    root = Path(repo_root).resolve()
    clip_path = Path(path)
    if not clip_path.is_absolute():
        clip_path = root / clip_path
    display = _display_path(clip_path, root)
    core_set = {_normalize_relative_path(value) for value in core_clip_paths}
    suffix = clip_path.suffix.lower()
    fmt = suffix[1:] if suffix.startswith(".") else suffix
    is_core = display in core_set

    if not clip_path.exists():
        return ClipInventoryEntry(
            path=display,
            name=clip_path.name,
            format=fmt,
            is_core=is_core,
            exists=False,
            load_status="missing",
            frame_count=None,
            sample_rate=None,
            frame_budget=None,
            sha256=None,
            byte_count=None,
            error="clip file does not exist",
        )
    if suffix not in MOTION_EXTENSIONS:
        return ClipInventoryEntry(
            path=display,
            name=clip_path.name,
            format=fmt,
            is_core=is_core,
            exists=True,
            load_status="unsupported_format",
            frame_count=None,
            sample_rate=None,
            frame_budget=None,
            sha256=_sha256_file(clip_path),
            byte_count=clip_path.stat().st_size,
            error=f"unsupported motion clip extension {suffix!r}",
        )

    try:
        if suffix == ".bvh":
            frame_count, sample_rate = _parse_bvh_metadata(clip_path)
        else:
            frame_count, sample_rate = _parse_csv_metadata(clip_path)
        budget = deterministic_frame_budget(frame_count)
        status = "loaded" if frame_count > 0 else "empty"
        error = None if frame_count > 0 else "clip has zero frames"
    except Exception as exc:
        frame_count = None
        sample_rate = None
        budget = None
        status = "unloadable"
        error = str(exc)

    return ClipInventoryEntry(
        path=display,
        name=clip_path.name,
        format=fmt,
        is_core=is_core,
        exists=True,
        load_status=status,
        frame_count=frame_count,
        sample_rate=sample_rate,
        frame_budget=budget,
        sha256=_sha256_file(clip_path),
        byte_count=clip_path.stat().st_size,
        error=error,
    )


def assert_core_clips_available(inventory: ClipInventory) -> None:
    """Raise if any required Step 3.1 core clip is missing or unloadable."""

    if inventory.missing_core_clips or inventory.unloadable_core_clips:
        parts = []
        if inventory.missing_core_clips:
            parts.append("missing: " + ", ".join(inventory.missing_core_clips))
        if inventory.unloadable_core_clips:
            parts.append("unloadable: " + ", ".join(inventory.unloadable_core_clips))
        raise ClipInventoryError("required core clip gate failed; " + "; ".join(parts))


def _parse_bvh_metadata(path: Path) -> tuple[int, float]:
    frame_count: int | None = None
    frame_time: float | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            lower = line.lower()
            if lower.startswith("frames:"):
                frame_count = int(line.split(":", 1)[1].strip())
            elif lower.startswith("frame time:"):
                frame_time = float(line.split(":", 1)[1].strip())
            if frame_count is not None and frame_time is not None:
                break
    if frame_count is None:
        raise ValueError("BVH metadata is missing Frames")
    if frame_time is None or frame_time <= 0.0:
        raise ValueError("BVH metadata has invalid Frame Time")
    return frame_count, 1.0 / frame_time


def _parse_csv_metadata(path: Path) -> tuple[int, float]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        return 0, DEFAULT_CSV_SAMPLE_RATE
    header_rows = 0 if _line_is_numeric_csv_row(lines[0]) else 1
    return max(0, len(lines) - header_rows), DEFAULT_CSV_SAMPLE_RATE


def _line_is_numeric_csv_row(line: str) -> bool:
    try:
        values = [value.strip() for value in line.split(",")]
        return bool(values) and all(value != "" and math.isfinite(float(value)) for value in values)
    except ValueError:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_relative_path(path: str | Path) -> str:
    return Path(path).as_posix().lstrip("./")
