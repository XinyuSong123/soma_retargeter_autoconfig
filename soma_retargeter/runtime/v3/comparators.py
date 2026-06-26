"""Small deterministic comparators for Step 3.1 artifacts."""

from __future__ import annotations

from typing import Any

from .fleet_inventory import stable_payload_hash


def deterministic_hash(payload: Any) -> str:
    return stable_payload_hash(payload)


def compare_hashes(left: Any, right: Any) -> dict[str, Any]:
    left_hash = deterministic_hash(left)
    right_hash = deterministic_hash(right)
    return {
        "left_hash": left_hash,
        "right_hash": right_hash,
        "matched": left_hash == right_hash,
    }
