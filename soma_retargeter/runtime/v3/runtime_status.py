"""Step 3.1 runtime status constants."""

FULL_FINAL_STATUSES = {
    "runtime_quality_passed",
    "runtime_quality_warned",
    "runtime_quality_failed",
}
PARTIAL_FINAL_STATUS = "partial_runtime_passed"
NEGATIVE_FINAL_STATUS = "negative_control_runtime_passed"
BLOCKED_FINAL_STATUS = "blocked_source_or_profile"

PROFILE_RESOLUTION_STATUSES = {
    "profile_match",
    "runtime_local_profile_generated",
    "runtime_local_profile_failed",
    "structured_partial_supported",
    "negative_control_rejected",
    "source_or_cache_unavailable",
    "runtime_model_load_failed",
}
