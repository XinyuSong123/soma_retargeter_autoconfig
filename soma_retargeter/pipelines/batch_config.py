CONTACT_AWARE_BATCH_SIZE_LOG = (
    "[INFO] contact_aware_foot_ik is enabled; forcing retarget batch_size=1 "
    "for per-motion contact locks."
)

_DISABLED_CONTACT_SOURCES = {"none", "disabled", "false", "null"}


def is_contact_aware_foot_ik_enabled(retarget_config):
    if not isinstance(retarget_config, dict):
        return False

    contact_cfg = retarget_config.get("contact_aware_foot_ik", {})
    if not isinstance(contact_cfg, dict):
        return False

    contact_source = str(contact_cfg.get("contact_source", "auto")).lower()
    return bool(contact_cfg.get("enabled", False)) and contact_source not in _DISABLED_CONTACT_SOURCES


def resolve_retarget_batch_size(configured_batch_size, retarget_config, *, log=True):
    batch_size = max(1, int(configured_batch_size))
    if is_contact_aware_foot_ik_enabled(retarget_config):
        if log:
            print(CONTACT_AWARE_BATCH_SIZE_LOG)
        return 1
    return batch_size
