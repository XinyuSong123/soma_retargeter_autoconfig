import numpy as np
import warp as wp


def _smooth_scores(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.clip(scores, 0.0, 1.0)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(scores, (window // 2, window - 1 - window // 2), mode="edge")
    return np.clip(np.convolve(padded, kernel, mode="valid"), 0.0, 1.0)


def _contact_score_from_positions(positions: np.ndarray, ground_height: float, velocity_dt: float) -> np.ndarray:
    z = positions[:, 2]
    xy = positions[:, :2]
    vel = np.zeros(len(positions), dtype=np.float32)
    if len(positions) > 1:
        vel[1:] = np.linalg.norm((xy[1:] - xy[:-1]) / max(velocity_dt, 1e-6), axis=1)
        vel[0] = vel[1]
    height_score = np.exp(-np.maximum(z - ground_height, 0.0) / 0.03)
    vel_score = np.exp(-vel / 0.2)
    return np.clip((height_score * 0.6 + vel_score * 0.4).astype(np.float32), 0.0, 1.0)


def infer_contacts_from_animation_buffer(buffer, root_tx=wp.transform_identity(), smoothing_window: int = 5) -> dict[str, np.ndarray]:
    skel = buffer.skeleton
    names = {
        "left_toe": ["LeftToeBase", "LeftToeEnd", "LeftToe"],
        "right_toe": ["RightToeBase", "RightToeEnd", "RightToe"],
        "left_heel": ["LeftFoot"],
        "right_heel": ["RightFoot"],
    }

    def pick(name_list):
        for n in name_list:
            idx = skel.joint_index(n)
            if idx != -1:
                return idx
        return -1

    joint_indices = {k: pick(v) for k, v in names.items()}
    if any(v == -1 for v in joint_indices.values()):
        missing = [k for k, v in joint_indices.items() if v == -1]
        raise ValueError(f"Cannot infer contacts, missing joints: {missing}")

    traces = {k: np.zeros((buffer.num_frames, 3), dtype=np.float32) for k in joint_indices}
    for frame in range(buffer.num_frames):
        gtx = buffer.compute_global_transforms(frame, root_tx=root_tx)
        for key, idx in joint_indices.items():
            traces[key][frame] = np.array(wp.transform_get_translation(gtx[idx]), dtype=np.float32)

    all_z = np.concatenate([v[:, 2] for v in traces.values()])
    ground = float(np.percentile(all_z, 2.0))
    dt = 1.0 / max(buffer.sample_rate, 1e-3)

    out = {}
    for key, pos in traces.items():
        out[f"{key}_contact_score"] = _smooth_scores(
            _contact_score_from_positions(pos, ground, dt), smoothing_window
        )
    return out


def contacts_from_npz_foot_contacts(foot_contacts: np.ndarray, smoothing_window: int = 5) -> dict[str, np.ndarray]:
    if foot_contacts.ndim != 2 or foot_contacts.shape[1] < 4:
        raise ValueError("foot_contacts must have shape [T,4+] in [L heel, L toe, R heel, R toe] order")
    mapped = {
        "left_heel_contact_score": foot_contacts[:, 0].astype(np.float32),
        "left_toe_contact_score": foot_contacts[:, 1].astype(np.float32),
        "right_heel_contact_score": foot_contacts[:, 2].astype(np.float32),
        "right_toe_contact_score": foot_contacts[:, 3].astype(np.float32),
    }
    return {k: _smooth_scores(np.clip(v, 0.0, 1.0), smoothing_window) for k, v in mapped.items()}
