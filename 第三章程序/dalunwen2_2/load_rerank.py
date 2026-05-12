import numpy as np


def _valid_q_values(q_values, avail_mask):
    q_values = np.asarray(q_values, dtype=np.float32).reshape(-1)
    valid_mask = np.isfinite(q_values)
    if avail_mask is not None:
        mask = np.asarray(avail_mask).reshape(-1)
        if mask.shape[0] == q_values.shape[0]:
            valid_mask = valid_mask & (mask > 0)
    return q_values, valid_mask


def _argmax_valid(q_values, valid_mask):
    if not bool(valid_mask.any()):
        return 0
    masked = np.where(valid_mask, q_values, -np.inf)
    return int(np.argmax(masked))


def _normalise_candidate_penalty(load_penalty, candidate_mask, action_count):
    penalties = np.asarray(load_penalty, dtype=np.float32).reshape(-1)
    if penalties.shape[0] != action_count:
        penalties = np.zeros(action_count, dtype=np.float32)
    penalties = np.where(np.isfinite(penalties), penalties, 0.0)
    normalised = np.zeros(action_count, dtype=np.float32)
    if not bool(candidate_mask.any()):
        return normalised
    candidate_values = penalties[candidate_mask]
    min_value = float(np.min(candidate_values))
    max_value = float(np.max(candidate_values))
    if max_value - min_value > 1e-8:
        normalised[candidate_mask] = (candidate_values - min_value) / (max_value - min_value)
    return normalised


def _candidate_mask(q_values, valid_mask, mode, margin_threshold, topk):
    candidate_mask = np.zeros_like(valid_mask, dtype=bool)
    if not bool(valid_mask.any()):
        return candidate_mask
    if mode == "topk_rerank":
        valid_indices = np.where(valid_mask)[0]
        k = max(1, min(int(topk), valid_indices.shape[0]))
        ranked = valid_indices[np.argsort(q_values[valid_indices])[::-1]]
        candidate_mask[ranked[:k]] = True
        return candidate_mask

    top_q = float(np.max(q_values[valid_mask]))
    threshold = float(margin_threshold)
    candidate_mask = valid_mask & ((top_q - q_values) <= threshold + 1e-8)
    return candidate_mask


def select_load_rerank_action(
    q_values,
    load_penalty=None,
    agent_num=0,
    mode="margin_gated",
    margin_threshold=0.5,
    topk=2,
    penalty_weight=0.0,
    avail_mask=None,
):
    q_values, valid_mask = _valid_q_values(q_values, avail_mask)
    action_count = q_values.shape[0]
    original_action = _argmax_valid(q_values, valid_mask)
    candidate_mask = _candidate_mask(q_values, valid_mask, mode, margin_threshold, topk)
    candidate_actions = [int(item) for item in np.where(candidate_mask)[0].tolist()]
    normalised_penalty = _normalise_candidate_penalty(
        load_penalty if load_penalty is not None else np.zeros(action_count),
        candidate_mask,
        action_count,
    )
    scores = np.full(action_count, -np.inf, dtype=np.float32)
    scores[candidate_mask] = (
        q_values[candidate_mask] - float(penalty_weight) * normalised_penalty[candidate_mask]
    )
    applied = bool(agent_num == 0 and float(penalty_weight) > 0.0 and load_penalty is not None)
    rerank_action = _argmax_valid(scores, candidate_mask) if applied else original_action
    info = {
        "applied": applied,
        "mode": mode,
        "original_action": int(original_action),
        "rerank_action": int(rerank_action),
        "action_changed": bool(rerank_action != original_action),
        "candidate_actions": candidate_actions,
        "candidate_size": len(candidate_actions),
        "q_values": [round(float(item), 6) if np.isfinite(item) else None for item in q_values],
        "load_penalty": [
            round(float(item), 6) if np.isfinite(item) else None
            for item in np.asarray(load_penalty if load_penalty is not None else np.zeros(action_count), dtype=np.float32).reshape(-1)[:action_count]
        ],
        "normalised_load_penalty": [
            round(float(item), 6) if np.isfinite(item) else None for item in normalised_penalty
        ],
        "scores": [round(float(item), 6) if np.isfinite(item) else None for item in scores],
    }
    if bool(valid_mask.any()):
        top_values = np.sort(q_values[valid_mask])[::-1]
        info["q_margin"] = round(float(top_values[0] - top_values[1]), 6) if top_values.shape[0] > 1 else 0.0
    else:
        info["q_margin"] = None
    return int(rerank_action), info
