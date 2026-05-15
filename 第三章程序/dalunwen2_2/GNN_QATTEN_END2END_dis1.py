import json
import os
import random
import shutil

import numpy as np
import torch

from agent import Agents
from config_gnn_qatten_end2end import Config
from utils import ReplayBuffer
import rollout_end2end


conf = Config()
init_pulse = 750


def set_seed(conf):
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)


def end2end_best_score(summary, mode="pulse_plus_si", si_weight=1.0):
    if int(summary.get("finished_order_count", 0)) != rollout_end2end.pro_num:
        return None
    if mode == "pulse_plus_si":
        return float(summary["final_pulse"]) + float(summary["smoothness_index"])
    if mode == "pulse_plus_weighted_si":
        return float(summary["final_pulse"]) + float(si_weight) * float(
            summary["smoothness_index"]
        )
    else:
        raise ValueError("unsupported end2end best score mode: {}".format(mode))


def is_better_end2end_summary(summary, best_score, mode="pulse_plus_si", si_weight=1.0):
    score = end2end_best_score(summary, mode, si_weight=si_weight)
    if score is None:
        return False, None
    return best_score is None or score < best_score, score


def online_pulse_score(summary):
    if int(summary.get("finished_order_count", 0)) != rollout_end2end.pro_num:
        return None
    return float(summary["final_pulse"]), float(summary["smoothness_index"])


def is_better_online_pulse_summary(summary, best_score):
    score = online_pulse_score(summary)
    if score is None:
        return False, None
    return best_score is None or score < best_score, score


def online_balanced_score(summary, si_weight=1.0):
    if int(summary.get("finished_order_count", 0)) != rollout_end2end.pro_num:
        return None
    return float(summary["final_pulse"]) + float(si_weight) * float(
        summary["smoothness_index"]
    )


def is_better_online_balanced_summary(summary, best_score, si_weight=1.0):
    score = online_balanced_score(summary, si_weight=si_weight)
    if score is None:
        return False, None
    return best_score is None or score < best_score, score


def _tag_summary_path(agents, tag):
    return os.path.join(agents.policy.model_dir, f"{tag}_summary.json")


def _load_tag_summary(agents, tag):
    summary_path = _tag_summary_path(agents, tag)
    if not os.path.exists(summary_path):
        return None
    with open(summary_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_saved_end2end_score(agents, tag, mode="pulse_plus_si", si_weight=1.0):
    summary = _load_tag_summary(agents, tag)
    if summary is None:
        return None
    return end2end_best_score(summary, mode, si_weight=si_weight)


def load_saved_online_pulse_score(agents, tag):
    summary = _load_tag_summary(agents, tag)
    if summary is None:
        return None
    return online_pulse_score(summary)


def load_saved_online_balanced_score(agents, tag, si_weight=1.0):
    summary = _load_tag_summary(agents, tag)
    if summary is None:
        return None
    return online_balanced_score(summary, si_weight=si_weight)


def load_best_existing_online_balanced_score(agents, tags, si_weight=1.0):
    best_score = None
    for tag in tags:
        score = load_saved_online_balanced_score(agents, tag, si_weight=si_weight)
        if score is not None and (best_score is None or score < best_score):
            best_score = score
    return best_score


def best_existing_online_balanced_tag(agents, tags, si_weight=1.0):
    best_tag = None
    best_score = None
    for tag in tags:
        score = load_saved_online_balanced_score(agents, tag, si_weight=si_weight)
        if score is not None and (best_score is None or score < best_score):
            best_tag = tag
            best_score = score
    return best_tag, best_score


def copy_tagged_model(agents, source_tag, target_tag):
    if source_tag == target_tag:
        return
    model_dir = agents.policy.model_dir
    filenames = [
        "{}_drqn_net_params.pkl",
        "{}_{}_mixer_params.pkl".format("{}", agents.policy.mixer),
        "{}_gnn_encoder_params.pkl",
        "{}_combo_scorer_params.pkl",
        "{}_si_predictor_params.pkl",
        "{}_summary.json",
    ]
    for pattern in filenames:
        src = os.path.join(model_dir, pattern.format(source_tag))
        dst = os.path.join(model_dir, pattern.format(target_tag))
        if os.path.exists(src):
            shutil.copyfile(src, dst)


def save_tagged_model(agents, conf, summary, score, tag):
    model_dir = agents.policy.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    agents.policy._save_state_dict(
        agents.policy.eval_drqn_net.state_dict(),
        os.path.join(model_dir, f"{tag}_drqn_net_params.pkl"),
    )
    agents.policy._save_state_dict(
        agents.policy.eval_mixer_net.state_dict(),
        os.path.join(model_dir, f"{tag}_{agents.policy.mixer}_mixer_params.pkl"),
    )
    if agents.policy.use_gnn:
        agents.policy._save_state_dict(
            agents.policy.eval_graph_encoder.state_dict(),
            os.path.join(model_dir, f"{tag}_gnn_encoder_params.pkl"),
        )
    if getattr(agents.policy, "use_combo_scorer", False):
        agents.policy._save_state_dict(
            agents.policy.eval_combo_scorer.state_dict(),
            os.path.join(model_dir, f"{tag}_combo_scorer_params.pkl"),
        )
    best_summary = dict(summary)
    best_summary["best_score"] = float(score)
    best_summary["best_score_mode"] = getattr(conf, "end2end_best_score_mode", "pulse_plus_si")
    best_summary["best_si_weight"] = float(getattr(conf, "end2end_best_si_weight", 1.0))
    with open(os.path.join(model_dir, f"{tag}_summary.json"), "w", encoding="utf-8") as file_obj:
        json.dump(best_summary, file_obj, ensure_ascii=False, indent=2)


def save_best_model(agents, conf, summary, score):
    save_tagged_model(agents, conf, summary, score, "best")


def resolve_expert_summary_path(agents, conf):
    summary_path = (getattr(conf, "expert_summary_path", "") or "").strip()
    if summary_path:
        if not os.path.isabs(summary_path):
            summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), summary_path))
        return summary_path
    return _tag_summary_path(agents, "best")


def save_expert_trace_snapshot(agents, conf, source_summary, source_summary_path, expert_init_pulse):
    model_dir = agents.policy.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    snapshot = dict(source_summary)
    snapshot["expert_source_summary_path"] = os.path.abspath(source_summary_path)
    snapshot["expert_source_tag"] = "best"
    snapshot["expert_init_pulse"] = int(expert_init_pulse)
    snapshot["expert_seed"] = int(getattr(conf, "expert_seed", conf.seed))
    snapshot["expert_disable_disturbance"] = bool(
        getattr(conf, "expert_disable_disturbance", True)
    )
    snapshot["expert_action_ids"] = rollout_end2end.extract_action_ids_from_summary(source_summary)
    snapshot_path = os.path.join(
        model_dir,
        "{}_teacher_summary.json".format(getattr(conf, "expert_model_tag", "expert_trace_best")),
    )
    with open(snapshot_path, "w", encoding="utf-8") as file_obj:
        json.dump(snapshot, file_obj, ensure_ascii=False, indent=2)
    return snapshot_path, snapshot


def action_trace_match_metrics(expected_actions, actual_actions):
    total = max(len(expected_actions), len(actual_actions), 1)
    matched = sum(
        1 for expected, actual in zip(expected_actions, actual_actions) if int(expected) == int(actual)
    )
    prefix_len = 0
    for expected, actual in zip(expected_actions, actual_actions):
        if int(expected) != int(actual):
            break
        prefix_len += 1
    first_divergence_step = None
    if prefix_len < min(len(expected_actions), len(actual_actions)):
        first_divergence_step = int(prefix_len)
    elif len(expected_actions) != len(actual_actions):
        first_divergence_step = int(prefix_len)
    return {
        "target_len": int(len(expected_actions)),
        "predicted_len": int(len(actual_actions)),
        "match_rate": float(matched / float(total)),
        "prefix_match_length": int(prefix_len),
        "first_divergence_step": first_divergence_step,
    }


def build_expert_batch(agents, conf):
    if not getattr(agents.policy, "use_combo_scorer", False):
        raise ValueError("expert distillation requires END2END_ACTION_VALUE_MODE=pair_scorer")
    summary_path = resolve_expert_summary_path(agents, conf)
    if not os.path.exists(summary_path):
        print("expert distillation skipped: missing summary", summary_path)
        return None, None
    with open(summary_path, "r", encoding="utf-8") as file_obj:
        expert_summary = json.load(file_obj)
    expert_action_ids = rollout_end2end.extract_action_ids_from_summary(expert_summary)
    expert_disturbances = rollout_end2end.extract_disturbances_from_summary(expert_summary)
    if not expert_action_ids:
        print("expert distillation skipped: empty expert action trace")
        return None, None
    expert_init_pulse = int(getattr(conf, "expert_init_pulse", 0))
    if expert_init_pulse <= 0:
        expert_init_pulse = rollout_end2end.infer_teacher_init_pulse(
            expert_summary,
            getattr(conf, "end2end_validation_init_pulse", 608),
        )
    snapshot_path, snapshot = save_expert_trace_snapshot(
        agents,
        conf,
        expert_summary,
        summary_path,
        expert_init_pulse,
    )

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(conf.expert_seed)
    np.random.seed(conf.expert_seed)
    torch.manual_seed(conf.expert_seed)
    try:
        expert_episode, _, _, replay_summary = rollout_end2end.generate_episode(
            agents,
            conf,
            [],
            expert_init_pulse,
            -1,
            [],
            evaluate=True,
            forced_action_trace=expert_action_ids,
            forced_disturbance_trace=expert_disturbances,
            strict_forced_trace=True,
            disable_disturbance=True,
        )
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)

    replay_action_ids = rollout_end2end.extract_action_ids_from_summary(replay_summary)
    if replay_action_ids != expert_action_ids:
        raise RuntimeError("expert replay diverged from frozen action trace")
    match_rate = 1.0 if expert_action_ids else 0.0
    metadata = {
        "summary_path": summary_path,
        "snapshot_path": snapshot_path,
        "snapshot": snapshot,
        "expert_init_pulse": int(expert_init_pulse),
        "expert_action_count": int(len(expert_action_ids)),
        "expert_action_ids": expert_action_ids,
        "expert_disturbances": expert_disturbances,
        "expert_match_rate": float(match_rate),
        "replay_summary": replay_summary,
    }
    print(
        "loaded expert trace",
        summary_path,
        "actions",
        metadata["expert_action_count"],
        "init_pulse",
        metadata["expert_init_pulse"],
        "snapshot",
        snapshot_path,
    )
    return expert_episode, metadata


def evaluate_expert_trace_match(agents, conf, expert_meta):
    if expert_meta is None:
        return None, None
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(conf.expert_seed)
    np.random.seed(conf.expert_seed)
    torch.manual_seed(conf.expert_seed)
    try:
        _, _, _, summary = rollout_end2end.generate_episode(
            agents,
            conf,
            [],
            int(expert_meta["expert_init_pulse"]),
            -1,
            [],
            evaluate=True,
            forced_disturbance_trace=expert_meta.get("expert_disturbances"),
            disable_disturbance=True,
        )
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
    actual_actions = rollout_end2end.extract_action_ids_from_summary(summary)
    metrics = action_trace_match_metrics(expert_meta["expert_action_ids"], actual_actions)
    return metrics, summary


def is_better_expert_trace(metrics, current_best):
    if metrics is None:
        return False, None
    score = (
        -float(metrics["match_rate"]),
        -int(metrics["prefix_match_length"]),
        int(metrics["predicted_len"]),
    )
    return current_best is None or score < current_best, score


def validate_current_model(agents, conf, epoch):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(conf.end2end_validation_seed)
    np.random.seed(conf.end2end_validation_seed)
    torch.manual_seed(conf.end2end_validation_seed)
    try:
        _, _, _, summary = rollout_end2end.generate_episode(
            agents,
            conf,
            [],
            conf.end2end_validation_init_pulse,
            epoch,
            [],
            evaluate=True,
        )
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
    return summary


def train():
    set_seed(conf)
    print(
        "training entry: GNN_QATTEN_END2END_dis1.py, mixer:",
        conf.mixer,
        "action_mode:",
        conf.action_mode,
        "n_agents:",
        conf.n_agents,
        "n_actions:",
        conf.n_actions,
    )
    agents = Agents(conf)
    buffer = ReplayBuffer(conf)

    pulses = []
    si_values = []
    train_steps = 0
    validation_tag = getattr(conf, "end2end_validation_best_tag", "validation_best")
    online_tag = getattr(conf, "end2end_online_pulse_best_tag", "online_pulse_best")
    online_balanced_tag = getattr(
        conf, "end2end_online_balanced_best_tag", "online_balanced_best"
    )
    online_balanced_si_weight = getattr(
        conf,
        "end2end_online_balanced_si_weight",
        getattr(conf, "end2end_best_si_weight", 1.0),
    )
    best_score = load_saved_end2end_score(
        agents,
        "best",
        getattr(conf, "end2end_best_score_mode", "pulse_plus_si"),
        getattr(conf, "end2end_best_si_weight", 1.0),
    )
    validation_best_score = load_saved_end2end_score(
        agents,
        validation_tag,
        getattr(conf, "end2end_best_score_mode", "pulse_plus_si"),
        getattr(conf, "end2end_best_si_weight", 1.0),
    )
    online_pulse_best_score = load_saved_online_pulse_score(agents, online_tag)
    online_balanced_source_tag, online_balanced_best_score = best_existing_online_balanced_tag(
        agents,
        [
            online_balanced_tag,
            validation_tag,
            online_tag,
        ],
        online_balanced_si_weight,
    )
    if online_balanced_source_tag is not None and online_balanced_source_tag != online_balanced_tag:
        copy_tagged_model(agents, online_balanced_source_tag, online_balanced_tag)
        print(
            "synced online balanced best from",
            online_balanced_source_tag,
            "score",
            online_balanced_best_score,
        )
    expert_batch = None
    expert_meta = None
    expert_trace_best_score = None
    if getattr(conf, "enable_expert_distillation", False):
        expert_batch, expert_meta = build_expert_batch(agents, conf)
    if validation_best_score is not None:
        print("loaded existing validation best score", validation_best_score)
    if online_pulse_best_score is not None:
        print(
            "loaded existing online pulse best pulse",
            online_pulse_best_score[0],
            "si",
            online_pulse_best_score[1],
        )
    if online_balanced_best_score is not None:
        print(
            "loaded existing online balanced best score",
            online_balanced_best_score,
            "si_weight",
            online_balanced_si_weight,
        )
    if expert_meta is not None:
        print(
            "expert pretrain epochs",
            getattr(conf, "expert_pretrain_epochs", 0),
            "pretrain imitation weight",
            getattr(conf, "expert_pretrain_imitation_weight", 1.0),
            "finetune imitation weight",
            getattr(conf, "expert_finetune_imitation_weight", 0.0),
        )

    for epoch in range(conf.n_epochs):
        if not pulses:
            now_pulse = init_pulse
        else:
            now_pulse = min(int(min(pulses) - 1), now_pulse)

        pulses = []
        episodes = []
        pretrain_phase = expert_batch is not None and epoch < int(
            getattr(conf, "expert_pretrain_epochs", 0)
        )
        if pretrain_phase:
            print("expert pretrain epoch", epoch, "using frozen trace only")
        else:
            for _ in range(conf.n_eposodes):
                episode, _, _, summary = rollout_end2end.generate_episode(
                    agents, conf, pulses, now_pulse, epoch, si_values, evaluate=False
                )
                episodes.append(episode)
                print(
                    "epoch",
                    epoch,
                    "pulse",
                    now_pulse,
                    "finished",
                    summary.get("finished_order_count"),
                    "si",
                    summary.get("smoothness_index"),
                )
                is_better, score = is_better_end2end_summary(
                    summary,
                    best_score,
                    getattr(conf, "end2end_best_score_mode", "pulse_plus_si"),
                    getattr(conf, "end2end_best_si_weight", 1.0),
                )
                if is_better:
                    best_score = score
                    save_best_model(agents, conf, summary, score)
                    print("saved best end2end checkpoint score", best_score)

            episode_batch = episodes[0]
            for episode in episodes[1:]:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            buffer.store_episode(episode_batch)
        for _ in range(conf.train_steps):
            if pretrain_phase:
                agents.train(
                    expert_batch,
                    train_steps,
                    td_weight=0.0,
                    imitation_weight=getattr(conf, "expert_pretrain_imitation_weight", 1.0),
                )
            else:
                mini_batch = buffer.sample(min(buffer.current_size, conf.batch_size))
                agents.train(
                    mini_batch,
                    train_steps,
                    expert_batch=expert_batch,
                    td_weight=1.0,
                    imitation_weight=(
                        getattr(conf, "expert_finetune_imitation_weight", 0.0)
                        if expert_batch is not None
                        else 0.0
                    ),
                )
            train_steps += 1

        validation_interval = int(getattr(conf, "end2end_validation_interval", 0))
        if validation_interval > 0 and (epoch + 1) % validation_interval == 0:
            validation_summary = validate_current_model(agents, conf, epoch)
            is_better, score = is_better_end2end_summary(
                validation_summary,
                validation_best_score,
                getattr(conf, "end2end_best_score_mode", "pulse_plus_si"),
                getattr(conf, "end2end_best_si_weight", 1.0),
            )
            print(
                "validation",
                epoch,
                "pulse",
                validation_summary.get("final_pulse"),
                "finished",
                validation_summary.get("finished_order_count"),
                "si",
                validation_summary.get("smoothness_index"),
            )
            if is_better:
                validation_best_score = score
                save_tagged_model(
                    agents, conf, validation_summary, score, validation_tag
                )
                print("saved validation best end2end checkpoint score", validation_best_score)
            is_online_better, online_score = is_better_online_pulse_summary(
                validation_summary, online_pulse_best_score
            )
            if is_online_better:
                online_pulse_best_score = online_score
                save_tagged_model(
                    agents, conf, validation_summary, online_score[0], online_tag
                )
                print(
                    "saved online pulse best end2end checkpoint pulse",
                    online_pulse_best_score[0],
                    "si",
                    online_pulse_best_score[1],
                )
            is_balanced_better, balanced_score = is_better_online_balanced_summary(
                validation_summary,
                online_balanced_best_score,
                online_balanced_si_weight,
            )
            if is_balanced_better:
                online_balanced_best_score = balanced_score
                tagged_summary = dict(validation_summary)
                tagged_summary["online_balanced_si_weight"] = float(online_balanced_si_weight)
                save_tagged_model(
                    agents,
                    conf,
                    tagged_summary,
                    balanced_score,
                    online_balanced_tag,
                )
                print(
                    "saved online balanced best end2end checkpoint score",
                    online_balanced_best_score,
                    "si_weight",
                    online_balanced_si_weight,
                )

        expert_trace_interval = int(getattr(conf, "expert_trace_interval", 0))
        should_eval_expert_trace = (
            expert_meta is not None
            and expert_trace_interval > 0
            and (epoch + 1) % expert_trace_interval == 0
        )
        if should_eval_expert_trace:
            trace_metrics, trace_summary = evaluate_expert_trace_match(agents, conf, expert_meta)
            is_trace_better, trace_score = is_better_expert_trace(
                trace_metrics, expert_trace_best_score
            )
            print(
                "expert trace",
                epoch,
                "match",
                trace_metrics.get("match_rate"),
                "prefix",
                trace_metrics.get("prefix_match_length"),
                "first_divergence",
                trace_metrics.get("first_divergence_step"),
            )
            if is_trace_better:
                expert_trace_best_score = trace_score
                tagged_summary = dict(trace_summary)
                tagged_summary["expert_trace_metrics"] = trace_metrics
                tagged_summary["expert_source_summary_path"] = expert_meta["summary_path"]
                save_tagged_model(
                    agents,
                    conf,
                    tagged_summary,
                    -float(trace_metrics["match_rate"]),
                    getattr(conf, "expert_model_tag", "expert_trace_best"),
                )
                print("saved expert trace best checkpoint match", trace_metrics.get("match_rate"))

    if train_steps > 0 and train_steps % conf.save_frequency != 0:
        agents.policy.save_model(max(train_steps, conf.save_frequency))
    print("end2end SI raw values:", si_values)


if __name__ == "__main__":
    train()
