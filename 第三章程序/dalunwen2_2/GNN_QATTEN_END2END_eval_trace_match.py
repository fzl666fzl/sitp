import json
import os
import random

import numpy as np
import torch

from agent import Agents
from config_gnn_qatten_end2end import Config
from GNN_QATTEN_END2END_online import configure_online_defaults
import rollout_end2end


def resolve_expert_summary_path(conf, agents):
    summary_path = (getattr(conf, "expert_summary_path", "") or "").strip()
    if summary_path:
        if not os.path.isabs(summary_path):
            summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), summary_path))
        return summary_path
    tagged_path = os.path.join(
        agents.policy.model_dir,
        "{}_teacher_summary.json".format(getattr(conf, "expert_model_tag", "expert_trace_best")),
    )
    if os.path.exists(tagged_path):
        return tagged_path
    return os.path.join(agents.policy.model_dir, "best_summary.json")


def load_expert_summary(conf, agents):
    summary_path = resolve_expert_summary_path(conf, agents)
    if not os.path.exists(summary_path):
        raise FileNotFoundError("missing expert summary: {}".format(summary_path))
    with open(summary_path, "r", encoding="utf-8") as file_obj:
        return summary_path, json.load(file_obj)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def trace_match_metrics(target_actions, predicted_actions):
    total = max(len(target_actions), len(predicted_actions), 1)
    matched = sum(
        1 for expected, actual in zip(target_actions, predicted_actions) if int(expected) == int(actual)
    )
    prefix_len = 0
    for expected, actual in zip(target_actions, predicted_actions):
        if int(expected) != int(actual):
            break
        prefix_len += 1
    first_divergence_step = None
    if prefix_len < min(len(target_actions), len(predicted_actions)):
        first_divergence_step = int(prefix_len)
    elif len(target_actions) != len(predicted_actions):
        first_divergence_step = int(prefix_len)
    return {
        "target_len": int(len(target_actions)),
        "predicted_len": int(len(predicted_actions)),
        "match_rate": float(matched / float(total)),
        "prefix_match_length": int(prefix_len),
        "first_divergence_step": first_divergence_step,
    }


def main():
    conf = configure_online_defaults(Config())
    conf.load_model = True
    agents = Agents(conf)
    summary_path, expert_summary = load_expert_summary(conf, agents)
    expert_actions = rollout_end2end.extract_action_ids_from_summary(expert_summary)
    expert_disturbances = rollout_end2end.extract_disturbances_from_summary(expert_summary)
    expert_init_pulse = int(getattr(conf, "expert_init_pulse", 0))
    if expert_init_pulse <= 0:
        expert_init_pulse = rollout_end2end.infer_teacher_init_pulse(
            expert_summary,
            getattr(conf, "end2end_validation_init_pulse", 608),
        )
    eval_seed = int(getattr(conf, "expert_seed", conf.seed))
    set_seed(eval_seed)
    _, _, _, summary = rollout_end2end.generate_episode(
        agents,
        conf,
        [],
        expert_init_pulse,
        0,
        [],
        evaluate=True,
        forced_disturbance_trace=expert_disturbances,
        disable_disturbance=True,
    )
    predicted_actions = rollout_end2end.extract_action_ids_from_summary(summary)
    metrics = trace_match_metrics(expert_actions, predicted_actions)

    print("\n===== End-to-end Trace Match =====")
    print("model_dir:", conf.model_dir)
    print("configured checkpoint:", conf.model_tag)
    print("loaded checkpoint:", getattr(conf, "loaded_model_tag", None))
    print("expert summary:", summary_path)
    print("eval seed:", eval_seed)
    print("initial pulse:", expert_init_pulse)
    print("target actions:", metrics["target_len"])
    print("predicted actions:", metrics["predicted_len"])
    print("action match rate:", metrics["match_rate"])
    print("prefix match length:", metrics["prefix_match_length"])
    print("first divergence step:", metrics["first_divergence_step"])
    print("final pulse:", summary.get("final_pulse"))
    print("smoothness index:", summary.get("smoothness_index"))
    print("station times:", summary.get("station_times"))
    print("finished order count:", summary.get("finished_order_count"))


if __name__ == "__main__":
    main()
