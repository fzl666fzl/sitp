import random

import numpy as np
import torch

from agent import Agents
from config_qatten_load_rerank import Config
import onlinerollout


INIT_PULSE = 608


def set_eval_seed(conf):
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)


def format_actions(summary):
    return [
        (item.get("station_id"), item.get("proc_rule"), item.get("team_rule"))
        for item in summary.get("station_actions", [])
    ]


def summarize_records(records):
    if not records:
        return {
            "total_decisions": 0,
            "changed_count": 0,
            "changed_rate": 0.0,
            "candidate_sizes": [],
            "changed_details": [],
        }
    changed = [item for item in records if item.get("executed_action_changed")]
    details = []
    for item in changed[:3]:
        details.append(
            {
                "step": item.get("step"),
                "station_id": item.get("station_id"),
                "q_margin": item.get("q_margin"),
                "original_action": item.get("original_action"),
                "rerank_action": item.get("rerank_action"),
                "candidate_actions": item.get("candidate_actions"),
                "load_penalty": item.get("load_penalty"),
                "normalised_load_penalty": item.get("normalised_load_penalty"),
            }
        )
    return {
        "total_decisions": len(records),
        "changed_count": len(changed),
        "changed_rate": round(len(changed) / max(len(records), 1), 6),
        "candidate_sizes": [item.get("candidate_size") for item in records],
        "changed_details": details,
    }


def run_online(mode=None, threshold=None, topk=None, penalty_weight=None):
    conf = Config()
    if mode is not None:
        conf.qatten_load_rerank_mode = mode
    if threshold is not None:
        conf.qatten_load_margin_threshold = float(threshold)
    if topk is not None:
        conf.qatten_load_topk = int(topk)
    if penalty_weight is not None:
        conf.qatten_load_penalty_weight = float(penalty_weight)
    conf.load_rerank_records = []
    set_eval_seed(conf)

    agents = Agents(conf)
    episode, _, _, summary = onlinerollout.generate_episode(
        agents, conf, [], INIT_PULSE, 0, [], evaluate=True
    )
    records = list(getattr(conf, "load_rerank_records", []))
    return conf, summary, records, episode


def main():
    conf, summary, records, _ = run_online()
    record_summary = summarize_records(records)

    print("\n===== QATTEN load-rerank online summary =====")
    print("checkpoint root:", conf.model_dir)
    print("model tag:", conf.model_tag)
    print("loaded checkpoint:", getattr(conf, "loaded_model_tag", None))
    print("loaded drqn path:", getattr(conf, "loaded_drqn_path", None))
    print("loaded mixer path:", getattr(conf, "loaded_mixer_path", None))
    print("mode:", conf.qatten_load_rerank_mode)
    print("margin threshold:", conf.qatten_load_margin_threshold)
    print("topk:", conf.qatten_load_topk)
    print("penalty weight:", conf.qatten_load_penalty_weight)
    print("initial pulse:", INIT_PULSE)
    print("final pulse:", summary.get("final_pulse"))
    print("smoothness index:", summary.get("smoothness_index"))
    print("station times:", summary.get("station_times"))
    print("station actions:", format_actions(summary))
    print("load rerank diagnostics:", record_summary)


if __name__ == "__main__":
    main()
