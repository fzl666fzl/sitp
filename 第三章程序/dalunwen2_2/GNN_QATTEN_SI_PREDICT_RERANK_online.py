import random

import numpy as np
import torch

from agent import Agents
from config_gnn_qatten_si_predict_rerank import Config
import onlinerollout
from utils import ReplayBuffer


INIT_PULSE = 608


def set_eval_seed(conf):
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)


def _actions(summary):
    return [
        (item.get("station_id"), item.get("proc_rule"), item.get("team_rule"))
        for item in summary.get("station_actions", [])
    ]


def train():
    conf = Config()
    conf.load_model = True
    conf.record_gnn_diagnostics = True
    conf.gnn_diagnostic_records = []
    set_eval_seed(conf)
    print(
        "online entry: GNN_QATTEN_SI_PREDICT_RERANK_online.py",
        "fusion_mode:",
        conf.gnn_action_fusion_mode,
        "margin_threshold:",
        getattr(conf, "gnn_margin_threshold", None),
        "topk:",
        getattr(conf, "gnn_topk", None),
        "gnn_weight:",
        getattr(conf, "gnn_rerank_weight", None),
        "si_penalty_weight:",
        getattr(conf, "si_predict_penalty_weight", None),
    )

    agents = Agents(conf)
    buffer = ReplayBuffer(conf)
    episode, _, _, summary = onlinerollout.generate_episode(
        agents, conf, [], INIT_PULSE, 0, [], evaluate=True
    )
    buffer.store_episode(episode)
    trace = summary.get("gnn_trace_summary") or {}

    print("\n===== GNN QATTEN SI-predict rerank online summary =====")
    print("loaded checkpoint:", getattr(conf, "loaded_model_tag", None))
    print("loaded drqn path:", getattr(conf, "loaded_drqn_path", None))
    print("loaded mixer path:", getattr(conf, "loaded_mixer_path", None))
    print("loaded gnn path:", getattr(conf, "loaded_gnn_path", None))
    print("loaded si predictor path:", getattr(conf, "loaded_si_predictor_path", None))
    print("final pulse:", summary.get("final_pulse"))
    print("smoothness index:", summary.get("smoothness_index"))
    print("station times:", summary.get("station_times"))
    print("station actions:", _actions(summary))
    print("agent0 executed action changed count:", trace.get("executed_action_changed_count"))
    print("agent0 changed count:", trace.get("agent0_changed_count"))
    print("changed_action_pred_si_gain_mean:", trace.get("changed_action_pred_si_gain_mean"))
    print(
        "selected_pred_si_lower_than_original_rate:",
        trace.get("selected_pred_si_lower_than_original_rate"),
    )
    print("rerank candidate sizes:", trace.get("rerank_candidate_size_per_decision"))
    return summary


if __name__ == "__main__":
    train()
