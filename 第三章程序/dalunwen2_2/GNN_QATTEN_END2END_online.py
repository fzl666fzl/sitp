import os
import random

import numpy as np
import torch

from agent import Agents
from config_gnn_qatten_end2end import Config
from utils import ReplayBuffer
import rollout_end2end


def _env_has_value(name):
    return os.getenv(name) not in (None, "")


def _tag_has_model_files(conf, tag):
    model_root = conf.model_dir
    if os.path.basename(model_root) != conf.map_name:
        model_root = os.path.join(model_root, conf.map_name)
    required = [
        os.path.join(model_root, f"{tag}_drqn_net_params.pkl"),
        os.path.join(model_root, f"{tag}_{conf.mixer}_mixer_params.pkl"),
    ]
    if getattr(conf, "use_gnn", False):
        required.append(os.path.join(model_root, f"{tag}_gnn_encoder_params.pkl"))
    if getattr(conf, "action_value_mode", None) == "pair_scorer":
        required.append(os.path.join(model_root, f"{tag}_combo_scorer_params.pkl"))
    return all(os.path.exists(path) for path in required)


def _tag_summary_score(conf, tag):
    model_root = conf.model_dir
    if os.path.basename(model_root) != conf.map_name:
        model_root = os.path.join(model_root, conf.map_name)
    summary_path = os.path.join(model_root, f"{tag}_summary.json")
    if not os.path.exists(summary_path):
        return None
    import json

    with open(summary_path, "r", encoding="utf-8") as file_obj:
        summary = json.load(file_obj)
    if int(summary.get("finished_order_count", 0)) != rollout_end2end.pro_num:
        return None
    si_weight = float(
        getattr(
            conf,
            "end2end_online_balanced_si_weight",
            getattr(conf, "end2end_best_si_weight", 1.0),
        )
    )
    return float(summary["final_pulse"]) + si_weight * float(summary["smoothness_index"])


def _default_online_model_tag(conf):
    candidates = [
        getattr(conf, "end2end_online_balanced_best_tag", "online_balanced_best"),
        getattr(conf, "end2end_validation_best_tag", "validation_best"),
        getattr(conf, "end2end_online_pulse_best_tag", "online_pulse_best"),
        "latest",
    ]
    scored_candidates = []
    for tag in candidates:
        if _tag_has_model_files(conf, tag):
            score = _tag_summary_score(conf, tag)
            if score is not None:
                scored_candidates.append((score, tag))
    if scored_candidates:
        scored_candidates.sort(key=lambda item: item[0])
        return scored_candidates[0][1]
    for tag in candidates:
        if _tag_has_model_files(conf, tag):
            return tag
    return candidates[0]


def configure_online_defaults(conf):
    conf.load_model = True
    if not _env_has_value("END2END_MODEL_TAG"):
        conf.model_tag = _default_online_model_tag(conf)
    return conf


conf = configure_online_defaults(Config())


def online_init_pulse(conf=None):
    if _env_has_value("END2END_ONLINE_INIT_PULSE"):
        return int(os.getenv("END2END_ONLINE_INIT_PULSE"))
    if conf is not None:
        return int(getattr(conf, "end2end_validation_init_pulse", 608))
    return int(os.getenv("END2END_VALIDATION_INIT_PULSE", "608"))


def set_eval_seed(conf):
    if _env_has_value("END2END_ONLINE_SEED"):
        conf.seed = int(os.getenv("END2END_ONLINE_SEED"))
    else:
        conf.seed = int(getattr(conf, "end2end_validation_seed", conf.seed))
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)


def print_online_summary(conf, init_pulse, summary):
    print("\n===== GNN Qatten end-to-end online summary =====")
    print("algorithm:", conf.mixer)
    print("action_mode:", conf.action_mode)
    print("action_value_mode:", getattr(conf, "action_value_mode", None))
    print("n_agents:", conf.n_agents)
    print("n_actions:", conf.n_actions)
    print("use_gnn:", conf.use_gnn)
    print("use_gnn_action_bias:", conf.use_gnn_action_bias)
    print("gnn_action_fusion_mode:", conf.gnn_action_fusion_mode)
    print("gnn_action_bias_weight:", conf.gnn_action_bias_weight)
    print("model_dir:", conf.model_dir)
    print("validation_best_tag:", getattr(conf, "end2end_validation_best_tag", None))
    print(
        "online_balanced_best_tag:",
        getattr(conf, "end2end_online_balanced_best_tag", None),
    )
    print("online_pulse_best_tag:", getattr(conf, "end2end_online_pulse_best_tag", None))
    print("load_model:", conf.load_model)
    print("configured checkpoint:", conf.model_tag)
    print("loaded checkpoint:", getattr(conf, "loaded_model_tag", None))
    print("loaded drqn path:", getattr(conf, "loaded_drqn_path", None))
    print("loaded mixer path:", getattr(conf, "loaded_mixer_path", None))
    print("loaded gnn path:", getattr(conf, "loaded_gnn_path", None))
    print("loaded combo scorer path:", getattr(conf, "loaded_combo_scorer_path", None))
    print("validation seed:", getattr(conf, "end2end_validation_seed", None))
    print("validation init pulse:", getattr(conf, "end2end_validation_init_pulse", None))
    print("eval seed:", conf.seed)
    print("initial pulse:", init_pulse)
    print("end2end_best_score_mode:", getattr(conf, "end2end_best_score_mode", None))
    print("end2end_best_si_weight:", getattr(conf, "end2end_best_si_weight", None))
    print(
        "end2end_final_si_penalty_weight:",
        getattr(conf, "end2end_final_si_penalty_weight", None),
    )
    print(
        "end2end_last_station_penalty_weight:",
        getattr(conf, "end2end_last_station_penalty_weight", None),
    )
    print("final pulse:", summary.get("final_pulse"))
    print("smoothness index:", summary.get("smoothness_index"))
    print("station times:", summary.get("station_times"))
    print("finished order count:", summary.get("finished_order_count"))
    print("actions:")
    for item in summary.get("actions", []):
        print(
            "  station {station}: action_id={action}, proc_id={proc}, team_id={team}, "
            "start={start:.2f}, finish={finish:.2f}, disturbance={disturbance:.2f}".format(
                station=item["station_id"] + 1,
                action=item["action_id"],
                proc=item["proc_id"],
                team=item["team_id"],
                start=item["start_time"],
                finish=item["finish_time"],
                disturbance=item.get("disturbance", 0.0),
            )
        )


def train():
    set_eval_seed(conf)
    agents = Agents(conf)
    buffer = ReplayBuffer(conf)
    pulses = []
    latest_summary = {}
    init_pulse = online_init_pulse(conf)

    for epoch in range(1):
        episode, _, _, latest_summary = rollout_end2end.generate_episode(
            agents, conf, pulses, init_pulse, epoch, [], evaluate=True
        )
        buffer.store_episode(episode)

    print_online_summary(conf, init_pulse, latest_summary)


if __name__ == "__main__":
    train()
