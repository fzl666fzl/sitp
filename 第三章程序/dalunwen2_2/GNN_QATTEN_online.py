import random

import numpy as np
import torch

from agent import Agents
from config_gnn_qatten import Config
import onlinerollout
from parameter import args_parser
from utils import ReplayBuffer


conf = Config()
conf.load_model = True
print(
    "online entry: GNN_QATTEN_online.py, current mixer:",
    conf.mixer,
    "use_gnn:",
    conf.use_gnn,
    "load_model:",
    conf.load_model,
    "model_tag:",
    conf.model_tag,
)

args = args_parser()

pro_num = args.pro_num
team_num = args.team_num
station_num = args.station_num
init_pulse = 608


def set_eval_seed(conf):
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)


def print_online_summary(conf, init_pulse, summary):
    print("\n===== GNN Qatten online summary =====")
    print("algorithm:", conf.mixer)
    print("use_gnn:", conf.use_gnn)
    print("use_gnn_graph_embedding:", getattr(conf, "use_gnn_graph_embedding", None))
    print("use_gnn_action_bias:", getattr(conf, "use_gnn_action_bias", None))
    print("gnn_action_fusion_mode:", getattr(conf, "gnn_action_fusion_mode", None))
    print("gnn_action_bias_weight:", getattr(conf, "gnn_action_bias_weight", None))
    print("gnn_margin_threshold:", getattr(conf, "gnn_margin_threshold", None))
    print("gnn_topk:", getattr(conf, "gnn_topk", None))
    print("gnn_rerank_weight:", getattr(conf, "gnn_rerank_weight", None))
    print("gnn_use_load_penalty:", getattr(conf, "gnn_use_load_penalty", None))
    print("gnn_load_penalty_weight:", getattr(conf, "gnn_load_penalty_weight", None))
    print("zero_gnn_embedding:", getattr(conf, "zero_gnn_embedding", False))
    print("load_model:", conf.load_model)
    print("configured checkpoint:", conf.model_tag)
    print("loaded checkpoint:", getattr(conf, "loaded_model_tag", None))
    print("loaded drqn path:", getattr(conf, "loaded_drqn_path", None))
    print("loaded mixer path:", getattr(conf, "loaded_mixer_path", None))
    print("loaded gnn path:", getattr(conf, "loaded_gnn_path", None))
    print("trainable parameters:", getattr(conf, "online_trainable_parameters", None))
    print("input shape:", getattr(conf, "online_input_shape", None))
    print("eval mode:", "fixed seed + epsilon=0")
    print("eval seed:", conf.seed)
    print("initial pulse:", init_pulse)
    print("final pulse:", summary.get("final_pulse"))
    print("smoothness index:", summary.get("smoothness_index"))
    print("station times:", summary.get("station_times"))
    print("station actions:")
    for item in summary.get("station_actions", []):
        station_no = item["station_id"] + 1
        print(
            f"  station {station_no}: proc_rule={item['proc_rule']}, "
            f"team_rule={item['team_rule']}"
        )


def train():
    set_eval_seed(conf)
    agents = Agents(conf)
    conf.online_trainable_parameters = sum(
        param.numel() for param in agents.policy.eval_parameters if param.requires_grad
    )
    input_shape = conf.obs_shape
    if conf.last_action:
        input_shape += conf.n_actions
    if conf.reuse_network:
        input_shape += conf.n_agents
    if getattr(conf, "use_gnn", False) and getattr(conf, "use_gnn_graph_embedding", False):
        input_shape += conf.gnn_embed_dim
    conf.online_input_shape = input_shape
    buffer = ReplayBuffer(conf)

    pulses = []
    latest_summary = {}

    for epoch in range(1):
        episodes = []

        if not pulses:
            now_pulse = init_pulse
        else:
            now_pulse = min(int(min(pulses) - 1), now_pulse)

        pulses = []
        for _ in range(conf.n_eposodes):
            episode, _, _, summary = onlinerollout.generate_episode(
                agents, conf, pulses, now_pulse, epoch, [], evaluate=True
            )
            episodes.append(episode)
            latest_summary = summary

        episode_batch = episodes[0]
        episodes.pop(0)
        for episode in episodes:
            for key in episode_batch.keys():
                episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

        buffer.store_episode(episode_batch)

    print_online_summary(conf, init_pulse, latest_summary)


if __name__ == "__main__":
    train()
