import random

import numpy as np
import torch

from agent import Agents
from config_gnn_qatten_si_aware import Config
import onlinerollout
from utils import ReplayBuffer


INIT_PULSE = 608


def set_eval_seed(conf):
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)


def train():
    conf = Config()
    conf.load_model = True
    set_eval_seed(conf)
    print(
        "online entry: GNN_QATTEN_SI_AWARE_online.py",
        "aux_target:",
        conf.gnn_aux_target_type,
        "model_dir:",
        conf.model_dir,
        "model_tag:",
        conf.model_tag,
        "fusion:",
        conf.gnn_action_fusion_mode,
    )
    agents = Agents(conf)
    buffer = ReplayBuffer(conf)
    pulses = []
    episode, _, _, summary = onlinerollout.generate_episode(
        agents, conf, pulses, INIT_PULSE, 0, [], evaluate=True
    )
    buffer.store_episode(episode)

    print("\n===== GNN Qatten SI-aware online summary =====")
    print("loaded checkpoint:", getattr(conf, "loaded_model_tag", None))
    print("loaded drqn path:", getattr(conf, "loaded_drqn_path", None))
    print("loaded mixer path:", getattr(conf, "loaded_mixer_path", None))
    print("loaded gnn path:", getattr(conf, "loaded_gnn_path", None))
    print("final pulse:", summary.get("final_pulse"))
    print("smoothness index:", summary.get("smoothness_index"))
    print("station times:", summary.get("station_times"))
    print("station actions:", [
        (item.get("station_id"), item.get("proc_rule"), item.get("team_rule"))
        for item in summary.get("station_actions", [])
    ])


if __name__ == "__main__":
    train()
