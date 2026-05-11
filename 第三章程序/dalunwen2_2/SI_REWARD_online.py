import random

import numpy as np
import torch

from agent import Agents
from config_si_reward_ablation import Config
from utils import ReplayBuffer
import onlinerollout


INIT_PULSE = 608


def set_eval_seed(conf):
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)


def train():
    conf = Config()
    conf.load_model = True
    conf.model_tag = "latest"
    set_eval_seed(conf)
    print(
        "online entry: SI_REWARD_online.py",
        "mode:",
        conf.final_reward_mode,
        "model_dir:",
        conf.model_dir,
        "model_tag:",
        conf.model_tag,
    )

    agents = Agents(conf)
    buffer = ReplayBuffer(conf)
    pulses = []
    episode, _, _, summary = onlinerollout.generate_episode(
        agents, conf, pulses, INIT_PULSE, 0, [], evaluate=True
    )
    buffer.store_episode(episode)

    print("\n===== SI reward ablation online summary =====")
    print("mode:", conf.final_reward_mode)
    print("loaded checkpoint:", getattr(conf, "loaded_model_tag", None))
    print("loaded drqn path:", getattr(conf, "loaded_drqn_path", None))
    print("loaded mixer path:", getattr(conf, "loaded_mixer_path", None))
    print("initial pulse:", INIT_PULSE)
    print("final pulse:", summary.get("final_pulse"))
    print("smoothness index:", summary.get("smoothness_index"))
    print("station times:", summary.get("station_times"))
    print("station actions:", [
        (item.get("station_id"), item.get("proc_rule"), item.get("team_rule"))
        for item in summary.get("station_actions", [])
    ])


if __name__ == "__main__":
    train()
