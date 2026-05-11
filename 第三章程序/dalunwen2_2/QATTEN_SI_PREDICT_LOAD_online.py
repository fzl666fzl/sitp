import random

import numpy as np
import torch

from agent import Agents
from config_qatten_si_predict_load import Config
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
        "online entry: QATTEN_SI_PREDICT_LOAD_online.py",
        "model_dir:",
        conf.model_dir,
        "model_tag:",
        conf.model_tag,
        "feature_dim:",
        conf.si_predict_feature_dim,
    )

    agents = Agents(conf)
    buffer = ReplayBuffer(conf)
    episode, _, _, summary = onlinerollout.generate_episode(
        agents, conf, [], INIT_PULSE, 0, [], evaluate=True
    )
    buffer.store_episode(episode)
    metrics = agents.policy.evaluate_si_prediction_batch(episode, conf.episode_limit)

    print("\n===== QATTEN SI-predict LOAD online summary =====")
    print("loaded checkpoint:", getattr(conf, "loaded_model_tag", None))
    print("loaded drqn path:", getattr(conf, "loaded_drqn_path", None))
    print("loaded mixer path:", getattr(conf, "loaded_mixer_path", None))
    print("loaded si predictor path:", getattr(conf, "loaded_si_predictor_path", None))
    print("final pulse:", summary.get("final_pulse"))
    print("smoothness index:", summary.get("smoothness_index"))
    print("station times:", summary.get("station_times"))
    print("station actions:", [
        (item.get("station_id"), item.get("proc_rule"), item.get("team_rule"))
        for item in summary.get("station_actions", [])
    ])
    if metrics:
        print("station_time_mae:", round(metrics["station_time_mae"], 6))
        print("si_mae:", round(metrics["si_mae"], 6))
        print("si_corr:", metrics["si_corr"])
        print("pred_station_times_mean:", [round(x, 6) for x in metrics["pred_station_times_mean"]])
        print("true_station_times_mean:", [round(x, 6) for x in metrics["true_station_times_mean"]])
        print("pred_si_mean:", round(metrics["pred_si_mean"], 6))
        print("true_si_mean:", round(metrics["true_si_mean"], 6))


if __name__ == "__main__":
    train()
