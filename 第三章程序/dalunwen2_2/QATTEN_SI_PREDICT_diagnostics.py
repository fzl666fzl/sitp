import random

import numpy as np
import torch

from agent import Agents
from config_qatten_si_predict import Config
import rollout_dis
from utils import ReplayBuffer


INIT_PULSE = 608


def set_eval_seed(conf):
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)


def main():
    conf = Config()
    conf.load_model = True
    set_eval_seed(conf)
    print(
        "diagnostics entry: QATTEN_SI_PREDICT_diagnostics.py",
        "model_dir:",
        conf.model_dir,
        "model_tag:",
        conf.model_tag,
    )

    agents = Agents(conf)
    buffer = ReplayBuffer(conf)
    pulses = []
    si_values = []
    episode = rollout_dis.generate_episode(
        agents, conf, pulses, INIT_PULSE, 0, si_values, evaluate=True
    )
    buffer.store_episode(episode)
    metrics = agents.policy.evaluate_si_prediction_batch(episode, conf.episode_limit)

    print("\n===== QATTEN SI-predict diagnostics =====")
    print("loaded checkpoint:", getattr(conf, "loaded_model_tag", None))
    print("loaded drqn path:", getattr(conf, "loaded_drqn_path", None))
    print("loaded mixer path:", getattr(conf, "loaded_mixer_path", None))
    print("loaded si predictor path:", getattr(conf, "loaded_si_predictor_path", None))
    print("final_pulse:", pulses[-1] if pulses else None)
    print("SI:", round(float(si_values[-1] ** 0.5), 6) if si_values else None)
    if metrics:
        print("station_time_mae:", round(metrics["station_time_mae"], 6))
        print("si_mae:", round(metrics["si_mae"], 6))
        print("si_corr:", metrics["si_corr"])
        print("pred_station_times_mean:", [round(x, 6) for x in metrics["pred_station_times_mean"]])
        print("true_station_times_mean:", [round(x, 6) for x in metrics["true_station_times_mean"]])
        print("pred_si_mean:", round(metrics["pred_si_mean"], 6))
        print("true_si_mean:", round(metrics["true_si_mean"], 6))


if __name__ == "__main__":
    main()
