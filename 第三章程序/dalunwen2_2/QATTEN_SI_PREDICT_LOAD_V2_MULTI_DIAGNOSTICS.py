import os
import random

import numpy as np
import torch

from agent import Agents
from config_qatten_si_predict_load_v2 import Config
import rollout_dis


INIT_PULSE = 608


def set_eval_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stack_episodes(episodes):
    keys = ["s", "u_onehot", "si_predict_features", "final_station_times", "final_si", "padded"]
    return {key: np.asarray([episode[key] for episode in episodes]) for key in keys}


def main():
    conf = Config()
    conf.load_model = True
    episode_count = int(os.getenv("SI_PREDICT_DIAG_EPISODES", "50"))
    epsilon = float(os.getenv("SI_PREDICT_DIAG_EPSILON", "0.4"))
    base_seed = int(os.getenv("SI_PREDICT_DIAG_SEED", str(conf.seed)))
    print(
        "diagnostics entry: QATTEN_SI_PREDICT_LOAD_V2_MULTI_DIAGNOSTICS.py",
        "model_dir:",
        conf.model_dir,
        "model_tag:",
        conf.model_tag,
        "episodes:",
        episode_count,
        "epsilon:",
        epsilon,
        "target_mode:",
        conf.si_predict_target_mode,
        "feature_dim:",
        conf.si_predict_feature_dim,
    )

    set_eval_seed(base_seed)
    agents = Agents(conf)
    episodes = []
    final_pulses = []
    si_values = []

    original_start_epsilon = conf.start_epsilon
    conf.start_epsilon = epsilon
    try:
        for episode_idx in range(episode_count):
            set_eval_seed(base_seed + episode_idx)
            pulses = []
            si_records = []
            episode = rollout_dis.generate_episode(
                agents, conf, pulses, INIT_PULSE, episode_idx, si_records, evaluate=False
            )
            episodes.append(episode)
            if pulses:
                final_pulses.append(float(pulses[-1]))
            if si_records:
                si_values.append(float(si_records[-1] ** 0.5))
    finally:
        conf.start_epsilon = original_start_epsilon

    episode_batch = stack_episodes(episodes)
    metrics = agents.policy.evaluate_si_prediction_batch(episode_batch, conf.episode_limit)

    print("\n===== QATTEN SI-predict LOAD V2 multi diagnostics =====")
    print("loaded checkpoint:", getattr(conf, "loaded_model_tag", None))
    print("loaded drqn path:", getattr(conf, "loaded_drqn_path", None))
    print("loaded mixer path:", getattr(conf, "loaded_mixer_path", None))
    print("loaded si predictor path:", getattr(conf, "loaded_si_predictor_path", None))
    print("episode_count:", len(episodes))
    print("final_pulse_mean:", round(float(np.mean(final_pulses)), 6) if final_pulses else None)
    print("final_pulse_min:", round(float(np.min(final_pulses)), 6) if final_pulses else None)
    print("final_pulse_max:", round(float(np.max(final_pulses)), 6) if final_pulses else None)
    print("SI_mean:", round(float(np.mean(si_values)), 6) if si_values else None)
    print("SI_min:", round(float(np.min(si_values)), 6) if si_values else None)
    print("SI_max:", round(float(np.max(si_values)), 6) if si_values else None)
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
