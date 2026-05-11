import random

import numpy as np
import torch

from agent import Agents
from config_qatten_si_predict import Config
import rollout_dis
from utils import ReplayBuffer


INIT_PULSE = 750


def set_train_seed(conf):
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)


def train():
    conf = Config()
    conf.load_model = False
    set_train_seed(conf)
    print(
        "training entry: QATTEN_SI_PREDICT_dis1.py",
        "model_dir:",
        conf.model_dir,
        "aux_weight:",
        conf.si_predict_aux_weight,
        "seed:",
        conf.seed,
    )

    agents = Agents(conf)
    buffer = ReplayBuffer(conf)
    train_steps = 0
    pulses = []
    all_pulse = []
    si_values = []

    for epoch in range(conf.n_epochs):
        episodes = []
        now_pulse = INIT_PULSE if not pulses else min(int(min(pulses) - 1), now_pulse)
        pulses = []
        for _ in range(conf.n_eposodes):
            episode = rollout_dis.generate_episode(
                agents, conf, pulses, now_pulse, epoch, si_values
            )
            episodes.append(episode)
            all_pulse.append(now_pulse)

        episode_batch = episodes[0]
        for episode in episodes[1:]:
            for key in episode_batch.keys():
                episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

        buffer.store_episode(episode_batch)
        if epoch > 20:
            mini_batch = buffer.sample(min(buffer.current_size, conf.batch_size))
            agents.train(mini_batch, train_steps)
            train_steps += 1

    agents.policy.save_model(conf.save_frequency)
    print("training finished:", {
        "train_steps": train_steps,
        "best_pulse_seen": min(all_pulse) if all_pulse else None,
        "si_records": si_values,
    })


if __name__ == "__main__":
    train()
