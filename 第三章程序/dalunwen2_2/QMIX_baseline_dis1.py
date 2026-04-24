import numpy as np
import pandas as pd

from parameter import args_parser
from agent import Agents
from utils import ReplayBuffer
import rollout_dis

from config_qmix_baseline import Config


conf = Config()
conf.load_model = False
print(
    "training entry: QMIX_baseline_dis1.py, current mixer:",
    conf.mixer,
    "model_tag:",
    conf.model_tag,
)

args = args_parser()

pro_num = args.pro_num
team_num = args.team_num
station_num = args.station_num
init_pulse = 750


def train():
    agents = Agents(conf)
    buffer = ReplayBuffer(conf)

    pulses = []
    all_pulse = []
    si_values = []
    train_steps = 0

    for epoch in range(conf.n_epochs):
        episodes = []

        if not pulses:
            now_pulse = init_pulse
        else:
            now_pulse = min(int(min(pulses) - 1), now_pulse)

        pulses = []
        for _ in range(conf.n_eposodes):
            episode = rollout_dis.generate_episode(agents, conf, pulses, now_pulse, epoch, si_values)
            episodes.append(episode)
            print("当前的节拍为", now_pulse)
            all_pulse.append(now_pulse)

        episode_batch = episodes[0]
        episodes.pop(0)
        for episode in episodes:
            for key in episode_batch.keys():
                episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

        buffer.store_episode(episode_batch)

        if epoch > 20:
            mini_batch = buffer.sample(min(buffer.current_size, conf.batch_size))
            agents.train(mini_batch, train_steps)
            train_steps += 1

    pd.DataFrame(all_pulse)
    print(si_values)


if __name__ == "__main__":
    train()
