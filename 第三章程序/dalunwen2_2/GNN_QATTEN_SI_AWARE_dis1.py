import numpy as np

from agent import Agents
from config_gnn_qatten_si_aware import Config
from parameter import args_parser
import rollout_dis
from utils import ReplayBuffer


conf = Config()
conf.load_model = False
print(
    "training entry: GNN_QATTEN_SI_AWARE_dis1.py",
    "aux_target:",
    conf.gnn_aux_target_type,
    "aux_weight:",
    conf.gnn_aux_weight,
    "model_dir:",
    conf.model_dir,
)

args = args_parser()
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
        now_pulse = init_pulse if not pulses else min(int(min(pulses) - 1), now_pulse)
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
