import numpy as np
import pandas as pd

from agent import Agents
from config_gnn_qatten import Config
from config import Config as QattenConfig
from parameter import args_parser
import rollout_dis
from utils import ReplayBuffer


conf = Config()
conf.load_model = False
print(
    "training entry: GNN_QATTEN_dis1.py, current mixer:",
    conf.mixer,
    "use_gnn:",
    conf.use_gnn,
    "use_gnn_graph_embedding:",
    getattr(conf, "use_gnn_graph_embedding", None),
    "use_gnn_action_bias:",
    getattr(conf, "use_gnn_action_bias", None),
    "gnn_aux_weight:",
    getattr(conf, "gnn_aux_weight", None),
    "model_tag:",
    conf.model_tag,
)

args = args_parser()

pro_num = args.pro_num
team_num = args.team_num
station_num = args.station_num
init_pulse = 750


def get_drqn_input_shape(conf):
    input_shape = conf.obs_shape
    if conf.last_action:
        input_shape += conf.n_actions
    if conf.reuse_network:
        input_shape += conf.n_agents
    if getattr(conf, "use_gnn", False) and getattr(conf, "use_gnn_graph_embedding", False):
        input_shape += conf.gnn_embed_dim
    return input_shape


def count_trainable_parameters(conf):
    conf.load_model = False
    agents = Agents(conf)
    return sum(param.numel() for param in agents.policy.eval_parameters if param.requires_grad)


def print_parity_report():
    qatten_conf = QattenConfig()
    parity_conf = Config()
    qatten_conf.load_model = False
    parity_conf.load_model = False

    qatten_params = count_trainable_parameters(qatten_conf)
    parity_params = count_trainable_parameters(parity_conf)
    print(f"QATTEN 参数量: {qatten_params}, Parity 参数量: {parity_params}")

    fields = [
        "seed",
        "continuous_final_reward",
        "pulse_reward_target",
        "pulse_reward_scale",
        "smoothness_reward_weight",
        "smoothness_reward_target",
        "gamma",
        "learning_rate",
        "start_epsilon",
        "end_epsilon",
        "anneal_steps",
        "anneal_epsilon",
        "epsilon_anneal_scale",
        "batch_size",
        "buffer_size",
        "train_steps",
        "update_target_params",
        "mixer",
        "qmix_hidden_dim",
        "two_hyper_layers",
        "hyper_hidden_dim",
        "n_attention_heads",
        "qatten_hidden_dim",
        "state_shape",
        "obs_shape",
        "n_actions",
        "n_agents",
        "episode_limit",
        "save_frequency",
    ]
    diffs = []
    for field in fields:
        qatten_value = getattr(qatten_conf, field)
        parity_value = getattr(parity_conf, field)
        if qatten_value != parity_value:
            diffs.append((field, qatten_value, parity_value))

    print("QATTEN input_shape:", get_drqn_input_shape(qatten_conf))
    print("Parity input_shape:", get_drqn_input_shape(parity_conf))
    print("QATTEN model_dir:", qatten_conf.model_dir)
    print("Parity model_dir:", parity_conf.model_dir)
    print("online rollout logic:", "both use Agents + ReplayBuffer + onlinerollout.generate_episode")
    if diffs:
        print("配置差异:")
        for field, qatten_value, parity_value in diffs:
            print(f"  {field}: QATTEN={qatten_value}, Parity={parity_value}")
    else:
        print("配置差异: none, except independent model_dir/model_tag/load_model path")

    if qatten_params != parity_params:
        raise RuntimeError("Parity 参数量与原 QATTEN 不一致，停止训练。")


def train():
    if conf.use_gnn:
        param_count_conf = Config()
        param_count_conf.load_model = False
        param_count_agents = Agents(param_count_conf)
        trainable_params = sum(
            param.numel()
            for param in param_count_agents.policy.eval_parameters
            if param.requires_grad
        )
        print("GNN-QATTEN trainable parameters:", trainable_params)
    else:
        print_parity_report()
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
            print("current pulse", now_pulse)
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
    agents.policy.save_model(conf.save_frequency)
    print(si_values)


if __name__ == "__main__":
    train()
