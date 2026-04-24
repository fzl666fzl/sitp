import random

import numpy as np
import torch

from parameter import args_parser
from agent import Agents
from utils import ReplayBuffer
import onlinerollout

from config_qmix_baseline import Config


conf = Config()
conf.load_model = True
print(
    "online entry: QMIX_baseline_online.py, current mixer:",
    conf.mixer,
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
    print("\n===== 在线结果摘要 =====")
    print("算法:", conf.mixer)
    print("是否加载模型:", conf.load_model)
    print("配置checkpoint:", conf.model_tag)
    print("实际加载checkpoint:", getattr(conf, "loaded_model_tag", None))
    print("评估模式:", "固定seed + epsilon=0")
    print("评估seed:", conf.seed)
    print("初始节拍:", init_pulse)
    print("最终节拍:", summary.get("final_pulse"))
    print("平滑指数:", summary.get("smoothness_index"))
    print("各站位完工时间:", summary.get("station_times"))
    print("各站位动作:")
    for item in summary.get("station_actions", []):
        station_no = item["station_id"] + 1
        print(
            f"  站位{station_no}: 工序规则={item['proc_rule']}, "
            f"班组规则={item['team_rule']}"
        )


def train():
    set_eval_seed(conf)
    agents = Agents(conf)
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
