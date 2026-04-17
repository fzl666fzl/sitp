'''
TODO:
    2022.11.28
    本文件用于基于DQN的强化学习指导装配线生产，在运行此文件的主函数时，通过改变self.action来改变工序排序的规则
    2023.1.6
    或许采样的方式不对，更新的方式不对，不能很好地收敛
    2023.2.20
    单智能体
'''
import simpy
import numpy as np
from parameter import args_parser
import pandas as pd
from agent import Agents
import random
from collections import deque
from utils import ReplayBuffer
import rollout0513
import simpy


from config import Config
conf = Config()


args = args_parser()

pro_num = args.pro_num
team_num = args.team_num
station_num = args.station_num
init_pulse = 710
action_set = [[1,0,0],[0.9,0.2,0.8],[0.9,0.4,0.8],
              [0.6,0.2,0.2],[0.5,0.2,0.3],[0.2,0.5,0.3],
              [0.2,0,0.8],[0.2,0.3,0.5],[0.2,0.8,0]]

def train():

    agents = Agents(conf)


    buffer = ReplayBuffer(conf)

    # save plt and pk

    win_rates = []
    episode_rewards = []
    train_steps = 0
    pulses = []
    all_pulse = []
    actions = []
    siis = [10]
    for epoch in range(conf.n_epochs):
        # print("train epoch: %d" % epoch)
        episodes = []

        if not pulses:
            now_pulse = init_pulse
        else:
            now_pulse = min((min(pulses)-1),now_pulse)
        pulses = []
        for episode_idx in range(conf.n_eposodes):##n_eposodes=1
            episode= rollout0513.generate_episode(agents, conf,pulses,now_pulse,epoch,actions,siis)
            episodes.append(episode)

            print("当前的节拍为",now_pulse)
            print("平滑指数是：", siis[-1])

            # if pulses[-1] < 620:
            #     print("当前的节拍为", pulses[-1])
            #     print("当前的动作为",actions[-1])
            all_pulse.append(now_pulse)
        episode_batch = episodes[0]

        episodes.pop(0)
        for episode in episodes:
            for key in episode_batch.keys():
                episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)


        buffer.store_episode(episode_batch)
        # print(episode_batch)
         # (1, 200, 3, 42)
        # print(episode_batch['s'].shape)  # (1, 200, 61)
        # print(episode_batch['u'].shape)  # (1, 200, 3, 1)
        # print(episode_batch['r'].shape)  # (1, 200, 1)
        # print(episode_batch['o_'].shape) # (1, 200, 3, 42)
        # print(episode_batch['s_'].shape) # (1, 200, 61)
        # print(episode_batch['avail_u'].shape)  # (1, 200, 3, 10)
        # print(episode_batch['avail_u_'].shape) # (1, 200, 3, 10)
        # print(episode_batch['padded'].shape)   # (1, 200, 1)
        # print(episode_batch['terminated'].shape) # (1, 200, 1)

        # for train_step in range(conf.train_steps):
        #     mini_batch = buffer.sample(min(buffer.current_size, conf.batch_size))  # obs； (64, 200, 3, 42)
        #     # print(mini_batch['o'])
        #     agents.train(mini_batch, train_steps)
        #     # train_steps += 1

        if epoch>20:
        # for train_step in range(conf.train_steps):
            mini_batch = buffer.sample(min(buffer.current_size, conf.batch_size))  # obs； (64, 200, 3, 42)
            # print(mini_batch['o'])
            agents.train(mini_batch, epoch)
            # train_steps += 1

    df_loss = pd.DataFrame(all_pulse)
    # df_loss.to_excel('节拍变化表0414.xlsx')



if __name__ == '__main__':
    train()




