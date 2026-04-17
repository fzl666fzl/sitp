'''
TODO:
    2023.11.30
    dalunwen在用的好程序！
'''
import simpy
import numpy as np
from parameter import args_parser
import pandas as pd
from agent import Agents
import random
from collections import deque
from utils import ReplayBuffer
# import rollout1130
import airrollout0113
import simpy


from config import Config
conf = Config()


args = args_parser()

pro_num = args.pro_num
team_num = args.team_num
station_num = args.station_num
init_pulse = 280
# action_set = [[1,0,0],[0.8,0.2,0],[0.8,0,0.2],
#               [0.6,0.2,0.2],[0.5,0.2,0.3],[0.2,0.5,0.3],
#               [0.2,0,0.8],[0.2,0.3,0.5],[0.2,0.8,0]]

def train():

    agents = Agents(conf)


    buffer = ReplayBuffer(conf)

    # save plt and pk

    win_rates = []
    episode_rewards = []
    train_steps = 0
    pulses = []
    all_pulse = []
    SI = []
    for epoch in range(conf.n_epochs):
        # print("train epoch: %d" % epoch)
        episodes = []

        if not pulses:
            now_pulse = init_pulse
        else:
            now_pulse = min(int(min(pulses)-1),now_pulse)


        pulses = []
        for episode_idx in range(conf.n_eposodes):##n_eposodes=1
            episode, times, pp = airrollout0113.generate_episode(agents, conf,pulses,now_pulse,epoch,SI)
            episodes.append(episode)

            print("当前的节拍为",now_pulse)
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
    # df_loss.to_excel('节拍变化表1130.xlsx')
    print(SI)



if __name__ == '__main__':
    train()




