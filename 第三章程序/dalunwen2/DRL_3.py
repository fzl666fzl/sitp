# -*- coding: utf-8 -*-
'''
TODO:2022/12/07暂时作为主函数
'''

import random

import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam


import simpy





EPISODES = 10


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200)  # 记忆体使用队列实现，队列满后根据插入顺序自动删除老数据
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.4  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.05
        self.bound_cnt = 50*0.9
        self.model = self._build_model()
        # 可视化MLP结构
        # plot_model(self.model, to_file='dqn-cartpole-v0-mlp.png', show_shapes=False)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model

        model = Sequential()  # 顺序模型，搭建神经网络（多层感知机）
        # model.add(Dense(43, input_dim=self.state_size))
        # model.add(LeakyReLU(alpha=0.05))
        # model.add(Dense(35))
        # model.add(LeakyReLU(alpha=0.05))
        # model.add(Dense(24, activation='relu'))
        model.add(Dense(16, input_shape=[1,self.state_size],activation='relu'))
        # model.add(Dense(12, activation='relu'))
        model.add(Dense(8))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))  # 指定损失函数以及优化器
        return model

    # 在记忆体（经验回放池）中保存具体某一时刻的当前状态信息
    def remember(self, state, action, reward, next_state, done):
        # 当前状态、动作、奖励、下一个状态、是否结束
        self.memory.append((state, action, reward, next_state, done))

    # 根据模型预测结果返回动作
    def act(self, state,cnt):
        if cnt<=self.bound_cnt and np.random.rand() <= self.epsilon:  # 如果随机数（0-1之间）小于epsilon，则随机返回一个动作
            return random.randrange(self.action_size)  # 随机返回动作0或1
        act_values = self.model.predict(state)  # eg:[[0.35821578 0.11153378]]
        print("model.predict act_values:",act_values)
        return np.argmax(act_values[0])  # returns action 返回价值最大的

    # 记忆回放，训练神经网络模型
    def replay(self, batch_size,losses):
        minibatch = random.sample(self.memory, batch_size)
        print("开始训练")
        loss_value = []
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:  # 没有结束
                # print(next_state)
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            state = np.array(state)

            state = state.reshape(1,self.state_size)
            states.append(state)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            targets.append(target_f)

        states = np.array(states)
        targets = np.array(targets)


        self.model.fit(states, targets, epochs=10, verbose=0,batch_size=32)  # 训练神经网络

        b = abs(float(self.model.history.history['loss'][0]))
        print(b)
        losses.append(b)
        # print("************训练损失***********")
        # print(loss_value)

    # 加载模型权重文件
    def load(self, name):
        self.model.load_weights(name)

    # 保存模型 （参数：filepath）
    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":

    state_size = 9
    action_size = 5


    agent = DQNAgent(state_size, action_size)

    done = False
    batch_size = 32
    avg = 0

    for e in range(EPISODES):  # 循环学习次数，每次学习都需要初始化环境

        total_reward, done = 0, False
        env = simpy.Environment()
        allstation, station_list, air = DRL_production.reset_env(env)
        DRL_production.reset_station(env, allstation, station_list)
        while not done:
            now_state,_ = DRL_production.get_states(env,air,allstation)
            now_state = np.array(now_state)
            now_state = now_state.astype(np.float32)
            now_state = now_state.reshape(1, state_size)
            action = agent.act(now_state)
            # action = 1
            now_state, next_state, reward, done = DRL_production.step(env, action, air, allstation)
            print(now_state)
            now_state = np.array(now_state)
            now_state = now_state.astype(np.float32)
            now_state = now_state.reshape(1, state_size)

            next_state = next_state.astype(np.float32)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(now_state, action, reward, next_state, done)  # 放入记忆体
            # print(now_state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                # print("episode: {}/{}, score（time）: {}".format(e, EPISODES))

                break
        # 定期检查记忆大小，进行记忆回放
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)


    print("Avg score:{}".format(avg / 1000))
