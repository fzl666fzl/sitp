import torch
from torch.distributions import one_hot_categorical
import time
import threading
import math




'''
TODO:
    2022.11.28
    本文件用于基于DQN的强化学习指导装配线生产，在运行此文件的主函数时，通过改变self.action来改变工序排序的规则
    2023.1.6
    或许采样的方式不对，更新的方式不对，不能很好地收敛
    2023.2.20
    单智能体
    2023.11.14
    rollout的改进版本
    
'''

import numpy as np
from parameter import args_parser
import simpy
import pandas

args = args_parser()

pro_num = args.pro_num
team_num = args.team_num
station_num = args.station_num
pulse = 710
# action_set = [[1,0,0],[0.8,0.2,0],[0.8,0,0.2],
#               [0.6,0.2,0.2],[0.5,0.2,0.3],[0.2,0.5,0.3],
#               [0.2,0,0.8],[0.2,0.3,0.5],[0.2,0.8,0]]
action_set = []
for i in np.arange(0.1,1,0.1):
    for j in np.arange(0.1,1,0.1):
        tmp_action = []
        tmp_action.append(i)
        if i + j > 1:
            continue
        else:
            tmp_action.append(j)
            tmp_action.append(1-i-j)
            action_set.append(tmp_action)
# print(len(action_set))##45

orderfreeair = []
orderfinishair = []
orderleftair = []

freeorders = args.freeorders
dict_isfirstprocedure = args.dict_isfirstprocedure
dict_time = args.dict_time
dict_preorder = args.dict_preorder
dict_team = args.dict_team
pro_id = args.pro_id
pro_init = args.pro_init
dict_ready = {}
dict_postorder = args.dict_postorder
dict_postnum = args.dict_postnum
dict_posttime = args.dict_posttime
dict_postsinktime = args.dict_postsinktime
dict_isctrl = args.pro_iscrtl
'''1.3定义三个类
class Team：小组类
class Singelstation：站位类，定义了每个站位的实际工作时间（此数据统计暂时未调通）
class SingelAircraft：飞机类，定义了飞机的id，每道工序的起止时间，每道工序是否完成，每道工序是否是不需要紧前的工序
'''


class Team:
    def __init__(self, id,station_id):
        self.id = id  ####一共有五个组
        self.cap = 1
        self.pro_num = 0  ###已经装配的工序
        self.busy_num = 0  #####被占用的工人数
        self.stfi = [0, 0]
        self.time_past = 0
        self.time_past1 = 0
        self.order_buffer = []
        self.order_finish = []
        self.finishtime = pulse * station_id


class WorkingArea:
    def __init__(self, env, id):
        self.id = id
        self.processlist = []
        self.priority = simpy.Container(env, capacity=1)
        self.into_machine = env.event()


# 创建装配的飞机类
class SingelAircraft:
    def __init__(self, env, aircarft_id):
        # self.mainbody = item
        self.env = env
        self.id = aircarft_id
        single_isfinish = [0 for _ in range(pro_num)]
        self.isfinish = dict(zip(pro_id, single_isfinish))
        self.startingtime = dict(zip(pro_id, single_isfinish))
        self.finishtime = dict(zip(pro_id, single_isfinish))
        self.team_id = dict(zip(pro_id, single_isfinish))
        self.station_id = dict(zip(pro_id, single_isfinish))
        self.isfirstprocedure = dict_isfirstprocedure
        self.orders_free = freeorders
        self.istrigger = []  # 保存了是否被触发了此工序事件，防止重复触发
        self.order_finish = []
        self.order_left = [i + 1 for i in range(pro_num)]
        for i in range(pro_num):
            self.istrigger.append(self.env.event())


class Station:
    def __init__(self, env, id, action):
        self.env = env
        self.id = id
        self.order_buffer = []
        self.order_finish = []
        self.isfirsttongji = 0
        self.pro = []  ##要装配的工序
        self.pro_fin = []  #####已经完成的工序
        self.time_past = 0  ##已经加工的时间
        self.time_remaining = 0  ####剩余的时间
        self.teams = []
        self.pro_rule = 0  #####初始化的工序分配规则为时间越长越优先
        self.store = simpy.Store(env, capacity=1)
        self.action = action

        self.aircraft = None
        self.orderfreeair = []
        self.orderfinishair = []
        self.orderleftair = []
        self.pro_all = 0##已经完成的工序数
        self.pro_isctrl = 0##已完成的有紧后工序数
        for i in range(team_num):
            self.teams.append(Team(i,id))


    def cal_pri1(self, order_free, action):  ###这里的action是向量
        pris = []
        tmp_order = list(order_free)
        for free in tmp_order:
            a = 1
            pri0 = (dict_time[free] - 33) / (303 - 33)
            pri1 = dict_postnum[free] / 4
            pri2 = (dict_posttime[free] - 33) / (1276 - 33)
            pri3 = (dict_postsinktime[free] - 33) / (5006 - 33)
            pri = [pri0, pri1, pri2, pri3]
            pri = np.dot(pri, action)
            pris.append(pri)
        act_idx = pris.index(max(pris))
        sel_act = tmp_order[act_idx]
        return sel_act

    def cal_pri(self, order_free, action):  ###计算自由工序的优先级
        pris = []
        pris = []
        tmp_order = list(order_free)
        for free in tmp_order:
            a = 1
            pri0 = (dict_time[free] - 33) / (303 - 33)
            pri1 = dict_postnum[free] / 4
            pri2 = (dict_posttime[free] - 33) / (1276 - 33)
            pri3 = (dict_postsinktime[free] - 33) / (5006 - 33)
            pri = [pri1, pri2, pri3]

            pri = np.dot(pri, action_set[action])
            pris.append(pri)
        act_idx = pris.index(max(pris))
        sel_act = tmp_order[act_idx]
        return sel_act


    def team_sel(self, team_free):  # 选择工人
        tmpteams = self.teams
        team_idtime = []
        for team in tmpteams:
            if team.id in team_free:
                team_idtime.append((team.id, team.time_past))
        team_idtime = sorted(team_idtime, key=lambda x: x[1])
        return team_idtime[0][0]

    def team_sel1(self, team_free, team_action):  # 选择工人
        team_action = team_action[:2]
        pris = []
        for team in self.teams:
            if team.id in team_free:
                pri0 = team.time_past / 100
                pri1 = (team.id - 1) / 10 + 1
                pri = [pri0, pri1]
                pri = np.dot(pri, team_action)
                pris.append(pri)
        act_idx = pris.index(min(pris))
        sel_act = team_free[act_idx]

        return sel_act

    def distribution(self, air, action):
        self.orderfreeair = air.orders_free[:]
        self.orderfinishair = air.order_finish[:]
        self.orderleftair = air.order_left[:]
        print(f'zhanwei{self.id}得到了飞机在时间{self.env.now}')
        order_stfi = {}
        for i in range(pro_num):
            order_stfi[i + 1] = [air.startingtime[i + 1], air.finishtime[i + 1]]

        orders_finish = air.order_finish[:]
        for j in range(team_num):
            self.teams[j].stfi = [self.id * pulse, self.id * pulse]
            self.teams[j].order_buffer = []
        team_free = [i for i in range(team_num)]
        order_free = air.orders_free[:]
        order_free = set(order_free)
        print(order_free)

        print("当前规则为", action)
        while order_free and team_free:

            # print(order_free)
            thisorder = self.cal_pri(order_free, action[0])

            t = self.team_sel(team_free)
            time_w = (t - 1) / 10 + 1

            ###processing
            tmp_team_stfi = self.teams[t].stfi
            if thisorder in freeorders:  ####初始的自由工序
                order_stfi[thisorder][0] = tmp_team_stfi[1]
                order_stfi[thisorder][1] = order_stfi[thisorder][0] + dict_time[thisorder] * time_w
                tmp_team_stfi[1] = order_stfi[thisorder][1]

                # print(
                #     f'工序：{thisorder}-站位：{self.id}-工种：{t}-开始时间：{order_stfi[thisorder][0]}-结束时间：{order_stfi[thisorder][1]}')

            else:
                tmp_maxtime = 0

                for tmpfather in dict_preorder[thisorder]:  # 遍历所有的父工序
                    # print('父工序为',order_stfi)
                    tmp_maxtime = max(tmp_maxtime, order_stfi[tmpfather][1])
                order_stfi[thisorder][0] = max(tmp_team_stfi[1], tmp_maxtime)
                order_stfi[thisorder][1] = order_stfi[thisorder][0] + dict_time[thisorder] * time_w
                tmp_team_stfi[1] = order_stfi[thisorder][1]
                # print(f'工序：{L[j]}-站位：{i}-工种：{G[j]}-结束时间：{tmp_station_stfi[G[j]][1]}')
                # print(
                # f'工序：{thisorder}-站位：{self.id}-工种：{t}-开始时间：{order_stfi[thisorder][0]}-结束时间：{order_stfi[thisorder][1]}')

            if self.id < station_num - 1:
                if tmp_team_stfi[1] <= pulse * (self.id + 1):
                    self.teams[t].order_buffer.append(thisorder)
                    self.teams[t].time_past += dict_time[thisorder] * time_w
                    self.teams[t].stfi[1] = tmp_team_stfi[1]
                    orders_finish.append(thisorder)
                    for order in dict_postorder[thisorder]:
                        flag = True
                        for preorder in dict_preorder[order]:
                            if preorder not in orders_finish:
                                flag = False
                                break
                        if flag:
                            order_free.add(order)

                    order_free.remove(thisorder)

                else:
                    team_free.remove(t)
                    if not team_free:
                        air.orders_free = list(order_free.copy())
                        self.orderfreeair = list(order_free.copy())
                        break
                    t = self.team_sel(team_free)  ###选择的工人序号tbuduiaaaa
                    order_stfi[thisorder][1] = order_stfi[thisorder][1] - dict_time[thisorder] * time_w


            else:
                self.teams[t].order_buffer.append(thisorder)
                self.teams[t].time_past += dict_time[thisorder] * time_w
                self.teams[t].stfi[1] = tmp_team_stfi[1]
                orders_finish.append(thisorder)

                ###update
                for order in dict_postorder[thisorder]:
                    flag = True
                    for preorder in dict_preorder[order]:
                        if preorder not in orders_finish:
                            flag = False
                            break
                    if flag:
                        order_free.add(order)
                order_free.remove(thisorder)
            # print(order_free)

    def station_decision(self, station_list):
        action = [[0, 1, 0, 0], [1, 0, 0, 0]]
        self.aircraft = yield station_list[self.id].get()
        # self.distribution(self.aircraft,action)
        for i in range(team_num):  # 每一个工人都创建一个进程,st1表示站位1的工序列表
            self.env.process(self.team_process(self.aircraft, i))
        station_maxtime = 0
        for j in range(team_num):
            station_maxtime = max(station_maxtime, self.teams[j].time_past)

        yield self.env.timeout(pulse)
        self.time_past = station_maxtime
        # station_list[self.id + 1].put(self.aircraft)

    def team_process(self, aircraft, team_id):
        env = self.env
        time_w = (team_id - 1) / 10 + 1
        for tmpwhichstation in self.teams[team_id].order_buffer:
            if aircraft.isfirstprocedure[tmpwhichstation] == 1:  # 如果工序没有紧前，直接生产
                if aircraft.isfinish[tmpwhichstation] == 0:
                    aircraft.startingtime[tmpwhichstation] = round(env.now, 2)
                    yield env.timeout(dict_time[tmpwhichstation] * time_w)  # 生成延时时间，在这里停留该工序的装配时间
                    # print(f'实际工序：{tmpwhichstation}-站位：{self.id}-工作组:{team_id}-时间：{self.env.now}')
                    #
                    aircraft.finishtime[tmpwhichstation] = round(env.now, 2)
                    aircraft.team_id[tmpwhichstation] = team_id
                    aircraft.station_id[tmpwhichstation] = self.id
                    aircraft.istrigger[tmpwhichstation - 1].succeed()
                    aircraft.isfinish[tmpwhichstation] = 1
                    aircraft.order_finish.append(tmpwhichstation)
                    aircraft.order_left.remove(tmpwhichstation)
                    self.orderfinishair.append(tmpwhichstation)
                    self.orderleftair.remove(tmpwhichstation)

            else:
                if aircraft.isfinish[tmpwhichstation] == 0:
                    if type(dict_preorder[tmpwhichstation]) == type(1):  # 如果只有一个紧前工序，等待它完成
                        yield aircraft.istrigger[dict_preorder[tmpwhichstation] - 1]
                    else:  # 等待全部的紧前工序完成
                        this_events = [aircraft.istrigger[k - 1] for k in dict_preorder[tmpwhichstation]]
                        yield simpy.events.AllOf(env, this_events)

                    aircraft.startingtime[tmpwhichstation] = round(env.now, 2)
                    yield env.timeout(dict_time[tmpwhichstation] * time_w)
                    # print(f'实际工序：{tmpwhichstation}-站位：{self.id}-工作组:{team_id}-时间：{self.env.now}')
                    aircraft.finishtime[tmpwhichstation] = round(env.now, 2)
                    self.teams[team_id].finishtime = round(env.now, 2)
                    aircraft.istrigger[tmpwhichstation - 1].succeed()
                    aircraft.isfinish[tmpwhichstation] = 1
                    aircraft.order_finish.append(tmpwhichstation)
                    aircraft.team_id[tmpwhichstation] = team_id
                    aircraft.station_id[tmpwhichstation] = self.id
                    aircraft.order_left.remove(tmpwhichstation)
                    self.orderfinishair.append(tmpwhichstation)
                    self.orderleftair.remove(tmpwhichstation)
            self.teams[team_id].time_past1 += dict_time[tmpwhichstation] * time_w

            self.teams[team_id].finishtime = round(env.now, 2) - pulse * self.id
            self.pro_all += 1
            if dict_isctrl[tmpwhichstation]==1:
                self.pro_isctrl += 1

    def station_production(self, station_list):
        env = self.env
        station_id = self.id
        while True:
            aircraft = yield station_list[station_id].get()
            for i in range(team_num):  # 每一个工人都创建一个进程,st1表示站位1的工序列表
                env.process(self.team_process(aircraft, env, i))
            station_maxtime = 0
            for j in range(team_num):
                station_maxtime = max(station_maxtime, self.teams[j].time_past)
            yield env.timeout(pulse - station_maxtime)
            self.time_past = station_maxtime
            station_list[station_id + 1].put(aircraft)





##获取排序 agent的状态
def get_states_station(tmpair, allstation,station_id):
    nowstation = allstation[station_id]
    st_time = [nowstation.teams[i].finishtime for i in range(team_num)]
    max_st_time = max(st_time)  ###当前站位实际加工的时间

    states = []
    ##当前工位剩余时间
    time_remain = pulse - max_st_time
    ##当前自由工序数量
    freeo_num = len(tmpair.orders_free)
    ##当前剩余工序数量
    remaino_num = pro_num - len(tmpair.order_finish)
    ##剩余工序最长序列长度maxdepth###剩余序列平均长度avgdepth
    maxdepth = 0
    avgdepth = 0
    maxdeptime = 0
    avgdeptime = 0
    for idd in tmpair.order_left:
        if dict_postorder[idd] != 0:
            root_depth = dict_postnum[idd] + 1
            deptime = dict_posttime[idd]
            maxdeptime = max(maxdeptime, deptime)
            maxdepth = max(root_depth, maxdepth)
            avgdepth += root_depth
            avgdeptime += deptime
    avgdepth = avgdepth / remaino_num if remaino_num != 0 else 0
    avgdeptime = avgdeptime / remaino_num if remaino_num != 0 else 0
    ##当前自由工序的后续*时间之和deptime
    states.append(nowstation.id/10)
    states.append(freeo_num/10)
    states.append(remaino_num/10)
    states.append(maxdepth/10)
    states.append(avgdepth/10)
    states.append(maxdeptime / 1000)
    states.append(avgdeptime / 1000)
    done = True if remaino_num == 0 else False
    return states, done

#获取分配agent的状态
def get_states_worker(tmpair, allstation,station_id):
    states = []
    nowstation = allstation[station_id]
    st_time = [nowstation.teams[i].finishtime for i in range(team_num)]
    past_time = [nowstation.teams[i].past_time for i in range(team_num)]
    averagetime = sum(st_time) / team_num  ###当前实际站位实际加工的时间
    difftime = max(st_time) - min(st_time) ##实际加工时间最大最小的差异
    averagepast = sum(past_time) / team_num
    diffpast = max(past_time)
    states.append(nowstation.id/10)
    states.append(averagetime/1000)
    states.append(difftime/1000)
    states.append(averagepast/1000)
    states.append(diffpast/1000)
    return states

def get_obs(tmpair, allstation,station_id):
    nowstation = allstation[station_id]
    st_time = [nowstation.teams[i].finishtime for i in range(team_num)]
    max_st_time = max(st_time)  ###当前站位实际加工的时间
    obs = []
    states = []
    ##当前工位剩余时间
    time_remain = pulse - max_st_time
    ##当前自由工序数量
    freeo_num = len(tmpair.orders_free)
    ##当前剩余工序数量
    remaino_num = pro_num - len(tmpair.order_finish)
    ##剩余工序最长序列长度maxdepth###剩余序列平均长度avgdepth
    maxdepth = 0
    avgdepth = 0
    maxdeptime = 0
    avgdeptime = 0
    for idd in tmpair.order_left:
        if dict_postorder[idd] != 0:
            root_depth = dict_postnum[idd] + 1
            deptime = dict_posttime[idd]
            maxdeptime = max(maxdeptime, deptime)
            maxdepth = max(root_depth, maxdepth)
            avgdepth += root_depth
            avgdeptime += deptime
    avgdepth = avgdepth / remaino_num if remaino_num != 0 else 0
    avgdeptime = avgdeptime / remaino_num if remaino_num != 0 else 0
    ##当前自由工序的后续*时间之和deptime
    states.append(nowstation.id / 10)
    states.append(freeo_num / 10)
    states.append(remaino_num / 10)
    states.append(maxdepth / 10)
    states.append(avgdepth / 10)
    states.append(maxdeptime / 1000)
    states.append(avgdeptime / 1000)
    done = True if remaino_num == 0 else False
    obs.append(states)
    # sss = states[:]
    states = []
    st_time = [nowstation.teams[i].finishtime for i in range(team_num)]
    past_time = [nowstation.teams[i].time_past for i in range(team_num)]
    averagetime = sum(st_time) / team_num  ###当前实际站位实际加工的时间
    difftime = max(st_time) - min(st_time) ##实际加工时间最大最小的差异
    averagepast = sum(past_time) / team_num
    diffpast = max(past_time)
    states.append(nowstation.id/10)
    states.append(averagetime/1000)
    states.append(difftime/1000)
    states.append(averagepast/1000)
    states.append(diffpast/1000)
    states = states + [0,0]
    obs.append(states)
    states_n = obs[0] + obs[1]
    return obs, states_n,done


def get_reward(now_time, allstation, air,station_id):


    nowstation = allstation[station_id]
    st_time = [nowstation.teams[i].finishtime for i in range(team_num)]

    max_st_time = max(st_time)
    avg_st_time = sum(st_time) / team_num
    si = 0
    for i in range(team_num):
        si += (nowstation.teams[i].finishtime / 100 - avg_st_time /100) ** 2

    rew = max_st_time / 100





    return -rew/10  ###奖励越小越好要取负



def reset_env(env):
    station_list = []  # 保存了所有的站位，每一个站位都是一个store用来存飞机实例，容量均为1，
    # #此处要小心，如果在所有站位的总体建模里面要注意,store的get和put机制会覆盖前一个对象，所以必须对站位的状态加以限制，
    # 否则出现阻塞时会出现某站位没有加工完就被覆盖掉的情况
    station_init = simpy.Store(env, capacity=10)  # 产生飞机的站位命名为station_init
    station_list.append(station_init)
    station_alltime = []
    for i in range(station_num):
        tmpstation = simpy.Store(env, capacity=1)
        station_list.append(tmpstation)
        station_alltime.append([0 for k in range(team_num)])

    allstation = []
    isfirsttongji = []

    for k in range(station_num):
        allstation.append(Station(env, k, 1))
        isfirsttongji.append([0 for i in range(team_num)])

    tmp_aircarft = SingelAircraft(env, 0)
    station_init.put(tmp_aircarft)
    return allstation, station_list, tmp_aircarft


def reset_station(env, allstation, station_list):
    for i in range(station_num):
        env.process(allstation[i].station_decision(station_list))  # 实际的站位处理函数




'''3.产生飞机的函数'''


def generate_item(env,
                  last_q: simpy.Store,
                  item_num: int,
                  all_aircraft
                  ):
    for i in range(item_num):
        print(f'{round(env.now, 2)} - item: item_{i} - created')
        tmp_aircarft = SingelAircraft(env, i)
        last_q.put(tmp_aircarft)
        tmp = tmp_aircarft
        all_aircraft.append(tmp)
        # t = random.expovariate(1 / MEAN_TIME)
        t = 10 * pulse
        yield env.timeout(round(t, 1))


def get_pulse(allstation):
    pulse_real = 0
    for i in range(station_num):
        time_tmp_station = 0
        for j in range(team_num):
            # if i == station_num-2:
            #     allstation[i].teams[j].finishtime += allstation[i+1].teams[j].finishtime
            time_tmp_station = max(time_tmp_station, allstation[i].teams[j].finishtime)
            # print("真正统计时的finishtime", allstation[i].teams[j].finishtime)
        pulse_real = max(pulse_real, time_tmp_station)

    return pulse_real


n_agents = 2

n_actions = 45





class RolloutWorker:
    def __init__(self, agents, conf,pulses):
        super().__init__()
        self.conf = conf
        self.agents = agents
        self.episode_limit = conf.episode_limit
        self.n_actions = conf.n_actions
        self.n_agents = conf.n_agents
        self.state_shape = conf.state_shape
        self.obs_shape = conf.obs_shape

        self.start_epsilon = conf.start_epsilon
        self.anneal_epsilon = conf.anneal_epsilon
        self.end_epsilon = conf.end_epsilon
        self.pulses = pulses
        print('Rollout Worker inited!')


def generate_episode(agents,conf,pulses,episode_num, thispulse):#evaluate=False
    env = simpy.Environment()
    ##TODO:
    #   o: 每个agent的状态的合集
    #   u：所有agent的动作合集
    #   s：全局状态
    #   o_:下一个时刻每个agent的状态的合集
    #   s_:下一个时刻的全局状态
    #   au: 可获得的动作（实际上就是action_set）
    #   u_onehot: 动作的独热编码
    oo, u, r, ss, o_,s_ = [], [], [], [], [], []
    au, avail_u_, u_onehot, terminate, padded = [], [], [], [], []
    n_orders = 10##每次分配的工序数

    ## 初始化站位和状态
    allstation, station_list, air = reset_env(env)
    #声明站位装配函数（env.process）
    reset_station(env, allstation, station_list)
    episode = {}

    last_action = np.zeros((n_agents, n_actions))
    epsilon = 0 if episode_num > int(conf.n_epochs * 0.95) else conf.start_epsilon
    actions, avail_actions, actions_onehot = [], [], []
    rules = []

    # print("当前的obs为", obs)
    # print("当前的state为", state)
    n_finish = 0 ##完成的工序数
    n_recur = 0 ##分配的步数，要在10步之内分配完
    freeo = freeorders[:]
    nonfreeo = []
    for o in pro_id:
        if dict_preorder[o]:
            nonfreeo.append(o)
    tmp_orders_finish = []
    rule1 = [0.2,0.3,0.5]
    rule2 = [0.8,0.1,0.1]
    station_id = 0
    freeworkers = [0,1,2]
    nextstaton_o = [] ##这个站位已经无法装配，保存到下一个站位生产的工序集
    unfinished_num = pro_num
    agents.policy.init_hidden(1)
    while n_finish < pro_num:
        station_id = station_id if station_id < station_num else station_num
        # print("初始的freeo是",freeo)
        recurrent_num = min(n_orders,unfinished_num)
        obs, states, _ = get_obs(air, allstation, station_id)
        # print("当前的obs是",obs)
        rules = []
        actions = []
        actions_onehot = []
        for agent_id in range(n_agents):
            avail_action = action_set[:]
            action = agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action,
                                          epsilon, evaluate=False)
            rules.append(action)
            # 生成动作的onehot编码
            action_onehot = np.zeros(n_actions)
            action_onehot[action] = 1
            actions.append(action)
            actions_onehot.append(action_onehot)
            avail_actions.append(avail_action)
            last_action[agent_id] = action_onehot

        if n_recur < 6:##要在10步之内分配完
        #*************每一步安排recurrent_num个工序***********#
            n_recur += 1
            for i in range(recurrent_num):
                if (not freeo) and not nextstaton_o:
                    break
                if (not freeo) and nextstaton_o:
                    station_id += 1
                    freeo = nextstaton_o[:]
                    nextstaton_o = []
                station_id = station_id if station_id < station_num else station_num - 1
                thisstation = allstation[station_id]

                #*************对所有的工序重排序***********#
                pris = []
                for o in freeo:
                    pri0 = (dict_time[o] - 0.2) / (51.9 - 0.2)
                    pri1 = dict_postnum[o] / 11
                    pri2 = (dict_posttime[o] - 0.2) / (160.9 - 0.2)
                    pri3 = (dict_postsinktime[o] - 0.2) / (79.85 - 0.2)
                    pri4 = (51.9 - dict_time[o]) / (51.9 - 0.2)
                    pri = [pri1, pri2, pri3]
                    pri = np.dot(pri, action_set[rules[0]])  ##按照给定的规则计算
                    pris.append(pri)

                act_idx = pris.index(max(pris))
                new_o = freeo[act_idx]

                # ******************为新的工序挑选工人******************#

                pris = []
                for team in thisstation.teams:
                    pri0 = team.time_past / 100
                    pri1 = (team.id - 1) / 10 + 1
                    pri2 = 0.2
                    pri = [pri0, pri0,pri0]
                    pri = np.dot(pri, action_set[rules[1]])
                    pris.append(pri)
                act_idx = pris.index(min(pris))
                new_w = freeworkers[act_idx]


                latestfi = thisstation.teams[new_w].finishtime
                thisworker = thisstation.teams[new_w]
                order_stfi = [0, 0]

                if new_o in freeorders:  ####初始的自由工序
                    order_stfi[0] = latestfi
                    order_stfi[1] = order_stfi[0] + dict_time[new_o] * ((thisworker.id - 1) / 10 + 1)

                else:
                    tmp_maxtime = 0
                    for tmpfather in dict_preorder[new_o]:  # 遍历所有的父工序
                        tmp_maxtime = max(tmp_maxtime, air.finishtime[tmpfather])
                    order_stfi[0] = max(latestfi, tmp_maxtime)
                    order_stfi[1] = order_stfi[0] + dict_time[new_o] * ((thisworker.id - 1) / 10 + 1)
                    # order_stfi[1] = order_stfi[0] + dict_time[new_o]

                # *************若没有超过节拍，从freeo中删除该工序，若超过了就放到nextstaton_o中**********#
                freeo.remove(new_o)
                if order_stfi[1] > thispulse * (station_id + 1) and station_id < station_num - 1 :
                    nextstaton_o.append(new_o)
                    continue

                else:
                    thisworker.time_past += dict_time[new_o]
                    thisworker.finishtime = order_stfi[1]
                    air.finishtime[new_o] = order_stfi[1]
                    air.startingtime[new_o] = order_stfi[0]
                    air.isfinish[new_o] = 1
                    tmp_orders_finish.append(new_o)
                    n_finish += 1
                # *************从nonfree更新freeo**********#
                for n in nonfreeo:
                    ffl = True
                    for pre in dict_preorder[n]:
                        if pre not in tmp_orders_finish:
                            ffl = False
                    if ffl:
                        freeo.append(n)
                        nonfreeo.remove(n)
                # print(freeo)
        else:
            while n_finish < pro_num:
                if (not freeo) and not nextstaton_o:
                    break
                if (not freeo) and nextstaton_o:
                    station_id += 1
                    freeo = nextstaton_o[:]
                    nextstaton_o = []
                station_id = station_id if station_id < station_num else station_num - 1
                thisstation = allstation[station_id]

                # *************对所有的工序重排序***********#
                pris = []
                for o in freeo:
                    pri0 = (dict_time[o] - 0.2) / (51.9 - 0.2)
                    pri1 = dict_postnum[o] / 11
                    pri2 = (dict_posttime[o] - 0.2) / (160.9 - 0.2)
                    pri3 = (dict_postsinktime[o] - 0.2) / (79.85 - 0.2)
                    pri4 = (51.9 - dict_time[o]) / (51.9 - 0.2)
                    pri = [pri1, pri2, pri3]
                    pri = np.dot(pri, action_set[rules[0]])  ##按照给定的规则计算
                    pris.append(pri)

                act_idx = pris.index(max(pris))
                new_o = freeo[act_idx]

                # ******************为新的工序挑选工人******************#

                pris = []
                for team in thisstation.teams:
                    pri0 = team.time_past / 100
                    pri1 = (team.id - 1) / 10 + 1
                    pri2 = 0.2
                    pri = [pri0, pri0, pri0]
                    pri = np.dot(pri, action_set[rules[1]])
                    pris.append(pri)
                act_idx = pris.index(min(pris))
                new_w = freeworkers[act_idx]

                latestfi = thisstation.teams[new_w].finishtime
                thisworker = thisstation.teams[new_w]
                order_stfi = [0, 0]

                if new_o in freeorders:  ####初始的自由工序
                    order_stfi[0] = latestfi
                    order_stfi[1] = order_stfi[0] + dict_time[new_o] * ((thisworker.id - 1) / 10 + 1)

                else:
                    tmp_maxtime = 0
                    for tmpfather in dict_preorder[new_o]:  # 遍历所有的父工序
                        tmp_maxtime = max(tmp_maxtime, air.finishtime[tmpfather])
                    order_stfi[0] = max(latestfi, tmp_maxtime)
                    order_stfi[1] = order_stfi[0] + dict_time[new_o] * ((thisworker.id - 1) / 10 + 1)
                    # order_stfi[1] = order_stfi[0] + dict_time[new_o]

                # *************若没有超过节拍，从freeo中删除该工序，若超过了就放到nextstaton_o中**********#
                freeo.remove(new_o)
                if order_stfi[1] > thispulse * (station_id + 1) and station_id < station_num - 1:
                    nextstaton_o.append(new_o)
                    continue

                else:
                    thisworker.time_past += dict_time[new_o]
                    thisworker.finishtime = order_stfi[1]
                    air.finishtime[new_o] = order_stfi[1]
                    air.startingtime[new_o] = order_stfi[0]
                    air.isfinish[new_o] = 1
                    tmp_orders_finish.append(new_o)
                    n_finish += 1
                # *************从nonfree更新freeo**********#
                for n in nonfreeo:
                    ffl = True
                    for pre in dict_preorder[n]:
                        if pre not in tmp_orders_finish:
                            ffl = False
                    if ffl:
                        freeo.append(n)
                        nonfreeo.remove(n)
                # print(freeo)

        obs_, states_,done = get_obs(air, allstation, station_id)
        reward = get_reward(env.now, allstation, air,station_id)
        print("actions: ", actions)


        oo.append(obs)
        ss.append(states)
        u.append(np.reshape(actions, [n_agents, 1]))
        u_onehot.append(actions_onehot)

        # r.append([reward])
        r.append([reward])
        o_.append(obs_)
        s_.append(states_)
        au.append(avail_actions)
        terminate.append([done])
        padded.append([0.])



    ##统计节拍
    now_pulse = 0
    for s in range(station_num):
        thiss = allstation[s]
        for w in range(team_num):
            now_pulse = max(now_pulse,thiss.teams[w].finishtime - s * thispulse)

    for rri in range(len(r)):
        if now_pulse > 750:
            rr = -1
        elif now_pulse > 700 and now_pulse <= 750:
            rr = -0.5
        elif now_pulse > 650 and now_pulse <=700:
            rr = 1
        else:
            rr = 3
        r[rri] = [rr]
    # df_start = pandas.DataFrame(air.startingtime, index=[0]).T
    # df_finish = pandas.DataFrame(air.finishtime, index=[0]).T
    # df_start.to_excel('工序时刻表1.xlsx')
    # df_finish.to_excel('工序结束时刻表1.xlsx')
    print("最终的节拍是",now_pulse)
    pulses.append(now_pulse)
    print(np.array(u_onehot).shape) ##7,2,45
    episode['o'] = oo.copy()
    episode['s'] = ss.copy()
    episode['u'] = u.copy()
    episode['r'] = r.copy()
    episode['o_'] = o_.copy()
    episode['s_'] = s_.copy()
    episode['avail_u'] = action_set.copy()
    episode['avail_u_'] = action_set.copy()
    episode['u_onehot'] = u_onehot.copy()
    episode['padded'] = padded.copy()
    episode['terminated'] = terminate.copy()
    return episode
if __name__ == '__main__':
    win_rates = []
    episode_rewards = []
    train_steps = 0
    pulses = []
    from config import Config
    conf = Config()

    generate_episode(conf,pulses,1, evaluate=False)
