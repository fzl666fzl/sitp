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

import random
from collections import deque

import simpy
import DDPGnet
import DRL_3
import tensorflow_probability as tfp
import tensorflow as tf
import math

args = args_parser()

pro_num = args.pro_num
team_num = args.team_num
station_num = args.station_num
pulse = 660
action_set = [[1,0,0],[0.8,0.2,0],[0.8,0,0.2],
              [0.6,0.2,0.2],[0.5,0.2,0.3],[0.2,0.5,0.3],
              [0.2,0,0.8],[0.2,0.3,0.5],[0.2,0.8,0]]

orderfreeair = []
orderfinishair = []
orderleftair = []

freeorders = [1, 2, 3, 4, 5, 6, 7]
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

'''1.3定义三个类
class Team：小组类
class Singelstation：站位类，定义了每个站位的实际工作时间（此数据统计暂时未调通）
class SingelAircraft：飞机类，定义了飞机的id，每道工序的起止时间，每道工序是否完成，每道工序是否是不需要紧前的工序
'''


class Team:
    def __init__(self, id):
        self.id = id  ####一共有五个组
        self.cap = 1  ###每个组有三个人
        self.pro_num = 0  ###已经装配的工序
        self.busy_num = 0  #####被占用的工人数
        self.stfi = [0, 0]
        self.time_past = 0
        self.order_buffer = []
        self.order_finish = []
        self.finishtime = 0


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
        for i in range(team_num):
            self.teams.append(Team(i))

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
            thisorder = self.cal_pri(order_free, action)

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
            self.teams[team_id].time_past += dict_time[tmpwhichstation] * time_w
            self.teams[team_id].finishtime = round(env.now, 2) - pulse * self.id

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

    def get_states(self, env, tmpair):

        states = []
        ##当前工位剩余时间
        time_remain = (self.id + 1) * pulse - env.now
        ##当前时间
        time_now = env.now
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
        states.append(self.id)
        states.append(time_now)
        states.append(time_remain)
        states.append(freeo_num)
        states.append(remaino_num)
        states.append(maxdepth)
        states.append(avgdepth)
        states.append(maxdeptime)
        states.append(avgdeptime)

        done = True if remaino_num == 0 else False
        return states, done

    def get_reward(self):
        ###最大生产时间最小
        st_time = [self.time_past for i in range(station_num)]
        non_zero = 0
        for i in range(station_num):
            if self.time_past != 0:
                non_zero += 1
        max_st_time = max(st_time)
        avg_st_time = sum(st_time) / non_zero if non_zero != 0 else sum(st_time)
        si = 0
        for i in range(station_num):
            if self.time_past != 0:
                si += (self.time_past - avg_st_time) ** 2
        si = si / non_zero if non_zero != 0 else 0
        rew = max_st_time / 100 + si / 1000
        return rew

    def step(self, env, action, tmpair, station_id):
        states, done = self.get_states(env, tmpair)
        print("是否完成", done)
        all_action = action  # 生产线获得决策
        before_time = env.now

        while True:
            if done:
                break
            try:
                env.step()
            except Exception as e:  # EmptySchedule,如果执行结束，则true，且reward = 0

                print('minifab finished')
                break

            time_change = env.now - before_time
            if time_change >= 400:
                break

        new_states, done = self.get_states(env, tmpair)
        print(new_states)
        new_states = np.float32(new_states)
        rew = self.get_reward()

        return states, new_states, rew, done


# TODO:因为权限的问题，将step函数放在外面更容易理解，但是站位类内部的那个方法也可以使用

###因为以下为主函数中采用的获取状态的函数
def get_states(env, tmpair, allstation):
    now_time = env.now
    if now_time >= 0 and now_time < pulse:
        station_id = 0
    elif now_time >= pulse and now_time < 2 * pulse:
        station_id = 1
    elif now_time >= 2 * pulse and now_time < 3 * pulse:
        station_id = 2
    else:
        station_id = 3

    nowstation = allstation[station_id]
    st_time = [nowstation.teams[i].finishtime for i in range(team_num)]
    max_st_time = max(st_time)  ###当前站位实际加工的时间
    state_n = []
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
    states.append(nowstation.id)
    states.append(now_time / 100)
    # states.append(max_st_time / 100)
    # states.append(time_remain / 100)
    states.append(freeo_num)
    states.append(remaino_num)
    states.append(maxdepth)
    states.append(avgdepth)
    states.append(maxdeptime / 100)
    states.append(avgdeptime / 100)
    done = True if remaino_num == 0 else False
    return states, done


def get_reward(now_time, allstation, air):
    ###最大生产时间最小
    if now_time > 0 and now_time <= pulse + 1:
        station_id = 0
    elif now_time > pulse + 1 and now_time <= 2 * pulse + 1:
        station_id = 1
    elif now_time > 2 * pulse + 1 and now_time <= 3 * pulse + 1:
        station_id = 2
    else:
        station_id = 3

    nowstation = allstation[station_id]
    # if station_id ==3:
    #     return -get_pulse(allstation)/100

    st_time = [nowstation.teams[i].finishtime for i in range(team_num)]
    # st_time = [nowstation.time_past for i in range(station_num)]
    max_st_time = max(st_time)
    avg_st_time = sum(st_time) / team_num
    si = 0
    for i in range(team_num):
        si += (nowstation.teams[i].finishtime - avg_st_time) ** 2
    # si = math.sqrt(si/team_num)
    # rew = max_st_time / 100 + si / 100
    rew = max_st_time / 100
    remaino = air.order_left[:]
    time_r = 0
    for o in remaino:
        time_r += dict_time[o]
    return -rew  ###奖励越小越好


def step(env, action, tmpair, allstation):
    now_time = env.now
    if now_time >= 0 and now_time < pulse:
        station_id = 0
    elif now_time >= pulse and now_time < 2 * pulse:
        station_id = 1
    elif now_time >= 2 * pulse and now_time < 3 * pulse:
        station_id = 2
    else:
        station_id = 3
    print('此时的站位为：', station_id)
    nowstation = allstation[station_id]
    nowstation.action = action
    before_time = env.now
    states, done = get_states(env, tmpair, allstation)
    while True:
        if done:
            break
        try:
            env.step()
        except Exception as e:  # EmptySchedule,如果执行结束，则true，且reward = 0

            print(e)
            print('minifab finished')
            break

        time_change = env.now - before_time
        if time_change >= 600:
            break

    new_states, done = get_states(env, tmpair, allstation)
    print(new_states)
    new_states = np.float32(new_states)
    rew = get_reward(env.now, allstation)

    return states, new_states, rew, done


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


##实例化站位，保存的统计数据是是否是第一次统计，为了防止后面的飞机进入站位更新此数据
# allstation = []
# isfirsttongji = []
# env = simpy.Environment()
# for k in range(station_num):
#     allstation.append(Station(env,k))
#     isfirsttongji.append([0 for i in range(team_num)])


# def aircraft_process(station_id,station_list):
#     allstation[station_id].station_decision()
#     allstation[station_id].station_production(station_list, station_id)


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
            print("真正统计时的finishtime", allstation[i].teams[j].finishtime)
        pulse_real = max(pulse_real, time_tmp_station)

    return pulse_real


SEED = 65535
ACTION_SPAN = 0.5
obs_dim = 8
action_dim = 9
agent_number = 1
init_pulse = 700
actor_learning_rate = 1e-2
critic_learning_rate = 1e-2
batch_size =32
train_start = 20


def pretrain(env, agent, air, allstation, station_list):
    total_reward, done = 0, False
    bg_noise = np.zeros(action_dim)
    init_action = True

    tmp_buffer = []
    now_state, done = get_states(env, air, allstation)
    while not done:
        print("***********")

        if init_action:
            action = [0, 1, 0, 0]
            init_action = False
        else:
            action = np.random.rand(1, 4)
            action = action / action.sum(axis=1, keepdims=1)
            action = action[0]
            print("选择的动作是：", action)

        # action = np.random.rand(1, 4)[0]
        print(action)

        now_time = env.now
        if now_time >= 0 and now_time < pulse:
            station_id = 0
        elif now_time >= pulse and now_time < 2 * pulse:
            station_id = 1
        elif now_time >= 2 * pulse and now_time < 3 * pulse:
            station_id = 2
        else:
            station_id = 3
        print('此时的站位为：', station_id)
        nowstation = allstation[station_id]
        nowstation.action = action[:]
        print("当前动作为：", nowstation.action)
        nowstation.distribution(air, action)
        if nowstation.id < 3:
            yield env.timeout(pulse)
        else:
            yield env.timeout(pulse + 1000)

        next_state, done = get_states(env, air, allstation)
        reward = get_reward(env.now, allstation)

        now_state = np.array(now_state)
        next_state = np.array(next_state)
        reward = np.array(reward)
        done = np.array(done)
        tmp_buffer.append([now_state, action, reward, next_state, done])
        # print("当前状态是：",now_state)
        # print("下一个时刻状态是",next_state)

        station_list[station_id + 1].put(air)

        if nowstation.id == 3:
            this_pulse = get_pulse(allstation)
            print("当前节拍是：", this_pulse)
            for tmp in tmp_buffer:
                tmp[-3] = 1 / (this_pulse / 100)
                agent.buffer.put(tmp)
            if agent.buffer.size() >= batch_size and agent.buffer.size() >= train_start:
                agent.replay()
            yield env.timeout(pulse + 2000)


def train(env, agent, air, allstation, station_list, losses,cnt,pulses):
    total_reward, done = 0, False
    bg_noise = np.zeros(action_dim)
    init_action = True

    tmp_buffer = []
    # now_state, done = get_states(env, air, allstation)
    # print(now_state)
    while not done:
        print("***********")

        # if init_action:
        #     action = 1
        #     init_action = False
        # else:
        now_state, done = get_states(env, air, allstation)
        now_state = np.array(now_state)
        now_state = now_state.reshape(1, obs_dim)
        action = agent.act(now_state,cnt)
        # u = tfp.distributions.Normal(a, 0.1)
        # action = tf.squeeze(u.sample(1), axis=0)
        # action = tf.clip_by_value(action, clip_value_min=0, clip_value_max=1)
        # action = action.numpy()
        print("选择的动作是：", action)

        # action = np.random.rand(1, 4)[0]

        now_time = env.now
        if now_time >= 0 and now_time < pulse:
            station_id = 0
        elif now_time >= pulse and now_time < 2 * pulse:
            station_id = 1
        elif now_time >= 2 * pulse and now_time < 3 * pulse:
            station_id = 2
        else:
            station_id = 3
        print('此时的站位为：', station_id)
        nowstation = allstation[station_id]
        nowstation.action = action
        print("当前动作为：", nowstation.action)
        nowstation.distribution(air, action)
        if nowstation.id < 3:
            yield env.timeout(pulse)
        else:
            yield env.timeout(pulse + 1000)

        next_state, done = get_states(env, air, allstation)
        reward = get_reward(env.now, allstation, air)

        next_state = np.array(next_state)
        next_state = next_state.reshape(1, obs_dim)
        reward = np.array(reward)
        done = np.array(done)
        tmp_buffer.append([now_state, action, reward, next_state, done])

        # agent.buffer.put([now_state, action, reward, next_state, done])
        # if agent.buffer.size() >= batch_size and agent.buffer.size() >= train_start:
        #     agent.replay()
        # print("当前状态是：",now_state)
        # print("下一个时刻状态是",next_state)

        station_list[station_id + 1].put(air)


        # print(now_state, action, reward, next_state, done)





        if nowstation.id == 3:
            this_pulse = get_pulse(allstation)
            pulses.append(this_pulse)
            print("当前节拍是：", this_pulse)
            for i in range(station_num):
                tmp = tmp_buffer[i]
                if (i == 0):
                    tmp[-3] = 0.5 * tmp[-3] + 0.3 * tmp_buffer[1][-3] + 0.1 * tmp_buffer[2][-3] + 0.1 * tmp_buffer[3][-3]
                elif (i == 1):
                    tmp[-3] = 0.5 * tmp[-3] + 0.3 * tmp_buffer[2][-3] + 0.2 * tmp_buffer[3][-3]
                elif (i == 2):
                    tmp[-3] = 0.6 * tmp[-3] + 0.4 * tmp_buffer[3][-3]

                ###训练效果比较好

                # tmp[-3] = -this_pulse/100


                # if (i == 0):
                #     tmp[-3] = 0.5 * tmp[-3] - 0.5 * this_pulse/100
                # elif (i == 1):
                #     tmp[-3] = 0.6 * tmp[-3] - 0.4 * this_pulse/100
                # elif (i == 2):
                #     tmp[-3] = 0.8 * tmp[-3] - 0.2 * this_pulse/100



                # agent.buffer.put(tmp)
                # agent.remember(now_state, action, reward, next_state, done)  # 放入记忆体

                agent.remember(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4])
            # if agent.buffer.size() >= batch_size:
            #     agent.replay(losses)

            if len(agent.memory) > batch_size:
                agent.replay(batch_size,losses)
            yield env.timeout(pulse + 2000)





def main():
    # policy初始化,所有的网络初始化
    maddpg_agents = DRL_3.DQNAgent(obs_dim, action_dim)

    ###暖机10次

    # for i in range(10):
    #     tmp_buffer = []
    #     env = simpy.Environment()
    #
    #     allstation, station_list, air = reset_env(env)
    #     reset_station(env, allstation, station_list)
    #     env.process(pretrain(env,maddpg_agents,air,allstation,station_list))
    #     env.run(5000)

    losses = []
    pulses = []
    for i_episode in range(100):
        tmp_buffer = []
        env = simpy.Environment()

        allstation, station_list, air = reset_env(env)
        reset_station(env, allstation, station_list)
        env.process(train(env, maddpg_agents, air, allstation, station_list, losses,i_episode,pulses))
        env.run(5000)

    df_loss = pd.DataFrame(losses)
    df_loss.to_excel('/home/wyl/nn_test/损失函数变化表0408.xlsx')

    df_loss = pd.DataFrame(pulses)
    df_loss.to_excel('/home/wyl/nn_test/节拍变化表0408.xlsx')


if __name__ == '__main__':
    main()

    # env = simpy.Environment()
    # allstation, station_list, air = reset_env(env)
    # reset_station(env, allstation, station_list)
    # env.run(5000)
    #
    # pulse_real = 0
    # time_station = []
    # time_team = []
    # time_tmp_team = []
    # for i in range(station_num):
    #     time_station.append(allstation[i].time_past)
    #     time_tmp_station = 0
    #     for j in range(team_num):
    #         time_tmp_team.append(allstation[i].teams[j].time_past)
    #         time_tmp_station = max(time_tmp_station,allstation[i].teams[j].finishtime-pulse*i)
    #     pulse_real = max(pulse_real, time_tmp_station)
    # pulse_real = get_pulse(allstation)
    # si = np.std(time_station)
    # sii = np.std(time_tmp_team)
    #
    #
    #
    # ##判断工序的开始时间是否早于他的紧前工序结束时间，若早于则返回False
    # flag = True
    # for i in pro_id:
    #     if i in freeorders == 1:
    #         break
    #     tmp_jinqian = dict_preorder[i]
    #     if not air.isfinish[i]:
    #         flag = False
    #         print("未完成的工序是",i)
    #
    #     for j in tmp_jinqian:
    #         if air.startingtime[i]<air.finishtime[j]:
    #             print("错误的工序是", i)
    #             flag = False
    #             break
    # print(flag)
    #
    # print("节拍",pulse_real)
    # print("站位SI",si)
    # print("人员SI",sii)
    # df_teamid = pd.DataFrame(air.team_id, index=[0])
    # df_stationid = pd.DataFrame(air.station_id, index=[0])
    # df_start = pd.DataFrame(air.startingtime, index=[0])
    # df_finish = pd.DataFrame(air.finishtime, index=[0])
    # df_start.to_excel('工序时刻表1.xlsx')
    # df_finish.to_excel('工序结束时刻表1.xlsx')
    # df_teamid.to_excel('工作组选择.xlsx')
    # df_stationid.to_excel('站位选择.xlsx')



