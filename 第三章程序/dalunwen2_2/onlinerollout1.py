import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time
import threading
import math
import random




'''
主要的仿真环境
'''

import numpy as np
from parameter import args_parser
import simpy


args = args_parser()

pro_num = args.pro_num
team_num = args.team_num
station_num = args.station_num
# pulse = 660
action_set = [[1,0,0],[0.8,0.2,0],[0.8,0,0.2],
              [0.6,0.2,0.2],[0.5,0.2,0.3],[0.2,0.5,0.3],
              [0.2,0,0.8],[0.2,0.3,0.5],[0.2,0.8,0]]
# action_set = []
# for i in np.arange(0.1,1,0.1):
#     for j in np.arange(0.1,1,0.1):
#         tmp_action = []
#         tmp_action.append(i)
#         if i + j > 1:
#             continue
#         else:
#             tmp_action.append(j)
#             tmp_action.append(1-i-j)
#             action_set.append(tmp_action)
# # print(len(action_set))##45
orderfreeair = []
orderfinishair = []
orderleftair = []

freeorders = args.freeorders[:]
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
    def __init__(self, id):
        self.id = id  ####一共有五个组
        self.cap = 1  ###每个组有三个人
        self.pro_num = 0  ###已经装配的工序
        self.busy_num = 0  #####被占用的工人数
        self.stfi = [0, 0]
        self.time_past = 0
        self.time_past1 = 0
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
    def __init__(self, env, id, action,station_pulse):
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
        self.action = [8,0] if (self.id == 0 or self.id == 2) else [0,0]
        self.pulse = station_pulse

        self.aircraft = None
        self.orderfreeair = []
        self.orderfinishair = []
        self.orderleftair = []
        self.pro_all = 0##已经完成的工序数
        self.pro_isctrl = 0##已完成的有紧后工序数
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

            pri = np.dot(pri, action_set[self.action[0]])
            # pri = np.dot(pri, action_set[action[0]])
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

        # 未完成的工序先加入到orders_free里面，防止因为手工作业的延长而遗漏
        ##再从里面选父工序完成了的
        air.orders_free = []
        for ppp in pro_id:
            if air.isfinish[ppp] == 0:
                if ppp in freeorders:
                    air.orders_free.append(ppp)
                    continue
                ff = True

                for preorder in dict_preorder[ppp]:
                    if air.isfinish[preorder] == 0:
                        ff = False
                        break
                if ff:
                    air.orders_free.append(ppp)

        self.orderfreeair = air.orders_free[:]
        self.orderfinishair = air.order_finish[:]
        self.orderleftair = air.order_left[:]
        print(f'zhanwei{self.id}得到了飞机在时间{self.env.now}')
        order_stfi = {}
        for i in range(pro_num):
            order_stfi[i + 1] = [air.startingtime[i + 1], air.finishtime[i + 1]]

        orders_finish = air.order_finish[:]
        for j in range(team_num):
            self.teams[j].stfi = [self.id * self.pulse, self.id * self.pulse]
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


            # 含扰动
            tmptime = 0
            if self.id < station_num - 1:
                tw = random.random()
                if self.id == 0 and tw < 0.2:
                    tmptime += dict_time[thisorder] * tw

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
                if tmp_team_stfi[1] <= self.pulse * (self.id + 1):
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

        yield self.env.timeout(self.pulse)
        self.time_past = station_maxtime
        # station_list[self.id + 1].put(self.aircraft)

    def team_process(self, aircraft, team_id):
        env = self.env
        time_w = (team_id - 1) / 10 + 1

        for tmpwhichstation in self.teams[team_id].order_buffer:
            tw = random.random()

            if aircraft.isfirstprocedure[tmpwhichstation] == 1:  # 如果工序没有紧前，直接生产
                if aircraft.isfinish[tmpwhichstation] == 0:
                    aircraft.startingtime[tmpwhichstation] = round(env.now, 2)

                    if (env.now + dict_time[tmpwhichstation] * time_w) > (self.pulse * (self.id + 1)) and self.id < (station_num-1):
                        continue

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

                    ##加扰动
                    if (env.now + dict_time[tmpwhichstation] * time_w) > (self.pulse * (self.id + 1)) and self.id < (station_num-1):
                        continue
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

            self.teams[team_id].finishtime = round(env.now, 2) - self.pulse * self.id
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
            yield env.timeout(self.pulse - station_maxtime)
            self.time_past = station_maxtime
            station_list[station_id + 1].put(aircraft)



# TODO:因为权限的问题，将step函数放在外面更容易理解，但是站位类内部的那个方法也可以使用

###因为以下为主函数中采用的获取状态的函数
def get_states(env, tmpair, allstation,thispulse):
    now_time = env.now
    if now_time >= 0 and now_time < thispulse:
        station_id = 0
    elif now_time >= thispulse and now_time < 2 * thispulse:
        station_id = 1
    elif now_time >= 2 * thispulse and now_time < 3 * thispulse:
        station_id = 2
    else:
        station_id = 3

    nowstation = allstation[station_id]
    st_time = [nowstation.teams[i].finishtime for i in range(team_num)]
    max_st_time = max(st_time)  ###当前站位实际加工的时间
    state_n = []
    states = []
    ##当前工位剩余时间
    time_remain = thispulse - max_st_time
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
    states.append(now_time / 1000)
    # states.append(max_st_time / 100)
    # states.append(time_remain / 100)
    states.append(freeo_num/10)
    states.append(remaino_num/10)
    states.append(maxdepth/10)
    states.append(avgdepth/10)
    states.append(maxdeptime / 1000)
    states.append(avgdeptime / 1000)
    done = True if remaino_num == 0 else False
    return states, done


def get_obs(env, tmpair, allstation,thispulse):
    now_time = env.now
    if now_time >= 0 and now_time < thispulse:
        station_id = 0
    elif now_time >= thispulse and now_time < 2 * thispulse:
        station_id = 1
    elif now_time >= 2 * thispulse and now_time < 3 * thispulse:
        station_id = 2
    else:
        station_id = 3

    nowstation = allstation[station_id]
    st_time = [nowstation.teams[i].finishtime for i in range(team_num)]
    timep = [nowstation.teams[i].time_past1 for i in range(team_num)]
    max_timep = max(timep)  ###当前站位实际加工的时间
    avg_timep = sum(timep) / team_num
    si = 0
    sifi = 0
    for i in range(team_num):
        si += (nowstation.teams[i].time_past1 - avg_timep) ** 2

    si = math.sqrt(si/team_num)/100
    states = []
    ##当前工位剩余时间

    ##当前自由工序数量
    freeo_num = len(tmpair.orders_free)
    ##当前剩余工序数量
    remaino_num = pro_num - len(tmpair.order_finish)
    critio = 0 if nowstation.pro_all==0 else nowstation.pro_isctrl/nowstation.pro_all
    ##当前自由工序的后续*时间之和deptime
    states.append(nowstation.id)
    states.append(now_time / 1000)
    states.append(freeo_num/10)
    states.append(remaino_num/10)
    states.append(si)
    states.append(avg_timep/100)
    states.append(max_timep/100)
    states.append(critio)
    done = True if remaino_num == 0 else False
    return states, done


def get_reward(now_time, allstation, air,thispulse):
    ###最大生产时间最小
    if now_time > 0 and now_time <= thispulse + 1:
        station_id = 0
    elif now_time > thispulse + 1 and now_time <= 2 * thispulse + 1:
        station_id = 1
    elif now_time > 2 * thispulse + 1 and now_time <= 3 * thispulse + 1:
        station_id = 2
    else:
        station_id = 3

    nowstation = allstation[station_id]
    # if station_id ==3:
    #     return -get_pulse(allstation)/100

    # st_time = [nowstation.teams[i].finishtime for i in range(team_num)]
    st_time = [nowstation.teams[i].time_past1 for i in range(team_num)]
    max_st_time = max(st_time)
    avg_st_time = sum(st_time) / team_num
    # si = 0
    # for i in range(team_num):
    #     si += (nowstation.teams[i].finishtime - avg_st_time) ** 2
    # si = math.sqrt(si/team_num)
    # rew = max_st_time / 100 + si / 100
    rew = max_st_time / 100
    remaino = air.order_left[:]
    time_r = 0
    for o in remaino:
        time_r += dict_time[o]


    rew = avg_st_time/1000*2
    return rew
    # return -rew/10  ###奖励越小越好要取负





def reset_env(env,thispulse):
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
        allstation.append(Station(env, k, 1,thispulse))
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
                  all_aircraft,thispulse
                  ):
    for i in range(item_num):
        print(f'{round(env.now, 2)} - item: item_{i} - created')
        tmp_aircarft = SingelAircraft(env, i)
        last_q.put(tmp_aircarft)
        tmp = tmp_aircarft
        all_aircraft.append(tmp)
        # t = random.expovariate(1 / MEAN_TIME)
        t = 10 * thispulse
        yield env.timeout(round(t, 1))


def get_pulse(allstation):
    pulse_real = 0
    si = 0 ##各站位的完工时间
    station_times = []
    for i in range(station_num):
        time_tmp_station = 0
        for j in range(team_num):
            # if i == station_num-2:
            #     allstation[i].teams[j].finishtime += allstation[i+1].teams[j].finishtime
            time_tmp_station = max(time_tmp_station, allstation[i].teams[j].finishtime)
            print("真正统计时的finishtime", allstation[i].teams[j].finishtime)
        station_times.append(time_tmp_station)
        pulse_real = max(pulse_real, time_tmp_station)
    avg_st_time = sum(station_times) / station_num
    for i in range(station_num):
        si += (station_times[i] - avg_st_time) ** 2
    si = si/station_num


    return pulse_real,si


n_agents = 2

n_actions = 9





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


def generate_episode(agents, conf,pulses,thispulse,episode_num, SI,evaluate=False):
        env = simpy.Environment()
        o, u, r, s, o_,s_ = [], [], [], [], [], []
        au, avail_u_, u_onehot, terminate, padded = [], [], [], [], []

        allstation, station_list, air = reset_env(env,thispulse)
        reset_station(env, allstation, station_list)
        episode = {}
        times = []
        pp = 0

        def production(env, agents, air, allstation, station_list, evaluate, start_epsilon):
            # o, u, o_, r, s, s_, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], [], [], []
            episode = {}
            now_state, done = get_states(env, air, allstation,thispulse)
            last_action = np.zeros((n_agents, n_actions))
            agents.policy.init_hidden(1)
            epsilon = 0 if episode_num>int(conf.n_epochs*0.95) else conf.start_epsilon
            # epsilon = 0 if evaluate else start_epsilon
            # print(now_state)
            pr = []
            while not done:
                obs, _ = get_obs(env, air, allstation,thispulse)
                state, _ = get_states(env, air, allstation,thispulse)
                actions, avail_actions, actions_onehot = [], [], []

                # print("当前的obs为", obs)
                # print("当前的state为", state)
                for agent_id in range(n_agents):
                    avail_action = action_set
                    action = agents.choose_action(state, last_action[agent_id], agent_id, avail_action,
                                                  epsilon, evaluate)

                    # 生成动作的onehot编码
                    action_onehot = np.zeros(n_actions)
                    action_onehot[action] = 1
                    actions.append(action)
                    actions_onehot.append(action_onehot)
                    avail_actions.append(avail_action)
                    last_action[agent_id] = action_onehot

                now_time = env.now
                if now_time >= 0 and now_time < thispulse:
                    station_id = 0
                elif now_time >= thispulse and now_time < 2 * thispulse:
                    station_id = 1
                elif now_time >= 2 * thispulse and now_time < 3 * thispulse:
                    station_id = 2
                else:
                    station_id = 3
                print('此时的站位为：', station_id)
                nowstation = allstation[station_id]
                nowstation.distribution(air, actions)
                if nowstation.id < 3:
                    yield env.timeout(thispulse)
                else:
                    yield env.timeout(thispulse + 1000)

                next_state, done = get_states(env, air, allstation,thispulse)
                next_obs,done = get_states(env,air,allstation,thispulse)
                reward = get_reward(env.now, allstation, air,thispulse)
                # print("actions: ", actions)

                o.append([state,state])
                s.append(state)
                u.append(np.reshape(actions, [n_agents, 1]))
                u_onehot.append(actions_onehot)

                # r.append([reward])
                pr.append(reward)
                o_.append([next_state,next_state])
                s_.append(next_state)
                au.append(avail_actions)
                terminate.append([done])
                padded.append([0.])

                # step += 1
                # if self.conf.epsilon_anneal_scale == 'step':
                #     epsilon = epsilon - self.anneal_epsilon if epsilon > self.end_epsilon else epsilon
                station_list[station_id + 1].put(air)

                if nowstation.id == station_num-1:
                    this_pulse,times = get_pulse(allstation)
                    pulses.append(this_pulse)
                    if this_pulse < 620:
                        # agents.policy.save_model(10)
                        SI.append(times)
                        print("平滑指数是：", math.sqrt(times))
                    # print("当前节拍是：", this_pulse)
                    pp = this_pulse
                    print(pr)


                    for i in range(station_num):
                        # if (i == 0):
                        #     tmp = 0.5 * pr[0] + 0.5 * 100/(this_pulse - 600)
                        # elif (i == 1):
                        #     tmp = 0.5 * pr[1] + 0.3 * 100/(this_pulse - 600)
                        # elif (i == 2):
                        #     tmp = 0.6 * pr[2] + 0.4 *100/(this_pulse - 600)


                        if this_pulse < 620:
                            tmp = 1
                        elif this_pulse < 650:
                            tmp = 0.5
                        else:
                            tmp = -1

                        r.append([tmp])

                    yield env.timeout(thispulse + 2000)
            # 最后一个动作

            # target q 在last obs需要avail_action



            # 当step<self.episode_limit时，输入数据加padding
            # for i in range(step, episode_limit):
            #     o.append(np.zeros((n_agents, obs_shape)))
            #     u.append(np.zeros([n_agents, 1]))
            #     s.append(np.zeros(state_shape))
            #     r.append([0.])
            #     o_.append(np.zeros((n_agents, obs_shape)))
            #     s_.append(np.zeros(state_shape))
            #     u_onehot.append(np.zeros((n_agents, n_actions)))
            #     avail_u.append(np.zeros((n_agents, n_actions)))
            #     avail_u_.append(np.zeros((n_agents, n_actions)))
            #     padded.append([1.])
            #     terminate.append([1.])



            for key in episode.keys():
                episode[key] = np.array([episode[key]])
            if not evaluate:
                start_epsilon = epsilon
            return episode,pp,times

        env.process(production(env, agents, air, allstation, station_list,evaluate,conf.start_epsilon))


        env.run(5000)

        episode['o'] = o.copy()
        episode['s'] = s.copy()
        episode['u'] = u.copy()
        episode['r'] = r.copy()
        episode['o_'] = o_.copy()
        episode['s_'] = s_.copy()
        episode['avail_u'] = action_set.copy()
        episode['avail_u_'] = action_set.copy()
        episode['u_onehot'] = u_onehot.copy()
        episode['padded'] = padded.copy()
        episode['terminated'] = terminate.copy()
        return episode,times,pp


class ReplayBuffer:
    def __init__(self, conf):
        self.conf = conf
        self.episode_limit = conf.episode_limit
        self.n_actions = conf.n_actions
        self.n_agents = conf.n_agents
        self.state_shape = conf.state_shape
        self.obs_shape = conf.obs_shape
        self.size = conf.buffer_size

        self.current_idx = 0
        self.current_size = 0

        self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
                        's': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'r': np.empty([self.size, self.episode_limit, 1]),
                        'o_': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        's_': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'avail_u_': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'u_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'padded': np.empty([self.size, self.episode_limit, 1]),
                        'terminated': np.empty([self.size, self.episode_limit, 1]),
                        }
        self.lock = threading.Lock()
        print("Replay Buffer inited!")

    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]  # 200
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['u'][idxs] = episode_batch['u']
            self.buffers['s'][idxs] = episode_batch['s']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['o_'][idxs] = episode_batch['o_']
            self.buffers['s_'][idxs] = episode_batch['s_']
            self.buffers['avail_u'][idxs] = episode_batch['avail_u']
            self.buffers['avail_u_'][idxs] = episode_batch['avail_u_']
            self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminated']

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
