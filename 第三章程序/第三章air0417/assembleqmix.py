import numpy as np

import time
import threading




'''
TODO:
    2023.04.19
'''

import numpy as np
from parameter import args_parser
from utils import ReplayBuffer
import simpy
from config import Config
from agent import Agents
import math


args = args_parser()

pro_num = args.pro_num
team_num = args.team_num
station_num = args.station_num
worker_num = args.worker_num
pulse = args.pulse
schedule_day = args.schtime

# action_set = [[1,0,0],[0.8,0.2,0],[0.8,0,0.2],
#               [0.6,0.2,0.2],[0.5,0.2,0.3],[0.2,0.5,0.3],
#               [0.2,0,0.8],[0.2,0.3,0.5],[0.2,0.8,0]]

action_set = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

# orderfreeair = []
# orderfinishair = []
# orderleftair = []



dict_time = args.dict_time
dict_preorder = args.dict_preorder
dict_team = args.dict_team
dict_workernum = args.dict_workernum
dict_free = args.dict_free
pro_id = args.pro_id

dict_ready = {}
dict_postorder = args.dict_postorder
dict_postnum = args.dict_postnum
dict_posttime = args.dict_posttime
dict_postsinktime = args.dict_postsinktime
init_frees = args.free_orders
schtime = args.schtime
schmaxcnt = 2

pro_a = [i+1 for i in range(283)]
pro_b = [i+284 for i in range(680-283)]
pro_c = [i+681 for i in range(2338-680)]
pro_d = [i+2339 for i in range(2520-2338)]
pro_e = [i+2521 for i in range(2526-2520)]
pro_f = [i+2527 for i in range(2721-2526)]
pro_g = [i+2722 for i in range(2836-2721)]
pro_h = [i+2837 for i in range(2966-2836)]
pro_i = [i+2967 for i in range(3032-2966)]
pro_j = [i+3033 for i in range(3144-3032)]
pro_k = [i+3145 for i in range(3169-3144)]
pro_l = [i+3170 for i in range(3182-3169)]

pack_len = [len(pro_a),len(pro_b),len(pro_c),len(pro_d),len(pro_e),len(pro_f),
            len(pro_g),len(pro_h),len(pro_i),len(pro_j),len(pro_k),len(pro_l)]

pack_num = 12


class Team:
    def __init__(self, id):
        self.id = id  ####一共有17个专业
        self.cap = 5 ###每个组有5人
        self.busy_num = 0  #####被占用的工人数
        self.workers = []
        self.wsource = []##把工人搞成资源，稍后请求的时候会用，在工序的属性中添加为他分配的工人
        self.islimit = False###team是否超时，若超时则不能再向其中添加
        self.assembletime = 0 ##装配时间
        self.timepast = 0 ##实际装配时间（不含等待时间）

        for i in range(worker_num):##初始化工人列表
            self.workers.append(worker(i))


    def worker_sel(self,worker_free,worker_num):  # worker_free:可选的工人，worker_num:所需的人数
        tmpworkers = self.workers[:]
        team_idtime = []
        for worker in tmpworkers:
            if worker.id in worker_free:
                team_idtime.append((worker.id, worker.time_past))
        team_idtime = sorted(team_idtime, key=lambda x: x[1])
        res = [ti[0] for ti in team_idtime]
        return res[:worker_num]

    def worker_sel1(self, team_free, team_action):  # 选择工人
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

    def worker_process(self, env,aircraft, worker_id):
        this_worker = self.workers[worker_id]
        time_w = this_worker.efi

        for tmpwhichstation in this_worker.order_buffer:
            w_list = aircraft.worker_list[tmpwhichstation][:]
            if worker_id not in w_list:
                print(tmpwhichstation,worker_id)
            w_list = w_list.remove(worker_id)
            if dict_free[tmpwhichstation] == 1:  # 如果工序没有紧前，直接生产
                if aircraft.isfinish[tmpwhichstation] == 0:
                    aircraft.startingtime[tmpwhichstation] = round(env.now, 2)
                    rs = []
                    if w_list:
                        for r in w_list:
                            request = self.wsource[r].request()  # 生成request事件申请资源
                            rs.append(request)
                            yield request  # 等待访问
                    yield env.timeout(dict_time[tmpwhichstation] * time_w)  # 生成延时时间，在这里停留该工序的装配时间
                    # print(f'实际工序：{tmpwhichstation}-站位：{self.id}-工作组:{team_id}-时间：{self.env.now}')
                    if w_list:
                        for i in range(len(w_list)):
                            self.wsource[w_list[i]].release(rs[i]) # 释放资源

                    aircraft.finishtime[tmpwhichstation] = round(env.now, 2)
                    aircraft.team_id[tmpwhichstation] = worker_id
                    aircraft.station_id[tmpwhichstation] = self.id
                    try:
                        aircraft.istrigger[tmpwhichstation - 1].succeed()
                    except:
                        pass
                    aircraft.isfinish[tmpwhichstation] = 1
                    aircraft.order_finish.append(tmpwhichstation)
                    # aircraft.order_left.remove(tmpwhichstation)
                    # self.orderfinishair.append(tmpwhichstation)
                    # self.orderleftair.remove(tmpwhichstation)

            else:
                if aircraft.isfinish[tmpwhichstation] == 0:
                    if type(dict_preorder[tmpwhichstation]) == type(1):  # 如果只有一个紧前工序，等待它完成
                        yield aircraft.istrigger[dict_preorder[tmpwhichstation] - 1]
                    else:  # 等待全部的紧前工序完成
                        this_events = [aircraft.istrigger[k - 1] for k in dict_preorder[tmpwhichstation]]
                        yield simpy.events.AllOf(env, this_events)

                    rs = []
                    if w_list:
                        for r in w_list:
                            request = self.wsource[r].request()  # 生成request事件申请资源
                            rs.append(request)
                            yield request  # 等待访问

                    # print(f'实际工序：{tmpwhichstation}-站位：{self.id}-工作组:{worker_id}-时间：{env.now}')



                    aircraft.startingtime[tmpwhichstation] = round(env.now, 2)
                    yield env.timeout(dict_time[tmpwhichstation] * time_w)
                    if w_list:
                        for i in range(len(w_list)):
                            self.wsource[w_list[i]].release(rs[i]) # 释放资源
                    # print(f'实际工序：{tmpwhichstation}-站位：{self.id}-工作组:{team_id}-时间：{self.env.now}')
                    aircraft.finishtime[tmpwhichstation] = round(env.now, 2)
                    try:
                        aircraft.istrigger[tmpwhichstation - 1].succeed()
                    except:
                        pass
                    aircraft.isfinish[tmpwhichstation] = 1
                    aircraft.order_finish.append(tmpwhichstation)
                    aircraft.team_id[tmpwhichstation] = worker_id
                    aircraft.station_id[tmpwhichstation] = self.id
                    # aircraft.order_left.remove(tmpwhichstation)#应该加上这一句
                    # self.orderfinishair.append(tmpwhichstation)
                    # self.orderleftair.remove(tmpwhichstation)
            self.workers[worker_id].time_past += dict_time[tmpwhichstation] * time_w
            self.workers[worker_id].finishtime = round(env.now, 2) - pulse * self.id






class worker:
    def __init__(self,id):
        self.id = id  ####一共有17个专业
        self.pro_num = 0  ###已经装配的工序
        self.busy_num = 0  #####被占用的工人数
        self.stfi = [0, 0] ##stfi[1]表示完工时间
        self.time_past = 0
        self.order_buffer = []
        self.order_finish = []
        self.finishtime = 0
        self.efi = self.id/10+0.8

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
        self.worker_list = dict(zip(pro_id, single_isfinish))##工序分配给工人的号
        self.pack_cnt = [0]*pack_num###统计各工序包完成的情况
        self.orders_free = init_frees[0]##初始的无紧前工序（仅有A工序包的工序）
        self.pack_id = 0##初始的工序包所在处为A
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
        self.assembletime = 0
        self.teams = []
        self.rule = [3,0]  #####初始化的工序分配规则为时间越长越优先
        self.store = simpy.Store(env, capacity=1)
        self.action = action

        self.aircraft = None
        self.orderfreeair = []
        self.orderfinishair = []
        self.orderleftair = []

        for i in range(team_num):
            self.teams.append(Team(i))
        self.schcnt = 0


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
        tmp_order =[]
        for free in order_free:
            tmp_order.append(free)
        # tmp_order = list(order_free)
        for free in tmp_order:
            # print(free)
            a = 1
            pri0 = (dict_time[free] - 0.2) / (51.9 - 0.2)
            pri1 = dict_postnum[free] / 11
            pri2 = (dict_posttime[free] - 0.2) / (160.9 - 0.2)
            pri3 = (dict_postsinktime[free] - 0.2) / (79.85 - 0.2)
            # pri = [pri1, pri2, pri3]
            pri = [pri0,pri1, pri2, pri3]

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

###scheduling agent


###模拟每隔一个节拍将飞机运到下一个站位
def transport(env,station_list,air):
    station_id = 0
    now_time = env.now
    while True:
        if now_time>pulse*(station_num-1)+10:
            timeouttime = pulse
        else:timeouttime = 1000
        yield env.timeout(timeouttime)
        station_list[station_id + 1].put(air)
        station_id += 1




###因为以下为主函数中采用的获取状态的函数
def get_states(env, tmpair, allstation):
    now_time = env.now
    if now_time >= 0 and now_time < pulse:
        station_id = 0
    elif now_time >= pulse and now_time < 2 * pulse:
        station_id = 1
    elif now_time >= 2 * pulse and now_time < 3 * pulse:
        station_id = 2
    elif now_time >= 3 * pulse and now_time < 4 * pulse:
        station_id = 3
    else:
        station_id = 4


    nowstation = allstation[station_id]
    # st_time = [nowstation.teams[i].finishtime for i in range(team_num)]
    for team in nowstation.teams:
        team.assembletime = max([w.stfi[1] - station_id * pulse for w in team.workers])
    max_st_time = max([t.assembletime for t in nowstation.teams])
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

    # st_time = [nowstation.teams[i].finishtime for i in range(team_num)]
    # st_time = [nowstation.time_past for i in range(station_num)]

    for team in nowstation.teams:
        team.assembletime = max([w.stfi[1] - station_id * pulse for w in team.workers])
    st_time = [t.assembletime for t in nowstation.teams]
    max_st_time = max([t.assembletime for t in nowstation.teams])
    # max_st_time = max(st_time)
    avg_st_time = sum(st_time) / team_num
    si = 0
    # for i in range(team_num):
    #     si += (nowstation.teams[i].finishtime - avg_st_time) ** 2
    # si = math.sqrt(si/team_num)
    # rew = max_st_time / 100 + si / 100
    rew = max_st_time / 100
    remaino = air.order_left[:]
    time_r = 0
    for o in remaino:
        time_r += dict_time[o]
    return -rew  ###奖励越小越好




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

    for station in allstation:
        for team in station.teams:
            team.assembletime = max([w.stfi[1] - station.id * pulse for w in team.workers])
        station.assembletime = max([t.assembletime for t in station.teams])
        print(f'站位{station.id}节拍是{station.assembletime}')
    pulses = [s.assembletime for s in allstation]
    pulse_real = max([s.assembletime for s in allstation])
    return pulse_real

n_agents = 2

n_actions = 9




class ScheduleWorker:
    def __init__(self, agents, conf):
        self.orders_finish = []
        self.order_left = []
        self.pack_cnt = [0] * pack_num
        self.pack_id = 0
        self.order_free = set(init_frees[0][:])
        self.is_first = True
        self.epsilon = conf.start_epsilon
        self.agents = agents
        self.episode_limit = conf.episode_limit
        self.n_actions = conf.n_actions
        self.n_agents = conf.n_agents
        self.state_shape = conf.state_shape
        self.obs_shape = conf.obs_shape

    def resetschedule(self):
        self.orders_finish = []
        self.order_left = []
        self.pack_cnt = [0] * pack_num
        self.pack_id = 0
        self.order_free = set(init_frees[0][:])
        self.is_first = True
        self.o, self.u, self.r, self.s, self.o_, self.s_ = [], [], [], [], [], []
        self.au, self.avail_u_, self.u_onehot, self.terminate, self.padded = [], [], [], [], []

    def production(self,env, air, agents, allstation, station_list,evaluate):
        episode = {}

        now_state, done = get_states(env, air, allstation)
        last_action = np.zeros((n_agents, n_actions))
        agents.policy.init_hidden(1)
        epsilon = 0 if evaluate else self.epsilon
        while True:

            if self.is_first:
                self.is_first = False
            else:
                self.order_free = set(self.order_left[:])
                self.order_left = []

            now_time = env.now
            if now_time >= 0 and now_time < pulse:
                station_id = 0
            elif now_time >= pulse and now_time < 2 * pulse:
                station_id = 1
            elif now_time >= 2 * pulse and now_time < 3 * pulse:
                station_id = 2
            elif now_time >= 3 * pulse and now_time < 4 * pulse:
                station_id = 3
            else:
                station_id = 4

            obs, _ = get_states(env, air, allstation)
            state, _ = get_states(env, air, allstation)
            actions, avail_actions, actions_onehot = [], [], []

            for agent_id in range(n_agents):
                avail_action = action_set
                action = agents.choose_action(obs, last_action[agent_id], agent_id, avail_action,
                                              epsilon, evaluate)
                # 生成动作的onehot编码
                action_onehot = np.zeros(n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            if now_time > pulse * (station_num - 1) + 10:
                last_schedule = True
            else:
                last_schedule = False  ####是否是最后一次调度，最后一次调度不限制时间

            this_station = allstation[station_id]
            this_station.schcnt += 1
            ##计算本次调度的作用时间
            if this_station.schcnt < schmaxcnt:
                this_schtime = this_station.schcnt * schtime
            else:
                this_schtime = pulse - schtime
            ##如果是最后一个调度，要把时间设的大一点，保证所有的工序都装配
            if last_schedule: this_schtime = 1000
            ##初始的free要从每个工序包的free开始
            # this_station.orderfreeair = air.orders_free[:]
            # this_station.orderfinishair = air.order_finish[:]
            # this_station.orderleftair = air.order_left[:]
            # pack_id = air.pack_id
            print("当前的调度时间是",this_schtime)
            print(f'站位{this_station.id}得到了飞机在时间{env.now}当前调度的包是{self.pack_id}')
            order_stfi = {}
            for i in range(pro_num):
                order_stfi[i + 1] = [air.startingtime[i + 1], air.finishtime[i + 1]]

            # orders_finish = air.order_finish[:]
            for j in range(team_num):
                tmp_team = this_station.teams[j]
                tmp_team.islimit = False
                for k in range(worker_num):
                    ####这个时间还有点问题，目前是调度次数为2的情况
                    tmp_team.workers[k].stfi = [station_id * pulse + schtime * (this_station.schcnt - 1),
                                                station_id * pulse + schtime * (this_station.schcnt - 1)]

            worker_free = [i for i in range(worker_num)]
            # order_free = set(air.orders_free[:])
            print(self.order_free)
            this_rule = actions
            print("当前规则为", this_rule)
            while self.order_free and this_station.schcnt <= schmaxcnt:

                # print(order_free)

                thisorder = this_station.cal_pri(self.order_free, this_rule[0])

                thisnum = dict_workernum[thisorder]  ###当前工序所需的人数
                thisteam = this_station.teams[dict_team[thisorder]]

                if thisteam.islimit:##当前的班组所有人都已经分配了任务，不能再分配了
                    self.order_free.remove(thisorder)
                    self.order_left.append(thisorder)
                    continue

                this_wnum = dict_workernum[thisorder]
                thisworkersid = thisteam.worker_sel(worker_free, this_wnum)  ##此处w可能是多个人，返回的是worker的ID
                thisworkers = [thisteam.workers[w] for w in thisworkersid]
                air.worker_list[thisorder] = thisworkersid[:]
                # print(air.worker_list[thisorder])
                for w in thisworkersid:
                    thisteam.workers[w].order_buffer.append(thisorder)

                ###processing

                latestfi = max([w.stfi[1] for w in thisworkers])
                maxefi = max([w.efi for w in thisworkers])
                if dict_free[thisorder] == 1:  ####初始的自由工序
                    order_stfi[thisorder][0] = latestfi
                    order_stfi[thisorder][1] = order_stfi[thisorder][0] + dict_time[thisorder] * maxefi


                else:
                    tmp_maxtime = 0
                    for tmpfather in dict_preorder[thisorder]:  # 遍历所有的父工序
                        # print('父工序为',order_stfi)
                        tmp_maxtime = max(tmp_maxtime, order_stfi[tmpfather][1])
                    order_stfi[thisorder][0] = max(latestfi, tmp_maxtime)
                    order_stfi[thisorder][1] = order_stfi[thisorder][0] + dict_time[thisorder] * maxefi

                plustime = pulse if this_station.schcnt == 2 else schtime
                if not last_schedule:
                    if order_stfi[thisorder][1] > station_id * pulse + plustime:
                        # print("当前的时间应不大于",station_id * pulse+plustime)
                        thisteam.islimit = True
                        air.orders_free = list(self.order_free.copy())
                        air.pack_id = self.pack_id
                        self.order_free.remove(thisorder)
                        self.order_left.append(thisorder)
                        continue



                print(
                    f'工序：{thisorder}-站位：{station_id}-开始时间：{order_stfi[thisorder][0]}-专业：{dict_team[thisorder]}')

                self.pack_cnt[self.pack_id] += 1
                ##更新所有相关工人的工作时间
                for k in range(thisnum):
                    thisworkers[k].order_buffer.append(thisorder)
                    thisworkers[k].stfi[1] = order_stfi[thisorder][1]
                    thisworkers[k].time_past += dict_time[thisorder] * maxefi
                    self.orders_finish.append(thisorder)
                for order in dict_postorder[thisorder]:
                    flag = True
                    for preorder in dict_preorder[order]:
                        if preorder not in self.orders_finish:
                            flag = False
                            break
                    if flag:
                        self.order_free.add(order)
                self.order_free.remove(thisorder)
                air.isfinish[thisorder] = 1
                if pack_len[self.pack_id] == self.pack_cnt[self.pack_id] and self.pack_id < pack_num - 1:
                    self.pack_id += 1
                    self.order_free = set(init_frees[self.pack_id][:])


            print(self.order_free)
            yield env.timeout(this_schtime)
            next_state, done = get_states(env, air, allstation)
            reward = get_reward(env.now, allstation, air)
            # print("actions: ", actions)

            self.o.append([obs, obs])
            self.s.append(state)
            self.u.append(np.reshape(actions, [n_agents, 1]))
            self.u_onehot.append(actions_onehot)

            self.r.append([reward])
            self.o_.append([next_state, next_state])
            self.s_.append(next_state)
            self.au.append(avail_actions)
            self.terminate.append([done])
            self.padded.append([0.])

            # step += 1
            # if self.conf.epsilon_anneal_scale == 'step':
            #     epsilon = epsilon - self.anneal_epsilon if epsilon > self.end_epsilon else epsilon
            station_list[station_id + 1].put(air)

            if this_station.id == station_num-1:
                this_pulse = get_pulse(allstation)
                print("当前节拍是：", this_pulse)
                yield env.timeout(pulse + 2000)

            for key in episode.keys():
                episode[key] = np.array([episode[key]])
            if not evaluate:
                start_epsilon = epsilon


    def generate_episode(self,agents,episode_num=None, evaluate=False):
        env = simpy.Environment()
        allstation, station_list, air = reset_env(env)
        # reset_station(env, allstation, station_list)
        episode = {}
        env.process(transport(env, station_list,air))
        env.process(self.production(env, air, agents, allstation, station_list,evaluate))
        env.run(2000)

        episode['o'] = self.o.copy()
        episode['s'] = self.s.copy()
        episode['u'] = self.u.copy()
        episode['r'] = self.r.copy()
        episode['o_'] = self.o_.copy()
        episode['s_'] = self.s_.copy()
        episode['avail_u'] = action_set.copy()
        episode['avail_u_'] = action_set.copy()
        episode['u_onehot'] = self.u_onehot.copy()
        episode['padded'] = self.padded.copy()
        episode['terminated'] = self.terminate.copy()
        return episode





if __name__ == '__main__':


    conf = Config()
    #调度一次的试验
    # agents = Agents(conf)
    # rollout_worker = ScheduleWorker(agents, conf)
    # rollout_worker.resetschedule()
    # episode = rollout_worker.generate_episode(agents)
    # print(episode)

    agents = Agents(conf)
    rollout_worker = ScheduleWorker(agents, conf)
    buffer = ReplayBuffer(conf)

    # save plt and pk

    win_rates = []
    episode_rewards = []
    train_steps = 0
    for epoch in range(conf.n_epochs):
        # print("train epoch: %d" % epoch)
        episodes = []
        for episode_idx in range(conf.n_eposodes):
            episode= rollout_worker.generate_episode(agents,episode_idx)
            episodes.append(episode)
            # print("当前的episode为",episode)

        episode_batch = episodes[0]

        episodes.pop(0)
        for episode in episodes:
            for key in episode_batch.keys():
                episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)


        buffer.store_episode(episode_batch)
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
        for train_step in range(conf.train_steps):
            mini_batch = buffer.sample(min(buffer.current_size, conf.batch_size))  # obs； (64, 200, 3, 42)
            # print(mini_batch['o'].shape)
            agents.train(mini_batch, train_steps)
            train_steps += 1
