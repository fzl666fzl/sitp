'''
2023.07.13
每个站位都有规则计算+装配仿真
'''
import simpy
import random
import pandas as pd
import math

import numpy as np
from parameter import args_parser
import simpy
import math


args = args_parser()

pro_num = args.pro_num
team_num = args.team_num
station_num = args.station_num
worker_num = args.worker_num

schedule_day = args.schtime

action_set = [[1,0,0],[0.8,0.2,0],[0.8,0,0.2],
              [0.6,0.2,0.2],[0.5,0.2,0.3],[0.2,0.5,0.3],
              [0.2,0,0.8],[0.2,0.3,0.5],[0.2,0.8,0]]


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

pack_len = args.pack_len
pack_num = 12

n_agents = 2
n_actions = 9

class Team:
    def __init__(self, id,env,station,thispulse):
        self.id = id  ####一共有17个班组
        self.station = station
        self.cap = 5 ###每个组有5人
        self.busy_num = 0  #####被占用的工人数
        self.workers = []
        self.islimit = False###team是否超时，若超时则不能再向其中添加
        self.assembletime = 0 ##装配时间
        self.timepast = 0 ##实际装配时间（不含等待时间）
        self.timelists = []
        self.singlepeak = 0##最多有几个工人在装配


        for i in range(worker_num):##初始化工人列表
            self.workers.append(Worker(i,env,station,thispulse))

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


class Worker:
    def __init__(self,id,env,station,thispulse):
        self.id = id
        self.env = env
        self.station = station
        self.pulse = thispulse
        self.pro_num = 0  ###已经装配的工序数
        self.busy_num = 0  #####被占用的工人数
        self.stfi = [0, 0] ##stfi[1]表示完工时间
        self.timelist = {}###每道工序的装配时间，因为考虑到与其他工人合作的情况，工序的加工时间不一定就是dicttime的时间
        self.time_past = 0
        self.order_buffer = [] ###工人要处理的工序列表
        self.order_finish = []
        self.finishtime = 0
        self.efi = self.id/10+0.8 ###熟练度
        self.resource = simpy.Resource(env, capacity=1)

    def woker_process(self, aircraft):
        env = self.env
        for order in self.order_buffer:
            if dict_free[order] != 1:
                if aircraft.isfinish[order] == 0:
                    ###等待所有的紧前工序完成
                    this_events = [aircraft.istrigger[int(k) - 1] for k in dict_preorder[order]]
                    yield simpy.events.AllOf(env, this_events)
                ###等待所有相关的工人空闲
            worker_list = aircraft.worker_list[order]


            ##给每道工序创建一个事件列表，只要就绪的工人就放到此列表中，列表满了就触发装配


            aircraft.workertrigger[order][self.id].succeed()
            this_events = [aircraft.workertrigger[order][k] for k in worker_list]
            yield simpy.events.AllOf(env, this_events)

            aircraft.startingtime[order] = round(env.now, 2)
            yield env.timeout(self.timelist[order])
            # if self.station == 0:
            # print(f'站位{self.station}工人{self.id}工序{order}时间{env.now}专业{dict_team[order]}')



            aircraft.finishtime[order] = round(env.now, 2)
            try:
                aircraft.istrigger[order - 1].succeed()
            except:
                pass
            aircraft.isfinish[order] = 1
            self.time_past += self.timelist[order]
        self.finishtime = round(env.now, 2) - self.pulse * self.id


class Station:
    def __init__(self, env, id, action,thispulse):
        self.env = env
        self.id = id
        self.pulse = thispulse
        self.order_buffer = []
        self.order_finish = []
        self.time_past = 0  ##已经加工的时间
        self.time_remaining = 0  ####剩余的时间
        self.assembletime = 0
        self.teams = []
        self.rule = [0,0]  #####初始化的工序分配规则为时间越长越优先
        self.store = simpy.Store(env, capacity=1)
        self.action = action
        self.schcnt = 1
        self.last_schedule = False
        self.aircraft = None


        for i in range(team_num):
            self.teams.append(Team(i,env,self.id,self.pulse))
        self.schcnt = 0


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
            pri = [pri1, pri2, pri3]
            # pri = [pri0,pri1, pri2, pri3]

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

    def calrule(self,air,nowrule):
        ##当前的调度时间间隔
        print(f'站位{self.id}调度飞机{air.id}在时间{self.env.now}')
        for j in range(team_num):
            tmp_team = self.teams[j]
            tmp_team.islimit = False##该班组还有工人能够装配
            for k in range(worker_num):
                # tmp_team.workers[k].stfi = [self.id * pulse + schtime * (self.schcnt - 1),
                #                             self.id * pulse + schtime * (self.schcnt - 1)]
                self.teams[j].workers[k].stfi = [self.id * self.pulse,self.id * self.pulse]
                self.teams[j].workers[k].order_buffer = []

        worker_free = [i for i in range(worker_num)]
        if air.isfirst:##如果不是第一次调度，orders-free就要从left里面挑选
            air.isfirst = False

        else:
            air.orders_free = set(air.orders_left)
            air.orders_left = []
        # order_free = set(order_free)
        # print(order_free)
        this_rule = nowrule

        while air.orders_free:
            thisorder = self.cal_pri(air.orders_free, this_rule[0])
            thisnum = dict_workernum[thisorder]  ###当前工序所需的人数
            thisteam = self.teams[dict_team[thisorder]]

            order_stfi = [0,0]


            ###??????
            # if thisteam.islimit:
            #     air.orders_free.remove(thisorder)
            #     air.orders_left.append(thisorder)
            #     continue

            this_wnum = dict_workernum[thisorder]
            thisworkersid = thisteam.worker_sel(worker_free, this_wnum)  ##此处w可能是多个人，返回的是worker的ID
            thisworkers = [thisteam.workers[w] for w in thisworkersid]

            # print(air.worker_list[thisorder])
            ###processing

            latestfi = max([w.stfi[1] for w in thisworkers])
            maxefi = max([w.efi for w in thisworkers])

            #含扰动
            # tw = random.random()
            # if tw < 0.2:
            #     dict_time[thisorder] += dict_time[thisorder] * tw

            if dict_free[thisorder] == 1:  ####初始的自由工序
                order_stfi[0] = latestfi
                order_stfi[1] = order_stfi[0] + dict_time[thisorder] * maxefi
            else:
                tmp_maxtime = 0
                for tmpfather in dict_preorder[thisorder]:  # 遍历所有的父工序
                    tmp_maxtime = max(tmp_maxtime, air.finishtime[tmpfather])
                order_stfi[0] = max(latestfi, tmp_maxtime)
                order_stfi[1] = order_stfi[0] + dict_time[thisorder] * maxefi

            # plustime = pulse if self.schcnt == 2 else schtime
            plustime = self.pulse
            if self.id < station_num - 1:
                if order_stfi[1] > self.id * self.pulse + plustime:
                    air.orders_free.remove(thisorder)
                    air.orders_left.append(thisorder)
                    # thisteam.islimit = True
                    continue


            air.pack_cnt[air.pack_id] += 1
            thisteam.singlepeak = max(thisteam.singlepeak, thisnum)
            ##更新所有相关工人的工作时间
            air.worker_list[thisorder] = thisworkersid[:]
            air.station_id[thisorder] = self.id
            air.startingtime[thisorder] = order_stfi[0]
            air.finishtime[thisorder] = order_stfi[1]
            air.workerset[thisorder]= thisworkers
            air.workertrigger[thisorder] = []

            for i in range(worker_num):
                air.workertrigger[thisorder].append(self.env.event())

            for k in range(thisnum):

                thisworkers[k].order_buffer.append(thisorder)
                thisworkers[k].stfi[1] = order_stfi[1]
                thisworkers[k].timelist[thisorder] = dict_time[thisorder] * maxefi
                thisworkers[k].time_past += dict_time[thisorder] * maxefi
                air.orders_finish.append(thisorder)
            for order in dict_postorder[thisorder]:
                flag = True
                for preorder in dict_preorder[order]:
                    if preorder not in air.orders_finish:
                        flag = False
                        break
                if flag:
                    air.orders_free.add(order)
            air.orders_free.remove(thisorder)
            # air.isfinish[thisorder] = 1
            if air.pack_len[air.pack_id] == air.pack_cnt[air.pack_id] and air.pack_id < pack_num - 1:
                print("工序包装配完成了",air.pack_id)
                air.pack_id += 1
                air.orders_free = set(init_frees[air.pack_id][:])
            self.order_buffer.append(thisorder)
        # print(self.order_buffer)
            # if last_schedule:
            #     this_pulse = get_pulse(allstation,pulse)
            #     print("当前节拍是：", this_pulse)
            #     pulses.append(this_pulse)
            #     yield env.timeout(pulse + 2000)



    def station_process(self,env,station_list):
        ####计算每个工人的order_buffer
        schtime = self.pulse
        aircraft = yield station_list[self.id].get()


        # ###创建每个工人装配的进程
        for i in range(team_num):
            this_team = self.teams[i]
            for w in range(worker_num):
                this_worker = this_team.workers[w]
                env.process(this_worker.woker_process(aircraft))

        yield env.timeout(schtime)





    ###计算该站位的实际装配时间
    def get_realtime(self):
        time_tmp_station = 0
        for j in range(team_num):
            tmp_team = self.teams[j]
            for k in range(worker_num):
                time_tmp_station = max(time_tmp_station, tmp_team.workers[k].stfi[1])
                # print("真正统计时的finishtime",tmp_team.workers[k].finishtime)
        return time_tmp_station


#创建装配的飞机类

class SingelAircraft:
    def __init__(self,env,aircarft_id):

        self.env = env
        self.id = aircarft_id
        single_isfinish = [0 for _ in range(pro_num)]
        self.isfinish = dict(zip(pro_id, single_isfinish))
        self.startingtime = dict(zip(pro_id, single_isfinish))
        self.finishtime = dict(zip(pro_id, single_isfinish))
        self.team_id = dict(zip(pro_id, single_isfinish))
        self.station_id = dict(zip(pro_id, single_isfinish))
        self.worker_list = dict(zip(pro_id, single_isfinish))  ##工序分配给工人的号
        self.pack_cnt = [0] * pack_num  ###统计各工序包完成的情况
        self.orders_free = set(init_frees[0][:])  ##初始的无紧前工序（仅有A工序包的工序）
        self.pack_id = 0  ##初始的工序包所在处为A
        self.workerset = {}  ###每道工序被分配给了哪些人
        self.istrigger = []  # 保存了是否被触发了此工序事件，防止重复触发
        self.workertrigger = {} ##保存工人的就绪状态
        self.orders_finish = []
        self.orders_left = []
        self.isfirst = True
        self.pack_len = pack_len[:]

        for i in range(pro_num):
            self.istrigger.append(self.env.event())


'''3.产生飞机的函数'''
def generate_item(env,
                  last_q: simpy.Store,
                  item_num: int,
                  all_aircraft
                  ):

    for i in range(item_num):
        print(f'{round(env.now, 2)} - item: item_{i} - created')
        tmp_aircarft = SingelAircraft(env,i)
        last_q.put(tmp_aircarft)
        tmp = tmp_aircarft
        all_aircraft.append(tmp)
        # t = random.expovariate(1 / MEAN_TIME)
        t = 300*10
        yield env.timeout(round(t, 1))

###因为以下为主函数中采用的获取状态的函数
def get_states(env, tmpair, allstation,thispulse):
    now_time = env.now
    if now_time >= 0 and now_time < thispulse:
        station_id = 0
    elif now_time >= thispulse and now_time < 2 * thispulse:
        station_id = 1
    elif now_time >= 2 * thispulse and now_time < 3 * thispulse:
        station_id = 2
    elif now_time >= 3 * thispulse and now_time < 4 * thispulse:
        station_id = 3
    else:
        station_id = 4


    nowstation = allstation[station_id]
    # st_time = [nowstation.teams[i].finishtime for i in range(team_num)]
    for team in nowstation.teams:
        team.assembletime = max([w.stfi[1] - station_id * thispulse for w in team.workers])
    max_st_time = max([t.assembletime for t in nowstation.teams])
    state_n = []
    states = []
    ##当前工位剩余时间
    time_remain = thispulse - max_st_time
    ##当前自由工序数量
    freeo_num = len(tmpair.orders_free)
    ##当前剩余工序数量
    remaino_num = 0
    lefts = []
    for oo in pro_id:
        if tmpair.isfinish[oo] == 0:
            remaino_num += 1
            lefts.append(oo)

    ##剩余工序最长序列长度maxdepth###剩余序列平均长度avgdepth
    maxdepth = 0
    avgdepth = 0
    maxdeptime = 0
    avgdeptime = 0
    for idd in lefts:
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
    states.append(now_time / 1000)
    # states.append(max_st_time / 100)
    # states.append(time_remain / 100)
    states.append(freeo_num/100)
    states.append(remaino_num/100)
    states.append(maxdepth/10)
    states.append(avgdepth/10)
    states.append(maxdeptime / 100)
    states.append(avgdeptime / 100)
    done = True if remaino_num == 0 else False
    return states, done


def get_reward(now_time, allstation, air,thispulse):

    if now_time >= 0 and now_time < thispulse:
        station_id = 0
    elif now_time >= thispulse and now_time < 2 * thispulse:
        station_id = 1
    elif now_time >= 2 * thispulse and now_time < 3 * thispulse:
        station_id = 2
    elif now_time >= 3 * thispulse and now_time < 4 * thispulse:
        station_id = 3
    else:
        station_id = 4

    nowstation = allstation[station_id]
    # if station_id ==3:
    #     return -get_pulse(allstation)/100

    # st_time = [nowstation.teams[i].finishtime for i in range(team_num)]
    # st_time = [nowstation.time_past for i in range(station_num)]

    for team in nowstation.teams:
        team.assembletime = max([w.finishtime - station_id * thispulse for w in team.workers])
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

    return -rew  ###奖励越小越好

def get_obs(env, tmpair, allstation,thispulse):
    now_time = env.now
    if now_time >= 0 and now_time < thispulse:
        station_id = 0
    elif now_time >= thispulse and now_time < 2 * thispulse:
        station_id = 1
    elif now_time >= 2 * thispulse and now_time < 3 * thispulse:
        station_id = 2
    elif now_time >= 3 * thispulse and now_time < 4 * thispulse:
        station_id = 3
    else:
        station_id = 4

    nowstation = allstation[station_id]

    timep = []
    for team in nowstation.teams:
        assembletime = sum([w.time_past for w in team.workers]) #每个班组的实际加工时间之和
        timep.append(assembletime)

    max_timep = max(timep)  ##当前站位最大实际加工时间
    avg_timep = sum(timep) / team_num
    si = 0
    sifi = 0
    for i in range(team_num):
        si += (nowstation.teams[i].time_past1 - avg_timep) ** 2

    si = math.sqrt(si/team_num)/100
    states = []
    ##当前工位剩余时间

    ##当前自由工序数量
    remaino_num = 0
    lefts = []
    for oo in pro_id:
        if tmpair.isfinish[oo] == 0:
            remaino_num += 1
            lefts.append(oo)
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



def get_pulse(allstation):

    for station in allstation:
        for team in station.teams:
            team.assembletime = max([w.finishtime - station.id * station.pulse for w in team.workers])
        station.assembletime = max([t.assembletime for t in station.teams])
        print(f'站位{station.id}节拍是{station.assembletime}')
    pulses = [s.assembletime for s in allstation]
    si = 0
    pulse_real = max([s.assembletime for s in allstation])
    avg_st_time = sum(pulses) / station_num
    for i in range(station_num):
        si += (pulses[i] - avg_st_time) ** 2
    si = si / station_num
    return pulse_real,si

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
        env.process(allstation[i].station_process(env,station_list))  # 实际的站位处理函数

def generate_episode(agents, conf, pulses, thispulse, episode_num, SI, evaluate=False):
    env = simpy.Environment()
    o, u, r, s, o_, s_ = [], [], [], [], [], []
    au, avail_u_, u_onehot, terminate, padded = [], [], [], [], []

    allstation, station_list, air = reset_env(env, thispulse)
    reset_station(env, allstation, station_list)
    episode = {}
    times = []
    pp = 0

    def production(env, agents, air, allstation, station_list, evaluate, start_epsilon):
        # o, u, o_, r, s, s_, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], [], [], []
        episode = {}
        now_state, done = get_states(env, air, allstation, thispulse)
        last_action = np.zeros((n_agents, n_actions))
        agents.policy.init_hidden(1)
        epsilon = 0 if episode_num > int(conf.n_epochs * 0.95) else conf.start_epsilon
        # epsilon = 0 if evaluate else start_epsilon
        # print(now_state)
        pr = []
        while not done:
            obs, _ = get_states(env, air, allstation, thispulse)
            print("当前的状态是",obs)
            state, _ = get_states(env, air, allstation, thispulse)
            actions, avail_actions, actions_onehot = [], [], []

            # print("当前的obs为", obs)
            # print("当前的state为", state)
            for agent_id in range(n_agents):
                avail_action = action_set
                action = agents.choose_action(state, last_action[agent_id], agent_id, avail_action,
                                              epsilon, evaluate)
                # action = 0
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
            elif now_time >= 3 * thispulse and now_time < 4 * thispulse:
                station_id = 3
            else:
                station_id = 4
            print('此时的站位为：', station_id)
            nowstation = allstation[station_id]
            nowstation.calrule(air, actions)
            if nowstation.id < station_num - 1:
                yield env.timeout(thispulse)
            else:
                yield env.timeout(thispulse + 1000)

            next_state, done = get_states(env, air, allstation, thispulse)
            next_obs, done = get_states(env, air, allstation, thispulse)
            reward = get_reward(env.now, allstation, air, thispulse)
            # print("actions: ", actions)

            o.append([state, state])
            s.append(state)
            u.append(np.reshape(actions, [n_agents, 1]))
            u_onehot.append(actions_onehot)

            # r.append([reward])
            pr.append(reward)
            o_.append([next_state, next_state])
            s_.append(next_state)
            au.append(avail_actions)
            terminate.append([done])
            padded.append([0.])

            # step += 1
            # if self.conf.epsilon_anneal_scale == 'step':
            #     epsilon = epsilon - self.anneal_epsilon if epsilon > self.end_epsilon else epsilon
            station_list[station_id + 1].put(air)

            if nowstation.id == station_num - 1:
                this_pulse, times = get_pulse(allstation)
                pulses.append(this_pulse)
                if this_pulse < 262:
                    # agents.policy.save_model(10)
                    oid = pro_id
                    sttime = list(air.startingtime.values())
                    fitime = list(air.finishtime.values())
                    st_id = list(air.station_id.values())
                    banzu_id = list(dict_team.values())
                    worlst = list(air.worker_list.values())

                    data = [oid,sttime, fitime, st_id, banzu_id, worlst]
                    df = pd.DataFrame(list(map(list, zip(*data))), columns=['工序号','工序开始时间', '工序结束时间', "站位号", "班组号", "工人列表"])
                    df.to_excel('/home/wyl/第三章air0417/output1.xlsx', index=False)
                    SI.append(times)
                    print("平滑指数是：", times)
                    print("动作是：", actions)
                print("当前节拍是：", this_pulse)
                pp = this_pulse

                for i in range(station_num):
                    # if (i == 0):
                    #     tmp = 0.5 * pr[0] + 0.3 * pr[1] + 0.1 * pr[2] + 0.1 * pr[3]
                    # elif (i == 1):
                    #     tmp = 0.5 * pr[1] + 0.3 * pr[2] + 0.2 * pr[3]
                    # elif (i == 2):
                    #     tmp = 0.6 * pr[2] + 0.4 * pr[3]

                    tmp = 0.8 * 100 / (this_pulse - 250)

                    if this_pulse <= 270 and this_pulse > 265:
                        tmp = 0.7
                    elif this_pulse <= 265:

                        tmp = 1
                    elif this_pulse <= 280 and this_pulse > 270:
                        tmp = 0.3
                    else:
                        tmp = 0.1

                    r.append([tmp])
                print(r)
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
        return episode, pp, times

    env.process(production(env, agents, air, allstation, station_list, evaluate, conf.start_epsilon))

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
    return episode, times, pp

from agent import Agents
from utils import ReplayBuffer
from config import Config
conf = Config()
init_pulse = 280
if __name__ == '__main__':

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
            now_pulse = min(int(min(pulses) - 1), now_pulse)

        pulses = []
        for episode_idx in range(1):  ##n_eposodes=1
            episode, times, pp = generate_episode(agents, conf, pulses, now_pulse, epoch, SI)
            episodes.append(episode)

            print("当前的节拍为", now_pulse)
            all_pulse.append(now_pulse)
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

