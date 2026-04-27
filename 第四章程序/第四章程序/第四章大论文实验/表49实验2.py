# -*- ecoding: utf-8 -*-
"""
@Time: 2024.04.10
@Scipt:合同网

"""

import random
import pandas as pd

from scipy.spatial.distance import euclidean
import copy
from collections import defaultdict
import numpy as np
from parameter import args_parser
import simpy

import matplotlib.pyplot as plt
import NSGA222
args = args_parser()

pro_num = args.pro_num
team_num = args.team_num
station_num = args.station_num
worker_num = args.worker_num
pulse = args.pulse
schedule_day = args.schtime
staticslist = []
action_set = [[1, 0, 0], [0.8, 0.2, 0], [0.8, 0, 0.2],
              [0.6, 0.2, 0.2], [0.5, 0.2, 0.3], [0.2, 0.5, 0.3],
              [0.2, 0, 0.8], [0.2, 0.3, 0.5], [0.2, 0.8, 0]]

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


class Team:
    def __init__(self, id, env, station):
        self.id = id  ####一共有17个班组
        self.station = station
        self.cap = 5  ###每个组有5人
        self.busy_num = 0  #####被占用的工人数
        self.workers = []
        self.islimit = False  ###team是否超时，若超时则不能再向其中添加
        self.assembletime = 0  ##装配时间
        self.timepast = 0  ##实际装配时间（不含等待时间）
        self.timelists = []
        self.singlepeak = 0  ##最多有几个工人在装配

        for i in range(worker_num):  ##初始化工人列表
            self.workers.append(Worker(i, env, station, self.id))

    def worker_sel(self, worker_free, worker_num):  # worker_free:可选的工人，worker_num:所需的人数
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


##当有一个工人不能装配时，将他的工序分配给别人，动态计算每一道工序的分配情况，每次都计算所有工人的优先级

class Worker:
    def __init__(self, id, env, station, profession):
        self.id = id
        self.profession = profession
        self.station = station
        self.profession = 0
        self.now_pro = 0  ###当前处理的工序id

        self.stfi = [0, 0]  ##stfi[1]表示完工时间
        self.timelist = {}  ###每道工序的装配时间，因为考虑到与其他工人合作的情况，工序的加工时间不一定就是dicttime的时间
        self.is_wait = False  ##开始时均处于不阻塞的状态
        self.is_pro = False

        self.time_past = 0
        self.order_buffer = []  ###工人要处理的所有的工序列表
        self.order_wait = []  ###工人还没处理的工序列表
        self.finishtime = 0
        self.efi = self.id / 10 + 0.8  ###熟练度
        self.resource = simpy.Resource(env, capacity=1)

    def woker_process(self, aircraft, this_team, dis_time=0):
        i = 0
        while self.order_buffer:
            # for i in range(len(self.order_buffer)):
            if i == len(self.order_buffer):
                yield env.timeout(2000)  ##测试扰动3
            self.now_pro = i

            # if self.is_wait == True:
            #     # yield env.timeout(dis_time + 0.01)
            #     print("受扰动工人的列表", self.order_buffer)
            #     for o in self.order_buffer[self.now_pro:]:
            #         aircraft.workertrigger[o][self.id].succeed()
            #     yield env.timeout(2000)  ##测试扰动3
            #     self.is_wait = False

            order = self.order_buffer[i]



            if dict_free[order] != 1:
                if aircraft.isfinish[order] == 0:
                    ###等待所有的紧前工序完成
                    this_events = [aircraft.istrigger[int(k) - 1] for k in dict_preorder[order]]
                    yield simpy.events.AllOf(env, this_events)

                ###等待所有相关的工人空闲
            worker_list = aircraft.worker_list[order]

            ##给每道工序创建一个事件列表，只要就绪的工人就放到此列表中，列表满了就触发装配，workerlist的问题
            try:
                aircraft.workertrigger[order][self.id].succeed()
            except:
                pass

            this_events = [aircraft.workertrigger[order][k] for k in worker_list]
            yield simpy.events.AllOf(env, this_events)

            aircraft.realstartime[order] = round(env.now, 2)
            # yield env.timeout(self.timelist[order])
            self.is_pro = True  ##工人进入加工状态
            yield env.timeout(dict_time[order])
            self.is_pro = False  ##工人进入加工状态

            # if self.station == 0 and this_team.id == 8 and self.id == 2:
            #     print("当前待装配",self.order_buffer)
            #     print(f'站位{self.station}工人{self.id}工序{order}时间{env.now}专业{dict_team[order]}')


            if env.now > (self.station + 1) * pulse:
                yield env.timeout(2000)

            # self.order_wait.remove(order)
            try:
                aircraft.istrigger[order - 1].succeed()
            except:
                pass
            aircraft.isfinish[order] = 1

            aircraft.realfinishtime[order] = round(env.now, 2)
            # self.time_past += self.timelist[order]
            self.time_past += dict_time[order]
            self.finishtime = round(env.now, 2) - pulse * self.station
            i += 1


class Station:
    def __init__(self, env, id, action):
        self.env = env
        self.id = id
        self.order_buffer = []
        self.order_finish = []
        self.time_past = 0  ##已经加工的时间
        self.time_remaining = 0  ####剩余的时间
        self.assembletime = 0
        self.teams = []
        self.rule = [3, 0]  ##初始化的工序分配规则为时间越长越优先
        self.store = simpy.Store(env, capacity=1)
        self.action = action
        self.schcnt = 1
        self.last_schedule = False
        self.aircraft = None

        for i in range(team_num):
            self.teams.append(Team(i, env, self.id))
        self.schcnt = 0

    def cal_pri(self, order_free, action):  ###计算自由工序的优先级
        pris = []
        tmp_order = []
        for free in order_free:
            tmp_order.append(free)
        # tmp_order = list(order_free)
        for free in tmp_order:
            # print(free)
            a = 1
            pri0 = (dict_time[free] - 0.2) / (51.9 - 0.2)
            pri1 = dict_postnum[free] / 11
            pri2 = (dict_posttime[free] - 0.2) / (160.9 - 0.2)
            # pri3 = (dict_postsinktime[free] - 0.2) / (79.85 - 0.2)
            # pri = [pri1, pri2, pri3]

            pri3 = (51.9-dict_time[free])/(51.9 - 0.2)
            pri = [pri0, pri1, pri2, pri3]



            pri = np.dot(pri, action)
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

    def calrule(self, air):
        ##当前的调度时间间隔
        # print(f'站位{self.id}调度飞机{air.id}在时间{env.now}')
        for j in range(team_num):
            tmp_team = self.teams[j]
            tmp_team.islimit = False  ##该班组还有工人能够装配
            for k in range(worker_num):
                # tmp_team.workers[k].stfi = [self.id * pulse + schtime * (self.schcnt - 1),
                #                             self.id * pulse + schtime * (self.schcnt - 1)]
                self.teams[j].workers[k].stfi = [self.id * pulse, self.id * pulse]
                self.teams[j].workers[k].order_buffer = []
                self.teams[j].workers[k].order_wait = []

        worker_free = [i for i in range(worker_num)]
        if air.isfirst:  ##如果不是第一次调度，orders-free就要从left里面挑选
            air.isfirst = False

        else:
            air.orders_free = set(air.orders_left)
            air.orders_left = []
        # order_free = set(order_free)
        # print(order_free)
        this_rule = self.rule

        while air.orders_free:
            thisorder = self.cal_pri(air.orders_free, [0.16608549, 0.4371956,  0.14417531, 0.10433768])
            thisnum = dict_workernum[thisorder]  ###当前工序所需的人数
            thisteam = self.teams[dict_team[thisorder]]

            order_stfi = [0, 0]

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

            if dict_free[thisorder] == 1:  ####初始的自由工序
                order_stfi[0] = latestfi
                # order_stfi[1] = order_stfi[0] + dict_time[thisorder] * maxefi
                order_stfi[1] = order_stfi[0] + dict_time[thisorder]
            else:
                tmp_maxtime = 0
                for tmpfather in dict_preorder[thisorder]:  # 遍历所有的父工序
                    tmp_maxtime = max(tmp_maxtime, air.finishtime[tmpfather])
                order_stfi[0] = max(latestfi, tmp_maxtime)
                # order_stfi[1] = order_stfi[0] + dict_time[thisorder] * maxefi
                order_stfi[1] = order_stfi[0] + dict_time[thisorder]

            # plustime = pulse if self.schcnt == 2 else schtime
            plustime = pulse
            if self.id < station_num - 1:
                if order_stfi[1] > self.id * pulse + plustime:
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
            air.workerset[thisorder] = thisworkers
            air.workertrigger[thisorder] = []

            for i in range(worker_num):
                air.workertrigger[thisorder].append(self.env.event())

            for k in range(thisnum):
                thisworkers[k].order_buffer.append(thisorder)
                thisworkers[k].order_wait.append(thisorder)
                thisworkers[k].stfi[1] = order_stfi[1]
                thisworkers[k].timelist[thisorder] = dict_time[thisorder] * maxefi
                # thisworkers[k].time_past += dict_time[thisorder] * maxefi
                thisworkers[k].time_past += dict_time[thisorder]
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
                # print("工序包装配完成了",air.pack_id)
                air.pack_id += 1
                air.orders_free = set(init_frees[air.pack_id][:])
            self.order_buffer.append(thisorder)
        # print(self.order_buffer)
        # if last_schedule:
        #     this_pulse = get_pulse(allstation,pulse)
        #     print("当前节拍是：", this_pulse)
        #     pulses.append(this_pulse)
        #     yield env.timeout(pulse + 2000)



    def station_process(self, env, station_list, dis_time):
        ####计算每个工人的order_buffer
        schtime = pulse
        while True:
            aircraft = yield station_list[self.id].get()
            self.calrule(aircraft)

            # ###创建每个工人装配的进程
            for i in range(team_num):
                this_team = self.teams[i]
                for w in range(worker_num):
                    this_worker = this_team.workers[w]
                    env.process(this_worker.woker_process(aircraft, this_team, dis_time))

            yield env.timeout(schtime)
            station_list[self.id + 1].put(aircraft)

    ###计算该站位的实际装配时间
    def get_realtime(self):
        time_tmp_station = 0
        for j in range(team_num):
            tmp_team = self.teams[j]
            for k in range(worker_num):
                time_tmp_station = max(time_tmp_station, tmp_team.workers[k].stfi[1])
                # print("真正统计时的finishtime",tmp_team.workers[k].finishtime)
        return time_tmp_station


# 创建装配的飞机类

class SingelAircraft:
    def __init__(self, env, aircarft_id):
        self.env = env
        self.id = aircarft_id
        single_isfinish = [0 for _ in range(pro_num)]
        self.isfinish = dict(zip(pro_id, single_isfinish))
        self.startingtime = dict(zip(pro_id, single_isfinish))
        self.finishtime = dict(zip(pro_id, single_isfinish))
        self.realstartime = dict(zip(pro_id, single_isfinish))
        self.realfinishtime = dict(zip(pro_id, single_isfinish))
        self.team_id = dict(zip(pro_id, single_isfinish))
        self.station_id = dict(zip(pro_id, single_isfinish))
        self.worker_list = dict(zip(pro_id, single_isfinish))  ##工序分配给工人的号
        self.pack_cnt = [0] * pack_num  ###统计各工序包完成的情况
        self.orders_free = set(init_frees[0][:])  ##初始的无紧前工序（仅有A工序包的工序）
        self.pack_id = 0  ##初始的工序包所在处为A
        self.workerset = {}  ###每道工序被分配给了哪些人
        self.istrigger = []  # 保存了是否被触发了此工序事件，防止重复触发
        self.workertrigger = {}  ##保存工人的就绪状态

        self.orders_finish = []
        self.orders_left = []
        self.isfirst = True
        self.pack_len = pack_len[:]

        for i in range(pro_num):
            self.istrigger.append(self.env.event())


'''产生飞机的函数'''


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
        t = pulse * 10
        yield env.timeout(round(t, 1))




# 定义自变量的类
class Individual(object):
    def __init__(self):
        self.solution = None  # 实际赋值中是一个 nparray 类型，方便进行四则运算
        self.objective = defaultdict()

        self.n = 0  # 解p被几个解所支配，是一个数值（左下部分点的个数）
        self.rank = 0  # 解所在第几层
        self.S = []  # 解p支配哪些解，是一个解集合（右上部分点的内容）
        self.distance = 0  # 拥挤度距离

    def bound_process(self, bound_min, bound_max):
        """
        对解向量 solution 中的每个分量进行定义域判断，超过最大值，将其赋值为最大值；小于最小值，赋值为最小值
        :param bound_min: 定义域下限
        :param bound_max:定义域上限
        :return:
        """
        for i, item in enumerate(self.solution):
            if item > bound_max:
                self.solution[i] = bound_max
            elif item < bound_min:
                self.solution[i] = bound_min

    def calculate_objective(self, allstation, airs,dis_station,dis_team,dis_worker,orderbufs,fintimes,timepsts,islast=False):
        objs = reschedule(allstation, airs, self.solution,dis_station,dis_team,dis_worker,orderbufs,fintimes,timepsts,islast)
        self.objective[1] = objs[0]
        self.objective[2] = objs[1]

    # 重载小于号“<”
    def __lt__(self, other):
        v1 = list(self.objective.values())
        v2 = list(other.objective.values())
        for i in range(len(v1)):
            if v1[i] > v2[i]:
                return 0  # 但凡有一个位置是 v1大于v2的 直接返回0,如果相等的话比较下一个目标值
        return 1

def fast_non_dominated_sort(P):
    """
    非支配排序
    :param P: 种群 P
    :return F: F=(F_1, F_2, ...) 将种群 P 分为了不同的层， 返回值类型是dict，键为层号，值为 List 类型，存放着该层的个体
    """
    F = defaultdict(list)

    for p in P:
        p.S = []
        p.n = 0
        for q in P:
            if p < q:  # if p dominate q
                p.S.append(q)  # Add q to the set of solutions dominated by p
            elif q < p:
                p.n += 1  # Increment the domination counter of p
        if p.n == 0:
            p.rank = 1
            F[1].append(p)

    i = 1
    while F[i]:
        Q = []
        for p in F[i]:
            for q in p.S:
                q.n = q.n - 1
                if q.n == 0:
                    q.rank = i + 1
                    Q.append(q)
        i = i + 1
        F[i] = Q

    return F


def crowding_distance_assignment(L):
    """ 传进来的参数应该是L = F(i)，类型是List"""
    l = len(L)  # number of solution in F

    for i in range(l):
        L[i].distance = 0  # initialize distance

    for m in L[0].objective.keys():
        L.sort(key=lambda x: x.objective[m])  # sort using each objective value
        L[0].distance = float('inf')
        L[l - 1].distance = float('inf')  # so that boundary points are always selected

        # 排序是由小到大的，所以最大值和最小值分别是 L[l-1] 和 L[0]
        f_max = L[l - 1].objective[m]
        f_min = L[0].objective[m]


        # 当某一个目标方向上的最大值和最小值相同时，此时会发生除零错，这里采用异常处理机制来解决
        try:
            for i in range(1, l - 1):  # for all other points
                L[i].distance = L[i].distance + (L[i + 1].objective[m] - L[i - 1].objective[m]) / (f_max - f_min)
        except Exception:
            print(str(m) + "目标方向上，最大值为" + str(f_max) + "最小值为" + str(f_min))


def binary_tournament(ind1, ind2):
    """
    二元锦标赛
    :param ind1:个体1号
    :param ind2: 个体2号
    :return:返回较优的个体
    """
    # if ind1.rank != ind2.rank:  # 如果两个个体有支配关系，即在两个不同的rank中，选择rank小的
    #     return ind1 if ind1.rank < ind2.rank else ind2
    #
    # elif ind1.objective[1] > 560 and ind2.objective[1] < 560:
    #     return ind2
    # elif ind2.objective[1] > 560 and ind1.objective[1] < 560:
    #     return ind1
    # elif ind1.objective[2] < ind2.objective[2]:
    #     return ind1
    # elif ind1.objective[2] > ind2.objective[2]:
    #     return ind2

    if ind1.objective[1] > 560 and ind2.objective[1] < 560:
        return ind2
    elif ind2.objective[1] > 560 and ind1.objective[1] < 560:
        return ind1
    elif ind1.objective[2] < ind2.objective[2]:
        return ind1
    elif ind1.objective[2] > ind2.objective[2]:
        return ind2

    elif ind1.distance != ind2.distance:  # 如果两个个体rank相同，比较拥挤度距离，选择拥挤读距离大的
        return ind1 if ind1.distance > ind2.distance else ind2
    else:  # 如果rank和拥挤度都相同，返回任意一个都可以
        return ind1


# TODO
def make_new_pop(P, eta, bound_min, bound_max,allstation, airs,dis_station,dis_team,dis_worker,orderbufs,fintimes,timepsts):
    """
    use select,crossover and mutation to create a new population Q
    :param P: 父代种群
    :param eta: 变异分布参数，该值越大则产生的后代个体逼近父代的概率越大。Deb建议设为 1
    :param bound_min: 定义域下限
    :param bound_max: 定义域上限
    :param objective_fun: 目标函数
    :return Q : 子代种群
    """
    popnum = len(P)
    Q = []
    # binary tournament selection
    for i in range(int(popnum / 2)):
        # 从种群中随机选择两个个体，进行二元锦标赛，选择出一个 parent1
        i = random.randint(0, popnum - 1)
        j = random.randint(0, popnum - 1)
        parent1 = binary_tournament(P[i], P[j])

        # 从种群中随机选择两个个体，进行二元锦标赛，选择出一个 parent2
        i = random.randint(0, popnum - 1)
        j = random.randint(0, popnum - 1)
        parent2 = binary_tournament(P[i], P[j])

        while (parent1.solution == parent2.solution).all():  # 如果选择到的两个父代完全一样，则重选另一个
            i = random.randint(0, popnum - 1)
            j = random.randint(0, popnum - 1)
            parent2 = binary_tournament(P[i], P[j])

        # parent1 和 parent1 进行交叉，变异 产生 2 个子代
        Two_offspring = crossover_mutation(parent1, parent2, eta, bound_min, bound_max,allstation, airs, dis_station,dis_team,dis_worker,orderbufs,fintimes,timepsts)

        # 产生的子代进入子代种群
        Q.append(Two_offspring[0])
        Q.append(Two_offspring[1])
    return Q


def crossover_mutation(parent1, parent2, eta, bound_min, bound_max,allstation, airs,dis_station,dis_team,dis_worker,orderbufs,fintimes,timepsts):
    """
    交叉方式使用二进制交叉算子（SBX），变异方式采用多项式变异（PM）
    :param parent1: 父代1
    :param parent2: 父代2
    :param eta: 变异分布参数，该值越大则产生的后代个体逼近父代的概率越大。Deb建议设为 1
    :param bound_min: 定义域下限
    :param bound_max: 定义域上限
    :param objective_fun: 目标函数
    :return: 2 个子代
    """
    poplength = len(parent1.solution)

    offspring1 = Individual()
    offspring2 = Individual()
    offspring1.solution = np.empty(poplength)
    offspring2.solution = np.empty(poplength)

    # 二进制交叉
    for i in range(poplength):
        rand = random.random()
        beta = (rand * 2) ** (1 / (eta + 1)) if rand < 0.5 else (1 / (2 * (1 - rand))) ** (1.0 / (eta + 1))
        offspring1.solution[i] = 0.5 * ((1 + beta) * parent1.solution[i] + (1 - beta) * parent2.solution[i])


        offspring2.solution[i] = 0.5 * ((1 - beta) * parent1.solution[i] + (1 + beta) * parent2.solution[i])

    offspring1.solution= offspring1.solution / sum(offspring1.solution)
    offspring2.solution = offspring2.solution/ sum(offspring2.solution)
    # 多项式变异

    for i in range(poplength):
        mu = random.random()
        delta = 2 * mu ** (1 / (eta + 1)) if mu < 0.5 else (1 - (2 * (1 - mu)) ** (1 / (eta + 1)))
        offspring1.solution[i] = offspring1.solution[i] + delta

    # 定义域越界处理
    offspring1.bound_process(bound_min, bound_max)
    offspring2.bound_process(bound_min, bound_max)

    # 计算目标函数值
    offspring1.calculate_objective(allstation, airs,dis_station,dis_team,dis_worker,orderbufs,fintimes,timepsts)
    offspring2.calculate_objective(allstation, airs,dis_station,dis_team,dis_worker,orderbufs,fintimes,timepsts)

    return [offspring1, offspring2]


def plot_P(P):
    """
    假设目标就俩,给个种群绘图
    :param P:
    :return:
    """
    X = []
    Y = []
    for ind in P:
        X.append(ind.objective[1])
        Y.append(ind.objective[2])

    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.scatter(X, Y)


def disturbance2(env, allstation, airs):
    yield env.timeout(430)##450,420,420,660
    air = airs[0]

    dis_station = 1
    dis_team = 5##13,0,5
    dis_workers = [1,2]  ##忙碌的工人是2号工人
    now_action = 0


    thisteam = allstation[dis_station].teams[dis_team]
    for dd in dis_workers:
        thisworker = thisteam.workers[dd]
        thisworker.is_wait = True  ##工人发生阻塞

    print("工人暂时离开")
    print("工人回来了")


    thisstation = allstation[dis_station]
    thisteam = thisstation.teams[dis_team]
    ##所有工人的orderbuf/finishtime/timepast
    orderbufs = []
    fintimes = []
    timepsts = []
    for w in thisteam.workers:
        orderbufs.append(w.order_buffer[:])
        fintimes.append(w.finishtime)
        timepsts.append(w.time_past)



    # NSGA-2
    generations = 20  # 迭代次数
    popnum = 50  # 种群大小
    eta = 1  # 变异分布参数，该值越大则产生的后代个体逼近父代的概率越大。Deb建议设为 1

    # poplength = 30  # 单个个体解向量的维数
    # bound_min = 0  # 定义域
    # bound_max = 1
    # objective_fun = ZDT1

    poplength = 6  # 单个个体解向量的维数
    bound_min = 0  # 定义域
    bound_max = 1







    # 生成第一代种群
    P = []
    for i in range(popnum):
        P.append(Individual())
        P[i].solution = np.random.rand(poplength) * (bound_max - bound_min) + bound_min  # 随机生成个体可行解
        P[i].bound_process(bound_min, bound_max)  # 定义域越界处理
        P[i].solution = P[i].solution/sum(P[i].solution)
        P[i].calculate_objective(allstation, airs,dis_station,dis_team,dis_workers,orderbufs,fintimes,timepsts)  # 计算目标函数值

    # 否 -> 非支配排序
    fast_non_dominated_sort(P)
    Q = make_new_pop(P, eta, bound_min, bound_max,allstation, airs, dis_station,dis_team,dis_workers,orderbufs,fintimes,timepsts)

    P_t = P  # 当前这一届的父代种群
    Q_t = Q  # 当前这一届的子代种群

    for gen_cur in range(generations):
        print("当前是第",gen_cur,"代")
        R_t = P_t + Q_t  # combine parent and offspring population
        F = fast_non_dominated_sort(R_t)

        P_n = []  # 即为P_t+1,表示下一届的父代
        i = 1
        while len(P_n) + len(F[i]) < popnum:  # until the parent population is filled
            crowding_distance_assignment(F[i])  # calculate crowding-distance in F_i
            P_n = P_n + F[i]  # include ith non dominated front in the parent pop
            i = i + 1  # check the next front for inclusion
        F[i].sort(key=lambda x: x.distance)  # sort in descending order using <n，因为本身就在同一层，所以相当于直接比拥挤距离
        P_n = P_n + F[i][:popnum - len(P_n)]
        Q_n = make_new_pop(P_n, eta, bound_min, bound_max,allstation, airs, dis_station,dis_team,dis_workers,orderbufs,fintimes,timepsts)  # use selection,crossover and mutation to create a new population Q_n

        # 求得下一届的父代和子代成为当前届的父代和子代，，进入下一次迭代 《=》 t = t + 1
        P_t = P_n
        Q_t = Q_n

        # 绘图
        plt.clf()
        plt.title('current generation:' + str(gen_cur + 1))
        plot_P(P_t)
        plt.pause(0.1)

    plt.show()


    for ind in P_t:
        if ind.objective[2] < 60 and ind.objective[1] < 570:
            print("当前的解及其性能")
            now_rule = ind.solution
        print(ind.solution)
        # break

    # now_rule = [0.16608549, 0.4371956,  0.14417531, 0.10433768, 0.14525334, 0.00295259]
    # now_rule = [1.,         0.   ,      0.     ,    0.   ,      0.10954231, 0.        ]
    # now_rule = [0.4,0.4,0.2,0.1,0.2,0.3]


    # bids = resel_station(allstation,dis_station,dis_team,dis_time)
    #
    # effis = read_bid(bids)


    reschedule(allstation, airs, now_rule, dis_station, dis_team, dis_workers, orderbufs, fintimes, timepsts,islast=True)
    # reschedule(allstation, airs, now_rule, dis_station, dis_team, dis_workers, orderbufs, fintimes, timepsts,
    #            islast=True, effis[1])

    staticsworker = {}
    for w in thisteam.workers:
        staticsworker[w.id] = " ".join(map(str, w.order_buffer[:]))
    staticsworker[5] = " ".join(map(str, staticslist[:]))
    print(staticsworker)
    df_statisworker = pd.DataFrame(staticsworker, index=[0])


    df_start = pd.DataFrame(air.startingtime, index=[0])
    df_finish = pd.DataFrame(air.finishtime, index=[0])

    df_start.to_excel('D:/研三/大论文第四章数据/工序开始时间_4910.xlsx')
    df_finish.to_excel('D:/研三/大论文第四章数据/工序结束时间_4910.xlsx')
    df_statisworker.to_excel('D:/研三/大论文第四章数据/要展示的工序扰动4910.xlsx')

#当装配过程中出现扰动时，给定相应的规则，重新将该班组的工序进行分配和排序
def reschedule(allstation, airs, rule,dis_station,dis_team,dis_workers,orderbufs,fintimes,timepsts,islast=False):#rule是一个[0,0,0,0,0,0]指定了工序选择规则与工序分配规则
    ##工人状态需要还原

    air = airs[0]

    # dis_station = 0
    # dis_team = 8
    # dis_worker = 2  ##忙碌的工人是2号工人

    thisstation = allstation[dis_station]
    thisteam = thisstation.teams[dis_team]


    virtual_starttime0 = 0
    virtual_finishtime0 = 0
    virtual_flag0 = False
    virtual_starttime1 = 0
    virtual_finishtime1 = 0
    virtual_flag1 = False


    #*********还原所有工人的orderbuf/finishtime/timepast*****#
    for id,w in enumerate(thisteam.workers):
        w.order_buffer = orderbufs[id][:]
        print("当前的信息是",w.order_buffer)
        w.finishtime = fintimes[id]
        w.time_past = timepsts[id]
    #**********************************************************#


    print("当前的规则是",rule)
    freeworkers = [0, 3, 4]
    allworkers = [0, 1, 2, 3, 4]
    rule1 = rule[:4]
    rule2 = rule[4:]




    thisorders = set()  ##待重新分配的工序集合

    for w in allworkers:
        ww = thisteam.workers[w]
        if air.realstartime[ww.now_pro] > 0:
            for oss in ww.order_buffer[ww.now_pro + 1:]:
                thisorders.add(oss)

            ww.order_buffer = ww.order_buffer[:ww.now_pro + 1]
            ww.finishtime = air.finishtime[ww.order_buffer[ww.now_pro]]
        else:
            for oss in ww.order_buffer[ww.now_pro:]:
                thisorders.add(oss)
            ww.finishtime = 430

            ww.order_buffer = ww.order_buffer[:ww.now_pro]

    thisorders = list(thisorders)
    print(thisorders)

    tmp_worlsts = {}
    for w in thisorders:
        tmp_worlsts[w] = air.worker_list[w][:]


    # for w in allworkers:
    #     ww = thisteam.workers[w]
    #
    #     now_p = ww.order_buffer[ww.now_pro]##各工人当前正在加工的工序
    #
    #     if air.realstartime[now_p] > 0:
    #         print("需要更新完工时间的工序是", now_p,air.finishtime[now_p])
    #         ww.finishtime = air.finishtime[now_p]
    #     else:
    #         ww.finishtime = 450







    pris = []
    freeo = thisorders[:]
    nonfreeo = []
    tmp_orders_finish = []#建立临时已完成工序集合

    for o in thisorders:
        for pre in dict_preorder[o]:
            if not air.isfinish[pre]:
                freeo.remove(o)
                nonfreeo.append(o)
                break
    print("当前的可装配工序", freeo)
    while freeo:
        #*************从nonfree更新freeo**********#
        for n in nonfreeo:
            ffl = True
            for pre in dict_preorder[n]:
                if pre not in tmp_orders_finish:
                    ffl = False
            if ffl:
                freeo.append(n)
                nonfreeo.remove(n)

        # *************对所有的工序重排序***********#
        pris = []
        for o in freeo:
            pri0 = (dict_time[o] - 0.2) / (51.9 - 0.2)
            pri1 = dict_postnum[o] / 11
            pri2 = (dict_posttime[o] - 0.2) / (160.9 - 0.2)
            pri3 = (dict_postsinktime[o] - 0.2) / (79.85 - 0.2)
            pri4 = (51.9-dict_time[o])/(51.9 - 0.2)
            pri = [pri1, pri2, pri3,pri4]
            pri = np.dot(pri, rule1)##按照给定的规则计算

            pris.append(pri)
        act_idx = pris.index(max(pris))
        new_o = freeo[act_idx]
        freeo.remove(new_o)

        #******************为新的工序挑选工人******************#
        pris = []
        tmp_worlsts[new_o] = []  ##清空该工序的工人列表待重新分配
        tmpworkers = thisteam.workers[:]
        team_idtime = []#将所有工人的时间排序，选出dict_workernum[new_o]前几的工人，如果该工人编号是那个扰动的工人，则降低其优先级
        for worker in tmpworkers:
            pri0 = worker.finishtime / 100
            pri1 = worker.time_past / 100
            pri2 = 2 if worker.id in dis_workers else 1
            pri = [pri0, pri2]
            pri = np.dot(pri,rule2)

            team_idtime.append((worker.id, pri))
        team_idtime = sorted(team_idtime, key=lambda x: x[1])
        res = [ti[0] for ti in team_idtime]
        new_w = res[:dict_workernum[new_o]]  ##new_w是新的工人序号列表

        ###设置标志位指示虚拟工人是是否被分配，以方便计算虚拟工人的总时长
        if dis_workers[0] in new_w:
            virtual_flag0 = True
        if dis_workers[1] in new_w:
            virtual_flag1 = True

        tmp_worlsts[new_o] = new_w[:]
        if islast:
            air.worker_list[new_o] = new_w[:]
        #************更新状态********************************#
        for w in new_w:
            ww = thisteam.workers[w]
            # ww.finishtime += dict_time[new_o]
            ww.order_buffer.append(new_o)
            # print("当前分配的order", new_o, "当前分配给的工人是", w)

        latestfi = max([thisteam.workers[w].finishtime for w in new_w])
        maxefi = max([thisteam.workers[w].efi for w in new_w])
        thisworkers = [thisteam.workers[w] for w in new_w]
        order_stfi = [0, 0]

        if dict_free[new_o] == 1:  ####初始的自由工序
            order_stfi[0] = latestfi
            # order_stfi[1] = order_stfi[0] + dict_time[new_o] * maxefi
            if dis_workers[0] in new_w or dis_workers[1] in new_w:
                order_stfi[1] = order_stfi[0] + dict_time[new_o]*1.2
            else:
                order_stfi[1] = order_stfi[0] + dict_time[new_o]

        else:
            tmp_maxtime = 0
            for tmpfather in dict_preorder[new_o]:  # 遍历所有的父工序
                tmp_maxtime = max(tmp_maxtime, air.finishtime[tmpfather])
            order_stfi[0] = max(latestfi, tmp_maxtime)
            # order_stfi[1] = order_stfi[0] + dict_time[new_o] * maxefi
            if dis_workers[0] in new_w or dis_workers[1] in new_w:
                order_stfi[1] = order_stfi[0] + dict_time[new_o]*1.2
            else:
                order_stfi[1] = order_stfi[0] + dict_time[new_o]

        print("当前的工序是",new_o,order_stfi[0])
        if virtual_flag0:
            if virtual_starttime0 == 0:
                virtual_starttime0 = order_stfi[0]
            virtual_finishtime0 = order_stfi[1]
        if virtual_flag1:
            if virtual_starttime1 == 0:
                virtual_starttime1 = order_stfi[0]
            virtual_finishtime1 = order_stfi[1]


        for k in range(dict_workernum[new_o]):
            # thisworkers[k].time_past += dict_time[new_o] * maxefi
            # thisworkers[k].finishtime += dict_time[new_o] * maxefi

            thisworkers[k].time_past += dict_time[new_o]
            thisworkers[k].finishtime = order_stfi[1]
        air.finishtime[new_o] = order_stfi[1]
        air.startingtime[new_o] = order_stfi[0]
    # for w in freeworkers:
    #     print(thisteam.workers[w].finishtime)
    #     print(thisteam.workers[w].order_buffer)


    fintime = 0
    for i in range(worker_num):
        fintime = max(fintime, thisteam.workers[i].finishtime)

    return fintime,virtual_finishtime0-virtual_starttime0+virtual_finishtime1-virtual_starttime1

'''
    查看站位状态
'''

def resel_station(allstations,dis_station,dis_team,cur_time):

    alls = [0,1,2,3,4]
    free_stations = [1,2,3,4]

    print("开始计算")
    bids = []
    for s in free_stations:

        t_station = allstations[s] #当前的站位
        t_team = t_station.teams[dis_team] #当前的班组

        ###标书< Z^0,Z,Agr,W^0,W,Effi>
        bid = []
        agr = 1
        maxlen = 2
        bidl = 0


        for w in range(worker_num):
            t_worker = t_team.workers[w]

            if t_worker.is_pro and t_worker.now_pro < len(t_worker.order_buffer) - 1:#当前工人处于加工状态，最早开工时间为当前工序的下一道的开工时间

                next_w = t_worker.order_buffer[t_worker.now_pro+1]

            else:

                if not t_worker.order_buffer:
                   if bidl < maxlen and t_worker.efi > 1:
                       bidl += 1
                       bid.append([dis_station,s,agr,1,w,t_worker.efi])
        if not bid:
            bid.append([dis_station,s,0,1,0,0])
        bids.append(bid)
    print(bids)

##标书解码
def read_bid(bids):

    b1 = []
    b2 = []
    efi_b11 = 0
    efi_b12 = 0

    efi_b21 = 0
    efi_b22 = 0


    for bb in bids:
        if bb[2]==1:
            if bb[3]==1:
                b1.append(bb)
            else:
                b2.append(bb)
    ##通过效率给标书排序
    tmpb1 = 0
    tmpind1 = 0
    for index, bb1 in enumerate(b1):
        if bb1[-1] >tmpb1:
            tmpb1 = bb1[-1]
            tmpind1 = index
    tmpbb1 = b1[tmpind1]
    tmpst = tmpbb1[1]
    tmpid = tmpbb1[4]


    tmpb2 = 0
    tmpind2 = 0
    for index, bb2 in enumerate(b2):
        if bb2[1] != tmpst and bb2[4] != tmpid:
            if bb2[-1] >tmpb2:
                tmpb2 = bb2[-1]
                tmpind2 = index
    tmpbb2 = b2[tmpind2]
    efi_b11 = tmpbb1[-1]
    efi_b12 = tmpbb2[-1]

    ##通过效率给标书排序
    tmpb2 = 0
    tmpind2 = 0
    for index, bb2 in enumerate(b2):
        if bb2[-1] > tmpb2:
            tmpb2 = bb2[-1]
            tmpind2 = index
    tmpbb2 = b2[tmpind2]
    tmpst = tmpbb2[1]
    tmpid = tmpbb2[4]

    tmpb1 = 0
    tmpind1 = 0
    for index, bb1 in enumerate(b1):
        if bb1[1] != tmpst and bb1[4] != tmpid:
            if bb1[-1] > tmpb1:
                tmpb1 = bb1[-1]
                tmpind1 = index
    tmpbb1 = b1[tmpind1]
    efi_b21 = tmpbb1[-1]
    efi_b22 = tmpbb2[-1]

    return [efi_b11,efi_b12],[efi_b21,efi_b22]








if __name__ == '__main__':
    all_states = []
    for i in range(1):
        ##实例化站位，保存的统计数据是是否是第一次统计，为了防止后面的飞机进入站位更新此数据
        allstation = []
        env = simpy.Environment()
        for k in range(station_num):
            allstation.append(Station(env, k, 0))

        # 实例环境
        station_list = []  # 保存了所有的站位，每一个站位都是一个store用来存飞机实例，容量均为1，
        # #此处要小心，如果在所有站位的总体建模里面要注意,store的get和put机制会覆盖前一个对象，所以必须对站位的状态加以限制，
        # 否则出现阻塞时会出现某站位没有加工完就被覆盖掉的情况
        station_init = simpy.Store(env, capacity=50)  # 产生飞机的站位命名为station_init
        station_list.append(station_init)

        station_time = [0 for _ in range(station_num)]
        all_aircraft = []
        station_alltime = []
        for i in range(station_num):
            tmp_workers = []
            tmpstation = simpy.Store(env, capacity=1)
            station_list.append(tmpstation)

        states = []

        env.process(generate_item(env, station_init, 10, all_aircraft))  # 产生飞机实例
        dis_time = random.randint(10, 300)

        # env.process((disturbance2(env, allstation, all_aircraft)))
        # env.process(resel_station(env,allstation,0,7))
        for i in range(station_num):
            env.process(allstation[i].station_process(env, station_list, dis_time))  # 实际的站位处理函数
        env.run(2000)

        is_reschedule = 0  # 0表示不需要重调度

        # print(allstation[station_num-1].get_realtime())
        ##判断所有工序是否完成，若存在未完成的则输出工序名
        for i in pro_id:
            if all_aircraft[0].isfinish[i] == 0:
                print("****")
                print(i)
                is_reschedule = 1  # 1表示需要重调度
        states.append(is_reschedule)

        ##判断工序的开始时间是否早于他的紧前工序结束时间，若早于则返回False
        flag = True
        for i in pro_id:
            if dict_free[i] == 1:
                break
            tmp_jinqian = dict_preorder[i]
            if type(tmp_jinqian) == type(1):
                if all_aircraft[0].startingtime[i] < all_aircraft[0].finishtime[tmp_jinqian]:
                    flag = False
                    break
            else:
                for j in tmp_jinqian:
                    if all_aircraft[0].startingtime[i] < all_aircraft[0].finishtime[j]:
                        flag = False
                        break
        print(flag)
        # 将各工序的开始结束时间写入表中

        # staticsset = set(staticslist)
        ##staticsworker字典保存五个工人的装配列表以及全部的工序列表
        staticstime = {}
        staticsworker = {}

        thisteam = allstation[3].teams[16]
        for w in thisteam.workers:
            staticsworker[w.id] = " ".join(map(str, w.order_buffer[:]))
        staticsworker[5] = " ".join(map(str, staticslist[:]))
        print(staticsworker)
        df_statisworker = pd.DataFrame(staticsworker, index=[0])

        df_start = pd.DataFrame(all_aircraft[0].realstartime, index=[0])
        df_finish = pd.DataFrame(all_aircraft[0].realfinishtime, index=[0])

        all_states.append(states)
        df_start.to_excel('D:/研三/大论文第四章数据/工序开始时间_4910.xlsx')
        df_finish.to_excel('D:/研三/大论文第四章数据/工序结束时间_4910.xlsx')
        df_statisworker.to_excel('D:/研三/大论文第四章数据/要展示的工序扰动4910.xlsx')
    indexes = ["扰动站位", "班组号", "待装配数量", "剩余时间", "受影响工序时间和", "受影响工序紧后时间和", "邻居待装配时间和", "扰动持续时间", "是否重调度"]
    # df_states = pd.DataFrame(all_states,columns=indexes)
    # df_states.to_excel("重调度数据收集站位1.xlsx")
