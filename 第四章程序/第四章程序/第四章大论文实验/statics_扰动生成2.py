'''
样本生成过程：
先用仿真方法模拟扰动，收集样本，若直接右移重调度可以解决则标记为无需重调度（0）；
若超过了节拍则尝试按照原规则重分配给其他工人，若通过则标记为需要站位内重调度（1）；
若没有通过则标记为需要其他站位协助（2）；

'''

import random
import pandas as pd
import math
import copy

import numpy as np
from parameter import args_parser
import simpy


args = args_parser()

pro_num = args.pro_num
team_num = args.team_num
station_num = args.station_num
worker_num = args.worker_num
pulse = args.pulse
schedule_day = args.schtime
staticslist = []
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


class Team:
    def __init__(self, id,env,station):
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
            self.workers.append(Worker(i,env,station,self.id))

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





##当有一个工人不能装配时，将他的工序分配给别人，动态计算每一道工序的分配情况，每次都计算所有工人的优先级

class Worker:
    def __init__(self,id,env,station,profession):
        self.id = id
        self.profession = profession
        self.station = station
        self.profession = 0
        self.now_pro = 0  ###当前处理的工序id

        self.stfi = [0, 0] ##stfi[1]表示完工时间
        self.timelist = {}###每道工序的装配时间，因为考虑到与其他工人合作的情况，工序的加工时间不一定就是dicttime的时间
        self.is_wait = False ##开始时均处于不阻塞的状态

        self.time_past = 0
        self.order_buffer = [] ###工人要处理的所有的工序列表
        self.order_wait = [] ###工人还没处理的工序列表
        self.finishtime = 0
        self.efi = self.id/10+0.8 ###熟练度
        self.resource = simpy.Resource(env, capacity=1)

    def woker_process(self, aircraft,this_team,dis_time=0):
        i = 0
        while self.order_buffer:
        # for i in range(len(self.order_buffer)):
            if i == len(self.order_buffer):
                yield env.timeout(2000)  ##测试扰动3
            self.now_pro = i

            if self.is_wait == True:
                # yield env.timeout(dis_time + 0.01)
                print("受扰动工人的列表",self.order_buffer)
                # for o in self.order_buffer[self.now_pro:]:
                #     aircraft.workertrigger[o][self.id].succeed()
                yield env.timeout(dis_time)##测试扰动1
                self.is_wait = False

            order = self.order_buffer[i]

            # if self.station == 0 and this_team.id == 8 and self.id == 0:
            #     print("当前待装配",self.order_buffer)
            #     print(f'站位{self.station}工人{self.id}工序{order}时间{env.now}正在*****')
            # if self.station == 0 and this_team.id == 8 and self.id == 2:
            #     print("当前待装配",self.order_buffer)
            #     print(f'站位{self.station}工人{self.id}工序{order}时间{env.now}正在*****')
            # if self.station == 0 and this_team.id == 8 and self.id == 4:
            #     print("当前待装配",self.order_buffer)
            #     print(f'站位{self.station}工人{self.id}工序{order}时间{env.now}正在*****')
            #
            # if self.station == 0 and this_team.id == 8 and self.id == 3:
            #     print("当前待装配",self.order_buffer)
            #     print(f'站位{self.station}工人{self.id}工序{order}时间{env.now}正在*****')
            #
            # if self.station == 0 and this_team.id == 8 and self.id == 1:
            #     print("当前待装配",self.order_buffer)
            #     print(f'站位{self.station}工人{self.id}工序{order}时间{env.now}正在*****')
            #


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

            aircraft.realstartingtime[order] = round(env.now, 2)
            # yield env.timeout(self.timelist[order])
            yield env.timeout(dict_time[order])


            # if self.station == 0 and this_team.id == 8 and self.id == 2:
            #     print("当前待装配",self.order_buffer)
            #     print(f'站位{self.station}工人{self.id}工序{order}时间{env.now}专业{dict_team[order]}')
            # if self.station == 0 and this_team.id == 8 and self.id == 4:
            #     print("当前待装配",self.order_buffer)
            #
            #     print(f'站位{self.station}工人{self.id}工序{order}时间{env.now}专业{dict_team[order]}')
            #     print(aircraft.worker_list[order])
            #
            # if self.station == 0 and this_team.id == 8 and self.id == 3:
            #     print("当前待装配",self.order_buffer)
            #     print(f'站位{self.station}工人{self.id}工序{order}时间{env.now}专业{dict_team[order]}')
            #
            # if self.station == 0 and this_team.id == 8 and self.id == 1:
            #     print("当前待装配",self.order_buffer)
            #     print(f'站位{self.station}工人{self.id}工序{order}时间{env.now}专业{dict_team[order]}')
            #
            # if self.station == 0 and this_team.id == 8 and self.id == 0:
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
        self.rule = [3,0]  ##初始化的工序分配规则为时间越长越优先
        self.store = simpy.Store(env, capacity=1)
        self.action = action
        self.schcnt = 1
        self.last_schedule = False
        self.aircraft = None


        for i in range(team_num):
            self.teams.append(Team(i,env,self.id))
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

    def calrule(self,air):
        ##当前的调度时间间隔
        # print(f'站位{self.id}调度飞机{air.id}在时间{env.now}')
        for j in range(team_num):
            tmp_team = self.teams[j]
            tmp_team.islimit = False##该班组还有工人能够装配
            for k in range(worker_num):
                # tmp_team.workers[k].stfi = [self.id * pulse + schtime * (self.schcnt - 1),
                #                             self.id * pulse + schtime * (self.schcnt - 1)]
                self.teams[j].workers[k].stfi = [self.id * pulse,self.id * pulse]
                self.teams[j].workers[k].order_buffer = []
                self.teams[j].workers[k].order_wait = []

        worker_free = [i for i in range(worker_num)]
        if air.isfirst:##如果不是第一次调度，orders-free就要从left里面挑选
            air.isfirst = False

        else:
            air.orders_free = set(air.orders_left)
            air.orders_left = []
        # order_free = set(order_free)
        # print(order_free)
        this_rule = self.rule

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
            air.workerset[thisorder]= thisworkers
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



    def station_process(self,env,station_list,dis_time):
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
                    env.process(this_worker.woker_process(aircraft,this_team,dis_time))

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


#创建装配的飞机类

class SingelAircraft:
    def __init__(self,env,aircarft_id):

        self.env = env
        self.id = aircarft_id
        single_isfinish = [0 for _ in range(pro_num)]
        self.isfinish = dict(zip(pro_id, single_isfinish))
        self.startingtime = dict(zip(pro_id, single_isfinish))
        self.finishtime = dict(zip(pro_id, single_isfinish))
        self.realstartingtime = dict(zip(pro_id, single_isfinish))
        self.realfinishtime = dict(zip(pro_id, single_isfinish))
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


'''产生飞机的函数'''
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
        t = pulse*10
        yield env.timeout(round(t, 1))



'''
    扰动1：站位0班组8 工人2和4的工序列表在180时刻被清除，过了50分钟后，恢复列表，装配右移50分钟，并未超过节拍，故可直接右移重调度
'''



'''
    扰动1数据收集版
'''


def disturbance(env, allstation, aircaraft, states, dis_concretetime,cur_time,dis_team,dis_worker):


    yield env.timeout(cur_time)
    air = aircaraft[0]


    dis_station = 3
    # dis_team = random.randint(0,14)##8:150-180；13;220-280;5:20-30
    # # dis_team = 13
    # dis_worker = random.randint(0,4)

    thisteam = allstation[dis_station].teams[dis_team]
    thisworker = thisteam.workers[dis_worker]
    thisworker.is_wait = True  ##发生了阻塞

    print("工人暂时离开")
    yield env.timeout(dis_concretetime)
    print("工人回来了")
    thisworker.is_wait = False
    ##采集数据
    # 当前剩余时间

    reservetime = (dis_station + 1) * pulse - cur_time

    order_wait = thisworker.order_buffer[thisworker.now_pro:]
    thisorders = copy.deepcopy(order_wait)
    allneighbortime = 0
    if thisorders:
        is_null = 0 ##工人的装配列表不为空
        alltime = sum(dict_time[o] for o in thisorders)  ##受影响工序的时间之和
        allposttime = sum(dict_posttime[o] for o in thisorders)  ##受影响工序紧后工序时间之和

        for o in thisorders:
            if dict_workernum[o] > 1:
                for w in air.worker_list[o]:
                    ww = thisteam.workers[w]
                    allneighbortime += sum(dict_time[wt] for wt in ww.order_buffer[ww.now_pro:])

    else:
        is_null = 1
        alltime = 0
        allposttime = 0

    states.append(dis_station)
    states.append(dis_team)
    states.append(len(thisorders)) ##工人的orderbuf是否为空
    states.append(reservetime) ##当前站位的剩余时间
    states.append(alltime)
    states.append(allposttime)
    states.append(allneighbortime) ##当受影响工序中有共同加工的工人，计算其他工人的装配序列时间和
    states.append(dis_concretetime) ##扰动持续时间



    yield env.timeout(2000)


'''
    扰动2：站位0班组8 工人2 不再回来了
    这里有几个想法是不对的不要再写了；
    1.该工人剩余的工序需要分配给班组的其他人，按照最早开始时间分配。这个应该是可以的，但是计算量会很大，且涉及到递归，
     因为要随时更新每个工人的时刻表
    

    扰动2：站位0班组8 工人2 该班组所有的剩余工序需要重新分配给同班组的别的工人
    优先分配所有正在加工的工序
    这里先考虑了只需要两个工人合作的情况
'''



def disturbance2(env, allstation, airs,states,dis_time,cur_time,dis_team,dis_worker):
    yield env.timeout(cur_time)
    air = airs[0]

    dis_station = 3
    # dis_team = 8
    # dis_worker = 2  ##忙碌的工人是2号工人
    now_action = 0

    freeworkers = [0, 1, 3, 4]
    allworkers = [0, 1, 2, 3, 4]
    thisteam = allstation[dis_station].teams[dis_team]
    thisworker = thisteam.workers[dis_worker]

    # thisworker.order_buffer = []
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


    ##采集数据
    # 当前剩余时间

    reservetime = (dis_station + 1) * pulse - cur_time

    order_wait = thisworker.order_buffer[thisworker.now_pro:]
    thisorders = copy.deepcopy(order_wait)
    allneighbortime = 0
    if thisorders:
        is_null = 0 ##工人的装配列表不为空
        alltime = sum(dict_time[o] for o in thisorders)  ##受影响工序的时间之和
        allposttime = sum(dict_posttime[o] for o in thisorders)  ##受影响工序紧后工序时间之和

        for o in thisorders:
            if dict_workernum[o] > 1:
                for w in air.worker_list[o]:
                    ww = thisteam.workers[w]
                    allneighbortime += sum(dict_time[wt] for wt in ww.order_buffer[ww.now_pro:])

    else:
        is_null = 1
        alltime = 0
        allposttime = 0

    states.append(dis_station)
    states.append(dis_team)
    states.append(len(thisorders)) ##工人的orderbuf是否为空
    states.append(reservetime) ##当前站位的剩余时间
    states.append(alltime)
    states.append(allposttime)
    states.append(allneighbortime) ##当受影响工序中有共同加工的工人，计算其他工人的装配序列时间和
    states.append(dis_time) ##扰动持续时间

    now_rule = [0.4,0.4,0.2,0.1,0.2,0.3]

    rreturn = reschedule(allstation, airs, now_rule, dis_station, dis_team, dis_worker, orderbufs, fintimes, timepsts,islast=True)
    if rreturn == 2:
        states.append(2)
    else:
        states.append(0)

def reschedule(allstation, airs, rule,dis_station,dis_team,dis_worker,orderbufs,fintimes,timepsts,islast=False):#rule是一个[0,0,0,0,0,0]指定了工序选择规则与工序分配规则
    ##工人状态需要还原

    air = airs[0]

    # dis_station = 0
    # dis_team = 8
    # dis_worker = 2  ##忙碌的工人是2号工人

    thisstation = allstation[dis_station]
    thisteam = thisstation.teams[dis_team]


    #*********还原所有工人的orderbuf/finishtime/timepast*****#
    for id,w in enumerate(thisteam.workers):
        w.order_buffer = orderbufs[id][:]
        print("当前的信息是",w.order_buffer)
        w.finishtime = fintimes[id]
        w.time_past = timepsts[id]
    #**********************************************************#


    print("当前的规则是",rule)
    freeworkers = [0, 1, 3, 4]
    allworkers = [0, 1, 2, 3, 4]
    rule1 = rule[:4]
    rule2 = rule[4:]

    thisworker = thisteam.workers[dis_worker]

    # thisworker.order_buffer = []
    thisworker.is_wait = True  ##工人发生阻塞


    thisorders = set()  ##待重新分配的工序集合
    for w in allworkers:
        ww = thisteam.workers[w]
        for oss in ww.order_buffer[ww.now_pro + 1:]:
            thisorders.add(oss)
        if w != dis_worker:
            ww.order_buffer = ww.order_buffer[:ww.now_pro + 1]
    thisorders = list(thisorders)
    print(thisorders)

    tmp_worlsts = {}
    for w in thisorders:
        tmp_worlsts[w] = air.worker_list[w][:]

    for w in freeworkers:
        ww = thisteam.workers[w]
        now_p = ww.order_buffer[ww.now_pro]##各工人当前正在加工的工序
        if now_p in thisorders:  ##说明正在加工的这道工序在等待别的工人，优先分配它，否则会死锁
            if dis_worker not in tmp_worlsts[now_p]:#若受扰动的工人不在待分配的工序工人列表中，则维持原调度
                new_w = tmp_worlsts[now_p][:]
                now_w = tmp_worlsts[now_p][:]

                now_w.remove(w)
                print(now_w,now_p)
                now_w = now_w[0]
                thisteam.workers[now_w].order_buffer.append(now_p)

            else:#若待分配的工序工人列表中包含受扰动的这个人，则根据规则重新分配
                wlist = freeworkers[:]
                wlist.remove(w)
                randomw = random.choice(wlist)  # 不按规则随机选取一个工人
                # randomw = thisteam.worker_sel(rule2)#按照规则选取


                randomww = thisteam.workers[randomw]
                randomww.order_buffer.append(now_p)

                new_w = [w, randomw]  ##新分配的工人序列
                tmp_worlsts[now_p] = new_w[:]
                if islast:
                    air.worker_list[now_p] = new_w[:]

            ##将该工序从带分配列表中删除
            thisorders.remove(now_p)

            #####以下为工人状态更新
            latestfi = max([thisteam.workers[i].finishtime for i in new_w])
            maxefi = max([thisteam.workers[w].efi for w in new_w])

            order_stfi = [0, 0]

            if dict_free[now_p] == 1:  ####初始的自由工序
                order_stfi[0] = latestfi
                # order_stfi[1] = order_stfi[0] + dict_time[new_o] * maxefi
                order_stfi[1] = order_stfi[0] + dict_time[now_p]
            else:
                tmp_maxtime = 0
                for tmpfather in dict_preorder[now_p]:  # 遍历所有的父工序
                    tmp_maxtime = max(tmp_maxtime, air.realfinishtime[tmpfather])
                order_stfi[0] = max(latestfi, tmp_maxtime)
                # order_stfi[1] = order_stfi[0] + dict_time[new_o] * maxefi
                order_stfi[1] = order_stfi[0] + dict_time[now_p]
            air.finishtime[now_p] = order_stfi[1]
            for k in new_w:
                # thisworkers[k].time_past += dict_time[new_o] * maxefi
                # thisworkers[k].finishtime += dict_time[new_o] * maxefi

                thisteam.workers[k].time_past += dict_time[now_p]
                thisteam.workers[k].finishtime = order_stfi[1]



    pris = []



    # freeo = thisorders[:]
    freeo = copy.deepcopy(thisorders)
    nonfreeo = []
    tmp_orders_finish = []#建立临时已完成工序集合

    for o in thisorders:
        # print("******", thisorders)
        # print("******", o)
        for pre in dict_preorder[o]:
            if not air.isfinish[pre]:
                freeo.remove(o)
                nonfreeo.append(o)
                break
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
        team_idtime = []#将所有工人的时间排序，选出dict_workernum[new_o]前几的工人
        for worker in tmpworkers:
            pri0 = worker.finishtime / 100
            pri1 = worker.time_past / 100
            pri2 = (200 - worker.time_past) / 100
            pri = [pri0, pri0]
            pri = np.dot(pri,rule2)
            if worker.id in freeworkers:
                team_idtime.append((worker.id, pri))
        team_idtime = sorted(team_idtime, key=lambda x: x[1])
        res = [ti[0] for ti in team_idtime]
        new_w = res[:dict_workernum[new_o]]  ##new_w是新的工人序号列表

        if dict_workernum[new_o] == 5:
            return 2
        else:

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
                order_stfi[1] = order_stfi[0] + dict_time[new_o]
            else:
                tmp_maxtime = 0
                for tmpfather in dict_preorder[new_o]:  # 遍历所有的父工序
                    tmp_maxtime = max(tmp_maxtime, air.finishtime[tmpfather])
                order_stfi[0] = max(latestfi, tmp_maxtime)
                # order_stfi[1] = order_stfi[0] + dict_time[new_o] * maxefi
                order_stfi[1] = order_stfi[0] + dict_time[new_o]
            for k in range(dict_workernum[new_o]):
                # thisworkers[k].time_past += dict_time[new_o] * maxefi
                # thisworkers[k].finishtime += dict_time[new_o] * maxefi

                thisworkers[k].time_past += dict_time[new_o]
                thisworkers[k].finishtime = order_stfi[1]
            air.finishtime[new_o] = order_stfi[1]
        for w in freeworkers:
            print(thisteam.workers[w].finishtime)
            print(thisteam.workers[w].order_buffer)
    return 0







'''
    扰动3：受扰动的不止一个人，工序也不是只有两个人,但是仍属于一个专业
'''
def disturbance3(env,allstation,airs):
    yield env.timeout(25)
    air = airs[0]

    dis_station = 0
    dis_team = 5
    dis_workers = [0,1]  ##忙碌的工人是2号工人
    now_action = 0

    freeworkers = [2, 3, 4]
    allworkers = [0,1,2,3,4]
    thisteam = allstation[dis_station].teams[dis_team]
    thisworkers = []##受影响的工人们
    for d in dis_workers:
        thisworkers.append(thisteam.workers[d])
        thisteam.workers[d].is_wait = True ##工人发生阻塞



    print("工人暂时离开")
    print("工人回来了")

    ##划定待重新分配的工序集合，将其余工人的order_buffer改为只有扰动前的集合
    thisorders = set() ##待重新分配的工序集合
    for w in allworkers:
        ww = thisteam.workers[w]
        for oss in ww.order_buffer[ww.now_pro+1:]:
            thisorders.add(oss)
        if w not in dis_workers:
            ww.order_buffer = ww.order_buffer[:ww.now_pro+1]
    thisorders = list(thisorders)
    print(thisorders)

    for w in freeworkers:
        ww = thisteam.workers[w]
        now_p = ww.order_buffer[ww.now_pro]
        if now_p in thisorders: ##说明正在加工的这道工序在等待别的工人，优先分配它，否则会死锁
            wlist = freeworkers[:]
            wlist.remove(w)
            new_w = [w]
            fflag = True
            for d in dis_workers:##假如正在等待受扰动的人，就要把工序先分配给其他人
                if d in air.worker_list[now_p]:
                    fflag = False
                    randomw = random.choice(wlist)  # 这里先随机选取一个工人
                    thisorders.remove(now_p)
                    wlist.remove(randomw)

                    randomww = thisteam.workers[randomw]
                    randomww.finishtime += dict_time[now_p]
                    randomww.order_buffer.append(now_p)

                    new_w.append(randomw) ##新分配的工人序列
            air.worker_list[now_p] = new_w[:]


            if fflag:
                new_w = air.worker_list[now_p][:]
                now_w = air.worker_list[now_p][:]

                now_w.remove(w)
                for n in now_w:
                    thisteam.workers[n].order_buffer.append(now_p)



            latestfi = max([thisteam.workers[i].finishtime for i in new_w])
            maxefi = max([thisteam.workers[w].efi for w in new_w])

            order_stfi = [0, 0]

            if dict_free[now_p] == 1:  ####初始的自由工序
                order_stfi[0] = latestfi
                # order_stfi[1] = order_stfi[0] + dict_time[new_o] * maxefi
                order_stfi[1] = order_stfi[0] + dict_time[now_p]
            else:
                tmp_maxtime = 0
                for tmpfather in dict_preorder[now_p]:  # 遍历所有的父工序
                    tmp_maxtime = max(tmp_maxtime, air.finishtime[tmpfather])
                order_stfi[0] = max(latestfi, tmp_maxtime)
                # order_stfi[1] = order_stfi[0] + dict_time[new_o] * maxefi
                order_stfi[1] = order_stfi[0] + dict_time[now_p]
            for k in new_w:
                # thisworkers[k].time_past += dict_time[new_o] * maxefi
                # thisworkers[k].finishtime += dict_time[new_o] * maxefi

                thisteam.workers[k].time_past += dict_time[now_p]
                thisteam.workers[k].finishtime += dict_time[now_p]



    ##对所有的工序重排序
    pris = []
    while thisorders:
        for o in thisorders:
            pri0 = (dict_time[o] - 0.2) / (51.9 - 0.2)
            pri1 = dict_postnum[o] / 11
            pri2 = (dict_posttime[o] - 0.2) / (160.9 - 0.2)
            pri3 = (dict_postsinktime[o] - 0.2) / (79.85 - 0.2)
            pri = [pri1, pri2, pri3]
            pri = np.dot(pri, action_set[now_action])
            pris.append(pri)
        ###从小到大排序
        pris_id = []
        # for pri in range(len(pris)):
        #     pris_id.append((thisorders[pri], pris[pri]))
        act_idx = pris.index(max(pris))
        new_o = thisorders[act_idx]
        # pris_id = sorted(pris_id, key=lambda x: x[1])
        # new_o = pris_id[0][0]


        thisorders.remove(new_o)

        ##为新的工序挑选工人
        air.worker_list[new_o] = []  ##清空该工序的工人列表待重新分配

        tmpworkers = thisteam.workers[:]
        team_idtime = []
        for worker in tmpworkers:
            if worker.id in freeworkers:
                team_idtime.append((worker.id, worker.finishtime))
        team_idtime = sorted(team_idtime, key=lambda x: x[1])
        res = [ti[0] for ti in team_idtime]
        new_w = res[:dict_workernum[new_o]]  ##new_w是新的工人序号列表
        air.worker_list[new_o] = new_w[:]
        for w in new_w:
            ww = thisteam.workers[w]
            ww.finishtime += dict_time[new_o]
            ww.order_buffer.append(new_o)
            # print("当前分配的order", new_o, "当前分配给的工人是", w)

        latestfi = max([thisteam.workers[w].finishtime for w in new_w])
        maxefi = max([thisteam.workers[w].efi for w in new_w])
        thisworkers = [thisteam.workers[w] for w in new_w]
        order_stfi = [0,0]

        if dict_free[new_o] == 1:  ####初始的自由工序
            order_stfi[0] = latestfi
            # order_stfi[1] = order_stfi[0] + dict_time[new_o] * maxefi
            order_stfi[1] = order_stfi[0] + dict_time[new_o]
        else:
            tmp_maxtime = 0
            for tmpfather in dict_preorder[new_o]:  # 遍历所有的父工序
                tmp_maxtime = max(tmp_maxtime, air.finishtime[tmpfather])
            order_stfi[0] = max(latestfi, tmp_maxtime)
            # order_stfi[1] = order_stfi[0] + dict_time[new_o] * maxefi
            order_stfi[1] = order_stfi[0] + dict_time[new_o]
        if len(new_w) < dict_workernum[new_o]:
            print("没有分配成功",new_o)
        for k in range(min(dict_workernum[new_o],len(new_w))):

            # thisworkers[k].time_past += dict_time[new_o] * maxefi
            # thisworkers[k].finishtime += dict_time[new_o] * maxefi

            thisworkers[k].time_past += dict_time[new_o]
            thisworkers[k].finishtime += dict_time[new_o]
    for w in freeworkers:
        print(thisteam.workers[w].finishtime)
        print(thisteam.workers[w].order_buffer)

##预先调度一次生成数据
if __name__ == '__main__':
    all_states = []
    for i in range(50):
        ##实例化站位
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
        dis_time = random.randint(70,100) #扰动持续时间
        cur_time = random.randint(1000,1100) ##扰动出现的时间

        # dis_team = random.randint(9,16)
        dis_team = 15
        dis_worker = random.randint(0,4)
        env.process(disturbance(env, allstation, all_aircraft,states,dis_time,cur_time,dis_team,dis_worker))
        # env.process((disturbance20(env,allstation,all_aircraft)))
        for i in range(station_num):
            env.process(allstation[i].station_process(env, station_list,dis_time))  # 实际的站位处理函数
        env.run(2000)


        is_reschedule = 0 #0表示不需要重调度

        # print(allstation[station_num-1].get_realtime())
        ##判断所有工序是否完成，若存在未完成的则输出工序名
        for i in pro_id:
            if all_aircraft[0].isfinish[i] == 0:
                print("****")
                print(i)
                is_reschedule = 1 #1表示需要重调度

        if is_reschedule == 0:

            states.append(is_reschedule)
            states.append(is_reschedule)

            ##staticsworker字典保存五个工人的装配列表以及全部的工序列表
            staticstime = {}
            staticsworker = {}


            df_statisworker = pd.DataFrame(staticsworker, index=[0])

            df_start = pd.DataFrame(all_aircraft[0].startingtime, index=[0])
            df_finish = pd.DataFrame(all_aircraft[0].finishtime, index=[0])

            all_states.append(states)

        ###进行第二重判断
        else:
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
            env.process(disturbance2(env, allstation, all_aircraft, states, dis_time,cur_time,dis_team,dis_worker))

            for i in range(station_num):
                env.process(allstation[i].station_process(env, station_list, dis_time=0))  # 实际的站位处理函数
            env.run(2000)

            is_reschedule = 1  # 1表示站位内即可完成任务
            print("********")
            # print(allstation[station_num-1].get_realtime())
            ##判断所有工序是否完成，若存在未完成的则输出工序名
            for i in pro_id:
                if all_aircraft[0].isfinish[i] == 0:
                    is_reschedule = 2  # 1表示需要重调度
            print("******",states)
            if states[-1] ==2:
                is_reschedule = 2
            states.append(is_reschedule)
            all_states.append(states)

    indexes = ["扰动站位", "班组号","待装配数量", "剩余时间", "受影响工序时间和", "受影响工序紧后时间和","邻居待装配时间和", "扰动持续时间","是否重调度","是否存在满班组人员"]
    df_states = pd.DataFrame(all_states,columns=indexes)
    # df_states.to_excel("D:/研三/扰动数据说明/重调度数据收集扰动2_站位3.xlsx")
    df_states.to_excel("D:/研三/扰动数据说明/扰动识别数据重生成/站位4.xlsx")
