'''
2023.07.13
在simulation的基础上进行改写，仿真工人不在的情境，首先工人缺失，但是该工人的order_buffer较少
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
            self.workers.append(Worker(i,env,station))

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
    def __init__(self,id,env,station):
        self.id = id
        self.station = station
        self.pro_num = 0  ###已经装配的工序数
        self.busy_num = 0  #####被占用的工人数
        self.stfi = [0, 0] ##stfi[1]表示完工时间
        self.timelist = {}###每道工序的装配时间，因为考虑到与其他工人合作的情况，工序的加工时间不一定就是dicttime的时间
        self.is_wait = True ##开始时均处于不阻塞的状态

        self.time_past = 0
        self.order_buffer = [] ###工人要处理的所有的工序列表
        self.order_wait = [] ###工人还没处理的工序列表
        self.finishtime = 0
        self.efi = self.id/10+0.8 ###熟练度
        self.resource = simpy.Resource(env, capacity=1)

    def woker_process(self, aircraft,this_team):
        for i in range(len(self.order_buffer)):
            if not self.order_buffer:
                yield env.timeout(2000)
            order = self.order_buffer[i]



            if dict_free[order] != 1:
                if aircraft.isfinish[order] == 0:

                    ###等待所有的紧前工序完成
                    this_events = [aircraft.istrigger[int(k) - 1] for k in dict_preorder[order]]
                    yield simpy.events.AllOf(env, this_events)
                    if order in [352, 356, 354,352,350]:
                        print("在装配了了啦啦啦啦", order, self.id)
                ###等待所有相关的工人空闲
            worker_list = aircraft.worker_list[order]
            if order in [352, 356, 354, 352, 350]:
                print("在装配了了啦啦啦啦2", worker_list)
            ##给每道工序创建一个事件列表，只要就绪的工人就放到此列表中，列表满了就触发装配
            try:
                aircraft.workertrigger[order][self.id].succeed()
            except:
                pass

            this_events = [aircraft.workertrigger[order][k] for k in worker_list]
            yield simpy.events.AllOf(env, this_events)

            aircraft.startingtime[order] = round(env.now, 2)
            # yield env.timeout(self.timelist[order])
            yield env.timeout(dict_time[order])
            # if self.station == 0 and dict_team[order] == 8 and self.id == 2:
            #     print("当前待装配",self.order_buffer)
            #     print(f'站位{self.station}工人{self.id}工序{order}时间{env.now}专业{dict_team[order]}')
            if self.station == 0 and dict_team[order] == 8 and self.id == 4:
                print("当前待装配",self.order_wait)
                print(f'站位{self.station}工人{self.id}工序{order}时间{env.now}专业{dict_team[order]}')

            if self.station == 0 and dict_team[order] == 8 and self.id == 3:
                print("当前待装配",self.order_wait)
                print(f'站位{self.station}工人{self.id}工序{order}时间{env.now}专业{dict_team[order]}')

            if self.station == 0 and dict_team[order] == 8 and self.id == 1:
                print("当前待装配",self.order_wait)
                print(f'站位{self.station}工人{self.id}工序{order}时间{env.now}专业{dict_team[order]}')

            if self.station == 0 and dict_team[order] == 8 and self.id == 0:
                print("当前待装配",self.order_wait)
                print(f'站位{self.station}工人{self.id}工序{order}时间{env.now}专业{dict_team[order]}')

            aircraft.finishtime[order] = round(env.now, 2)
            flag = True
            self.order_wait.remove(order)
            for w in aircraft.worker_list[order]:
                if not this_team.workers[w].is_wait:##假如有一个人在受扰动状态，则跳过下面的状态更新
                    flag = False

            if flag:
                try:
                    aircraft.istrigger[order - 1].succeed()
                except:
                    pass
                aircraft.isfinish[order] = 1
                # self.time_past += self.timelist[order]
                self.time_past += dict_time[order]
                self.finishtime = round(env.now, 2) - pulse * self.station

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
        self.rule = [3,0]  #####初始化的工序分配规则为时间越长越优先
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
        schtime = 298
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
                order_stfi[1] = order_stfi[0] + dict_time[thisorder] * maxefi
            else:
                tmp_maxtime = 0
                for tmpfather in dict_preorder[thisorder]:  # 遍历所有的父工序
                    tmp_maxtime = max(tmp_maxtime, air.finishtime[tmpfather])
                order_stfi[0] = max(latestfi, tmp_maxtime)
                order_stfi[1] = order_stfi[0] + dict_time[thisorder] * maxefi

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



    def station_process(self,env,station_list):
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
                    env.process(this_worker.woker_process(aircraft,this_team))

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
def disturbance(env,allstation,air):
    yield env.timeout(180)
    ###清空某个工人的orderbuffer然后timeout一段时间后再给他加上orderbuffer
    dis_station = 0
    dis_team = 8
    dis_worker1 = 2
    dis_worker2 = 4
    thisworker1 = allstation[dis_station].teams[dis_team].workers[dis_worker1]
    thisworker2 = allstation[dis_station].teams[dis_team].workers[dis_worker2]
    tmp_orderbuffer1 = copy.deepcopy(thisworker1.order_buffer)
    tmp_orderbuffer2 = copy.deepcopy(thisworker2.order_buffer)
    thisworker1.order_buffer = []
    thisworker2.order_buffer = []

    print("工人暂时离开")
    # yield env.timeout(50)
    print("工人回来了")
    thisworker1.order_buffer = tmp_orderbuffer1
    thisworker2.order_buffer = tmp_orderbuffer2

    # print(thisworker.order_buffer)
    yield env.timeout(2000)


'''
    扰动2：站位0班组8 工人2和4 工序需要被分配给别的工人(错误的，容易死锁)
'''
def disturbance2(env,allstation,airs):
    yield env.timeout(160)
    air = airs[0]

    dis_station = 0
    dis_team = 8
    dis_worker = 2##忙碌的工人是2号工人
    now_action = 0

    freeworkers = [0, 1, 3, 4]
    thisteam = allstation[dis_station].teams[dis_team]
    thisworker = thisteam.workers[dis_worker]

    thisorders = copy.deepcopy(thisworker.order_wait)
    thisworker.order_buffer = []
    thisworker.iswait = False ##工人发生阻塞

    print("工人暂时离开")
    # yield env.timeout(50)
    print("工人回来了")

    ###训练学习器实现重调度
    pris = []
    tmp_orderbuf = []
    tmp_tp = [] ##自由工人的timepast列表
    all_workers = [i for i in range(worker_num)]
    p0 = thisworker.time_past
    p1 = sum([dict_time[order] for order in thisorders])

    for k in range(5):
        tmp_orderbuf.append([])
    #
    # for b in freeworkers:
    #     tmp_orderbuf[b] = thisteam.workers[b].order_buffer[:]
    #     tmp_tp[b] = thisteam.workers[b].time_past

    for order in thisorders:
        workerlist = air.worker_list[order] ##该工序所有的工人列表
        air.worker_list[order].remove(dis_worker)
        all_workers = [i for i in range(worker_num)]
        for w in workerlist:
            all_workers.remove(w) ##all_workers此时变为不包含已经分配的工人

        pri0s = []
        pri1s = []
        pri2s = []
        for w in all_workers:
            worker = allstation[dis_station].teams[dis_team].workers[w]
            pri0 = worker.time_past ##实际加工时间越短的工人越优先
            pri0s.append(pri0)
            pri1 = sum([dict_time[order] for order in worker.order_buffer])##工人待装配的工序时间之和越短的越优先
            pri1s.append(pri1)
            pri2 = np.sqrt(np.sum(np.square(np.array([pri0,pri1])-np.array([p0,p1]))))###和自己越接近的越优先
            pri2s.append(pri2)
            # pri = np.dot([pri0,pri1], [1,0,0])
            # pris.append(pri)
        act_idx = pri0s.index(min(pri0s))
        sel_worker = thisteam.workers[all_workers[act_idx]]
        air.worker_list[order].append(all_workers[act_idx])
        sel_worker.finishtime += dict_time[order]
        sel_worker.order_wait.append(order)
        print("当前分配的order",order,"当前分配给的工人是",all_workers[act_idx])

    all_workers = [i for i in range(worker_num)]
    all_workers.remove(dis_worker)
    for w in all_workers:
        worker = thisteam.workers[w]
        tmp_wait = copy.deepcopy(worker.order_wait)
        # tmp_wait = [] ##该工人未完成的工序先提出来
        # for o in worker.order_buffer:
        #     if air.isfinish[o] != 1:
        #         tmp_wait.append(o)
        #         worker.order_buffer.remove(o)#先把未完成的工序删除，等待排序

        pris = []
        for o in tmp_wait:
            pri0 = (dict_time[o] - 0.2) / (51.9 - 0.2)
            pri1 = dict_postnum[o] / 11
            pri2 = (dict_posttime[o] - 0.2) / (160.9 - 0.2)
            pri3 = (dict_postsinktime[o] - 0.2) / (79.85 - 0.2)
            pri = [pri1, pri2, pri3]
            pri = np.dot(pri, action_set[now_action])
            pris.append(pri)
        ###从小到大排序
        pris_id = []
        for pri in range(len(pris)):
            pris_id.append((tmp_wait[pri], pris[pri]))
        pris_id = sorted(pris_id, key=lambda x: x[1])
        new_o = []
        for p in pris_id:
            new_o.append(p[0])
        worker.order_wait = new_o
        worker.order_buffer += new_o
        print(worker.id,new_o)
        ###继续安排上装配，继续仿真，如果仿真超过了节拍，说明不能按照规则进行分配


    # print(thisworker.order_buffer)
    yield env.timeout(2000)

'''
    扰动3：站位0班组8 工人2和4 该班组所有的工人的工序
'''

def disturbance3(env,allstation,airs):
    yield env.timeout(160)
    air = airs[0]

    dis_station = 0
    dis_team = 8
    dis_worker = 2  ##忙碌的工人是2号工人
    now_action = 0

    freeworkers = [0, 1, 3, 4]
    thisteam = allstation[dis_station].teams[dis_team]
    thisworker = thisteam.workers[dis_worker]


    thisworker.order_buffer = []
    thisworker.iswait = False  ##工人发生阻塞

    print("工人暂时离开")
    # yield env.timeout(50)
    print("工人回来了")

    thisorders = set() ##待重新分配的工序集合
    for w in freeworkers:
        ww = thisteam.workers[w]
        for oss in ww.order_wait:
            thisorders.add(oss)
        ww.order_wait = []
    thisorders = list(thisorders)
    print(thisorders)


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
            ww.order_wait.append(new_o)
            print("当前分配的order", new_o, "当前分配给的工人是", w)

        latestfi = max([thisteam.workers[w].finishtime for w in new_w])
        maxefi = max([thisteam.workers[w].efi for w in new_w])
        thisworkers = [thisteam.workers[w] for w in new_w]
        order_stfi = [0,0]

        if dict_free[new_o] == 1:  ####初始的自由工序
            order_stfi[0] = latestfi
            order_stfi[1] = order_stfi[0] + dict_time[new_o] * maxefi
        else:
            tmp_maxtime = 0
            for tmpfather in dict_preorder[new_o]:  # 遍历所有的父工序
                tmp_maxtime = max(tmp_maxtime, air.finishtime[tmpfather])
            order_stfi[0] = max(latestfi, tmp_maxtime)
            order_stfi[1] = order_stfi[0] + dict_time[new_o] * maxefi

        for k in range(dict_workernum[new_o]):

            thisworkers[k].time_past += dict_time[new_o] * maxefi
            thisworkers[k].finishtime += dict_time[new_o] * maxefi
    for w in freeworkers:
        print(thisteam.workers[w].finishtime)








    # for order in new_o:
    #
    #     air.worker_list[order] = [] ##清空该工序的工人列表待重新分配
    #     all_workers = [i for i in range(worker_num)]
    #
    #
    #     tmpworkers = thisteam.workers[:]
    #     team_idtime = []
    #     for worker in tmpworkers:
    #         if worker.id in freeworkers:
    #             team_idtime.append((worker.id, worker.finishtime))
    #     team_idtime = sorted(team_idtime, key=lambda x: x[1])
    #     res = [ti[0] for ti in team_idtime]
    #     new_w = res[:dict_workernum[order]] ##new_w是新的工人序号列表
    #     air.worker_list[order] = new_w[:]
    #     for w in new_w:
    #         ww = thisteam.workers[w]
    #         ww.finishtime += dict_time[order]
    #         ww.order_wait.append(order)
    #         print("当前分配的order",order,"当前分配给的工人是",w)
    #


if __name__ == '__main__':

    ##实例化站位，保存的统计数据是是否是第一次统计，为了防止后面的飞机进入站位更新此数据
    allstation = []
    env = simpy.Environment()
    for k in range(station_num):
        allstation.append(Station(env,k,0))

    # 实例环境
    station_list = []#保存了所有的站位，每一个站位都是一个store用来存飞机实例，容量均为1，
    # #此处要小心，如果在所有站位的总体建模里面要注意,store的get和put机制会覆盖前一个对象，所以必须对站位的状态加以限制，
    #否则出现阻塞时会出现某站位没有加工完就被覆盖掉的情况
    station_init = simpy.Store(env, capacity=50)#产生飞机的站位命名为station_init
    station_list.append(station_init)

    station_time = [0 for _ in range(station_num)]
    all_aircraft = []
    station_alltime = []
    for i in range(station_num):
        tmp_workers = []
        tmpstation = simpy.Store(env, capacity=1)
        station_list.append(tmpstation)

    env.process(generate_item(env, station_init,10,all_aircraft))#产生飞机实例
    env.process(disturbance3(env,allstation,all_aircraft))
    for i in range(station_num):
        env.process(allstation[i].station_process(env,station_list))#实际的站位处理函数
    env.run(2000)

    print(allstation[station_num-1].get_realtime())
    ##判断所有工序是否完成，若存在未完成的则输出工序名
    for i in pro_id:
        if all_aircraft[0].isfinish[i]==0:
            print("****")
            print(i)
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
                if all_aircraft[0].startingtime[i]<all_aircraft[0].finishtime[j]:
                    flag = False
                    break
    print(flag)
    #将各工序的开始结束时间写入表中
    df_start = pd.DataFrame(all_aircraft[0].startingtime,index = [0])
    df_finish = pd.DataFrame(all_aircraft[0].finishtime,index = [0])
    df_start.to_excel('工序时刻表1.xlsx')
    df_finish.to_excel('工序结束时刻表1.xlsx')
