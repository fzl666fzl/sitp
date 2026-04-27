'''
2023.07.03
写工人未
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


class Worker:
    def __init__(self,id,env,station):
        self.id = id
        self.station = station
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
        for order in self.order_buffer:
            if dict_free[order] != 1:
                if aircraft.isfinish[order] == 0:
                    ###等待所有的紧前工序完成
                    this_events = [aircraft.istrigger[k - 1] for k in dict_preorder[order]]
                    yield simpy.events.AllOf(env, this_events)
                ###等待所有相关的工人空闲
            workerset = aircraft.workerset[order]


            ##给每道工序创建一个事件列表，只要就绪的工人就放到此列表中，列表满了就触发装配


            wrs = []##待请求的事件列表
            # print(f'工人{self.id}工序{order}时间{env.now}')
            print([w.id for w in workerset],order)
            for w in workerset:
                r = w.resource.request()
                yield r
                wrs.append([w,r])
            # with [w.resource.request() for w in workerset] as req:

            aircraft.startingtime[order] = round(env.now, 2)
            yield env.timeout(self.timelist[order])
            print(f'站位{self.station}工人{self.id}工序{order}时间{env.now}专业{dict_team[order]}')


            aircraft.finishtime[order] = round(env.now, 2)
            try:
                aircraft.istrigger[order - 1].succeed()
            finally:
                pass
            aircraft.isfinish[order] = 1
            self.time_past += self.timelist[order]
        self.finishtime = round(env.now, 2) - pulse * self.id

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
        print(f'站位{self.id}调度飞机{air.id}在时间{env.now}')
        for j in range(team_num):
            tmp_team = self.teams[j]
            tmp_team.islimit = False##该班组还有工人能够装配
            for k in range(worker_num):
                # tmp_team.workers[k].stfi = [self.id * pulse + schtime * (self.schcnt - 1),
                #                             self.id * pulse + schtime * (self.schcnt - 1)]
                self.teams[j].workers[k].stfi = [self.id * pulse,self.id * pulse]

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
        print(self.order_buffer)
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
                    env.process(this_worker.woker_process(aircraft))

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
        self.orders_finish = []
        self.orders_left = []
        self.isfirst = True
        self.pack_len = pack_len[:]

        for i in range(pro_num):
            self.istrigger.append(self.env.event())

def assembleprocess(env,aircraft,order,allstation):
    thisstation = allstation[aircraft.station_id[order]]
    if dict_free[order] != 1:
        if aircraft.isfinish[order] == 0:
            ###等待所有的紧前工序完成
            this_events = [aircraft.istrigger[int(k) - 1] for k in dict_preorder[order]]
            yield simpy.events.AllOf(env, this_events)
        ###等待所有相关的工人空闲
    workerset = aircraft.workerset[order]

    ##给每道工序创建一个事件列表，只要就绪的工人就放到此列表中，列表满了就触发装配

    wrs = []  ##待请求的事件列表
    # print(f'工人{self.id}工序{order}时间{env.now}')
    # print([w.id for w in workerset], order)
    for w in workerset:
        r = w.resource.request()
        yield r
        wrs.append([w, r])
    # with [w.resource.request() for w in workerset] as req:

    aircraft.startingtime[order] = round(env.now, 2)
    yield env.timeout(dict_time[order])
    print(f'站位{thisstation.id}工序{order}时间{env.now}专业{dict_team[order]}')
    aircraft.finishtime[order] = round(env.now, 2)
    try:
        aircraft.istrigger[order - 1].succeed()
    finally:
        pass
    aircraft.isfinish[order] = 1






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
        t = pulse*10
        yield env.timeout(round(t, 1))

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

    # env.process(generate_item(env, station_init,10,all_aircraft))#产生飞机实例

    tmp_aircarft = SingelAircraft(env, 0)
    all_aircraft.append(tmp_aircarft)
    for i in range(station_num):
        allstation[i].calrule(all_aircraft[0])

    for i in range(1,pro_num+1):
        env.process(assembleprocess(env,all_aircraft[0],i,allstation))

    # for i in range(station_num):
    #     env.process(allstation[i].station_process(env,station_list))#实际的站位处理函数
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
    # df_start.to_excel('工序时刻表1.xlsx')
    # df_finish.to_excel('工序结束时刻表1.xlsx')

