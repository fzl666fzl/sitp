# parameters
import argparse
import os
import pandas as pd

def args_parser():
    parser = argparse.ArgumentParser()

    # pro_num = 3182
    # team_num = 5
    # station_num = 5
    # pulse = 1000
    # pro_none = [-1] * pro_num
    #
    # all_pro = pd.read_excel('紧前关系整理版.xlsx')  ####所有的工序序号及约束,pro代表procedure工序
    # pro_preorder = all_pro['紧前工序号'].values.tolist()
    # pro_id = all_pro['工序号'].values.tolist()
    # pro_time = all_pro['作业时间'].values.tolist()
    # pro_workernum = all_pro['需求人数'].values.tolist()
    # pro_team = all_pro['专业组别'].values.tolist()
    # pro_init = []
    #



    pro_num = 50
    team_num = 3
    station_num = 4
    pulse = 800
    pro_none = [-1] * pro_num
    freeorders = [1,2,3,4]

    # all_pro = pd.read_excel('/home/wyl/dalunwen2/工序约束_50.xlsx')  ####所有的工序序号及约束,pro代表procedure工序
    data_file = os.path.join(os.path.dirname(__file__), '工序约束_50.xlsx')
    all_pro = pd.read_excel(data_file)  ####所有的工序序号及约束,pro代表procedure工序
    # all_pro = pd.read_excel('工序约束_50_2.xlsx')  ####所有的工序序号及约束,pro代表procedure工序
    # all_pro = pd.read_excel('工序约束_50_10.xlsx')  ####所有的工序序号及约束,pro代表procedure工序
    pro_preorder = all_pro['紧前工序'].values.tolist()
    pro_postorder = all_pro['紧后工序'].values.tolist()
    pro_id = all_pro['工序'].values.tolist()
    pro_time = all_pro['时间'].values.tolist()
    pro_iscrtl = {}
    pro_init = []




    ####为整理紧前关系字典做准备
    single_isfirstpro = [0 for _ in range(pro_num)]
    single_pro_team = [0 for _ in range(pro_num)]
    for i in range(pro_num):
        if type(pro_preorder[i]) == type('ab'):
            pro_preorder[i] = pro_preorder[i].split(',')
            pro_preorder[i] = list(map(int, pro_preorder[i]))
        elif pro_preorder[i] == 0:
            pro_preorder[i] = []
            pro_init.append(pro_id[i])
            single_isfirstpro[i] = 1
        else:
            pro_preorder[i] = [pro_preorder[i]]

        if type(pro_postorder[i]) == type('ab'):
            pro_postorder[i] = pro_postorder[i].split(',')
            pro_postorder[i] = list(map(int, pro_postorder[i]))
            pro_iscrtl[i+1] = 1
        elif pro_postorder[i] == 0:
            pro_postorder[i] = []
            pro_iscrtl[i + 1] = 0

        else:
            pro_postorder[i] = [pro_postorder[i]]
            pro_iscrtl[i + 1] = 1

        # single_pro_team[i] = pro_team[i].split(',')
        # single_pro_team[i] = list(map(int, single_pro_team[i]))

    dict_isfirstprocedure = dict(zip(pro_id, single_isfirstpro))
    dict_time = dict(zip(pro_id, pro_time))
    dict_preorder = dict(zip(pro_id, pro_preorder))
    dict_team = dict(zip(pro_id, single_pro_team))
    # dict_workernum = dict(zip(pro_id, pro_workernum))

    dict_postorder = dict(zip(pro_id, pro_postorder))
    # for idd, pres in dict_preorder.items():
    #     if pres:
    #         for pre in pres:
    #             if dict_postorder[pre] == -1:
    #                 dict_postorder[pre] = [idd]
    #             else:
    #                 dict_postorder[pre].append(idd)
    print(dict_preorder)
    print(dict_postorder)

    dict_postnum = {}
    dict_posttime = {}
    dict_postsinktime = {}
    def find(root):
        tmps = dict_postorder[root]
        maxnum = 1

        if not tmps:
            return 0

        for tmp in tmps:
            tmpmaxnum = 1
            tmpmaxnum += find(tmp)
            maxnum = max(maxnum, tmpmaxnum)

        return maxnum
    def finddeptime(root):
        tmps = dict_postorder[root]
        tmpmaxnum = dict_time[root]
        if not tmps:
            return dict_time[root]
        for tmp in tmps:
            tmpmaxnum += finddeptime(tmp)
        return tmpmaxnum


    def maxDepth(self, root):
        if root is None:
            return 0
        else:
            left_height = self.maxDepth(root.left)
            right_height = self.maxDepth(root.right)
            return max(left_height, right_height) + 1

    def findsinktime(root):
        tmppath = []
        tmps = dict_postorder[root]
        if not tmps:
            return dict_time[root]
        for tmp in tmps:
            tmppath.append(findsinktime(tmp))
        return max(tmppath) + dict_time[root]


    for i in range(1,pro_num+1):
        sinktimeres = []
        # tmppath = []
        dict_postnum[i] = find(i)
        dict_posttime[i] = finddeptime(i)
        dict_postsinktime[i] = findsinktime(i)


    # print(max(dict_postsinktime.values()))
    # print(min(dict_postsinktime.values()))
    # print(max(dict_posttime.values()))
    # print(min(dict_posttime.values()))

    parser.add_argument('--pro_num', type=int, default=pro_num,
                        help="总的工序数")
    parser.add_argument('--team_num', type=int, default=team_num,
                        help="总的班组数")
    parser.add_argument('--station_num', type=int, default=station_num,
                        help="总的站位数")  # 固定时间间隔投料
    parser.add_argument('--pulse', type=int, default=pulse,
                        help="总的站位数")  # 固定时间间隔投料
    parser.add_argument('--pro_id', type=list, default=pro_id,
                        help="是否没有紧前工序")  # 调度时间
    parser.add_argument('--dict_isfirstprocedure', type=int, default=dict_isfirstprocedure,
                        help="是否没有紧前工序")  # 调度时间
    parser.add_argument('--dict_time', type=int, default=dict_time,
                        help="The time to charge the rules: performance_point")  # 调度时间
    parser.add_argument('--dict_preorder', type=int, default=dict_preorder,
                        help="The time to charge the rules: performance_point")  # 调度时间
    parser.add_argument('--dict_team', type=int, default=dict_team,
                        help="The time to charge the rules: performance_point")  # 调度时间
    # parser.add_argument('--dict_workernum', type=int, default=dict_workernum,
    #                     help="The time to charge the rules: performance_point")  # 调度时间
    parser.add_argument('--dict_postorder', type=int, default=dict_postorder,
                        help="The time to charge the rules: performance_point")  # 调度时间

    parser.add_argument('--pro_init', type=int, default=pro_init,
                        help="初始的可行解集合")  # 调度时间
    parser.add_argument('--dict_postnum', type=int, default=dict_postnum,
                        help="初始的可行解集合")  # 调度时间
    parser.add_argument('--dict_posttime', type=int, default=dict_posttime,
                        help="初始的可行解集合")  # 调度时间
    parser.add_argument('--dict_postsinktime',type = int,default=dict_postsinktime)
    parser.add_argument('--pro_iscrtl', type=int, default=pro_iscrtl)
    parser.add_argument('--freeorders', type=int, default=freeorders)
    print(pro_iscrtl)
    args = parser.parse_args()






    return args

args_parser()