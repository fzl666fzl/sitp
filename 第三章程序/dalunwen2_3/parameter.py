# parameters
import argparse
import os
import pandas as pd


def args_parser():
    parser = argparse.ArgumentParser()

    pro_num = 100
    team_num = 3
    station_num = 5
    pulse = 800
    pro_none = [-1] * pro_num
    freeorders = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    data_file = os.environ.get('DALUNWEN_DATA_FILE')
    if not data_file:
        data_file = os.path.join(os.path.dirname(__file__), '工序约束_10010.xlsx')
    elif not os.path.isabs(data_file):
        data_file = os.path.join(os.path.dirname(__file__), data_file)
    all_pro = pd.read_excel(data_file)
    pro_preorder = all_pro['紧前工序'].values.tolist()
    pro_postorder = all_pro['紧后工序'].values.tolist()
    pro_id = all_pro['工序'].values.tolist()
    pro_time = all_pro['时间'].values.tolist()
    pro_iscrtl = {}
    pro_init = []

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
            pro_iscrtl[i + 1] = 1
        elif pro_postorder[i] == 0:
            pro_postorder[i] = []
            pro_iscrtl[i + 1] = 0
        else:
            pro_postorder[i] = [pro_postorder[i]]
            pro_iscrtl[i + 1] = 1

    dict_isfirstprocedure = dict(zip(pro_id, single_isfirstpro))
    dict_time = dict(zip(pro_id, pro_time))
    dict_preorder = dict(zip(pro_id, pro_preorder))
    dict_team = dict(zip(pro_id, single_pro_team))
    dict_postorder = dict(zip(pro_id, pro_postorder))

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

    def findsinktime(root):
        tmps = dict_postorder[root]
        if not tmps:
            return dict_time[root]
        tmppath = []
        for tmp in tmps:
            tmppath.append(findsinktime(tmp))
        return max(tmppath) + dict_time[root]

    for i in range(1, pro_num + 1):
        dict_postnum[i] = find(i)
        dict_posttime[i] = finddeptime(i)
        dict_postsinktime[i] = findsinktime(i)

    parser.add_argument('--pro_num', type=int, default=pro_num, help='total procedure count')
    parser.add_argument('--team_num', type=int, default=team_num, help='total team count')
    parser.add_argument('--station_num', type=int, default=station_num, help='total station count')
    parser.add_argument('--pulse', type=int, default=pulse, help='station pulse')
    parser.add_argument('--pro_id', type=list, default=pro_id, help='procedure ids')
    parser.add_argument('--dict_isfirstprocedure', type=int, default=dict_isfirstprocedure, help='whether the procedure has no predecessor')
    parser.add_argument('--dict_time', type=int, default=dict_time, help='processing time')
    parser.add_argument('--dict_preorder', type=int, default=dict_preorder, help='predecessor map')
    parser.add_argument('--dict_team', type=int, default=dict_team, help='team map')
    parser.add_argument('--dict_postorder', type=int, default=dict_postorder, help='successor map')
    parser.add_argument('--pro_init', type=int, default=pro_init, help='initial feasible set')
    parser.add_argument('--dict_postnum', type=int, default=dict_postnum, help='post-order count')
    parser.add_argument('--dict_posttime', type=int, default=dict_posttime, help='post-order time sum')
    parser.add_argument('--dict_postsinktime', type=int, default=dict_postsinktime)
    parser.add_argument('--pro_iscrtl', type=int, default=pro_iscrtl)
    parser.add_argument('--freeorders', type=int, default=freeorders)

    print(pro_iscrtl)
    args = parser.parse_args()
    return args


args_parser()
