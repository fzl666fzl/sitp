# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import argparse
import os




# Press the green button in the gutter to run the script.
def args_parser():
    parser = argparse.ArgumentParser()
    script_dir = os.path.dirname(__file__)
    env_data_dir = os.environ.get('AIR0417_DATA_DIR', '').strip()
    # Keep historical paths as fallbacks, but prefer local/workspace paths first.
    data_dirs = [
        env_data_dir,
        script_dir,
        os.path.normpath(os.path.join(script_dir, '..', '..', '第三章', '源数据', '数据集数据')),
        os.path.normpath(os.path.join(script_dir, '..', '..', '实验', '3', '源数据', '飞机工艺', 'Data')),
        os.path.normpath(os.path.join(script_dir, '..', '..', '..', '..', '实验', '3', '源数据', '飞机工艺', 'Data')),
        os.path.normpath(os.path.join(os.path.expanduser('~'), 'Desktop', 'sitp', '实验', '3', '源数据', '飞机工艺', 'Data')),
        r"D:/飞机工艺/Data",
        r"/home/wyl/airassemble/Data",
    ]
    data_dirs = [d for d in data_dirs if d]

    def resolve_data_path(filename):
        for base in data_dirs:
            candidate = os.path.join(base, filename)
            if os.path.exists(candidate):
                return candidate
        searched = [os.path.join(base, filename) for base in data_dirs]
        raise FileNotFoundError(
            f"未找到数据文件: {filename}，已搜索路径: {searched}"
        )
    pro_num = 3182
    pack_num = 12
    team_num = 17
    worker_num = 5
    station_num = 5
    pulse = 275
    schtime = 160##五天调度一次
    pro_id = [i + 1 for i in range(pro_num)]
    pro_nonlist = [[] for i in range(pro_num)]
    dict_pro = {}
    dict_free = {}
    dict_post = dict(zip(pro_id, pro_nonlist))
    pro_init = []
    free_orders = [[] for _ in range(pack_num)]
    paths = ['A','B','C','D','E','F','G','H','I','J','K','L']
    prf = {'A':0,'B':1,'C':2,'D':3,'E':4,
           'F':5,'G':6,'J':7,'K':8,'L':9,
           'M':10,'N':11,'Q':12,'R':13,'W':14,
           'X':15,'Y':16}##把专业变成数字
    pro_a = [i + 1 for i in range(283)]
    pro_b = [i + 284 for i in range(680 - 283)]
    pro_c = [i + 681 for i in range(2338 - 680)]
    pro_d = [i + 2339 for i in range(2520 - 2338)]
    pro_e = [i + 2521 for i in range(2526 - 2520)]
    pro_f = [i + 2527 for i in range(2721 - 2526)]
    pro_g = [i + 2722 for i in range(2836 - 2721)]
    pro_h = [i + 2837 for i in range(2966 - 2836)]
    pro_i = [i + 2967 for i in range(3032 - 2966)]
    pro_j = [i + 3033 for i in range(3144 - 3032)]
    pro_k = [i + 3145 for i in range(3169 - 3144)]
    pro_l = [i + 3170 for i in range(3182 - 3169)]

    pack_len = [len(pro_a), len(pro_b), len(pro_c), len(pro_d), len(pro_e), len(pro_f),
                len(pro_g), len(pro_h), len(pro_i), len(pro_j), len(pro_k), len(pro_l)]

    # all_data = pd.read_excel("D:/飞机工艺/Data/3182工序总表2.xlsx")
    all_data = pd.read_excel(resolve_data_path("3182工序总表2.xlsx"))
    pro_workernum = all_data['需求人数'].values.tolist()
    pro_time = all_data['加工时间/h'].values.tolist()
    pro_prf = all_data['专业'].values.tolist()###15个专业
    for i in range(len(pro_prf)):
        pro_prf[i] = prf[pro_prf[i]]

    dict_time = dict(zip(pro_id,pro_time))
    dict_workernum =dict(zip(pro_id,pro_workernum))
    dict_team = dict(zip(pro_id,pro_prf))

    for i in range(len(paths)):
        path = resolve_data_path(paths[i] + "整合.xlsx")
        # print(paths[i])
        data = pd.read_excel(path)
        data['紧前工序号'].fillna(0, inplace=True)

        pro_preorder = data['紧前工序号'].values.tolist()
        pro_thisorder = data['工序号'].values.tolist()
        n = len(data)
        for j in range(n):
            idx = pro_thisorder[j]
            if type(pro_preorder[j]) == type('ab'):
                tmp = pro_preorder[j].split(',')
                dict_free[idx] = 0
                for k in range(len(tmp)):
                    tmp[k] = int(tmp[k])
                    dict_post[tmp[k]].append(idx)

            elif pro_preorder[j] == 0:
                tmp = []
                dict_free[idx] = 1##没有紧前的工序
                free_orders[i].append(pro_thisorder[j])
            else:
                tmp = [pro_preorder[j]]
                dict_post[pro_preorder[j]].append(idx)
                dict_free[idx] = 0

            dict_pro[idx] = tmp
    # print(dict_post)
    # print(free_orders)

    dict_postnum = {}
    dict_posttime = {}
    dict_postsinktime = {}


    def find(root):
        tmps = dict_post[root]
        maxnum = 1

        if not tmps:
            return 0

        for tmp in tmps:
            tmpmaxnum = 1
            tmpmaxnum += find(tmp)
            maxnum = max(maxnum, tmpmaxnum)

        return maxnum


    def finddeptime(root):
        tmps = dict_post[root]
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
        tmps = dict_post[root]
        if not tmps:
            return dict_time[root]
        for tmp in tmps:
            tmppath.append(findsinktime(tmp))
        return max(tmppath) + dict_time[root]


    for i in range(1, pro_num + 1):
        sinktimeres = []
        # tmppath = []
        dict_postnum[i] = find(i)
        dict_posttime[i] = finddeptime(i)##所有后续工序的和
        dict_postsinktime[i] = findsinktime(i)##到sink之前的所有后续工序的和
    disvalue = pd.read_excel(resolve_data_path("扰动工序固定.xlsx"))
    dis_o = disvalue['工序'].values
    dis_tw = [0.05] * len(dis_o)
    dict_dis = dict(zip(dis_o, dis_tw))


    #
    # print(dict_postnum)
    # print(dict_posttime)
    # print(dict_postsinktime)

    # print(max(dict_time.values()))##51.9
    # print(min(dict_time.values()))##0.2
    # print(max(dict_postnum.values()))##11
    # print(min(dict_postnum.values()))##0
    # print(max(dict_posttime.values()))##160.9
    # print(min(dict_posttime.values()))##0.2
    # print(max(dict_postsinktime.values()))##79.85
    # print(min(dict_postsinktime.values()))##0.2
    # print(free_orders)
    parser.add_argument('--pro_num', type=int, default=pro_num,
                        help="总的工序数")
    parser.add_argument('--team_num', type=int, default=team_num,
                        help="总的班组数")
    parser.add_argument('--station_num', type=int, default=station_num,
                        help="总的站位数")
    parser.add_argument('--pulse', type=int, default=pulse,
                        help="预设的节拍")
    parser.add_argument('--schtime', type=int, default=schtime,
                        help="调度的时间")
    parser.add_argument('--worker_num', type=int, default=worker_num,
                        help="每个班组工人的数量")

    parser.add_argument('--free_orders', type=int, default=free_orders,
                        help="按照工序包排好的没有紧前的工序")

    parser.add_argument('--pro_id', type=list, default=pro_id,
                        help="工序号")
    parser.add_argument('--pack_len',type = int,default=pack_len,help="每个工序包的工序数量")
    parser.add_argument('--dict_free', type=int, default=dict_free,
                        help="是否没有紧前工序")
    parser.add_argument('--dict_time', type=int, default=dict_time)
    parser.add_argument('--dict_preorder', type=int, default=dict_pro,
                        help="紧前工序号")  # 调度时间
    parser.add_argument('--dict_team', type=int, default=dict_team,
                        help="所属的专业")
    parser.add_argument('--dict_workernum', type=int, default=dict_workernum,
                        help="每个工序需要的人数")
    parser.add_argument('--dict_postorder', type=int, default=dict_post,
                        help="紧后工序号")  # 调度时间

    parser.add_argument('--pro_init', type=int, default=pro_init,
                        help="初始的可行解集合")  # 调度时间
    parser.add_argument('--dict_postnum', type=int, default=dict_postnum,
                        help="紧后工序数量")  # 调度时间
    parser.add_argument('--dict_posttime', type=int, default=dict_posttime,
                        help="紧后工序时间和")  # 调度时间
    parser.add_argument('--dict_postsinktime', type=int, default=dict_postsinktime,
                        help="紧后工序到sink的时间和")
    parser.add_argument('--dis_o', type=int, default=dis_o,
                        help="紧后工序到sink的时间和")
    parser.add_argument('--dis_tw', type=int, default=dis_tw,
                        help="紧后工序到sink的时间和")
    parser.add_argument('--dict_dis', type=int, default=dict_dis,
                        help="紧后工序到sink的时间和")
    args = parser.parse_args()

    return args

args_parser()

