'''
2023.07.28
绘制受扰动的情况
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
#
st_id =[i+1 for i in range(3182)]
worker_num = 5
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728','#d62728',"#FFC0CB"]

procedure_start = pd.read_excel('D:/研三/扰动数据说明/工序开始时间_nsga222.xlsx',index_col=0,header=0)
procedure_start = procedure_start.iloc[0].values.tolist()

dict_st = dict(zip(st_id,procedure_start))
print(dict_st)
procedure_finish = pd.read_excel('D:/研三/扰动数据说明/工序结束时间_nsga222.xlsx',index_col=0,header=0)
procedure_finish = procedure_finish.iloc[0].values.tolist()
dict_fi = dict(zip(st_id,procedure_finish))

dis_pro = pd.read_excel('D:/研三/扰动数据说明/要展示的工序扰动nsga222.xlsx',index_col=0,header=0)
print(dis_pro)
dict_order_buffer = {}
for i in range(worker_num):
    tmp = dis_pro[i].values.tolist()[0]
    tmplist = list(map(int,tmp.split(' ')))
    dict_order_buffer[i] = tmplist[:]





procedure_start = pd.read_excel('D:/研三/扰动数据说明/工序开始时间_nsga0.xlsx',index_col=0,header=0)
procedure_start = procedure_start.iloc[0].values.tolist()

dict_st = dict(zip(st_id,procedure_start))
print(dict_st)
procedure_finish = pd.read_excel('D:/研三/扰动数据说明/工序结束时间_nsga0.xlsx',index_col=0,header=0)
procedure_finish = procedure_finish.iloc[0].values.tolist()
dict_fi = dict(zip(st_id,procedure_finish))

dis_pro = pd.read_excel('D:/研三/扰动数据说明/要展示的工序扰动nsga0.xlsx',index_col=0,header=0)
print(dis_pro)
dict_order_buffer2 = {}
for i in range(worker_num):
    tmp = dis_pro[i].values.tolist()[0]
    tmplist = list(map(int,tmp.split(' ')))
    dict_order_buffer2[i] = tmplist[:]



def gatt(dict_pro1,dict_pro2):
    """甘特图
    m机器集
    t时间集
    """
    for wid,pro in dict_pro1.items():#w为工人id，p为该工人order_buf
        for p in pro:
            plt.barh(y=wid+0.5,width=dict_fi[p]-dict_st[p],left=dict_st[p], edgecolor='w',height=0.2,color=colors[wid]
            # plt.text(x=(dict_fi[p]+dict_st[p])/2,y=wid+1,s=str(p)+'\n',ha='center',
            #          fontdict=dict(fontsize=10)
                     )

    for wid,pro in dict_pro2.items():#w为工人id，p为该工人order_buf
        for p in pro:
            plt.barh(y=wid,width=dict_fi[p]-dict_st[p],left=dict_st[p], edgecolor='w',height=0.2,color=colors[wid]
            # plt.text(x=(dict_fi[p]+dict_st[p])/2,y=wid+1,s=str(p)+'\n',ha='center',
            #          fontdict=dict(fontsize=10)
                     )

    plt.ylabel('工人编号')
    plt.xlabel('加工时刻/h')
    plt.yticks([i+1 for i in range(worker_num)])
    plt.xlim(0,200)
    plt.rcParams['axes.facecolor'] = 'lightsteelblue'

    plt.show()


if __name__=="__main__":
    """测试代码"""

    gatt(dict_order_buffer,dict_order_buffer2)



