import numpy as np
# 极小型指标 -> 极大型指标
def Mintomax(datas):
    return np.max(datas) - datas


# 正向化矩阵标准化(去除量纲影响)
def Standard(datas):

    k = np.power(np.sum(pow(datas, 2), axis=0), 0.5)

    for i in range(len(k)):
        tmpki = k[i] if k[i]!=0 else 1
        datas[:, i] = datas[:, i] / tmpki

    return datas


# 计算得分并归一化
def Topsis(raw_data):
    # raw_data[:,1] = Mintomax(raw_data[:,1])
    print(raw_data)
    sta_data = Standard(raw_data)


    z_max = np.amax(sta_data, axis=0)
    z_min = np.amin(sta_data, axis=0)
    # 计算每一个样本点与最大值的距离

    tmpmaxdist = np.power(np.sum(np.power((z_max - sta_data), 2), axis=1), 0.5)
    tmpmindist = np.power(np.sum(np.power((z_min - sta_data), 2), axis=1), 0.5)

    score = []
    for i in range(len(raw_data)):
        minplusmax = tmpmaxdist[i] + tmpmindist[i] if (tmpmaxdist[i] + tmpmindist[i]) !=0 else 1
        score.append(tmpmindist[i]/minplusmax)

    print(score)
    return score

# data = [[89.0,2.0],[60.0,0.0],[74.0,1.0],[99.0,3.0]]
#
# print(Topsis(np.array(data)))
# a = np.array([1,2,3])
# b = np.array([4,5,6])
# print(np.transpose(np.vstack((a,b))))