from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
# 练习的数据：
# data = np.arange(25).reshape(5, 5)
# data = pd.DataFrame(data)
# data = [[0.6875,0.725,0.7375],[0.6875,0.7875,0.8],[0.8012,0.9004,0.8746],[0.8232,0.8673,0.85]]
data = [[0.725,0.75,0.7875],[0.6999,0.8,0.8374],[0.7875,0.8125,0.8249],[0.7624, 0.8525, 0.800]]
data = pd.DataFrame(data)
ytick = ["(8,16)","(8,16,24)","(8,24,32)","(8,16,24,32)"]
xtick = ["50","80","100"]
# 绘制热度图：
# plot = sns.heatmap(data)
# sns.color_palette("magma_r", as_cmap=True)
# 绘制热度图：


# 绘制添加数值和线条的热度图：
cmap = sns.heatmap(data, linewidths=0.8, annot=True)
plt.xlabel("迭代轮次", size=14,rotation=0)
plt.ylabel("隐藏层结构", size=14, rotation=90)

plt.xticks(np.arange(0.5,len(xtick)), labels=xtick,
                     rotation=45, rotation_mode="anchor", ha="right")
plt.yticks(np.arange(0.5,len(ytick)), labels=ytick)


# 调整色带的标签：
cbar = cmap.collections[0].colorbar

cbar.ax.tick_params(labelsize=20, labelcolor="blue")
cbar.ax.set_ylabel(ylabel="color scale", size=20, color="red", loc="center")

plt.show()