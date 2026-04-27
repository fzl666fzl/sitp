import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
df = pd.read_excel('扰动数据-甘特.xlsx')
# df = pd.read_excel('第四章实验综合验证表1.xlsx')
print(df)

# x1 = df['右移完工时间']
# x2 = df['NSGA完工时间']
# x3 = df['综合方法完工时间']
# data = [x1.values,x2.values,x3.values]

x1 = df['完工时间-原规则重调度']
x2 = df['完工时间-适应性调度']


data = [x1.values,x2.values]

plt.boxplot(data,patch_artist=False,widths=0.2,
            meanline=True,
            showmeans=True,
            meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
          )



# plt.xticks([1,2,3],['右移重调度','NSGA2重调度','扰动识别重调度'])
plt.xticks([1,2],['原规则重调度','本文方法重调度'])
plt.grid(axis='y',ls='--',alpha=0.5)
plt.ylabel('完工时间/h',fontsize=11)
# plt.ylim(0,600)
plt.tight_layout()
plt.show()