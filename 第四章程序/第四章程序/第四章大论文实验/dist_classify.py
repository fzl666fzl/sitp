import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_curve, auc

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 24


states = pd.read_excel("E:/研三/扰动识别数据/站位1.xlsx")
states = np.array(states)
np.random.shuffle(states)
X = states[:,:8]
y = states[:,-1]

print(X)
print(y)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)

parameters = {'splitter':('best','random')
              ,'criterion':("gini","entropy")
             }
# clf = GradientBoostingClassifier() #实例化GBDT分类模型对象
clf = DecisionTreeClassifier(random_state=25)
GS = GridSearchCV(clf, parameters, cv=10)
GS.fit(xtrain,ytrain)


# print(clf.score(xtest,ytest))
print(GS.best_score_)
# y_predict = clf.predict(xtest)

# 为了正常显示中文

