from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import pandas
import numpy as np
from imblearn.over_sampling import SMOTE

from sklearn.neighbors import KNeighborsClassifier


def _normalize(X, train=True, specified_column=None, X_max=None, X_min=None):
    # 参数:
    #     X: 需要标准化的数据
    #     train: 处理training data时为'True'，处理testing data时为‘False'.
    #     specific_column: 数据中需要标准化的列（feature）,所有列都需要标准化时为None
    #     X_mean:数据的均值
    #     X_std: 数据的标准差
    # 结果:
    #     X: 标准化后的数据
    #     X_mean:数据的均值
    #     X_std: 数据的标准差
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_max = np.max(X[:, specified_column], 0).reshape(1, -1)
        X_min = np.min(X[:, specified_column], 0).reshape(1, -1)
    X[:, specified_column] = (X[:, specified_column] - X_min) / (X_max - X_min + 1e-8)
    # if train:
    #     X_max = np.mean(X[:, specified_column], 0).reshape(1, -1)
    #     X_min = np.std(X[:, specified_column], 0).reshape(1, -1)
    # X[:, specified_column] = (X[:, specified_column] - X_max) / (X_min + 1e-8)

    return X, X_max, X_min

def data_processing(dataset,falg=False):#数据类型是pandas
    # load dataset

    my_values = dataset.values#pandas转array
    my_values = my_values.astype('float32')

    reframed = my_values
    reframed = pandas.DataFrame(reframed)
    # drop columns we don't want to predict
    #reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    # reframed.to_excel("E:\\wangsir\\时序数据" + ".xlsx")
    #
    # split into train and test sets
    # my_values = reframed.values
    n_train = 180
    train = my_values[:n_train, :]
    test = my_values[n_train:, :]


    train_X, train_y = train[:, 0:7], train[:, -3:-2]  # 用前面的状态+性能预测后面的性能
    test_X, test_y = test[:, 0:7], test[:, -3:-2]

    train_X, X_max, X_min= _normalize(train_X, train = True)
    # train_y, y_max, y_min = _normalize(train_y, train=True)
    test_X, _, _ = _normalize(test_X, train = False, specified_column = None, X_max = X_max, X_min = X_min)
    # test_y, _, _ = _normalize(test_y, train = False, specified_column = None, X_max = y_max, X_min = y_min)

    # tmp = np.zeros((train_y.shape[0], 3))  # 寄存器
    # for i in range(train_y.shape[0]):
    #     if train_y[i] == 0:
    #         tmp[i] = [1, 0, 0]
    #     if train_y[i] == 1:
    #         tmp[i] = [0, 1, 0]
    #     if train_y[i] == 2:
    #         tmp[i] = [0, 0, 1]
    # train_y = tmp[:]
    # tmp1 = np.zeros((train_y.shape[0], 3))  # 寄存器
    # for i in range(test_y.shape[0]):
    #     if test_y[i] == 0:
    #         tmp1[i] = [1, 0, 0]
    #     if test_y[i] == 1:
    #         tmp1[i] = [0, 1, 0]
    #     if test_y[i] == 2:
    #         tmp1[i] = [0, 0, 1]
    # test_y = tmp1[:]

    # print(train_y)
    # print(X_min)
    # reshape input to be 3D [samples, timesteps, features]

    return train_X,train_y,test_X,test_y






if __name__=='__main__':

    path = pandas.read_excel('总表.xlsx', header=0, index_col=0)
    train_X, train_Y, test_X, test_Y = data_processing(path)

    # 数据均衡处理
    smote = SMOTE(random_state=1337)
    train_X, train_Y = smote.fit_resample(train_X, train_Y)

    # 创建并训练SVM分类器
    svm_clf = svm.SVC(kernel='rbf', gamma='scale', C=1.0, probability=True)
    svm_clf.fit(train_X, train_Y)

    # 进行预测
    y_pred = svm_clf.predict(train_X)

    # 评估模型性能
    accuracy = accuracy_score(train_Y, y_pred)
    print(f"Accuracy: {accuracy}")

    knn = KNeighborsClassifier(n_neighbors=3, weights='distance', p=1, n_jobs=4)
    # 邻居数量可以调整 p=1 距离度量 采用的是:曼哈顿距离 闵可夫斯基提出的 默认2是欧式距离
    # 5个邻居都说是这种类型,那他就是那种 weights权重:uniform统一,一人一票 distance 离得近权重大 n_jobs=-1满进程执行 4 4个进程
    # 邻居数量最好不要超过样本数量的开方
    knn.fit(train_X, train_Y)
    y_ = knn.predict(train_X)
    print(knn.score(train_X, train_Y))





