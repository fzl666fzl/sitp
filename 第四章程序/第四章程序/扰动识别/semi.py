import random
import numpy as np
import sklearn.svm as svm
from sklearn.datasets import make_classification
import pandas

# 样本数据的抽取
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

    np.random.shuffle(my_values)

    n_train = 90
    train = my_values[:n_train, :]
    test = my_values[n_train:, :]


    train_X, train_y = train[:, 0:7], train[:, -1:]  # 用前面的状态+性能预测后面的性能
    test_X, test_y = test[:, 0:7], test[:, -1:]

    train_X, X_max, X_min= _normalize(train_X, train = True)
    # train_y, y_max, y_min = _normalize(train_y, train=True)
    test_X, _, _ = _normalize(test_X, train = False, specified_column = None, X_max = X_max, X_min = X_min)
    # test_y, _, _ = _normalize(test_y, train = False, specified_column = None, X_max = y_max, X_min = y_min)
    print(train_y)
    print(X_min)
    # reshape input to be 3D [samples, timesteps, features]

    return train_X,train_y,test_X,test_y


path = pandas.read_excel('重调度数据收集站位1.xlsx', header=0, index_col=0)
train_X, train_y, test_X, test_y = data_processing(path)






class TSVM(object):
    '''
    半监督TSVM
    '''

    def __init__(self, kernel='linear'):
        self.Cl, self.Cu = 1.5, 0.001
        self.kernel = kernel
        self.clf = svm.SVC(C=1.5, kernel=self.kernel)

    def train(self, X1, Y1, X2):
        N = len(X1) + len(X2)
        # 样本权值初始化
        sample_weight = np.ones(N)
        sample_weight[len(X1):] = self.Cu

        # 用已标注部分训练出一个初始SVM
        self.clf.fit(X1, Y1)

        # 对未标记样本进行标记
        Y2 = self.clf.predict(X2)
        Y2 = Y2.reshape(-1, 1)

        X = np.vstack([X1, X2])
        Y = np.vstack([Y1, Y2])

        # 未标记样本的序号
        Y2_id = np.arange(len(X2))

        while self.Cu < self.Cl:
            # 重新训练SVM, 之后再寻找易出错样本不断调整
            self.clf.fit(X, Y, sample_weight=sample_weight)
            while True:
                Y2_decision = self.clf.decision_function(X2)  # 参数实例到决策超平面的距离
                Y2 = Y2.reshape(-1)
                epsilon = 1 - Y2 * Y2_decision
                negative_max_id = Y2_id[epsilon == min(epsilon)]
                # print(epsilon[negative_max_id][0])
                if epsilon[negative_max_id][0] > 0:
                    # 寻找很可能错误的未标记样本，改变它的标记成其他标记
                    pool = list(set(np.unique(Y1)) - set(Y2[negative_max_id]))
                    Y2[negative_max_id] = random.choice(pool)
                    Y2 = Y2.reshape(-1, 1)
                    Y = np.vstack([Y1, Y2])

                    self.clf.fit(X, Y, sample_weight=sample_weight)
                else:
                    break
            self.Cu = min(2 * self.Cu, self.Cl)
            sample_weight[len(X1):] = self.Cu

    def score(self, X, Y):
        return self.clf.score(X, Y)

    def predict(self, X):
        return self.clf.predict(X)


if __name__ == '__main__':
    # features, labels = make_classification(n_samples=200, n_features=3,
    #                                        n_redundant=1, n_repeated=0,
    #                                        n_informative=2, n_clusters_per_class=2)
    n_given = 50
    # 取前n_given个数字作为标注集
    X1 = train_X[:n_given]
    X2 = train_X[n_given:]
    Y1 = train_y[:n_given].reshape(-1, 1)
    Y2_labeled = train_y[n_given:].reshape(-1, 1)
    model = TSVM()
    model.train(X1, Y1, X2)
    accuracy = model.score(X2, Y2_labeled)
    print(accuracy)