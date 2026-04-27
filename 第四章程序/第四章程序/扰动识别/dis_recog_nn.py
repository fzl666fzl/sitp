#TODO:此文件搭建神经网络，输出各场景的概率(不准确)
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
import pandas
import numpy as np

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
    n_train = 18
    train = my_values[:n_train, :]
    test = my_values[n_train:, :]


    train_X, train_y = train[:, 0:7], train[:, -1:]  # 用前面的状态+性能预测后面的性能
    test_X, test_y = test[:, 0:7], test[:, -1:]

    train_X, X_max, X_min= _normalize(train_X, train = True)
    # train_y, y_max, y_min = _normalize(train_y, train=True)
    test_X, _, _ = _normalize(test_X, train = False, specified_column = None, X_max = X_max, X_min = X_min)
    # test_y, _, _ = _normalize(test_y, train = False, specified_column = None, X_max = y_max, X_min = y_min)

    tmp = np.zeros((train_y.shape[0], 3))  # 寄存器
    for i in range(train_y.shape[0]):
        if train_y[i] == 0:
            tmp[i] = [1, 0, 0]
        if train_y[i] == 1:
            tmp[i] = [0, 1, 0]
        if train_y[i] == 2:
            tmp[i] = [0, 0, 1]
    train_y = tmp[:]
    tmp1 = np.zeros((train_y.shape[0], 3))  # 寄存器
    for i in range(test_y.shape[0]):
        if test_y[i] == 0:
            tmp1[i] = [1, 0, 0]
        if test_y[i] == 1:
            tmp1[i] = [0, 1, 0]
        if test_y[i] == 2:
            tmp1[i] = [0, 0, 1]
    test_y = tmp1[:]

    print(train_y)
    print(X_min)
    # reshape input to be 3D [samples, timesteps, features]

    return train_X,train_y,test_X,test_y






if __name__=='__main__':

    path = pandas.read_excel('重调度数据收集站位1.xlsx', header=0, index_col=0)
    train_X, train_Y, test_X, test_Y = data_processing(path)

    layer_len = 8
    output_len = 3
    # 构造模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # 训练并验证模型
    model.fit(train_X, train_Y, epochs=5)
    model.evaluate(test_X, test_Y, verbose=2)

    y = model.predict(train_X)
    print(y)
