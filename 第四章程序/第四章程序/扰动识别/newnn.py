#TODO:此文件为预测扰动场景的文件，输出为各扰动场景的概率（准确）
#   date：2023.10.17
import tensorflow as tf
import pandas
import numpy as np
import matplotlib as plt
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
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

    n_train = 180
    train = my_values[:n_train, :]
    test = my_values[n_train:, :]


    train_X, train_y = train[:, 0:7], train[:, -3:-2]  # 用前面的状态+性能预测后面的性能
    test_X, test_y = test[:, 0:7], test[:, -3:-2]

    train_X, X_max, X_min= _normalize(train_X, train = True)
    # train_y, y_max, y_min = _normalize(train_y, train=True)
    test_X, _, _ = _normalize(test_X, train = False, specified_column = None, X_max = X_max, X_min = X_min)
    # test_y, _, _ = _normalize(test_y, train = False, specified_column = None, X_max = y_max, X_min = y_min)
    # print(train_y)
    # print(X_min)
    # reshape input to be 3D [samples, timesteps, features]

    return train_X,train_y,test_X,test_y


path = pandas.read_excel('总表.xlsx', header=0, index_col=0)
train_X, train_y, test_X, test_y = data_processing(path)
print(test_X)
print(test_y)
# 使用Softmax解决多分类问题

# smote = SMOTE(random_state=1337)
# train_X, train_y = smote.fit_resample(train_X, train_y)
# test_X, test_y = smote.fit_resample(test_X, test_y)
#onehot编码
train_label_onehot = tf.keras.utils.to_categorical(train_y) #独热编码，隶属于自己类别为1，其他为0
test_label_onehot = tf.keras.utils.to_categorical(test_y)

#建立模型
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(24, activation='relu'))
# model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='linear'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              # optimizer = 'rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy']
)

#模型训练
history1 = model.fit(train_X, train_label_onehot, epochs=400,validation_data=(test_X, test_label_onehot),batch_size=32)

# #数据均衡处理
# smote = SMOTE(random_state=1337)
# train_X, train_y = smote.fit_resample(train_X, train_y)
# #onehot编码
# train_label_onehot = tf.keras.utils.to_categorical(train_y) #独热编码，隶属于自己类别为1，其他为0
# test_label_onehot = tf.keras.utils.to_categorical(test_y)
# #建立模型
# model = tf.keras.Sequential()
#
# model.add(tf.keras.layers.Dense(8, activation='relu'))
# model.add(tf.keras.layers.Dense(24, activation='relu'))
# model.add(tf.keras.layers.Dense(32, activation='linear'))
# model.add(tf.keras.layers.Dense(3, activation='softmax'))
#
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy']
# )

# #模型训练
# history2 = model.fit(train_X, train_label_onehot, epochs=64,validation_data=(test_X, test_label_onehot))
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
#
plt.plot(history1.history['accuracy'],label='测试集' )
# plt.plot(history1.history['val_accuracy'], label='验证集')

# 定义移动平均窗口大小
window_size = 3

# 计算简单移动平均
sma = np.convolve(history1.history['val_accuracy'], np.ones(window_size) / window_size, mode='valid')
plt.plot(np.arange(window_size - 1, 400), sma, label="验证集")

plt.legend()
plt.xlabel("迭代次数")
plt.ylabel("准确率")
plt.show()


# model.save('扰动分类模型.h5')