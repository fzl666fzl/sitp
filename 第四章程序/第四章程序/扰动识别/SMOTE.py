# 首先，导入所需的库
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

# 生成一个具有样本不均衡的分类数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
                           n_classes=2, weights=[0.05, 0.95], random_state=1337)

# 使用 SMOTE 类对数据进行自动重采样
smote = SMOTE(random_state=1337)
X_resampled, y_resampled = smote.fit_resample(X, y)


