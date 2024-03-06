import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso

# from stability_selection import RandomizedLasso
from sklearn.feature_selection import f_regression, RFECV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from minepy import MINE
from sklearn.feature_selection import mutual_info_classif


from compareperformance import run_logistic

np.random.seed(0)

# load dataset
data = pd.read_csv('/Users/cht/Desktop/IDH_HuaQIao/Huaqiao data.csv')
data = data.drop(labels=['Unnamed: 0'], axis=1)
data.info()

X, Y = data.drop(labels=['Standard 6'], axis=1), data['Standard 6']

feature_names = list(X.columns)

# Features Ranking
ranks = {}

# 将每个特征的得分缩放后,以字典形式存储
def rank_to_dict(score, feature_names, order=1):
    nd_array = (
        order * np.array([score]).T
    )  # [score]为二维行向量,但是fit_transform是以列来进行规范化的,所以需要转置
    ranks = (
        StandardScaler().fit_transform(nd_array).T[0]
    )  # T[0]返回ndarray的第一列,为1维数组;.T[0]等价于[:, 0]
    ranks = list(
        map(lambda x: round(x, 2), ranks)
    )  # 注意在python3中，map函数得到的是对象，需要用list()转化才能得到list中的数据
    return dict(zip(feature_names, ranks))


# 单变量特征选择
# 线性相关程度: 计算每个特征xi和应变量Y的相关程度;这里的f_regression通过F检验用于评估两个随机变量的线性相关性
f, pval = f_regression(X, Y, center=True)  # 注意X一定为二维ndarray(n_samples, n_features)
f = abs(f)
f[np.isinf(f)] = 0
f[np.isnan(f)] = 0
print(f)

ranks["Correlation"] = rank_to_dict(f, feature_names)

# Spearman Correlation Coefficient
from scipy.stats import spearmanr

corrs = []
for i in range(X.shape[1]):
    tau, p_value = spearmanr(X.iloc[:, i], Y)
    corrs.append(tau)
corrs = np.array(corrs)
corrs[np.isnan(corrs)] = 0
ranks["Spearman"] = rank_to_dict(corrs, feature_names)


# 最大信息系数(Maximal Information Coefficient): 计算每个特征xi和应变量Y的最大信息系数
mine = MINE()
mic_scores = []

for i in range(X.shape[1]):  # shape[0]为样本数,shape[1]为特征数

    mi = mutual_info_classif(
        np.array(X.iloc[:, i]).reshape(-1, 1), np.array(Y).reshape(-1, 1)
    )
    mi = float(mi)
    # mi = pd.Series(mi)
    # mine.compute_score(X.iloc[:,i], Y)
    # m = mine.mic()
    mic_scores.append(mi)
ranks["MIC"] = rank_to_dict(mic_scores, feature_names)
print(mic_scores)

#### 线性回归和正则化
# 回归系数: 根据线性回归的系数判断特征的重要性
# 递归特征消除(Recursive Feature Elimination): 普通线性回归(lr)实现递归特征消除
# stop the search when 5 features are left (they will get equal scores)

# l1正则: Lasso的参数
lasso = Lasso(alpha=0.001, random_state=0)
lasso.fit(X, Y)
ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), feature_names)

# l2正则: 岭回归的参数
ridge = Ridge(alpha=0.001, random_state=0)
ridge.fit(X, Y)
ridge_coef = ridge.coef_.tolist()
ridge_coef = eval(','.join(str(i) for i in ridge_coef))
ranks["Ridge"] = rank_to_dict(ridge_coef, feature_names)

# 随机森林特征选择
# 平均不纯度减少(Mean Decrease Impurity): 随机森林建树的过程中 根据不纯度选择特征的过程
rf = RandomForestClassifier(random_state=0, n_estimators=100, n_jobs=-1)
rf.fit(X, Y)
ranks["RandomForestClassifier"] = rank_to_dict(rf.feature_importances_, feature_names)

# Wrapper
# 稳定特征选择(Stability Selection)
# n): 随机lasso算法中实现稳定特征选择
# rlasso = RandomizedLasso(random_state=1234)

# rlasso.fit(X, Y)
# ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), feature_names)

# 递归特征消除(Recursive Feature Elimination)
estimator = LinearSVC(random_state=0)
selector = RFECV(estimator=estimator, cv=3, min_features_to_select=10, n_jobs=-1)
selector.fit(X, Y)
ranks["RFE"] = rank_to_dict(
    list(map(float, selector.ranking_)), feature_names, order=-1
)

final_rank = pd.DataFrame(ranks)

# 根据前面每个特征选择的方式的得到每个特征xi的平均得分
xi_mean = {}
for x_i in feature_names:
    xi_mean[x_i] = round(np.mean([ranks[method][x_i] for method in ranks.keys()]), 2)
ranks["mean_score"] = xi_mean

rank = pd.DataFrame(ranks)
features_rank = rank.sort_values(by='mean_score', ascending=False)
features_rank.to_csv('features_select.csv')

features_rank_Select = features_rank[features_rank['mean_score'] > 0]

# compare performance in train and test set

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['Standard 6'], axis=1),
    data['Standard 6'],
    test_size=0.2,
    random_state=0,
)

X_train_select = X_train[features_rank_Select.index]
X_test_select = X_test[features_rank_Select.index]

run_logistic(X_train, X_train_select, X_test, X_test_select, y_train, y_test)

selection = features_rank_Select.index
print(selection)
