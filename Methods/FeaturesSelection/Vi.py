import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.feature_selection import RFE, f_regression, RFECV
from sklearn.svm import SVC
from sklearn.svm import SVR, LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from minepy import MINE

import matplotlib.pyplot as plt
import seaborn as sns
import palettable

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from yellowbrick.features import FeatureImportances

from FeatureSelectwithML import selection

np.random.seed(1234)

# load dataset
data = pd.read_csv('/Users/cht/Desktop/IDH_HuaQIao/Huaqiao data.csv')
data = data.drop(labels=['Unnamed: 0'], axis=1)
data.info()

X, y = data.drop(labels=['Standard 6'], axis=1), data['Standard 6']

feature_names = list(X.columns)

# Lasso 权重
from yellowbrick.regressor import PredictionError
from sklearn.linear_model import Lasso, Ridge

# Create a new figure
fig = plt.figure(figsize=(15, 14))
ax = fig.add_subplot()
# Title case the feature for better display and create the visualizer
labels = list(map(lambda s: s.title(), feature_names))
viz = FeatureImportances(Lasso(alpha=0.001, random_state=0),
                         ax=ax,
                         labels=labels,
                         relative=True)

# Fit and show the feature importances
viz.fit(X, y)
fig = viz.poof()
fig.figure.savefig("image/Coefficients in the Lasso Model.png")

# 线性相关程度
plt.figure(figsize=(15, 14))
plt.title('Pearson correlation coefficient')
sns.heatmap(data.corr(),
            cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
            annot=True,
            fmt=".2f",
            annot_kws={
                'size': 8,
                'weight': 'normal'
            })
plt.savefig("image/Pearson correlation coefficient.png")

# Spearman Correlation Coefficient
plt.figure(figsize=(15, 11))
plt.title(' Spearman’s rank correlation coefficient')
sns.heatmap(data.corr(method='spearman'),
            cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
            annot=True,
            fmt=".2f",
            annot_kws={
                'size': 8,
                'weight': 'normal'
            })
plt.savefig("image/Spearman’s rank correlation coefficient.png")


# 最大信息系数(Maximal Information Coefficient): 计算每个特征xi和应变量Y的最大信息系数
def MIC_matirx_ShowHeatMap(DataFrame):
    colormap = plt.cm.RdBu
    ylabels = DataFrame.columns.values.tolist()
    f, ax = plt.subplots(figsize=(14, 14))
    plt.figure(figsize=(15, 11))
    plt.title(' Maximal information coefficient')
    plt.xlabel(feature_names)
    plt.ylabel(feature_names)
    sns.heatmap(DataFrame.astype(float),
                cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
                annot=True,
                fmt=".2f",
                annot_kws={
                    'size': 8,
                    'weight': 'normal'
                })
    plt.savefig("image/Maximal information coefficient.png")


def MIC_matirx(dataframe, mine):

    data = np.array(dataframe)
    n = len(data[0, :])
    result = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            mine.compute_score(data[:, i], data[:, j])
            result[i, j] = mine.mic()
            result[j, i] = mine.mic()
    RT = pd.DataFrame(result)
    return RT


mine = MINE()
data_mic = MIC_matirx(data, mine)
data_mic

MIC_matirx_ShowHeatMap(data_mic)

# 平均不纯度减少(Mean Decrease Impurity): 随机森林建树的过程中 根据不纯度选择特征的过程

fig = plt.figure(figsize=(15, 14))
ax = fig.add_subplot()

viz = FeatureImportances(RandomForestClassifier(random_state=0,
                                                n_estimators=400),
                         ax=ax)
viz.fit(X, y)
fig = viz.poof()
fig.figure.savefig(
    "image/Mean Decrease Impurity based on Random Forest Classifier.png")

# 稳定特征选择(Stability Selection): 随机lasso算法中实现稳定特征选择

# 递归特征消除(Recursive Feature Elimination): 普通线性回归(lr)实现递归特征消除
from sklearn.linear_model import LogisticRegression
from yellowbrick.features import RFECV

viz = RFECV(LinearSVC(random_state=0), cv=3, min_features_to_select=20)
viz.fit(X, y)
fig = viz.poof()
fig.figure.savefig("image/Recursive Feature Elimination based on SVM.png", figsize=(15, 14))

# l2正则: 岭回归的参数
from yellowbrick.regressor import PredictionError
from sklearn.linear_model import Lasso, Ridge

# Create a new figure
fig = plt.figure(figsize=(15, 14))
ax = fig.add_subplot()
# Title case the feature for better display and create the visualizer
labels = list(map(lambda s: s.title(), feature_names))
viz = FeatureImportances(Ridge(alpha=0.001, random_state=0),
                         ax=ax,
                         labels=labels,
                         relative=True)

# Fit and show the feature importances
viz.fit(X, y)
fig = viz.poof()
fig.figure.savefig("image/Coefficients in the Ridge Model.png")

# 回归系数: 根据线性回归的系数判断特征的重要性

lrcg = sns.pairplot(data, hue="Standard 6", diag_kind="hist")
lrcg.fig.set_size_inches(25,25)
lrcg.fig.set_dpi(1000)
lrcg.savefig("image/LR_c.png", figsize=(25, 25),dpi=1000)