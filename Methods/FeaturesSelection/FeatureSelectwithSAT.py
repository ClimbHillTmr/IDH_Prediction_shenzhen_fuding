import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import f_classif,mutual_info_classif

from sklearn.model_selection import train_test_split
from feature_engine.selection import (
    DropConstantFeatures,
    DropDuplicateFeatures,
    SmartCorrelatedSelection,
)

from compareperformance import run_logistic

# load dataset
data = pd.read_csv('/Users/cht/Desktop/IDH_HuaQIao/Huaqiao data.csv')
data.info()

# separate dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['Standard 6', 'Unnamed: 0'], axis=1),
    data['Standard 6'],
    test_size=0.2,
    random_state=0,
)

X_train.shape, X_test.shape

X_train_original = X_train.copy()
X_test_original = X_test.copy()

# remove constant features
constant_features = [feat for feat in X_train.columns if X_train[feat].std() == 0]

X_train.drop(labels=constant_features, axis=1, inplace=True)
X_test.drop(labels=constant_features, axis=1, inplace=True)

X_train.shape, X_test.shape

# remove quasi-constant features
sel = DropConstantFeatures(tol=0.99, variables=None, missing_values='raise')

sel.fit(X_train)

# number of quasi-constant features

len(sel.features_to_drop_)

# list of quasi-constant features

sel.features_to_drop_

# percentage of observations showing each of the different values
# of the variable

var = sel.features_to_drop_

X_train[var].value_counts(normalize=True)

# remove the quasi-constant features

X_train = sel.transform(X_train)
X_test = sel.transform(X_test)

X_train.shape, X_test.shape

# Remove duplicated features

# set up the selector
sel = DropDuplicateFeatures(variables=None, missing_values='raise')

# find the duplicate features, this might take a while
sel.fit(X_train)

# these are the pairs of duplicated features
# each set are duplicates

sel.duplicated_feature_sets_

# these are the features that will be dropped
# 1 from each of the pairs above

sel.features_to_drop_

# let's explore our list of duplicated features

len(sel.features_to_drop_)

# remove the duplicated features

X_train = sel.transform(X_train)
X_test = sel.transform(X_test)

X_train.shape, X_test.shape

# Remove correlated
# set up the selector

# visualise correlated features:

# I will build a correlation matrix, which examines the
# correlation of all features (that is, for all possible feature combinations)
# and then visualise the correlation matrix using a heatmap

# the default correlation method of pandas.corr is pearson
# I include it anyways for the demo
corrmat = X_train.corr(method='pearson')

# we can make a heatmap with the package seaborn
# and customise the colours of searborn's heatmap
cmap = sns.diverging_palette(200, 20, as_cmap=True)

# some more parameters for the figure
fig, ax = plt.subplots()
fig.set_size_inches(20, 20)
fig.set_dpi(1000)

# and now plot the correlation matrix
sns.heatmap(corrmat, cmap=cmap)

# correlation selector

sel = SmartCorrelatedSelection(
    variables=None,
    method="pearson",
    threshold=0.85,
    missing_values="raise",
    selection_method="variance",
    estimator=None,
    scoring="roc_auc",
    cv=1,
)

sel.fit(X_train, y_train)
# find correlated features
sel.fit(X_train)

# each set contains a group of correlated features

group = sel.correlated_feature_sets_[0]

X_train[group].std()

# the transformer selects 1 feature from each group.
# the rest will be removed and can be found in this attribute

len(sel.features_to_drop_)

# drop correlated features

X_train = sel.transform(X_train)
X_test = sel.transform(X_test)

X_train.shape, X_test.shape

# Select features based of anova
univariate_ = f_classif(X_train, y_train)

univariate_
univariate = pd.Series(univariate_[1])

univariate.index = X_train.columns
print(univariate)

univariate.sort_values(ascending=False).plot.bar(figsize=(20, 6))
univariate.to_csv('ANOVA_pvalue.csv')
plt.show()

X_train_anova = X_train[univariate.index[univariate <= 0.05]]

X_test_anova = X_test[univariate.index[univariate <= 0.05]]

run_logistic(
    X_train_original, X_train_anova, X_test_original, X_test_anova, y_train, y_test
)

list(univariate.index[univariate <= 0.05])

X_train_anova.shape, X_test_anova.shape

# calculate the mutual information between the variables and the target

# the smaller the value of the mi, the less information we can infer from
# the feature about the target

mi = mutual_info_classif(X_train, y_train, random_state=0)
mi
mi = pd.Series(mi)
mi.index = X_train.columns
mi.sort_values(ascending=False).plot.bar(figsize=(20, 6))
plt.ylabel('Mutual Information')
plt.show()


X_train_mi = X_train[mi.index[mi > 0.005]]
print(X_train_mi.columns)
X_test_mi = X_test[mi.index[mi > 0.005]]

# here we will select the top 10 features
# based on their mutual information value

run_logistic(X_train_original, X_train_mi, X_test_original, X_test_mi, y_train, y_test)

features_SAT_union = list(set(X_test_anova.columns).union(set(X_test_mi.columns)))

features_SAT_intersection = list(set(X_test_anova.columns).intersection(set(X_test_mi.columns)))

X_train_sat_in = X_train[features_SAT_intersection]
X_test_sat_in = X_test[features_SAT_intersection]
print(X_train_sat_in.columns)
X_train_sat_un = X_train[features_SAT_union]
X_test_sat_un = X_test[features_SAT_union]
print(X_train_sat_un.columns)

run_logistic(X_train_original, X_train_sat_in, X_test_original, X_test_sat_in, y_train, y_test)
run_logistic(X_train_original, X_train_sat_un, X_test_original, X_test_sat_un, y_train, y_test)