import os, sys
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ROCAUC, ClassificationReport

import sys


sys.path.append('..')
# from Methods.Models.NN import NN_model_train
from Methods.Models.LightGBM_model import LightGBM_model

from 透前预测_透中高血压.Results.demo.透前高血压数据准备 import X, y

X = X.astype('float64')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, shuffle=True
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.125, random_state=0, shuffle=True
)

print('训练集总长度：', len(X_train), '；label 为 1 的长度', np.sum(y_train))
print('验证集总长度：', len(X_val), '；label 为 1 的长度', np.sum(y_val))
print('测试集总长度：', len(X_test), '；label 为 1 的长度', np.sum(y_test))

pos = np.sum(y)
neg = len(X) - np.sum(y_train)
total = neg + pos

# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg) * (total) / 2.0
weight_for_1 = (1 / pos) * (total) / 2.0
print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

class_weight = {0: weight_for_0, 1: weight_for_1}

from yellowbrick.classifier import PrecisionRecallCurve, ROCAUC, ClassificationReport

LightGBM_model(X_train, X_test, y_train, y_test, X_val, y_val, class_weight)


with open('Results/LGBM/LGBM model.pickle', 'rb') as f:
    model = pickle.load(f)

viz = PrecisionRecallCurve(model)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show()


visualizer = ROCAUC(model, classes=["not_spam", "is_spam"])
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()  # Finalize and show the figure

# visualizer = ClassificationReport(model)

# visualizer.fit(X_train, y_train)  # Fit the visualizer and the model
# visualizer.score(X_test, y_test)  # Evaluate the model on the test data
# visualizer.show()  # Finalize and show the figure
