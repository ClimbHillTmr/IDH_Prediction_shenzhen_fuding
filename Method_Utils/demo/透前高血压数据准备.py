import os, sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('../Dataset/basic_dataset.csv')
dataset.info()

X_columns = [
    '年龄',
    '性别',
    '透析龄',
    '透析龄_天数',
    '传染病',
    '透析器',
    '透析方式',
    '透前舒张压',
    '透前收缩压',
    '透析前预设UFV',
    '透前体重',
    '透前呼吸频率',
    '透前体温',
    '干体重',
    '透前体重-干体重',
    '预设差值',
    '透析液钙浓度',
    '透析液电导率',
    '透析液钠浓度',
    '透析液钾浓度',
    '透析液碳酸氢根浓度',
    '抗凝剂类型',
    '抗凝剂使用总量',
    '抗凝剂维持量',
    '抗凝剂追加量',
]

y_columns = [
    '透中高血压',
]
dataset.info()
dataset[y_columns].info()

X_data = dataset
X_data = X_data[(X_data['透前舒张压'] > 0)]
X_data = X_data[(X_data['透析龄_天数'] >= 0)]
X_data = X_data[(X_data['干体重'] > 0)]
X_data = X_data[(X_data['透前体重'] > 0)]
X_data['透前体重-干体重'] = X_data['透前体重'] - X_data['干体重']
X_data['预设差值'] = X_data['透析前预设UFV'] - X_data['透前体重-干体重']
X_data['透前呼吸频率'].fillna(value=0, inplace=True)
X_data['透前体温'].fillna(value=0, inplace=True)
X_data['抗凝剂追加量'].fillna(value=0, inplace=True)

X_data['预设差值'].fillna(value=0, inplace=True)
X_data['透析前预设UFV'].fillna(value=0, inplace=True)
X_data[X_columns].info()

X = np.array(X_data[X_columns])
y = np.array(X_data[y_columns])
