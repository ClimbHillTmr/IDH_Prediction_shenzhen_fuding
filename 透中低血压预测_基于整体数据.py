import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import sys

sys.path.append("..")
from Method_Utils.train_untils import (
    remove_rows_with_few_duplicates,
    create_moved_dataframe,
    process_dataset,
    Align_standard,
    features_based,
    features,
    quantile_99,
    Iterate_columns,
    calculate_class_weights,
    preprocessing_data,
)

# from Methods.Models.LightGBM_model import LightGBM_model

# from Methods.Models.XGBoost import XGBoost_model
from Methods.split_methods.fuding_test import fuding_test
from Methods.split_methods.date_method import split_dataset_by_date

from Methods.Models.TabNet import TabNet_model
from Methods.Models.C_SVM_model import SVM_model

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Recognize_data/福鼎_IDH_data.csv")
# dataset = pd.read_csv(
#     "/Users/cht/Documents/GitHub/Dialysis_ML/透前预测_透中低血压/Final_data/深医_optimized_data 透前动脉压_HDH.csv"
# )

dataset = preprocessing_data(dataset)

columns_to_move = [
    "透中低血压_计算",
    # "涨幅时间点比值",
    "降幅时间点比值区间",
    # "涨幅时间点差值",
    "降幅时间点差值区间",
]


dataset = Align_standard(dataset)

dataset = process_dataset(df=dataset, columns_to_shift=columns_to_move)

dataset.info()

features = [
    "性别",
    "透析龄",
    # "透析日期",
    "传染病",
    # "患者状态",
    "透前收缩压",
    "透前舒张压",
    "透后收缩压",
    "透后舒张压",
    "透析龄_天数",
    "透析年龄",
    # "透析器",
    "透析方式",
    # "透析前预设UFV",
    "透前体重",
    "透前呼吸频率",
    # "透前体温",
    "干体重",
    "透析液钙浓度",
    "透析液电导率",
    # "透析液钠浓度",
    # "透析液钾浓度",
    # "透析液碳酸氢根浓度",
    "抗凝剂类型",
    # "抗凝剂使用总量",
    # "抗凝剂维持量",
    # "抗凝剂追加量",
    # "透析后体重",
    # "透后脉搏",
    # "透后体温",
    # "实际透析时长",
    "瘘管类型",
    "瘘管位置",
    "透前体重-干体重",
    "瘘管使用时间",
    "首次透析年龄",
    # "低血压诊断",
    "透中低血压_计算",
    # "降幅时间点比值",
    # "降幅时间点比值区间",
    # "降幅时间点差值",
    # "降幅时间点差值区间",
    "透析中收缩压_mean",
    "透析中收缩压_std",
    "透析中舒张压_mean",
    "透析中舒张压_std",
    "透析中脉搏_mean",
    "透析中脉搏_std",
    # "平均动脉压_mean",
    # "平均动脉压_std",
    "超滤率_mean",
    "超滤率_std",
    "超滤量MAX",
    # "超滤量_mean",
    # "超滤量_std",
    "静脉压_mean",
    "静脉压_std",
    "动脉压_mean",
    "动脉压_std",
    "血流速_mean",
    "血流速_std",
    "透析液温度_mean",
    "透析液温度_std",
    "跨膜压_mean",
    "跨膜压_std",
    # "历史平均患者状态",
    # "历史平均透析前预设UFV",
    "历史平均透前体重",
    "历史平均透前呼吸频率",
    "历史平均透前体温",
    "历史平均干体重",
    "历史平均透析液钙浓度",
    "历史平均透析液电导率",
    # "历史平均透析液钠浓度",
    # "历史平均透析液钾浓度",
    # "历史平均透析液碳酸氢根浓度",
    "历史平均实际透析时长",
    "历史平均透前收缩压",
    "历史平均透前舒张压",
    "历史平均降幅时间点比值区间",
    "历史平均降幅时间点差值区间",
    "历史平均透中低血压_计算",
    "历史平均透析中收缩压_mean",
    "历史平均透析中舒张压_mean",
    "历史平均透析中脉搏_mean",
    # "历史平均平均动脉压_mean",
    "历史平均超滤率_mean",
    # "历史平均超滤量_mean",
    "历史平均静脉压_mean",
    "历史平均动脉压_mean",
    "历史平均血流速_mean",
    "历史平均透析液温度_mean",
    "历史平均跨膜压_mean",
]

targets = [
    "新_透中低血压_计算",
    # "降幅时间点比值区间",
    # "降幅时间点差值区间",
]

dataset[features] = quantile_99(dataset[features])

for target in targets:

    # if target in [
    #     "降幅时间点比值区间",
    # ]:
    #     features = features + history_rate_proportion
    # elif target in [
    #     "降幅时间点差值区间",
    # ]:
    #     features = features + history_rate_diff

    # enc = preprocessing.OneHotEncoder()

    dataset = dataset[dataset[target] != 4]

    X = dataset[features]
    y = dataset[target]

    X = Iterate_columns(X)

    X.to_csv("fuding_dataset_" + str(target) + ".csv")
    y.to_csv("fuding_dataset_target_" + str(target) + ".csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    class_weight = calculate_class_weights(dataset[target])

    X.info()
    X_test.info()

    # if len(class_weight) > 2:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=0, shuffle=True
    )

    # from imblearn.combine import SMOTEENN, SMOTETomek

    # # from imblearn.over_sampling import SMOTE
    # # from imblearn.under_sampling import ClusterCentroids

    # smote_enn = SMOTEENN(random_state=0, sampling_strategy="minority")
    # X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
    # X_train, y_train = X_resampled, y_resampled
    # Check the number of classes in the target variable
    # print(np.unique(y_test))
    # print(np.unique(y_train))
    # print(np.unique(y_val))

    print("总长度：", len(X), "；label 为 1 的长度", np.sum(y))
    print("训练集总长度：", len(X_train), "；label 为 1 的长度", np.sum(y_train))
    print("验证集总长度：", len(X_val), "；label 为 1 的长度", np.sum(y_val))
    print("测试集总长度：", len(X_test), "；label 为 1 的长度", np.sum(y_test))

    # LightGBM_model(
    #     X_train,
    #     X_test,
    #     y_train,
    #     y_test,
    #     X_val,
    #     y_val,
    #     class_weights=class_weight,
    #     target=target,
    #     kinds="透后模型",
    # )

    # SVM_model(
    #     X_train=X_train,
    #     X_test=X_test,
    #     y_train=y_train,
    #     y_test=y_test,
    #     X_val=X_val,
    #     y_val=y_val,
    #     class_weights=class_weight,
    #     target=target,
    #     kinds="透后模型",
    # )

    TabNet_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val,
        class_weights=class_weight,
        target=target,kinds="透后模型",
    )
