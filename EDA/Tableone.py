import sys

sys.path.append("..")
import pandas as pd
import numpy as np
from Methods.split_methods.fuding_test import fuding_test
from Methods.split_methods.date_method import split_dataset_by_date

from tableone import TableOne
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

from Methods.Models.LightGBM_model import LightGBM_model
from Methods.Models.XGBoost import XGBoost_model
from Methods.split_methods.fuding_test import fuding_test
from Methods.split_methods.date_method import split_dataset_by_date

from Methods.Models.TabNet import TabNet_model
from Methods.Models.C_SVM_model import SVM_model

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("../Recognize_data/福鼎_IDH_data.csv")
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

history_rate_proportion = [
    "history_HBP_rate",
    "history_LBP_times_0_rate",
    "history_LBP_times_1_rate",
    "history_LBP_times_2_rate",
    "history_LBP_times_3_rate",
    "history_LBP_times_4_rate",
]
history_rate_diff = [
    "history_HBP",
    "history_LBP_times_0",
    "history_LBP_times_1",
    "history_LBP_times_2",
    "history_LBP_times_3",
    "history_LBP_times_4",
]
dataset[features] = quantile_99(dataset[features])

# Table one

categorical = [
    "性别",
    "传染病",
    "透析方式",
    "透析液电导率",
    "抗凝剂类型",
    "瘘管类型",
    "瘘管位置",
    # "高血压诊断",
    # "降幅时间点比值区间",
    # "降幅时间点差值区间",
]

groupby = ["透中低血压_计算"]
nonnormal = ["透析年龄"]
labels = {"透中低血压_计算": "透中低血压"}

mytable = TableOne(
    dataset[features],
    categorical=categorical,
    groupby=groupby,
    nonnormal=nonnormal,
    rename=labels,
    smd=True,
    htest_name=True,
    pval=True,
    pval_adjust="Bonferroni",
)

print(mytable.tabulate(tablefmt="simple"))

mytable.to_excel("Tableone.xlsx")
