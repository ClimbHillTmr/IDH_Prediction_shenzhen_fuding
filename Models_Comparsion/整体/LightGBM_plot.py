import sys

sys.path.append("..")
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# 导入所需库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
import pickle
import sys

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

target = [
    "新_透中低血压_计算",
    # "降幅时间点比值区间",
    # "降幅时间点差值区间",
]



def load_model(file_path):
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model


# 读取数据集
X = pd.read_csv(
    "/Users/cht/Documents/GitHub/IDH_Prediction_shenzhen_fuding/fuding_dataset_新_透中低血压_计算透后模型.csv"
)

y = pd.read_csv(
    "/Users/cht/Documents/GitHub/IDH_Prediction_shenzhen_fuding/fuding_dataset_target_新_透中低血压_计算透后模型.csv"
)

X_train, X_test, y_train, y_test = train_test_split(
    X[features], y[target], test_size=0.2, random_state=0, shuffle=True
)

y_test = y_test.values

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def old_models_result(method, X_test, y_test, target, name):
    y_test_preds = method.predict(X_test.values)
    try:
        y_test_predprob = method.predict_proba(X_test.values)[:, 1]
    except:
        y_test_predprob = method._predict_proba_lr(X_test.values)[:, 1]


    f1 = f1_score(y_test, y_test_preds)
    precision = precision_score(y_test, y_test_preds)
    recall = recall_score(y_test, y_test_preds)
    auc = roc_auc_score(y_test, y_test_predprob)

    result = [f1, precision, recall, auc]
    column = ["f1", "precision", "recall", "auc"]
    result = pd.DataFrame([result], columns=column)
    result.to_csv(str(name) + "Result" + str(target) + ".csv")

with open(
    "/Users/cht/Documents/GitHub/IDH_Prediction_shenzhen_fuding/Methods/Models/Results/LGBM/LGBM_model_Number_of_results_target=新_透中低血压_计算_透后模型.pickle",
    "rb",
) as f:
    LGBM_model = pickle.load(f)


with open(
    "/Users/cht/Documents/GitHub/IDH_Prediction_shenzhen_fuding/Methods/Models/Results/SVM/SVM_model_Number_of_results_target=新_透中低血压_计算_透后模型.pickle",
    "rb",
) as f:
    SVM_model = pickle.load(f)

with open(
    "/Users/cht/Documents/GitHub/IDH_Prediction_shenzhen_fuding/Methods/Models/Results/TabNet/TabNet_model_Number_of_results_target=新_透中低血压_计算_透后模型.pickle",
    "rb",
) as f:
    TabNet_model = pickle.load(f)

print(target)


old_models_result(LGBM_model, X_test=X_test, y_test=y_test, name="LGBM", target=target)
old_models_result(
    TabNet_model, X_test=X_test, y_test=y_test, name="TabNet", target=target
)
old_models_result(SVM_model, X_test=X_test, y_test=y_test, name="SVM", target=target)

model = LGBM_model

importance = model.booster_.feature_importance(importance_type="gain")
scaler = MinMaxScaler()

scaler = scaler.fit(importance.reshape(len(X_test.columns), 1))

importances = scaler.transform(importance.reshape(len(X_test.columns), 1))

importance = pd.DataFrame(
    importance.reshape(1, len(X_test.columns)), columns=list(X_test.columns)
)
importance.to_csv(
    str(target) + "LGBM_gain feature_importance.csv",
    index=False,
)
importance = model.booster_.feature_importance(importance_type="split")
scaler = MinMaxScaler()

scaler = scaler.fit(importance.reshape(len(X_test.columns), 1))

importances = scaler.transform(importance.reshape(len(X_test.columns), 1))

importance = pd.DataFrame(
    importance.reshape(1, len(X_test.columns)), columns=list(X_test.columns)
)
importance.to_csv(
    str(target) + "LGBM_split feature_importance.csv",
    index=False,
)

tree = lgb.create_tree_digraph(
    model,
    # feature_names=list(X_test.columns),
    orientation="vertical",
    tree_index=0,
    show_info=[
        "leaf_count",
        "data_percentage",
        "split_gain",
        "internal_count",
        "internal_value",
        "internal_weight",
        "leaf_weight",
    ],
)
tree.render(str(target) + "First_Tree")
