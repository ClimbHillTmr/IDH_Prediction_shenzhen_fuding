import sys

sys.path.append("..")
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

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
from Methods.split_methods.fuding_test import fuding_test
from Method_Utils.train_untils import (
    Align_standard,
    quantile_99,
    Iterate_columns,
)

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
dataset = pd.read_csv(
    "/Users/cht/Documents/GitHub/Intradialytic_Hypertension_Classification/透前预测_透中高血压/Final_data/深医_final_data 透前动脉压_HDH.csv"
)
# dataset = pd.read_csv(
#     "/Users/cht/Documents/GitHub/Dialysis_ML/透前预测_透中高血压/Final_data/深医_optimized_data 透前动脉压_HDH.csv"
# )

test_set = fuding_test(whole=True, standard="透前动脉压")

dataset = Align_standard(dataset)
test_set = Align_standard(test_set)

dataset = quantile_99(dataset)
test_set = quantile_99(test_set)

# dataset.info()
# test_set.info()

features = [
    "性别",
    "透析龄",
    # "透析日期",
    "传染病",
    # "患者状态",
    "透前收缩压",
    "透前舒张压",
    # "透后收缩压",
    # "透后舒张压",
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
    # "高血压诊断",
    # "透中高血压_计算",
    # "涨幅时间点比值",
    # "涨幅时间点比值区间",
    # "涨幅时间点差值",
    # "涨幅时间点差值区间",
    # "透析中收缩压_mean",
    # "透析中收缩压_std",
    # "透析中舒张压_mean",
    # "透析中舒张压_std",
    # "透析中脉搏_mean",
    # "透析中脉搏_std",
    # "平均动脉压_mean",
    # "平均动脉压_std",
    # "超滤率_mean",
    # "超滤率_std",
    # "超滤量MAX",
    # "超滤量_mean",
    # "超滤量_std",
    # "静脉压_mean",
    # "静脉压_std",
    # "动脉压_mean",
    # "动脉压_std",
    # "血流速_mean",
    # "血流速_std",
    # "透析液温度_mean",
    # "透析液温度_std",
    # "跨膜压_mean",
    # "跨膜压_std",
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
    "历史平均涨幅时间点比值区间",
    "历史平均涨幅时间点差值区间",
    "历史平均透中高血压_计算",
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
targets = [
    # "透中高血压_计算",
    "涨幅时间点比值区间",
    "涨幅时间点差值区间",
]

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def load_model(file_path):
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model


def models_roc(
    names, sampling_methods, colors, X_test, y_test, target, save=True, dpin=100
):
    plt.figure(figsize=(20, 20), dpi=dpin)

    for name, method, colorname in zip(names, sampling_methods, colors):
        y_test_preds = method.predict(X_test.values)
        try:
            y_pred = method.predict_proba(X_test.values)[:, 1]
        except:
            y_pred = method._predict_proba_lr(X_test)[:, 1]

        if target != "透中高血压_计算":
            try:
                y_pred = method.predict_proba(X_test.values)[:, 1]
            except:
                y_pred = method._predict_proba_lr(X_test)[:, 1]

        print(y_pred)

        fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)

        plt.plot(
            fpr,
            tpr,
            lw=5,
            label="{} (AUC={:.3f})".format(name, auc(fpr, tpr)),
            color=colorname,
        )
        plt.plot([0, 1], [0, 1], "--", lw=5, color="grey")
        plt.axis("square")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("False Positive Rate", fontsize=20)
        plt.ylabel("True Positive Rate", fontsize=20)
        plt.title("ROC Curve for " + str(target), fontsize=25)
        plt.legend(loc="lower right", fontsize=20)

    if save:
        plt.savefig("models_roc" + str(target) + ".pdf")

    return plt


for target in targets:
    test_set = test_set[test_set[target] != 4]
    X_test = test_set[features]
    X_test = Iterate_columns(X_test)

    y_test = test_set[target]
    output_file_evaluate_models = "results_evaluate_models" + str(target) + ".csv"
    model_paths = {
        "LGBM": "/Users/cht/Documents/GitHub/Intradialytic_Hypertension_Classification/透前预测_透中高血压/透前模型/Results/LGBM/LGBM_model Number of results target = "
        + str(target)
        + ".pickle",
        "TabNet": "/Users/cht/Documents/GitHub/Intradialytic_Hypertension_Classification/透前预测_透中高血压/透前模型/Results/TabNet/TabNet_model target = "
        + str(target)
        + ".pickle",
        "SVM": "/Users/cht/Documents/GitHub/Intradialytic_Hypertension_Classification/透前预测_透中高血压/透前模型/Results/SVM/SVM_model target = "
        + str(target)
        + ".pickle",
    }
    y_test = test_set[target]
    models = {name: load_model(path) for name, path in model_paths.items()}

    names = ["LGBM", "SVM", "TabNet"]
    models = [models["LGBM"], models["SVM"], models["TabNet"]]
    colors = ["r", "g", "b"]

    models_roc(names, models, colors, X_test, y_test, target)
