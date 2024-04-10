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


plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]

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
    # "低血压诊断",
    # "透中低血压_计算",
    # "降幅时间点比值",
    # "降幅时间点比值区间",
    # "降幅时间点差值",
    # "降幅时间点差值区间",
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
    "透中低血压_计算",
    # "降幅时间点比值区间",
    # "降幅时间点差值区间",
]


def load_model(file_path):
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model


# 读取数据集
X = pd.read_csv(
    "/Users/cht/Documents/GitHub/IDH_Prediction_shenzhen_fuding/fuding_dataset_透中低血压_计算透前模型.csv"
)

y = pd.read_csv(
    "/Users/cht/Documents/GitHub/IDH_Prediction_shenzhen_fuding/fuding_dataset_target_透中低血压_计算透前模型.csv"
)
X_train, X_test, y_train, y_test = train_test_split(
    X[features], y[target], test_size=0.2, random_state=0, shuffle=True
)


def models_roc(
    names, sampling_methods, colors, X_test, y_test, target, save=True, dpin=100
):
    plt.figure(figsize=(20, 20), dpi=dpin)

    for name, method, colorname in zip(names, sampling_methods, colors):
        y_test_preds = method.predict(X_test.values)
        try:
            y_pred = method.predict_proba(X_test.values)[:, 1]
        except:
            y_pred = method._predict_proba_lr(X_test.values)[:, 1]


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


output_file_evaluate_models = "results_evaluate_models" + str(target) + ".csv"
model_paths = {
    "LGBM": "/Users/cht/Documents/GitHub/IDH_Prediction_shenzhen_fuding/Methods/Models/Results/LGBM/LGBM_model_Number_of_results_target=透中低血压_计算_透前模型.pickle",
    "TabNet": "/Users/cht/Documents/GitHub/IDH_Prediction_shenzhen_fuding/Methods/Models/Results/TabNet/TabNet_model_Number_of_results_target=透中低血压_计算_透前模型.pickle",
    "SVM": "/Users/cht/Documents/GitHub/IDH_Prediction_shenzhen_fuding/Methods/Models/Results/SVM/SVM_model_Number_of_results_target=透中低血压_计算_透前模型.pickle",
}

models = {name: load_model(path) for name, path in model_paths.items()}

names = ["LGBM",  "TabNet","SVM",]
models = [models["LGBM"], models["TabNet"],models["SVM"]]
colors = ["r", "g", "b"]

models_roc(names, models, colors, X_test, y_test, target)
