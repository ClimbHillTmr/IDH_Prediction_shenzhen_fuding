import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from auc_delong_xu import auc_ci_Delong
import pickle
import sys
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append("..")
from Method_Utils.train_untils import Iterate_columns, quantile_99, Align_standard
from Methods.split_methods.fuding_test import fuding_test
from Models_Comparsion.compare_auc_delong_xu import delong_roc_variance, delong_roc_test

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


def load_model(file_path):
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model


targets = [
    "新_透中低血压_计算",
    # "降幅时间点比值区间",
    # "降幅时间点差值区间",
]
dataset_path = "/Users/cht/Documents/GitHub/Intradialytic_Hypertension_Classification/透前预测_透中高血压/Final_data/深医_final_data 透前动脉压_HDH.csv"

# 读取数据集
dataset = pd.read_csv(dataset_path)

X = dataset[features]
y = dataset[targets]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, shuffle=True
)

output_file_evaluate_models = "results_evaluate_models" + str(target) + ".csv"
model_paths = {
    "LGBM": "/Users/cht/Documents/GitHub/IDH_Prediction_shenzhen_fuding/Methods/Models/Results/LGBM/LGBM_model_Number_of_results_target=新_透中低血压_计算.pickle",
    "TabNet": "/Users/cht/Documents/GitHub/IDH_Prediction_shenzhen_fuding/Methods/Models/Results/SVM/SVM_model_Number_of_results_target=新_透中低血压_计算.pickle",
    "SVM": "/Users/cht/Documents/GitHub/IDH_Prediction_shenzhen_fuding/Methods/Models/Results/SVM/SVM_model_Number_of_results_target=新_透中低血压_计算.pickle",
}

models = {name: load_model(path) for name, path in model_paths.items()}

# # 评估多个模型
# model_names, aucs, auc_var_delongs = evaluate_models(
#     models, X_test, y_test, output_file_evaluate_models, target=target
# )

auc_var_delongs = []
aucs = []
model_names = []

# 创建一个空的DataFrame用于存储结果
results = pd.DataFrame(
    columns=[
        "Model",
        "ROC AUC",
        "Variance",
        "LGBM-SVM p-value",
        "LGBM-TabNet p-value",
    ]
)

# 计算p值
y_pred_LGBM = models["LGBM"].predict_proba(X_test)[:, 1]
if target != "透中高血压_计算":
    y_pred_LGBM = models["LGBM"].predict_proba(X_test)

auc_LGBM = roc_auc_score(
    y_test,
    y_pred_LGBM,
    average="weighted",
    multi_class="ovr",
)
_, auc_var_delong_LGBM = delong_roc_variance(y_test, y_pred_LGBM)

model_names.append("LGBM")
auc_var_delongs.append(auc_var_delong_LGBM)
aucs.append(auc_LGBM)

with open(
    "/Users/cht/Documents/GitHub/Intradialytic_Hypertension_Classification/透前预测_透中高血压/透前模型/Results/SVM/SVM_model target = "
    + str(target)
    + ".pickle",
    "rb",
) as f:
    SVM_model = pickle.load(f)

with open(
    "/Users/cht/Documents/GitHub/Intradialytic_Hypertension_Classification/透前预测_透中高血压/透前模型/Results/TabNet/TabNet_model target = "
    + str(target)
    + ".pickle",
    "rb",
) as f:
    TabNet_model = pickle.load(f)

for model, model_name in zip(
    [
        SVM_model,
        TabNet_model,
    ],
    ["SVM", "TabNet"],
):
    print(model_name)
    print(model)
    try:
        y_pred = model.predict_proba(X_test.values)[:, 1]
    except:
        y_pred = model._predict_proba_lr(X_test.values)[:, 1]

    auc = roc_auc_score(
        y_test,
        y_pred,
        average="weighted",
        multi_class="ovr",
    )

    _, auc_var_delong = delong_roc_variance(y_test, y_pred)

    model_names.append(model_name)
    auc_var_delongs.append(auc_var_delong)
    aucs.append(auc)

    print("auc_var_delong")
    print(auc_var_delong)

    # 计算p值
    p_value_LGBM = delong_roc_test(y_test, y_pred_LGBM, y_pred)
    print("p_value_LGBM")
    print(p_value_LGBM)
    results["LGBM-" + str(model_name) + " p-value"] = p_value_LGBM[0]
    continue

# 将结果添加到DataFrame
results["Model"] = [model_names]
results["ROC AUC"] = [aucs]
results["Variance"] = [[np.sqrt(i) for i in (auc_var_delongs)]]

# 打印结果
print(results)

# 将结果写入CSV文件
results.to_csv(output_file_evaluate_models, index=False)

print(model_names)
print("variance", auc_var_delongs)
print(aucs)
