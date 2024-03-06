import pickle
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    f1_score,
    make_scorer,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.class_weight import compute_class_weight
from pylab import mpl
from yellowbrick.classifier import ROCAUC, ClassificationReport, ConfusionMatrix
import warnings

warnings.filterwarnings("ignore")
mpl.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  ## mac

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def LightGBM_multi_model(X_train, X_test, y_train, y_test, X_val, y_val, class_weights):
    """
    Train and evaluate a LightGBM model.

    Parameters:
    - X_train, X_test, X_val: Training, testing, and validation data.
    - y_train, y_test, y_val: Training, testing, and validation labels.
    - class_weights: Class weights for handling class imbalance.

    Returns:
    - model: Trained LightGBM model.
    """

    # # 设置 LightGBM 参数
    # model = lgb.LGBMClassifier(
    #     n_jobs=-1,
    #     class_weight=class_weights,
    #     random_state=0,
    #     bagging_freq=2,
    #     bagging_fraction=0.5,
    #     verbose=-1,
    # )

    # # 网格搜索参数
    # param_dist = {
    #     "boosting_type": ["gbdt"],
    #     # "max_depth": [5, 10, 50, 100, 200, 500],
    #     "random_state": [0],
    #     # "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
    #     # "num_leaves": [2, 5, 10, 50, 100, 200, 500],
    #     # "n_estimators": [50, 100, 200, 500],
    #     "class_weight": ["balanced", class_weights],
    # }
    params_k = {
        "boosting_type": [
            "gbdt",
            # "dart",
            # "goss",
            # "rf",
        ],
        "objective": "multiclassova",
        "num_class": 5,
        "subsample": 0.5,
        "subsample_freq": 1,
        "learning_rate": 0.01,
        "num_leaves": 2**11 - 1,
        "min_data_in_leaf": 2**12 - 1,
        "feature_fraction": 0.5,
        "max_bin": 100,
        "n_estimators": 500,
        "boost_from_average": False,
        "random_seed": 0,
        "metric": "multi_logloss",
        "class_weight": class_weights,
    }
    train_data = lgb.Dataset(data=X_train, label=y_train, free_raw_data=False)
    print(train_data)
    valid_data = lgb.Dataset(data=X_val, label=y_val, free_raw_data=False)

    model_gbm = lgb.train(
        params_k,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=2000,
        early_stopping_rounds=25,
        verbose_eval=25,
    )

    model = model_gbm
    from sklearn.metrics import classification_report

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred))

    # # 模型评估
    # logger.info(f"Total training score: {model.score(X_train, y_train)}")
    # logger.info(f"Total testing score: {model.score(X_test, y_test)}")
    # logger.info(
    #     f"Total testing recall: {recall_score(y_test, model.predict(X_test), average='macro')}"
    # )
    # logger.info(
    #     f"Total testing confusion matrix: {confusion_matrix(y_test, model.predict(X_test))}"
    # )
    # logger.info(
    #     f"Total testing F1 score: {f1_score(y_test, model.predict(X_test), average='macro')}"
    # # )
    # if len(class_weights) <= 2:
    #     # ROC 曲线
    #     predictions = model.predict_proba(X_test)
    #     false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:, 1])
    #     roc_auc = auc(false_positive_rate, recall)
    #     plt.title("Receiver Operating Characteristic")
    #     plt.plot(false_positive_rate, recall, "b", label="AUC = %0.2f" % roc_auc)
    #     plt.legend(loc="lower right")
    #     plt.plot([0, 1], [0, 1], "r--")
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.0])
    #     plt.ylabel("Recall")
    #     plt.xlabel("Fall-out")
    #     plt.show()
    # if len(class_weights) <= 2:
    #     # 特征重要性
    #     lgb.plot_importance(
    #         model,
    #         dpi=1000,
    #         height=0.5,
    #         figsize=(12, 15),
    #         importance_type="gain",
    #         max_num_features=20,
    #     )
    #     plt.title("LightGBM model’s Gain Feature Importances")
    #     plt.show()
    #     plt.savefig(
    #         "LightGBM model’s Gain Feature Importances "
    #         + "Number of results"
    #         + str(len(class_weights))
    #         + ".png"
    #     )

    #     lgb.plot_importance(
    #         model,
    #         dpi=1000,
    #         height=0.5,
    #         figsize=(12, 12),
    #         importance_type="split",
    #         max_num_features=20,
    #     )
    #     plt.title("LightGBM model’s Split Feature Importances")
    #     plt.savefig(
    #         "LightGBM model’s Split Feature Importances "
    #         + "Number of results"
    #         + str(len(class_weights))
    #         + ".png"
    #     )
    #     plt.show()
    # # else:
    # #     # 特征重要性
    # #     lgb.plot_importance(
    # #         model["classifier"],
    # #         dpi=1000,
    # #         height=0.5,
    # #         figsize=(12, 15),
    # #         importance_type="gain",
    # #         max_num_features=20,
    # #     )
    # #     plt.title("LightGBM model’s Gain Feature Importances")
    # #     plt.show()
    # #     plt.savefig(
    # #         "LightGBM model’s Gain Feature Importances"
    # #         + "Number of results"
    # #         + str(len(class_weights))
    # #         + ".png"
    # #     )

    # #     lgb.plot_importance(
    # #         model["classifier"],
    # #         dpi=1000,
    # #         height=0.5,
    # #         figsize=(12, 12),
    # #         importance_type="split",
    # #         max_num_features=20,
    # #     )
    # #     plt.title("LightGBM model’s Split Feature Importances")
    # #     plt.show()
    # #     plt.savefig(
    # #         "LightGBM model’s Split Feature Importances"
    # #         + "Number of results"
    # #         + str(len(class_weights))
    # #         + ".png"
    # #     )

    # # 保存模型
    # with open(
    #     "Results/LGBM/LGBM_model Number of results"
    #     + str(len(class_weights))
    #     + ".pickle",
    #     "wb",
    # ) as f:
    #     pickle.dump(model, f)

    # # Classification Report
    # viz = ClassificationReport(model, title="Training Report of LGBM", cmap="PuBu")
    # viz.fit(X_train, y_train)
    # viz.score(X_train, y_train)
    # viz.show()
    # fig = viz.poof()
    # fig.figure.savefig(
    #     "Results/LGBM/Output/Training Report of LGBM "
    #     + "Number of results"
    #     + str(len(class_weights))
    #     + ".png"
    # )

    # # ROC
    # viz = ROCAUC(model, title="Training ROC of LGBM")
    # viz.fit(X_train, y_train)
    # viz.score(X_train, y_train)
    # viz.poof()
    # fig = viz.poof()
    # fig.figure.savefig(
    #     "Results/LGBM/Output/Training ROC of LGBM "
    #     + "Number of results"
    #     + str(len(class_weights))
    #     + ".png"
    # )

    # # Classification Report
    # viz = ClassificationReport(model, title="Test Report of LGBM", cmap="PuBu")
    # viz.fit(X_train, y_train)
    # viz.score(X_test, y_test)
    # viz.poof()
    # fig = viz.poof()
    # fig.figure.savefig(
    #     "Results/LGBM/Output/Test Report of LGBM"
    #     + "Number of results"
    #     + str(len(class_weights))
    #     + ".png"
    # )

    # # ROC
    # viz = ROCAUC(model, title="Test ROC of LGBM")
    # viz.fit(X_train, y_train)
    # viz.score(X_test, y_test)
    # viz.poof()
    # fig = viz.poof()
    # fig.figure.savefig(
    #     "Results/LGBM/Output/Test ROC of LGBM "
    #     + "Number of results"
    #     + str(len(class_weights))
    #     + ".png"
    # )

    # # Confusion Matrix
    # if len(class_weights) <= 2:
    #     cm = ConfusionMatrix(model, classes=[0, 1], title="Test Result of LGBM")
    # else:
    #     cm = ConfusionMatrix(
    #         model, classes=[0, 1, 2, 3, 4], title="Test Result of LGBM"
    #     )
    # cm.fit(X_train, y_train)
    # cm.score(X_test, y_test)
    # cm.poof()
    # fig = cm.poof()
    # fig.figure.savefig(
    #     "Results/LGBM/Output/Test Result of LGBM"
    #     + "Number of results"
    #     + str(len(class_weights))
    #     + ".png"
    # )

    # # Evaluation scores
    # eval_scores = {
    #     "training_score": model.score(X_train, y_train),
    #     "validation_score": model.score(X_val, y_val),
    #     "testing_score": model.score(X_test, y_test),
    #     # "roc_auc": roc_auc,
    # }
    # return model, eval_scores
    return classification_report(y_test, y_pred)
