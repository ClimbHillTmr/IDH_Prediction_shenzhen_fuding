import pickle
import logging
import os
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
mpl.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
## mac

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))


def LightGBM_model(
    X_train, X_test, y_train, y_test, X_val, y_val, class_weights, target, kinds
):
    """
    Train and evaluate a LightGBM model.

    Parameters:
    - X_train, X_test, X_val: Training, testing, and validation data.
    - y_train, y_test, y_val: Training, testing, and validation labels.
    - class_weights: Class weights for handling class imbalance.

    Returns:
    - model: Trained LightGBM model.

    """
    model = lgb.LGBMClassifier(
        n_jobs=-1,
        class_weight=class_weights,
        # objective="multiclass",
        random_state=0,
        num_class=1,
        # bagging_freq=2,
        # bagging_fraction=0.5,
        verbose=-1,
    )

    param_dist = {
        "boosting_type": ["gbdt", "dart", "goss", "rf"],
        "max_depth": [50, 100],
        "random_state": [0],
        "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1],
        "num_leaves": [50, 100],
        "n_estimators": [50, 100],
        "subsample": [0.6],
        "colsample_bytree": [0.6],
        "class_weight": ["balanced", class_weights],
    }

    scoring = {
        "recalled_weighted": make_scorer(
            recall_score,
            needs_proba=True,
            average="weighted",
        ),
        "f1_weighted": make_scorer(
            f1_score,
            needs_proba=True,
            average="weighted",
        ),
    }

    # Including evals in the parameters of the fit method
    grid_search = GridSearchCV(
        model,
        n_jobs=-1,
        param_grid=param_dist,
        cv=3,
        scoring=scoring,
        verbose=False,
        refit="f1_weighted",
    )

    grid_search.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        eval_metric="roc",
        early_stopping_rounds=10,
    )

    # Using the best estimator from the grid search
    model = grid_search.best_estimator_

    print("Total training", model.score(X_train, y_train))
    print("Total testing", model.score(X_test, y_test))
    print(
        "Total testing",
        recall_score(
            y_test,
            model.predict(X_test),
            average="weighted",
        ),
    )
    print("Total testing", confusion_matrix(y_test, model.predict(X_test)))
    print(
        "Total testing",
        f1_score(
            y_test,
            model.predict(X_test),
            average="weighted",
        ),
    )

    if len(class_weights) <= 2:
        # ROC 曲线
        predictions = model.predict_proba(X_test)
        false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:, 1])
        roc_auc = auc(false_positive_rate, recall)
        plt.title("Receiver Operating Characteristic")
        plt.plot(false_positive_rate, recall, "b", label="AUC = %0.2f" % roc_auc)
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.ylabel("Recall")
        plt.xlabel("Fall-out")
        plt.show()
        plt.savefig(
            os.path.join(
                current_dir,
                "Results/LGBM/LightGBM_ROC_target="
                + str(target)
                + "_"
                + str(kinds)
                + ".pdf",
            )
        )

        lgb.plot_importance(
            model,
            dpi=1000,
            height=0.5,
            figsize=(12, 15),
            importance_type="gain",
            max_num_features=20,
        )
        plt.title("LightGBM model’s Gain Feature Importances")
        plt.show()
        plt.savefig(
            os.path.join(
                current_dir,
                "Results/LGBM/LightGBM model’s Gain Feature Importances target="
                + str(target)
                + "_"
                + str(kinds)
                + ".pdf",
            )
        )

        lgb.plot_importance(
            model,
            dpi=1000,
            height=0.5,
            figsize=(12, 12),
            importance_type="split",
            max_num_features=20,
        )
        plt.title("LightGBM model’s Split Feature Importances")
        plt.savefig(
            os.path.join(
                current_dir,
                "Results/LGBM/LightGBM model’s Split Feature Importances target="
                + str(target)
                + "_"
                + str(kinds)
                + ".pdf",
            )
        )
        plt.show()
    # else:
    #     # 特征重要性
    #     lgb.plot_importance(
    #         model["classifier"],
    #         dpi=1000,
    #         height=0.5,
    #         figsize=(12, 15),
    #         importance_type="gain",
    #         max_num_features=20,
    #     )
    #     plt.title("LightGBM model’s Gain Feature Importances")
    #     plt.show()
    #     plt.savefig(
    #         "LightGBM model’s Gain Feature Importances"
    #         + "Number of results"
    #         + str(len(class_weights))
    #         + ".pdf"
    #     )

    #     lgb.plot_importance(
    #         model["classifier"],
    #         dpi=1000,
    #         height=0.5,
    #         figsize=(12, 12),
    #         importance_type="split",
    #         max_num_features=20,
    #     )
    #     plt.title("LightGBM model’s Split Feature Importances")
    #     plt.show()
    #     plt.savefig(
    #         "LightGBM model’s Split Feature Importances"
    #         + "Number of results"
    #         + str(len(class_weights))
    #         + ".pdf"
    #     )

    # 保存模型
    with open(
        os.path.join(
            current_dir,
            "Results/LGBM/LGBM_model_Number_of_results_target="
            + str(target)
            + "_"
            + str(kinds)
            + ".pickle",
        ),
        "wb",
    ) as f:
        pickle.dump(model, f)

        # Classification Report
    viz = ClassificationReport(model, title="Training Report of LGBM", cmap="PuBu")
    viz.fit(X_train, y_train)
    viz.score(X_train, y_train)
    viz.show()
    fig = viz.poof()
    fig.figure.savefig(
        os.path.join(
            current_dir,
            "Results/LGBM/Output/Training Report of LGBM target="
            + str(target)
            + "_"
            + str(kinds)
            + ".pdf",
        )
    )

    # ROC
    viz = ROCAUC(model, title="Training ROC of LGBM")
    viz.fit(X_train, y_train)
    viz.score(X_train, y_train)
    viz.poof()
    fig = viz.poof()
    fig.figure.savefig(
        os.path.join(
            current_dir,
            "Results/LGBM/Output/Training ROC of LGBM target="
            + str(target)
            + "_"
            + str(kinds)
            + ".pdf",
        )
    )

    # Classification Report
    viz = ClassificationReport(model, title="Test Report of LGBM", cmap="PuBu")
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.poof()
    fig = viz.poof()
    fig.figure.savefig(
        os.path.join(
            current_dir,
            "Results/LGBM/Output/Test Report of LGBM target="
            + str(target)
            + "_"
            + str(kinds)
            + ".pdf",
        )
    )

    # ROC
    viz = ROCAUC(model, title="Test ROC of LGBM")
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.poof()
    fig = viz.poof()
    fig.figure.savefig(
        os.path.join(
            current_dir,
            "Results/LGBM/Output/Test ROC of LGBM target="
            + str(target)
            + "_"
            + str(kinds)
            + ".pdf",
        )
    )

    # Confusion Matrix
    if len(class_weights) <= 2:
        cm = ConfusionMatrix(model, classes=[0, 1], title="Test Result of LGBM")
    else:
        cm = ConfusionMatrix(
            model, classes=[0, 1, 2, 3, 4], title="Test Result of LGBM"
        )
    cm.fit(X_train, y_train)
    cm.score(X_test, y_test)
    cm.poof()
    fig = cm.poof()
    fig.figure.savefig(
        os.path.join(
            current_dir,
            "Results/LGBM/Output/Test Result of LGBM target="
            + str(target)
            + "_"
            + str(kinds)
            + ".pdf",
        )
    )

    # Evaluation scores
    eval_scores = {
        "training_score": model.score(X_train, y_train),
        "validation_score": model.score(X_val, y_val),
        "testing_score": model.score(X_test, y_test),
        # "roc_auc": roc_auc,
    }

    return model, eval_scores
