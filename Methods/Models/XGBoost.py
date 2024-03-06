import pickle
import os
import xgboost as xgb
from sklearn.metrics import recall_score, confusion_matrix, f1_score
from xgboost import plot_importance
from sklearn.metrics import (
    f1_score,
    make_scorer,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
)
from yellowbrick.classifier import ROCAUC, ClassificationReport, ConfusionMatrix
import warnings

warnings.filterwarnings("ignore")
# mpl.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
## mac
import matplotlib.pyplot as plt

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))


def XGBoost_model(
    X_train, X_test, y_train, y_test, X_val, y_val, class_weights, target, kinds
):
    """
    Train and evaluate an XGBoost model.

    Parameters:
    - X_train, X_test, X_val: Training, testing, and validation data.
    - y_train, y_test, y_val: Training, testing, and validation labels.
    - class_weights: Class weights for handling class imbalance.

    Returns:
    - model: Trained XGBoost model.

    """
    # Convert data to DMatrix format (required by XGBoost)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Define XGBoost parameters
    params = {
        "booster": "gbtree",
        "objective": "multi:softmax" if len(class_weights) > 2 else "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "num_class": 4 if len(class_weights) > 2 else 1,
        "seed": 0,
        "nthread": -1,
    }

    # Train XGBoost model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, "eval")],
        early_stopping_rounds=20,
        verbose_eval=True,
    )

    # Evaluate the model
    # print(
    #     "Total training",
    #     recall_score(y_train, model.predict(X_train)),
    # )
    # print(
    #     "Total testing", recall_score(y_test, model.predict(X_test))
    # )
    # print("Total testing", confusion_matrix(y_test, model.predict(X_test)))
    # print("Total testing", f1_score(y_test, model.predict(X_test)))

    # Get probability estimates for the positive class
    y_prob = model.predict(dtest)

    # ROC curve
    false_positive_rate, recall, thresholds = roc_curve(y_test, y_prob)
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
            current_dir, "Results/XGBoost/XGBoost_ROC_target=" + str(target) + ".pdf"
        )
    )

    # Plot feature importances
    plot_importance(model)
    plt.title("XGBoost Feature Importances")
    plt.savefig(
        os.path.join(
            current_dir,
            f"Results/XGBoost/XGBoost_Feature_Importances_target={target}.pdf",
        )
    )

    # Save the XGBoost model
    model.save_model(
        os.path.join(
            current_dir, f"Results/XGBoost/XGBoost_model_target={target}.model"
        )
    )

    # Save the model with pickle (optional)
    with open(
        os.path.join(
            current_dir, f"Results/XGBoost/XGBoost_model_target={target}.pickle"
        ),
        "wb",
    ) as f:
        pickle.dump(model, f)

    # Classification Report
    viz = ClassificationReport(model, title="Training Report of XGBoost", cmap="PuBu")
    viz.fit(X_train, y_train)
    viz.score(X_train, y_train)
    viz.show()
    fig = viz.poof()
    fig.figure.savefig(
        os.path.join(
            current_dir,
            "Results/XGBoost/Output/Training Report of XGBoost target="
            + str(target)
            + ".pdf",
        )
    )

    # ROC
    viz = ROCAUC(model, title="Training ROC of XGBoost")
    viz.fit(X_train, y_train)
    viz.score(X_train, y_train)
    viz.poof()
    fig = viz.poof()
    fig.figure.savefig(
        os.path.join(
            current_dir,
            "Results/XGBoost/Output/Training ROC of XGBoost target="
            + str(target)
            + ".pdf",
        )
    )

    # Classification Report
    viz = ClassificationReport(model, title="Test Report of XGBoost", cmap="PuBu")
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.poof()
    fig = viz.poof()
    fig.figure.savefig(
        os.path.join(
            current_dir,
            "Results/XGBoost/Output/Test Report of XGBoost target="
            + str(target)
            + ".pdf",
        )
    )

    # ROC
    viz = ROCAUC(model, title="Test ROC of XGBoost")
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.poof()
    fig = viz.poof()
    fig.figure.savefig(
        os.path.join(
            current_dir,
            "Results/XGBoost/Output/Test ROC of XGBoost target=" + str(target) + ".pdf",
        )
    )

    # Confusion Matrix
    if len(class_weights) <= 2:
        cm = ConfusionMatrix(model, classes=[0, 1], title="Test Result of XGBoost")
    else:
        cm = ConfusionMatrix(
            model, classes=[0, 1, 2, 3, 4], title="Test Result of XGBoost"
        )
    cm.fit(X_train, y_train)
    cm.score(X_test, y_test)
    cm.poof()
    fig = cm.poof()
    fig.figure.savefig(
        os.path.join(
            current_dir,
            "Results/XGBoost/Output/Test Result of XGBoost target="
            + str(target)
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


# # 调用 XGBoost_model 函数
# XGBoost_model(
#     X_train, X_test, y_train, y_test, X_val, y_val, class_weights, target, kinds
# )
