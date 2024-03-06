# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC, LinearSVC
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    f1_score,
    make_scorer,
    roc_auc_score,
)

from yellowbrick.classifier import ROCAUC, ClassificationReport, ConfusionMatrix


def SVM_model(X_train, X_test, y_train, y_test, X_val, y_val, class_weights):
    """
    Train and evaluate a Support Vector Machine (SVM) model for IDH classification.

    Parameters:
    - X_train, X_test, X_val: Training, testing, and validation input features.
    - y_train, y_test, y_val: Training, testing, and validation target labels.
    - class_weights: Class weights for handling class imbalance.
    - is_new_patient (bool): Whether the patient is new. Default is True.

    Returns:
    - model: Trained SVM model.
    - eval_scores: Dictionary containing evaluation scores.
    """

    # Display the dataset statistics
    print("Training set length:", len(X_train), "; label 1 count:", np.sum(y_train))
    print("Validation set length:", len(X_val), "; label 1 count:", np.sum(y_val))
    print("Testing set length:", len(X_test), "; label 1 count:", np.sum(y_test))

    # Support Vector Machine (SVM) model with grid search for hyperparameter tuning
    svm = SVC(random_state=0)

    param_dist = {
        "C": [
            1e-6,
            1e6,
            "log-uniform",
            0.001,
            0.01,
            0.1,
            1,
        ],
        # "penalty": ["l1", "l2"],
        # "lose": ["hinge", "squared_hinge"],
        "class_weight": ["balanced", class_weights],
        # "gamma": ["auto"],
        "max_iter": [10, 100, 1000, 10000],
    }

    scoring = {
        "roc_score": make_scorer(roc_auc_score),
        "f1_weighted": make_scorer(f1_score),
    }

    grid_search = GridSearchCV(
        svm,
        n_jobs=-1,
        param_grid=param_dist,
        cv=2,
        scoring=scoring,
        verbose=0,
        refit="roc_score",
    )

    grid_search.fit(X_train, y_train)

    # Display the best estimator from grid search
    print(grid_search.best_estimator_, ":best estimator of SVM")

    # Use the best model for evaluation
    model = grid_search.best_estimator_

    # Display training, validation, and testing scores
    print("Total training score:", model.score(X_train, y_train))
    print("Total validation score:", model.score(X_val, y_val))
    print("Total testing score:", model.score(X_test, y_test))
    if len(class_weights) == 2:
        # ROC curve for testing set
        predictions = model.predict(X_test)
        false_positive_rate, recall, thresholds = roc_curve(y_test, predictions)
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

    # Classification Report
    viz = ClassificationReport(model, title="Training Report of SVM", cmap="PuBu")
    viz.fit(X_train, y_train)
    viz.score(X_train, y_train)
    viz.show()
    fig = viz.poof()
    fig.figure.savefig(
        "Results/SVM/Output/Training Report of SVM "
        + "Number of results"
        + str(len(class_weights))
        + ".png"
    )

    # ROC
    viz = ROCAUC(model, title="Training ROC of SVM")
    viz.fit(X_train, y_train)
    viz.score(X_train, y_train)
    viz.poof()
    fig = viz.poof()
    fig.figure.savefig(
        "Results/SVM/Output/Training ROC of SVM "
        + "Number of results"
        + str(len(class_weights))
        + ".png"
    )

    # Classification Report
    viz = ClassificationReport(model, title="Test Report of SVM", cmap="PuBu")
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.poof()
    fig = viz.poof()
    fig.figure.savefig(
        "Results/SVM/Output/Test Report of SVM"
        + "Number of results"
        + str(len(class_weights))
        + ".png"
    )

    # ROC
    viz = ROCAUC(model, title="Test ROC of SVM")
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.poof()
    fig = viz.poof()
    fig.figure.savefig(
        "Results/SVM/Output/Test ROC of SVM "
        + "Number of results"
        + str(len(class_weights))
        + ".png"
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
    fig.figure.savefig("Results/LGBM/Output/Test Result of LGBM" + ".png")

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    coef = np.maximum(model.coef_, -model.coef_)

    # 检查 coef 的形状
    print(coef.shape)  # 应该打印出 (1, n_features)，其中 n_features 是特征的数量

    # 标准化 coef
    coef_scaled = scaler.fit_transform(coef)

    # 创建一个 DataFrame 来保存标准化后的特征重要性
    feature_importances = pd.DataFrame(coef_scaled, columns=X_train.columns)

    # 打印标准化后的特征重要性
    print(feature_importances)

    feature_importances.to_csv(
        "Results/SVM/Importance/SVM_featureimportances "
        + "Number of results"
        + str(len(class_weights))
        + ".csv"
    )

    # Save the trained model
    # 保存模型
    with open("Results/SVM/SVM_model.pickle", "wb") as f:
        pickle.dump(model, f)

    # Evaluation scores
    eval_scores = {
        "training_score": model.score(X_train, y_train),
        "validation_score": model.score(X_val, y_val),
        "testing_score": model.score(X_test, y_test),
        "roc_auc": roc_auc,
    }

    return model, eval_scores
