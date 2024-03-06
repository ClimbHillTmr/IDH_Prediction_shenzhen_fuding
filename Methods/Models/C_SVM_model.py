# -*- coding: utf-8 -*-
import pickle
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.svm import SVC
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix
from pylab import mpl
import warnings

warnings.filterwarnings("ignore")
mpl.rcParams["font.sans-serif"] = ["Arial Unicode MS"]

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))


def SVM_model(
    X_train, X_test, y_train, y_test, X_val, y_val, class_weights, target, kinds
):
    """
    Train and evaluate an SVM model.

    Parameters:
    - X_train, X_test, X_val: Training, testing, and validation data.
    - y_train, y_test, y_val: Training, testing, and validation labels.
    - class_weights: Class weights for handling class imbalance.

    Returns:
    - model: Trained SVM model.

    """
    model = SVC(
        class_weight=class_weights,
        random_state=0,
        # Add other SVM parameters as needed
    )

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
        "max_iter": [
            10,
            100,
            1000,
        ],
    }

    # 定义评估指标
    scoring = {
        "recalled_weighted": make_scorer(
            recall_score,
            # needs_proba=True,
            # average="weighted",
        ),
        "f1_weighted": make_scorer(
            f1_score,
            # needs_proba=True,
            # average="weighted",
        ),
    }

    grid_search = GridSearchCV(
        model,
        n_jobs=-1,
        param_grid=param_dist,
        cv=3,
        scoring=scoring,
        # verbose=False,
        refit="f1_weighted",
    )

    grid_search.fit(
        X_train,
        y_train,
        # eval_set=[(X_val, y_val)],
        # verbose=False,
        # eval_metric="roc",
        # early_stopping_rounds=10,
    )

    grid_search.fit(X_train, y_train)

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

    # Save the SVM model
    with open(
        os.path.join(
            current_dir,
            "Results/SVM/SVM_model_Number_of_results_target="
            + str(target)
            + "_"
            + str(kinds)
            + ".pickle",
        ),
        "wb",
    ) as f:
        pickle.dump(model, f)

    # Classification Report
    viz = ClassificationReport(model, title="Test Report of SVM", cmap="PuBu")
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.poof()
    fig = viz.poof()
    fig.figure.savefig(
        os.path.join(
            current_dir,
            "Results/SVM/Output/Test Report of SVM target="
            + str(target)
            + "_"
            + str(kinds)
            + ".pdf",
        )
    )

    # Confusion Matrix
    cm = ConfusionMatrix(model, title="Test Result of SVM")
    cm.fit(X_train, y_train)
    cm.score(X_test, y_test)
    cm.poof()
    fig = cm.poof()
    fig.figure.savefig(
        os.path.join(
            current_dir,
            "Results/SVM/Output/Test Result of SVM target="
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
    }

    return model, eval_scores
