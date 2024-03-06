from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.augmentations import ClassificationSMOTE
from sklearn.preprocessing import MinMaxScaler
import torch
import os
import pickle
import numpy as np
import warnings
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix
from pylab import mpl

warnings.filterwarnings("ignore")
mpl.rcParams["font.sans-serif"] = ["Arial Unicode MS"]

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))


def TabNet_model(
    X_train, X_test, y_train, y_test, X_val, y_val, class_weights, target, kinds
):
    """
    Train and evaluate a TabNet model.

    Parameters:
    - X_train, X_test, X_val: Training, testing, and validation data.
    - y_train, y_test, y_val: Training, testing, and validation labels.
    - class_weights: Class weights for handling class imbalance.

    Returns:
    - model: Trained TabNet model.

    """
    aug = ClassificationSMOTE(p=0.5)

    features_col = X_train.columns

    X_train = X_train.values
    y_train = y_train.values

    X_val, y_val = X_val.values, y_val.values
    X_test, y_test = X_test.values, y_test.values

    # define the model
    TabNet = TabNetClassifier(
        n_d=18,
        n_a=18,
        optimizer_fn=torch.optim.RAdam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={
            "step_size": 16,  # how to use learning rate scheduler
            "gamma": 0.1,
        },
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax",  # "sparsemax"
        seed=0,
        lambda_sparse=1e-6,
    )

    # fit the model
    TabNet.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_name=["train", "valid"],
        eval_metric=["auc", "balanced_accuracy"],
        max_epochs=1000,
        patience=100,
        batch_size=64,
        virtual_batch_size=16,
        num_workers=0,
        weights=class_weights,
        drop_last=False,
        augmentations=aug,  # aug, None
    )

    model = TabNet

    print(
        "Total testing",
        recall_score(
            y_test,
            model.predict(X_test),
            average="weighted",
        ),
    )
    print("Total testing", confusion_matrix(y_test, model.predict(X_test)))

    # Save the TabNet model
    with open(
        os.path.join(
            current_dir,
            "Results/TabNet/TabNet_model_Number_of_results_target="
            + str(target)
            + "_"
            + str(kinds)
            + ".pickle",
        ),
        "wb",
    ) as f:
        pickle.dump(model, f)

    importance = model.feature_importances_

    scaler = MinMaxScaler()

    scaler = scaler.fit(importance.reshape(-1, 1))

    importances = scaler.transform(importance.reshape(-1, 1))

    importances = pd.DataFrame(
        importances.reshape(1, len(features_col)), columns=features_col
    )

    importances.to_csv(
        os.path.join(
            current_dir,
            "Results/TabNet/Importance/TabNet_feature target = "
            + str(target)
            + "_"
            + str(kinds)
            + " importances "
            + ".csv",
        ),
    )

    # Evaluation scores
    eval_scores = {
        # "training_score": model.score(X_train, y_train),
        # "validation_score": model.score(X_val, y_val),
        # "testing_score": model.score(X_test, y_test),
    }

    return model, eval_scores
