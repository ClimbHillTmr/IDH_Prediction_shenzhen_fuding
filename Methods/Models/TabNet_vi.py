from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import warnings

warnings.filterwarnings("ignore")

for target in ["新_透中低血压_计算", "透中低血压_计算"]:
    # Load TabNet model
    with open(
        "/Users/cht/Documents/GitHub/IDH_Prediction_shenzhen_fuding/Methods/Models/Results/TabNet/TabNet_model_Number_of_results_target="
        + str(target)
        + ".pickle",
        "rb",
    ) as f:
        tabnet_model = pickle.load(f)

    # Load dataset
    X = pd.read_csv(
        "/Users/cht/Documents/GitHub/IDH_Prediction_shenzhen_fuding/fuding_dataset_"
        + str(target)
        + ".csv"
    )
    X = X.drop(["患者id", "Unnamed: 1"], axis=1)
    y = pd.read_csv(
        "/Users/cht/Documents/GitHub/IDH_Prediction_shenzhen_fuding/fuding_dataset_target_"
        + str(target)
        + ".csv"
    )
    y = y.drop(["患者id", "Unnamed: 1"], axis=1)

    # Split the data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=0, shuffle=True
    )

    # Predict probabilities
    y_pred = tabnet_model.predict_proba(X_test.values)[:, 1]

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve and save to a separate PDF file
    plt.figure(figsize=(8, 8))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (AUC = {:.2f})".format(roc_auc),
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    pdf_filename_roc = "/TabNet/Output/output_roc_curve" + str(target) + ".pdf"
    plt.savefig(pdf_filename_roc)
    plt.close()
    print(f"ROC curve plot saved to {pdf_filename_roc}")

    # Calculate confusion matrix
    y_pred = tabnet_model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix and save to a separate PDF file
    plt.figure(figsize=(6, 6))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
    plt.yticks([0, 1], ["Actual 0", "Actual 1"])
    plt.xlabel("True label")
    plt.ylabel("Predicted label")

    pdf_filename_conf_matrix = (
        "/TabNet/Output/output_confusion_matrix" + str(target) + ".pdf"
    )
    plt.savefig(pdf_filename_conf_matrix)
    plt.close()
    print(f"Confusion matrix plot saved to {pdf_filename_conf_matrix}")
