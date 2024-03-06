import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import os
import tempfile
from sklearn.metrics import confusion_matrix

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]


def make_model(
    X_train,
    metrics=METRICS,
    output_bias=None,
):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential(
        [
            keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[-1],)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
        ]
    )
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )

    return model


def plot_metrics(history):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(
            history.epoch,
            history.history['val_' + metric],
            color=colors[0],
            linestyle="--",
            label='Val',
        )
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 20])
    plt.ylim([80, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def NN_model_train(X_train, X_test, y, y_train, y_test, X_val, y_val, class_weight):
    pos = np.sum(y)
    neg = len(y) - np.sum(y_train)
    model = make_model(X_train, output_bias=np.log([pos / neg]))
    initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
    model.save_weights(initial_weights)

    EPOCHS = 100
    BATCH_SIZE = 2048

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', verbose=1, patience=10, mode='max', restore_best_weights=True
    )

    weighted_model = make_model(X_train=X_train)
    weighted_model.load_weights(initial_weights)

    weighted_history = weighted_model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        validation_data=(X_val, y_val),
        # The class weights go here
        class_weight=class_weight,
    )

    plot_metrics(weighted_history)

    train_predictions_weighted = weighted_model.predict(X_train, batch_size=BATCH_SIZE)
    test_predictions_weighted = weighted_model.predict(X_test, batch_size=BATCH_SIZE)

    weighted_results = weighted_model.evaluate(
        X_test, y_test, batch_size=BATCH_SIZE, verbose=0
    )
    for name, value in zip(weighted_model.metrics_names, weighted_results):
        print(name, ': ', value)
    print()

    plot_cm(y_test, test_predictions_weighted)
