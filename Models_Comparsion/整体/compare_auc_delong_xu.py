import pandas as pd
import numpy as np
import scipy.stats


# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
import numpy as np


def compute_midrank(x):
    """Computes midranks for 4-class classification.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0

    while i < N:
        j = i
        while j < N and np.all(Z[j] == Z[i]):
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j

    T2 = np.empty(N, dtype=float)
    classes = np.unique(Z)

    for c in classes:
        class_indices = np.where(Z == c)[0]
        T2[class_indices] = T[class_indices] + 1

    return T2


# Example usage:
# x = np.array([3, 1, 4, 1, 5, 9, 2, 6])
# midranks = compute_midrank(x)
# print(midranks)


# def fastDeLong(predictions_sorted_transposed, label_1_count):
#     """
#     The fast version of DeLong's method for computing the covariance of
#     unadjusted AUC.
#     Args:
#        predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
#           sorted such as the examples with label "1" are first
#     Returns:
#        (AUC value, DeLong covariance)
#     Reference:
#      @article{sun2014fast,
#        title={Fast Implementation of DeLong's Algorithm for
#               Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
#        author={Xu Sun and Weichao Xu},
#        journal={IEEE Signal Processing Letters},
#        volume={21},
#        number={11},
#        pages={1389--1393},
#        year={2014},
#        publisher={IEEE}
#      }
#     """
#     # Short variables are named as they are in the paper
#     m = label_1_count
#     n = predictions_sorted_transposed.shape[1] - m
#     positive_examples = predictions_sorted_transposed[:, :m]
#     negative_examples = predictions_sorted_transposed[:, m:]
#     k = predictions_sorted_transposed.shape[0]


#     tx = np.empty([k, m], dtype=float)
#     ty = np.empty([k, n], dtype=float)
#     tz = np.empty([k, m + n], dtype=float)
#     for r in range(k):
#         tx[r, :] = compute_midrank(positive_examples[r, :])
#         ty[r, :] = compute_midrank(negative_examples[r, :])
#         tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
#     aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
#     v01 = (tz[:, :m] - tx[:, :]) / n
#     v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
#     sx = np.cov(v01)
#     sy = np.cov(v10)
#     delongcov = sx / m + sy / n
#     return aucs, delongcov

import traceback


def fastDeLong(predictions_sorted_transposed, label_counts):
    """
    The modified version of DeLong's method for computing the covariance of
    unadjusted AUC for 4-class classification.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such that the examples with label "1" are first,
          "2" are second, and so on.
       label_counts: a list containing the number of examples for each class
    Returns:
       (AUC value, DeLong covariance)
    """
    try:
        k = predictions_sorted_transposed.shape[0]
        n_classes = len(label_counts)
        n_total = predictions_sorted_transposed.shape[1]

        # Separate predictions for each class
        class_predictions = [
            predictions_sorted_transposed[
                :, sum(label_counts[:i]) : sum(label_counts[: i + 1])
            ]
            for i in range(n_classes)
        ]

        # Compute midranks for each class
        midranks = [
            [compute_midrank(class_predictions[i][r, :]) for r in range(k)]
            for i in range(n_classes)
        ]

        # Compute AUCs for each class
        aucs = np.empty(k)
        for r in range(k):
            aucs[r] = np.sum(
                [
                    midranks[i][r][: label_counts[i]].sum()
                    / (label_counts[i] * (n_total - label_counts[i]))
                    for i in range(n_classes)
                ]
            )
        print("aucs")
        print(aucs)

        # Compute DeLong covariance
        # Compute DeLong covariance
        delongcov = np.zeros((k, k))
        for r in range(k):
            for s in range(k):
                vrs = [
                    (
                        midranks[i][r][: label_counts[i]]
                        - np.nanmean(midranks[i][r][: label_counts[i]])
                    )
                    / label_counts[i]
                    if label_counts[i] > 1
                    and np.any(
                        midranks[i][r][: label_counts[i]]
                    )  # Check if there are at least two samples for this class and non-empty slice
                    else np.zeros(
                        0
                    )  # If not enough samples or empty slice, use an empty array
                    for i in range(n_classes)
                ]
                vss = [
                    (
                        midranks[i][s][: label_counts[i]]
                        - np.nanmean(midranks[i][s][: label_counts[i]])
                    )
                    / label_counts[i]
                    if label_counts[i] > 1
                    and np.any(
                        midranks[i][s][: label_counts[i]]
                    )  # Check if there are at least two samples for this class and non-empty slice
                    else np.zeros(
                        0
                    )  # If not enough samples or empty slice, use an empty array
                    for i in range(n_classes)
                ]

                # Debugging: Print relevant information
                print(f"r={r}, s={s}, vrs={vrs}, vss={vss}")

                # Check if there are non-empty arrays
                if any(vr.size > 0 for vr in vrs) and any(vs.size > 0 for vs in vss):
                    delongcov[r, s] = np.cov(np.concatenate(vrs), np.concatenate(vss))[
                        0, 1
                    ]
                else:
                    # If any array is empty, set the covariance to NaN
                    delongcov[r, s] = np.nan

        return aucs, delongcov

    except Exception as e:
        # Get detailed traceback information
        error_traceback = traceback.format_exc()
        # Raise an exception with detailed error message
        raise ValueError(
            f"Error in fastDeLong function: {e}\n\nTraceback:\n{error_traceback}"
        )


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / (np.sqrt(np.dot(np.dot(l, sigma), l.T)) + 1e-8)
    pvalue = 2 * (1 - scipy.stats.norm.cdf(np.abs(z)))
    #  print(10**(np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)))
    return pvalue


# def compute_ground_truth_statistics(ground_truth):
#     assert np.array_equal(np.unique(ground_truth), [0, 1])
#     order = (-ground_truth).argsort()
#     label_1_count = int(ground_truth.sum())
#     return order, label_1_count


def compute_ground_truth_statistics(ground_truth):
    """
    Computes ground truth statistics for multi-class classification.
    Args:
       ground_truth: np.array of class labels (0, 1, 2, 3, ..., n_classes-1)
    Returns:
       order: indices that sort the ground_truth array in descending order of class labels
       label_counts: a list containing the number of examples for each class
    """
    unique_labels = np.unique(ground_truth)
    n_classes = len(unique_labels)

    order = np.argsort(-ground_truth)  # Sort in descending order
    label_counts = [np.sum(ground_truth == label) for label in unique_labels]

    return order, label_counts


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    print(label_1_count)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    print(predictions_sorted_transposed)
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert (
        len(aucs) == 1
    ), "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of class labels
       predictions_one: predictions of the first model,
          np.array of floats of the probability for each class
       predictions_two: predictions of the second model,
          np.array of floats of the probability for each class
    """
    order, label_counts = compute_ground_truth_statistics(ground_truth)

    # Ensure predictions have the same number of classes as ground_truth
    predictions_one = predictions_one[: len(label_counts)]
    predictions_two = predictions_two[: len(label_counts)]

    # Check that order contains valid indices
    order = np.array(order)
    order = order[
        order < predictions_one.shape[0]
    ]  # Assuming predictions_one and predictions_two have the same shape

    print("predictions_one shape:", predictions_one.shape)
    print("order:", order)

    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[
        :, order
    ]
    print("predictions_sorted_transposed:", predictions_sorted_transposed)
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_counts)
    print("aucs:", aucs)

    return calc_pvalue(aucs, delongcov)
