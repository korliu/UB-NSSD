import argparse
import os
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import result
import sklearn.metrics as sk_metrics
import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    f1_score,
    fbeta_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelBinarizer

image_folder = Path("images")
image_folder.mkdir(exist_ok=True)


def ROC_curve(class_to_id, y_true, y_pred, results, title=""):
    # print(f"true vs. pred:", y_true, y_pred, sep="\n")
    # print(class_to_id)

    id_to_class = {int(v): k for k, v in class_to_id.items()}
    n_classes = len(class_to_id)

    y_true = [id_to_class[class_id] for class_id in y_true]
    y_pred = [id_to_class[class_id] for class_id in y_pred]

    # print(y_true, y_pred)

    target_values = [id_to_class[i] for i in range(3)]

    label_binarizer = LabelBinarizer().fit(target_values)
    y_onehot = label_binarizer.transform(y_true)
    # print(y_onehot_test)
    # print(y_onehot_test.shape)  # (n_samples, n_classes)

    y_target = y_onehot

    scores = [f"{variant}_score" for variant in target_values]
    y_scores = results[scores].to_numpy()

    # y_test = [x == "chewing" and 1 or 0 for x in y_true]
    # y_pred = results["chewing_score"]
    # RocCurveDisplay.from_predictions(y_test, y_pred)

    # y_test = [x == "biting" and 1 or 0 for x in y_true]
    # y_pred = results["biting_score"]
    # RocCurveDisplay.from_predictions(y_test, y_pred)

    # y_test = [x == "swallow" and 1 or 0 for x in y_true]
    # y_pred = results["swallow_score"]
    # RocCurveDisplay.from_predictions(y_test, y_pred)
    # plt.show()

    # print("NEXT")

    print(results)
    # print(scores)
    # print(y_scores)
    # y_scores2 = results[["biting_score", "chewing_score", "swallow_score"]].to_numpy()

    # print(list(y_target), y_scores)

    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_target[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")

    fig, ax = plt.subplots(figsize=(6, 6))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_target[:, class_id],
            y_scores[:, class_id],
            name=f"ROC curve for {id_to_class[class_id]}",
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 2),
        )

    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"One-vs-Rest multiclass ROC for {title}")
    plt.legend()
    plt.savefig(Path(image_folder, f"{title}_multi_ovr_roc.png"))


def confusion_matrix(class_to_id, y_true, y_pred, results, title=""):
    display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    display.plot()

    confusion_matrices = sk_metrics.multilabel_confusion_matrix(y_true, y_pred)
    # for i in zip(class_to_id.keys(), confusion_matrices):
    #     print(f"{i[0]} -> Confusion Matrix: {i[1]}")

    plt.title(f"Confusion Matrix for {title}")
    plt.savefig(Path(image_folder, f"{title}_confusion_matrix.png"))


def get_f1_score(class_to_id, y_true, y_pred, title):
    f1_scores = dict()

    average_types = ["micro", "macro", "weighted"]

    for a in average_types:
        score = f1_score(y_true, y_pred, average=a, pos_label=None)
        f1_scores[a] = score

    f1_score_data = f"F1 Score for {title}: {f1_scores} \n"
    with open(Path("outputs", "f1_scores.txt"), "a+") as f:
        f.write(f1_score_data)

    return f1_scores


def fbeta_score(class_to_id, y_true, y_pred):
    pass
