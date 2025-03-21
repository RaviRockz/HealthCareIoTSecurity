if True:
    from reset_random import reset_random

    reset_random()
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from sklearn.metrics import confusion_matrix

from model import buildModel
from utils import CLASSES

plt.rcParams["font.family"] = "IBM Plex Mono"


def compute_metrics(TP, FP, FN, TN, epoch, class_name):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) != 0
        else 0
    )

    return {
        "Epoch": epoch,
        "Class": class_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1Score": f1_score,
    }


def plot_confusion_matrix(data, y):
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    kys, vls = list(data.keys()), list(data.values())
    metrics = []
    for i, ax in enumerate(axes.flatten()):
        cm = confusion_matrix(vls[i]["pred"], y)
        sbn.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False, ax=ax)
        ax.set_xticklabels(CLASSES)
        ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix : Epoch - {kys[i]}")
        TP, TN, FP, FN = cm[0][0], cm[1][1], cm[1][0], cm[0][1]
        df = pd.DataFrame(
            [
                compute_metrics(TP, FP, FN, TN, vls[i]["epoch"], CLASSES[0]),
                compute_metrics(TN, FN, FP, TP, vls[i]["epoch"], CLASSES[1]),
            ],
        )
        df.loc[3] = [vls[i]["epoch"], "Average", *df.values[:, 2:].mean(axis=0)]
        metrics.append(df)
    fig.tight_layout()
    fig.savefig("results/confusion_matrix.png", bbox_inches="tight")
    df = pd.concat(metrics)
    df = df.round(4)
    df.to_csv("results/metrics.csv", index=False)


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    df = pd.read_csv("Data/features/features_selected.csv")
    df["class"].replace(
        {"environmentMonitoring": 0, "patientMonitoring": 1, "Attack": 2},
        inplace=True,
    )
    x, y = df.values[:, :-1], df.values[:, -1]
    x = np.expand_dims(x, axis=1)
    dat = {}
    keys = [500, 1000, 1500, 2000, 2500, 3000]
    for i, j in enumerate(np.linspace(1, 500, 6, dtype=int)):
        model = buildModel(x.shape[1:], len(CLASSES))
        model.load_weights(f"models/model_{j:03d}.h5")
        prob = np.array(model.predict(x, verbose=0))
        dat[keys[i]] = {
            "prob": prob,
            "pred": np.argmax(prob, axis=1),
            "epoch": keys[i],
        }
    plot_confusion_matrix(dat, y)
