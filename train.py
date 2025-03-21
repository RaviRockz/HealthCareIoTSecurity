if True:
    from reset_random import reset_random

    reset_random()
import os
import shutil

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from matplotlib import pyplot as plt

from model import buildModel
from utils import CLASSES

plt.rcParams["font.family"] = "IBM Plex Mono"


def train(df):
    reset_random()
    df["class"].replace(
        {"environmentMonitoring": 0, "patientMonitoring": 1, "Attack": 2},
        inplace=True,
    )
    x, y = df.values[:, :-1], df.values[:, -1]

    x = np.expand_dims(x, axis=1)
    y_cat = to_categorical(y, len(CLASSES))
    print("[INFO] X Shape :: {0}".format(x.shape))
    print("[INFO] Y Shape :: {0}".format(y.shape))

    model_dir = "models"
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.h5")
    model = buildModel(x.shape[1:], len(CLASSES))

    print("[INFO] Fitting Data")
    model.fit(
        x,
        y_cat,
        validation_data=(x, y_cat),
        batch_size=1024,
        epochs=500,
        verbose=1,
        callbacks=[
            ModelCheckpoint(
                model_path,
                save_best_only=True,
                save_weights_only=True,
                monitor="val_accuracy",
                mode="max",
                verbose=False,
            ),
            ModelCheckpoint(
                filepath=os.path.join(model_dir, "model_{epoch:03d}.h5"),
                save_weights_only=True,
                save_freq="epoch",
                monitor="val_accuracy",
                mode="max",
                verbose=False,
            ),
        ],
    )


if __name__ == "__main__":
    train(pd.read_csv("Data/features/features_selected.csv"))
