import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
import string

plt.rcParams["font.family"] = "IBM Plex Mono"
warnings.filterwarnings("ignore", category=Warning)


def plot_corr(df: pd.DataFrame):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    sbn.heatmap(
        df.corr(),
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 9},
        ax=ax,
        xticklabels=string.ascii_uppercase[:df.shape[1]],
        yticklabels=string.ascii_uppercase[:df.shape[1]],
        cmap="Purples",
        square=True,
        cbar=False,
    )
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    fig.savefig("Data/corr.png")


if __name__ == "__main__":
    plot_corr(pd.read_csv("Data/preprocessed.csv"))
