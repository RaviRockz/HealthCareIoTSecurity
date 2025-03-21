import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data():
    dp1 = "Data/environmentMonitoring.csv"
    dp2 = "Data/patientMonitoring.csv"
    dp3 = "Data/Attack.csv"
    print("[INFO] Loading Data From")
    df1 = pd.read_csv(dp1)
    df1.drop("label", axis=1, inplace=True)
    df2 = pd.read_csv(dp2)
    df2.drop("label", axis=1, inplace=True)
    df3 = pd.read_csv(dp3)
    df3.drop("label", axis=1, inplace=True)
    df = pd.concat([df1, df2, df3])
    print("[INFO] Data Shape :: {0}".format(df.shape))
    return df


def preprocess_data(df: pd.DataFrame):
    for c in df.columns:
        if df[c].dtype == "object":
            if c != "class":
                df[c] = df[c].astype("category")
                df[c] = df[c].cat.codes
    df = pd.concat(
        [
            df[df["class"] == "environmentMonitoring"].head(10000),
            df[df["class"] == "patientMonitoring"].head(10000),
            df[df["class"] == "Attack"].head(10000),
        ]
    ).sample(frac=1)
    x = df.values[:, :-1]
    print("[INFO] Scaling Data Using Min-Max Normalization")
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    df[df.columns.values[:-1]] = x
    sp = "Data/preprocessed.csv"
    print("[INFO] Saving Preprocessed Data To :: {0}".format(sp))
    df.to_csv(sp, index=False)
    print("[INFO] Data Shape :: {0}".format(df.shape))
    return df


if __name__ == "__main__":
    preprocess_data(load_data())
