import pandas as pd


def split(df, train_frac):
    df_shuffled = df.sample(frac=1, random_state=42)
    train_size = int(len(df_shuffled) * train_frac)
    df_train = df_shuffled.iloc[:train_size]
    df_val = df_shuffled.iloc[train_size:]
    df_train.to_csv("train.csv", index=False)
    df_val.to_csv("test.csv", index=False)