import pandas as pd
import os


def split(df: pd.DataFrame, train_frac: float = 0.8) -> None:
    """
    Split a DataFrame into training and test sets, then save them as CSV files.

    """

    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be a float between 0 and 1.")

    df_shuffled = df.sample(frac=1, random_state=42)
    train_size = int(len(df_shuffled) * train_frac)

    df_train = df_shuffled.iloc[:train_size]
    df_val = df_shuffled.iloc[train_size:]
    output_folder = "generated_files"
    os.makedirs(output_folder, exist_ok=True)
    file_path_train = os.path.join(output_folder, "train.csv")
    file_path_test = os.path.join(output_folder, "test.csv")
    df_train.to_csv(file_path_train, index=False)
    df_val.to_csv(file_path_test, index=False)
