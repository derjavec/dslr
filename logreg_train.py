import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils.filter_features import get_filter_features
from utils.logreg_train_utils import get_coeficients, generate_csv_file


def get_sk_coeficients(df, classes):
    """
    Train a sklearn LogisticRegression model and extract coefficients
    grouped by Hogwarts House.
    """

    if "Hogwarts House" not in df.columns:
        raise ValueError("The dataframe must\
                          contain a 'Hogwarts House' column.")

    data = df.drop(columns=["Hogwarts House"]).to_numpy(dtype=float)
    cls_to_id = {cls: i for i, cls in enumerate(classes)}
    index = df["Hogwarts House"].map(cls_to_id).to_numpy()

    model = LogisticRegression(max_iter=5000, solver="lbfgs")
    model.fit(data, index)

    weights = {}
    for i, house in enumerate(classes):
        weights[house] = {
            "intercept": model.intercept_[i],
            "coef": model.coef_[i],
        }

    return weights


def train_model(df, my_weights_csv_name, sk_weights_csv_name):
    """
    Generate CSV files with weight parameters learned from:
    - custom implementation (my_weights_csv)
    - sklearn implementation (sk_weights_csv)
    """

    if "Hogwarts House" not in df.columns:
        raise ValueError("The dataframe must contain\
                          a 'Hogwarts House' column.")

    classes = df["Hogwarts House"].unique()
    df_filtered = get_filter_features(df)

    weights = get_coeficients(df_filtered)
    sk_weights = get_sk_coeficients(df_filtered, classes)

    generate_csv_file(weights, my_weights_csv_name)
    generate_csv_file(sk_weights, sk_weights_csv_name)


def main():
    """
    Program entry point.
    """

    if len(sys.argv) != 2:
        raise ValueError("Usage: python train_logreg.py <dataset.csv>")

    dataset_path = sys.argv[1]
    df = pd.read_csv(dataset_path)

    train_model(df, "my_weights.csv", "sk_weights.csv")


if __name__ == "__main__":
    main()
