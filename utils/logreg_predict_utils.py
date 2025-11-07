import pandas as pd
import numpy as np
import os

from utils.filter_features import get_filter_features
from utils.split import split
from logreg_train import train_model


def split_and_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split dataset into 80% train / 20% test, train models
    and prepare test file.
    """
    split(df, 0.8)

    df_test = pd.read_csv("generated_files/test.csv")

    df_results = df_test['Hogwarts House'].rename('Result')

    df_results.to_csv('generated_files/results.csv', index=False)

    df_test['Hogwarts House'] = ''
    df_test.to_csv("generated_files/test.csv", index=False)

    df_train = pd.read_csv("generated_files/train.csv")
    train_model(df_train, 'my_weight_split.csv', 'sk_weight_split.csv')

    return df_test


def propose_to_test_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    If dataset has labels, ask user whether to erase
    them and prepare training split.
    """
    if df['Hogwarts House'].isna().all():
        return df

    while True:
        answer = input(
            "Hogwarts House column is not empty. Do you want to erase it "
            "and prepare a proper test set? (yes/no): "
        ).strip().lower()

        if answer not in ("yes", "no"):
            print("Please answer 'yes' or 'no'.")
            continue

        if answer == 'no':
            raise AssertionError(
                "Then please import './dataset/dataset_test.csv' as required."
            )

        print("Preparing split and erasing labels...")
        return split_and_train(df)


def sigmoid(x: float) -> float:
    """
    Compute sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


def get_weights(test: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load weight CSV files for custom and sklearn logistic regression.
    """
    my_csv = 'generated_files/my_weight_split.csv'\
             if test else 'generated_files/my_weights.csv'
    sk_csv = 'generated_files/sk_weight_split.csv'\
             if test else 'generated_files/sk_weights.csv'

    df_my = pd.read_csv(my_csv)
    df_sk = pd.read_csv(sk_csv)

    if df_my.empty or df_sk.empty:
        raise ValueError("Please run logreg_train.py first.")

    return df_my, df_sk


def create_csv_results(
    test: bool,
    my_predictions: list[str],
    sk_predictions: list[str]
) -> None:
    """
    Save CSV comparing custom vs sklearn predictions and compute accuracy.
    """
    df_cmp = pd.DataFrame({
        "My prediction": my_predictions,
        "SK prediction": sk_predictions
    })
    df_cmp['match sk'] = (
        df_cmp['My prediction'] == df_cmp['SK prediction']
    ).astype(int)

    score_sk = df_cmp['match sk'].mean()
    print(f"Accuracy compared to sklearn: {score_sk:.4f}")

    if test:
        df_result = pd.read_csv('generated_files/results.csv')
        df_cmp['Result'] = df_result['Result']
        df_cmp['match results'] = (
            df_cmp['My prediction'] == df_cmp['Result']
        ).astype(int)

        score_res = df_cmp['match results'].mean()
        print(f"Accuracy compared to ground truth: {score_res:.4f}")

    output_folder = "generated_files"
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, "predictions_comparison.csv")
    df_cmp.to_csv(file_path, index=False)


def get_features(test: bool) -> pd.DataFrame:
    """
    Load training dataset and extract filtered feature columns.
    """
    csv_name = 'generated_files/train.csv'\
               if test else './datasets/dataset_train.csv'
    df_train = pd.read_csv(csv_name)

    features = get_filter_features(df_train)

    if 'Hogwarts House' in features.columns:
        features = features.drop(columns='Hogwarts House')

    return features
