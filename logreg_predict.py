import sys
import pandas as pd
import numpy as np

from utils.logreg_predict_utils import (
    get_weights,
    get_features,
    sigmoid,
    propose_to_test_score,
    create_csv_results
)


def generate_predictions(
    filtered_features: pd.DataFrame,
    df: pd.DataFrame,
    df_weights: pd.DataFrame
) -> pd.Series:
    """
    Generate model predictions for each Hogwarts house.
    """
    df_scores = df.drop(['Index', 'Hogwarts House'], axis=1, errors='ignore')
    df_scores = df_scores.select_dtypes(include='number')
    df_scores.replace('', np.nan, inplace=True)
    df_scores.fillna(df_scores.mean(), inplace=True)

    X = df_scores[filtered_features.columns].to_numpy(dtype=float)

    df_w = df_weights.copy()

    intercept_data = df_w[['House', 'intercept']].to_numpy()
    df_w = df_w.drop(columns='intercept')

    houses = df_w['House'].unique()
    df_pred = pd.DataFrame(columns=houses)

    for house in houses:
        house_intercept = intercept_data[
            intercept_data[:, 0] == house
        ][0, 1]

        house_coef = df_w[df_w['House'] == house].iloc[0, 1:]\
            .to_numpy(dtype=float)
        scores = X @ house_coef + float(house_intercept)
        prob = sigmoid(scores)
        df_pred[house] = prob

    return df_pred.idxmax(axis=1)


def main() -> None:
    """
    Main CLI entrypoint.

    Steps:
    1. Load CSV file.
    2. Detect whether labels exist.
    3. If needed, ask user whether to erase and create split.
    4. Load weights and features.
    5. Predict with custom and sklearn models.
    6. Save comparison results.
    """
    if len(sys.argv) != 2:
        raise ValueError("Usage: python3 logreg_predict.py <dataset.csv>")

    df = pd.read_csv(sys.argv[1])

    if 'Hogwarts House' not in df.columns:
        raise ValueError("Dataset must contain a 'Hogwarts House' column.")

    test_mode = False
    df['Hogwarts House'].replace("", np.nan, inplace=True)

    if not df['Hogwarts House'].isna().all():
        test_mode = True
        df = propose_to_test_score(df)

    df_my_weights, df_sk_weights = get_weights(test_mode)
    filtered_features = get_features(test_mode)

    my_preds = generate_predictions(filtered_features, df, df_my_weights)
    sk_preds = generate_predictions(filtered_features, df, df_sk_weights)

    create_csv_results(test_mode, my_preds, sk_preds)


if __name__ == '__main__':
    main()
