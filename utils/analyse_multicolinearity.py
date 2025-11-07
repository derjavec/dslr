import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor,
)
from statsmodels.tools.tools import add_constant


def calculate_vif(df):
    """
    Compute Variance Inflation Factor (VIF) for each feature in a
    numerical dataframe.
    """
    X = add_constant(df)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    return vif_data


def analyse_multicolinearity(f_features, df):
    """
    Iteratively remove features with high multicollinearity based on VIF.

    The function removes the feature with the highest VIF until all
    remaining features have VIF < 5. NaN and infinite values are
    replaced to avoid computation issues.
    """
    df_f = df[f_features].copy()
    df_f.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_f.fillna(df_f.mean(), inplace=True)

    while True:
        df_vif = calculate_vif(df_f)
        df_vif_no_const = df_vif[df_vif["feature"] != "const"]
        max_vif = df_vif_no_const["VIF"].max()

        if max_vif < 5:
            break

        max_feature = (
            df_vif_no_const.loc[
                df_vif_no_const["VIF"] == max_vif, "feature"
            ].values[0]
        )
        df_f.drop(columns=max_feature, inplace=True)

    df_f["Hogwarts House"] = df["Hogwarts House"]
    return df_f
