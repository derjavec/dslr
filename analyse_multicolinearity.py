import pandas as pd
import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def calculate_vif(df):
    X = add_constant(df)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(X.shape[1])]
    return vif_data


def analyse_multicolinearity(f_features, df):
    df_f_features = df[f_features].copy()
    df_f_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_f_features.fillna(df_f_features.mean(), inplace=True)
    while True:
        df_vif = calculate_vif(df_f_features)
        df_vif_nc = df_vif[df_vif['feature'] != 'const']
        max_vif = df_vif_nc['VIF'].max()
        if max_vif < 5:
            break
        max_vif_feature = df_vif_nc.loc[df_vif['VIF'] == max_vif, 'feature'].values[0]
        df_f_features.drop(columns=max_vif_feature, inplace=True)
    df_f_features['Hogwarts House'] = df['Hogwarts House']
    return df_f_features
