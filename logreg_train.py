import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from filter_features import get_filter_features
from logistic_regresion import get_coeficients


def get_sk_coeficients(df, classes):
    data = df.drop(columns=['Hogwarts House']).to_numpy(dtype=float)
    cls_to_id = {cls: i for i, cls in enumerate(classes)}
    index = df['Hogwarts House'].map(cls_to_id).to_numpy()
    model = LogisticRegression(max_iter=5000, multi_class='multinomial', solver='lbfgs')
    model.fit(data, index)
    
    weights = {}
    for i, h in enumerate(classes):
        weights[h] = {
            'intercept': model.intercept_[i],
            'coef': model.coef_[i]
        }
    return weights

def generate_csv_file(weights, name):
    rows = []
    for house, params in weights.items():
        row = {'House': house, 'intercept': params['intercept']}
        for i, c in enumerate(params['coef']):
            row[f'coef{i}'] = c
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df)
    df.to_csv(name)

def main():
    if len(sys.argv) != 2:
        raise ValueError('Usage: please execute with the dataset path')

    df = pd.read_csv(sys.argv[1])

    if 'Hogwarts House' not in df.columns:
        raise ValueError("The dataframe must contain a 'Hogwarts House' column.")
    classes = df['Hogwarts House'].unique()
    df_f_features = get_filter_features(df)

    weights = get_coeficients(df_f_features)
    sk_weights = get_sk_coeficients(df_f_features, classes)
    generate_csv_file(weights, "my_weights")
    generate_csv_file(sk_weights, "sk_weights")


if __name__ == '__main__':
    main()