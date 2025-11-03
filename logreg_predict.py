import sys
import pandas as pd
import numpy as np

from filter_features import get_filter_features
from split import split
from logreg_train import train_model


def split_and_train(df):
    split(df, 0.8)

    df_test = pd.read_csv("test.csv")

    df_results = df_test['Hogwarts House'].rename('Result')
    df_results.to_csv('results.csv', index='False')
    df_test['Hogwarts House'] = ''

    df_train = pd.read_csv("train.csv")
    train_model(df_train, 'my_weight_split.csv', 'sk_weight_split.csv')
    return df_test

    


def propose_to_test_score(df):
    
    while(True):
        answer = input("Hogwarts House column is not empty so the file does not work for predicting, do you want to erase it? answer yes or no: ")
        answer.strip().lower()
        if not answer == 'yes' or answer == 'no':
            print("Come on.. answer yes or no please")
        if answer == 'no':
            raise AssertionError('Then import ./dataset/dataet_test.csv as you should')
        if answer == 'yes':
            print("ok, i'll make a copy with an empty 'Hogwarts House' column and use it to predict adding the score")
            df_cpy = split_and_train(df)
            break
    return df_cpy


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_predictions(f_features, df, df_weights):
    
    df_califications = df.drop(['Index', 'Hogwarts House'], axis=1, errors='ignore')
    df_califications = df_califications.select_dtypes(include='number')
    df_califications.replace('', np.nan, inplace=True)
    df_califications.fillna(df_califications.mean(), inplace=True)
      
    X_califications = df_califications[f_features.columns]
    intercepts = df_weights[['House', 'intercept']].to_numpy()
    
    df_weights.drop('intercept', axis=1, inplace=True)
    house_coef = df_weights.to_numpy()
    houses = df_weights['House'].unique()
    df_pred = pd.DataFrame(columns=houses)
    for h in houses:
        intercept = intercepts[intercepts[:,0] == h][0, 1]
        coef = house_coef[house_coef[:, 0] == h][0, 1:].astype(float)
        pred = X_califications @ coef + float(intercept)
        prob = sigmoid(pred)
        df_pred[h] = prob
    house_predictions = df_pred.idxmax(axis=1)
    return house_predictions

def get_weights(test):
    if test:
        my_csv_name = 'my_weight_split.csv'
        sk_csv_name = 'sk_weight_split.csv'
    else:
        my_csv_name = 'my_weights.csv'
        sk_csv_name = 'sk_weights.csv'
    
    df_my_weights = pd.read_csv(my_csv_name)
    df_sk_weights = pd.read_csv(sk_csv_name)
    
    if df_my_weights.empty or df_sk_weights.empty:
        raise ValueError('Please run the logreg_train.py first')
    
    return df_my_weights, df_sk_weights


def create_csv_results(test, my_predictions, sk_predictions):
    df_comparisson = pd.DataFrame({"My prediction" : my_predictions, "Sk prediction" : sk_predictions})
    df_comparisson['match sk'] = (df_comparisson['My prediction'] == df_comparisson['Sk prediction']).astype(int)
    if test:
        df_result = pd.read_csv('results.csv')
        df_comparisson['Result'] = df_result['Result']
        df_comparisson['match results'] = (df_comparisson['My prediction'] == df_comparisson['Result']).astype(int)
        score = df_comparisson['match results'].sum() / len(df_comparisson)
        print(f"score using my weights and the results:{score}")
    score = df_comparisson['match sk'].sum() / len(df_comparisson)
    print(f"score using my weights and sk weights:{score}")

def get_features(test):
    if test:
        csv_name = 'train.csv'
    else:
        csv_name = './datasets/dataset_train.csv'

    df_train = pd.read_csv(csv_name)
    f_features = get_filter_features(df_train)
    if 'Hogwarts House' in f_features.columns:
        f_features.drop('Hogwarts House', axis=1,inplace=True)
    return f_features

def main():
    if len(sys.argv) != 2:
        raise ValueError('Usage: please execute with the dataset path')
    
    df = pd.read_csv(sys.argv[1])
    test = False
    if 'Hogwarts House' in df.columns:
        df['Hogwarts House'].replace("", np.nan, inplace=True)
        if not df['Hogwarts House'].isna().all():
            test = True
            df = propose_to_test_score(df)
    else:
        raise ValueError("The dataframe must contain a 'Hogwarts House' column.")

    df_my_weights, df_sk_weights = get_weights(test)
    f_features = get_features(test)
    my_predictions = generate_predictions(f_features, df, df_my_weights)
    sk_predictions = generate_predictions(f_features, df, df_sk_weights)
    
    create_csv_results(test, my_predictions, sk_predictions)
    

if __name__ == '__main__':
    main()