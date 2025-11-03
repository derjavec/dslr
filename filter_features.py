import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from describe import describe
from analyse_multicolinearity import analyse_multicolinearity


def get_description_per_house(df):
    if 'Hogwarts House' not in df.columns:
        raise ValueError("The dataframe must contain a 'Hogwarts House' column.")

    df_num = df.select_dtypes(include='number')
    df_num['Hogwarts House'] = df['Hogwarts House']
    houses = df_num['Hogwarts House'].unique()
    stats = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    df_des_per_house = []
    for h in houses:
        df_house = df_num[df_num['Hogwarts House'] == h]
        df_des = describe(df_house, stats)
        df_des.insert(0, 'Hogwarts House', h)
        df_des_per_house.append(df_des)
    return df_des_per_house


def mean_despertion(df_des_per_house):

    mean = []
    for des in df_des_per_house:
        mean.append(des.loc['Mean'])
    mean_df = pd.DataFrame(mean)
    mean_df_num = mean_df.select_dtypes(include='number')
    scaler = MinMaxScaler()
    scaled_mean_df = pd.DataFrame(
        scaler.fit_transform(mean_df_num),
        columns=mean_df_num.columns,
        index=mean_df.index
    )
    dispersion = scaled_mean_df.std()
    return dispersion

def median_despertion(df_des_per_house):

    median = []
    for des in df_des_per_house:
        median.append(des.loc['50%'])
    median_df = pd.DataFrame(median)
    median_df_num = median_df.select_dtypes(include='number')
    scaler = MinMaxScaler()
    scaled_median_df = pd.DataFrame(
        scaler.fit_transform(median_df_num),
        columns=median_df_num.columns,
        index=median_df.index
    )
    dispersion = scaled_median_df.std()
    return dispersion

def scale(df_des_per_house, df):
    Max = []
    Min = []

    for des in df_des_per_house:
        Max.append(des.loc['Max'])
        Min.append(des.loc['Min'])
    max_df = pd.DataFrame(Max).select_dtypes(include='number')
    min_df = pd.DataFrame(Min).select_dtypes(include='number')
    Range = max_df.max() - min_df.min()
    scaled_df = df / Range
    return scaled_df


def std_noise(df_des_per_house):
    std = []
    
    for des in df_des_per_house:
        std.append(des.loc['Std'])
    
    std_df = pd.DataFrame(std)
    std_df_max = std_df.max()
    std_df_max = std_df_max.drop('Hogwarts House')
    scaled_std_df_max = scale(df_des_per_house, std_df_max)
    return scaled_std_df_max


def q_outliers(df_des_per_house):
    q25 = []
    q75 = []

    for des in df_des_per_house:
        q25.append(des.loc['25%'])
        q75.append(des.loc['75%'])
    
    q75_df = pd.DataFrame(q75).select_dtypes(include='number')
    q25_df = pd.DataFrame(q25).select_dtypes(include='number')
    
    iqr_df = q75_df.reset_index(drop=True) - q25_df.reset_index(drop=True)
    
    iqr_df_max = iqr_df.max()

    scaled_iqr_df_max = scale(df_des_per_house, iqr_df_max)
    return scaled_iqr_df_max


def filter_features_by_description(df_des_per_house):
    mean_data = mean_despertion(df_des_per_house)
    median_data = median_despertion(df_des_per_house)
    std_data = std_noise(df_des_per_house)
    iqr_data = q_outliers(df_des_per_house)

    f_mean_data = mean_data[mean_data >= 0.45]
    f_median_data = median_data[median_data >= 0.45]
    f_std_data = std_data[std_data <= 0.15]
    f_iqr_data = iqr_data[iqr_data <= 0.15]

    common_features = f_mean_data.index.intersection(f_median_data.index)
    common_features = common_features.intersection(f_std_data.index)
    common_features = common_features.intersection(f_iqr_data.index)

    return common_features



def get_filter_features(df):
    df_des_per_house = get_description_per_house(df)
    common_features = filter_features_by_description(df_des_per_house)
    f_features = analyse_multicolinearity(common_features, df)
    
    return f_features

def main():
    if len(sys.argv) != 2:
        raise ValueError('Usage: please execute with the dataset path')
    df = pd.read_csv(sys.argv[1])
    f_features = get_filter_features(df)
    print(f_features)


if __name__ == '__main__':
    main()