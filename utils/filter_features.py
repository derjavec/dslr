import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from describe import describe
from utils.analyse_multicolinearity import analyse_multicolinearity


def get_description_per_house(df):
    """
    Generate descriptive statistics for each Hogwarts house.
    """
    if "Hogwarts House" not in df.columns:
        raise ValueError("The dataframe must contain a\
                         'Hogwarts House' column.")

    df_num = df.select_dtypes(include="number").copy()
    df_num["Hogwarts House"] = df["Hogwarts House"]

    stats = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    df_des_per_house = []

    for house in df_num["Hogwarts House"].unique():
        df_house = df_num[df_num["Hogwarts House"] == house]
        df_des = describe(df_house, stats)
        df_des.insert(0, "Hogwarts House", house)
        df_des_per_house.append(df_des)

    return df_des_per_house


def mean_dispersion(df_des_per_house):
    """
    Compute dispersion of feature means across houses.
    """
    means = [des.loc["Mean"] for des in df_des_per_house]
    mean_df = pd.DataFrame(means).select_dtypes(include="number")

    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(mean_df),
        columns=mean_df.columns,
        index=mean_df.index,
    )
    return scaled.std()


def median_dispersion(df_des_per_house):
    """
    Compute dispersion of feature medians across houses.
    """
    medians = [des.loc["50%"] for des in df_des_per_house]
    median_df = pd.DataFrame(medians).select_dtypes(include="number")

    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(median_df),
        columns=median_df.columns,
        index=median_df.index,
    )
    return scaled.std()


def scale(df_des_per_house, df):
    """
    Scale a vector by feature ranges across houses.
    """
    max_list = [des.loc["Max"] for des in df_des_per_house]
    min_list = [des.loc["Min"] for des in df_des_per_house]

    max_df = pd.DataFrame(max_list).select_dtypes(include="number")
    min_df = pd.DataFrame(min_list).select_dtypes(include="number")

    value_range = max_df.max() - min_df.min()
    return df / value_range


def std_noise(df_des_per_house):
    """
    Compute noise score from standard deviations across houses.
    """
    std_values = [des.loc["Std"] for des in df_des_per_house]
    std_df = pd.DataFrame(std_values)

    if "Hogwarts House" in std_df.columns:
        std_df = std_df.drop(columns=["Hogwarts House"])

    max_std = std_df.max()
    return scale(df_des_per_house, max_std)


def q_outliers(df_des_per_house):
    """
    Measure outlier risk using IQR across houses.
    """
    q25 = [des.loc["25%"] for des in df_des_per_house]
    q75 = [des.loc["75%"] for des in df_des_per_house]

    q25_df = pd.DataFrame(q25)\
        .select_dtypes(include="number").reset_index(drop=True)
    q75_df = pd.DataFrame(q75)\
        .select_dtypes(include="number").reset_index(drop=True)

    iqr = q75_df - q25_df
    max_iqr = iqr.max()
    return scale(df_des_per_house, max_iqr)


def filter_features_by_description(df_des_per_house):
    """
    Select features that show clear discrimination between houses
    and low noise/outliers.
    """
    mean_data = mean_dispersion(df_des_per_house)
    median_data = median_dispersion(df_des_per_house)
    std_data = std_noise(df_des_per_house)
    iqr_data = q_outliers(df_des_per_house)

    keep_mean = mean_data[mean_data >= 0.45]
    keep_median = median_data[median_data >= 0.45]
    keep_std = std_data[std_data <= 0.15]
    keep_iqr = iqr_data[iqr_data <= 0.15]

    features = (
        keep_mean.index
        .intersection(keep_median.index)
        .intersection(keep_std.index)
        .intersection(keep_iqr.index)
    )
    return features


def get_filter_features(df):
    """
    Filter columns using statistical dispersion and multicollinearity removal.
    """
    df_des_per_house = get_description_per_house(df)
    features = filter_features_by_description(df_des_per_house)
    return analyse_multicolinearity(features, df)


def main():
    """CLI entrypoint."""
    if len(sys.argv) != 2:
        raise ValueError("Usage: please execute with the dataset path.")

    df = pd.read_csv(sys.argv[1])
    selected = get_filter_features(df)
    print(selected)


if __name__ == "__main__":
    main()
