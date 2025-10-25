import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from describe import describe


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


def scatter_plot_matrix(df):
    if 'Hogwarts House' not in df.columns:
        raise ValueError("The dataframe must contain a 'Hogwarts House' column.")

    house_colors = {
        'Gryffindor': '#7F0909',
        'Slytherin': '#0D6217',
        'Ravenclaw': '#000A90',
        'Hufflepuff': '#EEE117'
    }
    df_num = df.select_dtypes(include='number')
    if 'Index' in df_num.columns:
        df_num = df_num.drop(columns='Index')
    df_num['Hogwarts House'] = df['Hogwarts House']

    pair_grid = sns.pairplot(
        data = df_num,
        hue = 'Hogwarts House',
        palette = house_colors,

    )

    plt.suptitle("Pair Plot of Hogwarts Courses by House", y=1.02)
    plt.show()


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


def std_noise(df_des_per_house):
    std = []
    Max = []
    Min = []
    
    for des in df_des_per_house:
        std.append(des.loc['Std'])
        Max.append(des.loc['Max'])
        Min.append(des.loc['Min'])
    
    std_df = pd.DataFrame(std)
    max_df = pd.DataFrame(Max).select_dtypes(include='number')
    min_df = pd.DataFrame(Min).select_dtypes(include='number')
    Range = max_df.max() - min_df.min()
    std_df_max = std_df.max()
    std_df_max = std_df_max.drop('Hogwarts House')
    scaled_std_df_max = std_df_max / Range
    return scaled_std_df_max


def main():
    if len(sys.argv) != 2:
        raise ValueError('Usage: please execute with the dataset path')
    df = pd.read_csv(sys.argv[1])
    # scatter_plot_matrix(df)
    df_des_per_house = get_description_per_house(df)
    mean_data = mean_despertion(df_des_per_house)
    std_data = std_noise(df_des_per_house)


if __name__ == '__main__':
    main()