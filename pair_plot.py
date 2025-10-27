import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from filter_features import get_filter_features


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



def main():
    if len(sys.argv) != 2:
        raise ValueError('Usage: please execute with the dataset path')
    df = pd.read_csv(sys.argv[1])
    df_features = get_filter_features(df)
    scatter_plot_matrix(df)
    scatter_plot_matrix(df_features)


if __name__ == '__main__':
    main()