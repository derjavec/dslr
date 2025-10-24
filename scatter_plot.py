import matplotlib.pyplot as plt
import sys
import pandas as pd
from pandas.plotting import scatter_matrix


def scatter_plot(df: pd.DataFrame):
    """
    Plots a scatter matrix for the numeric features of the dataset
    and annotates the two most similar features.
    """
    feature_1, feature_2 = analyse_correlation(df)

    fig, axes = plt.subplots(figsize=(12, 12))
    axes = scatter_matrix(df, figsize=(12, 12), ax=axes)

    for ax in axes.flatten():
        ax.xaxis.label.set_rotation(45)
        ax.yaxis.label.set_rotation(0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.suptitle("Scatter Matrix of Hogwarts Courses", fontsize=16)

    fig.text(
        0.95,
        0.05,
        f"Most similar features: {feature_1} & {feature_2}",
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
    )

    plt.show()


def analyse_correlation(df: pd.DataFrame):
    """
    Finds the two features with the highest
    correlation (excluding self-correlation).
    """
    df_unstacked = df.unstack()
    df_unstacked = df_unstacked[df_unstacked < 1]

    value = df_unstacked.max()
    max_corr = df_unstacked.idxmax()
    print(
        f"The two most similar courses are: {max_corr[0]} "
        f"and {max_corr[1]} (corr={value:.3f})"
    )

    return max_corr[0], max_corr[1]


def compare_features(df: pd.DataFrame):
    """
    Computes the correlation matrix for numeric features and
    plots a scatter matrix.
    """
    df_num = df.select_dtypes(include='number')
    df_num = df_num.drop('Index', axis=1)

    corr = df_num.corr()

    scatter_plot(corr)


def main():
    """
    Main function to load dataset and visualize scatter matrix.
    """
    if len(sys.argv) != 2:
        raise ValueError('Usage: please execute with the dataset path')

    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)
    compare_features(df)


if __name__ == '__main__':
    main()
