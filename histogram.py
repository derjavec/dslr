import pandas as pd
import sys
import matplotlib.pyplot as plt

from describe import mean


def plot_histogram(df: pd.Series):
    """
    Plots a bar chart of scaled dispersion values for Hogwarts courses.
    """
    min_value = df.min()
    course = df.idxmin()

    plt.figure(figsize=(16, 8))
    plt.bar(df.index, df.values)
    plt.xticks(rotation=45)
    plt.ylabel("Scaled Dispersion (0 = most homogeneous)")
    plt.title("Homogeneity of Hogwarts Courses Between Houses")

    plt.text(
        0.95,
        0.95,
        f"Most homogeneous: {course}\nValue: {min_value:.2f}",
        horizontalalignment='right',
        verticalalignment='top',
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
    )

    plt.show()


def get_courses_mean(df: pd.DataFrame):
    """
    Computes the mean score for each course per Hogwarts house,
    calculates the dispersion, scales it, and plots it.
    """
    houses = df['Hogwarts House'].unique()
    df_num = df.select_dtypes(include='number')

    course_mean = pd.DataFrame(
        index=houses,
        columns=df_num.columns.drop('Index'),
        dtype=float
    )

    for h in houses:
        df_house = df[df['Hogwarts House'] == h]
        df_num_house = df_house.select_dtypes(include='number')
        for col in df_num_house.columns:
            course_mean.loc[h, col] = mean(df_num_house, col)

    course_dispersion = course_mean.max() - course_mean.min()
    course_dispersion = course_dispersion.drop('Index')
    course_dispersion_scaled = (
        (course_dispersion - course_dispersion.min()) /
        (course_dispersion.max() - course_dispersion.min())
    )

    plot_histogram(course_dispersion_scaled)


def main():
    """
    Main function: load dataset and compute course dispersion.
    """
    if len(sys.argv) != 2:
        raise ValueError('Usage: please execute with the dataset path')

    df = pd.read_csv(sys.argv[1])
    get_courses_mean(df)


if __name__ == '__main__':
    main()
