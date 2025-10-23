import pandas as pd
import sys
import matplotlib.pyplot as plt
from describe import mean, ft_min
from sklearn.preprocessing import MinMaxScaler

def plot_histogram(df):
    plt.bar(df.index, df.values)
    plt.ylabel("Scaled Dispersion (0 = most homogeneous)")
    plt.title("Homogeneity of Hogwarts Courses Between Houses")
    plt.show()

def get_courses_mean(df):
    houses = df['Hogwarts House'].unique()
    df_num = df.select_dtypes(include = 'number')
    course_mean = pd.DataFrame(index = houses, columns = df_num.columns.drop('Index'))
    for h in houses:
        df_house = df[df['Hogwarts House'] == h]
        df_num_house = df_house.select_dtypes(include = 'number')
        for col in df_num_house.columns:
            course_mean.loc[h, col] = mean(df_num_house, col)
    print(course_mean)
    course_dispersion = course_mean.max() - course_mean.min()
    course_dispersion = course_dispersion.drop('Index')
    course_dispersion_scaled = (course_dispersion - course_dispersion.min()) / (course_dispersion.max() - course_dispersion.min())
    print(course_dispersion_scaled)
    plot_histogram(course_dispersion_scaled)



def main():
    if len(sys.argv) != 2:
        raise ValueError('Usage: please execute with the dataset path')
    df = pd.read_csv(sys.argv[1])
    get_courses_mean(df)

if __name__ == '__main__':
    main()