import pandas as pd
import sys


def count(df, col):
    """Return the number of non-NaN values in the specified column."""
    total = 0
    for value in df[col]:
        if not pd.isna(value):
            total += 1
    return total


def mean(df, col):
    """Return the mean of the non-NaN values in the specified column."""
    total = 0
    cnt = 0
    for value in df[col]:
        if not pd.isna(value):
            total += value
            cnt += 1
    if cnt != 0:
        return total / cnt
    return 0


def std(df, col):
    """Return the sample standard deviation of the specified column."""
    m = mean(df, col)
    s = 0
    cnt = 0
    for value in df[col]:
        if not pd.isna(value):
            s += (value - m) ** 2
            cnt += 1
    if cnt <= 1:
        return 0
    s *= 1 / (cnt - 1)
    return s ** 0.5


def ft_min(df, col):
    """Return the minimum value in the specified column, ignoring NaNs."""
    m = None
    for value in df[col]:
        if pd.isna(value):
            continue
        if m is None or value < m:
            m = value
    return m


def ft_max(df, col):
    """Return the maximum value in the specified column, ignoring NaNs."""
    m = None
    for value in df[col]:
        if pd.isna(value):
            continue
        if m is None or value > m:
            m = value
    return m


def quartile(df, col, q):
    """
    Return the q-th quartile of the specified
    column using manual interpolation.
    q should be between 0 and 1 (e.g., 0.25, 0.5, 0.75).
    """
    data = sorted([x for x in df[col] if not pd.isna(x)])
    n = len(data)
    if n == 0:
        return 0
    pos = q * (n - 1)
    low = int(pos)
    high = min(low + 1, n - 1)
    frac = pos - low
    if frac == 0:
        return data[low]
    return data[low] + (data[high] - data[low]) * frac


def q25(df, col):
    """Return the 25th percentile of the column."""
    return quartile(df, col, 0.25)


def q50(df, col):
    """Return the 50th percentile (median) of the column."""
    return quartile(df, col, 0.5)


def q75(df, col):
    """Return the 75th percentile of the column."""
    return quartile(df, col, 0.75)


def describe(df, stats):
    """Compute a manual description table for the DataFrame."""
    df_num = df.select_dtypes(include='number')
    if 'Index' in df_num.columns:
        df_num = df_num.drop(columns='Index')
    df_des = pd.DataFrame(index=stats, columns=df_num.columns)

    func = {
        'Count': count,
        'Mean': mean,
        'Std': std,
        'Min': ft_min,
        '25%': q25,
        '50%': q50,
        '75%': q75,
        'Max': ft_max
    }

    for col in df_des.columns:
        for key in stats:
            df_des.loc[key, col] = func[key](df_num, col)

    return df_des


def main():
    """Main function to load the dataset and display description."""
    if len(sys.argv) != 2:
        raise ValueError('Usage: please execute with the dataset path')

    df = pd.read_csv(sys.argv[1])
    stats = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    df_des = describe(df, stats)
    print(df_des)


if __name__ == '__main__':
    main()
