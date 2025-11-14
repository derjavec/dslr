import pandas as pd
import sys

from typing import cast

def count(df, col) -> int:
    """Return the number of non-NaN values in the specified column."""
    total = 0
    for value in df[col]:
        if not pd.isna(value):
            total += 1
    return total


def mean(df, col) -> float:
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


def std(df, col) -> float:
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


def exc_kurtosis_bonus(df: pd.DataFrame, col) -> float:
    """ Returns the Excess-kurtosis of a distribution"""
    data = cast(pd.Series, df[col]).dropna()
    n = len(data)
    if n < 4:
        return float('nan')

    std_ = std(df, col)
    if  std_ == 0:
        return float('nan')
    mean_ = mean(df, col)

    # G2 ​= (n(n+1))/((n−1)(n−2)(n−3))  *  ∑​[((x​−μ)/s) ** 4]  -  (3(n−1)**2)​ / ((n−2)(n−3))
    #    = ∑​[(x​−μ)**4] / s**4  *  (n(n+1))/((n−1)(n−2)(n−3)) - (3(n−1)**2)​ / ((n−2)(n−3))
    #    = ((∑​[(x​−μ)**4] / s**4)  *  ((n**2+n) / (n−1)(n−2)(n−3)) ) - (3(n−1)**2)​ / (n−2)(n−3)
    #    = (∑​[(x​−μ)**4] * (n**2+n) / (n−1)(n−2)(n−3)*s**4 ) - (3(n−1)**2)​ / (n−2)(n−3)
    summed = sum( (x - mean_)**4 for x in data )
    e = (n-1) * (n-2) * (n-3) * std_**4
    excess = (3 * (n - 1)**2) / ((n-2) * (n-3))
    kurtosis = (summed * (n**2 + n)) / e - excess
    return kurtosis

# Skewness explains the asymetry of a distribution
# Positive Skew (Skew > 0): The distribution has a tail on the right the mean is greater than the median.
# Negative Skew (Skew < 0): The distribution has a tail on the left. The mean is less than the median.
# Zero Skew (Skew ≈ 0): The distribution is mostly symmetrical. The mean and median are close.
def skewness_bonus(df: pd.DataFrame, col):
    data = cast(pd.Series, df[col]).dropna()
    n = len(data)
    if n < 3:
        return float('nan')

    std_ = std(df, col)
    if  std_ == 0:
        return float('nan')
    mean_ = mean(df, col)

    # G1 ​= N/((N−1)(N−2)) * ∑​[ ((x​−μ) / s) **3 ]
    #    = N * ∑​[ ((x​−μ) / s) **3 ] / ((N−1)(N−2))
    #    = N * ∑​[ ((x​−μ)**3 / s**3] ) / ((N−1)(N−2))
    #    = N * ∑​[(x​−μ)**3] / s**3 / ((N−1)(N−2))
    #    = N * ∑​[(x​−μ)**3] / ((N−1)(N−2) s**3)

    summed = sum( (x - mean_)**3 for x in data )
    e = (n-1) * (n-2) * (std_**3)
    skew = n * summed / e
    return skew


def describe(df, stats):
    """Compute a manual description table for the DataFrame."""
    df_num = df.select_dtypes(include='number')
    if 'Index' in df_num.columns:
        df_num = df_num.drop(columns='Index')
    df_describe = pd.DataFrame(index=stats, columns=df_num.columns)

    func = {
        'Count': count,
        'Mean': mean,
        'Std': std,
        'Min': ft_min,
        '25%': q25,
        '50%': q50,
        '75%': q75,
        'Max': ft_max,
        'Kurt': exc_kurtosis_bonus,
        'Skew': skewness_bonus,
    }

    for col in df_describe.columns:
        for key in stats:
            df_describe.loc[key, col] = func[key](df_num, col)

    # To compare with pandas functions
    # for col in df_describe.columns:
    #     df_describe.loc['pdKurt', col] = df_num[col].kurt()
    #     df_describe.loc['pdSkew', col] = df_num[col].skew()

    return df_describe


def main():
    """Main function to load the dataset and display description."""
    if len(sys.argv) != 2:
        raise ValueError('Usage: please execute with the dataset path')

    df = pd.read_csv(sys.argv[1])
    stats = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'Kurt', 'Skew']
    df_des = describe(df, stats)
    print(df_des)


if __name__ == '__main__':
    main()
