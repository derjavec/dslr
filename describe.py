import pandas as pd
import sys


def describe(df):
    stats = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    df_des = pd.DataFrame(index=stats, columns=df.columns.drop('Index'))
    func = {
        "Count":count,
        "Mean":mean,
        "Std":std,
        "Min":ft_min
        "25%", "50%", "75%": quartile,
        "max": ft_max
    }
    
    for col in df_des.columns:
        for key in stats:
            if stats[key] in func:
                result = func[stats[key]]()
        




def main():
    if len(sys.argv) != 2:
        raise ValueError("Usage: please excecute with the dataset path")
    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)
    df_num = df.select_dtypes(include = 'number')
    # print(df_num)
    df_describe = describe(df_num)
    # print(df_describe)


if __name__ == "__main__" :
    main()