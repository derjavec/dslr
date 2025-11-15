import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import argparse
from pathlib import Path

def split_csv(input_file, file_a_path, file_b_path, test_size=0.3, random_state=42):
    """
    Loads a CSV, splits it into two DataFrames (70/30 ratio), and saves them
    to new CSV files, keeping headers.

    Args:
        input_file (str): The filepath of the source CSV.
        file_a_path (str): The output filepath for the 70% split (File A).
        file_b_path (str): The output filepath for the 30% split (File B).
        test_size (float): The proportion to allocate to File B (default is 0.3).
        random_state (int): Seed for the random split to ensure reproducibility.
    """
    try:
        # 1. Load the CSV file into a pandas DataFrame
        print(f"Loading '{input_file}'...")
        df = pd.read_csv(input_file)

        if df.empty:
            print(f"Error: The file '{input_file}' is empty.")
            return

        print(f"Total rows loaded: {len(df)}")

        # 2. Split the DataFrame.
        # 'train_test_split' is a convenient way to get a random stratified split.
        # We'll use 'test_size' for our 30% file (File B).
        df_a, df_b = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )

        # 3. Save the two separate DataFrames to new CSV files.
        # 'index=False' prevents pandas from writing the DataFrame index as a new column.
        print(f"Saving 70% ({len(df_a)} rows) to '{file_a_path}'...")
        df_a.to_csv(file_a_path, index=False)

        print(f"Saving 30% ({len(df_b)} rows) to '{file_b_path}'...")
        df_b.to_csv(file_b_path, index=False)

        print("\nSplit complete!")
        print(f"File A: {file_a_path}")
        print(f"File B: {file_b_path}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{input_file}' contains no data or columns.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Split a CSV file into two files (e.g., 70/30 split).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required argument
    parser.add_argument(
        "input_file",
        type=str,
        help="The filepath of the source CSV file to split."
    )

    # Optional arguments for output files
    parser.add_argument(
        "-a", "--fileA",
        type=str,
        default=None,
        help="Output path for the first split file (default: ./{input_name_stem}_split_A.csv)"
    )
    parser.add_argument(
        "-b", "--fileB",
        type=str,
        default=None,
        help="Output path for the second split file (default: ./{input_name_stem}_split_B.csv)"
    )

    # Optional arguments for split parameters
    parser.add_argument(
        "-s", "--split_size",
        type=float,
        default=0.3,
        help="Proportion for the second file (File B), e.g., 0.3 for a 70/30 split."
    )
    parser.add_argument(
        "-r", "--random_state",
        type=int,
        default=42,
        help="Seed for the random split for reproducibility."
    )

    args = parser.parse_args()

    # --- Determine output file paths ---
    # Get the stem (filename without extension) of the input file
    og_name_stem = Path(args.input_file).stem

    # Create default names in the current directory, as requested
    default_a = f"./{og_name_stem}_split_A.csv"
    default_b = f"./{og_name_stem}_split_B.csv"

    # Use user-provided path if available, otherwise use the default
    file_a_path = args.fileA if args.fileA is not None else default_a
    file_b_path = args.fileB if args.fileB is not None else default_b

    # Call the main function with parsed arguments
    split_csv(
        args.input_file,
        file_a_path,
        file_b_path,
        args.split_size,
        args.random_state
    )

if __name__ == "__main__":
    main()