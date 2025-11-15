import pandas as pd
import argparse
import sys
from pathlib import Path

def load_data(filepath, index_col, label_col):
    """
    Loads a CSV file, selecting only the index and label columns.
    Sets the index and handles potential errors.
    """
    try:
        print(f"Loading '{filepath}'...")
        df = pd.read_csv(filepath)

        # Check if required columns exist
        if index_col not in df.columns:
            print(f"Error: Index column '{index_col}' not found in '{filepath}'.")
            sys.exit(1)
        if label_col not in df.columns:
            print(f"Error: Label column '{label_col}' not found in '{filepath}'.")
            sys.exit(1)

        # --- New duplicate check ---
        # Check for duplicate values in the index column *before* setting it as the index.
        # This is the cause of the 'Reindexing only valid' error during the join.
        if df[index_col].duplicated().any():
            print(f"\n--- ERROR ---")
            print(f"Error: Duplicate values found in the index column ('{index_col}') of file: '{filepath}'.")
            print("Pandas cannot perform a join with non-unique index values.")
            print("Please clean the file and ensure all values in this column are unique.")

            # Show the user which index values are duplicated
            duplicated_indices = df[df[index_col].duplicated(keep=False)][index_col]
            print(f"\nDuplicated {index_col} entries found (showing first 5):")
            print(duplicated_indices.value_counts().head(5))
            print("------\n")
            sys.exit(1)
        # --- End new check ---

        # Select only the needed columns and set the index
        df = df[[index_col, label_col]].set_index(index_col)
        return df

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{filepath}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading '{filepath}': {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Compare prediction files against a truth file and calculate accuracy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Define arguments as requested
    parser.add_argument(
        '--truth',
        type=str,
        required=True,
        help="Path to the ground truth CSV file."
    )
    parser.add_argument(
        '--candidates',
        type=str,
        nargs='+',  # This allows for one or more arguments
        required=True,
        help="One or more file paths for the prediction (candidate) CSVs."
    )
    parser.add_argument(
        '--label',
        type=str,
        default="Hogwarts House",
        help="The name of the column header containing the class/label to compare."
    )
    parser.add_argument(
        '--index_col',
        type=str,
        default="Index",
        help="The name of the index/ID column to join on."
    )
    parser.add_argument(
        '-o', '--output_file',
        type=str,
        default=None,
        help="Optional: Path to save the merged DataFrame (Truth vs. Predictions)."
    )

    args = parser.parse_args()

    # 1. Load the truth file
    truth_df = load_data(args.truth, args.index_col, args.label)
    # Rename the label column to 'Truth' for clarity in the final merged file
    truth_df = truth_df.rename(columns={args.label: 'Truth'})

    all_dfs = [truth_df]
    candidate_names = []

    # 2. Load all candidate files
    for candidate_path in args.candidates:
        # Use the filename (without extension) as the column name
        candidate_name = Path(candidate_path).stem
        candidate_names.append(candidate_name)

        print(f"Processing candidate '{candidate_name}' from '{candidate_path}'...")

        candidate_df = load_data(candidate_path, args.index_col, args.label)
        # Rename the label column to the candidate's name
        candidate_df = candidate_df.rename(columns={args.label: candidate_name})

        all_dfs.append(candidate_df)

    # 3. Create the new combined DataFrame
    # We join on the index, and 'inner' join ensures we only keep
    # rows where the Index exists in ALL files (Truth + all candidates)
    try:
        merged_df = pd.concat(all_dfs, axis=1, join='inner')
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"Error merging DataFrames: {e}")
        print("This can happen if the Index column has duplicate values in one of the files.")
        print("The script tried to pre-check for this, but an error still occurred.")
        print("Please double-check your files for duplicate Index values.")
        print("------\n")
        sys.exit(1)

    if merged_df.empty:
        print("Error: The merged DataFrame is empty.")
        print("This likely means no 'Index' values were common across all files.")
        print(f"Truth file indices: {len(truth_df)}")
        for i, path in enumerate(args.candidates):
            print(f"'{Path(path).stem}' indices: {len(all_dfs[i+1])}")
        sys.exit(1)

    print(f"\nSuccessfully merged {len(merged_df)} common rows.")

    # 4. Calculate and print accuracy metrics
    print("\n--- Accuracy Report ---")
    total_rows = len(merged_df)

    for name in candidate_names:
        # Compare the 'Truth' column to the candidate's column
        correct_predictions = (merged_df['Truth'] == merged_df[name]).sum()
        accuracy = (correct_predictions / total_rows) * 100

        print(f"Model '{name}': {accuracy:.2f}% accuracy ({correct_predictions}/{total_rows})")

    # 5. Save the final merged DataFrame if requested
    if args.output_file:
        try:
            # The index (e.g., "Index") is valuable, so we keep it
            merged_df.to_csv(args.output_file, index=True)
            print(f"\nSaved merged results to '{args.output_file}'")
        except Exception as e:
            print(f"\nError saving output file: {e}")

if __name__ == "__main__":
    main()