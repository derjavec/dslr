Hogwarts House Prediction
Overview

This project is a machine learning pipeline that predicts the Hogwarts House for students based on numerical features extracted from a dataset. It uses custom logistic regression weights and optionally compares predictions with scikit-learn's logistic regression implementation.

The project also includes feature selection based on statistical description and multicollinearity analysis, ensuring that the model trains on the most relevant features.

Features

Train-test split with reproducible shuffling

Feature selection based on:

Mean and median dispersion

Standard deviation and interquartile range

Multicollinearity analysis

Custom logistic regression implementation

Comparison with scikit-learn's logistic regression

Generates CSV files with predictions and evaluation scores

Fully reproducible results

Requirements

Python 3.13+

pandas

numpy

scikit-learn

You can install the dependencies using:

pip install pandas numpy scikit-learn

Usage

Prepare your dataset
Your CSV file should include a Hogwarts House column and numeric features. The column names must match the training dataset.

Run the prediction script

python logreg_predict.py <path_to_dataset>


If the Hogwarts House column is not empty, the script will ask whether to erase it for prediction.

Output

results.csv: original Hogwarts House values for comparison (if test dataset)

Console output:

Predictions from custom weights

Predictions from scikit-learn weights

Score against actual results (if test dataset)

Score comparison between your weights and scikit-learn weights

File Structure
/datasets
    dataset_train.csv       # Training dataset
    dataset_test.csv        # Test dataset (optional)

filter_features.py          # Feature selection functions
logreg_train.py             # Train logistic regression weights
logreg_predict.py           # Generate predictions and compare scores
split.py                    # Split dataset into train/test

Example
python logreg_predict.py ./datasets/dataset_test.csv


Output:

score using my weights and the results: 0.971875
score using my weights and scikit-learn weights: 1.0

Notes

Feature selection is performed using the training set only to prevent data leakage.

The script supports comparison between custom logistic regression and scikit-learn's logistic regression.

Future warnings regarding multi_class in scikit-learn and inplace operations in pandas are harmless but can be fixed for cleaner output.

License

MIT License
