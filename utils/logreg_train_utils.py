import os
import numpy as np
import pandas as pd


def generate_csv_file(weights, name):
    """
    Save model weights to a CSV file.
    """
    output_folder = "generated_files"
    os.makedirs(output_folder, exist_ok=True)

    file_path = os.path.join(output_folder, name)
    rows = []
    for house, params in weights.items():
        row = {"House": house, "intercept": params["intercept"]}
        for i, coef_value in enumerate(params["coef"]):
            row[f"coef_{i}"] = coef_value
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)
    print(f"Weights saved to {file_path}")


def scale(X: np.ndarray):
    """
    Scale an array to the [0, 1] range.
    """
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    scaled = (X - x_min) / (x_max - x_min)
    return scaled, x_min, x_max


def predict_values(intercept: float,
                   coef: np.ndarray,
                   X: np.ndarray) -> np.ndarray:
    """
    Predict output for input matrix X using linear model parameters.
    """
    try:
        intercept = float(intercept)
        return intercept + X @ coef
    except Exception as err:
        raise ValueError("Invalid intercept or coefficient") from err


def calculate_error(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Compute prediction error: predicted - true.
    """
    if true.shape != pred.shape:
        raise ValueError("Data length mismatch\
                          between true and predicted arrays")
    return pred - true


def gradient_descent(
    X: np.ndarray,
    alpha: float,
    error: np.ndarray,
    intercept: float,
    coef: np.ndarray,
):
    """
    Perform one step of gradient descent update.
    """
    d_intercept = np.mean(error)
    d_coef = np.mean(error[:, None] * X, axis=0)

    intercept -= alpha * d_intercept
    coef -= alpha * d_coef
    return intercept, coef


def train_model(X: np.ndarray, Y: np.ndarray, alpha: float, iterations: int):
    """
    Train a linear model using gradient descent
    until convergence or max iteration.
    """
    X_array = np.array(X, dtype=float)
    Y_array = np.array(Y, dtype=float)

    if X_array.shape[0] != Y_array.shape[0]:
        raise ValueError("Data length mismatch between X and Y")

    intercept = 0.0
    coef = np.zeros(X_array.shape[1])
    epsilon = 1e-6
    mse_old = float("inf")

    scaled_X, x_min, x_max = scale(X_array)
    scaled_Y, y_min, y_max = scale(Y_array)

    for i in range(iterations):
        y = predict_values(intercept, coef, scaled_X)
        error = calculate_error(scaled_Y, y)
        mse = np.mean(error ** 2) / 2

        if abs(mse_old - mse) < epsilon:
            break

        mse_old = mse
        intercept, coef = gradient_descent(scaled_X,
                                           alpha, error,
                                           intercept, coef)

    coef_original = coef * (y_max - y_min) / (x_max - x_min)
    intercept_original = (
        y_min + intercept * (y_max - y_min) - np.sum(coef_original * x_min)
    )

    return intercept_original, coef_original, i + 1


def get_coeficients(df: pd.DataFrame):
    """
    Compute trained model coefficients per
    Hogwarts House using gradient descent.

    Automatically increases max iterations until convergence
    or until a hard maximum limit.
    """
    if "Hogwarts House" not in df.columns:
        raise ValueError("The dataframe must contain\
                          a 'Hogwarts House' column.")

    houses = df["Hogwarts House"].unique()
    data = df.drop(columns=["Hogwarts House"]).to_numpy(dtype=float)

    alpha = 0.1
    iterations = 1000
    max_iterations = 50000

    weights = {}

    for house in houses:
        y = (df["Hogwarts House"] == house).astype(int).to_numpy()

        while True:
            intercept, coef, steps = train_model(data, y, alpha, iterations)

            if steps < iterations or iterations >= max_iterations:
                break

            iterations *= 2

        weights[house] = {
            "intercept": intercept,
            "coef": coef,
        }

    return weights
