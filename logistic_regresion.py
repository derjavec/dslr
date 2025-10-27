import numpy as np

def scale(X: np.ndarray):
    """Scale a numpy array to [0, 1] range
    and return scaled array and bounds."""
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    scaled = (X - x_min) / (x_max - x_min)
    return scaled, x_min, x_max


def predict_values(intercept: float, coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Predict the output values for given inputs using a linear model."""
    try:
        intercept = float(intercept)
        return intercept + X @ coef
    except Exception as err:
        raise ValueError("Invalid intercept or coefficient") from err


def calculate_error(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Calculate the error between true and predicted values."""
    if true.shape != pred.shape:
        raise ValueError(
            "Data length mismatch between true and predicted arrays"
        )
    return pred - true


def gradient_descent(
    X: np.ndarray, alpha: float, error: np.ndarray,
    intercept: float, coef: np.ndarray
):
    """Perform one step of gradient descent for linear regression."""
    d_intercept = np.mean(error)
    d_coef = np.mean(error[:, None] * X, axis=0)

    intercept -= alpha * d_intercept
    coef -= alpha * d_coef
    return intercept, coef


def train_model(
    X: np.ndarray, Y: np.ndarray, alpha: float, iterations: int
):
    """Train a linear regression model using gradient descent."""
    X_array = np.array(X, dtype=float)
    Y_array = np.array(Y, dtype=float)

    if X_array.shape[0] != Y_array.shape[0]:
        raise ValueError("Data length mismatch between X and Y")

    intercept= 0.0
    coef = np.zeros(X_array.shape[1])
    epsilon = 1e-6
    mse_old = float("inf")

    scaled_X_array, x_min, x_max = scale(X_array)
    scaled_Y_array, y_min, y_max = scale(Y_array)

    for i in range(iterations):
        y = predict_values(intercept, coef, scaled_X_array)
        error = calculate_error(scaled_Y_array, y)
        mse = np.mean(error**2) / 2
        if abs(mse_old - mse) < epsilon:
            break
        mse_old = mse
        intercept, coef = gradient_descent(
            scaled_X_array, alpha, error, intercept, coef
        )
    coef_original = coef * (y_max - y_min) / (x_max - x_min)
    intercept_original = y_min + intercept * (y_max - y_min) - np.sum(coef_original * x_min)
    return intercept_original, coef_original, i + 1


def get_coeficients(df):
    """
    Get the trained coefficients (intercept and slope) for the data.

    Automatically increases iterations if convergence is not reached.
    """

    if 'Hogwarts House' not in df.columns:
        raise ValueError("The dataframe must contain a 'Hogwarts House' column.")
    houses = df['Hogwarts House'].unique()
    data = df.drop(columns=['Hogwarts House']).to_numpy(dtype=float)
    
    alpha = 0.1
    iterations = 1000
    max_iterations = 50000

    weights = {}

    for h in houses:
        y = (df['Hogwarts House'] == h).astype(int).to_numpy()

        while True:
            (
                intercept,
                coef,
                i,
            ) = train_model(data, y, alpha, iterations)

            if i < iterations or iterations >= max_iterations:
                break
            iterations *= 2
        weights[h] = {
            'intercept': intercept,
            'coef': coef
        }

    return weights