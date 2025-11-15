import numpy as np
import pandas as pd
import argparse
import os
from typing import Literal, Tuple, List, Optional, Sequence, Dict
from traceback import print_exc

# --- SKLEARN IMPORTS ---
# We'll need these for the main function
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


def get_features_test(df):
    kept_feats = ["Index", "Hogwarts House", "Herbology" ,"Defense Against the Dark Arts" ,"Divination" ,"Muggle Studies" ,"History of Magic" ,"Transfiguration"]
    features = [col for col in df.columns if col in kept_feats or col == "Hogwarts House"]
    return df[features]

def get_features_essential():
    return ["Herbology" ,"Defense Against the Dark Arts" ,"Divination" ,"Muggle Studies" ,"History of Magic" ,"Transfiguration"]

def get_filter_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the dataframe to keep only the specified features.
    """
    kept_feats = ["Hogwarts House", "Herbology" ,"Defense Against the Dark Arts" ,"Divination" ,"Muggle Studies" ,"History of Magic" ,"Transfiguration"]
    features = [col for col in df.columns if col in kept_feats or col == "Hogwarts House"]
    return df[features]

def scale(X: np.ndarray):
    """
    Scale an array to the [0, 1] range.
    """
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    scaled = (X - x_min) / (x_max - x_min)
    return scaled, x_min, x_max

class LogRegBuilder:

    __slots__ = (
        "X_array", "Y_array",
        "intercept", "coef",
        "scaled_X", "x_min", "x_max",
        "scaled_Y", "y_min", "y_max",
    )

    def __init__(self,
        X: np.ndarray,
        Y: np.ndarray,
        initialize: Literal["xavier", "rand"] = "xavier"
    ):
        self.X_array: np.ndarray = np.array(X, dtype=float)
        self.Y_array: np.ndarray = np.array(Y, dtype=float)

        if self.X_array.shape[0] != self.Y_array.shape[0]:
            raise ValueError("Data length mismatch between X and Y")

        self.intercept: float = 0.0
        self.coef: np.ndarray = np.zeros(self.X_array.shape[1])

        self.scaled_X, self.x_min, self.x_max = scale(self.X_array)
        self.scaled_Y, self.y_min, self.y_max = scale(self.Y_array)

        if initialize == "xavier":
            self._xavier_initialization()
        elif initialize == "rand":
            self._rand_initialization()
        else:
            raise ValueError("Invalid initialization method")

    def _xavier_initialization(self) -> None:
        limit = np.sqrt(6 / (self.X_array.shape[1] + 1))
        self.coef = np.random.uniform(-limit, limit, size=self.X_array.shape[1])

    def _rand_initialization(self) -> None:
        self.coef = np.random.normal(0, 1, size=self.X_array.shape[1])

    def _batch_gradient_descent(self,
        x_batch: np.ndarray,
        err: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Calculates gradients for the entire batch.
        err is (pred - true)
        """
        d_intercept = np.mean(err)
        d_coef = np.mean(err[:, None] * x_batch, axis=0)
        return d_intercept, d_coef

    def _stochastic_gradient_descent(self,
        x_batch: np.ndarray,
        err: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        d_intercept = err[0]
        d_coef = err[0] * x_batch[0]
        return d_intercept, d_coef

    def _mini_batch_gradient_descent(self,
        x_batch: np.ndarray,
        err: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        d_intercept = np.mean(err)
        d_coef = np.mean(err[:, None] * x_batch, axis=0)
        return d_intercept, d_coef

    def get_loss(self, y_mask: Optional[np.ndarray] = None, p: Optional[np.ndarray] = None) -> float:
        """
        Compute the Log Loss (Binary Cross-Entropy).
        """
        if y_mask is None or p is None:
            z = self.intercept + self.scaled_X @ self.coef
            p = 1 / (1 + np.exp(-z) + 1e-8) # Sigmoid
            y_mask = self.scaled_Y

        # Clip p to avoid log(0)
        p_clipped = np.clip(p, 1e-9, 1 - 1e-9)

        loss = -np.mean(
            y_mask * np.log(p_clipped) + (1 - y_mask) * np.log(1 - p_clipped) # type: ignore
        )
        return loss

    def train(self,
        epochs: int,
        alpha: float,
        method: Literal["batch", "stochastic", "mini-batch"] = "mini-batch",
        batch_size: int = 32
    ) -> Tuple[np.ndarray, float, int]:

        def _get_batch():
            if method == "batch":
                return self.scaled_X, self.scaled_Y
            elif method == "stochastic":
                idx = np.random.randint(0, self.scaled_X.shape[0])
                return self.scaled_X[idx:idx+1], self.scaled_Y[idx:idx+1]
            elif method == "mini-batch":
                idx = np.random.choice(
                    self.scaled_X.shape[0],
                    size=min(batch_size, self.scaled_X.shape[0]),
                    replace=False
                )
                return self.scaled_X[idx], self.scaled_Y[idx]
            else:
                raise ValueError("Invalid gradient descent method")

        fgrad = {
            "batch": self._batch_gradient_descent,
            "stochastic": self._stochastic_gradient_descent,
            "mini-batch": self._mini_batch_gradient_descent,
        }.get(method)
        if fgrad is None:
            raise ValueError("Invalid gradient descent method")

        loss = float("inf")
        itterations = epochs
        for e in range(epochs):

            x_mask, y_mask = _get_batch()
            z: float = self.intercept + x_mask @ self.coef
            p = 1 / (1 + np.exp(-z) + 1e-8) # Sigmoid
            err = p - y_mask  # Gradient

            loss_ = self.get_loss(y_mask, p)
            if abs(loss - loss_) < 1e-8:
                itterations = e + 1
                break
            loss = loss_

            intercept_grad, coef_grad = fgrad(x_mask, err)
            self.intercept -= alpha * intercept_grad
            self.coef -= alpha * coef_grad

        # Un-scale the parameters
        coef_original = self.coef * (self.y_max - self.y_min) / (self.x_max - self.x_min)
        intercept_original = (
            self.y_min + self.intercept * (self.y_max - self.y_min) - np.sum(coef_original * self.x_min)
        )
        return coef_original, intercept_original, itterations


class Perceptron:

    def __init__(self, coef: np.ndarray, intercept: float):
        self.coefs = coef
        self.intercept = intercept
        self.accuracy_history: List[float] = []

    def train(self,
        X: np.ndarray,
        Y: np.ndarray,
        alpha: float = 0.01,
        epochs: int = 1000,
        initialize: Literal["xavier", "rand"] = "xavier",
        method: Literal["batch", "stochastic", "mini-batch"] = "mini-batch",
        batch_size: int = 32
    ):
        # We must initialize X in the builder, even if coef shape is 0
        if X.shape[1] == 0:
             raise ValueError("Cannot train model with 0 features.")

        builder = LogRegBuilder(X, Y, initialize=initialize)
        self.coefs, self.intercept, itterations = builder.train(
            epochs, alpha, method=method, batch_size=batch_size
        )
        self.accuracy_history.append(builder.get_loss())

    def get_accuracy(self):
        if not self.accuracy_history:
            raise ValueError("Model has not been trained yet.")
        return self.accuracy_history[-1]

    def predict(self, X: np.ndarray) -> float:
        """
        Predict probability for a SINGLE input feature row X.
        """
        # Ensure X is a 1D array (a single row)
        if X.ndim != 1:
            raise ValueError(f"Perceptron.predict expected a 1D array (a single row), but got {X.ndim} dimensions.")

        # Ensure feature counts match
        if X.shape[0] != self.coefs.shape[0]:
            raise ValueError(f"Feature count mismatch. Model expects {self.coefs.shape[0]} features, row has {X.shape[0]}.")

        z = self.intercept + X @ self.coefs # Matrix multiplication, sums into a scalar
        return 1 / (1 + np.exp(-z) + 1e-8) # Sigmoid

class LogRegOvAClassifier:

    @classmethod
    def from_file(cls, path: str, class_col_name: str) -> 'LogRegOvAClassifier':
        """
        Load model weights from a CSV file.
        """
        df = pd.read_csv(path)
        classes = df["Class"].tolist()
        classifier = cls(classes, class_col_name)

        # Find all coefficient columns
        coef_cols = sorted(
            [col for col in df.columns if col.startswith("Coef_")],
            key=lambda c: int(c.split('_')[1]) # Sort them numerically
        )

        for _, row in df.iterrows():
            cls_name = row["Class"]
            intercept = row["Intercept"]
            coef = np.array([row[col] for col in coef_cols])
            classifier.models[cls_name] = Perceptron(coef=coef, intercept=intercept)
        return classifier

    def to_file(self, path: str) -> None:
        """
        Save the model weights to a CSV file.
        """
        data = []
        for cls, model in self.models.items():
            row = {"Class": cls, "Intercept": model.intercept}
            for i, coef in enumerate(model.coefs):
                row[f"Coef_{i}"] = coef
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    def __init__(self, classes: List[str], class_col_name: str):
        self.classes = classes
        self.class_col_name = class_col_name
        self.models = {
            cls: Perceptron(coef=np.zeros(0), intercept=0.0)
            for cls in classes
        }

    def train(self, df: pd.DataFrame, method: Literal["batch", "stochastic", "mini-batch"] = "mini-batch"):
        """
        Train one model for each class.
        """
        features = df.drop(columns=[self.class_col_name]).columns
        base_X = df[features].to_numpy(dtype=float)

        # Initialize all models with the correct coef shape
        n_features = base_X.shape[1]
        if n_features == 0:
            raise ValueError("No features found to train on.")

        for cls in self.classes:
            self.models[cls] = Perceptron(coef=np.zeros(n_features), intercept=0.0)

        for k in self.classes:
            print(f"  Training model for {k}...")
            # 1. For each class, set the target to 1 if it matches the class, else 0
            Y = (df[self.class_col_name] == k).astype(float).to_numpy(dtype=float)

            # 2. Now train
            model = self.models[k]
            model.train(base_X, Y, alpha=0.1, epochs=10000, method=method, batch_size=32)

    def predict(self, X: np.ndarray) -> str:
        """
        Predicts the class name for a row of features
        """
        if X.ndim != 1:
            raise ValueError(f"Perceptron.predict expected a 1D array (a single row), but got {X.ndim} dimensions.")

        probs = []
        for cls in self.classes:
            model = self.models[cls]
            probs.append(model.predict(X))
        max_val = 0
        max_idx = 0
        for idx, val in enumerate(probs):
            if val > max_val:
                max_val = val
                max_idx = idx
        return self.classes[max_idx]


### Clanker made functions to make pretty comparison between custom and sklearn models ###

def train_custom_model(df, save_path: str, method: Literal["batch", "stochastic", "mini-batch"] = "mini-batch"):
    """
    Train and save the custom LogRegOvAClassifier.
    """
    if "Hogwarts House" not in df.columns:
        raise ValueError("The dataframe must contain 'Hogwarts House' column.")

    classes = df["Hogwarts House"].unique()
    df_filtered: pd.DataFrame = get_filter_features(df)

    classifier = LogRegOvAClassifier(classes, "Hogwarts House")
    classifier.train(df_filtered, method=method)
    classifier.to_file(save_path)
    print(f"Custom model ({method}) trained and saved to {save_path}")


def train_sklearn_model(df: pd.DataFrame, save_path: str):
    """
    Train and save an SKLearn model with the *same custom format*.
    """
    class_col = "Hogwarts House"
    if class_col not in df.columns:
        raise ValueError(f"The dataframe must contain '{class_col}' column.")

    classes = df[class_col].unique()
    df_filtered = get_filter_features(df)

    X = df_filtered.drop(columns=[class_col]).to_numpy(dtype=float)
    Y = df_filtered[class_col].to_numpy(dtype=str)

    # Scale X
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Use 'saga' solver (good for this) and match epochs
    base_model = LogisticRegression(solver='saga', max_iter=10000, random_state=42)
    model = OneVsRestClassifier(base_model)
    model.fit(X_scaled, Y)

    # --- Extract weights and save in custom format ---
    data = []
    for i, cls_name in enumerate(model.classes_):
        estimator = model.estimators_[i]
        intercept = estimator.intercept_[0]

        # We need to un-scale the parameters to match the custom model's output
        # This is the reverse of the logic in LogRegBuilder.train
        # coef = coef_scaled * (y_max - y_min) / (x_max - x_min)
        # intercept = y_min + intercept_scaled * (y_max - y_min) - sum(coef * x_min)

        # Our custom model scales Y from 0 to 1, so y_min=0, y_max=1
        y_min, y_max = 0, 1
        x_min, x_max = scaler.data_min_, scaler.data_max_

        # Handle division by zero if a feature is constant
        scale_ = (y_max - y_min) / (x_max - x_min + 1e-9)
        coef_unscaled = estimator.coef_[0] * scale_

        intercept_unscaled = (
            y_min + intercept * (y_max - y_min) - np.sum(coef_unscaled * x_min)
        )

        row = {"Class": cls_name, "Intercept": intercept_unscaled}
        for j, coef_val in enumerate(coef_unscaled):
            row[f"Coef_{j}"] = coef_val
        data.append(row)

    df_weights = pd.DataFrame(data)
    df_weights.to_csv(save_path, index=False)
    print(f"SKLearn model trained and saved to {save_path}")


def test_models(df_test: pd.DataFrame, weights_paths: Dict[str, str], class_col_name: str):
    """
    Load all specified models and test their accuracy.
    """
    print("\n--- Running Model Accuracy Tests ---")
    df_filtered = get_features_test(df_test)
    print("Filtered test data shape:", df_filtered.shape)

    for name, path in weights_paths.items():
        if not os.path.exists(path):
            print(f"Warning: Weights file not found, skipping {name}: {path}")
            continue
        try:
            results = []
            model = LogRegOvAClassifier.from_file(path, class_col_name)
            for row in df_filtered.iterrows():
                ft_row = row[1].drop(labels=[class_col_name, "Index"]).to_numpy(dtype=float)
                pred_name = model.predict(ft_row)
                results.append((int(row[1]['Index']), pred_name))
            header = ["Index", "Hogwarts House"]
            the_dir = "my_predictions"
            os.makedirs(the_dir, exist_ok=True)
            with open(f"{the_dir}/model_predictions_{name}.csv", "w") as f:
                f.write(",".join(header) + "\n")
                for index, house in results:
                    f.write(f"{index},{house}\n")
        except Exception as e:
            print(f"Error testing model {name}: {e}\n{print_exc()}")

def main():
    parser = argparse.ArgumentParser(description="Train and/or test Logistic Regression models.")
    parser.add_argument("train_data", type=str, help="Path to the training CSV file.")
    parser.add_argument("test_data", type=str, help="Path to the testing CSV file.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "all"],
        default="all",
        help="Mode of operation: 'train' only, 'test' only, or 'all'."
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="models",
        help="Directory to save/load model weights."
    )
    args = parser.parse_args()

    # Ensure weights directory exists
    os.makedirs(args.weights_dir, exist_ok=True)

    # Define file paths
    weights_files = {
        "My_Model_(Batch)": os.path.join(args.weights_dir, "my_weights_batch.csv"),
        "My_Model_(Mini-Batch)": os.path.join(args.weights_dir, "my_weights_mini.csv"),
        "My_Model_(Stochastic)": os.path.join(args.weights_dir, "my_weights_stoch.csv"),
        "SKLearn_Model": os.path.join(args.weights_dir, "sk_weights.csv"),
    }
    class_col = "Hogwarts House"

    if args.mode in ["train", "all"]:
        print("--- 🚂 Training Models ---")
        try:
            df_train = pd.read_csv(args.train_data)
            print("Training data shape:", df_train.shape)
            df_train.dropna(inplace=True)
            print("Training data shape post drop:", df_train.shape)

            # Train custom models
            print("Training MyModel (Batch)...")
            train_custom_model(df_train.copy(), weights_files["My_Model_(Batch)"], method="batch")

            print("Training MyModel (Mini-Batch)...")
            train_custom_model(df_train.copy(), weights_files["My_Model_(Mini-Batch)"], method="mini-batch")

            print("Training MyModel (Stochastic)...")
            train_custom_model(df_train.copy(), weights_files["My_Model_(Stochastic)"], method="stochastic")

            # Train sklearn model
            print("Training SKLearn Model...")
            train_sklearn_model(df_train.copy(), weights_files["SKLearn_Model"])

            print("--- All models trained and saved. ---")

        except FileNotFoundError:
            print(f"Error: Training file not found at {args.train_data}")
            if args.mode == "train":
                return # Exit if in train-only mode
        except Exception as e:
            print(f"An error occurred during training: {e}\n{print_exc()}")
            if args.mode == "train":
                return

    if args.mode in ["test", "all"]:
        try:
            df_test = pd.read_csv(args.test_data)
            print("Testing data shape:", df_test.shape)
            all_cols = df_test.columns.tolist()

            exclude_cols = ["Hogwarts House"]
            cols_to_check = [col for col in all_cols if col not in exclude_cols]
            df_test.dropna(subset=cols_to_check, inplace=True)

            print("Testing data shape post drop:", df_test.shape)
            test_models(df_test, weights_files, class_col)

        except FileNotFoundError:
            print(f"Error: Testing file not found at {args.test_data}")
            return
        except Exception as e:
            print(f"An error occurred during testing: {e}\n{print_exc()}")

if __name__ == "__main__":
    main()