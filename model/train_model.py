import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DATASET_PATH = ROOT_DIR / "helpers" / "dataset" / "ai4i2020_normalized.csv"
MODEL_PATH = ROOT_DIR / "model" / "maintenance_model.pkl"

TARGET_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
INPUT_COLUMNS = [
    "Type",
    "Air temperature [C]",
    "Process temperature [C]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

RANDOM_STATE = 42
POSITIVE_WEIGHT_MULTIPLIER = 3.0
THRESHOLD_BETA = 2.0
MIN_THRESHOLD = 0.01
MAX_THRESHOLD = 0.65


# --- Custom Classes ---


class ThresholdedMultiOutputClassifier(BaseEstimator, ClassifierMixin):
    """Uses tuned probability cutoffs instead of the default 0.5 cutoff."""

    def __init__(self, estimator, thresholds=None):
        self.estimator = estimator
        self.thresholds = thresholds

    def fit(self, X, y):
        self.estimator.fit(X, y)
        if self.thresholds is None:
            self.thresholds_ = np.full(y.shape[1], 0.5)
        else:
            self.thresholds_ = np.asarray(self.thresholds, dtype=float)
        return self

    def predict(self, X):
        positive_probabilities = self._positive_probabilities(X)
        return (positive_probabilities >= self.thresholds_).astype(int)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def tune_thresholds(self, X, y, beta=THRESHOLD_BETA):
        positive_probabilities = self._positive_probabilities(X)
        thresholds = []

        for index, target in enumerate(TARGET_COLUMNS):
            y_true = y.iloc[:, index] if hasattr(y, "iloc") else y[:, index]

            if np.sum(y_true) == 0:
                thresholds.append(0.5)
                continue

            best_threshold = 0.5
            best_score = -1.0

            for threshold in np.linspace(MIN_THRESHOLD, MAX_THRESHOLD, 129):
                y_pred = (positive_probabilities[:, index] >= threshold).astype(int)
                score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

                if score > best_score:
                    best_score = score
                    best_threshold = threshold

            thresholds.append(best_threshold)
            print(
                f"{target} threshold tuned to {best_threshold:.3f} "
                f"(validation F{beta:g}: {best_score:.3f})"
            )

        self.thresholds_ = np.asarray(thresholds, dtype=float)
        self.thresholds = self.thresholds_
        return self

    def _positive_probabilities(self, X):
        probabilities = self.estimator.predict_proba(X)
        return np.column_stack(
            [target_probability[:, 1] for target_probability in probabilities]
        )


class MaintenanceFeatureBuilder(BaseEstimator, TransformerMixin):
    """Builds model features from dataset rows or live telemetry rows."""

    feature_names_ = [
        "type_l",
        "type_m",
        "type_h",
        "air_temperature_c",
        "process_temperature_c",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "power_w",
        "overstrain",
        "temp_diff_c",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()

        air_temp = self._temperature_column(df, "Air temperature")
        process_temp = self._temperature_column(df, "Process temperature")
        rotational_speed = self._numeric_column(df, "Rotational speed [rpm]")
        torque = self._numeric_column(df, "Torque [Nm]")
        tool_wear = self._numeric_column(df, "Tool wear [min]")

        product_type = (
            df["Type"].astype(str).str.upper()
            if "Type" in df.columns
            else pd.Series("M", index=df.index)
        )

        features = pd.DataFrame(index=df.index)
        features["type_l"] = (product_type == "L").astype(float)
        features["type_m"] = (product_type == "M").astype(float)
        features["type_h"] = (product_type == "H").astype(float)
        features["air_temperature_c"] = air_temp
        features["process_temperature_c"] = process_temp
        features["rotational_speed_rpm"] = rotational_speed
        features["torque_nm"] = torque
        features["tool_wear_min"] = tool_wear
        features["power_w"] = torque * rotational_speed * (2 * np.pi / 60)
        features["overstrain"] = tool_wear * torque
        features["temp_diff_c"] = process_temp - air_temp

        return features[self.feature_names_]

    @staticmethod
    def _numeric_column(df, column_name):
        if column_name not in df.columns:
            raise ValueError(f"Missing required input column: {column_name}")
        return pd.to_numeric(df[column_name], errors="raise")

    @staticmethod
    def _temperature_column(df, base_name):
        celsius_column = f"{base_name} [C]"
        kelvin_column = f"{base_name} [K]"

        if celsius_column in df.columns:
            return pd.to_numeric(df[celsius_column], errors="raise")

        if kelvin_column in df.columns:
            return pd.to_numeric(df[kelvin_column], errors="raise") - 273.15

        raise ValueError(
            f"Missing required temperature column: {celsius_column} or {kelvin_column}"
        )


# --- Helper Functions ---


def build_aggressive_class_weights(y):
    weights = []
    row_count = len(y)

    for target in TARGET_COLUMNS:
        positive_count = int(y[target].sum())
        negative_count = row_count - positive_count

        if positive_count == 0:
            weights.append({0: 1.0, 1: 1.0})
            continue

        positive_weight = (negative_count / positive_count) * POSITIVE_WEIGHT_MULTIPLIER
        weights.append({0: 1.0, 1: positive_weight})

    return weights


def load_dataset(path=DATASET_PATH):
    df = pd.read_csv(path)
    missing_targets = [column for column in TARGET_COLUMNS if column not in df.columns]
    if missing_targets:
        raise ValueError(f"Dataset is missing target columns: {missing_targets}")
    return df


def build_model(class_weights=None):
    if class_weights is None:
        class_weights = "balanced_subsample"

    return Pipeline(
        steps=[
            ("features", MaintenanceFeatureBuilder()),
            (
                "classifier",
                ThresholdedMultiOutputClassifier(
                    estimator=RandomForestClassifier(
                        n_estimators=1000,
                        class_weight=class_weights,
                        max_features="sqrt",
                        min_samples_leaf=5,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    )
                ),
            ),
        ]
    )


# --- Main Workflows ---


def optimize_spare_parts(data_path=None):
    print("Wczytywanie i przygotowywanie danych...")
    if data_path is None:
        data_path = ROOT_DIR / "helpers" / "dataset" / "ai4i2020.csv"

    df = pd.read_csv(data_path)
    features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    failure_types = ["TWF", "HDF", "PWF", "OSF"]
    spare_parts_mapping = {
        "TWF": "Narzędzia robocze (wymiana eksploatacyjna)",
        "HDF": "Komponenty układu chłodzenia",
        "PWF": "Części układu zasilania i napędu",
        "OSF": "Elementy konstrukcyjne i łożyska",
    }

    X = df[features]
    y = df[failure_types]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Trenowanie modelu Predictive Maintenance (Random Forest)...")
    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    print("\nRaport klasyfikacji (jakość przewidywania na danych testowych):")
    y_pred = model.predict(X_test)
    for i, col in enumerate(failure_types):
        print(f"\n--- Typ awarii: {col} ({spare_parts_mapping[col]}) ---")
        print(classification_report(y_test.iloc[:, i], y_pred[:, i], zero_division=0))

    print("\n=== SYMULACJA ZAMÓWIEŃ CZĘŚCI ZAMIENNYCH ===")
    current_sensors = pd.DataFrame(
        {
            "Air temperature [K]": [298.1, 303.5, 298.9],
            "Process temperature [K]": [308.6, 313.0, 309.1],
            "Rotational speed [rpm]": [1551, 1340, 2800],
            "Torque [Nm]": [42.8, 65.2, 15.0],
            "Tool wear [min]": [45, 215, 10],
        }
    )

    probabilities = model.predict_proba(current_sensors)
    THRESHOLD = 0.40

    for machine_id in range(len(current_sensors)):
        print(f"\n[ Maszyna ID: {machine_id + 1} ]")
        orders_placed = False

        for i, failure_mode in enumerate(failure_types):
            prob_of_failure = probabilities[i][machine_id][1]
            if prob_of_failure > THRESHOLD:
                part_needed = spare_parts_mapping[failure_mode]
                print(
                    f" -> ALARM! Ryzyko awarii {failure_mode}: {prob_of_failure * 100:.1f}%"
                )
                print(
                    f" -> AKCJA: Automatyczne przesunięcie z magazynu/zamówienie: {part_needed}"
                )
                orders_placed = True

        if not orders_placed:
            print(" -> Status: OK. Brak konieczności rezerwacji części.")


def train_and_save_model(dataset_path=DATASET_PATH, model_path=MODEL_PATH):
    df = load_dataset(dataset_path)

    X = df[INPUT_COLUMNS]
    y = df[TARGET_COLUMNS]

    print("\n--- Failure Counts ---")
    for target in TARGET_COLUMNS:
        print(f"{target}: {int(y[target].sum())} positive / {len(y)} total")

    stratify = df["Machine failure"] if "Machine failure" in df.columns else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    train_stratify = None
    if "Machine failure" in df.columns:
        train_stratify = df.loc[y_train.index, "Machine failure"]

    X_fit, X_threshold, y_fit, y_threshold = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_stratify,
    )

    class_weights = build_aggressive_class_weights(y_fit)
    model = build_model(class_weights=class_weights)
    model.fit(X_fit, y_fit)

    print("\n--- Tuned Decision Thresholds ---")
    threshold_features = model.named_steps["features"].transform(X_threshold)
    model.named_steps["classifier"].tune_thresholds(threshold_features, y_threshold)

    y_pred = model.predict(X_test)
    print("\n--- Evaluation Results ---")
    for index, target in enumerate(TARGET_COLUMNS):
        print(f"\nFailure mode: {target}")
        print(
            classification_report(
                y_test.iloc[:, index], y_pred[:, index], zero_division=0
            )
        )

    joblib.dump(model, model_path)
    print(f"\nSaved model to: {model_path}")


# --- Execution Entry Point ---

if __name__ == "__main__":
    # Fix module paths for joblib deserialization
    sys.modules.setdefault("model.train_model", sys.modules[__name__])
    MaintenanceFeatureBuilder.__module__ = "model.train_model"
    ThresholdedMultiOutputClassifier.__module__ = "model.train_model"

    # You can comment out the workflow you don't want to run
    # move_dataset() # Assuming this is imported properly

    print("--- Running Workflow 1: Spare Parts Optimization ---")
    optimize_spare_parts(data_path=ROOT_DIR / "helpers" / "dataset" / "ai4i2020.csv")

    print("\n--- Running Workflow 2: Advanced Pipeline Training ---")
    train_and_save_model()
