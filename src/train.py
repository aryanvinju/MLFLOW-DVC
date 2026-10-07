"""Train a heart disease classifier and log the run to MLflow."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def load_training_data(data_path: Path) -> tuple[pd.DataFrame, int]:
    """Read data and remove duplicate rows before the train/test split."""
    source_frame = pd.read_csv(data_path)
    if "target" not in source_frame.columns:
        raise ValueError("Dataset must include a 'target' column")
    if source_frame.isna().any().any():
        raise ValueError("Dataset contains missing values; clean it before training")
    frame = source_frame.drop_duplicates().reset_index(drop=True)
    if frame["target"].nunique() != 2:
        raise ValueError("Dataset must have two target classes")
    return frame, len(source_frame)


def fit_and_score(frame: pd.DataFrame, params: dict) -> tuple:
    """Fit a random forest and return its held-out metrics and feature names."""
    features = frame.drop(columns=["target"])
    target = frame["target"]
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=params["test_size"],
        random_state=params["random_state"], stratify=target
    )
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"], max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"], random_state=params["random_state"],
        class_weight="balanced",
    )
    model.fit(features_train, target_train)
    predicted = model.predict(features_test)
    metrics = {
        "accuracy": float(accuracy_score(target_test, predicted)),
        "precision": float(precision_score(target_test, predicted, zero_division=0)),
        "recall": float(recall_score(target_test, predicted, zero_division=0)),
        "f1": float(f1_score(target_test, predicted, zero_division=0)),
    }
    return model, metrics, list(features.columns)


def main() -> None:
    """Train the project dataset, save outputs, and log one MLflow run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/heart.csv")
    parser.add_argument("--params-path", default="params.yaml")
    parser.add_argument("--metrics-path", default="metrics.json")
    parser.add_argument("--model-path", default="artifacts/heart_model.joblib")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    frame, source_rows = load_training_data(data_path)
    settings = yaml.safe_load(Path(args.params_path).read_text(encoding="utf-8"))["train"]
    params = {
        "n_estimators": int(settings["n_estimators"]),
        "max_depth": int(settings["max_depth"]),
        "min_samples_leaf": int(settings["min_samples_leaf"]),
        "random_state": int(settings["random_state"]),
        "test_size": float(settings["test_size"]),
    }
    model, metrics, feature_names = fit_and_score(frame, params)
    checksum = hashlib.md5(data_path.read_bytes()).hexdigest()
    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_path).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_names}, model_path)

    # Use the same local database for training and the UI, across MLflow versions.
    mlflow.set_tracking_uri("sqlite:///mlflow_demo.db")
    mlflow.set_experiment("Heart Disease Classification")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("dataset_md5", checksum)
        mlflow.log_param("source_rows", source_rows)
        mlflow.log_param("unique_rows", len(frame))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(args.metrics_path))
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
    print(json.dumps({"dataset_md5": checksum, **metrics}, indent=2))


if __name__ == "__main__":
    main()
