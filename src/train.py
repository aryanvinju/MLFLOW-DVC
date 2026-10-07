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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/heart.csv")
    parser.add_argument("--params-path", default="params.yaml")
    parser.add_argument("--metrics-path", default="metrics.json")
    parser.add_argument("--model-path", default="artifacts/heart_model.joblib")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    source_frame = pd.read_csv(data_path)
    frame = source_frame.drop_duplicates().reset_index(drop=True)
    if "target" not in frame.columns:
        raise ValueError("Dataset must include a 'target' column")
    if frame.isna().any().any():
        raise ValueError("Dataset contains missing values; clean it before training")
    X = frame.drop(columns=["target"])
    y = frame["target"]
    settings = yaml.safe_load(Path(args.params_path).read_text(encoding="utf-8"))["train"]
    params = {
        "n_estimators": int(settings["n_estimators"]),
        "max_depth": int(settings["max_depth"]),
        "min_samples_leaf": int(settings["min_samples_leaf"]),
        "random_state": int(settings["random_state"]),
        "test_size": float(settings["test_size"]),
    }
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["random_state"], stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"], max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"], random_state=params["random_state"],
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
    }
    checksum = hashlib.md5(data_path.read_bytes()).hexdigest()
    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_path).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(X.columns)}, model_path)

    # Use the same local database for training and the UI, across MLflow versions.
    mlflow.set_tracking_uri("sqlite:///mlflow_demo.db")
    mlflow.set_experiment("Heart Disease Classification")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("dataset_md5", checksum)
        mlflow.log_param("source_rows", len(source_frame))
        mlflow.log_param("unique_rows", len(frame))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(args.metrics_path))
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
    print(json.dumps({"dataset_md5": checksum, **metrics}, indent=2))


if __name__ == "__main__":
    main()
