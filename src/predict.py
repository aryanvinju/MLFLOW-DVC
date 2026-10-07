"""Run one prediction using the model produced by src/train.py."""
import argparse
import joblib
import pandas as pd


def predict_one(saved_model: dict, raw_values: str) -> int:
    """Validate one CSV-style row and predict using a saved model bundle."""
    values = [float(value.strip()) for value in raw_values.split(",")]
    feature_names = saved_model["features"]
    if len(values) != len(feature_names):
        raise ValueError(
            f"Expected {len(feature_names)} values in order: {', '.join(feature_names)}"
        )
    row = pd.DataFrame([values], columns=feature_names)
    return int(saved_model["model"].predict(row)[0])


def main() -> None:
    """Load the model file and print one prediction."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/heart_model.joblib")
    parser.add_argument("--values", required=True, help="Comma-separated numeric feature values")
    args = parser.parse_args()
    saved_model = joblib.load(args.model)
    try:
        print(predict_one(saved_model, args.values))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
