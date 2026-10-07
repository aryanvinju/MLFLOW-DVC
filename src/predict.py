"""Run one prediction using the model produced by src/train.py."""
import argparse
import joblib
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="artifacts/heart_model.joblib")
parser.add_argument("--values", required=True, help="Comma-separated numeric feature values")
args = parser.parse_args()

saved = joblib.load(args.model)
values = [float(value.strip()) for value in args.values.split(",")]
if len(values) != len(saved["features"]):
    raise SystemExit(f"Expected {len(saved['features'])} values in order: {', '.join(saved['features'])}")
row = pd.DataFrame([values], columns=saved["features"])
print(int(saved["model"].predict(row)[0]))
