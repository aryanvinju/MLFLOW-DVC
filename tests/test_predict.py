"""Unit tests for single-row model inference."""

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.predict import predict_one


def test_predict_one_loads_saved_bundle(tmp_path):
    """A persisted model uses the saved feature names and predicts one row."""
    features = pd.DataFrame({"age": [30, 50], "chol": [180, 220]})
    model = DummyClassifier(strategy="constant", constant=1)
    model.fit(features, [0, 1])
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "features": list(features.columns)}, model_path)

    saved_model = joblib.load(model_path)

    assert predict_one(saved_model, "45,200") == 1


@pytest.mark.parametrize("raw_values", ["45", "45,200,1", "age,200"])
def test_predict_one_rejects_invalid_values(raw_values):
    """Incorrect length or nonnumeric values must not reach the model."""
    saved_model = {"model": None, "features": ["age", "chol"]}

    with pytest.raises(ValueError):
        predict_one(saved_model, raw_values)
