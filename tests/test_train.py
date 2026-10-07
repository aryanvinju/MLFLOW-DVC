"""Unit tests for dataset validation and duplicate removal."""

import pandas as pd
import pytest

from src.train import fit_and_score, load_training_data


def test_load_training_data_removes_duplicates(tmp_path):
    """Identical source records must not cross the train/test boundary."""
    data_path = tmp_path / "train.csv"
    pd.DataFrame(
        {"age": [30, 30, 50, 60], "chol": [180, 180, 220, 240],
         "target": [0, 0, 1, 0]}
    ).to_csv(data_path, index=False)

    frame, source_rows = load_training_data(data_path)

    assert source_rows == 4
    assert len(frame) == 3
    assert frame["target"].tolist() == [0, 1, 0]


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("age,label\n30,0\n50,1\n", "target"),
        ("age,target\n30,0\n,1\n", "missing values"),
        ("age,target\n30,1\n50,1\n", "two target classes"),
    ],
)
def test_load_training_data_rejects_invalid_input(tmp_path, contents, message):
    """Bad training inputs should fail before model fitting or MLflow logging."""
    data_path = tmp_path / "train.csv"
    data_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_training_data(data_path)


def test_fit_and_score_reports_held_out_metrics():
    """Training returns a fitted model and valid test metrics."""
    frame = pd.DataFrame(
        {"age": range(20), "chol": [180 + number for number in range(20)],
         "target": [0, 1] * 10}
    )
    params = {
        "n_estimators": 5, "max_depth": 3, "min_samples_leaf": 1,
        "random_state": 42, "test_size": 0.25,
    }

    model, metrics, feature_names = fit_and_score(frame, params)

    assert model.n_estimators == 5
    assert feature_names == ["age", "chol"]
    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    assert all(0 <= score <= 1 for score in metrics.values())
