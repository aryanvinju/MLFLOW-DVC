# Heart Disease Classification with Git, GitHub Actions, DVC, and MLflow

## Objective

This project demonstrates a repeatable machine learning workflow. Git records code and parameter changes, DVC identifies and restores the dataset and model outputs, GitHub Actions repeats the pipeline on a clean runner, and MLflow records each training run for comparison.

## Dataset and method

The included `heart.csv` has 1,025 rows, 13 numeric predictors, and a binary `target`. Of those rows, 723 are exact duplicates. The training script removes duplicates before splitting the 302 unique rows into stratified training and test sets. This avoids putting identical records in both sets. A random forest then predicts the target using a fixed random seed, a 20% test split, and the parameters in `params.yaml`.

The project's `data/heart.csv.dvc` file records the exact dataset checksum. `dvc.yaml` defines the training stage, and `dvc.lock` records its inputs, settings, and output hash. For this small classroom example, `demo_remote/` holds the DVC objects inside the Git repository so GitHub Actions can pull them without cloud credentials. A larger project should use external DVC storage.

## Results

Both local runs used the same deduplicated dataset and test split. Only the number of trees changed.

| Trees | Accuracy | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.7869 | 0.8125 | 0.7879 | 0.8000 |
| 200 | 0.8033 | 0.8387 | 0.7879 | 0.8125 |

The 200 tree run improved accuracy by about 1.6 percentage points on this one split. This is a small educational comparison, not evidence that the model is suitable for clinical use. MLflow stores the full precision metrics, settings, model, `source_rows`, `unique_rows`, and `dataset_md5` for each run. `metrics.json` contains the current run's metrics.

## Reproduce and inspect

Follow `README.md` to create a Windows virtual environment, install dependencies, and run `dvc pull` followed by `dvc repro --force`. `DEMO_RUNBOOK.md` gives the live presentation order. The GitHub Actions workflow runs the same DVC stage on pushes to `main` and uploads metrics, lock file, model, and MLflow files. The local MLflow UI uses `mlflow_demo.db`; each GitHub runner creates a separate database captured in its artifact.

## Source material

The five supplied MLflow and DVC tutorial documents informed the workflow. They were examples rather than a submission rubric. The `heart.csv` file came from the existing workspace; its original external source was not stated in the available materials.

## References

- [DVC data tracking](https://dvc.org/doc/command-reference/add)
- [DVC pipeline reproduction](https://dvc.org/doc/command-reference/repro)
- [MLflow tracking quickstart](https://www.mlflow.org/docs/latest/ml/getting-started/quickstart/)
- [GitHub Actions Python guide](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
