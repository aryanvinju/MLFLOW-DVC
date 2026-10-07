# Heart Disease Classification with Git, GitHub Actions, DVC, and MLflow

This Windows demonstration shows the same training pipeline locally and in GitHub Actions. Git versions the code and pipeline definitions. DVC connects each model to a dataset version and reproduces training. MLflow records each run's settings, dataset checksum, metrics, and model.

The dataset has 1,025 source rows, including duplicate records. Training removes exact duplicates before the train/test split to prevent identical rows from appearing on both sides. The prediction is an educational example, not a medical diagnosis.

## Quick start on Windows PowerShell

Clone [the repository](https://github.com/aryanvinju/MLFLOW-DVC) and run these commands from its root:

```powershell
git clone https://github.com/aryanvinju/MLFLOW-DVC.git
Set-Location MLFLOW-DVC
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:DVC_SITE_CACHE_DIR = "$PWD\.dvc\site-cache"
$env:DVC_NO_ANALYTICS = "1"
.\.venv\Scripts\dvc.exe pull
.\.venv\Scripts\dvc.exe repro --force
.\.venv\Scripts\dvc.exe metrics show
```

Before running the pipeline, you can run the same quality checks as GitHub Actions:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pylint --persistent=n src tests
```

`dvc pull` restores the dataset and current model from the small demo remote stored in this repository. The remote is kept in Git solely to make the classroom demo self-contained. A larger project would use separate storage such as S3 or another DVC remote.

The training stage writes `metrics.json` and `artifacts/heart_model.joblib`. Each forced reproduction makes a fresh MLflow run in the local `mlflow_demo.db` database. The model output is a trusted local training artifact; do not load model files from an unknown source.

## Open MLflow

In a second PowerShell window, from the repository root, run:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow_demo.db --port 5000
```

Open <http://127.0.0.1:5000>, choose **Heart Disease Classification**, and compare the runs. The `dataset_md5` parameter matches `data/heart.csv.dvc`. The `source_rows` and `unique_rows` parameters show how many duplicate rows were removed.

## Show GitHub Actions

Open the [Actions page](https://github.com/aryanvinju/MLFLOW-DVC/actions) and select **Heart disease ML pipeline**. Each push to `main`, or a manual **Run workflow**, checks out the Git version, installs dependencies, pulls the DVC data, runs training with MLflow, and uploads `metrics.json`, `dvc.lock`, the model, and the MLflow run files as an artifact named `heart-disease-run`.

The Actions runner has its own MLflow database. Open the local UI for local runs; use the Actions artifact and job log to show the remote run.

## Recreate both comparison runs on a fresh clone

The quick start creates the current 200 tree run. To see a second run in your local MLflow UI, change `n_estimators` in `params.yaml` from `200` to `100`, run `.\.venv\Scripts\dvc.exe repro`, then change it back to `200` and run the command again. The committed baseline settings can be inspected with `git show ea0c3a6:params.yaml`. This local experiment changes the working tree; the existing commits remain available in Git history.

## Show DVC and a prediction

```powershell
.\.venv\Scripts\dvc.exe status
Get-Content data\heart.csv.dvc
Get-Content params.yaml
Get-Content metrics.json
.\.venv\Scripts\python.exe src\predict.py --model artifacts\heart_model.joblib --values 52,1,0,125,212,0,1,168,0,1,2,2,3
```

Change `train.n_estimators` in `params.yaml`, then run `dvc repro` to create a new pipeline output and MLflow run. Use `git diff` to explain the code or parameter change and `git log --oneline` to show the two saved versions. See [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md) for a short presentation sequence.

## Project files

- `data/heart.csv.dvc` and `demo_remote/` hold the dataset version and its demo storage object.
- `dvc.yaml`, `dvc.lock`, `params.yaml`, and `metrics.json` describe and record the pipeline.
- `src/train.py` trains, evaluates, and logs the model; `src/predict.py` loads the exported model.
- `.github/workflows/mlops-pipeline.yml` runs the workflow on GitHub Actions.
- `final_report.md` records the project method and results.

## References

- [DVC data tracking](https://dvc.org/doc/command-reference/add)
- [DVC pipeline reproduction](https://dvc.org/doc/command-reference/repro)
- [MLflow tracking quickstart](https://www.mlflow.org/docs/latest/ml/getting-started/quickstart/)
- [GitHub Actions Python guide](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
