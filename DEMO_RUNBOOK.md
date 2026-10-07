# Professor Demonstration Runbook

Start in the `MLFLOW-DVC` repository folder on Windows PowerShell. Keep the [GitHub repository](https://github.com/aryanvinju/MLFLOW-DVC), its [Actions page](https://github.com/aryanvinju/MLFLOW-DVC/actions), and <http://127.0.0.1:5000> open in browser tabs. Complete the one-time setup in `README.md` before presenting.

## 1 Show Git

```powershell
git status --short --branch
git log --oneline -3
git show --stat HEAD
```

Explain that the commits save the pipeline and parameter versions. Open `params.yaml` to point out the change from 100 to 200 trees. Git tracks DVC's small pointer and lock files, while DVC identifies the dataset and model content by checksum.

## 2 Show GitHub Actions

Open the Actions tab and select the latest **Heart disease ML pipeline** run. Open **Fetch DVC versioned files**, **Train and log to MLflow**, and **Show metrics and prediction** in its job log. At the bottom of the run page, show the `heart-disease-run` artifact, which includes the metrics, model, and MLflow files.

If there is no recent run, use **Run workflow** on `main` and wait for the job to finish.

## 3 Show DVC locally

```powershell
$env:DVC_SITE_CACHE_DIR = "$PWD\.dvc\site-cache"
$env:DVC_NO_ANALYTICS = "1"
.\.venv\Scripts\dvc.exe status
Get-Content data\heart.csv.dvc
Get-Content params.yaml
Get-Content metrics.json
```

The checksum in `data/heart.csv.dvc` is the exact dataset version. `dvc.lock` records the input, code, parameter values, and model output for the latest reproduction. `dvc status` should say the pipeline is up to date.

## 4 Show MLflow

In another PowerShell window, start the UI if it is not already running:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow_demo.db --port 5000
```

Open the **Heart Disease Classification** experiment. Compare the runs with 100 and 200 trees. Point out `dataset_md5`, `source_rows`, `unique_rows`, accuracy, precision, recall, F1, and the logged model.

## 5 Re-run on demand

```powershell
.\.venv\Scripts\dvc.exe repro --force
.\.venv\Scripts\dvc.exe metrics show
```

This creates a fresh local MLflow run using the current Git and DVC versions. Refresh the MLflow page to show the new run. If `dvc` reports a site cache permission error, set `DVC_SITE_CACHE_DIR` as shown in step 3 and retry.

The GitHub runner and this laptop store separate MLflow histories. The Actions artifact proves the CI run; the local UI shows the local comparison.
