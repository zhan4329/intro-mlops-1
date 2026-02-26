# src/see_runs.py
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_mlflow_runs(experiment_name: str, max_runs: int = 20) -> pd.DataFrame:
    """Fetch recent runs for an experiment and return a DataFrame of params and metrics."""
    # creates the client to the MLflow server
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()

    # this is looking for all the runs for our experiment and ordering them by start time in descending order (recent first)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=max_runs,
    )

    # create a list of dictionaries with the run information to convert to a DataFrame
    rows = []
    for r in runs:
        rows.append({
            "run_id": r.info.run_id,
            "start_time": r.info.start_time,
            "epochs": r.data.params.get("epochs"),
            "learning_rate": float(r.data.params.get("learning_rate", float("nan"))),
            "val_loss": r.data.metrics.get("val_loss"),
            "val_accuracy": r.data.metrics.get("val_accuracy"),
        })

    return pd.DataFrame(rows)

def see_runs(experiment_name: str = "WineQualityClassifier", project_root=None, max_runs: int = 20):
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    mlflow.set_tracking_uri(f"file://{project_root / 'mlruns'}") # otherwise mlflow will not know where to find the runs

    # Load recent runs into a DataFrame
    df_runs = load_mlflow_runs(experiment_name, max_runs=max_runs)
    print(df_runs)

    # plotting the val loss and val accuracy for each run:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df_runs.plot(kind="bar", x="run_id", y="val_loss", ax=axes[0], title="Final val_loss", rot=45)
    axes[0].set_xlabel("Run ID")
    axes[0].set_ylabel("Validation Loss")
    df_runs.plot(kind="bar", x="run_id", y="val_accuracy", ax=axes[1], title="Final val_accuracy", ylim=(0, 1), rot=45)
    axes[1].set_xlabel("Run ID")
    axes[1].set_ylabel("Validation Accuracy")
    plt.tight_layout()
    plt.show()

    # you can also filter/sort runs to find the best one
    best_run = df_runs.loc[df_runs["val_loss"].idxmin()]
    print(f"Best run: {best_run['run_id']} with val_loss={best_run['val_loss']:.4f} and val_accuracy={best_run['val_accuracy']:.4f}")

    return df_runs

if __name__ == "__main__":
    see_runs()