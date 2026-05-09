import os
from pathlib import Path

import numpy as np


DEFAULT_TRACKING_URI = "https://kltn-mlflow-ui.tech/"


def mlflow_enabled() -> bool:
    return os.getenv("MLFLOW_ENABLE", "true").lower() in {"1", "true", "yes", "y"}


def get_mlflow():
    if not mlflow_enabled():
        print("MLflow logging is disabled by MLFLOW_ENABLE.")
        return None

    try:
        import mlflow
    except ImportError:
        print("MLflow package is not installed; skipping MLflow logging.")
        return None

    return mlflow


def start_run(default_experiment: str, default_run_name: str, tags: dict | None = None):
    mlflow = get_mlflow()
    if mlflow is None:
        return None

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", default_experiment)
    run_name = os.getenv("MLFLOW_RUN_NAME", default_run_name)
    run_id = os.getenv("MLFLOW_RUN_ID")

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags(tags)
        print(f"MLflow run started: {run.info.run_id}")
        print(f"MLflow tracking URI: {tracking_uri}")
        print(f"MLflow experiment: {experiment_name}")
        return mlflow
    except Exception as exc:
        print(f"Could not start MLflow run; skipping MLflow logging. Error: {exc}")
        return None


def end_run(mlflow):
    if mlflow is None:
        return
    try:
        mlflow.end_run()
    except Exception as exc:
        print(f"Could not end MLflow run cleanly: {exc}")


def _clean_value(value):
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value


def log_params_safe(mlflow, params: dict):
    if mlflow is None:
        return

    clean_params = {}
    for key, value in params.items():
        value = _clean_value(value)
        if value is not None:
            clean_params[key] = value

    if not clean_params:
        return

    try:
        mlflow.log_params(clean_params)
    except Exception as exc:
        print(f"Could not log MLflow params: {exc}")


def log_metrics_safe(mlflow, metrics: dict, prefix: str | None = None, step: int | None = None):
    if mlflow is None:
        return

    clean_metrics = {}
    for key, value in metrics.items():
        value = _clean_value(value)
        if isinstance(value, (int, float)) and np.isfinite(value):
            metric_name = f"{prefix}/{key}" if prefix else key
            clean_metrics[metric_name] = float(value)

    if not clean_metrics:
        return

    try:
        mlflow.log_metrics(clean_metrics, step=step)
    except Exception as exc:
        print(f"Could not log MLflow metrics: {exc}")


def log_artifacts_safe(mlflow, paths: list[str], artifact_path: str | None = None):
    if mlflow is None:
        return

    for path in paths:
        if not path or not Path(path).exists():
            continue
        try:
            mlflow.log_artifact(path, artifact_path=artifact_path)
        except Exception as exc:
            print(f"Could not log MLflow artifact {path}: {exc}")


def log_keras_model_safe(
    mlflow,
    model,
    artifact_path: str = "model",
    registered_model_name: str | None = None,
):
    if mlflow is None or model is None:
        return

    register_enabled = os.getenv("MLFLOW_REGISTER_MODEL", "true").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    if not register_enabled:
        registered_model_name = None

    if registered_model_name is None:
        registered_model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME")

    try:
        mlflow.keras.log_model(
            model,
            artifact_path=artifact_path,
        )
        print(f"MLflow model logged at artifact path: {artifact_path}")
    except Exception as exc:
        print(f"Could not log Keras model in MLflow: {exc}")
        return

    if not registered_model_name:
        return

    try:
        active_run = mlflow.active_run()
        if active_run is None:
            print("Could not register MLflow model: no active run.")
            return
        model_uri = f"runs:/{active_run.info.run_id}/{artifact_path}"
        result = mlflow.register_model(model_uri, registered_model_name)
        mlflow.set_tag("registered_model_name", registered_model_name)
        mlflow.set_tag("registered_model_version", result.version)
        print(f"MLflow model registered as: {registered_model_name} v{result.version}")
    except Exception as exc:
        print(f"Could not register MLflow model: {exc}")


def log_history_safe(mlflow, history: dict | None, phase_name: str, start_step: int = 0) -> int:
    if mlflow is None or not history:
        return start_step

    max_epochs = max((len(values) for values in history.values()), default=0)

    for epoch_idx in range(max_epochs):
        epoch_metrics = {
            metric_name: values[epoch_idx]
            for metric_name, values in history.items()
            if epoch_idx < len(values)
        }
        log_metrics_safe(
            mlflow,
            epoch_metrics,
            prefix=f"{phase_name}/epoch",
            step=start_step + epoch_idx,
        )

    summary_metrics = {}
    for metric_name, values in history.items():
        if not values:
            continue
        values_np = np.asarray(values, dtype=float)
        summary_metrics[f"{metric_name}_last"] = values_np[-1]
        if metric_name == "loss" or metric_name.startswith("val_loss"):
            summary_metrics[f"{metric_name}_best"] = np.nanmin(values_np)
        else:
            summary_metrics[f"{metric_name}_best"] = np.nanmax(values_np)

    log_metrics_safe(mlflow, summary_metrics, prefix=f"{phase_name}/summary")
    return start_step + max_epochs
