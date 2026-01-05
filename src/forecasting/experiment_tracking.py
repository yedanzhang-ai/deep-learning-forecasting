"""MLflow experiment tracking utilities."""

import mlflow
import pandas as pd

class ExperimentTracker:
    """Wrapper for MLflow experiment tracking."""

    def __init__(self, experiment_name="ts_forecasting", tracking_uri=None):
        """Initialize experiment tracker with MLflow."""
        
        self.experiment_name = experiment_name

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        self.active_run = None

    def start_run(self, run_name):
        """Start a new MLflow run."""
        # End any existing run first
        if mlflow.active_run():
            mlflow.end_run()
        
        self.active_run = mlflow.start_run(run_name=run_name)
        return self.active_run.info.run_id

    def end_run(self):
        """End the current run."""
        
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    def log_params(self, params):
        """Log parameters to current run."""
        
        flat_params = {}
        for key, value in params.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    flat_params[f"{key}.{k}"] = str(v)[:250]
            else:
                flat_params[key] = str(value)[:250]
        mlflow.log_params(flat_params)

    def log_metric(self, key, value, step=None):
        """Log a single metric."""
        
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics):
        """Log multiple metrics."""
        
        mlflow.log_metrics(metrics)

    def log_model_results(self, model_name, model_type, best_params, score, score_type="SMAPE"):
        """Log model tuning results as nested run."""
        
        with mlflow.start_run(run_name=f"tune_{model_name}", nested=True):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_type", model_type)
            self.log_params({"hp": best_params})
            mlflow.log_metric(f"best_{score_type.lower()}", score)

    def log_dataframe(self, df, name):
        """Log a DataFrame as CSV artifact."""
        
        import tempfile
        import os
        path = os.path.join(tempfile.gettempdir(), f"{name}.csv")
        df.to_csv(path, index=False)
        mlflow.log_artifact(path, artifact_path="data")

    def set_tags(self, tags):
        """Set run tags."""
        
        mlflow.set_tags(tags)


def get_tracker(experiment_name="ts_forecasting"):
    """Get experiment tracker instance."""
    
    return ExperimentTracker(experiment_name=experiment_name)
