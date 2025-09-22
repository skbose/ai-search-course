import mlflow

def start_experiment(experiment_name="ChatbotPipeline_V2", run_name=None):
    mlflow.set_experiment(experiment_name)

    if mlflow.active_run() is not None:
        mlflow.end_run()

    return mlflow.start_run(run_name=run_name)

def log_params(params: dict):
    for k, v in params.items():
        mlflow.log_param(k, v)

def log_metrics(metrics: dict):
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

def log_artifact(path: str):
    mlflow.log_artifact(path)
