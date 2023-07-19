import wandb
import mlflow
import mlflow.keras

from ..components import Logger


class WanDBLogger(Logger):
    def __init__(self):
        pass

    def log(self, args):
        wandb.init(config={"bs": 12})


class MLFlowLogger(Logger):
    def __init__(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("test-experiment")

    def log(self, args):
        mlflow.pytorch.autolog()
