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
        pass

    def log(self, args):
        mlflow.keras.autolog()
