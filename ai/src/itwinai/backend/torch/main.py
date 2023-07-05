import yaml

from models.mnist import MNISTModel
from trainer import TorchTrainer
from executor import TorchExecutor
from loggers import WanDBLogger, MLFlowLogger

if __name__ == '__main__':
    # Read config
    with open("config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)

            # Initialize logger
            logger = MLFlowLogger()

            # Create functional Pipeline
            pipeline = [
                TorchTrainer(logger)
            ]

            # Execute pipeline
            executor = TorchExecutor()
            executor.config(pipeline, config)
            executor.execute(pipeline)

        except yaml.YAMLError as exc:
            print(exc)