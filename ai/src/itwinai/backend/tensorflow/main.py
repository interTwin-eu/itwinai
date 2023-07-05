import yaml

from models.mnist import mnist_model, ModelConf
from trainer import TensorflowTrainer, TrainerConf
from dataloader import DataGetterConf, TensorflowDataGetter, TensorflowDataPreproc, DataPreprocConf
from executor import TensorflowExecutor
from utils import to_json, from_json
from loggers import WanDBLogger, MLFlowLogger

if __name__ == '__main__':
    # Read config
    with open("config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)

            # Load model from config
            model_config = ModelConf.Schema().load(config['ModelConf'])
            model = mnist_model(model_config)

            # Save, Load form JSON
            to_json(model, './model.json')
            from_json('./model.json')

            # Initialize logger
            logger = MLFlowLogger()

            # Create functional Pipeline
            pipeline = [
                TensorflowDataGetter(),
                TensorflowDataPreproc(),
                TensorflowTrainer(model, logger)
            ]

            # Execute pipeline
            executor = TensorflowExecutor()
            executor.config(pipeline, config)
            executor.execute(pipeline)

        except yaml.YAMLError as exc:
            print(exc)