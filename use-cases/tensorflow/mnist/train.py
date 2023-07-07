import yaml

from models.mnist import mnist_model, ModelConf
from trainer import TensorflowTrainer
from dataloader import TensorflowDataGetter, TensorflowDataPreproc
from utils import to_json, from_json
from itwinai.backend.tensorflow.executor import TensorflowExecutor
from itwinai.backend.tensorflow.loggers import WanDBLogger, MLFlowLogger

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
            logger = WanDBLogger()

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