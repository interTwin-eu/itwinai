import os

# TODO: Solve relative import
import sys
sys.path.append("..")
from components import Trainer

from models.mnist import MNISTModel
from dataloader import MNISTDataModule
from lightning.pytorch.cli import LightningCLI
from marshmallow_dataclass import dataclass
from omegaconf import DictConfig, OmegaConf
from utils import load_yaml_with_deps_from_dict

@dataclass
class TrainerConf:
    train: dict

class TorchTrainer(Trainer):
    def __init__(self, logger):
        self.trainer = None
        self.logger = logger
        self.conf = None

    def train(self, data):
        self.logger.log()

        cli = LightningCLI(
            args=self.conf,
            model_class=MNISTModel,
            datamodule_class=MNISTDataModule,
            run=False,
            save_config_kwargs={
                "overwrite": True,
                "config_filename": "pl-training.yml"
            },
            subclass_mode_model = True,
            subclass_mode_data = True
        )

        cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    def execute(self, data):
        self.train(data)

    def config(self, config):
        config = TrainerConf.Schema().load(config['TrainerConf'])

        # Convert from YAML
        train_config: DictConfig = load_yaml_with_deps_from_dict(
            config.train,
            os.path.dirname(__file__)
        )
        train_config = OmegaConf.to_container(train_config, resolve=True)

        self.conf = train_config
