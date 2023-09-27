import os

from itwinai.backend.components import Trainer
from itwinai.models.torch.mnist import MNISTModel
from dataloader import MNISTDataModule
from lightning.pytorch.cli import LightningCLI
from omegaconf import DictConfig, OmegaConf
from utils import load_yaml_with_deps_from_dict


class TorchTrainer(Trainer):
    def __init__(self, train: dict):
        # Convert from YAML
        train_config: DictConfig = load_yaml_with_deps_from_dict(
            train, os.path.dirname(__file__)
        )
        train_config = OmegaConf.to_container(train_config, resolve=True)
        self.conf = train_config

    def train(self, data):
        cli = LightningCLI(
            args=self.conf,
            model_class=MNISTModel,
            datamodule_class=MNISTDataModule,
            run=False,
            save_config_kwargs={
                "overwrite": True,
                "config_filename": "pl-training.yml",
            },
            subclass_mode_model=True,
            subclass_mode_data=True,
        )
        cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    def execute(self, data):
        self.train(data)
