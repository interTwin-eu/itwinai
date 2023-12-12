import os
from typing import Union, Dict, Any

from itwinai.components import Trainer, monitor_exec
from itwinai.torch.models.mnist import MNISTModel
from dataloader import MNISTDataModule
from lightning.pytorch.cli import LightningCLI
from utils import load_yaml


class LightningMNISTTrainer(Trainer):
    def __init__(self, config: Union[Dict, str]):
        super().__init__()
        self.save_parameters(**locals())
        if isinstance(config, str) and os.path.isfile(config):
            # Load from YAML
            config = load_yaml(config)
        self.conf = config

    @monitor_exec
    def execute(self) -> Any:
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

    def save_state(self):
        return super().save_state()

    def load_state(self):
        return super().load_state()
