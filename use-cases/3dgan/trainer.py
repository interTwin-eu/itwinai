import os
import sys
from typing import Union, Dict, Tuple, Optional, Any

from itwinai.components import Trainer
from model import ThreeDGAN
from dataloader import MyDataModule
from lightning.pytorch.cli import LightningCLI
from utils import load_yaml


class Lightning3DGANTrainer(Trainer):
    def __init__(self, config: Union[Dict, str]):
        super().__init__()
        if isinstance(config, str) and os.path.isfile(config):
            # Load from YAML
            config = load_yaml(config)
        self.conf = config

    def train(self) -> Any:
        old_argv = sys.argv
        sys.argv = ['some_script_placeholder.py']
        cli = LightningCLI(
            args=self.conf,
            model_class=ThreeDGAN,
            datamodule_class=MyDataModule,
            run=False,
            save_config_kwargs={
                "overwrite": True,
                "config_filename": "pl-training.yml",
            },
            subclass_mode_model=True,
            subclass_mode_data=True,
        )
        sys.argv = old_argv
        cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    def execute(
        self,
        config: Optional[Dict] = None
    ) -> Tuple[Any, Optional[Dict]]:
        result = self.train()
        return result, config

    def save_state(self):
        return super().save_state()

    def load_state(self):
        return super().load_state()
