import os
import sys
from typing import Union, Dict, Tuple, Optional, Any

import torch
from torch import Tensor
import lightning as pl
from lightning.pytorch.cli import LightningCLI

from itwinai.components import Trainer, Predictor
from itwinai.serialization import ModelLoader
from itwinai.torch.inference import TorchModelLoader

from model import ThreeDGAN
from dataloader import ParticlesDataModule
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
            datamodule_class=ParticlesDataModule,
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


class LightningModelLoader(TorchModelLoader):
    """Loads a torch lightning model from somewhere.

    Args:
        model_uri (str): Can be a path on local filesystem
            or an mlflow 'locator' in the form:
            'mlflow+MLFLOW_TRACKING_URI+RUN_ID+ARTIFACT_PATH'
    """

    def __call__(self) -> pl.LightningModule:
        """"Loads model from model URI.

        Raises:
            ValueError: if the model URI is not recognized
                or the model is not found.

        Returns:
            pl.LightningModule: torch lightning module.
        """
        # TODO: improve
        # # Load best model
        # loaded_model = cli.model.load_from_checkpoint(
        #     ckpt_path,
        #     lightning_conf['model']['init_args']
        # )
        return super().__call__()


class Lightning3DGANPredictor(Predictor):

    def __init__(
        self,
        model: Union[ModelLoader, pl.LightningModule],
        config: Union[Dict, str],
        name: Optional[str] = None
    ):
        super().__init__(model, name)
        if isinstance(config, str) and os.path.isfile(config):
            # Load from YAML
            config = load_yaml(config)
        self.conf = config

    def predict(
        self,
        datamodule: Optional[pl.LightningDataModule] = None,
        model: Optional[pl.LightningModule] = None
    ) -> Dict[str, Tensor]:
        old_argv = sys.argv
        sys.argv = ['some_script_placeholder.py']
        cli = LightningCLI(
            args=self.conf,
            model_class=ThreeDGAN,
            datamodule_class=ParticlesDataModule,
            run=False,
            save_config_kwargs={
                "overwrite": True,
                "config_filename": "pl-training.yml",
            },
            subclass_mode_model=True,
            subclass_mode_data=True,
        )
        sys.argv = old_argv

        # Override config file with inline arguments, if given
        if datamodule is None:
            datamodule = cli.datamodule
        if model is None:
            model = cli.model

        predictions = cli.trainer.predict(model, datamodule=datamodule)

        predictions_dict = dict()
        # TODO: postprocess predictions
        for idx, generated_img in enumerate(torch.cat(predictions)):
            predictions_dict[str(idx)] = generated_img
        return predictions_dict

    def execute(
        self,
        config: Optional[Dict] = None,
    ) -> Tuple[Optional[Tuple], Optional[Dict]]:
        """"Execute some operations.

        Args:
            config (Dict, optional): key-value configuration.
                Defaults to None.

        Returns:
            Tuple[Optional[Tuple], Optional[Dict]]: tuple structured as
                (results, config).
        """
        return self.predict(), config
