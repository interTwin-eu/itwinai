# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Kalliopi Tsolaki
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Kalliopi Tsolaki <kalliopi.tsolaki@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import os
import sys
import tempfile
from typing import Any, Dict, Optional, Union

import lightning as pl
import torch
import yaml
from dataloader import ParticlesDataModule
from lightning.pytorch import Trainer as LightningTrainer
from lightning.pytorch.cli import LightningCLI
from model import ThreeDGAN
from torch import Tensor

from itwinai.components import Predictor, Trainer, monitor_exec

# from itwinai.torch.mlflow import (
#     init_lightning_mlflow,
#     teardown_lightning_mlflow
# )
from itwinai.loggers import EmptyLogger, Logger
from itwinai.serialization import ModelLoader
from itwinai.torch.inference import TorchModelLoader
from itwinai.torch.type import Batch
from itwinai.utils import load_yaml


class Lightning3DGANTrainer(Trainer):
    def __init__(self, config: Union[Dict, str], itwinai_logger: Optional[Logger] = None):
        self.save_parameters(**self.locals2params(locals()))
        super().__init__()
        if isinstance(config, str) and os.path.isfile(config):
            # Load from YAML
            config = load_yaml(config)
        self.conf = config
        self.itwinai_logger = itwinai_logger if itwinai_logger else EmptyLogger()

    @monitor_exec
    def execute(self) -> Any:
        # Parse lightning configuration
        old_argv = sys.argv
        sys.argv = ["some_script_placeholder.py"]
        cli = LightningCLI(
            args=self.conf,
            model_class=ThreeDGAN,
            datamodule_class=ParticlesDataModule,
            trainer_class=LightningTrainer,
            run=False,
            save_config_kwargs={
                "overwrite": True,
                "config_filename": "pl-training.yml",
            },
            subclass_mode_model=True,
            subclass_mode_data=True,
        )
        sys.argv = old_argv

        with self.itwinai_logger.start_logging(rank=cli.trainer.global_rank):
            # Set the logger into the LightningTrainer
            cli.trainer.itwinai_logger = self.itwinai_logger

            # Start training
            cli.trainer.fit(cli.model, datamodule=cli.datamodule)

            self._log_config(self.itwinai_logger)
            self.itwinai_logger.log(
                cli.trainer.train_dataloader, "train_dataloader", kind="torch"
            )
            self.itwinai_logger.log(
                cli.trainer.val_dataloaders, "val_dataloader", kind="torch"
            )

    def _log_config(self, logger: Logger):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            local_yaml_path = os.path.join(tmp_dir, "pl-conf.yaml")
            with open(local_yaml_path, "w") as outfile:
                yaml.dump(self.conf, outfile, default_flow_style=False)
            logger.log(local_yaml_path, "lightning-config", kind="artifact")


class LightningModelLoader(TorchModelLoader):
    """Loads a torch lightning model from somewhere.

    Args:
        model_uri (str): Can be a path on local filesystem
            or an mlflow 'locator' in the form:
            'mlflow+MLFLOW_TRACKING_URI+RUN_ID+ARTIFACT_PATH'
    """

    def __call__(self) -> pl.LightningModule:
        """ "Loads model from model URI.

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
        name: Optional[str] = None,
    ):
        self.save_parameters(**self.locals2params(locals()))
        super().__init__(model, name)
        if isinstance(config, str) and os.path.isfile(config):
            # Load from YAML
            config = load_yaml(config)
        self.conf = config

    @monitor_exec
    def execute(
        self,
        datamodule: Optional[pl.LightningDataModule] = None,
        model: Optional[pl.LightningModule] = None,
    ) -> Dict[str, Tensor]:
        old_argv = sys.argv
        sys.argv = ["some_script_placeholder.py"]
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

        # Transpose predictions into images, energies and angles
        images = torch.cat(
            list(map(lambda pred: self.transform_predictions(pred["images"]), predictions))
        )
        energies = torch.cat(list(map(lambda pred: pred["energies"], predictions)))
        angles = torch.cat(list(map(lambda pred: pred["angles"], predictions)))

        predictions_dict = dict()
        for img, en, ang in zip(images, energies, angles):
            sample_key = f"energy={en.item()}&angle={ang.item()}"
            predictions_dict[sample_key] = img

        return predictions_dict

    def transform_predictions(self, batch: Batch) -> Batch:
        """
        Post-process the predictions of the torch model.
        """
        return batch.squeeze(1)
