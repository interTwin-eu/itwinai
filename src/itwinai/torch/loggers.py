# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Anna Elisa Lappe
#
# Credit:
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# -------------------------------------------------------------------------------------

import os
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import yaml
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger as LightningLogger
from lightning.pytorch.loggers.logger import rank_zero_experiment
from lightning.pytorch.loggers.utilities import _scan_checkpoints
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor
from typing_extensions import override

from itwinai.loggers import Logger


class PyTorchLightningLogger(LightningLogger):
    """Adapter between PyTorch Lightning logger and itwinai logger.

    This adapter forwards logging calls from PyTorch Lightning to the
    itwinai Logger instance, using the itwinai Logger's `log` method.
    It supports the lightning logging of metrics, hyperparameters, and checkpoints.
    Additionally, any function calls can be forwarded to the itwinai logger instance 
    though the `experiment` property of this Adapter.
    """

    def __init__(
        self,
        itwinai_logger: Logger,
        log_model: Union[Literal["all"], bool] = False
    ):
        """Initializes the adapter with an itwinai logger instance.

        Args:
            itwinai_logger (Logger): An instance of itwinai Logger.
            log_model (Union[Literal["all"], bool], optional):
                Specifies which checkpoints to log.
                If "all", logs all checkpoints; if True, logs the best k checkpoints according
                to the specifications given as `save_top_k` in the Lightning ModelCheckpoint;
                if False, does not log checkpoints.
        """
        self.itwinai_logger = itwinai_logger

        self._log_model = log_model
        self._logged_model_time = {}
        self._checkpoint_callback = None

        self._initialized = False

    @property
    def name(self):
        pass

    @property
    def version(self):
        pass

    @property
    @override
    @rank_zero_only
    def save_dir(self) -> Optional[str]:
        return self.experiment.savedir

    @property
    @rank_zero_experiment
    def experiment(self) -> Logger:
        """Lightning Logger function.
        Initializes and returns the itwinai Logger context for experiment tracking.

        Returns:
            Logger: The itwinai logger instance.
        """
        if not self._initialized:
            # With the rank_zero_decorator the rank will always be 0
            self.itwinai_logger.create_logger_context(rank=0)
            self._initialized = True

        return self.itwinai_logger

    @override
    @rank_zero_only
    def finalize(
        self,
        status: str
    ) -> None:
        """Lightning Logger function.
        Logs any remaining checkpoints and closes the logger context.

        Args:
            status (str): Status of the training (e.g., 'completed', 'failed').
        """
        if not self._initialized:
            return

        # Log checkpoints if the last checkpoint was saved but not logged
        if self._checkpoint_callback:
            self._scan_and_log_checkpoints(self._checkpoint_callback)

        self.experiment.destroy_logger_context()

    @override
    @rank_zero_only
    def log_metrics(
            self,
            metrics: Dict[str, float],
            step: Optional[int] = None
    ) -> None:
        """Lightning Logger function.
        Logs the given metrics and is usually called by the Lightning Trainer.

        Args:
            metrics (Dict[str, float]): Dictionary of metrics to log.
            step (Optional[int], optional): Training step associated with the metrics.
            Defaults to None.
        """
        for key, value in metrics.items():
            self.experiment.log(
                item=value,
                identifier=key,
                kind='metric',
                step=step
            )

    @override
    @rank_zero_only
    def log_hyperparams(
            self,
            params: Union[Dict[str, Any], Namespace]
    ) -> None:
        """Lightning Logger function. Logs hyperparameters for the experiment.

        Args:
            params (Union[Dict[str, Any], Namespace]): Hyperparameters dictionary or object.
        """
        if isinstance(params, Namespace):
            params = vars(params)

        self.experiment.save_hyperparameters(params)

    @override
    def after_save_checkpoint(
        self,
        checkpoint_callback: ModelCheckpoint
    ) -> None:
        """Lightning Logger function. Handles checkpoint saving to the logger after
        the ModelCheckpoint Callback of the Lightning Trainer is called.
        The checkpoints are logged as artifacts.

        Args:
            checkpoint_callback (ModelCheckpoint): Callback instance to manage checkpointing.
        """
        if self._log_model == "all" or (
            self._log_model is True and checkpoint_callback.save_top_k == -1
        ):
            self._scan_and_log_checkpoints(checkpoint_callback)
        elif self._log_model is True:
            self._checkpoint_callback = checkpoint_callback

    def _scan_and_log_checkpoints(
        self,
        checkpoint_callback: ModelCheckpoint
    ) -> None:
        """Scans and logs checkpoints as artifacts in the experiment.

        This function retrieves new checkpoints, logs them as artifacts, and saves
        related metadata, including checkpoint score, filename, and any model checkpoint
        parameters.

        Args:
            checkpoint_callback (ModelCheckpoint): Callback instance used for checkpointing.
        """
        # Get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # Log iteratively all new checkpoints
        for t, p, s, tag in checkpoints:
            metadata = {
                "score": s.item() if isinstance(s, Tensor) else s,
                "original_filename": Path(p).name,
                checkpoint_callback.__class__.__name__: {
                    k: getattr(checkpoint_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                    ]
                    # ensure it does not break if `ModelCheckpoint` args change
                    if hasattr(checkpoint_callback, k)
                },
            }
            if p == checkpoint_callback.best_model_path:
                aliases = ["latest", "best"]
            else:
                aliases = ["latest"]
            artifact_path = Path(p).stem

            # Log the checkpoint
            self.experiment.log(
                item=p,
                identifier=artifact_path,
                kind='artifact'
            )

            with tempfile.TemporaryDirectory(
                prefix="test",
                suffix="test",
                dir=os.getcwd()
            ) as tmp_dir:
                # Log the metadata
                with open(f"{tmp_dir}/metadata.yaml", "w") as tmp_file_metadata:
                    yaml.dump(metadata, tmp_file_metadata, default_flow_style=False)

                # Log the aliases
                with open(f"{tmp_dir}/aliases.txt", "w") as tmp_file_aliases:
                    tmp_file_aliases.write(str(aliases))

                # Log the metadata and aliases
                self.experiment.log(
                    item=tmp_dir,
                    identifier=artifact_path,
                    kind='artifact'
                )

            self._logged_model_time[p] = t
