import pathlib
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pathlib import Path
from argparse import Namespace
from torch import Tensor
import tempfile
import yaml
import os
from time import time

from lightning.pytorch.loggers.logger import Logger as LightningLogger
from lightning.pytorch.loggers.logger import rank_zero_experiment
from lightning.pytorch.loggers.utilities import _scan_checkpoints
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from lightning.pytorch.utilities import rank_zero_only
from typing_extensions import override

from itwinai.loggers import Logger

class PyTorchLightningLogger(LightningLogger):
    """Adapter between PyTorch Lightning logger and itwinai logger.

    This adapter class forwards logging calls from PyTorch Lightning to the
    itwinai Logger instance, using the itwinai Logger's `log` method.
    """

    def __init__(
            self, 
            itwinai_logger: Logger,
            log_model: Union[Literal["all"], bool] = False,
            experiment_name: str = "default_experiment"
        ):
        """Initialize the adapter with an itwinai logger instance.

        Args:
            itwinai_logger (Logger): An instance of itwinai Logger.
        """
        self.itwinai_logger = itwinai_logger 
        self._log_model = log_model
        self._checkpoint_name = experiment_name       # once loggers are refactored this can be something actually sensible
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
    @override
    @rank_zero_only
    def log_dir(self) -> Optional[str]:
        return self.experiment.savedir    

    @property
    @rank_zero_experiment
    def experiment(self) -> Logger:
        if not self._initialized:
            self.itwinai_logger.create_logger_context(rank=0)   # With the rank_zero_decorator the rank will always be 0
            self._initialized = True

        return self.itwinai_logger
    
    @override
    @rank_zero_only
    def finalize(self, status: str):
        if not self._initialized:
            return

        # Log checkpoints as artifacts if the last checkpoint was saved but not logged
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

        assert rank_zero_only.rank == 0

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
        """Logs hyperparameters.

        Args:
            params (Dict[str, Any], Namespace): Hyperparameters dictionary or object.
        """
        assert rank_zero_only.rank == 0

        if isinstance(params, Namespace):
            params = vars(params)

        self.experiment.save_hyperparameters(params)

    @override
    def after_save_checkpoint(
        self, 
        checkpoint_callback: ModelCheckpoint
    ) -> None:
        if self._log_model == "all" or self._log_model is True and checkpoint_callback.save_top_k == -1:
            self._scan_and_log_checkpoints(checkpoint_callback)
        elif self._log_model is True:
            self._checkpoint_callback = checkpoint_callback

    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # log iteratively all new checkpoints
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

            #if isinstance(self.itwinai_logger, MLFlowLogger) or isinstance(self.itwinai_logger, WandBLogger):
            # Create a temporary directory to log on mlflow

            aliases = ["latest", "best"] if p == checkpoint_callback.best_model_path else ["latest"]

            # Artifact path on mlflow
            artifact_path = Path(p).stem

            # Log the checkpoint
            self.experiment.log(
                item=p,
                identifier='best_model',
                kind='artifact'
            )

            with tempfile.TemporaryDirectory(prefix="test", suffix="test", dir=os.getcwd()) as tmp_dir:
                # Log the metadata
                with open(f"{tmp_dir}/metadata.yaml", "w") as tmp_file_metadata:
                    yaml.dump(metadata, tmp_file_metadata, default_flow_style=False)

                # Log the aliases
                with open(f"{tmp_dir}/aliases.txt", "w") as tmp_file_aliases:
                    tmp_file_aliases.write(str(aliases))

                # Log the metadata and aliases
                #self.experiment.log_artifacts(self._run_id, tmp_dir, artifact_path)
                self.experiment.log(
                    item=tmp_dir,
                    identifier=self._checkpoint_name,
                    kind='artifact'
                )

            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t

    