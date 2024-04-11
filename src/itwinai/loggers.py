"""Abstraction for loggers."""

import os
import csv
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union
import pickle
import pathlib

import wandb
import mlflow
# import mlflow.keras

BASE_EXP_NAME: str = 'unk_experiment'


class LogMixin(metaclass=ABCMeta):
    @abstractmethod
    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = 'metric',
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs
    ) -> None:
        """Log item."""
        pass


class Logger(LogMixin, metaclass=ABCMeta):
    """Base class for logger

    Args:
        savedir (str): disk location where logs are stored.
    """
    savedir: str = None
    supported_types: List[str]  # Supported logging 'kinds'
    _log_freq: Union[int, str]

    def __init__(
        self,
        savedir: str = 'mllogs',
        log_freq: Union[int, str] = 'epoch'
    ) -> None:
        self.savedir = savedir
        self.log_freq = log_freq

    @property
    def log_freq(self) -> Union[int, str]:
        return self._log_freq

    @log_freq.setter
    def log_freq(self, val: Union[int, str]):
        """Sanitize log_freq value."""
        if val in ['epoch', 'batch'] or (
                isinstance(val, int) and val > 0):
            self._log_freq = val
        else:
            raise ValueError(
                "Wrong value for 'log_freq'. Supported values are: "
                f"['epoch', 'batch'] or int > 0. Received: {val}"
            )

    @contextmanager
    def start_logging(self):
        try:
            self.create_logger_context()
            yield
        finally:
            self.destroy_logger_context()

    @abstractmethod
    def create_logger_context(self):
        pass

    @abstractmethod
    def destroy_logger_context(self):
        pass

    @abstractmethod
    def save_hyperparameters(self, params: Dict[str, Any]) -> None:
        pass

    def serialize(self, obj: Any, identifier: str) -> str:
        """Serializes object to disk and returns its path.

        Args:
            obj (Any): item to save.
            identifier (str): identifier of the item to log (expected to be a
            path under ``self.savedir``).

        Returns:
            str: local path of the serialized object to be logged.
        """
        itm_path = os.path.join(self.savedir, identifier)
        with open(itm_path, 'wb') as itm_file:
            pickle.dump(obj, itm_file)

    def should_log(
        self,
        batch_idx: Optional[int]
    ) -> bool:
        if batch_idx is not None:
            if isinstance(self.log_freq, int):
                if batch_idx % self.log_freq == 0:
                    return True
                return False
            if self.log_freq == 'batch':
                return True
            return False
        return True


class ConsoleLogger(Logger):
    """Simple logger for quick tests."""

    def __init__(
        self,
        savedir: str = 'mllogs',
        log_freq: Union[int, str] = 'epoch'
    ) -> None:
        savedir = os.path.join(savedir, 'simple-logger')
        super().__init__(savedir=savedir, log_freq=log_freq)
        self.supported_types = ['torch', 'artifact']

    def create_logger_context(self):
        os.makedirs(self.savedir, exist_ok=True)
        run_dirs = sorted([int(dir) for dir in os.listdir(self.savedir)])
        if len(run_dirs) == 0:
            self.run_id = 0
        else:
            self.run_id = int(run_dirs[-1]) + 1
        self.run_path = os.path.join(self.savedir, str(self.run_id))
        os.makedirs(self.run_path)

    def destroy_logger_context(self):
        pass

    def save_hyperparameters(self, params: Dict[str, Any]) -> None:
        pass

    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = 'metric',
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs
    ) -> None:
        if not self.should_log(batch_idx=batch_idx):
            return

        if kind == 'artifact':
            if isinstance(item, str) and os.path.isfile(item):
                import shutil
                identifier = os.path.join(
                    self.run_path,
                    identifier
                )
                print(f"ConsoleLogger: Serializing to {identifier}...")
                shutil.copyfile(item, identifier)
            else:
                identifier = os.path.join(
                    os.path.basename(self.run_path),
                    identifier
                )
                print(f"ConsoleLogger: Serializing to {identifier}...")
                self.serialize(item, identifier)
        elif kind == 'torch':
            identifier = os.path.join(self.run_path, identifier)
            print(f"ConsoleLogger: Saving to {identifier}...")
            import torch
            torch.save(item, identifier)
        else:
            print(f"ConsoleLogger: {identifier} = {item}")


class MLFlowLogger(Logger):
    """Abstraction for MLFlow logger."""

    active_run: mlflow.ActiveRun

    def __init__(
        self,
        savedir: str = 'mllogs',
        experiment_name: str = BASE_EXP_NAME,
        tracking_uri: Optional[str] = None,
        run_description: Optional[str] = None,
        log_freq: Union[int, str] = 'epoch'
    ):
        savedir = os.path.join(savedir, 'mlflow')
        super().__init__(savedir=savedir, log_freq=log_freq)
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_description = run_description

        if self.tracking_uri is None:
            # Default MLFLow tracking URI
            saved_abs_path = os.path.abspath(self.savedir)
            self.tracking_uri = pathlib.Path(saved_abs_path).as_uri()
            # self.tracking_uri = "file://" + self.savedir
        print(f'MLFLOW URI: {self.tracking_uri}')

        # TODO: for pytorch lightning:
        # mlflow.pytorch.autolog()

        self.supported_types = [
            'metric', 'figure', 'image', 'artifact', 'torch', 'dict', 'param',
            'text'
        ]

    def create_logger_context(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name=self.experiment_name)
        self.active_run: mlflow.ActiveRun = mlflow.start_run(
            description=self.run_description
        )

    def destroy_logger_context(self):
        mlflow.end_run()

    def save_hyperparameters(self, params: Dict[str, Any]) -> None:
        for param_name, val in params.items():
            self.log(item=val, identifier=param_name, step=0, kind='param')

    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = 'metric',
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs
    ) -> None:
        """Log with MLFlow.

        Args:
            item (Union[Any, List[Any]]): element to be logged (e.g., metric,
                image, artifact...).
            identifier (Union[str, List[str]]): unique identifier for the
                element to log(e.g., name of a metric, artifact path).
            kind (str, optional): type of the item to be logged. Must be one
                among the list of self.supported_types. Defaults to 'metric'.
            step (Optional[int], optional): logging step. Defaults to None.
            batch_idx (Optional[int], optional): batch counter (i.e., batch
                idx). Defaults to None.
        """
        if not self.should_log(batch_idx=batch_idx):
            return

        if kind == 'metric':
            # if isinstance(item, list) and isinstance(identifier, list):
            mlflow.log_metric(
                key=identifier,
                value=item,
                step=step
            )
        if kind == 'artifact':
            if not isinstance(item, str):
                # Save the object locally and then log it
                name = os.path.basename(identifier)
                save_path = os.path.join(self.savedir, '.trash', name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                item = self.serialize(item, save_path)
            mlflow.log_artifact(
                local_path=item,
                artifact_path=identifier
            )
        if kind == 'torch':
            import torch
            # Save the object locally and then log it
            name = os.path.basename(identifier)
            save_path = os.path.join(self.savedir, '.trash', name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(item, save_path)
            # Log into mlflow
            mlflow.log_artifact(
                local_path=save_path,
                artifact_path=identifier
            )
        if kind == 'dict':
            mlflow.log_dict(
                dictionary=item,
                artifact_file=identifier
            )
        if kind == 'figure':
            mlflow.log_figure(
                artifact_file=identifier,
                figure=item,
                save_kwargs=kwargs.get('save_kwargs')
            )
        if kind == 'image':
            mlflow.log_image(
                artifact_file=identifier,
                image=item
            )
        if kind == 'param':
            mlflow.log_param(
                key=identifier,
                value=item
            )
        if kind == 'text':
            mlflow.log_text(
                artifact_file=identifier,
                text=item
            )


class WanDBLogger(Logger):
    """Abstraction for WandB logger."""

    def __init__(
        self,
        savedir: str = 'mllogs',
        project_name: str = BASE_EXP_NAME,
        log_freq: Union[int, str] = 'epoch'
    ) -> None:
        savedir = os.path.join(savedir, 'wandb')
        super().__init__(savedir=savedir, log_freq=log_freq)
        self.project_name = project_name
        self.supported_types = [
            'watch', 'metric', 'figure', 'image', 'artifact', 'torch', 'dict',
            'param', 'text'
        ]

    def create_logger_context(self):
        self.active_run = wandb.init(
            dir=self.savedir,
            project=self.project_name
        )

    def destroy_logger_context(self):
        pass

    def save_hyperparameters(self, params: Dict[str, Any]) -> None:
        wandb.config.update(params)

    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = 'metric',
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs
    ) -> None:
        """Log with WandB. Wrapper of https://docs.wandb.ai/ref/python/log

        Args:
            item (Union[Any, List[Any]]): element to be logged (e.g., metric,
                image, artifact...).
            identifier (Union[str, List[str]]): unique identifier for the
                element to log(e.g., name of a metric, artifact path).
            kind (str, optional): type of the item to be logged. Must be one
                among the list of self.supported_types. Defaults to 'metric'.
            step (Optional[int], optional): logging step. Defaults to None.
            batch_idx (Optional[int], optional): batch counter (i.e., batch
                idx). Defaults to None.
        """
        if not self.should_log(batch_idx=batch_idx):
            return

        if kind == 'watch':
            wandb.watch(item)
        if kind in self.supported_types[1:]:
            wandb.log({identifier: item}, step=step, commit=True)


class TensorBoardLogger(Logger):
    """Abstraction for Tensorboard logger."""

    def __init__(
        self,
        savedir: str = 'mllogs',
        log_freq: Union[int, str] = 'epoch'
    ) -> None:
        savedir = os.path.join(savedir, 'tensorboard')
        super().__init__(savedir=savedir, log_freq=log_freq)

    def create_logger_context(self):
        pass

    def destroy_logger_context(self):
        pass

    def save_hyperparameters(self, params: Dict[str, Any]) -> None:
        pass

    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = 'metric',
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs
    ) -> None:
        """Log with Tensorboard.

        Args:
            item (Union[Any, List[Any]]): element to be logged (e.g., metric,
                image, artifact...).
            identifier (Union[str, List[str]]): unique identifier for the
                element to log(e.g., name of a metric, artifact path).
            kind (str, optional): type of the item to be logged. Must be one
                among the list of self.supported_types. Defaults to 'metric'.
            step (Optional[int], optional): logging step. Defaults to None.
            batch_idx (Optional[int], optional): batch counter (i.e., batch
                idx). Defaults to None.
        """
        if not self.should_log(batch_idx=batch_idx):
            return

        # TODO: complete


class LoggersCollection(Logger):
    """Contains a list of loggers. Never tested."""

    def __init__(
        self,
        loggers: List[Logger]
    ) -> None:
        super().__init__(savedir='/.tmp_mllogs_LoggersCollection', log_freq=0)
        self.loggers = loggers

    def should_log(self, batch_idx: int = None) -> bool:
        return True

    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = 'metric',
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs
    ) -> None:
        for logger in self.loggers:
            logger.log(
                item=item,
                identifier=identifier,
                kind=kind,
                step=step,
                batch_idx=batch_idx,
                **kwargs
            )


class EpochTimeTracker:
    def __init__(self, series_name: str, csv_file: str) -> None:
        self.series_name = series_name
        self._data = []
        self.csv_file = csv_file
        with open(csv_file, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['name', 'nodes', 'epoch_id', 'time'])

    def add_epoch_time(self, epoch_idx, time):
        n_nodes = os.environ.get('SLURM_NNODES', -1)
        fields = (self.series_name, n_nodes, epoch_idx, time)
        self._data.append(fields)
        with open(self.csv_file, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)

    def save(self, csv_file: Optional[str] = None):
        if not csv_file:
            csv_file = self.csv_file
        with open(csv_file, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['name', 'nodes', 'epoch_id', 'time'])
            csvwriter.writerows(self._data)
