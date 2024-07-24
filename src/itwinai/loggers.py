"""
``itwinai`` wrappers for well-known ML loggers.

A logger allows to save objects of different kinds:

.. list-table:: Logger kinds
   :widths: 25 25
   :header-rows: 1

   * - Object ``kind``
     - Description
   * - ``metric``
     - Number, usually representing a ML metric of interest (e.g., loss,
       accuracy).
   * - ``torch``
     - PyTorch object (e.g., tensor).
   * - ``artifact``
     - File on the local filesystem to be stored by the logger.
   * - ``figure``
     - Matplotlib of Plotly figure
   * - ``image``
     - PIL image or numpy array storing an image.
   * - ``param``
     - | Hyper-parameter (e.g., learning rate, batch size, number of layers)
       | as a primitive Python type.
   * - ``text``
     - Running text (string).
   * - ``dict``
     - Python dictionary.
   * - ``model``
     - ML model. At the moment only :class:`~torch.nn.Module` is supported.
   * - ``best_model``
     - Best ML model. At the moment only :class:`~torch.nn.Module` is
       supported.
   * - ``dataset``
     - Dataset object (e.g., objects of type :class:`~mlflow.data.Dataset`).
   * - ``watch``
     - | WandB ``watch``: Hook into the torch model to collect gradients and
       | the topology. `More info`_.
   * - ``flops_pb``
     - Flops per batch, used by :class:`~itwinai.loggers.Prov4MLLogger`.
   * - ``flops_pb``
     - Flops per batch, used by :class:`~itwinai.loggers.Prov4MLLogger`.
   * - ``flops_pe``
     - Flops per epoch, used by :class:`~itwinai.loggers.Prov4MLLogger`.
   * - ``system``
     - System metrics, used by :class:`~itwinai.loggers.Prov4MLLogger`.
   * - ``carbon``
     - Carbon footprint information, used
       by :class:`~itwinai.loggers.Prov4MLLogger`.
   * - ``execution_time``
     - Execution time, used by :class:`~itwinai.loggers.Prov4MLLogger`.

.. _More info:
    https://docs.wandb.ai/ref/python/watch
"""

import os
import csv
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union, Literal, Tuple
from typing_extensions import override
import pickle
import pathlib

import wandb
import mlflow
import prov4ml

BASE_EXP_NAME: str = 'default_experiment'


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
        """Log ``item`` with ``identifier`` name of ``kind`` type at ``step``
        time step.

        Args:
            item (Union[Any, List[Any]]): element to be logged (e.g., metric).
            identifier (Union[str, List[str]]): unique identifier for the
                element to log(e.g., name of a metric).
            kind (str, optional): type of the item to be logged. Must be one
                among the list of self.supported_kinds. Defaults to 'metric'.
            step (Optional[int], optional): logging step. Defaults to None.
            batch_idx (Optional[int], optional): DataLoader batch counter
                (i.e., batch idx), if available. Defaults to None.
        """


class Logger(LogMixin, metaclass=ABCMeta):
    """Base class for logger

    Args:
        savedir (str, optional): filesystem location where logs are stored.
            Defaults to 'mllogs'.
        log_freq (Union[int, Literal['epoch', 'batch']], optional):
            how often should the logger fulfill calls to the `log()`
            method:

            - When set to 'epoch', the logger logs only if ``batch_idx``
              is not passed to the ``log`` method.

            - When an integer
              is given, the logger logs if ``batch_idx`` is a multiple of
              ``log_freq``.

            - When set to ``'batch'``, the logger logs always.

            Defaults to 'epoch'.
        log_on_workers (Optional[Union[int, List[int]]]): if -1, log on all
            workers; if int log on worker with rank equal to log_on_workers;
            if List[int], log on workers which rank is in the list.
            Defaults to 0 (the global rank of the main worker).
    """
    #: Location on filesystem where to store data.
    savedir: str = None
    #: Supported logging 'kind's.
    supported_kinds: Tuple[str]
    #: Current worker global rank
    worker_rank: int

    _log_freq: Union[int, Literal['epoch', 'batch']]

    def __init__(
        self,
        savedir: str = 'mllogs',
        log_freq: Union[int, Literal['epoch', 'batch']] = 'epoch',
        log_on_workers: Union[int, List[int]] = 0
    ) -> None:
        self.savedir = savedir
        self.log_freq = log_freq
        self.log_on_workers = log_on_workers

    @property
    def log_freq(self) -> Union[int, Literal['epoch', 'batch']]:
        """Get ``log_feq``, namely how often should the logger
        fulfill or ignore calls to the `log()` method."""
        return self._log_freq

    @log_freq.setter
    def log_freq(self, val: Union[int, Literal['epoch', 'batch']]):
        """Sanitize log_freq value."""
        if val in ['epoch', 'batch'] or (isinstance(val, int) and val > 0):
            self._log_freq = val
        else:
            raise ValueError(
                "Wrong value for 'log_freq'. Supported values are: "
                f"['epoch', 'batch'] or int > 0. Received: {val}"
            )

    @contextmanager
    def start_logging(self, rank: Optional[int] = None):
        """Start logging context.

        Args:
            rank (Optional[int]): global rank of current process,
                used in distributed environments. Defaults to None.

        Example:


        >>> with my_logger.start_logging():
        >>>     my_logger.log(123, 'value', kind='metric', step=0)


        """
        try:
            self.create_logger_context(rank=rank)
            yield
        finally:
            self.destroy_logger_context()

    @abstractmethod
    def create_logger_context(self, rank: Optional[int] = None) -> Any:
        """
        Initializes the logger context.

        Args:
            rank (Optional[int]): global rank of current process,
                used in distributed environments. Defaults to None.
        """

    @abstractmethod
    def destroy_logger_context(self) -> None:
        """Destroy logger."""

    @abstractmethod
    def save_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Save hyperparameters.

        Args:
            params (Dict[str, Any]): hyperparameters dictionary.
        """

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
        batch_idx: Optional[int] = None
    ) -> bool:
        """Determines whether the logger should fulfill or ignore calls to the
        `log()` method, depending on the ``log_freq`` property:

        - When ``log_freq`` is set to 'epoch', the logger logs only if
          ``batch_idx`` is not passed to the ``log`` method.

        - When ``log_freq`` is an integer
          is given, the logger logs if ``batch_idx`` is a multiple of
          ``log_freq``.

        - When ``log_freq`` is set to ``'batch'``, the logger logs always.

        It also takes into account whether logging on the current worker
        rank is allowed by ``self.log_on_workers``.

        Args:
            batch_idx (Optional[int]): the dataloader batch idx, if available.
                Defaults to None.

        Returns:
            bool: True if the logger should log, False otherwise.
        """
        # Check worker's global rank
        worker_ok = (
            self.worker_rank is None or
            (isinstance(self.log_on_workers, int) and (
                self.log_on_workers == -1 or
                self.log_on_workers == self.worker_rank
            )
            )
            or
            (isinstance(self.log_on_workers, list)
             and self.worker_rank in self.log_on_workers)
        )
        if not worker_ok:
            return False

        # Check batch ID
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
    """Simplified logger.

    Args:
        savedir (str, optional): where to store artifacts.
            Defaults to 'mllogs'.
        log_freq (Union[int, Literal['epoch', 'batch']], optional):
            determines whether the logger should fulfill or ignore
            calls to the `log()` method. See ``Logger.should_log`` method for
            more details. Defaults to 'epoch'.
        log_on_workers (Optional[Union[int, List[int]]]): if -1, log on all
            workers; if int log on worker with rank equal to log_on_workers;
            if List[int], log on workers which rank is in the list.
            Defaults to 0 (the global rank of the main worker).
    """

    #: Supported kinds in the ``log`` method
    supported_kinds: Tuple[str] = ('torch', 'artifact', 'metric')

    def __init__(
        self,
        savedir: str = 'mllogs',
        log_freq: Union[int, Literal['epoch', 'batch']] = 'epoch',
        log_on_workers: Union[int, List[int]] = 0
    ) -> None:
        savedir = os.path.join(savedir, 'simple-logger')
        super().__init__(
            savedir=savedir,
            log_freq=log_freq,
            log_on_workers=log_on_workers
        )

    def create_logger_context(self, rank: Optional[int] = None):
        """
        Initializes the logger context.

        Args:
            rank (Optional[int]): global rank of current process,
                used in distributed environments. Defaults to None.
        """
        self.worker_rank = rank

        if not self.should_log():
            return

        os.makedirs(self.savedir, exist_ok=True)
        run_dirs = sorted([int(dir) for dir in os.listdir(self.savedir)])
        if len(run_dirs) == 0:
            self.run_id = 0
        else:
            self.run_id = int(run_dirs[-1]) + 1
        self.run_path = os.path.join(self.savedir, str(self.run_id))
        os.makedirs(self.run_path)

    def destroy_logger_context(self):
        """Destroy logger. Do nothing."""

    def save_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Save hyperparameters. Do nothing.

        Args:
            params (Dict[str, Any]): hyperparameters dictionary.
        """
        if not self.should_log():
            return

        # Save hyperparams

    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = 'metric',
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs
    ) -> None:
        """Print metrics to stdout and save artifacts to the filesystem.

        Args:
            item (Union[Any, List[Any]]): element to be logged (e.g., metric).
            identifier (Union[str, List[str]]): unique identifier for the
                element to log(e.g., name of a metric).
            kind (str, optional): type of the item to be logged. Must be
                one among the list of ``self.supported_kinds``.
                Defaults to 'metric'.
            step (Optional[int], optional): logging step. Defaults to None.
            batch_idx (Optional[int], optional): DataLoader batch counter
                (i.e., batch idx), if available. Defaults to None.
            kwargs: keyword arguments to pass to the logger.
        """
        if not self.should_log(batch_idx=batch_idx):
            return

        if kind == 'artifact':
            if isinstance(item, str) and os.path.isfile(item):
                import shutil
                identifier = os.path.join(
                    self.run_path,
                    identifier
                )
                if len(os.path.dirname(identifier)) > 0:
                    os.makedirs(os.path.dirname(identifier), exist_ok=True)
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
    """Abstraction around MLFlow logger.

    Args:
        savedir (str, optional): path on local filesystem where logs are
            stored. Defaults to 'mllogs'.
        experiment_name (str, optional): experiment name. Defaults to
            ``itwinai.loggers.BASE_EXP_NAME``.
        tracking_uri (Optional[str], optional): MLFLow tracking URI.
            Overrides ``savedir`` if given. Defaults to None.
        run_description (Optional[str], optional): run description.
            Defaults to None.
        log_freq (Union[int, Literal['epoch', 'batch']], optional):
            determines whether the logger should fulfill or ignore
            calls to the `log()` method. See ``Logger.should_log`` method for
            more details. Defaults to 'epoch'.
        log_on_workers (Optional[Union[int, List[int]]]): if -1, log on all
            workers; if int log on worker with rank equal to log_on_workers;
            if List[int], log on workers which rank is in the list.
            Defaults to 0 (the global rank of the main worker).
    """

    #: Supported kinds in the ``log`` method
    supported_kinds: Tuple[str] = (
        'metric', 'figure', 'image', 'artifact', 'torch', 'dict', 'param',
        'text', 'model', 'dataset')

    #: Current MLFLow experiment's run.
    active_run: mlflow.ActiveRun

    def __init__(
        self,
        savedir: str = 'mllogs',
        experiment_name: str = BASE_EXP_NAME,
        tracking_uri: Optional[str] = None,
        run_description: Optional[str] = None,
        log_freq: Union[int, Literal['epoch', 'batch']] = 'epoch',
        log_on_workers: Union[int, List[int]] = 0
    ):
        savedir = os.path.join(savedir, 'mlflow')
        super().__init__(
            savedir=savedir,
            log_freq=log_freq,
            log_on_workers=log_on_workers
        )
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_description = run_description

        if self.tracking_uri is None:
            # Default MLFLow tracking URI
            saved_abs_path = os.path.abspath(self.savedir)
            self.tracking_uri = pathlib.Path(saved_abs_path).as_uri()

        # TODO: for pytorch lightning:
        # mlflow.pytorch.autolog()

    def create_logger_context(
        self,
        rank: Optional[int] = None
    ) -> mlflow.ActiveRun:
        """
        Initializes the logger context. Start MLFLow run.

        Args:
            rank (Optional[int]): global rank of current process,
                used in distributed environments. Defaults to None.

        Returns:
            mlflow.ActiveRun: active MLFlow run.
        """
        self.worker_rank = rank

        if not self.should_log():
            return

        active_run = mlflow.active_run()
        if active_run:
            print("Detected an active MLFlow run. Attaching to it...")
            self.active_run = active_run
        else:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(experiment_name=self.experiment_name)
            self.active_run: mlflow.ActiveRun = mlflow.start_run(
                description=self.run_description
            )
        return self.active_run

    def destroy_logger_context(self):
        """Destroy logger. End current MLFlow run."""
        if not self.should_log():
            return

        mlflow.end_run()

    def save_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Save hyperparameters as MLFlow parameters.

        Args:
            params (Dict[str, Any]): hyperparameters dictionary.
        """
        if not self.should_log():
            return

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
            item (Union[Any, List[Any]]): element to be logged (e.g., metric).
            identifier (Union[str, List[str]]): unique identifier for the
                element to log(e.g., name of a metric).
            kind (str, optional): type of the item to be logged. Must be
                one among the list of ``self.supported_kinds``.
                Defaults to 'metric'.
            step (Optional[int], optional): logging step. Defaults to None.
            batch_idx (Optional[int], optional): DataLoader batch counter
                (i.e., batch idx), if available. Defaults to None.
            kwargs: keyword arguments to pass to the logger.
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
        if kind == 'model':
            import torch
            if isinstance(item, torch.nn.Module):
                mlflow.pytorch.log_model(item, identifier)
            else:
                print("WARNING: unrecognized model type")
        if kind == 'dataset':
            # Log mlflow dataset
            # https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_input
            # It may be needed to convert item into a mlflow dataset, e.g.:
            # https://mlflow.org/docs/latest/python_api/mlflow.data.html#mlflow.data.from_pandas
            # ATM delegated to the user
            if isinstance(item, mlflow.data.Dataset):
                mlflow.log_input(item)
            else:
                print("WARNING: unrecognized dataset type. "
                      "Must be an MLFlow dataset")
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


class WandBLogger(Logger):
    """Abstraction around WandB logger.

    Args:
        savedir (str, optional): location on local filesystem where logs
            are stored. Defaults to 'mllogs'.
        project_name (str, optional): experiment name. Defaults to
            ``itwinai.loggers.BASE_EXP_NAME``.
        log_freq (Union[int, Literal['epoch', 'batch']], optional):
            determines whether the logger should fulfill or ignore
            calls to the `log()` method. See ``Logger.should_log`` method for
            more details. Defaults to 'epoch'.
        log_on_workers (Optional[Union[int, List[int]]]): if -1, log on all
            workers; if int log on worker with rank equal to log_on_workers;
            if List[int], log on workers which rank is in the list.
            Defaults to 0 (the global rank of the main worker).
    """

    # TODO: add support for artifacts logging

    #: Supported kinds in the ``log`` method
    supported_kinds: Tuple[str] = (
        'watch', 'metric', 'figure', 'image', 'torch', 'dict',
        'param', 'text')

    def __init__(
        self,
        savedir: str = 'mllogs',
        project_name: str = BASE_EXP_NAME,
        log_freq: Union[int, Literal['epoch', 'batch']] = 'epoch',
        log_on_workers: Union[int, List[int]] = 0
    ) -> None:
        savedir = os.path.join(savedir, 'wandb')
        super().__init__(
            savedir=savedir,
            log_freq=log_freq,
            log_on_workers=log_on_workers
        )
        self.project_name = project_name

    def create_logger_context(self, rank: Optional[int] = None) -> None:
        """
        Initializes the logger context. Init WandB run.

        Args:
            rank (Optional[int]): global rank of current process,
                used in distributed environments. Defaults to None.
        """
        self.worker_rank = rank

        if not self.should_log():
            return

        os.makedirs(os.path.join(self.savedir, 'wandb'), exist_ok=True)
        self.active_run = wandb.init(
            dir=os.path.abspath(self.savedir),
            project=self.project_name
        )

    def destroy_logger_context(self):
        """Destroy logger."""
        if not self.should_log():
            return

    def save_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Save hyperparameters.

        Args:
            params (Dict[str, Any]): hyperparameters dictionary.
        """
        if not self.should_log():
            return

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
            item (Union[Any, List[Any]]): element to be logged (e.g., metric).
            identifier (Union[str, List[str]]): unique identifier for the
                element to log(e.g., name of a metric).
            kind (str, optional): type of the item to be logged. Must be
                one among the list of ``self.supported_kinds``.
                Defaults to 'metric'.
            step (Optional[int], optional): ignored by ``WandBLogger``.
            batch_idx (Optional[int], optional): DataLoader batch counter
                (i.e., batch idx), if available. Defaults to None.
            kwargs: keyword arguments to pass to the logger.
        """
        if not self.should_log(batch_idx=batch_idx):
            return

        if kind == 'watch':
            wandb.watch(item)
        elif kind in self.supported_kinds:
            # wandb.log({identifier: item}, step=step, commit=True)
            # Let WandB use its preferred step
            wandb.log({identifier: item}, commit=True)


class TensorBoardLogger(Logger):
    """Abstraction around TensorBoard logger, both for PyTorch and
    TensorFlow.

    Args:
        savedir (str, optional): location on local filesystem where logs
            are stored. Defaults to 'mllogs'.
        log_freq (Union[int, Literal['epoch', 'batch']], optional):
            determines whether the logger should fulfill or ignore
            calls to the `log()` method. See ``Logger.should_log`` method for
            more details. Defaults to 'epoch'.
        framework (Literal['tensorflow', 'pytorch'], optional):
            whether to log PyTorch or TensorFlow ML data.
            Defaults to 'pytorch'.
        log_on_workers (Optional[Union[int, List[int]]]): if -1, log on all
            workers; if int log on worker with rank equal to log_on_workers;
            if List[int], log on workers which rank is in the list.
            Defaults to 0 (the global rank of the main worker).

    Raises:
        ValueError: when ``framework`` is not recognized.
    """

    # TODO: decouple the logger into TorchTBLogger and TFTBLogger
    # and add the missing logging types supported by each.

    #: Supported kinds in the ``log`` method
    supported_kinds: Tuple[str] = (
        'metric', 'image', 'text', 'figure', 'torch')

    def __init__(
        self,
        savedir: str = 'mllogs',
        log_freq: Union[int, Literal['epoch', 'batch']] = 'epoch',
        framework: Literal['tensorflow', 'pytorch'] = 'pytorch',
        log_on_workers: Union[int, List[int]] = 0
    ) -> None:
        savedir = os.path.join(savedir, 'tensorboard')
        super().__init__(
            savedir=savedir,
            log_freq=log_freq,
            log_on_workers=log_on_workers
        )
        self.framework = framework
        if framework.lower() == 'tensorflow':
            import tensorflow as tf
            self.tf = tf
            self.writer = tf.summary.create_file_writer(savedir)
        elif framework.lower() == 'pytorch':
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(savedir)
        else:
            raise ValueError(
                "Framework must be either 'tensorflow' or 'pytorch'")

    def create_logger_context(self, rank: Optional[int] = None) -> None:
        """
        Initializes the logger context. Init Tensorboard run.

        Args:
            rank (Optional[int]): global rank of current process,
                used in distributed environments. Defaults to None.
        """
        self.worker_rank = rank

        if not self.should_log():
            return

        if self.framework == 'tensorflow':
            self.writer.set_as_default()

    def destroy_logger_context(self):
        """Destroy logger. Close SummaryWriter."""
        if not self.should_log():
            return

        self.writer.close()

    def save_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Save hyperparameters.

        Args:
            params (Dict[str, Any]): hyperparameters dictionary.
        """
        if not self.should_log():
            return

        if self.framework == 'tensorflow':
            from tensorboard.plugins.hparams import api as hp
            hparams = {hp.HParam(k): v for k, v in params.items()}
            with self.writer.as_default():
                hp.hparams(hparams)
        elif self.framework == 'pytorch':
            self.writer.add_hparams(params, {})

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
            item (Union[Any, List[Any]]): element to be logged (e.g., metric).
            identifier (Union[str, List[str]]): unique identifier for the
                element to log(e.g., name of a metric).
            kind (str, optional): type of the item to be logged. Must be
                one among the list of ``self.supported_kinds``.
                Defaults to 'metric'.
            step (Optional[int], optional): logging step. Defaults to None.
            batch_idx (Optional[int], optional): DataLoader batch counter
                (i.e., batch idx), if available. Defaults to None.
            kwargs: keyword arguments to pass to the logger.
        """
        if not self.should_log(batch_idx=batch_idx):
            return

        if self.framework == 'tensorflow':
            with self.writer.as_default():
                if kind == 'metric':
                    self.tf.summary.scalar(identifier, item, step=step)
                elif kind == 'image':
                    self.tf.summary.image(identifier, item, step=step)
                elif kind == 'text':
                    self.tf.summary.text(identifier, item, step=step)
                elif kind == 'figure':
                    self.tf.summary.figure(identifier, item, step=step)
        elif self.framework == 'pytorch':
            if kind == 'metric':
                self.writer.add_scalar(identifier, item, global_step=step)
            elif kind == 'image':
                self.writer.add_image(identifier, item, global_step=step)
            elif kind == 'text':
                self.writer.add_text(identifier, item, global_step=step)
            elif kind == 'figure':
                self.writer.add_figure(identifier, item, global_step=step)
            elif kind == 'torch':
                self.writer.add_graph(item)


class LoggersCollection(Logger):
    """Wrapper of a set of loggers, allowing to use them simultaneously.

    Args:
        loggers (List[Logger]): list of itwinai loggers.
    """

    #: Supported kinds are delegated to the loggers in the collection.
    supported_kinds: Tuple[str]

    def __init__(
        self,
        loggers: List[Logger]
    ) -> None:
        super().__init__(savedir='/tmp/mllogs_LoggersCollection', log_freq=1)
        self.loggers = loggers

    def should_log(self, batch_idx: int = None) -> bool:
        """Transparent method which delegates the `Logger.should_log``
        to individual loggers. Always returns True.

        Args:
            batch_idx (int, optional): dataloader batch index.
            Defaults to None.

        Returns:
            bool: always True.
        """
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
        """Log on all loggers.

        Args:
            item (Union[Any, List[Any]]): element to be logged (e.g., metric).
            identifier (Union[str, List[str]]): unique identifier for the
                element to log(e.g., name of a metric).
            kind (str, optional): type of the item to be logged. Must be
                one among the list of ``self.supported_kinds``.
                Defaults to 'metric'.
            step (Optional[int], optional): logging step. Defaults to None.
            batch_idx (Optional[int], optional): DataLoader batch counter
                (i.e., batch idx), if available. Defaults to None.
            kwargs: keyword arguments to pass to the logger.
        """
        for logger in self.loggers:
            logger.log(
                item=item,
                identifier=identifier,
                kind=kind,
                step=step,
                batch_idx=batch_idx,
                **kwargs
            )

    def create_logger_context(self, rank: Optional[int] = None) -> Any:
        """
        Initializes all loggers.

        Args:
            rank (Optional[int]): global rank of current process,
                used in distributed environments. Defaults to None.
        """
        for logger in self.loggers:
            logger.create_logger_context(rank=rank)

    def destroy_logger_context(self):
        """Destroy all loggers."""
        for logger in self.loggers:
            logger.destroy_logger_context()

    def save_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Save hyperparameters for all loggers.

        Args:
            params (Dict[str, Any]): hyperparameters dictionary.
        """
        for logger in self.loggers:
            logger.save_hyperparameters(params=params)


class Prov4MLLogger(Logger):
    """
    Abstraction around Prov4ML logger.

    Args:
        prov_user_namespace (str, optional): location to where provenance
            files will be uploaded. Defaults to "www.example.org".
        experiment_name (str, optional): experiment name.
            Defaults to "experiment_name".
        provenance_save_dir (str, optional): path where to store provenance
            files and logs. Defaults to "prov".
        save_after_n_logs (Optional[int], optional): how often to save
            logs to disk from main memory. Defaults to 100.
        create_graph (Optional[bool], optional): whether to create a
            provenance graph. Defaults to True.
        create_svg (Optional[bool], optional): whether to create an SVG
            representation of the provenance graph. Defaults to True.
        log_freq (Union[int, Literal['epoch', 'batch']], optional):
            determines whether the logger should fulfill or ignore
            calls to the `log()` method. See ``Logger.should_log`` method for
            more details. Defaults to 'epoch'.
        log_on_workers (Optional[Union[int, List[int]]]): if -1, log on all
            workers; if int log on worker with rank equal to log_on_workers;
            if List[int], log on workers which rank is in the list.
            Defaults to 0 (the global rank of the main worker).
    """

    #: Supported kinds in the ``log`` method
    supported_kinds: Tuple[str] = (
        'metric', 'flops_pb', 'flops_pe', 'system', 'carbon',
        'execution_time', 'model', 'best_model',
        'torch')

    def __init__(
        self,
        prov_user_namespace="www.example.org",
        experiment_name="experiment_name",
        provenance_save_dir="mllogs/prov_logs",
        save_after_n_logs: Optional[int] = 100,
        create_graph: Optional[bool] = True,
        create_svg: Optional[bool] = True,
        log_freq: Union[int, Literal['epoch', 'batch']] = 'epoch',
        log_on_workers: Union[int, List[int]] = 0
    ) -> None:
        super().__init__(
            savedir=provenance_save_dir,
            log_freq=log_freq,
            log_on_workers=log_on_workers
        )
        self.name = experiment_name
        self.version = None
        self.prov_user_namespace = prov_user_namespace
        self.provenance_save_dir = provenance_save_dir
        self.save_after_n_logs = save_after_n_logs
        self.create_graph = create_graph
        self.create_svg = create_svg

    @override
    def create_logger_context(self, rank: Optional[int] = None):
        """
        Initializes the logger context.

        Args:
            rank (Optional[int]): global rank of current process,
                used in distributed environments. Defaults to None.
        """
        self.worker_rank = rank

        if not self.should_log():
            return

        prov4ml.start_run(
            prov_user_namespace=self.prov_user_namespace,
            experiment_name=self.name,
            provenance_save_dir=self.provenance_save_dir,
            save_after_n_logs=self.save_after_n_logs,
            # This class will control which workers can log
            collect_all_processes=True,
            rank=rank
        )

    @override
    def destroy_logger_context(self):
        """
        Destroys the logger context.
        """
        if not self.should_log():
            return

        prov4ml.end_run(
            create_graph=self.create_graph,
            create_svg=self.create_svg)

    @override
    def save_hyperparameters(self, params: Dict[str, Any]) -> None:
        if not self.should_log():
            return

        # Save hyperparams
        for param_name, val in params.items():
            prov4ml.log_param(param_name, val)

    @override
    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = 'metric',
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        context: Optional[str] = 'training',
        **kwargs
    ) -> None:
        """Logs with Prov4ML.

        Args:
            item (Union[Any, List[Any]]): element to be logged (e.g., metric).
            identifier (Union[str, List[str]]): unique identifier for the
                element to log(e.g., name of a metric).
            kind (str, optional): type of the item to be logged. Must be
                one among the list of ``self.supported_kinds``.
                Defaults to 'metric'.
            step (Optional[int], optional): logging step. Defaults to None.
            batch_idx (Optional[int], optional): DataLoader batch counter
                (i.e., batch idx), if available. Defaults to None.
            kwargs: keyword arguments to pass to the logger.
        """

        if not self.should_log(batch_idx=batch_idx):
            return

        if kind == "metric":
            prov4ml.log_metric(key=identifier, value=item,
                               context=context, step=step)
        elif kind == "flops_pb":
            model, batch = item
            prov4ml.log_flops_per_batch(
                identifier, model=model,
                batch=batch, context=context, step=step)
        elif kind == "flops_pe":
            model, dataset = item
            prov4ml.log_flops_per_epoch(
                identifier, model=model,
                dataset=dataset, context=context, step=step)
        elif kind == "system":
            prov4ml.log_system_metrics(context=context, step=step)
        elif kind == "carbon":
            prov4ml.log_carbon_metrics(context=context, step=step)
        elif kind == "execution_time":
            prov4ml.log_current_execution_time(
                label=identifier, context=context, step=step)
        elif kind == 'model':
            prov4ml.save_model_version(
                model=item, model_name=identifier, context=context, step=step)
        elif kind == 'best_model':
            prov4ml.log_model(model=item, model_name=identifier,
                              log_model_info=True, log_as_artifact=True)
        elif kind == 'torch':
            from torch.utils.data import DataLoader
            if isinstance(item, DataLoader):
                prov4ml.log_dataset(dataset=item, label=identifier)
            else:
                prov4ml.log_param(key=identifier, value=item)


class EpochTimeTracker:
    """Profiler for epoch execution time used to support scaling tests.
    It uses CSV files to store, for each epoch, the ``name`` of the
    experiment, the number of compute ``nodes`` used, the ``epoch_id``,
    and the execution ``time`` in seconds.

    Args:
        series_name (str): name of the experiment/job.
        csv_file (str): path to CSV file to store experiments times.
    """

    def __init__(self, series_name: str, csv_file: str) -> None:
        self.series_name = series_name
        self._data = []
        self.csv_file = csv_file
        with open(csv_file, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['name', 'nodes', 'epoch_id', 'time'])

    def add_epoch_time(self, epoch_idx: int, time: float) -> None:
        """Add row to the current experiment's CSV file in append mode.

        Args:
            epoch_idx (int): epoch order idx.
            time (float): epoch execution time (seconds).
        """
        n_nodes = os.environ.get('SLURM_NNODES', -1)
        fields = (self.series_name, n_nodes, epoch_idx, time)
        self._data.append(fields)
        with open(self.csv_file, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)

    def save(self, csv_file: Optional[str] = None) -> None:
        """Save data to a new CSV file.

        Args:
            csv_file (Optional[str], optional): path to the CSV file.
                If not given, uses the one given in the constructor.
                Defaults to None.
        """
        if not csv_file:
            csv_file = self.csv_file
        with open(csv_file, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['name', 'nodes', 'epoch_id', 'time'])
            csvwriter.writerows(self._data)
