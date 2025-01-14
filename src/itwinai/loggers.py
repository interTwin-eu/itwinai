# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# -------------------------------------------------------------------------------------

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
     - ML model. At the moment only :class:`torch.nn.Module` is supported.
   * - ``best_model``
     - Best ML model. At the moment only :class:`torch.nn.Module` is
       supported.
   * - ``dataset``
     - Dataset object (e.g., objects of type :class:`mlflow.data.Dataset`).
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
   * - ``prov_documents``
     - Provenance documents, used by :class:`~itwinai.loggers.Prov4MLLogger`.

.. _More info:
    https://docs.wandb.ai/ref/python/watch
"""

import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

from typing_extensions import override

if TYPE_CHECKING:
    import mlflow


BASE_EXP_NAME: str = "default_experiment"


class LogMixin(ABC):
    @abstractmethod
    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = "metric",
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs,
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


class Logger(LogMixin):
    """Base class for logger

    Args:
        savedir (Union[Path, str], optional): filesystem location where logs are stored.
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
    savedir: Path
    #: Supported logging 'kind's.
    supported_kinds: Tuple[str]
    #: Current worker global rank
    worker_rank: int

    _log_freq: Union[int, Literal["epoch", "batch"]]

    def __init__(
        self,
        savedir: Union[Path, str] = "mllogs",
        log_freq: Union[int, Literal["epoch", "batch"]] = "epoch",
        log_on_workers: Union[int, List[int]] = 0,
        experiment_id: Optional[str] = None,
        run_id: Optional[Union[int, str]] = None,
    ) -> None:
        self.savedir = Path(savedir)
        self.log_freq = log_freq
        self.log_on_workers = log_on_workers
        self._experiment_id = experiment_id
        self._run_id = run_id

    @property
    def experiment_id(self) -> Optional[str]:
        """Return the experiment name."""
        return self._experiment_id

    @property
    def run_id(self) -> Optional[Union[int, str]]:
        """Return the experiment version."""
        return self._run_id

    @property
    def log_freq(self) -> Union[int, Literal["epoch", "batch"]]:
        """Get ``log_feq``, namely how often should the logger
        fulfill or ignore calls to the `log()` method."""
        return self._log_freq

    @log_freq.setter
    def log_freq(self, val: Union[int, Literal["epoch", "batch"]]):
        """Sanitize log_freq value."""
        if val in ["epoch", "batch"] or (isinstance(val, int) and val > 0):
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
        """Initializes the logger context.

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
        import pickle

        itm_path = self.savedir / identifier
        with open(itm_path, "wb") as itm_file:
            pickle.dump(obj, itm_file)

    def should_log(self, batch_idx: Optional[int] = None) -> bool:
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
            self.worker_rank is None
            or (
                isinstance(self.log_on_workers, int)
                and (self.log_on_workers == -1 or self.log_on_workers == self.worker_rank)
            )
            or (
                isinstance(self.log_on_workers, list)
                and self.worker_rank in self.log_on_workers
            )
        )
        if not worker_ok:
            return False

        # Check batch ID
        if batch_idx is not None:
            if isinstance(self.log_freq, int):
                if batch_idx % self.log_freq == 0:
                    return True
                return False
            if self.log_freq == "batch":
                return True
            return False
        return True


class ConsoleLogger(Logger):
    """Simplified logger.

    Args:
        savedir (Union[Path, str], optional): where to store artifacts.
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
    supported_kinds: Tuple[str] = ("torch", "artifact", "metric")

    def __init__(
        self,
        savedir: Union[Path, str] = "mllogs",
        log_freq: Union[int, Literal["epoch", "batch"]] = "epoch",
        log_on_workers: Union[int, List[int]] = 0,
    ) -> None:
        cl_savedir = Path(savedir) / "simple-logger"
        super().__init__(savedir=cl_savedir, log_freq=log_freq, log_on_workers=log_on_workers)

    def create_logger_context(self, rank: Optional[int] = None):
        """Initializes the logger context.

        Args:
            rank (Optional[int]): global rank of current process,
                used in distributed environments. Defaults to None.
        """
        self.worker_rank = rank

        if not self.should_log():
            return

        if self.savedir.is_dir():
            numeric_dirs = [
                int(exp_dir.name)
                for exp_dir in self.savedir.iterdir()
                if exp_dir.is_dir() and exp_dir.name.isdigit()
            ]
            self._experiment_id = max(numeric_dirs) + 1
        else:
            self._experiment_id = 0

        self.run_path = self.savedir / str(self.experiment_id)
        self.run_path.mkdir(exist_ok=True, parents=True)

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
        kind: str = "metric",
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs,
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

        if kind == "artifact":
            import shutil

            artifact_dir = self.run_path / "artifacts" / identifier
            artifact_dir.mkdir(exist_ok=True, parents=True)

            item_path = Path(item)
            if item_path.is_file():
                target_path = artifact_dir / identifier
                shutil.copyfile(item, target_path)

            elif item_path.is_dir():
                numeric_dirs = [
                    int(exp_dir.name)
                    for exp_dir in artifact_dir.iterdir()
                    if exp_dir.is_dir() and exp_dir.name.isdigit()
                ]
                child_id = max(numeric_dirs) + 1
                target_path = artifact_dir / f"{self._experiment_id}.{child_id}"
                shutil.copytree(item, target_path, dirs_exist_ok=True)
            else:
                print(
                    f"INFO: The ConsoleLogger expects an artifact to be either a path \
                      or a directory. Received instead an item of type {type(item)}. \
                        The item will be ignored and not logged."
                )

        elif kind == "torch":
            import torch

            target_path = self.run_path / identifier
            torch.save(item, target_path)
            print(f"INFO: ConsoleLogger saved to {target_path}...")

        elif kind == "metric":
            print(f"ConsoleLogger: {identifier} = {item}")


class MLFlowLogger(Logger):
    """Abstraction around MLFlow logger.

    Args:
        savedir (Union[Path, str], optional): path on local filesystem where logs are
            stored. Defaults to 'mllogs'.
        experiment_name (str, optional): experiment name. Defaults to
            ``itwinai.loggers.BASE_EXP_NAME``.
        tracking_uri (Optional[str], optional): MLFLow tracking URI.
            Overrides ``savedir`` if given. Defaults to None.
        run_description (Optional[str], optional): run description.
            Defaults to None.
        run_name (Optional[str], optional): run name.
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
        "metric",
        "figure",
        "image",
        "artifact",
        "torch",
        "dict",
        "param",
        "text",
        "model",
        "dataset",
    )

    #: Current MLFLow experiment's run.
    active_run: "mlflow.ActiveRun"

    def __init__(
        self,
        savedir: Union[Path, str] = "mllogs",
        experiment_name: str = BASE_EXP_NAME,
        tracking_uri: Optional[str] = None,
        run_description: Optional[str] = None,
        run_name: Optional[str] = None,
        log_freq: Union[int, Literal["epoch", "batch"]] = "epoch",
        log_on_workers: Union[int, List[int]] = 0,
    ):
        mfl_savedir = Path(savedir) / "mlflow"
        super().__init__(savedir=mfl_savedir, log_freq=log_freq, log_on_workers=log_on_workers)
        self.tracking_uri = tracking_uri
        self.run_description = run_description
        self.run_name = run_name
        self.experiment_name = experiment_name

        self.tracking_uri = (
            self.tracking_uri
            or os.environ.get("MLFLOW_TRACKING_URI")
            or Path(self.savedir).resolve().as_uri()
        )
        import mlflow

        self.mlflow = mlflow

    def create_logger_context(self, rank: Optional[int] = None) -> "mlflow.ActiveRun":
        """Initializes the logger context. Start MLFLow run.

        Args:
            rank (Optional[int]): global rank of current process,
                used in distributed environments. Defaults to None.

        Returns:
            mlflow.ActiveRun: active MLFlow run.
        """
        self.worker_rank = rank

        if not self.should_log():
            return

        active_run = self.mlflow.active_run()
        if active_run:
            print("Detected an active MLFlow run. Attaching to it...")
            self.active_run = active_run
        else:
            self.mlflow.set_tracking_uri(self.tracking_uri)
            self.mlflow.set_experiment(experiment_name=self.experiment_name)
            self.active_run: self.mlflow.ActiveRun = self.mlflow.start_run(
                description=self.run_description, run_name=self.run_name
            )
        self._run_id = self.active_run.info.run_id
        self._experiment_id = self.active_run.info.experiment_id

        return self.active_run

    def destroy_logger_context(self):
        """Destroy logger. End current MLFlow run."""
        if not self.should_log():
            return

        self.mlflow.end_run()

    def save_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Save hyperparameters as MLFlow parameters.

        Args:
            params (Dict[str, Any]): hyperparameters dictionary.
        """
        if not self.should_log():
            return

        for param_name, val in params.items():
            self.log(item=val, identifier=param_name, step=0, kind="param")

    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = "metric",
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs,
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

        if kind == "metric":
            self.mlflow.log_metric(key=identifier, value=item, step=step)
        elif kind == "artifact":
            if not isinstance(item, str):
                # Save the object locally and then log it
                name = os.path.basename(identifier)
                save_path = self.savedir / ".trash" / str(name)
                save_path.mkdir(os.path.dirname(save_path), exist_ok=True)
                item = self.serialize(item, save_path)
            self.mlflow.log_artifact(local_path=item, artifact_path=identifier)
        elif kind == "model":
            import torch

            if isinstance(item, torch.nn.Module):
                self.mlflow.pytorch.log_model(item, identifier)
            else:
                print("WARNING: unrecognized model type")
        elif kind == "dataset":
            # Log mlflow dataset
            # https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_input
            # It may be needed to convert item into a mlflow dataset, e.g.:
            # https://mlflow.org/docs/latest/python_api/mlflow.data.html#mlflow.data.from_pandas
            # ATM delegated to the user
            if isinstance(item, self.mlflow.data.Dataset):
                self.mlflow.log_input(item)
            else:
                print("WARNING: unrecognized dataset type. " "Must be an MLFlow dataset")
        elif kind == "torch":
            import torch

            # Save the object locally and then log it
            name = os.path.basename(identifier)
            save_path = self.savedir / ".trash" / str(name)
            save_path.mkdir(os.path.dirname(save_path), exist_ok=True)
            torch.save(item, save_path)
            # Log into mlflow
            self.mlflow.log_artifact(local_path=save_path, artifact_path=identifier)
        elif kind == "dict":
            self.mlflow.log_dict(dictionary=item, artifact_file=identifier)
        elif kind == "figure":
            self.mlflow.log_figure(
                artifact_file=identifier,
                figure=item,
                save_kwargs=kwargs.get("save_kwargs"),
            )
        elif kind == "image":
            self.mlflow.log_image(artifact_file=identifier, image=item)
        elif kind == "param":
            self.mlflow.log_param(key=identifier, value=item)
        elif kind == "text":
            self.mlflow.log_text(artifact_file=identifier, text=item)


class WandBLogger(Logger):
    """Abstraction around WandB logger.

    Args:
        savedir (Union[Path, str], optional): location on local filesystem where logs
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
        offline_mode (str, optional): Use this option if working on compute
            node without internet access. Saves logs locally.
            Defaults to 'False'.
    """

    # TODO: add support for artifacts logging

    #: Supported kinds in the ``log`` method
    supported_kinds: Tuple[str] = (
        "watch",
        "metric",
        "figure",
        "image",
        "torch",
        "dict",
        "param",
        "text",
    )

    def __init__(
        self,
        savedir: Union[Path, str] = "mllogs",
        project_name: str = BASE_EXP_NAME,
        log_freq: Union[int, Literal["epoch", "batch"]] = "epoch",
        log_on_workers: Union[int, List[int]] = 0,
        offline_mode: bool = False,
    ) -> None:
        wbl_savedir = Path(savedir) / "wandb"
        super().__init__(savedir=wbl_savedir, log_freq=log_freq, log_on_workers=log_on_workers)
        self.project_name = project_name
        self.offline_mode = offline_mode

        import wandb

        self.wandb = wandb

    def create_logger_context(self, rank: Optional[int] = None) -> None:
        """Initializes the logger context. Init WandB run.

        Args:
            rank (Optional[int]): global rank of current process,
                used in distributed environments. Defaults to None.
        """
        self.worker_rank = rank

        if not self.should_log():
            return

        (self.savedir / "wandb").mkdir(
            exist_ok=True,
            parents=True,
        )
        self.active_run = self.wandb.init(
            dir=self.savedir.resolve(),
            project=self.project_name,
            mode="offline" if self.offline_mode else "online",
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

        self.wandb.config.update(params)

    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = "metric",
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs,
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

        if kind == "watch":
            self.wandb.watch(item)
        elif kind in self.supported_kinds:
            # self.wandb.log({identifier: item}, step=step, commit=True)
            # Let WandB use its preferred step
            self.wandb.log({identifier: item}, commit=True)


class TensorBoardLogger(Logger):
    """Abstraction around TensorBoard logger, both for PyTorch and
    TensorFlow.

    Args:
        savedir (Union[Path, str], optional): location on local filesystem where logs
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
    supported_kinds: Tuple[str] = ("metric", "image", "text", "figure", "torch")

    def __init__(
        self,
        savedir: Union[Path, str] = "mllogs",
        log_freq: Union[int, Literal["epoch", "batch"]] = "epoch",
        framework: Literal["tensorflow", "pytorch"] = "pytorch",
        log_on_workers: Union[int, List[int]] = 0,
    ) -> None:
        tbl_savedir = Path(savedir) / "tensorboard"
        super().__init__(savedir=tbl_savedir, log_freq=log_freq, log_on_workers=log_on_workers)
        self.framework = framework
        if framework.lower() == "tensorflow":
            import tensorflow as tf

            self.tf = tf
            self.writer = tf.summary.create_file_writer(tbl_savedir.resolve().as_posix())
        elif framework.lower() == "pytorch":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(tbl_savedir.resolve().as_posix())
        else:
            raise ValueError("Framework must be either 'tensorflow' or 'pytorch'")

    def create_logger_context(self, rank: Optional[int] = None) -> None:
        """Initializes the logger context. Init Tensorboard run.

        Args:
            rank (Optional[int]): global rank of current process,
                used in distributed environments. Defaults to None.
        """
        self.worker_rank = rank

        if not self.should_log():
            return

        if self.framework == "tensorflow":
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

        if self.framework == "tensorflow":
            from tensorboard.plugins.hparams import api as hp

            hparams = {hp.HParam(k): v for k, v in params.items()}
            with self.writer.as_default():
                hp.hparams(hparams)
        elif self.framework == "pytorch":
            self.writer.add_hparams(params, {})

    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = "metric",
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs,
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

        if self.framework == "tensorflow":
            with self.writer.as_default():
                if kind == "metric":
                    self.tf.summary.scalar(identifier, item, step=step)
                elif kind == "image":
                    self.tf.summary.image(identifier, item, step=step)
                elif kind == "text":
                    self.tf.summary.text(identifier, item, step=step)
                elif kind == "figure":
                    self.tf.summary.figure(identifier, item, step=step)
        elif self.framework == "pytorch":
            if kind == "metric":
                self.writer.add_scalar(identifier, item, global_step=step)
            elif kind == "image":
                self.writer.add_image(identifier, item, global_step=step)
            elif kind == "text":
                self.writer.add_text(identifier, item, global_step=step)
            elif kind == "figure":
                self.writer.add_figure(identifier, item, global_step=step)
            elif kind == "torch":
                self.writer.add_graph(item)


class LoggersCollection(Logger):
    """Wrapper of a set of loggers, allowing to use them simultaneously.

    Args:
        loggers (List[Logger]): list of itwinai loggers.
    """

    #: Supported kinds are delegated to the loggers in the collection.
    supported_kinds: Tuple[str]

    def __init__(self, loggers: List[Logger]) -> None:
        super().__init__(savedir=Path("/tmp/mllogs_LoggersCollection"), log_freq=1)
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
        kind: str = "metric",
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs,
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
                **kwargs,
            )

    def create_logger_context(self, rank: Optional[int] = None) -> Any:
        """Initializes all loggers.

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
        provenance_save_dir (Union[Path, str], optional): path where to store provenance
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
        "metric",
        "flops_pb",
        "flops_pe",
        "system",
        "carbon",
        "execution_time",
        "model",
        "best_model",
        "torch",
    )

    def __init__(
        self,
        prov_user_namespace: str = "www.example.org",
        experiment_name: str = "experiment_name",
        provenance_save_dir: Union[Path, str] = "mllogs/prov_logs",
        save_after_n_logs: Optional[int] = 100,
        create_graph: Optional[bool] = True,
        create_svg: Optional[bool] = True,
        log_freq: Union[int, Literal["epoch", "batch"]] = "epoch",
        log_on_workers: Union[int, List[int]] = 0,
    ) -> None:
        super().__init__(
            savedir=provenance_save_dir,
            log_freq=log_freq,
            log_on_workers=log_on_workers,
        )
        self.prov_user_namespace = prov_user_namespace
        self.experiment_name = experiment_name
        self.provenance_save_dir = provenance_save_dir
        self.save_after_n_logs = save_after_n_logs
        self.create_graph = create_graph
        self.create_svg = create_svg

        import prov4ml

        self.prov4ml = prov4ml

    @override
    def create_logger_context(self, rank: Optional[int] = None):
        """Initializes the logger context.

        Args:
            rank (Optional[int]): global rank of current process,
                used in distributed environments. Defaults to None.
        """
        self.worker_rank = rank

        if not self.should_log():
            return

        self.prov4ml.start_run(
            prov_user_namespace=self.prov_user_namespace,
            experiment_name=self.experiment_name,
            provenance_save_dir=self.provenance_save_dir,
            save_after_n_logs=self.save_after_n_logs,
            # This class will control which workers can log
            collect_all_processes=True,
            rank=rank,
        )

    @override
    def destroy_logger_context(self):
        """
        Destroys the logger context.
        """
        if not self.should_log():
            return

        self.prov4ml.end_run(create_graph=self.create_graph, create_svg=self.create_svg)

    @override
    def save_hyperparameters(self, params: Dict[str, Any]) -> None:
        if not self.should_log():
            return

        # Save hyperparams
        for param_name, val in params.items():
            self.prov4ml.log_param(param_name, val)

    @override
    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = "metric",
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        context: Optional[str] = "training",
        **kwargs,
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
            self.prov4ml.log_metric(key=identifier, value=item, context=context, step=step)
        elif kind == "flops_pb":
            model, batch = item
            self.prov4ml.log_flops_per_batch(
                identifier, model=model, batch=batch, context=context, step=step
            )
        elif kind == "flops_pe":
            model, dataset = item
            self.prov4ml.log_flops_per_epoch(
                identifier, model=model, dataset=dataset, context=context, step=step
            )
        elif kind == "system":
            self.prov4ml.log_system_metrics(context=context, step=step)
        elif kind == "carbon":
            self.prov4ml.log_carbon_metrics(context=context, step=step)
        elif kind == "execution_time":
            self.prov4ml.log_current_execution_time(
                label=identifier, context=context, step=step
            )
        elif kind == "model":
            self.prov4ml.save_model_version(
                model=item, model_name=identifier, context=context, step=step
            )
        elif kind == "best_model":
            self.prov4ml.log_model(
                model=item,
                model_name=identifier,
                log_model_info=True,
                log_as_artifact=True,
            )
        elif kind == "torch":
            from torch.utils.data import DataLoader

            if isinstance(item, DataLoader):
                self.prov4ml.log_dataset(dataset=item, label=identifier)
            else:
                self.prov4ml.log_param(key=identifier, value=item)
        elif kind == "prov_documents":
            prov_docs = self.prov4ml.log_provenance_documents(
                create_graph=True, create_svg=True
            )

            # Upload to MLFlow
            if self.mlflow.active_run() is not None:
                for f in prov_docs:
                    if f:
                        self.mlflow.log_artifact(f)


class EpochTimeTracker:
    """Tracker for epoch execution time during training."""

    def __init__(self, strategy_name: str, save_path: Path | str, num_nodes: int) -> None:
        if isinstance(save_path, str):
            save_path = Path(save_path)

        self.save_path: Path = save_path
        self.strategy_name = strategy_name
        self.num_nodes = num_nodes
        self.data = {"epoch_id": [], "time": []}

    def add_epoch_time(self, epoch_idx: int, time: float) -> None:
        """Add epoch time to data."""
        self.data["epoch_id"].append(epoch_idx)
        self.data["time"].append(time)

    def save(self) -> None:
        """Save data to a new CSV file."""
        import pandas as pd

        df = pd.DataFrame(self.data)
        df["name"] = self.strategy_name
        df["nodes"] = self.num_nodes

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.save_path, index=False)
        print(f"Saving EpochTimeTracking data to '{self.save_path.resolve()}'.")


class EmptyLogger(Logger):
    """Dummy logger which can be used as a placeholder when a real logger is
    not available. All methods do nothing.
    """

    def __init__(
        self,
        savedir: Union[Path, str] = "mllogs",
        log_freq: int | Literal["epoch"] | Literal["batch"] = "epoch",
        log_on_workers: int | List[int] = 0,
    ) -> None:
        super().__init__(savedir, log_freq, log_on_workers)

    def create_logger_context(self, rank: Optional[int] = None):
        pass

    def destroy_logger_context(self):
        pass

    def save_hyperparameters(self, params: Dict[str, Any]) -> None:
        pass

    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = "metric",
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs,
    ) -> None:
        pass
