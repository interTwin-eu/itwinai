"""Provides training logic for PyTorch models via Trainer classes."""

from typing import (
    Optional, Dict, Union, Tuple, List, Any, Literal
)
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import torch.optim as optim
import pandas as pd
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import lightning as L
from lightning.pytorch.cli import LightningCLI

import horovod.torch as hvd

from ..components import Trainer, monitor_exec
from .type import (
    Batch, Loss, LrScheduler, Metric
)
from ..loggers import LogMixin, Logger
from .reproducibility import seed_worker, set_seed
from .distributed import (
    TorchDistributedStrategy,
    TorchDDPStrategy,
    HorovodStrategy,
    DeepSpeedStrategy,
    NonDistributedStrategy,
    distributed_resources_available
)
from ..utils import load_yaml
from .mlflow import (
    init_lightning_mlflow,
    teardown_lightning_mlflow
)
from .config import TrainingConfiguration


class TorchTrainer(Trainer, LogMixin):
    """Trainer class for torch training algorithms.

    Args:
        config (Union[Dict, TrainingConfiguration]): training configuration
            containing hyperparameters.
        epochs (int): number of training epochs.
        model (Optional[Union[nn.Module, str]], optional): pytorch model to
            train or a string identifier. Defaults to None.
        strategy (Literal['ddp', 'deepspeed', 'horovod'], optional):
            distributed strategy. Defaults to 'ddp'.
        validation_every (Optional[int], optional): run a validation epoch
            every ``validation_every`` epochs. Disabled if None. Defaults to 1.
        test_every (Optional[int], optional): run a test epoch
            every ``test_every`` epochs. Disabled if None. Defaults to None.
        random_seed (Optional[int], optional): set random seed for
            reproducibility. If None, the seed is not set. Defaults to None.
        logger (Optional[Logger], optional): logger for ML tracking.
            Defaults to None.
        log_all_workers (bool, optional): if True, the ``log`` method is
            called on all workers in the distributed context.
            Defaults to False.
        metrics (Optional[Dict[str, Metric]], optional): map of torchmetrics
            metrics. Defaults to None.
        checkpoints_location (str): path to checkpoints directory.
            Defaults to "checkpoints".
        checkpoint_every (Optional[int]): save a checkpoint every
            ``checkpoint_every`` epochs. Disabled if None. Defaults to None.
        name (Optional[str], optional): trainer custom name. Defaults to None.
    """
    # TODO:
    #   - extract BaseTorchTrainer and extend it creating a set of trainer
    #     templates (e.g.. GAN, Classifier, Transformer) allowing scientists
    #     to reuse ML algos.

    _strategy: TorchDistributedStrategy = None

    #: PyTorch ``DataLoader`` for training dataset.
    train_dataloader: DataLoader = None
    #: PyTorch ``DataLoader`` for validation dataset.
    validation_dataloader: DataLoader = None
    #: PyTorch ``DataLoader`` for test dataset.
    test_dataloader: DataLoader = None
    #: PyTorch model to train.
    model: nn.Module = None
    #: Loss criterion.
    loss: Loss = None
    #: Optimizer.
    optimizer: Optimizer = None
    #: Learning rate scheduler.
    lr_scheduler: LrScheduler = None
    #: PyTorch random number generator (PRNG).
    torch_rng: torch.Generator = None
    #: itwinai ``itwinai.Logger``
    logger: Logger = None
    #: Total number training batches used so far, across all epochs.
    train_glob_step: int = 0
    #: Total number validation batches used so far, across all epochs.
    validation_glob_step: int = 0
    #: Total number test batches used so far, across all epochs.
    test_glob_step: int = 0
    #: Dictionary of ``torchmetrics`` metrics, indexed by user-defined names.
    metrics: Dict[str, Metric]

    def __init__(
        self,
        config: Union[Dict, TrainingConfiguration],
        epochs: int,
        model: Optional[Union[nn.Module, str]] = None,
        strategy: Literal["ddp", "deepspeed", "horovod"] = 'ddp',
        validation_every: Optional[int] = 1,
        test_every: Optional[int] = None,
        random_seed: Optional[int] = None,
        logger: Optional[Logger] = None,
        log_all_workers: bool = False,
        metrics: Optional[Dict[str, Metric]] = None,
        checkpoints_location: str = "checkpoints",
        checkpoint_every: Optional[int] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name)
        self.save_parameters(**self.locals2params(locals()))

        # config is mean to store all hyperparameters, which can very from use
        # case to use case
        # and include learning_rate, batch_size....
        if isinstance(config, dict):
            self.config = TrainingConfiguration(**config)
        else:
            self.config = config
        self.epochs = epochs
        self.model = model
        self.strategy = strategy
        self.validation_every = validation_every
        self.test_every = test_every
        self.random_seed = random_seed
        self.logger = logger
        self.log_all_workers = log_all_workers
        self.metrics = metrics if metrics is not None else {}
        self.checkpoints_location = checkpoints_location
        os.makedirs(self.checkpoints_location, exist_ok=True)
        self.checkpoint_every = checkpoint_every

    @property
    def strategy(self) -> TorchDistributedStrategy:
        """Strategy currently in use."""
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Union[str, TorchDistributedStrategy]) -> None:
        if isinstance(strategy, TorchDistributedStrategy):
            self._strategy = strategy
        else:
            self._strategy = self._detect_strategy(strategy)

    @property
    def device(self) -> str:
        """Current device from distributed strategy."""
        return self.strategy.device()

    def _detect_strategy(self, strategy: str) -> TorchDistributedStrategy:
        if not distributed_resources_available():
            print("WARNING: falling back to non-distributed strategy.")
            dist_str = NonDistributedStrategy()
        elif strategy == 'ddp':
            dist_str = TorchDDPStrategy(backend='nccl')
        elif strategy == 'horovod':
            dist_str = HorovodStrategy()
        elif strategy == 'deepspeed':
            dist_str = DeepSpeedStrategy(backend='nccl')
        else:
            raise NotImplementedError(
                f"Strategy '{strategy}' is not recognized/implemented.")
        return dist_str

    def _init_distributed_strategy(self) -> None:
        if not self.strategy.is_initialized:
            self.strategy.init()

    def _optimizer_from_config(self) -> None:
        if self.config.optimizer == 'adadelta':
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.config.optim_lr,
                weight_decay=self.config.optim_weight_decay
            )
        elif self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.optim_lr,
                weight_decay=self.config.optim_weight_decay
            )
        elif self.config.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=self.config.optim_lr,
                weight_decay=self.config.optim_weight_decay,
                momentum=self.config.optim_momentum
            )
        elif self.config.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.optim_lr,
                weight_decay=self.config.optim_weight_decay,
                momentum=self.config.optim_momentum
            )
        else:
            raise ValueError(
                "Unrecognized self.config.optimizer! Check the docs for "
                "supported values and consider overriding "
                "create_model_loss_optimizer method for more flexibility."
            )

    def _loss_from_config(self) -> None:
        if self.config.loss == 'nllloss':
            self.loss = nn.functional.nll_loss
        elif self.config.loss == 'cross_entropy':
            self.loss = nn.functional.cross_entropy
        elif self.config.loss == 'mse':
            self.loss = nn.functional.mse_loss
        else:
            raise ValueError(
                "Unrecognized self.config.loss! Check the docs for "
                "supported values and consider overriding "
                "create_model_loss_optimizer method for more flexibility."
            )

    def create_model_loss_optimizer(self) -> None:
        """
        Instantiate a torch model, loss, optimizer, and LR scheduler using the
        configuration provided in the Trainer constructor.
        Generally a user-define method.
        """
        ###################################
        # Dear user, this is a method you #
        # may be interested to override!  #
        ###################################

        if self.model is None:
            raise ValueError(
                "self.model is None! Either pass it to the constructor or "
                "override create_model_loss_optimizer method."
            )

        # Parse optimizer from training configuration
        # Optimizer can be changed with a custom one here!
        self._optimizer_from_config()

        # Parse loss from training configuration
        # Loss can be changed with a custom one here!
        self._loss_from_config()

        # IMPORTANT: model, optimizer, and scheduler need to be distributed

        # First, define strategy-wise optional configurations
        if isinstance(self.strategy, DeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = dict(
                config_params=dict(
                    train_micro_batch_size_per_gpu=self.config.batch_size
                )
            )
        elif isinstance(self.strategy, HorovodStrategy):
            distribute_kwargs = dict(
                compression=(
                    hvd.Compression.fp16 if self.config.fp16_allreduce
                    else hvd.Compression.none
                ),
                op=hvd.Adasum if self.config.use_adasum else hvd.Average,
                gradient_predivide_factor=self.config.gradient_predivide_factor
            )
        else:
            distribute_kwargs = {}

        # Distributed model, optimizer, and scheduler
        (
            self.model,
            self.optimizer,
            self.lr_scheduler
        ) = self.strategy.distributed(
            self.model, self.optimizer, self.lr_scheduler, **distribute_kwargs
        )

    def create_dataloaders(
        self,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None
    ) -> None:
        """
        Create train, validation and test dataloaders using the
        configuration provided in the Trainer constructor.
        Generally a user-define method.

        Args:
            train_dataset (Dataset): training dataset object.
            validation_dataset (Optional[Dataset]): validation dataset object.
                Default None.
            test_dataset (Optional[Dataset]): test dataset object.
                Default None.
        """

        ###################################
        # Dear user, this is a method you #
        # may be interested to override!  #
        ###################################

        self.train_dataloader = self.strategy.create_dataloader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers_dataloader,
            pin_memory=self.config.pin_gpu_memory,
            generator=self.torch_rng,
            shuffle=self.config.shuffle_train
        )
        if validation_dataset is not None:
            self.validation_dataloader = self.strategy.create_dataloader(
                dataset=validation_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers_dataloader,
                pin_memory=self.config.pin_gpu_memory,
                generator=self.torch_rng,
                shuffle=self.config.shuffle_validation
            )
        if test_dataset is not None:
            self.test_dataloader = self.strategy.create_dataloader(
                dataset=test_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers_dataloader,
                pin_memory=self.config.pin_gpu_memory,
                generator=self.torch_rng,
                shuffle=self.config.shuffle_test
            )

    def _setup_metrics(self):
        """Move metrics to current device."""
        for m_name, metric in self.metrics.items():
            self.metrics[m_name] = metric.to(self.device)

    @monitor_exec
    def execute(
        self,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None
    ) -> Tuple[Dataset, Dataset, Dataset, Any]:
        """Prepares distributed environment and data structures
        for the actual training.

        Args:
            train_dataset (Dataset): training dataset.
            validation_dataset (Optional[Dataset], optional): validation
                dataset. Defaults to None.
            test_dataset (Optional[Dataset], optional): test dataset.
                Defaults to None.

        Returns:
            Tuple[Dataset, Dataset, Dataset, Any]: training dataset,
            validation dataset, test dataset, trained model.
        """
        self.torch_rng = set_seed(self.random_seed)
        self._init_distributed_strategy()
        self._setup_metrics()

        self.create_dataloaders(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset
        )
        self.create_model_loss_optimizer()

        if self.strategy.is_main_worker and self.logger:
            self.logger.create_logger_context()

            hparams = self.config.model_dump()
            hparams['distributed_strategy'] = self.strategy.__class__.__name__
            self.logger.save_hyperparameters(hparams)

        self.train()

        if self.strategy.is_main_worker and self.logger:
            self.logger.destroy_logger_context()
        self.strategy.clean_up()
        return train_dataset, validation_dataset, test_dataset, self.model

    def _set_epoch_dataloaders(self, epoch: int):
        """
        Sets epoch in the distributed sampler of a dataloader when using it.
        """
        if self.strategy.is_distributed:
            self.train_dataloader.sampler.set_epoch(epoch)
            if self.validation_dataloader is not None:
                self.validation_dataloader.sampler.set_epoch(epoch)
            if self.test_dataloader is not None:
                self.test_dataloader.sampler.set_epoch(epoch)

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch at the beginning of training.

        Args:
            epoch (int): epoch number, from 0 to ``epochs-1``.
        """
        self._set_epoch_dataloaders(epoch)

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
                among the list of self.supported_types. Defaults to 'metric'.
            step (Optional[int], optional): logging step. Defaults to None.
            batch_idx (Optional[int], optional): DataLoader batch counter
                (i.e., batch idx), if available. Defaults to None.
        """
        if self.logger and (
                self.strategy.is_main_worker or self.log_all_workers):
            self.logger.log(
                item=item,
                identifier=identifier,
                kind=kind,
                step=step,
                batch_idx=batch_idx,
                **kwargs
            )

    def save_checkpoint(
            self, name: str, epoch: int, loss: Optional[torch.Tensor] = None
    ) -> None:
        """Save training checkpoint.

        Args:
            name (str): name of the checkpoint.
            epoch (int): current training epoch.
            loss (Optional[torch.Tensor]): current loss (if available).
        """
        state = dict(
            epoch=epoch,
            loss=loss,
            optimizer=self.optimizer.state_dict(),
            model=self.model.state_dict(),
            lr_scheduler=self.lr_scheduler
        )
        ckpt_path = os.path.join(self.checkpoints_location, name)
        torch.save(state, ckpt_path)
        print(f"Saved '{name}' checkpoint at {ckpt_path}")

        # Save checkpoint to logger
        self.log(ckpt_path, name, kind='artifact')

    def load_checkpoint(self, name: str) -> None:
        """Load state from a checkpoint.

        Args:
            name (str): name of the checkpoint to load, assuming it
                is under ``self.checkpoints_location`` location.
        """
        ckpt_path = os.path.join(self.checkpoints_location, name)
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler = state['lr_scheduler']

    def compute_metrics(
        self,
        true: Batch,
        pred: Batch,
        logger_step: int,
        batch_idx: Optional[int],
        stage: str = 'train'
    ) -> Dict[str, Any]:
        """Compute and log metrics.

        Args:
            metrics (Dict[str, Metric]): metrics dict. Can be
                ``self.train_metrics`` or ``self.validation_metrics``.
            true (Batch): true values.
            pred (Batch): predicted values.
            logger_step (int): global step to pass to the logger.
            stage (str): 'train', 'validation'...

        Returns:
            Dict[str, Any]: metric values.
        """
        m_values = {}
        for m_name, metric in self.metrics.items():
            # metric = metric.to(self.device)
            m_val = metric(pred, true).detach().cpu().numpy()
            self.log(
                item=m_val,
                identifier=f'{stage}_{m_name}',
                kind='metric',
                step=logger_step,
                batch_idx=batch_idx
            )
            m_values[m_name] = m_val
        return m_values

    def train(self):
        """Trains a machine learning model.
        Main training loop/logic.

        Args:
            train_dataset (Dataset): training dataset.
            validation_dataset (Dataset): validation dataset.
            test_dataset (Dataset): test dataset.

        Returns:
            Tuple[Dataset, Dataset, Dataset, Any]: training dataset,
            validation dataset, test dataset, trained model.
        """
        best_loss = float('inf')
        for epoch in range(self.epochs):
            epoch_n = epoch + 1
            self.set_epoch(epoch)
            self.train_epoch(epoch)
            if self.validation_every and epoch_n % self.validation_every == 0:
                val_loss = self.validation_epoch(epoch)

                # Checkpointing current best model
                worker_val_losses = self.strategy.gather(
                    val_loss, dst_rank=0)
                if self.strategy.global_rank() == 0:
                    avg_loss = torch.mean(
                        torch.stack(worker_val_losses)
                    ).detach().cpu()
                    if avg_loss < best_loss:
                        ckpt_name = "best_model.pth"
                        self.save_checkpoint(
                            name=ckpt_name, epoch=epoch, loss=avg_loss)
                        best_loss = avg_loss

            if self.test_every and epoch_n % self.test_every == 0:
                self.test_epoch(epoch)

            # Periodic checkpointing
            if (self.strategy.is_main_worker and self.checkpoint_every
                    and epoch_n % self.checkpoint_every == 0):
                ckpt_name = f"epoch_{epoch}.pth"
                self.save_checkpoint(name=ckpt_name, epoch=epoch)

    def train_epoch(self, epoch: int) -> Loss:
        """Perform a complete sweep over the training dataset, completing an
        epoch of training.

        Args:
            epoch (int): current epoch number, from 0 to ``self.epochs - 1``.

        Returns:
            Loss: average training loss for the current epoch.
        """
        self.model.train()
        train_losses = []
        train_metrics = []
        for batch_idx, train_batch in enumerate(self.train_dataloader):
            loss, metrics = self.train_step(
                batch=train_batch,
                batch_idx=batch_idx
            )
            train_losses.append(loss)
            train_metrics.append(metrics)

            # Important: update counter
            self.train_glob_step += 1

        # Aggregate and log losses
        avg_loss = torch.mean(torch.stack(train_losses))
        self.log(
            item=avg_loss.item(),
            identifier='train_loss_epoch',
            kind='metric',
            step=self.train_glob_step,
        )
        # Aggregate and log metrics
        avg_metrics = pd.DataFrame(train_metrics).mean().to_dict()
        for m_name, m_val in avg_metrics.items():
            self.log(
                item=m_val,
                identifier='train_' + m_name + '_epoch',
                kind='metric',
                step=self.train_glob_step,
            )

        return avg_loss.item()

    def train_step(
        self,
        batch: Batch,
        batch_idx: int
    ) -> Tuple[Loss, Dict[str, Any]]:
        """Perform a single optimization step using a batch sampled from the
        training dataset.

        Args:
            batch (Batch): batch sampled by a dataloader.
            batch_idx (int): batch index in the dataloader.

        Returns:
            Tuple[Loss, Dict[str, Any]]: batch loss and dictionary of metric
            values with the same structure of ``self.metrics``.
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        self.optimizer.zero_grad()
        pred_y = self.model(x)
        loss = self.loss(pred_y, y)
        loss.backward()
        self.optimizer.step()

        # Log metrics
        self.log(
            item=loss.item(),
            identifier='train_loss',
            kind='metric',
            step=self.train_glob_step,
            batch_idx=batch_idx
        )
        metrics: Dict[str, Any] = self.compute_metrics(
            true=y,
            pred=pred_y,
            logger_step=self.train_glob_step,
            batch_idx=batch_idx,
            stage='train'
        )
        return loss, metrics

    def validation_epoch(self, epoch: int) -> Loss:
        """Perform a complete sweep over the validation dataset, completing an
        epoch of validation.

        Args:
            epoch (int): current epoch number, from 0 to ``self.epochs - 1``.

        Returns:
            Loss: average validation loss for the current epoch.
        """
        if self.validation_dataloader is not None:
            self.model.eval()
            validation_losses = []
            validation_metrics = []
            for batch_idx, val_batch \
                    in enumerate(self.validation_dataloader):
                loss, metrics = self.validation_step(
                    batch=val_batch,
                    batch_idx=batch_idx
                )
                validation_losses.append(loss)
                validation_metrics.append(metrics)

                # Important: update counter
                self.validation_glob_step += 1

            # Aggregate and log losses
            avg_loss = torch.mean(torch.stack(validation_losses))
            self.log(
                item=avg_loss.item(),
                identifier='validation_loss_epoch',
                kind='metric',
                step=self.validation_glob_step,
            )
            # Aggregate and log metrics
            avg_metrics = pd.DataFrame(validation_metrics).mean().to_dict()
            for m_name, m_val in avg_metrics.items():
                self.log(
                    item=m_val,
                    identifier='validation_' + m_name + '_epoch',
                    kind='metric',
                    step=self.validation_glob_step,
                )

            return avg_loss

    def validation_step(
        self,
        batch: Batch,
        batch_idx: int
    ) -> Tuple[Loss, Dict[str, Any]]:
        """Perform a single optimization step using a batch sampled from the
        validation dataset.

        Args:
            batch (Batch): batch sampled by a dataloader.
            batch_idx (int): batch index in the dataloader.

        Returns:
            Tuple[Loss, Dict[str, Any]]: batch loss and dictionary of metric
            values with the same structure of ``self.metrics``.
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        with torch.no_grad():
            pred_y = self.model(x)
            loss: Loss = self.loss(pred_y, y)
        self.log(
            item=loss.item(),
            identifier='validation_loss',
            kind='metric',
            step=self.validation_glob_step,
            batch_idx=batch_idx
        )
        metrics: Dict[str, Any] = self.compute_metrics(
            true=y,
            pred=pred_y,
            logger_step=self.validation_glob_step,
            batch_idx=batch_idx,
            stage='validation'
        )
        return loss, metrics

    def test_epoch(self, epoch: int) -> Loss:
        """Perform a complete sweep over the test dataset, completing an
        epoch of test.

        Args:
            epoch (int): current epoch number, from 0 to ``self.epochs - 1``.

        Returns:
            Loss: average test loss for the current epoch.
        """
        raise NotImplementedError()

    def test_step(
        self,
        batch: Batch,
        batch_idx: int
    ) -> Tuple[Loss, Dict[str, Any]]:
        """Perform a single predictions step using a batch sampled from the
        test dataset.

        Args:
            batch (Batch): batch sampled by a dataloader.
            batch_idx (int): batch index in the dataloader.

        Returns:
            Tuple[Loss, Dict[str, Any]]: batch loss and dictionary of metric
            values with the same structure of ``self.metrics``.
        """
        raise NotImplementedError()


class GANTrainer(TorchTrainer):
    """Trainer class for GAN models using pytorch.

    Args:
        config (Union[Dict, TrainingConfiguration]): training configuration
            containing hyperparameters.
        epochs (int): number of training epochs.
        discriminator (nn.Module): pytorch discriminator model to train GAN.
        generator (nn.Module): pytorch generator model to train GAN.
        strategy (Literal['ddp', 'deepspeed', 'horovod'], optional):
            distributed strategy. Defaults to 'ddp'.
        validation_every (Optional[int], optional): run a validation epoch
            every ``validation_every`` epochs. Disabled if None. Defaults to 1.
        test_every (Optional[int], optional): run a test epoch
            every ``test_every`` epochs. Disabled if None. Defaults to None.
        random_seed (Optional[int], optional): set random seed for
            reproducibility. If None, the seed is not set. Defaults to None.
        logger (Optional[Logger], optional): logger for ML tracking.
            Defaults to None.
        log_all_workers (bool, optional): if True, the ``log`` method is
            called on all workers in the distributed context.
            Defaults to False.
        metrics (Optional[Dict[str, Metric]], optional): map of torch metrics
            metrics. Defaults to None.
        checkpoints_location (str): path to checkpoints directory.
            Defaults to "checkpoints".
        checkpoint_every (Optional[int]): save a checkpoint every
            ``checkpoint_every`` epochs. Disabled if None. Defaults to None.
        name (Optional[str], optional): trainer custom name. Defaults to None.
    """

    def __init__(
            self,
            config: Union[Dict, TrainingConfiguration],
            epochs: int,
            discriminator: nn.Module,
            generator: nn.Module,
            strategy: Literal["ddp", "deepspeed"] = 'ddp',
            validation_every: Optional[int] = 1,
            test_every: Optional[int] = None,
            random_seed: Optional[int] = None,
            logger: Optional[Logger] = None,
            log_all_workers: bool = False,
            metrics: Optional[Dict[str, Metric]] = None,
            checkpoints_location: str = "checkpoints",
            checkpoint_every: Optional[int] = None,
            name: Optional[str] = None, **kwargs) -> None:
        super().__init__(
            config=config,
            epochs=epochs,
            model=None,
            strategy=strategy,
            validation_every=validation_every,
            test_every=test_every,
            random_seed=random_seed,
            logger=logger,
            log_all_workers=log_all_workers,
            metrics=metrics,
            checkpoints_location=checkpoints_location,
            checkpoint_every=checkpoint_every,
            name=name,
            **kwargs)
        self.save_parameters(**self.locals2params(locals()))
        self.discriminator = discriminator
        self.generator = generator

    def create_model_loss_optimizer(self) -> None:
        self.optimizerD = optim.Adam(
            self.discriminator.parameters(), lr=self.config.lr,
            betas=(0.5, 0.999)
        )
        self.optimizerG = optim.Adam(
            self.generator.parameters(), lr=self.config.lr,
            betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

        # https://stackoverflow.com/a/67437077
        self.discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            self.discriminator)
        self.generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            self.generator)

        # First, define strategy-wise optional configurations
        if isinstance(self.strategy, DeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = dict(
                config_params=dict(
                    train_micro_batch_size_per_gpu=self.config.batch_size
                )
            )
        else:
            distribute_kwargs = {}
        # Distribute discriminator and its optimizer
        self.discriminator, self.optimizerD, _ = self.strategy.distributed(
            self.discriminator, self.optimizerD, **distribute_kwargs)
        self.generator, self.optimizerG, _ = self.strategy.distributed(
            self.generator, self.optimizerG, **distribute_kwargs)

    def train_epoch(self, epoch: int):
        self.discriminator.train()
        self.generator.train()
        gen_train_losses = []
        disc_train_losses = []
        disc_train_accuracy = []
        for batch_idx, (real_images, _) in enumerate(self.train_dataloader):
            lossG, lossD, accuracy_disc = self.train_step(
                real_images, batch_idx)
            gen_train_losses.append(lossG)
            disc_train_losses.append(lossD)
            disc_train_accuracy.append(accuracy_disc)

            self.train_glob_step += 1
        # Aggregate and log losses and accuracy
        avg_disc_accuracy = torch.mean(torch.stack(disc_train_accuracy))
        self.log(
            item=avg_disc_accuracy.item(),
            identifier='disc_train_accuracy_per_epoch',
            kind='metric',
            step=epoch,
        )
        avg_gen_loss = torch.mean(torch.stack(gen_train_losses))
        self.log(
            item=avg_gen_loss.item(),
            identifier='gen_train_loss_per_epoch',
            kind='metric',
            step=epoch,
        )

        avg_disc_loss = torch.mean(torch.stack(disc_train_losses))
        self.log(
            item=avg_disc_loss.item(),
            identifier='disc_train_loss_per_epoch',
            kind='metric',
            step=epoch,
        )

        self.save_fake_generator_images(epoch)

    def validation_epoch(self, epoch: int):
        gen_validation_losses = []
        gen_validation_accuracy = []
        disc_validation_losses = []
        disc_validation_accuracy = []
        self.discriminator.eval()
        self.generator.eval()
        for batch_idx, (real_images, _) in enumerate(
                self.validation_dataloader):
            loss_gen, accuracy_gen, loss_disc, accuracy_disc = (
                self.validation_step(real_images, batch_idx))
            gen_validation_losses.append(loss_gen)
            gen_validation_accuracy.append(accuracy_gen)
            disc_validation_losses.append(loss_disc)
            disc_validation_accuracy.append(accuracy_disc)
            self.validation_glob_step += 1

        # Aggregate and log metrics
        disc_validation_loss = torch.mean(torch.stack(
            disc_validation_losses))
        self.log(
            item=disc_validation_loss.item(),
            identifier='disc_valid_loss_per_epoch',
            kind='metric',
            step=epoch,
        )
        disc_validation_accuracy = torch.mean(torch.stack(
            disc_validation_accuracy))
        self.log(
            item=disc_validation_accuracy.item(),
            identifier='disc_valid_accuracy_epoch',
            kind='metric',
            step=epoch,
        )
        gen_validation_loss = torch.mean(torch.stack(
            gen_validation_losses))
        self.log(
            item=gen_validation_loss.item(),
            identifier='gen_valid_loss_per_epoch',
            kind='metric',
            step=epoch,
        )
        gen_validation_accuracy = torch.mean(torch.stack(
            gen_validation_accuracy))
        self.log(
            item=gen_validation_accuracy.item(),
            identifier='gen_valid_accuracy_epoch',
            kind='metric',
            step=epoch,
        )

        return gen_validation_loss

    def train_step(self, real_images, batch_idx):
        real_images = real_images.to(self.device)
        batch_size = real_images.size(0)
        real_labels = torch.ones(
            (batch_size,),
            dtype=torch.float, device=self.device)
        fake_labels = torch.zeros(
            (batch_size,),
            dtype=torch.float, device=self.device)

        # Train Discriminator with real images
        output_real = self.discriminator(real_images)
        lossD_real = self.criterion(output_real, real_labels)
        # Generate fake images and train Discriminator
        noise = torch.randn(
            batch_size, self.config.z_dim, 1, 1, device=self.device)

        fake_images = self.generator(noise)
        output_fake = self.discriminator(fake_images.detach())
        lossD_fake = self.criterion(output_fake, fake_labels)

        lossD = (lossD_real+lossD_fake)/2

        self.optimizerD.zero_grad()
        lossD.backward()
        self.optimizerD.step()

        accuracy = ((output_real > 0.5).float() == real_labels).float().mean(
        ) + ((output_fake < 0.5).float() == fake_labels).float().mean()
        accuracy_disc = accuracy.mean()

        # Train Generator
        output_fake = self.discriminator(fake_images)
        lossG = self.criterion(output_fake, real_labels)
        self.optimizerG.zero_grad()
        lossG.backward()
        self.optimizerG.step()
        self.log(
            item=accuracy_disc,
            identifier='disc_train_accuracy_per_batch',
            kind='metric',
            step=self.train_glob_step,
            batch_idx=batch_idx
        )
        self.log(
            item=lossG,
            identifier='gen_train_loss_per_batch',
            kind='metric',
            step=self.train_glob_step,
            batch_idx=batch_idx
        )
        self.log(
            item=lossD,
            identifier='disc_train_loss_per_batch',
            kind='metric',
            step=self.train_glob_step,
            batch_idx=batch_idx
        )

        return lossG, lossD, accuracy_disc

    def validation_step(self, real_images, batch_idx):
        real_images = real_images.to(self.device)
        batch_size = real_images.size(0)
        real_labels = torch.ones((batch_size,),
                                 dtype=torch.float, device=self.device)
        fake_labels = torch.zeros((batch_size,),
                                  dtype=torch.float, device=self.device)

        # Validate with real images
        output_real = self.discriminator(real_images)
        loss_real = self.criterion(output_real, real_labels)

        # Generate and validate fake images
        noise = torch.randn(
            batch_size, self.config.z_dim, 1, 1, device=self.device)

        with torch.no_grad():
            fake_images = self.generator(noise)
            output_fake = self.discriminator(fake_images.detach())
        loss_fake = self.criterion(output_fake, fake_labels)

        # Generator's attempt to fool the discriminator
        loss_gen = self.criterion(output_fake, real_labels)
        accuracy_gen = (
            (output_fake > 0.5).float() == real_labels).float().mean()

        # Calculate total discriminator loss and accuracy
        d_total_loss = (loss_real + loss_fake) / 2
        accuracy = ((output_real > 0.5).float() == real_labels).float().mean(
        ) + ((output_fake < 0.5).float() == fake_labels).float().mean()
        d_accuracy = accuracy.item()/2

        self.log(
            item=loss_gen.item(),
            identifier='gen_valid_loss_per_batch',
            kind='metric',
            step=self.validation_glob_step,
            batch_idx=batch_idx
        )
        self.log(
            item=accuracy_gen.item(),
            identifier='gen_valid_accuracy_per_batch',
            kind='metric',
            step=self.validation_glob_step,
            batch_idx=batch_idx
        )

        self.log(
            item=d_total_loss.item(),
            identifier='disc_valid_loss_per_batch',
            kind='metric',
            step=self.validation_glob_step,
            batch_idx=batch_idx
        )
        self.log(
            item=d_accuracy,
            identifier='disc_valid_accuracy_per_batch',
            kind='metric',
            step=self.validation_glob_step,
            batch_idx=batch_idx
        )
        return loss_gen, accuracy_gen

    def save_checkpoint(self, name, epoch, loss=None):
        """Save training checkpoint with both optimizers."""
        if not os.path.exists(self.checkpoints_location):
            os.makedirs(self.checkpoints_location)

        checkpoint_path = os.path.join(self.checkpoints_location, f"{name}")
        checkpoint = {
            'epoch': epoch,
            'loss': loss.item() if loss is not None else None,
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_state_dict': self.generator.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if
            self.lr_scheduler else None
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load models and optimizers from checkpoint."""
        checkpoint = torch.load(checkpoint_path)

        self.discriminator.load_state_dict(
            checkpoint['discriminator_state_dict'])
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])

        if 'lr_scheduler' in checkpoint:
            if checkpoint['lr_scheduler'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        print(f"Checkpoint loaded from {checkpoint_path}")

    def save_fake_generator_images(self, epoch):
        """
        plot and save fake images from generator

         Args:
            epoch (int): epoch number, from 0 to ``epochs-1``.
        """
        self.generator.eval()
        noise = torch.randn(64, self.config.z_dim, 1, 1, device=self.device)
        fake_images = self.generator(noise)
        fake_images_grid = torchvision.utils.make_grid(
            fake_images, normalize=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_axis_off()
        ax.set_title(f'Fake images for epoch {epoch}')
        ax.imshow(np.transpose(fake_images_grid.cpu().numpy(), (1, 2, 0)))
        self.log(
            item=fig,
            identifier=f'fake_images_epoch_{epoch}.png',
            kind='figure',
            step=epoch,
        )


class TorchLightningTrainer(Trainer):
    """Generic trainer for torch Lightning workflows.

        Args:
            config (Union[Dict, str]): `Lightning configuration`_
                which can be the path to a file or a Python dictionary.
            mlflow_saved_model (str, optional): name of the model created in
                MLFlow. Defaults to 'my_model'.

        .. _Lightning configuration:
            https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html
        """

    def __init__(
        self,
        config: Union[Dict, str],
        mlflow_saved_model: str = 'my_model'
    ):
        self.save_parameters(**self.locals2params(locals()))
        super().__init__()
        if isinstance(config, str) and os.path.isfile(config):
            # Load from YAML
            config = load_yaml(config)
        self.conf = config
        self.mlflow_saved_model = mlflow_saved_model

    @monitor_exec
    def execute(self) -> Any:
        init_lightning_mlflow(
            self.conf,
            tmp_dir='/tmp',
            registered_model_name=self.mlflow_saved_model
        )
        old_argv = sys.argv
        sys.argv = ['some_script_placeholder.py']
        cli = LightningCLI(
            args=self.conf,
            model_class=L.LightningModule,
            datamodule_class=L.LightningDataModule,
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
        teardown_lightning_mlflow()


def _distributed_dataloader(dataloader: DataLoader, gwsize, grank):
    """Makes a Dataloader distributed."""
    sampler = DistributedSampler(
        dataloader.dataset,
        num_replicas=gwsize,
        rank=grank,
        shuffle=True
    )
    # Recreate dataloader, with updated sampler
    return DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        sampler=sampler,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=seed_worker,  # dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        generator=dataloader.generator,
        prefetch_factor=dataloader.prefetch_factor,
        persistent_workers=dataloader.persistent_workers,
        pin_memory_device=dataloader.pin_memory_device
    )


def distributed(func):
    """The decorated function must have a standard signature.
    Its first arguments must be:
    model, train_dataloader, validation_dataloader, device (in this order).

    Additional args or kwargs are allowed consistently with the signature
    of the decorated function.
    """
    def dist_train(
            model, train_dataloader, validation_dataloader=None, device='cpu',
            *args, **kwargs
    ):
        if torch.cuda.is_available():
            dist.init_process_group(backend='nccl')

        if torch.cuda.is_available():
            lwsize = torch.cuda.device_count()  # local world size - per node
            gwsize = dist.get_world_size()     # global world size - per run
            grank = dist.get_rank()            # global rank - assign per run
            lrank = dist.get_rank() % lwsize   # local rank - assign per node
        else:
            gwsize = 1
            grank = 0
            lrank = 0

        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu', lrank)
        if torch.cuda.is_available():
            torch.cuda.set_device(lrank)

        model = model.to(device)
        model = DDP(model, device_ids=[device], output_device=device)

        train_dataloader = _distributed_dataloader(
            train_dataloader, gwsize, grank)
        if validation_dataloader is not None:
            validation_dataloader = _distributed_dataloader(
                validation_dataloader, gwsize, grank)

        try:
            func(model, train_dataloader, validation_dataloader, device,
                 *args, **kwargs)
        finally:
            if torch.cuda.is_available():
                dist.barrier()
                dist.destroy_process_group()
    return dist_train
