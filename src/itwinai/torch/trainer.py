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

import lightning as L
from lightning.pytorch.cli import LightningCLI

import horovod.torch as hvd

from ..components import Trainer, monitor_exec
from .types import (
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


class Config:
    def __init__(self, my_dict: Optional[Dict] = None):
        my_dict = my_dict if my_dict is not None else {}
        self.__dict__.update(my_dict)


class TorchTrainer(Trainer, LogMixin):
    """Trainer class for torch training algorithms.

    Args:
        config (Dict): training configuration containing hyperparameters.
        epochs (int): number of training epochs.
        model (Optional[nn.Module], optional): model to train.
        Defaults to None.
        strategy (Literal[&quot;ddp&quot;, &quot;deepspeed&quot;,
        &quot;horovod&quot;], optional): distributed strategy.
        Defaults to 'ddp'.
        validation_every (Optional[int], optional): run a validation epoch
        every ``validation_every`` epochs. Disabled if None. Defaults to 1.
        test_every (Optional[int], optional): run a test epoch
        every ``test_every`` epochs. Disabled if None. Defaults to None.
        random_seed (Optional[int], optional): set random seed for
        reproducibility. If None, the seed is not set. Defaults to None.
        logger (Optional[Logger], optional): logger for ML tracking.
        Defaults to None.
        log_all_workers (bool, optional): if True, the ``log`` method is
        called on all workers in the distributed context. Defaults to False.
        metrics (Optional[Dict[str, Metric]], optional): map of torchmetrics
        metrics. Defaults to None.
        name (Optional[str], optional): trainer custom name. Defaults to None.
    """
    # TODO:
    #   - add checkpointing.
    #   - extract BaseTorchTrainer and extend it creating a set of trainer
    #     templates (e.g.. GAN, Classifier, Transformer) allowing scientists
    #     to reuse ML algos.
    #   - improve get from configuration object

    _strategy: TorchDistributedStrategy = None

    train_dataloader: DataLoader = None
    validation_dataloader: DataLoader = None
    test_dataloader: DataLoader = None

    model: nn.Module = None
    loss: Loss = None
    optimizer: Optimizer = None
    lr_scheduler: LrScheduler = None

    torch_rng: torch.Generator = None
    logger: Logger = None
    train_glob_step: int = 0
    validation_glob_step: int = 0
    test_glob_step: int = 0
    metrics: Dict[str, Metric]

    def __init__(
        self,
        config: Dict,
        epochs: int,
        model: Optional[nn.Module] = None,
        strategy: Literal["ddp", "deepspeed", "horovod"] = 'ddp',
        validation_every: Optional[int] = 1,
        test_every: Optional[int] = None,
        random_seed: Optional[int] = None,
        logger: Optional[Logger] = None,
        log_all_workers: bool = False,
        metrics: Optional[Dict[str, Metric]] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name)
        self.save_parameters(**self.locals2params(locals()))

        # config is mean to store all hyperparameters, which can very from use
        # case to use case
        # and include learning_rate, batch_size....
        self.config = Config(config)
        self.epochs = epochs
        self.model = model
        self.strategy = strategy
        self.validation_every = validation_every
        self.test_every = test_every
        self.random_seed = random_seed
        self.logger = logger
        self.log_all_workers = log_all_workers
        self.metrics = metrics if metrics is not None else {}

    @property
    def strategy(self) -> TorchDistributedStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Union[str, TorchDistributedStrategy]) -> None:
        if isinstance(strategy, TorchDistributedStrategy):
            self._strategy = strategy
        else:
            self._strategy = self._detect_strategy(strategy)

    @property
    def device(self) -> str:
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
            # Model was not passed to the constructor.
            # Create a model here
            raise ValueError(
                "self.model is None! Either pass it to the constructor or "
                "override this method."
            )

        # A simple NLLLoss
        self.loss = nn.functional.nll_loss

        # TODO: improve robustness of getting from config
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum
        )
        # Create self.lr_scheduler if needed

        # IMPORTANT: model, optimizer, and scheduler need to be distributed

        # First, define strategy-wise optional configurations
        # TODO: improve robustness of getting from config
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

        # TODO: improve robustness of getting from config
        self.train_dataloader = self.strategy.create_dataloader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            generator=self.torch_rng
        )
        if validation_dataset is not None:
            self.validation_dataloader = self.strategy.create_dataloader(
                dataset=train_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                generator=self.torch_rng
            )
        if test_dataset is not None:
            self.test_dataloader = self.strategy.create_dataloader(
                dataset=train_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                generator=self.torch_rng
            )

    def _setup_metrics(self):
        """Move metrics to current device."""
        for m_name, metric in self.metrics.items():
            self.metrics[m_name] = metric.to(self.device)

    @monitor_exec
    def execute(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        test_dataset: Dataset
    ) -> Tuple[Dataset, Dataset, Dataset, Any]:
        """Prepares distributed environment and data structures
        for the actual training.

        Args:
            train_dataset (Dataset): training dataset.
            validation_dataset (Dataset): validation dataset.
            test_dataset (Dataset): test dataset.

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

        if self.strategy.is_main_worker:
            self.logger.create_logger_context()

        self.train()

        if self.strategy.is_main_worker:
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

    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = 'metric',
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs
    ) -> None:
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
        # start_time = time.perf_counter()
        for epoch in range(self.epochs):
            epoch_n = epoch + 1
            self._set_epoch_dataloaders(epoch)
            self.train_epoch()
            if self.validation_every and self.validation_every % epoch_n == 0:
                self.validation_epoch()
            if self.test_every and self.test_every % epoch_n == 0:
                self.test_epoch()

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
                identifier=f'{m_name}_{stage}',
                kind='metric',
                step=logger_step,
                batch_idx=batch_idx
            )
            m_values[m_name] = m_val
        return m_values

    def training_step(
        self,
        batch: Batch,
        batch_idx: int
    ) -> Tuple[Loss, Dict[str, Any]]:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        pred_y = self.model(x)
        loss: Loss = self.loss(pred_y, y)
        self.log(
            item=loss.item(),
            identifier='training_loss',
            kind='metric',
            step=self.train_glob_step,
            batch_idx=batch_idx
        )
        metrics: Dict[str, Any] = self.compute_metrics(
            true=y,
            pred=pred_y,
            logger_step=self.train_glob_step,
            batch_idx=batch_idx,
            stage='training'
        )
        return loss, metrics

    def validation_step(
        self,
        batch: Batch,
        batch_idx: int
    ) -> Tuple[Loss, Dict[str, Any]]:
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

    def train_epoch(self) -> Loss:
        self.model.train()
        train_losses = []
        for batch_idx, train_batch in enumerate(self.train_dataloader):
            loss, metrics = self.training_step(
                batch=train_batch,
                batch_idx=batch_idx
            )
            # TODO: merge and log batch metrics and loss into epoch metrics
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss)
            # Important: update counter
            self.train_glob_step += 1

        # Aggregate and log losses
        avg_loss = torch.mean(torch.stack(train_losses)).detach().cpu()
        self.log(
            item=avg_loss.item(),
            identifier='training_loss_epoch',
            kind='metric',
            step=self.train_glob_step,
        )
        return avg_loss

    def validation_epoch(self) -> Loss:
        if self.validation_dataloader is not None:
            self.model.eval()
            validation_losses = []
            for batch_idx, val_batch \
                    in enumerate(self.validation_dataloader):
                # TODO: merge and log batch metrics and loss into epoch metrics
                loss, metrics = self.validation_step(
                    batch=val_batch,
                    batch_idx=batch_idx
                )
                validation_losses.append(loss)
                # Important: update counter
                self.validation_glob_step += 1

            # Aggregate and log losses
            avg_loss = torch.mean(
                torch.stack(validation_losses)
            ).detach().cpu()
            self.log(
                item=avg_loss.item(),
                identifier='validation_loss_epoch',
                kind='metric',
                step=self.validation_glob_step,
            )
            return avg_loss

    def test_epoch(self):
        # TODO: implement test epoch
        raise NotImplementedError()


class TorchLightningTrainer(Trainer):
    """Generic trainer for torch Lightning workflows.

        Args:
            config (Union[Dict, str]): (path to a) Lightning configuration
            https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html
            mlflow_saved_model (str, optional): name of the model created in
            MLFlow. Defaults to 'my_model'.
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


def preproc_dataloader(dataloader: DataLoader, gwsize, grank):
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

        train_dataloader = preproc_dataloader(train_dataloader, gwsize, grank)
        if validation_dataloader is not None:
            validation_dataloader = preproc_dataloader(
                validation_dataloader, gwsize, grank)

        try:
            func(model, train_dataloader, validation_dataloader, device,
                 *args, **kwargs)
        finally:
            if torch.cuda.is_available():
                dist.barrier()
                dist.destroy_process_group()
    return dist_train
