# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Henry Mutegeki <henry.mutegeki@cern.ch> - CERN
# - Iacopo Ferrario <iacopofederico.ferrario@eurac.edu> - EURAC
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import os
from pathlib import Path
from timeit import default_timer
from typing import Any, Dict, Literal, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from hython.losses import RMSELoss
from hython.metrics import MSEMetric
from hython.sampler import SamplerBuilder
from hython.trainer import ConvTrainer, RNNTrainer, RNNTrainParams
from ray import train
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from itwinai.distributed import suppress_workers_print
from itwinai.loggers import EpochTimeTracker, Logger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.distributed import (
    DeepSpeedStrategy,
    HorovodStrategy,
    NonDistributedStrategy,
    TorchDDPStrategy,
)
from itwinai.torch.profiling.profiler import profile_torch_trainer
from itwinai.torch.monitoring.monitoring import measure_gpu_utilization
from itwinai.torch.trainer import TorchTrainer
from itwinai.torch.type import Metric


class RNNDistributedTrainer(TorchTrainer):
    """Trainer class for RNN model using pytorch.

    Args:
        config (Union[Dict, TrainingConfiguration]): training configuration
            containing hyperparameters.
        epochs (int): number of training epochs.
        model (Optional[nn.Module], optional): model to train.
            Defaults to None.
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
        model: Optional[nn.Module] = None,
        strategy: Optional[Literal["ddp", "deepspeed", "horovod"]] = "ddp",
        validation_every: Optional[int] = 1,
        test_every: Optional[int] = None,
        random_seed: Optional[int] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[Dict[str, Metric]] = None,
        checkpoints_location: str = "checkpoints",
        checkpoint_every: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            epochs=epochs,
            model=model,
            strategy=strategy,
            validation_every=validation_every,
            test_every=test_every,
            random_seed=random_seed,
            logger=logger,
            metrics=metrics,
            checkpoints_location=checkpoints_location,
            checkpoint_every=checkpoint_every,
            name=name,
            **kwargs,
        )
        self.save_parameters(**self.locals2params(locals()))

    @suppress_workers_print
    def execute(
        self,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ) -> Tuple[Dataset, Dataset, Dataset, Any]:
        return super().execute(train_dataset, validation_dataset, test_dataset)

    def create_model_loss_optimizer(self) -> None:
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.lr_reduction_factor,
            patience=self.config.lr_reduction_patience,
        )

        target_weights = {
            t: 1 / len(self.config.target_names) for t in self.config.target_names
        }
        self.loss_fn = RMSELoss(target_weight=target_weights)
        self.metric_fn = MSEMetric()

        distribute_kwargs = {}
        if isinstance(self.strategy, DeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = {
                "config_params": {"train_micro_batch_size_per_gpu": self.config.batch_size}
            }
        elif isinstance(self.strategy, TorchDDPStrategy):
            if "find_unused_parameters" not in self.config.model_fields:
                self.config.find_unused_parameters = False
            distribute_kwargs = {"find_unused_parameters": self.config.find_unused_parameters}

        self.model, self.optimizer, _ = self.strategy.distributed(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            **distribute_kwargs,
        )

    def set_epoch(self, epoch: int):
        if self.profiler is not None and epoch > 0:
            # We don't want to start stepping until after the first epoch
            self.profiler.step()

        if self.strategy.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
            self.val_loader.sampler.set_epoch(epoch)

    @profile_torch_trainer
    @measure_gpu_utilization
    def train(self):
        """Override version of hython to support distributed strategy."""
        # Tracking epoch times for scaling test
        if self.strategy.is_main_worker:
            num_nodes = int(os.environ.get("SLURM_NNODES", "unk"))
            epoch_time_output_dir = Path("scalability-metrics/epoch-time")
            epoch_time_file_name = f"epochtime_{self.strategy.name}_{num_nodes}N.csv"
            epoch_time_output_path = epoch_time_output_dir / epoch_time_file_name

            epoch_time_tracker = EpochTimeTracker(
                strategy_name=self.strategy.name,
                save_path=epoch_time_output_path,
                num_nodes=num_nodes
            )

        trainer = RNNTrainer(
            RNNTrainParams(
                experiment=self.config.experiment,
                temporal_subsampling=self.config.temporal_subsampling,
                temporal_subset=self.config.temporal_subset,
                seq_length=self.config.seq_length,
                target_names=self.config.target_names,
                metric_func=self.metric_fn,
                loss_func=self.loss_fn,
            )
        )

        device = self.strategy.device()
        loss_history = {"train": [], "val": []}
        metric_history = {f"train_{target}": [] for target in trainer.P.target_names}
        metric_history.update({f"val_{target}": [] for target in trainer.P.target_names})

        best_loss = float("inf")
        for epoch in tqdm(range(self.epochs)):
            epoch_start_time = default_timer()
            self.set_epoch(epoch)
            self.model.train()

            # set time indices for training
            # This has effect only if the trainer overload the
            # method (i.e. for RNN)
            trainer.temporal_index([self.train_loader, self.val_loader])

            train_loss, train_metric = trainer.epoch_step(
                self.model, self.train_loader, device, opt=self.optimizer
            )

            self.model.eval()
            with torch.no_grad():
                # set time indices for validation
                # This has effect only if the trainer overload the method
                # (i.e. for RNN)
                trainer.temporal_index([self.train_loader, self.val_loader])

                val_loss, val_metric = trainer.epoch_step(
                    self.model, self.val_loader, device, opt=None
                )

            # gather losses from each worker and place them on the main worker.
            worker_val_losses = self.strategy.gather(val_loss, dst_rank=0)

            if not self.strategy.is_main_worker:
                continue

            # Moving them all to the cpu() before performing calculations
            avg_val_loss = torch.mean(torch.stack(worker_val_losses)).detach().cpu()
            self.lr_scheduler.step(avg_val_loss)
            loss_history["train"].append(train_loss)
            loss_history["val"].append(avg_val_loss)

            self.log(
                item=train_loss.item(),
                identifier="train_loss_per_epoch",
                kind="metric",
                step=epoch,
            )
            self.log(
                item=avg_val_loss.item(),
                identifier="val_loss_per_epoch",
                kind="metric",
                step=epoch,
            )

            for target in trainer.P.target_names:
                metric_history[f"train_{target}"].append(train_metric[target])
                metric_history[f"val_{target}"].append(val_metric[target])

            # Aggregate and log metrics
            avg_metrics = pd.DataFrame(metric_history).mean().to_dict()
            for m_name, m_val in avg_metrics.items():
                self.log(
                    item=m_val,
                    identifier=m_name + "_epoch",
                    kind="metric",
                    step=epoch,
                )

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model = self.model.state_dict()

            epoch_time = default_timer() - epoch_start_time
            epoch_time_tracker.add_epoch_time(epoch + 1, epoch_time)

        if self.strategy.is_main_worker:
            epoch_time_tracker.save()
            self.model.load_state_dict(best_model)
            self.log(item=self.model, identifier="LSTM", kind="model")

            # Report training metrics of last epoch to Ray
            train.report({"loss": avg_val_loss.item(), "train_loss": train_loss.item()})

        return loss_history, metric_history

    def create_dataloaders(self, train_dataset, validation_dataset, test_dataset):
        sampling_kwargs = {}
        if isinstance(self.strategy, HorovodStrategy):
            sampling_kwargs["num_replicas"] = self.strategy.global_world_size()
            sampling_kwargs["rank"] = self.strategy.global_rank()

        if isinstance(self.strategy, NonDistributedStrategy):
            processing = "single-gpu"
        else:
            processing = "multi-gpu"

        train_sampler_builder = SamplerBuilder(
            train_dataset,
            sampling="random",
            processing=processing,
            sampling_kwargs=sampling_kwargs,
        )

        val_sampler_builder = SamplerBuilder(
            validation_dataset,
            sampling="sequential",
            processing=processing,
            sampling_kwargs=sampling_kwargs,
        )

        train_sampler = train_sampler_builder.get_sampler()
        val_sampler = val_sampler_builder.get_sampler()

        self.train_loader = self.strategy.create_dataloader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers_dataloader,
            pin_memory=self.config.pin_gpu_memory,
            generator=self.torch_rng,
            sampler=train_sampler,
            drop_last=True,
        )

        if validation_dataset is not None:
            self.val_loader = self.strategy.create_dataloader(
                dataset=validation_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers_dataloader,
                pin_memory=self.config.pin_gpu_memory,
                generator=self.torch_rng,
                sampler=val_sampler,
                drop_last=True,
            )


class ConvRNNDistributedTrainer(TorchTrainer):
    """Trainer class for ConvRNN model using pytorch.

    Args:
        config (Union[Dict, TrainingConfiguration]): training configuration
            containing hyperparameters.
        epochs (int): number of training epochs.
        model (Optional[nn.Module], optional): model to train.
            Defaults to None.
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
        model: Optional[nn.Module] = None,
        strategy: Literal["ddp", "deepspeed", "horovod"] = "ddp",
        validation_every: Optional[int] = 1,
        test_every: Optional[int] = None,
        random_seed: Optional[int] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[Dict[str, Metric]] = None,
        checkpoints_location: str = "checkpoints",
        checkpoint_every: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            epochs=epochs,
            model=model,
            strategy=strategy,
            validation_every=validation_every,
            test_every=test_every,
            random_seed=random_seed,
            logger=logger,
            metrics=metrics,
            checkpoints_location=checkpoints_location,
            checkpoint_every=checkpoint_every,
            name=name,
            **kwargs,
        )
        self.save_parameters(**self.locals2params(locals()))

    def create_model_loss_optimizer(self) -> None:
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.lr_reduction_factor,
            patience=self.config.lr_reduction_patience,
        )

        target_weights = {
            t: 1 / len(self.config.target_names) for t in self.config.target_names
        }
        self.loss_fn = RMSELoss(target_weight=target_weights)
        self.metric_fn = MSEMetric()

        if isinstance(self.strategy, DeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = dict(
                config_params=dict(train_micro_batch_size_per_gpu=self.config.batch_size)
            )
        else:
            distribute_kwargs = {}  # dict(find_unused_parameters=True)

        self.model, self.optimizer, _ = self.strategy.distributed(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            **distribute_kwargs,
        )

    def train(self):
        """Override version of hython to support distributed strategy."""
        trainer = ConvTrainer(
            RNNTrainParams(
                experiment=self.config.experiment,
                temporal_subsampling=False,
                temporal_subset=1,
                target_names=self.config.target_names,
                metric_func=self.metric_fn,
                loss_func=self.loss_fn,
            )
        )

        device = self.strategy.device()
        loss_history = {"train": [], "val": []}
        metric_history = {f"train_{target}": [] for target in trainer.P.target_names}
        metric_history.update({f"val_{target}": [] for target in trainer.P.target_names})

        best_loss = float("inf")
        for epoch in tqdm(range(self.epochs)):
            if self.strategy.is_distributed:
                # *Added for distributed*
                self.train_loader.sampler.set_epoch(epoch)
                self.val_loader.sampler.set_epoch(epoch)

            self.model.train()

            # set time indices for training
            # This has effect only if the trainer overload the
            # method (i.e. for RNN)
            trainer.temporal_index([self.train_loader, self.val_loader])

            train_loss, train_metric = trainer.epoch_step(
                self.model, self.train_loader, device, opt=self.optimizer
            )

            self.model.eval()
            with torch.no_grad():
                # set time indices for validation
                # This has effect only if the trainer overload the method
                # (i.e. for RNN)
                trainer.temporal_index([self.train_loader, self.val_loader])

                val_loss, val_metric = trainer.epoch_step(
                    self.model, self.val_loader, device, opt=None
                )

            # gather losses from each worker and place them on the main worker.
            worker_val_losses = self.strategy.gather(val_loss, dst_rank=0)
            if not self.strategy.global_rank() == 0:
                continue

            avg_val_loss = torch.mean(torch.stack(worker_val_losses)).detach().cpu()
            self.lr_scheduler.step(avg_val_loss)
            loss_history["train"].append(train_loss)
            loss_history["val"].append(avg_val_loss)
            self.log(
                item=train_loss.item(),
                identifier="train_loss_per_epoch",
                kind="metric",
                step=epoch,
            )
            self.log(
                item=avg_val_loss.item(),
                identifier="val_loss_per_epoch",
                kind="metric",
                step=epoch,
            )

            for target in trainer.P.target_names:
                metric_history[f"train_{target}"].append(train_metric[target])
                metric_history[f"val_{target}"].append(val_metric[target])
            # Aggregate and log metrics
            avg_metrics = pd.DataFrame(metric_history).mean().to_dict()
            for m_name, m_val in avg_metrics.items():
                self.log(
                    item=m_val,
                    identifier=m_name + "_epoch",
                    kind="metric",
                    step=epoch,
                )

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                # self.model.load_state_dict(best_model_weights)

            # Report training metrics of last epoch to Ray
            train.report({"loss": avg_val_loss.item(), "train_loss": train_loss.item()})

        return loss_history, metric_history

    def create_dataloaders(self, train_dataset, validation_dataset, test_dataset):
        train_sampler_builder = SamplerBuilder(
            train_dataset,
            sampling="random",
            processing=("multi-gpu" if self.config.distributed else "single-gpu"),
        )

        val_sampler_builder = SamplerBuilder(
            validation_dataset,
            sampling="sequential",
            processing=("multi-gpu" if self.config.distributed else "single-gpu"),
        )

        train_sampler = train_sampler_builder.get_sampler()
        val_sampler = val_sampler_builder.get_sampler()

        self.train_loader = self.strategy.create_dataloader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers_dataloader,
            pin_memory=self.config.pin_gpu_memory,
            generator=self.torch_rng,
            sampler=train_sampler,
        )

        if validation_dataset is not None:
            self.val_loader = self.strategy.create_dataloader(
                dataset=validation_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers_dataloader,
                pin_memory=self.config.pin_gpu_memory,
                generator=self.torch_rng,
                sampler=val_sampler,
            )
