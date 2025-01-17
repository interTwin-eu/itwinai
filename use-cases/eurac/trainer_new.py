import os
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, Literal, Optional, Union, Any, Tuple
from tqdm.auto import tqdm
import copy
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import pandas as pd


from hython.sampler import SamplerBuilder
from hython.trainer import RNNTrainer, CalTrainer
from hython.models import get_model as get_hython_model

from itwinai.torch.distributed import (
    DeepSpeedStrategy,
    HorovodStrategy,
    NonDistributedStrategy,
    TorchDDPStrategy,
)
from itwinai.distributed import suppress_workers_print
from itwinai.loggers import EpochTimeTracker, Logger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.trainer import TorchTrainer
from itwinai.torch.type import Metric
from itwinai.torch.profiling.profiler import profile_torch_trainer

from omegaconf import OmegaConf
from hydra.utils import instantiate


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
        model: Optional[Union[str, nn.Module]] = None,
        strategy: Optional[
            Literal["ddp", "deepspeed", "horovod", "sequential"]
        ] = "ddp",
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

        self.model_class = get_hython_model(self.model)

    @suppress_workers_print
    # @profile_torch_trainer
    def execute(
        self,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ) -> Tuple[Dataset, Dataset, Dataset, Any]:
        self.init_hython_trainer()

        return super().execute(train_dataset, validation_dataset, test_dataset)

    def init_hython_trainer(self) -> None:
        self.config.loss_fn = instantiate(
            OmegaConf.create({"loss_fn": self.config.loss_fn})
        )["loss_fn"]

        self.config.metric_fn = instantiate(
            OmegaConf.create({"metric_fn": self.config.metric_fn})
        )["metric_fn"]

        if self.config.hython_trainer == "rnntrainer":
            self.model = self.model_class(
                hidden_size=self.config.hidden_size,
                dynamic_input_size=len(self.config.dynamic_inputs),
                static_input_size=len(self.config.static_inputs),
                output_size=len(self.config.target_variables),
                dropout=self.config.dropout,
            )
            self.hython_trainer = RNNTrainer(self.config)

        elif self.config.hython_trainer == "caltrainer":
            surrogate = get_hython_model(self.config.model_head)(
                hidden_size=self.config.model_head_hidden_size,
                dynamic_input_size=len(self.config.dynamic_inputs),
                static_input_size=len(self.config.head_model_inputs),
                output_size=len(self.config.target_variables),
                dropout=self.config.model_head_dropout,
            )

            surrogate.load_state_dict(
                torch.load(
                    f"{self.config.work_dir}/{self.config.model_head_dir}/{self.config.model_head_file}"
                )
            )

            transfer_nn = get_hython_model(self.config.model_transfer)(
                self.config.head_model_inputs,
                len(self.config.static_inputs),
                self.config.mt_output_dim,
                self.config.mt_hidden_dim,
                self.config.mt_n_layers,
            )

            self.model = self.model_class(
                transfernn=transfer_nn,
                head_layer=surrogate,
                freeze_head=self.config.freeze_head,
                scale_head_input_parameter=self.config.scale_head_input_parameter,
            )

            self.hython_trainer = CalTrainer(self.config)

        self.hython_trainer.init_trainer(self.model)

    def create_model_loss_optimizer(self) -> None:
        distribute_kwargs = {}
        if isinstance(self.strategy, DeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = {
                "config_params": {
                    "train_micro_batch_size_per_gpu": self.config.batch_size
                }
            }
        elif isinstance(self.strategy, TorchDDPStrategy):
            if "find_unused_parameters" not in self.config.model_fields:
                self.config.find_unused_parameters = False
            distribute_kwargs = {
                "find_unused_parameters": self.config.find_unused_parameters
            }

        self.model, self.optimizer, _ = self.strategy.distributed(
            model=self.model,
            optimizer=self.hython_trainer.optimizer,
            lr_scheduler=self.hython_trainer.lr_scheduler,
            **distribute_kwargs,
        )

    def set_epoch(self, epoch: int):
        if self.profiler is not None:
            self.profiler.step()

        if self.strategy.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
            self.val_loader.sampler.set_epoch(epoch)

    def train(self):
        """Override train_val version of hython to support distributed strategy."""

        # Tracking epoch times for scaling test
        if self.strategy.is_main_worker and self.strategy.is_distributed:
            num_nodes = os.environ.get("SLURM_NNODES", "unk")
            series_name = os.environ.get("DIST_MODE", "unk") + "-torch"
            file_name = f"epochtime_{series_name}_{num_nodes}N.csv"
            file_path = Path("logs_epoch") / file_name
            epoch_time_tracker = EpochTimeTracker(
                series_name=series_name, csv_file=file_path
            )

        device = self.strategy.device()
        loss_history = {"train": [], "val": []}
        metric_history = {
            f"train_{target}": [] for target in self.config.target_variables
        }
        metric_history.update(
            {f"val_{target}": [] for target in self.config.target_variables}
        )

        best_loss = float("inf")
        for epoch in tqdm(range(self.epochs)):
            epoch_start_time = timer()
            self.set_epoch(epoch)

            # run train_valid epoch step of hython trainer
            (
                train_loss,
                train_metric,
                val_loss,
                val_metric,
            ) = self.hython_trainer.train_valid_epoch(
                self.model, self.train_loader, self.val_loader, device
            )

            # gather losses from each worker and place them on the main worker.
            worker_val_losses = self.strategy.gather(val_loss, dst_rank=0)

            if not self.strategy.is_main_worker:
                continue

            # Moving them all to the cpu() before performing calculations
            avg_val_loss = torch.mean(torch.stack(worker_val_losses)).detach().cpu()
            self.hython_trainer.lr_scheduler.step(avg_val_loss)
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

            for target in self.config.target_variables:
                metric_history[f"train_{target}"].append(train_metric[target])
                metric_history[f"val_{target}"].append(val_metric[target])

            # Aggregate and log metrics
            metric_history_ = {}
            for period in metric_history:
                for target in metric_history[period]:
                    for key, value in target.items():
                        l = []
                        k = key.lower().split("metric")[0]
                        n = period + "_" + k
                        l.append(value)
                        metric_history_[n] = l

            avg_metrics = pd.DataFrame(metric_history_).mean().to_dict()
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

            epoch_end_time = timer()
            if self.strategy.is_distributed:
                epoch_time_tracker.add_epoch_time(
                    epoch - 1, epoch_end_time - epoch_start_time
                )

        if self.strategy.is_main_worker:
            self.model.load_state_dict(best_model)
            self.log(item=self.model, identifier="LSTM", kind="model")

            # Report training metrics of last epoch to Ray
            try:
                from ray import train

                train.report(
                    {"loss": avg_val_loss.item(), "train_loss": train_loss.item()}
                )
            except:
                pass

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
