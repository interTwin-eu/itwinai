import os
from copy import deepcopy
from pathlib import Path
from timeit import default_timer
from typing import Any, Dict, Literal, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
from hydra.utils import instantiate
from hython.models import ModelLogAPI
from hython.models import get_model_class as get_hython_model
from hython.sampler import SamplerBuilder
from hython.trainer import CalTrainer, RNNTrainer
from ray import tune
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from itwinai.constants import EPOCH_TIME_DIR
from itwinai.distributed import suppress_workers_print
from itwinai.loggers import EpochTimeTracker, Logger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.distributed import (
    DeepSpeedStrategy,
    HorovodStrategy,
    NonDistributedStrategy,
    TorchDDPStrategy,
)
from itwinai.torch.monitoring.monitoring import measure_gpu_utilization
from itwinai.torch.profiling.profiler import profile_torch_trainer
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
        model: Optional[Union[str, nn.Module]] = None,
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
        self.model_class = get_hython_model(model)
        self.model_class_name = model
        self.model_dict = {}

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
        self.config.loss_fn = instantiate({"loss_fn": self.config.loss_fn})["loss_fn"]

        self.config.metric_fn = instantiate({"metric_fn": self.config.metric_fn})["metric_fn"]

        self.model_api = ModelLogAPI(self.config)

        if self.config.hython_trainer == "rnntrainer":
            # LOAD MODEL
            self.model_logger = self.model_api.get_model_logger("model")
            self.model = self.model_class(self.config)

            self.hython_trainer = RNNTrainer(self.config)

        elif self.config.hython_trainer == "caltrainer":
            # LOAD MODEL HEAD/SURROGATE
            self.model_logger = self.model_api.get_model_logger("head")

            # TODO: to remove if condition, delegate logic to model api
            if self.model_logger == "mlflow":
                surrogate = self.model_api.load_model("head")
            else:
                # FIXME: There is a clash in "static_inputs" semantics between training and calibration
                # In the training the "static_inputs" are used to train the CudaLSTM model (main model - the surrogate -)
                # In the calibration the "static_inputs" are other input features that are used to train the TransferNN model.
                # Hence during calibration, when loading the weights of the surrogate,
                # I need to replace the CudaLSTM (now the head model) "static_inputs" with the correct "head_model_inputs"
                # in order to avoid clashes with the TransferNN model inputs
                # I think that if I used more modular config files, thanks to hydra, then I could import a surrogate_model.yaml
                # into both...
                config = deepcopy(self.config)
                config.static_inputs = config.head_model_inputs
                surrogate = get_hython_model(self.config.model_head)(config)

                surrogate = self.model_api.load_model("head", surrogate)

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
                "config_params": {"train_micro_batch_size_per_gpu": self.config.batch_size}
            }
        elif isinstance(self.strategy, TorchDDPStrategy):
            if "find_unused_parameters" not in self.config.model_fields:
                self.config.find_unused_parameters = False
            distribute_kwargs = {"find_unused_parameters": self.config.find_unused_parameters}

        self.model, self.optimizer, _ = self.strategy.distributed(
            model=self.model,
            optimizer=self.hython_trainer.optimizer,
            lr_scheduler=self.hython_trainer.lr_scheduler,
            **distribute_kwargs,
        )
        self.hython_trainer.optimizer = self.optimizer

    def set_epoch(self, epoch: int):
        if self.profiler is not None:
            self.profiler.step()

        if self.strategy.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
            self.val_loader.sampler.set_epoch(epoch)

    @profile_torch_trainer
    @measure_gpu_utilization
    def train(self):
        """Override train_val version of hython to support distributed strategy."""

        # Tracking epoch times for scaling test
        if self.strategy.is_main_worker and self.strategy.is_distributed:
            num_nodes = os.environ.get("SLURM_NNODES", "unk")
            epoch_time_output_dir = Path(f"scalability-metrics/{EPOCH_TIME_DIR}")
            epoch_time_file_name = f"epochtime_{self.strategy.name}_{num_nodes}N.csv"
            epoch_time_output_path = epoch_time_output_dir / epoch_time_file_name

            epoch_time_logger = EpochTimeTracker(
                strategy_name=self.strategy.name,
                save_path=epoch_time_output_path,
                num_nodes=num_nodes,
                should_log=self.measure_epoch_time,
            )

        device = self.strategy.device()
        loss_history = {"train": [], "val": []}
        metric_history = {f"train_{target}": [] for target in self.config.target_variables}
        metric_history.update({f"val_{target}": [] for target in self.config.target_variables})

        best_loss = float("inf")
        for epoch in tqdm(range(self.epochs)):
            epoch_start_time = default_timer()
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
                    for imetric, metric_value in target.items():
                        metric_key = imetric.lower().split("metric")[0]
                        new_metric_key = period + "_" + metric_key
                        metric_history_[new_metric_key] = [metric_value]

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
                # self.hython_trainer.save_weights(self.model)

            if self.strategy.is_distributed:
                epoch_time = default_timer() - epoch_start_time
                epoch_time_logger.add_epoch_time(epoch + 1, epoch_time)

        if self.strategy.is_main_worker:
            self.model.load_state_dict(best_model)

            # MODEL LOGGING
            model_log_names = self.model_api.get_model_log_names()
            for module_name, model_class_name in model_log_names.items():
                item = (
                    self.model
                    if module_name == "model"
                    else self.model.get_submodule(module_name)
                )

                if self.model_logger == "mlflow":
                    self.log(
                        item=item,
                        identifier=model_class_name,
                        kind="model",
                        registered_model_name=model_class_name,
                    )
                else:
                    self.model_api.log_model(module_name, item)

            # Report training metrics of last epoch to Ray
            tune.report({"loss": avg_val_loss.item(), "train_loss": train_loss.item()})

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

        batch_size = self.config.batch_size // self.strategy.global_world_size()
        self.train_loader = self.strategy.create_dataloader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=self.config.num_workers_dataloader,
            pin_memory=self.config.pin_gpu_memory,
            generator=self.torch_rng,
            sampler=train_sampler,
            drop_last=True,
        )

        if validation_dataset is not None:
            self.val_loader = self.strategy.create_dataloader(
                dataset=validation_dataset,
                batch_size=batch_size,
                num_workers=self.config.num_workers_dataloader,
                pin_memory=self.config.pin_gpu_memory,
                generator=self.torch_rng,
                sampler=val_sampler,
                drop_last=True,
            )
