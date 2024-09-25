import os
import pandas as pd
from typing import Dict, Literal, Optional, Union
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
from hython.losses import RMSELoss
from hython.metrics import MSEMetric
from hython.sampler import SamplerBuilder
from hython.trainer import HythonTrainer, RNNTrainer, RNNTrainParams
from itwinai.loggers import Logger, EpochTimeTracker
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.distributed import DeepSpeedStrategy, HorovodStrategy, TorchDDPStrategy
from itwinai.torch.trainer import TorchTrainer
from itwinai.torch.type import Metric
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm


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
            strategy: Literal["ddp", "deepspeed", "horovod"] = 'ddp',
            validation_every: Optional[int] = 1,
            test_every: Optional[int] = None,
            random_seed: Optional[int] = None,
            logger: Optional[Logger] = None,
            metrics: Optional[Dict[str, Metric]] = None,
            checkpoints_location: str = "checkpoints",
            checkpoint_every: Optional[int] = None,
            name: Optional[str] = None, **kwargs) -> None:
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
            **kwargs)
        self.save_parameters(**self.locals2params(locals()))

    def create_model_loss_optimizer(self) -> None:
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.lr
        )
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10)
        
        TARGET_WEIGHTS = {t: 1/len(self.config.target_names) for t in self.config.target_names}
        self.loss_fn = RMSELoss(target_weight=TARGET_WEIGHTS)
        self.metric_fn = MSEMetric()

        if isinstance(self.strategy, DeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = dict(
                config_params=dict(
                    train_micro_batch_size_per_gpu=self.config.batch_size
                )
            )
        elif isinstance(self.strategy, TorchDDPStrategy): 
            if 'find_unused_parameters' not in self.config.model_fields:
                self.config.find_unused_parameters = False
                distribute_kwargs = dict(
                    find_unused_parameters=self.config.find_unused_parameters
                )
        else:
            distribute_kwargs = {}

        
        # Distribute discriminator and its optimizer
        self.model, self.optimizer, _ = self.strategy.distributed(
            model=self.model, 
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler, 
            **distribute_kwargs
        )

    def train(self):
        """Override version of hython to support distributed strategy."""
        # Tracking epoch times for scaling test
        if self.strategy.is_main_worker: 
            num_nodes = os.environ.get("SLURM_NNODES", "unk")
            series_name = os.environ.get("DIST_MODE", "unk") + "-torch"
            file_name = f"epochtime_{series_name}_{num_nodes}N.csv"
            epoch_time_tracker = EpochTimeTracker(
                    series_name=series_name,
                    csv_file=file_name
            )
        trainer = RNNTrainer(
                RNNTrainParams(
                    experiment=self.config.experiment,
                    temporal_subsampling=self.config.temporal_subsampling,
                    temporal_subset=self.config.temporal_subset,
                    seq_length=self.config.seq_length,
                    target_names=self.config.target_names,
                    metric_func=self.metric_fn,
                    loss_func=self.loss_fn
                )
        )

        device = self.strategy.device()
        loss_history = {"train": [], "val": []}
        metric_history = {f"train_{target}": []
                          for target in trainer.P.target_names}
        metric_history.update({f"val_{target}": []
                              for target in trainer.P.target_names})

        best_loss = float("inf")
        for epoch in tqdm(range(self.epochs)):
            epoch_start_time = timer()
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


            if self.strategy.global_rank() != 0:
                # Logging time for scaling tests
                if self.strategy.is_main_worker: 
                    epoch_end_time = timer()
                    epoch_time_tracker.add_epoch_time(
                            epoch-1, 
                            epoch_end_time - epoch_start_time
                    ) 
                continue


            # Moving them all to the cpu() before performing calculations
            worker_val_losses = [wvl.cpu() for wvl in worker_val_losses]
            avg_val_loss = torch.mean(torch.stack(worker_val_losses)).detach().cpu()
            self.lr_scheduler.step(avg_val_loss)
            loss_history["train"].append(train_loss)
            loss_history["val"].append(avg_val_loss)
            self.log(
                item=train_loss.item(),
                identifier='train_loss_per_epoch',
                kind='metric',
                step=epoch,
            )
            self.log(
                item=avg_val_loss.item(),
                identifier='val_loss_per_epoch',
                kind='metric',
                step=epoch,
            )

            for target in trainer.P.target_names:
                metric_history[f"train_{target}"].append( train_metric[target])
                metric_history[f"val_{target}"].append( val_metric[target])
            # Aggregate and log metrics
            avg_metrics = pd.DataFrame(metric_history).mean().to_dict()
            for m_name, m_val in avg_metrics.items():
                self.log(
                    item=m_val,
                    identifier=m_name + '_epoch',
                    kind='metric',
                    step=epoch,
                )

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                print(f"train loss: {train_loss}")
                print(f"val loss: {avg_val_loss}")

            if self.strategy.is_main_worker: 
                epoch_end_time = timer()
                epoch_time_tracker.add_epoch_time(
                        epoch-1, 
                        epoch_end_time - epoch_start_time
                ) 

        return loss_history, metric_history

    def create_dataloaders(
            self, train_dataset, validation_dataset, test_dataset
        ):
        sampling_kwargs = {}
        if isinstance(self.strategy, HorovodStrategy): 
            sampling_kwargs["num_replicas"] = self.strategy.global_world_size()
            sampling_kwargs["rank"] = self.strategy.global_rank()

        train_sampler_builder = SamplerBuilder(
            train_dataset,
            sampling="random",
            processing="multi-gpu" if self.config.distributed else "single-gpu",
            sampling_kwargs=sampling_kwargs
        )  

        val_sampler_builder = SamplerBuilder(
            validation_dataset,
            sampling="sequential",
            processing="multi-gpu" if self.config.distributed else "single-gpu",
            sampling_kwargs=sampling_kwargs
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
            drop_last=True
        )

        if validation_dataset is not None:
            self.val_loader = self.strategy.create_dataloader(
                dataset=validation_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers_dataloader,
                pin_memory=self.config.pin_gpu_memory,
                generator=self.torch_rng,
                sampler=val_sampler,
               drop_last=True
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
            strategy: Literal["ddp", "deepspeed", "horovod"] = 'ddp',
            validation_every: Optional[int] = 1,
            test_every: Optional[int] = None,
            random_seed: Optional[int] = None,
            logger: Optional[Logger] = None,
            metrics: Optional[Dict[str, Metric]] = None,
            checkpoints_location: str = "checkpoints",
            checkpoint_every: Optional[int] = None,
            name: Optional[str] = None, **kwargs) -> None:
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
            **kwargs)
        self.save_parameters(**self.locals2params(locals()))

    def create_model_loss_optimizer(self) -> None:
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.lr
        )
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10)
        
        TARGET_WEIGHTS = {t: 1/len(self.config.rnn_config["target_names"]) for t in self.config.rnn_config["target_names"]}
        self.loss_fn = RMSELoss(target_weight=TARGET_WEIGHTS)
        self.metric_fn = MSEMetric()

        if isinstance(self.strategy, DeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = dict(
                config_params=dict(
                    train_micro_batch_size_per_gpu=self.config.batch_size
                )
            )
        else:
            distribute_kwargs = {} # dict(find_unused_parameters=True)
        # Distribute discriminator and its optimizer
        self.model, self.optimizer, _ = self.strategy.distributed(
            model=self.model, optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler, **distribute_kwargs)

    def train(self):
        """Override version of hython to support distributed strategy."""
        trainer = HythonTrainer(
                RNNTrainParams(
                    experiment=self.config.experiment,
                    temporal_subsampling=False,
                    temporal_subset=1,
                    target_names=self.config.rnn_config["target_names"],
                    metric_func=self.metric_fn,
                    loss_func=self.loss_fn)
        )

        device = self.strategy.device()
        loss_history = {"train": [], "val": []}
        metric_history = {f"train_{target}": []
                          for target in trainer.P.target_names}
        metric_history.update({f"val_{target}": []
                              for target in trainer.P.target_names})

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
            if self.strategy.global_rank() == 0:
                avg_val_loss = torch.mean(torch.stack(
                    worker_val_losses)).detach().cpu()
                self.lr_scheduler.step(avg_val_loss)
                loss_history["train"].append(train_loss)
                loss_history["val"].append(avg_val_loss)
                self.log(
                    item=train_loss.item(),
                    identifier='train_loss_per_epoch',
                    kind='metric',
                    step=epoch,
                )
                self.log(
                    item=avg_val_loss.item(),
                    identifier='val_loss_per_epoch',
                    kind='metric',
                    step=epoch,
                )

                for target in trainer.P.target_names:
                    metric_history[f"train_{target}"].append(
                        train_metric[target])
                    metric_history[f"val_{target}"].append(
                        val_metric[target])
                # Aggregate and log metrics
                avg_metrics = pd.DataFrame(metric_history).mean().to_dict()
                for m_name, m_val in avg_metrics.items():
                    self.log(
                        item=m_val,
                        identifier=m_name + '_epoch',
                        kind='metric',
                        step=epoch,
                    )

                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    # The code `best_model_weights` appears to be a variable
                    # name in Python. It is not assigned any value
                    # or operation in the provided snippet, so it is
                    # difficult to determine its specific purpose
                    # without additional context. It could potentially be
                    # used to store the weights of a machine learning model
                    # or any other relevant data  related to a model.
                    # best_model_weights = copy.deepcopy(self.model.state_dict())
                    #trainer.save_weights(self.model, self.config.dp_weights)
                    # print("Copied best model weights!")

                    print(f"train loss: {train_loss}")
                    print(f"val loss: {avg_val_loss}")

                #self.model.load_state_dict(best_model_weights)

        return loss_history, metric_history

    def create_dataloaders(
            self, train_dataset, validation_dataset, test_dataset):
        train_sampler_builder = SamplerBuilder(
            train_dataset,
            sampling="random",
            processing="multi-gpu" if self.config.rnn_config["distributed"] else "single-gpu") # 

        val_sampler_builder = SamplerBuilder(
            validation_dataset,
            sampling="sequential",
            processing="multi-gpu" if self.config.rnn_config["distributed"] else "single-gpu")

        train_sampler = train_sampler_builder.get_sampler()
        val_sampler = val_sampler_builder.get_sampler()

        self.train_loader = self.strategy.create_dataloader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers_dataloader,
            pin_memory=self.config.pin_gpu_memory,
            generator=self.torch_rng,
            sampler=train_sampler
        )

        if validation_dataset is not None:
            self.val_loader = self.strategy.create_dataloader(
                dataset=validation_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers_dataloader,
                pin_memory=self.config.pin_gpu_memory,
                generator=self.torch_rng,
                sampler=val_sampler
            )

