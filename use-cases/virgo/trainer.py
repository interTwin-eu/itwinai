# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------


import os
import time
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from ray import tune
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm

from itwinai.distributed import suppress_workers_print
from itwinai.loggers import EpochTimeTracker, Logger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.distributed import DeepSpeedStrategy, RayDDPStrategy, RayDeepSpeedStrategy
from itwinai.torch.monitoring.monitoring import measure_gpu_utilization
from itwinai.torch.profiling.profiler import profile_torch_trainer
from itwinai.torch.trainer import RayTorchTrainer, TorchTrainer
from itwinai.constants import EPOCH_TIME_DIR
from src.model import Decoder, Decoder_2d_deep, GeneratorResNet, UNet
from src.utils import init_weights


class VirgoTrainingConfiguration(TrainingConfiguration):
    """Virgo TrainingConfiguration"""

    #: Whether to save best model on validation dataset. Defaults to True.
    save_best: bool = True
    #: Loss function. Defaults to "l1".
    loss: Literal["l1", "l2"] = "l1"
    #: Generator to train. Defaults to "unet".
    generator: Literal["simple", "deep", "resnet", "unet"] = "unet"


class NoiseGeneratorTrainer(TorchTrainer):
    def __init__(
        self,
        num_epochs: int = 2,
        config: Dict | TrainingConfiguration | None = None,
        strategy: Literal["ddp", "deepspeed", "horovod"] | None = "ddp",
        checkpoint_path: str = "checkpoints/epoch_{}.pth",
        logger: Logger | None = None,
        random_seed: int | None = None,
        name: str | None = None,
        validation_every: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            epochs=num_epochs,
            config=config,
            strategy=strategy,
            logger=logger,
            random_seed=random_seed,
            name=name,
            validation_every=validation_every,
            **kwargs,
        )
        self.save_parameters(**self.locals2params(locals()))
        # Global training configuration

        if isinstance(config, dict):
            config = VirgoTrainingConfiguration(**config)
        self.config = config
        self.num_epochs = num_epochs
        self.checkpoints_location = checkpoint_path
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    def create_model_loss_optimizer(self) -> None:
        # Select generator

        generator = self.config.generator.lower()
        scaling = 0.02
        if generator == "simple":
            self.model = Decoder(3, norm=False)
        elif generator == "deep":
            self.model = Decoder_2d_deep(3)
        elif generator == "resnet":
            self.model = GeneratorResNet(3, 12, 1)
            scaling = 0.01
        elif generator == "unet":
            self.model = UNet(input_channels=3, output_channels=1, norm=False)
        else:
            raise ValueError("Unrecognized generator type! Got", generator)

        init_weights(self.model, "normal", scaling=scaling)

        loss = self.config.loss.lower()
        if loss == "l1":
            self.loss = nn.L1Loss()
        elif loss == "l2":
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Unrecognized loss type! Got", loss)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.optim_lr)

        # IMPORTANT: model, optimizer, and scheduler need to be distributed

        # First, define strategy-wise optional configurations
        if isinstance(self.strategy, DeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = dict(
                config_params=dict(train_micro_batch_size_per_gpu=self.config.batch_size)
            )
        else:
            distribute_kwargs = {}

        # Distributed model, optimizer, and scheduler
        self.model, self.optimizer, _ = self.strategy.distributed(
            self.model, self.optimizer, **distribute_kwargs
        )

    def create_dataloaders(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ) -> None:
        """Override the create_dataloaders function to use the custom_collate function."""
        # This is the case if a small dataset is used in-memory
        # - we can use the default collate_fn function
        if isinstance(train_dataset, TensorDataset):
            return super().create_dataloaders(
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                test_dataset=test_dataset,
            )
        else:
            # If we are using a custom dataset for the large dataset,
            # we need to overwrite the collate_fn function
            self.train_dataloader = self.strategy.create_dataloader(
                dataset=train_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers_dataloader,
                pin_memory=self.config.pin_gpu_memory,
                generator=self.torch_rng,
                shuffle=self.config.shuffle_train,
                collate_fn=self.custom_collate,
            )
            if validation_dataset is not None:
                self.validation_dataloader = self.strategy.create_dataloader(
                    dataset=validation_dataset,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers_dataloader,
                    pin_memory=self.config.pin_gpu_memory,
                    generator=self.torch_rng,
                    shuffle=self.config.shuffle_validation,
                    collate_fn=self.custom_collate,
                )
            if test_dataset is not None:
                self.test_dataloader = self.strategy.create_dataloader(
                    dataset=test_dataset,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers_dataloader,
                    pin_memory=self.config.pin_gpu_memory,
                    generator=self.torch_rng,
                    shuffle=self.config.shuffle_test,
                    collate_fn=self.custom_collate,
                )

    def custom_collate(self, batch):
        """
        Custom collate function to concatenate input tensors along their first dimension.
        """
        # Some batches contain None values, if any files from the dataset did not match the
        # criteria (i.e. three auxilliary channels)
        batch = [x for x in batch if x is not None]

        return torch.cat(batch)

    @suppress_workers_print
    def execute(
        self,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ) -> Tuple[Dataset, Dataset, Dataset, Any]:
        return super().execute(train_dataset, validation_dataset, test_dataset)

    @profile_torch_trainer
    @measure_gpu_utilization
    def train(self):
        # Start the timer for profiling
        #
        st = timer()
        # uncomment all lines relative to accuracy if you want to measure
        # IOU between generated and real spectrograms.
        # Note that it significantly slows down the whole process
        # it also might not work as the function has not been fully
        # implemented yet
        epoch_time_logger: EpochTimeTracker | None = None
        if self.strategy.is_main_worker and self.strategy.is_distributed:
            print("TIMER: broadcast:", timer() - st, "s")
            print("\nDEBUG: start training")
            print("--------------------------------------------------------")
            # nnod = os.environ.get('SLURM_NNODES', 'unk')
            # s_name = f"{os.environ.get('DIST_MODE', 'unk')}-torch"
            # save_path

            num_nodes = int(os.environ.get("SLURM_NNODES", 1))
            epoch_time_output_dir = Path(f"scalability-metrics/{self.run_id}/{EPOCH_TIME_DIR}")
            epoch_time_file_name = f"epochtime_{self.strategy.name}_{num_nodes}N.csv"
            epoch_time_output_path = epoch_time_output_dir / epoch_time_file_name
            epoch_time_logger = EpochTimeTracker(
                strategy_name=self.strategy.name,
                save_path=epoch_time_output_path,
                num_nodes=num_nodes,
                should_log=self.measure_epoch_time
            )
        loss_plot = []
        val_loss_plot = []
        acc_plot = []
        val_acc_plot = []
        best_val_loss = float("inf")

        for epoch in tqdm(range(self.num_epochs)):
            lt = timer()
            # itwinai - IMPORTANT: set current epoch ID
            self.set_epoch(epoch)
            t_list = []
            st = time.time()
            epoch_loss = []
            # epoch_acc = []
            for i, batch in enumerate(self.train_dataloader):
                t = timer()
                # The TensorDataset returns batches as lists of length 1
                if isinstance(batch, list):
                    batch = batch[0]
                # batch= transform(batch)
                target = batch[:, 0].unsqueeze(1).to(self.device)
                # print(f'TARGET ON DEVICE: {target.get_device()}')
                target = target.float()
                input = batch[:, 1:].to(self.device)
                # print(f'INPUT ON DEVICE: {input.get_device()}')

                self.optimizer.zero_grad()
                generated = self.model(input.float())
                # generated=normalize_(generated,1)
                loss = self.loss(generated, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.detach().cpu().numpy())
                t_list.append(timer() - t)
                # itwinai - log loss as metric
                self.log(
                    loss.detach().cpu().numpy(),
                    "epoch_loss_batch",
                    kind="metric",
                    step=epoch * len(self.train_dataloader) + i,
                    batch_idx=i,
                )
                # acc=accuracy(generated.detach().cpu().numpy(),target.detach().cpu().numpy(),20)
                # epoch_acc.append(acc)
            if self.strategy.is_main_worker:
                print("TIMER: train time", sum(t_list) / len(t_list), "s")
            val_loss = []
            # val_acc = []
            for i, batch in enumerate(self.validation_dataloader):
                # batch= transform(batch)
                if isinstance(batch, list):
                    batch = batch[0]
                target = batch[:, 0].unsqueeze(1).to(self.device)
                target = target.float()
                input = batch[:, 1:].to(self.device)
                with torch.no_grad():
                    generated = self.model(input.float())
                    # generated=normalize_(generated,1)
                    loss = self.loss(generated, target)
                val_loss.append(loss.detach().cpu().numpy())
                # itwinai -log loss as metric
                self.log(
                    loss.detach().cpu().numpy(),
                    "val_loss_batch",
                    kind="metric",
                    step=epoch * len(self.validation_dataloader) + i,
                    batch_idx=i,
                )
                # acc=accuracy(generated.detach().cpu().numpy(),target.detach().cpu().numpy(),20)
                # val_acc.append(acc)
            loss_plot.append(np.mean(epoch_loss))
            val_loss_plot.append(np.mean(val_loss))
            # acc_plot.append(np.mean(epoch_acc))
            # val_acc_plot.append(np.mean(val_acc))

            # itwinai - Log metrics/losses
            self.log(np.mean(epoch_loss), "epoch_loss", kind="metric", step=epoch)
            self.log(np.mean(val_loss), "val_loss", kind="metric", step=epoch)
            # self.log(np.mean(epoch_acc), 'epoch_acc',
            #          kind='metric', step=epoch)
            # self.log(np.mean(val_acc), 'val_acc',
            #          kind='metric', step=epoch)

            # print('epoch: {} loss: {} val loss: {} accuracy: {} val
            # accuracy: {}'.format(epoch,loss_plot[-1],val_loss_plot[-1],
            # acc_plot[-1],val_acc_plot[-1]))
            et = time.time()
            # itwinai - print() in a multi-worker context (distributed)
            if self.strategy.is_main_worker:
                print(
                    "epoch: {} loss: {} val loss: {} time:{}s".format(
                        epoch, loss_plot[-1], val_loss_plot[-1], et - st
                    )
                )

            # Save checkpoint every #validation_every epochs
            if self.validation_every and epoch % self.validation_every == 0:
                # uncomment the following if you want to save checkpoint every
                # 100 epochs regardless of the performance of the model
                # checkpoint = {
                #     'epoch': epoch,
                #     'model_state_dict': generator.state_dict(),
                #     'optim_state_dict': optimizer.state_dict(),
                #     'loss': loss_plot[-1],
                #     'val_loss': val_loss_plot[-1],
                # }
                # if self.strategy.is_main_worker:
                #     # Save only in the main worker
                #     checkpoint_filename = checkpoint_path.format(epoch)
                #     torch.save(checkpoint, checkpoint_filename)

                # Average loss among all workers
                # itwinai - gather local loss from all the workers
                worker_val_losses = self.strategy.gather_obj(val_loss_plot[-1])
                if self.strategy.is_main_worker:
                    # Save only in the main worker

                    # avg_loss has a meaning only in the main worker
                    avg_loss = np.mean(worker_val_losses)

                    # instead of val_loss and best_val loss we should
                    # use accuracy!!!
                    if self.config.save_best and avg_loss < best_val_loss:
                        # create checkpoint
                        checkpoint = {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optim_state_dict": self.optimizer.state_dict(),
                            "loss": loss_plot[-1],
                            "val_loss": val_loss_plot[-1],
                        }

                        # save checkpoint only if it is better than
                        # the previous ones
                        checkpoint_filename = self.checkpoints_location.format(epoch)
                        torch.save(checkpoint, checkpoint_filename)
                        # itwinai - log checkpoint as artifact
                        self.log(
                            checkpoint_filename,
                            os.path.basename(checkpoint_filename),
                            kind="artifact",
                        )

                        # update best model
                        best_val_loss = val_loss_plot[-1]
                        best_checkpoint_filename = self.checkpoints_location.format("best")
                        torch.save(checkpoint, best_checkpoint_filename)
                        # itwinai - log checkpoint as artifact
                        self.log(
                            best_checkpoint_filename,
                            os.path.basename(best_checkpoint_filename),
                            kind="artifact",
                        )
            # return (loss_plot, val_loss_plot,
            # acc_plot, val_acc_plot ,acc_plot, val_acc_plot)
            if self.strategy.is_main_worker and self.strategy.is_distributed:
                print("TIMER: epoch time:", timer() - lt, "s")
                assert epoch_time_logger is not None
                epoch_time_logger.add_epoch_time(epoch - 1, timer() - lt)

            # Report training metrics of last epoch to Ray
            tune.report({"loss": np.mean(val_loss)})

        return loss_plot, val_loss_plot, acc_plot, val_acc_plot


class RayNoiseGeneratorTrainer(RayTorchTrainer):
    def __init__(
        self,
        config: Dict,
        strategy: Optional[Literal["ddp", "deepspeed"]] = "ddp",
        name: Optional[str] = None,
        logger: Optional[Logger] = None,
        random_seed: int = 1234,
    ) -> None:
        super().__init__(
            config=config, strategy=strategy, name=name, logger=logger, random_seed=random_seed
        )

    def create_model_loss_optimizer(self) -> None:
        # Select generator
        generator = self.training_config.generator
        scaling = 0.02
        if generator == "simple":
            self.model = Decoder(3, norm=False)
        elif generator == "deep":
            self.model = Decoder_2d_deep(3)
        elif generator == "resnet":
            self.model = GeneratorResNet(3, 12, 1)
            scaling = 0.01
        elif generator == "unet":
            self.model = UNet(input_channels=3, output_channels=1, norm=False)
        else:
            raise ValueError("Unrecognized generator type! Got", generator)

        init_weights(self.model, "normal", scaling=scaling)

        # Select loss
        loss = self.training_config.loss
        if loss == "l1":
            self.loss = nn.L1Loss()
        elif loss == "l2":
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Unrecognized loss type! Got", loss)

        # Optimizer
        print(type(self.training_config.optim_lr), self.training_config.optim_lr)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.training_config.optim_lr
        )

        # First, define strategy-wise optional configurations
        if isinstance(self.strategy, RayDeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = dict(
                config_params=dict(train_micro_batch_size_per_gpu=self.config.batch_size)
            )
        else:
            distribute_kwargs = {}

        # Distributed model, optimizer, and scheduler
        self.model, self.optimizer, _ = self.strategy.distributed(
            self.model, self.optimizer, **distribute_kwargs
        )

    def custom_collate(self, batch):
        """Custom collate function to concatenate input tensors along their first dimension."""
        # Some batches contain None values,
        # if any files from the dataset did not match the criteria
        # (i.e. three auxilliary channels)
        batch = [x for x in batch if x is not None]

        return torch.cat(batch)

    def train(self, config, data):
        # Because of the way the ray cluster is set up, the strategy must be initialized within
        # the training function
        self.strategy.init()

        # Start the timer for profiling
        st = timer()

        self.training_config = VirgoTrainingConfiguration(**config)

        self.create_model_loss_optimizer()

        self.create_dataloaders(
            train_dataset=data[0],
            validation_dataset=data[1],
            test_dataset=data[2],
            collate_fn=self.custom_collate,
        )

        self.initialize_logger(hyperparams=config, rank=self.strategy.global_rank())

        if self.strategy.is_main_worker:
            print("TIMER: broadcast:", timer() - st, "s")
            print("\nDEBUG: start training")
            print("--------------------------------------------------------")

        loss_plot = []
        val_loss_plot = []
        acc_plot = []
        val_acc_plot = []
        best_val_loss = float("inf")

        for epoch in tqdm(range(self.training_config.num_epochs)):
            # lt = timer()

            if self.strategy.global_world_size() > 1:
                self.set_epoch(epoch)

            t_list = []
            st = time.time()
            epoch_loss = []

            for i, batch in enumerate(self.train_dataloader):
                t = timer()
                # The TensorDataset returns batches as lists of length 1
                if isinstance(batch, list):
                    batch = batch[0]

                if isinstance(self.strategy, RayDDPStrategy):
                    target = batch[:, 0].unsqueeze(1)
                    input = batch[:, 1:]
                else:
                    target = batch[:, 0].unsqueeze(1).to(self.device)
                    input = batch[:, 1:].to(self.device)

                target = target.float()

                self.optimizer.zero_grad()
                generated = self.model(input.float())
                loss = self.loss(generated, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.detach().cpu().numpy())
                t_list.append(timer() - t)
                # itwinai - log loss as metric
                self.log(
                    loss.detach().cpu().numpy(),
                    "epoch_loss_batch",
                    kind="metric",
                    step=epoch * len(self.train_dataloader) + i,
                    batch_idx=i,
                )

            if self.strategy.is_main_worker:
                print("TIMER: train time", sum(t_list) / len(t_list), "s")
            val_loss = []

            for i, batch in enumerate(self.validation_dataloader):
                if isinstance(batch, list):
                    batch = batch[0]

                if isinstance(self.strategy, RayDDPStrategy):
                    target = batch[:, 0].unsqueeze(1)
                    input = batch[:, 1:]
                else:
                    target = batch[:, 0].unsqueeze(1).to(self.device)
                    input = batch[:, 1:].to(self.device)

                target = target.float()
                with torch.no_grad():
                    generated = self.model(input.float())
                    # generated=normalize_(generated,1)
                    loss = self.loss(generated, target)
                val_loss.append(loss.detach().cpu().numpy())
                # itwinai -log loss as metric
                self.log(
                    loss.detach().cpu().numpy(),
                    "val_loss_batch",
                    kind="metric",
                    step=epoch * len(self.validation_dataloader) + i,
                    batch_idx=i,
                )

            loss_plot.append(np.mean(epoch_loss))
            val_loss_plot.append(np.mean(val_loss))

            # itwinai - Log metrics/losses
            self.log(np.mean(epoch_loss), "epoch_loss", kind="metric", step=epoch)
            self.log(np.mean(val_loss), "val_loss", kind="metric", step=epoch)

            et = time.time()
            # itwinai - print() in a multi-worker context (distributed)
            if self.strategy.is_main_worker:
                print(
                    "epoch: {} loss: {} val loss: {} time:{}s".format(
                        epoch, loss_plot[-1], val_loss_plot[-1], et - st
                    )
                )

                # uncomment the following if you want to save checkpoint every
                # 100 epochs regardless of the performance of the model
                # checkpoint = {
                #     'epoch': epoch,
                #     'model_state_dict': generator.state_dict(),
                #     'optim_state_dict': optimizer.state_dict(),
                #     'loss': loss_plot[-1],
                #     'val_loss': val_loss_plot[-1],
                # }
                # if self.strategy.is_main_worker:
                #     # Save only in the main worker
                #     checkpoint_filename = checkpoint_path.format(epoch)
                #     torch.save(checkpoint, checkpoint_filename)

                # Average loss among all workers
                # itwinai - gather local loss from all the workers
            # worker_val_losses = self.strategy.gather_obj(val_loss_plot[-1])

            checkpoint = None
            if self.strategy.is_main_worker:
                # save checkpoint only if it is better than
                # the previous ones
                if self.training_config["save_best"] and val_loss_plot[-1] < best_val_loss:
                    # create checkpoint
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optim_state_dict": self.optimizer.state_dict(),
                        "loss": loss_plot[-1],
                        "val_loss": val_loss_plot[-1],
                    }

            metrics = {"loss": val_loss_plot[-1]}
            self.checkpoint_and_report(
                epoch, tuning_metrics=metrics, checkpointing_data=checkpoint
            )

        self.close_logger()

        return loss_plot, val_loss_plot, acc_plot, val_acc_plot
