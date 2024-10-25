import os
import tempfile
import time
from timeit import default_timer as timer
from typing import Dict, Literal, Optional, Union
import mlflow
import numpy as np
import ray.train.torch
import torch
import torch.nn as nn
from src.model import Decoder, Decoder_2d_deep, GeneratorResNet, UNet
from src.utils import init_weights
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm
from ray.tune import Trainable


from itwinai.loggers import EpochTimeTracker
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.trainer_ray import RayTorchTrainer
from itwinai.loggers import Logger


class VirgoTrainingConfiguration(TrainingConfiguration):
    """Virgo TrainingConfiguration"""
    #: Whether to save best model on validation dataset. Defaults to True.
    save_best: bool = True
    #: Loss function. Defaults to "L1".
    loss: Literal["L1", "L2"] = "L1",
    #: Generator to train. Defaults to "unet".
    generator: Literal["simple", "deep", "resnet", "unet"] = "unet"


class NoiseGeneratorTrainer(RayTorchTrainer):

    def __init__(
        self,
        config: Dict,
        strategy: Optional[Literal["ddp", "deepspeed", "horovod"]] = 'ddp',
        name: Optional[str] = None,
        logger: Optional[Logger] = None
    ) -> None:
        super().__init__(
            config=config,
            strategy=strategy,
            name=name,
            logger=logger
        )
        # Global training configuration
        # if isinstance(virgo_config, dict):
        #    virgo_config = VirgoTrainingConfiguration(**virgo_config)

        # self.virgo_config = virgo_config

    def create_model_loss_optimizer(self) -> None:
        # Select generator
        generator = self.training_config["generator"]
        if generator == "simple":
            self.model = Decoder(3, norm=False)
            init_weights(self.model, 'normal', scaling=.02)
        elif generator == "deep":
            self.model = Decoder_2d_deep(3)
            init_weights(self.model, 'normal', scaling=.02)
        elif generator == "resnet":
            self.model = GeneratorResNet(3, 12, 1)
            init_weights(self.model, 'normal', scaling=.01)
        elif generator == "unet":
            self.model = UNet(
                input_channels=3, output_channels=1, norm=False)
            init_weights(self.model, 'normal', scaling=.02)
        else:
            raise ValueError("Unrecognized generator type! Got", generator)

        # Select loss
        loss = self.training_config["loss"]
        if loss == "L1":
            self.loss = nn.L1Loss()
        elif loss == "L2":
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Unrecognized loss type! Got", loss)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.training_config["learning_rate"])

        # IMPORTANT: model, optimizer, and scheduler need to be distributed

        # Distributed model, optimizer, and scheduler
        self.model, self.optimizer, _ = self.strategy.distributed(
            self.model,
            self.optimizer
        )

    def create_dataloaders(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None
    ) -> None:
        """Override the create_dataloaders function to use the custom_collate function.
        """
        # This is the case if a small dataset is used in-memory
        # - we can use the default collate_fn function
        if isinstance(train_dataset, TensorDataset):
            return super().create_dataloaders(
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                test_dataset=test_dataset
            )
        else:
            # If we are using a custom dataset for the large dataset,
            # we need to overwrite the collate_fn function
            self.train_dataloader = self.strategy.create_dataloader(
                dataset=train_dataset,
                batch_size=self.training_config["batch_size"],
                num_workers=self.training_config["num_workers_dataloader"],
                pin_memory=self.training_config["pin_gpu_memory"],
                # generator=self.torch_rng,
                shuffle=self.training_config["shuffle_train"],
                collate_fn=self.custom_collate
            )
            if validation_dataset is not None:
                self.validation_dataloader = self.strategy.create_dataloader(
                    dataset=validation_dataset,
                    batch_size=self.training_config["batch_size"],
                    num_workers=self.training_config["num_workers_dataloader"],
                    pin_memory=self.training_config["pin_gpu_memory"],
                    # generator=self.torch_rng,
                    shuffle=self.training_config["shuffle_validation"],
                    collate_fn=self.custom_collate
                )
            if test_dataset is not None:
                self.test_dataloader = self.strategy.create_dataloader(
                    dataset=test_dataset,
                    batch_size=self.training_config["batch_size"],
                    num_workers=self.training_config["num_workers_dataloader"],
                    pin_memory=self.training_config["pin_gpu_memory"],
                    # generator=self.torch_rng,
                    shuffle=self.training_config["shuffle_test"],
                    collate_fn=self.custom_collate
                )

    def custom_collate(self, batch):
        """
        Custom collate function to concatenate input tensors along their first dimension.
        """
        # Some batches contain None values, if any files from the dataset did not match the criteria
        # (i.e. three auxilliary channels)
        batch = [x for x in batch if x is not None]

        return torch.cat(batch)

    def train(self, config, data):
        # Start the timer for profiling
        st = timer()

        self.training_config = config

        self.create_model_loss_optimizer()

        self.create_dataloaders(
            train_dataset=data[0],
            validation_dataset=data[1],
            test_dataset=data[2]
        )

        self.initialize_logger(
            hyperparams=config, rank=self.strategy.global_rank())

        if self.strategy.is_main_worker:
            print('TIMER: broadcast:', timer()-st, 's')
            print('\nDEBUG: start training')
            print('--------------------------------------------------------')
            nnod = os.environ.get('SLURM_NNODES', 'unk')
            s_name = f"{os.environ.get('DIST_MODE', 'unk')}-torch"
            epoch_time_tracker = EpochTimeTracker(
                series_name=s_name,
                csv_file=f"epochtime_{s_name}_{nnod}N.csv"
            )
        loss_plot = []
        val_loss_plot = []
        acc_plot = []
        val_acc_plot = []
        best_val_loss = float('inf')

        for epoch in tqdm(range(self.training_config["epochs"])):
            lt = timer()
            # itwinai - IMPORTANT: set current epoch ID
            if self.strategy.global_world_size() > 1:
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
                target = batch[:, 0].unsqueeze(1)  # .to(self.device)
                # print(f'TARGET ON DEVICE: {target.get_device()}')
                target = target.float()
                input = batch[:, 1:]  # .to(self.device)
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
                self.log(loss.detach().cpu().numpy(),
                         'epoch_loss_batch',
                         kind='metric',
                         step=epoch*len(self.train_dataloader) + i,
                         batch_idx=i)
                # acc=accuracy(generated.detach().cpu().numpy(),target.detach().cpu().numpy(),20)
                # epoch_acc.append(acc)
            if self.strategy.is_main_worker:
                print('TIMER: train time', sum(t_list) / len(t_list), 's')
            val_loss = []
            # val_acc = []
            for i, batch in enumerate(self.validation_dataloader):
                # batch= transform(batch)
                if isinstance(batch, list):
                    batch = batch[0]
                target = batch[:, 0].unsqueeze(1)  # .to(self.device)
                target = target.float()
                input = batch[:, 1:]  # .to(self.device)
                with torch.no_grad():
                    generated = self.model(input.float())
                    # generated=normalize_(generated,1)
                    loss = self.loss(generated, target)
                val_loss.append(loss.detach().cpu().numpy())
                # itwinai -log loss as metric
                self.log(loss.detach().cpu().numpy(),
                         'val_loss_batch',
                         kind='metric',
                         step=epoch*len(self.validation_dataloader) + i,
                         batch_idx=i)
                # acc=accuracy(generated.detach().cpu().numpy(),target.detach().cpu().numpy(),20)
                # val_acc.append(acc)
            loss_plot.append(np.mean(epoch_loss))
            val_loss_plot.append(np.mean(val_loss))
            # acc_plot.append(np.mean(epoch_acc))
            # val_acc_plot.append(np.mean(val_acc))

            # itwinai - Log metrics/losses
            self.log(np.mean(epoch_loss), 'epoch_loss',
                     kind='metric', step=epoch)
            self.log(np.mean(val_loss), 'val_loss',
                     kind='metric', step=epoch)
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
                print('epoch: {} loss: {} val loss: {} time:{}s'.format(
                    epoch, loss_plot[-1], val_loss_plot[-1], et-st))

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
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optim_state_dict': self.optimizer.state_dict(),
                        'loss': loss_plot[-1],
                        'val_loss': val_loss_plot[-1],
                    }

            #             # save checkpoint only if it is better than
            #             # the previous ones
            #             checkpoint_filename = self.checkpoints_location.format(
            #                 epoch)
            #             torch.save(checkpoint, checkpoint_filename)
            #             # itwinai - log checkpoint as artifact
            #             self.log(checkpoint_filename,
            #                      os.path.basename(checkpoint_filename),
            #                      kind='artifact')

            #             # update best model
            #             best_val_loss = val_loss_plot[-1]
            #             best_checkpoint_filename = (
            #                 self.checkpoints_location.format('best')
            #             )
            #             torch.save(checkpoint, best_checkpoint_filename)
            #             # itwinai - log checkpoint as artifact
            #             self.log(best_checkpoint_filename,
            #                      os.path.basename(best_checkpoint_filename),
            #                      kind='artifact')
            # #return (loss_plot, val_loss_plot,

            # #acc_plot, val_acc_plot ,acc_plot, val_acc_plot)

            metrics = {
                "loss": val_loss_plot[-1]
            }
            self.checkpoint_and_report(
                epoch,
                tuning_metrics=metrics,
                checkpointing_data=checkpoint
            )

        if self.strategy.is_main_worker:
            mlflow.end_run()

        return loss_plot, val_loss_plot, acc_plot, val_acc_plot
