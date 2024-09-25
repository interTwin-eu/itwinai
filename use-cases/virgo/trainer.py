import os
import time
from timeit import default_timer as timer
from typing import Literal, Optional, Dict, Union

import numpy as np
import torch
import torch.nn as nn
from ray import train
from src.model import Decoder, Decoder_2d_deep, GeneratorResNet, UNet
from src.utils import init_weights
from tqdm import tqdm

from itwinai.loggers import EpochTimeTracker, Logger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.distributed import DeepSpeedStrategy
from itwinai.torch.trainer import TorchTrainer


class NoiseGeneratorTrainer(TorchTrainer):

    def __init__(
        self,
        config: Union[Dict, TrainingConfiguration],
        batch_size: int,
        learning_rate: float = 1e-3,
        num_epochs: int = 2,
        generator: Literal["simple", "deep", "resnet", "unet"] = "unet",
        loss: Literal["L1", "L2"] = "L1",
        strategy: Literal["ddp", "deepspeed", "horovod"] | None = 'ddp',
        checkpoint_path: str = "checkpoints/epoch_{}.pth",
        save_best: bool = True,
        logger: Optional[Logger] = None,
        random_seed: Optional[int] = None,
        name: str | None = None,
        validation_every: int = 0
    ) -> None:
        super().__init__(
            epochs=num_epochs,
            config=config,
            strategy=strategy,
            logger=logger,
            random_seed=random_seed,
            name=name,
            validation_every=validation_every
        )
        self.save_parameters(**self.locals2params(locals()))
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._generator = generator
        self._loss = loss
        self.checkpoints_location = checkpoint_path
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        # Global training configuration
        self.config = TrainingConfiguration(
            batch_size=batch_size,
            save_best=save_best,
            shuffle_train=True
        )

    def create_model_loss_optimizer(self) -> None:
        # Select generator
        generator = self._generator.lower()
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
        loss = self._loss.upper()
        if loss == "L1":
            self.loss = nn.L1Loss()
        elif loss == "L2":
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Unrecognized loss type! Got", loss)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

        # IMPORTANT: model, optimizer, and scheduler need to be distributed

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

        # Distributed model, optimizer, and scheduler
        self.model, self.optimizer, _ = self.strategy.distributed(
            self.model, self.optimizer, **distribute_kwargs
        )

    def train(self):
        # Start the timer for profiling
        st = timer()
        # uncomment all lines relative to accuracy if you want to measure
        # IOU between generated and real spectrograms.
        # Note that it significantly slows down the whole process
        # it also might not work as the function has not been fully
        # implemented yet

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
                target = batch[:, 0].unsqueeze(1).to(self.device)
                target = target.float()
                input = batch[:, 1:].to(self.device)
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
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optim_state_dict': self.optimizer.state_dict(),
                            'loss': loss_plot[-1],
                            'val_loss': val_loss_plot[-1],
                        }

                        # save checkpoint only if it is better than
                        # the previous ones
                        checkpoint_filename = self.checkpoints_location.format(
                            epoch)
                        torch.save(checkpoint, checkpoint_filename)
                        # itwinai - log checkpoint as artifact
                        self.log(checkpoint_filename,
                                 os.path.basename(checkpoint_filename),
                                 kind='artifact')

                        # update best model
                        best_val_loss = val_loss_plot[-1]
                        best_checkpoint_filename = (
                            self.checkpoints_location.format('best')
                        )
                        torch.save(checkpoint, best_checkpoint_filename)
                        # itwinai - log checkpoint as artifact
                        self.log(best_checkpoint_filename,
                                 os.path.basename(best_checkpoint_filename),
                                 kind='artifact')
            # return (loss_plot, val_loss_plot,
            # acc_plot, val_acc_plot ,acc_plot, val_acc_plot)
            if self.strategy.is_main_worker:
                print('TIMER: epoch time:', timer()-lt, 's')
                epoch_time_tracker.add_epoch_time(epoch-1, timer()-lt)

            # Report training metrics of last epoch to Ray
            train.report({"loss": np.mean(val_loss),
                          "train_loss": np.mean(epoch_loss)})

        return loss_plot, val_loss_plot, acc_plot, val_acc_plot
