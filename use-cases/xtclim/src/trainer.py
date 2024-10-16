# This script trains the Convolutional Variational Auto-Encoder (CVAE)
# network on preprocessed CMIP6 Data


"""
from codecarbon import EmissionsTracker

# Instantiate the tracker object
tracker = EmissionsTracker(
    output_dir="../code_carbon/",  # define the directory where to write the emissions results
    output_file="emissions.csv",  # define the name of the file containing the emissions results
    # log_level='error' # comment out this line to see regular output
)
tracker.start()
"""

from typing import Literal, Optional
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import sys
import os

from itwinai.components import monitor_exec
from itwinai.torch.trainer import TorchTrainer
from itwinai.torch.distributed import (
    distributed_resources_available,
    TorchDistributedStrategy,
    TorchDDPStrategy,
    HorovodStrategy,
    DeepSpeedStrategy,
    NonDistributedStrategy
)
from itwinai.loggers import Logger
from itwinai.torch.config import TrainingConfiguration

import model
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from engine import train, validate
from utils import save_reconstructed_images, save_loss_plot, save_ex
from initialization import beta, criterion, pixel_wise_criterion

class XTClimTrainer(TorchTrainer):
    def __init__(
            self,
            epochs: int,
            batch_size: int,
            lr: float,
            seasons: Literal["winter_", "spring_", "summer_", "autumn_"] = 'winter_',
            strategy: Literal["ddp", "deepspeed", "horovod"] = 'ddp',
            save_best: bool = True,
            logger: Optional[Logger] = None
    ):
        super().__init__(
            epochs=epochs,
            config={},
            strategy=strategy,
            logger=logger
        )
        self.epochs = epochs
        self.seasons = seasons
        # Global training configuration
        self.config = TrainingConfiguration(
            batch_size = batch_size,
            lr = lr,
            save_best=save_best,
            n_memb = 1, # number of members used in training the network
            stop_delta = 0.01,  # under 1% improvement consider the model starts converging
            patience = 15,  # wait for a few epochs to be sure before actually stopping
            early_count = 0,  # count when validation loss < stop_delta
            old_valid_loss = 0  # keep track of validation loss at t-1
        )

    def final_loss(self, bce_loss, mu, logvar, beta=0.1):
        """
        Adds up reconstruction loss (BCELoss) and Kullback-Leibler divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        Args:
            - bce_loss (torch.Tensor): recontruction loss
            - mu (torch.Tensor): the mean from the latent vector
            - logvar (torch.Tensor): log variance from the latent vector
            - beta (torch.Tensor): weight over the KL-Divergence

        Returns:
            - total loss (torch.Tensor)
        """
        BCE = bce_loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta*KLD

    @monitor_exec
    def execute(self):

        # pick the season to study among:
        season = self.seasons

        # Initialize distributed backend
        self._init_distributed_strategy()
        # initialize the model
        cvae_model = model.ConvVAE()
        optimizer = optim.Adam(cvae_model.parameters(), lr=self.config.lr)

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
        cvae_model, optimizer, _ = self.strategy.distributed(
            cvae_model, optimizer, **distribute_kwargs
        )

        # load training set and train data
        train_time = pd.read_csv(f"input/dates_train_{season}data_{self.config.n_memb}memb.csv")
        train_data = np.load(
            f"input/preprocessed_1d_train_{season}data_{self.config.n_memb}memb.npy"
        )
        n_train = len(train_data)
        trainset = [
            (torch.from_numpy(np.reshape(train_data[i], (2, 32, 32))), train_time["0"][i])
            for i in range(n_train)
        ]
        # load train set, shuffle it, and create batches
        trainloader = self.strategy.create_dataloader(trainset, batch_size=self.config.batch_size,
                shuffle=True, pin_memory=True)

        # load validation set and validation data
        test_time = pd.read_csv(f"input/dates_test_{season}data_{self.config.n_memb}memb.csv")
        test_data = np.load(f"input/preprocessed_1d_test_{season}data_{self.config.n_memb}memb.npy")
        n_test = len(test_data)
        testset = [
            (torch.from_numpy(np.reshape(test_data[i], (2, 32, 32))), test_time["0"][i])
            for i in range(n_test)
        ]
        testloader = self.strategy.create_dataloader(testset, batch_size=self.config.batch_size,
                shuffle=False, pin_memory=True)

        if self.strategy.is_main_worker and self.logger:
            self.logger.create_logger_context()

        # a list to save all the reconstructed images in PyTorch grid format
        grid_images = []
        # a list to save the loss evolutions
        train_loss = []
        valid_loss = []
        min_valid_epoch_loss = float('inf')  # random high value

        for epoch in range(self.epochs):
            if self.strategy.is_main_worker:
                print(f"Epoch {epoch+1} of {self.epochs}")

            if self.strategy.is_distributed:
                # Inform the sampler that a new epoch started: shuffle
                # may be needed
                trainloader.sampler.set_epoch(epoch)
                testloader.sampler.set_epoch(epoch)

            # train the model
            train_epoch_loss = train(
                cvae_model, trainloader, trainset, self.device, optimizer, criterion, beta
            )

            # evaluate the model on the test set
            valid_epoch_loss, recon_images = validate(
                cvae_model, testloader, testset, self.device, criterion, beta
            )

            self.log(train_epoch_loss,
                    'epoch_train_loss',
                    kind='metric'
                    )
            self.log(valid_epoch_loss,
                    'epoch_valid_loss',
                    kind='metric'
                    )

            # keep track of the losses
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)

            # save the reconstructed images from the validation loop
            #save_reconstructed_images(recon_images, epoch+1, season)

            # convert the reconstructed images to PyTorch image grid format
            image_grid = make_grid(recon_images.detach().cpu())
            grid_images.append(image_grid)
            # save one example of reconstructed image before and after training

            #if epoch == 0 or epoch == self.epochs-1:
            #    save_ex(recon_images[0], epoch, season)

            # decreasing learning rate
            if (epoch + 1) % 20 == 0:
                self.config.lr = self.config.lr / 5

            # early stopping to avoid overfitting
            if (
                epoch > 1
                and (self.config.old_valid_loss - valid_epoch_loss) / self.config.old_valid_loss < self.config.stop_delta
            ):
                # if the marginal improvement in validation loss is too small
                self.config.early_count += 1
                if self.config.early_count > self.config.patience:
                # if too small improvement for a few epochs in a row, stop learning
                    save_ex(recon_images[0], epoch, season)
                    break

            else:
                # if the condition is not verified anymore, reset the count
                self.config.early_count = 0
            self.config.old_valid_loss = valid_epoch_loss

            # save best model
            worker_val_losses = self.strategy.gather_obj(valid_epoch_loss)
            if self.strategy.is_main_worker:
                # Save only in the main worker
                # avg_loss has a meaning only in the main worker
                avg_loss = np.mean(worker_val_losses)
                if self.config.save_best and avg_loss < min_valid_epoch_loss:
                    min_valid_epoch_loss = avg_loss
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': cvae_model.state_dict(),
                        'optim_state_dict': optimizer.state_dict(),
                        'val_loss': valid_epoch_loss,
                    }
                    # save checkpoint only if it is better than
                    # the previous ones
                    checkpoint_filename = f"outputs/cvae_model_{season}1d_{self.config.n_memb}memb.pth"
                    torch.save(checkpoint, checkpoint_filename)
                    # itwinai - log checkpoint as artifact
                    self.log(checkpoint_filename,
                            os.path.basename(checkpoint_filename),
                            kind='artifact')

        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {valid_epoch_loss:.4f}")

        save_loss_plot(train_loss, valid_loss, season)
        # save the loss evolutions
        pd.DataFrame(train_loss).to_csv(
            f"outputs/train_loss_indiv_{season}1d_{self.config.n_memb}memb.csv"
        )
        pd.DataFrame(valid_loss).to_csv(
            f"outputs/test_loss_indiv_{season}1d_{self.config.n_memb}memb.csv"
        )

        # Clean-up
        if self.strategy.is_main_worker and self.logger:
            self.logger.destroy_logger_context()

        self.strategy.clean_up()

    # emissions = tracker.stop()
    # print(f"Emissions from this training run: {emissions:.5f} kg CO2eq")
