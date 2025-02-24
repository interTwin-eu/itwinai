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

from typing import Dict, List, Literal, Optional, Tuple
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

from itwinai.components import monitor_exec
from itwinai.torch.trainer import TorchTrainer
from itwinai.torch.distributed import DeepSpeedStrategy
from itwinai.loggers import Logger
from itwinai.torch.config import TrainingConfiguration

from ray import train as train_ray
import src.model as model
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from src.engine import train, validate
from src.utils import save_reconstructed_images, save_loss_plot, save_ex
from src.initialization import beta, criterion, pixel_wise_criterion

class XTClimTrainer(TorchTrainer):
    def __init__(
            self,
            epochs: int,
            config: Dict | TrainingConfiguration | None = None,
            seasons: Literal["winter_", "spring_", "summer_", "autumn_"] = 'winter_',
            strategy: Literal["ddp", "deepspeed", "horovod"] = 'ddp',
            save_best: bool = True,
            logger: Logger | None = None
    ):
        super().__init__(
            epochs=epochs,
            config=config,
            strategy=strategy,
            logger=logger
        )
        self.epochs = epochs
        self.seasons = seasons
        # Global training configuration
        if isinstance(config, dict):
            config = TrainingConfiguration(**config)
        self.config = config

    def final_loss(
        self,
        bce_loss: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 0.1
    ) -> torch.Tensor:
        """Adds up reconstruction loss (BCELoss) and Kullback-Leibler divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        Args:
            bce_loss (torch.Tensor): recontruction loss
            mu (torch.Tensor): the mean from the latent vector
            logvar (torch.Tensor): log variance from the latent vector
            beta (torch.Tensor): weight over the KL-Divergence

        Returns:
            total loss (torch.Tensor)
        """
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce_loss + beta*KLD

    def _load_data(self, input_dir: Path, season: str) -> Tuple[DataLoader, DataLoader, List, List]:
        """Load the training and test dataloaders.

        Args:
            input_dir (Path): directory where input data is stored.
            season (str): selected season ('winter_', 'spring_', etc.).

        Returns:
            Tuple[DataLoader, DataLoader, List, List]: tuple containing dataloaders and samples.
        """
        # Load train data
        train_time = pd.read_csv(
            input_dir / f"dates_train_{season}data_{self.config.n_memb}memb.csv"
        )
        train_data = np.load(
            input_dir / f"preprocessed_1d_train_{season}data_{self.config.n_memb}memb.npy"
        )
        n_train = len(train_data)
        trainset = [
            (torch.from_numpy(np.reshape(train_data[i], (2, 32, 32))), train_time["0"][i])
            for i in range(n_train)
        ]

        # Load test data
        test_time = pd.read_csv(
            input_dir / f"dates_test_{season}data_{self.config.n_memb}memb.csv"
        )
        test_data = np.load(
            input_dir / f"preprocessed_1d_test_{season}data_{self.config.n_memb}memb.npy"
        )
        n_test = len(test_data)
        testset = [
            (torch.from_numpy(np.reshape(test_data[i], (2, 32, 32))), test_time["0"][i])
            for i in range(n_test)
        ]

        # Create dataloaders
        trainloader = self.strategy.create_dataloader(
            trainset, batch_size=self.config.batch_size, shuffle=True, pin_memory=True
        )
        testloader = self.strategy.create_dataloader(
            testset, batch_size=self.config.batch_size, shuffle=False, pin_memory=True
        )

        return trainloader, testloader, trainset, testset

    def _handle_early_stopping(self, epoch: int, valid_epoch_loss: float) -> bool:
        """Handle early stopping based on validation loss.

        Args:
            epoch (int): Current epoch number.
            valid_epoch_loss (float): Validation loss for current epoch.

        Returns:
            bool: True if training should stop early, False otherwise.
        """
        min_epochs = epoch > 1
        improve_below_threshold = (
            (self.config.old_valid_loss - valid_epoch_loss) / self.config.old_valid_loss
        ) < self.config.stop_delta
        if min_epochs and improve_below_threshold:
            # if the marginal improvement in validation loss is too small
            self.config.early_count += 1
            if self.config.early_count > self.config.patience:
                # if too small improvement for a few epochs in a row, stop learning
                print(f"Early stopping at epoch {epoch}")
                save_ex(recon_images[0], epoch, season)
                return True
        else:
            # if the condition is not verified anymore, reset the count
            self.config.early_count = 0

    def _save_best_model(
        self,
        cvae_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        valid_epoch_loss: float,
        epoch: int,
        season: str,
        output_dir: Path
    ) -> None:
        """Save the model checkpoint if best.

        Args:
            cvae_model (torch.nn.Module): model under training.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            valid_epoch_loss (float): Validation loss for current epoch.
            epoch (int): Current epoch number.
            season (str): Season selected (e.g. 'winter_', 'spring_', etc.).
            output_dir (Path): directory to save checkpoints.

        Returns:
            None
        """
        worker_val_losses = self.strategy.gather_obj(valid_epoch_loss)
        if self.strategy.is_main_worker:
            # Save only in the main worker
            # avg_loss has a meaning only in the main worker
            avg_loss = np.mean(worker_val_losses)
            if self.config.save_best and avg_loss < self.min_valid_epoch_loss:
                self.min_valid_epoch_loss = avg_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': cvae_model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'val_loss': valid_epoch_loss,
                }
                # save checkpoint only if it is better than the previous ones
                checkpoint_filename = output_dir / f"cvae_model_{season}1d_{self.config.n_memb}memb.pth"
                torch.save(checkpoint, checkpoint_filename)
                # itwinai - log checkpoint as artifact
                self.log(checkpoint_filename, Path(checkpoint_filename).name, kind='artifact')

    @monitor_exec
    def execute(self):

        # Define the base directory for input and output data
        input_dir = Path("input")
        output_dir = Path("outputs")

        # Pick the season to study among:
        season = self.seasons

        # Initialize distributed backend
        self._init_distributed_strategy()

        # initialize the model
        cvae_model = model.ConvVAE()
        optimizer = optim.Adam(cvae_model.parameters(), lr=self.config.optim_lr)

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

        # Load training and validation dataloaders
        trainloader, testloader, trainset, testset = self._load_data(input_dir, season)

        if self.strategy.is_main_worker and self.logger:
            self.logger.create_logger_context()

        # Initialize lists to track loss and images
        grid_images = []
        train_loss = []
        valid_loss = []
        self.min_valid_epoch_loss = float('inf')  # random initial value

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

            self.log(train_epoch_loss, 'epoch_train_loss', kind='metric')
            self.log(valid_epoch_loss, 'epoch_valid_loss', kind='metric')

            # keep track of the losses
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)

            # Save reconstructed images
            #save_reconstructed_images(recon_images, epoch + 1, season)

            # Convert the reconstructed images to PyTorch grid format
            image_grid = make_grid(recon_images.detach().cpu())
            grid_images.append(image_grid)

            # decreasing learning rate
            if (epoch + 1) % self.config.lr_decay_interval == 0:
                self.config.optim_lr = self.config.optim_lr / self.config.lr_decay_rate

            # Early stopping and learning rate decay
            if self._handle_early_stopping(epoch, valid_epoch_loss):
                break
            self.config.old_valid_loss = valid_epoch_loss

            # Save the best model checkpoint
            self._save_best_model(
                cvae_model, optimizer, valid_epoch_loss, epoch, season, output_dir
            )

        # Final loss reports
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {valid_epoch_loss:.4f}")

        # Report training metrics of last epoch to Ray
        train_ray.report(
            {
                "loss": train_epoch_loss,
                "valid_loss": valid_epoch_loss
            }
        )

        save_loss_plot(train_loss, valid_loss, season)
        # save the loss evolutions
        pd.DataFrame(train_loss).to_csv(
            output_dir / f"train_loss_indiv_{season}1d_{self.config.n_memb}memb.csv"
        )
        pd.DataFrame(valid_loss).to_csv(
            output_dir / f"test_loss_indiv_{season}1d_{self.config.n_memb}memb.csv"
        )

        # Clean-up strategy and logger
        if self.strategy.is_main_worker and self.logger:
            self.logger.destroy_logger_context()

        self.strategy.clean_up()

