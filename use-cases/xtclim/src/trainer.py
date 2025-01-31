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

import torch
import torch.optim as optim
import numpy as np
import pandas as pd

from itwinai.components import Trainer, monitor_exec
import model
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from engine import train, validate
from utils import save_reconstructed_images, save_loss_plot, save_ex
from initialization import device, beta, criterion

class TorchTrainer(Trainer):
    def __init__(
            self,
            epochs: int,
            batch_size: int,
            lr: float
    ):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    @monitor_exec
    def execute(self):
        # pick the season to study among:
        # '' (none, i.e. full dataset), 'winter_', 'spring_', 'summer_', 'autumn_'
        seasons = ["winter_", "spring_", "summer_", "autumn_"]

        # number of members used for the training of the network
        n_memb = 1

        # initialize learning parameters
        #lr0 = 0.001
        #batch_size = 64
        #epochs = 100
        #early stopping parameters
        stop_delta = 0.01  # under 1% improvement consider the model starts converging
        patience = 15  # wait for a few epochs to be sure before actually stopping
        early_count = 0  # count when validation loss < stop_delta
        old_valid_loss = 0  # keep track of validation loss at t-1

        for season in seasons:

            # initialize the model
            cvae_model = model.ConvVAE().to(device)
            optimizer = optim.Adam(cvae_model.parameters(), lr=self.lr)

            # load training set and train data
            train_time = pd.read_csv(f"input/dates_train_{season}data_{n_memb}memb.csv")
            train_data = np.load(
                f"input/preprocessed_1d_train_{season}data_{n_memb}memb.npy"
            )
            n_train = len(train_data)
            trainset = [
                (torch.from_numpy(np.reshape(train_data[i], (2, 32, 32))), train_time["0"][i])
                for i in range(n_train)
            ]
            # load train set, shuffle it, and create batches
            trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

            # load validation set and validation data
            test_time = pd.read_csv(f"input/dates_test_{season}data_{n_memb}memb.csv")
            test_data = np.load(f"input/preprocessed_1d_test_{season}data_{n_memb}memb.npy")
            n_test = len(test_data)
            testset = [
                (torch.from_numpy(np.reshape(test_data[i], (2, 32, 32))), test_time["0"][i])
                for i in range(n_test)
            ]
            testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)

            # a list to save all the reconstructed images in PyTorch grid format
            grid_images = []
            # a list to save the loss evolutions
            train_loss = []
            valid_loss = []
            min_valid_epoch_loss = 100  # random high value

            for epoch in range(self.epochs):
                print(f"Epoch {epoch+1} of {self.epochs}")

                # train the model
                train_epoch_loss = train(
                    cvae_model, trainloader, trainset, device, optimizer, criterion, beta
                )

                # evaluate the model on the test set
                valid_epoch_loss, recon_images = validate(
                    cvae_model, testloader, testset, device, criterion, beta
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
            lr = lr / 5

#-------


        # early stopping to avoid overfitting
#        if (
#            epoch > 1
#            and (old_valid_loss - valid_epoch_loss) / old_valid_loss < stop_delta
#        ):
            # if the marginal improvement in validation loss is too small
#            early_count += 1

        #if early_count > patience:
                # if too small improvement for a few epochs in a row, stop learning
        #        save_ex(recon_images[0], epoch, season)
                #break

#        else:
            # if the condition is not verified anymore, reset the count
#            early_count = 0
#        old_valid_loss = valid_epoch_loss

#---------------

        # save best model
        if valid_epoch_loss < min_valid_epoch_loss:
            min_valid_epoch_loss = valid_epoch_loss
            torch.save(
                cvae_model.state_dict(),
                f"outputs/cvae_model_{season}1d_{n_memb}memb.pth",
            )

            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f"Val Loss: {valid_epoch_loss:.4f}")

            save_loss_plot(train_loss, valid_loss, season)
            # save the loss evolutions
            pd.DataFrame(train_loss).to_csv(
                f"outputs/train_loss_indiv_{season}1d_{n_memb}memb.csv"
            )
            pd.DataFrame(valid_loss).to_csv(
                f"outputs/test_loss_indiv_{season}1d_{n_memb}memb.csv"
            )

        # emissions = tracker.stop()
        # print(f"Emissions from this training run: {emissions:.5f} kg CO2eq")
