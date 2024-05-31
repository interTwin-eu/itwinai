"""
Simplified training script: data generation + training in one
procedural script. This is an INTERMEDIATE step of integration in itwinai.
"""
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.dataset import (
    generate_dataset_aux_channels,
    generate_dataset_main_channel,
    generate_cut_image_dataset,
    normalize_
)
from src.model import UNet  # ,Decoder, Decoder_2d_deep, GeneratorResNet
from src.utils import init_weights, calculate_iou_2d

# Global parameters
DATA_ROOT = "data"
LOAD_DATASET = True
BATCH_SIZE = 20
LR = 0.00005
SAVE_CHECKPOINT = 'choose_your_path.checkpoint_epoch_{}.pth'
N_EPOCHS = 50


def generate_dataset():
    df_aux_ts = generate_dataset_aux_channels(
        1000, 3, duration=16, sample_rate=500,
        num_waves_range=(20, 25), noise_amplitude=0.6
    )
    df_main_ts = generate_dataset_main_channel(
        df_aux_ts, weights=None, noise_amplitude=0.1
    )

    # save datasets
    save_name_main = 'TimeSeries_dataset_synthetic_main.pkl'
    save_name_aux = 'TimeSeries_dataset_synthetic_aux.pkl'
    df_main_ts.to_pickle(os.path.join(DATA_ROOT, save_name_main))
    df_aux_ts.to_pickle(os.path.join(DATA_ROOT, save_name_aux))

    # Transform to images and save to disk
    df_ts = pd.concat([df_main_ts, df_aux_ts], axis=1)
    df = generate_cut_image_dataset(
        df_ts, list(df_ts.columns),
        num_processes=20, square_size=64
    )
    save_name = 'Image_dataset_synthetic_64x64.pkl'
    df.to_pickle(os.path.join(DATA_ROOT, save_name))


def train_decoder(
    num_epochs, generator, criterion, optimizer, dataloader,
    val_loader, accuracy, checkpoint_path, save_best=True
):
    # num_epochs: (int) number of epochs for training
    # generator: (NN.Module) NN model to train
    # criterion: (torch.optim) optimizer to use in training
    # dataloader: (DataLoader) training data
    # val_loader: (Dataloader) validation data
    # accuracy: (function) metric to measure performance of the model
    #   (Note not to be confused with loss)
    # checkpoint_path: (str) full path (including filename in the form
    #   filename_{}.pkl so to insert num_epoch) to save checkpoints at
    # save_best: (bool) if you want to save best performing model

    # uncomment all lines relative to accuracy if you want to measure IOU
    #   between generated and real spectrograms.
    # Note that it significantly slows down the whole process
    # it also might not work as the function has not been fully implemented yet

    loss_plot = []
    val_loss_plot = []
    acc_plot = []
    val_acc_plot = []
    best_val_loss = 5000000
    for epoch in tqdm(range(1, num_epochs+1)):
        st = time.time()
        epoch_loss = []
        epoch_acc = []
        for i, batch in enumerate(dataloader):
            # batch= transform(batch)
            target = batch[:, 0].unsqueeze(1).to(device)
            # print(f'TARGET ON DEVICE: {target.get_device()}')
            target = target.float()
            input = batch[:, 1:].to(device)
            # print(f'INPUT ON DEVICE: {input.get_device()}')

            optimizer.zero_grad()
            generated = generator(input.float())
            # generated=normalize_(generated,1)
            loss = criterion(generated, target)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.detach().cpu().numpy())
            # acc=accuracy(generated.detach().cpu().numpy(),target.detach().cpu().numpy(),20)
            # epoch_acc.append(acc)
        val_loss = []
        val_acc = []
        for batch in (val_loader):
            # batch= transform(batch)
            target = batch[:, 0].unsqueeze(1).to(device)
            target = target.float()
            input = batch[:, 1:].to(device)
            with torch.no_grad():
                generated = generator(input.float())
                # generated=normalize_(generated,1)
                loss = criterion(generated, target)
                val_loss.append(loss.detach().cpu().numpy())
                # acc=accuracy(generated.detach().cpu().numpy(),target.detach().cpu().numpy(),20)
                # val_acc.append(acc)
        loss_plot.append(np.mean(epoch_loss))
        val_loss_plot.append(np.mean(val_loss))
        acc_plot.append(np.mean(epoch_acc))
        val_acc_plot.append(np.mean(val_acc))
        # print('epoch: {} loss: {} val loss: {} accuracy: {} val accuracy:
        # {}'.format(epoch,loss_plot[-1],val_loss_plot[-1],acc_plot[-1],val_acc_plot[-1]))
        et = time.time()
        print('epoch: {} loss: {} val loss: {} time:{}s'.format(
            epoch, loss_plot[-1], val_loss_plot[-1], et-st))

        # Save checkpoint every 100 epochs
        if (epoch+1) % 100 == 0:
            # uncomment the following if you want to save checkpoint every 100
            # epochs regardless of the performance of the model
            # checkpoint = {
            #     'epoch': epoch,
            #     'model_state_dict': generator.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': loss_plot[-1],
            #     'val_loss': val_loss_plot[-1],
            # }

            # checkpoint_filename = checkpoint_path.format(epoch)
            # torch.save(checkpoint, checkpoint_filename)

            # instead of val_loss and best_val loss we should use accuracy!!!
            if save_best and val_loss_plot[-1] < best_val_loss:
                # create checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': generator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_plot[-1],
                    'val_loss': val_loss_plot[-1],
                }

                # save checkpoint only if it is better than the previous ones
                checkpoint_filename = checkpoint_path.format(epoch)
                torch.save(checkpoint, checkpoint_filename)

                # update best model
                best_val_loss = val_loss_plot[-1]
                best_checkpoint_filename = checkpoint_path.format('best')
                torch.save(checkpoint, best_checkpoint_filename)

    return loss_plot, val_loss_plot, acc_plot, val_acc_plot


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        device = 'cpu'

    # Generate dataset
    if not LOAD_DATASET:
        generate_dataset()

    # Split data
    file_path = os.path.join(DATA_ROOT, 'Image_dataset_synthetic_64x64.pkl')
    df = pd.read_pickle(file_path)
    # Convert data to torch
    df = df.applymap(lambda x: torch.tensor(x))

    # Divide Image dataset in main and aux channels. Note that df generated in
    # the section Generate Synthetic Dataset will always have the main channel
    # as its first column
    main_channel = list(df.columns)[0]
    aux_channels = list(df.columns)[1:]

    df_aux_all_2d = pd.DataFrame(df[aux_channels])
    df_main_all_2d = pd.DataFrame(df[main_channel])

    X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
        df_aux_all_2d, df_main_all_2d, test_size=0.1, random_state=42)

    # TRAINING SET

    # smaller dataset
    signal_data_train_small_2d = torch.stack([torch.stack(
        [y_train_2d[main_channel].iloc[i]]) for i in range(100)])
    # for i in range(y_train.shape[0])
    aux_data_train_small_2d = torch.stack([torch.stack(
        [X_train_2d.iloc[i][0], X_train_2d.iloc[i][1], X_train_2d.iloc[i][2]])
        for i in range(100)])  # for i in range(X_train.shape[0])

    # whole dataset
    signal_data_train_2d = torch.stack([torch.stack(
        [y_train_2d[main_channel].iloc[i]])
        for i in range(y_train_2d.shape[0])])
    aux_data_train_2d = torch.stack([torch.stack(
        [X_train_2d.iloc[i][0], X_train_2d.iloc[i][1], X_train_2d.iloc[i][2]])
        for i in range(X_train_2d.shape[0])])

    # concatenate torch.tensors
    train_data_2d = torch.cat([signal_data_train_2d, aux_data_train_2d], dim=1)
    train_data_small_2d = torch.cat(
        [signal_data_train_small_2d, aux_data_train_small_2d], dim=1)

    print(signal_data_train_2d.shape)
    print(aux_data_train_2d.shape)
    # --------------------------------------------

    # TEST SET

    # smaller dataset
    signal_data_test_small_2d = torch.stack([torch.stack(
        [y_test_2d[main_channel].iloc[i]]) for i in range(100)])
    # for i in range(y_test.shape[0])
    aux_data_test_small_2d = torch.stack([torch.stack(
        [X_test_2d.iloc[i][0], X_test_2d.iloc[i][1], X_test_2d.iloc[i][2]])
        for i in range(100)])  # for i in range(X_test.shape[0])

    # whole dataset
    signal_data_test_2d = torch.stack([torch.stack(
        [y_test_2d[main_channel].iloc[i]]) for i in range(y_test_2d.shape[0])])
    aux_data_test_2d = torch.stack([torch.stack(
        [X_test_2d.iloc[i][0], X_test_2d.iloc[i][1], X_test_2d.iloc[i][2]])
        for i in range(X_test_2d.shape[0])])

    test_data_2d = torch.cat([signal_data_test_2d, aux_data_test_2d], dim=1)
    test_data_small_2d = torch.cat(
        [signal_data_test_small_2d, aux_data_test_small_2d], dim=1)

    print(signal_data_test_2d.shape)
    print(aux_data_test_2d.shape)
    # -----------------------------------------------

    # Normalize
    train_data_2d = normalize_(train_data_2d)
    test_data_2d = normalize_(test_data_2d)

    # Create dataloader objects with preprocessed dataset
    dataloader = DataLoader(
        train_data_2d,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_data_2d,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Model to train
    generator_2d = UNet(
        input_channels=3, output_channels=1, norm=False).to(device)
    init_weights(generator_2d, 'normal', scaling=.02)

    # loss function, learning rate, and optimizer
    l2_loss = nn.MSELoss()  # this is l2!!!
    l1_loss = nn.L1Loss()  # this is L1!!!
    loss = l1_loss  # LogCoshLoss()
    G_optimizer = torch.optim.Adam(generator_2d.parameters(), lr=LR)

    # Training
    loss_plot, val_loss_plot, acc_plot, val_acc_plot = train_decoder(
        N_EPOCHS, generator_2d, loss, G_optimizer, dataloader, test_dataloader,
        calculate_iou_2d, SAVE_CHECKPOINT
    )
