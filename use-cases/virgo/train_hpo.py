import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray import train
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from itwinai.torch.reproducibility import set_seed

from src.dataset import (
    generate_dataset_aux_channels,
    generate_dataset_main_channel,
    generate_cut_image_dataset,
    normalize_
)
from src.model import UNet, GeneratorResNet, Decoder
from src.utils import init_weights, calculate_iou_2d

# Global parameters
DATA_ROOT = "/p/scratch/intertwin/datasets/virgo/test_data"
LOAD_DATASET = True
N_EPOCHS = 100
SAVE_CHECKPOINT = 'checkpoints/checkpoint_epoch_{}.pth'

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def generate_dataset():
    raise RuntimeError("generate_dataset should not be called!")
    df_aux_ts = generate_dataset_aux_channels(
        1000, 3, duration=16, sample_rate=500,
        num_waves_range=(20, 25), noise_amplitude=0.6
    )
    df_main_ts = generate_dataset_main_channel(
        df_aux_ts, weights=None, noise_amplitude=0.1
    )

    # Save datasets
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


def train_decoder(num_epochs, generator, criterion, optimizer, dataloader,
                  val_loader, accuracy, checkpoint_path, save_best=True):
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

        # Report training metrics of last epoch to Ray
        train.report({"loss": np.mean(val_loss),
                     "train_loss": np.mean(epoch_loss)})

        # all_val_losses_the_same = all(loss == val_loss_plot[0] for loss in val_loss_plot)
        # print(
        #     f"Were all recorded results of validation loss in epoch {epoch} the same? - {all_val_losses_the_same}")

        # all_training_losses_the_same = all(loss == loss_plot[0] for loss in loss_plot)
        # print(
        #     f"Were all recorded results of training loss in epoch {epoch} the same? - {all_training_losses_the_same}")

    return loss_plot, val_loss_plot, acc_plot, val_acc_plot


def main(config):
    g = set_seed(500)

    if not LOAD_DATASET:
        generate_dataset()

    file_path = os.path.join(DATA_ROOT, 'Image_dataset_synthetic_64x64.pkl')
    df = pd.read_pickle(file_path)
    df = df.applymap(lambda x: torch.tensor(x).float())

    main_channel = list(df.columns)[0]
    print(f'The main channel is: {main_channel}')
    aux_channels = list(df.columns)[1:]
    print(f"The AUX channels: {aux_channels}")

    df_main_all_2d = pd.DataFrame(df[main_channel])
    df_aux_all_2d = pd.DataFrame(df[aux_channels])

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

    print(f"Train set has size: {train_data_2d.size()}")
    print(f"Test set has size {test_data_2d.size()}")

    # Create dataloader objects with preprocessed dataset
    dataloader = DataLoader(
        train_data_2d,
        batch_size=int(config['batch_size']),
        shuffle=True,
        worker_init_fn=seed_worker,
        num_workers=2,
        generator=g
    )
    test_dataloader = DataLoader(
        test_data_2d,
        batch_size=int(config['batch_size']),
        shuffle=False,
        worker_init_fn=seed_worker,
        num_workers=2,
        generator=g
    )

    # Model to train
    generator_2d = Decoder(3, norm=False).to(device)
    init_weights(generator_2d, 'normal', scaling=.02, generator=g)

    # loss function, learning rate, and optimizer
    l2_loss = nn.MSELoss()  # this is l2!!!
    l1_loss = nn.L1Loss()  # this is L1!!!
    loss = l1_loss  # LogCoshLoss()
    G_optimizer = torch.optim.Adam(generator_2d.parameters(), lr=config['lr'])

    train_decoder(N_EPOCHS, generator_2d, loss, G_optimizer, dataloader,
                  test_dataloader, calculate_iou_2d, SAVE_CHECKPOINT.format('_best'))


def run_hpo(args):

    if not args.load_old_results:

        # Initialize Ray with cluster configuration from environment variables
        ray.init(
            address=os.environ["ip_head"],  
            _node_ip_address=os.environ["head_node_ip"],
        )

        # Define the search space for hyperparameters
        search_space = {
            'batch_size': tune.choice([64]),
            'lr': tune.uniform(1e-4, 1e-2)
        }

        # TuneConfig for configuring search algorithm and scheduler
        tune_config = tune.TuneConfig(
            metric=args.metric,  # Metric to optimize (loss by default)
            mode="min",  # Minimize the loss
            search_alg=args.search_alg,
            scheduler=args.scheduler,
            num_samples=args.num_samples  # Number of trials to run
        )

        # Ray's RunConfig for experiment name and stopping criteria
        run_config = train.RunConfig(
            name="Virgo-Ray-Experiment",
            stop={"training_iteration": args.max_iterations}
        )

        # Set resource allocation for each trial (number of GPUs and/or number of CPUs)
        resources_per_trial = {"gpu": args.ngpus}

        # Set up Ray Tune Tuner
        tuner = tune.Tuner(
            tune.with_resources(main, resources=resources_per_trial),
            tune_config=tune_config,
            run_config=run_config,
            param_space=search_space  # Search space defined above
        )

        # Run the hyperparameter optimization and get results
        result_grid = tuner.fit()

    else:
        # Load results from an earlier Ray Tune run
        print(f"Loading results from {args.experiment_path}...")

        # Restore tuner from saved results
        restored_tuner = tune.Tuner.restore(
            args.experiment_path, trainable=main)
        result_grid = restored_tuner.get_results()

    # Display experiment statistics
    print(f"Number of errored trials: {result_grid.num_errors}")
    print(f"Number of terminated trials: {result_grid.num_terminated}")
    print(f"Ray Tune experiment path: {result_grid.experiment_path}")

    # Get the best result based on the last 10 iterations' average
    best_result = result_grid.get_best_result(
        scope="last-10-avg", metric=args.metric, mode="min")
    print(f"Best result: {best_result}")

    # Print a dataframe with all trial results
    result_df = result_grid.get_dataframe()
    print(f"All results dataframe: {result_df}")
    print(f"All result columns: {result_df.columns}")

    # Plot the results for all trials
    plot_results(result_grid, metric=args.metric, filename="ray-loss-plot.png")
    plot_results(result_grid, metric="train_loss",
                 filename="ray-train_loss-plot.png")

    debug_weird_validation_loss(result_grid)
    debug_weird_validation_loss(result_grid, metric="train_loss")


# Function to plot the results for all trials
def plot_results(result_grid, metric="loss", filename="plot.png"):
    ax = None
    for result in result_grid:
        label = f"lr={result.config['lr']:.6f}, batch size={result.config['batch_size']}"
        if ax is None:
            ax = result.metrics_dataframe.plot(
                "training_iteration", metric, label=label)
        else:
            result.metrics_dataframe.plot(
                "training_iteration", metric, ax=ax, label=label)

    ax.set_title(
        f"{metric.capitalize()} vs. Training Iteration for All Trials")
    ax.set_ylabel(metric.capitalize())

    # Save the plot to a file
    plt.savefig(filename)

    # Show the plot
    plt.show()


def debug_weird_validation_loss(result_grid, metric="loss"):
    for i in range(len(result_grid)):
        result = result_grid[i]
        all_recorded_metrics = result.metrics_dataframe[metric].tolist()
        #print(f"Trial {i}: {all_recorded_metrics}")

        if all(metric == all_recorded_metrics[0] for metric in all_recorded_metrics):
            print(
                f"In trial {i}, all recorded results of metric {metric} were the same.")
            print(
                f"There were a total of {len(all_recorded_metrics)} losses recorded for this trial.")


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Hyperparameter Optimization with Ray Tune')
    parser.add_argument('--load_old_results', type=bool,
                        default=False,
                        help='Set this to true if you want to load results from an older ray run.')
    parser.add_argument('--experiment_path', type=str,
                        default='~/ray_results/Virgo-Ray-Experiment',
                        help='Directory where the results of the previous run are stored. Set this only if load_old_results is set to True. Defaults to ~/ray_results/Virgo-Ray-Experiment')
    parser.add_argument('--num-samples', type=int,
                        default=10, help='Number of trials to run')
    parser.add_argument('--ngpus', type=int, default=1,
                        help='Number of GPUs per trial')
    parser.add_argument('--metric', type=str, default='loss',
                        help='Metric to optimise.')
    parser.add_argument('--scheduler', type=str, default=None,
                        choices=['ASHA', 'FIFO'], help='Scheduler to use for tuning')
    parser.add_argument('--search_alg', type=str, default=None,
                        choices=['BayesOpt', 'HyperOpt'], help='Optimizer to use for tuning')
    parser.add_argument('--max-iterations', type=int,
                        default='20', help='...')
    args = parser.parse_args()  # Parse the command-line arguments

    # Check for available GPU
    if torch.cuda.is_available():
        device = 'cuda'
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        device = 'cpu'


    run_hpo(args)

