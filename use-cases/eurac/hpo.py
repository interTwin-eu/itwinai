import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import ray
import torch
from ray import train, tune

from itwinai.parser import ConfigParser

# Global variable for data root directory - this is the synthetic Virgo test data,
# which can generally be used so that new data does not need to be generated for every run
DATA_ROOT = "/p/scratch/intertwin/datasets/virgo/test_data"


def run_trial(config):
    """
    Execute a single trial using the given configuration (config).
    This runs a full training pipeline - you can also specify a pipeline as a dictionary, 
    e.g. if you only want to run certain parts without changing your config.yaml file (see below).

    Args:
    - config: Dictionary with hyperparameters (e.g., 'batch_size', 'lr').

    Example to run with a manual pipeline:

    my_pipeline = Pipeline(
        [
            TimeSeriesDatasetSplitter(
                train_proportion=0.9,
                root_folder="/p/scratch/intertwin/datasets/virgo"
            ),
            TimeSeriesProcessor(),
            NoiseGeneratorTrainer(
                config=config,
                num_epochs=4,
                strategy=None,
                checkpoint_path='checkpoints/checkpoint_epoch_{}.pth',
                validation_every=20
            )
        ]
    )
    """

    # Passing a seed to TimeSeriesDatasetSplitter and NoiseGeneratorTrainer
    # will make runs uniform across trials
    # (reducing the variablility to the hyperparameter settings)

    # Note: Comment out the TimeSeriesDatasetGenerator class and the
    # WandBLogger in the config.yaml file to make it run on hdfml and pre-generated dataset
    parser = ConfigParser(
        config=Path('config.yaml'),
        override_keys={
            'batch_size': config['batch_size'],
            'learning_rate': config['lr']
        }
    )
    my_pipeline = parser.parse_pipeline(
        pipeline_nested_key='training_pipeline',
        verbose=False
    )

    my_pipeline.execute()


def run_hpo(args):
    """
    Run hyperparameter optimization using Ray Tune.
    Either starts a new optimization run or resumes from previous results.

    Args:
    - args: Command-line arguments parsed by argparse.
    """
    if not args.load_old_results:

        # Initialize Ray with cluster configuration from environment variables
        ray.init(
            address=os.environ["ip_head"],
            _node_ip_address=os.environ["head_node_ip"],
        )

        # Define the search space for hyperparameters
        search_space = {
            'batch_size': tune.choice([64, 128, 256]),
            'lr': tune.uniform(1e-5, 1e-3)
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
            name="Eurac-Ray-Experiment",
            stop={"training_iteration": args.max_iterations}
        )

        # Set resource allocation for each trial (number of GPUs and/or number of CPUs)
        resources_per_trial = {"gpu": args.ngpus}

        # Set up Ray Tune Tuner
        tuner = tune.Tuner(
            tune.with_resources(run_trial, resources=resources_per_trial),
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
            args.experiment_path,
            trainable=run_trial
        )
        result_grid = restored_tuner.get_results()

    # Display experiment statistics
    print(f"Number of errored trials: {result_grid.num_errors}")
    print(f"Number of terminated trials: {result_grid.num_terminated}")
    print(f"Ray Tune experiment path: {result_grid.experiment_path}")

    # Get the best result based on the last 10 iterations' average
    best_result = result_grid.get_best_result(
        scope="last-10-avg",
        metric=args.metric,
        mode="min"
    )
    print(f"Best result: {best_result}")

    # Print a dataframe with all trial results
    result_df = result_grid.get_dataframe()
    print(f"All results dataframe: {result_df}")
    print(f"All result columns: {result_df.columns}")

    # Plot the results for all trials
    plot_results(result_grid, metric=args.metric, filename="ray-loss-plot.png")
    plot_results(result_grid, metric="train_loss",
                 filename="ray-train_loss-plot.png")


def plot_results(result_grid, metric="loss", filename="plot.png"):
    """
    Plot the results for all trials and save the plot to a file.

    Args:
    - result_grid: Results from Ray Tune trials.
    - metric: The metric to plot (e.g., 'loss').
    - filename: Name of the file to save the plot.
    """
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


# Main entry point for script execution
if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Hyperparameter Optimization with Ray Tune')
    parser.add_argument(
        '--load_old_results', type=bool,
        default=False,
        help='Set this to true if you want to load results from an older ray run.')
    parser.add_argument(
        '--experiment_path', type=str,
        default='~/ray_results/Virgo-Ray-Experiment',
        help='Directory where the results of the previous run are stored. \
        Set this only if load_old_results is set to True. \
        Defaults to ~/ray_results/Virgo-Ray-Experiment')
    parser.add_argument(
        '--num_samples', type=int,
        default=10, help='Number of trials to run')
    parser.add_argument(
        '--ngpus', type=int, default=1,
        help='Number of GPUs per trial')
    parser.add_argument(
        '--metric', type=str, default='loss',
        help='Metric to optimise.')
    parser.add_argument(
        '--scheduler', type=str, default=None,
        choices=['ASHA', 'FIFO'], help='Scheduler to use for tuning')
    parser.add_argument(
        '--search_alg', type=str, default=None,
        choices=['BayesOpt', 'HyperOpt'], help='Optimizer to use for tuning')
    parser.add_argument(
        '--max_iterations', type=int,
        default='20', help='Maximum iterations per trial')

    args = parser.parse_args()  # Parse the command-line arguments

    # Check for available GPU
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        device = 'cpu'
        print("Using CPU")

    run_hpo(args)
