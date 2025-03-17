# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Anna Lappe
#
# Credit:
# - Anna Lappe <anna.lappe@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import argparse
from typing import Dict

import matplotlib.pyplot as plt
import ray
import torch
from hydra import compose, initialize
from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler

from itwinai.cli import exec_pipeline_with_compose


def run_trial(config: Dict, data: Dict):
    """Execute a single trial using the given configuration (config).
    This runs a full training pipeline - you can also specify a pipeline as a dictionary,
    e.g. if you only want to run certain parts without changing your config.yaml file
    (see below).

    Args:
        config (dict): A dictionary containing hyperparameters, such as:
            - 'batch_size' (int): The size of the batch for training.
            - 'lr' (float): The learning rate for the optimizer.
        data (dict): A dictionary containing a "pipeline_path" field, which points to the yaml
            file containing the pipeline definition
    """
    pipe_key = data["pipeline_name"]

    with initialize():
        cfg = compose(
            "config.yaml",
            overrides=[
                f"batch={config['batch']}",
                f"learning_rate={config['learning_rate']}",
                f"hidden_size={config['hidden_size']}",
                f"dropout={config['dropout']}",
                f"interval_value={config['interval_value']}",
                f"+pipe_key={pipe_key}",
                f"work_dir={args.workdir}",
                f"dataset_root={args.dataset_path}",
            ],
        )
        exec_pipeline_with_compose(cfg)


def run_hpo(args):
    """Run hyperparameter optimization using Ray Tune.
    Either starts a new optimization run or resumes from previous results.

    Args:
    - args: Command-line arguments parsed by argparse.
    """
    if not args.load_old_results:
        # Initialize Ray with cluster configuration from environment variables
        ray.init()

        # Define the search space for hyperparameters
        search_space = {
            "batch": tune.qrandint(128, 1024, 128),
            "learning_rate": tune.uniform(1e-4, 1e-2),
            "hidden_size": tune.randint(16, 512),
            "dropout": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
            "seq_length": tune.qrandint(90, 365, 30),
            "interval_value": tune.choice([3, 5, 10]),
        }

        # TuneConfig for configuring search algorithm and scheduler
        tune_config = tune.TuneConfig(
            metric=args.metric,  # Metric to optimize (loss by default)
            mode="min",  # Minimize the loss
            num_samples=args.num_samples,  # Number of trials to run
            scheduler=AsyncHyperBandScheduler(
                max_t=args.max_iterations, grace_period=args.grace_period, reduction_factor=4
            ),
        )

        # Ray's RunConfig for experiment name and stopping criteria
        run_config = train.RunConfig(stop={"training_iteration": args.max_iterations})

        # Set up Ray Tune Tuner with resources and parameters
        resources_per_trial = {"gpu": args.ngpus, "cpu": args.ncpus}
        trainable_with_resources = tune.with_resources(
            run_trial, resources=resources_per_trial
        )

        data = {
            "pipeline_name": args.pipeline_name,
            "dataset_path": args.dataset_path,
            "workdir": args.workdir,
        }
        trainable_with_parameters = tune.with_parameters(trainable_with_resources, data=data)

        tuner = tune.Tuner(
            trainable_with_parameters,
            tune_config=tune_config,
            run_config=run_config,
            param_space=search_space,
        )

        # Run the hyperparameter optimization and get results
        result_grid = tuner.fit()

    else:
        # Load results from an earlier Ray Tune run
        print(f"Loading results from {args.experiment_path}...")

        # Restore tuner from saved results
        restored_tuner = tune.Tuner.restore(args.experiment_path, trainable=run_trial)
        result_grid = restored_tuner.get_results()

    # Display experiment statistics
    print(f"Number of errored trials: {result_grid.num_errors}")
    print(f"Number of terminated trials: {result_grid.num_terminated}")
    print(f"Ray Tune experiment path: {result_grid.experiment_path}")

    # Get the best result based on the last 10 iterations' average
    best_result = result_grid.get_best_result(
        scope="last-10-avg", metric=args.metric, mode="min"
    )
    print(f"Best result: {best_result}")

    # Print a dataframe with all trial results
    result_df = result_grid.get_dataframe()
    print(f"All results dataframe: {result_df}")
    print(f"All result columns: {result_df.columns}")

    # Plot the results for all trials
    # plot_results(result_grid, metric=args.metric, filename="ray-loss-plot.png")
    # plot_results(result_grid, metric="train_loss", filename="ray-train_loss-plot.png")


def plot_results(result_grid, metric="loss", filename="plot.png"):
    """Plot the results for all trials and save the plot to a file.

    Args:
    - result_grid: Results from Ray Tune trials.
    - metric: The metric to plot (e.g., 'loss').
    - filename: Name of the file to save the plot.
    """
    ax = None
    for result in result_grid:
        label = f"lr={result.config['lr']:.6f}, batch size={result.config['batch_size']}"
        if ax is None:
            ax = result.metrics_dataframe.plot("training_iteration", metric, label=label)
        else:
            result.metrics_dataframe.plot("training_iteration", metric, ax=ax, label=label)

    ax.set_title(f"{metric.capitalize()} vs. Training Iteration for All Trials")
    ax.set_ylabel(metric.capitalize())

    plt.savefig(filename)

    # Show the plot
    plt.show()


# Main entry point for script execution
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization with Ray Tune")
    parser.add_argument(
        "--load_old_results",
        type=bool,
        default=False,
        help="Set this to true if you want to load results from an older ray run.",
    )
    parser.add_argument(
        "--pipeline_name",
        type=str,
        default="training_pipeline",
        help='Name of the training pipeline to be used. \
            This pipeline has to be defined in a file called "config.yaml". \
            Defaults to "training_pipeline"',
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        default="~/ray_results/Eurac-Ray-Experiment",
        help="Directory where the results of the previous run are stored. \
        Set this only if load_old_results is set to True. \
        Defaults to ~/ray_results/Eurac-Ray-Experiment",
    )
    parser.add_argument("--num_samples", type=int, default=10, help="Number of trials to run")
    parser.add_argument("--ngpus", type=int, help="Number of GPUs per trial")
    parser.add_argument("--ncpus", type=int, help="Number of CPUs per trial")
    parser.add_argument("--metric", type=str, default="loss", help="Metric to optimise.")
    parser.add_argument(
        "--max_iterations", type=int, default=20, help="Maximum iterations per trial"
    )
    parser.add_argument(
        "--grace_period", type=int, default=10, help="Grace period for AsyncHyperBandScheduler"
    )
    parser.add_argument(
        "--dataset_path", default="/dataset", type=str, help="Path to the EURAC dataset"
    )
    parser.add_argument("--workdir", default="/app", type=str, help="Path to the workdir")

    args = parser.parse_args()  # Parse the command-line arguments

    # Check for available GPU
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        device = "cpu"
        print("Using CPU")

    run_hpo(args)
