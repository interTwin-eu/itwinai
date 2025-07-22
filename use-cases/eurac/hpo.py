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
import logging
import os
from typing import Dict

import matplotlib.pyplot as plt
import ray
import torch
from hydra import compose, initialize
from ray import tune

from itwinai.cli import exec_pipeline_with_compose

py_logger = logging.getLogger(__name__)

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
                f"batch_size={config['batch_size']}",
                f"optim_lr={config['optim_lr']}",
                f"+pipe_key={pipe_key}",
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
        ray.init(
            address=os.environ["ip_head"],
            _node_ip_address=os.environ["head_node_ip"],
        )

        # Define the search space for hyperparameters
        search_space = {
            "batch_size": tune.choice([64, 128, 256]),
            "lr": tune.uniform(1e-5, 1e-3),
        }

        # TuneConfig for configuring search algorithm and scheduler
        tune_config = tune.TuneConfig(
            metric=args.metric,  # Metric to optimize (loss by default)
            mode="min",  # Minimize the loss
            num_samples=args.num_samples,  # Number of trials to run
        )

        # Ray's RunConfig for experiment name and stopping criteria
        run_config = tune.RunConfig(
            name="Eurac-Ray-Experiment",
            stop={"training_iteration": args.max_iterations},
        )

        # Determine GPU and CPU utilization per trial
        # We are allocating all available ressources per node evenly across trials
        ngpus_per_trial = max(1, args.ngpus // args.num_samples)
        ncpus_per_trial = max(1, args.ncpus // args.num_samples)

        # Set up Ray Tune Tuner with resources and parameters
        resources_per_trial = {"gpu": ngpus_per_trial, "cpu": ncpus_per_trial}
        trainable_with_resources = tune.with_resources(
            run_trial, resources=resources_per_trial
        )

        data = {"pipeline_name": args.pipeline_name}
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
        py_logger.info(f"Loading results from {args.experiment_path}...")

        # Restore tuner from saved results
        restored_tuner = tune.Tuner.restore(args.experiment_path, trainable=run_trial)
        result_grid = restored_tuner.get_results()

    # Display experiment statistics
    py_logger.info(f"Number of errored trials: {result_grid.num_errors}")
    py_logger.info(f"Number of terminated trials: {result_grid.num_terminated}")
    py_logger.info(f"Ray Tune experiment path: {result_grid.experiment_path}")

    # Get the best result based on the last 10 iterations' average
    best_result = result_grid.get_best_result(
        scope="last-10-avg", metric=args.metric, mode="min"
    )
    py_logger.info(f"Best result: {best_result}")

    # Print a dataframe with all trial results
    result_df = result_grid.get_dataframe()
    py_logger.info(f"All results dataframe: {result_df}")
    py_logger.info(f"All result columns: {result_df.columns}")

    # Plot the results for all trials
    plot_results(result_grid, metric=args.metric, filename="ray-loss-plot.png")
    plot_results(result_grid, metric="train_loss", filename="ray-train_loss-plot.png")


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
    parser.add_argument("--ngpus", type=int, help="Number of GPUs available on node.")
    parser.add_argument("--ncpus", type=int, help="Number of CPUs available on node.")
    parser.add_argument("--metric", type=str, default="loss", help="Metric to optimise.")
    parser.add_argument(
        "--max_iterations", type=int, default="20", help="Maximum iterations per trial"
    )

    args = parser.parse_args()  # Parse the command-line arguments

    # Check for available GPU
    if torch.cuda.is_available():
        device = "cuda"
        py_logger.info(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        device = "cpu"
        py_logger.info("Using CPU")

    run_hpo(args)
