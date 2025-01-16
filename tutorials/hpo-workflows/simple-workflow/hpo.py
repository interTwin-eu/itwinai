# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Anna Lappe
#
# Credit:
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import argparse
from typing import Dict

import ray
import torch
from ray import train, tune

from itwinai.parser import ConfigParser


def run_trial(config: Dict, data: Dict):
    """Execute a single trial using the given configuration (config).
    This runs a full training pipeline - you can also specify a pipeline as a dictionary,
    e.g. if you only want to run certain parts without changing your config.yaml file
    (see below).

    Args:
        config (dict): A dictionary containing hyperparameters, such as:
            - 'batch_size' (int): The size of the batch for training.
            - 'lr' (float): The learning rate for the optimizer.
        data (dict): A dictionary containing a "pipeline_name" field,
            which specifies the training pipeline to be used.
            Must be defined in a file called "config.yaml"

    You can also run a manual pipeline by directly creating it with imported classes:

    Example:
        >>> my_pipeline = Pipeline(
        >>>    [
        >>>        FashionMNISTGetter(),
        >>>        FashionMNISTSplitter(train_proportion=0.9, validation_proportion=0.1),
        >>>        MyTrainer(
        >>>             config=config,
        >>>             epochs=10,
        >>>             strategy="ddp",
        >>>        ),
        >>>    ]
        >>> )

        >>> my_pipeline.execute()
        ```
    """

    pipeline_name = data["pipeline_name"]
    parser = ConfigParser(
        config="config.yaml",
        override_keys={
            # Set hyperparameters controlled by ray
            "batch_size": config["batch_size"],
            "optim_lr": config["optim_lr"],
        },
    )
    my_pipeline = parser.parse_pipeline(pipeline_nested_key=pipeline_name, verbose=False)

    my_pipeline.execute()


def run_hpo(args):
    """Run hyperparameter optimization using Ray Tune.
    Either starts a new optimization run or resumes from previous results.

    Args:
    - args: Command-line arguments parsed by argparse.
    """
    if not args.load_old_results:
        # Initialize Ray with cluster configuration from environment variables
        ray.init(address="auto")

        # Define the search space for hyperparameters
        search_space = {
            "batch_size": tune.choice([3, 4, 5, 6]),
            "optim_lr": tune.uniform(1e-5, 1e-3),
        }

        tune_config = tune.TuneConfig(
            metric=args.metric,  # Metric to optimize (loss by default)
            mode="min",  # Minimize the loss
            num_samples=args.num_samples,  # Number of trials to run
        )

        run_config = train.RunConfig(
            name="Virgo-Ray-Basic-Experiment",
            stop={"training_iteration": args.max_iterations},
        )

        # Determine GPU and CPU utilization per trial
        # We are allocating all available resources per node evenly across trials
        ngpus_per_trial = max(1, args.ngpus // args.num_samples)
        ncpus_per_trial = max(1, args.ncpus // args.num_samples)

        # Set resource allocation for each trial (number of GPUs and/or number of CPUs)
        resources_per_trial = {"gpu": ngpus_per_trial, "cpu": ncpus_per_trial}
        run_with_resources = tune.with_resources(run_trial, resources=resources_per_trial)

        data = {"pipeline_name": args.pipeline_name}
        trainable_with_parameters = tune.with_parameters(run_with_resources, data=data)

        # Set up Ray Tune Tuner
        tuner = tune.Tuner(
            trainable_with_parameters,
            tune_config=tune_config,
            run_config=run_config,
            param_space=search_space,  # Search space defined above
        )

        # Run the hyperparameter optimization and get results
        result_grid = tuner.fit()

    else:
        # Load results from an earlier Ray Tune run
        print(f"Loading results from {args.experiment_path}...")

        # Restore tuner from saved results
        restored_tuner = tune.Tuner.restore(args.experiment_path, trainable=run_trial)
        result_grid = restored_tuner.get_results()

    # Print a dataframe with all trial results
    result_df = result_grid.get_dataframe()
    print(f"All results dataframe: {result_df}")
    print(f"All result columns: {result_df.columns}")


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
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        device = "cpu"
        print("Using CPU")

    run_hpo(args)
