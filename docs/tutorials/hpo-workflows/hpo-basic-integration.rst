.. _hpo_basic_workflow:

Getting Started with Hyperparameter Optimization in itwinai
=============================================================

**Author(s)**: Anna Lappe (CERN)

This tutorial will walk you through setting up and running hyperparameter optimization (HPO)
for your itwinai pipeline using Ray Tune. By the end, you'll be able to customize the provided
template code for your own pipeline and start optimizing your models.
You can find the code for the tutorial `on Github <https://github.com/interTwin-eu/itwinai/blob/main/tutorials/hpo-workflows/simple-workflow>`_.
It includes a complete example of running a simple custom itwinai ``TorchTrainer`` on the MNIST dataset.


Here we will go through the ``hpo.py`` template step-by-step to help you understand the workflow
and guide you in customizing it for your specific needs. That said, the script is designed 
to be general-purpose, so with minimal adjustments (just one or two lines), you should be able 
to run your own training pipeline by using the
`slurm script <https://github.com/interTwin-eu/itwinai/blob/main/tutorials/hpo-workflows/simple-workflow/slurm_hpo.sh>`_ and the 
`HPO template <https://github.com/interTwin-eu/itwinai/blob/main/tutorials/hpo-workflows/simple-workflow/hpo.py>`_  provided in the tutorial,.

**Prerequisites**

Before getting started, ensure you have:

#.  A basic understanding of itwinai pipelines and trainers. If not, refer to the tutorials on :doc:`workflows <../../tutorials/tutorials>`.
#.  A YAML file defining your pipeline.
#.  All necessary components implemented (e.g., data getter, loader, trainer, etc.).

Step 1: Import the Required Libraries:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Let's start by importing all the libraries needed for this script.

.. code-block:: python 

    import argparse
    from typing import Dict

    import torch
    from ray import train, tune

    from itwinai.parser import ConfigParser


Step 2: Define the Trial Execution Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``run_trial`` function executes a single HPO trial using the provided configuration and data.
``config`` contains the hyperparameters (e.g., learning rate, batch size). ``data`` is a general 
dictionary that can contain any values you need to pass to the function. In our template, 
it includes only the pipeline name defined in your config.yaml, but you may pass additional 
values here. 

**Note:** Keep the size of the data dictionary small because it is serialized by Ray Tune 
when sent to workers. Avoid including large objects like datasets here. 

.. code-block:: python

    def run_trial(config: Dict, data: Dict):
        pipeline_name = data["pipeline_name"]
        parser = ConfigParser(
            config="config.yaml",
            override_keys={
                # Set hyperparameters controlled by ray
                "batch_size": config["batch_size"],
                "optim_lr": config["optim_lr"]
            },
        )
        my_pipeline = parser.parse_pipeline(
            pipeline_nested_key=pipeline_name, verbose=False
        )

        my_pipeline.execute()

**Adaptation**: 

*    If neccessary, replace ``config.yaml`` with the path to your own configuration file.
*    Adjust hyperparameter keys (e.g., ``batch_size``, ``learning_rate``) to match the keys used in your pipeline.
*    If your pipeline requires additional configurations, add them to the ``override_keys`` dictionary.

For a detailed explanation of defining and parsing pipelines, refer to the :doc:`Workflow Page <../../how-it-works/workflows/explain_workflows>`.


.. _Step 3:

Step 3: Configure and Run HPO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``run_hpo`` function is the core of this script. It sets up Ray Tune to manage the hyperparameter optimization process.

**1.  Define the search space:** 
Specify the range of hyperparameters to explore. In our example:

.. code-block:: python

    search_space = {
            "batch_size": tune.choice([3, 4, 5, 6]),
            "optim_lr": tune.uniform(1e-5, 1e-3),
        }

**Adaptation**: 

*    Add or remove hyperparameters to suit your model.
*    Use ``tune.choice`` for discrete parameters and ``tune.uniform`` for continuous, uniform ranges. You can add different sampling distributions and types. For additional search space options, please refer to the `Ray Tune documentation <https://docs.ray.io/en/latest/tune/api/search_space.html>`_. Any search space definition supported by Ray Tune can be used in this template as well.

**2.  Define ray configs:** 
Set up the tuning and run configurations to control the number of trials, optimization goal, and stopping criteria.

.. code-block:: python

    tune_config = tune.TuneConfig(
                metric=args.metric,  # Metric to optimize (loss by default)
                mode="min",  # Minimize the loss
                num_samples=args.num_samples,  # Number of trials to run
            )

            run_config = train.RunConfig(
                name="Virgo-Ray-Experiment", 
                stop={"training_iteration": args.max_iterations}
            )

**Adaptation**: 

*    Replace ``args.metric`` with the metric relevant to your optimization task. This metric has to be reported at the end of each iteration (by default this is one epoch) in your trainer. We explain how to do this in `Step 4`_.
*    If needed, replace the stopping criteria defined in ``stop``. If set lower than your trainer's epochs, trials stop early.


**3.  Allocate Resources:** 
Assign GPUs and CPUs for each trial. In this example we dynamically allocate resources based on the total available number of GPUs and CPUs.

.. code-block:: python

    ngpus_per_trial = max(1, args.ngpus // args.num_samples)
    ncpus_per_trial = max(1, args.ncpus // args.num_samples)

    # Set resource allocation for each trial (number of GPUs and/or number of CPUs)
    resources_per_trial = {"gpu": ngpus_per_trial, "cpu": ncpus_per_trial}
    run_with_resources = tune.with_resources(run_trial, resources=resources_per_trial)

**Adaptation**:

*    If not using GPUs, set ``ngpus_per_trial`` to 0


**4.  Set Up and Execute the Tuner:** 
Combine the trial function, configurations, and search space into a ``Tuner`` object, and run it.

.. code-block:: python

    data = {"pipeline_name": args.pipeline_name}
    trainable_with_parameters = tune.with_parameters(run_with_resources, data=data)

    # Set up Ray Tune Tuner
    tuner = tune.Tuner(
        trainable_with_parameters,
        tune_config=tune_config,
        run_config=run_config,
        param_space=search_space,  # Search space defined above
    )

    result_grid = tuner.fit()

Finally, we can call our function:

.. code-block:: python

    # Main entry point for script execution
    if __name__ == "__main__":
        # Parse command-line arguments
        ...

        # Check for available GPU
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            device = "cpu"
            print("Using CPU")

        run_hpo(args)


.. _Step 4:

Step 4: Add Reporting Call to Your Trainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To manage trials, i.e. know when to stop bad performing ones and where to search next for good 
hyperparameter configurations, Ray Tune needs to know how our model is doing during training.  
For this, we add the following to report metrics at the end of each training iteration:

.. code-block:: python

    class MyItwinaiTrainer(TorchTrainer):
        ...

        def train():
            for epoch in epochs:
            ...

            # Report training metrics of last epoch to Ray
            train.report({"loss": epoch_val_loss})

It is important that this metric is the same that you specify when setting up your Tune Config, as described in `Step 3`_.

**Running the Script**:  
Once you have your HPO script and your trainer set up, you can launch your training by executing the 
`slurm script <https://github.com/interTwin-eu/itwinai/blob/main/tutorials/hpo-workflows/simple-workflow/slurm_hpo.sh>`_:

.. code-block:: bash

    cd tutorials/hpo-workflows/simple-workflow
    sbatch slurm_hpo.sh

This script launches a ray cluster and executes the ``hpo.py`` script. 
For more details, see the :doc:`HPO introduction <../../how-it-works/hpo/explain-hpo>`.