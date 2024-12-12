.. _explain_hpo:

Hyperparameter Optimization
============================

**Author(s)**: Anna Lappe (CERN)

Hyperparameter optimization (HPO) is a core technique for improving machine learning model 
performance. This page introduces the concepts behind HPO, covering key elements like 
hyperparameters, search algorithms, schedulers, and more. 
It also outlines the benefits and drawbacks of HPO, helping you make informed decisions when 
applying it with itwinai. 


Key Concepts
-------------

**Hyperparameters** are parameters in a machine learning model or training process that are set 
before training begins. They are not learned from the data but can have a significant impact 
on the model's performance and training efficiency. Examples of hyperparameters include:

*    Learning rate
*    Batch size
*    Number of layers in a neural network
*    Regularization coefficients (e.g., L2 penalty)

HPO is the process of systematically searching for the optimal set of hyperparameters to 
maximize model performance on a given task.

The **search space** defines the range of values each hyperparameter can take. It may include 
discrete values (e.g., [32, 64, 128]) or continuous ranges (e.g., learning rate from 1e-5 to 1e-1).

**Search algorithms** explore the hyperparameter search space to identify the best configuration. 
Common approaches include:

*    Grid Search: Exhaustive search over all combinations of hyperparameter values.
*    Random Search: Randomly samples combinations from the search space.
*    Bayesian Optimization: Uses a probabilistic surrogate model to learn the shape of the search space and predict promising configurations.
*    Evolutionary Algorithms: Use search algorithms based on evolutionary concepts to evolve hyperparameter configurations, e.g. genetic programming.

**Schedulers** manage the allocation of computational resources across multiple hyperparameter 
configurations. They help prioritize promising configurations and terminate less effective 
ones early to save resources. 
Examples include:

*    ASHA (Asynchronous Successive Halving Algorithm): Allocates resources by successively discarding the lowest-performing hyperparameter combinations.
*    Median Stopping Rule: Stops trials that perform below the median of completed trials.

The **evaluation metric** determines the performance of a model on a validation set. 
Common metrics include accuracy, F1 score, and mean squared error. 
The choice of metric depends on the task and its goals.

A **trial** is the evaluation of one set of hyperparameters. Depending on whether you are 
using a scheduler, this could be the entire training run, so as many epochs as you 
have specified, or it could be terminated early and thus run for fewer epochs.


When to Use HPO and Key Considerations
---------------------------------------

HPO can significantly enhance a model's predictive accuracy and generalization to unseen data 
by finding the best hyperparameter settings.
However, there are some drawbacks, especially with regards to computational cost and resource 
management. Especially distributed HPO requires careful planning of computational resources 
to avoid bottlenecks or excessive costs. You should consider that if you want to run four different trials, 
and you run them on the same amount of resources as you normally would, your training would run four times as long.

Because of this, we want to design our HPO training wisely, so that we avoid unneseccary 
computational cost. These are some things that might help you when you are getting started with HPO.

HPO is beneficial when:

*    model performance is sensitive to hyperparameter settings.
*    you have access to sufficient computational resources.
*    good hyperparameter settings are difficult to determine manually.

To make sure you get the most out of your HPO training, the are some considerations you might want to make are to

*    define the search space wisely: Narrow the search space to plausible ranges to improve efficiency.
*    choose appropriate metrics: Use metrics aligned with your task's goals. With some search algorithms it is also possible to do multi-objective optimization, e.g. if you want to track both accuracy and L1 loss - just make sure that the metric that you track conforms with your search algorithm.
*    allocate your resources strategically: Balance the computational cost with the expected performance gains. This is what the scheduler is for, and it is generally always a good idea to use one, unless you expect your objective function to be extremely heterogenous, i.e. that the performance of a hyperparameter configuration on the first (for example) ten epochs is not a good indicator for its future performance at all. You might also have experience in training your model and want to account for additional behaviors  - for this there are additional parameters you may set, such as a grace period (a minimum number of iterations a configuration is allowed to run).


Hyperparameter Optimization in itwinai
---------------------------------------

Now that we know the key concepts behind HPO, we can explore how these are implemented in itwinai. 
itwinai provides two ways of adding HPO to your machine learning training. If you already have an itwinai trainer and pipeline set up, then the first and easier approach is to use our simple template function to wrap the assembling and executing of your pipeline. This function is called for each trial - so the pipeline stays almost exactly as-is, we just add hyperparameters to tune and we are ready to go.
The second method uses the itwinai ``RayTorchTrainer``, an alternative to the ``TorchTrainer``. This trainer has HPO functionalities already built-in, which means no extra scripts, just replace the trainer in your pipeline and run it as you normally would with any itwinai pipeline.
In the next section we'll introduce distributed HPO, and discover how we can easily start optimizing hyperparameters 
in our exisiting itwinai pipeline with just a few lines of code. We will then describe the 
architecture and operation of the ``RayTorchTrainer`` and talk about what to consider when choosing the best HPO integration for you.

Ray Overview
^^^^^^^^^^^^^

We use an open-source framework called Ray to facilitate distributed HPO. Ray provides two key 
components used in itwinai:

*    **Ray Train**: A module for distributed model training.
*    **Ray Tune**: A framework for hyperparameter optimization, supporting a variety of search algorithms and schedulers.

Ray uses its own cluster architecture to distribute training. A ray cluster consists of a group 
of nodes that work together to execute distributed tasks. Each node can contribute computational 
resources, and Ray schedules and manages these resources.

How a Ray Cluster Operates:

#.    **Node Roles**: A cluster includes a head node (orchestrator) and worker nodes (executors). 
#.    **Task Scheduling**: Ray automatically schedules trials across nodes based on available resources.
#.    **Shared State**: Nodes share data such as checkpoints and trial results via a central storage path.

We launch a ray cluster using a dedicated slurm job script. You may refer to `this script <https://github.com/interTwin-eu/itwinai/blob/main/tutorials/hpo-workflows/slurm_hpo.sh>`_.
It should be suitable for almost any 
time you wish to run an itwinai pipeline with Ray, the only thing you may have to change is the ``#SBATCH`` directives to set the proper resource requirements.
We use this script to launch both of our HPO integrations, changing only the final command, depending on which script we want to execute once our ray cluster is set up.
Also refer to the `ray documentation <https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html>`_ 
on this topic, if you want to learn more about how to launch a ray cluster with slurm.


How to Run Your Pipeline with Ray Tune
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The easiest way to start running HPO with itwinai is to use our template to wrap a 
pipeline in a simple function to pass it to a Ray Tune ``Tuner``. This method is suitable for users who want a quick, lightweight setup -
if you already have an itwinai trainer and pipeline, setting up this integration should not take you
much more than a few minutes. This method uses only Ray Tune for trial distribution and hyperparameter sampling, 
and does not distribute the trials themselves. 
If you are new to HPO or working with a relatively small model and dataset, it is recommended that you start with this integration. 
Advanced users with distributed training requirements can skip ahead to the `distributed method`_.

**How It Works**

You can set up this integration by wrapping your existing itwinai trainer and pipeline in a function
that Ray Tune can call for each trial. 
Here's a summary:

#.   **Define the search space**: Specify the hyperparameters and their possible values or ranges.
#.   **Wrap the pipeline in a trial function**: Use the provided ``run_trial`` function as a template to adapt to your pipeline.
#.   **Set up the tuner**: Configure the ray tune ``Tuner`` to manage trials, allocate resources, and evaluate results.

Refer to the
:doc:`tutorial <../../tutorials/hpo-workflows/hpo-basic-integration>` on getting started with hyperparameter optimization in itwinai
for the quick-start integration.
The following section explains the more advanced distributed integration for users with multi-node 
setups and higher computational requirements.


.. _distributed method:

How to Run Distributed HPO with the RayTorchTrainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``RayTorchTrainer`` combines components from **Ray Train** and **Ray Tune**, providing a more advanced approach leveraging them together
for fully distributed HPO. This method is suitable for larger-scale experiments requiring 
optimized resource utilization across multiple nodes in a cluster.
Because it implements the same interface as the itwinai ``TorchTrainer``, you can easily
replace the itwinai ``TorchTrainer`` with the ``RayTorchTrainer`` in your pipeline with only a few modifications. 
The key features of this trainer are:

#.    **Compatibility**: Use all itwinai components—loggers, data getters, splitters, and so on, with the ``RayTorchTrainer``.
#.    **Flexibility**: Distributed HPO works with various search algorithms and schedulers supported by Ray Tune.
#.    **Minimal Code Changes**: Replace the ``TorchTrainer`` with the ``RayTorchTrainer`` with very minimal code changes and you're ready to run HPO.

In the ``TorchTrainer``, initialization tasks (e.g., model creation, logger setup) are done 
outside of the ``train()`` function. However, in the ``RayTorchTrainer``, this logic must be 
moved inside ``train()`` because Ray executes only the ``train()`` function for each trial independently, so allocation of trial resources is done only once ``train()`` is called.
Furthermore distribution frameworks, such as DDP or DeepSpeed, are agnostic of the other trials, so they should be initialized only once the trial resources are allocated.

For a hands-on tutorial for how to change your existing itwinai pipeline code to additionally 
run HPO, or how to set up an HPO integration with itwinai from scratch, have a look at the 
:doc:`distributed HPO tutorial <../../tutorials/hpo-workflows/hpo-torchtrainer-integration>`.
