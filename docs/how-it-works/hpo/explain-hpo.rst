.. _explain_hpo:

Hyperparameter Optimization
============================

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
and you run them on the same amount of resources as you normally would for only one set of hyperparameters, your training would run four times as long (unless you use a scheduler to elimate some trials early).

Because of this, we want to design our HPO training wisely, so that we avoid unneseccary 
computational cost. These are some things that might help you when you are getting started with HPO.

HPO is beneficial when:

*    model performance is sensitive to hyperparameter settings.
*    you have access to sufficient computational resources.
*    good hyperparameter settings are difficult to determine manually.

To make sure you get the most out of your HPO training, the are some considerations you might want to make are to

*    define the search space wisely: Narrow the search space to plausible ranges to improve efficiency.
*    choose appropriate metrics: Use metrics aligned with your task's goals.
*    allocate your resources strategically: Balance the computational cost with the expected performance gains.
*    interpret your results carefully: Ensure that the improvements are meaningful and not due to overfitting.


Hyperparameter Optimization in itwinai
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that we know the key concepts behind HPO, we can explore how these are implemented in itwinai. 
We'll introduce distributed HPO, describe the architecture and operation of the ``RayTorchTrainer``,
and see that with the itwinai HPO integration, you can start optimising the hyperparameters of your 
models with very minimal changes to your existing itwinai pipeline.


Ray Overview
-------------

We use an open-source framework called Ray to facilitate distributed HPO. Ray provides two key 
components used in itwinai:

*    **Ray Train**: A module for distributed model training.
*    **Ray Tune**: A framework for hyperparameter optimization, supporting a variety of search algorithms and schedulers.

Ray uses its own cluster architecture to distribute training. A ray cluster consists of a group 
of nodes that work together to execute distributed tasks. Each node can contribute computational 
resources, and Ray schedules and manages these resources.

How a Ray Cluster Operates:

#.    **Node Roles**: A cluster includes a head node (orchestrator) and worker nodes (executors). 
#.     **Task Scheduling**: Ray automatically schedules trials across nodes based on available resources.
#.     **Shared State**: Nodes share data such as checkpoints and trial results via a central storage path.

We launch a ray cluster using a dedicated slurm job script. You may refer to `this script <https://github.com/interTwin-eu/itwinai/blob/main/tutorials/hpo-workflows/slurm_hpo.sh>`_ It should be suitable for almost any 
time you wish to run an itwinai pipeline with Ray, the only thing you may have to change is the ``#SBATCH`` directives to set the proper resource requirements. 
Also refer to the `ray documentation <https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html>`_ 
on this topic, if you want to learn more about how to launch a ray cluster with slurm.


How Distributed Training Works with the RayTorchTrainer
--------------------------------------------------------

The ``RayTorchTrainer`` combines components from **Ray Train** and **Ray Tune**, enabling 
distributed HPO to run within your pipeline while maintaining compatibility with other itwinai features. 
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
:doc:`HPO tutorial <../../tutorials/hpo-workflows/hpo-workflows>`.
