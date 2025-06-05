.. _hpo_torchtrainer_workflow:

Hyperparameter Optimization with TorchTrainer on MNIST
======================================================

**Author(s)**: Anna Lappe (CERN), Matteo Bunino (CERN)

This tutorial provides a step-by-step guide to using the
:class:`~itwinai.torch.trainer.TorchTrainer` class for running a hyperparameter optimization
(HPO) study. We assume that you are familiar with the
:class:`~itwinai.torch.trainer.TorchTrainer` class, and the itwinai training pipeline. If you
are not, you might want to go through the tutorials on these first. To illustrate the process,
we will work with the FashionMNIST dataset.

By the end of this tutorial, you will:

*   Understand how the :class:`~itwinai.torch.trainer.TorchTrainer` functions.
*   Know how to create a configuration file to define your HPO study.
*   Understand the steps required to define and run an HPO study with the itwinai training
    pipeline.

You can find the full code for this tutorial `on Github
<https://github.com/interTwin-eu/itwinai/blob/main/tutorials/hpo-workflows/fashion-mnist>`_.


Setting up the Trainer
----------------------

Let's start by defining our trainer class. When you extend the itwinai
:class:`~itwinai.torch.trainer.TorchTrainer`, you inherit all the necessary logic to connect to
an existing Ray cluster, perform distributed machine learning (ML), hyperparameter optimization
(HPO), or both simultaneously. 

When selecting a strategy (e.g., Horovod or DeepSpeed) to distribute training, the
:class:`~itwinai.torch.trainer.TorchTrainer` ensures that workers can communicate via the
underlying Ray cluster. Similarly, the :class:`~itwinai.torch.trainer.TorchTrainer` allows you
to run HPO **without requiring any code changes**.

During HPO, Ray executes the :meth:`~itwinai.torch.trainer.TorchTrainer.train` method for each
trial **independently**, meaning that trials are completely agnostic of one another. For more
details, see the :doc:`HPO introduction <../../how-it-works/hpo/explain-hpo>`.

If you need to implement custom training logic not supported by
:class:`~itwinai.torch.trainer.TorchTrainer`, you can create a new trainer that inherits from
it and override the :meth:`~itwinai.torch.trainer.TorchTrainer.train` method.

.. important::

   **For Ray to correctly execute the training code,** you must call the
   :meth:`~itwinai.torch.trainer.TorchTrainer.ray_report` method **at the end of each epoch**
   to report the validation metric that you want to optimize during tuning (typically, the
   validation loss).  
   
   This is essential for both **distributed ML training and HPO**, as it allows Ray workers to
   communicate back with the head process, keeping it updated on the validation loss evolution.
   Additionally, this method enables checkpoint saving to a persistent storage location. See
   the official Ray documentation for more details: `Saving Checkpoints in Ray
   <https://docs.ray.io/en/latest/train/user-guides/checkpoints.html>`_.

   Also, consider that when a Ray cluster is not available and you are not running HPO, the
   :meth:`~itwinai.torch.trainer.TorchTrainer.train` method is automatically ignored. In other
   words, you don't need to remove the call to
   :meth:`~itwinai.torch.trainer.TorchTrainer.ray_report` when you are not using Ray for
   distributed ML training or HPO.

In this tutorial, we will tune two hyperparameters: **batch size** and **learning rate**.  
Our model will be a **ResNet18**, trained on the **FashionMNIST** dataset.

Below you can find an example of how the :meth:`~itwinai.torch.trainer.TorchTrainer.train`
method can be overridden:

.. code-block:: python
    
    def train(self) -> None:
        device = self.strategy.device()

        for self.current_epoch in range(self.epochs):
            self.set_epoch()

            train_losses = []
            val_losses = []

            # Training epoch
            self.model.train()
            for images, labels in self.train_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                train_loss = self.loss(outputs, labels)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                train_losses.append(train_loss.detach().cpu().numpy())

            # Validation epoch
            self.model.eval()
            for images, labels in self.validation_dataloader:
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = self.model(images)
                    val_loss = self.loss(outputs, labels)
                val_losses.append(val_loss.detach().cpu().numpy())

            # Log metrics with itwinai loggers
            self.log(
                np.mean(train_losses), "train_loss", kind="metric", step=self.current_epoch
            )
            self.log(np.mean(val_losses), "val_loss", kind="metric", step=self.current_epoch)

            # Report metrics and checkpoint to Ray head
            checkpoint = {
                "epoch": self.current_epoch,
                "loss": train_loss,
                "val_loss": val_loss,
            }
            metrics = {"loss": val_loss.item()}
            self.ray_report(metrics=metrics, checkpoint_data=checkpoint)


Configuring our Trainer
-----------------------
Now that we have our Trainer set up, the next step is to define a configuration file
for our HPO pipeline. Once again, this configuration will look very similar to any other
itwinai pipeline configuration, but we will add some HPO-specific parameters to define our
search space, search algorithm and scheduling algorithm.

When you want to run distributed ML training or HPO with Ray, you can specify additional
Ray-specific configuration objects that can be passed as arguments to the
:class:`~itwinai.torch.trainer.TorchTrainer`, using the arguments starting with ``ray_`` prefix
(e.g., ``ray_tune_config``).

In the configuration file, Ray configurations can be defined using Hydra syntax for objects.
The Ray search space is needed to define the domains of all the hyperparameters that we want to
tune. Once the Ray's search algorithm samples an hyperparameter set for a trial, the sampled
hyperparameter values will be used to override the default value in the
:class:`~itwinai.torch.config.TrainingConfiguration`, which is passed using the ``config``
argument of the :class:`~itwinai.torch.trainer.TorchTrainer`.


.. code-block:: yaml

    # For more info: https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html
    ray_scaling_config:
        _target_: ray.train.ScalingConfig
        num_workers: 1
        use_gpu: true
        resources_per_worker:
            CPU: 8
            GPU: 1

    # For more info: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html
    ray_tune_config:
        _target_: ray.tune.TuneConfig
        num_samples: 2
        scheduler:
            _target_: ray.tune.schedulers.ASHAScheduler
            metric: loss  # name of the metric to optimize during HPO
            mode: min
            max_t: 10
            grace_period: 5
            reduction_factor: 4
            brackets: 1

    # For more info: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.RunConfig.html
    ray_run_config:
        _target_: ray.tune.RunConfig
        storage_path: ${itwinai.cwd:}/ray_checkpoints
        name: FashionMNIST-HPO-Experiment

    # For more info: https://docs.ray.io/en/latest/tune/api/search_space.html
    ray_search_space:
        batch_size:
            type: choice
            categories: [32, 64, 128]
        learning_rate:
            type: uniform
            lower: 1e-5
            upper: 1e-3


Okay, let's break down the Ray configuration objects. 

*   The ``ray_scaling_config`` argument defines how we distribute resources between our trials.
    To learn more about the options for setting resources, please refer to the `ray train
    documentation <https://docs.ray.io/en/latest/train/user-guides/using-gpus.html>`_ on this
    topic. It is important that you ensure that you have allocated suffiecient resources on
    your cluster to be able to execute at least one trial. This means that if your
    configuration demands 4 GPUs and 32 CPUs per trial under ``resources_per_worker``, you
    should make sure that you have allocated at least this many GPUs and CPUs for your job.
*   In the ``ray_tune_config`` we configure which search algorithm and scheduler to use to
    search the hyperparameter space and sample new configurations. You can refer to the ray
    documentation to learn more about the supported `search algorithms
    <https://docs.ray.io/en/latest/tune/api/suggestion.html#tune-search-al_>`_ and `schedulers
    <https://docs.ray.io/en/latest/tune/api/schedulers.html>`_. In the ``num_samples`` argument
    you can specify how many trials you wish to run, the default is one. Ray will queue trials
    if they cannot all be executed at once.
*   The ``ray_run_config`` defines a path that is used for checkpointing. This is mandatory to
    set if you want to distribute any one trial across more than one node, because ray uses
    this as a shared directory to coordinate and share data generated on each of the nodes.
    The ``ray_run_config`` is of type ``ray.tune.RunConfig`` even for Ray distributed training
    without HPO. Find out more about the ``RunConfig``
    `here <https://docs.ray.io/en/latest/tune/api/doc/ray.tune.RunConfig.html>`_.
*   In the ``ray_search_space`` we define which hyperparameters we want to tune. For the
    tunable parameters we have to specify the type and define their domain. For more
    information on which parameter types are possible and how to define their domains, have a
    look at `this page <https://docs.ray.io/en/latest/tune/api/search_space.html>`_.


.. danger::
   
   **IMPORTANT:** When tuning, **you must use the exact hyperparameter names** as defined in 
   :class:`~itwinai.torch.config.TrainingConfiguration`. If you use different names,
   **the hyperparameters will be ignored**, 
   making the entire tuning process **invalid**.

   **Example:** In :class:`~itwinai.torch.config.TrainingConfiguration`, the learning rate is
   defined as ``optim_lr``.
   Therefore, when defining a search space for the learning rate, you **must** use ``optim_lr``
   as the name for the learning rate.  

   **Why?** The trainer accesses the learning rate using ``self.config.optim_lr``. If you
   define it with different names (e.g., ``lr`` or ``learning_rate``), the tuner will set the
   learning rate with the wrong name in the training configuration, **and it will be ignored by
   the trainer**.

.. note::
    Notice how in the ``ray_run_config`` we use the custom OmegaConf resolver ``${itwinai.cwd:}``
    provided by itwinai to dynamically compute the absolute path to the current working
    directory, depending on where the pipeline is executed. It is important to use an absolute
    path because the run config expects a URI for the ``storage_path``.



Running our Code
----------------

Great! So we have created our custom trainer inheriting from the
:class:`~itwinai.torch.trainer.TorchTrainer`, and we have defined our pipeline in a
configuration file. Now, all that is left to do is launch our training on HPC:

.. code-block:: bash

    cd tutorials/hpo-workflows/fashion-mnist sbatch slurm_hpo.sh
