.. _hpo_torchtrainer_workflow:

Hyperparameter Optimization with RayTorchTrainer on MNIST
=========================================================

**Author(s)**: Anna Lappe (CERN)

This tutorial provides a step-by-step guide to using the ``RayTorchTrainer`` class for running 
a hyperparameter optimization (HPO) study. We assume that you are familiar with the ``TorchTrainer`` class, and the itwinai training pipeline. If you are not, you might want to go through the tutorials on these first.
To illustrate the process, we will work with the FashionMNIST dataset.

The code we write here will closely resemble what you would implement using the itwinai 
``TorchTrainer``, with minor modifications required to make it compatible with the Ray framework. 
By the end of this tutorial, you will:

*   Understand how the ``RayTorchTrainer`` functions.
*   Know how to create a configuration file to define your HPO study.
*   Understand the steps required to define and run an HPO study with the itwinai training pipeline.

You can find the full code for this tutorial `on Github <https://github.com/interTwin-eu/itwinai/blob/main/tutorials/hpo-workflows/distributed-workflow>`_.


Setting up the Trainer
-----------------------

Let’s start by defining our trainer class. Its structure will closely mirror that of 
the ``TorchTrainer``, but with a few key adjustments to integrate seamlessly with the Ray backend. 
The most notable change is that we move certain initialization steps into the ``train()`` function. 
Ray executes the ``train()`` function for each trial independently,
which means that the trials are completetely agnostic of one another. This means that
the distributed strategy and thus the model, optimizer, logger, etc., can only be instantiated
within the ``train()`` function.
For more details on this, see the :doc:`HPO introduction <../../how-it-works/hpo/explain-hpo>`.

In this tutorial, we will tune two hyperparameters, the batch size and the learning rate.
Our model will be a ResNet18, and we will train it on the FashionMNIST dataset.


Code Comparison: RayTorchTrainer vs TorchTrainer
------------------------------------------------------

.. tabs::

    .. tab:: RayTorchTrainer

        .. code-block:: python

            class MyRayTorchTrainer(RayTorchTrainer):
                def __init__(
                    self,
                    config: Dict,
                    strategy: Literal["ddp", "deepspeed"] = "ddp",
                    name: str | None = None,
                    logger: Logger | None = None,
                ) -> None:
                    super().__init__(config=config, strategy=strategy, name=name, logger=logger)

                def create_model_loss_optimizer(self):
                    model = resnet18(num_classes=10)
                    model.conv1 = torch.nn.Conv2d(
                        1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False
                    )
                    # First, define strategy-wise optional configurations
                    if isinstance(self.strategy, RayDeepSpeedStrategy):
                        distribute_kwargs = dict(
                            config_params=dict(
                                train_micro_batch_size_per_gpu=self.training_config["batch_size"]
                            )
                        )
                    else:
                        distribute_kwargs = {}
                    optimizer = Adam(model.parameters(), lr=self.training_config["learning_rate"])
                    self.model, self.optimizer, _ = self.strategy.distributed(
                        model, optimizer, **distribute_kwargs
                    )
                    self.loss = CrossEntropyLoss()

                def train(self, config, data):

                    ################## This is unique to the RayTorchTrainer #####################
                    self.training_config = config
                    self.strategy.init()
                    self.initialize_logger(
                        hyperparams=self.training_config, rank=self.strategy.global_rank()
                    )
                    self.create_model_loss_optimizer()
                    self.create_dataloaders(
                        train_dataset=data[0], validation_dataset=data[1], test_dataset=data[2]
                    )
                    ###############################################################################

                    for epoch in range(self.training_config["epochs"]):
                        if self.strategy.global_world_size() > 1:
                            self.set_epoch(epoch)

                        train_losses = []
                        val_losses = []

                        for images, labels in self.train_dataloader:
                            ############### This is unique to the RayTorchTrainer ##################
                            if isinstance(self.strategy, RayDeepSpeedStrategy):
                                device = self.strategy.device()
                                images, labels = images.to(device), labels.to(device)
                            ########################################################################

                            outputs = self.model(images)
                            train_loss = self.loss(outputs, labels)
                            self.optimizer.zero_grad()
                            train_loss.backward()
                            self.optimizer.step()
                            train_losses.append(train_loss.detach().cpu().numpy())

                        for images, labels in self.validation_dataloader:
                            ############### This is unique to the RayTorchTrainer ##################
                            if isinstance(self.strategy, RayDeepSpeedStrategy):
                                device = self.strategy.device()
                                images, labels = images.to(device), labels.to(device)
                            ########################################################################

                            with torch.no_grad():
                                outputs = self.model(images)
                                val_loss = self.loss(outputs, labels)
                            val_losses.append(val_loss.detach().cpu().numpy())

                        self.log(np.mean(train_losses), "train_loss", kind="metric", step=epoch)
                        self.log(np.mean(val_losses), "val_loss", kind="metric", step=epoch)
                        checkpoint = {
                            "epoch": epoch,
                            "loss": train_loss,
                            "val_loss": val_loss,
                        }
                        ############### This is unique to the RayTorchTrainer ##################
                        metrics = {"loss": val_loss.item()}
                        self.checkpoint_and_report(
                            epoch, tuning_metrics=metrics, checkpointing_data=checkpoint
                        )
                        ########################################################################


    .. tab:: TorchTrainer

        .. code-block:: python

            class MyTrainer(TorchTrainer):
                def __init__(
                    self,
                    config: Dict | TrainingConfiguration | None = None,
                    strategy: Literal["ddp", "deepspeed", "horovod"] = "ddp",
                    name: str | None = None,
                    logger: Logger | None = None,
                ) -> None:
                    self.config = config
                    super().__init__(config=config, strategy=strategy, name=name, logger=logger)

                def create_model_loss_optimizer(self):
                    model = resnet18(num_classes=10)
                    model.conv1 = torch.nn.Conv2d(
                        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                    )
                    # First, define strategy-wise optional configurations
                    if isinstance(self.strategy, DeepSpeedStrategy):
                        distribute_kwargs = dict(
                            config_params=dict(train_micro_batch_size_per_gpu=self.config.batch_size)
                        )
                    else:
                        distribute_kwargs = {}
                    optimizer = Adam(model.parameters(), lr=self.config.learning_rate)
                    self.model, self.optimizer, _ = self.strategy.distributed(
                        model, optimizer, **distribute_kwargs
                    )
                    self.loss = CrossEntropyLoss()

                def train(self):
                    
                    for epoch in range(self.config.epochs):
                        if self.strategy.global_world_size() > 1:
                            self.set_epoch(epoch)

                        train_losses = []
                        val_losses = []

                        for images, labels in enumerate(self.train_dataloader):
                            
                            device = self.strategy.device()
                            images, labels = images.to(device), labels.to(device)

                            outputs = self.model(images)
                            train_loss = self.loss(outputs, labels)
                            self.optimizer.zero_grad()
                            train_loss.backward()
                            self.optimizer.step()
                            train_losses.append(train_loss.detach().cpu().numpy())

                        for images, labels in enumerate(self.validation_dataloader):

                            device = self.strategy.device()
                            images, labels = images.to(device), labels.to(device)

                            with torch.no_grad():
                                outputs = self.model(images)
                                val_loss = self.loss(outputs, labels)
                            val_losses.append(val_loss.detach().cpu().numpy())

                        self.log(np.mean(train_losses), "train_loss", kind="metric", step=epoch)
                        self.log(np.mean(val_losses), "val_loss", kind="metric", step=epoch)
                        checkpoint = {
                            "epoch": epoch,
                            "loss": train_loss,
                            "val_loss": val_loss,
                        }
                        checkpoint_filename = self.checkpoints_location.format(epoch)
                        torch.save(checkpoint, checkpoint_filename)
                        self.log(
                            checkpoint_filename,
                            os.path.basename(checkpoint_filename),
                            kind="artifact",
                        )


Configuring our Training
-------------------------
Amazing! Now that we have our Trainer set up, the next step is to define a configuration file 
for our HPO pipeline. Once again, this configuration will look very similar to any other 
itwinai pipeline configuration, but we will add some HPO-specific parameters to define our 
search space, search algorithm and scheduling algorithm. 


Code Comparison: HPO Config vs TorchTrainer Config
----------------------------------------------------------

.. tabs::

    .. tab:: HPO Config

        .. code-block:: yaml

            ray_training_pipeline:
            class_path: itwinai.pipeline.Pipeline
            init_args:
                steps:
                - class_path: data.FashionMNISTGetter
                - class_path: data.FashionMNISTSplitter
                    init_args: 
                    train_proportion: 0.9
                    validation_proportion: 0.1
                - class_path: trainer.MyRayTorchTrainer
                    init_args:
                    config:
                        scaling_config:
                            num_workers: 4
                            use_gpu: true
                            resources_per_worker:
                                CPU: 5
                                GPU: 1
                        train_loop_config:
                            batch_size:
                                type: choice
                                options: [32, 64, 128]
                            learning_rate:
                                type: uniform
                                min: 1e-5
                                max: 1e-3
                            epochs: 20
                        tune_config:
                            num_samples: 2
                            scheduler:
                                name: asha
                                max_t: 20
                                grace_period: 10
                                reduction_factor: 4
                                brackets: 1
                            search_alg:
                                name: bayes
                                metric: loss
                                mode: min
                                n_random_steps: 5
                        run_config:
                            storage_path: ray_checkpoints
                            name: Virgo-HPO-Experiment
                    strategy: ddp
                    logger:
                        class_path: itwinai.loggers.LoggersCollection
                        init_args:
                        loggers:
                            - class_path: itwinai.loggers.MLFlowLogger
                            init_args:
                                experiment_name: MNIST HPO Experiment
                                log_freq: batch

    .. tab:: TorchTrainer Config

        .. code-block:: yaml

            training_pipeline:
            class_path: itwinai.pipeline.Pipeline
            init_args:
                steps:
                - class_path: data.FashionMNISTGetter
                - class_path: data.FashionMNISTSplitter
                    init_args: 
                    train_proportion: 0.9
                    validation_proportion: 0.1
                - class_path: trainer.MyRayTrainer
                    init_args:
                    strategy: ddp
                    epochs: 20
                    checkpoints_location: checkpoints
                    logger:
                        class_path: itwinai.loggers.LoggersCollection
                        init_args:
                        loggers:
                            - class_path: itwinai.loggers.MLFlowLogger
                            init_args:
                                experiment_name: MNIST Experiment
                                log_freq: batch


Okay, let's break down the arguments to our ``MyRayTorchTrainer`` class. 

*   The ``scaling_config`` argument defines how we distribute resources between our trials. To learn more about the options for setting resources, please refer to the `ray train documentation <https://docs.ray.io/en/latest/train/user-guides/using-gpus.html>`_ on this topic. It is important that you ensure that you have allocated suffiecient resources on your cluster to be able to execute at least one trial. This means that if your configuration demands 4 GPUs and 32 CPUs per trial under ``resources_per_worker``, you should make sure that you have allocated at least this many GPUs and CPUs for your job.
*   In the ``train_loop_config`` we define which hyperparameters we want to tune, as well as any additional parameters that we want to pass to our ``train()`` function. For the tunable parameters we have to specify the type and define their domain. For more information on which parameter types are possible and how to define their domains, have a look at `this page <https://docs.ray.io/en/latest/tune/api/search_space.html>`_, and learn how to define their domains according to the ``RayTorchTrainer``'s specifications `here <https://github.com/interTwin-eu/itwinai/blob/main/src/itwinai/torch/trainer.py>`_.
*   In the ``tune_config`` we configure which search algorithm and scheduler to use to search the hyperparameter space and sample new configurations. Almost all search algorithms and schedulers supported by ray tune are also supported by us. You can refer to the ray documentation to learn more about the supported `search algorithms <https://docs.ray.io/en/latest/tune/api/suggestion.html#tune-search-al_>`_ and `schedulers <https://docs.ray.io/en/latest/tune/api/schedulers.html>`_. In the ``num_samples`` argument you can specify how many trials you wish to run, the default is one. Ray will queue trials if they cannot all be executed at once.
*   The ``run_config`` defines a path that is used for checkpointing. This is mandatory to set if you want to distribute any one trial across more than one node, because ray uses this as a shared directory to coordinate and share data generated on each of the nodes.


Running our Code
----------------

Great! So we have created our custom trainer inheriting from the ``RayTorchTrainer``, and we 
have defined our pipeline in a configuration file. 
Now, all that is left to do is launch our training:

.. code-block:: bash

    cd tutorials/hpo-workflows/distributed-workflow
    sbatch slurm_hpo.sh