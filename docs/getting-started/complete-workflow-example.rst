Complete Workflow Example
=========================

This page shows you how to run a complete machine learning workflow with itwinai in just a few
steps. This example demonstrates distributed training with SLURM integration, plugin usage, and
pipeline configuration all in a single configuration file.

Prerequisites
-------------

- itwinai installed: ``pip install itwinai``
- Access to an HPC cluster with SLURM (and a corresponding pre-execution script)
- Git access for plugin installation

Steps
-----

1. Create a configuration file, e.g. ``config.yaml``, that contains a pipeline, the required
   plugins and a SLURM configuration, as follows:

.. code-block:: yaml


   plugins:
   - <my first required plugin>
   - <my second required plugin>

   slurm_config:
     job_name: my-job-name
     account: my-billing-account
     partition: my-partition
     submit_job: true # This has to be set if you actually want to launch the job
     save_script: true # If you wish to store the resulting SLURM script(s) to file
     pre_exec_file: <path or URL to pre-execution file for current system>

     # and any other SLURM configuration options you want to set. Check out the SLURM
     # builder for more information
     ...
        


   # Your pipeline. You can name it whatever you want, but make sure to change the 
   # ``pipe_key`` variable to match it.
   training_pipeline:
     _target_: itwinai.pipeline.Pipeline
     steps:
       dataloading_step:
         ...
      

2. Run the workflow:

.. code-block:: bash

   itwinai run -c run_config.yaml

MNIST Example
-------------

Here's a concrete example showing how to run distributed MNIST training on the Vega HPC system
using the itwinai MNIST plugin. This simplified example shows the key components:

.. code-block:: yaml

   # General training parameters
   run_id: mnist-usecase-0
   dataset_root: .tmp/
   batch_size: 128
   lr: 0.001
   epochs: 5
   strategy: ddp
   # The rest of the general config is omitted for the sake of readability

   # Install the itwinai MNIST plugin from GitHub
   plugins:
   - git+https://github.com/matbun/itwinai-mnist-plugin.git

   # SLURM configuration for Vega HPC system
   slurm_config:
     job_name: mnist-job
     account: s24r05-03-users  # Replace with your account
     partition: gpu
     memory: 64G
     num_nodes: 2
     submit_job: true
     save_script: true
     pre_exec_file: https://raw.githubusercontent.com/interTwin-eu/itwinai/refs/heads/main/src/itwinai/slurm/system-base-scripts/vega_pre_exec.sh

   # Define the ML pipeline
   training_pipeline:
     _target_: itwinai.pipeline.Pipeline
     steps:
       dataloading_step:
         _target_: itwinai.plugins.mnist.dataloader.MNISTDataModuleTorch
         save_path: ${dataset_root}
       training_step:
         _target_: itwinai.torch.trainer.TorchTrainer
         run_id: ${run_id}
         config:
           batch_size: ${batch_size}
           optimizer: sgd
           optim_lr: ${lr}
         model:
           _target_: itwinai.plugins.mnist.model.Net
         epochs: ${epochs}
         strategy: ${strategy}
         logger:
           _target_: itwinai.loggers.LoggersCollection
           loggers:
             - _target_: itwinai.loggers.ConsoleLogger
             - _target_: itwinai.loggers.MLFlowLogger
               experiment_name: MNIST classifier
        ... # The rest of the pipeline is omitted for the sake of readability

.. note::
   
   This example has been simplified for readability. The full configuration includes additional 
   parameters, hyperparameter optimization settings, detailed metrics, and more complete pipeline 
   steps. For a working example, please refer to ``use-cases/mnist/torch/run-example.yaml``.

**Key Components:**

- **Plugin**: Uses the ``itwinai-mnist-plugin`` which provides MNIST-specific components (based on the code in ``use-cases/mnist/torch/``)
- **SLURM**: Configured for Vega HPC system with 2 GPU nodes  
- **Pipeline**: Two-step workflow with data loading and distributed training
- **Logging**: Combines console output with MLFlow experiment tracking

**Full Example Configuration**

For a complete configuration with hyperparameter optimization, advanced metrics, and more detailed settings, 
see the full example at ``use-cases/mnist/torch/run-example.yaml``. You can run it directly with:

.. code-block:: bash

   itwinai run -c https://raw.githubusercontent.com/interTwin-eu/itwinai/refs/heads/main/use-cases/mnist/torch/run-example.yaml

What This Example Does
----------------------

This configuration demonstrates several key itwinai features:

**Plugin Integration**
  The example uses the MNIST plugin from GitHub, showing how to extend itwinai with external components.

**SLURM Integration** 
  The ``slurm_config`` section automatically generates and submits SLURM jobs for HPC execution,
  including multi-node distributed training setup.

**Unified Configuration**
  All training parameters, infrastructure settings, and pipeline definitions are in one file,
  making it easy to reproduce experiments.

**Distributed Training**
  Configured for 2-node distributed training using DDP (Distributed Data Parallel) strategy.

Expected Output
---------------

When you run this example, itwinai will:

1. Download and install the MNIST plugin
2. Generate a SLURM job script
3. Submit the job to your HPC cluster
4. Run distributed MNIST training across 2 nodes
5. Save checkpoints and training logs
