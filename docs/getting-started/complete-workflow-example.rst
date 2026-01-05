Complete Workflow Example
=========================

This page shows you how to run a complete machine learning workflow with itwinai in just a few
steps. This example demonstrates distributed training with SLURM integration, plugin usage, and
pipeline configuration all in a single configuration file.

This guide shows how to use the ``itwinai run`` command, which is a single entry
point to install an itwinai plugin, define an ML workflow and its
hyperparameters, and submit a distributed AI job on a SLURM cluster. This is
made possible thanks to the integration within the same configuration file of:
plugin configuration, SLURM cluster configuration, and ML workflow definition
(including hyperparameters).

If you want to know more about how to use the ``itwinai run`` command, please
refer to the official CLI reference for `run
<https://itwinai.readthedocs.io/stable/api/cli_reference.html#run>`__.

.. note::

   **Difference between** ``itwinai run`` **and** ``itwinai exec-pipeline``

   - ``itwinai run`` receives a configuration file that includes configuration for plugins, the
     SLURM cluster, and ML workflows, and it takes care of executing the complete workflow
     end-to-end.

     See the official CLI reference for
     `run <https://itwinai.readthedocs.io/stable/api/cli_reference.html#run>`__.

   - ``itwinai exec-pipeline`` is more low-level and is designed to receive only the
     configuration of an ML workflow (a.k.a. a "pipeline"). It does not take care of SLURM
     submission or plugin installation: it only executes the ML workload described by the
     pipeline, i.e., a subset of the configuration that ``itwinai run`` would use.

     See the official CLI reference for
     `exec-pipeline <https://itwinai.readthedocs.io/stable/api/cli_reference.html#exec-pipeline>`__.

Prerequisites
-------------

- itwinai installed: ``pip install itwinai``
- Access to an HPC cluster with SLURM (and a corresponding pre-execution script)
- Git access for plugin installation

Steps
-----

1. Create a configuration file, e.g. ``config.yaml``, that contains a pipeline, the required
   plugins, and a SLURM configuration, as follows:

   .. code-block:: yaml

      # Default fields (always needed)
      strategy: ddp
      run_name: "mnist"
      
      plugins:
        - <my first required plugin>
        - <my second required plugin>

      slurm_config:
        job_name: my-job-name
        account: my-billing-account
        partition: my-partition
        submit_job: true   # Set this if you actually want to submit the job
        save_script: true  # Set this if you want to store the generated SLURM script(s)
        pre_exec_file: <path or URL to pre-execution file for current system>
        
        # Fields referring to this config file
        pipe_key: training_pipeline
        config_name: <name of this config file>
        config_path: <path to the directory containing this config file>

        # Propagate global config to this slurm_config
        distributed_strategy: ${strategy}
        run_name: ${run_name}

        # This provides a template for the training command launched using:
        # $ itwinai exec-pipeline -c <this config>.yaml [ARGS]
        # The template is filled in using the fields of this slurm_config.
        # Example:
        training_cmd: >
          {itwinai_launcher} exec-pipeline
          --config-name={config_name}
          --config-path={config_path}
          --strategy={distributed_strategy}
          --run-name={run_name}
          +pipe_key={pipe_key}

        # Any other SLURM configuration options you want to set.
        # Check out the SLURM builder for more information.
        ...

      # Your pipeline. You can name it whatever you want, but make sure to set
      # the ``pipe_key`` variable accordingly.
      training_pipeline:
        _target_: itwinai.pipeline.Pipeline
        steps:
          dataloading_step:
            ...

2. Run the workflow:

   .. code-block:: bash

      itwinai run -c run_config.yaml

   The command above will install the dependencies and produce a SLURM job script, but it will not
   submit the job to SLURM. To also submit the job to the SLURM queue, add the ``-j`` option:

   .. code-block:: bash

      itwinai run -jc run_config.yaml

   The ``slurm_config`` section follows :class:`itwinai.slurm.configuration.MLSlurmBuilderConfig`
   (extending :class:`itwinai.slurm.configuration.SlurmScriptConfiguration`), which documents
   each field and its default. Use the YAML to set values; ``-j`` and ``-s`` are the only CLI
   overrides applied on top of the config for submission and saving.

MNIST Example
-------------

Here's a concrete example showing how to run distributed MNIST training on the Vega HPC system
using the itwinai MNIST plugin. This simplified example shows the key components:

.. code-block:: yaml

    # Default fields (always needed)
    strategy: ddp
    run_name: "mnist"

    # General config
    dataset_root: .tmp/
    num_classes: 10
    batch_size: 128
    num_workers_dataloader: 4
    pin_memory: False
    lr: 0.001
    momentum: 0.9
    fp16_allreduce: False
    use_adasum: False
    gradient_predivide_factor: 1.0
    epochs: 5
    test_data_path: mnist-sample-data
    inference_model_mlflow_uri: mnist-pre-trained.pth
    predictions_dir: mnist-predictions
    predictions_file: predictions.csv
    class_labels: null
    checkpoints_location: checkpoints
    checkpoint_every: 1

    plugins:
      - git+https://github.com/matbun/itwinai-mnist-plugin.git

    slurm_config:
      job_name: mnist-job
      account: s24r05-03-users
      partition: gpu
      memory: 64G
      mode: single
      num_nodes: 2
      submit_job: true
      save_script: true
      pre_exec_file: https://raw.githubusercontent.com/interTwin-eu/itwinai/refs/heads/main/src/itwinai/slurm/system-base-scripts/vega_pre_exec.sh

      # Fields referring to this config file
      pipe_key: training_pipeline
      config_name: run-example # Assuming this is the name of this file
      config_path: . # Assuming that run-example.yaml is in the current directory

      # Propagate global config to this slurm_config
      distributed_strategy: ${strategy}
      run_name: ${run_name}

      # This provides a template for the training command launched
      # using itwinai exec-pipeline -c <this config>.yaml
      # The template is filled in using the fields of this slurm_config.
      training_cmd: >
        {itwinai_launcher} exec-pipeline
        --config-name={config_name}
        --config-path={config_path}
        --strategy={distributed_strategy}
        --run-name={run_name}
        +pipe_key={pipe_key}

    # Workflows configuration
    training_pipeline:
      _target_: itwinai.pipeline.Pipeline
      steps:
        dataloading_step:
          _target_: itwinai.plugins.mnist.dataloader.MNISTDataModuleTorch
          save_path: ${dataset_root}
        training_step:
          _target_: itwinai.torch.trainer.TorchTrainer
          strategy: ${strategy}
          measure_gpu_data: False
          enable_torch_profiling: False
          store_torch_profiling_traces: False
          measure_epoch_time: False
          run_name: ${run_name}
          time_ray: True # track time for ray report and fit
          # from_checkpoint: ${itwinai.cwd:}/checkpoints_ddp/best_model/
          config:
            batch_size: ${batch_size}
            num_workers_dataloader: ${num_workers_dataloader}
            pin_gpu_memory: ${pin_memory}

        ... # The rest of the pipeline is omitted for the sake of readability

.. note::

   This example has been simplified for readability. The full configuration includes additional
   parameters, hyperparameter optimization settings, detailed metrics, and more complete pipeline
   steps. For a working example, please refer to ``use-cases/mnist/torch/run-example.yaml``.

Key Components
~~~~~~~~~~~~~~

- **Plugin**: Uses the ``itwinai-mnist-plugin`` which provides MNIST-specific components (based on the code in ``use-cases/mnist/torch/``)
- **SLURM**: Configured for Vega HPC system with 2 GPU nodes  
- **Pipeline**: Two-step workflow with data loading and distributed training
- **Logging**: Combines console output with MLFlow experiment tracking

**Full Example Configuration**

For a complete configuration with hyperparameter optimization, advanced metrics, and more detailed settings, 
see the full example at ``use-cases/mnist/torch/run-example.yaml``. You can run it directly with:

.. code-block:: bash

   itwinai run -jc https://raw.githubusercontent.com/interTwin-eu/itwinai/refs/heads/main/use-cases/mnist/torch/run-example.yaml

What This Example Does
~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~

When you run this example, itwinai will:

1. Download and install the MNIST plugin
2. Generate a SLURM job script
3. Submit the job to your HPC cluster
4. Run distributed MNIST training across 2 nodes
5. Save checkpoints and training logs
