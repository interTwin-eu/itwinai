HPC
===

On HPC you ask SLURM for nodes and GPUs, and SLURM starts your processes. itwinai generates
the job script for you, so you describe the allocation in YAML instead of writing ``#SBATCH``
directives by hand.

itwinai has been run on JSC, Vega, LUMI and Deucalion.

Launching
---------

One file holds everything: the pipeline, the plugins it needs, and the allocation. The
``slurm_config`` section is where you describe the hardware.

.. code-block:: yaml
   :caption: ``config.yaml``

   strategy: ddp          # ddp, deepspeed or horovod
   run_name: mnist

   plugins:
     - git+https://github.com/matbun/itwinai-mnist-plugin.git

   slurm_config:
     job_name: mnist-job
     account: intertwin
     partition: develbooster
     time: "01:00:00"

     num_nodes: 2
     gpus_per_node: 4
     cpus_per_task: 16
     pre_exec_file: <path or URL to the pre-exec script for this system>

     # Fields pointing back at this file
     pipe_key: training_pipeline
     config_name: config
     config_path: .

     # Propagate the global config into this section
     distributed_strategy: ${strategy}
     run_name: ${run_name}

     # Template for the command the job script will launch
     training_cmd: >
       {itwinai_launcher} exec-pipeline
       --config-name={config_name}
       --config-path={config_path}
       --strategy={distributed_strategy}
       --run-name={run_name}
       +pipe_key={pipe_key}

   training_pipeline:
     _target_: itwinai.pipeline.Pipeline
     steps:
       ...

Then install the plugins, build the script and submit it, in one command:

.. code-block:: bash

   itwinai run -c config.yaml     # install plugins, print the script
   itwinai run -sc config.yaml    # also save it, to read before submitting
   itwinai run -jc config.yaml    # also submit it

The builder picks the launcher from ``distributed_strategy``: ``torchrun`` for ``ddp`` and
``deepspeed``, plain ``srun`` for ``horovod``, and a Ray head plus workers when Ray is
requested. You do not call ``torchrun`` yourself.

.. note::

   ``itwinai generate-slurm -c slurm_config.yaml`` takes the same fields from a **standalone**
   file instead, and only builds the script: no plugin installation, no pipeline in the file.
   Use it when the environment is already set up and you want the script and nothing else.

The ``slurm_config`` fields are
:class:`~itwinai.slurm.configuration.MLSlurmBuilderConfig`, which extends
:class:`~itwinai.slurm.configuration.SlurmScriptConfiguration`. See
:doc:`../getting-started/complete-workflow-example` for the single-file walkthrough and
:doc:`../getting-started/slurm` for the SLURM primer and the full flag list.

Containers on HPC
-----------------

Compute nodes usually have no internet access and no writable site-packages, so the common
pattern is to convert the itwinai image to Singularity once, put the ``.sif`` on the shared
filesystem, and run it from the job script. See :doc:`../getting-started/containers`.

.. note::

   Container images for HPC are published as Singularity images, which are a different
   artifact from the Docker images used in the :doc:`cloud <cloud>` modes.

What to read next
-----------------

- :doc:`../how-it-works/training/training` for what the strategies do.
- :doc:`../how-it-works/hpo/explain-hpo` for hyperparameter optimization on HPC.
- :doc:`../how-it-works/scalability-report/scalability_report` to measure whether scaling out
  actually helped.
