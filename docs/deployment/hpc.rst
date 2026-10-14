HPC
===

On HPC you ask SLURM for nodes and GPUs, and SLURM starts your processes. itwinai generates
the job script for you, so you describe the allocation in YAML instead of writing ``#SBATCH``
directives by hand.

itwinai has been run on LUMI, JSC, Vega and Deucalion.

Launching
---------

Describe the allocation once:

.. code-block:: yaml
   :caption: ``slurm_config.yaml``

   account: intertwin
   time: 01:00:00
   partition: develbooster

   distributed_strategy: ddp   # ddp, deepspeed or horovod

   num_nodes: 2
   gpus_per_node: 4
   cpus_per_task: 16

   training_cmd: "itwinai exec-pipeline +pipe_key=training_pipeline strategy=ddp"

Then generate the script, inspect it, and submit it:

.. code-block:: bash

   itwinai generate-slurm -c slurm_config.yaml        # print it
   itwinai generate-slurm -c slurm_config.yaml -s     # save it
   itwinai generate-slurm -c slurm_config.yaml -j     # submit it

The builder picks the launcher from ``distributed_strategy``: ``torchrun`` for ``ddp`` and
``deepspeed``, plain ``srun`` for ``horovod``, and a Ray head plus workers when Ray is
requested. You do not call ``torchrun`` yourself.

See :doc:`../getting-started/slurm` for the SLURM primer, the full flag list and the
configuration reference.

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
