HPC
===

On HPC you ask SLURM for nodes and GPUs, and SLURM starts your processes. itwinai generates
the job script for you, so you describe the allocation in YAML instead of writing ``#SBATCH``
directives by hand.

itwinai has been run on the following EuroHPC sites: JSC, Vega, LUMI and Deucalion.

Launching
---------

One file holds the pipeline, the plugins it needs and the allocation: the ``slurm_config``
section describes the hardware, and ``itwinai run -jc config.yaml`` installs the plugins,
generates the job script and submits it. See
:doc:`../getting-started/complete-workflow-example` for a full config file and
:doc:`../getting-started/slurm` for the field reference.

The builder picks the launcher from ``distributed_strategy``: ``torchrun`` for ``ddp`` and
``deepspeed``, plain ``srun`` for ``horovod``, and a Ray head plus workers when Ray is
requested. You do not call ``torchrun`` yourself.

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
