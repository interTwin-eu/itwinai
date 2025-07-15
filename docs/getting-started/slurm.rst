Submitting jobs to SLURM on HPC
====================================

**Author(s)**: Matteo Bunino (CERN)

Here you can find a minimal set of resources to use SLURM job scheduler on an HPC cluster.

What is SLURM? See this quickstart: https://slurm.schedmd.com/quickstart.html

SLURM cheatsheets:

- https://slurm.schedmd.com/pdfs/summary.pdf
- https://www.carc.usc.edu/user-information/user-guides/hpc-basics/slurm-cheatsheet

Commands
--------

- ``sinfo``: get cluster status (e.g., number of free nodes at the moment).
- ``squeue -u USERNAME``: visualize the queue of jobs of ``USERNAME`` user.
- ``sbatch JOBSCRIPT``: submit a `job script`_ to the SLURM queue.
- ``scontrol show job JOBID``: get detailed info of job with ``JOBID`` id.
- ``scancel JOBID``: cancel job with ``JOBID`` id.
- ``scancel -u USERNAME``: cancel all jobs of ``USERNAME`` user.
- ``srun``: is used to execute a command in a SLURM job script. Example: ``srun python train.py``.
- ``sacct -j JOBID``: Get job stats after completion/when running.
- ``scontrol write batch_script JOBID -``: show the jos script associated with some job.

More commands here: https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands/

SLURM commands on JSC: https://apps.fz-juelich.de/jsc/hps/juwels/batchsystem.html#slurm-commands

Job scripts for batch jobs
--------------------------

SLURM job scripts are regular shell files enriched with some ``#SBATCH`` directives at the top.

To know more, see this: https://www.osc.edu/book/export/html/2861

Check job status
----------------

Once a job is submitted to the SLURM queue, it goes through a number of states before finishing.
You can check in which state is a job of interest using the following command:

.. code-block:: bash

   scontrol show job JOBID

To interpret the state code, use this guide: https://confluence.cscs.ch/display/KB/Meaning+of+Slurm+job+state+codes 

Interactive shell on a compute node
-----------------------------------

Allocate a compute node with 4 GPUs for 1 hour:

.. note::
   make sure to adapt the ``--account`` in the code snippets below to your allocation account

.. tab-set::

   .. tab-item:: Juelich (JSC)

      On the `JUWELS <https://apps.fz-juelich.de/jsc/hps/juwels/configuration.html>`_ system at Juelich Supercomputer (JSC):

      .. code-block:: bash

         salloc --account=intertwin --partition=develbooster --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --gpus-per-node=4 --time=01:00:00

         
   .. tab-item:: Vega

      On `Vega <https://doc.vega.izum.si/introduction/>`_ Supercomputer:

      .. code-block:: bash

         salloc --account=s24r05-03-users --partition=gpu --nodes=1 --cpus-per-gpu=4 --gres=gpu:4 --time=1:00:00

   .. tab-item:: LUMI

      On `LUMI <https://docs.lumi-supercomputer.eu/hardware/lumig/>`_ Supercomputer:

      .. code-block:: bash

         salloc --account=project_123456 --partition=dev-g --nodes=1  --gres=gpu:4 --cpus-per-gpu=16 --time=1:00:00
   

Once resources are available, the command will return a ``JOBID``. Use it to jump into the compute node with the 4 GPUs in this way:

.. code-block:: bash

   srun --jobid JOBID --overlap --pty /bin/bash

   # Check that you are in the compute node and you have 4 GPUs
   nvidia-smi

Remember to load the correct environment modules before activating the python virtual environment.

Alternatively, if you don’t need to open an interactive shell on the compute node allocated
with the ``salloc`` command,
you can directly run a command on the allocated node(s) by prefixing your command with ``srun``.
This approach ensures that your command is executed on the compute node rather than on the login node.

Example:

.. code-block:: bash  

   srun YOUR_COMMAND

Environment variables
---------------------

Before running a job, SLURM will set some environment variables in the job environment.

You can see a table of them here: https://www.glue.umd.edu/hpcc/help/slurmenv.html

Job arrays
----------

Job arrays allow to conveniently submit a collection of similar and independent jobs.

For more information on job arrays, see the following documentation:
https://slurm.schedmd.com/job_array.html

Job array example: https://guiesbibtic.upf.edu/recerca/hpc/array-jobs

.. _job script: #job-scripts-for-batch-jobs

itwinai SLURM Script Builder
----------------------------

``itwinai`` includes a SLURM script builder to simplify the management of SLURM scripts.
It provides a default method for generating and submitting simple scripts, but also
allows you to customize and launch multiple jobs with different configurations in a single
command.

Generating SLURM Script
+++++++++++++++++++++++

To generate and submit a SLURM script, you can use the following command:

.. code-block:: bash

   itwinai generate-slurm

This will use the default variables for everything, and will save the script for
reproducibility. You can override variables by setting flags. For example, to set
the job name to ``my_test_job``, you can do the following:

.. code-block:: bash

   itwinai generate-slurm --job-name my_test_job

For a full list of options, add the ``--help`` or equivalently ``-h`` flag:

.. code-block:: bash

   itwinai generate-slurm --help

Preview SLURM Scripts
+++++++++++++++++++++

A common workflow is to preview the SLURM script before saving or submitting it. This
can be done by adding ``--no-submit-job`` and ``--no-save-script`` as follows:

.. code-block:: bash

   itwinai generate-slurm --no-submit-job --no-save-script

This will print the script in the console for inspection without saving the script or
submitting the job. These arguments provide a quick way to verify that your script is
configured correctly. 

SLURM Configuration File
++++++++++++++++++++++++

The ``itwinai`` SLURM Script builder allows you to store your SLURM variables in a
configuration file, letting you easily manage the different parameters without the noise
of the ``SBATCH`` syntax. You can add a configuration file using ``--config`` or ``-c``.
This configuration file uses ``yaml`` syntax. The following is an example of a SLURM
configuration file: 

.. code-block:: yaml
   :caption: ``slurm_config.yaml``
   :name: Example SLURM Config

    account: intertwin
    time: 01:00:00
    partition: develbooster

    # Which distributed strategy/framework to use, controlling how the communication
    # between workers is implemented. The acronym 'ddp' refers to PyTorch's Distributed
    # Data Parallel.
    dist_strat: ddp # "ddp", "deepspeed" or "horovod"

    std_out: slurm_job_logs/${dist_strat}.out
    err_out: slurm_job_logs/${dist_strat}.err
    job_name: ${dist_strat}-job

    num_nodes: 1
    num_tasks_per_node: 1
    gpus_per_node: 4
    cpus_per_task: 16

    training_cmd: "train.py"

If this file is called ``slurm_config.yaml``, then you would specify it as follows:

.. code-block:: bash

   itwinai generate-slurm -c slurm_config.yaml

You can override arguments from the configuration file in the CLI if you pass them
after the config file. For example, if you want to use everything from the configuration
file but want a different job name without changing the config, you can do the following:

.. code-block:: bash

   itwinai generate-slurm -c slurm_config.yaml --job-name different_job_name


The resulting SLURM script generated using the ``slurm_config.yaml`` file above is:

.. code-block:: bash
   :caption: ``ddp-1x4.sh``
   :name: SLURM Job Script generated using the configuration above

      #!/bin/bash

      # Job configuration
      #SBATCH --job-name=ddp-job
      #SBATCH --account=intertwin
      #SBATCH --partition=develbooster
      #SBATCH --time=01:00:00

      #SBATCH --output=slurm_job_logs/ddp.out
      #SBATCH --error=slurm_job_logs/ddp.err

      # Resource allocation
      #SBATCH --nodes=1
      #SBATCH --ntasks-per-node=1
      #SBATCH --cpus-per-task=16
      #SBATCH --gpus-per-node=4
      #SBATCH --gres=gpu:4
      #SBATCH --exclusive

      # Pre-execution command
      ml --force purge
      ml Stages/2025 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA Python CMake HDF5 PnetCDF libaio mpi4py git
      source .venv/bin/activate
      export OMP_NUM_THREADS=4

      # Job execution command
      srun --cpu-bind=none --ntasks-per-node=1 \
      bash -c "torchrun \
      --log_dir='logs_torchrun' \
      --nnodes=$SLURM_NNODES \
      --nproc_per_node=$SLURM_GPUS_PER_NODE \
      --rdzv_id=$SLURM_JOB_ID \
      --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
      --rdzv_backend=c10d \
      --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)'i:29500 \
      train.py"