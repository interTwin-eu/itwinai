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

Allocate a compute node with 4 GPUs on JSC supercomputer:

.. code-block:: bash

   salloc --account=intertwin --partition=batch --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --gpus-per-node=4 --time=01:00:00

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

See here a table of them: https://www.glue.umd.edu/hpcc/help/slurmenv.html

Job arrays
----------

Job arrays allow to conveniently submit a collection of similar and independent jobs.

To know more, see this: https://slurm.schedmd.com/job_array.html

Job array example: https://guiesbibtic.upf.edu/recerca/hpc/array-jobs

.. _job script: #job-scripts-for-batch-jobs
