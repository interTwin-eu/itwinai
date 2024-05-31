PyTorch basics example
======================

Tutorial: distributed strategies for PyTorch
--------------------------------------------

In this tutorial we show how to use torch ``DistributedDataParallel`` (DDP), Horovod and
DeepSpeed from the same client code.
Note that the environment is tested on the HDFML system at JSC. For other systems,
the module versions might need change accordingly.

Setup
+++++

First, from the root of this repository, build the environment containing
pytorch, horovod and deepspeed. You can *try* with:

.. code-block:: bash

    # Creates a Python venv called envAI_hdfml
    make torch-gpu-jsc


Distributed training
++++++++++++++++++++

Each distributed strategy has its own SLURM job script, which
should be used to run it:

If you want to distribute the code in ``train.py`` with **torch DDP**, run from terminal:
  
.. code-block:: bash

    export DIST_MODE="ddp"
    export RUN_NAME="ddp-itwinai"
    export TRAINING_CMD="train.py -s ddp"
    export PYTHON_VENV="../../../envAI_hdfml"
    sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
        --job-name="$RUN_NAME-n$N" \
        --output="logs_slurm/job-$RUN_NAME-n$N.out" \
        --error="logs_slurm/job-$RUN_NAME-n$N.err" \
        slurm.sh


If you want to distribute the code in ``train.py`` with **DeepSpeed**, run from terminal:
  
.. code-block:: bash

    export DIST_MODE="deepspeed"
    export RUN_NAME="deepspeed-itwinai"
    export TRAINING_CMD="train.py -s deepspeed"
    export PYTHON_VENV="../../../envAI_hdfml"
    sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
        --job-name="$RUN_NAME-n$N" \
        --output="logs_slurm/job-$RUN_NAME-n$N.out" \
        --error="logs_slurm/job-$RUN_NAME-n$N.err" \
        slurm.sh


If you want to distribute the code in ``train.py`` with **Horovod**, run from terminal:
  
.. code-block:: bash

    export DIST_MODE="deepspeed"
    export RUN_NAME="deepspeed-itwinai"
    export TRAINING_CMD="train.py -s deepspeed"
    export PYTHON_VENV="../../../envAI_hdfml"
    sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
        --job-name="$RUN_NAME-n$N" \
        --output="logs_slurm/job-$RUN_NAME-n$N.out" \
        --error="logs_slurm/job-$RUN_NAME-n$N.err" \
        slurm.sh


**You can run all of them with:**

.. code-block:: bash

    bash runall.sh


train.py
++++++++

.. literalinclude:: ../tutorials/distributed-ml/torch-tutorial-0-basics/train.py
   :language: python


runall.sh
+++++++++

.. literalinclude:: ../tutorials/distributed-ml/torch-tutorial-0-basics/runall.sh
   :language: bash


slurm.sh
++++++++

.. literalinclude:: ../tutorials/distributed-ml/torch-tutorial-0-basics/slurm.sh
   :language: bash

