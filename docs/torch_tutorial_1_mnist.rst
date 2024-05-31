PyTorch MNIST example
=====================

Tutorial: distributed strategies for PyTorch model trained on MNIST dataset
---------------------------------------------------------------------------

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


Before launching training, since on JSC's compute nodes there is not internet connection,
you need to download the dataset before while on the login lode:

.. code-block:: bash

    source ../../../envAI_hdfml/bin/activate
    python train.py --download-only


This command creates a local folder called "MNIST" with the dataset.

Distributed training
++++++++++++++++++++

Each distributed strategy has its own SLURM job script, which
should be used to run it:

If you want to distribute the code in ``train.py`` with **torch DDP**, run from terminal:
  
.. code-block:: bash

    export DIST_MODE="ddp"
    export RUN_NAME="ddp-itwinai"
    export TRAINING_CMD="train.py -s ddp -c config.yaml"
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
    export TRAINING_CMD="train.py -s deepspeed -c config.yaml"
    export PYTHON_VENV="../../../envAI_hdfml"
    sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
        --job-name="$RUN_NAME-n$N" \
        --output="logs_slurm/job-$RUN_NAME-n$N.out" \
        --error="logs_slurm/job-$RUN_NAME-n$N.err" \
        slurm.sh


If you want to distribute the code in ``train.py`` with **Horovod**, run from terminal:
  
.. code-block:: bash

    export DIST_MODE="horovod"
    export RUN_NAME="horovod-itwinai"
    export TRAINING_CMD="train.py -s horovod -c config.yaml"
    export PYTHON_VENV="../../../envAI_hdfml"
    sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
        --job-name="$RUN_NAME-n$N" \
        --output="logs_slurm/job-$RUN_NAME-n$N.out" \
        --error="logs_slurm/job-$RUN_NAME-n$N.err" \
        slurm.sh


**You can run all of them with:**

.. code-block:: bash

    bash runall.sh


config.yaml
+++++++++++

.. literalinclude:: ../tutorials/distributed-ml/torch-tutorial-1-mnist/config.yaml
   :language: yaml


runall.sh
+++++++++

.. literalinclude:: ../tutorials/distributed-ml/torch-tutorial-1-mnist/runall.sh
   :language: bash


slurm.sh
++++++++

.. literalinclude:: ../tutorials/distributed-ml/torch-tutorial-1-mnist/slurm.sh
   :language: bash


train.py
++++++++

.. literalinclude:: ../tutorials/distributed-ml/torch-tutorial-1-mnist/train.py
   :language: python

