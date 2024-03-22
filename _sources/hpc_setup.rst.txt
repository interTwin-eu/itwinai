🌐 HPC systems
---------------
How to use torch `DistributedDataParallel` (DDP), Horovod and DeepSpeed from the same client code.
Note that the environment is tested on the HDFML system at JSC. For other systems, the module versions might need change accordingly.


.. toctree::
   :maxdepth: 5


Environments
++++++++++++

Install PyTorch env (GPU support) on Juelich Super Computer (tested on HDFML system)

.. code-block:: bash

    torch-gpu-jsc: env-files/torch/createEnvJSC.sh
    sh env-files/torch/createEnvJSC.sh


Install Tensorflow env (GPU support) on Juelich Super Computer (tested on HDFML system)

.. code-block:: bash

    tf-gpu-jsc: env-files/tensorflow/createEnvJSCTF.sh
    sh env-files/tensorflow/createEnvJSCTF.sh



Setup
+++++

First, from the root of this `repository <https://github.com/interTwin-eu/itwinai/tree/distributed-strategy-launcher>`_, build the environment containing pytorch, horovod and deepspeed. You can try with:

.. code-block:: bash

    # Creates a Python venv called envAI_hdfml
    make torch-gpu-jsc


Distributed training
++++++++++++++++++++

Each distributed strategy has its own SLURM job script, which should be used to run it:

If you want to distribute the code in `train.py` with **torch DDP**, run from terminal:

.. code-block:: bash

    sbatch ddp_slurm.sh

If you want to distribute the code in `train.py` with **DeepSpeed**, run from terminal:

.. code-block:: bash

    sbatch deepspeed_slurm.sh

If you want to distribute the code in `train.py` with **Horovod**, run from terminal:

.. code-block:: bash

    sbatch hvd_slurm.sh

You can run all of them with:

.. code-block:: bash

    bash runall.sh