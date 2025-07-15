.. note:: 
    If you want to use the Prov4ML logger, you need to install it explicitly since it is only
    available on GitHub:

    For systems with Nvidia GPUs:

    .. code-block:: bash

       uv pip install "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@v0.0.1"

    For macOS:

    .. code-block:: bash

       uv pip install "prov4ml[apple]@git+https://github.com/matbun/ProvML@v0.0.1"


Installing Horovod and Microsoft DeepSpeed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you also want to install Horovod and Microsoft DeepSpeed for distributed ML with
PyTorch, then make sure to install them **after** ``itwinai``. You can choose if you
want to do this with or without GPU (CUDA) support: 

.. tab-set:: 

    .. tab-item:: CPU

        .. code-block:: bash

            uv pip install --no-cache-dir --no-build-isolation git+https://github.com/horovod/horovod.git
            uv pip install --no-cache-dir --no-build-isolation deepspeed

    
    .. tab-item:: CUDA

        .. code-block:: bash

            curl -fsSL https://github.com/interTwin-eu/itwinai/raw/main/env-files/torch/install-horovod-deepspeed-cuda.sh | bash


.. warning::
   
    Horovod requires ``CMake>=1.13`` and 
    `other packages <https://horovod.readthedocs.io/en/latest/install_include.html#requirements>`_
    Make sure to have them installed in your environment before proceeding.


.. warning::
   The installation of Horovod and DeepSpeed needs to be executed on a machine/node where GPUs
   are available. On some HPC systems, such as the `JUWELS <https://apps.fz-juelich.de/jsc/hps/juwels/configuration.html>`_
   system on JSC, GPUs **are not available on login nodes** (the host you connect to when you
   SSH into the system), only on **compute nodes**. On the JUWELS system, run this command to
   install DeepSpeed and Horovod directly **from the repository's root**:

        .. code-block:: bash

            curl -fsSL https://github.com/interTwin-eu/itwinai/raw/main/env-files/torch/horovod-deepspeed-JSC.slurm | sbatch

