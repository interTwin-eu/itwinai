Installing Horovod and Microsoft DeepSpeed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you also want to install Horovod and Microsoft DeepSpeed for distributed ML with
PyTorch, then make sure to install them **after** ``itwinai``. You can choose if you
want to do this with or without GPU (CUDA) support: 

.. tab-set:: 

    .. tab-item:: CPU

        .. code-block:: bash

            pip install --no-cache-dir --no-build-isolation git+https://github.com/horovod/horovod.git
            pip install --no-cache-dir --no-build-isolation deepspeed

    
    .. tab-item:: CUDA

        .. code-block:: bash

            curl -fsSL https://github.com/interTwin-eu/itwinai/raw/main/env-files/torch/install-horovod-deepspeed-cuda.sh | bash


.. warning::
   
   Horovod requires ``CMake>=1.13`` and 
   `other packages <https://horovod.readthedocs.io/en/latest/install_include.html#requirements>`_
   Make sure to have them installed in your environment before proceeding.

.. warning::

   If you run the installation script for CUDA above, then make sure that you actually have
   CUDA installed. For some HPC systems, we found it better to install this via a SLURM script
   since they only have CUDA on their compute nodes. You can find a sample SLURM script that
   we use for the Jülich Supercomputing Centre (JSC) here:
   `horovod-deepspeed-JSC.slurm <https://github.com/interTwin-eu/itwinai/blob/main/env-files/torch/horovod-deepspeed-JSC.slurm>`_.
