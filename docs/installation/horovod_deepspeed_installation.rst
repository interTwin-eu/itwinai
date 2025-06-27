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
