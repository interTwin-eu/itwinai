User Installation (for non-developers)
======================================
**Author(s)**: Jarl Sondre Sæther (CERN)

This guide provides step-by-step instructions for installing the ``itwinai`` library for
users. 

Requirements: Linux or macOS environment. Windows is not supported. 

Creating a Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
While not mandatory, creating a virtual environment is highly recommended to isolate
dependencies and prevent conflicts with other Python projects.

.. warning::

    On high-performance computing (HPC) systems, you must load the appropriate modules
    before activating your virtual environment to ensure compatibility with system
    libraries. See the dropdown below for detailed instructions:

    .. dropdown:: HPC Systems

       .. include:: ./hpc_modules.rst


If you don't already have a virtual environment, you can create one with the following
command:

.. code-block:: bash 

   python -m venv <name-of-venv>

Remember to replace ``<name-of-venv>`` with the name you want for your virtual
environment. Now, you can start your virtual environment with the following command: 

.. code-block:: bash 

   source <name-of-venv>/bin/activate


Installing the ``itwinai`` library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can choose if you want to install ``itwinai`` with support for either PyTorch or
TensorFlow by using extras:

.. tab-set:: 

    .. tab-item:: PyTorch

        To install ``itwinai`` with PyTorch without GPU acceleration, you can use the
        following command:

        .. code-block:: bash
            
            pip install "itwinai[torch]"

        To enable GPU acceleration, you can use the following command:

        .. code-block:: bash

            pip install ".[torch]" \
                --extra-index-url https://download.pytorch.org/whl/cu121


    .. tab-item:: TensorFlow

        To install ``itwinai`` with TensorFlow without GPU acceleration, you can use the
        following command:

        .. code-block:: bash
            
            pip install "itwinai[tf]"

        To enable GPU acceleration, you can use the following command:

        .. code-block:: bash

            pip install ".[tf-cuda]"


.. note:: 
    If you want to use the Prov4ML logger, you need to install it explicitly since it is only
    available on GitHub:

    For systems with Nvidia GPUs:

    .. code-block:: bash

       pip install "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@new-main"

    For macOS:

    .. code-block:: bash

       pip install "prov4ml[apple]@git+https://github.com/matbun/ProvML@new-main"

Installing Horovod and Microsoft DeepSpeed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you also want to install Horovod and Microsoft DeepSpeed for distributed ML with
PyTorch, then make sure to install them **after** ``itwinai``. You can do this with the
following command:

.. code-block:: bash

    curl -fsSL https://github.com/interTwin-eu/itwinai/raw/main/env-files/torch/install-horovod-deepspeed-cuda.sh | bash

.. warning::
   
   Horovod requires ``CMake>=1.13`` and 
   `other packages <https://horovod.readthedocs.io/en/latest/install_include.html#requirements>`_
   Make sure to have them installed in your environment before proceeding.
