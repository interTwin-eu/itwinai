Developer Installation
======================

This guide provides step-by-step instructions for installing the ``itwinai`` library for
developers. 

Cloning the Repository
~~~~~~~~~~~~~~~~~~~~~~
When cloning the repository, you have to make sure to also clone the submodules. You can
do both with the following command: 

.. code-block:: bash

  git clone [--recurse-submodules] git@github.com:interTwin-eu/itwinai.git


.. The explanation for creating a venv is the same for developers and users
.. include:: ./creating_venv.rst

Installing the ``itwinai`` Library as a Developer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In this project, we use ``uv`` as a project-wide package manager. Therefore, we suggest
that you skim through the :doc:`uv tutorial </installation/uv_tutorial>` before
continuing this tutorial.

Optional Dependencies (extras)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``itwinai`` library has numerous optional dependencies that can be activated when
installing with ``pip`` through extras: 

* ``dev``: for developers, including libraries for running tests etc.
* ``torch``: for installation with PyTorch support.
* ``tf``: for installation with TensorFlow support.
* ``tf-cuda``: for installation with TensorFlow support with GPU acceleration enabled.
* ``docs``: for installation of packages required to build the docs locally.
* ``hpo``: for installation of packages used for hyperparameter optimization (HPO).

You can at any point in time find (or update) the optional dependencies in the
``pyproject.toml`` file in the root of the repository. 

Installing the Library Itself
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As a developer, you will also install the library using ``pip`` (or ``uv pip`` if you
wish), but the main difference is that you need to install it as *editable* using the
``-e`` flag. Another difference is that you also need the ``dev`` extra. 

.. note:: 

   When installing on HPC, it is sometimes an advantage to use the ``--no-cache-dir``
   option to avoid filling up your ``~/.cache`` directory. Filling up this directory
   will often lead to you use up your disk quota, especially in terms of inodes. 

Below you can find complete commands for installation, depending on if you are installing
``itwinai`` with or without GPU (CUDA) support and locally or on HPC:

.. tab-set:: 

    .. tab-item:: Local (CPU)
    
        .. code-block:: bash

            pip install -e ".[torch,dev,tf]"

    
    .. tab-item:: Local (CUDA)

        .. code-block:: bash

            pip install -e ".[torch,dev,tf]" \
                --extra-index-url https://download.pytorch.org/whl/cu121


    .. tab-item:: HPC (CUDA)

        Note: This is the same as ``Local (CUDA)`` but without using the cache directory.

        .. code-block:: bash

            pip install -e ".[torch,dev,tf]" \
                --no-cache-dir \
                --extra-index-url https://download.pytorch.org/whl/cu121


.. note:: 
    If you want to use the Prov4ML logger, you need to install it explicitly since it is only
    available on GitHub:

    For systems with Nvidia GPUs:

    .. code-block:: bash

       pip install "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@new-main"

    For macOS:

    .. code-block:: bash

       pip install "prov4ml[apple]@git+https://github.com/matbun/ProvML@new-main"


.. The explanation for installing horovod and DS is the same for developers and users
.. include:: ./horovod_deepspeed_installation.rst
