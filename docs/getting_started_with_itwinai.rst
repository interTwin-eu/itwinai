Getting started with itwinai
============================

In this section, we will run you through the installation and give some instructions for the use of the itwinai framework for HPC and local systems.


User Installation
-----------------

Requirements:

- Linux environment. Windows and macOS were never tested.

Python virtual environment
++++++++++++++++++++++++++

Depending on your environment, there are different ways to
select a specific python version.

💻 Laptop or GPU node
++++++++++++++++++++++

If you are working on a laptop
or on a simple on-prem setup, you could consider using
`pyenv <https://github.com/pyenv/pyenv>`_. See the
`installation instructions <https://github.com/pyenv/pyenv?tab=readme-ov-file#installation>`_. If you are using pyenv,
make sure to read `this <https://github.com/pyenv/pyenv/wiki#suggested-build-environment>`_.

🌐 HPC environment
+++++++++++++++++++

In HPC systems it is more popular to load dependencies using
Environment Modules or Lmod. Contact the system administrator
to learn how to select the proper python modules.

On JSC, we activate the required modules in this way:

.. code-block:: bash

    ml --force purge
    ml Stages/2024 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
    ml Python CMake HDF5 PnetCDF libaio mpi4py


Install itwinai
+++++++++++++++

Install itwinai and its dependencies using the
following command, and follow the instructions:

.. code-block:: bash

    # Create a python virtual environment and activate it
    $ python -m venv ENV_NAME
    $ source ENV_NAME/bin/activate

    # Install itwinai inside the environment
    (ENV_NAME) $ export ML_FRAMEWORK="pytorch" # or "tensorflow"
    (ENV_NAME) $ curl -fsSL https://github.com/interTwin-eu/itwinai/raw/main/env-files/itwinai-installer.sh | bash


The ``ML_FRAMEWORK`` environment variable controls whether you are installing
itwinai for PyTorch or TensorFlow.

.. warning::
   itwinai depends on Horovod, which requires ``CMake>=1.13`` and 
   `other packages <https://horovod.readthedocs.io/en/latest/install_include.html#requirements>`_. 
   Make sure to have them installed in your environment before proceeding.



Developer Installation
----------------------

If you are contributing to this repository, please continue below for
more advanced instructions.

.. warning::
   Branch protection rules are applied to all branches which names 
   match this regex: ``[dm][ea][vi]*`` . When creating new branches, 
   please avoid using names that match that regex.


Install itwinai environment
+++++++++++++++++++++++++++

Regardless of how you loaded your environment, you can create the
python virtual environments with the following commands.
Once the correct Python version is loaded, create the virtual
environments using our pre-make Makefile:

.. code-block:: bash

    make torch-env # or make torch-env-cpu
    make tensorflow-env # or make tensorflow-env-cpu

    # Juelich supercomputer
    make torch-gpu-jsc
    make tf-gpu-jsc


TensorFlow
++++++++++

Installation:

.. code-block:: bash

    # Install TensorFlow 2.13
    make tensorflow-env

    # Activate env
    source .venv-tf/bin/activate


A CPU-only version is available at the target ``tensorflow-env-cpu``.

PyTorch (+ Lightning)
+++++++++++++++++++++

Installation:

.. code-block:: bash

    # Install PyTorch + lightning
    make torch-env

    # Activate env
    source .venv-pytorch/bin/activate


A CPU-only version is available at the target ``torch-env-cpu``.

Development environment
+++++++++++++++++++++++

This is for developers only. To have it, update the installed ``itwinai`` package
adding the ``dev`` extra:

.. code-block:: bash

    pip install -e .[dev]


Test with ``pytest``
++++++++++++++++++++

Do this only if you are a developer wanting to test your code with pytest.

First, you need to create virtual environments both for torch and tensorflow.
For instance, you can use:

.. code-block:: bash

    make torch-env-cpu
    make tensorflow-env-cpu


To select the name of the torch and tf environments you can set the following
environment variables, which allow to run the tests in environments with
custom names which are different from ``.venv-pytorch`` and ``.venv-tf``.

.. code-block:: bash

    export TORCH_ENV="my_torch_env"
    export TF_ENV="my_tf_env"


Functional tests (marked with ``pytest.mark.functional``) will be executed under
``/tmp/pytest`` location to guarantee they are run in a clean environment.

To run functional tests use:

.. code-block:: bash

    pytest -v tests/ -m "functional"


To run all tests on itwinai package:

.. code-block:: bash

    make test


Run tests in JSC virtual environments:

.. code-block:: bash

    make test-jsc








.. 🌐 HPC systems
.. ---------------
   
.. Here, we lay out how to use torch DistributedDataParallel (DDP), Horovod, and DeepSpeed from the same client code.
.. Note that the environment is tested on the HDFML system at JSC. For other systems, the module versions might need change accordingly.


.. Environments
.. ++++++++++++

.. Install PyTorch env (GPU support) on Juelich Super Computer (tested on HDFML system)

.. .. code-block:: bash

..     torch-gpu-jsc: env-files/torch/createEnvJSC.sh
..     sh env-files/torch/createEnvJSC.sh


.. Install Tensorflow env (GPU support) on Juelich Super Computer (tested on HDFML system)

.. .. code-block:: bash

..     tf-gpu-jsc: env-files/tensorflow/createEnvJSCTF.sh
..     sh env-files/tensorflow/createEnvJSCTF.sh


.. Setup
.. +++++

.. First, from the root of `this repository <https://github.com/interTwin-eu/itwinai/tree/distributed-strategy-launcher>`_, build the environment containing pytorch, horovod, and deepspeed. You can try with:

.. .. code-block:: bash

..     # Creates a Python environment called envAI_hdfml
..     make torch-gpu-jsc


.. Distributed training
.. ++++++++++++++++++++

..  Each distributed strategy is described with a SLURM job script used to run that strategy.

.. So if you want to distribute the code in `train.py` with, for example, **torch DDP**, run from terminal:

.. .. code-block:: bash

..     sbatch ddp_slurm.sh

.. Similarly, if you want to distribute the code in `train.py` with **DeepSpeed**, run from terminal:

.. .. code-block:: bash

..     sbatch deepspeed_slurm.sh

.. To distribute the code in `train.py` with **Horovod**, run from terminal:

.. .. code-block:: bash

..     sbatch hvd_slurm.sh

.. Finally, you can run all of them with:

.. .. code-block:: bash

..     bash runall.sh





.. 💻 Local systems
.. -----------------

.. **Requirements**

.. * Linux environment. 

.. Windows and macOS were never tested.
   

.. Micromamba installation
.. +++++++++++++++++++++++

.. To manage Conda environments we use micromamba, a lightweight version of Conda.

.. In order to install micromamba, please refer to the `Manual installation guide <https://mamba.readthedocs.io/en/latest/micromamba-installation.html#umamba-install/>`_.

.. Consider that Micromamba can eat a lot of space when building environments because packages are cached on the local filesystem after being downloaded. To clear cache, you can use `micromamba clean -a`.
.. Micromamba data are kept under the `$HOME` location. However, in some systems, `$HOME` has a limited storage space so it is recommended to install Micromamba in another location with more storage space by changing the `$MAMBA_ROOT_PREFIX` variable. 
.. Below is a complete installation example where the default `$MAMBA_ROOT_PREFIX` is overridden for Linux:


.. .. code-block:: bash

..     cd $HOME

..     # Download micromamba (This command is for Linux Intel (x86_64) systems. Find the right one for your system!)
..     curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

..     # Install micromamba in a custom directory
..     MAMBA_ROOT_PREFIX='my-mamba-root'
..     ./bin/micromamba shell init $MAMBA_ROOT_PREFIX

..     # To invoke micromamba from Makefile, you need to add explicitly to $PATH
..     echo 'PATH="$(dirname $MAMBA_EXE):$PATH"' >> ~/.bashrc

.. **Reference**: `Micromamba installation guide <https://mamba.readthedocs.io/en/latest/installation.html#micromamba>`_.


.. Environment setup
.. +++++++++++++++++

.. **Requirements:**

.. * Linux environment. Windows and macOS were never tested.
.. * Micromamba: see the installation instructions above.
.. * VS Code, for development.

.. Tensorflow
.. ++++++++++

.. Installation:

.. .. code-block:: bash

..     # Install TensorFlow 2.13
..     make tf-2.13

..     # Activate env
..     micromamba activate ./.venv-tf

.. Other TensorFlow versions are available, using the following targets `tf-2.10`, and `tf-2.11`.


.. PyTorch (+ Lightning)
.. +++++++++++++++++++++

.. Installation:

.. .. code-block:: bash

..     # Install PyTorch + lightning
..     make torch-gpu

..     # Activate env
..     micromamba activate ./.venv-pytorch

.. Other similarly CPU-only version is available at the target `torch-cpu`.


.. Development environment
.. +++++++++++++++++++++++

.. This is for developers only. To have it, update the installed `itwinai` package adding the `dev` extra:

.. .. code-block:: bash

..     pip install -e .[dev]


.. **Test with `pytest`**
.. To run tests on itwinai package:

.. .. code-block:: bash

..     # Activate env
..     micromamba activate ./.venv-pytorch # or ./.venv-tf

..     pytest -v -m "not slurm" tests/


.. However, some tests are intended to be executed only on HPC systems, where SLURM is available. They are marked with "slurm" tags. To run these tests, use the dedicated job script:

.. .. code-block:: bash

..     sbatch tests/slurm_tests_startscript

..     # Upon completion, check the output:
..     cat job.err
..     cat job.out




