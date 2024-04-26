Getting started with itwinai
============================

In this section, we will run you through the installation and give some instructions for the use of the itwinai framework for HPC and local systems.


🌐 HPC systems
---------------
   
Here, we lay out how to use torch `DistributedDataParallel` (DDP), Horovod and DeepSpeed from the same client code.
Note that the environment is tested on the HDFML system at JSC. For other systems, the module versions might need change accordingly.


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

First, from the root of `this repository <https://github.com/interTwin-eu/itwinai/tree/distributed-strategy-launcher>`_, build the environment containing pytorch, horovod, and deepspeed. You can try with:

.. code-block:: bash

    # Creates a Python environment called envAI_hdfml
    make torch-gpu-jsc


Distributed training
++++++++++++++++++++

 Each distributed strategy is described with a SLURM job script used to run that strategy.

So if you want to distribute the code in `train.py` with, for example, **torch DDP**, run from terminal:

.. code-block:: bash

    sbatch ddp_slurm.sh

Similarly, if you want to distribute the code in `train.py` with **DeepSpeed**, run from terminal:

.. code-block:: bash

    sbatch deepspeed_slurm.sh

To distribute the code in `train.py` with **Horovod**, run from terminal:

.. code-block:: bash

    sbatch hvd_slurm.sh

Finally, you can run all of them with:

.. code-block:: bash

    bash runall.sh





💻 Local systems
-----------------

**Requirements**

* Linux environment. 

Windows and macOS were never tested.
   

Micromamba installation
+++++++++++++++++++++++

To manage Conda environments we use micromamba, a lightweight version of Conda.

In order to install micromamba, please refer to the `Manual installation guide <https://mamba.readthedocs.io/en/latest/micromamba-installation.html#umamba-install/>`_.

Consider that Micromamba can eat a lot of space when building environments because packages are cached on the local filesystem after being downloaded. To clear cache, you can use `micromamba clean -a`.
Micromamba data are kept under the `$HOME` location. However, in some systems, `$HOME` has a limited storage space so it is recommended to install Micromamba in another location with more storage space by changing the `$MAMBA_ROOT_PREFIX` variable. 
Below is a complete installation example where the default `$MAMBA_ROOT_PREFIX` is overridden for Linux:


.. code-block:: bash

    cd $HOME

    # Download micromamba (This command is for Linux Intel (x86_64) systems. Find the right one for your system!)
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

    # Install micromamba in a custom directory
    MAMBA_ROOT_PREFIX='my-mamba-root'
    ./bin/micromamba shell init $MAMBA_ROOT_PREFIX

    # To invoke micromamba from Makefile, you need to add explicitly to $PATH
    echo 'PATH="$(dirname $MAMBA_EXE):$PATH"' >> ~/.bashrc

**Reference**: `Micromamba installation guide <https://mamba.readthedocs.io/en/latest/installation.html#micromamba>`_.


Environment setup
+++++++++++++++++

**Requirements:**

* Linux environment. Windows and macOS were never tested.
* Micromamba: see the installation instructions above.
* VS Code, for development.

Tensorflow
++++++++++

Installation:

.. code-block:: bash

    # Install TensorFlow 2.13
    make tf-2.13

    # Activate env
    micromamba activate ./.venv-tf

Other TensorFlow versions are available, using the following targets `tf-2.10`, and `tf-2.11`.


PyTorch (+ Lightning)
+++++++++++++++++++++

Installation:

.. code-block:: bash

    # Install PyTorch + lightning
    make torch-gpu

    # Activate env
    micromamba activate ./.venv-pytorch

Other similarly CPU-only version is available at the target `torch-cpu`.


Development environment
+++++++++++++++++++++++

This is for developers only. To have it, update the installed `itwinai` package adding the `dev` extra:

.. code-block:: bash

    pip install -e .[dev]


**Test with `pytest`**
To run tests on itwinai package:

.. code-block:: bash

    # Activate env
    micromamba activate ./.venv-pytorch # or ./.venv-tf

    pytest -v -m "not slurm" tests/


However, some tests are intended to be executed only on HPC systems, where SLURM is available. They are marked with "slurm" tags. To run these tests, use the dedicated job script:

.. code-block:: bash

    sbatch tests/slurm_tests_startscript

    # Upon completion, check the output:
    cat job.err
    cat job.out




