On HPC systems, it is common to manage dependencies using **Environment Modules**
or **Lmod**. These tools allow you to dynamically load and unload software
modules, such as compilers, CUDA drivers, and MPI libraries. If you are unsure which modules
to load for your application, contact your system administrator or refer to your HPC system's
documentation for specific guidance.

Find below the modules you should load on the supercomputers where we tested itwinai, depending
on whether you want ``PyTorch`` or ``TensorFlow`` support:


.. tab-set::

    .. tab-item:: Juelich (JSC)

        Modules for the `JUWELS <https://apps.fz-juelich.de/jsc/hps/juwels/configuration.html>`_ 
        system at Juelich Supercomputer (JSC):

        .. tab-set::

            .. tab-item:: PyTorch

                .. code-block:: bash

                    ml --force purge
                    ml Stages/2025 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
                    ml Python CMake HDF5 PnetCDF libaio mpi4py git

                    # Now you can create or active the python environment here

            .. tab-item:: TensorFlow

                .. code-block:: bash

                    ml --force purge
                    ml Stages/2024 GCC/12.3.0 OpenMPI CUDA/12 MPI-settings/CUDA
                    ml Python/3.11 HDF5 PnetCDF libaio mpi4py CMake cuDNN/8.9.5.29-CUDA-12

                    # Now you can create or active the python environment here

    .. tab-item:: Vega

        Modules for `Vega <https://doc.vega.izum.si/introduction/>`_ Supercomputer:

        .. tab-set::

            .. tab-item:: PyTorch

                .. code-block:: bash

                    ml --force purge
                    ml CMake/3.29.3-GCCcore-13.3.0
                    ml mpi4py/3.1.5
                    ml OpenMPI/4.1.6-GCC-13.2.0
                    ml cuDNN/8.9.7.29-CUDA-12.3.0
                    ml CUDA/12.6.0
                    ml NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
                    ml Python/3.12.3-GCCcore-13.3.0

                    # Now you can create or active the python environment here


            .. tab-item:: TensorFlow

                .. code-block:: bash

                    ml --force purge
                    ml CMake/3.29.3-GCCcore-13.3.0
                    ml mpi4py/3.1.5
                    ml OpenMPI/4.1.6-GCC-13.2.0
                    ml cuDNN/8.9.7.29-CUDA-12.3.0
                    ml CUDA/12.6.0
                    ml NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
                    ml Python/3.12.3-GCCcore-13.3.0

                    # Now you can create or active the python environment here


    .. tab-item:: LUMI

        On `LUMI <https://docs.lumi-supercomputer.eu/hardware/lumig/>`_, Python virtual
        environments are discouraged in favour of containers. Load the following modules before
        running commands in your AI containers:

            .. code-block:: bash

                ml --force purge
                ml LUMI partition/G
                module use /appl/local/containers/ai-modules
                module load singularity-AI-bindings

        These modules are needed to bind into the container the correct software suite on LUMI.
        More info can be found `here <https://lumi-supercomputer.github.io/LUMI-training-materials/ai-20250204/extra_05_RunningContainers/>`_.

After using the commands above to load the modules, check which modules you loaded by running
the ``ml`` command in the terminal.