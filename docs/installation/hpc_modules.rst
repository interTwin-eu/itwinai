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
                    ml Stages/2024 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
                    ml Python CMake HDF5 PnetCDF libaio mpi4py

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
                    ml Python/3.11.5-GCCcore-13.2.0 CMake/3.24.3-GCCcore-11.3.0 mpi4py OpenMPI CUDA/12.3
                    ml GCCcore/11.3.0 NCCL cuDNN/8.9.7.29-CUDA-12.3.0 UCX-CUDA/1.15.0-GCCcore-13.2.0-CUDA-12.3.0

                    # Now you can create or active the python environment here

            .. tab-item:: TensorFlow

                .. code-block:: bash

                    ml --force purge
                    ml Python/3.11.5-GCCcore-13.2.0 CMake/3.24.3-GCCcore-11.3.0 mpi4py OpenMPI CUDA/12.3
                    ml GCCcore/11.3.0 NCCL cuDNN/8.9.7.29-CUDA-12.3.0 UCX-CUDA/1.15.0-GCCcore-13.2.0-CUDA-12.3.0

                    # Now you can create or active the python environment here


    .. tab-item:: LUMI

        On `LUMI <https://docs.lumi-supercomputer.eu/hardware/lumig/>`_, Python virtual
        environments are not allowed, in favour of containers. Therefore,
        the software modules are a bit different (WIP).