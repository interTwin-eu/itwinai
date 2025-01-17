On HPC systems, it is common to manage dependencies using **Environment Modules**
or **Lmod**. These tools allow you to dynamically load and unload software
environments. If you are unsure which modules to load for your application, contact
your system administrator or refer to your HPC system's documentation for specific
guidance.

For Juelich Supercomputer (JSC) and Vega Supercomputer, here are the modules you
should load, depending on whether you want ``PyTorch`` or ``TensorFlow`` support:

.. tab-set::

    .. tab-item:: PyTorch

        For JSC:
        
        .. code-block:: bash

            ml --force purge
            ml Stages/2024 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
            ml Python CMake HDF5 PnetCDF libaio mpi4py

        For Vega:
        
        .. code-block:: bash

            ml --force purge
            ml Python/3.11.5-GCCcore-13.2.0 CMake/3.24.3-GCCcore-11.3.0 mpi4py OpenMPI CUDA/12.3
            ml GCCcore/11.3.0 NCCL cuDNN/8.9.7.29-CUDA-12.3.0 UCX-CUDA/1.15.0-GCCcore-13.2.0-CUDA-12.3.0

    .. tab-item:: TensorFlow

        For JSC:
        
        .. code-block:: bash

            ml --force purge
            ml Stages/2024 GCC/12.3.0 OpenMPI CUDA/12 MPI-settings/CUDA
            ml Python/3.11 HDF5 PnetCDF libaio mpi4py CMake cuDNN/8.9.5.29-CUDA-12

        For Vega:
        
        .. code-block:: bash

            ml --force purge
            ml Python/3.11.5-GCCcore-13.2.0 CMake/3.24.3-GCCcore-11.3.0 mpi4py OpenMPI CUDA/12.3
            ml GCCcore/11.3.0 NCCL cuDNN/8.9.7.29-CUDA-12.3.0 UCX-CUDA/1.15.0-GCCcore-13.2.0-CUDA-12.3.0
