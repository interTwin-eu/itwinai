On HPC systems, it is common to manage dependencies using **Environment Modules**
or **Lmod**. These tools allow you to dynamically load and unload software
modules, such as compilers, CUDA drivers, and MPI libraries. If you are unsure which modules
to load for your application, contact your system administrator or refer to your HPC system's
documentation for specific guidance.

Find below the modules you should load on the supercomputers where we tested itwinai, depending
on whether you want ``PyTorch`` or ``TensorFlow`` support. If you are deploying itwinai on a
different HPC system, please refer to the **Other HPCs** tab.


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

        Modules for `Vega <https://doc.vega.izum.si/introduction/>`_ Supercomputer.

        When installing the environment from the login node, make sure that the
        CUDA drivers are loaded correctly and the GPU is visible by running the
        `nvidia-smi` command. This is very important for a successful
        installation of DeepSpeed and Horovod. If the GPU is not correctly
        visualized, consider logging-in again to another login node.
        Alternatively, consider running the installation on a compute node.

        .. tab-set::

            .. tab-item:: PyTorch

                .. code-block:: bash

                    ml --force purge
                    ml CMake/3.29.3-GCCcore-13.3.0
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
                    ml OpenMPI/4.1.6-GCC-13.2.0
                    ml cuDNN/8.9.7.29-CUDA-12.3.0
                    ml CUDA/12.6.0
                    ml NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
                    ml Python/3.12.3-GCCcore-13.3.0

                    # Now you can create or active the python environment here

        Currently, the latest version of ``mpi4py`` on Vega is not compatible with Python 3.12,
        therefore you'll have to build it yourself in your python environment:

            .. code-block:: bash

                # Create the venv
                uv venv

                uv pip install --no-cache-dir --force-reinstall --no-binary=mpi4py mpi4py


    .. tab-item:: LUMI

        On `LUMI <https://docs.lumi-supercomputer.eu/hardware/lumig/>`_, Python virtual
        environments are discouraged in favour of containers as they create a large number
        of files, which affect the performances of the distributed storage system.
        Load the following modules before running commands in your AI containers:

            .. code-block:: bash

                ml --force purge
                ml LUMI partition/G
                module use /appl/local/containers/ai-modules
                module load singularity-AI-bindings

        These modules are needed to bind into the container the correct software suite on LUMI.
        More info can be found `here <https://lumi-supercomputer.github.io/LUMI-training-materials/ai-20250204/extra_05_RunningContainers/>`_.


    .. tab-item:: Other HPCs

        Module names and packaging conventions vary across HPC centres, but the underlying
        software requirements for running ``itwinai`` are usually the same. The goal is to
        load a consistent toolchain (compiler + MPI), a Python runtime, and (optionally) the
        GPU communication stack.

        The version numbers below are **reference values** known to work on at least one
        modern HPC software stack; other versions may work as well.

        **Recommended baseline (reference versions)**

        * **Python**: >= 3.12 (e.g., 3.12.3)
        * **CMake**: >= 3.29 (e.g., 3.29.3) — required to compile Horovod from source
        * **git**: recent (e.g., 2.45.x)

        **Compiler + MPI (ABI compatibility matters)**

        * **GCC**: a recent major release (e.g., 13.3.0)
        * **MPI**: a compatible MPI implementation (e.g., OpenMPI 5.0.x)
        * **mpi4py**: built/installed against the *same* MPI you loaded (e.g., 4.0.1)

        Many sites provide these as a single toolchain module (e.g., a compiler+MPI bundle).
        That is fine as long as the compiler and MPI are internally compatible.

        **GPU software stack (only if you use GPUs)**

        * **CUDA toolkit/runtime**: >= 12.6 (e.g., 12.6)
        * **cuDNN**: a CUDA-matched build (e.g., 9.5.0.* for CUDA 12)
        * **NCCL**: recommended for multi-GPU communication (version typically follows CUDA)

        Notes:

        * For **PyTorch**, it is often possible to install wheels built for a CUDA version that
          differs from the system CUDA. However, **DeepSpeed** is more sensitive: it generally
          requires **compatibility between the CUDA runtime on the system and the CUDA version
          used by the installed PyTorch build**. In ``itwinai`` this is typically handled by
          selecting a PyTorch build compatible with your target CUDA (the project pins the
          PyTorch version in ``pyproject.toml``).
        * Some systems expose a separate module/setting to enable **CUDA-aware MPI**
          communication (distinct from the CUDA toolkit itself). If your site provides such a
          module, load it when running distributed GPU workloads.

        **Common scientific I/O libraries (as needed by your workflow/plugins)**

        * **HDF5**: (e.g., 1.14.x)
        * **PnetCDF** (Parallel NetCDF): (e.g., 1.13.x)

        **DeepSpeed optional dependency**

        * **libaio**: (e.g., 0.3.113) — enables asynchronous disk I/O for certain DeepSpeed
          features (optimizer states / checkpointing). DeepSpeed can still work without it,
          but some features may be disabled. If you do not have ``libaio`` on your system,
          consider disabling AIO at build time (e.g., by not enabling ``DS_BUILD_AIO``).
          This setting is activated by default in our
          `DeepSpeed installation script <https://github.com/interTwin-eu/itwinai/blob/main/env-files/torch/install-horovod-deepspeed-cuda.sh>`_.

        A *typical* starting point (adapt module names to your site) is:

            .. code-block:: bash

                ml --force purge

                # Toolchain + MPI (or a site-provided toolchain module)
                ml <gcc-or-toolchain> <mpi>

                # Python runtime + build tools
                ml <python> <cmake> <git>

                # Optional: CUDA-aware MPI setting (only if your site provides it)
                ml <cuda-aware-mpi-setting>

                # Common HPC libraries (as needed by your workflow/plugins)
                ml <hdf5> <pnetcdf> <mpi4py>

                # GPU stack (only if running on GPUs)
                ml <cuda> <cudnn> <nccl>

                # Optional (only if needed by your setup)
                ml <libaio>

        .. note::
        
            While itwinai does not strictly require specific versions of CUDA, MPI
            or related libraries, deploying on a new HPC system can still expose
            compatibility issues (e.g., between the CUDA runtime, the framework
            build, and MPI/toolchain choices). We therefore encourage users to try a
            small set of software versions when setting up a new environment.
            Support for additional systems depends on contributor availability, but
            we welcome reports and improvements: please open a GitHub issue to share
            your findings, or submit a pull request with documentation updates that
            worked on your platform.

After using the commands above to load the modules, check which modules you loaded by running
the ``ml`` command in the terminal.
