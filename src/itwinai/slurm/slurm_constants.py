# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

SLURM_TEMPLATE = r"""#!/bin/bash

# Job configuration
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --time={time}

#SBATCH --output={std_out}
#SBATCH --error={err_out}

# Resource allocation
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={num_tasks_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --exclusive

# Pre-execution command
{pre_exec_command}

# Job execution command
{exec_command}"""

JUWELS_HPC_MODULES = [
    "Stages/2025",
    "GCC",
    "OpenMPI",
    "CUDA/12",
    "cuDNN",
    "MPI-settings/CUDA",
    "Python",
    "CMake",
    "HDF5",
    "PnetCDF",
    "libaio",
    "mpi4py",
    "git",
]
