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

# Resources allocation
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={num_tasks_per_node}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --cpus-per-gpu={cpus_per_gpu}
#SBATCH --exclusive

{pre_exec_command}

{exec_command}"""

JUWELS_HPC_MODULES = [
    "Stages/2024",
    "GCC",
    "OpenMPI",
    "CUDA/12",
    "MPI-settings/CUDA",
    "Python/3.11.3",
    "HDF5",
    "PnetCDF",
    "libaio",
    "mpi4py",
]
