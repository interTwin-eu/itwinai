# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

DEFAULT_SLURM_LOG_DIR = "slurm-job-logs"
DEFAULT_SLURM_SAVE_DIR = "slurm-scripts"
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

{pre_exec_command}

{exec_command}"""
