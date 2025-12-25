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
DEFAULT_PY_SPY_DIR = "py-spy-output"
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
#SBATCH --ntasks={num_tasks}
#SBATCH --ntasks-per-node={num_tasks_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --mem={memory}
{exclusive_line}

# Make sure that Hydra produces the whole output
export HYDRA_FULL_ERROR=1

# Set itwinai log level to "INFO" by default. For richer logs use "DEBUG"
export ITWINAI_LOG_LEVEL=INFO

{pre_exec_command}

{exec_command}
"""
