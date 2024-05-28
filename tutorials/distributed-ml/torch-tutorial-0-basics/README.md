# Tutorial: distributed strategies for PyTorch

In this tutorial we show how to use torch `DistributedDataParallel` (DDP), Horovod and
DeepSpeed from the same client code.
Note that the environment is tested on the HDFML system at JSC. For other systems,
the module versions might need change accordingly.

## Setup

First, from the root of this repository, build the environment containing
pytorch, horovod and deepspeed. You can *try* with:

```bash
# Creates a Python venv called envAI_hdfml
make torch-gpu-jsc
```

## Distributed training on a single node (interactive)

If you want to use SLURM in interactive mode, do the following:

```bash
# Allocate resources
$ salloc --partition=batch --nodes=1 --account=intertwin  --gres=gpu:4 --time=1:59:00
job ID is XXXX
# Get a shell in the compute node (if using SLURM)
$ srun --jobid XXXX --overlap --pty /bin/bash 
# Now you are inside the compute node

# On JSC, you may need to load some modules...
ml --force purge
ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py

# ...before activating the Python environment (adapt this to your env name/path)
source ../../../envAI_hdfml/bin/activate
```

To launch the training with torch DDP use:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=gpu train.py -s ddp

# Optional -- from a SLURM login node:
srun --jobid XXXX --ntasks-per-node=1 torchrun --standalone --nnodes=1 --nproc-per-node=gpu train.py -s ddp
```

To launch the training with Microsoft DeepSpeed use:

```bash
deepspeed train.py -s deepspeed --deepspeed

# Optional -- from a SLURM login node:
srun --jobid XXXX --ntasks-per-node=1 deepspeed train.py -s deepspeed --deepspeed 
```

To launch the training with Horovod use:

> [!NOTE]  
> NOTE: Assuming 4 GPUs are available.

If your setup has a different number of GPUs, change the `-np 4 -H localhost:4` part.

> [!WARNING]  
> To use `horovodrun`, make sure that `mpirun` is available in your environment. Otherwise
> you cannot use Horovod in interactive mode.

```bash
# Assuming 4 GPUs are available (-np=4)
horovodrun -np 4 -H localhost:4 train.py -s horovod

# Optional -- from a SLURM login node:
srun --jobid XXXX --ntasks-per-node=1 horovodrun -np 4 -H localhost:4 python -u train.py -s horovod
```

## Distributed training with SLURM (batch mode)

Each distributed strategy has its own SLURM job script, which
should be used to run it:

If you want to distribute the code in `train.py` with **torch DDP**, run from terminal:
  
```bash
export DIST_MODE="ddp"
export RUN_NAME="ddp-itwinai"
export TRAINING_CMD="train.py -s ddp"
export PYTHON_VENV="../../../envAI_hdfml"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.sh
```

If you want to distribute the code in `train.py` with **DeepSpeed**, run from terminal:
  
```bash
export DIST_MODE="deepspeed"
export RUN_NAME="deepspeed-itwinai"
export TRAINING_CMD="train.py -s deepspeed"
export PYTHON_VENV="../../../envAI_hdfml"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.sh
```

If you want to distribute the code in `train.py` with **Horovod**, run from terminal:
  
```bash
export DIST_MODE="deepspeed"
export RUN_NAME="deepspeed-itwinai"
export TRAINING_CMD="train.py -s deepspeed"
export PYTHON_VENV="../../../envAI_hdfml"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.sh
```

You can run all of them with:

```bash
bash runall.sh
```
