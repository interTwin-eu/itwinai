# Tutorial: distributed strategies for PyTorch

**Author(s)**: Matteo Bunino (CERN), Jarl Sondre Sæther (CERN)

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

You can run your training with SLURM by using the `itwinai` SLURM Builder. Use the
`slurm_config.yaml` file to specify your SLURM parameters and then preview your script
with the following command:

```bash
itwinai generate-slurm -c slurm_config.yaml --no-save-script --no-submit-job
```

If you are happy with the script, you can then run it by omitting `--no-submit-job`:

```bash
itwinai generate-slurm -c slurm_config.yaml --no-save-script
```

If you want to store a copy of the script in a folder, then you can similarly omit
`--no-save-script`:

```bash
itwinai generate-slurm -c slurm_config.yaml
```
