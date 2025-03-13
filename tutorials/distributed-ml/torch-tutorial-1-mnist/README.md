# Tutorial: distributed strategies for PyTorch model trained on MNIST dataset

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

Before launching training, since on JSC's compute nodes there is not internet connection,
you need to download the dataset before while on the login lode:

```bash
source ../../../envAI_hdfml/bin/activate
python train.py --download-only
```

This command creates a local folder called "MNIST" with the dataset.

## Distributed training

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
