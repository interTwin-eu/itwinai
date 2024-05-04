# Using itwinai container to run training on MNIST

## Docker (non-HPC environment)

```bash
bash run_docker.sh
```

The script above runs the following command in the itwinai torch container
in this folder:

```bash
itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline
```

## Singularity (HPC environment)

Distributed ML on multiple compute nodes:

```bash
# Uncomment this line to repeat pull every time
# rm -rf itwinai_torch.sif

# Pull Docker image and convert it to Singularity on login node
singularity pull itwinai_torch.sif docker://ghcr.io/intertwin-eu/itwinai:0.0.1-torch-2.1

# Download dataset locally
singularity run itwinai_torch.sif /bin/bash -c \
    "itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline --steps dataloading_step"

# Run on distributed ML job (torch DDP is the default one)
sbatch slurm.sh

# Run all distributed jobs
bash runall.sh
```
