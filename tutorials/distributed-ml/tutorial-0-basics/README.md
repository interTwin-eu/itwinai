# Tutorial: distributed strategies for PyTorch

In this tutorial we show how to use torch `DistributedDataParallel` (DDP), Horovod and DeepSpeed from the same client code. Note that the environment is tested on the HDFML system at JSC. For other systems, the module versions might need change accordingly. 

First, from the root of this repo, build the environment containing
pytorch, horovod and deepspeed. You can *try* with:

```bash
# Creates a Python venv called envAI_hdfml
make torch-gpu-jsc
```

Each distributed strategy has its own SLURM job script, which
should be used to run it:

If you want to distribute the code in `train.py` with **torch DDP**, run from terminal:
  
```bash
sbatch ddp_slurm.sh
```

If you want to distribute the code in `train.py` with **DeepSpeed**, run from terminal:
  
```bash
sbatch deepspeed_slurm.sh
```

If you want to distribute the code in `train.py` with **Horovod**, run from terminal:
  
```bash
sbatch hvd_slurm.sh
```
