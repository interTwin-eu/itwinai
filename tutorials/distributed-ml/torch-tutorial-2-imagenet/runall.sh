#!/bin/bash
# Run all versions of distributed ML version
rm *checkpoint.pth.tar *.out *.err *.csv
echo "Torch DDP training: $(sbatch ddp_slurm.sh)"
echo "DeepSpeed training: $(sbatch deepspeed_slurm.sh)"
echo "Horovod training: $(sbatch hvd_slurm.sh)"