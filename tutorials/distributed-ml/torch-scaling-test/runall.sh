#!/bin/bash
# Run all versions of distributed ML version

rm *checkpoint.pth.tar *.out *.err *.csv

for name in ddp horovod deepspeed
do
    # echo $fold" training: $(sbatch --nodes=1 $fold"_slurm.sh")"
    echo $name" training: $(sbatch $name"_slurm.sh")"
done