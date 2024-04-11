#!/bin/bash
# Run all versions of distributed ML version

for name in ddp hvd deepspeed
do
    rm *checkpoint.pth.tar *.out *.err *.csv
    # echo $fold" training: $(sbatch --nodes=1 $fold"_slurm.sh")"
    echo $name" training: $(sbatch $name"_slurm.sh")"
done