#!/bin/bash

rm *checkpoint.pth.tar *.out *.err

timeout="00:01:00"
for N in 1 2 4 8
do
    sbatch --job-name="HVD-imagenet-pure-n$N" --nodes=$N --output="job-Phvd-n$N.out" --error="job-Phvd-n$N.err" --time=$timeout hvd_slurm.sh
done