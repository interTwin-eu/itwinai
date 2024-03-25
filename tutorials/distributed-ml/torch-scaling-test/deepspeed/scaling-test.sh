#!/bin/bash

rm *checkpoint.pth.tar *.out *.err *.csv

timeout="01:01:00"
for N in 1 2 4 8
do
    sbatch --job-name="DS-imagenet-pure-n$N" --nodes=$N --output="job-Pds-n$N.out" --error="job-Pds-n$N.err" --time=$timeout deepspeed_slurm.sh
done