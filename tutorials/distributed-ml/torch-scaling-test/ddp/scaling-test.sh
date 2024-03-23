#!/bin/bash

rm *checkpoint.pth.tar *.out *.err *.csv

timeout="01:01:00"
for N in 1 2 4 8
do
    sbatch --job-name="DDP-imagenet-pure-n$N" --nodes=$N --output="job-Pddp-n$N.out" --error="job-Pddp-n$N.err" --time=$timeout ddp_slurm.sh
done