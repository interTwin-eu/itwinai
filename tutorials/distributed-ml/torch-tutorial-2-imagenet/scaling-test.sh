#!/bin/bash

rm *checkpoint.pth.tar *.out *.err

timeout="00:01:00"
for N in 1 2 4 8
do
    sbatch --job-name="DDP-imagenet-n$N" --nodes=$N --output="job-ddp-n$N.out" --error="job-ddp-n$N.err" --time=$timeout ddp_slurm.sh
    sbatch --job-name="DS-imagenet-n$N" --nodes=$N --output="job-ds-n$N.out" --error="job-ds-n$N.err" --time=$timeout deepspeed_slurm.sh
    sbatch --job-name="HVD-imagenet-n$N" --nodes=$N --output="job-hvd-n$N.out" --error="job-hvd-n$N.err" --time=$timeout hvd_slurm.sh
done