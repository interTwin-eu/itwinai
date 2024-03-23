#!/bin/bash
# Run all versions of distributed ML version

for fold in ddp horovod deepspeed
do
    cd $fold
    rm *checkpoint.pth.tar *.out *.err *.csv
    if [ $fold == "horovod" ]
    then
        fold="hvd"
    fi
    echo $fold" training: $(sbatch $fold"_slurm.sh")"
    cd ..
done