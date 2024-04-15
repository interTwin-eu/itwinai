#!/bin/bash
# Run all versions of distributed ML version
# $1 (Optional[int]): number of nodes. Default: 2
# $2 (Optional[str]): timeout. Default: "00:30:00"

if [ -z "$1" ] ; then
    N=2
else
    N=$1
fi
if [ -z "$2" ] ; then
    T="00:30:00"
else
    T=$2
fi

# Common options
CMD="--nodes=$N --time=$T --account=intertwin --partition=batch slurm.sh"

echo "Distributing training over $N nodes. Timeout set to: $T"

rm *.out *.err *.csv #*checkpoint.pth.tar 

# DDP baseline
DIST_MODE="ddp"
RUN_NAME="ddp-bl-imagenent"
TRAINING_CMD="ddp_trainer.py -c config/base.yaml -c config/ddp.yaml"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD" --job-name="$RUN_NAME-n$N" $CMD

# DeepSpeed baseline
DIST_MODE="deepspeed"
RUN_NAME="deepspeed-bl-imagenent"
TRAINING_CMD="deepspeed_trainer.py -c config/base.yaml -c config/deepspeed.yaml"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD" --job-name="$RUN_NAME-n$N" $CMD

# Horovod baseline
DIST_MODE="horovod"
RUN_NAME="horovod-bl-imagenent"
TRAINING_CMD="horovod_trainer.py -c config/base.yaml -c config/horovod.yaml"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD" --job-name="$RUN_NAME-n$N" $CMD

# DDP itwinai
DIST_MODE="ddp"
RUN_NAME="ddp-itwinai-imagenent"
TRAINING_CMD="itwinai_trainer.py -c config/base.yaml -c config/ddp.yaml -s ddp"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD" --job-name="$RUN_NAME-n$N" $CMD

# DeepSpeed itwinai
DIST_MODE="deepspeed"
RUN_NAME="deepspeed-itwinai-imagenent"
TRAINING_CMD="itwinai_trainer.py -c config/base.yaml -c config/deepspeed.yaml -s deepspeed"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD" --job-name="$RUN_NAME-n$N" $CMD

# Horovod itwinai
DIST_MODE="horovod"
RUN_NAME="horovod-itwinai-imagenent"
TRAINING_CMD="itwinai_trainer.py -c config/base.yaml -c config/horovod.yaml -s horovod"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD" --job-name="$RUN_NAME-n$N" $CMD