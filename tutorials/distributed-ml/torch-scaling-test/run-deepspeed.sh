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
    T="01:00:00"
else
    T=$2
fi


# Common options
CMD="--nodes=$N --time=$T --account=intertwin --partition=batch slurm.sh"
PYTHON_VENV="../../../envAI_hdfml"

echo "Distributing training over $N nodes. Timeout set to: $T"

# Clear SLURM logs (*.out and *.err files)
rm -rf logs_slurm
mkdir logs_slurm
rm -rf logs_torchrun

# DeepSpeed itwinai
DIST_MODE="deepspeed"
RUN_NAME="deepspeed-itwinai-imagenent"
TRAINING_CMD="itwinai_trainer.py -c config/base.yaml -c config/deepspeed.yaml -s deepspeed"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.out" \
    $CMD

#--error="logs_slurm/job-$RUN_NAME-n$N.err" \