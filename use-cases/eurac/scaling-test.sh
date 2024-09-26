#!/bin/bash
#
# Script for performing a scalability test. Runs distributed training with a different
# number of nodes with the same number of GPUs. 

export NUM_GPUS=4
export DEBUG=false
export TIME=0:20:00
export PYTHON_VENV="../../envAI_hdfml"

# Creating scalability report folder incase it doesn't exist 
# and then removing old logs
mkdir -p logs_epoch 
rm -rf logs_epoch/*

for NUM_NODES in 1 2 4
do
		# "hacky" way of passing it as keyword argument
		export NUM_NODES=$NUM_NODES
		bash runall.sh 
		echo
done
