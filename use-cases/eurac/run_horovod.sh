#!/bin/bash

NUM_NODES=2
GPUS_PER_NODE=2
CPUS_PER_TASK=1

srun --jobid 70649 \
	--nodes=$NUM_NODES \
	--ntasks-per-node=1 \
	--cpus-per-task=$CPUS_PER_TASK \
	python -u \
	$(which itwinai) exec-pipeline \
	--config config.yaml \
	--pipe-key training_pipeline \
	-o strategy=horovod

	
