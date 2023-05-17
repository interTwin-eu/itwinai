#!/bin/bash
# Launch some workflows

################################
###    Training examples     ###
################################

# MNIST training
conda run -p ./.venv python run-workflow.py -f ./use-cases/mnist/training-workflow.yml

# MNIST training (with CWL workflow definition)
conda run -p ./.venv python run-workflow.py -f ./use-cases/mnist/training-workflow.yml --cwl

# AI commnd: from within AI venv
itwinai train --input ./data/mnist/preproc-images-train --output ./data/mnist/ml-logs --config ./use-cases/mnist/mnist-ai.yml

################################
###   Inference examples     ###
################################

# MNIST inference
conda run -p ./.venv python run-workflow.py -f ./use-cases/mnist/inference-workflow.yml

################################
###  Visualization examples  ###
################################

# Visualize logs
conda activate ./ai/.venv-dev && itwinai visualize --path ./data/mnist/ml-logs