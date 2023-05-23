#!/bin/bash

# Setup workflow runner
make

# Launch some workflows

################################
###    Training examples     ###
################################

# MNIST training
conda run -p ./.venv \
    python run-workflow.py -f ./use-cases/mnist/training-workflow.yml

# MNIST training (with CWL workflow definition)
conda run -p ./.venv \
    python run-workflow.py -f ./use-cases/mnist/training-workflow.yml --cwl

# AI commnd: from within AI venv
itwinai train --train-dataset ./data/mnist/preproc-images \
    --ml-logs ./data/mnist/ml-logs \
    --config ./use-cases/mnist/mnist-ai-train.yml

################################
###   Inference examples     ###
################################

# MNIST inference
conda run -p ./.venv \
    python run-workflow.py -f ./use-cases/mnist/inference-workflow.yml

# MNIST inference
itwinai predict --input-dataset ./data/mnist/preproc-images \
    --predictions-location ./data/mnist/ml-predictions \
    --config ./use-cases/mnist/mnist-ai-inference.yml \
    --ml-logs ./data/mnist/ml-logs

################################
###  Visualization examples  ###
################################

# Visualize logs
conda activate ./ai/.venv-pytorch && \
    itwinai visualize --path ./data/mnist/ml-logs

# Datasets registry
conda activate ./ai/.venv-pytorch && \
    itwinai datasets --use-case use-cases/mnist/

# Workflows (any file '*-worlflow.yml')
conda activate ./ai/.venv-pytorch && \
    itwinai workflows --use-case use-cases/mnist/

################################
###        Build docs        ###
################################

# According to docs/README.md
cd docs && bundle exec jekyll serve