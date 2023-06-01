#!/bin/bash

# Setup workflow runner
make

# Launch some workflows

################################
###    Training examples     ###
################################

# MNIST training
micromamba run -p ./.venv \
    python run-workflow.py -f ./use-cases/mnist/training-workflow.yml

# MNIST training (with CWL workflow definition)
micromamba run -p ./.venv \
    python run-workflow.py -f ./use-cases/mnist/training-workflow.yml --cwl

# AI commnd: from within AI venv
itwinai train --train-dataset ./use-cases/mnist/data/preproc-images \
    --ml-logs ./use-cases/mnist/data/ml-logs \
    --config ./use-cases/mnist/mnist-ai-train.yml

################################
###   Inference examples     ###
################################

# MNIST inference
micromamba run -p ./.venv \
    python run-workflow.py -f ./use-cases/mnist/inference-workflow.yml

# MNIST inference
itwinai predict --input-dataset ./use-cases/mnist/data/preproc-images \
    --predictions-location ./use-cases/mnist/data/ml-predictions \
    --config ./use-cases/mnist/mnist-ai-inference.yml \
    --ml-logs ./use-cases/mnist/data/ml-logs

################################
###  Visualization examples  ###
################################

# Visualize logs
micromamba activate ./ai/.venv-pytorch && \
    itwinai mlflow-ui --path ./use-cases/mnist/data/ml-logs

# Datasets registry
micromamba activate ./ai/.venv-pytorch && \
    itwinai datasets --use-case use-cases/mnist/

# Workflows (any file '*-workflow.yml')
micromamba activate ./ai/.venv-pytorch && \
    itwinai workflows --use-case use-cases/mnist/

################################
###        Build docs        ###
################################

# According to docs/README.md
cd docs && bundle exec jekyll serve