#!/bin/bash
# Launch some workflows

# MNIST training
conda run -p ./.venv python run-workflow.py -f ./use-cases/mnist/training-workflow.yml

# MNIST inference
conda run -p ./.venv python run-workflow.py -f ./use-cases/mnist/inference-workflow.yml

# AI commnd: from within AI venv
itwinai train --input ./data/mnist/preproc-images-train --output ./data/mnist/ml-logs --config ./use-cases/mnist/mnist-ai.yml

# Visualize logs
itwinai visualize --path ./data/mnist/ml-logs