#!/bin/bash

# Simple installation for users
# It expects ML_FRAMEWORK env variable to be set

# echo "Please choose an option:"
# echo "1. Install itwinai with PyTorch support"
# echo "2. Install itwinai with TensorFlow support"
# read -p "Enter your choice (1 or 2): " choice

if [ -z "$ML_FRAMEWORK" ]; then
    echo "Error: ML_FRAMEWORK env variable not set. Accepted values are 'pytorch' and 'tensorflow'"
    exit 1
fi


# Detect python env and CUDA env
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Not in a Python virtual environment."
    exit 1
fi
export ENV_NAME=$VIRTUAL_ENV

# Check if any CUDA-capable device is detected by nvidia-smi
if ! nvidia-smi > /dev/null 2>&1; then
    echo "CUDA is not available. Installing cpu-only version."
    export NO_CUDA=1
fi

# Begin installation of dependencies + itwinai
if [ "$ML_FRAMEWORK" == "pytorch" ]; then
    echo "Installing itwinai with PyTorch support..."
    # Skip last line (head -n -1) because it contains the istallation of itwinai
    curl -fsSL https://github.com/interTwin-eu/itwinai/raw/main/env-files/torch/generic_torch.sh | head -n -1 | bash
    # cat ../env-files/torch/generic_torch.sh | head -n -1 | bash
    # Install from PyPI
    pip install itwinai[torch]
elif [ "$ML_FRAMEWORK" == "tensorflow" ]; then
    echo "Installing itwinai with TensorFlow support..."
    # Skip last line (head -n -1) because it contains the istallation of itwinai
    curl -fsSL https://github.com/interTwin-eu/itwinai/raw/main/env-files/tensorflow/generic_tf.sh | head -n -1 | bash
    # cat ../env-files/tensorflow/generic_tf.sh | head -n -1 | bash
    # Install from PyPI
    pip install itwinai
else
    echo "Invalid choice. Please run the script again and select either 1 or 2."
fi
