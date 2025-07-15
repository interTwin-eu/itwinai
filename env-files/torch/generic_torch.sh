#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

if [ -z "$ENV_NAME" ]; then
  ENV_NAME=".venv-pytorch"
fi

work_dir=$PWD

# Create the python venv if it doesn't already exist
if [ -d "${work_dir}/$ENV_NAME" ];then
  echo "env $ENV_NAME already exists"
else
  python3 -m venv $ENV_NAME
  echo "$ENV_NAME environment is created in ${work_dir}"
fi

# Activate the venv and then install itwinai as editable
source $ENV_NAME/bin/activate
pip install uv

if [ -z "$NO_CUDA" ]; then
  # Install with CUDA support
  uv pip install -e ".[torch,dev]" \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu126
else
  # Install without CUDA support
  uv pip install -e ".[torch,dev]" --no-cache-dir
fi


# Install Prov4ML
if [[ "$(uname)" == "Darwin" ]]; then
  uv pip install --no-cache-dir  "prov4ml[apple]@git+https://github.com/matbun/ProvML@v0.0.2"
else
  # Assuming Nvidia GPUs are available
  uv pip install --no-cache-dir  "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@v0.0.2"
fi
