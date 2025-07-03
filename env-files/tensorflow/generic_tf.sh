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
  ENV_NAME=".venv-tf"
fi

work_dir=$PWD

# Create the python venv if it doesn't already exist
if [ -d "${work_dir}/$ENV_NAME" ];then
  echo "env $ENV_NAME already exists"
else
  python3 -m venv $ENV_NAME
  echo "$ENV_NAME environment is created in ${work_dir}"
fi

source $ENV_NAME/bin/activate

if [ -z "$NO_CUDA" ]; then
  TF_EXTRA="tf"
else
  TF_EXTRA="tf-cuda"
fi
pip install --no-cache-dir -e ".[$TF_EXTRA,dev]"

# Install Prov4ML
if [[ "$(uname)" == "Darwin" ]]; then
  pip install --no-cache-dir "prov4ml[apple]@git+https://github.com/matbun/ProvML@v0.0.1"
else
  # Assuming Nvidia GPUs are available
  pip install --no-cache-dir "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@v0.0.1"
fi