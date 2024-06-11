#!/bin/bash

# ENV VARIABLES:
#   - ENV_NAME: set custom name for virtual env. Default: ".venv-tf"
#   - NO_CUDA: if set, install without cuda support

# Detect custom env name from env
if [ -z "$ENV_NAME" ]; then
  ENV_NAME=".venv-tf"
fi

if [ -z "$NO_CUDA" ]; then
  echo "Installing itwinai and its dependencies in '$ENV_NAME' virtual env (CUDA enabled)"
else
  echo "Installing itwinai and its dependencies in '$ENV_NAME' virtual env (CUDA disabled)"
fi

# get python version
pver="$(python --version 2>&1 | awk '{print $2}' | cut -f1-2 -d.)"

# use pyenv if exist
if [ -d "$HOME/.pyenv" ];then
  export PYENV_ROOT="$HOME/.pyenv"
  export PATH="$PYENV_ROOT/bin:$PATH"
fi

# set dir
cDir=$PWD

# create environment
if [ -d "${cDir}/$ENV_NAME" ];then
  echo "env $ENV_NAME already exists"

  source $ENV_NAME/bin/activate
else
  python3 -m venv $ENV_NAME

  # activate env
  source $ENV_NAME/bin/activate

  echo "$ENV_NAME environment is created in ${cDir}"
fi

pip3 install --upgrade pip

# get wheel -- setuptools extension
pip3 install --no-cache-dir wheel

# install TF 
if [ -f "${cDir}/$ENV_NAME/bin/tensorboard" ]; then
  echo 'TF already installed'
  echo
else
  if [ -z "$NO_CUDA" ]; then
    pip3 install tensorflow[and-cuda]==2.16 --no-cache-dir
  else
    # CPU only installation
    pip3 install tensorflow==2.16 --no-cache-dir
  fi
fi

# CURRENTLY, horovod is not used with TF. Skipped.
# # install horovod
# if [ -f "${cDir}/$ENV_NAME/bin/horovodrun" ]; then
#   echo 'Horovod already installed'
#   echo
# else
#   if [ -z "$NO_CUDA" ]; then
#     export HOROVOD_GPU=CUDA
#     export HOROVOD_GPU_OPERATIONS=NCCL
#     export HOROVOD_WITH_TENSORFLOW=1
#     # export TMPDIR=${cDir}
#   else
#     # CPU only installation
#     export HOROVOD_WITH_TENSORFLOW=1
#     # export TMPDIR=${cDir}
#   fi

#   pip3 install --no-cache-dir horovod[tensorflow,keras] # --ignore-installed
# fi

# WHEN USING TF >= 2.16:
# # install legacy version of keras (2.16)
# # Since TF 2.16, keras updated to 3.3,
# # which leads to an error when more than 1 node is used
# # https://keras.io/getting_started/
pip3 install tf_keras

# itwinai
pip3 install -e .[dev]
