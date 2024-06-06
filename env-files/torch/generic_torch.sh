#!/bin/bash

# ENV VARIABLES:
#   - ENV_NAME: set custom name for virtual env. Default: ".venv-pytorch"
#   - NO_CUDA: if set, install without cuda support

# Detect custom env name from env
if [ -z "$ENV_NAME" ]; then
  ENV_NAME=".venv-pytorch"
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

# install Torch
if [ -f "${cDir}/$ENV_NAME/bin/torchrun" ]; then
  echo 'Torch already installed'
else
  if [ -z "$NO_CUDA" ] ; then
    pip3 install --no-cache-dir \
    torch==2.1.* torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  else
    # CPU only installation for MacOS
    if [[ "$OSTYPE" =~ ^darwin ]] ; then
      pip3 install --no-cache-dir \
        torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
    else
    # CPU only installation for other OSs
      pip3 install --no-cache-dir \
         torch==2.1.* torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
  fi
fi

# HPO - RayTune
if [ -f "${cDir}/$ENV_NAME/bin/ray" ]; then
  echo 'Ray already installed'
else
  if [[ "$OSTYPE" =~ ^darwin ]] ; then
    echo 'Installation issues: Skipping Ray installation for MacOS'
  else
    pip3 install --no-cache-dir ray ray[tune]
  fi
fi

# install deepspeed
if [ -f "${cDir}/$ENV_NAME/bin/deepspeed" ]; then
  echo 'DeepSpeed already installed'
else
  if [ -z "$NO_CUDA" ]; then
    export DS_BUILD_CCL_COMM=1
    export DS_BUILD_UTILS=1
    export DS_BUILD_AIO=1
    export DS_BUILD_FUSED_ADAM=1
    export DS_BUILD_FUSED_LAMB=1
    export DS_BUILD_TRANSFORMER=1
    export DS_BUILD_STOCHASTIC_TRANSFORMER=1
    export DS_BUILD_TRANSFORMER_INFERENCE=1
    pip3 install --no-cache-dir DeepSpeed
  else
    # CPU only installation
    pip3 install deepspeed
  fi

  # fix .triton/autotune/Fp16Matmul_2d_kernel.pickle bug
  line=$(cat -n $ENV_NAME/lib/python${pver}/site-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py | grep os.rename | awk '{print $1}' | head -n 1)
  sed -i "${line}s|^|#|" $ENV_NAME/lib/python${pver}/site-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py
fi

# install horovod
if [ -f "${cDir}/$ENV_NAME/bin/horovodrun" ]; then
  echo 'Horovod already installed'
else

  if [ -z "$NO_CUDA" ]; then
    # compiler vars
    export LDSHARED="$CC -shared" &&
    export CMAKE_CXX_STANDARD=17 

    # CPU vars
    export HOROVOD_MPI_THREADS_DISABLE=1
    export HOROVOD_CPU_OPERATIONS=MPI

    # GPU vars
    export HOROVOD_GPU_ALLREDUCE=NCCL
    export HOROVOD_NCCL_LINK=SHARED
    export HOROVOD_NCCL_HOME=$EBROOTNCCL

    # Host language vars
    export HOROVOD_WITH_PYTORCH=1
    export HOROVOD_WITHOUT_TENSORFLOW=1
    export HOROVOD_WITHOUT_MXNET=1
  else
    # CPU only installation
    export HOROVOD_WITH_PYTORCH=1
    export HOROVOD_WITHOUT_TENSORFLOW=1
    export HOROVOD_WITHOUT_MXNET=1
  fi
    
#   # need to modify for torch 2.1.0 
#   git clone --recurse-submodules https://github.com/horovod/horovod.git
#   line=$(cat -n horovod/CMakeLists.txt | grep CMAKE_CXX_STANDARD | awk '{print $1}' | head -n 1)
#   var='set(CMAKE_CXX_STANDARD 17)'
#   sed -i "${line}s|.*|$var|" horovod/CMakeLists.txt
#   line=$(cat -n horovod/horovod/torch/CMakeLists.txt | grep CMAKE_CXX_STANDARD | awk '{print $1}' | head -n 1)
#   var='    set(CMAKE_CXX_STANDARD 17)'
#   sed -i "${line}s|.*|$var|" horovod/horovod/torch/CMakeLists.txt

#   # create tar!
#   rm -rf horovod.tar.gz
#   tar czf horovod.tar.gz horovod
  
#   # install
#   pip3 install --no-cache-dir horovod.tar.gz
#   rm -rf horovod horovod.tar.gz

  # Cleaner Horovod installation
	# https://github.com/horovod/horovod/pull/3998
  # Assume that Horovod env vars are already in the current env!
  pip3 install --no-cache-dir git+https://github.com/thomas-bouvier/horovod.git@compile-cpp17
fi

# get required libraries in reqs.txt
if [ -f "${cDir}/$ENV_NAME/lib/python${pver}/site-packages/torchnlp/_third_party/weighted_random_sampler.py" ]; then
   echo 'required libs already exist'
else
#   pip3 install -r Scripts/reqs.txt --no-cache-dir

  # fix int bug: modify l.4 of /torchnlp/_third_party/weighted_random_sampler.py
  var='int_classes = int'
  sed -i "4s|.*|$var|" \
    ${cDir}/$ENV_NAME/lib/python${pver}/site-packages/torchnlp/_third_party/weighted_random_sampler.py
fi

# Install itwinai
pip3 install -e .[dev,torch]
