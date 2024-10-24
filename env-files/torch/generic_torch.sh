#!/bin/bash
# ENV VARIABLES:
#   - ENV_NAME: set custom name for virtual env. Default: ".venv-pytorch"
#   - NO_CUDA: if set, install without cuda support
#   - PIP_INDEX_TORCH_CUDA: pip index to be used to install torch with CUDA. Defaults to https://download.pytorch.org/whl/cu121
# Detect custom env name from env
if [ -z "$ENV_NAME" ]; then
  ENV_NAME=".venv-pytorch"
fi

if [ -z "$NO_CUDA" ]; then
  echo "Installing itwinai and its dependencies in '$ENV_NAME' virtual env (CUDA enabled)"
else
  echo "Installing itwinai and its dependencies in '$ENV_NAME' virtual env (CUDA disabled)"
fi

if [ -z "$PIP_INDEX_TORCH_CUDA" ]; then
  PIP_INDEX_TORCH_CUDA="https://download.pytorch.org/whl/cu121"
fi

# get python version
pver="$(python3 --version 2>&1 | awk '{print $2}' | cut -f1-2 -d.)"

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

pip install --no-cache-dir --upgrade pip 
pip install --no-cache-dir packaging wheel 

# Adding this constraint as numpy >= 2 seems to clash with DeepSpeed
pip install --no-cache-dir 'numpy<2.0.0' || exit 1

# install Torch
if [ -f "${cDir}/$ENV_NAME/bin/torchrun" ]; then
  echo 'Torch already installed'
else
  if [ -z "$NO_CUDA" ] ; then
    pip install --no-cache-dir \
    'torch==2.4.*' torchvision torchaudio --index-url "$PIP_INDEX_TORCH_CUDA" || exit 1
  else
    # CPU only installation for MacOS
    if [[ "$OSTYPE" =~ ^darwin ]] ; then
      pip install --no-cache-dir \
        'torch==2.4.*' torchvision torchaudio || exit 1
    else
    # CPU only installation for other OSs
      pip install --no-cache-dir \
         'torch==2.4.*' torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || exit 1
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
    pip install --no-cache-dir ray ray[tune] || exit 1
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
	fi
	pip install --no-cache-dir py-cpuinfo || exit 1
	pip install --no-cache-dir deepspeed || exit 1

	# fix .triton/autotune/Fp16Matmul_2d_kernel.pickle bug
	line=$(cat -n $ENV_NAME/lib/python${pver}/site-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py | grep os.rename | awk '{print $1}' | head -n 1)

	# 'sed' is implemented differently on MacOS than on Linux (https://stackoverflow.com/questions/4247068/sed-command-with-i-option-failing-on-mac-but-works-on-linux)
	if [[ "$OSTYPE" =~ ^darwin ]] ; then
		sed -i '' "${line}s|^|#|" $ENV_NAME/lib/python${pver}/site-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py || exit 1
	else
	  sed -i "${line}s|^|#|" $ENV_NAME/lib/python${pver}/site-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py || exit 1
	fi
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
#   pip install --no-cache-dir horovod.tar.gz
#   rm -rf horovod horovod.tar.gz

  # Cleaner Horovod installation
	# https://github.com/horovod/horovod/pull/3998
  # Assume that Horovod env vars are already in the current env!
  pip install --no-cache-dir git+https://github.com/horovod/horovod.git || exit 1
  # pip install --no-cache-dir git+https://github.com/thomas-bouvier/horovod.git@compile-cpp17 || exit 1
fi

# get required libraries in reqs.txt
# if [ -f "${cDir}/$ENV_NAME/lib/python${pver}/site-packages/torchnlp/_third_party/weighted_random_sampler.py" ]; then
   # echo 'required libs already exist'
# else
#   pip install -r Scripts/reqs.txt --no-cache-dir

  # fix int bug: modify l.4 of /torchnlp/_third_party/weighted_random_sampler.py
  # var='int_classes = int'
  # sed -i .backup_file "4s|.*|$var|" \
    # ${cDir}/$ENV_NAME/lib/python${pver}/site-packages/torchnlp/_third_party/weighted_random_sampler.py || exit 1
  # Deleting unnecessary backup file
  # rm ${cDir}/$ENV_NAME/lib/python${pver}/site-packages/torchnlp/_third_party/weighted_random_sampler.py.backup_file
# fi

# Install Pov4ML
if [[ "$OSTYPE" =~ ^darwin ]] ; then
  pip install --no-cache-dir "prov4ml[apple]@git+https://github.com/matbun/ProvML" || exit 1
else
  pip install --no-cache-dir "prov4ml[linux]@git+https://github.com/matbun/ProvML" || exit 1
fi

# Install itwinai: MUST be last line of the script for the user installation script to work!
pip install --no-cache-dir -e .[torch,dev] || exit 1
