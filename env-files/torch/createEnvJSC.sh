#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI, RS, Matteo Bunino

# set dir
cDir=$PWD

# environmental variables
mkdir -p tmp
export TMPDIR=${cDir}/tmp # set tmp dir env var

# get sys info
sysN="$(uname -n | cut -f2- -d.)"
sysN="${sysN%%[0-9]*}"

# load modules
ml Stages/2024 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
ml Python CMake HDF5 PnetCDF libaio mpi4py
# echo "these modules are loaded:"
# ml

# get python version
pver="$(python --version 2>&1 | awk '{print $2}' | cut -f1-2 -d.)"

# use pyenv if exist
if [ -d "$HOME/.pyenv" ];then
  export PYENV_ROOT="$HOME/.pyenv"
  export PATH="$PYENV_ROOT/bin:$PATH"
fi

# create environment
if [ -d "${cDir}/envAI_${sysN}" ];then
  echo 'env already exist'

  source envAI_${sysN}/bin/activate
else
  python3 -m venv envAI_${sysN}

  # activate env
  source envAI_${sysN}/bin/activate

  echo "envAI_${sysN} environment is created in ${cDir}"
fi

# get wheel -- setuptools extension
pip3 install --no-cache-dir wheel

# install Torch
if [ -f "${cDir}/envAI_${sysN}/bin/torchrun" ]; then
  echo 'Torch already installed'
else
  pip3 install --no-cache-dir \
     torch==2.1.0+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# HPO - RayTune
if [ -f "${cDir}/envAI_${sysN}/bin/ray" ]; then
  echo 'Ray already installed'
else
  pip3 install --no-cache-dir ray ray[tune]
fi

# install deepspeed
if [ -f "${cDir}/envAI_${sysN}/bin/deepspeed" ]; then
  echo 'DeepSpeed already installed'
else
  export DS_BUILD_CCL_COMM=1
  export DS_BUILD_UTILS=1
  export DS_BUILD_AIO=1
  export DS_BUILD_FUSED_ADAM=1
  export DS_BUILD_FUSED_LAMB=1
  export DS_BUILD_TRANSFORMER=1
  export DS_BUILD_STOCHASTIC_TRANSFORMER=1
  export DS_BUILD_TRANSFORMER_INFERENCE=1

  # this will pass
  pip3 install --no-cache-dir DeepSpeed

  # fix .triton/autotune/Fp16Matmul_2d_kernel.pickle bug
  line=$(cat -n envAI_${sysN}/lib/python${pver}/site-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py | grep os.rename | awk '{print $1}' | head -n 1)
  sed -i "${line}s|^|#|" envAI_${sysN}/lib/python${pver}/site-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py
fi

# # install heat
# if [ -d "${cDir}/envAI_${sysN}/lib/python${pver}/site-packages/heat" ]; then
#   echo 'HeAT already installed'
# else
#   # need to modify setup.py to accep torch>2.1 for heat
#   git clone --recurse-submodules https://github.com/helmholtz-analytics/heat.git
#   line=$(cat -n heat/setup.py | grep torch | awk '{print $1}' | head -n 1)
#   var='        "torch>=2.1.0",'
#   sed -i "${line}s|.*|$var|" heat/setup.py

#   # create tar!
#   rm -rf heat.tar.gz
#   tar czf heat.tar.gz heat

#   # install
#   pip3 install --no-cache-dir 'heat.tar.gz[hdf5,netcdf]'
# fi

# install horovod
if [ -f "${cDir}/envAI_${sysN}/bin/horovodrun" ]; then
  echo 'Horovod already installed'
else
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

  # need to modify for torch 2.1.0 
  git clone --recurse-submodules https://github.com/horovod/horovod.git
  line=$(cat -n horovod/CMakeLists.txt | grep CMAKE_CXX_STANDARD | awk '{print $1}' | head -n 1)
  var='set(CMAKE_CXX_STANDARD 17)'
  sed -i "${line}s|.*|$var|" horovod/CMakeLists.txt
  line=$(cat -n horovod/horovod/torch/CMakeLists.txt | grep CMAKE_CXX_STANDARD | awk '{print $1}' | head -n 1)
  var='    set(CMAKE_CXX_STANDARD 17)'
  sed -i "${line}s|.*|$var|" horovod/horovod/torch/CMakeLists.txt

  # create tar!
  rm -rf horovod.tar.gz
  tar czf horovod.tar.gz horovod
  
  # install
  pip3 install --no-cache-dir horovod.tar.gz
fi

# get required libraries in reqs.txt
if [ -f "${cDir}/envAI_${sysN}/lib/python${pver}/site-packages/torchnlp/_third_party/weighted_random_sampler.py" ]; then
   echo 'required libs already exist'
else
  pip3 install -r Scripts/reqs.txt --no-cache-dir

  # fix int bug: modify l.4 of /torchnlp/_third_party/weighted_random_sampler.py
  var='int_classes = int'
  sed -i "4s|.*|$var|" \
    ${cDir}/envAI_${sysN}/lib/python${pver}/site-packages/torchnlp/_third_party/weighted_random_sampler.py
fi

# fix IB IP config - FZJ specific
if [ -f "${cDir}/envAI_${sysN}/bin/torchrun" ]; then
  sed -i -e '5,100s/^/#/' ${cDir}/envAI_${sysN}/bin/torchrun
  echo """
import re
import sys
from torch.distributed.run import main
from torch.distributed.elastic.agent.server import api as sapi

def new_get_fq_hostname():
    return _orig_get_fq_hostname().replace('.', 'i.', 1)

if __name__ == '__main__':
    _orig_get_fq_hostname = sapi._get_fq_hostname
    sapi._get_fq_hostname = new_get_fq_hostname
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
""" >> ${cDir}/envAI_${sysN}/bin/torchrun
fi

# JUBE benchmarking environment
if [ -f "${cDir}/envAI_${sysN}/bin/jube" ]; then
  echo 'JUBE already installed'
else
  pip3 install --no-cache-dir http://apps.fz-juelich.de/jsc/jube/jube2/download.php?version=latest
fi

# some tests
echo "unit tests:"
for item in 'torch' 'deepspeed' 'horovod';do
  python3 -c "import $item; print('$item version:',$item.__version__)"
done

# Install itwinai
pip install --upgrade pip
pip install -e .[dev]

# cleanup
rm -rf horovod *.tar.gz
