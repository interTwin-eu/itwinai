#!/bin/bash
# -*- coding: utf-8 -*-

if [ ! -f "env-files/torch/generic_torch.sh" ]; then
  echo "ERROR: env-files/torch/generic_torch.sh not found!"
  exit 1
fi

# set dir
cDir=$PWD

# get sys info
sysN="$(uname -n | cut -f2- -d.)"
sysN="${sysN%%[0-9]*}"

# load modules
# NOTE: REFLECT THEM IN THE MAIN README! 
ml --force purge
ml Stages/2025 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
ml Python CMake HDF5 PnetCDF libaio mpi4py git

# Create and install torch env
export ENV_NAME="envAI_$sysN"
bash env-files/torch/generic_torch.sh
source $ENV_NAME/bin/activate

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




