#!/bin/bash
# -*- coding: utf-8 -*-
# author: RS
# version: 220302a
# creates machine specific python env

# set modules
ml --force purge

# get sys info
cDir=$PWD
sysN="$(uname -n | cut -f2- -d.)"
echo "system:${sysN}"
echo

cont1=false
if [ "$sysN" = 'deepv' ] ; then
  ml use "$OTHERSTAGES"
  ml Stages/2022 GCC OpenMPI cuDNN NCCL Python CMake
  cont1=true
elif [ "$sysN" = 'juwels' ] ; then
  ml Stages/2022 GCC ParaStationMPI Python CMake NCCL libaio cuDNN
  cont1=true
elif [ "$sysN" = 'hdfml' ] ; then
  #ml Stages/2022 GCC OpenMPI Python NCCL cuDNN libaio CMake
  #ml Stages/2023 NVHPC/23.1 ParaStationMPI/5.8.0-1-mt NCCL/default-CUDA-11.7 cuDNN/8.6.0.163-CUDA-11.7 Python CMake
  ml Stages/2024 GCC/12.3.0 OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py CMake cuDNN/8.9.5.29-CUDA-12
  cont1=true
else
  echo
  echo 'unknown system detected'
  echo 'canceling'
  echo
fi
echo "modules loaded"
echo

# get python version
pver="$(python --version 2>&1 | awk '{print $2}' | cut -f1-2 -d.)"
echo "python version is ${pver}"
echo

if [ "$cont1" = true ] ; then
  if [ -d "${cDir}/envAItf_${sysN}" ];then
    echo 'env already exist'
    echo

    source envAItf_${sysN}/bin/activate
  else
    # create env
    python3 -m venv envAItf_${sysN}

    # get headers for pip
    if [ -f "${cDir}/envAItf_${sysN}/bin/pip3" ]; then
      echo 'pip already exist'
    else
      cp "$(which pip3)" $cDir/envAItf_${sysN}/bin/
      ln -s $cDir/envAItf_${sysN}/bin/pip3 $cDir/envAItf_${sysN}/bin/pip${pver}
      var="#!$cDir/envAItf_${sysN}/bin/python${pver}"
      sed -i "1s|.*|$var|" $cDir/envAItf_${sysN}/bin/pip3
    fi

    # activate env
    source envAItf_${sysN}/bin/activate

    echo "a new env is created in ${cDir}"
    echo "activation is done via:"
    echo "source ${cDir}/envAItf_${sysN}/bin/activate"
  fi
fi

# install TF 
if [ -f "${cDir}/envAItf_${sysN}/bin/tensorboard" ]; then
  echo 'TF already installed'
  echo
else
  export TMPDIR=${cDir}

  pip3 install --upgrade tensorflow[and-cuda] --no-cache-dir
fi

# install horovod
if [ -f "${cDir}/envAItf_${sysN}/bin/horovodrun" ]; then
  echo 'Horovod already installed'
  echo
else
  export HOROVOD_GPU=CUDA
  export HOROVOD_GPU_OPERATIONS=NCCL
  export HOROVOD_WITH_TENSORFLOW=1
  export TMPDIR=${cDir}

  pip3 install --no-cache-dir horovod --ignore-installed
fi

# JUBE benchmarking environment
if [ -f "${cDir}/envAI_${sysN}/bin/jube" ]; then
  echo 'JUBE already installed'
else
  pip3 install --no-cache-dir http://apps.fz-juelich.de/jsc/jube/jube2/download.php?version=latest
fi

# get rest of the libraries$
if [ "$cont1" = true ] ; then
  pip3 install -r reqs_TF.txt --ignore-installed
fi

# Install itwinai
pip install --upgrade pip
pip install -e .[dev]

# eof
