#!/bin/bash
# -*- coding: utf-8 -*-

if [ ! -f "env-files/tensorflow/generic_tf.sh" ]; then
  echo "ERROR: env-tensorflow/torch/generic_tf.sh not found!"
  exit 1
fi

# set modules
ml --force purge

# get sys info
cDir=$PWD
sysN="$(uname -n | cut -f2- -d.)"
echo "system:${sysN}"
echo

cont1=false
if [ "$sysN" = 'hdfml' ] ; then
  # NOTE: REFLECT THEM IN THE MAIN README! 
  ml Stages/2024 GCC/12.3.0 OpenMPI CUDA/12 MPI-settings/CUDA Python/3.11 HDF5 PnetCDF libaio mpi4py CMake cuDNN/8.9.5.29-CUDA-12
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
    python -m venv envAItf_${sysN}

    # get headers for pip
    if [ -f "${cDir}/envAItf_${sysN}/bin/pip" ]; then
      echo 'pip already exist'
    else
      cp "$(which pip)" $cDir/envAItf_${sysN}/bin/
      ln -s $cDir/envAItf_${sysN}/bin/pip $cDir/envAItf_${sysN}/bin/pip${pver}
      var="#!$cDir/envAItf_${sysN}/bin/python${pver}"
      sed -i "1s|.*|$var|" $cDir/envAItf_${sysN}/bin/pip
    fi

    # activate env
    source envAItf_${sysN}/bin/activate

    echo "a new env is created in ${cDir}"
    echo "activation is done via:"
    echo "source ${cDir}/envAItf_${sysN}/bin/activate"
  fi
fi

# Install TF dependencies in env
export ENV_NAME="envAItf_$sysN"
bash env-files/tensorflow/generic_tf.sh
source $ENV_NAME/bin/activate

# JUBE benchmarking environment
if [ -f "${cDir}/envAI_${sysN}/bin/jube" ]; then
  echo 'JUBE already installed'
else
  pip install --no-cache-dir http://apps.fz-juelich.de/jsc/jube/jube2/download.php?version=latest
fi

# # get rest of the libraries$
# if [ "$cont1" = true ] ; then
#   pip install -r reqs_TF.txt #--ignore-installed
# fi

