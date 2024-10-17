#!/bin/bash

pip install --no-cache-dir --upgrade pip 
pip install --no-cache-dir packaging wheel 

# Adding this constraint as numpy >= 2 seems to clash with DeepSpeed
pip install --no-cache-dir 'numpy<2.0.0' || exit 1

# HPO - RayTune
pip install --no-cache-dir ray ray[tune] || exit 1


# DeepSpeed
pip install --no-cache-dir py-cpuinfo || exit 1
pip install --no-cache-dir deepspeed || exit 1

# # fix .triton/autotune/Fp16Matmul_2d_kernel.pickle bug
# line=$(cat -n $ENV_NAME/lib/python${pver}/site-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py | grep os.rename | awk '{print $1}' | head -n 1)

# # 'sed' is implemented differently on MacOS than on Linux (https://stackoverflow.com/questions/4247068/sed-command-with-i-option-failing-on-mac-but-works-on-linux)
# if [[ "$OSTYPE" =~ ^darwin ]] ; then
#     sed -i '' "${line}s|^|#|" $ENV_NAME/lib/python${pver}/site-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py || exit 1
# else
#     sed -i "${line}s|^|#|" $ENV_NAME/lib/python${pver}/site-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py || exit 1
# fi


# install horovod
export HOROVOD_WITH_PYTORCH=1
export HOROVOD_WITHOUT_TENSORFLOW=1
export HOROVOD_WITHOUT_MXNET=1

pip install --no-cache-dir git+https://github.com/horovod/horovod.git || exit 1

# Install Pov4ML
pip install --no-cache-dir "prov4ml[linux]@git+https://github.com/matbun/ProvML" || exit 1

# Install itwinai
pip install --no-cache-dir ".[torch,dev]" || exit 1

# Check itwinai installation
itwinai sanity-check --torch || exit 1