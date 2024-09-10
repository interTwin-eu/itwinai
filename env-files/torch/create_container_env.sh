#!/bin/bash

# Install dependencies in container, assuming that the container image
# is from NGC and pytorch is already installed.

if [ -z "$1" ]; then
    echo "ERROR: NGC tag not specified"
    exit 2
fi

if [ "$1" == "23.09-py3" ]; then
    # Tested for torch==2.1.0

    pip install --no-cache-dir --upgrade pip
    # pip install --no-cache-dir lightning torchmetrics wheel ray ray[tune]

    # DeepSpeed
    
    # export DS_BUILD_CCL_COMM=1 # temporarily disabled. 
    # To install it see:
    # https://github.com/intel/torch-ccl?tab=readme-ov-file#install-prebuilt-wheel
    # https://github.com/oneapi-src/oneCCL?tab=readme-ov-file#installation
    
    export DS_BUILD_UTILS=1
    export DS_BUILD_AIO=1
    export DS_BUILD_FUSED_ADAM=1
    export DS_BUILD_FUSED_LAMB=1
    export DS_BUILD_TRANSFORMER=1
    export DS_BUILD_STOCHASTIC_TRANSFORMER=1
    export DS_BUILD_TRANSFORMER_INFERENCE=1

    pip3 install --no-cache-dir deepspeed || exit 1

    # fix .triton/autotune/Fp16Matmul_2d_kernel.pickle bug
    pver="$(python --version 2>&1 | awk '{print $2}' | cut -f1-2 -d.)"
    line=$(cat -n /usr/lib/python${pver}/site-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py | grep os.rename | awk '{print $1}' | head -n 1)
    sed -i "${line}s|^|#|" /usr/lib/python${pver}/site-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py || exit 1

    # Horovod
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

    # Fix needed to compile horovod with torch >= 2.1
    # https://github.com/horovod/horovod/pull/3998
    # Assume that Horovod env vars are already in the current env!
    pip3 install --no-cache-dir git+https://github.com/thomas-bouvier/horovod.git@compile-cpp17 || exit 1

    # Install Pov4ML
    if [[ "$OSTYPE" =~ ^darwin ]] ; then
      pip install "prov4ml[apple]@git+https://github.com/matbun/ProvML@main" || exit 1
    else
      pip install "prov4ml[linux]@git+https://github.com/matbun/ProvML@main" || exit 1
    fi

    # Install itwinai
    # $(python -c 'import torch;print(torch.__version__)') serves to enforce that the current version of
    # torch in the container is preserved, otherwise, if updated, Horovod will complain.
    pip install .[torch] torch==$(python -c 'import torch;print(torch.__version__)') --no-cache-dir || exit 1

else
    echo "ERROR: unrecognized tag."
    exit 2
fi
