ARG IMG_TAG=24.05-py3 
# ARG IMG_TAG=23.09-py3

# 23.09-py3: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-09.html
# 24.04-py3: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-04.html

FROM nvcr.io/nvidia/pytorch:${IMG_TAG}

LABEL org.opencontainers.image.source="https://github.com/interTwin-eu/itwinai"
LABEL org.opencontainers.image.description="Base itwinai image with torch dependencies and CUDA drivers"
LABEL org.opencontainers.image.licenses=MIT
LABEL maintainer="Matteo Bunino - matteo.bunino@cern.ch"

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    # Needed by Prov4ML/yProvML to generate provenance graph
    dot2tex \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# https://github.com/mpi4py/mpi4py/pull/431
RUN pip install --no-cache-dir --upgrade pip \
    && env SETUPTOOLS_USE_DISTUTILS=local python -m pip install --no-cache-dir mpi4py

# DeepSpeed, Horovod and other deps
ENV HOROVOD_WITH_PYTORCH=1 \
    HOROVOD_WITHOUT_TENSORFLOW=1 \
    HOROVOD_WITHOUT_MXNET=1 \
    CMAKE_CXX_STANDARD=17 \
    HOROVOD_MPI_THREADS_DISABLE=1 \
    HOROVOD_CPU_OPERATIONS=MPI \
    HOROVOD_GPU_ALLREDUCE=NCCL \
    HOROVOD_NCCL_LINK=SHARED \
    # DeepSpeed
    # DS_BUILD_CCL_COMM=1 \
    DS_BUILD_UTILS=1 \
    DS_BUILD_AIO=1 \
    DS_BUILD_FUSED_ADAM=1 \
    DS_BUILD_FUSED_LAMB=1 \
    DS_BUILD_TRANSFORMER=1 \
    DS_BUILD_STOCHASTIC_TRANSFORMER=1 \
    DS_BUILD_TRANSFORMER_INFERENCE=1
# Torch: reuse the global torch in the container
RUN CONTAINER_TORCH_VERSION="$(python -c 'import torch;print(torch.__version__)')" \
    && pip install --no-cache-dir torch=="$CONTAINER_TORCH_VERSION" \
    deepspeed==0.15.* \
    git+https://github.com/horovod/horovod.git@3a31d93 \
    "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@new-main" \
    ray[tune] \
    # fix .triton/autotune/Fp16Matmul_2d_kernel.pickle bug
    && pver="$(python --version 2>&1 | awk '{print $2}' | cut -f1-2 -d.)" \
    && line=$(cat -n "/usr/local/lib/python${pver}/dist-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py" | grep os.rename | awk '{print $1}' | head -n 1) \
    && sed -i "${line}s|^|#|" "/usr/local/lib/python${pver}/dist-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py"

# Install itwinai
COPY pyproject.toml pyproject.toml
COPY src src
# Torch: reuse the global torch in the container
RUN CONTAINER_TORCH_VERSION="$(python -c 'import torch;print(torch.__version__)')" \
    && pip install --no-cache-dir torch=="$CONTAINER_TORCH_VERSION" \
    .[torch,dev] \
    && itwinai sanity-check --torch \
    --optional-deps deepspeed \
    --optional-deps horovod \
    --optional-deps ray

# Additional pip deps
ARG REQUIREMENTS=env-files/torch/requirements/requirements.txt
COPY "${REQUIREMENTS}" additional-requirements.txt
RUN pip install --no-cache-dir -r additional-requirements.txt

# Add tests
COPY tests tests
