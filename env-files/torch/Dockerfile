# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

ARG BASE_IMG_NAME=nvcr.io/nvidia/pytorch:24.05-py3 
# ARG IMG_TAG=23.09-py3

# 23.09-py3: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-09.html
# 24.04-py3: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-04.html

FROM ${BASE_IMG_NAME}
ARG BASE_IMG_NAME

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y \
    # Needed by Prov4ML/yProvML to generate provenance graph
    dot2tex \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# TODO: replace mpi4py build with install of linux binary "python3-mpi4py"
# https://github.com/mpi4py/mpi4py/pull/431
RUN pip install --no-cache-dir --upgrade pip \
    && env SETUPTOOLS_USE_DISTUTILS=local python -m pip install --no-cache-dir mpi4py

# Install itwinai with torch
COPY pyproject.toml pyproject.toml
COPY src src
RUN pip install --no-cache-dir .[torch] --extra-index-url https://download.pytorch.org/whl/cu124

# Install DeepSpeed, Horovod and Ray
ENV HOROVOD_WITH_PYTORCH=1 \
    HOROVOD_WITHOUT_TENSORFLOW=1 \
    HOROVOD_WITHOUT_MXNET=1 \
    CMAKE_CXX_STANDARD=17 \
    HOROVOD_MPI_THREADS_DISABLE=1 \
    HOROVOD_CPU_OPERATIONS=MPI \
    HOROVOD_GPU_ALLREDUCE=NCCL \
    HOROVOD_NCCL_LINK=SHARED \
    # DeepSpeed
    # This is disabled as it needs OneCCL
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
    && echo "CONTAINER_TORCH_VERSION: $CONTAINER_TORCH_VERSION" \
    && pip install --no-cache-dir torch=="$CONTAINER_TORCH_VERSION" \
    deepspeed==0.15.* \
    git+https://github.com/horovod/horovod.git@3a31d93 \
    "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@v0.0.1" \
    pytest \
    pytest-xdist \
    psutil \
    # fix .triton/autotune/Fp16Matmul_2d_kernel.pickle bug
    && pver="$(python --version 2>&1 | awk '{print $2}' | cut -f1-2 -d.)" \
    && line=$(cat -n "/usr/local/lib/python${pver}/dist-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py" | grep os.rename | awk '{print $1}' | head -n 1) \
    && sed -i "${line}s|^|#|" "/usr/local/lib/python${pver}/dist-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py"

# Installation sanity check
RUN itwinai sanity-check --torch \
    # Microsoft DeepSpeed for distributed ML
    --optional-deps deepspeed \
    # Horovod for distributed ML
    --optional-deps horovod \
    # Prov4ML provenance logger
    --optional-deps prov4ml \
    # Ray for disrtibuted ML and hyperparameter-tuning
    --optional-deps ray

# Additional pip deps
ARG REQUIREMENTS=env-files/torch/requirements/requirements.txt
COPY "${REQUIREMENTS}" additional-requirements.txt
RUN pip install --no-cache-dir -r additional-requirements.txt

# Env vars for robustness
ENV PYTHONNOUSERSITE=1 \
    # Prevent silent override of PYTHONPATH by Singularity/Apptainer
    PYTHONPATH="" \
    SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"

# Add tests
COPY tests tests
# Add Dockerfile
COPY env-files/torch/Dockerfile Dockerfile


# Labels
ARG CREATION_DATE
ARG COMMIT_HASH
ARG ITWINAI_VERSION
ARG IMAGE_FULL_NAME
ARG BASE_IMG_DIGEST

# https://github.com/opencontainers/image-spec/blob/main/annotations.md#pre-defined-annotation-keys
LABEL org.opencontainers.image.created=${CREATION_DATE}
LABEL org.opencontainers.image.authors="Matteo Bunino - matteo.bunino@cern.ch"
LABEL org.opencontainers.image.url="https://github.com/interTwin-eu/itwinai"
LABEL org.opencontainers.image.documentation="https://itwinai.readthedocs.io/"
LABEL org.opencontainers.image.source="https://github.com/interTwin-eu/itwinai"
LABEL org.opencontainers.image.version=${ITWINAI_VERSION}
LABEL org.opencontainers.image.revision=${COMMIT_HASH}
LABEL org.opencontainers.image.vendor="CERN - European Organization for Nuclear Research"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.ref.name=${IMAGE_FULL_NAME}
LABEL org.opencontainers.image.title="itwinai"
LABEL org.opencontainers.image.description="Base itwinai image with torch dependencies and CUDA drivers"
LABEL org.opencontainers.image.base.digest=${BASE_IMG_DIGEST}
LABEL org.opencontainers.image.base.name=${BASE_IMG_NAME}
