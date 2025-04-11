# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

FROM ubuntu:jammy AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    # Needed (at least) by horovod wheel builder
    cmake \
    git \
    # Needed (at least) by horovod wheel builder
    libopenmpi-dev \
    # Needed (at least) by horovod wheel builder
    python3-mpi4py \
    # Needed to build horovod
    python3.10-dev \
    # Needed to create virtual envs
    python3.10-venv \
    wget \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install ROCm
# https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/amdgpu-installer/amdgpu-installer-ubuntu.html
WORKDIR /tmp/rocm
RUN wget https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/jammy/amdgpu-install_6.3.60303-1_all.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
    ./amdgpu-install_6.3.60303-1_all.deb \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* && apt-get update \
    # Install ROCm
    && amdgpu-install -y --usecase=rocm \
    # Check installation
    && echo -e "\nCheck installation" && dkms status


# Cleanup
RUN rm -rf /tmp/*

# Install itwinai
WORKDIR /app
COPY pyproject.toml pyproject.toml
COPY src src

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN /usr/bin/python3.10 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip wheel \
    # Needed to run deepspeed (and Horovod?) with MPI backend
    && pip install --no-cache-dir mpi4py 

RUN pip install --no-cache-dir .[torch] --extra-index-url https://download.pytorch.org/whl/rocm6.1 \
    "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@new-main" \
    pytest \
    pytest-xdist \
    psutil \
    wheel

# DeepSpeed, Horovod and other deps
ENV HOROVOD_GPU=ROCM \
    HOROVOD_WITH_PYTORCH=1 \
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
    # Disable some DeepSpeed OPS since apex and transformers are not installed in the current venv
    # DS_BUILD_FUSED_ADAM=0 \
    # DS_BUILD_FUSED_LAMB=0 \
    # DS_BUILD_TRANSFORMER=0 \
    DS_BUILD_STOCHASTIC_TRANSFORMER=0 \
    DS_BUILD_TRANSFORMER_INFERENCE=0

# Install DeepSpeed, Horovod and Ray
RUN CONTAINER_TORCH_VERSION="$(python -c 'import torch;print(torch.__version__)')" \
    && pip install --no-cache-dir torch=="$CONTAINER_TORCH_VERSION" \
    deepspeed==0.15.* \
    git+https://github.com/horovod/horovod.git@3a31d93 

# Installation sanity check
RUN itwinai sanity-check --torch \
    --optional-deps deepspeed \
    --optional-deps horovod \
    --optional-deps prov4ml \
    --optional-deps ray