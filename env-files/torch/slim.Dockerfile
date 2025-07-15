# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

# Dockerfile for slim itwinai image. MPI, CUDA and other need to be mounted from the host machine.

ARG BASE_IMG_NAME=python:3.12-slim

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS build

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    # Needed (at least) by horovod wheel builder
    cmake \
    git \
    # Needed (at least) by horovod wheel builder
    libopenmpi-dev \
    python3 \
    # Needed (at least) by horovod wheel builder
    python3-mpi4py \
    # Needed to build horovod
    python3.12-dev \
    # # Needed to create virtual envs
    # python3.12-venv \
    wget \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Cleanup
RUN rm -rf /tmp/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    # https://docs.astral.sh/uv/reference/environment/#uv_no_cache
    UV_NO_CACHE=1

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

# Install itwinai with torch
WORKDIR /app
COPY pyproject.toml pyproject.toml
COPY src src
# install uv so that uv → /usr/local/bin/uv
RUN wget -qO - https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh
RUN uv venv /opt/venv 
ENV UV_PYTHON=/opt/venv/bin/python
RUN uv pip install --no-cache-dir --upgrade pip wheel \
    # Needed to run deepspeed (and Horovod?) with MPI backend
    && uv pip install --no-cache-dir mpi4py \
    && uv pip install --no-cache-dir \
    # Select from which index to install torch
    --index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.org/simple \
    # This is needed by UV to trust all indexes:
    --index-strategy unsafe-best-match \
    # Install packages
    .[torch] \
    "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@v0.0.2" \
    # Minimal installation to run CI tests in the container with pytest
    pytest \
    pytest-xdist \
    psutil

# Install DeepSpeed and Horovod
RUN uv pip install --no-cache-dir \
    # Needed when working with uv venv
    --no-build-isolation \
    deepspeed==0.16.8 \
    git+https://github.com/horovod/horovod.git@3a31d93 


# Installation sanity check
RUN itwinai sanity-check --torch \
    --optional-deps deepspeed \
    --optional-deps horovod \
    --optional-deps prov4ml \
    --optional-deps ray





# App image
FROM ${BASE_IMG_NAME}
ARG BASE_IMG_NAME

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY --from=build /opt/venv /opt/venv

# Override symlink in the venv
RUN ln -sf /usr/local/bin/python3.12 /opt/venv/bin/python

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    # OpenMPI dev library needed to build Horovod
    libopenmpi-dev \
    # mpi4py, which may be needed and also installs mpirun
    python3-mpi4py \
    # Needed to pull OpenMPI and to use this container in ray k8s cluster as Head/Worker container
    wget \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*


# # Singularity may change the $PATH, hence this env var may increase the chances that the venv
# # is actually recognised
# ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONNOUSERSITE=1 \
    # Prevent silent override of PYTHONPATH by Singularity/Apptainer
    PYTHONPATH="" \
    SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt" \
    # https://docs.astral.sh/uv/reference/environment/#uv_no_cache
    UV_NO_CACHE=1 \
    UV_PYTHON=/opt/venv/bin/python

# Install uv
RUN wget -qO - https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh

# Make sure that the virualenv is reacheable also from login shell
# This is needed to use this container as Ray Worker/Head on k8s
RUN echo 'PATH="/opt/venv/bin:$PATH"' >> /etc/profile

RUN itwinai sanity-check --torch \
    --optional-deps deepspeed \
    --optional-deps horovod \
    --optional-deps prov4ml \
    --optional-deps ray

WORKDIR /app
COPY pyproject.toml pyproject.toml
COPY tests tests
# Add Dockerfile
COPY env-files/torch/slim.Dockerfile Dockerfile

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
LABEL org.opencontainers.image.description="Lightweight base itwinai image with torch dependencies without CUDA drivers"
LABEL org.opencontainers.image.base.digest=${BASE_IMG_DIGEST}
LABEL org.opencontainers.image.base.name=${BASE_IMG_NAME}
