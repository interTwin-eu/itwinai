# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

# Dockerfile for slim itwinai image. MPI, CUDA and other need to be mounted from the host machine.

ARG BASE_IMG_NAME=python:3.10-slim

FROM nvcr.io/nvidia/pytorch:24.05-py3 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    # build-essential \
    # curl \
    # libopenmpi-dev \
    # python3-mpi4py \
    python3.10-venv \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# # DeepSpeed, Horovod and other deps
# ENV HOROVOD_WITH_PYTORCH=1 \
#     HOROVOD_WITHOUT_TENSORFLOW=1 \
#     HOROVOD_WITHOUT_MXNET=1 \
#     CMAKE_CXX_STANDARD=17 \
#     HOROVOD_MPI_THREADS_DISABLE=1 \
#     HOROVOD_CPU_OPERATIONS=MPI \
#     HOROVOD_GPU_ALLREDUCE=NCCL \
#     HOROVOD_NCCL_LINK=SHARED \
#     # DeepSpeed
#     # This is disabled as it needs OneCCL
#     # DS_BUILD_CCL_COMM=1 \
#     DS_BUILD_UTILS=1 \
#     DS_BUILD_AIO=1 \
#     # Disable some DeepSpeed OPS since apex and transformers are not installed in the current venv
#     DS_BUILD_FUSED_ADAM=0 \
#     DS_BUILD_FUSED_LAMB=0 \
#     DS_BUILD_TRANSFORMER=0 \
#     DS_BUILD_STOCHASTIC_TRANSFORMER=0 \
#     DS_BUILD_TRANSFORMER_INFERENCE=0

# # TODO: replace mpi4py build with install of linux binary "python3-mpi4py"
# # User /usr/local/bin/python3.10 explicitly to force /opt/venv/bin/python to point to python3.10. Needed to link
# # /usr/local/bin/python3.10 (in the app image) to /usr/bin/python3.10 (in the builder image)
# RUN /usr/bin/python3.10 -m venv /opt/venv \
#     && pip install --no-cache-dir --upgrade pip \
#     # https://github.com/mpi4py/mpi4py/pull/431
#     && env SETUPTOOLS_USE_DISTUTILS=local python -m pip install --no-cache-dir mpi4py \
#     && pip install --no-cache-dir \
#     # Needed to install horovod
#     wheel

# RUN /usr/bin/python3.10 -m venv /opt/venv \
#     # wheel needed to install horovod
#     && pip install --no-cache-dir --upgrade pip wheel 

# Install itwinai with torch
WORKDIR /app
COPY pyproject.toml pyproject.toml
COPY src src
RUN /usr/bin/python3.10 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .[torch] --extra-index-url https://download.pytorch.org/whl/cu124 \
    "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@v0.0.1" \
    pytest

# # Install DeepSpeed, Horovod and Ray
# RUN CONTAINER_TORCH_VERSION="$(python -c 'import torch;print(torch.__version__)')" \
#     && pip install --no-cache-dir torch=="$CONTAINER_TORCH_VERSION" \
#     deepspeed==0.15.* \
#     git+https://github.com/horovod/horovod.git@3a31d93 \
#     "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@v0.0.1" \
#     pytest

# Installation sanity check
RUN itwinai sanity-check --torch \
    # --optional-deps deepspeed \
    # --optional-deps horovod \
    --optional-deps prov4ml \
    --optional-deps ray

# App image
FROM ${BASE_IMG_NAME}
ARG BASE_IMG_NAME

COPY --from=build /opt/venv /opt/venv

# Link /usr/local/bin/python3.10 (in the app image) to /usr/bin/python3.10 (in the builder image)
RUN ln -s /usr/local/bin/python3.10 /usr/bin/python3.10

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     # OpenMPI dev library needed to build Horovod
#     libopenmpi-dev \
#     # mpi4py, which may be needed and also installs mpirun
#     python3-mpi4py \
#     # # Needed to pull OpenMPI
#     # wget \
#     && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# # TODO: consider installing OpenMPI through libopenmpi-dev and python3-mpi4py instead
# # OpenMPI: consider installing mpi4py binary for linux instead
# WORKDIR /tmp/ompi
# ENV OPENMPI_VERSION=4.1.6 \
#     OPENMPI_MINOR=4.1 
# ENV OPENMPI_URL="https://download.open-mpi.org/release/open-mpi/v${OPENMPI_MINOR}/openmpi-${OPENMPI_VERSION}.tar.gz" 
# ENV OPENMPI_DIR=/opt/openmpi-${OPENMPI_VERSION} 
# ENV PATH="${OPENMPI_DIR}/bin:${PATH}" 
# ENV LD_LIBRARY_PATH="${OPENMPI_DIR}/lib:${LD_LIBRARY_PATH}" 
# ENV MANPATH=${OPENMPI_DIR}/share/man:${MANPATH}
# RUN wget -q -O openmpi-$OPENMPI_VERSION.tar.gz $OPENMPI_URL && tar xzf openmpi-$OPENMPI_VERSION.tar.gz \
#     && cd openmpi-$OPENMPI_VERSION && ./configure --prefix=$OPENMPI_DIR && make install

# Activate the virtualenv in the container
# See here for more information:
# https://pythonspeed.com/articles/multi-stage-docker-python/
ENV PATH="/opt/venv/bin:$PATH"

# # Singularity may change the $PATH, hence this env var may increase the chances that the venv
# # is actually recognised
# ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONNOUSERSITE=1 \
    # Prevent silent override of PYTHONPATH by Singularity/Apptainer
    PYTHONPATH="" \
    SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"

RUN itwinai sanity-check --torch \
    # --optional-deps deepspeed \
    # --optional-deps horovod \
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
