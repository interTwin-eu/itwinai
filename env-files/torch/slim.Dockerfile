# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Alex Krochak <o.krochak@fz-juelich.de> - JSC
# --------------------------------------------------------------------------------------

# Dockerfile for slim itwinai image. MPI, CUDA and other need to be mounted from the host machine.

ARG BASE_IMG_NAME=ubuntu:24.04

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
ARG RUCIO_CLIENTS_VERSION=39.*
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
    # RUCIO clients go into the itwinai venv so that user code can use the RUCIO Python API
    # (rucio.client.Client, DownloadClient) alongside torch in a single interpreter. Resolving
    # them together with itwinai here means a future incompatibility fails the build rather than
    # silently diverging: rucio-clients declares all its dependencies unpinned.
    "rucio-clients[argcomplete]==${RUCIO_CLIENTS_VERSION}" \
    # "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@v0.0.2" \
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
    --optional-deps yprov4ml \
    --optional-deps ray


# App image
FROM ${BASE_IMG_NAME}
ARG BASE_IMG_NAME

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY --from=build /opt/venv /opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    # OpenMPI dev library needed to build Horovod
    libopenmpi-dev \
    # Install Python
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    # mpi4py, which may be needed and also installs mpirun
    python3-mpi4py \
    # Needed to pull OpenMPI and to use this container in ray k8s cluster as Head/Worker container
    wget \
    # RUCIO transfer stack. Ubuntu ships gfal2, its plugins and the XRootD client natively, so
    # no separate conda env (and no second Python interpreter) is needed for them.
    gfal2 \
    gfal2-util-scripts \
    python3-gfal2 \
    python3-gfal2-util \
    # One plugin per transfer protocol; without these gfal2 cannot talk to any RSE
    gfal2-plugin-file \
    gfal2-plugin-gridftp \
    gfal2-plugin-http \
    gfal2-plugin-srm \
    gfal2-plugin-xrootd \
    xrootd-client \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# IGTF-accredited CA bundle - needed for GFAL2 to trust RSE storage endpoints
RUN set -euo pipefail && \
    mkdir -p /etc/grid-security/certificates /etc/apt/keyrings && \
    wget -qO /etc/apt/keyrings/egi-igtf.asc \
    https://repository.egi.eu/sw/production/cas/1/current/GPG-KEY-EUGridPMA-RPM-4 && \
    echo "deb [signed-by=/etc/apt/keyrings/egi-igtf.asc] https://repository.egi.eu/sw/production/cas/1/current egi-igtf core" \
    > /etc/apt/sources.list.d/egi-igtf.list && \
    apt-get update && apt-get install -y ca-policy-egi-core && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/* && \
    test "$(find /etc/grid-security/certificates -name '*.0' | wc -l)" -gt 50


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
    --optional-deps yprov4ml \
    --optional-deps ray

# Expose the apt-installed gfal2 bindings inside the itwinai venv
RUN set -euo pipefail && \
    /opt/venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' && \
    SITE="$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" && \
    ln -s /usr/lib/python3/dist-packages/gfal2.so "${SITE}/gfal2.so" && \
    ln -s /usr/lib/python3/dist-packages/gfal2_util "${SITE}/gfal2_util"

ENV GFAL_PYTHONBIN=/opt/venv/bin/python \
    X509_CERT_DIR=/etc/grid-security/certificates

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
LABEL org.opencontainers.image.authors="Matteo Bunino - matteo.bunino@cern.ch & Alex Krochak - o.krochak@fz-juelich.de"
LABEL org.opencontainers.image.url="https://github.com/interTwin-eu/itwinai"
LABEL org.opencontainers.image.documentation="https://itwinai.readthedocs.io/"
LABEL org.opencontainers.image.source="https://github.com/interTwin-eu/itwinai"
LABEL org.opencontainers.image.version=${ITWINAI_VERSION}
LABEL org.opencontainers.image.revision=${COMMIT_HASH}
LABEL org.opencontainers.image.vendor="CERN - European Organization for Nuclear Research & JSC - Jülich Supercomputing Centre"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.ref.name=${IMAGE_FULL_NAME}
LABEL org.opencontainers.image.title="itwinai"
LABEL org.opencontainers.image.description="Lightweight base itwinai image with torch dependencies without CUDA drivers"
LABEL org.opencontainers.image.base.digest=${BASE_IMG_DIGEST}
LABEL org.opencontainers.image.base.name=${BASE_IMG_NAME}
