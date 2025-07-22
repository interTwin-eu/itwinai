# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------


ARG BASE_IMG_NAME=python:3.12-slim

FROM ${BASE_IMG_NAME}
ARG BASE_IMG_NAME

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*


ENV \
    # https://docs.astral.sh/uv/reference/environment/#uv_no_cache
    UV_NO_CACHE=1

# Install UV package manager and upgrade pip
RUN pip install --no-cache-dir --upgrade pip uv

# Install itwinai with torch
WORKDIR /app
COPY pyproject.toml pyproject.toml
COPY src src
RUN uv venv \
    && uv pip install --no-cache-dir --upgrade pip \
    && uv pip install --no-cache-dir \
    # Select from which index to install torch
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    # This is needed by UV to trust all indexes:
    --index-strategy unsafe-best-match \
    # Install packages
    .[torch] \
    "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@v0.0.2" \
    pytest \
    pytest-xdist \
    psutil

# Make uv venv the default python env
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Installation sanity check
RUN itwinai sanity-check --torch \
    --optional-deps prov4ml \
    --optional-deps ray

COPY env-files/torch/skinny.Dockerfile Dockerfile
COPY tests tests

# This is here because the skinny container could be used to run PR tests
COPY use-cases use-cases
# This is needed to override the default venv path used by functional
# tests for torch-based use cases (under ./use-cases)
ENV TORCH_ENV="/app/.venv"

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
LABEL org.opencontainers.image.description="Skinny base itwinai image for torch"
LABEL org.opencontainers.image.base.digest=${BASE_IMG_DIGEST}
LABEL org.opencontainers.image.base.name=${BASE_IMG_NAME}
