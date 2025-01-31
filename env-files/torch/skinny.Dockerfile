# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------


ARG BASE_IMG_NAME=python:3.10-slim

FROM ${BASE_IMG_NAME}
ARG BASE_IMG_NAME


# Install itwinai with torch
WORKDIR /app
COPY pyproject.toml pyproject.toml
COPY src src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .[torch] --extra-index-url https://download.pytorch.org/whl/cu124 \
    "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@new-main" \
    pytest \
    pytest-xdist \
    psutil

# Installation sanity check
RUN itwinai sanity-check --torch \
    --optional-deps prov4ml \
    --optional-deps ray

COPY tests tests
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
LABEL org.opencontainers.image.description="Skinny base itwinai image for torch"
LABEL org.opencontainers.image.base.digest=${BASE_IMG_DIGEST}
LABEL org.opencontainers.image.base.name=${BASE_IMG_NAME}
