# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

ARG BASE_IMG_NAME=nvcr.io/nvidia/tensorflow:24.08-tf2-py3

# 23.09-tf2-py3: https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-23-09.html
# 24.04-tf2-py3: https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-24-0.html
# 24.08-tf2-py3: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-08.html

FROM  ${BASE_IMG_NAME}
ARG BASE_IMG_NAME

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    # Needed by Prov4ML/yProvML to generate provenance graph
    dot2tex \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
    tf_keras==2.16.* \
    "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@v0.0.1"

# Inprove robustness: avoid silent override by Singularity/Apptainer
ENV PYTHONPATH=""

# Install itwinai
COPY pyproject.toml pyproject.toml
COPY src src
RUN pip install --no-cache-dir .[tf-cuda] \
    && itwinai sanity-check --tensorflow --optional-deps ray

# Additional pip deps
ARG REQUIREMENTS=env-files/torch/requirements/requirements.txt
COPY "${REQUIREMENTS}" additional-requirements.txt
RUN pip install --no-cache-dir -r additional-requirements.txt

# Add tests
COPY tests tests

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
LABEL org.opencontainers.image.description="Base itwinai image with tensorflow dependencies and CUDA drivers"
LABEL org.opencontainers.image.base.digest=${BASE_IMG_DIGEST}
LABEL org.opencontainers.image.base.name=${BASE_IMG_NAME}
