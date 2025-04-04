# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

FROM python:3.11-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml pyproject.toml
COPY src src
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir ".[torch]" --extra-index-url https://download.pytorch.org/whl/cpu

COPY tutorials/distributed-ml/torch-k8s/train-cpu.py train-cpu.py

LABEL org.opencontainers.image.authors="Matteo Bunino - matteo.bunino@cern.ch"
LABEL org.opencontainers.image.url="https://github.com/interTwin-eu/itwinai"
LABEL org.opencontainers.image.documentation="https://itwinai.readthedocs.io/"
LABEL org.opencontainers.image.source="https://github.com/interTwin-eu/itwinai"
LABEL org.opencontainers.image.vendor="CERN - European Organization for Nuclear Research"