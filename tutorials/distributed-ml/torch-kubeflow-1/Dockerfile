# FROM ghcr.io/intertwin-eu/itwinai-dev:d01124feb254c6c98a1e5b3ecc76b63b410d03cb-torch-slim
FROM python:3.11-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y \
    # build-essential \
    git \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
#     && pip install --no-cache-dir "itwinai[torch]@git+https://github.com/interTwin-eu/itwinai@33f13c8" --extra-index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml pyproject.toml
COPY src src
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir ".[torch]" --extra-index-url https://download.pytorch.org/whl/cpu

COPY tutorials/distributed-ml/torch-k8s/train-cpu.py train-cpu.py
