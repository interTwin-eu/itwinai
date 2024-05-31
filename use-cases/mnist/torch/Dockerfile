# FROM python:3.9
FROM nvcr.io/nvidia/pytorch:23.09-py3

WORKDIR /usr/src/app

# Install pytorch (cpuonly)
# Ref:https://pytorch.org/get-started/previous-versions/#linux-and-windows-5
RUN pip install --no-cache-dir torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

# Install itwinai and dependencies
COPY pyproject.toml ./
COPY src ./
RUN  pip install --no-cache-dir .

# Add torch MNIST use case
COPY use-cases/mnist/torch/* ./
