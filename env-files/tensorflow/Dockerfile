ARG IMG_TAG=24.08-tf2-py3

# 23.09-tf2-py3: https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-23-09.html
# 24.04-tf2-py3: https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-24-0.html
# 24.08-tf2-py3: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-08.html

FROM nvcr.io/nvidia/tensorflow:${IMG_TAG}

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

LABEL org.opencontainers.image.source="https://github.com/interTwin-eu/itwinai"
LABEL org.opencontainers.image.description="Base itwinai image with tensorflow dependencies and CUDA drivers"
LABEL org.opencontainers.image.licenses=MIT
LABEL maintainer="Matteo Bunino - matteo.bunino@cern.ch"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    # Needed by Prov4ML/yProvML to generate provenance graph
    dot2tex \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
    tf_keras==2.16.* \
    "prov4ml[linux]@git+https://github.com/matbun/ProvML@new-main" 

# Install itwinai
COPY pyproject.toml pyproject.toml
COPY src src
RUN pip install . \
    && itwinai sanity-check --tensorflow

# Additional pip deps
ARG REQUIREMENTS=env-files/torch/requirements/requirements.txt
COPY "${REQUIREMENTS}" additional-requirements.txt
RUN pip install --no-cache-dir -r additional-requirements.txt

# Add tests
COPY tests tests

# ENTRYPOINT [ "/bin/sh" ]
# CMD [  ]
