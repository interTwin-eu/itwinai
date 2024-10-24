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

# Install itwinai
COPY pyproject.toml pyproject.toml
COPY src src
RUN pip install --no-cache-dir . \
    && itwinai sanity-check --tensorflow --optional-deps ray

# Additional pip deps
ARG REQUIREMENTS=env-files/torch/requirements/requirements.txt
COPY "${REQUIREMENTS}" additional-requirements.txt
RUN pip install --no-cache-dir -r additional-requirements.txt

# ENTRYPOINT [ "/bin/sh" ]
# CMD [  ]
