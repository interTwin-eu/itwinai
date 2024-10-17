ARG IMG_TAG=23.09-py3

# 23.09-py3: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-09.html
# 24.04-py3: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-04.html

FROM nvcr.io/nvidia/pytorch:${IMG_TAG}

LABEL org.opencontainers.image.source=https://github.com/interTwin-eu/itwinai
LABEL org.opencontainers.image.description="Base itwinai image with torch dependencies and CUDA drivers"
LABEL org.opencontainers.image.licenses=MIT
LABEL maintainer="Matteo Bunino - matteo.bunino@cern.ch"

# https://stackoverflow.com/a/56748289
ARG IMG_TAG

WORKDIR /app

# https://github.com/mpi4py/mpi4py/pull/431
RUN env SETUPTOOLS_USE_DISTUTILS=local python -m pip install --no-cache-dir mpi4py

# Install itwinai
COPY pyproject.toml ./
COPY src ./
COPY env-files/torch/create_container_env.sh ./
RUN bash create_container_env.sh ${IMG_TAG}

# Create non-root user
RUN groupadd -g 10001 jarvis \
    && useradd -m -u 10000 -g jarvis jarvis \
    && chown -R jarvis:jarvis /app
USER jarvis:jarvis
