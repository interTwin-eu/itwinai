ARG IMG_TAG=24.09-py3

# 23.09-py3: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-09.html
# 24.04-py3: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-04.html

FROM nvcr.io/nvidia/pytorch:${IMG_TAG} as build

# https://stackoverflow.com/a/56748289
ARG IMG_TAG

WORKDIR /app

# Virtual env
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
# User python3.10 explicitly to force /opt/venv/bin/python to point to python3.10. Needed to link
# /usr/local/bin/python3.10 (in the app image) to /usr/bin/python3.10 (in the builder image)
RUN python3.10 -m venv /opt/venv

# https://github.com/mpi4py/mpi4py/pull/431
RUN env SETUPTOOLS_USE_DISTUTILS=local python -m pip install --no-cache-dir mpi4py

# Install itwinai
COPY pyproject.toml ./
COPY src ./
COPY env-files/torch/create_container_env.sh ./
RUN bash create_container_env.sh ${IMG_TAG}

# App image
FROM python:3.10-slim

LABEL org.opencontainers.image.source=https://github.com/interTwin-eu/itwinai
LABEL org.opencontainers.image.description="Base itwinai image with torch dependencies and CUDA drivers"
LABEL org.opencontainers.image.licenses=MIT
LABEL maintainer="Matteo Bunino - matteo.bunino@cern.ch"

# Copy virtual env
COPY --from=build /opt/venv /opt/venv

# Link /usr/local/bin/python3.10 (in the app image) to /usr/bin/python3.10 (in the builder image)
RUN ln -s /usr/local/bin/python3.10 /usr/bin/python3.10

# Activate the virtualenv in the container
# See here for more information:
# https://pythonspeed.com/articles/multi-stage-docker-python/
ENV PATH="/opt/venv/bin:$PATH"
