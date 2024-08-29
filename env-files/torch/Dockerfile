ARG IMG_TAG=23.09-py3

# 23.09-py3: torch==2.1.0
# 24.04-py3: torch==2.3.0

FROM nvcr.io/nvidia/pytorch:${IMG_TAG}

WORKDIR /usr/src/app

# https://github.com/mpi4py/mpi4py/pull/431
RUN env SETUPTOOLS_USE_DISTUTILS=local python -m pip install --no-cache-dir mpi4py

# Install itwinai
COPY pyproject.toml ./
COPY src ./
COPY env-files/torch/create_container_env.sh ./
RUN bash create_container_env.sh ${IMG_TAG}

# Create non-root user
RUN groupadd -g 10001 jovyan \
    && useradd -m -u 10000 -g jovyan jovyan \
    && chown -R jovyan:jovyan /usr/src/app
USER jovyan:jovyan
