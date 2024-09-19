ARG IMG_TAG=24.08-tf2-py3

# 23.09-tf2-py3: https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-23-09.html
# 24.04-tf2-py3: https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-24-0.html
# 24.08-tf2-py3: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-08.html

FROM nvcr.io/nvidia/tensorflow:${IMG_TAG}

WORKDIR /usr/src/app

# Install itwinai
COPY pyproject.toml ./
COPY src ./
COPY env-files/tensorflow/create_container_env.sh ./
RUN bash create_container_env.sh

# Create non-root user
RUN groupadd -g 10001 jovyan \
    && useradd -m -u 10000 -g jovyan jovyan \
    && chown -R jovyan:jovyan /usr/src/app
USER jovyan:jovyan

# ENTRYPOINT [ "/bin/sh" ]
# CMD [  ]

LABEL org.opencontainers.image.source=https://github.com/interTwin-eu/itwinai
LABEL org.opencontainers.image.description="Base itwinai image with tensorflow dependencies and CUDA drivers"
LABEL org.opencontainers.image.licenses=MIT
LABEL maintainer="Matteo Bunino - matteo.bunino@cern.ch"