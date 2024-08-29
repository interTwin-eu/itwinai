ARG IMG_TAG=24.08-tf2-py3

# 23.09-tf2-py3: tensorflow==2.13.0
# 24.04-tf2-py3: tensorflow==2.15.0

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