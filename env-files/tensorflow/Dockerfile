FROM nvcr.io/nvidia/tensorflow:23.09-tf2-py3

# 24.04-tf2-py3: tensorflow==2.15.0
# 23.09-tf2-py3: tensorflow==2.13.0

WORKDIR /usr/src/app

# Install itwinai
COPY pyproject.toml ./
COPY src ./
COPY env-files/tensorflow/create_container_env.sh ./
RUN bash create_container_env.sh

# Create non-root user
RUN groupadd -g 10001 dotnet \
    && useradd -m -u 10000 -g dotnet dotnet \
    && chown -R dotnet:dotnet /usr/src/app
USER dotnet:dotnet

# ENTRYPOINT [ "/bin/sh" ]
# CMD [  ]