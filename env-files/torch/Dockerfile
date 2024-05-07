# FROM nvcr.io/nvidia/pytorch:24.04-py3 
FROM nvcr.io/nvidia/pytorch:23.09-py3 

# 23.09-py3: torch==2.1.0
# 24.04-py3: torch==2.3.0

WORKDIR /usr/src/app

# Install itwinai
COPY pyproject.toml ./
COPY src ./
COPY env-files/torch/create_container_env.sh ./
RUN bash create_container_env.sh 23.09-py3

# Create non-root user
RUN groupadd -g 10001 dotnet \
    && useradd -m -u 10000 -g dotnet dotnet \
    && chown -R dotnet:dotnet /usr/src/app
USER dotnet:dotnet

# ENTRYPOINT [ "/bin/sh" ]
# CMD [  ]