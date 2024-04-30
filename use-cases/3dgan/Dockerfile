FROM nvcr.io/nvidia/pytorch:23.09-py3
# FROM python:3.11

WORKDIR /usr/src/app

# Install itwinai
COPY pyproject.toml ./
COPY src ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir lightning \
    && pip install --no-cache-dir .

# Add 3DGAN use case files and install additional requirements
COPY use-cases/3dgan/requirements.txt ./
COPY use-cases/3dgan/* ./
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN groupadd -g 10001 dotnet \
    && useradd -m -u 10000 -g dotnet dotnet \
    && chown -R dotnet:dotnet /usr/src/app
USER dotnet:dotnet

# ENTRYPOINT [ "itwinai", "exec-pipeline" ]
# CMD [ "--config", "pipeline.yaml" ]