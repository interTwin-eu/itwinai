# FROM python:3.9.12
FROM nvcr.io/nvidia/pytorch:23.09-py3

WORKDIR /usr/src/app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir lightning

# Add 3DGAN custom requirements
COPY use-cases/3dgan/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install itwinai and dependencies
COPY pyproject.toml ./
COPY src ./
RUN  pip install --no-cache-dir .

# Add 3DGAN use case files
COPY use-cases/3dgan/* ./