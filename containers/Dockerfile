# Set the base image to Python 3.9.17-slim
#FROM python:3.9.17-slim AS build


#Create from the following container
#FROM nvcr.io/nvidia/pytorch:21.12-py3 AS build
FROM nvcr.io/nvidia/pytorch:23.09-py3 AS build

WORKDIR ai

# Copy the required files to the container
COPY env-files      env-files
COPY src            src
COPY tests          tests
COPY pyproject.toml pyproject.toml 
#COPY setup.py   setup.py
#COPY README.md  README.md

# Set environment variable
ENV LC_ALL=C

# Install packages
#RUN apt update && apt install -y iputils-ping && apt install -y telnet && apt install -y krb5-user && apt install -y nano
#RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Update pip and install python requirements
RUN pip install --upgrade pip
#RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN ls 
#RUN pip install -r env-files/torch/pytorch-env-gpu-container.txt
RUN pip install -e .

# Define the run command when the container starts
#CMD ["sleep", "300"]
#CMD ["python", "-V"]
CMD ["python", "train.py", "-p", "pipeline.yaml"]

# Additional information labels
LABEL Author="azoechba"
LABEL Version="v0.0.1"
LABEL Description="This is the base image of the itwinai library in Docker."
