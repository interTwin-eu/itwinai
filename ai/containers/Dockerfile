# Set the base image to Python 3.9.17-slim
FROM python:3.9.17-slim AS build


WORKDIR ai

# Copy the required files to the container
COPY env-files  env-files
COPY src        src
COPY tests      tests
COPY setup.py   setup.py
COPY README.md  README.md

# Set environment variable
ENV LC_ALL=C

# Install packages
RUN apt update && apt install -y iputils-ping && apt install -y telnet && apt install -y krb5-user && apt install -y nano
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Update pip and install python requirements
RUN pip install --upgrade pip
RUN pip install -r /ai/env-files/pytorch-env-container.txt
RUN pip install -e /ai

# Define the run command when the container starts
CMD ["python", "-V"]

# Additional information labels
LABEL Author="azoechba"
LABEL Version="v0.0.1"
LABEL Description="This is the base image of the itwinai library in Docker."
