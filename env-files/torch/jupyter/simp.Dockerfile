FROM ubuntu:jammy

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pip \
    python3.10 

RUN pip install --no-cache-dir "jupyterhub<2" jupyterlab 

