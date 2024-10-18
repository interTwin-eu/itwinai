FROM nvcr.io/nvidia/pytorch:24.09-py3

SHELL ["/bin/bash", "-c"]

WORKDIR /opt/mamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba \
    && ./bin/micromamba shell init -s bash -r /opt/mamba/micromamba
ENV PATH="/opt/mamba/bin:$PATH"