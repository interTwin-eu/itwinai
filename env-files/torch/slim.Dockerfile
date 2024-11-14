# Dockerfile for slim itwinai image. MPI, CUDA and other need to be mounted from the host machine.

FROM nvcr.io/nvidia/pytorch:24.05-py3 AS build

RUN apt-get update && apt-get install -y \
    build-essential \
    # cargo \
    curl \
    python3.10-venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# DeepSpeed, Horovod and other deps
ENV HOROVOD_WITH_PYTORCH=1 \
    HOROVOD_WITHOUT_TENSORFLOW=1 \
    HOROVOD_WITHOUT_MXNET=1 \
    CMAKE_CXX_STANDARD=17 \
    HOROVOD_MPI_THREADS_DISABLE=1 \
    HOROVOD_CPU_OPERATIONS=MPI \
    HOROVOD_GPU_ALLREDUCE=NCCL \
    HOROVOD_NCCL_LINK=SHARED \
    # DeepSpeed
    # This is disabled as it needs OneCCL
    # DS_BUILD_CCL_COMM=1 \
    DS_BUILD_UTILS=1 \
    DS_BUILD_AIO=1 \
    DS_BUILD_FUSED_ADAM=1 \
    DS_BUILD_FUSED_LAMB=1 \
    DS_BUILD_TRANSFORMER=1 \
    DS_BUILD_STOCHASTIC_TRANSFORMER=1 \
    DS_BUILD_TRANSFORMER_INFERENCE=1

# User /usr/local/bin/python3.10 explicitly to force /opt/venv/bin/python to point to python3.10. Needed to link
# /usr/local/bin/python3.10 (in the app image) to /usr/bin/python3.10 (in the builder image)
RUN /usr/bin/python3.10 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.4.*" \
    torchvision \
    torchaudio

# Rust compiler
RUN curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh -s -- -y

# Disable some DeepSpeed OPS as apex and transformers are not installed in the current venv
ENV DS_BUILD_FUSED_ADAM=0 \
    DS_BUILD_FUSED_LAMB=0 \
    DS_BUILD_TRANSFORMER=0 \
    DS_BUILD_STOCHASTIC_TRANSFORMER=0 \
    DS_BUILD_TRANSFORMER_INFERENCE=0

RUN pip install --no-cache-dir --global-option="-j8" --global-option="build_ext" \
    "deepspeed==0.15.*" \
    "torch==2.4.*"
RUN pip install --no-cache-dir --global-option="-j8" \
    "horovod[pytorch]@git+https://github.com/horovod/horovod.git@3a31d93" \
    "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@new-main" \
    ray[tune] \
    "torch==2.4.*"

COPY src src
COPY pyproject.toml pyproject.toml

RUN pip install --no-cache-dir .[torch,dev] \
    && itwinai sanity-check --torch \
    --optional-deps deepspeed \
    --optional-deps horovod \
    --optional-deps ray

# App image
FROM python:3.10-slim
COPY --from=build /opt/venv /opt/venv

# Link /usr/local/bin/python3.10 (in the app image) to /usr/bin/python3.10 (in the builder image)
RUN ln -s /usr/local/bin/python3.10 /usr/bin/python3.10

# Activate the virtualenv in the container
# See here for more information:
# https://pythonspeed.com/articles/multi-stage-docker-python/
ENV PATH="/opt/venv/bin:$PATH"

RUN itwinai sanity-check --torch \
    --optional-deps deepspeed \
    --optional-deps horovod \
    --optional-deps ray