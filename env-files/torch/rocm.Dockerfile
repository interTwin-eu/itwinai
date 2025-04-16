# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

# https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html#docker-image-support
FROM rocm/pytorch:rocm6.3.4_ubuntu22.04_py3.10_pytorch_release_2.4.0

WORKDIR /app
COPY pyproject.toml pyproject.toml
COPY src src
RUN pip install --no-cache-dir .[torch] --extra-index-url https://download.pytorch.org/whl/rocm6.1

# Installation sanity check
RUN itwinai sanity-check --torch \
    # Ray for disrtibuted ML and hyperparameter-tuning
    --optional-deps ray
