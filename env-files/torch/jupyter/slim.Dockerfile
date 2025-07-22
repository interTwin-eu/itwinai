# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - VRE Team @ CERN 23/24 - E. Garcia, G. Guerrieri
# --------------------------------------------------------------------------------------

# Container image for JupyterHub 2.5.1 -- supports JupyterLab 4
# This generates an image that can be both offloaded via interLink and started on a local cloud

ARG BASE_IMG_NAME=quay.io/jupyter/minimal-notebook:python-3.12

FROM ${BASE_IMG_NAME}
ARG BASE_IMG_NAME

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Improve robustness: avoid silent override by Singularity/Apptainer
    PYTHONPATH="" \
    # User-fiendly page for Rucio clients
    PAGER=cat \
    # Install uv packages system wide (no need for .venv):
    # https://docs.astral.sh/uv/reference/environment/#uv_system_python
    UV_SYSTEM_PYTHON=true \
    # https://docs.astral.sh/uv/reference/environment/#uv_no_cache
    UV_NO_CACHE=1

# OS deps
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    wget \
    git \
    ca-certificates \
    software-properties-common \
    libnss3 \
    libffi-dev \
    libssl-dev \
    dot2tex \
    python3-mpi4py \
    voms-clients-java \
    gnupg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up CERN/ESCAPE CA certs
# RUN wget -q -O - https://dist.eugridpma.info/distribution/igtf/current/GPG-KEY-EUGridPMA-RPM-3 | apt-key add - && \
#     add-apt-repository 'deb http://repository.egi.eu/sw/production/cas/1/current egi-igtf core' && \
#     apt-get update && \
#     apt-get -y install ca-policy-egi-core && \
#     rm -rf /var/lib/apt/lists/*
# Since they are not available for Debian as an apt package, we need to install them manually
RUN mkdir -p /etc/grid-security/certificates && \
    wget -q https://dist.eugridpma.info/distribution/igtf/current/igtf-policy-installation-bundle.tar.gz && \
    tar -xzf igtf-policy-installation-bundle.tar.gz -C /etc/grid-security/certificates --strip-components=1 && \
    rm igtf-policy-installation-bundle.tar.gz

# VOMS setup
RUN mkdir -p /etc/vomses && \
    wget -q https://indigo-iam.github.io/escape-docs/voms-config/voms-escape.cloud.cnaf.infn.it.vomses -O /etc/vomses/voms-escape.cloud.cnaf.infn.it.vomses && \
    mkdir -p /etc/grid-security/vomsdir/escape && \
    wget -q https://indigo-iam.github.io/escape-docs/voms-config/voms-escape.cloud.cnaf.infn.it.lsc -O /etc/grid-security/vomsdir/escape/voms-escape.cloud.cnaf.infn.it.lsc

# CERN CA certs
RUN mkdir /certs && touch /certs/rucio_ca.pem && \
    curl -fsSL 'https://cafiles.cern.ch/cafiles/certificates/CERN%20Root%20Certification%20Authority%202.crt' | openssl x509 -inform DER -out /tmp/cernrootca2.crt && \
    curl -fsSL 'https://cafiles.cern.ch/cafiles/certificates/CERN%20Grid%20Certification%20Authority(1).crt' -o /tmp/cerngridca.crt && \
    curl -fsSL 'https://cafiles.cern.ch/cafiles/certificates/CERN%20Certification%20Authority.crt' -o /tmp/cernca.crt && \
    cat /tmp/cernrootca2.crt >> /certs/rucio_ca.pem && \
    cat /tmp/cerngridca.crt >> /certs/rucio_ca.pem && \
    cat /tmp/cernca.crt >> /certs/rucio_ca.pem && \
    rm /tmp/*.crt && \
    update-ca-certificates

# Add custom asyncssh config (interLink)
COPY env-files/torch/jupyter/asyncssh_config.py /opt/ssh/jupyterhub-singleuser
RUN chmod +x /opt/ssh/jupyterhub-singleuser && chown -R ${NB_UID}:${NB_GID} /opt/ssh/jupyterhub-singleuser

# Add Rucio setup
COPY env-files/torch/jupyter/configure.py /opt/setup-rucio-jupyterlab/configure.py
RUN chmod +x /opt/setup-rucio-jupyterlab/configure.py && chown -R ${NB_UID}:${NB_GID} /opt/setup-rucio-jupyterlab
COPY env-files/torch/jupyter/setup.sh /usr/local/bin/setup.sh
RUN chmod +x /usr/local/bin/setup.sh
RUN mkdir -p /opt/rucio/etc && chown -R ${NB_UID}:${NB_GID} /opt/rucio/etc
# Wrap Rucio setup.sh ans start.sh under a single file (which is called from ENTRYPOINT)
RUN mv /usr/local/bin/start.sh /usr/local/bin/start-original.sh
COPY env-files/torch/jupyter/start-cloud.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Enable JupyterLab
ENV JUPYTER_ENABLE_LAB=yes

# install uv so that uv → /usr/local/bin/uv
RUN curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh

# Install jupyter ecosystem
USER $NB_UID
RUN uv pip install --upgrade pip && \
    uv pip install \
    "jupyterhub==5.2.1" \
    "notebook>=7.0.0" \
    "jupyterlab>=4.1,<4.2" \
    "jupyterlab-git" \
    "jupyter-server-proxy" \
    "ipywidgets" \
    "PyJWT" \
    "asyncssh" \
    "peewee" \
    "numpy" \
    "pandas" \
    "matplotlib" \
    "scikit-learn" \
    "nbformat" \
    "ipykernel" \
    "jsonschema" \
    "traitlets"

# Needs to be installed separated from the rest of the jupyterlab ecosystem to avoid conflicts...
RUN uv pip install rucio-jupyterlab 

# Install itwinai and prov4ml
WORKDIR "$HOME/itwinai"
COPY --chown=${NB_UID} pyproject.toml pyproject.toml
COPY --chown=${NB_UID} src src


RUN uv pip install --no-cache-dir --upgrade pip \
    && uv pip install --no-cache-dir \
    # Select from which index to install torch
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    # This is needed by UV to trust all indexes:
    --index-strategy unsafe-best-match \
    # Install packages
    .[torch] \
    "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@v0.0.2" \
    pytest \
    pytest-xdist \
    psutil

RUN itwinai sanity-check --torch \
    --optional-deps prov4ml \
    --optional-deps ray

# Add tests
WORKDIR /app
COPY --chown=${NB_UID} tests tests
COPY --chown=${NB_UID} env-files/torch/jupyter/slim.Dockerfile Dockerfile

# This is most likely ignored when jupyterlab is launched from jhub, in favour of jupyterhub-singleuser
CMD ["start-notebook.sh"]


# Labels
ARG CREATION_DATE
ARG COMMIT_HASH
ARG ITWINAI_VERSION
ARG IMAGE_FULL_NAME
ARG BASE_IMG_DIGEST

# https://github.com/opencontainers/image-spec/blob/main/annotations.md#pre-defined-annotation-keys
LABEL org.opencontainers.image.created=${CREATION_DATE}
LABEL org.opencontainers.image.authors="Matteo Bunino - matteo.bunino@cern.ch, VRE Team @ CERN 23/24 - E. Garcia, G. Guerrieri"
LABEL org.opencontainers.image.url="https://github.com/interTwin-eu/itwinai"
LABEL org.opencontainers.image.documentation="https://itwinai.readthedocs.io/"
LABEL org.opencontainers.image.source="https://github.com/interTwin-eu/itwinai"
LABEL org.opencontainers.image.version=${ITWINAI_VERSION}
LABEL org.opencontainers.image.revision=${COMMIT_HASH}
LABEL org.opencontainers.image.vendor="CERN - European Organization for Nuclear Research"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.ref.name=${IMAGE_FULL_NAME}
LABEL org.opencontainers.image.title="itwinai"
LABEL org.opencontainers.image.description="slim itwinai image with torch dependencies, and Rucio client for jupyterlab v4 singleuser server enabled for interLink offloading"
LABEL org.opencontainers.image.base.digest=${BASE_IMG_DIGEST}
LABEL org.opencontainers.image.base.name=${BASE_IMG_NAME}
