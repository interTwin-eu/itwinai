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

# IGTF-accredited CA bundle - the trust anchors GFAL2 needs to talk to RSE storage endpoints.
#
# Installed from the signed EGI apt repository. The previous approach unpacked
# igtf-policy-installation-bundle.tar.gz, which is a SOURCE tree needing ./configure && make
# install: with --strip-components=1 the certificates landed under
# /etc/grid-security/certificates/src/accredited/ and no <hash>.0 symlinks were ever created,
# so the directory was unusable as an X509_CERT_DIR.
#
# NOTE: ca-policy-egi-core pulls the individual CAs via Recommends, hence no
# --no-install-recommends, and the CA packages do not create their own parent directory,
# hence the mkdir.
RUN set -euo pipefail && \
    mkdir -p /etc/grid-security/certificates /etc/apt/keyrings && \
    curl -fsSL https://repository.egi.eu/sw/production/cas/1/current/GPG-KEY-EUGridPMA-RPM-4 \
    -o /etc/apt/keyrings/egi-igtf.asc && \
    echo "deb [signed-by=/etc/apt/keyrings/egi-igtf.asc] https://repository.egi.eu/sw/production/cas/1/current egi-igtf core" \
    > /etc/apt/sources.list.d/egi-igtf.list && \
    apt-get update && apt-get install -y ca-policy-egi-core && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    test "$(find /etc/grid-security/certificates -name '*.0' | wc -l)" -gt 50

ENV X509_CERT_DIR=/etc/grid-security/certificates

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

USER $NB_UID

# RUCIO transfer stack (GFAL2 + XRootD) installed into the BASE conda environment - the one
# the notebook kernel runs on - rather than a side environment. This is what lets a notebook
# do `import gfal2` and drive RUCIO's DownloadClient in-process, alongside torch and itwinai.
#
# Deliberately placed before every pip layer below: mamba re-solves the environment, and doing
# that while it is still close to the upstream image is far safer than re-solving on top of
# torch. Anything pip installs afterwards lands in site-packages and leaves these packages be.
#
# conda-forge rather than apt: the kernel is /opt/conda/bin/python, while Ubuntu's gfal2.so is
# built against the system libpython. Mixing the two runtimes in one process is not worth the
# risk, and conda-forge is also newer (gfal2 2.23.5 vs 2.22.1) with more protocol plugins.
RUN mamba install -y -n base -c conda-forge \
    gfal2 \
    python-gfal2 \
    gfal2-util \
    xrootd && \
    mamba clean -afy && \
    # Only CONDA_DIR: mamba touches nothing else, and $HOME holds root-owned files
    # (uv's receipt) that ${NB_USER} cannot chmod.
    fix-permissions "${CONDA_DIR}"

# Install jupyter ecosystem
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
# rucio-clients is pinned rather than left to float: rucio-jupyterlab only asks for >=32.0.
ARG RUCIO_CLIENTS_VERSION=39.*
RUN uv pip install rucio-jupyterlab "rucio-clients[argcomplete]==${RUCIO_CLIENTS_VERSION}"

# Install itwinai
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
    # "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@v0.0.2" \
    pytest \
    pytest-xdist \
    psutil

RUN itwinai sanity-check --torch \
    --optional-deps yprov4ml \
    --optional-deps ray

# RUCIO sanity check: the Python API and the CLIs must both work from the kernel interpreter.
# Mirrors the check in env-files/torch/slim.Dockerfile; here the interpreter assertion also
# guards against a side conda environment being prepended to PATH and shadowing the kernel,
# which is what used to break `import gfal2` in notebooks.
RUN test "$(command -v python)" = "${CONDA_DIR}/bin/python" && \
    python -c "import gfal2, gfal2_util, torch, itwinai; from rucio.client.client import Client; \
    from rucio.client.downloadclient import DownloadClient" && \
    rucio --version && \
    gfal-copy --version && \
    test "$(find "${X509_CERT_DIR}" -name '*.0' | wc -l)" -gt 50

# Add tests
WORKDIR /app
COPY --chown=${NB_UID} tests tests
COPY --chown=${NB_UID} env-files/torch/jupyter/slim.Dockerfile Dockerfile

# RUCIO client configuration. This is a template - 'account' is a placeholder - so mount your
# own over /app/rucio.cfg, or point RUCIO_CONFIG at it. See env-files/torch/rucio-testing.txt.
COPY --chown=${NB_UID} env-files/torch/rucio.cfg rucio.cfg

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
