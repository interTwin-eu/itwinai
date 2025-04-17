# FROM python:3.12-slim

# # Set environment variables
# ENV DEBIAN_FRONTEND=noninteractive \
#     PYTHONUNBUFFERED=1 \
#     PIP_NO_CACHE_DIR=1

# # Install OS dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     git \
#     curl \
#     ca-certificates \
#     libglib2.0-0 \
#     libxext6 \
#     libsm6 \
#     libxrender1 \
#     libssl-dev \
#     libffi-dev \
#     libnss3 \
#     wget \
#     && apt-get clean && rm -rf /var/lib/apt/lists/*

# # Create a user
# ARG NB_USER=jovyan
# ARG NB_UID=1000
# ENV USER=${NB_USER}
# ENV HOME=/home/${NB_USER}
# RUN groupadd --gid ${NB_UID} ${NB_USER} && \
#     useradd -m -s /bin/bash -N -u ${NB_UID} -g ${NB_USER} ${NB_USER}

# WORKDIR ${HOME}

# # Install JupyterHub singleuser requirements and JupyterLab
# RUN pip install --upgrade pip && \
#     pip install \
#     jupyterhub==5.2.1 \
#     "notebook>=7.0.0" \
#     "jupyterlab>=4.1,<4.2" \
#     jupyterlab-git \
#     jupyter-server-proxy \
#     ipywidgets \
#     numpy pandas matplotlib seaborn scikit-learn

# # Set permissions
# RUN chown -R ${NB_USER}:${NB_USER} ${HOME}

# # Switch to non-root user
# USER ${NB_USER}

# # Default command (needed for JupyterHub singleuser)
# CMD ["jupyterhub-singleuser"]


# =====================================================================================

# Container image for JupyterHub 2.5.1 -- supports JupyterLab 4

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NB_USER=jovyan \
    NB_UID=1000 \
    NB_GID=1000

# OS deps
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

# Create user and group
RUN groupadd -g ${NB_GID} ${NB_USER} && \
    useradd -m -s /bin/bash -u ${NB_UID} -g ${NB_GID} ${NB_USER}

# Install jupyter ecosystem
RUN pip install --upgrade pip && \
    pip install \
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

# Add custom asyncssh config
COPY env-files/torch/jupyter/asyncssh_config.py /opt/ssh/jupyterhub-singleuser
RUN chmod +x /opt/ssh/jupyterhub-singleuser && chown -R ${NB_UID}:${NB_GID} /opt/ssh/jupyterhub-singleuser

# Add rucio setup
COPY env-files/torch/jupyter/configure.py /opt/setup-rucio-jupyterlab/configure.py
RUN chmod +x /opt/setup-rucio-jupyterlab/configure.py && chown -R ${NB_UID}:${NB_GID} /opt/setup-rucio-jupyterlab

COPY env-files/torch/jupyter/setup.sh /usr/local/bin/setup.sh
RUN chmod +x /usr/local/bin/setup.sh

RUN mkdir -p /opt/rucio/etc && chown -R ${NB_UID}:${NB_GID} /opt/rucio/etc

# Enable JupyterLab
ENV JUPYTER_ENABLE_LAB=yes

# Switch to user
USER ${NB_UID}
WORKDIR /home/${NB_USER}

# Rucio JupyterLab extension
RUN pip install rucio-jupyterlab

# Install itwinai and prov4ml
WORKDIR /home/${NB_USER}/itwinai
COPY --chown=${NB_UID} pyproject.toml pyproject.toml
COPY --chown=${NB_UID} src src

RUN pip install ".[torch]" --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip install \
    "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@new-main" \
    pytest pytest-xdist psutil

ENV PATH="${PATH}:/home/${NB_USER}/.local/bin"
RUN itwinai sanity-check --torch \
    --optional-deps prov4ml \
    --optional-deps ray

# # Additional requirements
# ARG REQUIREMENTS=env-files/torch/requirements/requirements.txt
# COPY --chown=${NB_UID} "${REQUIREMENTS}" additional-requirements.txt
# RUN pip install -r additional-requirements.txt

# Add tests
WORKDIR /app
COPY pyproject.toml pyproject.toml
COPY tests tests
COPY env-files/torch/jupyter/new.Dockerfile Dockerfile

WORKDIR /home/${NB_USER}
CMD ["setup.sh", "start-notebook.sh"]
