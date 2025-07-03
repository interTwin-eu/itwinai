# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - VRE Team @ CERN 23/24 - E. Garcia, G. Guerrieri
# --------------------------------------------------------------------------------------

# https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-pytorch-notebook
# https://github.com/jupyter/docker-stacks/wiki

# FROM quay.io/jupyter/pytorch-notebook:x86_64-cuda12-python-3.11
ARG BASE_IMG_NAME=jupyter/scipy-notebook:python-3.10.11

FROM ${BASE_IMG_NAME}
ARG BASE_IMG_NAME

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER $NB_UID

# jupyterhub must be < 2
RUN conda install -y -n base mamba \
    && mamba install -y -c conda-forge python-gfal2 \
    nodejs \
    jupyterlab"<4" \
    notebook"<7" \
    jupyterhub"<2" \
    jsonschema \
    jupyterlab_server \
    jupyter_server \
    traitlets \
    nbformat \
    ipykernel \
    PyJWT \
    ipywidgets \
    asyncssh \
    peewee \
    && conda clean --all -f -y

USER root

RUN apt-get update -y && apt-get install -y \
    build-essential \
    cmake \
    curl \
    software-properties-common \
    voms-clients-java \
    && rm /opt/conda/bin/voms-proxy-init \
    && ln -s /usr/bin/voms-proxy-init /opt/conda/bin/voms-proxy-init \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*


# ESCAPE grid-security and VOMS setup
RUN wget -q -O - https://dist.eugridpma.info/distribution/igtf/current/GPG-KEY-EUGridPMA-RPM-3 | apt-key add -

RUN apt-get update \
    && add-apt-repository 'deb http://repository.egi.eu/sw/production/cas/1/current egi-igtf core' \
    && apt-get -y install ca-policy-egi-core \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/vomses \
    && wget -q https://indigo-iam.github.io/escape-docs/voms-config/voms-escape.cloud.cnaf.infn.it.vomses -O /etc/vomses/voms-escape.cloud.cnaf.infn.it.vomses \
    && mkdir -p /etc/grid-security/vomsdir/escape \
    && wget -q https://indigo-iam.github.io/escape-docs/voms-config/voms-escape.cloud.cnaf.infn.it.lsc -O /etc/grid-security/vomsdir/escape/voms-escape.cloud.cnaf.infn.it.lsc

# Setup merged CERN CA file on Ubuntu based images.
# This file is contained in the `CERN-bundle.pem` file downloaded using 
RUN mkdir /certs \
    && touch /certs/rucio_ca.pem \
    && curl -fsSL 'https://cafiles.cern.ch/cafiles/certificates/CERN%20Root%20Certification%20Authority%202.crt' | openssl x509 -inform DER -out /tmp/cernrootca2.crt \
    && curl -fsSL 'https://cafiles.cern.ch/cafiles/certificates/CERN%20Grid%20Certification%20Authority(1).crt' -o /tmp/cerngridca.crt \
    && curl -fsSL 'https://cafiles.cern.ch/cafiles/certificates/CERN%20Certification%20Authority.crt' -o /tmp/cernca.crt \
    && cat /tmp/cernrootca2.crt >> /certs/rucio_ca.pem \
    && cat /tmp/cerngridca.crt >> /certs/rucio_ca.pem \
    && cat /tmp/cernca.crt >> /certs/rucio_ca.pem \
    && rm /tmp/*.crt \
    && update-ca-certificates

# # Install trust anchors 
# RUN curl https://ca.cern.ch/cafiles/certificates/CERN%20Root%20Certification%20Authority%202.crt -o /etc/pki/ca-trust/source/anchors/1.crt &&\
#     curl https://ca.cern.ch/cafiles/certificates/CERN%20Grid%20Certification%20Authority.crt -o /etc/pki/ca-trust/source/anchors/2.crt &&\
#     curl https://ca.cern.ch/cafiles/certificates/CERN%20Grid%20Certification%20Authority\(1\).crt -o /etc/pki/ca-trust/source/anchors/3.crt &&\
#     curl http://signet-ca.ijs.si/pub/cacert/signet02cacert.crt -o /etc/pki/ca-trust/source/anchors/4.crt &&\
#     curl https://doku.tid.dfn.de/_media/de:dfnpki:ca:tcs-server-certificate-ca-bundle.tar -o geant-bundle.tar &&\
#     tar xf geant-bundle.tar &&\
#     cp tcs-server-certificate-ca-bundle/*.pem /etc/pki/ca-trust/source/anchors/ &&\
#     rm -rf geant-bundle.tar tcs-server-certificate-ca-bundle &&\
#     update-ca-trust

# Add async ssh script
COPY env-files/torch/jupyter/asyncssh_config.py /opt/ssh/jupyterhub-singleuser
RUN fix-permissions /opt/ssh/jupyterhub-singleuser \
    && chown -R $NB_UID /opt/ssh/jupyterhub-singleuser \
    && chmod +x /opt/ssh/jupyterhub-singleuser

# Setup extension Rucio instance config
COPY env-files/torch/jupyter/configure.py /opt/setup-rucio-jupyterlab/configure.py
RUN fix-permissions /opt/setup-rucio-jupyterlab/configure.py \
    && chown -R $NB_UID /opt/setup-rucio-jupyterlab/configure.py \
    && chmod +x /opt/setup-rucio-jupyterlab/configure.py

COPY env-files/torch/jupyter/setup.sh /usr/local/bin/setup.sh
RUN fix-permissions /usr/local/bin/setup.sh \
    && sed -i -e 's/\r$/\n/' /usr/local/bin/setup.sh \
    && chmod +x /usr/local/bin/setup.sh

RUN mkdir -p /opt/rucio/etc \
    #    && touch /opt/rucio/etc/rucio.cfg \
    && fix-permissions /opt/rucio/etc \
    && chown -R ${NB_UID}:${NB_GID} /opt/rucio/etc 

#    && /usr/local/bin/setup.sh 
#RUN chown -R $NB_UID $HOME/.jupyter/jupyter_notebook_config.json 
#    && chown -R $NB_UID /etc/jupyter/jupyter_notebook_config.py


#ENV IPYTHONDIR=/etc/ipython
#ADD ipython_kernel_config.json /etc/ipython/profile_default/ipython_kernel_config.json
#RUN chown -R $NB_UID /etc/ipython
ENV JUPYTER_ENABLE_LAB=yes

USER $NB_UID
WORKDIR $HOME

# Install rucio-jupyterlab with jlab v=3
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir rucio-jupyterlab==0.10.0 \
    && jupyter serverextension enable --py rucio_jupyterlab --sys-prefix 

#############################
###        itwinai        ###
#############################

# Improve robustness: avoid silent override by Singularity/Apptainer
ENV PYTHONPATH=""

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Needed by Prov4ML/yProvML to generate provenance graph
    dot2tex \
    # mpi4py system-wide -- also installs mpirun
    python3-mpi4py \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install MPI4py also in the conda env
RUN mamba install -y -c conda-forge mpi4py openmpi \
    && mamba clean --all -f -y

USER $NB_UID

# Install itwinai with torch
WORKDIR "$HOME/itwinai"
COPY --chown=${NB_UID} pyproject.toml pyproject.toml
COPY --chown=${NB_UID} src src

# DO NOT install dev extras as they may change some versions of ipython and ipykernel
RUN pip install --no-cache-dir .[torch] --extra-index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir \
    "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@v0.0.1" \
    pytest \
    pytest-xdist \
    psutil

# Installation sanity check
RUN itwinai sanity-check --torch \
    --optional-deps prov4ml \
    --optional-deps ray

# Additional pip deps
ARG REQUIREMENTS=env-files/torch/requirements/requirements.txt
COPY --chown=${NB_UID} "${REQUIREMENTS}" additional-requirements.txt
RUN pip install --no-cache-dir -r additional-requirements.txt

# Add tests
WORKDIR /app
COPY pyproject.toml pyproject.toml
COPY tests tests
# Add Dockerfile
COPY env-files/torch/jupyter/slim.Dockerfile Dockerfile

WORKDIR $HOME

CMD ["setup.sh", "start-notebook.sh"]


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
LABEL org.opencontainers.image.description="slim itwinai image with torch dependencies, and Rucio client for jupyterlab singleuser server enabled for interLink offloading"
LABEL org.opencontainers.image.base.digest=${BASE_IMG_DIGEST}
LABEL org.opencontainers.image.base.name=${BASE_IMG_NAME}
