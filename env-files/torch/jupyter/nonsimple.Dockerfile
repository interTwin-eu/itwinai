FROM jupyter/scipy-notebook:python-3.10.11 AS nb
# FROM quay.io/jupyter/pytorch-notebook:x86_64-cuda12-python-3.11

FROM nvcr.io/nvidia/pytorch:24.09-py3 AS nvidia

#################################################################################################
# docker-stacks-foundation:                                                                     #
# https://github.com/jupyter/docker-stacks/blob/main/images/docker-stacks-foundation/Dockerfile #
#################################################################################################

ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# Install all OS dependencies for the Server that starts
# but lacks all features (e.g., download as all possible file formats)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --yes && \
    # - `apt-get upgrade` is run to patch known vulnerabilities in system packages
    #   as the Ubuntu base image is rebuilt too seldom sometimes (less than once a month)
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    # - bzip2 is necessary to extract the micromamba executable.
    bzip2 \
    ca-certificates \
    locales \
    # - `netbase` provides /etc/{protocols,rpc,services}, part of POSIX
    #   and required by various C functions like getservbyname and getprotobyname
    #   https://github.com/jupyter/docker-stacks/pull/2129
    netbase \
    sudo \
    # - `tini` is installed as a helpful container entrypoint,
    #   that reaps zombie processes and such of the actual executable we want to start
    #   See https://github.com/krallin/tini#why-tini for details
    tini \
    wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    echo "C.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen

# Configure environment
ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    NB_USER="${NB_USER}" \
    NB_UID=${NB_UID} \
    NB_GID=${NB_GID} \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    LANGUAGE=C.UTF-8
ENV PATH="${CONDA_DIR}/bin:${PATH}" \
    HOME="/home/${NB_USER}"

# Copy a script that we will use to correct permissions after running certain commands
COPY --from=nb /usr/local/bin/fix-permissions /usr/local/bin/fix-permissions
RUN chmod a+rx /usr/local/bin/fix-permissions

# Enable prompt color in the skeleton .bashrc before creating the default NB_USER
# hadolint ignore=SC2016
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc && \
    # More information in: https://github.com/jupyter/docker-stacks/pull/2047
    # and docs: https://docs.conda.io/projects/conda/en/latest/dev-guide/deep-dives/activation.html
    echo 'eval "$(conda shell.bash hook)"' >> /etc/skel/.bashrc

# Delete existing user with UID="${NB_UID}" if it exists
# hadolint ignore=SC2046
RUN if grep -q "${NB_UID}" /etc/passwd; then \
    userdel --remove $(id -un "${NB_UID}"); \
    fi

# Create "${NB_USER}" user (`jovyan` by default) with UID="${NB_UID}" (`1000` by default) and in the 'users' group
# and make sure these dirs are writable by the `users` group.
RUN echo "auth requisite pam_deny.so" >> /etc/pam.d/su && \
    sed -i.bak -e 's/^%admin/#%admin/' /etc/sudoers && \
    sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers && \
    useradd --no-log-init --create-home --shell /bin/bash --uid "${NB_UID}" --no-user-group "${NB_USER}" && \
    mkdir -p "${CONDA_DIR}" && \
    chown "${NB_USER}:${NB_GID}" "${CONDA_DIR}" && \
    chmod g+w /etc/passwd && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

USER ${NB_UID}

# Pin the Python version here, or set it to "default"
ARG PYTHON_VERSION=3.10

# Setup work directory for backward-compatibility
RUN mkdir "/home/${NB_USER}/work" && \
    fix-permissions "/home/${NB_USER}"

# Download and install Micromamba, and initialize the Conda prefix.
#   <https://github.com/mamba-org/mamba#micromamba>
#   Similar projects using Micromamba:
#     - Micromamba-Docker: <https://github.com/mamba-org/micromamba-docker>
#     - repo2docker: <https://github.com/jupyterhub/repo2docker>
# Install Python, Mamba, and jupyter_core
# Cleanup temporary files and remove Micromamba
# Correct permissions
# Do all this in a single RUN command to avoid duplicating all of the
# files across image layers when the permissions change
COPY --from=nb --chown="${NB_UID}:${NB_GID}" "${CONDA_DIR}/.condarc" "${CONDA_DIR}/.condarc"
WORKDIR /tmp
RUN set -x && \
    arch=$(uname -m) && \
    if [ "${arch}" = "x86_64" ]; then \
    # Should be simpler, see <https://github.com/mamba-org/mamba/issues/1437>
    arch="64"; \
    fi && \
    # https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#linux-and-macos
    wget --progress=dot:giga -O - \
    "https://micro.mamba.pm/api/micromamba/linux-${arch}/latest" | tar -xvj bin/micromamba && \
    PYTHON_SPECIFIER="python=${PYTHON_VERSION}" && \
    if [[ "${PYTHON_VERSION}" == "default" ]]; then PYTHON_SPECIFIER="python"; fi && \
    # Install the packages
    ./bin/micromamba install \
    --root-prefix="${CONDA_DIR}" \
    --prefix="${CONDA_DIR}" \
    --yes \
    'jupyter_core' \
    # excluding mamba 2.X due to several breaking changes
    # https://github.com/jupyter/docker-stacks/pull/2147
    'mamba<2.0.0' \
    "${PYTHON_SPECIFIER}" && \
    rm -rf /tmp/bin/ && \
    # Pin major.minor version of python
    # https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#preventing-packages-from-updating-pinning
    mamba list --full-name 'python' | awk 'END{sub("[^.]*$", "*", $2); print $1 " " $2}' >> "${CONDA_DIR}/conda-meta/pinned" && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Copy local files as late as possible to avoid cache busting
#COPY --from=nb /usr/local/bin/run-hooks.sh /usr/local/bin/run-hooks.sh
COPY --from=nb /usr/local/bin/start.sh /usr/local/bin/start.sh

# Configure container entrypoint
ENTRYPOINT ["tini", "-g", "--", "start.sh"]

USER root

# Create dirs for startup hooks
RUN mkdir /usr/local/bin/start-notebook.d && \
    mkdir /usr/local/bin/before-notebook.d

#COPY --from=nb /usr/local/bin/before-notebook.d/10activate-conda-env.sh /usr/local/bin/before-notebook.d/10activate-conda-env.sh

# Switch back to jovyan to avoid accidental container runs as root
USER ${NB_UID}

WORKDIR "${HOME}"


#################################################################################################
# base-notebook:                                                                                #
# https://github.com/jupyter/docker-stacks/blob/main/images/base-notebook/Dockerfile            #
#################################################################################################

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# Install all OS dependencies for the Server that starts
# but lacks all features (e.g., download as all possible file formats)
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    # - Add necessary fonts for matplotlib/seaborn
    #   See https://github.com/jupyter/docker-stacks/pull/380 for details
    fonts-liberation \
    # - `pandoc` is used to convert notebooks to html files
    #   it's not present in the aarch64 Ubuntu image, so we install it here
    pandoc \
    # - `run-one` - a wrapper script that runs no more
    #   than one unique instance of some command with a unique set of arguments,
    #   we use `run-one-constantly` to support the `RESTARTABLE` option
    run-one \
    npm && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

# Install JupyterHub, JupyterLab, NBClassic and Jupyter Notebook
# Generate a Jupyter Server config
# Cleanup temporary files
# Correct permissions
# Do all this in a single RUN command to avoid duplicating all of the
# files across image layers when the permissions change
WORKDIR /tmp
RUN mamba install --yes \
    'jupyterhub' \
    'jupyterlab' \
    'nbclassic' \
    'notebook' && \
    jupyter server --generate-config && \
    mamba clean --all -f -y && \
    npm cache clean --force && \
    jupyter lab clean && \
    rm -rf "/home/${NB_USER}/.cache/yarn" && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

ENV JUPYTER_PORT=8888
EXPOSE $JUPYTER_PORT

# Configure container startup
#CMD ["start-notebook.py"]

# Copy local files as late as possible to avoid cache busting
#COPY --from=nb /usr/local/bin/start-notebook.py /usr/local/bin/start-notebook.py
COPY --from=nb /usr/local/bin/start-notebook.sh /usr/local/bin/start-notebook.sh
#COPY --from=nb /usr/local/bin/start-singleuser.py /usr/local/bin/start-singleuser.py
COPY --from=nb /usr/local/bin/start-singleuser.sh /usr/local/bin/start-singleuser.sh
COPY --from=nb /etc/jupyter/jupyter_server_config.py /etc/jupyter/jupyter_server_config.py
COPY --from=nb /etc/jupyter/docker_healthcheck.py /etc/jupyter/docker_healthcheck.py

# Fix permissions on /etc/jupyter as root
USER root
RUN fix-permissions /etc/jupyter/

# HEALTHCHECK documentation: https://docs.docker.com/engine/reference/builder/#healthcheck
# This healtcheck works well for `lab`, `notebook`, `nbclassic`, `server`, and `retro` jupyter commands
# https://github.com/jupyter/docker-stacks/issues/915#issuecomment-1068528799
HEALTHCHECK --interval=3s --timeout=1s --start-period=3s --retries=3 \
    CMD /etc/jupyter/docker_healthcheck.py || exit 1

# Switch back to jovyan to avoid accidental container runs as root
USER ${NB_UID}

WORKDIR "${HOME}"


#################################################################################################
# minimal-notebook:                                                                             #
# https://github.com/jupyter/docker-stacks/blob/main/images/minimal-notebook/Dockerfile         #
#################################################################################################

LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# Install all OS dependencies for a fully functional Server
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    # Common useful utilities
    curl \
    git \
    nano-tiny \
    tzdata \
    unzip \
    vim-tiny \
    # git-over-ssh
    openssh-client \
    # `less` is needed to run help in R
    # see: https://github.com/jupyter/docker-stacks/issues/1588
    less \
    # `nbconvert` dependencies
    # https://nbconvert.readthedocs.io/en/latest/install.html#installing-tex
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-plain-generic \
    # Enable clipboard on Linux host systems
    xclip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create alternative for nano -> nano-tiny
RUN update-alternatives --install /usr/bin/nano nano /bin/nano-tiny 10

# Switch back to jovyan to avoid accidental container runs as root
USER ${NB_UID}

# Add an R mimetype option to specify how the plot returns from R to the browser
COPY --from=nb --chown=${NB_UID}:${NB_GID} /opt/conda/lib/R/etc/Rprofile.site /opt/conda/lib/R/etc/Rprofile.site

# Add setup scripts that may be used by downstream images or inherited images
#COPY --from=nb /opt/setup-scripts/ /opt/setup-scripts/







#####################
# custom and fixes  #
#####################

USER $NB_UID

# jupyterhub must be < 2
RUN conda install -y -n base mamba \
    && mamba install -y -c conda-forge python-gfal2 \
    nodejs \
    #jupyterlab"<4" \
    #notebook"<7" \
    #jupyterhub"<2" \
    jsonschema \
    #jupyterlab_server \
    #jupyter_server"<2" \
    traitlets \
    nbformat \
    ipykernel \
    PyJWT \
    ipywidgets \
    asyncssh \
    peewee \
    jupyter_client==8.2.0 \
    jupyter_core==5.3.0 \
    jupyter_events==0.6.3 \
    jupyter-lsp==2.1.0 \
    jupyter_server==1.24.0 \
    jupyter_server_fileid==0.9.2 \
    jupyter-server-mathjax==0.2.6 \
    jupyter_server_terminals==0.4.4 \
    jupyter_server_ydoc==0.8.0 \
    jupyter_telemetry==0.1.0 \
    jupyter_ydoc==0.2.4 \
    jupyterhub==1.5.0 \
    jupyterlab==3.6.8 \
    jupyterlab-git==0.41.0 \
    jupyterlab_pygments==0.2.2 \
    jupyterlab_server==2.22.1 \
    jupyterlab_widgets==3.0.7 \
    notebook==6.5.4 \
    notebook-shim==0.2.3 \
    ipython==8.13.2 \
    ipython_genutils==0.2.0 \
    && conda clean --all -f -y


USER root

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN apt update -y \
    && apt install -y curl voms-clients-java software-properties-common \
    && rm /opt/conda/bin/voms-proxy-init \
    && ln -s /usr/bin/voms-proxy-init /opt/conda/bin/voms-proxy-init \
    && apt clean -y \
    && rm -rf /var/lib/apt/lists/*


# ESCAPE grid-security and VOMS setup
RUN wget -q -O - https://dist.eugridpma.info/distribution/igtf/current/GPG-KEY-EUGridPMA-RPM-3 | apt-key add -

RUN apt update \
    && add-apt-repository 'deb http://repository.egi.eu/sw/production/cas/1/current egi-igtf core' \
    && apt -y install ca-policy-egi-core \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/vomses \
    && wget https://indigo-iam.github.io/escape-docs/voms-config/voms-escape.cloud.cnaf.infn.it.vomses -O /etc/vomses/voms-escape.cloud.cnaf.infn.it.vomses \
    && mkdir -p /etc/grid-security/vomsdir/escape \
    && wget https://indigo-iam.github.io/escape-docs/voms-config/voms-escape.cloud.cnaf.infn.it.lsc -O /etc/grid-security/vomsdir/escape/voms-escape.cloud.cnaf.infn.it.lsc

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

# Fix jupyter server extensions
RUN rm -rf /usr/local/share/jupyter/labextensions/* 
# && mamba install -y -c conda-forge "ipympl>=0.9.3"
USER $NB_UID
WORKDIR $HOME
COPY env.txt ./
RUN mamba install -y --file env.txt

# USER $NB_UID
# WORKDIR $HOME

CMD ["setup.sh", "start-notebook.sh"]
