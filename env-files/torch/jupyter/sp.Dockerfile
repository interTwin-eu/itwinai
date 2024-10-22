# https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-pytorch-notebook
# https://github.com/jupyter/docker-stacks/wiki

FROM jupyter/scipy-notebook:python-3.10.11
# FROM quay.io/jupyter/pytorch-notebook:x86_64-cuda12-python-3.11

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

USER $NB_UID
WORKDIR $HOME

# Install rucio-jupyterlab with jlab v=3
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir rucio-jupyterlab==0.10.0 \
    && jupyter serverextension enable --py rucio_jupyterlab --sys-prefix 

#############################
###        itwinai        ###
#############################

USER root

WORKDIR /tmp/cuda
# CUDA Toolkit:
# - https://developer.nvidia.com/cuda-downloads
# - Installation guide: https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#ubuntu
# - cuda-toolkit metapackage: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#meta-packages
# cuDNN:
# - https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html#installing-cudnn-on-linux
# NCCL: 
# - https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#debian
# *NOTE* to correctly install Apex below, CUDA toolkit version must match with the torch CUDA backend version
RUN wget -O cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring.deb \
    && apt-get update && apt-get install -y \
    # CUDA toolkit metapackage (does not include the Nvidia driver)
    cuda-toolkit-12-4 \
    # cuDNN
    cudnn-cuda-12 \
    # NCCL
    libnccl2 \
    libnccl-dev \
    # Nvidia driver, as explained here: https://developer.nvidia.com/cuda-downloads
    nvidia-open \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}\
    LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# OpenMPI
WORKDIR /tmp/ompi
ENV OPENMPI_VERSION=5.0.5 \
    OPENMPI_MINOR=5.0 
ENV OPENMPI_URL="https://download.open-mpi.org/release/open-mpi/v${OPENMPI_MINOR}/openmpi-${OPENMPI_VERSION}.tar.gz" 
ENV OPENMPI_DIR=/opt/openmpi-${OPENMPI_VERSION} 
ENV PATH="${OPENMPI_DIR}/bin:${PATH}" 
ENV LD_LIBRARY_PATH="${OPENMPI_DIR}/lib:${LD_LIBRARY_PATH}" 
ENV MANPATH=${OPENMPI_DIR}/share/man:${MANPATH}
RUN wget -O openmpi-$OPENMPI_VERSION.tar.gz $OPENMPI_URL && tar xzf openmpi-$OPENMPI_VERSION.tar.gz \
    && cd openmpi-$OPENMPI_VERSION && ./configure --prefix=$OPENMPI_DIR && make install

# Cleanup
RUN rm -rf /tmp/*

USER $NB_UID

# Torch and smaller deps
RUN pip install --no-cache-dir \
    'numpy<2' \
    packaging \
    py-cpuinfo \
    torch==2.4.* torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    wheel

# Apex: https://github.com/NVIDIA/apex
# (needed for DeepSpeed *_FUSED optinal build options)
# Note: it will take more than an hour to build
RUN cd /tmp && git clone https://github.com/NVIDIA/apex && cd apex \
    && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ \
    && rm -rf /tmp/apex
# Transformer engine: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html
# (needed for DeepSpeed *TRANSFORMER* optinal build options)
RUN pip install --no-cache-dir transformer_engine[pytorch]

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
    DS_BUILD_UTILS=1 \
    DS_BUILD_AIO=1 \
    DS_BUILD_FUSED_ADAM=1 \
    DS_BUILD_FUSED_LAMB=1 \
    DS_BUILD_TRANSFORMER=1 \
    DS_BUILD_STOCHASTIC_TRANSFORMER=1 \
    DS_BUILD_TRANSFORMER_INFERENCE=1
RUN pip install --no-cache-dir \
    deepspeed \
    git+https://github.com/horovod/horovod.git \
    "prov4ml[linux]@git+https://github.com/matbun/ProvML" \
    ray ray[tune]
# RUN pip install --no-cache-dir \
#     'numpy<2' \
#     packaging \
#     py-cpuinfo \
#     torch==2.4.* torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
#     wheel \
#     && pip install --no-cache-dir \
#     deepspeed \
#     git+https://github.com/horovod/horovod.git \
#     "prov4ml[linux]@git+https://github.com/matbun/ProvML" \
#     ray ray[tune]
# To test that Horovod was succesfully built with NCCL run: horovodrun --check-build

# Core itwinai lib
WORKDIR $HOME/itwinai
# COPY env-files/torch/jupyter/install_itwinai_torch.sh ./
COPY pyproject.toml ./
COPY src ./
RUN pip install --no-cache-dir ".[torch,dev]" && itwinai sanity-check --torch
# RUN bash install_itwinai_torch.sh && rm install_itwinai_torch.sh 
# Additional pip deps
ARG REQUIREMENTS=env-files/torch/jupyter/requirements.txt
COPY ${REQUIREMENTS} ./
RUN pip install --no-cache-dir -r $(basename ${REQUIREMENTS})

WORKDIR $HOME

CMD ["setup.sh", "start-notebook.sh"]
