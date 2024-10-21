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
    jupyter_server"<2" \
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

CMD ["setup.sh", "start-notebook.sh"]
