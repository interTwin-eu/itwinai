FROM ghcr.io/intertwin-eu/itwinai:jlab-slim-latest

USER root

# Setup RUCIO dependencies 
RUN ARCH=$(uname -m) && \
    case "${ARCH}" in \
    x86_64)  MAMBA_ARCH=64 ;; \
    aarch64) MAMBA_ARCH=aarch64 ;; \
    *) echo "Unsupported arch: ${ARCH}" && exit 1 ;; \
    esac && \
    curl -Ls "https://micro.mamba.pm/api/micromamba/linux-${MAMBA_ARCH}/latest" \
    | tar -xvj -C /usr/local bin/micromamba && \
    /usr/local/bin/micromamba create -y -p /opt/conda/envs/rucio -c conda-forge \
    python=3.10 gfal2 python-gfal2 gfal2-util xrootd && \
    /opt/conda/envs/rucio/bin/pip install --no-cache-dir "rucio-clients[argcomplete]==39.*" && \
    /opt/conda/envs/rucio/bin/python -c "import gfal2" && \
    ln -sf /opt/conda/envs/rucio/bin/python3 /usr/bin/python && \
    fix-permissions "/opt/conda/envs/rucio"

# IGTF-accredited CA bundle — needed for GFAL2 to trust RSE storage endpoints
# (e.g. TUBITAK_WEBDAV's TR-Grid CA 2024). Separate from the system trust store.
RUN mkdir -p /opt/certs && \
    curl -s https://repository.egi.eu/sw/production/cas/1/current/tgz/ -o /tmp/listing.html && \
    grep -io 'href="ca_[^"]*\.tar\.gz"' /tmp/listing.html | cut -d'"' -f2 > /tmp/ca_files.txt && \
    for f in $(cat /tmp/ca_files.txt); do \
    curl -sO "https://repository.egi.eu/sw/production/cas/1/current/tgz/$f" && \
    tar xzf "$f" --strip-components=1 -C /opt/certs && rm "$f"; \
    done && \
    fix-permissions "/opt/certs"

ENV X509_CERT_DIR="/opt/certs"

# Ensure the rucio env's PATH wins over the base-env reactivation that happens
# on every container start via before-notebook.d/10activate-conda-env.sh
RUN echo 'export PATH="/opt/conda/envs/rucio/bin:$PATH"' > /usr/local/bin/before-notebook.d/20-rucio-path.sh && \
    chmod +x /usr/local/bin/before-notebook.d/20-rucio-path.sh

USER ${NB_UID}
WORKDIR /app
RUN rm -rf tests src pyproject.toml Dockerfile
COPY --chown=${NB_UID} pyproject.toml pyproject.toml
COPY --chown=${NB_UID} src src
RUN pip install --no-cache-dir .
# rucio.cfg file should be present 
COPY --chown=${NB_UID} rucio.cfg rucio.cfg