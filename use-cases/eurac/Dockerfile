# FROM ghcr.io/intertwin-eu/itwinai:torch-slim-latest
FROM ghcr.io/intertwin-eu/itwinai-dev:73c47fe634f0208309ee1e9475630ac410476f0a-torch-slim

# Make sure that the virualenv is reacheable also from login shell
# This is needed to use this container as Ray Worker/Head on k8s
RUN echo 'PATH="/opt/venv/bin:$PATH"' >> /etc/profile

WORKDIR /app
COPY use-cases/eurac/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY use-cases/eurac eurac