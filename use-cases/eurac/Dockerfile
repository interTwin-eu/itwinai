# FROM ghcr.io/intertwin-eu/itwinai:torch-slim-latest
FROM ghcr.io/intertwin-eu/itwinai-dev:73c47fe634f0208309ee1e9475630ac410476f0a-torch-slim
WORKDIR /app
COPY use-cases/eurac/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY use-cases/eurac eurac