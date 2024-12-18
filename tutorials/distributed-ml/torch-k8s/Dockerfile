# FROM ghcr.io/intertwin-eu/itwinai-dev:d01124feb254c6c98a1e5b3ecc76b63b410d03cb-torch-slim
FROM python:3.11-slim-bullseye

WORKDIR /app

RUN pip install --no-cache-dir numpy torch --index-url https://download.pytorch.org/whl/cpu

COPY main.py .
